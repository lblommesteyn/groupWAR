"""Who wins the rebound, from positions at release, and what a miss is worth.

Landing model: secured-ball locations of training misses (other game folds), expressed relative to the rim in a
frame rotated to the shot direction (long/short, left/right of the shot line), pooled by shot-distance bin, then
rotated back onto each test shot and smoothed onto a 1 ft grid within 24 ft of the rim -> weights w(q).
Control of the landing zone at release (numpy port of holes_field):
  voronoi  nearest player wins q          kinematic  smallest mean arrival time wins (velocity-aware)
  prob     P(min_O T < min_D T), Clark    each summarised as sum_q w(q) * control(q)
Geometry: attackers / defenders within 12 ft of the rim, attackers boxed out (a defender closer to the rim within
4 ft of the attacker-rim line), offensive crash speed toward the rim, attackers back (x < 20 ft from half court).
Arms (5-fold GroupKFold by game; logistic and gradient boosting): base (shot distance, angle, type, block)
  + geometry, + voronoi, + kinematic, + prob, + all.
Value: E[second-chance points | OREB] and E[opponent points | DREB] regressed on release geometry
(floor balance, crash), giving value_of_miss = P(OREB) E[pts|OREB] - P(DREB) E[opp pts|DREB].
"""
import json, math, pickle
import numpy as np
from scipy.special import erf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from masked_paths import ROOT

HOOP = np.array([41.75, 0.0])
V_MAX, A_MAX, T_REACT, S0, V_REF, K_DIST = 20.0, 20.0, 0.2, 0.15, 10.0, 0.1
DBINS = [0, 4, 10, 16, 22, 26, 99]
OUT = ROOT / 'results/rebounds'

gx, gy = np.meshgrid(np.arange(17.5, 47.0, 1.0), np.arange(-24.5, 25.0, 1.0), indexing='ij')
GRID = np.column_stack([gx.ravel(), gy.ravel()])
GRID = GRID[np.hypot(GRID[:, 0] - HOOP[0], GRID[:, 1]) <= 24]


def arrival(pos, vel, q):
    """pos, vel [10, 2]; q [G, 2] -> [10, G]"""
    sp = np.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * np.minimum(1, V_MAX / np.maximum(sp, 1e-6))
    pr = pos + vel * T_REACT
    dx = q[None, :, 0] - pr[:, :1]; dy = q[None, :, 1] - pr[:, 1:]
    d = np.maximum(np.hypot(dx, dy), 1e-3)
    u0 = np.clip((vel[:, :1] * dx + vel[:, 1:] * dy) / d, -V_MAX, V_MAX)
    t1 = (V_MAX - u0) / A_MAX; s1 = u0 * t1 + 0.5 * A_MAX * t1 ** 2
    ta = (-u0 + np.sqrt(u0 ** 2 + 2 * A_MAX * d)) / A_MAX
    return T_REACT + np.where(s1 >= d, ta, t1 + (d - s1) / V_MAX)


Phi = lambda x: 0.5 * (1 + erf(x / math.sqrt(2)))
phi = lambda x: np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def clark(mu, var):
    m, v = mu[0], var[0]
    for k in range(1, len(mu)):
        a = np.sqrt(v + var[k]); al = (m - mu[k]) / a
        P1, P2, ph = Phi(-al), Phi(al), phi(al)
        mean = m * P1 + mu[k] * P2 - a * ph
        sec = (m * m + v) * P1 + (mu[k] ** 2 + var[k]) * P2 - (m + mu[k]) * a * ph
        m, v = mean, np.maximum(sec - mean * mean, 1e-6)
    return m, v


def frame_of(r):
    """unit vectors (along shot direction from shooter to rim, lateral)"""
    sh = r['pos'][0].astype(float); u = HOOP - sh
    n = np.linalg.norm(u)
    u = u / n if n > 0.5 else np.array([1.0, 0.0])
    return u, np.array([-u[1], u[0]])


def dbin(d):
    return int(np.searchsorted(DBINS, d, 'right') - 1)


def landing_bank(R):
    bank = {b: [] for b in range(len(DBINS) - 1)}
    for r in R:
        if r['secure_xy'] is None or r['secure_ms'] > 4000:
            continue
        u, l = frame_of(r); rel = r['secure_xy'].astype(float) - HOOP
        bank[dbin(r['dist'])].append((rel @ u, rel @ l))
    return {b: np.array(v) for b, v in bank.items()}


def landing_weights(r, bank, bw=2.0):
    u, l = frame_of(r); pts = bank[dbin(r['dist'])]
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(r['event_id']).choice(len(pts), 1500, replace=False)]
    xy = HOOP + pts[:, :1] * u + pts[:, 1:] * l
    d2 = ((GRID[:, None, :] - xy[None]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / bw ** 2).sum(1)
    return w / w.sum()


def geometry(r):
    pos, vel = r['pos'].astype(float), r['vel'].astype(float)
    drim = np.hypot(pos[:, 0] - HOOP[0], pos[:, 1])
    boxed = 0
    for i in range(5):
        a = pos[i]; v = HOOP - a; L = np.linalg.norm(v)
        if L < 1e-3:
            continue
        for j in range(5, 10):
            w = pos[j] - a; t = (w @ v) / L
            if 0 < t < L and abs(v[0] * w[1] - v[1] * w[0]) / L < 4 and drim[j] < drim[i]:
                boxed += 1; break
    toward = ((HOOP - pos) / np.maximum(drim[:, None], 1e-3) * vel).sum(1)
    return [(drim[:5] < 12).sum(), (drim[5:] < 12).sum(), boxed, toward[1:5].clip(0).sum(), toward[5:].clip(0).sum(),
            (pos[:5, 0] < 20).sum(), (pos[5:, 0] < 20).sum(), drim[1:5].min(), drim[5:].min()]


def control_feats(r, w):
    pos, vel, pe = r['pos'].astype(float), r['vel'].astype(float), r['pe'].astype(float)
    m = w > 1e-4; q = GRID[m]; wq = w[m] / w[m].sum()
    d = np.hypot(q[None, :, 0] - pos[:, :1], q[None, :, 1] - pos[:, 1:])
    vor = (d[:5].min(0) < d[5:].min(0)).astype(float)
    mu = arrival(pos, vel, q); var = S0 ** 2 + (pe[:, None] / V_REF) ** 2 + (K_DIST * mu) ** 2
    kin = (mu[:5].min(0) < mu[5:].min(0)).astype(float)
    mo, vo = clark(mu[:5], var[:5]); md, vd = clark(mu[5:], var[5:])
    prob = Phi((md - mo) / np.sqrt(vo + vd))
    # shooter excluded (follow-ups are rare; tests whether the four teammates control the zone)
    mo4, vo4 = clark(mu[1:5], var[1:5])
    prob4 = Phi((md - mo4) / np.sqrt(vo4 + vd))
    return [wq @ vor, wq @ kin, wq @ prob, wq @ prob4]


def base_feats(r):
    sh = r['pos'][0].astype(float)
    ang = abs(math.atan2(sh[1], HOOP[0] - sh[0]))
    st = (r['subtype'] or '').lower()
    return [r['dist'], ang, r['three'], r['blocked'], 'layup' in st, 'dunk' in st, 'hook' in st, 'tip' in st,
            r['flight_ms'] / 1000]


def main():
    R = pickle.load(open(OUT / 'rebounds.pkl', 'rb'))
    R = [r for r in R if r['dist'] is not None]
    G = np.array([r['game_id'] for r in R]); y = np.array([r['oreb'] for r in R])
    games = np.array(sorted(set(G))); fold_of = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(games))}
    F = np.array([fold_of[g] for g in G])
    XB = np.array([base_feats(r) for r in R], float)
    XG = np.array([geometry(r) for r in R], float)
    XC = np.zeros((len(R), 4))
    for f in range(5):
        bank = landing_bank([r for r, ff in zip(R, F) if ff != f])
        for i in np.where(F == f)[0]:
            XC[i] = control_feats(R[i], landing_weights(R[i], bank))
        print('fold', f, 'control done', flush=True)
    np.savez(OUT / 'features.npz', XB=XB, XG=XG, XC=XC, y=y, F=F)
    sets = {'base': XB, '+geometry': np.hstack([XB, XG]), '+voronoi': np.hstack([XB, XC[:, :1]]),
            '+kinematic': np.hstack([XB, XC[:, 1:2]]), '+prob': np.hstack([XB, XC[:, 2:4]]),
            '+geometry+prob': np.hstack([XB, XG, XC[:, 2:4]]), '+all': np.hstack([XB, XG, XC])}
    res = {}; P = {}
    print(f'misses {len(R)} games {len(games)} oreb rate {y.mean():.3f}')
    for mname in ('linear', 'gbm'):
        for sname, X in sets.items():
            p = np.zeros(len(y))
            for f in range(5):
                tr, te = F != f, F == f
                mdl = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if mname == 'linear'
                       else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50))
                p[te] = mdl.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            P[(mname, sname)] = p
            res[f'{mname}|{sname}'] = dict(auc=float(roc_auc_score(y, p)), ll=float(log_loss(y, p)))
            print(f'{mname:6s} {sname:16s} AUC {res[f"{mname}|{sname}"]["auc"]:.4f} ll {res[f"{mname}|{sname}"]["ll"]:.4f}', flush=True)
    # value of a miss
    pa = np.array([r.get('pts_after', np.nan) for r in R], float)
    m_o = (y == 1) & np.isfinite(pa); m_d = (y == 0) & np.isfinite(pa)
    cols = ['att_rim12', 'def_rim12', 'boxed', 'off_crash', 'def_crash', 'att_back', 'def_back', 'att_min_rim', 'def_min_rim']
    for lab, m in (('second-chance pts | OREB', m_o), ('opp pts | DREB', m_d)):
        lr = LinearRegression().fit(np.hstack([XB[m][:, :1], XG[m]]), pa[m])
        print(f'{lab}: mean {pa[m].mean():.3f}; per unit', dict(zip(['dist'] + cols, lr.coef_.round(3))))
        res[f'value|{lab}'] = dict(mean=float(pa[m].mean()), coef=dict(zip(['dist'] + cols, lr.coef_.round(4).tolist())))
    by_back = {int(k): float(pa[m_d][XG[m_d][:, 5] == k].mean()) for k in range(0, 6) if (XG[m_d][:, 5] == k).sum() > 100}
    print('opp pts after DREB by # attackers back at release:', {k: round(v, 3) for k, v in by_back.items()})
    res['opp_pts_by_back'] = by_back
    p = P[('gbm', '+all')]
    eo = np.full(len(R), pa[m_o].mean()); ed = np.full(len(R), pa[m_d].mean())
    vmiss = p * eo - (1 - p) * ed
    print(f'value of a miss: mean {vmiss.mean():.3f}, sd {vmiss.std():.3f}, 10-90% {np.percentile(vmiss, [10, 90]).round(3)}')
    json.dump(res, open(OUT / 'rebound_model.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
