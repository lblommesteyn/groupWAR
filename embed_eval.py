"""Do tracking player descriptions close the gap to RAPM? (GroupWAR embedding test)

Windows:  A = before Mar 1,  B = March,  T = Apr 1 onward (late regular season, play-in, playoffs).
Protocol (no description ever sees the games it is scored on):
  1. descriptions from A; learn description->impact weights on B stints; calibrate win prob on B games
  2. descriptions (and RAPM) from A+B; predict T games with the same weights / calibration
Rows are stint-sides: offense points per 100 on its possessions, weighted by possessions.
Game prediction sums stint predictions over the actual test-game lineups and possessions.
"""
import json, pickle, sys
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack as sphstack
from scipy.sparse.linalg import lsqr
from sklearn.metrics import roc_auc_score, log_loss
from masked_paths import ROOT

E = ROOT / 'results/embed'
A_END, B_END = '2026-03-01', '2026-04-01'
LAM_RAPM = 1000.0

BOX = ['mpg', 'gp_share', 'usage', 'ts', 'pts36', 'rim_share', 'smid_share', 'lmid_share', 'c3_share', 'a3_share',
       'rim_acc', 'fg3_acc', 'fta_rate', 'ast36', 'ast3_share', 'astrim_share', 'oreb36', 'dreb36', 'stl36',
       'blk36', 'tov36', 'pf36', 'height', 'weight', 'isG', 'isF', 'isC']
TRK_OFF = ['shot_making', 'shot_quality', 'own_d1', 'own_touch', 'gravity', 'off_speed', 'onball_share', 'spacing']
TRK_DEF = ['nd_rate', 'contest_d1', 'def_xsupp', 'def_fg', 'rim_prot', 'rim_presence', 'dmatch', 'paint_share',
           'def_speed', 'onball_def', 'matchup_usage']


def inches(h):
    try:
        a, b = str(h).split('-'); return int(a) * 12 + int(b)
    except Exception:
        return np.nan


def describe(G, gids, shots):
    """Per-player feature dict from games gids."""
    gids = set(gids)
    S = defaultdict(lambda: defaultdict(float))
    prof, gp, team_games = {}, defaultdict(int), defaultdict(int)
    pairs = defaultdict(float)
    lg_cnt, lg_dsum = np.zeros(11), np.zeros(11)
    for g in gids:
        d = G[g]
        for side in ('home', 'away'):
            team_games[d[side]] += 1
        for pid, b in d['box'].items():
            if b['sec'] > 0:
                gp[pid] += 1
                for k, v in b.items():
                    S[pid][k] += float(v)
        prof.update(d['profile'])
        for st in d['stints']:  # on-court possessions for usage / rates
            for k, side in enumerate(st['lineup']):
                for p in side:
                    S[p]['own_poss'] += st['poss_team'][k]; S[p]['def_poss'] += st['poss_team'][1 - k]
        for pid, v in d['trk_off'].items():
            for k in ('n', 'onball', 'speed', 'nspeed', 'ball_dist'):
                S[pid]['o_' + k] += v[k]
            S[pid].setdefault('cnt', np.zeros(11)); S[pid].setdefault('dsum', np.zeros(11))
            S[pid]['cnt'] = S[pid]['cnt'] + v['cnt']; S[pid]['dsum'] = S[pid]['dsum'] + v['dsum']
            lg_cnt += v['cnt']; lg_dsum += v['dsum']
        for pid, v in d['trk_def'].items():
            for k in v:
                S[pid]['d_' + k] += v[k]
        for (q, o), c in d['pairs'].items():
            pairs[(q, o)] += c
    for s in shots:
        if s['game_id'] not in gids:
            continue
        p = S[s['shooter']]
        p['sa_n'] += 1; p['sa_res'] += s['made'] - s['xa']
        if s['trk_ok']:
            p['sb_n'] += 1; p['sb_res'] += s['made'] - s['xb']; p['sb_x'] += s['xb']
            p['sb_d1'] += s['d1']; p['sb_touch'] += s['touch_s']
            q = S[s['nd_id']]
            q['nd_n'] += 1; q['nd_d1'] += s['d1']; q['nd_supp'] += s['xb'] - s['xa']; q['nd_res'] += s['made'] - s['xa']
            if s['dist'] < 6:
                for r in s['rim_def_ids']:
                    S[r]['rim_n'] += 1; S[r]['rim_res'] += s['made'] - s['xa']
    lg_bin = lg_dsum / np.maximum(lg_cnt, 1)
    tg = np.median(list(team_games.values())) if team_games else 1
    allsb = [s for s in shots if s['game_id'] in gids and s['trk_ok']]
    lg_xb = np.mean([s['xb'] for s in allsb]); lg_d1 = np.mean([s['d1'] for s in allsb])
    lg_touch = np.mean([s['touch_s'] for s in allsb])
    out = {}
    for pid, s in S.items():
        if s['sec'] < 60:
            continue
        m = max(s['sec'] / 60, 1); fga = max(s['fga'], 1); ast = max(s['ast'], 1)
        pr = prof.get(pid, {}); pos = pr.get('pos') or ''
        f = dict(mpg=m / max(gp[pid], 1), gp_share=gp[pid] / tg,
                 usage=(s['fga'] + 0.44 * s['fta'] + s['tov']) / max(s['own_poss'], 1),
                 ts=s['pts'] / max(2 * (s['fga'] + 0.44 * s['fta']), 1), pts36=36 * s['pts'] / m,
                 rim_share=s['rim_a'] / fga, smid_share=s['smid_a'] / fga, lmid_share=s['lmid_a'] / fga,
                 c3_share=s['c3_a'] / fga, a3_share=s['a3_a'] / fga,
                 rim_acc=(s['rim_m'] + 50 * 0.66) / (s['rim_a'] + 50), fg3_acc=(s['tpm'] + 50 * 0.36) / (s['tpa'] + 50),
                 fta_rate=s['fta'] / fga, ast36=36 * s['ast'] / m, ast3_share=s['ast3'] / ast,
                 astrim_share=s['astrim'] / ast, oreb36=36 * s['oreb'] / m, dreb36=36 * s['dreb'] / m,
                 stl36=36 * s['stl'] / m, blk36=36 * s['blk'] / m, tov36=36 * s['tov'] / m, pf36=36 * s['pf'] / m,
                 height=inches(pr.get('ht')), weight=float(pr['wt']) if str(pr.get('wt') or '').isdigit() else np.nan,
                 isG=float('G' in pos), isF=float('F' in pos), isC=float('C' in pos))
        cnt = s.get('cnt', np.zeros(11)); dsum = s.get('dsum', np.zeros(11))
        f.update(shot_making=s['sb_res'] / (s['sb_n'] + 50),
                 shot_quality=(s['sb_x'] + 30 * lg_xb) / (s['sb_n'] + 30),
                 own_d1=(s['sb_d1'] + 30 * lg_d1) / (s['sb_n'] + 30),
                 own_touch=(s['sb_touch'] + 30 * lg_touch) / (s['sb_n'] + 30),
                 gravity=float((dsum - cnt * lg_bin).sum() / (cnt.sum() + 200)),
                 off_speed=s['o_speed'] / max(s['o_nspeed'], 1),
                 onball_share=s['o_onball'] / max(s['o_n'] + s['o_onball'], 1),
                 spacing=s['o_ball_dist'] / max(s['o_n'], 1),
                 nd_rate=s['nd_n'] / max(s['def_poss'], 1),
                 contest_d1=(s['nd_d1'] + 30 * lg_d1) / (s['nd_n'] + 30),
                 def_xsupp=s['nd_supp'] / (s['nd_n'] + 30), def_fg=s['nd_res'] / (s['nd_n'] + 100),
                 rim_prot=s['rim_res'] / (s['rim_n'] + 50), rim_presence=s['rim_n'] / max(s['def_poss'], 1),
                 dmatch=s['d_dmatch'] / max(s['d_n'], 1), paint_share=s['d_paint'] / max(s['d_n'], 1),
                 def_speed=s['d_speed'] / max(s['d_nspeed'], 1), onball_def=s['d_onball'] / max(s['d_n'], 1),
                 minutes=m, has_trk=float(s['o_n'] + s['d_n'] > 50))
        out[pid] = f
    # matchup difficulty: usage of the players this defender was matched to
    num, den = defaultdict(float), defaultdict(float)
    for (q, o), c in pairs.items():
        if o in out:
            num[q] += c * out[o]['usage']; den[q] += c
    for pid, f in out.items():
        f['matchup_usage'] = num[pid] / den[pid] if den[pid] else np.nan
    return out


def matrix(desc, cols, pids):
    X = np.array([[desc.get(p, {}).get(c, np.nan) for c in cols] for p in pids], float)
    w = np.array([desc.get(p, {}).get('minutes', 0) for p in pids])
    mu = np.array([np.nansum(X[:, j] * w) / w[np.isfinite(X[:, j])].sum() for j in range(len(cols))])
    sd = np.array([np.sqrt(np.nansum((X[:, j] - mu[j]) ** 2 * w) / w[np.isfinite(X[:, j])].sum()) for j in range(len(cols))])
    Z = (X - mu) / np.where(sd > 0, sd, 1)
    return np.nan_to_num(Z)  # unseen players / missing values = league average


def rows(G, gids):
    R = []
    for g in gids:
        d = G[g]
        for st in d['stints']:
            h, a = st['lineup']
            for k, (o, de) in enumerate(((h, a), (a, h))):
                if st['poss_team'][k] > 0:
                    R.append((g, o, de, 1 - k, 100 * st['pts'][k] / st['poss_team'][k], st['poss_team'][k]))
    return R


class Lin:
    """y = mu + h*home + Xo·wo + Xd·wd  (ridge on w), optional player-indicator ridge on the residual."""
    def __init__(self, lam_w=None, lam_p=None):
        self.lam_w, self.lam_p = lam_w, lam_p

    def design(self, R, Z, pidx):
        if Z is None:
            return np.zeros((len(R), 0))
        F = Z.shape[1]
        X = np.zeros((len(R), 2 * F))
        for i, (_, o, d, *_r) in enumerate(R):
            X[i, :F] = Z[[pidx[p] for p in o]].sum(0); X[i, F:] = Z[[pidx[p] for p in d]].sum(0)
        return X

    def ind(self, R, pidx):
        n = len(pidx); r, c, v = [], [], []
        for i, (_, o, d, *_r) in enumerate(R):
            for p in o:
                r.append(i); c.append(pidx[p]); v.append(1.0)
            for p in d:
                r.append(i); c.append(n + pidx[p]); v.append(1.0)
        return csr_matrix((v, (r, c)), shape=(len(R), 2 * n))

    def fit(self, R, Z, pidx, lam_w):
        y = np.array([r[4] for r in R]); w = np.array([r[5] for r in R]); hm = np.array([r[3] for r in R], float)
        X = np.column_stack([np.ones(len(R)), hm, self.design(R, Z, pidx)])
        P = np.diag([0, 0] + [lam_w] * (X.shape[1] - 2))
        self.beta = np.linalg.solve(X.T @ (X * w[:, None]) + P, X.T @ (w * y))
        return self

    def predict(self, R, Z, pidx):
        hm = np.array([r[3] for r in R], float)
        X = np.column_stack([np.ones(len(R)), hm, self.design(R, Z, pidx)])
        return X @ self.beta


def rapm_fit(R, pidx, prior=None, lam=LAM_RAPM):
    """Ridge O/D ratings on residual y - prior; intercept and home unpenalized via centering."""
    y = np.array([r[4] for r in R]); w = np.array([r[5] for r in R]); hm = np.array([r[3] for r in R], float)
    if prior is not None:
        y = y - prior
    M = Lin().ind(R, pidx)
    C = np.column_stack([np.ones(len(R)), hm])
    b0 = np.linalg.lstsq(C * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    res = y - C @ b0
    sw = np.sqrt(w)
    from scipy.sparse import diags
    A = diags(sw) @ M
    theta = lsqr(A, res * sw, damp=np.sqrt(lam), atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    return b0, theta


def rapm_pred(R, pidx, b0, theta):
    hm = np.array([r[3] for r in R], float)
    return b0[0] + b0[1] * hm + Lin().ind(R, pidx) @ theta


def game_margin(R, pred):
    """Predicted home margin per game from stint-side predictions."""
    m = defaultdict(float)
    for (g, o, d, home, y, w), p in zip(R, pred):
        m[g] += (1 if home else -1) * p * w / 100
    return m


def cv_lambda(R, Z, pidx, grid=(10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000)):
    gids = np.array(sorted({r[0] for r in R})); rng = np.random.default_rng(0); fold = dict(zip(gids, rng.integers(0, 5, len(gids))))
    best = None
    for lam in grid:
        err = 0
        for k in range(5):
            tr = [r for r in R if fold[r[0]] != k]; te = [r for r in R if fold[r[0]] == k]
            p = Lin().fit(tr, Z, pidx, lam).predict(te, Z, pidx)
            err += sum(r[5] * (r[4] - q) ** 2 for r, q in zip(te, p))
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


def main(arms=None, describe_fn=None, out_name='embed_eval.json'):
    describe_fn = describe_fn or describe
    G = pickle.load(open(E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    gA = sorted(g for g, d in G.items() if d['date'] < A_END)
    gB = sorted(g for g, d in G.items() if A_END <= d['date'] < B_END)
    gT = sorted(g for g, d in G.items() if d['date'] >= B_END)
    print(f'games A {len(gA)}  B {len(gB)}  T {len(gT)}')
    RA, RB, RT = rows(G, gA), rows(G, gB), rows(G, gT)
    pids = sorted({p for R in (RA, RB, RT) for r in R for p in r[1] + r[2]})
    pidx = {p: i for i, p in enumerate(pids)}
    dA, dAB = describe_fn(G, gA, shots), describe_fn(G, gA + gB, shots)
    win = lambda gs: np.array([G[g]['final'][0] > G[g]['final'][1] for g in gs], float)
    yB, yT = win(gB), win(gT)

    arms = arms or {'box': BOX, 'trk': TRK_OFF + TRK_DEF, 'box+trk_off': BOX + TRK_OFF, 'box+trk_def': BOX + TRK_DEF,
                    'box+trk': BOX + TRK_OFF + TRK_DEF}
    preds, info = {}, {}

    def calibrate(mB):
        x = np.array([mB[g] for g in gB])
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1e6).fit(x[:, None], yB)

    def finish(name, mB, mT):
        cal = calibrate(mB)
        xT = np.array([mT[g] for g in gT])
        preds[name] = dict(margin=xT, prob=cal.predict_proba(xT[:, None])[:, 1])

    # RAPM: A -> B for calibration, A+B -> T
    b0, th = rapm_fit(RA, pidx); mB = game_margin(RB, rapm_pred(RB, pidx, b0, th))
    b0, th = rapm_fit(RA + RB, pidx); finish('rapm', mB, game_margin(RT, rapm_pred(RT, pidx, b0, th)))
    rapm_theta = th
    for name, cols in arms.items():
        ZA, ZAB = matrix(dA, cols, pids), matrix(dAB, cols, pids)
        lam = cv_lambda(RB, ZA, pidx)
        mdl = Lin().fit(RB, ZA, pidx, lam)
        info[name] = dict(lam=lam, n_feat=len(cols))
        mB = game_margin(RB, mdl.predict(RB, ZA, pidx))
        finish(name, mB, game_margin(RT, mdl.predict(RT, ZAB, pidx)))
        # feature-prior RAPM: prior from descriptions, ratings on the residual
        priorA = mdl.predict(RA, ZA, pidx)
        b0, th = rapm_fit(RA, pidx, priorA)
        mB2 = game_margin(RB, mdl.predict(RB, ZA, pidx) + rapm_pred(RB, pidx, b0, th) - b0[0] - b0[1] * np.array([r[3] for r in RB]))
        priorAB = mdl.predict(RA + RB, ZAB, pidx)
        b0, th = rapm_fit(RA + RB, pidx, priorAB)
        hT = np.array([r[3] for r in RT], float)
        mT2 = game_margin(RT, mdl.predict(RT, ZAB, pidx) + rapm_pred(RT, pidx, b0, th) - b0[0] - b0[1] * hT)
        finish(f'rapm+{name}_prior', mB2, mT2)
        if name in ('box', 'box+trk'):
            F = len(cols); beta = mdl.beta[2:]
            info[name]['weights_off'] = dict(zip(cols, np.round(beta[:F], 3).tolist()))
            info[name]['weights_def'] = dict(zip(cols, np.round(beta[F:], 3).tolist()))

    # metrics + paired game bootstrap vs RAPM
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(gT), (2000, len(gT)))
    ll = lambda y, p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    base = preds['rapm']
    res = dict(n_games=dict(A=len(gA), B=len(gB), T=len(gT)), info=info, models={})
    for k, v in preds.items():
        d_ll = ll(yT, v['prob']) - ll(yT, base['prob'])
        aucs = [roc_auc_score(yT[b], v['margin'][b]) - roc_auc_score(yT[b], base['margin'][b]) for b in B[:500]]
        res['models'][k] = dict(auc=float(roc_auc_score(yT, v['margin'])), log_loss=float(log_loss(yT, v['prob'])),
                                d_ll_vs_rapm=float(d_ll.mean()),
                                d_ll_ci=[float(x) for x in np.percentile([d_ll[b].mean() for b in B], [2.5, 97.5])],
                                d_auc_ci=[float(x) for x in np.percentile(aucs, [2.5, 97.5])])
    res['models']['home_only'] = dict(auc=0.5, log_loss=float(log_loss(yT, np.full(len(yT), yB.mean()))))
    json.dump(res, open(E / out_name, 'w'), indent=1)
    print(f"{'model':28s} {'AUC':>6s} {'logloss':>8s}  Δll vs RAPM [95% CI]        ΔAUC CI")
    for k, v in res['models'].items():
        extra = f"{v['d_ll_vs_rapm']:+.4f} [{v['d_ll_ci'][0]:+.4f},{v['d_ll_ci'][1]:+.4f}]  [{v['d_auc_ci'][0]:+.3f},{v['d_auc_ci'][1]:+.3f}]" if 'd_ll_ci' in v and k != 'rapm' else ''
        print(f"{k:28s} {v['auc']:6.3f} {v['log_loss']:8.4f}  {extra}")
    print(json.dumps({k: v['lam'] for k, v in info.items()}))
    for k in [a for a in ('box+trk',) if a in info]:
        wo, wd = info[k]['weights_off'], info[k]['weights_def']
        print('largest |w| offense:', sorted(wo.items(), key=lambda kv: -abs(kv[1]))[:8])
        print('largest |w| defense:', sorted(wd.items(), key=lambda kv: -abs(kv[1]))[:8])


if __name__ == '__main__':
    main()
