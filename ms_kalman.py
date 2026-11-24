"""Time-varying player ratings: a full-covariance Kalman filter over every player's offense and defense, the
structure DARKO uses, with our box prior and luck-adjusted targets inside it.

State: delta (2n) = each player's offense / defense deviation from the box prior m, plus league intercept and home
edge. Each stint-side row observes
    y - sum m(off five, o) - sum m(def five, d) = b + b_h home + sum delta_o(off) + sum delta_d(def) + noise,
noise variance SIGMA2 / possessions. Between game days delta follows an Ornstein-Uhlenbeck process toward 0
(i.e. the rating reverts toward the box prior): delta <- rho delta, P <- rho^2 P + (1 - rho^2) tau^2 I,
rho = exp(-days / T). The intercept random-walks slowly. One game = one batch update of all its rows.
Prior mean m and the luck-adjusted target are point-in-time: both come from the latest "grid" cutoff before the
game (grid = every block start, every validation start and every ~30 days in season), using data before it.

Ratings are snapshotted (theta = m + delta) at every block's validation start and block start; per block the
(tau, T) pair is chosen on the validation window exactly as lambda is for the ridge arms, then logistic calibration,
then the block is predicted from its realised stints (same games / calibration protocol as darko_cal).
Arms: kal (half-luck target, box prior), kal_bio (+ draft / physical prior), kal_sq (+ shot-quality target).
Also saves, for every row, the one-step-ahead prediction (made before that game's update) of the best kal_sq
setting, for the lineup-fit study (ms_fit.py).
Outputs: adds keys to results/latent/msb_*.npz; results/latent/kal_snap.pkl; results/latent/kal_onestep.npz.
  python ms_kalman.py      (PC)
"""
import pickle, sys, time
from datetime import date, timedelta
import numpy as np
from scipy.linalg.blas import sgemm
from sklearn.linear_model import LogisticRegression

import ms_spm as sp
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
SIGMA2 = 115.0 ** 2       # per-possession points variance x 100^2 (Y is points per 100)
TAUS = [1.5, 2.25, 3.0]   # prior sd of a player's deviation from the box prior, points per 100
TS = [120.0, 365.0, 1000.0]  # OU timescale, days
B_DRIFT = 0.02            # intercept random-walk variance per day
ARMS = {'kal': dict(target='half', bio=False), 'kal_bio': dict(target='half', bio=True),
        'kal_sq': dict(target='sq', bio=True)}


def blocks():
    out = []
    for f in sorted(OUT.glob('msb_*.npz')):
        season, b0 = int(f.stem.split('_')[1]), f.stem.split('_')[2]
        vcut = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS))
        out.append((f, season, b0, vcut))
    return out


def grid_cuts(Dt, B):
    days = np.unique(Dt.DAY)
    fixed = {sp.dnum(b0) for _, _, b0, _ in B} | {sp.dnum(v) for _, _, _, v in B}
    g, last = set(fixed), days[0] + 45
    for d in days:
        if d >= last + 30:
            g.add(int(d)); last = d
    g = sorted(c for c in g if c > days[0] + 40)
    keep = []
    for c in g:  # drop non-fixed cuts within 10 days of a fixed one
        if c in fixed or all(abs(c - f) > 10 for f in fixed):
            keep.append(c)
    return keep


class Priors:
    """Point-in-time targets and box priors at each grid cut (cached on disk: one SPM fit per cut)."""

    def __init__(self, Dt, cuts, target, bio):
        self.Dt, self.cuts = Dt, cuts
        f = OUT / f'kal_prior_{target}_{int(bio)}.pkl'
        self.m = pickle.load(open(f, 'rb')) if f.exists() else {}
        t0 = time.time()
        for c in cuts:
            if c not in self.m:
                y = self.target(c, target)
                self.m[c] = (Dt.spm_prior(c, y, bio=bio).astype(np.float32), None)
                pickle.dump(self.m, open(f, 'wb'))
                print(f'  prior {target} bio={bio} {date.fromordinal(c)} ({time.time() - t0:.0f}s)', flush=True)
        self.kind = target

    def target(self, c, kind=None):
        kind = kind or self.kind
        return self.Dt.luck_target(c, 0.5) if kind == 'half' else self.Dt.shotq_target(c, 0.5)


def run(Dt, pri, tau, T, snap_days, keep_onestep=False):
    """One forward pass. Returns {snap_day: (theta, b)} and optionally the one-step-ahead prediction per row."""
    n2 = 2 * Dt.n; N = n2 + 2
    x = np.zeros(N); x[n2] = 108.0
    P = np.asfortranarray(np.eye(N, dtype=np.float32) * tau ** 2); P[n2, n2] = 25.0; P[n2 + 1, n2 + 1] = 4.0
    cuts = pri.cuts; ci = -1; m = np.zeros(n2); y = Dt.Y
    snaps, todo = {}, sorted(snap_days)
    onestep = np.full(len(Dt.Y), np.nan) if keep_onestep else None
    gdays = np.array([sp.dnum(Dt.G[g]['date']) for g in Dt.gids])
    last = gdays[0]

    def drift(to):
        nonlocal last
        dt = to - last
        if dt > 0:
            rho = np.exp(-dt / T)
            x[:n2] *= rho
            P[:n2, :n2] *= rho * rho
            P[:n2, n2:] *= rho; P[n2:, :n2] *= rho
            P[np.arange(n2), np.arange(n2)] += (1 - rho * rho) * tau ** 2
            P[n2, n2] += B_DRIFT * dt
            last = to

    for gi in range(len(Dt.gids)):
        d = gdays[gi]
        while todo and todo[0] <= d:  # snapshot before this day's games
            drift(todo[0])
            while ci + 1 < len(cuts) and cuts[ci + 1] <= todo[0]:
                ci += 1; m = pri.m[cuts[ci]][0].astype(float); y = pri.target(cuts[ci])
            snaps[todo.pop(0)] = (m + x[:n2], x[n2:].copy())
        while ci + 1 < len(cuts) and cuts[ci + 1] <= d:
            ci += 1; m = pri.m[cuts[ci]][0].astype(float); y = pri.target(cuts[ci])
        r = Dt.rows_of.get(gi)
        if not r:
            continue
        drift(d)
        r = np.asarray(r)
        Oc, Dc = Dt.O[r], Dt.n + Dt.D[r]
        idx = np.unique(np.concatenate([Oc.ravel(), Dc.ravel(), [n2, n2 + 1]]))
        pos = {j: k for k, j in enumerate(idx)}
        H = np.zeros((len(r), len(idx)))
        for i in range(len(r)):
            for j in Oc[i]:
                H[i, pos[j]] += 1
            for j in Dc[i]:
                H[i, pos[j]] += 1
            H[i, pos[n2]] = 1; H[i, pos[n2 + 1]] = Dt.H[r[i]]
        z = y[r] - m[Oc].sum(1) - m[Dc].sum(1)
        pred = H @ x[idx]
        if onestep is not None:
            onestep[r] = pred + m[Oc].sum(1) + m[Dc].sum(1)
        PHt = P[:, idx].astype(float) @ H.T         # N x k
        S = H @ PHt[idx] + np.diag(SIGMA2 / Dt.W[r])
        K = PHt @ np.linalg.inv(S)                  # N x k (S is small: rows of one game)
        x += K @ (z - pred)
        sgemm(-1.0, np.asfortranarray(K, np.float32), np.asfortranarray(PHt, np.float32), beta=1.0, c=P,
              trans_b=True, overwrite_c=True)       # P -= K PHt^T in place
    while todo:
        drift(todo[0]); snaps[todo.pop(0)] = (m + x[:n2], x[n2:].copy())
    return snaps, onestep


def main():
    Dt = sp.Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    B = blocks(); cuts = grid_cuts(Dt, B)
    snap_days = sorted({sp.dnum(b0) for _, _, b0, _ in B} | {sp.dnum(v) for _, _, _, v in B})
    print(len(B), 'blocks', len(cuts), 'grid cuts', flush=True)
    sf = OUT / 'kal_snap.pkl'
    allsnap = pickle.load(open(sf, 'rb')) if sf.exists() else {}
    for arm, cfg in ARMS.items():
        pri = Priors(Dt, cuts, cfg['target'], cfg['bio'])
        res = {}
        for tau in TAUS:
            for T in TS:
                t0 = time.time()
                keep = arm == 'kal_sq'
                snaps, one = run(Dt, pri, tau, T, snap_days, keep_onestep=keep)
                res[(tau, T)] = snaps
                if keep:
                    np.save(OUT / f'kal_onestep_{tau}_{int(T)}.npy', one)
                print(f'{arm} tau {tau} T {T}: {time.time() - t0:.0f}s', flush=True)
        allsnap[arm] = res; pickle.dump(allsnap, open(sf, 'wb'))
        for f, season, b0, vcut in B:
            P = dict(np.load(f)); test = list(P['gid'])
            val = [g for g in Dt.gids if vcut <= G[g]['date'] < b0]
            best = None
            for hp, snaps in res.items():
                th, b = snaps[sp.dnum(vcut)]
                xv = Dt.margins(th, np.array([b[0], b[1]]), [gix[g] for g in val])
                cal = LogisticRegression(C=1e4).fit(xv[:, None], win(val))
                l = -np.mean(np.log(np.clip(cal.predict_proba(xv[:, None])[np.arange(len(val)), win(val)], 1e-9, 1)))
                if best is None or l < best[0]:
                    best = (l, hp, cal)
            _, hp, cal = best
            th, b = res[hp][sp.dnum(b0)]
            P[arm] = cal.predict_proba(Dt.margins(th, np.array([b[0], b[1]]), [gix[g] for g in test])[:, None])[:, 1]
            P[arm + '_hp'] = np.array(hp)
            np.savez(f, **P)
        print(arm, 'scored', flush=True)


if __name__ == '__main__':
    main()
