"""POP part 1: are there skill-level lineup synergies that DUNCAN's additive ratings miss?

Pair-level teammate effects were noise (pair_validate.py, cross-season r -0.06). This tests the low-dimensional
version (Maymin, Maymin & Shen 2013, "NBA chemistry"): synergy as a function of the SKILLS on the floor, which seven
seasons of stints can identify.

Roles (point-in-time at each cutoff, recency-decayed with DUNCAN's half-life, shrunk to league):
  usage, 3PA, rim FGA (<= 4 ft)   per 100 on-court possessions, from the stints
  ast, oreb, dreb, blk, stl       per 36 minutes, from the game logs
z-scored (minutes-weighted). For a five-man unit with role matrix Z (5 x K):
  F_kl = S_k S_l - Q_kl  (k <= l),  S = column sums, Q = Z'Z  ->  sum over teammate PAIRS of z_ik z_jl
These are pure interactions (main effects stay in the player ratings): 36 offense-unit terms (points scored) and 36
defense-unit terms (points allowed). Arm 'cross' adds offense x defense unit products S_off_k S_def_l (64 terms).
Fit: one ridge with DUNCAN-2's player terms (lambda 3000, SPM prior, half-strength luck target) plus the synergy
columns at their own penalty lam_s (columns standardised).

Test: rolling blocks (every 3rd ms_bench block): fit on everything before b0, predict the 30 days from b0.
lam_s chosen on the 2020-21 and 2021-22 blocks; 2022-23 onward are scored with that choice fixed.
Metric: possession-weighted SSE of points per 100 on test stint rows, vs the additive base, all rows and rows with a
player who changed team in the last 120 days. 95% CI by resampling games.
Output: results/pop/synergy.json (+ per-block coefficient vectors in synergy_coef.npz)     python pop_synergy.py (PC)
"""
import json, sys, time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

import ms_spm as sp
from ms_fix import FData
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
DST = ROOT / 'results/pop'
LAM = 3000.0
LAM_S = [3e4, 3e5, 1e6, 3e6]
ROLES = ['usage', 'fg3a', 'rim', 'ast', 'oreb', 'dreb', 'blk', 'stl']
K_POSS, K_MIN = 300.0, 250.0
TUNE_SEASONS = {2021, 2022}


class PData(FData):
    """FData + per-player-day role events from the stints (on-court possessions, usage, 3PA, rim FGA)."""

    def __init__(self):
        super().__init__()
        import pickle
        ev = defaultdict(lambda: np.zeros(4))
        for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl', 'ms2_stints_2627.pkl'):
            d = pickle.load(open(OUT / f, 'rb'))
            for s in d['stints']:
                dy = sp.dnum(d['games'][s['game_id']]['date'])
                for k in (0, 1):
                    pp = s['poss_team'][k]
                    if pp <= 0:
                        continue
                    rim = defaultdict(int)
                    for sh in s.get('shots', ([], []))[k]:
                        if sh[1] <= 4 and not sh[2]:
                            rim[sh[0]] += 1
                    for p in s['lineup'][k]:
                        if p not in self.pix:
                            continue
                        u = s['use'].get(p, (0, 0, 0)); sh3 = s['shoot'].get(p, (0, 0, 0, 0))
                        ev[(self.pix[p], dy)] += [pp, u[0] + 0.44 * u[1] + u[2], sh3[0], rim[p]]
        keys = np.array(list(ev.keys())); vals = np.array(list(ev.values()))
        self.RE_p, self.RE_day, self.RE = keys[:, 0], keys[:, 1], vals
        print('role events', len(vals), flush=True)

    def roles(self, cut):
        n = self.n; Z = np.zeros((n, len(ROLES)))
        m = self.RE_day < cut; w = 0.5 ** ((cut - self.RE_day[m]) / sp.HL)
        c = [np.bincount(self.RE_p[m], w * self.RE[m, j], n) for j in range(4)]
        lg = [c[j].sum() / c[0].sum() for j in (1, 2, 3)]
        for j in range(3):
            Z[:, j] = 100 * (c[j + 1] + K_POSS * lg[j]) / (c[0] + K_POSS)
        g = self.gl_day < cut; wg = 0.5 ** ((cut - self.gl_day[g]) / sp.HL)
        mins = np.bincount(self.gl_p[g], wg * self.gl_min[g], n)
        for j, col in enumerate(['ast', 'oreb', 'dreb', 'blk', 'stl']):
            tot = np.bincount(self.gl_p[g], wg * self.gl[col][g], n); lgr = tot.sum() / mins.sum()
            Z[:, 3 + j] = 36 * (tot + K_MIN * lgr) / (mins + K_MIN)
        ok = mins > 200; wt = np.minimum(mins[ok], 3000)
        mu = np.average(Z[ok], 0, wt); sd = np.sqrt(np.average((Z[ok] - mu) ** 2, 0, wt)) + 1e-9
        return (Z - mu) / sd


IU = np.triu_indices(len(ROLES))


def unit_feats(Zr, U):
    Z = Zr[U]                                    # rows x 5 x K
    S = Z.sum(1); Q = np.einsum('rik,ril->rkl', Z, Z)
    F = S[:, :, None] * S[:, None, :] - Q
    return F[:, IU[0], IU[1]], S


def feats(Zr, Dt, rows, cross):
    Fo, So = unit_feats(Zr, Dt.O[rows]); Fd, Sd = unit_feats(Zr, Dt.D[rows])
    F = [Fo, Fd]
    if cross:
        F.append((So[:, :, None] * Sd[:, None, :]).reshape(len(rows), -1))
    return np.concatenate(F, 1)


def fit(Dt, rows, y, m, cut, F=None, lam_s=None):
    w = Dt.W[rows] * 0.5 ** ((cut - Dt.DAY[rows]) / sp.HL); sw = np.sqrt(w)
    X = Dt.X[rows]; r = y[rows] - X @ m
    A = csr_matrix(X.multiply(sw[:, None]))
    C = np.column_stack([np.ones(len(rows)), Dt.H[rows]]) * sp.SC * sw[:, None]
    n2 = 2 * Dt.n; p = 0 if F is None else F.shape[1]
    # linear operator [A | s*sw*F | C]: no dense-as-sparse copy of F (memory)
    from scipy.sparse.linalg import LinearOperator
    fs = (np.sqrt(LAM / lam_s) * sw).astype(np.float32) if p else None
    def mv(v):
        out = A @ v[:n2] + C @ v[n2 + p:]
        if p:
            out += fs * (F @ v[n2:n2 + p].astype(np.float32))
        return out
    def rmv(u):
        parts = [A.T @ u]
        if p:
            parts.append((F.T @ (fs * u).astype(np.float32)).astype(float))
        parts.append(C.T @ u)
        return np.concatenate(parts)
    op = LinearOperator((len(rows), n2 + p + 2), matvec=mv, rmatvec=rmv, dtype=float)
    sol = lsqr(op, sw * r, damp=np.sqrt(LAM), atol=1e-8, btol=1e-8, iter_lim=4000)[0]
    g = sol[n2:n2 + p] * (np.sqrt(LAM / lam_s) if p else 1)
    return m + sol[:n2], g, sol[n2 + p:] * sp.SC


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    Dt = PData(); n = Dt.n
    blocks = sorted({f.stem.split('_')[2]: int(f.stem.split('_')[1]) for f in OUT.glob('msb_*.npz')}.items())
    blocks = blocks[::every]
    gid_row = Dt.GI
    arms = ['base'] + [f'syn_{l:g}' for l in LAM_S] + [f'cross_{l:g}' for l in LAM_S]
    per = []          # (block, season, game index, mover flag, w, sq errors per arm)
    coefs = {}
    DST.mkdir(parents=True, exist_ok=True)
    for b0, season in blocks:
        t0 = time.time(); cut = sp.dnum(b0)
        tr = np.flatnonzero(Dt.DAY < cut); te = np.flatnonzero((Dt.DAY >= cut) & (Dt.DAY < cut + 30))
        if len(te) < 1000:
            continue
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        Zr = Dt.roles(cut)
        # movers: players whose most-possession team in the 120 days before cut differs from the team they play for now
        rec = np.flatnonzero((Dt.DAY < cut) & (Dt.DAY >= cut - 120)); cnt = defaultdict(float)
        for i in rec:
            for p in Dt.O[i]:
                cnt[(p, Dt.off_team[i])] += Dt.W[i]
        last = {}
        for (p, t), w_ in cnt.items():
            if w_ > last.get(p, (0, 0))[1]:
                last[p] = (t, w_)
        mv = np.array([any(p in last and last[p][0] != Dt.off_team[i] for p in Dt.O[i]) or
                       any(p in last and last[p][0] != Dt.def_team[i] for p in Dt.D[i]) for i in te])
        yt, wt, H = Dt.Y[te], Dt.W[te], Dt.H[te]
        pred = {}
        th, _, b = fit(Dt, tr, y, m, cut)
        pred['base'] = b[0] + b[1] * H + th[Dt.O[te]].sum(1) + th[n + Dt.D[te]].sum(1)
        for cross in (False, True):
            Ftr = feats(Zr, Dt, tr, cross); Fte = feats(Zr, Dt, te, cross)
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
            Ftr = ((Ftr - mu) / sd).astype(np.float32); Fte = (Fte - mu) / sd
            for l in LAM_S:
                th, g, b = fit(Dt, tr, y, m, cut, Ftr, l)
                k = f"{'cross' if cross else 'syn'}_{l:g}"
                pred[k] = b[0] + b[1] * H + th[Dt.O[te]].sum(1) + th[n + Dt.D[te]].sum(1) + Fte @ g
                coefs[f'{b0}_{k}'] = g / sd
            del Ftr, Fte
        E = np.column_stack([(yt - pred[a]) ** 2 for a in arms])
        per.append(dict(b0=b0, season=season, gi=gid_row[te], mv=mv, w=wt, E=E))
        d = {a: float(np.sum(wt * (E[:, 0] - E[:, j]))) / 1e6 for j, a in enumerate(arms)}
        print(b0, season, f'{len(te)} rows, {mv.mean():.2f} movers;', 'SSE gain vs base (x1e6):',
              {a: round(v, 2) for a, v in d.items() if a != 'base'}, f'{time.time() - t0:.0f}s', flush=True)
        np.savez(DST / 'synergy_coef.npz', **coefs)

    def pooled(sel_blocks, mask_fn=None):
        P = [p for p in per if p['season'] in sel_blocks]
        if not P:
            return None
        gi = np.concatenate([p['gi'] for p in P]); w = np.concatenate([p['w'] for p in P])
        E = np.concatenate([p['E'] for p in P]); mk = np.concatenate([mask_fn(p) for p in P]) if mask_fn else np.ones(len(w), bool)
        gi, w, E = gi[mk], w[mk], E[mk]
        ug, inv = np.unique(gi, return_inverse=True)
        G = np.stack([np.bincount(inv, w * E[:, j], len(ug)) for j in range(E.shape[1])], 1)   # per game SSE
        tot = G.sum(0); rng = np.random.default_rng(0)
        bs = np.array([G[rng.integers(0, len(ug), len(ug))].sum(0) for _ in range(2000)])
        out = {}
        for j, a in enumerate(arms):
            if a == 'base':
                continue
            rel = (bs[:, 0] - bs[:, j]) / bs[:, 0]
            out[a] = dict(sse_gain_pct=round(100 * float((tot[0] - tot[j]) / tot[0]), 4),
                          ci_pct=np.round(100 * np.percentile(rel, [2.5, 97.5]), 4).tolist())
        out['games'] = int(len(ug))
        return out

    seasons = sorted({p['season'] for p in per})
    tune = pooled(TUNE_SEASONS)
    best = {c: max((a for a in arms if a.startswith(c)), key=lambda a: tune[a]['sse_gain_pct']) for c in ('syn', 'cross')}
    test_s = set(seasons) - TUNE_SEASONS
    res = dict(blocks=[p['b0'] for p in per], roles=ROLES, lam_s=LAM_S, tune=tune, chosen=best,
               test_all=pooled(test_s), test_movers=pooled(test_s, lambda p: p['mv']),
               test_by_season={int(s): pooled({s}) for s in sorted(test_s)})
    json.dump(res, open(DST / 'synergy.json', 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('tune', 'chosen', 'test_all', 'test_movers')}, indent=1))


if __name__ == '__main__':
    main()
