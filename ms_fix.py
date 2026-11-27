"""Fixing context leakage: two candidate repairs for ratings that travel badly to new teams (ms_movers.py: slope of
actual on predicted 0.64 for team-changers, DARKO 1.01).

  team_T   the same ridge plus offense and defense team-season effects (penalty lambda_T, set by rescaling those
           columns: lsqr has one damping term, a column scaled by s gets effective penalty lambda / s^2). A team's
           system / coaching / fit is absorbed by the team term instead of its players. Predictions include the team
           term where it exists (the new team's current-season effect for a deadline move; 0 for a new season).
  shrink_k only for players who changed teams: rating <- prior + k * (rating - prior), prior = the box/minutes prior.
  both     team_T with shrink_k on top.
Tuning: lambda_T and k chosen on the 2021-2023 deadlines only; reported on the held-out moves (2024-2026 deadlines,
2025 and 2026 offseasons). Scored like ms_movers.py (stints with a mover; per-mover on-court net; slope), and on
post-cutoff stints WITHOUT a mover, so a fix that hurts ordinary predictions shows. 95% CIs resample movers.
Output: results/latent/ms_fix.json     python ms_fix.py   (PC)
"""
import json
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

import ms_spm as sp
import ms_stint_eval as se
from ms_movers import EVENTS, MIN_PRE, MIN_POST, team_ids
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
LAM = 1000.0
LAM_T = [300.0, 1000.0, 3000.0]
KS = [0.5, 0.65, 0.8, 1.0]
TUNE = {'2021 deadline', '2022 deadline', '2023 deadline'}


class FData(sp.Data):
    def __init__(self):
        super().__init__(live=True)
        TM = team_ids()
        gid = np.array(self.gids)[self.GI]
        season = np.array([self.G[g]['season'] for g in self.gids])[self.GI]
        h = np.array([TM.get(g, (0, 0))[0] for g in gid]); a = np.array([TM.get(g, (0, 0))[1] for g in gid])
        self.off_team = np.where(self.H == 1, h, a); self.def_team = np.where(self.H == 1, a, h)
        self.season_r = season
        keys = sorted({(t, s) for t, s in zip(self.off_team, season) if t} | {(t, s) for t, s in zip(self.def_team, season) if t})
        self.tix = {k: i for i, k in enumerate(keys)}; self.nt = len(keys)
        self.to = np.array([self.tix.get((t, s), -1) for t, s in zip(self.off_team, season)])
        self.td = np.array([self.tix.get((t, s), -1) for t, s in zip(self.def_team, season)])

    def ridge_team(self, rows, y, m, lam, lam_t, cut):
        w = self.W[rows] * 0.5 ** ((cut - self.DAY[rows]) / sp.HL); sw = np.sqrt(w)
        X = self.X[rows]; r = y[rows] - X @ m
        A = csr_matrix(X.multiply(sw[:, None]))
        s = np.sqrt(lam / lam_t); nt = self.nt
        ri, ci, v = [], [], []
        for j, (o, d) in enumerate(zip(self.to[rows], self.td[rows])):
            if o >= 0:
                ri.append(j); ci.append(o); v.append(s * sw[j])
            if d >= 0:
                ri.append(j); ci.append(nt + d); v.append(s * sw[j])
        T = csr_matrix((v, (ri, ci)), shape=(len(rows), 2 * nt))
        C = np.column_stack([np.ones(len(rows)), self.H[rows]]) * sp.SC * sw[:, None]
        sol = lsqr(hstack([A, T, csr_matrix(C)]).tocsr(), sw * r, damp=np.sqrt(lam), atol=1e-7, btol=1e-7, iter_lim=3000)[0]
        n2 = 2 * self.n
        return m + sol[:n2], sol[n2:n2 + 2 * nt] * s, sol[n2 + 2 * nt:] * sp.SC


def main():
    Dt = FData(); n = Dt.n; Dk = se.load_darko_od()
    season = Dt.season_r
    has = Dt.to >= 0
    rec = defaultdict(list)   # arm -> per-mover records
    other = defaultdict(lambda: [0.0, 0.0])  # arm -> [sse, sst] on non-mover rows
    ev_log = {}
    for name, cut_s, days in EVENTS:
        cut = sp.dnum(cut_s)
        if days:
            post = (Dt.DAY >= cut) & (Dt.DAY < cut + days)
        else:
            s0 = season[np.flatnonzero(Dt.DAY >= cut)[0]]; post = (Dt.DAY >= cut) & (season == s0)
        pre = (Dt.DAY < cut) & (Dt.DAY >= cut - (120 if not days else 300)) & has
        post &= has
        def poss_by(mask):
            c = defaultdict(float)
            for i in np.flatnonzero(mask):
                for p in Dt.O[i]:
                    c[(p, Dt.off_team[i])] += Dt.W[i]
                for p in Dt.D[i]:
                    c[(p, Dt.def_team[i])] += Dt.W[i]
            return c
        a, b = poss_by(pre), poss_by(post)
        last = {}
        for (p, t), w in a.items():
            if w >= MIN_PRE and w > last.get(p, (0, 0))[1]:
                last[p] = (t, w)
        movers = {p for (p, t), w in b.items() if p in last and t != last[p][0] and w >= MIN_POST}
        mv = np.zeros(n, bool); mv[list(movers)] = True
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        trn = np.flatnonzero(Dt.DAY < cut)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            lastd = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > lastd - sp.VAL_DAYS) & (Dt.DAY <= lastd))
        rows = np.flatnonzero(post)
        mrow = np.array([mv[Dt.O[i]].any() or mv[Dt.D[i]].any() for i in rows])
        th0, b0 = Dt.ridge(trn, y, m, LAM, cut); fits = {'base': (th0, None, b0)}
        for lt in LAM_T:
            th, te, bb = Dt.ridge_team(trn, y, m, LAM, lt, cut); fits[f'team_{int(lt)}'] = (th, te, bb)
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)

        def pred(th, te, bb, r):
            p = bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
            if te is not None:
                p = p + np.where(Dt.to[r] >= 0, te[np.maximum(Dt.to[r], 0)], 0) * (Dt.to[r] >= 0) + \
                    np.where(Dt.td[r] >= 0, te[Dt.nt + np.maximum(Dt.td[r], 0)], 0) * (Dt.td[r] >= 0)
            return p
        arms = {}
        for fk, (th, te, bb) in fits.items():
            for k in KS:
                th2 = th.copy()
                if k < 1:
                    for blk in (0, 1):
                        sl = slice(blk * n, (blk + 1) * n)
                        th2[sl] = np.where(mv, m[sl] + k * (th[sl] - m[sl]), th[sl])
                arms[f'{fk}|k{k}'] = (th2, te, bb)
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        Y, W = Dt.Y[rows], Dt.W[rows]
        preds = {}
        for an, (th, te, bb) in arms.items():
            c = se.lin_fit([pred(th, te, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
            preds[an] = se.lin_pred(c, [pred(th, te, bb, rows), Dt.H[rows]])
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        preds['darko'] = se.lin_pred(se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]), dkf(rows))
        held = name not in TUNE
        for an, pv in preds.items():
            nm = ~mrow
            other[(an, held)][0] += float(np.sum(W[nm] * (Y[nm] - pv[nm]) ** 2)); other[(an, held)][1] += float(np.sum(W[nm] * (Y[nm] - ybar) ** 2))
        for p in movers:
            on_o = np.array([p in Dt.O[i] for i in rows]); on_d = np.array([p in Dt.D[i] for i in rows]); on = on_o | on_d
            def net(v):
                return np.sum(W[on_o] * v[on_o]) / max(W[on_o].sum(), 1) - np.sum(W[on_d] * v[on_d]) / max(W[on_d].sum(), 1)
            act = net(Y); sst = float(np.sum(W[on] * (Y[on] - ybar) ** 2))
            for an, pv in preds.items():
                rec[an].append(dict(event=name, held=held, pid=int(Dt.pids[p]), poss=float(W[on].sum()), actual=act, pred=net(pv),
                                    sse=float(np.sum(W[on] * (Y[on] - pv[on]) ** 2)), sst=sst))
        ev_log[name] = dict(movers=len(movers), rows=int(len(rows)))
        print(name, ev_log[name], ' '.join(f"{an}:{1 - sum(r['sse'] for r in rec[an] if r['event'] == name) / sum(r['sst'] for r in rec[an] if r['event'] == name):.5f}"
                                           for an in ['base|k1.0', 'team_1000|k1.0', 'base|k0.65', 'darko']), flush=True)

    def summary(R):
        a = np.array([r['actual'] for r in R]); p = np.array([r['pred'] for r in R]); w = np.array([r['poss'] for r in R])
        X = np.c_[np.ones(len(a)), p]; sl = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (a * w))[1]
        return dict(stint_r2=1 - sum(r['sse'] for r in R) / sum(r['sst'] for r in R), slope=sl,
                    mse=float(np.average((a - p) ** 2, weights=w)))
    tune = {an: summary([r for r in R if not r['held']]) for an, R in rec.items()}
    best = max((an for an in tune if an != 'darko'), key=lambda an: tune[an]['stint_r2'])
    best_team = max((an for an in tune if an.startswith('team') and an.endswith('k1.0')), key=lambda an: tune[an]['stint_r2'])
    best_shr = max((an for an in tune if an.startswith('base')), key=lambda an: tune[an]['stint_r2'])
    show = list(dict.fromkeys(['base|k1.0', best_shr, best_team, best, 'darko']))
    res = dict(events=ev_log, tuned_on=sorted(TUNE), chosen=dict(best=best, team=best_team, shrink=best_shr), tune=tune, held={})
    rng = np.random.default_rng(0)
    H = {an: [r for r in rec[an] if r['held']] for an in show}
    k = len(H['darko'])
    for an in show:
        s = summary(H[an]); o = other[(an, True)]
        s['nonmover_stint_r2'] = 1 - o[0] / o[1]
        diffs = {ref: [] for ref in ('base|k1.0', 'darko') if ref != an}
        for _ in range(1000):
            i = rng.integers(0, k, k)
            for ref in diffs:
                A_ = [H[an][j] for j in i]; B_ = [H[ref][j] for j in i]
                diffs[ref].append((1 - sum(r['sse'] for r in A_) / sum(r['sst'] for r in A_)) - (1 - sum(r['sse'] for r in B_) / sum(r['sst'] for r in B_)))
        s['dr2_ci'] = {ref: np.percentile(v, [2.5, 97.5]).round(5).tolist() for ref, v in diffs.items()}
        res['held'][an] = {kk: (round(float(vv), 5) if not isinstance(vv, dict) else vv) for kk, vv in s.items()}
        print('HELD', an, res['held'][an], flush=True)
    res['held_movers'] = k
    json.dump(res, open(OUT / 'ms_fix.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
