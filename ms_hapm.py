"""Held-out adjusted plus-minus (HAPM) on our model, tested where LAPM is weakest: players who change teams.

Idea (Starro / lumahoops, Nov 2026): a joint ridge splits credit between players who share the floor, so part of a
player's rating belongs to his teammates and system. HAPM instead fits the model on every stint WITHOUT player j,
predicts j's stints with an average player in his slot, and calls the gap j's value.
Here: same target (half luck-adjusted points per 100), same box/minutes prior and recency weights as LAPM.
  without-j fit  ridge on training rows without j (lambda 1000, warm-started from the full LAPM solution)
  j's value      offense: possession- and recency-weighted mean of (actual - prediction with j's slot = 0) over j's
                 offensive rows, shrunk toward j's box prior with weight LAM_H; defense the same on defensive rows.
Arms (only the evaluated players' ratings change; everyone else keeps LAPM):
  lapm, lapm_k0.5 (movers halfway to prior, ms_fix), hapm_L for LAM_H in LAMS_H, darko.
Evaluated players: all movers of each event (ms_movers definition) + a sample of non-movers active after the cutoff
(N_STAY), so the non-mover check covers players whose rating actually changed.
Tuning: LAM_H on the 2021-2023 deadlines; report held-out events (2024-26 deadlines, 2025 and 2026 offseasons).
Output: results/latent/ms_hapm.json     python ms_hapm.py   (PC, a few hours)
"""
import json, time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

import ms_spm as sp
import ms_stint_eval as se
from ms_fix import FData, TUNE
from ms_movers import EVENTS, MIN_PRE, MIN_POST
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
LAM = 1000.0
LAMS_H = [300.0, 1000.0, 3000.0]
N_STAY = 80
WINDOW = 3 * sp.HL   # rows older than this weigh < 1/8 and are dropped from the held-out fits


def design(Dt, rows, cut):
    w = Dt.W[rows] * 0.5 ** ((cut - Dt.DAY[rows]) / sp.HL); sw = np.sqrt(w)
    A = csr_matrix(Dt.X[rows].multiply(sw[:, None]))
    C = csr_matrix(np.column_stack([np.ones(len(rows)), Dt.H[rows]]) * sp.SC * sw[:, None])
    return hstack([A, C]).tocsr(), w, sw


def main():
    Dt = FData(); n = Dt.n; Dk = se.load_darko_od(); season = Dt.season_r; has = Dt.to >= 0
    rng = np.random.default_rng(0)
    rec = defaultdict(list); log = {}
    for name, cut_s, days in EVENTS:
        t0 = time.time(); cut = sp.dnum(cut_s)
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
        movers = sorted({p for (p, t), w in b.items() if p in last and t != last[p][0] and w >= MIN_POST})
        stay_pool = sorted({p for (p, t), w in b.items() if p in last and t == last[p][0] and w >= MIN_POST} - set(movers))
        stay = sorted(rng.choice(stay_pool, min(N_STAY, len(stay_pool)), replace=False).tolist())
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        trn = np.flatnonzero((Dt.DAY < cut) & (Dt.DAY >= cut - WINDOW))
        A, w, sw = design(Dt, trn, cut)
        r0 = sw * (y[trn] - Dt.X[trn] @ m)
        full = lsqr(A, r0, damp=np.sqrt(LAM), atol=1e-7, btol=1e-7, iter_lim=3000)[0]
        th0 = m + full[:2 * n]; b0 = full[2 * n:] * sp.SC
        # per evaluated player: held-out fit and residual sums
        H = {}
        On, Dn = Dt.O[trn], Dt.D[trn]
        for j in movers + stay:
            in_o = (On == j).any(1); in_d = (Dn == j).any(1); keep = ~(in_o | in_d)
            x0 = full.copy()
            sol = lsqr(A[keep], r0[keep], damp=np.sqrt(LAM), atol=1e-6, btol=1e-6, iter_lim=400, x0=x0)[0]
            th, bb = m + sol[:2 * n], sol[2 * n:] * sp.SC
            th[j] = 0.0; th[n + j] = 0.0                         # average player in j's slot
            ro = trn[in_o]; rd = trn[in_d]
            pred = lambda r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
            wo, wd = w[in_o], w[in_d]
            H[j] = (float(np.sum(wo * (y[ro] - pred(ro)))), float(wo.sum()), float(np.sum(wd * (y[rd] - pred(rd)))), float(wd.sum()))
        print(name, len(movers), 'movers', len(stay), 'stayers', f'{time.time() - t0:.0f}s', flush=True)
        mv = np.zeros(n, bool); mv[movers] = True
        arms = {'lapm': th0.copy()}
        k = th0.copy()
        for blk in (0, 1):
            sl = slice(blk * n, (blk + 1) * n); k[sl] = np.where(mv, m[sl] + 0.5 * (th0[sl] - m[sl]), th0[sl])
        arms['lapm_k0.5'] = k
        for L in LAMS_H:
            t = th0.copy()
            for j, (so, wo_, sd, wd_) in H.items():
                t[j] = (so + L * m[j]) / (wo_ + L)
                t[n + j] = (sd + L * m[n + j]) / (wd_ + L)
            arms[f'hapm_{int(L)}'] = t
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            ld = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > ld - sp.VAL_DAYS) & (Dt.DAY <= ld))
        rows = np.flatnonzero(post); Y, W = Dt.Y[rows], Dt.W[rows]
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        pr = lambda th, r: b0[0] + b0[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        preds = {an: se.lin_pred(se.lin_fit([pr(th, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow]), [pr(th, rows), Dt.H[rows]])
                 for an, th in arms.items()}
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        preds['darko'] = se.lin_pred(se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]), dkf(rows))
        held = name not in TUNE
        for grp, ps in (('mover', movers), ('stay', stay)):
            for p in ps:
                on_o = (Dt.O[rows] == p).any(1); on_d = (Dt.D[rows] == p).any(1); on = on_o | on_d
                net = lambda v: np.sum(W[on_o] * v[on_o]) / max(W[on_o].sum(), 1) - np.sum(W[on_d] * v[on_d]) / max(W[on_d].sum(), 1)
                act = net(Y); sst = float(np.sum(W[on] * (Y[on] - ybar) ** 2))
                for an, pv in preds.items():
                    rec[(grp, an)].append(dict(event=name, held=held, poss=float(W[on].sum()), actual=act, pred=net(pv),
                                               sse=float(np.sum(W[on] * (Y[on] - pv[on]) ** 2)), sst=sst))
        log[name] = dict(movers=len(movers), stay=len(stay), secs=round(time.time() - t0))

    def summ(R):
        a = np.array([r['actual'] for r in R]); p = np.array([r['pred'] for r in R]); w = np.array([r['poss'] for r in R])
        X = np.c_[np.ones(len(a)), p]
        return dict(stint_r2=1 - sum(r['sse'] for r in R) / sum(r['sst'] for r in R),
                    slope=float(np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (a * w))[1]))
    tune = {an: summ([r for r in R if not r['held']]) for (g, an), R in rec.items() if g == 'mover'}
    hs = [an for an in tune if an.startswith('hapm')]
    best = max(hs, key=lambda an: tune[an]['stint_r2'])
    show = ['lapm', 'lapm_k0.5', best, 'darko']
    res = dict(events=log, tune=tune, chosen=best, held={})
    for g in ('mover', 'stay'):
        Hr = {an: [r for r in rec[(g, an)] if r['held']] for an in show}; k_ = len(Hr['lapm'])
        out = {}
        for an in show:
            s = summ(Hr[an]); diffs = {ref: [] for ref in ('lapm', 'darko') if ref != an}
            for _ in range(1000):
                i = rng.integers(0, k_, k_)
                for ref in diffs:
                    A_ = [Hr[an][x] for x in i]; B_ = [Hr[ref][x] for x in i]
                    diffs[ref].append((1 - sum(r['sse'] for r in A_) / sum(r['sst'] for r in A_)) -
                                      (1 - sum(r['sse'] for r in B_) / sum(r['sst'] for r in B_)))
            s['dr2_ci'] = {ref: np.percentile(v, [2.5, 97.5]).round(5).tolist() for ref, v in diffs.items()}
            out[an] = s
            print('HELD', g, an, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in s.items()}, flush=True)
        res['held'][g] = dict(n=k_, arms=out)
    json.dump(res, open(OUT / 'ms_hapm.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
