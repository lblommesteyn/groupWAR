"""DUNCAN shrinkage, tuned on what we care about: out-of-sample stint prediction for everyone.

ms_movers / ms_hapm showed DUNCAN's ratings are too extreme (slope of actual on predicted ~0.7 for players who stay,
~0.6 for movers; DARKO ~1.0). Lambda was picked on game log loss, where the logistic calibration hides scale.
Grid: ridge lambda in LAMS x global pull toward the box prior k_all (theta <- m + k_all (theta - m)) x the extra
mover pull k_mv (applied on top for players who changed teams).
Events: the ms_movers cutoffs. Tuned on the 2021-2023 deadlines (all post-cutoff stints); reported on the held-out
events: all stints, stints with a mover, and the per-player slope. 95% CIs resample games.
Output: results/latent/ms_shrink2.json     python ms_shrink.py   (PC)
"""
import json
from collections import defaultdict
import numpy as np

import ms_spm as sp
import ms_stint_eval as se
from ms_fix import FData, TUNE
from ms_movers import EVENTS, MIN_PRE, MIN_POST
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
LAMS = [1000.0, 10000.0, 30000.0, 100000.0]
K_ALL = [1.0, 0.85, 0.7, 0.55]
K_MV = [1.0, 0.7]


def main():
    Dt = FData(); n = Dt.n; Dk = se.load_darko_od(); season = Dt.season_r; has = Dt.to >= 0
    store = defaultdict(lambda: defaultdict(list))   # arm -> field -> list over events
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
        mv = np.zeros(n, bool)
        for (p, t), w in b.items():
            if p in last and t != last[p][0] and w >= MIN_POST:
                mv[p] = True
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        trn = np.flatnonzero(Dt.DAY < cut)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            ld = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > ld - sp.VAL_DAYS) & (Dt.DAY <= ld))
        rows = np.flatnonzero(post); Y, W = Dt.Y[rows], Dt.W[rows]
        mrow = mv[Dt.O[rows]].any(1) | mv[Dt.D[rows]].any(1)
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        g = Dt.GI[rows]
        pr = lambda th, bb, r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        preds = {}
        for lam in LAMS:
            th, bb = Dt.ridge(trn, y, m, lam, cut)
            for ka in K_ALL:
                for km in K_MV:
                    kv = np.where(np.r_[mv, mv], ka * km, ka)
                    t2 = m + kv * (th - m)
                    c = se.lin_fit([pr(t2, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
                    preds[f'lam{int(lam)}_k{ka}_mv{km}'] = (se.lin_pred(c, [pr(t2, bb, rows), Dt.H[rows]]), c[1])
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        cd = se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]); preds['darko'] = (se.lin_pred(cd, dkf(rows)), np.nan)
        for an, (pv, slope) in preds.items():
            S = store[an]
            S['event'].append(np.full(len(rows), name)); S['g'].append(g); S['mrow'].append(mrow)
            S['se'].append(W * (Y - pv) ** 2); S['st'].append(W * (Y - ybar) ** 2); S['map_slope'].append(np.full(len(rows), slope))
        print(name, int(mv.sum()), 'movers', len(rows), 'rows', flush=True)

    def r2(S, mask):
        return 1 - S['se'][mask].sum() / S['st'][mask].sum()
    cat = {an: {f: np.concatenate(v) for f, v in S.items()} for an, S in store.items()}
    ev = cat['darko']['event']; held = ~np.isin(ev, list(TUNE))
    tune = {an: float(r2(S, ~held)) for an, S in cat.items() if an != 'darko'}
    best = max(tune, key=tune.get)  # extended grid: ms_shrink.py chose the largest lambda it had (10000)
    base = 'lam1000_k1.0_mv1.0'; mvfix = 'lam1000_k1.0_mv0.5'
    print('tuning stint R2 top 5:', sorted(tune.items(), key=lambda t: -t[1])[:5], 'base', tune[base], flush=True)
    rng = np.random.default_rng(0)
    gh = cat['darko']['g'][held]; ug = np.unique(gh); gi = np.searchsorted(ug, gh)
    res = dict(tune=tune, chosen=best, held={})
    for an in dict.fromkeys([base, mvfix, best, 'darko']):
        S = {f: v[held] for f, v in cat[an].items()}
        sse = np.bincount(gi, S['se'], len(ug)); sst = np.bincount(gi, S['st'], len(ug))
        sse_m = np.bincount(gi, S['se'] * S['mrow'], len(ug)); sst_m = np.bincount(gi, S['st'] * S['mrow'], len(ug))
        e = dict(all=float(1 - sse.sum() / sst.sum()), movers=float(1 - sse_m.sum() / sst_m.sum()),
                 nonmovers=float(1 - (sse - sse_m).sum() / (sst - sst_m).sum()))
        if an != base:
            Sb = {f: v[held] for f, v in cat[base].items()}
            sb = np.bincount(gi, Sb['se'], len(ug))
            bs = [(sb[i].sum() - sse[i].sum()) / sst[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
            e['d_all_vs_base_ci'] = np.percentile(bs, [2.5, 97.5]).round(5).tolist()
        if an != 'darko':
            Sd = {f: v[held] for f, v in cat['darko'].items()}
            sd_ = np.bincount(gi, Sd['se'], len(ug))
            bs = [(sd_[i].sum() - sse[i].sum()) / sst[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
            e['d_all_vs_darko_ci'] = np.percentile(bs, [2.5, 97.5]).round(5).tolist()
        res['held'][an] = e
        print('HELD', an, e, flush=True)
    json.dump(res, open(OUT / 'ms_shrink2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
