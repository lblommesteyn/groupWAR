"""Interior-weighted defense on the full 6,057-game benchmark (2020-21 .. 2025-26).

Found on the tracking-era development blocks (trk_bench.py, 2026 cutoffs only): giving more of each stint's defensive
credit to defenders who anchor the interior improves held-out stint prediction, and the box-score proxy (blocks +
defensive rebounds) does as well as tracking paint presence. These blocks were not used in finding it.
  w_j = max(1 + g (load_j / mean load of the five defenders - 1), 0.05),  load = decayed, shrunk (blk + dreb) per 36
  relative to the minutes-weighted league mean, from box scores before the cutoff (the SPM box features).
Arms, all lambda 3000, half luck target, SPM prior, no mover pull (as ms_duncan2):  g = 0 (DUNCAN-2 core), 0.6, 1.2.
PRE-DECLARED: primary g = 0.6. It passes if its game log loss is not worse than g = 0 (paired CI not above zero) AND
its stint R2 gain over g = 0 has a game-resampled CI above zero. g = 1.2 is reported, not used for the decision.
Same blocks, validation windows and calibration protocol as ms_duncan2 / ms_bench.
Output: results/latent/ms_defw.json     python ms_defw.py   (PC)
"""
import json
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression

import ms_spm as sp
import ms_stint_eval as se
from trk_bench import design, ridge
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
GS = [0.0, 0.6, 1.2]


def weights(Dt, cut, g):
    if g == 0:
        return None
    F, mins = Dt.box_features(cut)
    bp = F[:, sp.BOX.index('blk')] + F[:, sp.BOX.index('dreb')]
    bp = bp / np.average(bp, weights=np.maximum(mins, 1))
    L = bp[Dt.D]
    return np.maximum(1 + g * (L / L.mean(1, keepdims=True) - 1), 0.05)


def main():
    Dt = sp.Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    Dk = se.load_darko_od()
    P = {g: [] for g in GS}; P['darko_cal'] = []; WIN = []
    SE = {g: [] for g in GS}; SE['darko'] = []; ST, SG = [], []

    def fit(cut, g):
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        X = design(Dt, wD=weights(Dt, cut, g))
        th, b = ridge(Dt, X, np.flatnonzero(Dt.DAY < cut), y, m, cut)
        return X, th, b

    def margins(X, th, b, gis):
        out = np.zeros(len(gis))
        for k, gi in enumerate(gis):
            r = np.asarray(Dt.rows_of[gi])
            yh = b[0] + b[1] * Dt.H[r] + X[r] @ th
            out[k] = np.sum(np.where(Dt.H[r] == 1, 1, -1) * Dt.W[r] * yh / 100)
        return out

    for f in sorted(OUT.glob('msb_*.npz')):
        b0 = f.stem.split('_')[2]; Z = dict(np.load(f)); test = list(Z['gid'])
        vcut_s = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS))
        val = [g for g in Dt.gids if vcut_s <= G[g]['date'] < b0]
        vr = np.concatenate([Dt.rows_of[gix[g]] for g in val if gix[g] in Dt.rows_of]).astype(int)
        tr = np.concatenate([Dt.rows_of[gix[g]] for g in test if gix[g] in Dt.rows_of]).astype(int)
        for g in GS:
            Xv, thv, bv = fit(sp.dnum(vcut_s), g); xv = margins(Xv, thv, bv, [gix[q] for q in val])
            cal = LogisticRegression(C=1e4).fit(xv[:, None], win(val))
            X, th, b = fit(sp.dnum(b0), g); xt = margins(X, th, b, [gix[q] for q in test])
            P[g].append(cal.predict_proba(xt[:, None])[:, 1])
            pv = bv[0] + bv[1] * Dt.H[vr] + Xv[vr] @ thv; pt = b[0] + b[1] * Dt.H[tr] + X[tr] @ th
            c = se.lin_fit([pv, Dt.H[vr]], Dt.Y[vr], Dt.W[vr]); SE[g].append(Dt.W[tr] * (Dt.Y[tr] - se.lin_pred(c, [pt, Dt.H[tr]])) ** 2)
        o_, d_ = se.darko_vec(Dk, Dt.pids, vcut_s); fv = [o_[Dt.O[vr]].sum(1), d_[Dt.D[vr]].sum(1), Dt.H[vr]]
        o_, d_ = se.darko_vec(Dk, Dt.pids, b0); ft = [o_[Dt.O[tr]].sum(1), d_[Dt.D[tr]].sum(1), Dt.H[tr]]
        SE['darko'].append(Dt.W[tr] * (Dt.Y[tr] - se.lin_pred(se.lin_fit(fv, Dt.Y[vr], Dt.W[vr]), ft)) ** 2)
        ybar = np.average(Dt.Y[vr], weights=Dt.W[vr]); ST.append(Dt.W[tr] * (Dt.Y[tr] - ybar) ** 2); SG.append(Dt.GI[tr])
        P['darko_cal'].append(Z['darko_cal']); WIN.append(Z['win'])
        print(b0, len(test), flush=True)

    w = np.concatenate(WIN); p = {k: np.concatenate(v) for k, v in P.items()}
    ll = {k: -np.log(np.clip(np.where(w == 1, v, 1 - v), 1e-9, 1)) for k, v in p.items()}
    rng = np.random.default_rng(0); bs = rng.integers(0, len(w), (2000, len(w)))
    sg = np.concatenate(SG); ug, gi = np.unique(sg, return_inverse=True)
    t = np.bincount(gi, np.concatenate(ST), len(ug))
    sse = {k: np.bincount(gi, np.concatenate(v), len(ug)) for k, v in SE.items()}
    rg = np.random.default_rng(1); gidx = [rg.integers(0, len(ug), len(ug)) for _ in range(2000)]
    res = dict(games=int(len(w)), primary='g0.6', arms={})
    for k in list(GS) + ['darko']:
        kk = 'darko_cal' if k == 'darko' else k
        e = dict(log_loss=round(float(ll[kk].mean()), 5), stint_r2=round(float(1 - sse[k].sum() / t.sum()), 6))
        for ref in (0.0, 'darko'):
            if k == ref:
                continue
            rr = 'darko_cal' if ref == 'darko' else ref
            d = ll[kk] - ll[rr]
            e[f'dll_vs_{ref}_ci'] = np.percentile([d[i].mean() for i in bs], [2.5, 97.5]).round(5).tolist()
            e[f'r2_gain_vs_{ref}_ci'] = np.percentile([(sse[ref][i].sum() - sse[k][i].sum()) / t[i].sum() for i in gidx], [2.5, 97.5]).round(6).tolist()
        res['arms'][f'g{k}' if k != 'darko' else 'darko'] = e
        print(k, e, flush=True)
    a = res['arms']['g0.6']
    res['primary_passes'] = bool(a['dll_vs_0.0_ci'][0] <= 0 and a['r2_gain_vs_0.0_ci'][0] > 0)
    json.dump(res, open(OUT / 'ms_defw.json', 'w'), indent=1)
    print('primary passes:', res['primary_passes'])


if __name__ == '__main__':
    main()
