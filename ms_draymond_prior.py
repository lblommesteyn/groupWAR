"""Does BOWEN (formerly DRAYMOND) (world_model.py) predict defense beyond DUNCAN? A defensive-prior test like ms_trk2.

At each cutoff inside the tracked seasons, per defender, from BOWEN (formerly DRAYMOND) frames of games BEFORE the cutoff only:
  exec_depth (signed distance to rim vs scheme spot), exec_ft (distance to scheme spot), scheme_depth, and the
  same split by situation (defender within 12 ft of the rim at the scheme spot vs outside).
Prior adjustment: RidgeCV of (plain RAPM defense - box prior defense) on those features, weights min(frames/2500, 1)
x possessions, added to the box prior for defense only; then the DUNCAN v2 fit (lambda LAM2, pull K_ALL).
Scored on the following 30 days: stint R2 (linear map fit on the 30 days before the cutoff), game-clustered CIs.
Caveat: the world model is cross-fitted by game over both seasons, so its notion of normal positioning saw later
games (positions only, never outcomes).
Output: results/latent/ms_draymond_prior.json     python ms_draymond_prior.py [LAM2 K_ALL]   (PC)
"""
import json, sys
from datetime import date
import numpy as np
from sklearn.linear_model import RidgeCV

import ms_spm as sp
import ms_stint_eval as se
from masked_paths import ROOT

OUT = ROOT / 'results/latent'; WD = ROOT / 'results/world'
LAM2 = float(sys.argv[1]) if len(sys.argv) > 1 else 10000.0
K_ALL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
CUTS = ['2026-02-18', '2026-03-20', '2026-04-19', '2026-11-11', '2026-11-18']
RIM = np.array([41.75, 0.0])


def main():
    Dt = sp.Data(live=True); n = Dt.n
    z = np.load(WD / 'pred.npz')
    poss, k = z['poss'], z['k'].astype(int)
    y, ms = z['y'].astype(np.float32), z['ms'].astype(np.float32)
    pid = z['meta_deff'][poss, k]; dnum = np.array([sp.dnum(d) for d in z['meta_date']])[poss]
    dep = lambda p: np.hypot(*(p - RIM).T)
    ex_d = dep(y) - dep(ms); ex = np.hypot(*(y - ms).T); sh_d = dep(ms) - dep(z['mb'].astype(np.float32)); near = dep(ms) < 12
    pix = np.array([Dt.pix.get(int(p), -1) for p in pid])
    rows_all = []
    res = {}
    for cut_s in CUTS:
        cut = sp.dnum(cut_s); sel = (dnum < cut) & (pix >= 0)
        ii = pix[sel]
        cnt = np.bincount(ii, minlength=n).astype(float)
        f = lambda v, m=None: np.bincount(ii, (v if m is None else v * m)[sel], n) / np.maximum(np.bincount(ii, None if m is None else m[sel].astype(float), n), 1)
        F = np.column_stack([f(ex_d), f(ex), f(sh_d), f(ex_d, near), f(ex_d, ~near), f(ex, near), f(ex, ~near)])
        have = cnt >= 1500
        y_t = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y_t)
        trn = np.flatnonzero(Dt.DAY < cut)
        thr, _ = Dt.ridge(trn, y_t, np.zeros(2 * n), sp.LAM_TARGET, cut)
        res_d = -(thr[n:] - m[n:])                      # what the box prior misses on defense (+ = better)
        Z = (F[have] - F[have].mean(0)) / (F[have].std(0) + 1e-9)
        wts = np.minimum(cnt[have] / 2500, 1.0)
        reg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Z, res_d[have], sample_weight=wts)
        adj = np.zeros(2 * n); adj[n + np.flatnonzero(have)] = -reg.predict(Z) * wts   # defence theta: + = worse
        arms = {}
        for name, mm in (('duncan2', m), ('duncan2_draymond', m + adj)):
            th, bb = Dt.ridge(trn, y_t, mm, LAM2, cut); arms[name] = (mm + K_ALL * (th - mm), bb)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        trow = np.flatnonzero((Dt.DAY >= cut) & (Dt.DAY < cut + 30))
        pr = lambda th, bb, r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        for name, (th, bb) in arms.items():
            c = se.lin_fit([pr(th, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
            p = se.lin_pred(c, [pr(th, bb, trow), Dt.H[trow]])
            rows_all.append((cut_s, name, Dt.GI[trow], Dt.W[trow] * (Dt.Y[trow] - p) ** 2, Dt.W[trow] * (Dt.Y[trow] - ybar) ** 2))
        res[cut_s] = dict(players=int(have.sum()), cv_r2=round(float(reg.score(Z, res_d[have], sample_weight=wts)), 3),
                          coef=dict(zip(['exec_depth', 'exec_ft', 'scheme_depth', 'exec_depth_near', 'exec_depth_far', 'exec_ft_near', 'exec_ft_far'],
                                        np.round(reg.coef_, 3).tolist())), test_rows=int(len(trow)))
        print(cut_s, res[cut_s], flush=True)
    out = {}
    for name in ('duncan2', 'duncan2_draymond'):
        R = [r for r in rows_all if r[1] == name]
        out[name] = dict(g=np.concatenate([r[2] for r in R]), se=np.concatenate([r[3] for r in R]), st=np.concatenate([r[4] for r in R]))
    g = out['duncan2']['g']; ug, gi = np.unique(g, return_inverse=True)
    S = {k_: (np.bincount(gi, v['se'], len(ug)), np.bincount(gi, v['st'], len(ug))) for k_, v in out.items()}
    r2 = {k_: float(1 - a.sum() / b.sum()) for k_, (a, b) in S.items()}
    rng = np.random.default_rng(0)
    bs = [(S['duncan2'][0][i].sum() - S['duncan2_draymond'][0][i].sum()) / S['duncan2'][1][i].sum()
          for i in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    res['stint_r2'] = r2; res['gain_ci'] = np.percentile(bs, [2.5, 97.5]).round(5).tolist()
    print('stint R2', r2, 'gain CI', res['gain_ci'], flush=True)
    json.dump(res, open(OUT / 'ms_draymond_prior.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
