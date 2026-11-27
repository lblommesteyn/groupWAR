"""DUNCAN v2 on the 6,057-game benchmark: does the stint-tuned shrinkage (ms_shrink.py) keep or improve win and
margin prediction? Same blocks, validation windows and calibration protocol as ms_spm / ms_bench:
  duncan2  lambda LAM2, ratings pulled toward the box prior: theta <- m + K_ALL (theta - m)
Read-only on results/latent/msb_*.npz; saves results/latent/duncan2_{LAM2}_{K_ALL}.npz and prints pooled log loss / margin corr against spm_lpit_h
(current DUNCAN) and darko_cal with paired bootstrap CIs.   python ms_duncan2.py [LAM2 K_ALL]   (PC)
"""
import sys
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
import ms_spm as sp
import ms_stint_eval as se
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
LAM2 = float(sys.argv[1]) if len(sys.argv) > 1 else 10000.0
K_ALL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
KEY = 'duncan2'
TAG = f'{int(LAM2)}_{K_ALL}'


def fit(Dt, cut):
    y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
    th, b = Dt.ridge(np.flatnonzero(Dt.DAY < cut), y, m, LAM2, cut)
    return m + K_ALL * (th - m), b


def main():
    Dt = sp.Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    W, P_, D_, S_, M_ = [], [], [], [], []
    Dk = se.load_darko_od(); SE, SD, ST, SG = [], [], [], []
    for f in sorted(OUT.glob('msb_*.npz')):
        b0 = f.stem.split('_')[2]; P = dict(np.load(f)); test = list(P['gid'])
        vcut_s = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS))
        val = [g for g in Dt.gids if vcut_s <= G[g]['date'] < b0]
        thv, bv = fit(Dt, sp.dnum(vcut_s)); xv = Dt.margins(thv, bv, [gix[g] for g in val])
        cal = LogisticRegression(C=1e4).fit(xv[:, None], win(val))
        th, b = fit(Dt, sp.dnum(b0)); xt = Dt.margins(th, b, [gix[g] for g in test])
        # stint R2 on the same blocks: linear map fit on the validation stints, as ms_stint_eval
        vr = np.concatenate([Dt.rows_of[gix[g]] for g in val if gix[g] in Dt.rows_of]).astype(int)
        tr = np.concatenate([Dt.rows_of[gix[g]] for g in test if gix[g] in Dt.rows_of]).astype(int)
        pr = lambda t_, b_, r: b_[0] + b_[1] * Dt.H[r] + t_[Dt.O[r]].sum(1) + t_[Dt.n + Dt.D[r]].sum(1)
        c = se.lin_fit([pr(thv, bv, vr), Dt.H[vr]], Dt.Y[vr], Dt.W[vr]); pt = se.lin_pred(c, [pr(th, b, tr), Dt.H[tr]])
        o_, d_ = se.darko_vec(Dk, Dt.pids, vcut_s); fv = [o_[Dt.O[vr]].sum(1), d_[Dt.D[vr]].sum(1), Dt.H[vr]]
        o_, d_ = se.darko_vec(Dk, Dt.pids, b0); ft = [o_[Dt.O[tr]].sum(1), d_[Dt.D[tr]].sum(1), Dt.H[tr]]
        pd_ = se.lin_pred(se.lin_fit(fv, Dt.Y[vr], Dt.W[vr]), ft)
        ybar = np.average(Dt.Y[vr], weights=Dt.W[vr])
        SE.append(Dt.W[tr] * (Dt.Y[tr] - pt) ** 2); SD.append(Dt.W[tr] * (Dt.Y[tr] - pd_) ** 2); ST.append(Dt.W[tr] * (Dt.Y[tr] - ybar) ** 2); SG.append(Dt.GI[tr])
        P[KEY] = cal.predict_proba(xt[:, None])[:, 1]   # read-only: several settings run in parallel
        W.append(P['win']); P_.append(P[KEY]); D_.append(P['darko_cal']); S_.append(P['spm_lpit_h']); M_.append(P['margin'])
        print(b0, len(test), flush=True)
    w = np.concatenate(W); p = {'duncan2': np.concatenate(P_), 'darko_cal': np.concatenate(D_), 'spm_lpit_h': np.concatenate(S_)}
    mg = np.concatenate(M_)
    ll = {k: -np.log(np.clip(np.where(w == 1, v, 1 - v), 1e-9, 1)) for k, v in p.items()}
    lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / np.clip(1 - v, 1e-6, 1))
    rng = np.random.default_rng(0); bs = rng.integers(0, len(w), (2000, len(w)))
    se_, sd_, st_, sg_ = (np.concatenate(v) for v in (SE, SD, ST, SG))
    ug, gi = np.unique(sg_, return_inverse=True)
    a, d, t = (np.bincount(gi, v, len(ug)) for v in (se_, sd_, st_))
    r2o, r2d = 1 - a.sum() / t.sum(), 1 - d.sum() / t.sum()
    rg = np.random.default_rng(1)
    bsg = [(d[i].sum() - a[i].sum()) / t[i].sum() for i in (rg.integers(0, len(ug), len(ug)) for _ in range(1000))]  # one generator: fresh draws
    print(f'stint R2 duncan2 {r2o:.5f}  darko {r2d:.5f}  diff CI {np.percentile(bsg, [2.5, 97.5]).round(5)}')
    np.savez(OUT / f'duncan2_{TAG}.npz', win=w, p=p['duncan2'], margin=mg, stint_r2=r2o, stint_r2_darko=r2d)
    print(f'{len(w)} games  LAM2 {LAM2} K_ALL {K_ALL}')
    for k in p:
        print(f'{k:11s} ll {ll[k].mean():.4f}  margin r {np.corrcoef(lg(p[k]), mg)[0, 1]:.3f}')
    for ref in ('spm_lpit_h', 'darko_cal'):
        d = ll['duncan2'] - ll[ref]
        dc = [np.corrcoef(lg(p['duncan2'][i]), mg[i])[0, 1] - np.corrcoef(lg(p[ref][i]), mg[i])[0, 1] for i in bs[:1000]]
        print(f'duncan2 - {ref}: dll {d.mean():+.4f} CI {np.percentile([d[i].mean() for i in bs], [2.5, 97.5]).round(4)}  '
              f'dr CI {np.percentile(dc, [2.5, 97.5]).round(3)}')


if __name__ == '__main__':
    main()
