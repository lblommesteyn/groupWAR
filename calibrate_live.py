"""Freeze the win-probability mapping for the live forecasts (live_forecast.py), from the pregame backtest.

For every 2024-25 and 2025-26 test block: our pregame margin (spm_lpit_h ratings saved at the block start, each
player weighted by projected minutes from his previous 10 games, players who played only) and DARKO's (o + d DPM as
of the block start, same weights). One logistic fit per model on all those games -> results/latent/live_calib.json.
Run once, before the live test starts; never refit during it.   python calibrate_live.py   (PC)
"""
import json, pickle
from bisect import bisect_left
import numpy as np
from sklearn.linear_model import LogisticRegression

from masked_paths import ROOT
from ms_fit import Pregame
from ms_stint_eval import load_darko_od

OUT = ROOT / 'results/latent'
SEASONS = (2025, 2026)


def our_margin(pix, theta, b, T):
    n = len(pix)
    val = lambda side, k: sum(s * theta[k * n + pix[p]] for p, s in side if p in pix)
    h, a = T
    return (b[1] + val(h, 0) + val(a, 1)) - (val(a, 0) + val(h, 1))


def darko_margin(Dk, asof, T):
    def net(side):
        s = 0.0
        for p, w in side:
            o, d = -1.0, -1.0
            if p in Dk:
                j = bisect_left(Dk[p][0], asof) - 1
                if j >= 0:
                    o, d = Dk[p][1][j]
            s += w * (o + d)
        return s
    return net(T[0]) - net(T[1])


def main():
    import argparse
    import ms_spm as sp
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, help='refit each block at this lambda (else the saved duncan-1 ratings)')
    ap.add_argument('--k', type=float, default=1.0, help='pull toward the box prior: theta <- m + k (theta - m)')
    ap.add_argument('--out', default='live_calib.json')
    a = ap.parse_args()
    saved = pickle.load(open(OUT / 'spm_ratings.pkl', 'rb'))
    Dk = load_darko_od(); PG = Pregame(None)
    Dt = sp.Data() if a.lam else None
    xo, xd, y = [], [], []
    for f in sorted(OUT.glob('msb_*.npz')):
        season, b0 = int(f.stem.split('_')[1]), f.stem.split('_')[2]
        if season not in SEASONS or 'spm_lpit_h' not in saved.get(b0, {}):
            continue
        r = saved[b0]['spm_lpit_h']; pix = {p: i for i, p in enumerate(r['pids'])}
        if a.lam:
            cut = sp.dnum(b0); y_ = Dt.luck_target(cut, 0.5); m_ = Dt.spm_prior(cut, y_)
            th_, b_ = Dt.ridge(np.flatnonzero(Dt.DAY < cut), y_, m_, a.lam, cut)
            r = dict(theta=m_ + a.k * (th_ - m_), b=b_); pix = Dt.pix
        P = np.load(f)
        for g, w in zip(P['gid'], P['win']):
            T = PG.teams(g)
            if T is None:
                continue
            xo.append(our_margin(pix, r['theta'], r['b'], T)); xd.append(darko_margin(Dk, b0, T)); y.append(int(w))
    xo, xd, y = np.array(xo), np.array(xd), np.array(y)
    out = {'games': len(y), 'seasons': SEASONS}
    for k, x in (('ours', xo), ('darko', xd)):
        m = LogisticRegression(C=1e4).fit(x[:, None], y)
        p = m.predict_proba(x[:, None])[:, 1]
        ll = -np.mean(np.log(np.where(y == 1, p, 1 - p)))
        out[k] = {'intercept': float(m.intercept_[0]), 'slope': float(m.coef_[0, 0]), 'in_sample_ll': float(ll)}
    out.update(lam=a.lam or 1000.0, k=a.k)
    json.dump(out, open(OUT / a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
