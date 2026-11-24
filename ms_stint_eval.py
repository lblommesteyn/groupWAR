"""Phase 3 test with more power than wins: predict the points per 100 possessions of every held-out stint.

For each ms_bench block (ratings frozen at b0, lambda from the block's npz): every stint-side row of the block's games
is predicted from the ten players on the floor. DARKO gets the same treatment from its o_dpm / d_dpm dated before b0,
mapped to points with a linear fit (intercept, home, sum offense o_dpm, sum defense d_dpm) on the 30 days before b0;
our arms get the same linear recalibration of their own prediction, so scale differences do not matter.
Score: possession-weighted R2 of actual points per 100 (and of the luck-adjusted points, as a lower-noise target),
pooled over blocks; paired bootstrap over games of the R2 difference vs DARKO.
Player-level (2025-26): each player's on-court net per 100 after the cutoff vs the net his ratings predict
(sum of his lineups' predictions), correlation across players with >= 300 possessions after the cutoff.
Output: results/latent/ms_stint_eval.json   (PC: python ms_stint_eval.py)
"""
import json, pickle
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
import numpy as np

import ms_spm as sp
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
ARMS = ['rapm', 'spm', 'spm_lpit_h']


def load_darko_od():
    H = json.load(open(ROOT / 'results/benchmarks/darko_history.json'))
    out = {}
    for p, h in H.items():
        if 'o_dpm' not in h:
            continue
        rows = sorted((d, o, de) for d, o, de, f in zip(h['date'], h['o_dpm'], h['d_dpm'], h.get('future_game', [0] * len(h['date'])))
                      if o is not None and de is not None and not f)
        out[int(p)] = ([r[0] for r in rows], [(r[1], r[2]) for r in rows])
    return out


def darko_vec(Dk, pids, asof):
    o = np.full(len(pids), -1.0); d = np.full(len(pids), -1.0)  # unrated: below average at both ends
    for i, p in enumerate(pids):
        if p in Dk:
            j = bisect_left(Dk[p][0], asof) - 1
            if j >= 0:
                o[i], d[i] = Dk[p][1][j]
    return o, d


def lin_fit(Xf, y, w):
    A = np.column_stack([np.ones(len(y))] + Xf); sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


def lin_pred(c, Xf):
    return np.column_stack([np.ones(len(Xf[0]))] + Xf) @ c


def main():
    Dt = sp.Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    Dk = load_darko_od()
    acc = defaultdict(list)  # per block arrays
    player = {}
    for f in sorted(OUT.glob('msb_*.npz')):
        P = dict(np.load(f)); season, b0 = int(f.stem.split('_')[1]), f.stem.split('_')[2]
        if not all(a in P for a in ARMS):
            continue
        vcut_s = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS)); cut, vcut = sp.dnum(b0), sp.dnum(vcut_s)
        test_rows = np.concatenate([Dt.rows_of[gix[g]] for g in P['gid'] if gix[g] in Dt.rows_of]).astype(int)
        val_rows = np.flatnonzero((Dt.DAY >= vcut) & (Dt.DAY < cut))
        H = Dt.H; W = Dt.W
        preds = {}
        for a in ARMS:
            lam = float(P[a + '_lam'])
            out = []
            for c_, rows in ((vcut, val_rows), (cut, test_rows)):
                th, b = sp.fit_arm(Dt, a, c_, lam)
                out.append(b[0] + b[1] * H[rows] + th[Dt.O[rows]].sum(1) + th[Dt.n + Dt.D[rows]].sum(1))
            c = lin_fit([out[0], H[val_rows]], Dt.Y[val_rows], W[val_rows])
            preds[a] = lin_pred(c, [out[1], H[test_rows]])
            if a == 'spm_lpit_h':
                preds['_raw_spm_lpit_h'] = out[1]
        feats = []
        for asof, rows in ((vcut_s, val_rows), (b0, test_rows)):
            o, d = darko_vec(Dk, Dt.pids, asof)
            feats.append([o[Dt.O[rows]].sum(1), d[Dt.D[rows]].sum(1), H[rows]])
        c = lin_fit(feats[0], Dt.Y[val_rows], W[val_rows])
        preds['darko'] = lin_pred(c, feats[1])
        acc['season'].append(np.full(len(test_rows), season)); acc['g'].append(Dt.GI[test_rows])
        acc['y'].append(Dt.Y[test_rows]); acc['yl'].append(Dt.YL[test_rows]); acc['w'].append(W[test_rows])
        for k, v in preds.items():
            if not k.startswith('_'):
                acc[k].append(v)
        if b0 == '2026-02-18':  # player-level: ratings at the 2026 deadline block, rest of season
            later = np.flatnonzero(Dt.DAY >= cut)
            pr = {}
            for a in ARMS:
                th, b = sp.fit_arm(Dt, a, cut, float(P[a + '_lam']))
                pr[a] = th
            o, d = darko_vec(Dk, Dt.pids, b0)
            player = dict(rows=later, theta=pr, darko=(o, d))
        print(season, b0, len(test_rows), 'rows', flush=True)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    res = {}
    rng = np.random.default_rng(0)
    games = np.unique(A['g']); gpos = {g: i for i, g in enumerate(games)}; gi = np.array([gpos[g] for g in A['g']])
    names = ARMS + ['darko']
    for tgt in ('y', 'yl'):
        y, w = A[tgt], A['w']
        sse = {k: np.bincount(gi, w * (y - A[k]) ** 2, len(games)) for k in names}
        sst = np.bincount(gi, w * (y - np.average(y, weights=w)) ** 2, len(games))
        r2 = {k: 1 - sse[k].sum() / sst.sum() for k in names}
        bs = rng.integers(0, len(games), (1000, len(games)))
        res[tgt] = {}
        for k in names:
            e = dict(r2=float(r2[k]))
            if k != 'darko':
                dd = [(sse['darko'][b].sum() - sse[k][b].sum()) / sst[b].sum() for b in bs]
                e['d_r2_vs_darko'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            res[tgt][k] = e
        print(f'target {tgt}:', ' '.join(f"{k} R2 {res[tgt][k]['r2']:.5f} {np.round(res[tgt][k].get('d_r2_vs_darko', []), 5)}" for k in names))
        for s in sorted(set(A['season'])):
            m = A['season'] == s; gs = np.unique(gi[m])
            print(f'  {s}', ' '.join(f"{k} {1 - sse[k][gs].sum() / sst[gs].sum():.5f}" for k in names))
            res[tgt][str(s)] = {k: float(1 - sse[k][gs].sum() / sst[gs].sum()) for k in names}
    if player:
        rows = player['rows']; o, d = player['darko']
        act = defaultdict(lambda: np.zeros(3))  # player -> [sum w*on-court net contribution, sum w, n rows]
        pred = {k: defaultdict(float) for k in ARMS + ['darko']}
        for k in ARMS + ['darko']:
            if k == 'darko':
                po = o[Dt.O[rows]].sum(1) - d[Dt.D[rows]].sum(1)  # higher = offense better
            else:
                th = player['theta'][k]; po = th[Dt.O[rows]].sum(1) + th[Dt.n + Dt.D[rows]].sum(1)
            for r, v in zip(rows, po):
                for p in Dt.O[r]:
                    pred[k][p] += Dt.W[r] * v
                for p in Dt.D[r]:
                    pred[k][p] -= Dt.W[r] * v
        for r in rows:
            for p in Dt.O[r]:
                act[p] += [Dt.W[r] * Dt.Y[r], Dt.W[r], 1]
            for p in Dt.D[r]:
                act[p] += [-Dt.W[r] * Dt.Y[r], Dt.W[r], 1]
        ps = [p for p, v in act.items() if v[1] >= 600]  # >= 300 possessions each way
        y = np.array([act[p][0] / act[p][1] for p in ps])
        res['player_2026'] = dict(n=len(ps))
        for k in ARMS + ['darko']:
            x = np.array([pred[k][p] / act[p][1] for p in ps])
            res['player_2026'][k] = float(np.corrcoef(x, y)[0, 1])
        print('player-level after 2026-02-18, corr(predicted, actual on-court net):', res['player_2026'])
    json.dump(res, open(OUT / 'ms_stint_eval.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
