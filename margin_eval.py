"""Margin-level re-scoring of every saved rolling test: same predictions, the final margin instead of the winner.

Every arm's saved test probability is monotone in its predicted margin (the GroupWAR-T heads regress
sigmoid(margin / SCALE); the latent arms are logistic calibrations of a margin), so logit(p) is an affine
function of the predicted margin. Score = correlation of logit(p) with the actual final margin (affine-invariant,
so no extra calibration is fitted on test data); R^2 = corr^2. Paired game-bootstrap CIs for differences.
  2025-26 tracking season, 423 games: rolling_prior + benchmark + latent block files (same test order).
  2019-23 lineup data, 984 games in 2022-23: ms_latent block files.
Output: results/latent/margin_eval.json.   Usage (PC): python margin_eval.py
"""
import json, pickle
import numpy as np

import embed_eval as ee
import groupwar_t as gw
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
B = 4000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def compare(y, X, pairs, rng):
    idx = rng.integers(0, len(y), (B, len(y)))
    cor = lambda x, i: np.corrcoef(x[i], y[i])[0, 1]
    res = {k: dict(corr=float(np.corrcoef(x, y)[0, 1]), r2=float(np.corrcoef(x, y)[0, 1] ** 2)) for k, x in X.items()}
    for a, b in pairs:
        d = np.array([cor(X[a], i) - cor(X[b], i) for i in idx])
        res[a][f'd_corr_vs_{b}'] = [float(np.mean(d))] + [float(q) for q in np.percentile(d, [2.5, 97.5])]
        res[a][f'p_better_than_{b}'] = float((d > 0).mean())
    return res


def show(title, n, res):
    print(f'\n{title} ({n} games)')
    for k, r in sorted(res.items(), key=lambda t: -t[1]['corr']):
        extra = '  '.join(f"Δ vs {kk[11:]} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for kk, v in r.items() if kk.startswith('d_corr_vs_'))
        print(f'  {k:24s} corr {r["corr"]:.3f}  R2 {r["r2"]:.3f}  {extra}')


def main():
    rng = np.random.default_rng(0)
    out = {}
    # ---- 2025-26
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb')); G = {g: d for g, d in G.items() if d['stints']}
    recs = gw.game_records(G)
    y, X = [], {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        order = [r['game_id'] for r in recs if b0 <= r['date'] < b1]
        rp = np.load(gw.OUT / 'rolling_prior' / f'block_{b0}.npz')
        bm = np.load(gw.OUT / 'benchmark' / f'block_{b0}.npz')
        lt = np.load(gw.OUT / 'latent' / f'block_{b0}.npz')
        m = np.array([G[g]['final'][0] - G[g]['final'][1] for g in order], float)
        assert len(m) == len(rp['win']) and ((m > 0).astype(int) == rp['win']).all() and (bm['win'] == rp['win']).all()
        y.append(m)
        seen = set()
        for src, keys in ((rp, rp.files), (bm, bm.files), (lt, [k for k in lt.files if k.startswith('lat_')])):
            for k in keys:
                if k != 'win' and k not in seen:
                    seen.add(k); X.setdefault(k, []).append(logit(src[k]))
    y = np.concatenate(y); X = {k: np.concatenate(v) for k, v in X.items() if len(v) == len(BLOCKS) - 1}
    pairs = [(k, 'darko_pit') for k in X if k != 'darko_pit'] + [('lat_best_load', 'anchored_comp_hist_trk'),
             ('lat_usage_ms', 'lat_additive'), ('lat_best_trk', 'anchored_comp_hist_trk'), ('lat_usage_trk', 'lat_additive'), ('lat_usage', 'lat_additive'), ('anchored_comp_hist_trk', 'rapm')]
    out['2025_26'] = dict(n=len(y), arms=compare(y, X, pairs, rng))
    show('2025-26 rolling test, margin correlation', len(y), out['2025_26']['arms'])
    # ---- 2019-23 multi-season latent
    D = pickle.load(open(OUT / 'ms_stints.pkl', 'rb')); GM = D['games']
    import ms_latent as ml
    y, X = [], {}
    for b0, b1 in zip(ml.BLOCKS[:-1], ml.BLOCKS[1:]):
        order = sorted((g for g in GM if b0 <= GM[g]['date'] < b1), key=lambda g: (GM[g]['date'], g))
        P = np.load(OUT / f'ms_block_{b0}.npz')
        m = np.array([GM[g]['final'][0] - GM[g]['final'][1] for g in order], float)
        assert ((m > 0).astype(int) == P['win']).all()
        y.append(m)
        for k in ('additive', 'usage', 'fit', 'full'):
            X.setdefault(k, []).append(logit(P[k]))
    y = np.concatenate(y); X = {k: np.concatenate(v) for k, v in X.items()}
    out['2022_23'] = dict(n=len(y), arms=compare(y, X, [('usage', 'additive'), ('full', 'additive'), ('fit', 'additive')], rng))
    show('2022-23 multi-season latent test, margin correlation', len(y), out['2022_23']['arms'])
    json.dump(out, open(OUT / 'margin_eval.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
