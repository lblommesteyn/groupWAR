"""Block bootstrap for the headline DUNCAN-2 vs DARKO game numbers (AUDIT.md I5). Read-only on saved results.

The original CIs (ms_duncan2.py) resample games independently, ignoring that games in one 30-day block share one
fitted rating set and calibration. Here the resampling unit is the block (44 blocks). Given-lineup evaluation
(realised stints), as in the original.
Inputs: results/latent/duncan2_3000_1.0.npz (per-game p, concatenated in msb file order), results/latent/msb_*.npz.
Also the skill-synergy result (pop_synergy, AUDIT S1) from the per-block gains printed in logs/pop_synergy_all.log.
Output: results/corrected/block_bootstrap.json     python block_bootstrap.py   (PC)
"""
import ast, json, re
import numpy as np

from masked_paths import ROOT
from provenance import provenance

LAT = ROOT / 'results/latent'


def main():
    z = np.load(LAT / 'duncan2_3000_1.0.npz'); p_d2, w = z['p'], z['win']
    files = sorted(LAT.glob('msb_*.npz'))
    blocks, pdk = [], []
    for i, f in enumerate(files):
        P = np.load(f); blocks.append(np.full(len(P['gid']), i)); pdk.append(P['darko_cal'])
    blk = np.concatenate(blocks); pdk = np.concatenate(pdk)
    assert len(blk) == len(p_d2), (len(blk), len(p_d2))
    ll = lambda p: -np.log(np.clip(np.where(w == 1, p, 1 - p), 1e-9, 1))
    d = ll(p_d2) - ll(pdk)
    lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / np.clip(1 - v, 1e-6, 1))
    mg = z['margin']
    nb = len(files); rng = np.random.default_rng(0)
    ids = [np.flatnonzero(blk == b) for b in range(nb)]
    bd, br = [], []
    for _ in range(4000):
        s = np.concatenate([ids[b] for b in rng.integers(0, nb, nb)])
        bd.append(d[s].mean())
        br.append(np.corrcoef(lg(p_d2[s]), mg[s])[0, 1] - np.corrcoef(lg(pdk[s]), mg[s])[0, 1])
    gd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    res = dict(provenance=provenance(__file__, ['results/latent/duncan2_3000_1.0.npz', 'results/latent/ms_bench.json']),
               games=int(len(d)), blocks=nb, evaluation='given realised lineups (not pregame)',
               ll_duncan2_minus_darko=dict(mean=round(float(d.mean()), 5), game_ci=np.percentile(gd, [2.5, 97.5]).round(5).tolist(),
                                           block_ci=np.percentile(bd, [2.5, 97.5]).round(5).tolist()),
               margin_r_duncan2_minus_darko=dict(mean=round(float(np.corrcoef(lg(p_d2), mg)[0, 1] - np.corrcoef(lg(pdk), mg)[0, 1]), 4),
                                                 block_ci=np.percentile(br, [2.5, 97.5]).round(4).tolist()))
    # synergy: per-block SSE gains (x1e6) for each arm, as printed by pop_synergy.py
    log = ROOT / 'logs/pop_synergy_all.log'
    if log.exists():
        gains = []
        for line in log.read_text().splitlines():
            m = re.search(r"^(\d{4}-\d\d-\d\d) (\d{4}) .*SSE gain vs base \(x1e6\): (\{.*\})", line)
            if m:
                gains.append((int(m.group(2)), ast.literal_eval(m.group(3))))
        test = [g for s, g in gains if s not in (2021, 2022)]
        for arm in ('syn_1e+06', 'syn_300000'):
            v = np.array([g[arm] for g in test])
            bs = [v[rng.integers(0, len(v), len(v))].sum() for _ in range(4000)]
            res[f'synergy_{arm}_test_blocks'] = dict(blocks=len(v), total_gain_x1e6=round(float(v.sum()), 3),
                                                     block_ci_total=np.percentile(bs, [2.5, 97.5]).round(3).tolist(),
                                                     blocks_positive=int((v > 0).sum()))
    (ROOT / 'results/corrected').mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / 'results/corrected/block_bootstrap.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
