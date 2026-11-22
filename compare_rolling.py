"""Paired comparison of rolling_prior.py runs against a baseline run (same 423 games, same blocks).

  python compare_rolling.py rolling_pbp rolling_epv_onball rolling_epv_offball ...
Reads results/groupwar_t/<run>/block_*.npz (pull from the PC first). For each arm present in both runs: AUC, log
loss, and 95% bootstrap CIs (games resampled, 2000 draws) of the difference vs the baseline run's same arm.
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from masked_paths import ROOT

D = ROOT / 'results/groupwar_t'
ARMS = ['rapm_hist_trk', 'anchored_comp_hist_trk', 'anchored_comp_hist_trk_cal']


def load(run):
    P = {}
    for f in sorted((D / run).glob('block_*.npz')):
        for k, v in np.load(f).items():
            P.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in P.items()}


def main():
    base, runs = sys.argv[1], sys.argv[2:]
    B = load(base); win = B['win']
    rng = np.random.default_rng(0); bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    print(f'{len(win)} games; baseline {base}')
    for a in ARMS:
        print(f"  {base:28s} {a:28s} auc {roc_auc_score(win, B[a]):.4f} ll {ll(B[a]).mean():.4f}")
    for r in runs:
        R = load(r)
        assert (R['win'] == win).all(), r
        for a in ARMS:
            if a not in R:
                continue
            d = ll(R[a]) - ll(B[a])
            dl = np.percentile([d[b].mean() for b in bs], [2.5, 97.5])
            da = np.percentile([roc_auc_score(win[b], R[a][b]) - roc_auc_score(win[b], B[a][b]) for b in bs[:1000]], [2.5, 97.5])
            print(f"  {r:28s} {a:28s} auc {roc_auc_score(win, R[a]):.4f} ll {ll(R[a]).mean():.4f} "
                  f"| dAUC [{da[0]:+.4f}, {da[1]:+.4f}] dll [{dl[0]:+.4f}, {dl[1]:+.4f}]")


if __name__ == '__main__':
    main()
