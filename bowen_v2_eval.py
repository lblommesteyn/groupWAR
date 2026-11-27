"""BOWEN v2 evaluation: do live-only training (A) and a 3-option mixture head (B) fix the misses?

Compared on the SAME live-play frames (dead ball = shot clock frozen over the last two frames, the rule in
world_model.py), out-of-fold for every model (same game folds, seed 0):
  v1  results/world/pred.npz                      original: trained on all frames, single spot
  A   results/world/bowen_live/pred.npz           trained on live frames only, single spot
  B   results/world/bowen_live_mix3/pred.npz      trained on live frames only, 3-component mixture (expected spot =
                                                  most likely component)
Metrics: miss quantiles, x (along the court) and y (across) medians and 90th percentiles, share of misses > 15 ft,
mean negative log-likelihood of where the defender was (lower = better; for B the full mixture), by situation too.
Output: results/world/bowen_v2_eval.json     python bowen_v2_eval.py   (PC)
"""
import json
import numpy as np

from masked_paths import ROOT
from provenance import provenance

W = ROOT / 'results/world'
RIM = np.array([41.75, 0.0])


def key(z):
    return z['poss'].astype(np.int64) * 600 + z['t'].astype(np.int64) * 5 + z['k'].astype(np.int64)


def main():
    T = [np.load(W / f'traj_{s}.npz') for s in (2026, 2027)]
    sc = np.concatenate([x['sc'] for x in T]).astype(np.float32)
    P = [np.load(ROOT / f'results/archetypes/poss_{s}.npz', allow_pickle=True) for s in (2026, 2027)]
    fd = np.concatenate([p['fd'] for p in P]); dfn = list(P[0]['def_f'])
    models = {'v1': W / 'pred.npz', 'A_live': W / 'bowen_live/pred.npz', 'B_live_mix3': W / 'bowen_live_mix3/pred.npz'}
    Z = {m: np.load(f) for m, f in models.items()}
    keys = {m: key(z) for m, z in Z.items()}
    common = keys['v1']
    for m in ('A_live', 'B_live_mix3'):
        common = np.intersect1d(common, keys[m])
    z1 = Z['v1']; i1 = np.searchsorted(keys['v1'], common) if np.all(np.diff(keys['v1']) > 0) else None
    out = dict(models={m: str(f.relative_to(ROOT)) for m, f in models.items()}, results={})
    aligned = {}
    for m, z in Z.items():
        o = np.argsort(keys[m]); ks = keys[m][o]; ix = o[np.searchsorted(ks, common)]
        aligned[m] = (z, ix)
    z, ix = aligned['v1']
    poss, t, k = z['poss'][ix], z['t'][ix].astype(int), z['k'][ix].astype(int)
    s0, s1, s2 = sc[poss, t], sc[poss, np.maximum(t - 1, 0)], sc[poss, np.maximum(t - 2, 0)]
    dead = np.isfinite(s0) & np.isfinite(s1) & np.isfinite(s2) & (t >= 2) & (np.abs(s0 - s1) < 0.05) & (np.abs(s1 - s2) < 0.05)
    live = ~dead
    f = fd[poss, k]
    oh, pa, dr = f[:, dfn.index('on_handler')], f[:, dfn.index('paint')], f[:, dfn.index('d_rim')]
    sit = np.where(oh >= 0.4, 'On ball', np.where(pa >= 0.5, 'Interior', np.where(dr >= 22, 'Perimeter stay-home', 'Help / rotation')))
    out['frames'] = dict(common=int(len(common)), live=int(live.sum()), dead=int(dead.sum()))
    for m, (z, ix) in aligned.items():
        y = z['y'][ix].astype(np.float32); ms = z['ms'][ix].astype(np.float32)
        d = y - ms; e = np.hypot(*d.T)
        if 'nll_s' in z.files:
            nll = z['nll_s'][ix].astype(np.float32)
        else:
            ls = z['ls'][ix].astype(np.float32); nll = (d ** 2).sum(-1) / (2 * np.exp(ls)) + ls
        r = {}
        for lab, mk in (('live', live), ('dead', dead)):
            ee = e[mk]
            r[lab] = dict(median=round(float(np.median(ee)), 2), p75=round(float(np.percentile(ee, 75)), 2), p90=round(float(np.percentile(ee, 90)), 2),
                          p99=round(float(np.percentile(ee, 99)), 2), share_over_15ft=round(float((ee > 15).mean()), 4),
                          x_median=round(float(np.median(np.abs(d[mk, 0]))), 2), x_p90=round(float(np.percentile(np.abs(d[mk, 0]), 90)), 2),
                          y_median=round(float(np.median(np.abs(d[mk, 1]))), 2), y_p90=round(float(np.percentile(np.abs(d[mk, 1]), 90)), 2),
                          mean_nll=round(float(np.mean(nll[mk])), 3))
        r['live_by_situation'] = {s: dict(median=round(float(np.median(e[live & (sit == s)])), 2), p90=round(float(np.percentile(e[live & (sit == s)], 90)), 2),
                                          x_p90=round(float(np.percentile(np.abs(d[live & (sit == s), 0]), 90)), 2),
                                          mean_nll=round(float(np.mean(nll[live & (sit == s)])), 3))
                                  for s in ('On ball', 'Interior', 'Help / rotation', 'Perimeter stay-home')}
        out['results'][m] = r
        print(m, json.dumps(r['live']), flush=True)
    out['provenance'] = provenance(__file__, [str(f.relative_to(ROOT)) for f in models.values()])
    json.dump(out, open(W / 'bowen_v2_eval.json', 'w'), indent=1)
    print(json.dumps({m: r['live_by_situation'] for m, r in out['results'].items()}, indent=1))


if __name__ == '__main__':
    main()
