"""Where BOWEN misses: distributions of the gap between where BOWEN expected each defender (scheme-conditioned spot)
and where he was, over every tracked half-court frame, out-of-fold (results/world/pred.npz, cross-fit by game).

A "miss" mixes model error with real positioning deviation; the human review (bowen_review) is what separates them.
Writes one {rows, meta} JSON per breakdown to results/bowen_review/miss/ (each carries this script's code):
  miss_summary    quantiles of the miss, scheme vs league baseline
  miss_hist       frames by miss size (1 ft bins, 30+ pooled)
  miss_court      mean miss and mean depth bias by the defender's actual spot (2.5 ft cells; x = ft from midcourt
                  toward the attacked rim at 41.75, y = ft from the middle of the floor)
  miss_situation  by defensive situation (from the defender's tracking features on that possession)
  miss_time       by seconds into the possession
  miss_ball       by the defender's distance from the ball
  miss_direction  direction of big misses (> 6 ft): toward the rim, away from it, sideways
  python bowen_miss_report.py   (Mac)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
W = ROOT / 'results/world'
OUT = ROOT / 'results/bowen_review/miss'
RIM = np.array([41.75, 0.0])


def main():
    code = Path(__file__).read_text()
    z = np.load(W / 'pred.npz')
    y, ms, mb = z['y'].astype(np.float32), z['ms'].astype(np.float32), z['mb'].astype(np.float32)
    poss, k, t = z['poss'], z['k'].astype(int), z['t'].astype(int)
    P = [np.load(ROOT / f'results/archetypes/poss_{s}.npz', allow_pickle=True) for s in (2026, 2027)]
    fd = np.concatenate([p['fd'] for p in P]); dfn = list(P[0]['def_f'])
    T = [np.load(W / f'traj_{s}.npz') for s in (2026, 2027)]
    ball = np.concatenate([x['ball'][:, :, :2] for x in T]).astype(np.float32)
    e, eb = np.hypot(*(y - ms).T), np.hypot(*(y - mb).T)
    depth = np.hypot(*(y - RIM).T) - np.hypot(*(ms - RIM).T)       # + = further from the rim than expected
    meta = lambda inp: [{'label': 'bowen_miss_report.py', 'value': code, 'type': 'code', 'lang': 'python'},
                        {'label': 'Input', 'value': inp},
                        {'label': 'Frames', 'value': f'{len(e):,} defender-frames, {len(np.unique(poss)):,} half-court possessions, 2025-26 and 2026-27, out-of-fold'}]
    src = 'results/world/pred.npz (BOWEN, 2-fold cross-fit by game)'
    out = {}
    out['miss_summary'] = [dict(stat=f'p{q}', percentile=q, scheme_ft=round(float(np.percentile(e, q)), 2),
                                baseline_ft=round(float(np.percentile(eb, q)), 2)) for q in (10, 25, 50, 75, 90, 95, 99)]
    out['miss_summary'].append(dict(stat='mean', percentile=None, scheme_ft=round(float(e.mean()), 2), baseline_ft=round(float(eb.mean()), 2)))
    edges = np.arange(0, 31, 1.0)
    hs = np.histogram(np.minimum(e, 30.0 - 1e-6), edges)[0]; hb = np.histogram(np.minimum(eb, 30.0 - 1e-6), edges)[0]
    out['miss_hist'] = [dict(miss_ft=int(a), label=f'{int(a)}-{int(a) + 1}' if a < 29 else '29+', scheme_share=round(float(s / len(e)), 5),
                             baseline_share=round(float(b / len(eb)), 5)) for a, s, b in zip(edges[:-1], hs, hb)]
    gx, gy = np.arange(0, 47.5 + 1e-6, 2.5), np.arange(-25, 25 + 1e-6, 2.5)
    ix, iy = np.digitize(y[:, 0], gx) - 1, np.digitize(y[:, 1], gy) - 1
    ok = (ix >= 0) & (ix < len(gx) - 1) & (iy >= 0) & (iy < len(gy) - 1)
    cell = ix[ok] * (len(gy) - 1) + iy[ok]; nC = (len(gx) - 1) * (len(gy) - 1)
    cnt = np.bincount(cell, minlength=nC); me = np.bincount(cell, e[ok], nC); md = np.bincount(cell, depth[ok], nC)
    out['miss_court'] = [dict(cell=f'{a}', x=float(gx[a // (len(gy) - 1)] + 1.25), y=float(gy[a % (len(gy) - 1)] + 1.25), frames=int(cnt[a]),
                              mean_miss_ft=round(float(me[a] / cnt[a]), 2), mean_depth_ft=round(float(md[a] / cnt[a]), 2))
                         for a in range(nC) if cnt[a] >= 2000]
    f = fd[poss, k]
    oh, pa, dr = f[:, dfn.index('on_handler')], f[:, dfn.index('paint')], f[:, dfn.index('d_rim')]
    sit = np.where(oh >= 0.4, 'On ball', np.where(pa >= 0.5, 'Interior', np.where(dr >= 22, 'Perimeter stay-home', 'Help / rotation')))
    out['miss_situation'] = []
    for s in ('On ball', 'Interior', 'Help / rotation', 'Perimeter stay-home'):
        m = sit == s
        out['miss_situation'].append(dict(situation=s, frames=int(m.sum()), share_of_frames=round(float(m.mean()), 3),
                                          median_ft=round(float(np.median(e[m])), 2), p90_ft=round(float(np.percentile(e[m], 90)), 2),
                                          baseline_median_ft=round(float(np.median(eb[m])), 2), mean_depth_ft=round(float(depth[m].mean()), 2),
                                          share_over_10ft=round(float((e[m] > 10).mean()), 3)))
    tb = np.minimum(t // 5, 11)
    out['miss_time'] = [dict(seconds=2 * b, label=f'{2 * b}-{2 * b + 2} s' if b < 11 else '22+ s', frames=int((tb == b).sum()),
                             median_ft=round(float(np.median(e[tb == b])), 2), p90_ft=round(float(np.percentile(e[tb == b], 90)), 2)) for b in range(12)]
    db = np.hypot(*(y - ball[poss, t]).T); okb = np.isfinite(db); bb = np.minimum(np.where(okb, db, 0) // 4, 9).astype(int)
    out['miss_ball'] = [dict(ball_ft=4 * b, label=f'{4 * b}-{4 * b + 4} ft' if b < 9 else '36+ ft', frames=int((okb & (bb == b)).sum()),
                             median_ft=round(float(np.median(e[okb & (bb == b)])), 2), p90_ft=round(float(np.percentile(e[okb & (bb == b)], 90)), 2))
                        for b in range(10)]
    d = y - ms; ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360; big = e > 6
    dirs = {'Toward the rim': (ang < 45) | (ang >= 315), 'Away from the rim': (ang >= 135) & (ang < 225),
            'Sideways': ((ang >= 45) & (ang < 135)) | ((ang >= 225) & (ang < 315))}
    out['miss_direction'] = [dict(direction=n, share_of_big_misses=round(float(m[big].mean()), 3)) for n, m in dirs.items()]
    out['miss_direction'].append(dict(direction='All big misses (> 6 ft), share of all frames', share_of_big_misses=round(float(big.mean()), 3)))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        json.dump(dict(rows=rows, meta=meta(src)), open(OUT / f'{name}.json', 'w'), allow_nan=False)
        print(name, len(rows))


if __name__ == '__main__':
    main()
