"""EPV v3 samples: every 5 Hz half-court frame with its 2 s movement history, in the MAE's input format.

Frames: results/holes/{gid}.npz (half-court, offense attacks +x): pos/vel [n,10,2], ball [n,2]; ball velocity by
differencing. A sample is frame j with >= 3 continuous frames ending at j (same possession, gaps <= 250 ms); its
window is the last min(10, run) frames, left-padded by repeating the earliest one.
Label: by possession (holes and epv_data share possessions() numbering): possession points y and luck-removed y2
(epv2_label.py); kind 2 if the frame is after the possession's last matched shot release, 1 if no shot, else 0.
Globals from the nearest v1 frame of the possession (game clock, period, margin, seconds into possession shifted
by dt; shot clock from the holes frame). v1idx = index of a v1 frame at the same moment (|dt| <= 100 ms) in the
concatenated v1 arrays, else -1, so v2 can be scored on exactly the same moments.
Output: results/epv/v3.npz  F [Nf,11,4] f16 frames, fi [N] index of "now" in F, L [N], glob [N,5], y, y2, kind,
gi (game index into games), v1idx, meta [N, 13] (poss k, t, holder, off5, def5)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT

H = ROOT / 'results/holes'
E = ROOT / 'results/epv'


def game(g):
    if not (H / f'{g}.npz').exists():
        return None
    h = np.load(H / f'{g}.npz'); d = np.load(E / 'data' / f'{g}.npz'); l = np.load(E / 'labels2' / f'{g}.npz')
    meta = h['meta']; t = meta[:, 1]; k = meta[:, 0]
    pos, vel, ball = h['pos'].astype(np.float32), h['vel'].astype(np.float32), h['ball'].astype(np.float32)
    bv = np.zeros_like(ball); bv[1:] = (ball[1:] - ball[:-1]) / np.maximum(np.diff(t)[:, None] / 1000, 0.1)
    F = np.concatenate([np.concatenate([pos, vel], -1), np.concatenate([ball, bv], -1)[:, None]], 1)
    ok = np.r_[False, (k[1:] == k[:-1]) & (np.diff(t) <= 250)]
    run = np.ones(len(t), int)
    for j in range(1, len(t)):
        if ok[j]:
            run[j] = run[j - 1] + 1
    bv[~ok] = 0; F[~ok, 10, 2:] = 0
    vm, vt = d['meta'], d['meta'][:, 1]
    by_k = {}
    for i in range(len(vm)):
        by_k.setdefault(int(vm[i, 0]), []).append(i)
    last_shot = {kk: max([l['shot_t'][i] for i in v] + [-1]) for kk, v in by_k.items()}
    rows = []
    for j in np.flatnonzero(run >= 3):
        cand = by_k.get(int(k[j]))
        if cand is None:
            continue  # possession dropped by v1 (no attack direction, > 4 pts)
        cand = np.array(cand); c = cand[np.argmin(np.abs(vt[cand] - t[j]))]
        dt = (t[j] - vt[c]) / 1000
        ls = last_shot[int(k[j])]
        kd = 1 if ls < 0 else (2 if t[j] > ls else 0)
        gl = d['glob'][c].astype(np.float32).copy()
        gl[0] = meta[j, 4] / 1000 if meta[j, 4] >= 0 else max(gl[0] - dt, 0)
        gl[1] = max(gl[1] - dt, 0); gl[4] = max(gl[4] + dt, 0)
        rows.append((j, min(10, run[j]), gl, d['y'][c], l['y2'][c], kd, c if abs(dt) <= 0.1 else -1,
                     [vm[c, 0], t[j], meta[j, 2]] + meta[j, 5:15].tolist()))
    if not rows:
        return None
    j, L, gl, y, y2, kind, c, mt = map(np.array, zip(*rows))
    return g, F.astype(np.float16), j, L, gl.astype(np.float32), y.astype(np.float32), y2.astype(np.float32), kind.astype(np.int8), c, mt


def main():
    games = sorted(f.stem for f in (E / 'labels2').glob('*.npz'))
    v1n = {g: len(np.load(E / 'labels2' / f'{g}.npz')['y2']) for g in games}
    v1off = dict(zip(games, np.cumsum([0] + [v1n[g] for g in games])[:-1]))
    out = {k: [] for k in ('F', 'fi', 'L', 'glob', 'y', 'y2', 'kind', 'gi', 'v1idx', 'meta')}
    nf = 0
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for i, r in enumerate(ex.map(game, games, chunksize=2)):
            if r is None:
                continue
            g, F, j, L, gl, y, y2, kind, c, mt = r
            gi = games.index(g)
            out['F'].append(F); out['fi'].append(j + nf); nf += len(F)
            for kk, v in (('L', L), ('glob', gl), ('y', y), ('y2', y2), ('kind', kind), ('v1idx', np.where(c >= 0, c + v1off[g], -1)), ('meta', mt)):
                out[kk].append(v)
            out['gi'].append(np.full(len(j), gi, np.int32))
            if i % 100 == 0:
                print(i, nf, sum(len(x) for x in out['fi']), flush=True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(E / 'v3.npz', games=np.array(games), **res)
    n = len(res['fi']); print(f'frames {nf}  samples {n}  kind {np.bincount(res["kind"])}  full 2 s history {np.mean(res["L"] == 10):.1%}')


if __name__ == '__main__':
    main()
