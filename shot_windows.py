"""Shot-aligned tracking windows for the MAE encoder: the 10 half-court frames (2 s at 5 Hz) ending at release.

For every tracked shot (results/shots/features.jsonl, trk_ok), take the last holes frame at or before the release
time, take up to 10 consecutive frames (at least 3) (same possession, gaps <= 250 ms) with the shooter among the attackers.
Shots without such a window (transition, early-clock, tracking gaps) get no window; the shot model sees NaN.
Output: results/jepa/shot_windows.npz  X [N,10,11,4] float16, ids [N,10] (player index into windows.npz pids, -1
unknown), shooter_slot [N], L [N] frames present (left zero-padded), key game_id/event_id.   Usage: python shot_windows.py
"""
import json, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import holes_validate as hv
from masked_paths import ROOT, ARCHIVE

H = ROOT / 'results/holes'
OUT = ROOT / 'results/jepa'
W, WMIN = 10, 3


def game(args):
    g, base, shots = args
    if not (H / f'{g}.npz').exists():
        return g, []
    _, T = hv.timestamps(base)
    d = np.load(H / f'{g}.npz')
    meta = d['meta']; t = meta[:, 1]; k = meta[:, 0]
    pos = d['pos'].astype(np.float32); vel = d['vel'].astype(np.float32); ball = d['ball'].astype(np.float32)
    bv = np.zeros_like(ball); bv[1:] = (ball[1:] - ball[:-1]) / np.maximum(np.diff(t)[:, None] / 1000, 0.1)
    ok = np.r_[False, (k[1:] == k[:-1]) & (np.diff(t) <= 250)]
    run = np.zeros(len(t), int)
    for j in range(len(t)):
        run[j] = run[j - 1] + 1 if (j and ok[j]) else 1
    out = []
    for ev, ri, sh in shots:
        if ri >= len(T):
            continue
        j = np.searchsorted(t, T[ri], 'right') - 1
        if j < 0 or T[ri] - t[j] > 300 or run[j] < WMIN:
            continue
        off = list(meta[j, 5:10])
        if sh not in off:
            continue
        L = min(W, run[j]); idx = np.arange(j - L + 1, j + 1)
        X = np.zeros((W, 11, 4), np.float32)  # left-padded: the last L rows are frames ending at release
        X[W - L:] = np.concatenate([np.concatenate([pos[idx], vel[idx]], -1), np.concatenate([ball[idx], bv[idx]], -1)[:, None]], 1)
        out.append((ev, X.astype(np.float16), meta[j, 5:15].astype(np.int64), off.index(sh), L))
    return g, out


def main():
    by = defaultdict(list)
    for l in open(ROOT / 'results/shots/features.jsonl'):
        r = json.loads(l)
        if r.get('trk_ok') and r.get('release_idx') is not None:
            by[r['game_id']].append((r['event_id'], r['release_idx'], r['shooter']))
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    pids = np.load(OUT / 'windows.npz')['pids']; pidx = {int(p): i for i, p in enumerate(pids)}
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for g, res in ex.map(game, [(g, bases[g], s) for g, s in by.items() if g in bases], chunksize=4):
            rows += [(g,) + r for r in res]
    X = np.stack([r[2] for r in rows])
    ids = np.array([[pidx.get(int(p), -1) for p in r[3]] for r in rows], np.int32)
    np.savez(OUT / 'shot_windows.npz', X=X, ids=ids, shooter_slot=np.array([r[4] for r in rows], np.int8), L=np.array([r[5] for r in rows], np.int8),
             game_id=np.array([r[0] for r in rows]), event_id=np.array([r[1] for r in rows], np.int64))
    n = sum(len(v) for v in by.values())
    print(f'shots {n} with window {len(rows)} ({len(rows) / n:.2f})')


if __name__ == '__main__':
    main()
