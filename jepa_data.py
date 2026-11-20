"""Windows for tracking JEPA / world-model pretraining and frozen probes, from the holes frames (results/holes).

Window: 20 consecutive half-court frames at 5 Hz (2 s context = frames 0-9, "now" = frame 9, 2 s future = 10-19)
inside one possession, consecutive timestamps <= 250 ms apart, stride 10 frames. Offense attacks +x.
Stored (results/jepa/windows.npz):
  X      [N, 20, 11, 4] float16  x, y, vx, vy for 5 attackers, 5 defenders, ball (ball velocity by differencing)
  ids    [N, 10] int32 player index (into pids), game [N] game index, date-sorted
  probe labels at "now": shot3, open3, rim3 (offense shot within 3 s), shot_q (xFG x pts of that shot, nan if
  none), A_now, A_fut (holes A at frame 9 / 19); baseline [N, 16] (holes_validate.baseline) and field [N, 11]
  features at frame 9 for the comparison probes.
Usage: python jepa_data.py
"""
import json, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import holes_validate as hv
from masked_paths import ROOT, ARCHIVE

H = ROOT / 'results/holes'
OUT = ROOT / 'results/jepa'
W, NOW, STRIDE = 20, 9, 10


def game(args):
    g, base, shots = args
    _, T = hv.timestamps(base)
    d = np.load(H / f'{g}.npz')
    meta = d['meta']; t = meta[:, 1]; k = meta[:, 0]
    pos = d['pos'].astype(np.float32); vel = d['vel'].astype(np.float32); ball = d['ball'].astype(np.float32)
    ok = np.r_[False, (k[1:] == k[:-1]) & (np.diff(t) <= 250)]  # frame i continues frame i-1
    starts, i = [], 0
    run = 0
    for j in range(len(t)):
        run = run + 1 if ok[j] else 1
        if run >= W and (j - W + 1) % 1 == 0:
            s = j - W + 1
            if not starts or s - starts[-1] >= STRIDE:
                starts.append(s)
    if not starts:
        return g, None
    st = np.array(starts)
    idx = st[:, None] + np.arange(W)[None]
    bv = np.zeros_like(ball); bv[1:] = (ball[1:] - ball[:-1]) / np.maximum(np.diff(t)[:, None] / 1000, 0.1)
    X = np.concatenate([np.concatenate([pos[idx], vel[idx]], -1),
                        np.concatenate([ball[idx], bv[idx]], -1)[:, :, None]], 2)  # [N,20,11,4]
    now = st + NOW
    shots = sorted((T[f['release_idx']], f['shooter'], f['d1'], f['dist_trk'], q) for f, q in shots if f['release_idx'] < len(T))
    stt = np.array([s[0] for s in shots]) if shots else np.zeros(0)
    lab = defaultdict(list)
    for n in now:
        off = meta[n, 5:10]
        j0, j1 = np.searchsorted(stt, t[n], 'right'), np.searchsorted(stt, t[n] + 3000, 'right')
        nxt = [s for s in shots[j0:j1] if s[1] in off]
        lab['shot3'].append(bool(nxt)); lab['open3'].append(any(s[2] > 6 for s in nxt))
        lab['rim3'].append(any(s[3] < 6 for s in nxt)); lab['shot_q'].append(nxt[0][4] if nxt else np.nan)
    A = d['A'].astype(np.float32)
    base_f = hv.baseline(pos[now], vel[now], ball[now], meta[now]).astype(np.float32)
    field = np.column_stack([d[c][now].astype(np.float32) for c in hv.FIELD])
    return g, dict(X=X.astype(np.float16), ids=meta[now, 5:15], date_t=t[now], A_now=A[now], A_fut=A[st + W - 1],
                   base=base_f, field=field, **{k_: np.array(v, np.float32) for k_, v in lab.items()})


def main():
    feats = {(r['game_id'], r['event_id']): r for r in map(json.loads, open(ROOT / 'results/shots/features.jsonl'))}
    by = defaultdict(list)
    for a, x in zip(map(json.loads, open(ROOT / 'results/shots/alignment.jsonl')), map(json.loads, open(ROOT / 'results/embed/shots_xfg.jsonl'))):
        f = feats.get((a['game_id'], a['event_id']))
        if f and f.get('trk_ok'):
            by[a['game_id']].append((f, x['xb'] * (3 if x['three'] else 2) if x.get('xb') is not None else np.nan))
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    dates = {}
    for l in open(ROOT / 'results/shots/alignment.jsonl'):
        r = json.loads(l); dates[r['game_id']] = r['date']
    gids = sorted((f.stem for f in H.glob('*.npz')), key=lambda g: (dates.get(g, '9999'), g))
    parts = {}
    with ProcessPoolExecutor(8) as ex:
        for g, r in ex.map(game, [(g, bases[g], by.get(g, [])) for g in gids], chunksize=4):
            if r is not None:
                parts[g] = r
    order = [g for g in gids if g in parts]
    pids = sorted({int(p) for g in order for p in np.unique(parts[g]['ids'])})
    pidx = {p: i for i, p in enumerate(pids)}
    out = {k: np.concatenate([parts[g][k] for g in order]) for k in parts[order[0]]}
    out['ids'] = np.vectorize(pidx.get)(out['ids']).astype(np.int32)
    out['game'] = np.concatenate([np.full(len(parts[g]['X']), i, np.int32) for i, g in enumerate(order)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'windows.npz', **out, pids=np.array(pids), games=np.array(order), dates=np.array([dates.get(g, '') for g in order]))
    print('windows', len(out['X']), 'games', len(order), 'players', len(pids), 'shot3 rate', out['shot3'].mean().round(3))


if __name__ == '__main__':
    main()
