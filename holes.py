"""Holes field over the whole NBA archive at 5 Hz, live half-court frames.

Frames: inside a possession (epv_data.possessions), offense attacking +x (attack_dirs), all ten on-court players
locatable (observed, or interpolated across a gap <= 1 s), ball in the frontcourt and both teams' mean x > 12 ft
(chance set up, not transition). Velocity: centred difference of the interpolated tracks over +-0.2 s.
Per frame stored (float16): summaries from holes_field.Field.summarize; ids, holder, timestamps (int64).
Output: results/holes/{game_id}.npz.  Usage: python holes.py [n_games] [cpu_workers] [device]
"""
import json, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from masked_paths import ROOT, ARCHIVE
from epv_data import load, possessions
from player_games import attack_dirs
from holes_field import Field

OUT = ROOT / 'results/holes'


MAX_GAP = 1000.0  # ms; longer player gaps are not interpolated
PE_OBS, PE_RATE = 0.5, 3.0  # ft; + ft per second away from the nearest real sample


class Tracks:
    """Per-player time series; position at any time by linear interpolation across gaps <= MAX_GAP."""

    def __init__(self, T, players):
        acc = {}
        for t, fr in zip(T, players):
            for p, xy in fr.items():
                acc.setdefault(p, ([], []))[0].append(t); acc[p][1].append(xy)
        self.d = {p: (np.array(ts, float), np.array(xy, float)) for p, (ts, xy) in acc.items()}

    def at(self, p, t):
        """-> (xy, pe) or None."""
        if p not in self.d:
            return None
        ts, xy = self.d[p]
        j = np.searchsorted(ts, t)
        if j < len(ts) and ts[j] == t:
            return xy[j], PE_OBS
        if j == 0 or j == len(ts) or ts[j] - ts[j - 1] > MAX_GAP:
            return None
        q = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
        return (1 - q) * xy[j - 1] + q * xy[j], PE_OBS + PE_RATE * min(t - ts[j - 1], ts[j] - t) / 1000


def frames(task):
    base, shots = task
    gid = base.rstrip('/').rsplit('/', 1)[1]
    with zipfile.ZipFile(ARCHIVE) as z:
        T, P, B, players, on, SC, GC = load(z, base)
    if len(T) < 3:
        return gid, None
    dirs = attack_dirs(shots)
    hold, poss = possessions(T, P, B, players, on)
    tr = Tracks(T, players)
    pos, vel, ball, pes, meta = [], [], [], [], []
    for k, ps in enumerate(poss):
        sg = dirs.get((ps['period'], ps['team']))
        if sg is None:
            continue
        side = 0 if ps['team'] == 'home' else 1
        for i in range(ps['start'], ps['end'] + 1):
            if not np.isfinite(B[i, 0]):
                continue
            bx = B[i, :2] * sg
            if bx[0] <= 0:
                continue
            h, a = on[i]
            off, de = (h, a) if side == 0 else (a, h)
            if len(off) != 5 or len(de) != 5:
                continue
            ids = off + de; t = T[i]
            now = [tr.at(p, t) for p in ids]
            if any(r is None for r in now):
                continue
            x = np.array([r[0] for r in now]) * sg
            if x[:5, 0].mean() < 12 or x[5:, 0].mean() < 12:
                continue
            lo = [tr.at(p, t - 200) for p in ids]; hi = [tr.at(p, t + 200) for p in ids]
            v = np.zeros((10, 2))
            for j in range(10):
                a_, b_ = (lo[j][0] if lo[j] else now[j][0]), (hi[j][0] if hi[j] else now[j][0])
                span = (0.2 if lo[j] else 0) + (0.2 if hi[j] else 0)
                if span:
                    v[j] = (b_ - a_) * sg / span
            pos.append(x); vel.append(v); ball.append(bx); pes.append([r[1] for r in now])
            meta.append([k, t, hold[i] or 0, P[i], SC[i] if np.isfinite(SC[i]) else -1] + list(ids))
    if not pos:
        return gid, None
    f32 = lambda a: np.array(a, np.float32)
    return gid, (f32(pos), f32(vel), f32(ball), f32(pes), np.array(meta, np.int64))


def run(field, data, bs=512):
    pos, vel, ball, pe, meta = data
    outs = {}
    for s in range(0, len(pos), bs):
        o = field.summarize(*(torch.from_numpy(a[s:s + bs]).to(field.dev) for a in (pos, vel, ball, pe)))
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v).astype(np.float16) for k, v in outs.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    dev = sys.argv[3] if len(sys.argv) > 3 else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    xf = [json.loads(l) for l in open(ROOT / 'results/embed/shots_xfg.jsonl')]
    by = {}
    for s in xf:
        if s['trk_ok']:
            by.setdefault(s['game_id'], []).append(s)
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(nm.rsplit('tracking.jsonl.gz', 1)[0] for nm in zz.namelist() if nm.endswith('tracking.jsonl.gz'))
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(b, by.get(b.rstrip('/').rsplit('/', 1)[1], [])) for b in bases]
    tasks = [t for t in tasks if not (OUT / f"{t[0].rstrip('/').rsplit('/', 1)[1]}.npz").exists()]
    if n:
        tasks = tasks[:n]
    field = Field(dev); t0 = time.time(); tot = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (gid, data) in enumerate(ex.map(frames, tasks), 1):
            if data is None:
                print(gid, 'no frames', flush=True); continue
            res = run(field, data)
            np.savez_compressed(OUT / f'{gid}.npz', meta=data[4], ball=data[2].astype(np.float16), pe=data[3].astype(np.float16),
                                pos=data[0].astype(np.float16), vel=data[1].astype(np.float16), **res)
            tot += len(data[0])
            print(f'{i}/{len(tasks)} {gid} frames {len(data[0])} A mean {res["A"].astype(float).mean():.0f} '
                  f'total {tot} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
