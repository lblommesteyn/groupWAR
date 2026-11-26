"""Possession-level player roles from tracking, for fluid archetypes (archetype_mix.py clusters and aggregates).

Both archives (2025-26 Game Pulse, 2026-27 CourtSketch converted). Per frame: ball holder = nearest on-court player
within 4 ft of the ball with the ball below 8 ft. A possession is a run of frames where every holder belongs to one
team (holder-less gaps up to 3 s are bridged), cropped to its half-court part (from the ball crossing into the
half it ends in) and kept if that part lasts at least 4 s. Frames are thinned to 2.5 Hz.
Per offensive player and possession: on-ball share, touches, mean distance to ball / to the attacked rim, share of
time in the paint, in the corners, beyond the arc above the break, near the ball handler (<= 6 ft, off ball:
screens, handoffs), and mean speed.
Per defender: mean distance to own rim, paint share, share of frames as the ball handler's nearest defender, mean
distance to the nearest attacker, share of frames closer to the rim than that attacker ("between man and basket"),
and mean speed.
Output: results/archetypes/poss_{season}.npz  (python archetypes.py, Mac; about 5 minutes)
"""
import gzip, json, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/archetypes'
ARCHIVES = {2026: Path.home() / 'Downloads/NBA-2025-26-Available-Tracking.zip',
            2027: Path.home() / 'Downloads/NBA-2026-27-CourtSketch-Tracking.zip'}
HOOP_X = 41.75
OFF_F = ['onball', 'touches', 'd_ball', 'd_rim', 'paint', 'corner', 'above_break', 'near_handler', 'speed']
DEF_F = ['d_rim', 'paint', 'on_handler', 'd_man', 'between', 'speed']


def frames(z, base, max_age=1000):
    """Frames with all ten on-court players located; a player missing from a frame keeps his last position for up
    to max_age ms (the 2025-26 feed sends positions in fragments)."""
    T, ball, pos, on = [], [], [], []
    last = {}
    for line in gzip.open(z.open(base + 'tracking.jsonl.gz'), 'rt'):
        f = json.loads(line)
        if not f.get('period'):
            continue
        t = f['timestamp_ms']
        for side in ('home_players', 'away_players'):
            for p in f[side]:
                if p.get('xyz'):
                    last[p['player_id']] = (t, p['xyz'][:2])
        o = f.get('on_court') or {}
        h, a = tuple(o.get('home') or []), tuple(o.get('away') or [])
        if len(h) != 5 or len(a) != 5:
            continue
        if not all(p in last and t - last[p][0] <= max_age for p in h + a):
            continue
        b = f.get('ball')
        T.append(t); ball.append(b['xyz'] if b and b.get('xyz') else [np.nan] * 3)
        pos.append(np.array([last[p][1] for p in h + a], float)); on.append(h + a)
    return np.array(T), np.array(ball, float), pos, on


def game(task):
    path, base = task
    gid = base.rstrip('/').rsplit('/', 1)[1]
    with zipfile.ZipFile(path) as z:
        T, B, X, ON = frames(z, base)
    if len(T) < 500:
        return gid, []
    keep = np.r_[True, np.diff(T // 400) > 0]  # thin to ~2.5 Hz
    idx = np.flatnonzero(keep)
    T, B = T[idx], B[idx]; X = [X[i] for i in idx]; ON = [ON[i] for i in idx]
    n = len(T)
    holder = np.full(n, -1)
    for i in range(n):
        if np.isfinite(B[i, 0]) and B[i, 2] < 8:
            d = np.hypot(*(X[i] - B[i, :2]).T)
            j = int(np.argmin(d))
            if d[j] <= 4:
                holder[i] = j
    side = np.where(holder < 0, -1, (holder >= 5).astype(int))  # 0 home has ball, 1 away
    out, i = [], 0
    while i < n:
        if side[i] < 0:
            i += 1; continue
        s, j, last = side[i], i, i
        while j < n and (side[j] == s or (side[j] < 0 and T[j] - T[last] <= 3000)):
            if side[j] == s:
                last = j
            if ON[j] != ON[i]:
                break
            j += 1
        seg = np.arange(i, last + 1); i = last + 1
        bx = B[seg, 0]
        tail = bx[len(bx) // 2:]; tail = tail[np.isfinite(tail)]
        if len(tail) < 3:
            continue
        hx = HOOP_X * np.sign(np.median(tail))
        front = np.flatnonzero(np.isfinite(bx) & (np.sign(bx) == np.sign(hx)))
        if not len(front):
            continue
        seg = seg[front[0]:]                            # the half-court part: from the ball crossing half
        if T[seg[-1]] - T[seg[0]] < 4000 or len(seg) < 8:
            continue
        off = slice(0, 5) if s == 0 else slice(5, 10); de = slice(5, 10) if s == 0 else slice(0, 5)
        P = np.stack([X[k] for k in seg])              # [t, 10, 2]
        O, D = P[:, off], P[:, de]
        rim = np.array([hx, 0.0])
        drim_o = np.hypot(*(O - rim).transpose(2, 0, 1)); drim_d = np.hypot(*(D - rim).transpose(2, 0, 1))
        hb = holder[seg] - (0 if s == 0 else 5)        # holder index within offense, or out of range
        has = (hb >= 0) & (hb < 5)
        ballxy = B[seg, :2]
        dball = np.hypot(*(O - ballxy[:, None, :]).transpose(2, 0, 1))
        onb = np.zeros((len(seg), 5)); onb[np.flatnonzero(has), hb[has]] = 1
        touches = (np.diff(np.vstack([np.zeros(5), onb]), axis=0) > 0).sum(0)
        dx = np.abs(O[..., 0] - hx); ay = np.abs(O[..., 1])
        paint = (dx < 19) & (ay < 8)
        corner = (ay > 21) & (dx < 14)
        abv = (drim_o > 23.75) & ~corner & (dx < 47)
        hpos = np.where(has[:, None], O[np.arange(len(seg)), np.clip(hb, 0, 4)], np.nan)
        dh = np.hypot(*(O - hpos[:, None, :]).transpose(2, 0, 1))
        nearh = (dh <= 6) & (onb == 0)
        dt = np.maximum(np.diff(T[seg]) / 1000, 1e-3)[:, None]
        sp_o = np.r_[np.zeros((1, 5)), np.hypot(*np.diff(O, axis=0).transpose(2, 0, 1)) / dt]
        sp_d = np.r_[np.zeros((1, 5)), np.hypot(*np.diff(D, axis=0).transpose(2, 0, 1)) / dt]
        dd = np.hypot(*(D[:, :, None, :] - O[:, None, :, :]).transpose(3, 0, 1, 2))   # [t, def, off]
        man = dd.argmin(2); dman = dd.min(2)
        manrim = np.take_along_axis(drim_o, man, 1)
        between = drim_d < manrim
        onh = np.zeros((len(seg), 5))
        for t in np.flatnonzero(has):
            onh[t, int(np.argmin(dd[t, :, hb[t]]))] = 1
        fo = np.column_stack([onb.mean(0), touches, np.nanmean(dball, 0), drim_o.mean(0), paint.mean(0), corner.mean(0),
                              abv.mean(0), nearh.mean(0), sp_o[1:].mean(0)])
        fd = np.column_stack([drim_d.mean(0), ((np.abs(D[..., 0] - hx) < 19) & (np.abs(D[..., 1]) < 8)).mean(0), onh.mean(0),
                              dman.mean(0), between.mean(0), sp_d[1:].mean(0)])
        lineup = ON[seg[0]]
        offp = lineup[:5] if s == 0 else lineup[5:]; defp = lineup[5:] if s == 0 else lineup[:5]
        out.append((int(T[seg[0]]), int(T[seg[-1]] - T[seg[0]]), int(s), offp, defp, fo.astype(np.float32), fd.astype(np.float32)))
    return gid, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for season, path in ARCHIVES.items():
        with zipfile.ZipFile(path) as z:
            bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in z.namelist() if n.endswith('tracking.jsonl.gz'))
        G, T0, DUR, SIDE, OP, DP, FO, FD, DATE = [], [], [], [], [], [], [], [], []
        with ProcessPoolExecutor(8) as ex:
            for k, (gid, rows) in enumerate(ex.map(game, [(path, b) for b in bases], chunksize=4)):
                day = next(b for b in bases if gid in b).rstrip('/').rsplit('/', 2)[1]
                for t0, dur, s, op, dp, fo, fd in rows:
                    G.append(gid); DATE.append(day); T0.append(t0); DUR.append(dur); SIDE.append(s)
                    OP.append(op); DP.append(dp); FO.append(fo); FD.append(fd)
                if k % 100 == 0:
                    print(season, k, len(bases), len(G), flush=True)
        np.savez_compressed(OUT / f'poss_{season}.npz', gid=np.array(G), date=np.array(DATE), t0=np.array(T0),
                            dur=np.array(DUR), side=np.array(SIDE), off=np.array(OP, np.int64), deff=np.array(DP, np.int64),
                            fo=np.array(FO), fd=np.array(FD), off_f=np.array(OFF_F), def_f=np.array(DEF_F))
        print(season, len(bases), 'games', len(G), 'possessions', flush=True)


if __name__ == '__main__':
    main()
