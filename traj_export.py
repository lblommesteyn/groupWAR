"""Half-court possession trajectories for the defensive world model (world_model.py).

Same possession definition as archetypes.py (holder runs, half-court part only, >= 4 s), thinned to 2.5 Hz, at
most MAX_T frames (24 s). Coordinates are rotated so the offense always attacks the rim at (+41.75, 0).
Per possession:
  pos [T, 10, 2]  offense slots 0-4 then defense 5-9 (feet), ball [T, 3], holder [T] (offense slot or -1),
  shot clock [T] (s, NaN if unknown), off/def player ids [5], off/def team ids, game id, date, season, period.
Output: results/world/traj_{season}.npz (padded to MAX_T with a length vector).   python traj_export.py  (Mac)
"""
import gzip, json, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from archetypes import ARCHIVES, HOOP_X

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/world'
MAX_T = 60


def frames(z, base, max_age=1000):
    T, ball, pos, on, sc, per = [], [], [], [], [], []
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
        if len(h) != 5 or len(a) != 5 or not all(p in last and t - last[p][0] <= max_age for p in h + a):
            continue
        b = f.get('ball')
        T.append(t); ball.append(b['xyz'] if b and b.get('xyz') else [np.nan] * 3)
        pos.append(np.array([last[p][1] for p in h + a], float)); on.append(h + a)
        s = f.get('shot_clock_ms'); sc.append(s / 1000 if s is not None else np.nan); per.append(f['period'])
    return np.array(T), np.array(ball, float), pos, on, np.array(sc, float), np.array(per)


def game(task):
    path, base = task
    gid = base.rstrip('/').rsplit('/', 1)[1]
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read(base + 'metadata.json'))
        T, B, X, ON, SC, PER = frames(z, base)
    g = meta.get('game') or {}
    teams = ((g.get('homeTeam') or {}).get('teamId'), (g.get('awayTeam') or {}).get('teamId'))
    if len(T) < 500:
        return gid, []
    idx = np.flatnonzero(np.r_[True, np.diff(T // 400) > 0])
    T, B, SC, PER = T[idx], B[idx], SC[idx], PER[idx]; X = [X[i] for i in idx]; ON = [ON[i] for i in idx]
    n = len(T); holder = np.full(n, -1)
    for i in range(n):
        if np.isfinite(B[i, 0]) and B[i, 2] < 8:
            d = np.hypot(*(X[i] - B[i, :2]).T); j = int(np.argmin(d))
            if d[j] <= 4:
                holder[i] = j
    side = np.where(holder < 0, -1, (holder >= 5).astype(int))
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
        bx = B[seg, 0]; tail = bx[len(bx) // 2:]; tail = tail[np.isfinite(tail)]
        if len(tail) < 3:
            continue
        sgn = np.sign(np.median(tail))
        front = np.flatnonzero(np.isfinite(bx) & (np.sign(bx) == sgn))
        if not len(front):
            continue
        seg = seg[front[0]:][:MAX_T]
        if T[seg[-1]] - T[seg[0]] < 4000 or len(seg) < 8:
            continue
        order = list(range(5)) + list(range(5, 10)) if s == 0 else list(range(5, 10)) + list(range(5))
        P = np.stack([X[k][order] for k in seg]) * np.array([sgn, sgn])   # rotate 180 degrees if attacking -x
        Bb = B[seg].copy(); Bb[:, :2] *= np.array([sgn, sgn])
        hb = holder[seg]; hb = np.where(hb < 0, -1, np.where(s == 0, hb, hb - 5))
        hb = np.where((hb >= 0) & (hb < 5), hb, -1)
        lu = ON[seg[0]]; ids = [lu[k] for k in order]
        out.append(dict(pos=P.astype(np.float16), ball=Bb.astype(np.float16), holder=hb.astype(np.int8), sc=SC[seg].astype(np.float16),
                        off=ids[:5], deff=ids[5:], off_team=teams[s] or 0, def_team=teams[1 - s] or 0, period=int(PER[seg[0]]), t0=int(T[seg[0]])))
    return gid, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for season, path in ARCHIVES.items():
        with zipfile.ZipFile(path) as z:
            bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in z.namelist() if n.endswith('tracking.jsonl.gz'))
        rows, gids, dates = [], [], []
        with ProcessPoolExecutor(8) as ex:
            for k, (gid, rs) in enumerate(ex.map(game, [(path, b) for b in bases], chunksize=4)):
                day = next(b for b in bases if gid in b).rstrip('/').rsplit('/', 2)[1]
                rows += rs; gids += [gid] * len(rs); dates += [day] * len(rs)
                if k % 100 == 0:
                    print(season, k, len(bases), len(rows), flush=True)
        N = len(rows)
        pos = np.full((N, MAX_T, 10, 2), np.nan, np.float16); ball = np.full((N, MAX_T, 3), np.nan, np.float16)
        hold = np.full((N, MAX_T), -1, np.int8); sc = np.full((N, MAX_T), np.nan, np.float16); L = np.zeros(N, np.int16)
        for k, r in enumerate(rows):
            t = len(r['pos']); L[k] = t
            pos[k, :t] = r['pos']; ball[k, :t] = r['ball']; hold[k, :t] = r['holder']; sc[k, :t] = r['sc']
        np.savez_compressed(OUT / f'traj_{season}.npz', pos=pos, ball=ball, holder=hold, sc=sc, length=L,
                            off=np.array([r['off'] for r in rows], np.int64), deff=np.array([r['deff'] for r in rows], np.int64),
                            off_team=np.array([r['off_team'] for r in rows], np.int64), def_team=np.array([r['def_team'] for r in rows], np.int64),
                            period=np.array([r['period'] for r in rows], np.int8), t0=np.array([r['t0'] for r in rows], np.int64),
                            gid=np.array(gids), date=np.array(dates))
        print(season, len(bases), 'games', N, 'possessions', flush=True)


if __name__ == '__main__':
    main()
