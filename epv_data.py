"""EPV training samples: 1 Hz frames inside possessions, offense attacking +x, label = possession points.

Possessions come from the ball-holder sequence (tracking_metrics.holders, momentary contacts dropped);
a possession is a run of same-team holding, with opposing runs shorter than 1 s ignored (deflections,
scrambles), so offensive rebounds stay in the same possession. Points = offense score change from
possession start until the next possession starts (covers free throws and and-ones), from the PBP
running score matched by timestamp.

Per sample (float16):  tok[11, 8] = x, y, vx, vy, z(ball only), holder, offense(1)/defense(0)/ball(-1), dist to rim
                       glob[5]   = shot clock, game clock, period, score margin (offense - defense), seconds into possession
Also stored: game_id, possession id, timestamp, holder pid, offense player ids, defense player ids, label.
Output shards: results/epv/data/{game_id}.npz
"""
import bisect, gzip, json, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT, ARCHIVE
from shot_align import HOOP_X
from tracking_metrics import holders
from player_games import attack_dirs

OUT = ROOT / 'results/epv/data'


def load(z, base):
    T, P, B, players, on, SC, GC = [], [], [], [], [], [], []
    for line in gzip.open(z.open(base + 'tracking.jsonl.gz'), 'rt'):
        f = json.loads(line)
        if not f.get('period'):
            continue
        T.append(f['timestamp_ms']); P.append(f['period'])
        b = f.get('ball'); B.append(b['xyz'] if b and b.get('xyz') else [np.nan] * 3)
        players.append({p['player_id']: p['xyz'][:2] for s in ('home_players', 'away_players') for p in f[s] if p.get('xyz')})
        o = f.get('on_court') or {}
        on.append((tuple(o.get('home') or []), tuple(o.get('away') or [])))
        SC.append(f.get('shot_clock_ms') if f.get('shot_clock_ms') is not None else np.nan)
        GC.append(f.get('game_clock_ms') if f.get('game_clock_ms') is not None else np.nan)
    return np.array(T), np.array(P), np.array(B, float), players, on, np.array(SC, float), np.array(GC, float)


def score_track(z, base):
    pbp = {}
    for line in gzip.open(z.open(base + 'raw/stats.jsonl.gz'), 'rt'):
        r = json.loads(line)
        if r['event'] == 'PlayByPlay#show':
            pbp[r['metadata']['id']] = (r['timestamp'], r['metadata'])
    ts, sc = [], []
    for t, m in sorted(pbp.values(), key=lambda x: x[0]):
        try:
            sc.append((int(m['scoreHome']), int(m['scoreAway']))); ts.append(t)
        except (KeyError, TypeError, ValueError):
            pass
    return ts, sc


def possessions(T, P, B, players, on):
    """Ball-holder sequence -> possessions (same-team runs merged, opposing runs < 1 s ignored)."""
    hold, segs = holders(T, P, B, players, on)
    segs = [s for s in segs if s[2] - s[1] >= 1]
    team = lambda i, p: 'home' if p in on[i][0] else 'away' if p in on[i][1] else None
    # possessions: merge same-team runs, ignore opposing runs < 1 s
    poss = []
    for p, i0, i1 in segs:
        tm = team(i0, p)
        if tm is None:
            continue
        if poss and poss[-1]['team'] == tm and P[i0] == poss[-1]['period']:
            poss[-1]['end'] = i1
        elif poss and poss[-1]['team'] != tm and T[i1] - T[i0] < 1000 and P[i0] == poss[-1]['period']:
            continue
        else:
            poss.append(dict(team=tm, start=i0, end=i1, period=P[i0]))
    return hold, poss


def game(task):
    base, shots = task
    gid = base.rstrip('/').rsplit('/', 1)[1]
    with zipfile.ZipFile(ARCHIVE) as z:
        T, P, B, players, on, SC, GC = load(z, base)
        sts, ssc = score_track(z, base)
    if len(T) == 0 or not sts:
        return gid, 0
    dirs = attack_dirs(shots)
    hold, poss = possessions(T, P, B, players, on)
    def score_at(t):
        j = bisect.bisect_right(sts, t) - 1
        return ssc[j] if j >= 0 else (0, 0)
    # assign each scoring change to the scoring team's latest possession that started before it
    # and ended at most 8 s earlier (PBP stamps can trail the inbound)
    pts_of = [0] * len(poss); unassigned = 0
    starts = {tm: [(T[ps['start']], k) for k, ps in enumerate(poss) if ps['team'] == tm] for tm in ('home', 'away')}
    prev = (0, 0)
    for t, s_ in zip(sts, ssc):
        d = (s_[0] - prev[0], s_[1] - prev[1]); prev = s_
        for side_, tm in ((0, 'home'), (1, 'away')):
            if d[side_] <= 0:
                continue
            lst = starts[tm]; j = bisect.bisect_right(lst, (t, 10 ** 9)) - 1
            if j >= 0 and t - T[poss[lst[j][1]]['end']] <= 8000:
                pts_of[lst[j][1]] += d[side_]
            else:
                unassigned += d[side_]
    toks, globs, meta, labels = [], [], [], []
    for k, ps in enumerate(poss):
        nxt = poss[k + 1]['start'] if k + 1 < len(poss) and poss[k + 1]['period'] == ps['period'] else None
        t0 = T[ps['start']]
        side = 0 if ps['team'] == 'home' else 1
        pts = pts_of[k]
        if pts > 4:
            continue  # merged possessions / corrections
        sg = dirs.get((ps['period'], ps['team']))
        if sg is None:
            continue
        last = nxt if nxt is not None else ps['end']
        for i in range(ps['start'], last, 5):
            h, a = on[i]
            off, de = (h, a) if side == 0 else (a, h)
            if len(off) != 5 or len(de) != 5 or not np.isfinite(B[i, 0]) or i < 1 or P[i - 1] != P[i]:
                continue
            if not all(p in players[i] and p in players[i - 1] for p in off + de):
                continue
            dt = max((T[i] - T[i - 1]) / 1000, 0.1)
            tok = np.zeros((11, 8), np.float32)
            for j, p in enumerate(off + de):
                x, y = np.array(players[i][p]) * sg; vx, vy = (np.array(players[i][p]) - players[i - 1][p]) * sg / dt
                tok[j] = [x, y, vx, vy, 0, float(hold[i] == p), 1.0 if j < 5 else 0.0, np.hypot(x - HOOP_X, y)]
            bx, by, bz = B[i, 0] * sg, B[i, 1] * sg, B[i, 2]
            bv = (B[i, :2] - B[i - 1, :2]) * sg / dt if np.isfinite(B[i - 1, 0]) else np.zeros(2)
            tok[10] = [bx, by, bv[0], bv[1], bz, 0, -1.0, np.hypot(bx - HOOP_X, by)]
            sc = score_at(T[i])
            margin = (sc[side] - sc[1 - side])
            toks.append(tok)
            globs.append([SC[i] / 1000 if np.isfinite(SC[i]) else 24.0, GC[i] / 1000 if np.isfinite(GC[i]) else 360.0,
                          P[i], margin, (T[i] - t0) / 1000])
            meta.append([k, T[i], hold[i] or 0] + list(off) + list(de))
            labels.append(pts)
    if not toks:
        return gid, 0
    print(gid, 'assigned', sum(pts_of), 'unassigned', unassigned, 'final', ssc[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{gid}.npz', tok=np.array(toks, np.float16), glob=np.array(globs, np.float32),
                        meta=np.array(meta, np.int64), y=np.array(labels, np.int8))
    return gid, len(toks)


if __name__ == '__main__':
    xf = [json.loads(l) for l in open(ROOT / 'results/embed/shots_xfg.jsonl')]
    by = {}
    for s in xf:
        if s['trk_ok']:
            by.setdefault(s['game_id'], []).append(s)
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in zz.namelist() if n.endswith('tracking.jsonl.gz'))
    tasks = [(b, by.get(b.rstrip('/').rsplit('/', 1)[1], [])) for b in bases]
    if len(sys.argv) > 1:
        tasks = tasks[:int(sys.argv[1])]
    tot = 0
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
        for i, (g, n) in enumerate(ex.map(game, tasks, chunksize=2), 1):
            tot += n
            if i % 100 == 0:
                print(i, tot, flush=True)
    print('samples', tot)
