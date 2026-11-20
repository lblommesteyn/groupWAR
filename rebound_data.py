"""Missed field goals -> rebound samples: positions at release, where the ball was secured, who won it, what followed.

Per aligned, tracked miss (results/shots/features.jsonl, made = false):
  label         PBP rebound after the shot (next REBOUND event by id; latest correction of each id wins):
                oreb = 1 offensive, 0 defensive (team rebounds included)
  release       ten players (shooting team first, shooter at slot 0), positions/velocities/positional error at
                release, oriented so the shooting team attacks +x (hoop at +HOOP_X); interpolated tracks (holes.Tracks)
  secure        first frame after the rim frame with a ball holder: ball location, time since rim, holder, team
  value         points the shooting team scores from the rebound until the next opposing possession starts
                (second-chance points, credited to anyone), and if defensive, the points the opponent scores on the
                possession the rebound starts (possessions from epv_data; PBP score track, +4 s slack for PBP lag)
Output: results/rebounds/rebounds.pkl (list of dicts).
"""
import bisect, gzip, json, pickle, sys, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from masked_paths import ROOT, ARCHIVE
from epv_data import load, possessions
from holes import Tracks
from player_games import attack_dirs

OUT = ROOT / 'results/rebounds'
SLACK = 4000


def pbp(z, base):
    ev = {}
    for line in gzip.open(z.open(base + 'raw/stats.jsonl.gz'), 'rt'):
        r = json.loads(line)
        if r['event'] == 'PlayByPlay#show':
            ev[r['metadata']['id']] = (r['timestamp'], r['metadata'])
    return ev


def game(task):
    base, misses, xshots = task
    gid = base.rstrip('/').rsplit('/', 1)[1]
    with zipfile.ZipFile(ARCHIVE) as z:
        T, P, B, players, on, SC, GC = load(z, base)
        ev = pbp(z, base)
    if len(T) < 3 or not misses:
        return []
    ids = sorted(ev)
    sts, ssc = [], []
    for t, m in sorted(ev.values(), key=lambda x: x[0]):
        try:
            ssc.append((int(m['scoreHome']), int(m['scoreAway']))); sts.append(t)
        except (KeyError, TypeError, ValueError):
            pass
    score = lambda t: ssc[bisect.bisect_right(sts, t) - 1] if bisect.bisect_right(sts, t) else (0, 0)
    dirs = attack_dirs(xshots)
    hold, poss = possessions(T, P, B, players, on)
    pstart = [T[p['start']] for p in poss]
    tr = Tracks(T, players)
    out = []
    for s in misses:
        r, k = s['release_idx'], s['rim_idx']
        if k >= len(T) - 1:
            continue
        j = bisect.bisect_right(ids, s['event_id'])
        reb = None
        for e in ids[j:j + 6]:
            m = ev[e][1]
            if m.get('subType') in ('offensive', 'defensive') and 'REBOUND' in (m.get('description') or '').upper():
                reb = m; break
        if reb is None:
            continue
        side = s['team']; sg = dirs.get((s['period'], side))
        if sg is None:
            continue
        h, a = on[r]
        off, de = (h, a) if side == 'home' else (a, h)
        if len(off) != 5 or len(de) != 5 or s['shooter'] not in off:
            continue
        off = (s['shooter'],) + tuple(p for p in off if p != s['shooter'])
        if len(off) != 5 or len(set(off + de)) != 10:
            continue  # duplicated ids in the on-court list
        now = [tr.at(p, T[r]) for p in off + de]
        if any(x is None for x in now):
            continue
        lo = [tr.at(p, T[r] - 200) for p in off + de]; hi = [tr.at(p, T[r] + 200) for p in off + de]
        vel = np.zeros((10, 2))
        for q in range(10):
            a_, b_ = (lo[q][0] if lo[q] else now[q][0]), (hi[q][0] if hi[q] else now[q][0])
            span = (0.2 if lo[q] else 0) + (0.2 if hi[q] else 0)
            if span:
                vel[q] = (b_ - a_) * sg / span
        pos = np.array([x[0] for x in now]) * sg; pe = np.array([x[1] for x in now])
        # secure: first holder after the rim frame (within 6 s)
        sec = None
        for i in range(k + 1, len(T)):
            if T[i] - T[k] > 6000 or P[i] != P[k]:
                break
            if hold[i] is not None and np.isfinite(B[i, 0]):
                sec = i; break
        tm_of = lambda p: 'off' if p in off else 'def' if p in de else None
        oreb = int(reb['subType'] == 'offensive')
        rec = dict(game_id=gid, event_id=s['event_id'], date=s['date'], shooter=s['shooter'], three=s['three'],
                   dist=s['dist_trk'], subtype=s['subtype'], blocked='BLOCK' in (s.get('desc') or '').upper(),
                   oreb=oreb, rebounder=int(reb.get('personId') or 0), off=off, de=de,
                   pos=pos.astype(np.float32), vel=vel.astype(np.float32), pe=pe.astype(np.float32),
                   ball_rel=(B[r, :2] * sg).astype(np.float32), flight_ms=int(T[k] - T[r]),
                   secure_xy=None if sec is None else (B[sec, :2] * sg).astype(np.float32),
                   secure_ms=None if sec is None else int(T[sec] - T[k]),
                   secure_team=None if sec is None else tm_of(hold[sec]), secure_pid=None if sec is None else hold[sec])
        # what followed: score changes until the next possession of the other team starts
        t0 = T[sec] if sec is not None else T[k]
        si = 0 if side == 'home' else 1
        nxt_other = next((pstart[q] for q in range(bisect.bisect_right(pstart, t0), len(poss))
                          if poss[q]['team'] != (side if oreb else ('away' if side == 'home' else 'home'))), None)
        if nxt_other is not None:
            s0, s1 = score(t0 - 500), score(nxt_other + SLACK)
            if oreb:
                rec['pts_after'] = s1[si] - s0[si]
            else:
                rec['pts_after'] = s1[1 - si] - s0[1 - si]
            rec['after_ms'] = int(nxt_other - t0)
        out.append(rec)
    return out


def main():
    feats = defaultdict(list)
    for f in map(json.loads, open(ROOT / 'results/shots/features.jsonl')):
        if f.get('trk_ok') and not f['made'] and f.get('release_idx') is not None:
            feats[f['game_id']].append(f)
    xs = defaultdict(list)
    for s in map(json.loads, open(ROOT / 'results/embed/shots_xfg.jsonl')):
        if s['trk_ok']:
            xs[s['game_id']].append(s)
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in zz.namelist() if n.endswith('tracking.jsonl.gz'))
    tasks = [(b, feats.get(g, []), xs.get(g, [])) for b in bases for g in [b.rstrip('/').rsplit('/', 1)[1]]]
    if len(sys.argv) > 1:
        tasks = tasks[:int(sys.argv[1])]
    res = []
    with ProcessPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(game, tasks), 1):
            res += r
            if i % 50 == 0:
                print(i, len(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / 'rebounds.pkl', 'wb'))
    o = np.array([r['oreb'] for r in res]); st = [r['secure_team'] for r in res]
    agree = np.mean([(t == 'off') == bool(y) for t, y in zip(st, o) if t is not None])
    pa = [r for r in res if 'pts_after' in r]
    print(f'misses {len(res)} oreb rate {o.mean():.3f} tracking/PBP agreement {agree:.3f}; '
          f'pts after OREB {np.mean([r["pts_after"] for r in pa if r["oreb"]]):.3f}, '
          f'opp pts after DREB {np.mean([r["pts_after"] for r in pa if not r["oreb"]]):.3f}')


if __name__ == '__main__':
    main()
