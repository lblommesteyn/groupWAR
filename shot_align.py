"""Align play-by-play field-goal attempts to the tracking frame at release.

For each 2pt/3pt attempt: find the ball arriving at a rim near the PBP timestamp,
walk back to the last frame where the ball was within reach of the listed shooter,
and validate the release spot against the PBP shot location.
"""
import gzip, json, re, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT, ARCHIVE

HOOP_X = 41.75          # feet from midcourt to rim center
REACH = 4.0             # ball-to-shooter distance that counts as possession
FPS_MS = 200
OUT = ROOT / 'results/shots'


def load_pbp(z, base):
    pbp = {}
    for line in gzip.open(z.open(base + 'raw/stats.jsonl.gz'), 'rt'):
        r = json.loads(line)
        if r['event'] == 'PlayByPlay#show':
            pbp[r['metadata']['id']] = r  # later rows are corrections
    return sorted(pbp.values(), key=lambda r: r['timestamp'])


def load_frames(z, base):
    T, P, ball, players, team_of, on = [], [], [], [], {}, []
    for line in gzip.open(z.open(base + 'tracking.jsonl.gz'), 'rt'):
        f = json.loads(line)
        if not f.get('period'):
            continue
        T.append(f['timestamp_ms']); P.append(f['period'])
        b = f.get('ball'); ball.append(b['xyz'] if b and b.get('xyz') else [np.nan] * 3)
        pos = {}
        for side in ('home_players', 'away_players'):
            for p in f[side]:
                if p.get('xyz'):
                    pos[p['player_id']] = p['xyz'][:2]
                team_of[p['player_id']] = side[:4]
        players.append(pos)
        o = f.get('on_court') or {}
        on.append((tuple(o.get('home') or []), tuple(o.get('away') or [])))
    return np.array(T), np.array(P), np.array(ball, float), players, on


def track(players, T, pid, idx, max_gap_ms=600):
    """Player xy at frame idx; fill gaps <= max_gap_ms by linear interpolation."""
    if pid in players[idx]:
        return np.array(players[idx][pid]), False
    lo = idx
    while lo > 0 and pid not in players[lo] and T[idx] - T[lo] <= max_gap_ms:
        lo -= 1
    hi = idx
    while hi < len(T) - 1 and pid not in players[hi] and T[hi] - T[idx] <= max_gap_ms:
        hi += 1
    if pid in players[lo] and pid in players[hi] and T[hi] - T[lo] <= max_gap_ms:
        q = (T[idx] - T[lo]) / (T[hi] - T[lo])
        return (1 - q) * np.array(players[lo][pid]) + q * np.array(players[hi][pid]), True
    return None, False


def align_game(name):
    base = name.rsplit('tracking.jsonl.gz', 1)[0]
    gid = base.rstrip('/').rsplit('/', 1)[1]
    date = base.rstrip('/').rsplit('/', 2)[1]
    with zipfile.ZipFile(ARCHIVE) as z:
        pbp = load_pbp(z, base)
        T, P, B, players, on = load_frames(z, base)
    if len(T) == 0:
        return gid, [], {}
    rim = (np.abs(np.abs(B[:, 0]) - HOOP_X) < 3) & (np.abs(B[:, 1]) < 3) & (B[:, 2] > 8)
    rows, why = [], {}

    def fail(reason):
        why[reason] = why.get(reason, 0) + 1

    for e in pbp:
        m = e['metadata']
        if m['eventType'] not in ('2pt', '3pt'):
            continue
        made = 'MISS' not in m['description']
        base_row = dict(game_id=gid, date=date, event_id=m['id'], period=m['period'], clock=m['gameClock'],
                        shooter=m['personId'], team=m.get('team'), three=m['eventType'] == '3pt',
                        subtype=m['subType'], desc=m['description'], made=made,
                        pbp_x=m['locX'] / 10, pbp_y=m['locY'] / 10)
        t = e['timestamp']
        w = np.flatnonzero((T > t - 3000) & (T < t + 1500) & (P == m['period']) & rim)
        if not len(w):
            fail('no_rim_arrival'); rows.append(base_row | dict(aligned=False, reason='no_rim_arrival')); continue
        k = w[np.argmin(np.abs(T[w] - t))]
        # first frame of that rim arrival
        while k > 0 and rim[k - 1] and T[k] - T[k - 1] <= 2 * FPS_MS:
            k -= 1
        hoop = np.array([np.sign(B[k, 0]) * HOOP_X, 0.0])
        r = None
        j = k - 1
        while j >= 0 and T[k] - T[j] <= 3000:
            if P[j] == m['period'] and m['personId'] in players[j] and np.isfinite(B[j, 0]):
                if np.linalg.norm(B[j, :2] - np.array(players[j][m['personId']])) <= REACH:
                    r = j; break
            j -= 1
        if r is None:
            fail('no_release'); rows.append(base_row | dict(aligned=False, reason='no_release')); continue
        sxy = np.array(players[r][m['personId']])
        dist_trk = float(np.linalg.norm(sxy - hoop))
        dist_pbp = float(np.hypot(m['locX'], m['locY']) / 10)
        # PBP loc: x lateral, y toward midcourt, tenths of feet, in the attacking hoop's frame
        along, lat = -np.sign(hoop[0]) * (sxy[0] - hoop[0]), -np.sign(hoop[0]) * sxy[1]
        loc_err = float(np.hypot(along - m['locY'] / 10, lat - m['locX'] / 10))
        ok = loc_err <= 4.5
        if not ok:
            fail('location_mismatch')
        rows.append(base_row | dict(aligned=bool(ok), reason=None if ok else 'location_mismatch',
                                   release_idx=int(r), rim_idx=int(k), flight_ms=int(T[k] - T[r]),
                                   pbp_offset_ms=int(t - T[k]), dist_trk=dist_trk, dist_pbp=dist_pbp, loc_err=loc_err))
    return gid, rows, why


if __name__ == '__main__':
    with zipfile.ZipFile(ARCHIVE) as z:
        names = sorted(n for n in z.namelist() if n.endswith('tracking.jsonl.gz'))
    if len(sys.argv) > 1:
        names = names[:int(sys.argv[1])]
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    with ProcessPoolExecutor(10) as ex:
        for i, (gid, rows, why) in enumerate(ex.map(align_game, names), 1):
            allrows += rows
            if i % 50 == 0:
                print(i, 'games', flush=True)
    with open(OUT / 'alignment.jsonl', 'w') as f:
        for r in allrows:
            f.write(json.dumps(r) + '\n')
    a = np.array([r['aligned'] for r in allrows])
    print(f'{len(allrows)} attempts, aligned {a.mean():.1%}')
    reasons = {}
    for r in allrows:
        reasons[r.get('reason')] = reasons.get(r.get('reason'), 0) + 1
    print(reasons)
