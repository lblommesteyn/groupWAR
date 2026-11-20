"""Event-level tracking metrics per (game, player): passes, drives, cuts, screens,
closeouts, help rotations, offensive-rebound crashes, transition shots.

Builds a ball-holder sequence (nearest on-court player within 3.5 ft, ball below 8 ft,
carried through <=0.6 s ball dropouts) and derives events from it. Shots are linked
by their aligned release frame. Output: results/metrics/event_counts.pkl
"""
import json, pickle, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT, ARCHIVE
from shot_align import OUT as SHOTS, HOOP_X, load_frames
from player_games import attack_dirs

OUT = ROOT / 'results/metrics'
HOLD_R, HOLD_Z = 3.5, 8.0


def holders(T, P, B, players, on):
    n = len(T); hold = [None] * n; last, last_t = None, -1e18
    for i in range(n):
        h, a = on[i]
        if np.isfinite(B[i, 0]):
            best, bd = None, HOLD_R
            if B[i, 2] < HOLD_Z:
                for p in h + a:
                    if p in players[i]:
                        d = np.hypot(players[i][p][0] - B[i, 0], players[i][p][1] - B[i, 1])
                        if d < bd:
                            best, bd = p, d
            hold[i] = best
            last, last_t = best, T[i]
        elif last is not None and T[i] - last_t <= 600:
            hold[i] = last
    segs, cur = [], None  # (pid, start, end)
    for i, p in enumerate(hold):
        if p is not None and cur and cur[0] == p and i - cur[2] <= 3 and P[i] == P[cur[2]]:
            cur[2] = i
        elif p is not None:
            cur = [p, i, i]; segs.append(cur)
    return hold, segs


def team_of(on, i, p):
    return 'home' if p in on[i][0] else 'away' if p in on[i][1] else None


def new():
    return defaultdict(float)


def game(task):
    base, shots = task
    with zipfile.ZipFile(ARCHIVE) as z:
        T, P, B, players, on = load_frames(z, base)
    C = defaultdict(new)
    if len(T) == 0:
        return base, {}
    dirs = attack_dirs([s for s in shots if s.get('trk_ok')])
    hold, segs = holders(T, P, B, players, on)
    rel = {(s['shooter'], s['release_idx']) for s in shots}
    # drop momentary contacts (<0.2 s) unless the segment contains a linked shot release
    segs = [sg for sg in segs if sg[2] - sg[1] >= 1 or any((sg[0], r) in rel for r in range(sg[1] - 2, sg[2] + 4))]
    seg_team = [team_of(on, s[1], s[0]) for s in segs]

    def hoop(i, team):
        sg = dirs.get((P[i], team))
        return None if sg is None else np.array([sg * HOOP_X, 0.0])

    def frontcourt(i, team, back=10):
        sg = dirs.get((P[i], team)); j = i - back
        return (sg is not None and j >= 0 and P[j] == P[i] and np.isfinite(B[i, 0]) and np.isfinite(B[j, 0])
                and B[i, 0] * sg > 0 and B[j, 0] * sg > 0)

    def xy(i, p):
        v = players[i].get(p)
        return None if v is None else np.asarray(v)

    # shots -> segment index
    shot_at = {}
    for s in shots:
        r = s['release_idx']
        for k, sg in enumerate(segs):
            if sg[0] == s['shooter'] and sg[1] - 2 <= r <= sg[2] + 3:
                shot_at[k] = s; break
    for k, sg in enumerate(segs):
        C[sg[0]]['touches'] += 1
        C[sg[0]]['hold_s'] += (T[sg[2]] - T[sg[1]]) / 1000
    # passes, lost passes, potential assists, transition
    poss_start = {}
    for k in range(len(segs)):
        if k == 0 or seg_team[k] != seg_team[k - 1] or P[segs[k][1]] != P[segs[k - 1][2]]:
            poss_start[k] = T[segs[k][1]]
        else:
            poss_start[k] = poss_start[k - 1]
    for k in range(len(segs) - 1):
        a, b = segs[k], segs[k + 1]
        if k in shot_at or P[a[2]] != P[b[1]] or T[b[1]] - T[a[2]] > 2500 or a[0] == b[0]:
            continue
        if seg_team[k] == seg_team[k + 1]:
            C[a[0]]['passes'] += 1; C[b[0]]['catches'] += 1
            s = shot_at.get(k + 1)
            if s and T[s['release_idx']] - T[b[1]] <= 3000:
                x = s['xb'] if s.get('xb') is not None else s['xa']
                C[a[0]]['pot_ast'] += 1; C[a[0]]['pot_ast_x'] += x; C[a[0]]['pot_ast_made'] += s['made']
        elif T[b[1]] - T[a[2]] <= 1200:
            mid = B[a[2]:b[1] + 1]
            near_rim = np.any((np.abs(np.abs(mid[:, 0]) - HOOP_X) < 6) & (np.abs(mid[:, 1]) < 6) & (mid[:, 2] > 7))
            if not near_rim:
                C[a[0]]['lost_pass'] += 1
    for k, s in shot_at.items():
        if T[s['release_idx']] - poss_start[k] <= 7000:
            C[s['shooter']]['trans_shots'] += 1
        C[s['shooter']]['linked_shots'] += 1
    # drives
    for k, (p, i0, i1) in enumerate(segs):
        tm = seg_team[k]; hp = hoop(i0, tm) if tm else None
        if hp is None:
            continue
        dist = [(i, np.linalg.norm(xy(i, p) - hp)) for i in range(i0, i1 + 1) if xy(i, p) is not None]
        start = end = None
        for i, d in dist:
            if d >= 20:
                start = i
            elif start is not None and d <= 10 and T[i] - T[start] <= 3000:
                end = i; break
        if end is None or not frontcourt(start, tm):
            continue
        opp = on[start][1] if tm == 'home' else on[start][0]
        Ds, De = {q: xy(start, q) for q in opp}, {q: xy(end, q) for q in opp}
        C[p]['drives'] += 1
        if all(v is not None for v in Ds.values()) and all(v is not None for v in De.values()):
            ps = xy(start, p)
            onball = min(opp, key=lambda q: np.linalg.norm(Ds[q] - ps))
            C[onball]['beaten'] += 1
            near_s = sum(np.linalg.norm(Ds[q] - hp) < 12 for q in opp)
            near_e = sum(np.linalg.norm(De[q] - hp) < 12 for q in opp)
            C[p]['collapse'] += near_e - near_s; C[p]['drives_full'] += 1
            for q in opp:
                if q != onball:
                    C[q]['help_n'] += 1
                    C[q]['help_move'] += np.linalg.norm(Ds[q] - hp) - np.linalg.norm(De[q] - hp)
        # outcome within 3 s of reaching 10 ft
        s = shot_at.get(k)
        if s:
            x = s['xb'] if s.get('xb') is not None else s['xa']
            C[p]['drive_shot'] += 1; C[p]['drive_shot_x'] += x; C[p]['drive_shot_made'] += s['made']
        elif k + 1 < len(segs) and T[segs[k + 1][1]] - T[i1] <= 2500:
            if seg_team[k + 1] == tm:
                s2 = shot_at.get(k + 1)
                if s2 and T[s2['release_idx']] - T[end] <= 4000:
                    C[p]['drive_kick'] += 1
                    C[p]['drive_kick_x'] += s2['xb'] if s2.get('xb') is not None else s2['xa']
            else:
                C[p]['drive_tov'] += 1
    # cuts and screens (off-ball, during own team's holding)
    last_cut, last_screen = {}, {}
    for i in range(6, len(T)):
        hp_ = hold[i]
        if hp_ is None:
            continue
        tm = team_of(on, i, hp_); hp = hoop(i, tm) if tm else None
        if hp is None or P[i - 6] != P[i] or not frontcourt(i - 6, tm, 15):
            continue
        mates = on[i][0] if tm == 'home' else on[i][1]
        opp = on[i][1] if tm == 'home' else on[i][0]
        hxy = xy(i, hp_)
        if hxy is None:
            continue
        for m in mates:
            if m == hp_:
                continue
            a, b = xy(i - 6, m), xy(i, m)
            if a is None or b is None:
                continue
            C[m]['offball_frames'] += 1
            if (np.linalg.norm(a - hp) - np.linalg.norm(b - hp) >= 10 and np.linalg.norm(b - hp) <= 8
                    and T[i] - last_cut.get(m, -1e18) > 3000):
                C[m]['cuts'] += 1; last_cut[m] = T[i]
                for k2 in range(i + 1, min(i + 6, len(T))):
                    if hold[k2] == m:
                        C[m]['cut_fed'] += 1; break
        if i % 2:
            continue
        dpos = {q: xy(i, q) for q in opp}
        dpos = {q: v for q, v in dpos.items() if v is not None}
        if not dpos:
            continue
        ondef = min(dpos, key=lambda q: np.linalg.norm(dpos[q] - hxy))
        for m in mates:
            if m == hp_:
                continue
            a, b = xy(i - 2, m), xy(i, m)
            if a is None or b is None:
                continue
            if (np.linalg.norm(b - a) < 1.5 and np.linalg.norm(b - dpos[ondef]) < 5
                    and np.linalg.norm(b - hxy) < 10 and T[i] - last_screen.get(m, -1e18) > 3000):
                C[m]['screens'] += 1; last_screen[m] = T[i]
    # closeouts (catch-and-shoot) and offensive-rebound crashes
    for s in shots:
        if not s.get('trk_ok'):
            continue
        r = s['release_idx']
        if s['touch_s'] <= 1.2 and r >= 5 and P[r - 5] == P[r]:
            q = s['nd_id']; a, b = xy(r - 5, q), xy(r, q)
            sa, sb = xy(r - 5, s['shooter']), xy(r, s['shooter'])
            if a is not None and b is not None and sa is not None and sb is not None:
                C[q]['co_n'] += 1
                C[q]['co_speed'] += (np.linalg.norm(a - sa) - np.linalg.norm(b - sb)) / ((T[r] - T[r - 5]) / 1000)
                C[q]['co_d1'] += s['d1']; C[q]['co_res'] += s['made'] - s['xa']
        if not s['made']:
            k = s['rim_idx'] + 3
            if k < len(T):
                hp = np.array([s['hoop_sign'] * HOOP_X, 0.0])
                side = on[r][0] if s['shooter_side'] == 'home' else on[r][1]
                for m in side:
                    if m != s['shooter']:
                        C[m]['crash_opp'] += 1
                        v = xy(k, m)
                        if v is not None and np.linalg.norm(v - hp) < 10:
                            C[m]['crash'] += 1
    return base, {p: dict(v) for p, v in C.items()}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    al = [json.loads(l) for l in open(SHOTS / 'alignment.jsonl')]
    xf = [json.loads(l) for l in open(ROOT / 'results/embed/shots_xfg.jsonl')]
    feats = {(r['game_id'], r['event_id']): r for r in map(json.loads, open(SHOTS / 'features.jsonl'))}
    by = defaultdict(list)
    for a, x in zip(al, xf):
        f = feats.get((a['game_id'], a['event_id']))
        if f is None:
            continue  # unaligned: no release frame
        by[a['game_id']].append(dict(x, release_idx=f['release_idx'], rim_idx=f['rim_idx'], event_id=a['event_id']))
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in zz.namelist() if n.endswith('tracking.jsonl.gz'))
    tasks = [(b, by.get(b.rstrip('/').rsplit('/', 1)[1], [])) for b in bases]
    import sys
    if len(sys.argv) > 1:
        tasks = tasks[:int(sys.argv[1])]
    res = {}
    with ProcessPoolExecutor(10) as ex:
        for i, (b, c) in enumerate(ex.map(game, tasks, chunksize=2), 1):
            res[b.rstrip('/').rsplit('/', 1)[1]] = c
            if i % 100 == 0:
                print(i, flush=True)
    pickle.dump(res, open(OUT / 'event_counts.pkl', 'wb'))
    print('done', len(res))
