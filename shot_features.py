"""Tracking features at release for aligned field-goal attempts.

Defender positions are observed or filled by <=0.6 s linear interpolation
(benchmarked at ~0.3 ft p90). A shot gets tracking features only when all five
defenders are accounted for at release, per the defender-replay result.
"""
import json, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ARCHIVE
from shot_align import OUT, HOOP_X, REACH, load_frames, track

LOC_TOL = 6.0
LAG = 2  # frames (0.4 s) for velocities


def hull_area(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return 0.0
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    x, y = np.array(hull).T
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def features_for_game(task):
    name, shots = task
    with zipfile.ZipFile(ARCHIVE) as z:
        T, P, B, players, on = load_frames(z, name.rsplit('tracking.jsonl.gz', 1)[0])
    out = []
    for s in shots:
        r, k, pid = s['release_idx'], s['rim_idx'], s['shooter']
        home, away = on[r]
        if pid in home:
            mates, defs = [p for p in home if p != pid], list(away)
        elif pid in away:
            mates, defs = [p for p in away if p != pid], list(home)
        else:
            out.append(s | dict(trk_ok=False, trk_reason='shooter_not_on_court')); continue
        hoop = np.array([np.sign(B[k, 0]) * HOOP_X, 0.0])
        sxy = np.array(players[r][pid])
        D = [track(players, T, d, r)[0] for d in defs]
        M = [track(players, T, m, r)[0] for m in mates]
        n_filled = sum(track(players, T, d, r)[1] for d in defs)
        if len(defs) != 5 or any(d is None for d in D):
            out.append(s | dict(trk_ok=False, trk_reason='defender_missing')); continue
        D = np.array(D); dd = np.linalg.norm(D - sxy, axis=1); order = np.argsort(dd)
        nd = order[0]
        to_hoop = hoop - sxy; to_hoop /= np.linalg.norm(to_hoop) + 1e-9
        to_def = D[nd] - sxy; cos_def = float(to_def @ to_hoop / (np.linalg.norm(to_def) + 1e-9))
        # defenders roughly between shooter and hoop
        rel = D - sxy; proj = rel @ to_hoop; perp = np.abs(rel @ np.array([-to_hoop[1], to_hoop[0]]))
        between = int(((proj > 0) & (perp < 3) & (proj < np.linalg.norm(hoop - sxy))).sum())
        f = dict(d1=float(dd[order[0]]), d2=float(dd[order[1]]),
                 n_def_4=int((dd < 4).sum()), n_def_6=int((dd < 6).sum()), n_def_10=int((dd < 10).sum()),
                 def_cos=cos_def, def_between=between, def_filled=int(n_filled),
                 nd_id=int(defs[nd]), def_ids=[int(d) for d in defs], def_dists=[float(v) for v in dd],
                 rim_def_ids=[int(defs[i]) for i in range(5) if np.linalg.norm(D[i] - hoop) < 8],
                 shooter_side='home' if pid in home else 'away', hoop_sign=int(np.sign(hoop[0])))
        # closing speed of the release-time nearest defender, and shooter speed
        prev = r - LAG
        if prev >= 0 and T[r] - T[prev] <= LAG * 250 and P[prev] == P[r]:
            sp = track(players, T, pid, prev)[0]; dp = track(players, T, defs[nd], prev)[0]
            dt = (T[r] - T[prev]) / 1000
            if sp is not None and dp is not None:
                f['close_speed'] = float((np.linalg.norm(dp - sp) - dd[nd]) / dt)
                f['shooter_speed'] = float(np.linalg.norm(sxy - sp) / dt)
                f['shooter_toward_hoop'] = float((sxy - sp) @ to_hoop / dt)
                f['d1_prev'] = float(np.min(np.linalg.norm(
                    np.array([track(players, T, d, prev)[0] if track(players, T, d, prev)[0] is not None else [np.nan, np.nan] for d in defs]) - sp, axis=1)))
        # touch time: consecutive frames before release with ball in shooter's reach
        j, hold = r, 0
        while j > 0 and T[r] - T[j - 1] <= 6000 and P[j - 1] == P[r]:
            j -= 1
            if pid in players[j] and np.isfinite(B[j, 0]) and np.linalg.norm(B[j, :2] - players[j][pid]) <= REACH:
                hold = T[r] - T[j]
            elif np.isfinite(B[j, 0]):
                break
        f['touch_s'] = hold / 1000
        f['ball_z_release'] = float(B[r, 2]) if np.isfinite(B[r, 2]) else np.nan
        if all(m is not None for m in M) and len(M) == 4:
            M = np.array(M)
            f['mate_mean_dist'] = float(np.linalg.norm(M - sxy, axis=1).mean())
            f['off_hull'] = hull_area(np.vstack([M, sxy]))
            f['mates_open'] = int((np.linalg.norm(M[:, None] - D[None], axis=2).min(1) > 6).sum())
        f['def_hull'] = hull_area(D)
        f['def_paint'] = int(((np.abs(D[:, 0] - hoop[0]) < 19) & (np.abs(D[:, 1]) < 8)).sum())
        out.append(s | dict(trk_ok=True) | f)
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(OUT / 'alignment.jsonl')]
    good = [r for r in rows if r.get('loc_err', 99) <= LOC_TOL]
    with zipfile.ZipFile(ARCHIVE) as z:
        names = {n.split('/')[-2]: n for n in z.namelist() if n.endswith('tracking.jsonl.gz')}
    by = defaultdict(list)
    for r in good:
        by[r['game_id']].append(r)
    tasks = [(names[g], v) for g, v in by.items()]
    res = []
    with ProcessPoolExecutor(10) as ex:
        for i, o in enumerate(ex.map(features_for_game, tasks), 1):
            res += o
            if i % 100 == 0:
                print(i, flush=True)
    with open(OUT / 'features.jsonl', 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    ok = np.array([r['trk_ok'] for r in res])
    print(f'{len(rows)} attempts -> {len(good)} aligned (<= {LOC_TOL} ft) -> {ok.sum()} with all 5 defenders ({ok.mean():.1%})')
