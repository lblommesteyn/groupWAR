"""Pick-and-roll detection and coverage labels from 5 Hz half-court frames (holes.py) + defender assignments (matchups.py).

Screen: ball handler H at least 17 ft from the rim; a teammate S within 4.5 ft of H's defender DH and within 8 ft of
H, S nearly stationary (< 6 ft/s) and H moving (> 4 ft/s). Consecutive qualifying frames are one screen, located at
the frame where S is closest to DH (t0). Same handler/screener within 3 s = same screen. DS = S's defender 0.4 s
before t0 (before any switch).
Coverage, from 0.2-1.6 s after t0 while H still has the ball (rules applied in this order):
  blitz   DH and DS both within 6 ft of H in >= 3 frames
  switch  after 0.6 s, DS is H's assigned defender and DH is not, in most frames
  drop    DS stays deep: median (H rim distance - DS rim distance) >= 8 ft and DS never within 7 ft of H
  show    DS gets within 7 ft of H at or above the level of the screen (depth < 5 ft), then goes back
  soft    anything in between (DS deep-ish but engages, or late)
Screener action: roll (rim distance falls >= 8 ft within 1.6 s, ends < 15 ft), pop (ends >= 20 ft from the rim), else stay.
Value: calibrated cross-fit EPV v3 (poss_lab calibration) at t0 and 2 s later, and possession points.
Output: results/pnr/events.parquet-like npz (one row per screen) + results/pnr/summary.json
  python pnr.py
"""
import json, pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT

H = ROOT / 'results/holes'; M = ROOT / 'results/matchups'; OUT = ROOT / 'results/pnr'
HOOP = np.array([41.75, 0.0])
COV = ['blitz', 'switch', 'drop', 'show', 'soft']
ACT = ['roll', 'pop', 'stay']


def game(gid):
    try:
        d = np.load(H / f'{gid}.npz'); a = np.load(M / f'{gid}.npz')
    except FileNotFoundError:
        return gid, []
    pos = d['pos'].astype(np.float32); vel = d['vel'].astype(np.float32); meta = d['meta']; asg = a['assign']
    k, t, holder = meta[:, 0], meta[:, 1], meta[:, 2]; off = meta[:, 5:10]; de = meta[:, 10:15]
    T = len(meta)
    hs = np.full(T, -1)
    for j in range(5):
        hs[off[:, j] == holder] = j
    rim = np.linalg.norm(pos - HOOP, axis=-1)  # [T, 10]
    spd = np.linalg.norm(vel, axis=-1)
    # contiguous same-possession frames
    same = np.r_[False, (k[1:] == k[:-1]) & (np.abs(t[1:] - t[:-1] - 200) < 60)]
    cand = defaultdict(list)  # (possession run start, handler slot, screener slot) -> frames
    run = np.cumsum(~same)
    for i in range(T):
        h = hs[i]
        if h < 0 or rim[i, h] < 17 or spd[i, h] < 4:
            continue
        dh = np.flatnonzero(asg[i] == h)
        if len(dh) == 0:
            continue
        dh = dh[np.argmin(np.linalg.norm(pos[i, 5 + dh] - pos[i, h], axis=1))]
        for s in range(5):
            if s == h or spd[i, s] > 6:
                continue
            if np.linalg.norm(pos[i, s] - pos[i, 5 + dh]) < 4.5 and np.linalg.norm(pos[i, s] - pos[i, h]) < 8:
                cand[(run[i], h, s)].append((i, dh))
    ev = []
    for (r, h, s), fr in cand.items():
        groups, cur = [], [fr[0]]
        for f in fr[1:]:
            if t[f[0]] - t[cur[-1][0]] <= 3000:
                cur.append(f)
            else:
                groups.append(cur); cur = [f]
        groups.append(cur)
        for gq in groups:
            i0, dh = min(gq, key=lambda f: np.linalg.norm(pos[f[0], s] - pos[f[0], 5 + f[1]]))
            pre = max(i0 - 2, 0) if run[max(i0 - 2, 0)] == r else i0
            ds_c = np.flatnonzero(asg[pre] == s)
            ds_c = ds_c[ds_c != dh]
            if len(ds_c) == 0:
                continue
            ds = ds_c[np.argmin(np.linalg.norm(pos[pre, 5 + ds_c] - pos[pre, s], axis=1))]
            post = [i for i in range(i0 + 1, min(i0 + 9, T)) if run[i] == r]
            held = [i for i in post if hs[i] == h]
            if len(held) < 3:
                continue
            dH = lambda i, d_: float(np.linalg.norm(pos[i, 5 + d_] - pos[i, h]))
            both = sum(dH(i, dh) < 6 and dH(i, ds) < 6 for i in held)
            late = [i for i in held if i >= i0 + 3]
            sw = sum(asg[i, ds] == h and asg[i, dh] != h for i in late)
            depth = [rim[i, h] - rim[i, 5 + ds] for i in held[:5]]
            near = [dH(i, ds) for i in held[:5]]
            if both >= 3:
                c = 'blitz'
            elif late and sw > len(late) / 2:
                c = 'switch'
            elif np.median(depth) >= 8 and min(near) >= 7:
                c = 'drop'
            elif min(near) < 7 and depth[int(np.argmin(near))] < 5:
                c = 'show'
            else:
                c = 'soft'
            fut = [i for i in range(i0, min(i0 + 9, T)) if run[i] == r]
            dr = rim[fut[-1], s] - rim[i0, s]
            a_ = 'roll' if dr <= -8 and rim[fut[-1], s] < 15 else 'pop' if rim[fut[-1], s] >= 20 else 'stay'
            i2 = max(i for i in range(i0, min(i0 + 11, T)) if run[i] == r)
            ev.append(dict(gid=gid, k=int(k[i0]), t=int(t[i0]), t2=int(t[i2]), handler=int(off[i0, h]), screener=int(off[i0, s]),
                           dh=int(de[i0, dh]), ds=int(de[i0, ds]), cov=c, act=a_, x=float(pos[i0, h, 0]), y=float(pos[i0, h, 1]),
                           depth=float(np.median(depth)), near=float(min(near))))
    return gid, ev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gids = sorted(f.stem for f in M.glob('*.npz'))
    E = []
    with ProcessPoolExecutor(8) as ex:
        for gid, ev in ex.map(game, gids, chunksize=4):
            E += ev
    print('games', len(gids), 'screens', len(E), f'({len(E) / len(gids):.1f} per game)', flush=True)
    # value: calibrated cross-fit EPV v3 at the screen and 2 s later, possession points
    v3 = np.load(ROOT / 'results/epv/v3.npz'); c = np.load(ROOT / 'results/epv/out3/epv3_cf.npz')
    pickle.dump(E, open(OUT / 'events.pkl', 'wb'))
    games = v3['games']; gi = c['gi']; meta = c['meta']; fold = c['fold']; ok = c['kind'] != 2; yv = c['y'].astype(float)
    epv = c['epv'].astype(float); cal = np.zeros_like(epv)
    for f in np.unique(fold):
        tr = ok & (fold != f); a, b = np.polyfit(epv[tr], yv[tr], 1); cal[fold == f] = a * epv[fold == f] + b
    E_at = {}; P_at = {}
    for i in range(len(epv)):
        key = (str(games[gi[i]]), int(meta[i, 0]))
        E_at[key + (int(meta[i, 1]) // 200,)] = cal[i]
        P_at[key] = yv[i]
    for e in E:
        key = (e['gid'], e['k'])
        e['epv0'] = E_at.get(key + (e['t'] // 200,), np.nan); e['epv2'] = E_at.get(key + (e['t2'] // 200,), np.nan)
        e['pts'] = P_at.get(key, np.nan)
    pickle.dump(E, open(OUT / 'events.pkl', 'wb'))
    # summaries
    names = {}
    for v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).values():
        names.update(v['players'])
    team = {}
    for g, v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).items():
        for p, tm in v.get('team', {}).items():
            team[(g, p)] = tm
    def tab(rows):
        n = len(rows); cc = defaultdict(list)
        for e in rows:
            cc[e['cov']].append(e)
        out = {}
        for cv in COV:
            r = cc.get(cv, [])
            de = [e['epv2'] - e['epv0'] for e in r if np.isfinite(e['epv0']) and np.isfinite(e['epv2'])]
            out[cv] = dict(n=len(r), share=len(r) / max(n, 1), d_epv=float(np.mean(de)) if de else None,
                           pts=float(np.nanmean([e['pts'] for e in r])) if r else None)
        return out
    S = dict(n=len(E), league=tab(E))
    act = defaultdict(lambda: defaultdict(int))
    for e in E:
        act[e['cov']][e['act']] += 1
    S['action_by_cov'] = {k: dict(v) for k, v in act.items()}
    # screener defenders (big men): coverage mix when they guard the screener
    by_ds = defaultdict(list)
    for e in E:
        by_ds[e['ds']].append(e)
    S['screener_defenders'] = sorted([dict(name=names.get(p, str(p)), n=len(r), **{cv: sum(e['cov'] == cv for e in r) / len(r) for cv in COV})
                                      for p, r in by_ds.items() if len(r) >= 150], key=lambda d: -d['n'])
    json.dump(S, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    print(json.dumps(S['league'], indent=1, default=float))
    print('action by coverage', json.dumps(S['action_by_cov']))
    for key in ('drop', 'switch', 'show', 'blitz'):
        top = sorted(S['screener_defenders'], key=lambda d: -d[key])[:8]
        print(f'\nmost {key} (as screener defender, >=150 screens):', ', '.join(f"{d['name']} {d[key]:.0%}" for d in top))


if __name__ == '__main__':
    main()
