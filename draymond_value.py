"""DRAYMOND value: what do execution gaps cost in points?

1. Shot timestamps: for every aligned shot (results/shots/alignment.jsonl, both seasons) the release timestamp from
   the game's tracking frames (shot_align.load_frames, the index the alignment used).
2. Each half-court possession in results/world (traj_*.npz) gets the shot released inside it by one of its five
   attackers: points scored on that shot (made x 2/3) and its location-only xFG (xa).
3. For every defender on that possession: mean execution gap (distance and signed depth vs the scheme spot, from
   pred.npz) over the frames from 3.0 s to 0.8 s before release (before the shot itself moves anyone), and whether
   he is guarding the shooter (nearest defender at release).
4. Value model, cross-fit by game halves: points ~ location xFG x value + per-defender gap terms, separately for the
   shooter's defender and the four help defenders (ridge). Contribution of a defender on a possession =
   coefficients x his gaps - the same at zero gap; per player: points per 100 shot possessions, + = costs points.
5. Reliability: odd / even games and season to season.
Output: results/world/draymond_value.json, results/world/draymond_value_players.json     python draymond_value.py
"""
import json, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

from archetypes import ARCHIVES
from shot_align import load_frames

ROOT = Path(__file__).resolve().parent
W = ROOT / 'results/world'
SHOTS = {2026: ROOT / 'results/shots/alignment.jsonl', 2027: ROOT.parent / 'court-model-2627/results/shots/alignment.jsonl'}
XFG = {2026: ROOT / 'results/embed/shots_xfg.jsonl', 2027: ROOT.parent / 'court-model-2627/results/embed/shots_xfg.jsonl'}
RIM = np.array([41.75, 0.0])


def release_times(task):
    path, base, rows = task
    with zipfile.ZipFile(path) as z:
        T = load_frames(z, base)[0]
    return [(r['game_id'], r['event_id'], int(T[r['release_idx']])) for r in rows if r['release_idx'] is not None and r['release_idx'] < len(T)]


def shots():
    out = {}
    for s in (2026, 2027):
        al = [json.loads(l) for l in open(SHOTS[s])]
        xa = [json.loads(l) for l in open(XFG[s])]
        by = defaultdict(list)
        for a, x in zip(al, xa):
            if a['aligned'] and a.get('release_idx') is not None:
                by[a['game_id']].append(dict(a, xa=x['xa'], nd=x.get('nd_id')))
        with zipfile.ZipFile(ARCHIVES[s]) as z:
            base = {n.rsplit('/', 2)[-2]: n.rsplit('tracking.jsonl.gz', 1)[0] for n in z.namelist() if n.endswith('tracking.jsonl.gz')}
        tasks = [(ARCHIVES[s], base[g], rows) for g, rows in by.items() if g in base]
        meta = {(r['game_id'], r['event_id']): r for rows in by.values() for r in rows}
        with ProcessPoolExecutor(8) as ex:
            for res in ex.map(release_times, tasks, chunksize=4):
                for g, e, t in res:
                    out[(g, e)] = dict(meta[(g, e)], t=t)
        print(s, len(out), 'shots with release time', flush=True)
    return out


def main():
    S = shots()
    by_game = defaultdict(list)
    for (g, e), r in S.items():
        by_game[g].append(r)
    z = np.load(W / 'pred.npz')
    trj = {s: np.load(W / f'traj_{s}.npz') for s in (2026, 2027)}
    off = np.concatenate([trj[s]['off'] for s in (2026, 2027)]); t0 = np.concatenate([trj[s]['t0'] for s in (2026, 2027)])
    L = np.concatenate([trj[s]['length'] for s in (2026, 2027)]); gid = z['meta_gid']; deff = z['meta_deff']
    season = z['meta_season']
    # possession -> shot
    shot_of = {}
    for i in range(len(gid)):
        for r in by_game.get(gid[i], []):
            if t0[i] <= r['t'] <= t0[i] + int(L[i]) * 400 + 400 and r['shooter'] in off[i]:
                shot_of[i] = r; break
    print(len(shot_of), 'possessions with a tracked shot of', len(gid), flush=True)
    # per (possession, defender) gaps over the pre-shot window
    poss, k, t = z['poss'], z['k'].astype(int), z['t'].astype(int)
    y, ms = z['y'].astype(np.float32), z['ms'].astype(np.float32)
    dep = lambda p: np.hypot(*(p - RIM).T)
    gap = np.hypot(*(y - ms).T); gapd = dep(y) - dep(ms)
    ts = t0[poss] + t * 400
    rel = np.array([shot_of[p]['t'] if p in shot_of else -1 for p in range(len(gid))])[poss]
    win = (rel > 0) & (ts >= rel - 3000) & (ts <= rel - 800)
    key = poss[win].astype(np.int64) * 5 + k[win]
    uk, inv = np.unique(key, return_inverse=True)
    G = np.bincount(inv, gap[win]) / np.bincount(inv); GD = np.bincount(inv, gapd[win]) / np.bincount(inv)
    P_, K_ = (uk // 5).astype(int), (uk % 5).astype(int)
    # assemble possession rows: shooter's defender (nearest at release from xfg nd_id) + help defenders
    rows = defaultdict(dict)
    for p, kk, g_, gd in zip(P_, K_, G, GD):
        rows[p][kk] = (g_, gd)
    X, Yp, XA, meta = [], [], [], []
    for p, d in rows.items():
        if len(d) < 5:
            continue
        r = shot_of[p]; val = 3 if r['three'] else 2
        on = [kk for kk in range(5) if deff[p, kk] == r.get('nd')]
        if not on:
            continue
        o = on[0]; help_ = [kk for kk in range(5) if kk != o]
        f = [d[o][0], max(d[o][1], 0), max(-d[o][1], 0)] + list(np.sum([[d[h][0], max(d[h][1], 0), max(-d[h][1], 0)] for h in help_], 0))
        X.append(f); Yp.append(val * r['made']); XA.append(val * r['xa']); meta.append((p, o, help_))
    X, Yp, XA = np.array(X), np.array(Yp, float), np.array(XA)
    games = np.array([gid[m[0]] for m in meta]); fold = np.array([hash(g) % 2 for g in games])
    names = ['on_gap', 'on_too_far', 'on_too_close', 'help_gap', 'help_too_far', 'help_too_close']
    coefs = []
    contrib = np.zeros((len(X), 2))
    for f in (0, 1):
        tr = fold != f
        A = np.column_stack([XA, X])
        m = Ridge(alpha=10.0).fit(A[tr], Yp[tr]); coefs.append(m.coef_[1:].tolist())
        c = m.coef_[1:]
        te = ~tr
        contrib[te, 0] = X[te, :3] @ c[:3]                       # shooter's defender
        contrib[te, 1] = X[te, 3:] @ c[3:] / 4                    # per help defender (averaged)
    res = dict(shot_possessions=int(len(X)), coef_by_fold=[dict(zip(names, np.round(cf, 4).tolist())) for cf in coefs])
    print(res, flush=True)
    # per player-season value: points per 100 shot possessions (+ = costs points), out-of-fold coefficients
    acc = defaultdict(lambda: np.zeros(2))
    for i, (p, o, help_) in enumerate(meta):
        s = int(season[p]); odd = int(games[i][-1]) % 2; c = np.array(coefs[fold[i]])
        acc[(int(deff[p, o]), s, odd)] += [contrib[i, 0], 1]
        for h in help_:
            g_, gd = rows[p][h]
            acc[(int(deff[p, h]), s, odd)] += [g_ * c[3] + max(gd, 0) * c[4] + max(-gd, 0) * c[5], 1]
    P = defaultdict(dict)
    for (pid, s, odd), (v, n) in acc.items():
        P[(pid, s)][odd] = (v, n)
    out = {}
    for (pid, s), d in P.items():
        n = sum(x[1] for x in d.values())
        if n >= 300:
            out[f'{pid}_{s}'] = dict(n=int(n), value100=100 * sum(x[0] for x in d.values()) / n,
                                     half=[100 * d[o][0] / d[o][1] if o in d and d[o][1] else None for o in (1, 0)])
    def corr(a, b):
        a, b = np.array(a, float), np.array(b, float); ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None
    for s in (2026, 2027):
        v = [x for k_, x in out.items() if k_.endswith(str(s)) and None not in x['half']]
        res[f'split_half_r_{s}'] = corr([x['half'][0] for x in v], [x['half'][1] for x in v])
    both = [k_[:-5] for k_ in out if k_.endswith('2026') and k_[:-5] + '_2027' in out]
    res['cross_season_r'] = corr([out[p + '_2026']['value100'] for p in both], [out[p + '_2027']['value100'] for p in both])
    res['players'] = len(out)
    print(res, flush=True)
    json.dump(res, open(W / 'draymond_value.json', 'w'), indent=1)
    json.dump(out, open(W / 'draymond_value_players.json', 'w'))


if __name__ == '__main__':
    main()
