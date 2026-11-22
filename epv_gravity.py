"""Gravity: how much more the defense leans toward a player than a typical defense would (epv_ghost2.py ghosts).

For moment s, attacker i and defender j: pull[s, i, j] = |ghost_j - O_i| - |D_j - O_i|  (feet; positive = the real
defender is closer to i than the identity-blind MAE expects a defender to be, given everyone else and the ball).
gravity[s, i] = sum over the five defenders of pull[s, i, j].
The ghost is a blurry average, which biases distances by spot, so (as in epv_offball.py) each moment is measured
against a cross-fitted GBM baseline of gravity for any attacker at that spot in that situation (position, rim and
ball distances, ball rim distance, speed, holder flag, shot clock, time into possession, nearest teammate).
Per player: off-ball and on-ball means over moments (possession-weighted), split-half reliability (odd/even games).
Output: results/epv/gravity.json, gravity_pg.pkl (game -> pid -> grav, grav_n: off-ball adjusted sums for metrics2)
"""
import json, os, pickle
from collections import defaultdict
import numpy as np
from masked_paths import ROOT

E = ROOT / 'results/epv'
HOOP = np.array([41.75, 0.0])


def main():
    g = np.load(E / 'out4/ghost2.npz'); s = g['s']; gpos = g['gpos'].astype(np.float32)
    v3 = np.load(E / 'v3.npz'); meta, gi, games = v3['meta'][s], v3['gi'][s], v3['games']
    Fn = v3['F'][v3['fi'][s]].astype(np.float32); glob = v3['glob'][s]
    fold = np.load(E / 'out3/epv3_cf.npz')['fold'][s]
    O, Dr, ball = Fn[:, :5, :2], Fn[:, 5:10, :2], Fn[:, 10, :2]
    dr = np.linalg.norm(Dr[:, None] - O[:, :, None], axis=-1)    # [n, 5 att, 5 def]
    dg = np.linalg.norm(gpos[:, None] - O[:, :, None], axis=-1)
    pull = dg - dr
    print('ghost-real defender gap at now, median ft: %.2f' % np.median(np.linalg.norm(gpos - Dr, axis=-1)))
    grav = pull.sum(-1)                                          # [n, 5]
    near = np.where(dr < 12, pull, 0).sum(-1)                    # only defenders within 12 ft of him
    holder = meta[:, 2]; ids = meta[:, 3:8]; is_h = ids == holder[:, None]
    X, Y, Y2, R, Fo = [], [], [], [], []
    for i in range(5):
        p_ = O[:, i]
        tm = np.linalg.norm(O[:, [j for j in range(5) if j != i]] - p_[:, None], axis=-1).min(1)
        X.append(np.c_[p_[:, 0], np.abs(p_[:, 1]), np.linalg.norm(p_ - HOOP, axis=1), np.linalg.norm(p_ - ball, axis=1),
                       np.linalg.norm(ball - HOOP, axis=1), np.linalg.norm(Fn[:, i, 2:], axis=1), is_h[:, i], glob[:, 0], glob[:, 4], tm])
        Y.append(grav[:, i]); Y2.append(near[:, i]); R.append(np.c_[np.arange(len(s)), np.full(len(s), i)]); Fo.append(fold)
    X, Y, Y2, R, Fo = map(np.concatenate, (X, Y, Y2, R, Fo))
    from sklearn.ensemble import HistGradientBoostingRegressor
    adj = {}
    for name, y in (('grav', Y), ('near', Y2)):
        base = np.zeros(len(y))
        for k in np.unique(Fo):
            base[Fo == k] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=500).fit(X[Fo != k], y[Fo != k]).predict(X[Fo == k])
        print(f'{name}: baseline explains {1 - np.var(y - base) / np.var(y):.1%}', flush=True)
        a = np.zeros((len(s), 5)); a[R[:, 0], R[:, 1]] = y - base; adj[name] = a
    variants = {'raw': grav, 'adj': adj['grav'], 'near_adj': adj['near']}
    names = {}
    for v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).values():
        names.update(v['players'])
    res = {'reliability': {}, 'players': []}
    per = {}
    for vn, V in variants.items():
        acc = defaultdict(lambda: [0.0, 0])  # (game, poss, pid, on/off) -> sum, n
        for r in range(len(s)):
            key0 = (int(gi[r]), int(meta[r, 0]))
            for i in range(5):
                a = acc[key0 + (int(ids[r, i]), 'on' if is_h[r, i] else 'off')]; a[0] += V[r, i]; a[1] += 1
        P = defaultdict(lambda: defaultdict(lambda: np.zeros((2, 2))))
        pg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for (gg, k, pid, side), (sm, n) in acc.items():
            P[pid][side][gg % 2] += [sm / n, 1]
            if vn == 'adj' and side == 'off':
                c = pg[str(games[gg])][pid]; c['grav'] += sm / n; c['grav_n'] += 1
        if vn == 'adj':
            pickle.dump({g_: {p: dict(c) for p, c in v.items()} for g_, v in pg.items()}, open(E / 'gravity_pg.pkl', 'wb'))
        for side, mn in (('off', 300), ('on', 100)):
            a = [(h[0, 0] / h[0, 1], h[1, 0] / h[1, 1]) for d in P.values() for h in [d[side]] if h[0, 1] >= mn and h[1, 1] >= mn]
            a = np.array(a); res['reliability'][f'{vn}_{side}'] = dict(r=float(np.corrcoef(a.T)[0, 1]), n=len(a))
        per[vn] = P
    print('split-half r:', json.dumps(res['reliability']))
    for pid in per['raw']:
        d = {vn: {sd: per[vn][pid][sd].sum(0) for sd in ('off', 'on')} for vn in variants}
        if d['raw']['off'][1] < 600:
            continue
        row = dict(pid=int(pid), name=names.get(pid, str(pid)), off_poss=int(d['raw']['off'][1]), on_poss=int(d['raw']['on'][1]))
        for vn in variants:
            for sd in ('off', 'on'):
                row[f'{vn}_{sd}'] = float(d[vn][sd][0] / d[vn][sd][1]) if d[vn][sd][1] >= 50 else None
        res['players'].append(row)
    json.dump(res, open(E / 'gravity.json', 'w'), indent=1)
    for key in ('raw_off', 'adj_off', 'near_adj_off', 'adj_on'):
        rs = sorted([r for r in res['players'] if r[key] is not None], key=lambda r: -r[key])
        print(f'\n{key} (feet of extra defensive lean), top / bottom')
        print('  ' + ', '.join(f"{r['name']} {r[key]:+.2f}" for r in rs[:15]))
        print('  ... ' + ', '.join(f"{r['name']} {r[key]:+.2f}" for r in rs[-6:]))


if __name__ == '__main__':
    main()
