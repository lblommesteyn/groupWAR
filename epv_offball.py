"""Aggregate ghosting credit (epv_ghost.py) into per-player off-ball offense and defense value.

dV[s, i] = EPV(actual) - EPV(player i replaced by the MAE ghost), raw EPV scale -> scaled by the calibration slope
(actual points per raw EPV point, fit across folds as in poss_lab.py) so credit is in points.
  offense, slot 0-4, only moments where he is NOT the ball holder:  credit = dV
  defense, slot 5-9, all moments:                                   credit = -dV
Per possession: mean credit over its moments (a possession counts once however long it is). Per player: sum over
possessions / possessions * 100 = points per 100 possessions on the floor (offense: per 100 offensive
possessions; defense: per 100 defensive possessions).
Reliability: odd vs even games, players with >= 300 possessions in each half.
ADJUST=1 (default): credit is measured against a cross-fitted GBM baseline of dV for a player at that spot in that\nsituation (position, distance to rim and ball, ball distance to rim, his speed, shot clock, time into possession,\nEPV), so it compares him with typical players in the same role and spot. ADJUST=0 gives the raw role-laden version.\nOutput: results/epv/offball.json (offball_raw.json with ADJUST=0)
"""
import json, pickle
from collections import defaultdict
import numpy as np
from masked_paths import ROOT

E = ROOT / 'results/epv'
import os
ADJUST = os.environ.get('ADJUST', '1') == '1'


def main():
    g = np.load(E / 'out4/ghost.npz'); s, dV = g['s'], g['dV'].astype(float)
    v3 = np.load(E / 'v3.npz'); meta, gi, games, kind = v3['meta'][s], v3['gi'][s], v3['games'], v3['kind']
    c = np.load(E / 'out3/epv3_cf.npz'); ok = c['kind'] != 2
    slope = np.polyfit(c['epv'][ok].astype(float), c['y'][ok], 1)[0]
    dV *= slope
    print(f'moments {len(s)}  calibration slope {slope:.3f}', flush=True)
    holder = meta[:, 2]; ids = meta[:, 3:13]
    off_mask = ids[:, :5] != holder[:, None]
    raw = dV.copy()
    if ADJUST:
        # role/spot baseline: expected dV for a player at this spot in this situation, cross-fitted by EPV fold
        from sklearn.ensemble import HistGradientBoostingRegressor
        Fn = v3['F'][v3['fi'][s]].astype(np.float32)  # [n, 11, 4] at "now"
        glob = v3['glob'][s]; fold = c['fold'][s]
        ball = Fn[:, 10, :2]
        for side, sl in (('off', range(5)), ('def', range(5, 10))):
            X, Y, R, Fo = [], [], [], []
            for i in sl:
                m = off_mask[:, i] if side == 'off' else np.ones(len(s), bool)
                p_ = Fn[m, i, :2]; r_ = np.flatnonzero(m)
                X.append(np.c_[p_[:, 0], np.abs(p_[:, 1]), np.hypot(p_[:, 0] - 41.75, p_[:, 1]), np.linalg.norm(p_ - ball[m], axis=1),
                               np.hypot(ball[m, 0] - 41.75, ball[m, 1]), np.linalg.norm(Fn[m, i, 2:], axis=1), glob[m, 0], glob[m, 4], g['epv'][m]])
                Y.append(dV[m, i]); R.append(np.c_[r_, np.full(len(r_), i)]); Fo.append(fold[m])
            X, Y, R, Fo = np.concatenate(X), np.concatenate(Y), np.concatenate(R), np.concatenate(Fo)
            base = np.zeros(len(Y))
            for k in np.unique(Fo):
                mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=500).fit(X[Fo != k], Y[Fo != k])
                base[Fo == k] = mdl.predict(X[Fo == k])
            print(f'{side}: baseline explains {1 - np.var(Y - base) / np.var(Y):.1%} of moment-level dV', flush=True)
            dV[R[:, 0], R[:, 1]] = Y - base
    print('mean dV offense off-ball %.4f  holder %.4f  defense %.4f' % (dV[:, :5][off_mask].mean(), dV[:, :5][~off_mask].mean(), dV[:, 5:].mean()))
    # possession means
    acc = defaultdict(lambda: [0.0, 0])  # (game, poss, pid, side) -> [sum, n]
    for r in range(len(s)):
        key0 = (int(gi[r]), int(meta[r, 0]))
        for i in range(10):
            if i < 5 and not off_mask[r, i]:
                continue
            a = acc[key0 + (int(ids[r, i]), 'off' if i < 5 else 'def')]
            a[0] += dV[r, i] if i < 5 else -dV[r, i]; a[1] += 1
    P = defaultdict(lambda: defaultdict(lambda: np.zeros((2, 2))))  # pid -> side -> [half][sum, poss]
    for (gg, k, pid, side), (sm, n) in acc.items():
        P[pid][side][gg % 2] += [sm / n, 1]
    rel = {}
    for side in ('off', 'def'):
        a, b = [], []
        for pid, d in P.items():
            h = d[side]
            if h[0, 1] >= 300 and h[1, 1] >= 300:
                a.append(h[0, 0] / h[0, 1] * 100); b.append(h[1, 0] / h[1, 1] * 100)
        rel[side] = dict(r=float(np.corrcoef(a, b)[0, 1]), n=len(a))
    print('split-half r per 100 possessions:', json.dumps(rel))
    names = {}
    for v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).values():
        names.update(v['players'])
    rows = []
    for pid, d in P.items():
        o, df = d['off'].sum(0), d['def'].sum(0)
        if o[1] + df[1] < 1500:
            continue
        rows.append(dict(pid=int(pid), name=names.get(pid, str(pid)), off_poss=int(o[1]), def_poss=int(df[1]),
                         off_100=o[0] / max(o[1], 1) * 100, def_100=df[0] / max(df[1], 1) * 100))
    for r in rows:
        r['total_100'] = r['off_100'] + r['def_100']
    json.dump(dict(slope=slope, reliability=rel, players=sorted(rows, key=lambda r: -r['total_100'])), open(E / ('offball.json' if ADJUST else 'offball_raw.json'), 'w'), indent=1)
    for key, lab in (('off_100', 'OFF-BALL OFFENSE'), ('def_100', 'DEFENSE')):
        rs = sorted(rows, key=lambda r: -r[key])
        print(f'\n{lab} (points per 100 possessions), top / bottom')
        for r in rs[:12]:
            print(f"  {r['name'][:24]:24s} {r[key]:+6.2f}")
        print('  ...')
        for r in rs[-6:]:
            print(f"  {r['name'][:24]:24s} {r[key]:+6.2f}")


if __name__ == '__main__':
    main()
