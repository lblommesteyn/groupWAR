"""On-ball EPV credit from cross-fitted EPV v2 (out2/epv_cf.npz, every frame scored by a model that never saw its game).

Per possession, frames at 1 Hz (kind != 2, i.e. up to the possession's last shot):
  move credit     dEPV between consecutive frames -> the ball holder at the earlier frame (or the last holder if
                  the ball was loose / in a pass; a pass's value lands on the passer)
  finish credit   y2 (possession points with shot luck removed) - EPV at the last frame -> last holder
                  (shot selection + free throws + putbacks; turnovers count as 0 - EPV)
Both credits are measured against a cross-fitted GBM baseline of the expected change given the state (ball location,\ntime into possession, shot clock, holder distance to rim, current EPV): raw dEPV drifts +0.008/s while the ball is\nbrought up, which would otherwise pay whoever dribbles it up.\nPossession start value (EPV at the first frame) is credited to nobody: it is the state the team inherited.
Sum of credits + start value = y2 for every possession.

Per player per game -> results/epv/credit_pg.pkl ; season table + split-half reliability (odd/even games)
-> results/epv/credit.json
"""
import json, pickle
from collections import defaultdict
import numpy as np
from masked_paths import ROOT

CF = ROOT / 'results/epv/out2/epv_cf.npz'
OUT = ROOT / 'results/epv'


def state_features(gid):
    """Per frame: ball x, |ball y|, seconds into possession, shot clock, holder distance to rim (from the v1 shards)."""
    out = []
    for g in sorted(set(gid)):
        t = np.load(OUT / 'data' / f'{g}.npz'); tok, gl = t['tok'].astype(float), t['glob'].astype(float)
        hold = tok[:, :5, 5].argmax(1); hd = np.where(tok[:, :5, 5].max(1) > 0, tok[np.arange(len(tok)), hold, 7], np.nan)
        out.append(np.c_[tok[:, 10, 0], np.abs(tok[:, 10, 1]), gl[:, 4], gl[:, 0], hd])
    return np.concatenate(out)


def baseline(X, y, fold):
    """Expected value of y given state, cross-fitted by the EPV folds (a player is credited only beyond it)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    b = np.zeros(len(y))
    for k in np.unique(fold):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=200).fit(X[fold != k], y[fold != k])
        b[fold == k] = m.predict(X[fold == k])
    return b


def main():
    d = np.load(CF, allow_pickle=True)
    epv, gid, meta, y2, kind, fold = d['epv'], d['gid'], d['meta'], d['y2'], d['kind'], d['fold']
    Z = state_features(gid); assert len(Z) == len(epv)
    keys = list(zip(gid, meta[:, 0]))
    starts = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    steps, ends, poss = [], [], []  # steps: (i, j, holder) ; ends: (i, holder)
    for a, b in zip(starts[:-1], starts[1:]):
        idx = [i for i in range(a, b) if kind[i] != 2]
        if not idx:
            continue
        poss.append(idx[0]); last_h = 0
        for i, j in zip(idx[:-1], idx[1:]):
            h = int(meta[i, 2]) or last_h
            last_h = h or last_h
            if h:
                steps.append((i, j, h))
        ends.append((idx[-1], int(meta[idx[-1], 2]) or last_h))
    si, sj, sh = map(np.array, zip(*steps)); ei, eh = map(np.array, zip(*ends))
    dE = epv[sj] - epv[si]
    Xs = np.c_[Z[si], epv[si]]; bs = baseline(Xs, dE, fold[si])
    fin = y2[ei] - epv[ei]
    Xe = np.c_[Z[ei], epv[ei]]; be = baseline(Xe, fin, fold[ei])
    print('raw mean dEPV/step %.4f -> residual %.4f ; raw finish %.3f -> residual %.3f' % (dE.mean(), (dE - bs).mean(), fin.mean(), (fin - be).mean()))
    pg = defaultdict(lambda: defaultdict(float))  # (gid, pid) -> stats
    for i in poss:
        for p in meta[i, 3:8]:
            pg[(gid[i], int(p))]['off_poss'] += 1
    for i, r, h in zip(si, dE - bs, sh):
        pg[(gid[i], int(h))]['move'] += float(r); pg[(gid[i], int(h))]['touch_s'] += 1
    for i, r, h in zip(ei, fin - be, eh):
        if h:
            pg[(gid[i], int(h))]['finish'] += float(r); pg[(gid[i], int(h))]['ends'] += 1
    n_poss, total = len(poss), float((fin - be).sum())
    pickle.dump({k: dict(v) for k, v in pg.items()}, open(OUT / 'credit_pg.pkl', 'wb'))

    games = sorted({g for g, _ in pg}); half = {g: i % 2 for i, g in enumerate(games)}
    S = defaultdict(lambda: np.zeros((2, 5)))  # pid -> [half][off_poss, move, finish, touch_s, ends]
    for (g, p), v in pg.items():
        S[p][half[g]] += [v.get('off_poss', 0), v.get('move', 0), v.get('finish', 0), v.get('touch_s', 0), v.get('ends', 0)]
    rel = {}
    for name, col in (('move', 1), ('finish', 2), ('total', None)):
        a, b = [], []
        for p, s in S.items():
            if s[0, 0] >= 400 and s[1, 0] >= 400:
                f = (lambda h: (s[h, 1] + s[h, 2]) / s[h, 0] * 100) if col is None else (lambda h: s[h, col] / s[h, 0] * 100)
                a.append(f(0)); b.append(f(1))
        rel[name] = dict(r=float(np.corrcoef(a, b)[0, 1]), n=len(a))
    names = {}
    for v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).values():
        names.update({str(k): n for k, n in v['players'].items()})
    table = []
    for p, s in S.items():
        t = s.sum(0)
        if t[0] < 1000:
            continue
        table.append(dict(pid=int(p), name=names.get(str(p), str(p)), off_poss=int(t[0]),
                          move_100=t[1] / t[0] * 100, finish_100=t[2] / t[0] * 100, total_100=(t[1] + t[2]) / t[0] * 100,
                          touch_s_100=t[3] / t[0] * 100))
    table.sort(key=lambda r: -r['total_100'])
    res = dict(n_poss=n_poss, mean_finish=total / max(n_poss, 1), reliability=rel, players=table)
    json.dump(res, open(OUT / 'credit.json', 'w'), indent=1)
    print('possessions', n_poss, 'mean finish residual %.3f' % (total / n_poss))
    print('split-half r (per 100 poss, >=400 poss each half):', json.dumps(rel))
    for r in table[:15]:
        print(f"{r['name'][:22]:22s} total {r['total_100']:+6.2f}  move {r['move_100']:+6.2f}  finish {r['finish_100']:+6.2f}  touch {r['touch_s_100']:5.0f}s")
    print('...')
    for r in table[-5:]:
        print(f"{r['name'][:22]:22s} total {r['total_100']:+6.2f}  move {r['move_100']:+6.2f}  finish {r['finish_100']:+6.2f}")


if __name__ == '__main__':
    main()
