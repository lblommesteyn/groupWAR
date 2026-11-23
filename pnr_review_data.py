"""Sample ball screens per coverage label for hand review (review/index.html): 30 per label, seed 0.
Each clip: 5 Hz frames from 1 s before the screen to 2.4 s after (same possession), 10 players + ball, rounded 0.1 ft.
  python pnr_review_data.py  -> review/clips.json
"""
import json, pickle
from collections import defaultdict
import numpy as np
from masked_paths import ROOT

PER = 30
COV = ['blitz', 'switch', 'drop', 'show', 'soft']


def main():
    E = pickle.load(open(ROOT / 'results/pnr/events.pkl', 'rb'))
    names, dates = {}, {}
    for g, v in pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb')).items():
        names.update(v['players']); dates[g] = v['date']
    rng = np.random.default_rng(0)
    by = defaultdict(list)
    for e in E:
        by[e['cov']].append(e)
    pick = []
    for c in COV:
        idx = rng.permutation(len(by[c]))[:PER * 2]
        pick += [by[c][i] for i in idx]
    by_g = defaultdict(list)
    for e in pick:
        by_g[e['gid']].append(e)
    clips = []
    got = defaultdict(int)
    for g, es in by_g.items():
        h = np.load(ROOT / 'results/holes' / f'{g}.npz'); meta = h['meta']; pos = h['pos']; ball = h['ball']
        for e in es:
            if got[e['cov']] >= PER:
                continue
            i0 = np.flatnonzero((meta[:, 0] == e['k']) & (meta[:, 1] == e['t']))
            if len(i0) == 0:
                continue
            i0 = int(i0[0]); fr = []
            for i in range(i0 - 5, i0 + 13):
                if 0 <= i < len(meta) and meta[i, 0] == e['k'] and abs(int(meta[i, 1]) - e['t'] - (i - i0) * 200) < 120:
                    fr.append(i)
            if len(fr) < 10:
                continue
            off = [int(x) for x in meta[i0, 5:10]]; de = [int(x) for x in meta[i0, 10:15]]
            if e['handler'] not in off or e['screener'] not in off or e['dh'] not in de or e['ds'] not in de:
                continue
            P = np.concatenate([pos[fr], ball[fr][:, None]], 1).round(1)
            clips.append(dict(id=f"{g}-{e['k']}-{e['t']}", g=g, date=dates.get(g, ''), cov=e['cov'], act=e['act'],
                              t0=fr.index(i0), f=P.reshape(len(fr), -1).tolist(),
                              h=off.index(e['handler']), s=off.index(e['screener']), dh=5 + de.index(e['dh']), ds=5 + de.index(e['ds']),
                              n=[names.get(p, str(p)) for p in off + de], depth=round(e['depth'], 1), near=round(e['near'], 1)))
            got[e['cov']] += 1
    rng.shuffle(clips)
    json.dump(clips, open(ROOT / 'review/clips.json', 'w'), separators=(',', ':'))
    print(len(clips), dict(got))


if __name__ == '__main__':
    main()
