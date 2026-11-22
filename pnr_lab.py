"""Tag possession-lab library possessions with their ball screens (pnr.py events), without rebuilding the library.

Each library possession is located in its game's 5 Hz frames (results/holes/{gid}.npz) by matching its first frame;
screens whose screen frame falls inside are attached:
  index.json  possessions[i].pc = sorted list of coverages seen in the possession
  p/{id}.json pr = [[frame, coverage, screener action, handler slot, screener slot], ...]
  python pnr_lab.py [library dir]
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from masked_paths import ROOT

LIB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'code/possession-lab/public/data'


def main():
    E = pickle.load(open(ROOT / 'results/pnr/events.pkl', 'rb'))
    by_g = defaultdict(list)
    for e in E:
        by_g[e['gid']].append(e)
    ix = json.load(open(LIB / 'index.json'))
    per_g = defaultdict(list)
    for m in ix['possessions']:
        per_g[m['g']].append(m)
    n_tag = n_scr = miss = 0
    for g, ms in per_g.items():
        h = np.load(ROOT / 'results/holes' / f'{g}.npz')
        pos = h['pos'].astype(np.float32); ball = h['ball'].astype(np.float32); meta = h['meta']; t = meta[:, 1]
        F = np.round(np.concatenate([pos, ball[:, None]], 1).reshape(len(pos), -1), 1)
        for m in ms:
            fp = LIB / 'p' / f"{m['id']}.json"; p = json.load(open(fp))
            f0 = np.array(p['f'][0], np.float32)
            a = int(np.argmin(np.abs(F - f0).max(1)))
            if np.abs(F[a] - f0).max() > 0.11:
                miss += 1; continue
            ta, tb = t[a], t[min(a + len(p['f']) - 1, len(t) - 1)]
            off = [int(x) for x in meta[a, 5:10]]
            pr = []
            for e in by_g.get(g, []):
                if e['k'] == meta[a, 0] and ta <= e['t'] <= tb and e['handler'] in off and e['screener'] in off:
                    fr = int(np.searchsorted(t[a:a + len(p['f'])], e['t']))
                    pr.append([fr, e['cov'], e['act'], off.index(e['handler']), off.index(e['screener'])])
            pr.sort()
            p['pr'] = pr; m['pc'] = sorted({r[1] for r in pr})
            json.dump(p, open(fp, 'w'), separators=(',', ':'))
            n_tag += bool(pr); n_scr += len(pr)
    json.dump(ix, open(LIB / 'index.json', 'w'), separators=(',', ':'))
    print(f"{len(ix['possessions'])} possessions, {n_tag} with ball screens ({n_scr} screens), {miss} not located")


if __name__ == '__main__':
    main()
