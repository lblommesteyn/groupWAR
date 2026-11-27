"""Do BORIS context shifts replicate? For every (player, teammate) context in mix.json-style terms, compute the
archetype-mix shift (with - without) separately in odd and even games, and across seasons, and correlate.
Same for the 'with another lead handler' shift and for the overall mix (split-half).
  python boris_validate.py   (Mac)"""
import json
from collections import defaultdict
import numpy as np
from archetype_mix import load, prep, fit, K_OFF, K_DEF

D, OFF_F, DEF_F = load()
N = len(D['gid']); poss = np.repeat(np.arange(N), 5)
odd = np.array([int(g[-1]) % 2 for g in D['gid']])
res = {}
for side, key, feats, k in (('off', 'fo', OFF_F, K_OFF), ('def', 'fd', DEF_F, K_DEF)):
    Z = prep(D[key].reshape(-1, len(feats)), feats); gm, R = fit(Z, k)
    pid = (D['off'] if side == 'off' else D['deff']).reshape(-1); mates = D['off'] if side == 'off' else D['deff']
    by = defaultdict(list)
    for r, p in enumerate(pid):
        by[int(p)].append(r)
    mix_h, shifts = [], defaultdict(list)
    for p, rows in by.items():
        rows = np.array(rows)
        if len(rows) < 400:
            continue
        P_ = poss[rows]; Rp = R[rows]; o = odd[P_]; s = D['season'][P_]
        mix_h.append((Rp[o == 1].mean(0), Rp[o == 0].mean(0)))
        five = mates[P_]
        cnt = defaultdict(int)
        for row in five:
            for q in row:
                if q != p:
                    cnt[int(q)] += 1
        for q, c in sorted(cnt.items(), key=lambda t: -t[1])[:3]:
            w = (five == q).any(1)
            for lab, a, b in (('half', o == 1, o == 0), ('season', s == 2026, s == 2027)):
                ok = all(((w & m_).sum() >= 60, (~w & m_).sum() >= 60) for m_ in (a, b)) if False else \
                     min((w & a).sum(), (~w & a).sum(), (w & b).sum(), (~w & b).sum()) >= 60
                if ok:
                    sa = Rp[w & a].mean(0) - Rp[~w & a].mean(0); sb = Rp[w & b].mean(0) - Rp[~w & b].mean(0)
                    shifts[lab].append((sa, sb))
    mh = np.array(mix_h)
    out = dict(mix_split_half_r=[round(float(np.corrcoef(mh[:, 0, c], mh[:, 1, c])[0, 1]), 3) for c in range(k)])
    for lab, v in shifts.items():
        v = np.array(v)
        out[f'teammate_shift_{lab}_r'] = [round(float(np.corrcoef(v[:, 0, c], v[:, 1, c])[0, 1]), 3) for c in range(k)]
        out[f'teammate_shift_{lab}_n'] = len(v)
    res[side] = out
    print(side, out, flush=True)
json.dump(res, open('results/archetypes/validate.json', 'w'), indent=1)
