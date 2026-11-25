"""Where do we lose / win against DARKO? Per-game log-loss gap (spm_lpit_h - darko_cal) split by context.
  python diag_vs_darko.py   (PC)"""
import pickle
from collections import defaultdict
from datetime import date
import numpy as np
from masked_paths import ROOT
OUT = ROOT / 'results/latent'
G = {}
for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl'):
    G.update(pickle.load(open(OUT / f, 'rb'))['games'])
first = defaultdict(lambda: '9999')
for g in G.values():
    first[g['season']] = min(first[g['season']], g['date'][:10])
ll = lambda p, y: -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))
rows = []
for f in sorted(OUT.glob('msb_*.npz')):
    P = np.load(f)
    if 'darko_cal' not in P or 'spm_lpit_h' not in P:
        continue
    for i, g in enumerate(P['gid']):
        y = P['win'][i]; o, d = P['spm_lpit_h'][i], P['darko_cal'][i]
        day = (date.fromisoformat(G[g]['date'][:10]) - date.fromisoformat(first[G[g]['season']])).days
        rows.append(dict(season=G[g]['season'], day=day, gap=ll(o, y) - ll(d, y), conf=abs(d - .5),
                         dis=abs(o - d), blow=abs(G[g]['final'][0] - G[g]['final'][1]) >= 20))
def show(name, key):
    grp = defaultdict(list)
    for r in rows:
        grp[key(r)].append(r['gap'])
    print(name)
    rng = np.random.default_rng(0)
    for k in sorted(grp):
        v = np.array(grp[k]); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        print(f'  {str(k):>14} n {len(v):5d} gap {v.mean():+.4f} [{np.percentile(bs, 2.5):+.4f} {np.percentile(bs, 97.5):+.4f}]')
print(len(rows), 'games; gap = our log loss - DARKO (negative = we win)')
show('season', lambda r: r['season'])
show('days into season', lambda r: '0-30' if r['day'] < 30 else '30-75' if r['day'] < 75 else '75-150' if r['day'] < 150 else '150+ (late/playoffs)')
show('how confident DARKO is', lambda r: 'tossup <.1' if r['conf'] < .1 else 'lean .1-.25' if r['conf'] < .25 else 'strong .25+')
show('how much we disagree', lambda r: '<.05' if r['dis'] < .05 else '.05-.1' if r['dis'] < .1 else '.1+')
show('blowout (20+)', lambda r: r['blow'])
