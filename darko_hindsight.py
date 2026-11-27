"""Is DARKO's per-game history recomputed after the fact? Re-download 40 players' histories from darko.app and
compare dpm on the same past dates with results/benchmarks/darko_history.json (saved earlier).
  python darko_hindsight.py   (PC)"""
import json, os, time, urllib.request
from datetime import datetime
import numpy as np
from fetch_darko import decode
from masked_paths import ROOT

F = ROOT / 'results/benchmarks/darko_history.json'
H = json.load(open(F)); saved_at = datetime.fromtimestamp(os.path.getmtime(F)).isoformat(timespec='minutes')
ok = [p for p, h in H.items() if 'dpm' in h and len(h['date']) > 200]
rng = np.random.default_rng(1); pick = rng.choice(ok, 40, replace=False)
diffs, by_age = [], []
for p in pick:
    req = urllib.request.Request(f'https://darko.app/player/{p}/__data.json', headers={'User-Agent': 'Mozilla/5.0 (research script)'})
    r = decode(json.load(urllib.request.urlopen(req, timeout=30))); h = r.get('history') or {}
    cols = {k: h['values'][i] for i, k in enumerate(h.get('keys', []))}
    new = {d: v for d, v, f in zip(cols['date'], cols['dpm'], cols.get('future_game', [0] * len(cols['date']))) if v is not None and not f}
    old = {d: v for d, v, f in zip(H[p]['date'], H[p]['dpm'], H[p].get('future_game', [0] * len(H[p]['date']))) if v is not None and not f}
    for d in set(new) & set(old):
        diffs.append(new[d] - old[d]); by_age.append(d[:4])
    time.sleep(1.0)
d = np.array(diffs); yrs = np.array(by_age)
print('history saved', saved_at, '| compared', len(d), 'player-dates over', len(pick), 'players')
print('identical share %.3f, mean abs diff %.4f, max %.3f' % ((np.abs(d) < 1e-6).mean(), np.abs(d).mean(), np.abs(d).max()))
for y in sorted(set(yrs)):
    s = yrs == y; print(' ', y, 'n', s.sum(), 'identical %.3f mean abs diff %.4f' % ((np.abs(d[s]) < 1e-6).mean(), np.abs(d[s]).mean()))
