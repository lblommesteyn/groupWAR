"""Where do we lose to DARKO? Per-game log-loss gap (our usage model minus DARKO's bayes_rapm_total, both calibrated
the same way on the ms_bench blocks), sliced by game situation.
  phase       days since the season's first game
  thin        share of game seconds by players with < 500 minutes in our training stints (rookies, fringe)
  moved       share of seconds by players whose teammates differ from their previous game's (trades/signings)
  confidence  |DARKO's probability - 0.5|
  disagree    |our probability - DARKO's|; who was right when we disagree
Output: results/latent/ms_diag.json + per-game table ms_diag_games.pkl   (PC: python ms_diag.py)"""
import json, pickle
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
import ms_bench as mb, ms_latent as ml
import darko_parts_lib as dp

S, G, secs = mb.load()
D = dp.load('bayes_rapm_total')
win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
# per game: players by side, in date order; cumulative seconds per player by date
games = sorted(G, key=lambda g: (G[g]['date'], g))
side_of = {g: [set(secs[g][0]), set(secs[g][1])] for g in games}
prev_mates = {}  # (player, game) -> teammates in the player's previous game
last = {}
for g in games:
    for side in side_of[g]:
        for p in side:
            prev_mates[(p, g)] = last.get(p)
        for p in side:
            last[p] = side - {p}
psecs = defaultdict(list)  # player -> [(date, cumulative secs)]
cum = defaultdict(float)
for g in games:
    for side in secs[g]:
        for p, t in side.items():
            cum[p] += t; psecs[p].append((G[g]['date'], cum[p]))


def hist_secs(p, b0):
    L = psecs[p]; i = bisect_left(L, (b0,)) - 1
    return L[i][1] if i >= 0 else 0.0


rows = []
for f in sorted(mb.OUT.glob('msb_*.npz')):
    B = dict(np.load(f)); season = int(f.stem.split('_')[1]); b0 = f.stem.split('_')[2]
    vcut = str(date.fromisoformat(b0) - timedelta(days=ml.VAL_DAYS))
    val = [g for g in G if vcut <= G[g]['date'] < b0]; test = list(B['gid'])
    cal = LogisticRegression(C=1e4).fit(mb.darko_strength(D, secs, val, vcut)[:, None], win(val))
    pd_ = cal.predict_proba(mb.darko_strength(D, secs, test, b0)[:, None])[:, 1]
    start = min(G[g]['date'] for g in G if G[g]['season'] == season)
    for k, g in enumerate(test):
        tot = thin = moved = 0.0
        for side, mates in zip(secs[g], side_of[g]):
            for p, t in side.items():
                tot += t
                if hist_secs(p, b0) < 500 * 60:
                    thin += t
                pm = prev_mates.get((p, g))
                if pm is not None and len((mates - {p}) & pm) < 0.5 * len(mates - {p}):
                    moved += t
        rows.append(dict(g=g, season=season, playoff=G[g]['playoff'], days=(date.fromisoformat(G[g]['date']) - date.fromisoformat(start)).days,
                         win=int(B['win'][k]), ours=float(B['usage'][k]), darko=float(pd_[k]), thin=thin / tot, moved=moved / tot))
ll = lambda w, p: -(w * np.log(max(p, 1e-6)) + (1 - w) * np.log(max(1 - p, 1e-6)))
for r in rows:
    r['gap'] = ll(r['win'], r['ours']) - ll(r['win'], r['darko'])
pickle.dump(rows, open(mb.OUT / 'ms_diag_games.pkl', 'wb'))
gap = np.array([r['gap'] for r in rows]); rng = np.random.default_rng(0)
out = {'n': len(rows), 'mean_gap': float(gap.mean()), 'slices': {}}


def report(name, key, edges):
    v = np.array([key(r) for r in rows]); out['slices'][name] = []
    print(f'\n{name}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() < 30:
            continue
        x = gap[m]; bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
        e = dict(lo=lo, hi=hi, n=int(m.sum()), gap=float(x.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 share_of_total=float(x.sum() / gap.sum()))
        out['slices'][name].append(e)
        print(f'  [{lo:>6}, {hi:>6})  n {e["n"]:5d}  gap {e["gap"]:+.4f} [{e["ci"][0]:+.4f}, {e["ci"][1]:+.4f}]  share of total loss {e["share_of_total"]:+.0%}')


report('days into season', lambda r: r['days'], [0, 45, 90, 135, 180, 400])
report('share of minutes by thin-history players', lambda r: r['thin'], [0, 0.02, 0.08, 0.2, 1.01])
report('share of minutes by players with new teammates', lambda r: r['moved'], [0, 0.01, 0.1, 0.3, 1.01])
report('DARKO confidence |p-0.5|', lambda r: abs(r['darko'] - 0.5), [0, 0.1, 0.2, 0.3, 0.51])
report('disagreement |ours-darko|', lambda r: abs(r['ours'] - r['darko']), [0, 0.03, 0.06, 0.1, 1])
report('playoffs', lambda r: int(r['playoff']), [0, 1, 2])
# sharpness: spread of predictions
lo = lambda p: np.log(p / (1 - p))
out['spread'] = {k: float(np.std(lo(np.clip([r[k] for r in rows], 1e-6, 1 - 1e-6)))) for k in ('ours', 'darko')}
big = [r for r in rows if abs(r['ours'] - r['darko']) > 0.1]
out['big_disagreements'] = dict(n=len(big), darko_right=float(np.mean([(r['darko'] > r['ours']) == r['win'] for r in big])))
print('\nspread of logits', out['spread'], '\nwhen we disagree by >10 pts:', out['big_disagreements'])
json.dump(out, open(mb.OUT / 'ms_diag.json', 'w'), indent=1)
