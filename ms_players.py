"""Player-level comparison: our stint ratings vs DARKO's bayes_rapm_total at mid-season block starts.
Fits ms_latent's usage model on all stints before b0 (both prior variances), nets o - d (per 100 poss), and compares
with DARKO RAPM dated before b0: shrinkage slope, gap by minutes, gap vs DARKO's one-year change (do we lag
improvers/decliners?), and the players who move test-game predictions most.  (PC: python ms_players.py)"""
import json
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
import pandas as pd
import latent_skill as ls, ms_bench as mb, ms_latent as ml
import darko_parts_lib as dp

S, G, secs = mb.load(); D = dp.load('bayes_rapm_total')
names = pd.read_parquet(r'C:\Users\LukeB\dev\groupWAR\v3\data\nba\players.parquet')
nm = dict(zip(names.person_id.astype(int), names.display_first_last))


def dk(p, d):
    if p not in D:
        return None
    i = bisect_left(D[p][0], d) - 1
    return D[p][1][i] if i >= 0 else None


out = {}
for b0 in ('2024-02-21', '2025-02-19', '2026-02-18'):
    train = sorted((g for g in G if G[g]['date'] < b0), key=lambda g: G[g]['date'])
    R = ml.rows(S, G, train)
    tsecs = defaultdict(float); rec = defaultdict(float)
    for g in train:
        for side in secs[g]:
            for p, t in side.items():
                tsecs[p] += t
                if G[g]['date'] >= str(date.fromisoformat(b0) - timedelta(days=120)):
                    rec[p] += t
    res = {}
    for v in ml.V_GRID:
        mdl, pidx = ls.fit(R, ml.ex_for(R, v), True, False, wmult=ml.decay(R, G, b0))
        th = mdl.th.detach().numpy(); n = len(pidx)
        ours = {p: th[i] - th[n + i] for p, i in pidx.items()}
        P = [p for p in ours if rec[p] > 300 * 60 and dk(p, b0) is not None]
        o = np.array([ours[p] for p in P]); d = np.array([dk(p, b0) for p in P])
        yr = str(date.fromisoformat(b0) - timedelta(days=365))
        chg = np.array([d[k] - (dk(p, yr) if dk(p, yr) is not None else d[k]) for k, p in enumerate(P)])
        mins = np.array([tsecs[p] / 60 for p in P])
        slope = np.polyfit(d, o, 1)[0]; gap = o - (np.polyfit(d, o, 1)[1] + slope * d)
        r = dict(n=len(P), corr=float(np.corrcoef(o, d)[0, 1]), sd_ours=float(o.std()), sd_darko=float(d.std()),
                 slope_ours_on_darko=float(slope), gap_vs_change_slope=float(np.polyfit(chg, o - d * o.std() / d.std(), 1)[0]),
                 corr_gap_change=float(np.corrcoef(o / o.std() - d / d.std(), chg)[0, 1]),
                 corr_gap_logmin=float(np.corrcoef(o / o.std() - d / d.std(), np.log(mins))[0, 1]))
        z = o / o.std() - d / d.std(); order = np.argsort(z)
        r['we_lower'] = [(nm.get(P[k], P[k]), round(float(o[k]), 1), round(float(d[k]), 1), round(float(chg[k]), 1), int(mins[k])) for k in order[:12]]
        r['we_higher'] = [(nm.get(P[k], P[k]), round(float(o[k]), 1), round(float(d[k]), 1), round(float(chg[k]), 1), int(mins[k])) for k in order[::-1][:12]]
        res[str(v)] = r
        print(b0, 'v', v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if not k.startswith('we_')}, flush=True)
        if v == ml.V_GRID[-1]:
            print('  we rate LOWER than DARKO (name, ours, darko, darko 1y change, career min):'); [print('   ', x) for x in r['we_lower']]
            print('  we rate HIGHER:'); [print('   ', x) for x in r['we_higher']]
    out[b0] = res
json.dump(out, open(mb.OUT / 'ms_players.json', 'w'), indent=1, default=str)
