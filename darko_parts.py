"""Which part of DARKO wins? Score DARKO's own components on the ms_bench blocks (same games, same calibration):
dpm (full), box_dpm (box-score component), bayes_rapm_total (its plus-minus component), and the box + our stint
model average. Output: results/latent/darko_parts.json   (PC: python darko_parts.py)"""
import json
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
import ms_bench as mb, ms_latent as ml
from masked_paths import ROOT


def load(col):
    out = {}
    for p, h in json.load(open(mb.DARKO)).items():
        if col not in h:
            continue
        rows = sorted((d, v) for d, v, f in zip(h['date'], h[col], h.get('future_game', [0] * len(h['date']))) if v is not None and not f)
        out[int(p)] = ([r[0] for r in rows], [r[1] for r in rows])
    return out


S, G, secs = mb.load()
win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
cols = ['dpm', 'box_dpm', 'bayes_rapm_total']
D = {c: load(c) for c in cols}
P = defaultdict(list)
for f in sorted(mb.OUT.glob('msb_*.npz')):
    B = dict(np.load(f)); b0 = f.stem.split('_')[2]
    vcut = str(date.fromisoformat(b0) - timedelta(days=ml.VAL_DAYS))
    val = [g for g in G if vcut <= G[g]['date'] < b0]; test = list(B['gid'])
    for c in cols:
        cal = LogisticRegression(C=1e4).fit(mb.darko_strength(D[c], secs, val, vcut)[:, None], win(val))
        P[c].append(cal.predict_proba(mb.darko_strength(D[c], secs, test, b0)[:, None])[:, 1])
    P['win'].append(B['win']); P['usage'].append(B['usage'])
    P['box+usage'].append(1 / (1 + np.exp(-(lg(P['box_dpm'][-1]) + lg(B['usage'])) / 2)))
P = {k: np.concatenate(v) for k, v in P.items()}
w = P['win']; ll = lambda p: -(w * np.log(np.clip(p, 1e-6, 1)) + (1 - w) * np.log(np.clip(1 - p, 1e-6, 1)))
bs = np.random.default_rng(0).integers(0, len(w), (2000, len(w)))
res = {}
for k in cols + ['usage', 'box+usage']:
    d = ll(P[k]) - ll(P['dpm'])
    res[k] = dict(log_loss=float(ll(P[k]).mean()), d_vs_dpm=[float(x) for x in np.percentile([d[b].mean() for b in bs], [2.5, 97.5])])
    print(f"{k:18s} ll {res[k]['log_loss']:.4f}  vs full DARKO {np.round(res[k]['d_vs_dpm'], 4)}")
json.dump(res, open(mb.OUT / 'darko_parts.json', 'w'), indent=1)
