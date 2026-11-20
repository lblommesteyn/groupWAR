"""Latent-skill model on four seasons of lineups (2019-20 .. 2022-23): does usage/load or learned fit help with volume?

Rolling test through 2022-23 (monthly blocks, ~1,200 test games): train on every stint before the block start
(recency-weighted, half-life HALF_LIFE days), predict each test game's margin from its realized stints.
Players: additive o/d with a ridge prior N(0, V) (no history file: the training window is the history), free usage
logits, free style vectors z (k = 3, no descriptions before tracking existed) for the fit terms.
Hyperparameters (V; fit penalty) and the probability calibration are chosen on the last VAL_DAYS of training
(model refit on that shorter window), then the model is refit on the full window.
Arms: additive, usage (+ load cost), fit (+ pairwise styles), full. Paired game-bootstrap CIs vs additive.
Output: results/latent/ms_latent.json, ms_block_*.npz.   Usage (PC): python ms_latent.py
"""
import json, pickle
from datetime import date, timedelta
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import latent_skill as ls
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
BLOCKS = ['2022-11-15', '2022-12-15', '2023-01-15', '2023-02-15', '2023-03-15', '2023-04-15', '2023-07-01']
HALF_LIFE, VAL_DAYS = 365.0, 30
V_GRID = [1.0, 4.0]
L2_GRID = [100.0, 1000.0, 10000.0]


def rows(S, G, gids):
    gs = set(gids); R, ages = [], []
    for st in S:
        if st['game_id'] not in gs:
            continue
        h, a = st['lineup']
        for k, (o, de) in enumerate(((h, a), (a, h))):
            if st['poss_team'][k] > 0:
                c = [(lambda u: u[0] + 0.44 * u[1] + u[2])(st['use'].get(p, [0, 0, 0])) for p in o]
                R.append((st['game_id'], tuple(o), tuple(de), 1 - k, 100 * st['pts'][k] / st['poss_team'][k], st['poss_team'][k], c))
    return R


def decay(R, G, ref):
    r0 = date.fromisoformat(ref)
    return [0.5 ** ((r0 - date.fromisoformat(G[r[0]]['date'])).days / HALF_LIFE) for r in R]


def ex_for(R, v):
    pids = sorted({p for r in R for p in r[1] + r[2]})
    n = len(pids)
    return dict(pids=pids, m=np.zeros(2 * n), v=np.full(2 * n, v), Z=np.zeros((n, 1)))


def margins(mdl, pidx, R):
    return ls.predict_games(mdl, pidx, R)


def run_arm(S, G, train, val_tr, val, test, b0, usage, fit_):
    vt = (val_tr[-1] if val_tr else b0)
    best = None
    for v in V_GRID:
        for lw in (L2_GRID if fit_ else [ls.L2_W]):
            Rv = rows(S, G, val_tr)
            mdl, pidx = ls.fit(Rv, ex_for(Rv, v), usage, fit_, l2w=lw, wmult=decay(Rv, G, G[vt]['date']))
            pm = margins(mdl, pidx, rows(S, G, val))
            x = np.array([pm.get(g, 0.0) for g in val]); y = np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in val])
            cal = LogisticRegression(C=1e4).fit(x[:, None], y)
            l = float(-np.mean(np.log(np.clip(cal.predict_proba(x[:, None])[np.arange(len(y)), y], 1e-9, 1))))
            if best is None or l < best[0]:
                best = (l, v, lw, cal)
    _, v, lw, cal = best
    Rt = rows(S, G, train)
    mdl, pidx = ls.fit(Rt, ex_for(Rt, v), usage, fit_, l2w=lw, wmult=decay(Rt, G, b0))
    pm = margins(mdl, pidx, rows(S, G, test))
    x = np.array([pm.get(g, 0.0) for g in test])
    info = dict(v=v, l2w=lw, g0=float(torch.nn.functional.softplus(mdl.g0)), Wo=float(mdl.Wo.detach().abs().max()),
                Wd=float(mdl.Wd.detach().abs().max()), M=float(mdl.M.detach().abs().max()))
    return cal.predict_proba(x[:, None])[:, 1], info


def main():
    D = pickle.load(open(OUT / 'ms_stints.pkl', 'rb'))
    S, G = D['stints'], D['games']
    allp = {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        f = OUT / f'ms_block_{b0}.npz'
        if f.exists():
            allp[b0] = dict(np.load(f)); continue
        train = sorted((g for g in G if G[g]['date'] < b0), key=lambda g: G[g]['date'])
        vcut = str(date.fromisoformat(b0) - timedelta(days=VAL_DAYS))
        val_tr = [g for g in train if G[g]['date'] < vcut]; val = [g for g in train if G[g]['date'] >= vcut]
        test = sorted((g for g in G if b0 <= G[g]['date'] < b1), key=lambda g: (G[g]['date'], g))
        P = {'win': np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in test])}
        print(f'block {b0}: train {len(train)} val {len(val)} test {len(test)}', flush=True)
        for arm, (usage, fit_) in ls.ARMS.items():
            P[arm], info = run_arm(S, G, train, val_tr, val, test, b0, usage, fit_)
            print(f'  {arm:9s} auc {roc_auc_score(P["win"], P[arm]):.3f} {info}', flush=True)
        np.savez(f, **P); allp[b0] = P
    names = list(ls.ARMS)
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {k: np.concatenate([p[k] for p in allp.values()]) for k in names}
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    Bs = np.random.default_rng(0).integers(0, len(win), (2000, len(win)))
    res = {'n_test': int(len(win)), 'arms': {}}
    print(f'\npooled {len(win)} games')
    for k in names:
        d = ll(pr[k]) - ll(pr['additive'])
        da = [roc_auc_score(win[b], pr[k][b]) - roc_auc_score(win[b], pr['additive'][b]) for b in Bs[:1000]]
        res['arms'][k] = dict(auc=float(roc_auc_score(win, pr[k])), log_loss=float(ll(pr[k]).mean()),
                              d_ll=[float(x) for x in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])],
                              d_auc=[float(x) for x in np.percentile(da, [2.5, 97.5])])
        r = res['arms'][k]
        print(f"{k:9s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | vs additive ΔAUC {np.round(r['d_auc'], 3)} Δll {np.round(r['d_ll'], 4)}")
    json.dump(res, open(OUT / 'ms_latent.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
