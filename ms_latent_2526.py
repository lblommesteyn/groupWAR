"""Payoff test: the usage/load lineup model with seven seasons of lineups, on the 2025-26 rolling test.

History: play-by-play stints 2019-20 .. 2025-26 (ms_stints.pkl from v3 for 2019-23, ms_stints_2325.pkl and
ms_stints_2526.pkl from fetch_pbp.py). For each 2025-26 rolling block (groupwar_rolling.BLOCKS, the same 423 test
games, same order as rolling_prior), fit the additive and usage arms on every stint before the block (recency-weighted,
hyperparameters and calibration on the last 30 days, exactly ms_latent.run_arm) and predict each test game from its
realised stints.
Stack on the current best model (rolling_cal: anchored_comp_hist_trk_cal, calibrated on history):
  best_load = logit(best_cal) + [logit(usage) - logit(additive)]  (the load adjustment, in the arms' own calibrated units)
Scores: log loss / AUC and margin correlation, paired game-bootstrap CIs vs additive, vs best_cal, vs DARKO (calibrated).
Output: results/latent/ms2526_block_*.npz, ms2526.json.   Usage (PC): python ms_latent_2526.py
"""
import json, pickle
from datetime import date, timedelta
import numpy as np
from sklearn.metrics import roc_auc_score

import embed_eval as ee
import groupwar_t as gw
import ms_latent as ml
import latent_skill as ls
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
ARMS = {'additive': (False, False), 'usage': (True, False)}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def main():
    S, G = [], {}
    for f in ('ms_stints.pkl', 'ms_stints_2325.pkl', 'ms_stints_2526.pkl'):
        d = pickle.load(open(OUT / f, 'rb')); S += d['stints']; G.update(d['games'])
        print(f, len(d['games']), 'games', flush=True)
    PG = pickle.load(open(ee.E / 'player_games.pkl', 'rb')); PG = {g: d for g, d in PG.items() if d['stints']}
    recs = gw.game_records(PG)
    res_blocks = {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        f = OUT / f'ms2526_block_{b0}.npz'
        order = [r['game_id'] for r in recs if b0 <= r['date'] < b1]
        base = np.load(gw.OUT / 'rolling_cal' / f'block_{b0}.npz')
        bench = np.load(gw.OUT / 'benchmark_cal' / f'block_{b0}.npz')
        if f.exists():
            res_blocks[b0] = dict(np.load(f)); continue
        missing = [g for g in order if g not in G]
        train = sorted((g for g in G if G[g]['date'] < b0), key=lambda g: G[g]['date'])
        vcut = str(date.fromisoformat(b0) - timedelta(days=ml.VAL_DAYS))
        val_tr = [g for g in train if G[g]['date'] < vcut]; val = [g for g in train if G[g]['date'] >= vcut]
        print(f'block {b0}: train {len(train)} games, test {len(order)} ({len(missing)} without fetched stints)', flush=True)
        test = [g for g in order if g in G]
        P = {'win': base['win'], 'margin': np.array([PG[g]['final'][0] - PG[g]['final'][1] for g in order], float),
             'best_cal': base['anchored_comp_hist_trk_cal'], 'best': base['anchored_comp_hist_trk'],
             'darko_cal': bench['darko_pit_cal'], 'rapm_hist_trk_cal': base['rapm_hist_trk_cal']}
        for arm, (usage, fit_) in ARMS.items():
            p, info = ml.run_arm(S, G, train, val_tr, val, test, b0, usage, fit_)
            full = np.full(len(order), 0.5); full[[order.index(g) for g in test]] = p
            P[arm] = full
            print(f'  {arm:9s} auc {roc_auc_score(P["win"], full):.3f} {info}', flush=True)
        P['best_load'] = 1 / (1 + np.exp(-(logit(P['best_cal']) + logit(P['usage']) - logit(P['additive']))))
        print(f'  best_cal auc {roc_auc_score(P["win"], P["best_cal"]):.3f}  best_load auc {roc_auc_score(P["win"], P["best_load"]):.3f}', flush=True)
        np.savez(f, **P); res_blocks[b0] = P
    K = ['additive', 'usage', 'best', 'best_cal', 'best_load', 'darko_cal', 'rapm_hist_trk_cal']
    win = np.concatenate([p['win'] for p in res_blocks.values()]); m = np.concatenate([p['margin'] for p in res_blocks.values()])
    pr = {k: np.concatenate([p[k] for p in res_blocks.values()]) for k in K}
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (4000, len(win)))
    cor = lambda x, b: np.corrcoef(x[b], m[b])[0, 1]
    out = {'n': int(len(win)), 'arms': {}}
    print(f'\npooled {len(win)} games')
    for k in K:
        r = out['arms'][k] = dict(auc=float(roc_auc_score(win, pr[k])), log_loss=float(ll(pr[k]).mean()),
                                  margin_corr=float(np.corrcoef(logit(pr[k]), m)[0, 1]))
        for b in ('additive', 'best_cal', 'darko_cal'):
            if b == k:
                continue
            d = ll(pr[k]) - ll(pr[b])
            r[f'd_ll_vs_{b}'] = [float(np.mean(d))] + [float(q) for q in np.percentile([d[i].mean() for i in Bs], [2.5, 97.5])]
            dc = [cor(logit(pr[k]), i) - cor(logit(pr[b]), i) for i in Bs[:2000]]
            r[f'd_corr_vs_{b}'] = [float(np.mean(dc))] + [float(q) for q in np.percentile(dc, [2.5, 97.5])]
        print(f"{k:18s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} corr {r['margin_corr']:.3f} | "
              + ' | '.join(f"vs {b}: Δll {np.round(r[f'd_ll_vs_{b}'], 4)} Δcorr {np.round(r[f'd_corr_vs_{b}'], 3)}"
                           for b in ('additive', 'best_cal', 'darko_cal') if f'd_ll_vs_{b}' in r))
    json.dump(out, open(OUT / 'ms2526.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
