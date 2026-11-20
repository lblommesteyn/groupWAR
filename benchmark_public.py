"""Public-metric benchmark on the same rolling blocks and 423 test games as rolling_prior.

Ratings-only scorer for every arm (same as our rating arms), so only the player ratings differ.
  darko_pit    DARKO DPM point-in-time: each game uses each player's latest DPM dated before that game;
               test games use values before the block start (no test-block information)
  epm_prev     EPM 2024-25 end of season (leak-free, stale)
  epm_end      EPM 2025-26 end of season (LEAKY: includes the test games; reference only)
  ours_plus_darko   mean of our rapm_hist_trk and darko_pit (does ours add to DARKO?)
Baselines reloaded from rolling_prior: rapm, rapm_hist_trk, anchored_comp_hist_trk.
Unrated players get the replacement-ish value -2.0 (both public metrics are per 100 possessions).
"""
import bisect, csv, json, pickle
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import groupwar_t as gw
import rolling_prior as rp
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

B = ROOT / 'results/benchmarks'
UNRATED = -2.0
SEEDS = 3


def load_darko():
    H = json.load(open(B / 'darko_history.json'))
    out = {}
    for p, h in H.items():
        if 'dpm' not in h:
            continue
        rows = sorted((d, v) for d, v, f in zip(h['date'], h['dpm'], h.get('future_game', [0] * len(h['date'])))
                      if v is not None and not f)
        out[int(p)] = ([r[0] for r in rows], [r[1] for r in rows])
    return out


def darko_at(D, p, date):
    if p not in D:
        return UNRATED
    ds, vs = D[p]; j = bisect.bisect_left(ds, date) - 1  # strictly before `date`
    return vs[j] if j >= 0 else UNRATED


def load_epm(f):
    return {int(r['player_id']): float(r['tot']) for r in csv.DictReader(open(B / f, encoding='utf-8')) if r['tot'] not in ('', 'NA')}


def main():
    out = gw.OUT / 'benchmark'; out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    D = load_darko(); epm_prev, epm_end = load_epm('epm_2025.csv'), load_epm('epm_2026.csv')
    recs = gw.game_records(G); A, ok = gw.rolling_A(recs)
    y = np.array([r['margin'] for r in recs], np.float32); gid = [r['game_id'] for r in recs]
    allp = {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        f = out / f'block_{b0}.npz'
        if f.exists():
            allp[b0] = dict(np.load(f)); continue
        hist = sorted(g for g, d in G.items() if d['date'] < b0)
        mid = sorted(G[g]['date'] for g in hist)[len(hist) // 2]
        h1 = [g for g in hist if G[g]['date'] < mid]; h2 = [g for g in hist if G[g]['date'] >= mid]
        blk = [g for g, d in G.items() if b0 <= d['date'] < b1]
        keep = [k for k, g in enumerate(gid) if g in set(hist + blk)]
        sub = [recs[k] for k in keep]; bset = set(blk)
        tr = np.array([i for i, r in enumerate(sub) if r['game_id'] not in bset and ok[keep[i]]])
        te = np.array([i for i, r in enumerate(sub) if r['game_id'] in bset])
        A_, y_ = A[keep], y[keep]
        base = dict(np.load(gw.OUT / 'rolling_prior' / f'block_{b0}.npz'))
        P = {'win': (y_[te] > 0).astype(int)}
        assert (P['win'] == base['win']).all()
        for k in ('rapm', 'rapm_hist_trk', 'anchored_comp_hist_trk'):
            P[k] = base[k]
        # our rapm_hist_trk tables (same cross-fit as rolling_prior) for the combined arm
        T1, T2, TT = rp.tables(G, shots, h2, h1), rp.tables(G, shots, h1, h2), rp.tables(G, shots, h1 + h2, h2)
        ours = {g: T1[0]['rapm_hist_trk'] for g in h1} | {g: T2[0]['rapm_hist_trk'] for g in h2} | {g: TT[0]['rapm_hist_trk'] for g in blk}
        players = {r['game_id']: set(r['secs']) for r in sub}
        def darko_table(g):
            date = b0 if g in bset else G[g]['date']
            return {p: darko_at(D, p, date) for p in players[g]}
        arms = {
            'darko_pit': darko_table,
            'epm_prev': lambda g: {p: epm_prev.get(p, UNRATED) for p in players[g]},
            'epm_end_LEAKY': lambda g: {p: epm_end.get(p, UNRATED) for p in players[g]},
            'ours_plus_darko': lambda g: {p: 0.5 * (ours[g].get(p, 0.0) + darko_at(D, p, b0 if g in bset else G[g]['date']))
                                          for p in players[g]},
        }
        for name, fn in arms.items():
            tab = {}
            for r in sub:
                t = fn(r['game_id']); tab[r['game_id']] = (t, t, {'x': {p: np.zeros(1, np.float32) for p in t}})
            M, Rts, Xs = gw.tensors(sub, tab)
            zeros = np.zeros_like(Xs['x'])
            P[name] = np.mean([gw.train_arm(zeros, A_, M, Rts['rapm'], y_, tr, te, False, True, seed=s) for s in range(SEEDS)], 0)
            print(f'{b0} {name:18s} auc {roc_auc_score(P["win"], P[name]):.3f}', flush=True)
        np.savez(f, **P); allp[b0] = P
    names = [k for k in next(iter(allp.values())) if k != 'win']
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {n: np.concatenate([p[n] for p in allp.values()]) for n in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'arms': {}}
    for n in names:
        r = res['arms'][n] = dict(auc=float(roc_auc_score(win, pr[n])), log_loss=float(log_loss(win, np.clip(pr[n], 1e-6, 1 - 1e-6))))
        for base in ('darko_pit', 'anchored_comp_hist_trk'):
            d = ll(pr[n]) - ll(pr[base])
            r[f'd_ll_vs_{base}'] = [float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])]
            r[f'd_auc_vs_{base}'] = [float(v) for v in np.percentile(
                [roc_auc_score(win[b], pr[n][b]) - roc_auc_score(win[b], pr[base][b]) for b in Bs[:1000]], [2.5, 97.5])]
    json.dump(res, open(out / 'benchmark.json', 'w'), indent=1)
    print(f'\npooled {len(win)} games')
    for n, r in res['arms'].items():
        print(f"{n:24s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | vs DARKO ΔAUC {np.round(r['d_auc_vs_darko_pit'], 3)} Δll {np.round(r['d_ll_vs_darko_pit'], 4)} | vs ours ΔAUC {np.round(r['d_auc_vs_anchored_comp_hist_trk'], 3)} Δll {np.round(r['d_ll_vs_anchored_comp_hist_trk'], 4)}")


if __name__ == '__main__':
    main()
