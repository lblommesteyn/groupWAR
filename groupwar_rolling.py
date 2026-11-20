"""Rolling-origin evaluation of GroupWAR-T arms: refit everything before each ~2-week block, predict the block.

Inside each block's history: split at the median date into H1/H2; H1 games get H2-built tables and vice
versa (cross-fit); block games get tables built on H1+H2 with the description map learned on H2.
"""
import json, pickle, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import groupwar_t as gw

BLOCKS = ['2026-03-01', '2026-03-15', '2026-03-29', '2026-04-12', '2026-04-26', '2026-05-10', '2026-07-01']
SEEDS = 3


def main():
    out = gw.OUT / 'rolling'; out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    cols_list = {'box': ee.BOX, 'trk': gw.cols_all() if hasattr(gw, 'cols_all') else
                 ee.TRK_OFF + ee.TRK_DEF + gw.m2.NEW_OFF + gw.m2.NEW_DEF,
                 'box_trk': ee.BOX + ee.TRK_OFF + ee.TRK_DEF + gw.m2.NEW_OFF + gw.m2.NEW_DEF}
    recs = gw.game_records(G)
    A, ok = gw.rolling_A(recs)
    Ash = gw.shuffle_graph(torch.tensor(A)).numpy()
    y = np.array([r['margin'] for r in recs], np.float32)
    date = np.array([r['date'] for r in recs]); gid = [r['game_id'] for r in recs]
    allp = {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        f = out / f'block_{b0}.npz'
        if f.exists():
            allp[b0] = dict(np.load(f)); print('cached', b0); continue
        hist = sorted(g for g, d in G.items() if d['date'] < b0)
        mid = sorted(G[g]['date'] for g in hist)[len(hist) // 2]
        h1 = [g for g in hist if G[g]['date'] < mid]; h2 = [g for g in hist if G[g]['date'] >= mid]
        blk = [g for g, d in G.items() if b0 <= d['date'] < b1]
        print(f'block {b0}: history {len(hist)} (split {mid}), test {len(blk)}', flush=True)
        t1 = gw.ratings_and_features(G, shots, h2, h1, cols_list)  # for H1 games
        t2 = gw.ratings_and_features(G, shots, h1, h2, cols_list)  # for H2 games
        tt = gw.ratings_and_features(G, shots, h1 + h2, h2, cols_list)
        which = {g: t1 for g in h1} | {g: t2 for g in h2} | {g: tt for g in blk}
        keep = [k for k, g in enumerate(gid) if g in which]
        sub = [recs[k] for k in keep]
        M, Rts, Xs = gw.tensors(sub, {r['game_id']: which[r['game_id']] for r in sub})
        A_, Ash_, y_, ok_ = A[keep], Ash[keep], y[keep], ok[keep]
        bset = set(blk)
        tr = np.array([i for i, r in enumerate(sub) if r['game_id'] not in bset and ok_[i]])
        te = np.array([i for i, r in enumerate(sub) if r['game_id'] in bset])
        zeros = np.zeros_like(Xs['box'][..., :1])
        arms = {'ratings_rapm': (zeros, A_, 'rapm', False, True),
                'ratings_rapm_trkprior': (zeros, A_, 'rapm_t', False, True),
                'gcn_only_box': (Xs['box'], A_, 'rapm', True, False),
                'gcn_only_trk': (Xs['trk'], A_, 'rapm', True, False),
                'gcn_only_trk_SHUFFLED': (Xs['trk'], Ash_, 'rapm', True, False),
                'anchored_box_trk': (Xs['box_trk'], A_, 'rapm_t', True, True),
                'anchored_box_trk_SHUFFLED': (Xs['box_trk'], Ash_, 'rapm_t', True, True)}
        P = {'win': (y_[te] > 0).astype(int), 'gid': np.array([sub[i]['game_id'] for i in te])}
        for name, (X, Ag, rk, gcn, rat) in arms.items():
            P[name] = np.mean([gw.train_arm(X, Ag, M, Rts[rk], y_, tr, te, gcn, rat, seed=s) for s in range(SEEDS)], 0)
            print(f'  {name:26s} auc {roc_auc_score(P["win"], P[name]):.3f}', flush=True)
        np.savez(f, **P); allp[b0] = P
    # pooled metrics + game bootstrap
    names = [k for k in next(iter(allp.values())) if k not in ('win', 'gid')]
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {n: np.concatenate([p[n] for p in allp.values()]) for n in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'blocks': BLOCKS, 'arms': {}}
    base = pr['ratings_rapm']
    for n in names:
        d = ll(pr[n]) - ll(base)
        res['arms'][n] = dict(auc=float(roc_auc_score(win, pr[n])), log_loss=float(log_loss(win, np.clip(pr[n], 1e-6, 1 - 1e-6))),
                              d_ll_ci=[float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])],
                              d_auc_ci=[float(v) for v in np.percentile([roc_auc_score(win[b], pr[n][b]) - roc_auc_score(win[b], base[b]) for b in Bs[:1000]], [2.5, 97.5])],
                              block_auc=[float(roc_auc_score(p['win'], p[n])) for p in allp.values()])
    # paired: real graph vs shuffled
    for a, b in [('gcn_only_trk', 'gcn_only_trk_SHUFFLED'), ('anchored_box_trk', 'anchored_box_trk_SHUFFLED'),
                 ('gcn_only_trk', 'gcn_only_box')]:
        d = ll(pr[a]) - ll(pr[b])
        res[f'{a}_vs_{b}'] = dict(d_ll=float(d.mean()), ci=[float(v) for v in np.percentile([d[x].mean() for x in Bs], [2.5, 97.5])],
                                  d_auc_ci=[float(v) for v in np.percentile([roc_auc_score(win[x], pr[a][x]) - roc_auc_score(win[x], pr[b][x]) for x in Bs[:1000]], [2.5, 97.5])])
    json.dump(res, open(out / 'rolling.json', 'w'), indent=1)
    print(f"\npooled test games: {len(win)}")
    for n, r in res['arms'].items():
        print(f"{n:26s} auc {r['auc']:.3f} ll {r['log_loss']:.4f}  ΔAUC vs RAPM {np.round(r['d_auc_ci'], 3)}  Δll {np.round(r['d_ll_ci'], 4)}  blocks {np.round(r['block_auc'], 2)}")
    for k, v in res.items():
        if '_vs_' in k:
            print(k, np.round(v['d_ll'], 4), 'll CI', np.round(v['ci'], 4), 'AUC CI', np.round(v['d_auc_ci'], 3))


if __name__ == '__main__':
    main()
