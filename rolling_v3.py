"""Rolling-origin test of embedding v3 (shrunk / compressed) vs raw, in the GroupWAR-T anchored scorer."""
import json, pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import embed_v3 as e3
import groupwar_t as gw
import metrics2 as m2
from groupwar_rolling import BLOCKS

SEEDS = 3


def tables(G, shots, win, mapg):
    R_win, R_map = ee.rows(G, win), ee.rows(G, mapg)
    pids = sorted({p for Rr in (R_win, R_map) for r in Rr for p in r[1] + r[2]} | {p for d in G.values() for p in d['box']})
    pidx = {p: i for i, p in enumerate(pids)}; n = len(pids)
    desc = m2.describe2(G, win, shots)
    _, th = ee.rapm_fit(R_win, pidx)
    rapm = {p: th[pidx[p]] - th[n + pidx[p]] for p in pids}
    embs, info = e3.build(G, shots, win, mapg, R_win, R_map, pids, pidx, desc)
    out = {}
    for name in ('raw', 'shrunk'):
        Z = embs[name][:, :len(e3.ALL)]
        lam = ee.cv_lambda(R_map, Z, pidx); mdl = ee.Lin().fit(R_map, Z, pidx, lam)
        F_ = len(e3.ALL); wo, wd = mdl.beta[2:2 + F_], mdl.beta[2 + F_:]
        pv = Z @ wo - Z @ wd
        _, th2 = ee.rapm_fit(R_win, pidx, mdl.predict(R_win, Z, pidx))
        out[name] = {p: pv[pidx[p]] + th2[pidx[p]] - th2[n + pidx[p]] for p in pids}
    feats = {k: dict(zip(pids, v)) for k, v in embs.items()}
    return (rapm, out, feats, info)


def to_gw(tab, rating_key):
    rapm, rt, feats, _ = tab
    return (rapm, rt[rating_key], feats)


def main():
    out = gw.OUT / 'rolling_v3'; out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    recs = gw.game_records(G); A, ok = gw.rolling_A(recs)
    Ash = gw.shuffle_graph(torch.tensor(A)).numpy()
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
        print(f'block {b0}: history {len(hist)}, test {len(blk)}', flush=True)
        T1, T2, TT = tables(G, shots, h2, h1), tables(G, shots, h1, h2), tables(G, shots, h1 + h2, h2)
        if b0 == BLOCKS[-2]:
            json.dump({k: (v if not isinstance(v, dict) else {kk: float(vv) if np.isscalar(vv) else vv for kk, vv in v.items()})
                       for k, v in TT[3].items()}, open(out / 'embedding_info.json', 'w'), indent=1, default=str)
        keep = [k for k, g in enumerate(gid) if g in set(h1 + h2 + blk)]
        sub = [recs[k] for k in keep]
        which = {g: T1 for g in h1} | {g: T2 for g in h2} | {g: TT for g in blk}
        P = {}
        bset = set(blk)
        tr = np.array([i for i, r in enumerate(sub) if r['game_id'] not in bset and ok[keep[i]]])
        te = np.array([i for i, r in enumerate(sub) if r['game_id'] in bset])
        A_, Ash_, y_ = A[keep], Ash[keep], y[keep]
        P['win'] = (y_[te] > 0).astype(int)
        for rk in ('raw', 'shrunk'):
            M, Rts, Xs = gw.tensors(sub, {r['game_id']: to_gw(which[r['game_id']], rk) for r in sub})
            zeros = np.zeros_like(Xs['raw'][..., :1])
            arms = {f'ratings_prior_{rk}': (zeros, A_, False)}
            if rk == 'raw':
                arms['ratings_rapm'] = None
                arms['anchored_raw'] = (Xs['raw'], A_, True)
            else:
                arms['anchored_shrunk'] = (Xs['shrunk'], A_, True)
                arms['anchored_comp'] = (Xs['comp'], A_, True)
                arms['anchored_comp_SHUFFLED'] = (Xs['comp'], Ash_, True)
            for name, spec in arms.items():
                if spec is None:
                    P[name] = np.mean([gw.train_arm(zeros, A_, M, Rts['rapm'], y_, tr, te, False, True, seed=s) for s in range(SEEDS)], 0)
                else:
                    X, Ag, gcn = spec
                    P[name] = np.mean([gw.train_arm(X, Ag, M, Rts['rapm_t'], y_, tr, te, gcn, True, seed=s) for s in range(SEEDS)], 0)
                print(f'  {name:24s} auc {roc_auc_score(P["win"], P[name]):.3f}', flush=True)
        np.savez(f, **P); allp[b0] = P
    names = [k for k in next(iter(allp.values())) if k != 'win']
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {n: np.concatenate([p[n] for p in allp.values()]) for n in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'arms': {}}
    for n in names:
        res['arms'][n] = dict(auc=float(roc_auc_score(win, pr[n])), log_loss=float(log_loss(win, np.clip(pr[n], 1e-6, 1 - 1e-6))))
        for base in ('ratings_rapm', 'anchored_raw'):
            d = ll(pr[n]) - ll(pr[base])
            res['arms'][n][f'd_ll_vs_{base}'] = [float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])]
            res['arms'][n][f'd_auc_vs_{base}'] = [float(v) for v in np.percentile(
                [roc_auc_score(win[b], pr[n][b]) - roc_auc_score(win[b], pr[base][b]) for b in Bs[:1000]], [2.5, 97.5])]
    json.dump(res, open(out / 'rolling_v3.json', 'w'), indent=1)
    print(f'\npooled {len(win)} games')
    for n, r in res['arms'].items():
        print(f"{n:24s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | vs RAPM ΔAUC {np.round(r['d_auc_vs_ratings_rapm'], 3)} Δll {np.round(r['d_ll_vs_ratings_rapm'], 4)} | vs raw ΔAUC {np.round(r['d_auc_vs_anchored_raw'], 3)} Δll {np.round(r['d_ll_vs_anchored_raw'], 4)}")


if __name__ == '__main__':
    main()
