"""Rolling test of EPV-target ratings (denoised RAPM) against the current best backbone.

Possession rows: for each tracked possession, the cross-fitted EPV at the last sample where the offense
holds the ball (shot quality and decisions, no shot luck), lineup at that sample. Row weight 1, y = 100*EPV.
Arms (ratings-only scorer, same rolling blocks; baselines reloaded from rolling_prior for paired comparison):
  epv_hist        EPV-target ridge with the multi-season prior
  epv_hist_trk    + tracking descriptions (fit on what history leaves unexplained, points rows)
  blend           mean of rapm_hist_trk and epv_hist_trk
  anchored_comp_blend   compressed embeddings + GCN on the blend backbone
EPV for block b comes from out_cf/epv_{b}.npz (models trained on games before b only).
"""
import json, pickle
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import groupwar_t as gw
import hist_prior as hp
import rolling_prior as rp
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

CF = ROOT / 'results/epv/out_cf'
SEEDS = 3


def pull(b):
    import subprocess
    CF.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp', '-q', f'pc:C:/Users/LukeB/dev/court-epv/out_cf/epv_{b}.npz', str(CF / f'epv_{b}.npz')],
                   stderr=subprocess.DEVNULL)


def epv_rows(npz, G, games):
    d = np.load(npz, allow_pickle=True)
    epv, gid, meta = d['epv'], d['gid'], d['meta']
    games = set(games)
    home = {g: {p for st in G[g]['stints'] for p in st['lineup'][0]} for g in games if g in G}
    last = {}
    for i in range(len(epv)):
        g = str(gid[i])
        if g not in games or not np.isfinite(epv[i]):
            continue
        off = tuple(int(x) for x in meta[i, 3:8]); de = tuple(int(x) for x in meta[i, 8:13])
        if int(meta[i, 2]) in off:  # offense holds the ball
            last[(g, int(meta[i, 0]))] = (off, de, float(epv[i]))
    R = []
    for (g, k), (off, de, v) in last.items():
        R.append((g, off, de, 1 if off[0] in home[g] else 0, 100 * v, 1.0))
    return R


def epv_ratings(G, R_epv, pids, pidx, Z, R_map):
    n = len(pids)
    m, v, _ = hp.prior_vectors(pids)
    _, th = hp.prior_rapm(R_epv, pidx, m, v)
    out = {'epv_hist': hp.net(th, pids)}
    Xm = ee.Lin().ind(R_map, pidx); Rm2 = [r[:4] + (r[4] - h,) + r[5:] for r, h in zip(R_map, Xm @ m)]
    mdl = ee.Lin().fit(Rm2, Z, pidx, ee.cv_lambda(Rm2, Z, pidx)); F_ = Z.shape[1]
    m2 = m + np.concatenate([Z @ mdl.beta[2:2 + F_], Z @ mdl.beta[2 + F_:]])
    _, th2 = hp.prior_rapm(R_epv, pidx, m2, v)
    out['epv_hist_trk'] = hp.net(th2, pids)
    return out


def tables(G, shots, win, mapg, npz):
    import embed_v3 as e3, metrics2 as m2
    rat, feats = rp.tables(G, shots, win, mapg)
    R_win, R_map = ee.rows(G, win), ee.rows(G, mapg)
    pids = sorted({p for Rr in (R_win, R_map) for r in Rr for p in r[1] + r[2]} | {p for d in G.values() for p in d['box']})
    pidx = {p: i for i, p in enumerate(pids)}
    desc = m2.describe2(G, win, shots)
    embs, _ = e3.build(G, shots, win, mapg, R_win, R_map, pids, pidx, desc)
    Z = embs['shrunk'][:, :len(e3.ALL)]
    R_epv = [r for r in epv_rows(npz, G, win) if all(p in pidx for p in r[1] + r[2])]
    rat.update(epv_ratings(G, R_epv, pids, pidx, Z, R_map))
    rat['blend'] = {p: 0.5 * (rat['rapm_hist_trk'].get(p, 0) + rat['epv_hist_trk'].get(p, 0)) for p in pids}
    rat['blend_rel'] = {p: 0.5 * (rat['rapm_hist_trk_rel'].get(p, 0) + rat['epv_hist_trk'].get(p, 0)) for p in pids}
    return rat, feats, len(R_epv)


def main():
    out = gw.OUT / 'rolling_epv'; out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    recs = gw.game_records(G); A, ok = gw.rolling_A(recs)
    y = np.array([r['margin'] for r in recs], np.float32); gid = [r['game_id'] for r in recs]
    allp = {}
    for b0, b1 in zip(BLOCKS[:-1], BLOCKS[1:]):
        f = out / f'block_{b0}.npz'
        if f.exists():
            allp[b0] = dict(np.load(f)); continue
        npz = CF / f'epv_{b0}.npz'
        import time
        while not npz.exists():  # cross-fit EPV for this block still training on the PC
            pull(b0); time.sleep(60)
        hist = sorted(g for g, d in G.items() if d['date'] < b0)
        mid = sorted(G[g]['date'] for g in hist)[len(hist) // 2]
        h1 = [g for g in hist if G[g]['date'] < mid]; h2 = [g for g in hist if G[g]['date'] >= mid]
        blk = [g for g, d in G.items() if b0 <= d['date'] < b1]
        T1, T2, TT = tables(G, shots, h2, h1, npz), tables(G, shots, h1, h2, npz), tables(G, shots, h1 + h2, h2, npz)
        print(f'block {b0}: history {len(hist)}, test {len(blk)}, EPV possession rows {TT[2]}', flush=True)
        which = {g: T1 for g in h1} | {g: T2 for g in h2} | {g: TT for g in blk}
        keep = [k for k, g in enumerate(gid) if g in which]
        sub = [recs[k] for k in keep]; bset = set(blk)
        tr = np.array([i for i, r in enumerate(sub) if r['game_id'] not in bset and ok[keep[i]]])
        te = np.array([i for i, r in enumerate(sub) if r['game_id'] in bset])
        A_, y_ = A[keep], y[keep]
        base = dict(np.load(gw.OUT / 'rolling_prior' / f'block_{b0}.npz'))
        P = {'win': (y_[te] > 0).astype(int), 'rapm': base['rapm'], 'rapm_hist_trk': base['rapm_hist_trk'],
             'anchored_comp_hist_trk': base['anchored_comp_hist_trk']}
        assert (P['win'] == base['win']).all()
        for rk in ('rapm_hist_trk_rel', 'epv_hist', 'epv_hist_trk', 'blend', 'blend_rel'):
            tab = {r['game_id']: (which[r['game_id']][0][rk], which[r['game_id']][0][rk], which[r['game_id']][1]) for r in sub}
            M, Rts, Xs = gw.tensors(sub, tab)
            zeros = np.zeros_like(Xs['comp'][..., :1])
            P[rk] = np.mean([gw.train_arm(zeros, A_, M, Rts['rapm'], y_, tr, te, False, True, seed=s) for s in range(SEEDS)], 0)
            print(f'  {rk:24s} auc {roc_auc_score(P["win"], P[rk]):.3f}', flush=True)
            if rk == 'blend_rel':
                P['anchored_comp_blend_rel'] = np.mean([gw.train_arm(Xs['comp'], A_, M, Rts['rapm'], y_, tr, te, True, True, seed=s)
                                                    for s in range(SEEDS)], 0)
                print(f'  anchored_comp_blend_rel  auc {roc_auc_score(P["win"], P["anchored_comp_blend_rel"]):.3f}', flush=True)
        np.savez(f, **P); allp[b0] = P
    names = [k for k in next(iter(allp.values())) if k != 'win']
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {n: np.concatenate([p[n] for p in allp.values()]) for n in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'arms': {}}
    for n in names:
        r = res['arms'][n] = dict(auc=float(roc_auc_score(win, pr[n])), log_loss=float(log_loss(win, np.clip(pr[n], 1e-6, 1 - 1e-6))))
        for base in ('rapm_hist_trk', 'anchored_comp_hist_trk'):
            d = ll(pr[n]) - ll(pr[base])
            r[f'd_ll_vs_{base}'] = [float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])]
            r[f'd_auc_vs_{base}'] = [float(v) for v in np.percentile(
                [roc_auc_score(win[b], pr[n][b]) - roc_auc_score(win[b], pr[base][b]) for b in Bs[:1000]], [2.5, 97.5])]
    json.dump(res, open(out / 'rolling_epv.json', 'w'), indent=1)
    print(f'\npooled {len(win)} games')
    for n, r in res['arms'].items():
        print(f"{n:24s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | vs rapm_hist_trk ΔAUC {np.round(r['d_auc_vs_rapm_hist_trk'], 3)} Δll {np.round(r['d_ll_vs_rapm_hist_trk'], 4)} | vs best ΔAUC {np.round(r['d_auc_vs_anchored_comp_hist_trk'], 3)} Δll {np.round(r['d_ll_vs_anchored_comp_hist_trk'], 4)}")


if __name__ == '__main__':
    main()
