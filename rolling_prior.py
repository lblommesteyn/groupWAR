"""Rolling-origin test of the multi-season prior in the GroupWAR-T rating backbone.

Arms (ratings-only scorer unless noted):
  rapm            ridge to zero (current backbone, no history)
  rapm_trk        + tracking-description prior (previous best backbone)
  hist_only       history projection alone, no 2025-26 data
  rapm_hist       RAPM shrunk to each player's history
  rapm_hist_trk   history + tracking descriptions fit on what history leaves unexplained
  anchored_comp_hist_trk   compressed embeddings + GCN on the rapm_hist_trk backbone
"""
import json, os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import embed_v3 as e3
import groupwar_t as gw
import hist_prior as hp
import metrics2 as m2
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

SEEDS = 3
K_REL = 1500.0  # minutes at which a description gets half weight


_EXTRA = None


def extra_rows():
    """EXTRA_PBP=1: 2025-26 stints from the fetched play-by-play (fetch_pbp.py -> ms_stints_2526.pkl) for games before
    the tracking archive starts (2026-01-19), earlier than every test block, added to the in-season rating fits."""
    global _EXTRA
    if _EXTRA is None:
        _EXTRA = []
        if os.environ.get('EXTRA_PBP') == '1':
            D = pickle.load(open(ROOT / 'results/latent/ms_stints_2526.pkl', 'rb'))
            pre = {g for g, d in D['games'].items() if d['date'] < '2026-01-19' and not d['playoff']}
            for st in D['stints']:
                if st['game_id'] in pre:
                    h, a = st['lineup']
                    for k, (o, de) in enumerate(((h, a), (a, h))):
                        if st['poss_team'][k] > 0:
                            _EXTRA.append((st['game_id'], tuple(o), tuple(de), 1 - k, 100 * st['pts'][k] / st['poss_team'][k], st['poss_team'][k]))
            print(f'extra 2025-26 play-by-play rows: {len(_EXTRA)} from {len(pre)} games', flush=True)
    return _EXTRA


def tables(G, shots, win, mapg, extra=False):
    R_win, R_map = ee.rows(G, win), ee.rows(G, mapg)
    R_ext = extra_rows()
    pids = sorted({p for Rr in (R_win, R_map, R_ext) for r in Rr for p in r[1] + r[2]} | {p for d in G.values() for p in d['box']})
    pidx = {p: i for i, p in enumerate(pids)}; n = len(pids)
    desc = m2.describe2(G, win, shots)
    embs, _ = e3.build(G, shots, win, mapg, R_win, R_map, pids, pidx, desc)
    Z = embs['shrunk'][:, :len(e3.ALL)]
    m, v, known = hp.prior_vectors(pids)
    rat = {}
    _, th = ee.rapm_fit(R_win + R_ext, pidx); rat['rapm'] = hp.net(th, pids)
    # tracking prior without history (previous best)
    lam = ee.cv_lambda(R_map, Z, pidx); mdl = ee.Lin().fit(R_map, Z, pidx, lam)
    F_ = Z.shape[1]; pv = Z @ mdl.beta[2:2 + F_] - Z @ mdl.beta[2 + F_:]
    _, th2 = ee.rapm_fit(R_win, pidx, mdl.predict(R_win, Z, pidx))
    rat['rapm_trk'] = {p: pv[pidx[p]] + th2[pidx[p]] - th2[n + pidx[p]] for p in pids}
    rat['hist_only'] = hp.net(m, pids)
    _, th3 = hp.prior_rapm(R_win + R_ext, pidx, m, v); rat['rapm_hist'] = hp.net(th3, pids)
    # descriptions explain what history does not: fit mapping on map rows after removing history
    Xm = ee.Lin().ind(R_map, pidx); hist_map = Xm @ m
    Rm2 = [r[:4] + (r[4] - h,) + r[5:] for r, h in zip(R_map, hist_map)]
    lam2 = ee.cv_lambda(Rm2, Z, pidx); mdl2 = ee.Lin().fit(Rm2, Z, pidx, lam2)
    add_o = Z @ mdl2.beta[2:2 + F_]; add_d = Z @ mdl2.beta[2 + F_:]
    m2_ = m + np.concatenate([add_o, add_d])
    _, th4 = hp.prior_rapm(R_win + R_ext, pidx, m2_, v); rat['rapm_hist_trk'] = hp.net(th4, pids)
    # reliability-scaled description adjustment: rho = minutes / (minutes + K_REL)
    mins = np.array([desc.get(p, {}).get('minutes', 0.0) for p in pids]); rho = mins / (mins + K_REL)
    m3_ = m + np.concatenate([rho * add_o, rho * add_d])
    _, th5 = hp.prior_rapm(R_win + R_ext, pidx, m3_, v); rat['rapm_hist_trk_rel'] = hp.net(th5, pids)
    feats = {'comp': dict(zip(pids, embs['comp']))}
    if extra:  # prior means/variances of rapm_hist_trk and the shrunk descriptions (latent_skill)
        return rat, feats, dict(pids=pids, m=m2_, v=v, Z=Z)
    return rat, feats


def main():
    import os
    out = gw.OUT / os.environ.get('ROLLING_OUT', 'rolling_prior'); out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    recs = gw.game_records(G); A, ok = gw.rolling_A(recs)
    y = np.array([r['margin'] for r in recs], np.float32); gid = [r['game_id'] for r in recs]
    allp = {}
    RATS = ['rapm', 'rapm_trk', 'hist_only', 'rapm_hist', 'rapm_hist_trk']
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
        which = {g: T1 for g in h1} | {g: T2 for g in h2} | {g: TT for g in blk}
        keep = [k for k, g in enumerate(gid) if g in which]
        sub = [recs[k] for k in keep]; bset = set(blk)
        tr = np.array([i for i, r in enumerate(sub) if r['game_id'] not in bset and ok[keep[i]]])
        te = np.array([i for i, r in enumerate(sub) if r['game_id'] in bset])
        A_, y_ = A[keep], y[keep]
        P = {'win': (y_[te] > 0).astype(int)}
        ytr = (y_[tr] > 0).astype(int)

        def run(name, X, gcn):
            outs = [gw.train_arm(X, A_, M, Rts['rapm'], y_, tr, te, gcn, True, seed=s, oof=True) for s in range(SEEDS)]
            P[name] = np.mean([o[0] for o in outs], 0)
            # calibration fit on history only: logistic on the out-of-fold logit of the training games
            oo = np.mean([o[1][tr] for o in outs], 0)
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            cal = LogisticRegression(C=1e4).fit(lg(oo)[:, None], ytr)
            P[f'{name}_cal'] = cal.predict_proba(lg(P[name])[:, None])[:, 1]
            print(f'  {name:24s} auc {roc_auc_score(P["win"], P[name]):.3f}  calibration slope {cal.coef_[0, 0]:.2f}', flush=True)

        for rk in RATS:
            tab = {r['game_id']: (which[r['game_id']][0][rk], which[r['game_id']][0][rk], which[r['game_id']][1]) for r in sub}
            M, Rts, Xs = gw.tensors(sub, tab)
            zeros = np.zeros_like(Xs['comp'][..., :1])
            run(rk, zeros, False)
            if rk == 'rapm_hist_trk':
                run('anchored_comp_hist_trk', Xs['comp'], True)
        np.savez(f, **P); allp[b0] = P
    names = [k for k in next(iter(allp.values())) if k != 'win']
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {n: np.concatenate([p[n] for p in allp.values()]) for n in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'arms': {}}
    for n in names:
        r = res['arms'][n] = dict(auc=float(roc_auc_score(win, pr[n])), log_loss=float(log_loss(win, np.clip(pr[n], 1e-6, 1 - 1e-6))))
        for base in ('rapm', 'rapm_trk'):
            d = ll(pr[n]) - ll(pr[base])
            r[f'd_ll_vs_{base}'] = [float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])]
            r[f'd_auc_vs_{base}'] = [float(v) for v in np.percentile(
                [roc_auc_score(win[b], pr[n][b]) - roc_auc_score(win[b], pr[base][b]) for b in Bs[:1000]], [2.5, 97.5])]
    json.dump(res, open(out / 'rolling_prior.json', 'w'), indent=1)
    print(f'\npooled {len(win)} games')
    for n, r in res['arms'].items():
        print(f"{n:24s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | vs rapm ΔAUC {np.round(r['d_auc_vs_rapm'], 3)} Δll {np.round(r['d_ll_vs_rapm'], 4)} | vs rapm_trk ΔAUC {np.round(r['d_auc_vs_rapm_trk'], 3)} Δll {np.round(r['d_ll_vs_rapm_trk'], 4)}")


if __name__ == '__main__':
    main()
