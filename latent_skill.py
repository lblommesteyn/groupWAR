"""Latent-skill lineup model v1: opportunity (usage) + efficiency under load + learned fit, on stint rows.

Per stint and offense side (rows as embed_eval.rows: y = points per 100 possessions, weight = possessions):
  additive   b + h*home + sum_O o_i - sum_D d_j            o, d ~ N(m, v) = rapm_hist_trk prior (history + descriptions)
  usage      share_i = softmax(u_O)_i, fit to who used the possessions (FGA + 0.44 FTA + TOV per player, PBP)
             natural share s0_i = e^u_i / (e^u_i + 4): the share next to four average-usage teammates
  load       - sum_i share_i * g_i * (share_i - s0_i) * 100,   g_i = softplus(g0 + dg_i), dg_i ~ N(0, TAU_G^2)
             (forced above natural share costs efficiency; learned from lineups that push players up or down)
  fit        pairwise style interactions: z_i = A desc_i + dz_i (k dims, dz ~ N(0, TAU_Z^2))
             off-off 1/2[(sum z)'W_o(sum z) - sum z'W_o z], def-def same with W_d, off-def (sum z_O)' M (sum z_D)
Loss: sum w (y - yhat)^2 / (2 s2) + ALPHA * multinomial NLL of usage counts + Gaussian priors (MAP, L-BFGS).
Arms: additive, +usage, +fit, full. Rolling evaluation on the same blocks/games as rolling_prior, cross-fitted like it
(fit on h2 -> predict h1, h1 -> h2, h1+h2 -> test block); game margin = sum over the game's stints of
poss * (yhat_home - yhat_away) / 100; probability via a logistic fitted on the cross-fitted history predictions.
"""
import json, os, pickle, sys
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import embed_eval as ee
import groupwar_t as gw
import rolling_prior as rp
from groupwar_rolling import BLOCKS
from masked_paths import ROOT

torch.set_default_dtype(torch.float64)
OUT = gw.OUT / 'latent'
K, TAU_G, TAU_Z, TAU_U, ALPHA, L2_W = 3, 0.5, 0.3, 1.5, 1.0, 1.0
L2_GRID = [10.0, 100.0, 1000.0, 10000.0]
ARMS = {'additive': (False, False), 'usage': (True, False), 'fit': (False, True), 'full': (True, True)}
# load slope transferred from the four-season fit (ms_latent.py: softplus(g0) 0.93-1.15 across 2022-23 blocks)
G0_MS = 1.08
ARMS_FIX = {'usage_ms': G0_MS}  # usage arm with the population load slope fixed, not learned from one season


def stint_rows(GU, gids):
    """-> list of (g, off5, def5, home, y per 100, w, use counts per off player)"""
    R = []
    for g in gids:
        for st in GU[g]['stints']:
            h, a = st['lineup']
            for k, (o, de) in enumerate(((h, a), (a, h))):
                if st['poss_team'][k] > 0:
                    c = [(lambda u: u[0] + 0.44 * u[1] + u[2])(st['use'].get(p, [0, 0, 0, 0, []])) for p in o]
                    R.append((g, tuple(o), tuple(de), 1 - k, 100 * st['pts'][k] / st['poss_team'][k], st['poss_team'][k], c))
    return R


class Model(torch.nn.Module):
    def __init__(self, n, m, v, Z, usage, fit, l2w=L2_W, g0_fix=None):
        super().__init__()
        self.l2w = l2w
        self.usage, self.fit_ = usage, fit
        f64 = lambda a: torch.tensor(np.asarray(a, np.float64))
        self.m, self.v, self.Z = f64(m), f64(v), f64(Z)
        self.th = torch.nn.Parameter(self.m.clone())  # [o (n), d (n)] in RAPM sign convention (d: + = allows more)
        self.b = torch.nn.Parameter(torch.zeros(2))
        self.u = torch.nn.Parameter(torch.zeros(n))
        if g0_fix is None:
            self.g0 = torch.nn.Parameter(torch.tensor(-1.0))
        else:  # fixed population slope: softplus(g0) = g0_fix; player deviations dg still free under their prior
            self.register_buffer('g0', torch.tensor(float(np.log(np.expm1(g0_fix))), dtype=torch.float64))
        self.dg = torch.nn.Parameter(torch.zeros(n))
        F = Z.shape[1]
        self.A = torch.nn.Parameter(0.01 * torch.randn(F, K, generator=torch.Generator().manual_seed(0)))
        # small random start: with z = 0 and W = 0 the fit term has zero gradient (saddle)
        self.dz = torch.nn.Parameter(0.1 * torch.randn(n, K, generator=torch.Generator().manual_seed(1)))
        self.Wo = torch.nn.Parameter(torch.zeros(K, K)); self.Wd = torch.nn.Parameter(torch.zeros(K, K))
        self.M = torch.nn.Parameter(torch.zeros(K, K))
        self.n = n

    def forward(self, O, D, H):
        n = self.n
        yhat = self.b[0] + self.b[1] * H + self.th[O].sum(1) + self.th[n + D].sum(1)
        out = {}
        if self.usage:
            uO = self.u[O]; share = torch.softmax(uO, 1)
            s0 = torch.exp(uO) / (torch.exp(uO) + 4)
            g = torch.nn.functional.softplus(self.g0 + self.dg[O])
            yhat = yhat - 100 * (share * g * (share - s0)).sum(1)
            out['logshare'] = torch.log_softmax(uO, 1)
        if self.fit_:
            z = self.Z @ self.A + self.dz
            zO, zD = z[O], z[D]
            Wo = 0.5 * (self.Wo + self.Wo.T); Wd = 0.5 * (self.Wd + self.Wd.T)
            pair = lambda zz, W: 0.5 * (torch.einsum('bk,kl,bl->b', zz.sum(1), W, zz.sum(1)) - torch.einsum('bik,kl,bil->b', zz, W, zz))
            yhat = yhat + pair(zO, Wo) + pair(zD, Wd) + torch.einsum('bk,kl,bl->b', zO.sum(1), self.M, zD.sum(1))
        out['yhat'] = yhat
        return out

    def penalty(self, s2):
        p = ((self.th - self.m) ** 2 / (2 * self.v)).sum()  # = prior_rapm penalty (s2/v)(theta-m)^2 after dividing by 2 s2
        p = p + (self.u ** 2).sum() / (2 * TAU_U ** 2)
        if self.usage:
            p = p + (self.dg ** 2).sum() / (2 * TAU_G ** 2)
        if self.fit_:
            p = p + (self.dz ** 2).sum() / (2 * TAU_Z ** 2) + self.l2w * ((self.Wo ** 2).sum() + (self.Wd ** 2).sum() + (self.M ** 2).sum()) \
                + 0.01 * self.l2w * (self.A ** 2).sum()
        return p


def fit(R, ex, usage, fit_, iters=300, l2w=L2_W, wmult=None, g0_fix=None):
    pidx = {p: i for i, p in enumerate(ex['pids'])}; n = len(pidx)
    O = torch.tensor([[pidx[p] for p in r[1]] for r in R]); D = torch.tensor([[pidx[p] for p in r[2]] for r in R])
    H = torch.tensor([r[3] for r in R], dtype=torch.float64); y = torch.tensor([r[4] for r in R]); w = torch.tensor([r[5] for r in R])
    C = torch.tensor([[float(c) for c in r[6]] for r in R])
    if wmult is not None:  # e.g. recency decay; also scales the usage counts
        w = w * torch.tensor(np.asarray(wmult, np.float64)); C = C * torch.tensor(np.asarray(wmult, np.float64))[:, None]
    Zs = ex['Z']; Zs = (Zs - Zs.mean(0)) / (Zs.std(0) + 1e-9)
    mdl = Model(n, ex['m'], ex['v'], Zs, usage, fit_, l2w, g0_fix)
    with torch.no_grad():  # noise scale as in hist_prior.prior_rapm
        res = y - mdl(O, D, H)['yhat']; res = res - (w * res).sum() / w.sum()
        s2 = float((w * res ** 2).sum() / len(R))
    opt = torch.optim.LBFGS(mdl.parameters(), lr=1, max_iter=iters, history_size=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        o = mdl(O, D, H)
        loss = (w * (y - o['yhat']) ** 2).sum() / (2 * s2) + mdl.penalty(s2)
        if usage:
            loss = loss - ALPHA * (C * o['logshare']).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mdl, pidx


def predict_games(mdl, pidx, R):
    known = [r for r in R if all(p in pidx for p in r[1] + r[2])]
    O = torch.tensor([[pidx[p] for p in r[1]] for r in known]); D = torch.tensor([[pidx[p] for p in r[2]] for r in known])
    H = torch.tensor([r[3] for r in known], dtype=torch.float64)
    with torch.no_grad():
        yh = mdl(O, D, H)['yhat'].numpy()
    m = defaultdict(float)
    for r, v in zip(known, yh):
        m[r[0]] += (1 if r[3] else -1) * r[5] * v / 100
    return m


def main():
    out = OUT; out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    GU = {g['game_id']: g for g in pickle.load(open(ROOT / 'results/latent/games_use.pkl', 'rb'))}
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints'] and g in GU}
    recs = gw.game_records(G); gid = [r['game_id'] for r in recs]
    nblocks = int(sys.argv[1]) if len(sys.argv) > 1 else len(BLOCKS) - 1
    allp = {}
    for b0, b1 in list(zip(BLOCKS[:-1], BLOCKS[1:]))[:nblocks]:
        f = out / f'block_{b0}.npz'
        P = dict(np.load(f)) if f.exists() else {}
        todo = [a for a in list(ARMS) + list(ARMS_FIX) if f'lat_{a}' not in P] + (['additive'] if 'm_additive' not in P else [])
        if not todo and 'lat_best_load' in P:
            allp[b0] = P; continue
        print(f'block {b0}: arms {todo}', flush=True)
        hist = sorted(g for g, d in G.items() if d['date'] < b0)
        mid = sorted(G[g]['date'] for g in hist)[len(hist) // 2]
        h1 = [g for g in hist if G[g]['date'] < mid]; h2 = [g for g in hist if G[g]['date'] >= mid]
        blk = [g for g, d in G.items() if b0 <= d['date'] < b1]
        base = dict(np.load(gw.OUT / 'rolling_prior' / f'block_{b0}.npz'))
        test_order = [g for g in gid if g in set(blk)]
        P.setdefault('win', np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in test_order]))
        assert (P['win'] == base['win']).all(), 'test order differs from rolling_prior'
        for k in ('rapm_hist_trk', 'anchored_comp_hist_trk'):
            P[k] = base[k]
        ex = {}
        for name, (win, mapg) in {'h1': (h2, h1), 'h2': (h1, h2), 'all': (h1 + h2, h2)}.items():
            ex[name] = rp.tables(G, shots, win, mapg, extra=True)[2]
        from sklearn.linear_model import LogisticRegression
        hy = np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in h1 + h2])
        R_ = {k: stint_rows(GU, v) for k, v in (('h1', h1), ('h2', h2), ('hist', h1 + h2), ('blk', blk))}
        arms = {**{a: (u, f_, None) for a, (u, f_) in ARMS.items()}, **{a: (True, False, g) for a, g in ARMS_FIX.items()}}
        for arm, (usage, fit_, g0f) in arms.items():
            if arm not in todo:
                continue
            best = None
            for lw in (L2_GRID if fit_ else [L2_W]):  # pick the fit penalty on cross-fitted history log loss
                pr_h = {}
                for name, win, tgt in (('h1', 'h2', 'h1'), ('h2', 'h1', 'h2')):
                    mdl, pidx = fit(R_[win], ex[name], usage, fit_, l2w=lw, g0_fix=g0f)
                    pr_h.update(predict_games(mdl, pidx, R_[tgt]))
                hx_ = np.array([pr_h.get(g, 0.0) for g in h1 + h2])
                c_ = LogisticRegression(C=1e4).fit(hx_[:, None], hy)
                l_ = float(-np.mean(np.log(np.clip(c_.predict_proba(hx_[:, None])[np.arange(len(hy)), hy], 1e-9, 1))))
                if best is None or l_ < best[0]:
                    best = (l_, lw, pr_h)
            _, lw, preds = best
            mdl, pidx = fit(R_['hist'], ex['all'], usage, fit_, l2w=lw, g0_fix=g0f)
            preds = dict(preds); preds.update(predict_games(mdl, pidx, R_['blk']))
            if fit_ or usage:
                print(f'  {arm}: l2w {lw} g0 {torch.nn.functional.softplus(mdl.g0).item():.3f} '
                      f'|Wo| {mdl.Wo.detach().abs().max().item():.3f} |Wd| {mdl.Wd.detach().abs().max().item():.3f} '
                      f'|M| {mdl.M.detach().abs().max().item():.3f}', flush=True)
            hx = np.array([preds.get(g, 0.0) for g in h1 + h2])
            cal = LogisticRegression(C=1e4).fit(hx[:, None], hy)
            tx = np.array([preds.get(g, 0.0) for g in test_order])
            P[f'lat_{arm}'] = cal.predict_proba(tx[:, None])[:, 1]
            P[f'm_{arm}'] = tx; P[f'slope_{arm}'] = np.float64(cal.coef_[0, 0])
            print(f'{b0} lat_{arm:9s} auc {roc_auc_score(P["win"], P[f"lat_{arm}"]):.3f}', flush=True)
        # load term on top of the best model: the usage_ms margin minus the additive margin (same priors, same data) is
        # the lineup load adjustment; convert to logit with the usage_ms calibration slope and add to the best logit
        pb = np.clip(P['anchored_comp_hist_trk'], 1e-6, 1 - 1e-6)
        z = np.log(pb / (1 - pb)) + P['slope_usage_ms'] * (P['m_usage_ms'] - P['m_additive'])
        P['lat_best_load'] = 1 / (1 + np.exp(-z))
        print(f'{b0} lat_best_load  auc {roc_auc_score(P["win"], P["lat_best_load"]):.3f} '
              f'(best {roc_auc_score(P["win"], pb):.3f})', flush=True)
        np.savez(f, **P); allp[b0] = P
    names = [k for k in next(iter(allp.values())) if k != 'win' and not k.startswith(('m_', 'slope_'))]
    win = np.concatenate([p['win'] for p in allp.values()])
    pr = {k: np.concatenate([p[k] for p in allp.values()]) for k in names}
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {'n_test': int(len(win)), 'arms': {}}
    for k in names:
        r = res['arms'][k] = dict(auc=float(roc_auc_score(win, pr[k])), log_loss=float(ll(pr[k]).mean()))
        for b in ('lat_additive', 'anchored_comp_hist_trk'):
            d = ll(pr[k]) - ll(pr[b])
            r[f'd_ll_vs_{b}'] = [float(x) for x in np.percentile([d[i].mean() for i in Bs], [2.5, 97.5])]
        print(f"{k:24s} auc {r['auc']:.3f} ll {r['log_loss']:.4f} | Δll vs additive {np.round(r['d_ll_vs_lat_additive'], 4)} "
              f"vs best {np.round(r['d_ll_vs_anchored_comp_hist_trk'], 4)}")
    json.dump(res, open(out / 'latent.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
