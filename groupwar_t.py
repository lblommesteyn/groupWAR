"""GroupWAR-T: the v3 anchored graph scorer with tracking-era inputs.

Scorer, shuffle control and rolling deployments follow groupWAR v3
(v3/core.py RelGraphScorer, v3/anchored.py AnchoredScorer, v3/prior_graphs.py).
Inputs here come from the 2025-26 tracking archive:
  backbone rating : RAPM (net per 100), optionally with the tracking-description prior
  node features   : box (v3 definitions) / tracking v2 / both
  deployments     : rolling, from each team's earlier games (pre-game only)
Cross-fit: Jan-Feb games use March descriptions+ratings and vice versa; test games
(Apr 1+) use Jan-Mar. Time is measured in possessions.
"""
import argparse, json, pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss

import embed_eval as ee
import metrics2 as m2
from masked_paths import ROOT

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = ROOT / 'results/groupwar_t'
R, SCALE, RSCALE = 12, 10.0, 5.0


# ------------------------------------------------------------------ model (from v3)
class RelGraphScorer(nn.Module):
    def __init__(self, f_in, hidden=128, layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(f_in + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.w_pos = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.w_neg = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def side_values(self, X, A, node_mask):
        minutes = torch.diagonal(A, dim1=1, dim2=2).unsqueeze(-1)
        H = self.embed(torch.cat([X, minutes], -1))
        off = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        A_pos, A_neg = off.clamp(min=0), (-off).clamp(min=0)
        for ws, wp, wn, ln in zip(self.w_self, self.w_pos, self.w_neg, self.norms):
            H = H + self.drop(F.relu(ln(ws(H) + wp(A_pos @ H) + wn(A_neg @ H))))
        P = self.phi(H) * (minutes * node_mask.unsqueeze(-1))
        return self.rho(P[:, :R].sum(1)).squeeze(-1), self.rho(P[:, R:].sum(1)).squeeze(-1)


class Anchored(nn.Module):
    """value = a * sum_i m_i r_i + GCN fit term (starts at zero). use_rating=False -> plain v1-style GCN."""
    def __init__(self, f_in, use_gcn=True, use_rating=True):
        super().__init__()
        self.use_rating = use_rating
        self.gcn = RelGraphScorer(f_in) if use_gcn else None
        if use_gcn and use_rating:
            nn.init.zeros_(self.gcn.rho[-1].weight); nn.init.zeros_(self.gcn.rho[-1].bias)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.home = nn.Parameter(torch.zeros(1))
        self.last_fit = None

    def forward(self, X, A, M, rating, home=True):
        m = torch.diagonal(A, dim1=1, dim2=2) * M
        vh = va = 0.0
        if self.use_rating:
            base = m * rating * RSCALE
            vh, va = self.a * base[:, :R].sum(1), self.a * base[:, R:].sum(1)
        self.last_fit = None
        if self.gcn is not None:
            gh, ga = self.gcn.side_values(X, A, M)
            self.last_fit = gh - ga
            vh, va = vh + gh, va + ga
        return torch.sigmoid(vh - va + (self.home if home else 0.0))


def shuffle_graph(A):
    d = torch.diagonal(A, dim1=1, dim2=2)
    out = d.unsqueeze(-1) * d.unsqueeze(1)
    out[:, :R, R:] *= -1; out[:, R:, :R] *= -1
    return out - torch.diag_embed(torch.diagonal(out, dim1=1, dim2=2)) + torch.diag_embed(d)


# ------------------------------------------------------------------ data
def game_records(G):
    recs = []
    for g, d in sorted(G.items(), key=lambda kv: (kv[1]['date'], kv[0])):
        secs, shared, team_of, L = defaultdict(float), defaultdict(float), {}, 0.0
        for st in d['stints']:
            w = st['poss']
            if w <= 0:
                continue
            L += w
            for side, team in zip(st['lineup'], (d['home'], d['away'])):
                for i, p in enumerate(side):
                    secs[p] += w; team_of[p] = team
                    for q in side[i + 1:]:
                        shared[(min(p, q), max(p, q))] += w
        if L > 0:
            recs.append(dict(game_id=g, date=d['date'], home=d['home'], away=d['away'], length=L,
                             margin=d['final'][0] - d['final'][1], secs=dict(secs), shared=dict(shared),
                             team_of=team_of))
    return recs


def side_players(g, team):
    ps = sorted([p for p, t in g['team_of'].items() if t == team], key=lambda p: -g['secs'][p])[:R]
    return ps + [None] * (R - len(ps))


def rolling_A(recs, decay=0.9, min_hist=3):
    """Pre-game deployment: decayed average of each team's earlier shared-time matrices."""
    st = {}
    A = np.zeros((len(recs), 2 * R, 2 * R), np.float32); ok = np.zeros(len(recs), bool)
    for k, g in enumerate(recs):
        sides, enough = [], True
        for team in (g['home'], g['away']):
            s = st.get(team); W = np.zeros((R, R), np.float32)
            players = side_players(g, team)
            if s is None or s['n'] < min_hist:
                enough = False
            if s is not None and s['L'] > 0:
                for i, p in enumerate(players):
                    if p is None:
                        continue
                    W[i, i] = s['secs'].get(p, 0) / s['L']
                    for j in range(i + 1, R):
                        q = players[j]
                        if q is not None:
                            W[i, j] = W[j, i] = s['shared'].get((min(p, q), max(p, q)), 0) / s['L']
                tot = np.trace(W)
                if tot > 0:
                    W *= 5.0 / tot
            sides.append(W)
        mh, ma = np.diag(sides[0]), np.diag(sides[1])
        cross = -np.outer(mh, ma)  # independence: P(i and j on court together)
        A[k] = np.block([[sides[0], cross], [cross.T, sides[1]]])  # diag = fraction of game on court
        ok[k] = enough
        for team in (g['home'], g['away']):
            s = st.setdefault(team, dict(secs=defaultdict(float), shared=defaultdict(float), L=0.0, n=0))
            for dct in (s['secs'], s['shared']):
                for key in dct:
                    dct[key] *= decay
            s['L'] *= decay
            for p, v in g['secs'].items():
                if g['team_of'][p] == team:
                    s['secs'][p] += v
            for (p, q), v in g['shared'].items():
                if g['team_of'].get(p) == team:
                    s['shared'][(p, q)] += v
            s['L'] += g['length']; s['n'] += 1
    return A, ok


def ratings_and_features(G, shots, win_games, map_games, cols_list):
    """Ratings (plain RAPM and tracking-prior RAPM, net per 100) + standardized features from win_games.
    The description->impact map for the prior is learned on map_games with win_games descriptions."""
    R_win = ee.rows(G, win_games); R_map = ee.rows(G, map_games)
    pids = sorted({p for Rr in (R_win, R_map) for r in Rr for p in r[1] + r[2]} |
                  {p for d in G.values() for p in d['box']})
    pidx = {p: i for i, p in enumerate(pids)}; n = len(pids)
    desc = m2.describe2(G, win_games, shots)
    _, th = ee.rapm_fit(R_win, pidx)
    rapm = {p: th[pidx[p]] - th[n + pidx[p]] for p in pids}
    cols = ee.BOX + ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF
    Z = ee.matrix(desc, cols, pids)
    lam = ee.cv_lambda(R_map, Z, pidx)
    mdl = ee.Lin().fit(R_map, Z, pidx, lam)
    F_ = len(cols); wo, wd = mdl.beta[2:2 + F_], mdl.beta[2 + F_:]
    prior_val = Z @ wo - Z @ wd
    prior_rows = mdl.predict(R_win, Z, pidx)
    _, th2 = ee.rapm_fit(R_win, pidx, prior_rows)
    rapm_t = {p: prior_val[pidx[p]] + th2[pidx[p]] - th2[n + pidx[p]] for p in pids}
    feats = {name: dict(zip(pids, ee.matrix(desc, c, pids))) for name, c in cols_list.items()}
    return rapm, rapm_t, feats


def tensors(recs, tables):
    """tables: per game -> (rating dict, feature dicts)."""
    G_ = len(recs)
    M = np.zeros((G_, 2 * R), np.float32)
    out_r = {k: np.zeros((G_, 2 * R), np.float32) for k in ('rapm', 'rapm_t')}
    out_x = defaultdict(lambda: None)
    for k, g in enumerate(recs):
        rapm, rapm_t, feats = tables[g['game_id']]
        players = side_players(g, g['home']) + side_players(g, g['away'])
        for i, p in enumerate(players):
            if p is None:
                continue
            M[k, i] = 1
            out_r['rapm'][k, i] = rapm.get(p, 0.0) / RSCALE
            out_r['rapm_t'][k, i] = rapm_t.get(p, 0.0) / RSCALE
            for name, fd in feats.items():
                if out_x[name] is None:
                    out_x[name] = np.zeros((G_, 2 * R, len(next(iter(fd.values())))), np.float32)
                v = fd.get(p)
                if v is not None:
                    out_x[name][k, i] = v
    return M, out_r, dict(out_x)


# ------------------------------------------------------------------ training
def train_arm(X, A, M, Rt, y, tr_idx, te_idx, use_gcn, use_rating, folds=6, epochs=40, lr=1e-3, shrink=0.01, seed=0):
    X, A, M, Rt = (torch.tensor(v) for v in (X, A, M, Rt))
    yt = torch.sigmoid(torch.tensor(y, dtype=torch.float32) / SCALE)
    rng = np.random.default_rng(seed); fs = np.array_split(rng.permutation(tr_idx), folds)
    preds = []
    for k in range(folds):
        torch.manual_seed(100 * seed + k)
        val = torch.tensor(fs[k]); tr = torch.tensor(np.concatenate([fs[j] for j in range(folds) if j != k]))
        model = Anchored(X.shape[-1], use_gcn, use_rating).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr if use_gcn else 3e-2, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, state = 1e9, None
        for ep in range(epochs):
            model.train()
            perm = tr[torch.randperm(len(tr))]
            for s in range(0, len(perm), 64):
                b = perm[s:s + 64]
                p = model(X[b].to(DEV), A[b].to(DEV), M[b].to(DEV), Rt[b].to(DEV))
                loss = F.mse_loss(p, yt[b].to(DEV))
                if model.last_fit is not None and use_rating and shrink > 0:
                    loss = loss + shrink * (model.last_fit ** 2).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = F.mse_loss(model(X[val].to(DEV), A[val].to(DEV), M[val].to(DEV), Rt[val].to(DEV)), yt[val].to(DEV)).item()
            if vl < best:
                best, state = vl, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            ti = torch.tensor(te_idx)
            preds.append(model(X[ti].to(DEV), A[ti].to(DEV), M[ti].to(DEV), Rt[ti].to(DEV)).cpu().numpy())
    return np.mean(preds, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    gA = sorted(g for g, d in G.items() if d['date'] < ee.A_END)
    gB = sorted(g for g, d in G.items() if ee.A_END <= d['date'] < ee.B_END)
    gT = sorted(g for g, d in G.items() if d['date'] >= ee.B_END)
    cols_list = {'box': ee.BOX, 'trk': ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF,
                 'box_trk': ee.BOX + ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF}
    cache = OUT / 'tables.pkl'
    if cache.exists():
        tabA, tabB, tabT = pickle.load(open(cache, 'rb'))
    else:
        print('building cross-fit ratings/features ...', flush=True)
        tabB = ratings_and_features(G, shots, gB, gA, cols_list)   # used for A games
        tabA = ratings_and_features(G, shots, gA, gB, cols_list)   # used for B games
        tabT = ratings_and_features(G, shots, gA + gB, gB, cols_list)
        pickle.dump((tabA, tabB, tabT), open(cache, 'wb'))
    recs = game_records(G)
    A, ok = rolling_A(recs)
    which = {g: tabB for g in gA} | {g: tabA for g in gB} | {g: tabT for g in gT}
    M, Rts, Xs = tensors(recs, {r['game_id']: which[r['game_id']] for r in recs})
    y = np.array([r['margin'] for r in recs], np.float32)
    gid = [r['game_id'] for r in recs]
    setT = set(gT)
    tr_idx = np.array([k for k, g in enumerate(gid) if g not in setT and ok[k]])
    te_idx = np.array([k for k, g in enumerate(gid) if g in setT])
    print(f'train {len(tr_idx)} games (rolling history ok), test {len(te_idx)}', flush=True)
    Ash = shuffle_graph(torch.tensor(A)).numpy()
    zeros = np.zeros_like(Xs['box'][..., :1])
    arms = {
        'ratings_rapm': (zeros, A, 'rapm', False, True),
        'ratings_rapm_trkprior': (zeros, A, 'rapm_t', False, True),
        'gcn_only_box (v1-style)': (Xs['box'], A, 'rapm', True, False),
        'gcn_only_trk': (Xs['trk'], A, 'rapm', True, False),
        'anchored_box': (Xs['box'], A, 'rapm', True, True),
        'anchored_trk': (Xs['trk'], A, 'rapm_t', True, True),
        'anchored_box_trk': (Xs['box_trk'], A, 'rapm_t', True, True),
        'anchored_box_trk_SHUFFLED': (Xs['box_trk'], Ash, 'rapm_t', True, True),
    }
    win = (y[te_idx] > 0).astype(int)
    preds, res = {}, {}
    for name, (X, A_, rk, gcn, rat) in arms.items():
        p = np.mean([train_arm(X, A_, M, Rts[rk], y, tr_idx, te_idx, gcn, rat, epochs=args.epochs, seed=s)
                     for s in range(args.seeds)], 0)
        preds[name] = p
        res[name] = dict(auc=float(roc_auc_score(win, p)), log_loss=float(log_loss(win, np.clip(p, 1e-6, 1 - 1e-6))))
        print(f"{name:28s} auc {res[name]['auc']:.3f}  ll {res[name]['log_loss']:.4f}", flush=True)
    rng = np.random.default_rng(0); Bs = rng.integers(0, len(win), (2000, len(win)))
    base = preds['ratings_rapm']
    ll = lambda p: -(win * np.log(np.clip(p, 1e-6, 1)) + (1 - win) * np.log(np.clip(1 - p, 1e-6, 1)))
    for name, p in preds.items():
        d = ll(p) - ll(base)
        res[name]['d_ll_vs_rapm_ci'] = [float(v) for v in np.percentile([d[b].mean() for b in Bs], [2.5, 97.5])]
        res[name]['d_auc_vs_rapm_ci'] = [float(v) for v in np.percentile(
            [roc_auc_score(win[b], p[b]) - roc_auc_score(win[b], base[b]) for b in Bs[:500]], [2.5, 97.5])]
    json.dump(dict(n_train=len(tr_idx), n_test=len(te_idx), arms=res), open(OUT / 'validity.json', 'w'), indent=1)
    np.savez(OUT / 'test_preds.npz', win=win, **{k.replace(' ', '_'): v for k, v in preds.items()})
    for name, r in res.items():
        print(f"{name:28s} auc {r['auc']:.3f} ll {r['log_loss']:.4f}  Δll CI {np.round(r['d_ll_vs_rapm_ci'], 4)}  ΔAUC CI {np.round(r['d_auc_vs_rapm_ci'], 3)}")


if __name__ == '__main__':
    main()
