"""Anchored graph scorer: the additive on/off rating is the backbone, the graph
network only learns a fit correction on top of it.

  side value = a * sum_i m_i r_i  +  rho(sum_i m_i phi(GCN(x_i, r_i)))
  P(home win) = sigmoid(value_home - value_away + home)

r_i is the player's ridge stint rating (pts/48) fitted WITHOUT the game's season
(training seasons: cross-fit on the other training seasons; test season: fit on
earlier seasons only). The GCN's output layer starts at zero and is shrunk with
`--shrink`, so with no evidence of fit the model is exactly the ratings model.
Deployments are the pre-game prior graphs (as main_pre).

  python v3/anchored.py --test-season 2023 --tag holdout     # validity: AUC vs ratings only
  python v3/anchored.py --tag main                           # all seasons, for the Paris search
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from core import DEV, RelGraphScorer
from data_registry import load_sport
from dataset import side_players
from train import shuffle_graph, target
from values import ValueModel, value_of

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "v3" / "models"
CACHE = MODELS / "anchored"
RSCALE = 5.0  # rating column is r / RSCALE so it sits on the feature scale


class AnchoredScorer(nn.Module):
    def __init__(self, f_in, hidden=128, layers=4, dropout=0.1, use_gcn=True, hide_rating=False):
        super().__init__()
        self.hide_rating = hide_rating
        self.gcn = RelGraphScorer(f_in, hidden, layers, dropout) if use_gcn else None
        if use_gcn:  # fit term starts at exactly zero
            nn.init.zeros_(self.gcn.rho[-1].weight)
            nn.init.zeros_(self.gcn.rho[-1].bias)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.home_adv = nn.Parameter(torch.zeros(1))
        self.last_fit = None

    def side_values(self, X, A, node_mask, R):
        minutes = torch.diagonal(A, dim1=1, dim2=2) * node_mask
        base = minutes * X[..., -1] * RSCALE
        bh, ba = base[:, :R].sum(1), base[:, R:].sum(1)
        if self.gcn is None:
            self.last_fit = None
            return self.a * bh, self.a * ba
        Xg = torch.cat([X[..., :-1], torch.zeros_like(X[..., -1:])], -1) if self.hide_rating else X
        gh, ga = self.gcn.side_values(Xg, A, node_mask, R)
        self.last_fit = gh - ga
        return self.a * bh + gh, self.a * ba + ga

    def forward(self, X, A, node_mask, R, home=True):
        vh, va = self.side_values(X, A, node_mask, R)
        return torch.sigmoid(vh - va + (self.home_adv if home else 0.0))


def rating_tables(seasons, cap, cross_fit):
    """{season: {player: rating}}. cross_fit: ratings for season s fitted on
    seasons <= cap except s; otherwise one fit on seasons <= cap for all."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in seasons:
        excl = s if cross_fit else None
        f = CACHE / f"ratings_cap{cap}_ex{excl}.pkl"
        if f.exists():
            out[s] = pickle.load(open(f, "rb"))
            continue
        print(f"fitting ratings cap {cap} exclude {excl}", flush=True)
        vm = ValueModel("nba", cap, exclude_season=excl, lineup_seasons=[])
        b, r, repl = vm.fit()
        tab = {"r": r, "b": b, "repl": repl}
        pickle.dump(tab, open(f, "wb"))
        out[s] = tab
    return out


def node_ratings(S, games, tabs, R):
    G = len(games)
    Rt = np.zeros((G, 2 * R), np.float32)
    for k, g in enumerate(games):
        t = tabs.get(g["season"])
        if t is None:
            continue
        players = side_players(g, g["home"], R) + side_players(g, g["away"], R)
        for i, p in enumerate(players):
            if p is not None:
                # players never rated count as average (r = 0), as in the value scorer
                Rt[k, i] = value_of(S, p, g["season"], t["b"], t["r"], t["repl"]) if p in t["r"] else 0.0
    return torch.tensor(Rt)


def run_epoch(model, X, A, M, y, R, scale, idx, opt=None, bs=128, shrink=0.0):
    train = opt is not None
    model.train(train)
    perm = idx[torch.randperm(len(idx))] if train else idx
    tot = 0.0
    for s in range(0, len(perm), bs):
        b = perm[s:s + bs]
        xb, ab, mb, yb = X[b].to(DEV), A[b].to(DEV), M[b].to(DEV), y[b].to(DEV)
        with torch.set_grad_enabled(train):
            p = model(xb, ab, mb, R, home=True)
            loss = F.mse_loss(p, target(yb, scale))
            if model.last_fit is not None and shrink > 0:
                loss = loss + shrink * (model.last_fit ** 2).mean()
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * len(b)
    return tot / len(perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-season", type=int, default=None)
    ap.add_argument("--lag", type=int, default=1, help="feature season lag (1 = strictly pre-game)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shrink", type=float, default=0.01)
    ap.add_argument("--arms", default="ratings,anchored,anchored_shuffled")
    ap.add_argument("--hide-rating", action="store_true", help="GCN does not see the rating column")
    ap.add_argument("--drop-feats", default="", help="feature columns hidden from the GCN (status/volume), e.g. mpg,gp_share,usage,pts36")
    args = ap.parse_args()

    S = load_sport("nba")
    games = S.games()
    X, A, M, y, meta, cov = S.tensors(lag=args.lag)
    from prior_graphs import prior_graph_tensors
    A = prior_graph_tensors(S, X, A, M, meta, lag=args.lag)
    R, scale = S.R, S.scale
    seasons = np.array([m["season"] for m in meta])
    playoff = np.array([m["playoff"] for m in meta])
    all_seasons = sorted(set(seasons.tolist()))
    if args.test_season is not None:
        train_s = [s for s in all_seasons if s < args.test_season]
        tabs = rating_tables(train_s, args.test_season - 1, cross_fit=True)
        tabs.update(rating_tables([args.test_season], args.test_season - 1, cross_fit=False))
    else:
        train_s = all_seasons
        tabs = rating_tables(train_s, max(all_seasons), cross_fit=True)
    Rt = node_ratings(S, games, tabs, R)
    drop = [c for c in args.drop_feats.split(",") if c]
    if drop:  # the rating carries status; the GCN sees style only
        X = X.clone()
        X[..., [S.feature_cols.index(c) for c in drop]] = 0.0
    Xr = torch.cat([X, (Rt / RSCALE).unsqueeze(-1)], -1)
    print(f"games {len(meta)}, feature coverage {cov:.3f}, rated nodes {(Rt != 0).float().mean():.3f}", flush=True)

    pool = ~playoff & np.isin(seasons, train_s)
    pool_idx = np.where(pool)[0]
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(pool_idx), args.folds)
    test_idx = (np.where(~playoff & (seasons == args.test_season))[0] if args.test_season is not None else None)

    res = {"args": vars(args), "arms": {}}
    for arm in args.arms.split(","):
        A_use = shuffle_graph(A, M, R) if arm.endswith("shuffled") else A
        out = MODELS / "nba" / f"anchored_{args.tag}_{arm}"
        out.mkdir(parents=True, exist_ok=True)
        preds, logs = [], []
        for k in range(args.folds):
            torch.manual_seed(100 + k)
            val = torch.tensor(folds[k])
            tr = torch.tensor(np.concatenate([folds[j] for j in range(args.folds) if j != k]))
            model = AnchoredScorer(Xr.shape[-1], use_gcn=(arm != "ratings"), hide_rating=args.hide_rating).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr if arm != "ratings" else 3e-2,
                                    weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
            best, best_state = 1e9, None
            for ep in range(args.epochs):
                run_epoch(model, Xr, A_use, M, y, R, scale, tr, opt, shrink=args.shrink)
                vl = run_epoch(model, Xr, A_use, M, y, R, scale, val)
                sched.step()
                if vl < best:
                    best, best_state = vl, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            torch.save(best_state, out / f"fold{k}.pt")
            logs.append({"fold": k, "best_val_mse": best})
            if test_idx is not None:
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    ti = torch.tensor(test_idx)
                    preds.append(model(Xr[ti].to(DEV), A_use[ti].to(DEV), M[ti].to(DEV), R, home=True).cpu().numpy())
            print(f"{arm} fold {k}: val mse {best:.5f}", flush=True)
        cfg = {"n_feat": Xr.shape[-1], "R": R, "scale": scale, "folds": args.folds, "arm": arm, "drop_feats": drop, "hide_rating": args.hide_rating,
               "lag": args.lag, "folds_log": logs}
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        entry = {"val_mse": float(np.mean([l["best_val_mse"] for l in logs]))}
        if test_idx is not None:
            p = np.mean(preds, 0)
            win = (y[test_idx].numpy() > 0).astype(int)
            entry.update(auc=float(roc_auc_score(win, p)), logloss=float(log_loss(win, np.clip(p, 1e-6, 1 - 1e-6))),
                         n=int(len(test_idx)))
            np.save(out / "test_pred.npy", p)
            np.save(out / "test_win.npy", win)
        res["arms"][arm] = entry
        print(arm, entry, flush=True)
    (ROOT / "v3" / "results" / f"anchored_{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
