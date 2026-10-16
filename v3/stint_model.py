"""Stint-level player-impact model (prototype).

Each stint has exactly the units on the floor and its own point margin, so the
lineup-to-outcome mapping is learned from ~135k NBA stints instead of ~4k game
results, and in-game deployment never enters the target. A player's value is a
box-score prior u(x) plus a shrunken per-player residual (the RAPM idea with a
learned prior); an optional interaction term rho(sum phi(x)) lets unit value
depend on the mix of player types.

Validation is game-level on a held-out season with the rolling pre-game
deployment (the same games and AUC as eval_scorer E1):
  game strength = sum_i share_i * value_i  (home minus away).

  python v3/stint_model.py --sport nba --test-season 2023 --lag 1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from data_registry import load_sport

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v3" / "data"


class StintModel(nn.Module):
    def __init__(self, n_feat, n_players, hidden=64, interact=True):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.emb = nn.Embedding(n_players + 1, 1)  # last row = unknown player, fixed at 0
        nn.init.zeros_(self.emb.weight)
        self.interact = interact
        if interact:
            self.phi = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, 16))
            self.rho = nn.Sequential(nn.Linear(16, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.home = nn.Parameter(torch.zeros(()))

    def player_value(self, X, I):
        return self.u(X).squeeze(-1) + self.emb(I).squeeze(-1)

    def unit_value(self, X, I):
        """X (B,5,F), I (B,5) -> points per 100 seconds for the unit."""
        v = self.player_value(X, I).sum(-1)
        if self.interact:
            v = v + self.rho(self.phi(X).sum(1)).squeeze(-1)
        return v

    def forward(self, Xh, Ih, Xa, Ia):
        return self.unit_value(Xh, Ih) - self.unit_value(Xa, Ia) + self.home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--test-season", type=int, default=2023)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--id-l2", type=float, default=3e-3, help="shrinkage on per-player residuals")
    ap.add_argument("--no-ids", action="store_true")
    ap.add_argument("--no-interact", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--id-lr", type=float, default=2e-2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    S = load_sport(args.sport)
    games = S.games()
    gseason = {g["game_id"]: g["season"] for g in games}
    gplayoff = {g["game_id"]: g["playoff"] for g in games}
    st = pd.read_pickle(DATA / args.sport / "stints.pkl")
    st = st[st["game_id"].isin(gseason)].copy()
    st["season"] = st["game_id"].map(gseason)
    st["playoff"] = st["game_id"].map(gplayoff)
    st = st[st["secs"] >= 10]

    F = len(S.feature_cols)
    pids = sorted({p for u in st["home5"] for p in u} | {p for u in st["away5"] for p in u})
    pidx = {p: k for k, p in enumerate(pids)}
    unk = len(pids)
    zero = np.zeros(F, np.float32)
    repl_cache = {}

    def fv(p, season):
        v = S.vec(p, season - args.lag)
        if v is None:  # no feature season: replacement-level (mean of low-minute players)
            if season not in repl_cache:
                cur = S.ps[(S.ps["season"] == season - args.lag)]
                lo, hi = cur["minutes_total"].quantile([0.10, 0.30])
                rp = cur[(cur["minutes_total"] >= lo) & (cur["minutes_total"] <= hi)]
                repl_cache[season] = ((rp[S.feature_cols] - S.mu) / S.sd).fillna(0).mean().to_numpy(np.float32)
            return repl_cache[season]
        return v

    def unit_arrays(df, col):
        X = np.stack([[fv(p, s) for p in u] for u, s in zip(df[col], df["season"])]).astype(np.float32)
        I = np.array([[pidx[p] for p in u] for u in df[col]])
        return torch.tensor(X), torch.tensor(I)

    tr = st[(st["season"] < args.test_season) & ~st["playoff"]]
    Xh, Ih = unit_arrays(tr, "home5")
    Xa, Ia = unit_arrays(tr, "away5")
    secs = torch.tensor(tr["secs"].to_numpy(np.float32))
    margin = torch.tensor((tr["pts_home"] - tr["pts_away"]).to_numpy(np.float32))
    if args.no_ids:
        Ih[:] = unk
        Ia[:] = unk
    print(f"{args.sport}: train stints {len(tr)}, players {len(pids)}, lag {args.lag}", flush=True)

    model = StintModel(F, len(pids), interact=not args.no_interact)
    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if n != "emb.weight"],
                            lr=args.lr, weight_decay=1e-3)
    opt_id = torch.optim.Adam([model.emb.weight], lr=args.id_lr)
    n = len(tr)
    # rate is points per 100 seconds; prediction for a stint = rate * secs / 100
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, 4096):
            b = perm[s:s + 4096]
            rate = model(Xh[b], Ih[b], Xa[b], Ia[b])
            pred = rate * secs[b] / 100
            # Gaussian likelihood with variance proportional to stint length
            loss = ((margin[b] - pred) ** 2 / secs[b]).mean()
            if not args.no_ids:
                loss = loss + args.id_l2 * (model.emb.weight[:-1] ** 2).sum() / n * 4096
            opt.zero_grad()
            opt_id.zero_grad()
            loss.backward()
            opt.step()
            if not args.no_ids:
                opt_id.step()
            with torch.no_grad():
                model.emb.weight[-1] = 0
            tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  epoch {ep + 1}: loss {tot / n:.4f}", flush=True)

    # ── game-level validation with rolling pre-game deployment ──
    from dataset import side_players
    from prior_graphs import rolling_graph_tensors
    A, ok = rolling_graph_tensors(S)
    R = S.R
    model.eval()
    strength, gs, gp, gm = [], [], [], []
    with torch.no_grad():
        for k, g in enumerate(games):
            if not ok[k]:
                continue
            pos = S.hockey_pos(g) if S.hockey_pos else None
            vals = []
            for side, team in enumerate((g["home"], g["away"])):
                pl = side_players(g, team, R, pos)
                share = torch.diagonal(A[k])[side * R:(side + 1) * R]
                X = torch.stack([torch.tensor(fv(p, g["season"])) if p is not None else torch.tensor(zero)
                                 for p in pl])
                I = torch.tensor([pidx.get(p, unk) if (p is not None and not args.no_ids) else unk for p in pl])
                v = model.player_value(X, I)
                vals.append(float((share * v).sum()))
            strength.append(vals[0] - vals[1])
            gs.append(g["season"]); gp.append(g["playoff"]); gm.append(g["margin"])
    strength, gs, gp, gm = map(np.array, (strength, gs, gp, gm))
    win = (gm > 0).astype(int)
    trg = (gs < args.test_season) & ~gp & (gm != 0)
    lr = LogisticRegression().fit(strength[trg, None], win[trg])
    res = {"args": vars(args)}
    for name, m in (("test_regular", (gs == args.test_season) & ~gp),
                    ("test_playoffs", (gs == args.test_season) & gp),
                    ("train_games", trg)):
        if m.sum() == 0:
            continue
        p = lr.predict_proba(strength[m, None])[:, 1]
        res[name] = {"n": int(m.sum()), "auc": roc_auc_score(win[m], p), "logloss": log_loss(win[m], p),
                     "acc": float(((p > 0.5) == win[m]).mean())}
        print(f"  {name:14s} n={m.sum():5d} auc {res[name]['auc']:.3f} logloss {res[name]['logloss']:.4f} "
              f"acc {res[name]['acc']:.3f}", flush=True)
    if args.out:
        out = ROOT / "v3" / "results" / f"stint_{args.sport}_{args.out}.json"
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
