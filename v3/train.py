"""Train an ensemble of graph scorers for one sport.

  python v3/train.py --sport nba --tag main                   # all training seasons
  python v3/train.py --sport nba --tag holdout --test-season 2023   # E1 validity split
Options --layers/--hidden/--lag/--shuffle-graph are used for sensitivity runs.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import DEV, RelGraphScorer
from data_registry import load_sport

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "v3" / "models"


def target(margin, scale):
    return torch.sigmoid(margin / scale)


def run_epoch(model, X, A, M, y, R, scale, idx, opt=None, bs=128):
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
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * len(b)
    return tot / len(perm)


def shuffle_graph(A, M, R):
    """Control: replace each game's time matrix with a matrix whose off-diagonal
    structure is uninformative (each pair gets the product of minutes)."""
    d = torch.diagonal(A, dim1=1, dim2=2)
    out = d.unsqueeze(-1) * d.unsqueeze(1)
    out[:, :R, R:] *= -1
    out[:, R:, :R] *= -1
    out = out - torch.diag_embed(torch.diagonal(out, dim1=1, dim2=2)) + torch.diag_embed(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-season", type=int, default=None)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shuffle-graph", action="store_true")
    ap.add_argument("--prior-deploy", action="store_true",
                    help="train on pre-game prior deployments instead of observed time")
    ap.add_argument("--include-playoffs", action="store_true")
    args = ap.parse_args()

    S = load_sport(args.sport)
    X, A, M, y, meta, coverage = S.tensors(lag=args.lag)
    R, scale = S.R, S.scale
    if args.prior_deploy:
        from prior_graphs import prior_graph_tensors
        A = prior_graph_tensors(S, X, A, M, meta, lag=args.lag)
    if args.shuffle_graph:
        A = shuffle_graph(A, M, R)
    seasons = np.array([m["season"] for m in meta])
    playoff = np.array([m["playoff"] for m in meta])
    pool = np.ones(len(meta), dtype=bool)
    if not args.include_playoffs:
        pool &= ~playoff
    if args.test_season is not None:
        pool &= seasons < args.test_season
    pool_idx = np.where(pool)[0]
    print(f"{args.sport}: {len(meta)} games, train pool {len(pool_idx)}, feature coverage {coverage:.3f}",
          flush=True)

    out = MODELS / args.sport / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(pool_idx), args.folds)
    log = []
    for k in range(args.folds):
        torch.manual_seed(100 + k)
        val = torch.tensor(folds[k])
        tr = torch.tensor(np.concatenate([folds[j] for j in range(args.folds) if j != k]))
        model = RelGraphScorer(X.shape[-1], args.hidden, args.layers).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        best, best_state = 1e9, None
        for ep in range(args.epochs):
            tl = run_epoch(model, X, A, M, y, R, scale, tr, opt)
            vl = run_epoch(model, X, A, M, y, R, scale, val)
            sched.step()
            if vl < best:
                best, best_state = vl, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        torch.save(best_state, out / f"fold{k}.pt")
        log.append({"fold": k, "best_val_mse": best, "final_train_mse": tl})
        print(f"fold {k}: best val mse {best:.5f}", flush=True)
    cfg = vars(args) | {"n_feat": X.shape[-1], "R": R, "scale": scale, "folds_log": log,
                        "coverage": coverage}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
