"""E1 (scorer validity) and E9 (does the graph matter) on a held-out season.

  python v3/eval_scorer.py --sport nba --test-season 2023 --tags holdout,holdout_shuf,holdout_lag1

Each tag must be an ensemble trained with --test-season equal to the one given.
Baselines are fit on the same training games:
  home      : constant home win rate
  rapm_sum  : minute-weighted sum of RAPM (fit on training seasons), logistic
  feat_lin  : minute-weighted mean feature vector per side, logistic on the difference
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from core import DEV
from data_registry import load_sport
from rapm import fit as fit_rapm
from rosters import load_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "results"
OUT.mkdir(exist_ok=True, parents=True)


def metrics(p, y_win, margin, scale):
    keep = margin != 0
    p, yw = np.clip(p[keep], 1e-4, 1 - 1e-4), y_win[keep]
    return {"n": int(keep.sum()), "log_loss": log_loss(yw, p), "brier": brier_score_loss(yw, p),
            "auc": roc_auc_score(yw, p), "acc": float(((p > 0.5) == yw).mean()),
            "mse_sigma": float(np.mean((p - 1 / (1 + np.exp(-margin[keep] / scale))) ** 2))}


def predict(ens, X, A, M, R, idx, bs=256):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            out.append(ens(X[b].to(DEV), A[b].to(DEV), M[b].to(DEV), R, home=True).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True)
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()
    S = load_sport(args.sport)
    R, scale = S.R, S.scale
    res = {}
    cache = {}
    from prior_graphs import prior_graph_tensors
    prior_A = {}
    for lag in (0, 1):
        cache[lag] = S.tensors(lag=lag)
        Xl, Al, Ml, _, ml, _ = cache[lag]
        prior_A[lag] = prior_graph_tensors(S, Xl, Al, Ml, ml, lag=lag)
    X, A, M, y, meta, cov = cache[0]
    season = np.array([m["season"] for m in meta])
    playoff = np.array([m["playoff"] for m in meta])
    margin = y.numpy()
    win = (margin > 0).astype(int)
    tr = np.where((season < args.test_season) & ~playoff)[0]
    subsets = {"test_regular": np.where((season == args.test_season) & ~playoff)[0],
               "test_playoffs": np.where((season == args.test_season) & playoff)[0]}

    # RAPM baseline (training seasons only)
    seasons_tr = sorted(set(season[tr].tolist()))
    rp, alpha, _ = fit_rapm(args.sport, seasons_tr)
    rmap = dict(zip(rp["player_id"], rp["rapm"]))
    games = S.games()
    from dataset import side_players
    strength = np.zeros(len(games))
    for k, g in enumerate(games):
        pos = S.hockey_pos(g) if S.hockey_pos else None
        hs = side_players(g, g["home"], R, pos)
        as_ = side_players(g, g["away"], R, pos)
        sh = sum(g["secs"].get(p, 0) / g["length"] * rmap.get(p, 0) for p in hs if p)
        sa = sum(g["secs"].get(p, 0) / g["length"] * rmap.get(p, 0) for p in as_ if p)
        strength[k] = sh - sa
    # high-talent subset: both sides in the top quartile of minute-weighted RAPM level
    lvl_h = np.zeros(len(games)); lvl_a = np.zeros(len(games))
    for k, g in enumerate(games):
        pos = S.hockey_pos(g) if S.hockey_pos else None
        lvl_h[k] = sum(g["secs"].get(p, 0) / g["length"] * rmap.get(p, 0)
                       for p in side_players(g, g["home"], R, pos) if p)
        lvl_a[k] = sum(g["secs"].get(p, 0) / g["length"] * rmap.get(p, 0)
                       for p in side_players(g, g["away"], R, pos) if p)
    te = subsets["test_regular"]
    q75 = np.quantile(np.concatenate([lvl_h[te], lvl_a[te]]), 0.75)
    subsets["test_high_talent"] = te[(lvl_h[te] >= q75) & (lvl_a[te] >= q75)]

    def mw_feat(Xl, Al, Ml):
        d = torch.diagonal(Al, dim1=1, dim2=2)
        fh = (Xl[:, :R] * d[:, :R, None]).sum(1) / d[:, :R].sum(1, keepdim=True)
        fa = (Xl[:, R:] * d[:, R:, None]).sum(1) / d[:, R:].sum(1, keepdim=True)
        return (fh - fa).numpy()

    for lag in (0, 1):
        Xl, Al, Ml, _, _, covl = cache[lag]
        for suffix, Ause in (("", Al), ("_pre", prior_A[lag])):
            F = mw_feat(Xl, Ause, Ml)
            lin = LogisticRegression(max_iter=2000, C=1.0).fit(F[tr][margin[tr] != 0], win[tr][margin[tr] != 0])
            for name, idx in subsets.items():
                res.setdefault(name, {})[f"feat_lin_lag{lag}{suffix}"] = metrics(
                    lin.predict_proba(F[idx])[:, 1], win[idx], margin[idx], scale)
    # RAPM-sum with pre-game minutes (prior deployment) instead of observed minutes
    dpre = torch.diagonal(prior_A[0], dim1=1, dim2=2).numpy()
    strength_pre = np.zeros(len(games))
    for k, g in enumerate(games):
        pos = S.hockey_pos(g) if S.hockey_pos else None
        ps_ = side_players(g, g["home"], R, pos) + side_players(g, g["away"], R, pos)
        v = np.array([rmap.get(p, 0.0) if p else 0.0 for p in ps_])
        strength_pre[k] = (dpre[k, :R] * v[:R]).sum() - (dpre[k, R:] * v[R:]).sum()
    lr_pre = LogisticRegression(max_iter=2000).fit(strength_pre[tr, None][margin[tr] != 0],
                                                   win[tr][margin[tr] != 0])
    lr = LogisticRegression(max_iter=2000).fit(strength[tr, None][margin[tr] != 0], win[tr][margin[tr] != 0])
    home_rate = win[tr][margin[tr] != 0].mean()
    for name, idx in subsets.items():
        res[name]["home_only"] = metrics(np.full(len(idx), home_rate), win[idx], margin[idx], scale)
        res[name]["rapm_sum"] = metrics(lr.predict_proba(strength[idx, None])[:, 1], win[idx], margin[idx], scale)
        res[name]["rapm_sum_pre"] = metrics(lr_pre.predict_proba(strength_pre[idx, None])[:, 1],
                                            win[idx], margin[idx], scale)

    for tag in args.tags.split(","):
        ens, cfg = load_ensemble(args.sport, tag)
        assert cfg["test_season"] == args.test_season, tag
        Xl, Al, Ml, _, _, _ = cache[cfg["lag"]]
        variants = {"": Al, "@pre": prior_A[cfg["lag"]]}
        if cfg.get("prior_deploy"):
            variants = {"": prior_A[cfg["lag"]]}
        for suffix, Ause in variants.items():
            if cfg.get("shuffle_graph"):
                from train import shuffle_graph
                Ause = shuffle_graph(Ause, Ml, R)
            for name, idx in subsets.items():
                res[name][tag + suffix] = metrics(predict(ens, Xl, Ause, Ml, R, torch.tensor(idx)),
                                                  win[idx], margin[idx], scale)
    res["meta"] = {"rapm_alpha": alpha, "train_games": int(len(tr)),
                   "coverage_lag0": cache[0][5], "coverage_lag1": cache[1][5],
                   "n": {k: int(len(v)) for k, v in subsets.items()}}
    (OUT / f"E1_{args.sport}_{args.test_season}.json").write_text(json.dumps(res, indent=2))
    for name in subsets:
        print(f"\n== {name} (n={len(subsets[name])})")
        for k, v in res[name].items():
            print(f"  {k:24s} logloss {v['log_loss']:.4f}  brier {v['brier']:.4f}  auc {v['auc']:.3f}  acc {v['acc']:.3f}")


if __name__ == "__main__":
    main()
