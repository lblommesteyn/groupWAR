"""EPM vs our stint ratings as GroupWAR's step-one rating, on held-out seasons.

Each game in season s is rated with information from before s only:
  ridge : stint ridge ratings fitted on seasons <= s-1 (ValueModel, as the value scorer)
  epm   : each player's end-of-season EPM for s-1 (dunksandthrees)
  blend : average of the two
Game strength = sum over each side of pre-game prior minutes x rating. Since AUC
only depends on the ordering of the strength difference, no fitting is needed.

  python v3/epm_eval.py --epm <path to epm_2019_2024.json> --seasons 2021,2022,2023
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anchored import node_ratings, rating_tables
from data_registry import load_sport
from prior_graphs import prior_graph_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epm", required=True)
    ap.add_argument("--seasons", default="2021,2022,2023")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    S = load_sport("nba")
    games = S.games()
    X, A, M, y, meta, _ = S.tensors(lag=1)
    A = prior_graph_tensors(S, X, A, M, meta, lag=1)
    mins = torch.diagonal(A, dim1=1, dim2=2) * M
    R = S.R
    seasons = np.array([m["season"] for m in meta])
    playoff = np.array([m["playoff"] for m in meta])
    win = (y.numpy() > 0).astype(int)
    epm = {int(s): {int(d["player_id"]): float(d["tot"]) for d in rows if d["tot"] is not None}
           for s, rows in json.load(open(args.epm)).items()}

    def strength(Rt):
        v = (mins * Rt).numpy()
        return v[:, :R].sum(1) - v[:, R:].sum(1)

    out = {}
    rng = np.random.default_rng(0)
    for s in [int(x) for x in args.seasons.split(",")]:
        idx = np.where((seasons == s) & ~playoff)[0]
        ridge_tab = rating_tables([s], s - 1, cross_fit=False)
        R_ridge = node_ratings(S, games, ridge_tab, R)
        epm_tab = {s: {"r": epm.get(s - 1, {}), "b": np.zeros(len(S.feature_cols)), "repl": 0.0}}
        R_epm = node_ratings(S, games, epm_tab, R)
        st = {"ridge": strength(R_ridge), "epm": strength(R_epm)}
        st["blend"] = 0.5 * st["ridge"] + 0.5 * st["epm"]
        w = win[idx]
        res = {k: float(roc_auc_score(w, v[idx])) for k, v in st.items()}
        diffs = {k: [] for k in ("epm", "blend")}
        for _ in range(args.boot):
            i = idx[rng.integers(0, len(idx), len(idx))]
            base = roc_auc_score(win[i], st["ridge"][i])
            for k in diffs:
                diffs[k].append(roc_auc_score(win[i], st[k][i]) - base)
        for k, d in diffs.items():
            res[f"{k}_minus_ridge"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res["n"] = int(len(idx))
        out[s] = res
        print(s, json.dumps(res), flush=True)
    from pathlib import Path
    (Path(__file__).resolve().parent / "results" / "epm_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
