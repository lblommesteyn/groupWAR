"""Compare RAPM ratings fit on different stint targets (actual vs luck-adjusted vs xPts).

Same evaluation as stint_ridge.py with --box-scale 0: game strength is the rolling
pre-game minute-share-weighted sum of player ratings; a logistic fit on training
seasons maps it to a win probability on held-out regular-season games. Every
variant is scored on the same games against the actual result, and AUC differences
against the actual-points ratings get a paired bootstrap over games.

  python v3/luck_eval.py --test-season 2023 --alphas 30000,100000,300000 [--std-days 14]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score

from data_registry import load_sport
from dataset import side_players
from prior_graphs import rolling_graph_tensors
from rapm import PER, load_stints

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["actual", "ctg", "inpred", "xpts", "xpts_shooter"]


def fit(df, alpha):
    pl = sorted({p for t in df["home5"] for p in t} | {p for t in df["away5"] for p in t})
    ix = {p: i for i, p in enumerate(pl)}
    rows, cols, vals = [], [], []
    for r, (h, a) in enumerate(zip(df["home5"], df["away5"])):
        for p in h:
            rows.append(r); cols.append(ix[p]); vals.append(1.0)
        for p in a:
            rows.append(r); cols.append(ix[p]); vals.append(-1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(df), len(pl)))
    y = (df["pts_home"] - df["pts_away"]).to_numpy() / df["secs"].to_numpy() * PER["nba"]
    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=df["secs"].to_numpy().astype(float))
    return dict(zip(pl, m.coef_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=2023)
    ap.add_argument("--alphas", default="30000,100000,300000")
    ap.add_argument("--std-days", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T = args.test_season

    S = load_sport("nba")
    games = S.games()
    A, ok = rolling_graph_tensors(S)
    R = S.R
    G = []
    for k, g in enumerate(games):
        if not ok[k]:
            continue
        sides = []
        for side, team in enumerate((g["home"], g["away"])):
            pl = side_players(g, team, R, None)
            share = np.diagonal(A[k].numpy())[side * R:(side + 1) * R]
            sides.append([(p, float(sh)) for p, sh in zip(pl, share) if p is not None])
        G.append((g["season"], g["playoff"], g["margin"], sides, pd.Timestamp(g["date"]), g["game_id"]))
    gs = np.array([x[0] for x in G]); gp = np.array([x[1] for x in G]); gm = np.array([x[2] for x in G])
    win = (gm > 0).astype(int)
    trg = (gs < T) & ~gp & (gm != 0)
    test = (gs == T) & ~gp
    gdate = {g["game_id"]: pd.Timestamp(g["date"]) for g in games}
    test_dates = sorted({x[4] for x in G if x[0] == T})

    def strength(rating_for):
        out = np.zeros(len(G))
        for i, (season, _, _, sides, date, _) in enumerate(G):
            r = rating_for(season, date)
            v = [sum(sh * r.get(p, 0.0) for p, sh in side) for side in sides]
            out[i] = v[0] - v[1]
        return out

    res = {"args": vars(args), "results": {}}
    probs = {}
    for var in args.variants.split(","):
        st_all = load_stints("nba", f"stints_{var}.pkl")
        st_all["date"] = st_all["game_id"].map(gdate)
        st_all = st_all[~st_all["playoff"] & (st_all["secs"] >= 5)]
        for alpha in [float(a) for a in args.alphas.split(",")]:
            base = fit(st_all[st_all["season"] < T], alpha)
            cache = {}

            def rating_for(season, date, base=base, cache=cache, alpha=alpha, st_all=st_all):
                if not args.std_days or season != T:
                    return base
                kk = (date - test_dates[0]).days // args.std_days
                if kk not in cache:
                    cut = test_dates[0] + pd.Timedelta(days=kk * args.std_days)
                    cache[kk] = fit(st_all[(st_all["season"] <= T) & (st_all["date"] < cut)], alpha)
                return cache[kk]

            s = strength(rating_for)
            lr = LogisticRegression().fit(s[trg, None], win[trg])
            p = lr.predict_proba(s[test, None])[:, 1]
            key = f"{var}@{int(alpha)}"
            probs[key] = p
            res["results"][key] = {"variant": var, "alpha": alpha, "n": int(test.sum()),
                                   "auc": roc_auc_score(win[test], p), "logloss": log_loss(win[test], p),
                                   "ratings_n": len(base)}
            res["results"][key]["ratings"] = {str(k): float(v) for k, v in base.items()}
            print(f"{key:24s} auc {res['results'][key]['auc']:.4f} logloss {res['results'][key]['logloss']:.4f}",
                  flush=True)

    # paired bootstrap vs actual at the same alpha
    y = win[test]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
    for key, p in probs.items():
        var, a = key.split("@")
        ref = probs.get(f"actual@{a}")
        if var == "actual" or ref is None:
            continue
        d = []
        for b in idx:
            if y[b].min() == y[b].max():
                continue
            d.append(roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], ref[b]))
        d = np.array(d)
        dl = [log_loss(y[b], p[b], labels=[0, 1]) - log_loss(y[b], ref[b], labels=[0, 1]) for b in idx[:500]]
        res["results"][key]["auc_diff"] = float(roc_auc_score(y, p) - roc_auc_score(y, ref))
        res["results"][key]["auc_diff_ci"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        res["results"][key]["logloss_diff_ci"] = [float(np.quantile(dl, 0.025)), float(np.quantile(dl, 0.975))]
        print(f"{key:24s} dAUC {res['results'][key]['auc_diff']:+.4f} "
              f"[{res['results'][key]['auc_diff_ci'][0]:+.4f}, {res['results'][key]['auc_diff_ci'][1]:+.4f}]",
              flush=True)
    np.savez(ROOT / "v3" / "results" / f"luck_{args.out}_probs.npz", y=y, **probs)
    (ROOT / "v3" / "results" / f"luck_{args.out}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
