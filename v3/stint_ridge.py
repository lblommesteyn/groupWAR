"""Box-prior RAPM: one ridge regression on stints with two blocks of columns.

  margin rate = sum_i b . x_i (box-score prior, lightly penalised)
              + sum_i r_i     (per-player residual, RAPM penalty)
              + home

Player value = b . x_i + r_i. Evaluated like eval_scorer E1: game strength is
the rolling pre-game minute-share-weighted sum of player values, on held-out
season games.

  python v3/stint_ridge.py --sport nba --test-season 2023 --lag 1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score

from data_registry import load_sport
from rapm import PER, load_stints
from values import state_dummies

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--test-season", type=int, default=2023)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--alphas", default="1000,3000,10000,30000")
    ap.add_argument("--box-scale", default="0,1,3,10,30")
    ap.add_argument("--std-days", type=int, default=0,
                    help="season-to-date: refit every N days of the test season on all earlier stints")
    ap.add_argument("--stints", default="stints.pkl", help="stints_xg.pkl for NHL expected goals")
    ap.add_argument("--states", action="store_true", help="NHL manpower-state terms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    S = load_sport(args.sport)
    F = len(S.feature_cols)
    repl = {}

    def fv(p, season):
        v = S.vec(p, season - args.lag)
        if v is not None:
            return v
        s = season - args.lag
        if s not in repl:
            cur = S.ps[S.ps["season"] == s]
            lo, hi = cur["minutes_total"].quantile([0.10, 0.30])
            rp = cur[(cur["minutes_total"] >= lo) & (cur["minutes_total"] <= hi)]
            # no feature season at all (first season of data): league average
            repl[s] = (((rp[S.feature_cols] - S.mu) / S.sd).fillna(0).mean().to_numpy(np.float32)
                       if len(rp) else np.zeros(F, np.float32))
        return repl[s]

    st_all = load_stints(args.sport, args.stints)
    gdate = {g["game_id"]: pd.Timestamp(g["date"]) for g in S.games()}
    st_all["date"] = st_all["game_id"].map(gdate)
    st_all = st_all[~st_all["playoff"] & (st_all["secs"] >= 5)]
    st = st_all[st_all["season"] < args.test_season]
    players = sorted({p for t in st["home5"] for p in t} | {p for t in st["away5"] for p in t})
    idx = {p: i for i, p in enumerate(players)}
    rows, cols, vals = [], [], []
    B = np.zeros((len(st), F), np.float32)
    for r, (h, a, s) in enumerate(zip(st["home5"], st["away5"], st["season"])):
        for p in h:
            rows.append(r); cols.append(idx[p]); vals.append(1.0)
            B[r] += fv(p, s)
        for p in a:
            rows.append(r); cols.append(idx[p]); vals.append(-1.0)
            B[r] -= fv(p, s)
    Xid = csr_matrix((vals, (rows, cols)), shape=(len(st), len(players)))
    y = (st["pts_home"] - st["pts_away"]).to_numpy() / st["secs"].to_numpy() * PER[args.sport]
    w = st["secs"].to_numpy().astype(float)
    print(f"{args.sport}: {len(st)} stints, {len(players)} players, lag {args.lag}", flush=True)

    # game-level evaluation set (rolling pre-game deployment)
    from dataset import side_players
    from prior_graphs import rolling_graph_tensors
    games = S.games()
    A, ok = rolling_graph_tensors(S)
    R = S.R
    G = []
    for k, g in enumerate(games):
        if not ok[k]:
            continue
        pos = S.hockey_pos(g) if S.hockey_pos else None
        sides = []
        for side, team in enumerate((g["home"], g["away"])):
            pl = side_players(g, team, R, pos)
            share = np.diagonal(A[k].numpy())[side * R:(side + 1) * R]
            sides.append([(p, float(sh)) for p, sh in zip(pl, share) if p is not None])
        G.append((g["season"], g["playoff"], g["margin"], sides, pd.Timestamp(g["date"])))

    def evaluate(b, r_map, by_date=None):
        strength = []
        for season, _, _, sides, date in G:
            if by_date is not None and season == args.test_season:
                b, r_map = by_date(date)
            v = [sum(sh * (float(fv(p, season) @ b) + r_map.get(p, 0.0)) for p, sh in side) for side in sides]
            strength.append(v[0] - v[1])
        strength = np.array(strength)
        gs = np.array([x[0] for x in G]); gp = np.array([x[1] for x in G]); gm = np.array([x[2] for x in G])
        win = (gm > 0).astype(int)
        trg = (gs < args.test_season) & ~gp & (gm != 0)
        lr = LogisticRegression().fit(strength[trg, None], win[trg])
        out = {}
        for name, m in (("test_regular", (gs == args.test_season) & ~gp),
                        ("test_playoffs", (gs == args.test_season) & gp)):
            if m.sum():
                p = lr.predict_proba(strength[m, None])[:, 1]
                out[name] = {"n": int(m.sum()), "auc": roc_auc_score(win[m], p),
                             "logloss": log_loss(win[m], p), "acc": float(((p > 0.5) == win[m]).mean())}
        return out

    def build(df):
        pl = sorted({p for t in df["home5"] for p in t} | {p for t in df["away5"] for p in t})
        ix = {p: i for i, p in enumerate(pl)}
        rows, cols, vals = [], [], []
        Bm = np.zeros((len(df), F), np.float32)
        for r, (h, a, s_) in enumerate(zip(df["home5"], df["away5"], df["season"])):
            for p in h:
                rows.append(r); cols.append(ix[p]); vals.append(1.0); Bm[r] += fv(p, s_)
            for p in a:
                rows.append(r); cols.append(ix[p]); vals.append(-1.0); Bm[r] -= fv(p, s_)
        yy = (df["pts_home"] - df["pts_away"]).to_numpy() / df["secs"].to_numpy() * PER[args.sport]
        return csr_matrix((vals, (rows, cols)), shape=(len(df), len(pl))), Bm, pl, yy, df["secs"].to_numpy().astype(float)

    def std_fitter(c, a):
        """Refit on all stints before each cutoff date; cache by cutoff."""
        test_dates = sorted({x[4] for x in G if x[0] == args.test_season})
        start = test_dates[0]
        cache = {}

        def get(date):
            k = (date - start).days // args.std_days
            if k not in cache:
                cut = start + pd.Timedelta(days=k * args.std_days)
                df = st_all[(st_all["season"] <= args.test_season) & (st_all["date"] < cut)]
                Xi, Bm, pl, yy, ww = build(df)
                X = hstack([csr_matrix(Bm * c), Xi]).tocsr() if c > 0 else Xi
                if args.states:
                    X = hstack([X, state_dummies(df)]).tocsr()
                m = Ridge(alpha=a, fit_intercept=True).fit(X, yy, sample_weight=ww)
                bb = m.coef_[:F] * c if c > 0 else np.zeros(F)
                r_ = m.coef_[(F if c > 0 else 0):(F if c > 0 else 0) + len(pl)]
                cache[k] = (bb, dict(zip(pl, r_)))
            return cache[k]
        return get

    res = {"args": vars(args), "grid": []}
    for c in [float(x) for x in args.box_scale.split(",")]:
        for a in [float(x) for x in args.alphas.split(",")]:
            # scaling the box block by c divides its effective penalty by c^2
            X = hstack([csr_matrix(B * c), Xid]).tocsr() if c > 0 else Xid
            if args.states:
                X = hstack([X, state_dummies(st)]).tocsr()
            m = Ridge(alpha=a, fit_intercept=True).fit(X, y, sample_weight=w)
            b = m.coef_[:F] * c if c > 0 else np.zeros(F)
            rr = m.coef_[(F if c > 0 else 0):(F if c > 0 else 0) + len(players)]
            by_date = None
            if args.std_days:
                by_date = std_fitter(c, a)
            ev = evaluate(b, dict(zip(players, rr)), by_date)
            row = {"box_scale": c, "alpha": a, **{k: v for k, v in ev.items()}}
            res["grid"].append(row)
            t = ev["test_regular"]
            print(f"  box x{c:<4g} alpha {a:<7g} auc {t['auc']:.3f} logloss {t['logloss']:.4f} "
                  f"acc {t['acc']:.3f}", flush=True)
    if args.out:
        (ROOT / "v3" / "results" / f"stintridge_{args.sport}_{args.out}.json").write_text(
            json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
