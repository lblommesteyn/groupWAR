"""Ridge-regularised adjusted plus-minus from stints.

NBA: target = home margin per 48 minutes of the stint, weight = stint seconds.
NHL: target = home goal differential per 60 minutes, weight = stint seconds.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v3" / "data"
PER = {"nba": 2880.0, "nhl": 3600.0}


def load_stints(sport, name="stints.pkl"):
    st = pd.read_pickle(DATA / sport / name)
    with open(DATA / sport / "game_graphs.pkl", "rb") as f:
        g = pickle.load(f)
    meta = pd.DataFrame([{"game_id": x["game_id"], "season": x["season"], "playoff": x["playoff"]}
                         for x in g])
    return st.merge(meta, on="game_id", how="inner")


def design(st):
    players = sorted({p for t in st["home5"] for p in t} | {p for t in st["away5"] for p in t})
    idx = {p: i for i, p in enumerate(players)}
    rows, cols, vals = [], [], []
    for r, (h, a) in enumerate(zip(st["home5"], st["away5"])):
        for p in h:
            rows.append(r); cols.append(idx[p]); vals.append(1.0)
        for p in a:
            rows.append(r); cols.append(idx[p]); vals.append(-1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(st), len(players)))
    return X, players


def fit(sport, seasons, alpha=None, regular_only=True):
    st = load_stints(sport)
    st = st[st["season"].isin(seasons)]
    if regular_only:
        st = st[~st["playoff"]]
    st = st[st["secs"] >= 5]
    X, players = design(st)
    y = (st["pts_home"] - st["pts_away"]).to_numpy() / st["secs"].to_numpy() * PER[sport]
    w = st["secs"].to_numpy()
    if alpha is None:  # choose by holding out 20% of games
        games = st["game_id"].unique()
        rng = np.random.default_rng(0)
        test = set(rng.choice(games, len(games) // 5, replace=False))
        te = st["game_id"].isin(test).to_numpy()
        best = None
        for a in [300, 1000, 3000, 10000, 30000, 100000, 300000]:
            m = Ridge(alpha=a, fit_intercept=True).fit(X[~te], y[~te], sample_weight=w[~te])
            err = np.average((m.predict(X[te]) - y[te]) ** 2, weights=w[te])
            if best is None or err < best[0]:
                best = (err, a)
        alpha = best[1]
    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
    secs = pd.Series(np.asarray(abs(X).multiply(w[:, None]).sum(0)).ravel(), index=players)
    return pd.DataFrame({"player_id": players, "rapm": m.coef_, "secs": secs.values}), alpha, m.intercept_


if __name__ == "__main__":
    import sys
    sport = sys.argv[1]
    seasons = [int(s) for s in sys.argv[2].split(",")]
    df, a, b0 = fit(sport, seasons)
    print("alpha", a, "home intercept", b0)
    print(df.sort_values("rapm", ascending=False).head(20))
