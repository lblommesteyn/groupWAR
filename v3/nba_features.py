"""NBA player-season features from pbpstats totals + Kaggle bio data.

Features deliberately exclude any on-court plus-minus quantity, so that RAPM
computed from stints remains an independent benchmark.

Usage: python v3/nba_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "v3" / "data" / "nba"

SUM_COLS = ["Minutes", "GamesPlayed", "Points", "FG2A", "FG2M", "FG3A", "FG3M", "FTA",
            "AtRimFGA", "AtRimFGM", "ShortMidRangeFGA", "LongMidRangeFGA", "Corner3FGA",
            "Arc3FGA", "Corner3FGM", "Arc3FGM", "Assists", "ThreePtAssists", "AtRimAssists",
            "OffRebounds", "DefRebounds", "Steals", "Blocks", "Turnovers", "Fouls",
            "OffPoss", "DefPoss"]

FEATURES = ["mpg", "gp_share", "usage", "ts", "pts36",
            "rim_share", "smid_share", "lmid_share", "c3_share", "a3_share",
            "rim_acc", "fg3_acc", "fta_rate",
            "ast36", "ast3_share", "astrim_share",
            "oreb36", "dreb36", "stl36", "blk36", "tov36", "pf36",
            "height", "weight", "isG", "isF", "isC"]


def load_totals():
    rows = []
    for f in sorted((D / "totals").glob("*.json")):
        season = f.name.split("_")[0]
        for r in json.loads(f.read_text()):
            r["season_str"] = season
            rows.append(r)
    df = pd.DataFrame(rows)
    for c in SUM_COLS:
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["player_id"] = df["EntityId"].astype(int)
    df["usage_x_min"] = pd.to_numeric(df.get("Usage", 0), errors="coerce").fillna(0) * df["Minutes"]
    g = df.groupby(["player_id", "season_str"])
    out = g[SUM_COLS + ["usage_x_min"]].sum().reset_index()
    out["name"] = g["Name"].first().values
    out["team"] = g["TeamAbbreviation"].last().values
    out["season"] = out["season_str"].str[:4].astype(int) + 1  # 2019-20 -> 2020
    return out


def bio():
    p = pd.read_parquet(D / "players.parquet")
    p = p.rename(columns={"person_id": "player_id"})

    def inches(h):
        try:
            a, b = str(h).split("-")
            return int(a) * 12 + int(b)
        except Exception:
            return np.nan
    p["height"] = p["height"].apply(inches)
    p["weight"] = pd.to_numeric(p["weight"], errors="coerce")
    pos = p["position"].fillna("")
    p["isG"] = pos.str.contains("Guard").astype(float)
    p["isF"] = pos.str.contains("Forward").astype(float)
    p["isC"] = pos.str.contains("Center").astype(float)
    p["name_bio"] = p["display_first_last"]
    return p[["player_id", "name_bio", "height", "weight", "isG", "isF", "isC", "country",
              "birthdate", "position"]]


def build():
    t = load_totals()
    b = bio()
    df = t.merge(b, on="player_id", how="left")
    m = df["Minutes"].clip(lower=1)
    fga = (df["FG2A"] + df["FG3A"]).clip(lower=1)
    lg_rim = df["AtRimFGM"].sum() / df["AtRimFGA"].sum()
    lg_3 = df["FG3M"].sum() / df["FG3A"].sum()
    k = 50.0  # shrinkage toward league average for low-volume shooters
    df["mpg"] = df["Minutes"] / df["GamesPlayed"].clip(lower=1)
    df["gp_share"] = df["GamesPlayed"] / 82.0
    df["usage"] = df["usage_x_min"] / m
    df["ts"] = df["Points"] / (2 * (fga + 0.44 * df["FTA"])).clip(lower=1)
    df["pts36"] = 36 * df["Points"] / m
    for src, dst in [("AtRimFGA", "rim_share"), ("ShortMidRangeFGA", "smid_share"),
                     ("LongMidRangeFGA", "lmid_share"), ("Corner3FGA", "c3_share"),
                     ("Arc3FGA", "a3_share")]:
        df[dst] = df[src] / fga
    df["rim_acc"] = (df["AtRimFGM"] + k * lg_rim) / (df["AtRimFGA"] + k)
    df["fg3_acc"] = (df["FG3M"] + k * lg_3) / (df["FG3A"] + k)
    df["fta_rate"] = df["FTA"] / fga
    df["ast36"] = 36 * df["Assists"] / m
    ast = df["Assists"].clip(lower=1)
    df["ast3_share"] = df["ThreePtAssists"] / ast
    df["astrim_share"] = df["AtRimAssists"] / ast
    for src, dst in [("OffRebounds", "oreb36"), ("DefRebounds", "dreb36"), ("Steals", "stl36"),
                     ("Blocks", "blk36"), ("Turnovers", "tov36"), ("Fouls", "pf36")]:
        df[dst] = 36 * df[src] / m
    df["height"] = df["height"].fillna(df["height"].median())
    df["weight"] = df["weight"].fillna(df["weight"].median())
    for c in ["isG", "isF", "isC"]:
        df[c] = df[c].fillna(0.0)
    df.to_parquet(D / "player_seasons.parquet", index=False)
    print(df.groupby("season").size())
    print(df[FEATURES].describe().T[["mean", "std"]])


if __name__ == "__main__":
    build()
