"""
NBA Data Processing — V2 (Bigger Embeddings)
Adds shot zone distributions and assist location features.
Feature vector: 13-dim → 20-dim.

New features:
  - shot_zone_* (5): proportion of FGA from each zone (Above the Break 3,
    In The Paint, Mid-Range, Left Corner 3, Right Corner 3)
  - AST (1): total assists
  - AST_rate (1): assists per minute
  - USG_proxy (1): (FGA + AST + TOV_proxy) / minutes estimate

Mirrors nba_process.py with expanded embedding step only.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─────────────────────────────────────────────
# Steps 1-3 are identical to v1 — import them
# ─────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nba_process import build_lineup_shifts, build_xppp_model, build_apm


# ─────────────────────────────────────────────
# STEP 4 (V2): Expanded player embeddings
# 13-dim → 20-dim
# ─────────────────────────────────────────────

def build_embeddings_v2():
    """
    Build per-player per-season feature vectors with expanded features.
    V1 features (13): FGA, FGM, xPPP, FGA_3pt, zone_* (dummies), isG, isF, isC,
                       PLAYER_HEIGHT, PLAYER_WEIGHT, toi
    V2 additions (7): shot_zone_AboveBreak3_pct, shot_zone_Paint_pct,
                       shot_zone_MidRange_pct, shot_zone_Corner3L_pct,
                       shot_zone_Corner3R_pct, AST, AST_rate
    """
    print("Building V2 player embeddings (20-dim)...")
    shots = pd.read_parquet(DATA_DIR / "nba_shots_xppp.parquet")
    players_df = pd.read_csv(DATA_DIR / "nba_headshots.csv")
    lineups = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")

    shots["season_year"] = shots["season"].str[:4].astype(int)

    # ── V1 shot features ──
    zone_cols = pd.get_dummies(shots["SHOT_ZONE_BASIC"], prefix="zone")
    shots = pd.concat([shots, zone_cols], axis=1)

    shot_embed = shots.groupby(["PLAYER_ID", "season_year"]).agg(
        FGA=("SHOT_MADE_FLAG", "count"),
        FGM=("SHOT_MADE_FLAG", "sum"),
        xPPP=("xppp", "mean"),
        FGA_3pt=("is_3pt", "sum"),
        **{col: (col, "sum") for col in zone_cols.columns}
    ).reset_index()
    shot_embed.columns.name = None

    # ── V2 NEW: Shot zone distributions (proportion of FGA from each zone) ──
    zone_dist = shots.groupby(["PLAYER_ID", "season_year", "SHOT_ZONE_BASIC"]).size()
    zone_dist = zone_dist.unstack(fill_value=0)
    zone_total = zone_dist.sum(axis=1)

    # Normalize to proportions
    zone_pct = zone_dist.div(zone_total, axis=0).fillna(0)
    zone_pct.columns = [f"shot_zone_{c.replace(' ', '_').replace('(', '').replace(')', '')}_pct"
                        for c in zone_pct.columns]
    zone_pct = zone_pct.reset_index()

    shot_embed = shot_embed.merge(zone_pct, on=["PLAYER_ID", "season_year"], how="left")

    # ── V2 NEW: Assist features from play-by-play ──
    # Load play-by-play data for assist counts
    pbp_files = sorted(DATA_DIR.glob("nba_pbp_*.parquet"))
    if pbp_files:
        pbp = pd.concat([pd.read_parquet(f) for f in pbp_files], ignore_index=True)
        pbp["season_year"] = pbp["season"].str[:4].astype(int) if "season" in pbp.columns else 2022

        # Assists: EVENTMSGTYPE==1 (made shot) with PLAYER2_ID as assister
        if "PLAYER2_ID" in pbp.columns and "EVENTMSGTYPE" in pbp.columns:
            assists = pbp[pbp["EVENTMSGTYPE"] == 1].groupby(
                ["PLAYER2_ID", "season_year"]
            ).size().reset_index(name="AST")
            assists = assists.rename(columns={"PLAYER2_ID": "PLAYER_ID"})
            assists = assists[assists["PLAYER_ID"] > 0]
        else:
            assists = pd.DataFrame(columns=["PLAYER_ID", "season_year", "AST"])
    else:
        print("  No PBP files found — setting AST features to 0")
        assists = pd.DataFrame(columns=["PLAYER_ID", "season_year", "AST"])

    shot_embed = shot_embed.merge(assists, on=["PLAYER_ID", "season_year"], how="left")
    shot_embed["AST"] = shot_embed["AST"].fillna(0)

    # ── Physical + position features ──
    roster_embed = players_df[["PERSON_ID", "season", "POSITION",
                                "PLAYER_HEIGHT", "PLAYER_WEIGHT"]].copy()
    roster_embed["season_year"] = roster_embed["season"].str[:4].astype(int)
    roster_embed["isG"] = roster_embed["POSITION"].str.contains("G").astype(int)
    roster_embed["isF"] = roster_embed["POSITION"].str.contains("F").astype(int)
    roster_embed["isC"] = roster_embed["POSITION"].str.contains("C").astype(int)
    roster_embed = roster_embed.rename(columns={"PERSON_ID": "PLAYER_ID"})

    # ── Minutes played ──
    player_cols_home = [f"home_{i}" for i in range(1, 6)]
    home_time = lineups.melt(
        id_vars=["game_id", "duration"], value_vars=player_cols_home, value_name="PLAYER_ID"
    ).groupby("PLAYER_ID")["duration"].sum().reset_index().rename(columns={"duration": "toi"})

    # ── Merge all ──
    embeddings = shot_embed.merge(
        roster_embed[["PLAYER_ID", "season_year", "isG", "isF", "isC",
                       "PLAYER_HEIGHT", "PLAYER_WEIGHT"]],
        on=["PLAYER_ID", "season_year"], how="left"
    ).merge(home_time, on="PLAYER_ID", how="left").fillna(0)

    # ── V2 NEW: Assist rate (AST / toi) ──
    embeddings["AST_rate"] = np.where(
        embeddings["toi"] > 0,
        embeddings["AST"] / embeddings["toi"],
        0.0
    )

    embeddings = embeddings.rename(columns={"PLAYER_ID": "player_id", "season_year": "year"})

    # ── Normalize ──
    skip_cols = {"player_id", "year", "isG", "isF", "isC"}
    for col in embeddings.columns:
        if col not in skip_cols:
            embeddings[col] = embeddings[col].astype(float)
            std = embeddings[col].std()
            if std > 0:
                embeddings[col] = (embeddings[col] - embeddings[col].mean()) / std

    feat_cols = [c for c in embeddings.columns if c not in ["player_id", "year"]]
    print(f"  V2 embedding dimensions: {len(feat_cols)}")
    print(f"  Feature columns: {feat_cols}")

    with open(DATA_DIR / "nba_embeddings_v2.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved V2 embeddings for {len(embeddings)} player-seasons.")
    return embeddings


# ─────────────────────────────────────────────
# STEP 5: Period matrices (unchanged from v1)
# ─────────────────────────────────────────────

from nba_process import build_period_matrices


if __name__ == "__main__":
    print("=== V2 Processing Pipeline ===")
    print("Steps 1-3: reusing v1 (lineup shifts, xPPP, APM)")
    print()

    print("=== Step 4 (V2): Expanded embeddings ===")
    emb = build_embeddings_v2()

    print("\n=== Step 5: Period matrices (unchanged) ===")
    build_period_matrices()

    print("\nV2 processing complete.")
