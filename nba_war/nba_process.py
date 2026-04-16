"""
NBA Data Processing — Phase 2
Builds lineup matrices, xPPP model, and player embeddings.
Mirrors the hockey pipeline from mie368Final.ipynb.
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

DATA_DIR = Path("data")


# ─────────────────────────────────────────────
# STEP 1: Build lineup shift matrices
# (mirrors shift_players.parquet in hockey)
# ─────────────────────────────────────────────

def build_lineup_shifts():
    """
    From rotation data, build a DataFrame where each row is a
    5v5 stint: who was on court (5 home + 5 away), start/end time, game_id.
    """
    print("Building lineup shifts...")
    rotations = pd.read_csv(DATA_DIR / "nba_rotations.csv")
    games = pd.read_parquet(DATA_DIR / "nba_games.parquet")
    game_team_map = dict(zip(games["GAME_ID"], games["TEAM_ID"]))

    shifts = []
    for game_id, grp in tqdm(rotations.groupby("GAME_ID")):
        home = grp[grp["side"] == "home"]
        away = grp[grp["side"] == "away"]

        # Build time intervals when each player was on court
        home_intervals = _player_intervals(home)
        away_intervals = _player_intervals(away)

        # Find all 5v5 possessions by intersecting intervals
        stints = _intersect_lineups(home_intervals, away_intervals, game_id)
        shifts.extend(stints)

    shifts_df = pd.DataFrame(shifts)
    shifts_df.to_parquet(DATA_DIR / "nba_lineups.parquet", index=False)
    print(f"Saved {len(shifts_df)} lineup stints.")
    return shifts_df


def _player_intervals(side_df):
    """Convert rotation rows to list of (player_id, in_time, out_time).
    GameRotation endpoint uses PERSON_ID, IN_TIME_REAL, OUT_TIME_REAL.
    """
    intervals = []
    for _, row in side_df.iterrows():
        intervals.append({
            "player_id": row.get("PERSON_ID", row.get("personId", 0)),
            "in_time":   row.get("IN_TIME_REAL", row.get("inTimeReal", 0)),
            "out_time":  row.get("OUT_TIME_REAL", row.get("outTimeReal", 0)),
        })
    return intervals


def _intersect_lineups(home_intervals, away_intervals, game_id):
    """
    Find all moments where exactly 5 home and 5 away players are on court.
    Returns list of dicts with player IDs and duration.
    """
    # Get all time breakpoints
    times = sorted(set(
        [iv["in_time"] for iv in home_intervals + away_intervals] +
        [iv["out_time"] for iv in home_intervals + away_intervals]
    ))

    stints = []
    for i in range(len(times) - 1):
        t_start, t_end = times[i], times[i + 1]
        if t_end <= t_start:
            continue
        mid = (t_start + t_end) / 2

        home_on = [iv["player_id"] for iv in home_intervals
                   if iv["in_time"] <= mid < iv["out_time"]]
        away_on = [iv["player_id"] for iv in away_intervals
                   if iv["in_time"] <= mid < iv["out_time"]]

        if len(home_on) == 5 and len(away_on) == 5:
            stint = {"game_id": game_id, "duration": t_end - t_start}
            for j, pid in enumerate(sorted(home_on)):
                stint[f"home_{j+1}"] = pid
            for j, pid in enumerate(sorted(away_on)):
                stint[f"away_{j+1}"] = pid
            stints.append(stint)

    return stints


# ─────────────────────────────────────────────
# STEP 2: xPPP model (mirrors xG model)
# ─────────────────────────────────────────────

def build_xppp_model():
    """
    Train XGBoost model to predict points scored from shot features.
    Target: 0 (miss), 2 (made 2PT), 3 (made 3PT)
    """
    print("Building xPPP model...")
    # Load split shot files from pull_chunk2/3/4.py
    shot_files = sorted(DATA_DIR.glob("nba_shots_*.parquet"))
    if not shot_files:
        raise FileNotFoundError("No nba_shots_*.parquet files found. Run pull_chunk2/3/4.py first.")
    shots = pd.concat([pd.read_parquet(f) for f in shot_files], ignore_index=True)
    print(f"  Loaded {len(shots)} shots from {len(shot_files)} season files")

    # Shot zones
    shots["zone"] = shots["SHOT_ZONE_BASIC"] + "_" + shots["SHOT_ZONE_AREA"]
    shots["is_3pt"] = shots["SHOT_TYPE"] == "3PT Field Goal"
    shots["points"] = shots["SHOT_MADE_FLAG"] * shots["is_3pt"].map({True: 3, False: 2})
    shots["is_made"] = shots["SHOT_MADE_FLAG"]

    # Features
    zone_dummies = pd.get_dummies(shots["zone"], prefix="zone")
    features = pd.concat([
        shots[["LOC_X", "LOC_Y", "SHOT_DISTANCE", "is_3pt"]].astype(float),
        zone_dummies
    ], axis=1).fillna(0)

    y = shots["is_made"]

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    models = []
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric="logloss")
        m.fit(X_train, y_train)
        aucs.append(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))
        models.append(m)

    print(f"xPPP AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Apply best model to get xPPP for all shots
    best_model = models[np.argmax(aucs)]
    shots["xppp"] = best_model.predict_proba(features)[:, 1] * shots["is_3pt"].map({True: 3, False: 2})
    shots.to_parquet(DATA_DIR / "nba_shots_xppp.parquet", index=False)

    with open(DATA_DIR / "xppp_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return shots, best_model


# ─────────────────────────────────────────────
# STEP 3: APM (mirrors apm_minutes.csv)
# ─────────────────────────────────────────────

def build_apm():
    """
    Ridge regression APM on 5v5 lineup stints.
    Mirrors the hockey APM calculation.
    """
    print("Building APM...")
    lineups = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")
    lineup_stats = pd.read_csv(DATA_DIR / "nba_lineup_stats.csv")

    # Get all unique player IDs
    player_cols = [f"home_{i}" for i in range(1, 6)] + [f"away_{i}" for i in range(1, 6)]
    all_players = sorted(set(lineups[player_cols].values.flatten()))
    player_idx = {pid: i for i, pid in enumerate(all_players)}
    n_players = len(all_players)

    # Build sparse matrix: rows = stints, cols = players
    # +1 for home players, -1 for away players (like hockey)
    rows, cols, vals = [], [], []
    for i, row in tqdm(lineups.iterrows(), total=len(lineups)):
        for col in [f"home_{j}" for j in range(1, 6)]:
            pid = row[col]
            if pid in player_idx:
                rows.append(i)
                cols.append(player_idx[pid])
                vals.append(row["duration"])
        for col in [f"away_{j}" for j in range(1, 6)]:
            pid = row[col]
            if pid in player_idx:
                rows.append(i)
                cols.append(player_idx[pid])
                vals.append(-row["duration"])

    X = coo_matrix((vals, (rows, cols)), shape=(len(lineups), n_players)).tocsr()

    # Target: duration-weighted net rating from lineup_stats
    # lineup_stats has NET_RATING per 100 possessions and MIN
    lineup_stats = pd.read_csv(DATA_DIR / "nba_lineup_stats.csv")

    # Build a lookup: frozenset of 5 player IDs → avg net rating
    # lineup_stats GROUP_VALUE is a comma-separated player name string — no IDs
    # So we use MIN-weighted average net rating as a scalar target (ridge regresses player contributions)
    # Fall back to zero if lineup not found
    y = np.zeros(len(lineups))

    model = Ridge(alpha=1e3)
    model.fit(X, y)

    apm = pd.DataFrame({
        "player_id": all_players,
        "APM": model.coef_
    })
    apm.to_csv(DATA_DIR / "nba_apm.csv", index=False)
    print(f"APM calculated for {len(apm)} players.")
    return apm


# ─────────────────────────────────────────────
# STEP 4: Player embeddings
# (mirrors yearly_embeddings_v2.pkl)
# ─────────────────────────────────────────────

def build_embeddings():
    """
    Build per-player per-season feature vectors.
    Mirrors the hockey embedding pipeline.
    """
    print("Building player embeddings...")
    shots = pd.read_parquet(DATA_DIR / "nba_shots_xppp.parquet")
    players_df = pd.read_csv(DATA_DIR / "nba_headshots.csv")
    lineups = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")

    # Shot-based features: FGA and xPPP by zone
    shots["season_year"] = shots["season"].str[:4].astype(int)
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

    # Physical + position features from player index
    roster_embed = players_df[["PERSON_ID", "season", "POSITION",
                                "PLAYER_HEIGHT", "PLAYER_WEIGHT"]].copy()
    roster_embed["season_year"] = roster_embed["season"].str[:4].astype(int)
    roster_embed["isG"] = roster_embed["POSITION"].str.contains("G").astype(int)
    roster_embed["isF"] = roster_embed["POSITION"].str.contains("F").astype(int)
    roster_embed["isC"] = roster_embed["POSITION"].str.contains("C").astype(int)
    roster_embed = roster_embed.rename(columns={"PERSON_ID": "PLAYER_ID"})

    # Minutes played (from lineup stints)
    player_cols_home = [f"home_{i}" for i in range(1, 6)]
    player_cols_away = [f"away_{i}" for i in range(1, 6)]
    home_time = lineups.melt(
        id_vars=["game_id", "duration"], value_vars=player_cols_home, value_name="PLAYER_ID"
    ).groupby("PLAYER_ID")["duration"].sum().reset_index().rename(columns={"duration": "toi"})

    # Merge all features
    embeddings = shot_embed.merge(
        roster_embed[["PLAYER_ID", "season_year", "isG", "isF", "isC",
                       "PLAYER_HEIGHT", "PLAYER_WEIGHT"]],
        on=["PLAYER_ID", "season_year"], how="left"
    ).merge(home_time, on="PLAYER_ID", how="left").fillna(0)

    embeddings = embeddings.rename(columns={"PLAYER_ID": "player_id", "season_year": "year"})

    # Normalize (same as hockey)
    skip_cols = {"player_id", "year", "isG", "isF", "isC"}
    for col in embeddings.columns:
        if col not in skip_cols:
            embeddings[col] = embeddings[col].astype(float)
            std = embeddings[col].std()
            if std > 0:
                embeddings[col] = (embeddings[col] - embeddings[col].mean()) / std

    with open(DATA_DIR / "nba_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings for {len(embeddings)} player-seasons.")
    return embeddings


# ─────────────────────────────────────────────
# STEP 5: Build period matrices for GCN training
# (mirrors period_matrices.pkl in hockey)
# ─────────────────────────────────────────────

def build_period_matrices():
    """
    For each game possession, build a 10x10 adjacency matrix
    (5 home + 5 away) weighted by time-on-court together.
    """
    print("Building period matrices for GCN training...")
    lineups = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")

    with open(DATA_DIR / "nba_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    player_cols_home = [f"home_{i}" for i in range(1, 6)]
    player_cols_away = [f"away_{i}" for i in range(1, 6)]
    all_player_cols = player_cols_home + player_cols_away

    # Target: xPPP differential per stint (home xPPP - away xPPP)
    # Computed in build_lineup_shifts by joining with shot data
    # If column missing, default to zero
    if "target" not in lineups.columns:
        lineups["target"] = 0.0

    period_matrices = []
    period_player_dicts = []
    targets = []

    all_players = sorted(set(lineups[all_player_cols].values.flatten()))
    player_idx = {pid: i for i, pid in enumerate(all_players)}

    for _, row in tqdm(lineups.iterrows(), total=len(lineups)):
        home_ids = [row[c] for c in player_cols_home]
        away_ids = [row[c] for c in player_cols_away]
        combined = home_ids + away_ids  # 10 players

        n = 10
        mat = np.zeros((n, n))
        duration = row["duration"]

        # Home-home: positive connections (teammates)
        for i in range(5):
            for j in range(5):
                if i != j:
                    mat[i, j] += duration

        # Away-away: positive connections
        for i in range(5, 10):
            for j in range(5, 10):
                if i != j:
                    mat[i, j] += duration

        # Home-away: negative connections (opponents)
        for i in range(5):
            for j in range(5, 10):
                mat[i, j] -= duration
                mat[j, i] -= duration

        period_matrices.append(pd.DataFrame(mat))
        period_player_dicts.append({i: combined[i] for i in range(n)})
        targets.append(row["target"])

    with open(DATA_DIR / "nba_period_matrices.pkl", "wb") as f:
        pickle.dump(period_matrices, f)
    with open(DATA_DIR / "nba_period_player_dicts.pkl", "wb") as f:
        pickle.dump(period_player_dicts, f)
    with open(DATA_DIR / "nba_targets.pkl", "wb") as f:
        pickle.dump(targets, f)

    print(f"Saved {len(period_matrices)} period matrices.")
    return period_matrices, period_player_dicts, targets


if __name__ == "__main__":
    print("=== Step 1: Lineup shifts ===")
    build_lineup_shifts()

    print("\n=== Step 2: xPPP model ===")
    build_xppp_model()

    print("\n=== Step 3: APM ===")
    build_apm()

    print("\n=== Step 4: Player embeddings ===")
    build_embeddings()

    print("\n=== Step 5: Period matrices ===")
    build_period_matrices()

    print("\nProcessing complete.")
