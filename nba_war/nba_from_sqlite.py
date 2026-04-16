"""
NBA Data Processing — from Kaggle SQLite
Reads nba.sqlite (wyattowalsh/basketball dataset) and builds all
pipeline outputs needed for GCN training and the roster search.

Outputs:
  data/nba_headshots.csv         — player info + nationality
  data/nba_games.parquet         — game IDs for 2021-22, 2022-23
  data/nba_player_stats.csv      — per-player per-season counting stats
  data/nba_lineups.parquet       — per-game starting lineup stints + target
  data/nba_apm.csv               — Ridge APM (player value)
  data/nba_embeddings.pkl        — per-player-season feature vectors
  data/nba_period_matrices.pkl   — 10x10 adjacency matrices for GCN
  data/nba_period_player_dicts.pkl
  data/nba_targets.pkl
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "nba.sqlite"

SEASONS = ["22021", "22022"]   # 2021-22, 2022-23 regular season
YEAR_MAP = {"22021": 2022, "22022": 2023}   # season_id → calendar year label


def get_con():
    return sqlite3.connect(DB_PATH)


# ─────────────────────────────────────────────────────────────────
# STEP 1: Player headshots / info
# ─────────────────────────────────────────────────────────────────

def build_headshots():
    out = DATA_DIR / "nba_headshots.csv"
    if out.exists():
        print("nba_headshots.csv already exists, skipping.")
        return pd.read_csv(out)

    print("Building nba_headshots.csv...")
    con = get_con()
    df = pd.read_sql("""
        SELECT
            person_id   AS PERSON_ID,
            first_name  AS PLAYER_FIRST_NAME,
            last_name   AS PLAYER_LAST_NAME,
            country     AS birthCountry,
            height,
            weight,
            position    AS POSITION,
            from_year,
            to_year
        FROM common_player_info
    """, con)
    con.close()

    # Derive position flags
    df["isG"] = df["POSITION"].str.contains("G", na=False).astype(int)
    df["isF"] = df["POSITION"].str.contains("F", na=False).astype(int)
    df["isC"] = df["POSITION"].str.contains("C", na=False).astype(int)

    # Add season column (replicate row for each season player was active)
    rows = []
    for _, row in df.iterrows():
        for sid, yr in YEAR_MAP.items():
            season_str = f"{yr-1}-{str(yr)[2:]}"
            if row["from_year"] <= yr <= row["to_year"] + 1:
                r = row.copy()
                r["season"] = season_str
                rows.append(r)

    players_df = pd.DataFrame(rows).reset_index(drop=True)
    players_df.to_csv(out, index=False)
    print(f"  Saved {len(players_df)} player-season rows.")
    return players_df


# ─────────────────────────────────────────────────────────────────
# STEP 2: Game IDs
# ─────────────────────────────────────────────────────────────────

def build_games():
    out = DATA_DIR / "nba_games.parquet"
    if out.exists():
        print("nba_games.parquet already exists, skipping.")
        return pd.read_parquet(out)

    print("Building nba_games.parquet...")
    con = get_con()
    season_filter = ",".join(f"'{s}'" for s in SEASONS)
    df = pd.read_sql(f"""
        SELECT game_id AS GAME_ID, season_id, game_date,
               team_id_home, team_abbreviation_home,
               team_id_away, team_abbreviation_away,
               pts_home, pts_away,
               (pts_home - pts_away) AS home_margin
        FROM game
        WHERE season_id IN ({season_filter})
    """, con)
    con.close()
    df["year"] = df["season_id"].map(YEAR_MAP)
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df)} games.")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 3: Per-player counting stats from play_by_play
# eventmsgtype: 1=FGM, 2=FGmiss, 3=FTM, 4=FTmiss, 5=TO, 6=foul,
#               8=sub, 10=jumpball, 11=ejection, 12=period start
# ─────────────────────────────────────────────────────────────────

def build_player_stats():
    out = DATA_DIR / "nba_player_stats.csv"
    if out.exists():
        print("nba_player_stats.csv already exists, skipping.")
        return pd.read_csv(out)

    print("Building nba_player_stats.csv from play_by_play...")
    con = get_con()
    season_filter = ",".join(f"'{s}'" for s in SEASONS)

    # Join PBP with game to get season_id
    pbp = pd.read_sql(f"""
        SELECT p.game_id, p.eventmsgtype, p.eventmsgactiontype,
               p.player1_id, p.player1_team_id,
               p.player2_id, p.player2_team_id,
               g.season_id
        FROM play_by_play p
        JOIN game g ON p.game_id = g.game_id
        WHERE g.season_id IN ({season_filter})
          AND p.player1_id IS NOT NULL
          AND p.player1_id != 0
    """, con)
    con.close()

    pbp["year"] = pbp["season_id"].map(YEAR_MAP)

    # Field goals (player1 is shooter)
    fgm  = pbp[pbp["eventmsgtype"] == 1][["player1_id","year"]].copy()
    fgms = pbp[pbp["eventmsgtype"] == 2][["player1_id","year"]].copy()

    # 3-point: actiontype 1=2pt, 79/80/97/98/99/100/101 ≈ 3pt
    THREE_ACTIONS = {79, 80, 97, 98, 99, 100, 101}
    fg_all = pbp[pbp["eventmsgtype"].isin([1, 2])].copy()
    fg_all["is_3pt"] = fg_all["eventmsgactiontype"].isin(THREE_ACTIONS).astype(int)
    fg_all["is_made"] = (fg_all["eventmsgtype"] == 1).astype(int)

    stats = fg_all.groupby(["player1_id", "year"]).agg(
        FGA=("is_made", "count"),
        FGM=("is_made", "sum"),
        FGA_3pt=("is_3pt", "sum"),
        FGM_3pt=("is_3pt", lambda x: (x * (fg_all.loc[x.index, "is_made"])).sum()),
    ).reset_index()

    stats["eFG"] = (stats["FGM"] + 0.5 * stats["FGM_3pt"]) / stats["FGA"].clip(lower=1)
    stats["three_rate"] = stats["FGA_3pt"] / stats["FGA"].clip(lower=1)
    stats = stats.rename(columns={"player1_id": "player_id"})

    stats.to_csv(out, index=False)
    print(f"  Saved stats for {len(stats)} player-seasons.")
    return stats


# ─────────────────────────────────────────────────────────────────
# STEP 4: Starting lineups per game (5v5) + score margin target
# Strategy: for each game, find first 5 distinct players per team
#           who appear in period-1 PBP events (subs excluded).
# ─────────────────────────────────────────────────────────────────

def build_lineups():
    out = DATA_DIR / "nba_lineups.parquet"
    if out.exists():
        print("nba_lineups.parquet already exists, skipping.")
        return pd.read_parquet(out)

    print("Building nba_lineups.parquet (starting lineups from PBP)...")
    games = build_games()
    con = get_con()
    season_filter = ",".join(f"'{s}'" for s in SEASONS)

    pbp = pd.read_sql(f"""
        SELECT p.game_id, p.period, p.eventnum, p.eventmsgtype,
               p.player1_id, p.player1_team_id,
               p.player2_id, p.player2_team_id,
               p.player3_id
        FROM play_by_play p
        JOIN game g ON p.game_id = g.game_id
        WHERE g.season_id IN ({season_filter})
          AND p.period = 1
          AND p.eventmsgtype NOT IN (8, 12)
          AND p.player1_id IS NOT NULL AND p.player1_id != 0
        ORDER BY p.game_id, p.eventnum
    """, con)
    con.close()

    # Normalize team IDs to int for comparison (game table=str, pbp=float)
    games["team_id_home"] = games["team_id_home"].astype(float).astype(int)
    games["team_id_away"] = games["team_id_away"].astype(float).astype(int)
    game_meta = games.set_index("GAME_ID")
    lineups = []

    for game_id, grp in tqdm(pbp.groupby("game_id"), desc="Games"):
        if game_id not in game_meta.index:
            continue
        meta = game_meta.loc[game_id]
        home_id = int(meta["team_id_home"])
        away_id = int(meta["team_id_away"])
        margin  = meta["home_margin"]

        home_players, away_players = [], []
        for _, row in grp.iterrows():
            pid = row["player1_id"]
            tid = row["player1_team_id"]
            if pid is None or tid is None or pid == 0:
                continue
            try:
                tid = int(float(tid))
                pid = int(float(pid))
            except (ValueError, TypeError):
                continue
            if tid == home_id and pid not in home_players:
                home_players.append(pid)
            elif tid == away_id and pid not in away_players:
                away_players.append(pid)
            if len(home_players) >= 5 and len(away_players) >= 5:
                break

        if len(home_players) >= 5 and len(away_players) >= 5:
            row_dict = {"game_id": game_id, "target": margin, "year": meta["year"]}
            for j in range(5):
                row_dict[f"home_{j+1}"] = home_players[j]
                row_dict[f"away_{j+1}"] = away_players[j]
            lineups.append(row_dict)

    df = pd.DataFrame(lineups)
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df)} game lineups.")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 5: APM via Ridge regression on game-level lineup indicator
# ─────────────────────────────────────────────────────────────────

def build_apm():
    out = DATA_DIR / "nba_apm.csv"
    if out.exists():
        print("nba_apm.csv already exists, skipping.")
        return pd.read_csv(out)

    print("Building nba_apm.csv...")
    lineups = build_lineups()

    home_cols = [f"home_{i}" for i in range(1, 6)]
    away_cols = [f"away_{i}" for i in range(1, 6)]

    all_players = sorted(set(lineups[home_cols + away_cols].values.flatten().tolist()))
    pid_to_idx  = {p: i for i, p in enumerate(all_players)}
    n = len(all_players)
    m = len(lineups)

    X = lil_matrix((m, n), dtype=np.float32)
    y = lineups["target"].values.astype(np.float32)

    for i, row in enumerate(lineups.itertuples(index=False)):
        for col in home_cols:
            pid = getattr(row, col)
            if pid in pid_to_idx:
                X[i, pid_to_idx[pid]] = 1.0
        for col in away_cols:
            pid = getattr(row, col)
            if pid in pid_to_idx:
                X[i, pid_to_idx[pid]] = -1.0

    model = Ridge(alpha=1000)
    model.fit(X.tocsr(), y)

    apm = pd.DataFrame({"player_id": all_players, "APM": model.coef_})
    apm.to_csv(out, index=False)
    print(f"  APM for {len(apm)} players.")
    return apm


# ─────────────────────────────────────────────────────────────────
# STEP 6: Player embeddings (feature vectors per player-season)
# ─────────────────────────────────────────────────────────────────

def build_embeddings():
    out = DATA_DIR / "nba_embeddings.pkl"
    if out.exists():
        print("nba_embeddings.pkl already exists, skipping.")
        with open(out, "rb") as f:
            return pickle.load(f)

    print("Building nba_embeddings.pkl...")
    players_df  = pd.read_csv(DATA_DIR / "nba_headshots.csv")
    stats       = pd.read_csv(DATA_DIR / "nba_player_stats.csv")
    apm         = pd.read_csv(DATA_DIR / "nba_apm.csv")
    lineups     = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")

    # Minutes proxy: count of games appeared in from lineups
    home_cols = [f"home_{i}" for i in range(1, 6)]
    away_cols = [f"away_{i}" for i in range(1, 6)]
    all_pids = pd.concat([
        lineups.melt(id_vars=["year"], value_vars=home_cols+away_cols, value_name="player_id"),
    ])
    games_played = all_pids.groupby(["player_id","year"])["year"].count().reset_index(name="games_played")

    # Roster attributes
    roster = players_df[["PERSON_ID","season","isG","isF","isC","height","weight"]].copy()
    roster["year"] = roster["season"].str[:4].astype(int) + 1
    roster = roster.rename(columns={"PERSON_ID": "player_id"})

    emb = stats.merge(roster, on=["player_id","year"], how="left")
    emb = emb.merge(games_played, on=["player_id","year"], how="left")
    emb = emb.merge(apm[["player_id","APM"]], on="player_id", how="left")
    emb = emb.drop(columns=["season"], errors="ignore")   # drop string column

    # Convert height "6-0" → inches (72), handle missing
    def height_to_inches(h):
        try:
            ft, ins = str(h).split("-")
            return int(ft) * 12 + int(ins)
        except Exception:
            return 0
    if emb["height"].dtype == object:
        emb["height"] = emb["height"].apply(height_to_inches)

    emb = emb.fillna(0)

    # Normalize numeric columns
    skip = {"player_id","year","isG","isF","isC"}
    for col in emb.columns:
        if col not in skip and emb[col].dtype in [np.float64, np.int64, np.float32]:
            std = emb[col].std()
            if std > 0:
                emb[col] = (emb[col] - emb[col].mean()) / std

    with open(out, "wb") as f:
        pickle.dump(emb, f)
    print(f"  Saved embeddings: {emb.shape}")
    return emb


# ─────────────────────────────────────────────────────────────────
# STEP 7: Period matrices for GCN training
# ─────────────────────────────────────────────────────────────────

def build_period_matrices():
    out_mat  = DATA_DIR / "nba_period_matrices.pkl"
    out_pids = DATA_DIR / "nba_period_player_dicts.pkl"
    out_tgt  = DATA_DIR / "nba_targets.pkl"

    if out_mat.exists():
        print("Period matrices already exist, skipping.")
        return

    print("Building period matrices for GCN training...")
    lineups = pd.read_parquet(DATA_DIR / "nba_lineups.parquet")

    home_cols = [f"home_{i}" for i in range(1, 6)]
    away_cols = [f"away_{i}" for i in range(1, 6)]

    matrices, player_dicts, targets = [], [], []

    for _, row in tqdm(lineups.iterrows(), total=len(lineups)):
        home_ids = [row[c] for c in home_cols]
        away_ids = [row[c] for c in away_cols]
        combined = home_ids + away_ids   # indices 0-4 home, 5-9 away

        mat = np.zeros((10, 10))
        d = 1.0   # weight = 1 game

        for i in range(5):
            for j in range(5):
                if i != j:
                    mat[i, j] += d        # home teammates
        for i in range(5, 10):
            for j in range(5, 10):
                if i != j:
                    mat[i, j] += d        # away teammates
        for i in range(5):
            for j in range(5, 10):
                mat[i, j] -= d            # opponents
                mat[j, i] -= d

        matrices.append(pd.DataFrame(mat))
        player_dicts.append({k: combined[k] for k in range(10)})
        targets.append(float(row["target"]))

    with open(out_mat,  "wb") as f: pickle.dump(matrices, f)
    with open(out_pids, "wb") as f: pickle.dump(player_dicts, f)
    with open(out_tgt,  "wb") as f: pickle.dump(targets, f)
    print(f"  Saved {len(matrices)} matrices.")


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Player headshots ===")
    build_headshots()

    print("\n=== Step 2: Games ===")
    build_games()

    print("\n=== Step 3: Player stats ===")
    build_player_stats()

    print("\n=== Step 4: Lineups ===")
    build_lineups()

    print("\n=== Step 5: APM ===")
    build_apm()

    print("\n=== Step 6: Embeddings ===")
    build_embeddings()

    print("\n=== Step 7: Period matrices ===")
    build_period_matrices()

    print("\nDone. All outputs saved to data/")
