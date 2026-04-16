"""
NBA Data Pull — Chunk 1: Metadata
Pulls player info, nationality, lineup stats, and game IDs.
Output: nba_headshots.csv, nba_lineup_stats.csv, nba_games.parquet
Estimated size: ~20MB  |  Time: ~15 min
"""

import pandas as pd
import pickle
import time
import os
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguegamelog,
    playerindex,
    commonplayerinfo,
    leaguedashlineups,
)
from nba_api.stats.library.http import NBAStatsHTTP

# stats.nba.com blocks requests without browser-like headers
NBAStatsHTTP.headers = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASONS = ["2021-22", "2022-23", "2023-24"]
DELAY = 1.0
TIMEOUT = 90  # stats.nba.com is slow; 30s default times out


def with_retry(fn, retries=5, wait=10):
    """Call fn(), retry on timeout/connection error."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt+1} failed ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
                wait *= 2
            else:
                raise


# ── Game IDs ────────────────────────────────────────────────────────────────
def pull_game_ids():
    out = DATA_DIR / "nba_games.parquet"
    if out.exists():
        print("nba_games.parquet already exists, skipping.")
        return

    all_games = []
    for season in SEASONS:
        for season_type in ["Regular Season", "Playoffs"]:
            print(f"  {season} {season_type}...")
            log = with_retry(lambda s=season, st=season_type: leaguegamelog.LeagueGameLog(
                season=s,
                season_type_all_star=st,
                league_id="00",
                timeout=TIMEOUT
            ).get_data_frames()[0])
            log["season"] = season
            log["season_type"] = season_type
            all_games.append(log)
            time.sleep(DELAY)

    games = pd.concat(all_games).drop_duplicates(subset="GAME_ID")
    games.to_parquet(out, index=False)
    print(f"Saved {len(games)} games → nba_games.parquet")


# ── Lineup stats ─────────────────────────────────────────────────────────────
def pull_lineup_stats():
    out = DATA_DIR / "nba_lineup_stats.csv"
    if out.exists():
        print("nba_lineup_stats.csv already exists, skipping.")
        return

    all_lineups = []
    for season in SEASONS:
        print(f"  Lineup stats {season}...")
        lineups = with_retry(lambda s=season: leaguedashlineups.LeagueDashLineups(
            season=s,
            season_type_all_star="Regular Season",
            measure_type_simple="Advanced",
            per_mode_simple="Per100Possessions",
            timeout=TIMEOUT
        ).get_data_frames()[0])
        lineups["season"] = season
        all_lineups.append(lineups)
        time.sleep(DELAY)

    df = pd.concat(all_lineups, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} lineup records → nba_lineup_stats.csv")


# ── Player index + nationality ───────────────────────────────────────────────
def pull_player_info():
    out = DATA_DIR / "nba_headshots.csv"
    checkpoint = DATA_DIR / "nba_nationality_raw.pkl"

    # Step 1: player index (bulk)
    all_players = []
    for season in SEASONS:
        print(f"  Player index {season}...")
        idx = with_retry(lambda s=season: playerindex.PlayerIndex(
            season=s, league_id="00", timeout=TIMEOUT
        ).get_data_frames()[0])
        idx["season"] = season
        all_players.append(idx)
        time.sleep(DELAY)

    players_df = pd.concat(all_players).drop_duplicates(subset=["PERSON_ID", "season"])
    unique_ids = players_df["PERSON_ID"].unique()
    print(f"  {len(unique_ids)} unique players")

    # Step 2: nationality via commonplayerinfo
    if checkpoint.exists():
        with open(checkpoint, "rb") as f:
            nationality = pickle.load(f)
        print(f"  Resuming nationality — {len(nationality)} already done")
    else:
        nationality = {}

    remaining = [pid for pid in unique_ids if pid not in nationality]
    print(f"  Pulling nationality for {len(remaining)} players...")

    for i, pid in enumerate(tqdm(remaining)):
        try:
            info = with_retry(lambda p=pid: commonplayerinfo.CommonPlayerInfo(
                player_id=p, timeout=TIMEOUT
            ).get_data_frames()[0])
            nationality[pid] = {
                "birthCountry": info["COUNTRY"].iloc[0],
                "birthCity":    info["BIRTHCITY"].iloc[0],
                "birthDate":    info["BIRTHDATE"].iloc[0],
            }
            time.sleep(DELAY)
        except Exception as e:
            print(f"  Failed player {pid}: {e}")
            time.sleep(2)

        if (i + 1) % 100 == 0:
            with open(checkpoint, "wb") as f:
                pickle.dump(nationality, f)

    with open(checkpoint, "wb") as f:
        pickle.dump(nationality, f)

    nat_df = pd.DataFrame.from_dict(nationality, orient="index").reset_index()
    nat_df.columns = ["PERSON_ID"] + list(nat_df.columns[1:])

    players_df = players_df.merge(nat_df, on="PERSON_ID", how="left")
    players_df.to_csv(out, index=False)
    print(f"Saved {len(players_df)} player records → nba_headshots.csv")


if __name__ == "__main__":
    print("=== Chunk 1: Game IDs ===")
    pull_game_ids()

    print("\n=== Chunk 1: Lineup stats ===")
    pull_lineup_stats()

    print("\n=== Chunk 1: Player info + nationality ===")
    pull_player_info()

    print("\nChunk 1 complete. Files to SCP:")
    print("  data/nba_games.parquet")
    print("  data/nba_lineup_stats.csv")
    print("  data/nba_headshots.csv")
