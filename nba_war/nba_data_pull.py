"""
NBA Data Pull — Phase 1
Pulls play-by-play, lineup rotations, shot charts, and player info
for NBA seasons 2022-2024 from nba_api.
"""

import pandas as pd
import numpy as np
import json
import time
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguegamelog,
    playbyplayv3,
    gamerotation,
    shotchartdetail,
    playerindex,
    commonplayerinfo,
    leaguedashlineups,
)
from nba_api.stats.static import players, teams

# Output directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Rate limit — nba_api gets blocked if you hit it too fast
DELAY = 0.7  # seconds between calls


def get_game_ids(seasons):
    """Get all regular season + playoff game IDs."""
    all_games = []
    for season in seasons:
        print(f"Fetching game IDs for {season}...")
        for season_type in ["Regular Season", "Playoffs"]:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                league_id="00"
            ).get_data_frames()[0]
            log["season"] = season
            log["season_type"] = season_type
            all_games.append(log)
            time.sleep(DELAY)
    games = pd.concat(all_games).drop_duplicates(subset="GAME_ID")
    games.to_parquet(DATA_DIR / "nba_games.parquet", index=False)
    print(f"Saved {len(games)} games.")
    return games["GAME_ID"].tolist()


def pull_pbp(game_ids, checkpoint_file="data/nba_pbp_raw.pkl"):
    """Pull play-by-play for all games with checkpointing."""
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            done = pickle.load(f)
        print(f"Resuming from checkpoint — {len(done)} games already done.")
    else:
        done = {}

    remaining = [g for g in game_ids if g not in done]
    print(f"Pulling PBP for {len(remaining)} games...")

    for game_id in tqdm(remaining):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
            done[game_id] = pbp
            time.sleep(DELAY)
        except Exception as e:
            print(f"Failed {game_id}: {e}")
            time.sleep(2)
            continue

        # Save checkpoint every 100 games
        if len(done) % 100 == 0:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(done, f)

    with open(checkpoint_file, "wb") as f:
        pickle.dump(done, f)

    pbp_all = pd.concat(done.values(), ignore_index=True)
    pbp_all.to_parquet(DATA_DIR / "nba_pbp.parquet", index=False)
    print(f"Saved PBP — {len(pbp_all)} events.")
    return pbp_all


def pull_rotations(game_ids, checkpoint_file="data/nba_rotations_raw.pkl"):
    """Pull lineup rotation (substitution) data for all games."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            done = pickle.load(f)
        print(f"Resuming rotations checkpoint — {len(done)} games done.")
    else:
        done = {}

    remaining = [g for g in game_ids if g not in done]
    print(f"Pulling rotations for {len(remaining)} games...")

    for game_id in tqdm(remaining):
        try:
            rot = gamerotation.GameRotation(game_id=game_id).get_data_frames()
            # Returns [home_rotation, away_rotation]
            home = rot[0].copy()
            away = rot[1].copy()
            home["side"] = "home"
            away["side"] = "away"
            done[game_id] = pd.concat([home, away])
            time.sleep(DELAY)
        except Exception as e:
            print(f"Failed {game_id}: {e}")
            time.sleep(2)
            continue

        if len(done) % 100 == 0:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(done, f)

    with open(checkpoint_file, "wb") as f:
        pickle.dump(done, f)

    rot_all = pd.concat(done.values(), ignore_index=True)
    rot_all.to_parquet(DATA_DIR / "nba_rotations.parquet", index=False)
    print(f"Saved rotations — {len(rot_all)} rows.")
    return rot_all


def pull_shots(seasons):
    """Pull shot chart data (location + outcome) for xPPP model."""
    all_shots = []
    for season in seasons:
        print(f"Pulling shots for {season}...")
        for team in teams.get_teams():
            try:
                shots = shotchartdetail.ShotChartDetail(
                    team_id=team["id"],
                    player_id=0,
                    season_nullable=season,
                    season_type_all_star="Regular Season",
                    context_measure_simple="FGA"
                ).get_data_frames()[0]
                shots["season"] = season
                all_shots.append(shots)
                time.sleep(DELAY)
            except Exception as e:
                print(f"Failed shots {season} {team['abbreviation']}: {e}")
                time.sleep(2)

    shots_df = pd.concat(all_shots, ignore_index=True)
    shots_df.to_parquet(DATA_DIR / "nba_shots.parquet", index=False)
    print(f"Saved {len(shots_df)} shots.")
    return shots_df


def pull_player_info(seasons):
    """Pull player index (bulk) + nationality for all players."""
    all_players = []
    for season in seasons:
        print(f"Pulling player index for {season}...")
        idx = playerindex.PlayerIndex(
            season=season,
            league_id="00"
        ).get_data_frames()[0]
        idx["season"] = season
        all_players.append(idx)
        time.sleep(DELAY)

    players_df = pd.concat(all_players).drop_duplicates(subset=["PERSON_ID", "season"])

    # Pull individual player info for nationality (birthCountry)
    print("Pulling individual player info for nationality...")
    unique_ids = players_df["PERSON_ID"].unique()
    nationality = {}
    checkpoint_nat = "data/nba_nationality_raw.pkl"

    if os.path.exists(checkpoint_nat):
        with open(checkpoint_nat, "rb") as f:
            nationality = pickle.load(f)

    remaining = [pid for pid in unique_ids if pid not in nationality]
    for pid in tqdm(remaining):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            nationality[pid] = {
                "birthCountry": info["COUNTRY"].iloc[0],
                "birthCity": info["BIRTHCITY"].iloc[0],
                "birthDate": info["BIRTHDATE"].iloc[0],
            }
            time.sleep(DELAY)
        except Exception as e:
            print(f"Failed player {pid}: {e}")
            time.sleep(2)

        if len(nationality) % 100 == 0:
            with open(checkpoint_nat, "wb") as f:
                pickle.dump(nationality, f)

    with open(checkpoint_nat, "wb") as f:
        pickle.dump(nationality, f)

    nat_df = pd.DataFrame.from_dict(nationality, orient="index").reset_index()
    nat_df.columns = ["PERSON_ID"] + list(nat_df.columns[1:])
    players_df = players_df.merge(nat_df, on="PERSON_ID", how="left")
    players_df.to_csv(DATA_DIR / "nba_headshots.csv", index=False)
    print(f"Saved {len(players_df)} player records.")
    return players_df


def pull_lineup_stats(seasons):
    """Pull pre-aggregated 5-man lineup stats for APM baseline."""
    all_lineups = []
    for season in seasons:
        print(f"Pulling lineup stats for {season}...")
        lineups = leaguedashlineups.LeagueDashLineups(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_simple="Advanced",
            per_mode_simple="Per100Possessions"
        ).get_data_frames()[0]
        lineups["season"] = season
        all_lineups.append(lineups)
        time.sleep(DELAY)

    lineups_df = pd.concat(all_lineups, ignore_index=True)
    lineups_df.to_csv(DATA_DIR / "nba_lineup_stats.csv", index=False)
    print(f"Saved {len(lineups_df)} lineup records.")
    return lineups_df


if __name__ == "__main__":
    SEASONS = ["2021-22", "2022-23", "2023-24"]

    print("=== Step 1: Game IDs ===")
    game_ids = get_game_ids(SEASONS)

    print("\n=== Step 2: Play-by-play ===")
    pbp = pull_pbp(game_ids)

    print("\n=== Step 3: Lineup rotations ===")
    rotations = pull_rotations(game_ids)

    print("\n=== Step 4: Shot charts ===")
    shots = pull_shots(SEASONS)

    print("\n=== Step 5: Player info + nationality ===")
    players_df = pull_player_info(SEASONS)

    print("\n=== Step 6: Lineup stats ===")
    lineup_stats = pull_lineup_stats(SEASONS)

    print("\nDone. All data saved to data/")
