"""
NBA Data Pull — Chunk 5: Game Rotations (all 3 seasons)
Saves directly to CSV in batches of 500 games to avoid large pkl in memory.
Output: nba_rotations.csv
Estimated size: ~80MB  |  Time: ~2 hours (3 seasons × ~1230 games × 0.7s)

Resume-safe: tracks done game IDs in a small checkpoint file.
"""

import pandas as pd
import pickle
import time
import os
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import gamerotation
from nba_api.stats.library.http import NBAStatsHTTP

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

DELAY = 0.7
BATCH_SIZE = 500
CHECKPOINT = DATA_DIR / "rotations_done_ids.pkl"
OUT_FILE = DATA_DIR / "nba_rotations.csv"


def pull_rotations():
    # Load game IDs
    games = pd.read_parquet(DATA_DIR / "nba_games.parquet")
    game_ids = games["GAME_ID"].tolist()
    print(f"Total games: {len(game_ids)}")

    # Load checkpoint
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            done_ids = pickle.load(f)
        print(f"Resuming — {len(done_ids)} games already done.")
    else:
        done_ids = set()

    # If output exists, we already have some rows — don't rewrite header
    write_header = not OUT_FILE.exists()

    remaining = [g for g in game_ids if g not in done_ids]
    print(f"Pulling rotations for {len(remaining)} remaining games...")

    batch = []
    for game_id in tqdm(remaining):
        try:
            rot = gamerotation.GameRotation(game_id=game_id).get_data_frames()
            home = rot[0].copy()
            away = rot[1].copy()
            home["side"] = "home"
            away["side"] = "away"
            home["GAME_ID"] = game_id
            away["GAME_ID"] = game_id
            batch.append(pd.concat([home, away]))
            done_ids.add(game_id)
            time.sleep(DELAY)
        except Exception as e:
            print(f"  Failed {game_id}: {e}")
            time.sleep(2)
            continue

        if len(batch) >= BATCH_SIZE:
            chunk_df = pd.concat(batch, ignore_index=True)
            chunk_df.to_csv(OUT_FILE, mode="a", header=write_header, index=False)
            write_header = False
            batch = []

            with open(CHECKPOINT, "wb") as f:
                pickle.dump(done_ids, f)
            print(f"  Flushed batch — {len(done_ids)} games done total")

    # Flush remaining
    if batch:
        chunk_df = pd.concat(batch, ignore_index=True)
        chunk_df.to_csv(OUT_FILE, mode="a", header=write_header, index=False)

    with open(CHECKPOINT, "wb") as f:
        pickle.dump(done_ids, f)

    total_rows = sum(1 for _ in open(OUT_FILE)) - 1
    print(f"Saved {total_rows} rotation rows → nba_rotations.csv")


if __name__ == "__main__":
    print("=== Chunk 5: Game rotations ===")
    pull_rotations()

    print("\nChunk 5 complete. File to SCP:")
    print("  data/nba_rotations.csv")
