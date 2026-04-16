"""
NBA Data Pull — Chunk 4: Shot Charts (2023-24)
Output: nba_shots_202324.parquet
Estimated size: ~150MB  |  Time: ~30 min
"""

import pandas as pd
import time
from pathlib import Path

from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
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

SEASON = "2023-24"
DELAY = 0.7


def pull_shots(season):
    out = DATA_DIR / f"nba_shots_{season.replace('-', '')}.parquet"
    if out.exists():
        print(f"{out.name} already exists, skipping.")
        return

    all_shots = []
    all_teams = teams.get_teams()
    print(f"Pulling shots for {season} — {len(all_teams)} teams...")

    for team in all_teams:
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
            print(f"  {team['abbreviation']}: {len(shots)} shots")
            time.sleep(DELAY)
        except Exception as e:
            print(f"  Failed {team['abbreviation']}: {e}")
            time.sleep(2)

    df = pd.concat(all_shots, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} shots → {out.name}")


if __name__ == "__main__":
    print(f"=== Chunk 4: Shot charts {SEASON} ===")
    pull_shots(SEASON)

    print("\nChunk 4 complete. File to SCP:")
    print(f"  data/nba_shots_{SEASON.replace('-', '')}.parquet")
