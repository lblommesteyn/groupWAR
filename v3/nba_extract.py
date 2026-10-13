"""Extract 2019-20 .. 2023-24 NBA play-by-play (regular season + playoffs) from the
Kaggle wyattowalsh/basketball CSV dump into compact parquet.

Usage: python v3/nba_extract.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "nba_war" / "data" / "raw_kaggle"
OUT = ROOT / "v3" / "data" / "nba"
OUT.mkdir(parents=True, exist_ok=True)

# game_id = '00' + type + yy + nnnnn ; type 2 = regular season, 4 = playoffs
SEASON_YY = ["19", "20", "21", "22", "23"]
PREFIXES = tuple(f"00{t}{yy}" for t in "24" for yy in SEASON_YY)

KEEP = [
    "game_id", "eventnum", "eventmsgtype", "eventmsgactiontype", "period",
    "pctimestring", "homedescription", "neutraldescription", "visitordescription",
    "score", "person1type", "player1_id", "player1_team_id",
    "player2_id", "player2_team_id", "player3_id", "player3_team_id",
]


def main():
    parts = []
    reader = pd.read_csv(RAW / "play_by_play.csv", usecols=KEEP, dtype={"game_id": str},
                         chunksize=2_000_000, low_memory=False)
    for i, chunk in enumerate(reader):
        sel = chunk[chunk["game_id"].str.startswith(PREFIXES)]
        if len(sel):
            parts.append(sel)
        print(f"chunk {i}: kept {len(sel):,}", flush=True)
    pbp = pd.concat(parts, ignore_index=True)
    for c in ["player1_id", "player2_id", "player3_id",
              "player1_team_id", "player2_team_id", "player3_team_id"]:
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype("int64")
    pbp["season"] = 2000 + pbp["game_id"].str[3:5].astype(int) + 1  # 2019-20 -> 2020
    pbp["playoff"] = pbp["game_id"].str[2] == "4"
    pbp.to_parquet(OUT / "pbp.parquet", index=False)
    print(pbp.groupby(["season", "playoff"])["game_id"].nunique())

    games = pd.read_csv(RAW / "game.csv", dtype={"game_id": str, "season_id": str})
    games = games[games["game_id"].str.startswith(PREFIXES)]
    games = games[["game_id", "game_date", "season_id", "team_id_home", "team_abbreviation_home",
                   "team_id_away", "team_abbreviation_away", "pts_home", "pts_away"]]
    games.to_parquet(OUT / "games.parquet", index=False)

    cpi = pd.read_csv(RAW / "common_player_info.csv")
    cpi.to_parquet(OUT / "players.parquet", index=False)
    print("games", len(games), "players", len(cpi))


if __name__ == "__main__":
    main()
