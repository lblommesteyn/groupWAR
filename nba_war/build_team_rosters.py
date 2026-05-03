"""
Build nba_team_rosters.csv with per-team top player lists by APM.
Derives player-team assignments from play_by_play (most-frequent team per
player in 2022-23). Joins with nba_apm.csv for rankings.

Output columns: team, player_id, APM, rank (1 = highest APM on team)
"""

import sqlite3
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Top NBA teams in 2022-23 regular season to use as opponent pool
TOP_TEAMS = ["MIL", "BOS", "DEN", "PHI", "MEM", "GSW", "LAL", "PHX",
             "CLE", "NYK", "MIA", "SAC"]


def main():
    con = sqlite3.connect(DATA / "nba.sqlite")

    # Most-frequent team per player in 2022-23
    # game_id prefix 0022200 = regular season 2022-23
    print("Querying play-by-play for player-team assignments...")
    team_counts = pd.read_sql("""
        SELECT player1_id    AS player_id,
               player1_team_abbreviation AS team,
               COUNT(*) AS events
        FROM play_by_play
        WHERE player1_id IS NOT NULL
          AND player1_team_abbreviation IS NOT NULL
          AND game_id LIKE '0022200%'
        GROUP BY player1_id, player1_team_abbreviation
    """, con)
    con.close()

    # For each player, keep team with most events
    team_counts = team_counts.sort_values(["player_id", "events"], ascending=[True, False])
    primary = team_counts.drop_duplicates(subset=["player_id"], keep="first")
    primary["player_id"] = primary["player_id"].astype(int)
    print(f"Derived primary team for {len(primary)} players.")

    # Join with APM data
    apm = pd.read_csv(DATA / "nba_apm.csv")
    apm["player_id"] = apm["player_id"].astype(int)
    merged = primary.merge(apm, on="player_id", how="inner")
    print(f"After APM join: {len(merged)} players with APM data.")

    # Keep only top opponent teams
    merged = merged[merged["team"].isin(TOP_TEAMS)].copy()
    print(f"After team filter: {len(merged)} players across {merged['team'].nunique()} teams.")

    # Rank by APM within each team
    merged = merged.sort_values(["team", "APM"], ascending=[True, False])
    merged["rank"] = merged.groupby("team").cumcount() + 1

    out = merged[["team", "player_id", "APM", "rank"]].reset_index(drop=True)
    out.to_csv(DATA / "nba_team_rosters.csv", index=False)

    print(f"\nWrote {len(out)} rows to nba_team_rosters.csv")
    print("\nTop 10 by team:")
    for team in TOP_TEAMS:
        top10 = out[out["team"] == team].head(10)
        print(f"\n{team}:")
        print(top10[["player_id", "APM", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
