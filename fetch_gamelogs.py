"""Player game logs (box score per player per game) for 2019-20 .. 2025-26, regular season + playoffs, from NBA stats
LeagueGameLog (one request per season and type). Source of the box-score prior in ms_bench.py.
Output: results/latent/gamelogs.parquet   (PC: python fetch_gamelogs.py)"""
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from masked_paths import ROOT

OUT = ROOT / 'results/latent/gamelogs.parquet'
frames = []
for y in range(2019, 2026):
    season = f'{y}-{str(y + 1)[2:]}'
    for typ in ('Regular Season', 'Playoffs'):
        for attempt in range(4):
            try:
                df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=typ, player_or_team_abbreviation='P',
                                                 timeout=60).get_data_frames()[0]
                break
            except Exception as e:
                print('retry', season, typ, e, flush=True); time.sleep(10)
        df['season'] = y + 1; df['playoff'] = typ == 'Playoffs'
        frames.append(df); print(season, typ, len(df), flush=True); time.sleep(2)
g = pd.concat(frames, ignore_index=True)
g.columns = [c.lower() for c in g.columns]
g.to_parquet(OUT); print('saved', len(g), g.columns.tolist())
