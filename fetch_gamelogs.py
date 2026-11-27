"""Player game logs (box score per player per game), regular season + playoffs, from NBA stats LeagueGameLog (one
request per season and type). Source of the box-score prior (ms_spm.Data.box_features).

  python fetch_gamelogs.py              full rebuild, 2019-20 .. 2025-26
  python fetch_gamelogs.py 2027         refresh only those seasons (end year) inside the existing file: their rows are
                                        replaced, everything else kept. Run daily for the live season (daily_pc*.bat),
                                        so the live box prior sees this season's games as the backtest's did.
Backtests only read rows dated before each cutoff, so adding later rows changes no past result.
Output: results/latent/gamelogs.parquet   (PC)"""
import sys, time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from masked_paths import ROOT

OUT = ROOT / 'results/latent/gamelogs.parquet'


def fetch(end_year):
    y = end_year - 1; season = f'{y}-{str(end_year)[2:]}'; out = []
    for typ in ('Regular Season', 'Playoffs'):
        df = None
        for attempt in range(4):
            try:
                df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=typ, player_or_team_abbreviation='P',
                                                 timeout=60).get_data_frames()[0]
                break
            except Exception as e:
                print('retry', season, typ, e, flush=True); time.sleep(10)
        if df is None:
            raise SystemExit(f'failed {season} {typ}')
        df.columns = [c.lower() for c in df.columns]
        df['season'] = end_year; df['playoff'] = typ == 'Playoffs'
        if len(df):
            out.append(df)
        print(season, typ, len(df), flush=True); time.sleep(2)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    seasons = [int(a) for a in sys.argv[1:]]
    if not seasons:
        g = pd.concat([fetch(y) for y in range(2020, 2027)], ignore_index=True)
    else:
        old = pd.read_parquet(OUT)
        new = [fetch(y) for y in seasons]
        new = [n for n in new if len(n)]
        g = pd.concat([old[~old.season.isin(seasons)]] + [n[old.columns.intersection(n.columns)] for n in new], ignore_index=True)
    g.to_parquet(OUT); print('saved', len(g), 'rows; last game', g.game_date.max(), flush=True)


if __name__ == '__main__':
    main()
