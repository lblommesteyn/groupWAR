"""Player bio + draft + combine measurements for the rookie / low-minute prior (ms_spm box features).

LeagueDashPlayerBioStats per season (age that season, height, weight, draft year / round / pick) and
DraftCombineStats per draft (wingspan, standing reach). Everything here is known before a player's first game,
except age, which is taken per season.
Output: results/latent/bio.parquet (player_id, season, age, height, weight, draft_year, draft_pick, undrafted,
        wingspan, reach).   Usage (PC): python fetch_bio.py
"""
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerbiostats, draftcombinestats

from masked_paths import ROOT

OUT = ROOT / 'results/latent'
SEASONS = {f'{y}-{str(y + 1)[2:]}': y + 1 for y in range(2015, 2026)}  # label -> season end year (ms stints 'season')


def num(x):
    return pd.to_numeric(x, errors='coerce')


def main():
    rows = []
    for lab, se in SEASONS.items():
        for st in ('Regular Season', 'Playoffs'):
            d = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=lab, season_type_all_star=st,
                                                                  timeout=60).get_data_frames()[0]
            time.sleep(1.0)
            d['season'] = se
            rows.append(d)
        print(lab, len(rows[-2]), flush=True)
    b = pd.concat(rows).drop_duplicates(['PLAYER_ID', 'season'])
    out = pd.DataFrame(dict(player_id=b.PLAYER_ID.astype(int), season=b.season, age=num(b.AGE),
                            height=num(b.PLAYER_HEIGHT_INCHES), weight=num(b.PLAYER_WEIGHT),
                            draft_year=num(b.DRAFT_YEAR), draft_pick=num(b.DRAFT_NUMBER)))
    out['undrafted'] = out.draft_pick.isna().astype(int)
    comb = []
    for y in range(2000, 2026):
        try:
            c = draftcombinestats.DraftCombineStats(season_all_time=f'{y}-{str(y + 1)[2:]}', timeout=60).get_data_frames()[0]
            comb.append(c[['PLAYER_ID', 'WINGSPAN', 'STANDING_REACH']])
        except Exception as e:  # noqa: BLE001
            print('combine', y, e)
        time.sleep(1.0)
    c = pd.concat(comb).drop_duplicates('PLAYER_ID', keep='last')
    c = c.rename(columns={'PLAYER_ID': 'player_id', 'WINGSPAN': 'wingspan', 'STANDING_REACH': 'reach'})
    out = out.merge(c.astype(float).astype({'player_id': int}), on='player_id', how='left')
    out.to_parquet(OUT / 'bio.parquet')
    print(len(out), 'rows', out.player_id.nunique(), 'players; wingspan known', round(out.wingspan.notna().mean(), 3),
          'draft pick known', round(1 - out.undrafted.mean(), 3))


if __name__ == '__main__':
    main()
