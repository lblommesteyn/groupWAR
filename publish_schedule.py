"""Publish the season schedule into court-live, so the live position capture and the cloud backup forecast know the
games without reaching stats.nba.com (blocked from GitHub's runners) or depending on the PC's forecast file.

Writes {LIVE}/data/schedule/{season}.json: [{gid, date (US Eastern), tip_utc, home_id, away_id, home, away}, ...].
Rewritten only when something changed (postponements, NBA Cup knockout games added).
  python publish_schedule.py [--season 2026-27] [--live C:\\Users\\LukeB\\dev\\court-live]   (PC, daily)
"""
import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', default='2026-27')
    ap.add_argument('--live', default=r'C:\Users\LukeB\dev\court-live')
    a = ap.parse_args()
    from nba_api.stats.endpoints import scheduleleaguev2
    df = None
    for attempt in range(4):
        try:
            df = scheduleleaguev2.ScheduleLeagueV2(season=a.season, league_id='00', timeout=60).get_data_frames()[0]
            break
        except Exception as e:
            print('retry', e, flush=True); time.sleep(15)
    if df is None:
        raise SystemExit('schedule fetch failed')
    rows = []
    for g in df.itertuples():
        gid = str(g.gameId)
        if not gid.startswith(('002', '004', '006')):      # regular season, playoffs, NBA Cup final
            continue
        rows.append(dict(gid=gid, date=str(g.gameDateEst)[:10], tip_utc=str(g.gameDateTimeUTC),
                         home_id=int(g.homeTeam_teamId), away_id=int(g.awayTeam_teamId),
                         home=str(g.homeTeam_teamTricode), away=str(g.awayTeam_teamTricode)))
    rows.sort(key=lambda r: (r['date'], r['gid']))
    f = Path(a.live) / 'data' / 'schedule' / f'{a.season}.json'
    f.parent.mkdir(parents=True, exist_ok=True)
    new = json.dumps(rows, indent=0)
    if not f.exists() or f.read_text() != new:
        f.write_text(new); print('wrote', f, len(rows), 'games')
    else:
        print('schedule unchanged,', len(rows), 'games')


if __name__ == '__main__':
    main()
