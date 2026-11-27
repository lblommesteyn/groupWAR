"""DUNCAN score: one row per player from the day's DUNCAN-2 ratings (live_ratings_{date}.npz, written by the morning
live_forecast run).

  O-DUNCAN / D-DUNCAN / DUNCAN   points per 100 possessions vs an average player (offense, defense with + = better,
                                 net = O + D), centred so the season's possession-weighted average player is 0.
  WAR                            wins above replacement so far this season:
                                     (DUNCAN - replacement) x on-court possessions / 100 / points per win
  WAR_ROS, WAR_82                rest-of-season projection (minutes: mean of last 10 games, x the team's remaining
                                 games, on-court possessions per minute from his own stints) and season total.
Replacement level: the possession-weighted mean DUNCAN of players outside the top 13 x 30 by possessions over the
last two seasons (the pool a team can sign at the minimum). Points per win: slope of team win% on average margin
over every team-season in the stint data (fitted, not a rule of thumb).
Output: results/duncan/score_{date}.json and, with --live, {LIVE}/data/duncan/{date}.json
  python duncan_score.py [--date YYYY-MM-DD] [--live C:\\Users\\LukeB\\dev\\court-live]   (PC)
"""
import argparse, json, pickle
from collections import defaultdict
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from masked_paths import ROOT

OUT = ROOT / 'results/latent'
DST = ROOT / 'results/duncan'
SEASON = 2027
FILES = ('ms2_stints_2526.pkl', 'ms2_stints_2627.pkl')


def team_ids():
    from ms_fix import team_ids as t
    return t()


def points_per_win():
    """1 / slope of win% on margin per game, all regular-season team-seasons in the stint data."""
    G = {}
    for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl'):
        G.update(pickle.load(open(OUT / f, 'rb'))['games'])
    TM = team_ids(); acc = defaultdict(lambda: [0, 0, 0.0])
    for g, m in G.items():
        if m['playoff'] or g not in TM or not m.get('final'):
            continue
        h, a = TM[g]; sh, sa = m['final']
        for t, d in ((h, sh - sa), (a, sa - sh)):
            r = acc[(t, m['season'])]; r[0] += 1; r[1] += d > 0; r[2] += d
    v = np.array([(w / n, d / n) for n, w, d in acc.values() if n >= 50])
    k = np.polyfit(v[:, 1], v[:, 0], 1)[0]   # win% per point of margin per game
    return 1.0 / k, len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=date.today().isoformat())
    ap.add_argument('--live')
    a = ap.parse_args()
    z = np.load(OUT / f'live_ratings_{a.date}.npz', allow_pickle=True)
    pids = [int(p) for p in z['pids']]; n = len(pids); pix = {p: i for i, p in enumerate(pids)}
    o, d = z['theta'][:n], -z['theta'][n:]           # defense: + = fewer points allowed
    # on-court possessions per player: this season, and the last two seasons (replacement pool)
    TM = team_ids()
    poss_now, poss_2y, sec_now, team_now = np.zeros(n), np.zeros(n), np.zeros(n), {}
    G = {}
    for f in FILES:
        D = pickle.load(open(OUT / f, 'rb')); G.update(D['games'])
        for s in D['stints']:
            m = G[s['game_id']]
            if m['date'] >= a.date:
                continue
            pp = float(np.mean(s['poss_team'])); se = m['season']
            for k, side in enumerate(s['lineup']):
                for p in side:
                    i = pix.get(p)
                    if i is None:
                        continue
                    if se >= SEASON - 1:
                        poss_2y[i] += pp
                    if se == SEASON:
                        poss_now[i] += pp; sec_now[i] += s['t1'] - s['t0']
                        if s['game_id'] in TM:
                            team_now[p] = (m['date'], TM[s['game_id']][k])
    net = o + d
    w = poss_now / poss_now.sum()
    mo, md = float(w @ o), float(w @ d)
    o, d = o - mo, d - md; net = o + d
    rank = np.argsort(-poss_2y); pool = rank[13 * 30:]; pool = pool[poss_2y[pool] > 0]
    repl = float(np.average(net[pool], weights=poss_2y[pool]))
    ppw, nts = points_per_win()
    war = (net - repl) * poss_now / 100 / ppw
    # rest of season: last-10-game minutes x team games left x his possessions per minute
    gl = pd.read_parquet(OUT / 'gamelogs.parquet')
    gl = gl[(gl.season == SEASON) & ~gl.playoff & (gl.game_date < a.date)].sort_values('game_date')
    abbr = gl.groupby('team_id').team_abbreviation.last().to_dict()
    name = gl.groupby('player_id').player_name.last().to_dict()
    team_games = gl.groupby('team_id').game_id.nunique().to_dict()
    last10 = gl.groupby('player_id').tail(10).groupby('player_id')['min'].mean().to_dict()
    gp = gl.groupby('player_id').game_id.nunique().to_dict()
    ppm = np.where(sec_now > 0, poss_now / np.maximum(sec_now / 60, 1e-9), 0.0)
    lg_ppm = poss_now.sum() / max(sec_now.sum() / 60, 1)
    rows = []
    for p, i in pix.items():
        if poss_now[i] <= 0:
            continue
        t = team_now.get(p, (None, None))[1]
        left = max(82 - team_games.get(t, 0), 0)
        rpm = ppm[i] if sec_now[i] > 600 else lg_ppm
        ros_poss = last10.get(p, 0.0) * left * rpm
        ros = (net[i] - repl) * ros_poss / 100 / ppw
        rows.append(dict(pid=p, name=name.get(p, str(p)), team=abbr.get(t, ''), gp=int(gp.get(p, 0)),
                         poss=round(float(poss_now[i])), o=round(float(o[i]), 2), d=round(float(d[i]), 2),
                         duncan=round(float(net[i]), 2), war=round(float(war[i]), 2),
                         war_ros=round(float(ros), 2), war_82=round(float(war[i] + ros), 2),
                         mpg_last10=round(float(last10.get(p, 0.0)), 1)))
    rows.sort(key=lambda r: -r['war_82'])
    out = dict(date=a.date, ratings_through=str(z['through']), model_version='duncan-2', replacement=round(repl, 2),
               points_per_win=round(ppw, 1), team_seasons=nts, centre=dict(o=round(mo, 3), d=round(md, 3)), players=rows)
    DST.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DST / f'score_{a.date}.json', 'w'), indent=0)
    if a.live:
        dd = Path(a.live) / 'data' / 'duncan'; dd.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dd / f'{a.date}.json', 'w'), indent=0)
    print(f'replacement {repl:+.2f}/100, {ppw:.1f} points per win ({nts} team-seasons), {len(rows)} players')
    for r in rows[:15]:
        print(f"{r['name']:24s} {r['team']:4s} DUNCAN {r['duncan']:+5.1f} (O {r['o']:+.1f} D {r['d']:+.1f})  "
              f"WAR {r['war']:4.1f}  proj {r['war_82']:4.1f}")


if __name__ == '__main__':
    main()
