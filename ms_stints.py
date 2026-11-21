"""Multi-season stints with per-player usage, from groupWAR v3's play-by-play (2019-20 .. 2022-23).

Replays v3/nba_build_graphs lineup reconstruction (its period_starters / clock helpers, unchanged) and records per
stint: lineups, seconds, points, per-team possessions (FGA + 0.44 FTA - OREB + TOV) and per-player usage
[FGA, FTA, TOV]. Event types: 1 made FG, 2 missed FG, 3 FT, 4 rebound, 5 turnover, 8 substitution.
Output (court-model on the PC): results/latent/ms_stints.pkl = {'stints': [...], 'games': {gid: meta}}.
Usage (PC): python ms_stints.py                       (v3 2019-23 -> ms_stints.pkl)
            python ms_stints.py PBP GAMES OUT.pkl     (e.g. fetch_pbp.py's 2023-25 files)
"""
import pickle, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

V3 = Path(r'C:\Users\LukeB\dev\groupWAR\v3')
sys.path.insert(0, str(V3))
from nba_build_graphs import period_starters, clock_to_elapsed, parse_score, SUB  # noqa: E402

OUT = Path(__file__).parent / 'results/latent'


def new_stint(gid, on, h, a, t0, score):
    return dict(game_id=gid, lineup=(tuple(sorted(on[h])), tuple(sorted(on[a]))), t0=t0, t1=t0, score0=score,
                fga=[0, 0], fta=[0, 0], oreb=[0, 0], tov=[0, 0], use={})


def close(st, t1, score):
    st['t1'] = t1
    st['pts'] = [score[1] - st['score0'][1], score[0] - st['score0'][0]]  # score is (visitor, home)
    st['poss_team'] = [st['fga'][k] + 0.44 * st['fta'][k] - st['oreb'][k] + st['tov'][k] for k in (0, 1)]
    return st


def process(gid, ev_all, h, a):
    stints, prev_end, score = [], {}, (0, 0)
    for period, ev in ev_all.groupby('period', sort=True):
        rows = ev.sort_values('eventnum').to_dict('records')
        on = period_starters(rows, [h, a], prev_end)
        if len(on[h]) != 5 or len(on[a]) != 5:
            return None
        base = (period - 1) * 720 if period <= 4 else 2880 + (period - 5) * 300
        st = new_stint(gid, on, h, a, base, score); last_miss = None
        for r in rows:
            t = base + clock_to_elapsed(r['pctimestring'], period)
            sc = parse_score(r['score'])
            if sc is not None:
                score = sc
            et, p1, t1 = r['eventmsgtype'], r['player1_id'], r['player1_team_id']
            side = 0 if t1 == h else 1 if t1 == a else None
            if et == SUB:
                tm, out_p, in_p = r['player1_team_id'], r['player1_id'], r['player2_id']
                if tm in on and out_p in on[tm] and in_p and in_p not in on[tm]:
                    stints.append(close(st, t, score))
                    on[tm] = [in_p if p == out_p else p for p in on[tm]]
                    st = new_stint(gid, on, h, a, t, score)
                continue
            if side is None:
                if et == 4 and r['player1_id'] in (h, a):  # team rebound: player1_id holds the team id
                    side = 0 if r['player1_id'] == h else 1
                elif et == 5 and r['player1_id'] in (h, a):
                    side = 0 if r['player1_id'] == h else 1
                    st['tov'][side] += 1
                    continue
                else:
                    continue
            u = st['use'].setdefault(p1, [0, 0, 0]) if p1 and p1 in on.get(t1, []) else [0, 0, 0]
            if et in (1, 2):
                st['fga'][side] += 1; u[0] += 1
                last_miss = side if et == 2 else None
            elif et == 3:
                st['fta'][side] += 1; u[1] += 1
                desc = (r['homedescription'] if side == 0 else r['visitordescription']) or ''
                last_miss = side if 'MISS' in desc and r['eventmsgactiontype'] in (10, 12, 15) else None
            elif et == 5:
                st['tov'][side] += 1; u[2] += 1
            elif et == 4:
                if last_miss is not None and side == last_miss:
                    st['oreb'][side] += 1
                last_miss = None
        stints.append(close(st, base + (720 if period <= 4 else 300), score))
        prev_end = {tm: list(on[tm]) for tm in on}
    return stints


def main():
    a = sys.argv[1:]
    pbp = pd.read_parquet(a[0] if a else V3 / 'data/nba/pbp.parquet')
    games = pd.read_parquet(a[1] if a else V3 / 'data/nba/games.parquet')
    out_f = OUT / (a[2] if a else 'ms_stints.pkl')
    meta = games.drop_duplicates('game_id').set_index('game_id')
    out, G, fail = [], {}, 0
    for n, (gid, ev) in enumerate(pbp.groupby('game_id', sort=False)):
        if gid not in meta.index:
            fail += 1; continue
        m = meta.loc[gid]
        s = process(gid, ev, int(m['team_id_home']), int(m['team_id_away']))
        if not s:
            fail += 1; continue
        pts_h = sum(x['pts'][0] for x in s); pts_a = sum(x['pts'][1] for x in s)
        if (pts_h, pts_a) != (int(m['pts_home']), int(m['pts_away'])):
            fail += 1; continue  # lineup/score reconstruction does not reconcile
        out += [x for x in s if x['t1'] > x['t0']]
        G[gid] = dict(date=str(m['game_date'])[:10], season=int(ev['season'].iloc[0]), playoff=bool(ev['playoff'].iloc[0]),
                      final=(pts_h, pts_a))
        if n % 500 == 0:
            print(n, len(G), fail, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump({'stints': out, 'games': G}, open(out_f, 'wb'))
    fga = sum(x['fga'][0] + x['fga'][1] for x in out); att = sum(u[0] for x in out for u in x['use'].values())
    print(f'games {len(G)} failed {fail} stints {len(out)} fga {fga} attributed {att / max(fga, 1):.3f} '
          f'poss/game {sum(sum(x["poss_team"]) for x in out) / 2 / max(len(G), 1):.1f}')


if __name__ == '__main__':
    main()
