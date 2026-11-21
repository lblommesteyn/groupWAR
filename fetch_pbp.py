"""Fetch 2023-24 and 2024-25 play-by-play (NBA stats PlayByPlayV3 via nba_api) and convert it to the v2-style rows
that v3's lineup builder and ms_stints.py read. PlayByPlayV2 no longer returns data, hence the conversion.

V3 -> v2 mapping: actionType -> eventmsgtype (Made Shot 1, Missed Shot 2, Free Throw 3, Rebound 4, Turnover 5,
Foul 6, Violation 7, Substitution 8, Timeout 9, Jump Ball 10, Ejection 11, period start/end 12/13, Instant Replay 18);
free-throw subType "k of n" -> v2 action codes (1 of 1: 10, 1/2 of 2: 11/12, 1/2/3 of 3: 13/14/15, technical 16);
technical fouls -> action 11 (a TECH_ACTIONS code). Substitution rows carry the OUTGOING player's personId and only
the incoming player's last name ("SUB: Niang FOR Allen"); the incoming id is resolved from that team's players
in the same game (playerName / playerNameI). Games where a name is ambiguous or missing are kept but flagged;
ms_stints drops any game whose lineup/score reconstruction does not reconcile.
Cache: results/latent/pbp_cache/{game_id}.json (resumable).
Output: results/latent/pbp_2325.parquet, games_2325.parquet.   Usage (PC): python fetch_pbp.py
"""
import json, re, time
from pathlib import Path
import pandas as pd

from masked_paths import ROOT

OUT = ROOT / 'results/latent'
CACHE = OUT / 'pbp_cache'
import os
# SEASONS='2025-26:2026' TAG=2526 DATE_MAX=2026-01-19 python fetch_pbp.py  -> the current season before tracking starts
SEASONS = dict((k, int(v)) for k, v in (x.split(':') for x in os.environ.get('SEASONS', '2023-24:2024,2024-25:2025').split(',')))
TAG = os.environ.get('TAG', '2325')
DATE_MAX = os.environ.get('DATE_MAX')
TYPE = {'Made Shot': 1, 'Missed Shot': 2, 'Free Throw': 3, 'Rebound': 4, 'Turnover': 5, 'Foul': 6, 'Violation': 7,
        'Substitution': 8, 'Timeout': 9, 'Jump Ball': 10, 'Ejection': 11, 'Instant Replay': 18}
FT = {('1', '1'): 10, ('1', '2'): 11, ('2', '2'): 12, ('1', '3'): 13, ('2', '3'): 14, ('3', '3'): 15}


def games():
    from nba_api.stats.endpoints import leaguegamefinder
    rows = []
    for s, sn in SEASONS.items():
        for st in ('Regular Season', 'Playoffs'):
            d = leaguegamefinder.LeagueGameFinder(season_nullable=s, league_id_nullable='00', season_type_nullable=st,
                                                  timeout=60).get_data_frames()[0]
            time.sleep(1.0)
            for gid, g in d.groupby('GAME_ID'):
                if len(g) != 2:
                    continue
                h = g[g['MATCHUP'].str.contains('vs.')]; a = g[g['MATCHUP'].str.contains('@')]
                if len(h) != 1 or len(a) != 1:
                    continue
                rows.append(dict(game_id=gid, game_date=h['GAME_DATE'].iloc[0], season=sn, playoff=st == 'Playoffs',
                                 team_id_home=int(h['TEAM_ID'].iloc[0]), team_id_away=int(a['TEAM_ID'].iloc[0]),
                                 pts_home=int(h['PTS'].iloc[0]), pts_away=int(a['PTS'].iloc[0])))
    return pd.DataFrame(rows)


def fetch(gid):
    f = CACHE / f'{gid}.json'
    if f.exists():
        return json.load(open(f))
    from nba_api.stats.endpoints import playbyplayv3
    for k in range(4):
        try:
            d = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=60).get_data_frames()[0]
            recs = d.to_dict('records')
            json.dump(recs, open(f, 'w'))
            return recs
        except Exception as e:
            print(f'  {gid} retry {k}: {repr(e)[:120]}', flush=True); time.sleep(5 * (k + 1))
    return None


def clock(s):
    m = re.match(r'PT(\d+)M([\d.]+)S', s or '')
    return f'{int(m.group(1))}:{float(m.group(2)):05.2f}' if m else '0:00'


SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def norm(name):
    """'Vučević' -> 'vucevic', 'Butler III' -> 'butler', 'G. Antetokounmpo' -> 'g antetokounmpo'"""
    import unicodedata
    s = unicodedata.normalize('NFKD', name or '').encode('ascii', 'ignore').decode().lower().replace('.', ' ').replace("'", '')
    t = [w for w in s.split() if w not in SUFFIX]
    return ' '.join(t)


_FULL = None
UNRESOLVED = []  # (game, description, candidates) for the log


def full_names():
    global _FULL
    if _FULL is None:
        from nba_api.stats.static import players
        _FULL = {int(q['id']): q['full_name'] for q in players.get_players()}
    return _FULL


def name_keys(pid, r=None):
    """keys a substitution description may use for this person, all normalised: last name, initial + last,
    any first-name prefix + last ('Ja. Green', 'Jay. Williams'), and the play-by-play's own playerName / playerNameI."""
    ks = set()
    full = full_names().get(pid)
    if full:
        w = full.split(); first, last = w[0], ' '.join(w[1:])
        ks |= {norm(last)} | {norm(first[:k] + ' ' + last) for k in range(1, len(first) + 1)}  # 'J.', 'Ja.', 'Jay.', full
    if r is not None:
        ks |= {norm(r.get('playerName')), norm(r.get('playerNameI'))}
    return {k for k in ks if k}


def has_suffix(name):
    import unicodedata
    s = unicodedata.normalize('NFKD', name or '').encode('ascii', 'ignore').decode().lower().replace('.', ' ')
    return any(w in SUFFIX for w in s.split())


def name_map(recs, teams):
    names = {}  # (team, normalised name) -> set of ids
    for r in recs:
        if r['personId'] and r['teamId'] in teams:
            pid = int(r['personId'])
            for k in name_keys(pid, r):
                names.setdefault((int(r['teamId']), k), set()).add(pid)
    return names


def resolve(desc_name, cand):
    """several candidates: use the suffix ('Williams Jr.' vs 'Williams') to choose"""
    if len(cand) <= 1:
        return cand
    want = has_suffix(desc_name)
    c2 = {p for p in cand if has_suffix(full_names().get(p, '')) == want}
    return c2 if c2 else cand


def convert(recs, gid, home, away, season, playoff, season_names):
    names = name_map(recs, (home, away))
    out, bad = [], 0
    for r in recs:
        at = r['actionType']
        if at == 'period':
            et = 12 if r['subType'] == 'start' else 13
        else:
            et = TYPE.get(at, 0)
        if et == 0 and not r['personId']:
            continue
        team, pid = int(r['teamId'] or 0), int(r['personId'] or 0)
        act = 0
        if et == 3:
            m = re.search(r'(\d) of (\d)', r['subType'] or '')
            act = FT.get(m.groups(), 0) if m else (16 if 'Technical' in (r['subType'] or '') else 0)
        elif et == 6 and 'Technical' in (r['subType'] or '') or et == 6 and 'Technical' in (r['description'] or ''):
            act = 11
        p2 = 0
        if et == 8:
            m = re.match(r'SUB:\s*(.+?)\s+FOR\s+(.+)$', r['description'] or '')
            key = (team, norm(m.group(1))) if m else None
            cand = resolve(m.group(1), names.get(key, set()) - {pid}) if m else set()
            if len(cand) != 1 and key:  # no other event in this game: that team's players over the season
                cand = resolve(m.group(1), season_names.get(key, set()) - {pid})
            if len(cand) == 1:
                p2 = cand.pop()
            else:
                bad += 1; UNRESOLVED.append((gid, r['description'], sorted(cand)))
        desc = r['description'] or ''
        if et == 3 and r.get('shotResult') == 'Missed' and 'MISS' not in desc:
            desc = 'MISS ' + desc
        score = f"{r['scoreAway']} - {r['scoreHome']}" if str(r.get('scoreHome') or '').strip() and str(r.get('scoreAway') or '').strip() else None
        out.append(dict(game_id=gid, eventnum=int(r['actionNumber']), eventmsgtype=et, eventmsgactiontype=act,
                        period=int(r['period']), pctimestring=clock(r['clock']),
                        homedescription=desc if team == home else None, neutraldescription=None,
                        visitordescription=desc if team == away else None, score=score, person1type=None,
                        player1_id=pid, player1_team_id=team if team else None, player2_id=p2,
                        player2_team_id=team if p2 else None, player3_id=0, player3_team_id=None,
                        season=season, playoff=playoff))
    return out, bad


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    gf = OUT / f'games_{TAG}.parquet'
    G = pd.read_parquet(gf) if gf.exists() else games()
    if DATE_MAX:
        G = G[G['game_date'] < DATE_MAX]
    G.to_parquet(gf)
    print(f'games {len(G)}', flush=True)
    rows, bad_games, miss = [], 0, 0
    t0 = time.time(); R = {}
    for n, g in enumerate(G.itertuples()):
        cached = (CACHE / f'{g.game_id}.json').exists()
        R[g.game_id] = fetch(g.game_id)
        if not cached:
            time.sleep(0.7)
        if n % 100 == 0:
            print(f'{n}/{len(G)} games fetched, {time.time() - t0:.0f}s', flush=True)
    season_names = {}
    for g in G.itertuples():
        for k, v in name_map(R[g.game_id] or [], (g.team_id_home, g.team_id_away)).items():
            season_names.setdefault((g.season,) + k, set()).update(v)
    for g in G.itertuples():
        recs = R[g.game_id]
        if recs is None:
            miss += 1; continue
        sn = {k[1:]: v for k, v in season_names.items() if k[0] == g.season and k[1] in (g.team_id_home, g.team_id_away)}
        conv, bad = convert(recs, g.game_id, g.team_id_home, g.team_id_away, g.season, g.playoff, sn)
        rows += conv; bad_games += bad > 0
    pd.DataFrame(rows).to_parquet(OUT / f'pbp_{TAG}.parquet')
    print(f'done: {len(G) - miss} games, {len(rows)} rows, games with an unresolved sub {bad_games} ({len(UNRESOLVED)} subs), failed {miss}')


if __name__ == '__main__':
    main()
