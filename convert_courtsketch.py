"""Convert the 2026-27 CourtSketch download (Codex, 247 games) into the 2025-26 Game Pulse archive layout, so the
tracking pipeline (shot_align -> ... -> metrics2) runs on both seasons unchanged.

Input  ~/Downloads/nba-tracking-extension-2026-27/{courtsketch-tracking/{gid}.jsonl.gz, nba-stats/{gid}.json.gz,
       game-catalog.json}. CourtSketch frames: integer inches, ball [x, y, z, holder], players {pid: [x, y, z]}.
Output a zip with  ROOT/games/{date}/{gid}/tracking.jsonl.gz  (schema nba-gamepulse-tracking/1.0 fields the pipeline
       reads: timestamp_ms, period, phase, game_clock_ms, shot_clock_ms, home_players / away_players [{player_id,
       team_id, xyz (feet), on_court}], ball {xyz}, on_court {home, away} from Lineup#show, last_touch, quality),
       metadata.json, coverage.json (periods with no coordinates -> rapm.py coverage warning), raw/stats.jsonl.gz
       (the Ease Live event list, one per line, as in 2025-26).
Then run shot_align, shot_features, player_games (XFG_SPLIT=2099-01-01), tracking_metrics with
TRK_ARCHIVE=<this zip> in a SEPARATE copy of the repo (real files, not symlinks: results/ goes next to the code).
  python convert_courtsketch.py [SRC] [OUT.zip] [--append-to EXISTING.zip]
"""
import bisect, gzip, io, json, sys, zipfile
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Downloads/nba-tracking-extension-2026-27'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / 'Downloads/NBA-2026-27-CourtSketch-Tracking.zip'
PREFIX = 'NBA-2026-27-CourtSketch/games'
IN = 12.0


def timeline(ev, kind):
    rows = sorted((e['timestamp'], e['metadata']) for e in ev if e.get('event') == kind and e.get('metadata'))
    return [r[0] for r in rows], [r[1] for r in rows]


def at(tl, t):
    i = bisect.bisect_right(tl[0], t) - 1
    return (tl[0][i], tl[1][i]) if i >= 0 else (None, None)


def convert(gid, game):
    ev = json.load(gzip.open(SRC / 'nba-stats' / f'{gid}.json.gz', 'rt'))
    info = next((e['metadata'] for e in ev if e.get('event') == 'GameInfo#show'), {})
    home_id, away_id = info.get('homeId'), info.get('awayId')
    lineups = timeline(ev, 'Lineup#show'); clock = timeline(ev, 'Clock#show'); shot = timeline(ev, 'ShotClock#show')
    side = {}
    for m in lineups[1]:
        for s in ('home', 'away'):
            for p in m.get(s) or []:
                side[int(p)] = s
    for e in ev:
        if e.get('event') == 'Teams#show':
            for s in ('home', 'away'):
                for p in (e['metadata'].get(s) or {}).get('players') or []:
                    side.setdefault(int(p['personId']), s)
    team = {'home': home_id, 'away': away_id}
    out = io.BytesIO(); n = full = 0; nonempty = {}
    with gzip.open(out, 'wt', compresslevel=6) as w:
        for line in gzip.open(SRC / 'courtsketch-tracking' / f'{gid}.jsonl.gz', 'rt'):
            f = json.loads(line); t = f['timestamp_ms']
            _, lu = at(lineups, t); lu = lu or {}
            on = {'home': [int(p) for p in lu.get('home') or []], 'away': [int(p) for p in lu.get('away') or []]}
            onset = set(on['home'] + on['away'])
            ct, cm = at(clock, t)
            gc = None
            if cm is not None:
                gc = cm.get('value')
                if gc is not None and cm.get('running'):
                    gc = max(0, gc - (t - ct))
            st, sm = at(shot, t)
            sc = sm.get('value') if sm else None
            if sc is not None and cm is not None and cm.get('running'):
                sc = max(0, sc - (t - st))
            fs = {'home': [], 'away': []}
            for pid, v in f['players'].items():
                if v is not None and side.get(int(pid)):
                    fs[side[int(pid)]].append(int(pid))
            if len(fs['home']) == 5 and len(fs['away']) == 5:  # tracking sees exactly ten: trust it over lagging Lineup events
                on = {'home': sorted(fs['home']), 'away': sorted(fs['away'])}; onset = set(on['home'] + on['away'])
            hp, ap = [], []
            for pid, v in f['players'].items():
                p = int(pid); s = side.get(p)
                if s is None or v is None:
                    continue
                (hp if s == 'home' else ap).append({'player_id': p, 'team_id': team[s], 'xyz': [v[0] / IN, v[1] / IN, v[2] / IN],
                                                    'on_court': p in onset})
            b = f.get('ball')
            ball = {'xyz': [b[0] / IN, b[1] / IN, b[2] / IN]} if b else None
            observed = sum(1 for x in hp + ap if x['on_court'])
            rec = {'schema_version': 'nba-gamepulse-tracking/1.0', 'game_id': gid, 'timestamp_ms': t, 'units': 'feet',
                   'period': f['period'], 'phase': 'period', 'game_clock_ms': gc, 'shot_clock_ms': sc,
                   'home_players': hp, 'away_players': ap, 'other_players': [], 'ball': ball, 'on_court': on,
                   'last_touch': int(b[3]) if b and len(b) > 3 and b[3] else None,
                   'quality': {'observed_players': observed, 'lineup_known': len(onset) == 10,
                               'all_ten_and_ball_observed': observed == 10 and ball is not None,
                               'source_interpolated': None, 'clock_interpolated': True, 'source': 'courtsketch'}}
            full += rec['quality']['all_ten_and_ball_observed']; n += 1
            nonempty[f['period']] = nonempty.get(f['period'], 0) + bool(hp or ap)
            w.write(json.dumps(rec, separators=(',', ':')) + '\n')
    stats = io.BytesIO()
    with gzip.open(stats, 'wt', compresslevel=6) as w:
        for e in sorted(ev, key=lambda e: e.get('timestamp') or 0):
            w.write(json.dumps(e, separators=(',', ':')) + '\n')
    meta = {'schema_version': 'nba-gamepulse-tracking/1.0', 'game_id': gid, 'game_date': game['date'], 'source': 'courtsketch',
            'game': {'gameId': gid, 'gameCode': f"{game['date'].replace('-', '')}/{game['away']}{game['home']}",
                     'homeTeam': {'teamId': home_id, 'teamTricode': game['home'], 'score': game['home_score']},
                     'awayTeam': {'teamId': away_id, 'teamTricode': game['away'], 'score': game['away_score']}},
            'tracking_gaps': game.get('tracking_gaps', [])}
    periods = sorted({e['metadata']['period'] for e in ev if e.get('event') == 'PlayByPlay#show'
                      and isinstance((e.get('metadata') or {}).get('period'), int) and e['metadata']['period'] > 0})
    cov = {'schema_version': 'nba-gamepulse-tracking/1.0', 'game_id': gid, 'source': 'courtsketch', 'frames': n,
           'nonempty_frames_by_period_or_phase': {str(k): v for k, v in nonempty.items()},
           'periods_without_coordinate_observations': [p for p in periods if not nonempty.get(p)],
           'tracking_gaps': game.get('tracking_gaps', [])}
    return out.getvalue(), stats.getvalue(), json.dumps(meta, indent=1).encode(), json.dumps(cov, indent=1).encode(), n, full


def main():
    cat = {g['id']: g for g in json.load(open(SRC / 'game-catalog.json'))}
    gids = sorted(g for g in cat if (SRC / 'courtsketch-tracking' / f'{g}.jsonl.gz').exists()
                  and (SRC / 'nba-stats' / f'{g}.json.gz').exists())
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED) as z:
        for k, gid in enumerate(gids):
            tr, st, meta, cov, n, full = convert(gid, cat[gid])
            base = f"{PREFIX}/{cat[gid]['date']}/{gid}/"
            z.writestr(base + 'tracking.jsonl.gz', tr); z.writestr(base + 'raw/stats.jsonl.gz', st)
            z.writestr(base + 'metadata.json', meta); z.writestr(base + 'coverage.json', cov)
            if k % 25 == 0:
                print(k, gid, n, 'frames', full, 'full', flush=True)
    print('wrote', OUT, len(gids), 'games')


if __name__ == '__main__':
    main()
