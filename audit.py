"""Stream Neilball normalized tracking, using only Python's standard library."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import math
from pathlib import Path
import zipfile


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def position(entity):
    xyz = entity.get('xyz') if isinstance(entity, dict) else None
    return isinstance(xyz, list) and len(xyz) == 3 and all(number(v) for v in xyz)


def roster(frame):
    sides = []
    for side in ('home_players', 'away_players'):
        players = frame.get(side, [])
        if not isinstance(players, list):
            return None
        players = [p for p in players if isinstance(p, dict) and p.get('on_court') is True]
        if len(players) != 5 or any(p.get('player_id') is None or not position(p) for p in players):
            return None
        sides.append(tuple(sorted(str(p['player_id']) for p in players)))
    return tuple(sides) if len(set(sides[0] + sides[1])) == 10 else None


def audit_rows(rows, max_gap_ms=300):
    counts, gaps, flags = Counter(), Counter(), Counter()
    run_lengths, periods, players = Counter(), Counter(), Counter()
    decoded_bytes = 0
    previous = None
    run_frames = longest = 0
    game_ids = set()
    for line in rows:
        decoded_bytes += len(line.encode('utf-8'))
        counts['lines'] += 1
        try:
            frame = json.loads(line)
            if not isinstance(frame, dict):
                raise ValueError('not an object')
        except (ValueError, TypeError):
            counts['invalid_json'] += 1
            if run_frames: run_lengths[run_frames] += 1
            previous, run_frames = None, 0
            continue
        counts['frames'] += 1
        game_ids.add(str(frame.get('game_id', 'missing')))
        for key, value in (frame.get('quality') or {}).items():
            if value is True:
                flags[key] += 1
        lineup = roster(frame)
        complete = lineup is not None and position(frame.get('ball'))
        counts['ball_observed_frames'] += int(position(frame.get('ball')))
        counts['ten_identified_players_frames'] += int(lineup is not None)
        counts['complete_geometry_frames'] += int(complete)
        ts, period, clock = (frame.get(k) for k in ('timestamp_ms', 'period', 'game_clock_ms'))
        valid = complete and number(ts) and isinstance(period, int) and period >= 1
        periods[str(period)] += 1
        counts['labeled_period_frames'] += int(isinstance(period, int) and period >= 1)
        counts['complete_labeled_period_frames'] += int(valid)
        counts['geometry_quality_flag_disagreements'] += int(complete != bool((frame.get('quality') or {}).get('all_ten_and_ball_observed')))
        counts['zero_clock_period_frames'] += int(period is not None and clock == 0)
        if valid:
            for side in lineup:
                for pid in side: players[pid] += 1
        contiguous = False
        if previous and number(ts):
            dt = ts - previous['timestamp_ms']
            if dt <= 0:
                counts['nonincreasing_timestamps'] += 1
            else:
                bucket = '<=250ms' if dt <= 250 else '<=300ms' if dt <= 300 else '<=1s' if dt <= 1000 else '>1s'
                gaps[bucket] += 1
            same_period = period == previous.get('period')
            if same_period and number(clock) and number(previous.get('game_clock_ms')):
                counts['repeated_clock_transitions'] += int(clock == previous['game_clock_ms'])
                counts['clock_increases'] += int(clock > previous['game_clock_ms'])
            contiguous = (0 < dt <= max_gap_ms and same_period
                          and frame.get('game_id') == previous.get('game_id')
                          and lineup is not None and lineup == roster(previous)
                          and position(previous.get('ball')))
        if valid:
            if not contiguous and run_frames: run_lengths[run_frames] += 1
            run_frames = run_frames + 1 if contiguous else 1
            counts['complete_runs'] += int(run_frames == 1)
            longest = max(longest, run_frames)
        else:
            if run_frames: run_lengths[run_frames] += 1
            run_frames = 0
        previous = frame if number(ts) else None
    if run_frames: run_lengths[run_frames] += 1
    return {
        'decoded_tracking_bytes': decoded_bytes,
        'frames_by_period': dict(periods),
        'complete_period_frames_by_player': dict(players),
        'complete_run_length_histogram': dict(run_lengths),
        'nonoverlapping_windows_by_frames': {str(w): sum((n // w)*v for n,v in run_lengths.items()) for w in (11, 26, 51, 121)},
        'game_ids': sorted(game_ids), 'counts': dict(counts),
        'complete_geometry_fraction': counts['complete_geometry_frames'] / max(counts['frames'], 1),
        'timestamp_gap_histogram': dict(gaps), 'true_quality_flags': dict(flags),
        'longest_complete_stable_lineup_run_frames': longest,
        'note': 'Runs are not possessions. Clock repetition alone does not establish stale labels. Event alignment is required.',
    }


@contextmanager
def sources(path):
    path = Path(path)
    if path.is_dir():
        yield [(str(p), lambda p=p: gzip.open(p, 'rt')) for p in sorted(path.rglob('tracking.jsonl.gz'))]
    elif path.name.endswith('.jsonl.gz'):
        yield [(str(path), lambda: gzip.open(path, 'rt'))]
    else:
        with zipfile.ZipFile(path) as archive:
            yield [(n, lambda n=n: gzip.open(archive.open(n), 'rt'))
                   for n in sorted(archive.namelist()) if n.endswith('tracking.jsonl.gz')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Complete ZIP, extracted directory, or tracking.jsonl.gz')
    parser.add_argument('--output', default='audit.json')
    parser.add_argument('--limit-games', type=int)
    args = parser.parse_args()
    report = {'schema_version': 1, 'source': str(Path(args.source).resolve()), 'games': [], 'errors': []}
    try:
        with sources(args.source) as files:
            report['tracking_files_available'] = len(files)
            if not files:
                parser.error('No normalized tracking files found')
            for name, opener in files[:args.limit_games]:
                try:
                    with opener() as rows:
                        result = audit_rows(rows)
                    result['member'] = name
                    report['games'].append(result)
                except (OSError, EOFError, ValueError, AttributeError) as exc:
                    report['errors'].append({'member': name, 'error': str(exc)})
                print('Audited:', name, flush=True)
    except zipfile.BadZipFile:
        parser.error('Not a complete valid ZIP. Wait for the download to finish.')
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('Report:', args.output)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
