"""Full archive audit with bounded process parallelism and incremental results."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import zipfile
from audit import audit_rows


def work(task):
    path, member = task
    with zipfile.ZipFile(path) as z:
        with gzip.open(z.open(member), 'rt') as rows:
            result = audit_rows(rows)
        coverage = json.loads(z.read(member.replace('tracking.jsonl.gz', 'coverage.json')))
        result['source_coverage'] = coverage
        result['source_count_matches'] = {
            'frames': result['counts']['frames'] == coverage.get('frames'),
            'complete': result['counts']['complete_geometry_frames'] == coverage.get('frames_with_all_ten_and_ball'),
        }
        result['member'] = member
        result['date'] = member.split('/')[-3]
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--output', default='results')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        members = sorted(n for n in z.namelist() if n.endswith('/tracking.jsonl.gz'))
        inventory = {'archive_bytes': Path(args.archive).stat().st_size,
                     'extracted_member_bytes': sum(n.file_size for n in z.infolist()),
                     'members': len(z.infolist()), 'tracking_files': len(members)}
        # Preserve small source documents, without extracting raw datasets.
        for name in ('manifest.json', 'availability.json', 'README.md'):
            member = next(n for n in z.namelist() if n.count('/') == 1 and n.endswith('/' + name))
            (out / ('source_' + name)).write_bytes(z.read(member))
    counts, gaps, runs, windows, players, periods = (Counter() for _ in range(6))
    summaries, decoded = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (out/'games.jsonl').open('w') as f:
        for i, result in enumerate(pool.map(work, ((args.archive,n) for n in members)), 1):
            f.write(json.dumps(result) + '\n'); f.flush()
            counts.update(result['counts']); gaps.update(result['timestamp_gap_histogram'])
            runs.update(result['complete_run_length_histogram']); windows.update(result['nonoverlapping_windows_by_frames'])
            players.update(result['complete_period_frames_by_player']); periods.update(result['frames_by_period'])
            decoded += result['decoded_tracking_bytes']
            summaries.append({k: result[k] for k in ('game_ids','date','complete_geometry_fraction','source_count_matches','nonoverlapping_windows_by_frames')})
            if i % 10 == 0 or i == 1: print(f'{i}/{len(members)} games audited', flush=True)
    report = dict(inventory=inventory, counts=dict(counts), decoded_tracking_bytes=decoded,
                  timestamp_gap_histogram=dict(gaps), complete_run_length_histogram=dict(runs),
                  nonoverlapping_windows_by_frames=dict(windows), complete_period_frames_by_player=dict(players),
                  frames_by_period=dict(periods), games=summaries)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Completed:', out/'summary.json', flush=True)


if __name__ == '__main__': main()
