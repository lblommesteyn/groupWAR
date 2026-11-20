"""Retrospective masked-span benchmark. Run from the court-model directory."""
import gzip
import json
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LENGTHS = (1, 3, 5, 10)


def score_segment(rows, errors):
    if len(rows) < 3:
        return
    times = np.array([t for t, _ in rows])
    xyz = np.stack([coords for _, coords in rows])
    for length in LENGTHS:
        width = length + 2
        count = len(rows) // width
        if not count:
            continue
        paths = xyz[:count*width].reshape(count, width, 11, 3)
        clock = times[:count*width].reshape(count, width)
        elapsed = clock[:, -1] - clock[:, 0]
        regular = np.all(np.abs(np.diff(clock) - 200) <= 25, axis=1)
        alpha = ((clock[:, 1:-1] - clock[:, :1]) / elapsed[:, None])[:, :, None, None]
        estimate = paths[:, :1] + alpha * (paths[:, -1:] - paths[:, :1])
        residual = estimate - paths[:, 1:-1]
        xy_error = np.linalg.norm(residual[:, :, :, :2], axis=-1)
        # Ground-truth path speed is used for evaluation strata only.
        distance = np.linalg.norm(np.diff(paths[:, :, :, :2], axis=1), axis=-1).sum(axis=1)
        moving = distance / (elapsed[:, None] / 1000) >= 3
        valid = np.isfinite(paths).all(axis=(1, 3)) & regular[:, None]
        for label, matrix, mask in (
                ('players', xy_error[:, :, :10], valid[:, :10]),
                ('moving_players', xy_error[:, :, :10], valid[:, :10] & moving[:, :10]),
                ('slower_players', xy_error[:, :, :10], valid[:, :10] & ~moving[:, :10]),
                ('ball_xy', xy_error[:, :, 10:], valid[:, 10:]),
                ('ball_xyz', np.linalg.norm(residual[:, :, 10:], axis=-1), valid[:, 10:])):
            values = matrix.transpose(0, 2, 1)[mask]
            if values.size:
                errors[length, label].append(values.ravel().astype('float32'))


def run_game(task):
    archive, member = task
    errors = defaultdict(list)
    segment = []
    last_key = None
    with zipfile.ZipFile(archive) as z, gzip.open(z.open(member), 'rt') as f:
        for line in f:
            row = json.loads(line)
            sides = row.get('on_court') or {}
            home, away = sides.get('home') or [], sides.get('away') or []
            period = row.get('period')
            ts = row['timestamp_ms']
            valid = (row['phase'] == 'period' and isinstance(period, int) and period > 0
                     and len(home) == len(away) == 5 and len(set(home+away)) == 10
                     and None not in home+away and row['quality'].get('source_interpolated') is not True)
            ids = sorted(home)+sorted(away) if valid else []
            key = (period, tuple(ids))
            if segment and (not valid or key != last_key or not 175 <= ts-segment[-1][0] <= 225):
                score_segment(segment, errors)
                segment = []
            if valid:
                seen = {p['player_id']:p['xyz'] for side in ('home_players','away_players')
                        for p in row[side] if p.get('on_court') and p.get('xyz')}
                missing = [np.nan]*3
                coords = [seen.get(pid,missing) for pid in ids]
                coords.append((row.get('ball') or {}).get('xyz') or missing)
                segment.append((ts, np.asarray(coords, dtype=float)))
            last_key = key
    score_segment(segment, errors)
    return {key:np.concatenate(parts) for key,parts in errors.items()}


def main():
    archive = sys.argv[1]
    screening = json.loads(Path('results/screening.json').read_text())
    candidates = sorted(g['game_id'] for g in screening['candidate_games_without_quarter_warnings']
                        if g['date'] >= '2026-05-01')
    selected = np.random.default_rng(77).choice(candidates, min(24,len(candidates)), replace=False)
    with zipfile.ZipFile(archive) as z:
        files = [n for n in sorted(z.namelist()) if n.endswith('/tracking.jsonl.gz') and n.split('/')[-2] in selected]
    assert len(files) == len(selected)
    pooled = defaultdict(list)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run_game, [(archive,n) for n in files]):
            for key,values in result.items():
                pooled[key].append(values)
    stats = {}
    for (length,label),chunks in sorted(pooled.items()):
        values = np.concatenate(chunks)
        median,p90 = np.percentile(values,[50,90])
        stats.setdefault(str(length/5),{})[label] = dict(windows=values.size//length,
            games=len(chunks), median_ft=float(median), p90_ft=float(p90),
            within_one_foot=float(np.mean(values<=1)))
    report = dict(game_ids=sorted(selected.tolist()), eligible_games=len(candidates), results=stats,
                  method='Seed 77 sample of 24 screened games dated May 1 onward. Mask 1,3,5,10 samples in disjoint blocks per duration per segment. Stable lineup/period and 175–225 ms steps. Exclude explicitly source-interpolated frames. Moving means observed path speed >=3 ft/s. Errors pooled by hidden sample; players XY, ball XY and XYZ. Both endpoints used.')
    Path('results/interpolation_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
