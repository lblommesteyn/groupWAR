"""Estimate per-entity missing episodes, bounded by observations, in continuous feed segments."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import zipfile
from audit import position, number


def analyze(rows):
    hist = {'players': Counter(), 'ball': Counter()}
    censored = Counter()
    states = {}
    prev_t = prev_key = None
    frames = Counter()

    def close():
        for pid, state in states.items():
            if state['missing']:
                censored['ball' if pid == 'ball' else 'players'] += 1
        states.clear()

    for line in rows:
        f = json.loads(line)
        t, period = f.get('timestamp_ms'), f.get('period')
        lineup = f.get('on_court') or {}
        h, a = lineup.get('home') or [], lineup.get('away') or []
        ids = tuple(sorted(str(v) for v in h+a))
        valid = (number(t) and isinstance(period, int) and period > 0 and len(h)==5
                 and len(a)==5 and len(set(ids))==10 and all(v is not None for v in h+a))
        if not valid:
            close(); prev_t = prev_key = None
            frames['excluded_unknown_period_or_lineup'] += 1
            continue
        key = (f.get('game_id'), period, ids)
        continuous = prev_t is not None and key == prev_key and 0 < t-prev_t <= 300
        if not continuous:
            close()
        observed = {str(p['player_id']) for side in ('home_players','away_players','other_players')
                    for p in f.get(side,[]) if p.get('player_id') is not None and position(p)}
        if position(f.get('ball')): observed.add('ball')
        frames['included'] += 1
        for pid in ids+('ball',):
            kind = 'ball' if pid == 'ball' else 'players'
            missing = pid not in observed
            frames[kind+'_missing_samples'] += int(missing)
            if pid not in states:
                states[pid] = {'missing':missing,'start':None}
            else:
                state = states[pid]
                if missing and not state['missing']:
                    state['start'] = (prev_t+t)/2
                elif not missing and state['missing']:
                    if state['start'] is None:
                        censored[kind] += 1
                    else:
                        duration = round((prev_t+t)/2-state['start'])
                        hist[kind][duration] += 1
                    state['start'] = None
                state['missing'] = missing
        prev_t, prev_key = t,key
    close()
    return {'duration_ms_histogram':{k:dict(v) for k,v in hist.items()},
            'censored_episodes':dict(censored), 'frames':dict(frames)}


def work(task):
    path,name = task
    with zipfile.ZipFile(path) as z:
        with gzip.open(z.open(name),'rt') as rows: result = analyze(rows)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',default='results/gaps.json')
    args=p.parse_args()
    with zipfile.ZipFile(args.archive) as z: names=sorted(n for n in z.namelist() if n.endswith('/tracking.jsonl.gz'))
    hist={'players':Counter(),'ball':Counter()};censored=Counter();frames=Counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(work,((args.archive,n) for n in names)),1):
            for k in hist:hist[k].update(r['duration_ms_histogram'][k])
            censored.update(r['censored_episodes']);frames.update(r['frames'])
            if i%50==0 or i==1:print(f'{i}/{len(names)} games',flush=True)
    result={'games':len(names),'duration_ms_histogram':{k:dict(v) for k,v in hist.items()},
            'censored_episodes':dict(censored),'frames':dict(frames),
            'method':'Per-entity absence with observed coordinates on both sides, within fixed lineup/period and source timestamp steps <=300ms. Boundary times are midpoints between adjacent samples. Other episodes are censored and excluded from the histogram. Wall-clock seconds; includes stoppages and source-interpolated observations.'}
    Path(args.output).write_text(json.dumps(result)+'\n'); print('Saved',args.output,flush=True)


if __name__=='__main__':main()
