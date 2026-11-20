"""Audit 26-frame windows and compare fixed ridge forecasters on later games."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
from audit import position

WIDTH = 26
ORIGIN = 20
OUTPUT = Path(__file__).parent / 'results' / 'clip_experiment'


def pack(mask, boundary):
    """Greedy disjoint windows in each uninterrupted valid run."""
    start = np.flatnonzero(mask & (boundary | ~np.r_[False, mask[:-1]]))
    end = np.flatnonzero(mask & (np.r_[boundary[1:], True] | ~np.r_[mask[1:], False])) + 1
    groups = [np.arange(a, b-WIDTH+1, WIDTH) for a,b in zip(start,end) if b-a >= WIDTH]
    return np.concatenate(groups) if groups else np.empty(0,dtype=int)


def fill_short(x, times, boundary):
    filled = x.copy()
    segment = np.cumsum(boundary)
    right_anchor = np.zeros(x.shape[:2], dtype=int)
    for player in range(10):
        observed = np.flatnonzero(np.isfinite(x[:,player,0]))
        left, right = observed[:-1], observed[1:]
        candidate = (right-left > 1) & (segment[left] == segment[right])
        left, right = left[candidate], right[candidate]
        duration = (times[right]+times[right-1]-times[left]-times[left+1])/2
        for a,b in zip(left[duration<=600],right[duration<=600]):
            weight = ((times[a+1:b]-times[a])/(times[b]-times[a]))[:,None]
            filled[a+1:b,player] = x[a,player]+weight*(x[b,player]-x[a,player])
            right_anchor[a+1:b,player] = b
    return filled,right_anchor


def features(x, times, starts):
    origin = starts+ORIGIN
    p = x[origin,:10]
    columns = [np.ones((*p.shape[:2],1)), p/50]
    for lag in (1,3,5,10):
        dt = (times[origin]-times[origin-lag])/1000
        columns.append((p-x[origin-lag,:10])/dt[:,None,None]/10)
    ball = x[origin,10:11]
    columns.append((ball-p)/30)
    teams = p.reshape(-1,2,5,2).mean(axis=2)
    for centers in (teams,teams[:,::-1]):
        columns.append((centers.repeat(5,axis=1)-p)/30)
    return np.concatenate(columns,axis=2)


def statistics(x, raw, anchor, times, starts):
    # Evaluate roughly one second ahead. Every target and origin must be observed.
    elapsed = times[starts+25]-times[starts+20]
    starts = starts[(elapsed>=900)&(elapsed<=1100)]
    f = features(x,times,starts)
    d = raw[starts+25,:10]-raw[starts+20,:10]
    anchored = anchor[starts[:,None]+np.array([10,15,17,19,20]),:10]
    safe = (anchored <= (starts+20)[:,None,None]).all(axis=(1,2))
    valid = np.isfinite(d).all(axis=2) & safe[:,None]
    f,d = f[valid],d[valid]
    return (f.T@f,f.T@d,np.sum(d*d),len(d))


def work(task):
    archive, name = task
    times,positions,boundaries = [],[],[]
    previous = None
    with zipfile.ZipFile(archive) as z, gzip.open(z.open(name),'rt') as rows:
        for line in rows:
            f=json.loads(line)
            lineup=f.get('on_court') or {}
            h,a=lineup.get('home') or [],lineup.get('away') or []
            ts=f['timestamp_ms']
            if not (f.get('period') and len(h)==len(a)==5 and len(set(h+a))==10 and None not in h+a):
                previous=None
                continue
            ids=tuple(sorted(h)+sorted(a))
            key=(f['period'],ids)
            boundaries.append(previous is None or previous!=key or not 0<ts-times[-1]<=300)
            times.append(ts)
            seen={p['player_id']:p['xyz'][:2] for side in ('home_players','away_players') for p in f[side]
                  if p.get('on_court') is True and position(p)}
            coords=[seen.get(pid,(np.nan,np.nan)) for pid in ids]
            coords.append(f['ball']['xyz'][:2] if position(f.get('ball')) else (np.nan,np.nan))
            positions.append(coords)
            previous=key
    if not times:
        return name,{},None
    t=np.asarray(times)
    raw=np.asarray(positions)
    boundary=np.asarray(boundaries)
    full=np.isfinite(raw[:,:,0]).all(axis=1)
    original=pack(full,boundary)
    x,anchor=fill_short(raw,t,boundary)
    players=np.isfinite(x[:,:10,0]).all(axis=1)
    ball=np.isfinite(raw[:,10,0])
    all_ok=players & ball
    augmented=pack(all_ok,boundary)
    available=all_ok.copy()
    for i in original:available[i:i+WIDTH]=False
    extra=pack(available,boundary)
    sample=extra[:,None]+np.arange(WIDTH)
    counts=dict(original=len(original),repacked=len(augmented),extra_disjoint=len(extra),
                imputed_player_samples_in_extra=int((anchor[sample,:10]>0).sum()),
                only_ball_blocks=int((~ball[pack(players,boundary)[:,None]+np.arange(WIDTH)]).any(axis=1).sum()))
    stats=[statistics(x,raw,anchor,t,starts) for starts in (original,extra)]
    return name,counts,stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    warnings=set(json.loads((OUTPUT.parent/'coverage_distribution.json').read_text())['quarter_warning_game_ids'])
    with zipfile.ZipFile(args.archive) as z:
        names=sorted(n for n in z.namelist() if n.endswith('/tracking.jsonl.gz') and n.split('/')[-2] not in warnings)
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(name,counts,stats) in enumerate(pool.map(work,((args.archive,n) for n in names)),1):
            records.append((name,counts,stats))
            if i%100==0:print(i,'games',flush=True)
    dates=lambda n:n.split('/')[-3]
    strata=lambda n:'train' if dates(n)<'2026-05-01' else 'test'
    def combine(parts):
        return [sum(p[i] for p in parts) for i in range(4)]
    train=[s for n,c,s in records if strata(n)=='train' and s]
    test=[s[0] for n,c,s in records if strata(n)=='test' and s and s[0][3]]
    def fit(parts):
        xx,xy,_,_=combine(parts)
        return np.linalg.solve(xx+np.eye(len(xx))*10,xy)
    def mse(w,s):
        xx,xy,yy,k=s
        return (yy-2*(xy*w).sum()+(w*(xx@w)).sum())/k
    report={}
    errors=[]
    for label,parts in [('original',[s[0] for s in train]),('augmented',[p for s in train for p in s])]:
        w=fit(parts)
        errors.append(np.array([mse(w,s) for s in test]))
        report[label]=dict(train_player_targets=sum(p[3] for p in parts),test_rmse_ft=float(np.sqrt(mse(w,combine(test)))))
    sizes=np.array([s[3] for s in test])
    rng=np.random.default_rng(741)
    draws=rng.integers(len(test),size=(5000,len(test)))
    paired=[np.sqrt(np.average(e[draws],axis=1,weights=sizes[draws])) for e in errors]
    report['paired_95pct_rmse_change_ft']=np.percentile(paired[1]-paired[0],[2.5,97.5]).tolist()
    report['test_games_scored']=len(test)
    report['test_player_targets']=int(sizes.sum())
    totals=Counter()
    games=[]
    for n,c,s in records:
        totals.update(c)
        games.append(dict(game_id=n.split('/')[-2],date=dates(n),split=strata(n),**c))
    result=dict(counts=dict(totals),forecast=report,games=games)
    (OUTPUT/'comparison.json').write_text(json.dumps(result,indent=2))
    print(dict(totals),report,flush=True)


if __name__=='__main__':main()
