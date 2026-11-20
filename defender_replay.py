"""Replay full multi-player visibility masks on clean windows, keeping ball reference fixed."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip,json,zipfile
import numpy as np
from masked_paths import ROOT,ARCHIVE

W,C=26,13

def read_game(args):
    game,name=args
    rng=np.random.default_rng(int(game))
    xy=[];times=[];segments=[];prev=None;seg=0
    with zipfile.ZipFile(ARCHIVE) as z,gzip.open(z.open(name),'rt') as f:
        for line in f:
            r=json.loads(line); on=r.get('on_court') or {}
            h,a=on.get('home') or [],on.get('away') or []
            if not r.get('period') or len(h)!=5 or len(a)!=5 or len(set(h+a))!=10 or None in h+a or r['quality'].get('source_interpolated') is True:
                prev=None;continue
            ids=tuple(sorted(h)+sorted(a));key=(ids,r['period']);t=r['timestamp_ms']
            if prev!=key or not 175<=t-times[-1]<=225:seg+=1
            prev=key
            obs={p['player_id']:p['xyz'][:2] for side in ['home_players','away_players'] for p in r[side] if p.get('on_court') and p.get('xyz')}
            xy.append([obs.get(pid,[np.nan]*2) for pid in ids]+[(r.get('ball') or {}).get('xyz',[np.nan]*2)[:2]])
            times.append(t);segments.append(seg)
    if len(xy)<W:return [],[]
    xy=np.asarray(xy);times=np.array(times);seg=np.array(segments);ok=np.isfinite(xy).all(2)
    s=np.arange(len(xy)-W+1);s=s[seg[s]==seg[s+W-1]]
    donors=s[ok[s+C,10]&~ok[s+C,:10].all(1)]
    donors=rng.choice(donors,min(300,len(donors)),replace=False)
    masks=ok[donors[:,None]+np.arange(W),:10]
    bad=np.r_[0,np.cumsum(~ok.all(1))]; starts=s[bad[s+W]==bad[s]]
    # Keep receiver clips disjoint and near a possible ball handler.
    clean=[];last=-W
    for i in starts:
        if i<last+W:continue
        dist=np.linalg.norm(xy[i+C,:10]-xy[i+C,10],axis=1)
        if dist.min()>4:continue
        clean.append((xy[i:i+W].astype('f'),times[i:i+W]-times[i],int(np.argmin(dist))))
        last=i
    if len(clean)>160:clean=[clean[i] for i in rng.choice(len(clean),160,replace=False)]
    return masks,clean


def recover(x,t,mask,p):
    valid=np.flatnonzero(mask[:,p]);before=valid[valid<C];after=valid[valid>C]
    if len(before)<2 or len(after)<2:return None
    l,r=before[-1],after[0];l0,r0=before[-2],after[1]
    if l-l0>2 or r0-r>2:return None
    dt=(t[r]-t[l])/1000
    if (t[r]+t[r-1]-t[l]-t[l+1])/2>2050:return None
    q=(t[C]-t[l])/(t[r]-t[l]);chord=x[r,p]-x[l,p]
    base=x[l,p]+q*chord
    u=(x[l,p]-x[l0,p])*1000/(t[l]-t[l0]);v=(x[r0,p]-x[r,p])*1000/(t[r0]-t[r])
    U=(dt*u-chord)*q*(1-q)**2;V=(chord-dt*v)*q*q*(1-q)
    features=np.stack([U,V,U*dt,V*dt,U*np.linalg.norm(u)/10,V*np.linalg.norm(v)/10],axis=1)
    return base,U+V,features


def scenarios(data,gids,donor_ids,seed):
    rng=np.random.default_rng(seed)
    pool=np.concatenate([data[g][0] for g in donor_ids if len(data[g][0])])
    for g in gids:
        for x,t,handler in data[g][1]:
            for _ in range(4):
                mask=pool[rng.integers(len(pool))][:,np.r_[rng.permutation(5),5+rng.permutation(5)]]
                defenders=np.arange(5)+(5 if handler<5 else 0)
                yield g,x,t,mask,defenders


def summarize(rows):
    a=np.array(rows);out={'cases':len(a)}
    for i,key in enumerate(['visible','curve','learned']):
        dist,idok=a[:,2*i],a[:,2*i+1];valid=np.isfinite(dist)
        out[key]=dict(answer_rate=float(valid.mean()),correct_per_case=float(np.nansum(idok)/len(a)),
                      accuracy_when_answered=float(np.nanmean(idok)) if valid.any() else None,
                      mae_ft=float(np.nanmean(dist)) if valid.any() else None,
                      p90_ft=float(np.percentile(dist[valid],90)) if valid.any() else None)
    return out


def main():
    splits=json.loads((ROOT/'results/masked_path_comparison.json').read_text())['splits']
    train,test=splits['train'],splits['test']
    with zipfile.ZipFile(ARCHIVE) as z:
        names={n.split('/')[-2]:n for n in z.namelist() if n.endswith('/tracking.jsonl.gz')}
    with ProcessPoolExecutor(max_workers=5) as pool:
        data=dict(zip(train+test,pool.map(read_game,[(g,names[g]) for g in train+test])))
    xx=np.eye(6)*10;xy=np.zeros(6);count=0
    for g,x,t,mask,defs in scenarios(data,train,train,918):
        for p in defs[~mask[C,defs]]:
            fit=recover(x,t,mask,p)
            if fit:
                base,_,features=fit
                xx+=features.T@features;xy+=features.T@(x[C,p]-base);count+=1
    weights=np.linalg.solve(xx,xy)
    groups=defaultdict(list);clips=sum(len(data[g][1]) for g in test)
    for g,x,t,mask,defs in scenarios(data,test,train,919):
        ball=x[C,10];truth=np.linalg.norm(x[C,defs]-ball,axis=1);best=truth.argmin()
        missing=~mask[C,defs]
        if not missing.any():continue
        pred=x[C,defs].copy();pred[missing]=np.nan
        predictions=[pred,pred.copy(),pred.copy()]
        for j in np.flatnonzero(missing):
            fit=recover(x,t,mask,defs[j])
            if fit:
                base,offset,feat=fit
                predictions[1][j]=base+offset;predictions[2][j]=base+feat@weights
        scores=[]
        for positions in predictions:
            dist=np.linalg.norm(positions-ball,axis=1)
            if np.isfinite(dist).any():
                j=np.nanargmin(dist);scores.extend([abs(float(dist[j]-truth[best])),int(j==best)])
            else:scores.extend([np.nan,np.nan])
        labels=['all','one_missing' if missing.sum()==1 else 'multiple_missing']
        if missing[best]:
            labels.append('nearest_hidden')
            seen=np.flatnonzero(mask[:,defs[best]]);l=seen[seen<C];r=seen[seen>C]
            if len(l) and len(r):
                a,b=l[-1],r[0];ms=(t[b]+t[b-1]-t[a]-t[a+1])/2
                labels.append('nearest_gap_'+('le0.6s' if ms<=600 else '0.6_to_1s' if ms<=1000 else '1_to_2s' if ms<=2000 else 'over2s'))
            else:labels.append('nearest_gap_unbounded')
        if np.isfinite(predictions[1]).all():labels.append('all_defenders_available_after_fill')
        for label in labels:groups[label].append(scores)
    report=dict(test_clips=clips,test_games_with_clips=sum(bool(len(data[g][1])) for g in test),
                donor_masks=sum(len(data[g][0]) for g in train),train_reconstructable_points=count,
                method='Four mask draws per disjoint receiver clip. Ball within 4 feet of nearest player; opposing side held fixed. Player positions in feet. Learned model is a ridge fit of velocity corrections.',
                groups={k:summarize(v) for k,v in groups.items()})
    (ROOT/'results/defender_replay.json').write_text(json.dumps(report,indent=2))
    print(clips,[(k,r['cases'],[(m,round(r[m]['correct_per_case'],3),r[m]['mae_ft']) for m in ('visible','curve','learned')]) for k,r in report['groups'].items()])
if __name__=='__main__':main()
