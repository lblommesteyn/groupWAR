"""Matched coordinate and EMA latent prediction pilot on masked player paths."""
import json
import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).parent
ARCHIVE = Path.home()/'Downloads/NBA-2025-26-Available-Tracking.zip'
SAMPLES = 300


def extract(task):
    name, gid, *options = task
    include_history = bool(options and options[0])
    lineups = []
    rng = np.random.default_rng(int(gid))
    frames, times, cuts = [], [], []
    key = None
    with zipfile.ZipFile(ARCHIVE) as z, gzip.open(z.open(name),'rt') as stream:
        if include_history:
            meta=json.loads(z.read(name.rsplit('/',1)[0]+'/metadata.json'))
            teams=[meta['game'][side]['teamId'] for side in ('homeTeam','awayTeam')]
        for line in stream:
            f=json.loads(line)
            on=f.get('on_court') or {}
            h,a=on.get('home') or [],on.get('away') or []
            if (not f.get('period') or len(h)!=5 or len(a)!=5 or len(set(h+a))!=10
                    or None in h+a or f['quality'].get('source_interpolated') is True):
                key=None
                continue
            ids=tuple(sorted(h)+sorted(a))
            k=(f['period'],ids)
            t=f['timestamp_ms']
            cuts.append(k!=key or not times or not 175<=t-times[-1]<=225)
            key=k
            pos={p['player_id']:p['xyz'][:2] for side in ('home_players','away_players')
                 for p in f[side] if p.get('on_court') and p.get('xyz')}
            ball=f.get('ball') or {}
            frames.append([pos.get(pid,[np.nan]*2) for pid in ids]+[ball.get('xyz',[np.nan]*2)[:2]])
            times.append(t)
            if include_history: lineups.append(ids)
    if not frames:return {}
    xy=np.asarray(frames);t=np.asarray(times);seg=np.cumsum(cuts)
    output={}
    for n in (5,10):
        # Two visible samples before and after each endpoint; continuous target truth.
        w=n+6
        starts=np.arange(len(xy)-w+1)
        starts=starts[seg[starts]==seg[starts+w-1]]
        missing=np.vstack([np.zeros((1,10)),np.cumsum(~np.isfinite(xy[:,:10]).all(axis=2),axis=0)])
        s,p=np.where((missing[starts+w]-missing[starts])==0)
        if not len(s):continue
        pick=rng.choice(len(s),min(SAMPLES,len(s)),replace=False)
        s,p=starts[s[pick]],p[pick]
        idx=np.arange(len(s)); path=xy[s[:,None]+np.arange(w),p[:,None]]
        left,right=2,n+3
        ts=t[s[:,None]+np.arange(w)]
        alpha=(ts-ts[:,left,None])/(ts[:,right,None]-ts[:,left,None])
        base=path[:,left,None]+alpha[:,:,None]*(path[:,right,None]-path[:,left,None])
        truth=path[:,3:right]-base[:,3:right]
        # Teacher and student share the token encoder: five XY offsets from the left endpoint.
        visible=path[:,[0,1,left,right,right+2]]-path[:,left,None]
        hidden=path[:,np.linspace(3,right-1,5).round().astype(int)]-path[:,left,None]
        scene=xy[s[:,None]+np.array([left,(left+right)//2,right])]
        features=[]
        for j in range(3):
            delta=scene[:,j]-base[:,[left,(left+right)//2,right][j],None]
            dist=np.linalg.norm(delta,axis=2)
            dist[~np.isfinite(dist)]=np.inf
            dist[idx,p]=np.inf
            mate=(np.arange(10)[None,:]//5==p[:,None]//5)
            for mask in [mate,~mate]:
                best=np.argmin(np.where(mask,dist[:,:10],np.inf),axis=1)
                ok=np.isfinite(np.where(mask,dist[:,:10],np.inf)[idx,best])
                features.extend([np.where(ok[:,None],delta[idx,best],0)/10,ok[:,None]])
            ball=delta[:,10];ok=np.isfinite(ball).all(axis=1)
            features.extend([np.where(ok[:,None],ball,0)/10,ok[:,None]])
        features.extend([path[:,left]/40,alpha[:,3:right]])
        # Direction change between net approach and departure across the two halves.
        mid=(left+right)//2
        u=path[:,mid]-path[:,left];v=path[:,right]-path[:,mid]
        lengths=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        angle=np.degrees(np.arccos(np.clip((u*v).sum(axis=1)/np.maximum(lengths,1e-9),-1,1)))
        active=(np.linalg.norm(u,axis=1)>=1)&(np.linalg.norm(v,axis=1)>=1)
        group=np.where(active&(angle>=60),2,np.where(active&(angle<=20),1,0))
        output[n]=tuple(np.asarray(arr,dtype=np.float32) for arr in (
            visible.reshape(-1,10)/5,hidden.reshape(-1,10)/5,np.concatenate(features,axis=1),
            truth.reshape(len(s),-1),group))
        if include_history:
            dt=(ts[:,right]-ts[:,left])[:,None,None]
            speed_in=(path[:,left]-path[:,0])/(ts[:,left]-ts[:,0])[:,None]
            speed_out=(path[:,right+2]-path[:,right])/(ts[:,right+2]-ts[:,right])[:,None]
            q=alpha[:,3:right,None]
            chord=path[:,right,None]-path[:,left,None]
            correction=q*(1-q)*((1-q)*(dt*speed_in[:,None]-chord)+q*(chord-dt*speed_out[:,None]))
            speed=np.linalg.norm(speed_in,axis=1)*1000
            location=np.abs(path[:,left,0])
            ranges=np.linalg.norm(scene[:,0,10]-path[:,left],axis=1)
            context=np.digitize(speed,[3,8])*12+np.digitize(location,[15,32])*4+np.where(np.isfinite(ranges),np.digitize(ranges,[5,15]),3)
            player=np.array([lineups[i][j] for i,j in zip(s,p)])
            keys=np.column_stack((player,np.asarray(teams)[p//5],np.asarray(teams)[1-p//5],context))
            direction=chord[:,0]/np.maximum(np.linalg.norm(chord[:,0],axis=1,keepdims=True),1e-8)
            output[n] += (correction,keys,direction)
    return output


def fit(train,val,n,aux):
    rng=np.random.default_rng(900+n)
    x,h,c,y,_=train
    def weight(a,b):return (rng.normal(size=(a,b))/np.sqrt(a)).astype('f')
    params=[weight(10,32),weight(32+c.shape[1],64),weight(64,2*n)*.1,weight(64,32)]
    teacher=params[0].copy()
    m=[np.zeros_like(p) for p in params];v=[a.copy() for a in m]
    best=np.inf;step=0
    def forward(x,c):
        enc=np.tanh(x@params[0]);ctx=np.column_stack((enc,c))
        z=np.tanh(ctx@params[1])
        return enc,ctx,z,z@params[2]
    for epoch in range(40):
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
            enc,ctx,z,out=forward(x[batch],c[batch])
            dy=2*(out-y[batch])/out.size
            target=np.tanh(h[batch]@teacher)
            dzaux=aux*2*(z@params[3]-target)/target.size
            dz=(dy@params[2].T+dzaux@params[3].T)*(1-z*z)
            de=(dz@params[1][:32].T)*(1-enc*enc)
            grads=[x[batch].T@de,ctx.T@dz,z.T@dy,z.T@dzaux]
            step+=1
            for i,g in enumerate(grads):
                m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                params[i]-=.001*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
            teacher=.99*teacher+.01*params[0]
        error=np.mean((forward(val[0],val[2])[-1]-val[3])**2)
        if error<best:
            best=error;saved=[p.copy() for p in params];best_epoch=epoch+1
    params[:]=saved
    return lambda x,c:forward(x,c)[-1],best_epoch


def main():
    eligible=json.loads((ROOT/'results/screening.json').read_text())['candidate_games_without_quarter_warnings']
    rng=np.random.default_rng(615)
    splits={}
    for split,start,end,count in [('train','2026-01','2026-04',100),('validation','2026-04','2026-05',20),('test','2026-05','2026-07',43)]:
        games=sorted(g['game_id'] for g in eligible if start<=g['date']<end)
        splits[split]=sorted(rng.choice(games,min(len(games),count),replace=False).tolist())
    with zipfile.ZipFile(ARCHIVE) as z:
        members={n.split('/')[-2]:n for n in z.namelist() if n.endswith('/tracking.jsonl.gz')}
    data={}
    ids=sum(splits.values(),[])
    with ProcessPoolExecutor(max_workers=5) as pool:
        for i,(gid,examples) in enumerate(zip(ids,pool.map(extract,[(members[g],g) for g in ids])),1):
            data[gid]=examples
            if i%30==0:print(i,'games',flush=True)
    report={'splits':splits,'results':{}}
    for n in (5,10):
        def collect(split):
            return [np.concatenate(parts) for parts in zip(*(data[g][n] for g in splits[split] if n in data[g]))]
        train,val,test=map(collect,('train','validation','test'))
        models=[fit(train,val,n,a) for a in (0,1)]
        x,h,c,y,group=test
        errors=[np.linalg.norm((pred-y).reshape(-1,n,2),axis=2) for pred in
                [np.zeros_like(y)]+[fn(x,c) for fn,_ in models]]
        res={'selected_epochs':[epoch for _,epoch in models],'windows':[len(d[0]) for d in [train,val,test]]}
        for label,mask in [('all',np.ones(len(y),bool)),('straight',group==1),('turn',group==2)]:
            res[label]={'windows':int(mask.sum()),'median_p90_feet':{
                name:np.percentile(e[mask],[50,90]).tolist() for name,e in zip(['linear','coordinate','jepa'],errors)}}
        gid=np.concatenate([np.repeat(g,len(data[g][n][0])) for g in splits['test'] if n in data[g]])
        diff=(errors[2]-errors[1]).mean(axis=1)
        sums=np.array([diff[gid==g].sum() for g in np.unique(gid)])
        sizes=np.array([(gid==g).sum() for g in np.unique(gid)])
        picks=rng.integers(len(sums),size=(5000,len(sums)))
        res['jepa_minus_coordinate_mean_ft_ci']=np.percentile(sums[picks].sum(1)/sizes[picks].sum(1),[2.5,97.5]).tolist()
        report['results'][n]=res
        print(n,json.dumps(res),flush=True)
    (ROOT/'results/masked_path_comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
