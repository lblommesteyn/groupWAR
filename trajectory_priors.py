"""Compare cubic Hermite interpolation and historical residual corrections on fixed masked paths."""
from concurrent.futures import ProcessPoolExecutor
import json
import zipfile
import numpy as np
from masked_paths import extract, fit, ROOT, ARCHIVE


def evaluate(parts,n,game_ids):
    train,val,test=parts
    def components(data):
        truth=data[3].reshape(-1,n,2)
        residual=truth-data[5]
        fwd=data[7]
        side=np.column_stack((-fwd[:,1],fwd[:,0]))
        rotation=np.stack((fwd,side),axis=2)
        return residual@rotation,rotation
    historical,_=components(train)
    league=historical.mean(axis=0)
    # Shrink per-group mean curve toward league. Count windows, not individual frames.
    tables=[]
    for col in range(4):
        keys=train[6][:,col]
        table={}
        for k in np.unique(keys):
            sample=historical[keys==k]
            table[int(k)]=(sample.sum(axis=0)+50*league)/(len(sample)+50)-league
        tables.append(table)
    def predictions(data):
        _,rotation=components(data)
        zero=np.zeros_like(league)
        estimates=[np.broadcast_to(league,(len(rotation),n,2))]
        for i,table in enumerate(tables):
            estimates.append(np.stack([table.get(int(k),zero) for k in data[6][:,i]]))
        return np.stack([arr@rotation.transpose(0,2,1) for arr in estimates],axis=-1)
    V,T=predictions(val),predictions(test)
    y=val[3].reshape(-1,n,2)-val[5]
    truth=test[3].reshape(-1,n,2)
    preds={'linear':np.zeros_like(truth),'curve':test[5]}
    for label,cols in {'league':[0],'player':[0,1],'team':[0,2],'opponent':[0,3],'context':[0,4],'all_priors':list(range(5))}.items():
        design=V[...,cols].reshape(-1,len(cols))
        weights=np.linalg.solve(design.T@design+10*np.eye(len(cols)),design.T@y.ravel())
        preds[label]=test[5]+T[...,cols]@weights
    for label,scene in [('own_motion',False),('own_motion_and_scene',True)]:
        def inputs(d):
            # Fixed-width zeroed social features maintain architecture and initialization.
            ctx=d[2].copy()
            if not scene:ctx[:,:27]=0
            return [d[0],d[1],ctx,d[3],d[4]]
        fn,_=fit(inputs(train),inputs(val),n,0)
        x,_,c,_,_=inputs(test)
        preds[label]=fn(x,c).reshape(-1,n,2)
    err={k:np.linalg.norm(v-truth,axis=2) for k,v in preds.items()}
    res={}
    for label,mask in [('all',np.ones(len(truth),bool)),('straight',test[4]==1),('turn',test[4]==2)]:
        res[label]={k:np.percentile(v[mask],[50,90]).tolist() for k,v in err.items()}
    rng=np.random.default_rng(748)
    games=np.unique(game_ids)
    picks=rng.integers(len(games),size=(5000,len(games)))
    res['mean_differences']={}
    for label,base in [('curve','linear'),('player','league'),('team','league'),('opponent','league'),('context','league'),('all_priors','curve'),('own_motion_and_scene','own_motion')]:
        diff=(err[label]-err[base]).mean(axis=1)
        sums=np.array([diff[game_ids==g].sum() for g in games])
        sizes=np.array([(game_ids==g).sum() for g in games])
        res['mean_differences'][label+' vs '+base]=[float(diff.mean()),*np.percentile(sums[picks].sum(1)/sizes[picks].sum(1),[2.5,97.5])]
    res['seen_player_share']=float(np.isin(test[6][:,0],train[6][:,0]).mean())
    return res


if __name__=='__main__':
    splits=json.loads((ROOT/'results/masked_path_comparison.json').read_text())['splits']
    gids=sum(splits.values(),[])
    with zipfile.ZipFile(ARCHIVE) as z:
        names={n.split('/')[-2]:n for n in z.namelist() if n.endswith('/tracking.jsonl.gz')}
    with ProcessPoolExecutor(max_workers=5) as pool:
        data=dict(zip(gids,pool.map(extract,[(names[g],g,True) for g in gids])))
    report={}
    for n in (5,10):
        parts=[[np.concatenate(a) for a in zip(*(data[g][n] for g in gids if n in data[g]))] for gids in splits.values()]
        ids=np.concatenate([np.repeat(g,len(data[g][n][0])) for g in splits['test'] if n in data[g]])
        report[n]=evaluate(parts,n,ids)
        print(n,json.dumps(report[n]),flush=True)
    (ROOT/'results/trajectory_priors.json').write_text(json.dumps(report,indent=2)+'\n')
