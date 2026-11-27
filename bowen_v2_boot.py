import numpy as np
from pathlib import Path
W=Path('results/world')
def key(z): return z['poss'].astype(np.int64)*600+z['t'].astype(np.int64)*5+z['k'].astype(np.int64)
F={'v1':('pred.npz',None),'A':('bowen_live/pred.npz',None),'B':('bowen_live_mix3/pred.npz','nll_s'),'Vs':('bowen_v3_hier/pred.npz','nll_s'),'Vp':('bowen_v3_hier/pred.npz','nll_p')}
Z={};N={}
for m,(f,nk) in F.items():
    z=np.load(W/f); k=key(z); o=np.argsort(k); Z[m]=(k[o],z,o)
    if nk: N[m]=z[nk][o].astype(np.float32)
    else:
        ls=z['ls'][o].astype(np.float32); d=(z['y'][o]-z['ms'][o]).astype(np.float32); N[m]=(d**2).sum(-1)/(2*np.exp(ls))+ls
common=Z['v1'][0]
for m in Z: common=np.intersect1d(common,Z[m][0])
for m in Z: N[m]=N[m][np.searchsorted(Z[m][0],common)]
z=Z['v1'][1]; o=Z['v1'][2]; ix=np.searchsorted(Z['v1'][0],common)
live=Z['B'][1]['live'][Z['B'][2]][np.searchsorted(Z['B'][0],common)].astype(bool)
gid=z['meta_gid'][z['poss'][o][ix]]
u,g=np.unique(gid[live],return_inverse=True); rng=np.random.default_rng(0)
for a,b in [('A','v1'),('B','v1'),('Vs','B'),('Vp','B'),('Vp','Vs')]:
    d=(N[a]-N[b])[live]; s=np.bincount(g,d); c=np.bincount(g)
    bs=[]
    for _ in range(1000):
        w=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)); bs.append((w*s).sum()/(w*c).sum())
    print(a,'-',b,round(d.mean(),4),np.round(np.percentile(bs,[2.5,97.5]),4),'games',len(u))
