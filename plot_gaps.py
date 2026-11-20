"""Render measured missing-episode distributions; no inferred observations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent/'results'
r=json.loads((root/'gaps.json').read_text())
edges=np.array([0,1,2,3,5,10,20,60,np.inf])
labels=['≤1 s','1–2 s','2–3 s','3–5 s','5–10 s','10–20 s','20–60 s','>60 s']
data={}
for kind,hist in r['duration_ms_histogram'].items():
    times=np.array([int(k)/1000 for k in hist]);counts=np.array(list(hist.values()))
    bins=np.searchsorted(edges[1:],times,side='left')
    number=np.bincount(bins,weights=counts,minlength=8)
    duration=np.bincount(bins,weights=counts*times,minlength=8)
    order=np.argsort(times);cum=np.cumsum(counts[order]);n=int(counts.sum())
    quantiles={str(q):float(times[order][np.searchsorted(cum,q*n)]) for q in (.5,.9,.95,.99)}
    data[kind]={'episodes':n,'censored_episodes':r['censored_episodes'].get(kind,0),
                'percent_at_most_1s':float(counts[times<=1].sum()/n*100),
                'percent_at_most_2s':float(counts[times<=2].sum()/n*100),
                'quantiles_seconds':quantiles,'bins':labels,'episode_counts':number.astype(int).tolist(),
                'missing_seconds':duration.tolist(),'episode_percent':(number/n*100).tolist(),
                'missing_time_percent':(duration/duration.sum()*100).tolist()}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(13,9.3),layout='constrained')
fig.set_facecolor('#f8fafc')
x=np.arange(len(labels))
for ax,metric,title in zip(axes,['episode_percent','missing_time_percent'],
                         ['Measurable gaps: percentage of disappearance episodes',
                          'Measurable gaps: percentage of missing time']):
    ax.set_facecolor('#f8fafc')
    for offset,kind,color,label in [(-.19,'players','#2563eb','Individual players'),(.19,'ball','#e87924','Ball')]:
        bars=ax.bar(x+offset,data[kind][metric],width=.36,color=color,label=label,zorder=3)
        ax.bar_label(bars,labels=[(f'{v:.1f}%' if v >= .05 else '<0.1%') for v in data[kind][metric]],padding=4,fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylabel('Share (%)');ax.set_title(title,loc='left',pad=15,weight='bold')
    ax.grid(axis='y',alpha=.18,zorder=0);ax.set_ylim(0,max(max(data[k][metric]) for k in data)*1.24)
axes[0].legend(frameon=False,loc='upper right')
axes[1].set_xlabel('Missing episode duration · source timestamp seconds (not game-clock seconds)')
fig.suptitle('NBA tracking: how long do players and the ball go missing?\n'
             f"666 games · {data['players']['episodes']:,} player gaps · {data['ball']['episodes']:,} ball gaps",
             fontsize=17,weight='bold',ha='left',x=.04)
fig.supxlabel('Only gaps bounded by observed positions, within unchanged lineups/periods and timestamp steps ≤300 ms.\n'
              'Excluded: 17.95M player and 1.79M ball boundary-truncated fragments. Stoppages may remain. Bin upper bounds inclusive.',
              fontsize=9,color='#475569')
fig.savefig(root/'missing_duration_distribution.png',dpi=180)
fig.savefig(root/'missing_duration_distribution.svg')
(root/'gap_statistics.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
