import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

out = Path(__file__).parent / 'results'
data = json.loads((out / 'trajectory_priors.json').read_text())
fig = plt.figure(figsize=(12.8, 8.8), layout='constrained')
grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=.17)
colors = ['#146d82', '#ad4b20']
methods = ['linear','curve','all_priors','own_motion','own_motion_and_scene']
labels = ['Straight line','Velocity-based curve','Curve + all priors','Learned own motion','Own motion + scene']
fig.suptitle('Reconstructing missing NBA player paths', x=.02, ha='left', weight='bold', fontsize=21)
for j,(duration,group,title) in enumerate([('5','all','1-second gaps'),('10','all','2-second gaps'),('10','turn','2-second sharp turns')]):
    ax = fig.add_subplot(grid[0,j]); ax.set_title(title, loc='left', weight='bold', pad=12)
    for i,method in enumerate(methods):
        m,p = data[duration][group][method]
        ax.plot([m,p],[i,i],color='#9abcc2',lw=2.4)
        ax.scatter(m,i,color=colors[0],s=43,zorder=3)
        ax.scatter(p,i,facecolor='white',edgecolor=colors[0],linewidth=1.7,s=43,zorder=3)
        ax.annotate(f'{m:.2f} / {p:.2f}',(p,i),xytext=(0,9),textcoords='offset points',ha='right',fontsize=9)
    ax.set(xlim=(0,5.85),ylim=(4.55,-.75),yticks=range(5),yticklabels=labels if j==0 else ['']*5,xlabel='Position error (feet)')
    ax.grid(axis='x',alpha=.17)
    if j==0:
        ax.legend(handles=[Line2D([],[],marker='o',ls='',color=colors[0],label='Median'),Line2D([],[],marker='o',ls='',mfc='white',color=colors[0],label='90th percentile')],loc='lower right',fontsize=8,frameon=False)
ax=fig.add_subplot(grid[1,:])
ax.set_title('What each historical prior adds to the league-average curve correction',loc='left',weight='bold',pad=10)
for j,(duration,name) in enumerate([('5','1-second gaps'),('10','2-second gaps')]):
    vals=np.array([data[duration]['mean_differences'][k+' vs league'] for k in ['player','team','opponent','context']])*304.8
    y=np.arange(4)+(j-.5)*.26
    mean,lo,hi=vals.T
    ax.errorbar(mean,y,xerr=[mean-lo,hi-mean],fmt='o' if j==0 else 's',color=colors[j],capsize=4,label=name)
ax.axvline(0,color='#454545',ls='--',lw=1)
ax.set(yticks=range(4),yticklabels=['Player','Own team','Opponent','Situation'],ylim=(3.6,-.6),xlim=(-.83,.18),xlabel='Change in mean position error (millimeters)     ·     Negative values indicate improvement')
ax.legend(loc='upper left',frameon=False)
ax.grid(axis='x',alpha=.17)
for a in fig.axes:
    a.spines[['top','right','left']].set_visible(False)
    a.tick_params(axis='y',length=0)
fig.supxlabel('43 held-out playoff games  •  12,900 masked paths per duration  •  Lower-panel whiskers show 95% intervals from resampling games.',fontsize=10,color='#45535a')
for ext in ('png','svg'):
    fig.savefig(out / ('trajectory_prior_ablation.'+ext),dpi=165,bbox_inches='tight')
