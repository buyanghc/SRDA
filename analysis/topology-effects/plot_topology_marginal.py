"""Balanced topology summaries; keep original review bars and manuscript intact."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'by_model_topology_condition.csv'
with SOURCE.open(newline='') as f:
    raw = list(csv.DictReader(f))
cells = {(r['campaign'], r['graph'], r['condition']): r for r in raw}
models = sorted({r['campaign'] for r in raw})
assert len(cells) == 504 and len(models) == 6
assert all(int(r['n']) == 135 for r in raw)
def effects(model, graph):
    def s(c):
        r = cells[model, graph, c]
        return 100 * Fraction(int(r['successes']), int(r['n']))
    st, se, sn, ht, he, hn = map(s, ['system-terminal', 'system-empty',
        'system-natural', 'forward-terminal', 'forward-empty', 'forward-natural'])
    return [ht-st, se-st, sn-st, he-ht-se+st, hn-ht-sn+st]

SPEC = [
    ('outdegree', 'Out-degree', [1,2,3],
     [[f'G({b},3)'] for b in [1,2,3]],
     ['1','2','3'], 'Depth fixed at 3'),
    ('depth', 'Depth', [2,3,6],
     [[f'G(1,{d})'] for d in [2,3,6]],
     ['2','3','6'], 'Out-degree fixed at 1'),
    ('paths', 'Added links', [0,1,2,3],
     [[g] for g in ['G33-X2','G33-P2','G33-X3','G33-P3']],
     ['1 dead-end','1 target','2 dead-end','2 target'], 'Matched one-link and two-link pairs'),
]
SERIES = [('H','#376795','o','-'), ('E','#72BCD5','s','--'),
          ('N','#FFE6B7','^','-.'), ('H × E','#FFD06F','o','-'),
          ('H × N','#E76254','D','--')]
plt.rcParams.update({'font.family':'Times New Roman','font.size':15,
    'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,
    'axes.linewidth':1,'pdf.fonttype':42,'svg.fonttype':'path',
    'axes.unicode_minus':False,'hatch.linewidth':.8})
summary, detail, vals = [], [], {}
for slug, _, xs, groups, _, _ in SPEC:
    arr = np.zeros((5,len(groups)))
    for j, graphs in enumerate(groups):
        contributions = [effects(m,g) for g in graphs for m in models]
        for i, (name,_,_,_) in enumerate(SERIES):
            value = sum(r[i] for r in contributions) / len(contributions)
            arr[i,j] = float(value)
            summary.append(dict(chart=slug,x=xs[j],effect=name,value_pp=float(value),
                exact_value_pp=str(value),graphs=';'.join(graphs),
                configurations=6,executions_per_condition=810*len(graphs)))
        for g in graphs:
            for m in models:
                for i, (name,_,_,_) in enumerate(SERIES):
                    detail.append(dict(chart=slug,x=xs[j],graph=g,campaign=m,
                                       effect=name,value_pp=float(effects(m,g)[i])))
    vals[slug] = arr

def panel(ax, spec):
    slug, xlabel, xs, groups, labels, subtitle = spec
    indices = list(range(5))
    values = vals[slug]
    if slug == 'paths':
        # Touching thin bars; one bar-width gap between neighboring groups.
        count=len(indices); centers=np.arange(4)*(count+1)
        for k,i in enumerate(indices):
            name,color,marker,ls=SERIES[i]
            xpos=centers+k-(count-1)/2
            ax.bar(xpos,values[i],width=1,color=color,edgecolor='black',
                   linewidth=.7,hatch=['/','.','+','xx','\\'][i],
                   hatchcolor='white',label=name,zorder=3)
            label_small(ax,xpos,values[i],k)
        ax.set_xticks(centers,labels,rotation=20,ha='right',fontsize=18)
        ax.set_xlim(-count/2-.5,centers[-1]+count/2+.5)
        ax.axvline((centers[1]+centers[2])/2,color='#BBBBBB',ls=':',lw=.8)
    else:
        for k,i in enumerate(indices):
            name,color,marker,ls=SERIES[i]
            ax.plot(xs,values[i],color=color,marker=marker,linestyle=ls,
                    linewidth=2,markersize=7,markeredgecolor='#333333',
                    markeredgewidth=.6,label=name,zorder=3)
        ax.set_xticks(xs,labels)
        ax.set_xlim(min(xs)-.2,max(xs)+.2)
    if slug == 'paths':
        ax.set_ylim(-1,14)
        ax.set_yticks([0,2,4,6,8,10,12,14])
    else:
        ax.set_ylim(-2,28)
        ax.set_yticks([0,5,10,15,20,25])
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E3E3E3',lw=.65)
    ax.axhline(0,color='#AAAAAA',ls=(0,(4,4)),lw=.8)
    ax.set_ylabel('Effect on ASR (pp)')
    ax.legend(loc='upper right' if slug in ('outdegree', 'paths') else 'upper left',frameon=True,fancybox=False,edgecolor='#BBBBBB',
              framealpha=1,fontsize=11,ncol=1,handlelength=1.5,borderpad=.25,labelspacing=.12)
    ax.set_xlabel(xlabel)
    if slug == 'paths':
        ax.xaxis.label.set_fontsize(19)
    ax.set_title(subtitle,fontsize=14,pad=9)

def label_small(ax,xs,ys,series_index):
    for x,y in zip(xs,ys):
        if abs(y)<1:
            ax.annotate(f'{y:.2f}',(x,max(y,0)),xytext=(0,4),
                        textcoords='offset points',ha='center',va='bottom',rotation=90,fontsize=12,
                        fontweight='bold',zorder=6)

for spec in SPEC:
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    fig.subplots_adjust(left=.16,right=.98,bottom=.24,top=.90)
    panel(ax,spec)
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/f'{spec[0]}-marginal.{ext}',dpi=170)
    plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
fig.subplots_adjust(left=.055,right=.99,bottom=.24,top=.90,wspace=.30)
for j,spec in enumerate(SPEC):
    panel(axes[j],spec)
for ext in ['png','pdf','svg']:
    fig.savefig(ROOT/f'topology-marginal-overview.{ext}',dpi=150)
plt.close(fig)
for name,rows in [('marginal-summary.csv',summary),('marginal-by-model.csv',detail)]:
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'marginal-audit.json').write_text(json.dumps(dict(source=str(SOURCE),
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    formulas={'H':'HT-ST','E':'SE-ST','N':'SN-ST','H×E':'HE-HT-SE+ST','H×N':'HN-HT-SN+ST'},
    averaging='Equal weight over the listed topologies and six model configurations.',
    exclusion='Degree comparison fixes depth at 3; depth comparison fixes degree at 1. No unmeasured graph cells are imputed.',
    uncertainty='Descriptive means, no confidence intervals or significance claims.',
    manuscript_updated=False),indent=2)+'\n')
