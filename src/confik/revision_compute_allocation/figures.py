"""Quantitative evidence panels; all rows retained, no synthetic measurements."""
from __future__ import annotations
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DISPLAY={'always_hard':'Always-hard','geometry_threshold':'Geometry rule','reject_only_hard':'Reject-only + hard',
         'routing_only':'Routing-only','p50_selection':'P50-selection','full_cghik':'CG-HIK'}
ORDER=list(DISPLAY)
COLORS=['#777777','#AAA79E','#B4B2C4','#6B9EAF','#D1A361','#215A89']
ROBOT={'panda':'Panda','ur5e':'UR5e'}


def save(fig,root,name):
    for ext in ['pdf','svg','png']:
        path=root/f'{name}.{ext}'
        if path.exists():raise FileExistsError(path)
    fig.savefig(root/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(root/f'{name}.svg',bbox_inches='tight')
    fig.savefig(root/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def draw(root,out,tables,dest=None):
    dest=out/'reports' if dest is None else dest
    dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.titlesize':10,'axes.labelsize':8,
                         'xtick.labelsize':7,'ytick.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    with (out/'01_existing_result_decomposition/trajectory_savings_decomposition.csv').open() as f:old=list(csv.DictReader(f))
    groups=['both_complete','only_cghik_complete','only_hard_complete','neither_complete']
    labels=['Both complete','Only CG-HIK','Only hard','Neither complete']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for ax,robot,letter in zip(axes,['panda','ur5e'],'ab'):
        rows=[next(r for r in old if r['robot']==robot and r['group']==g) for g in groups]
        savings=np.array([float(r['saved_latency_ns'])/1e9 for r in rows])
        ax.barh(np.arange(4),savings,color=['#6B9EAF','#D1A361','#AAAAAA','#215A89'])
        ax.axvline(0,color='#444444',lw=.7)
        ax.set_yticks(range(4),[f'{s} (n={r["trajectory_count"]})' for s,r in zip(labels,rows)])
        ax.invert_yaxis();ax.set_xlabel('Hard − CG-HIK cumulative time (s)')
        ax.set_title(f'{letter}   {ROBOT[robot]} - original trajectories',loc='left')
        for y,x in enumerate(savings):
            ax.annotate(f'{x:.2f}',(x,y),xytext=(3 if x>=0 else -3,0),textcoords='offset points',ha='left' if x>=0 else 'right',va='center',fontsize=7)
        ax.margins(x=.18)
    save(fig,dest,'cost_sources_decomposition')

    fig,axes=plt.subplots(2,3,figsize=(9.2,5.8),layout='constrained')
    for ri,robot in enumerate(['panda','ur5e']):
        for ci,(metric,title) in enumerate([('latency_ratio','Mean end-to-end cost / hard'),('fev_ratio','Mean FEV / hard'),('p95_ms','Query-call P95 (ms)')]):
            ax=axes[ri,ci]
            if ci<2:
                data=[next(r for r in tables['point_paired_comparisons'] if r['robot']==robot and r['subset']=='feasible' and r['method']==name and r['reference']=='always_hard') for name in ORDER[1:]]
                values=[r[metric] for r in data]
                lo=[r[metric]-r[metric+'_ci_low'] for r in data];hi=[r[metric+'_ci_high']-r[metric] for r in data]
                ax.errorbar(values,range(5),xerr=np.array([lo,hi]),fmt='none',ecolor='#666666',capsize=2,lw=1)
                ax.scatter(values,range(5),c=COLORS[1:],s=30,zorder=3)
                ax.axvline(1,color='#999999',ls='--',lw=.8);ax.set_yticks(range(5),[DISPLAY[n] for n in ORDER[1:]])
            else:
                data=[next(r for r in tables['point_main_table'] if r['robot']==robot and r['subset']=='feasible' and r['method']==name) for name in ORDER]
                ax.barh(range(6),[r[metric] for r in data],color=COLORS)
                ax.set_yticks(range(6),[DISPLAY[n] for n in ORDER])
            ax.invert_yaxis();ax.set_xlabel(title)
            ax.set_title(f'{"abcdef"[ri*3+ci]}   {ROBOT[robot]} · 2,500 witnesses',loc='left')
    save(fig,dest,'internal_strategy_mechanisms')

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained')
    names=['always_hard','full_cghik','trac_ik_5ms','trac_ik_20ms','trac_ik_100ms','trac_ik_400ms']
    colors=['#777777','#215A89','#4D887A','#6B9EAF','#B59259','#AA8091']
    for ax,robot,letter in zip(axes,['panda','ur5e'],'ab'):
        for j,(name,color) in enumerate(zip(names,colors)):
            rr=[r for r in tables['external_success_time_curve'] if r['robot']==robot and r['subset']=='feasible' and r['method']==name]
            x=np.array([r['elapsed_limit_ms'] for r in rr]);y=[100*r['verified_within_rate'] for r in rr]
            if np.any(x<=0):raise ValueError('Elapsed-time limits must be strictly positive for the log axis')
            assert np.all(np.diff(x)>0) and np.all(np.diff(y)>=0)
            label=DISPLAY.get(name,'TRAC-IK '+name.split('_')[-1])
            ax.plot(x,y,label=label,color=color,lw=1.5 if name=='full_cghik' else 1.,marker=['o','s','^','v','D','x'][j],ms=3)
        ax.set_xscale('log');ax.set_ylim(-2,102);ax.axvline(20,color='#BBBBBB',ls=':',lw=.7)
        ax.set_xticks([.1,1,10,100,800],['0.1','1','10','100','800'])
        ax.set_xlabel('Actual elapsed-time limit (ms)');ax.set_ylabel('Verified within limit (%)')
        ax.set_title(f'{letter}   {ROBOT[robot]} · witness-feasible queries',loc='left')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    save(fig,dest,'trac_ik_success_time')

    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
    for ri,robot in enumerate(['panda','ur5e']):
        rr=[r for r in tables['trajectory_main_table'] if r['robot']==robot and r['family']=='all']
        rr=sorted(rr,key=lambda r:['always_hard','routing_only','full_cghik'].index(r['method']) if not r['method'].startswith('trac_ik') else 3)
        labels=[DISPLAY.get(r['method'],'TRAC-IK') for r in rr]
        colors=['#777777','#6B9EAF','#215A89','#4D887A']
        axes[ri,0].bar(range(4),[r['completed'] for r in rr],color=colors)
        axes[ri,0].set_ylim(0,44);axes[ri,0].set_ylabel('Completed trajectories / 40')
        axes[ri,1].bar(range(4),[r['total_latency_ns']/1e9 for r in rr],color=colors)
        axes[ri,1].set_ylabel('Cumulative time, all frames (s)')
        for ci in range(2):
            axes[ri,ci].set_xticks(range(4),labels,rotation=20,ha='right',rotation_mode='anchor')
            axes[ri,ci].set_title(f'{"abcd"[ri*2+ci]}   {ROBOT[robot]} · feasible reference paths',loc='left')
    save(fig,dest,'feasible_reference_trajectories')
