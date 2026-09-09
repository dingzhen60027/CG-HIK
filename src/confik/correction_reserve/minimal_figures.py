"""Paired tradeoff and one UID-fixed real trace, not a success-only illustration.

Figure contract: ask whether reduced intervention preserves completion at lower
cost. Python quantitative grids, 183 mm, >=7 pt text, editable SVG/PDF, 600 dpi
PNG previews. Bootstrap uncertainty is over whole UIDs, not frames. The trace
is a descriptive first-UID high-curvature run, not an independent experiment.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from . import study as old
from .minimal_study import configurations

METHODS=['cr_ik','two_step_predictive','trac_task_5ms','trac_task_20ms','pink_qp',
         'cr_ik_minimal_fixed','cr_ik_minimal_no_shortcut']
LABELS=['Original CR-IK','Two-step predictive','TRAC-IK 5 ms','TRAC-IK 20 ms','Pink QP',
        'Fixed-demand control','No-shortcut control']
ROBOTS={'panda':'Panda','ur5e':'UR5e'}


def save(fig,out,name):
    fig.savefig(out/f'{name}.svg',facecolor='white')
    fig.savefig(out/f'{name}.pdf',facecolor='white')
    fig.savefig(out/f'{name}.png',dpi=600,facecolor='white')
    plt.close(fig)


def main():
    _,_,root=configurations();out=root/'figures';out.mkdir(exist_ok=False)
    source=root/'reports/source_data.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2047244,5.51),sharey=True)
    fig.subplots_adjust(left=.26,right=.97,top=.93,bottom=.11,wspace=.24,hspace=.32)
    extent=5.
    for col,robot in enumerate(ROBOTS):
        for j,baseline in enumerate(METHODS):
            for row,metric in enumerate(('completion','total_latency_ns')):
                result=next(r for r in data['paired_comparisons'] if r['robot']==robot and
                    r['method']=='cr_ik_minimal' and r['baseline']==baseline and r['metric']==metric)
                stat='difference' if row==0 else 'ratio';factor=100 if row==0 else 1
                center=result[stat]*factor;lo,hi=np.asarray(result[stat+'_ci'])*factor
                if row==0:extent=max(extent,abs(lo)*1.1,abs(hi)*1.1)
                else:assert min(center,lo,hi)>0
                axes[row,col].errorbar(center,6-j,xerr=[[max(0,center-lo)],[max(0,hi-center)]],
                    fmt='o',ms=3.5,color='#B56528' if baseline=='cr_ik' else '#397DA8',capsize=2,lw=.8)
        for row in range(2):
            ax=axes[row,col];ax.set_yticks(range(7),list(reversed(LABELS)));ax.set_ylim(-.6,6.6)
            ax.grid(axis='x',color='#DDDDDD',lw=.5)
            ax.set_title(f'{"abcd"[2*row+col]}  {ROBOTS[robot]}: n=40 trajectories',loc='left')
        axes[0,col].axvline(0,color='#666666',ls='--',lw=.8)
        axes[0,col].set_xlabel('Minimal − comparator TSR (pp)')
        axes[1,col].axvline(1,color='#666666',ls='--',lw=.8)
        axes[1,col].set_xscale('log');axes[1,col].set_xlim(.1,10)
        axes[1,col].set_xticks([.1,.3,1,3,10],['0.1','0.3','1','3','10'])
        axes[1,col].set_xlabel('Cumulative cost ratio (log scale)')
    for ax in axes[0]:ax.set_xlim(-extent,extent)
    save(fig,out,'paired_completion_and_cost')
    fig,axes=plt.subplots(3,2,figsize=(7.2047244,5.51),sharex='col')
    fig.subplots_adjust(left=.12,right=.97,top=.88,bottom=.10,wspace=.22,hspace=.25)
    traces=[]
    for col,robot in enumerate(ROBOTS):
        folder=root/robot;items=json.loads((folder/'online_inputs.json').read_text())
        item=min([i for i in items if i['family']=='high_curvature'],key=lambda i:i['uid'])
        jobs=json.loads((folder/'summaries.json').read_text())
        selected={m:next(s for s in jobs if s['uid']==item['uid'] and s['method']==m and s['repeat']==0)
                  for m in ('cr_ik','cr_ik_minimal')}
        rows={m:old.read_rows(folder/s['raw_file']) for m,s in selected.items()}
        r=rows['cr_ik_minimal'];time=np.arange(len(r))*.02
        demand=[x['demand'] for x in r];error=[x.get('offline_next_prediction_error',np.nan) for x in r]
        gamma=[x['predicted_gamma'] if x['predicted_gamma'] is not None else np.nan for x in r]
        axes[0,col].plot(time,demand,color='#222222',lw=1,label='Causal demand')
        axes[0,col].plot(time,error,color='#999999',lw=.8,ls='--',label='Actual next error (offline)')
        axes[0,col].plot(time,gamma,color='#397DA8',lw=.8,label='Found local reserve')
        axes[0,col].set_yscale('symlog',linthresh=1)
        axes[0,col].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        axes[1,col].step(time,[x['conic_calls'] for x in r],where='post',color='#397DA8',lw=.8)
        axes[1,col].set_yticks([0,1,2]);axes[1,col].set_ylim(-.1,2.2)
        for method,color in [('cr_ik','#B56528'),('cr_ik_minimal','#397DA8')]:
            axes[2,col].plot(time,[x['total_latency_ns']/1e6 for x in rows[method]],color=color,lw=.8,
                label='Original CR-IK' if method=='cr_ik' else 'Minimal CR-IK')
        axes[2,col].axhline(20,color='#666666',lw=.6,ls='--')
        axes[2,col].set_xlabel('Target time (s)')
        axes[0,col].set_title(f'{ROBOTS[robot]}: first-UID curvature trajectory',loc='left')
        for ax in axes[:,col]:ax.grid(color='#DDDDDD',lw=.5)
        traces.append(dict(robot=robot,uid=item['uid'],time_s=time.tolist(),demand=demand,
            actual_next_error=[None if not np.isfinite(x) else x for x in error],
            found_gamma=[None if not np.isfinite(x) else x for x in gamma],
            sources={str(folder/s['raw_file']):old.sha(folder/s['raw_file']) for s in selected.values()}))
    axes[0,0].set_ylabel('Tolerance units\n(linear to 1, log above 1)')
    axes[1,0].set_ylabel('SOCP calls')
    axes[2,0].set_ylabel('Whole-call latency (ms)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    more,names=axes[2,0].get_legend_handles_labels()
    fig.legend(handles+more,labels+names,loc='upper center',bbox_to_anchor=(.54,1),ncol=3)
    save(fig,out,'causal_demand_and_actual_work')
    old.write_json(out/'trace_source_data.json',traces)
    old.write_json(out/'manifest.json',dict(source=str(source),source_sha256=old.sha(source),
        plotting_source_sha256=old.sha(__file__),width_mm=183,minimum_font_pt=7,preview_dpi=600,
        archetype='quantitative grids',uncertainty='paired trajectory bootstrap 95% intervals; no multiplicity correction',
        trace_selection='first UID in high-curvature family, repeat 0; no success-based selection',
        trace_interpretation='one observed development path; actual next error joined offline; missing gamma left as gaps',
        exclusions='none in full paired analysis; one prespecified-geometry trace per robot for illustration'))


if __name__=='__main__':main()
