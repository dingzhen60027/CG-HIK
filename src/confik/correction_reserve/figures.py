"""Quantitative figure contract: completion, cost and actual correction ability.

Archetype: quantitative grid. No illustrative/simulated observations. Every
panel uses all relevant frozen units, with paired comparisons where appropriate.
Python-only export: editable SVG/PDF and 600-dpi PNG, 183 mm overall width,
minimum 7 pt labels (no reduced mathematical subscripts). Figures test the
reserve hypothesis; their titles do not presume it succeeded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import read_rows,write_json,sha
from .reporting import LABELS

ORDER=['trac_task_5ms','trac_task_20ms','pink_qp','ranged_ik_upstream','ranged_ik_positive_range',
       'two_step_predictive','cr_ik','single_step_reserve','two_step_sigma']
SHORT=['TRAC-IK 5 ms','TRAC-IK 20 ms','Pink QP','RangedIK original','RangedIK range adapter',
       'Two-step predictive','CR-IK','Single-step reserve','Two-step sigma-min']
FOCUS=['trac_task_5ms','two_step_predictive','cr_ik']
COLORS={'trac_task_5ms':'#666666','two_step_predictive':'#397DA8','cr_ik':'#B56528'}
FAMILIES=['smooth','near_singular','joint_limit_return','high_curvature']
FAMILY_LABELS=['Smooth','Near singular','Joint-limit return','Curvature / speed']
ROBOT_LABEL={'panda':'Panda','ur5e':'UR5e'}


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,
        'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False})


def save(fig,folder,name):
    fig.savefig(folder/f'{name}.svg',facecolor='white')
    fig.savefig(folder/f'{name}.pdf',facecolor='white')
    fig.savefig(folder/f'{name}.png',dpi=600,facecolor='white')
    plt.close(fig)


def primary(data,out):
    fig,axes=plt.subplots(2,2,figsize=(7.2047244,6.1023622),sharey='row')
    fig.subplots_adjust(left=.24,right=.98,bottom=.09,top=.94,wspace=.18,hspace=.30)
    rng=np.random.default_rng(981009902)
    for col,robot in enumerate(('panda','ur5e')):
        ax=axes[0,col];lat_ax=axes[1,col]
        units=[u for u in data['units'] if u['robot']==robot]
        for j,method in enumerate(ORDER):
            row=next(r for r in data['main'] if r['robot']==robot and r['method']==method)
            rows=[u for u in units if u['method']==method]
            y=len(ORDER)-1-j
            # Bootstrap independent UIDs; all nested searches were averaged first.
            groups=[np.flatnonzero(np.array([r['family'] for r in rows])==f) for f in FAMILIES]
            idx=np.concatenate([rng.choice(g,(4000,len(g))) for g in groups],axis=1)
            for metric,offset,color,marker,label in (
                ('completion',.12,'#397DA8','o','TSR'),
                ('deadline_completion',-.12,'#B56528','s','DTSR20')):
                values=np.array([r[metric] for r in rows])*100
                center=values.mean();lo,hi=np.percentile(values[idx].mean(axis=1),[2.5,97.5])
                ax.errorbar(center,y+offset,xerr=np.array([[max(0,center-lo)],[max(0,hi-center)]]),
                    fmt=marker,color=color,markersize=3,capsize=2,lw=.7,label=label if j==0 else None)
            # Diamond is the pooled P95, a descriptive finite-benchmark quantile,
            # not an inferential mean; individual trajectory data stay in CSV.
            color=COLORS.get(method,'#888888')
            lat_ax.scatter(row['p95_ms'],y,color=color,s=19,marker='D')
            lat_ax.text(row['p95_ms']*1.15,y,f"{row['p95_ms']:.2f}",va='center',fontsize=7,color=color)
        for a in (ax,lat_ax):
            a.set_yticks(range(len(ORDER)),list(reversed(SHORT)))
            a.set_ylim(-.6,len(ORDER)-.4);a.grid(axis='x',color='#DDDDDD',lw=.5)
        ax.set_xlim(-3,106);ax.set_xlabel('Complete trajectories (%)')
        ax.set_title(f'{"a" if col==0 else "b"}  {ROBOT_LABEL[robot]} — n={len(rows)} trajectories',loc='left')
        ax.legend(loc='upper left',fontsize=7)
        assert all(r['p95_ms']>0 for r in data['main'])
        lat_ax.set_xscale('log');lat_ax.set_xlim(.1,42);lat_ax.set_xlabel('Pooled frame P95 (ms, log scale)')
        lat_ax.set_xticks([.1,1,10,20],['0.1','1','10','20'])
        lat_ax.axvline(20,color='#555555',ls='--',lw=.7)
        lat_ax.set_title(f'{"c" if col==0 else "d"}  Whole-call timing, including failures',loc='left')
    save(fig,out,'completion_and_latency')


def family_comparison(data,out):
    fig,axes=plt.subplots(1,2,figsize=(7.2047244,3.2677165),sharey=True)
    fig.subplots_adjust(left=.23,right=.98,bottom=.18,top=.87,wspace=.2)
    for col,robot in enumerate(('panda','ur5e')):
        for j,family in enumerate(FAMILIES):
            units=[u for u in data['units'] if u['robot']==robot and u['family']==family]
            cr={u['uid']:u['completion'] for u in units if u['method']=='cr_ik'}
            for baseline,offset,marker,color in [('trac_task_5ms',.13,'o','#666666'),
                                                ('two_step_predictive',-.13,'s','#397DA8')]:
                base={u['uid']:u['completion'] for u in units if u['method']==baseline}
                d=np.array([cr[uid]-base[uid] for uid in sorted(cr)])*100
                rng=np.random.default_rng(981009902+j)
                boot=d[rng.integers(0,len(d),(4000,len(d)))].mean(axis=1)
                lo,hi=np.percentile(boot,[2.5,97.5]);center=d.mean()
                axes[col].errorbar(center,3-j+offset,xerr=np.array([[max(0,center-lo)],[max(0,hi-center)]]),
                    fmt=marker,ms=4,color=color,capsize=2,lw=.8,label=LABELS[baseline] if j==0 else None)
        axes[col].axvline(0,color='#AAAAAA',lw=.8)
        axes[col].set_yticks(range(4),list(reversed(FAMILY_LABELS)));axes[col].set_ylim(-.6,3.6)
        axes[col].set_xlabel('CR-IK completion difference (pp)')
        axes[col].set_title(f'{"a" if col==0 else "b"}  {ROBOT_LABEL[robot]}',loc='left')
        axes[col].grid(axis='x',color='#DDDDDD',lw=.5)
    limits=np.array([a.get_xlim() for a in axes]);bound=max(5,np.max(np.abs(limits)))
    for a in axes:a.set_xlim(-bound,bound)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.99),ncol=2)
    save(fig,out,'paired_family_completion')


def mechanism(folder,out):
    rows=json.loads((folder/'summaries.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.2047244,3.5433071),sharey=True)
    fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.85,wspace=.18)
    counts={}
    for col,robot in enumerate(('panda','ur5e')):
        for method,marker in zip(FOCUS,['o','s','^']):
            valid=[r for r in rows if r['robot']==robot and r['method']==method and r['current_accepted']
                   and r['predicted_gamma'] is not None and r['all_directions_radius'] is not None]
            counts[f'{robot}/{method}']=len(valid)
            axes[col].scatter([r['predicted_gamma'] for r in valid],[r['all_directions_radius'] for r in valid],
                marker=marker,s=24,color=COLORS[method],alpha=.72,label=LABELS[method])
        axes[col].set_ylim(-.25,4.8);axes[col].set_yticks([0,.5,1,2,4],['0','0.5','1','2','≥4'])
        axes[col].set_xlabel('Local linear correction reserve γ')
        axes[col].set_title(f'{"a" if col==0 else "b"}  {ROBOT_LABEL[robot]}: 12 states / 4 trajectories',loc='left')
        axes[col].grid(color='#DDDDDD',lw=.5)
    axes[0].set_ylabel('Found radius across all signed axes\n(all three searches succeed; tolerance units)')
    axes[1].legend(loc='upper left')
    save(fig,out,'predicted_and_empirical_reserve')
    write_json(out/'mechanism_panel_counts.json',counts)


def representative(folders,out):
    source=[]
    fig,axes=plt.subplots(3,2,figsize=(7.2047244,6.1023622),sharex='col')
    fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.90,wspace=.25,hspace=.25)
    for col,folder in enumerate(map(Path,folders)):
        inputs=json.loads((folder/'online_inputs.json').read_text())
        item=min([i for i in inputs if i['family']=='high_curvature'],key=lambda i:i['uid'])
        summaries=json.loads((folder/'summaries.json').read_text())
        for method in FOCUS:
            run=next(r for r in summaries if r['uid']==item['uid'] and r['method']==method and r['repeat']==0)
            rows=read_rows(folder/run['raw_file']);t=np.arange(len(rows))*.02
            residual=np.array([max((r['position_error'] if r['position_error'] is not None else np.nan)/.001,
                                  (r['orientation_error'] if r['orientation_error'] is not None else np.nan)/np.deg2rad(.5)) for r in rows])
            motion=np.array([r.get('velocity_utilization',np.nan) for r in rows],float)
            latency=np.array([r['total_latency_ns']/1e6 for r in rows])
            for ax,value in zip(axes[:,col],[residual,motion,latency]):
                ax.plot(t,value,color=COLORS[method],lw=.8,label=LABELS[method])
            source.append(dict(robot=item['robot'],uid=item['uid'],method=method,raw_file=str(folder/run['raw_file']),
                               raw_sha256=sha(folder/run['raw_file']),time_s=t.tolist(),
                               returned_residual_utilization=residual.tolist(),returned_step_utilization=motion.tolist(),latency_ms=latency.tolist()))
            if run['first_failure_frame'] is not None:
                ft=run['first_failure_frame']*.02
                axes[0,col].axvline(ft,color=COLORS[method],ls=':',lw=.8)
        axes[0,col].set_yscale('symlog',linthresh=1);axes[0,col].axhline(1,color='#555555',ls='--',lw=.7)
        from matplotlib.ticker import ScalarFormatter
        axes[0,col].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        axes[1,col].axhline(1,color='#555555',ls='--',lw=.7)
        axes[2,col].axhline(20,color='#555555',ls='--',lw=.7)
        axes[0,col].set_title(f'{ROBOT_LABEL[item["robot"]]} — fixed first-UID curvature path',loc='left')
        axes[2,col].set_xlabel('Target time (s)')
        for ax in axes[:,col]:ax.grid(color='#DDDDDD',lw=.5)
    axes[0,0].set_ylabel('Returned pose tolerance use\n(linear to 1, log above 1)')
    axes[1,0].set_ylabel('Returned joint-step use')
    axes[2,0].set_ylabel('Whole-call latency (ms)')
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.99),ncol=3)
    save(fig,out,'representative_tracking')
    write_json(out/'representative_source_data.json',source)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mechanism',default='outputs/correction_reserve_ik/development_mechanism')
    p.add_argument('--runs',nargs='+',required=True);args=p.parse_args()
    report=Path(args.report);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=json.loads((report/'source_data.json').read_text());style()
    primary(data,out);family_comparison(data,out);mechanism(Path(args.mechanism),out);representative(args.runs,out)
    write_json(out/'figure_manifest.json',dict(source_data=str(report/'source_data.json'),
        source_sha256=sha(report/'source_data.json'),plotting_source_sha256=sha(__file__),
        backend='matplotlib, editable SVG/PDF, 600 dpi PNG',
        claim='test whether current/next reserve yields completion and nonlinear correction benefits, with costs visible',
        uncertainty='TSR/DTSR and paired completion: trajectory-bootstrap 95% intervals; frame P95 descriptive, not an inferential mean',
        selection='all independent units for aggregate plots; representative trace selected by UID, not outcomes; mechanism fixed before probes',
        exclusions='mechanism scatter requires an admissible current and a found nominal/unperturbed next; counts saved; all failures remain in mechanism and main tables',
        limits='local gamma is not nonlinear certification; empirical radius is censored at four tolerance units'))


if __name__=='__main__':main()
