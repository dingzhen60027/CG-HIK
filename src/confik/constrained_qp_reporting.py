"""Read-only tables/one mechanism figure for the fixed local-QP follow-up."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from .correction_reserve.study import read_rows,write_json,sha,ROOT
from .correction_reserve.reporting import csv_write


def strata(c):
    a=c['reference_activity'];n=a['active_count']
    labels=['all','unqualified' if not a['qualified'] else 'unconstrained' if n==0 else 'one_active' if n==1 else 'multiple_active']
    if c['internal_additions'] or c['internal_releases']:labels.append('internal_set_change')
    if c.get('sequence_additions',0) or c.get('sequence_releases',0):labels.append('sequence_set_change')
    return labels


def descriptive_ratio(pairs):
    """Repeat -> QP -> UID averages supplied by caller; UID is bootstrap unit."""
    x=np.array(pairs,float);ratio=float(x[:,0].mean()/x[:,1].mean())
    if len(x)<2:return ratio,None,None
    rng=np.random.default_rng(2026091502)
    ids=rng.integers(0,len(x),(4000,len(x)))
    boot=x[ids].mean(axis=1);ratios=boot[:,0]/boot[:,1]
    return ratio,*np.quantile(ratios,[.025,.975]).tolist()


def report(out):
    out=Path(out);folder=out/'reports';folder.mkdir(exist_ok=False)
    bench=out/'benchmark_quality_cap'
    classification=read_rows(bench/'classification.jsonl.gz')
    classes={(c['robot'],c['cohort'],c['index']):c for c in classification}
    calls=[r for f in sorted(bench.glob('*_calls.jsonl.gz')) for r in read_rows(f)]
    groups=defaultdict(list)
    for r in calls:
        for group in strata(classes[r['robot'],r['cohort'],r['index']]):
            groups[r['robot'],r['cohort'],group,r['method']].append(r)
    tables=[];paired=[];units=[]
    for (robot,cohort,group,method),rs in sorted(groups.items()):
        row=dict(robot=robot,cohort=cohort,group=group,method=method,
            unique_qps=len({r['index'] for r in rs}),source_uids=len({r['uid'] for r in rs}),
            calls=len(rs),quality_failed_calls=sum(not r['quality_pass'] for r in rs),
            quality_failed_qps=len({r['index'] for r in rs if not r['quality_pass']}),
            max_box=max(r['raw_box_violation'] for r in rs),
            max_raw_kkt=max(r['raw_projected_kkt'] for r in rs),
            max_normalized_kkt=max(r['raw_normalized_kkt'] for r in rs),
            max_scaled_objective_gap=max(r['scaled_objective_gap'] for r in rs))
        for metric in ['kernel_ns','return_ns','common_check_ns','total_ns']:
            values=np.array([r[metric] for r in rs])/1000
            row.update({f'{metric[:-3]}_{s}_us':float(v) for s,v in zip(['mean','p50','p95','p99'],[values.mean(),*np.quantile(values,[.5,.95,.99])])})
        for initial in [True,False]:
            vals=[r['total_ns']/1000 for r in rs if r['first_in_uid']==initial]
            row[('first' if initial else 'steady')+'_call_mean_us']=float(np.mean(vals)) if vals else None
        tables.append(row)
        byuid=defaultdict(list)
        for r in rs:byuid[r['uid']].append(r)
        for uid,subset in byuid.items():
            units.append(dict(robot=robot,cohort=cohort,group=group,method=method,uid=uid,
                unique_qps=len({r['index'] for r in subset}),
                **{k:float(np.mean([r[k] for r in subset])) for k in ['kernel_ns','return_ns','total_ns']},
                all_quality_pass=all(r['quality_pass'] for r in subset)))
    for robot in ['panda','ur5e']:
        for cohort in ['natural','targeted']:
            for group in sorted({k[2] for k in groups if k[:2]==(robot,cohort)}):
                a=groups.get((robot,cohort,group,'active_numpy'),[])
                for method in ['qpoases','osqp_cached','clip']:
                    b=groups.get((robot,cohort,group,method),[])
                    aidx=defaultdict(list);bidx=defaultdict(list)
                    for r in a:aidx[r['index']].append(r)
                    for r in b:bidx[r['index']].append(r)
                    valid=[i for i in aidx if all(r['quality_pass'] for r in aidx[i]+bidx[i])]
                    for metric in ['kernel_ns','return_ns','total_ns']:
                        uidpairs=defaultdict(list)
                        for i in valid:
                            uidpairs[aidx[i][0]['uid']].append([np.mean([r[metric] for r in aidx[i]]),np.mean([r[metric] for r in bidx[i]])])
                        if not uidpairs:continue
                        pairs=[np.mean(v,axis=0).tolist() for v in uidpairs.values()]
                        estimate,lo,hi=descriptive_ratio(pairs)
                        paired.append(dict(robot=robot,cohort=cohort,group=group,comparison='active_numpy / '+method,
                            metric=metric,all_qps=len(aidx),quality_paired_qps=len(valid),
                            source_uids=len(pairs),ratio=estimate,ci95_low=lo,ci95_high=hi,
                            unit='UID equal-weight; five calls averaged within each QP; QPs averaged within UID'))
    census=[]
    for robot in ['panda','ur5e']:
        allmeta=read_rows(out/f'capture/{robot}_chronological_metadata.jsonl.gz')
        for group in ['all','unconstrained','one_active','multiple_active','sequence_set_change']:
            rows=[r for r in allmeta if group=='all' or
                  (group=='unconstrained' and r['active_count']==0) or
                  (group=='one_active' and r['active_count']==1) or
                  (group=='multiple_active' and r['active_count']>1) or
                  (group=='sequence_set_change' and r['sequence_additions']+r['sequence_releases']>0)]
            census.append(dict(robot=robot,group=group,qps=len(rows),source_uids=len({r['uid'] for r in rows}),
                physical_active_qps=sum(bool(r['physical_joints']) for r in rows),
                rate_active_qps=sum(bool(r['rate_joints']) for r in rows),
                trust_active_qps=sum(bool(r['trust_joints']) for r in rows),
                weak_contacts=sum(r['weak_contacts'] for r in rows),
                unqualified=sum(not r['qualified'] for r in rows)))
    cases=json.loads((out/'mechanism/six_same_input_cases.json').read_text());local=[]
    for c in cases:
        d=np.array(c['bounded_direction']);clip=np.array(c['clipped_direction'])
        row={k:c[k] for k in ['robot','site_id','uid','frame','outer_iteration','previous_q_max_difference','before_cost']}
        row.update(active_joints=json.dumps([i for i,s in enumerate(c['state']) if s]),
            physical_joints=json.dumps(c['physical_joints']),rate_joints=json.dumps(c['rate_joints']),
            trust_joints=json.dumps(c['trust_joints']),direction_difference_norm=float(np.linalg.norm(d-clip)),
            other_joint_redistribution_norm=float(np.linalg.norm(c['free_joint_redistribution'])))
        for label,method in [('bounded','single_gn_k1'),('clip','single_gn_clip')]:
            first=c['trials'][label][0]
            accepted=next((t for t in c['trials'][label] if t['line_search_accept']),None)
            row[label+'_fullstep_cost']=first['task_cost'];row[label+'_fullstep_task_verified']=first['task_verified']
            row[label+'_backtrack_alpha']=accepted['alpha'] if accepted else None
            row[label+'_historical_first_failure']=c['historical_outcomes'][method][0]['first_failure']
            row[label+'_historical_complete_runs']=sum(s['complete'] for s in c['historical_outcomes'][method])
        local.append(row)
    initial=[]
    for file in sorted((out/'benchmark').glob('*_calls.jsonl.gz')):
        rs=read_rows(file)
        for method in ['active_numpy','osqp_cached','qpoases','clip']:
            ss=[r for r in rs if r['method']==method]
            initial.append(dict(robot=ss[0]['robot'],cohort=ss[0]['cohort'],method=method,
                failed_qps=len({r['index'] for r in ss if not r['quality_pass']}),
                failed_calls=sum(not r['quality_pass'] for r in ss),calls=len(ss)))
    csv_write(folder/'quality_time.csv',tables)
    csv_write(folder/'paired_time.csv',paired)
    csv_write(folder/'uid_timing_units.csv',units)
    csv_write(folder/'replay_census.csv',census)
    csv_write(folder/'local_to_trajectory.csv',local)
    csv_write(folder/'original_cap_quality.csv',initial)
    setups=json.loads((bench/'setups.json').read_text());csv_write(folder/'workspace_setup.csv',setups)
    write_json(folder/'source_data.json',dict(quality_time=tables,paired=paired,census=census,local=local,initial_quality=initial,setups=setups))
    figure(out,cases)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write_json(folder/'manifest.json',dict(files=hashes,new_ik_performance_runs=0,
        old_outputs_untouched=True,figure_data='mechanism/six_same_input_cases.json',
        code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/run_constrained_qp_mechanism.py',ROOT/'src/confik/constrained_qp_reference.py']}))
    print('reports',len(tables),'table rows',len(paired),'paired summaries',flush=True)


def figure(out,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(7.2047,4.0945),gridspec_kw={'width_ratios':[1.35,1,1]})
    fig.subplots_adjust(left=.16,right=.985,bottom=.30,top=.83,wspace=.32)
    labels=[('P' if c['robot']=='panda' else 'U')+' '+c['site_id'][-3:]+' / f'+str(c['frame']) for c in cases]
    differences=np.full((6,7),np.nan)
    for i,c in enumerate(cases):differences[i,:len(c['state'])]=np.array(c['bounded_direction'])-c['clipped_direction']
    bound=float(np.nanmax(abs(differences)))
    im=axes[0].imshow(differences,aspect='auto',cmap='PuOr_r',norm=SymLogNorm(.01,vmin=-bound,vmax=bound))
    for i,c in enumerate(cases):
        for j,s in enumerate(c['state']):
            if s:axes[0].add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor='black',lw=1.1))
    axes[0].set_yticks(range(6),labels);axes[0].set_xticks(range(7),range(1,8))
    axes[0].set_xlabel('Joint (black outline: active)')
    cax=fig.add_axes([.175,.155,.25,.026]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_ticks([-1,-.1,0,.1,1],labels=['−1','−0.1','0','0.1','1'])
    colors={'bounded':'#21618C','clip':'#A75A35'}
    for i,c in enumerate(cases):
        xs=[c['trials'][m][0]['task_cost']/c['before_cost'] for m in ['bounded','clip']]
        assert np.all(np.asarray(xs)>0)
        axes[1].plot(xs,[i,i],color='0.75',lw=.8)
        for x,m,marker in zip(xs,['bounded','clip'],['o','s']):
            axes[1].plot(x,i,marker,color=colors[m],ms=4,label=m.capitalize() if i==0 else None)
        for m,method,dy,marker in [('bounded','single_gn_k1',-.10,'o'),('clip','single_gn_clip',.10,'s')]:
            o=c['historical_outcomes'][method][0];x=o['first_failure'] if not o['complete'] else 150
            axes[2].plot([0,x],[i+dy,i+dy],color=colors[m],lw=1,alpha=.7)
            axes[2].plot(x,i+dy,marker,color=colors[m],ms=4)
    axes[1].set_xscale('log');axes[1].set_xlabel('Full-step FK cost / initial cost')
    axes[1].set_xticks([.0001,.001,.01,.1],['0.0001','0.001','0.01','0.1'])
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.23),fontsize=7,ncol=2,frameon=False)
    axes[2].set_xlim(0,156);axes[2].set_xticks([0,50,100,150],['0','50','100','All 150'])
    axes[2].set_xlabel('Historical successful prefix')
    for ax,title in zip(axes,['a  Bounded − clip, scaled step','b  Nonlinear residual reduction','c  Recorded later outcome']):
        ax.set_title(title,fontsize=7.3,fontweight='bold',loc='left',pad=12)
        ax.set_ylim(5.5,-.5)
    for ax in axes[1:]:ax.set_yticks(range(6),['']*6);ax.grid(axis='x',color='.9',lw=.5)
    fig.text(.16,.045,'Six prescribed cases; exact common previous state at each local comparison.\nAll three historical repeats agree. Lower local residual does not guarantee completion (U 131).',fontsize=7)
    folder=out/'reports'
    fig.savefig(folder/'local_to_tracking.svg',facecolor='white')
    fig.savefig(folder/'local_to_tracking.pdf',facecolor='white')
    fig.savefig(folder/'local_to_tracking.png',dpi=600,facecolor='white')
    plt.close(fig)
