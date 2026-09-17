"""Read-only Phase 2 aggregation, evidence checks and recorded-state previews."""
from pathlib import Path
import gzip
import json
import subprocess
import numpy as np
import h5py
from scipy.interpolate import BSpline
from .phase2 import OUT,SOURCE,ROOT,METHODS,sources
from .study import write,sha
from .reporting import csv_write,scene_rows,summary,avg,quantile
from .task import ScanTask
from .models import context
from .phase2_graph import time_lower_bound,transition_controls


def read_gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)


def augment(raw):
    for r in raw:
        root=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        r['quality_completion_time_s']=r.get('simulated_completion_time_s') if r.get('quality_completed') else None
        r['raw_motion_completion_time_s']=r.get('simulated_completion_time_s')
        tracepath=root/'optimizer_trace.json.gz'
        if tracepath.exists():
            tr=read_gz(tracepath)
            r['optimizer_native_status']=tr.get('status')
            r['optimizer_elapsed_s']=tr.get('elapsed')
            r['optimizer_stop_at_budget']=tr.get('status') in ('User_Requested_Stop','Maximum_WallTime_Exceeded')
            r['optimizer_elapsed_beyond_30s']=max(0.,tr.get('elapsed',0.)-30.)
        if not r.get('physical_run'):continue
        task=ScanTask(**json.loads((OUT/'inputs'/r['slot']/'identity.json').read_text())['task'])
        plan=np.load(root/'planned_path.npz')
        with h5py.File(root/'execution.h5') as f:
            t=f['t'][:];s=f['s'][:];q=f['q'][:];qr=f['qr'][:]
        dt=np.diff(t);wait=t[:-1]>plan['t'][-1];scan=task.references(s[:-1])[4]&~wait;turn=~scan&~wait
        r.update(actual_scan_s=float(sum(dt[scan])),actual_transition_s=float(sum(dt[turn])),actual_wait_s=float(sum(dt[wait])),
                 tracking_p95_rad=float(np.quantile(np.max(abs(q-qr),axis=1),.95)),
                 actual_endpoint_error_rad=float(np.max(abs(q[-1]-plan['coeff'][-1]))))
        np.testing.assert_allclose(sum(r[k] for k in ('actual_scan_s','actual_transition_s','actual_wait_s')),t[-1]-t[0],atol=1e-9)
        r['quality_transition_s']=r['actual_transition_s'] if r['quality_completed'] else None


def collect():
    raw=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/*/metrics.json'))]
    if len(raw)!=432:raise RuntimeError(f'Expected 432 completed conditions, found {len(raw)}')
    augment(raw);rows=scene_rows(raw,METHODS)
    for r in rows:
        group=[x for x in raw if x['slot']==r['slot'] and x['method']==r['method']]
        first=next(x for x in group if x['repeat']==0)
        for key in ('first_legal_path_s','selected_path_available_s','adopted_updates','active_control_count'):
            r[key]=avg(group,key)
        r['first_path_time_scope']='Returned densely verified graph-stage path; may upper-bound an earlier internal fitting incumbent' if r['method'] not in ('B0','B1') else 'Frozen baseline did not instrument earliest common-margin incumbent; final returned path time reported separately'
        r['final_returned_legal_path_s']=r['total_plan_wall_s'] if r['planner_feasible_repeat_mean']>0 else None
        for key in ('actual_scan_s','actual_transition_s','actual_wait_s','quality_transition_s','tracking_p95_rad',
                    'endpoint_closure_error_rad','actual_endpoint_error_rad','transitions_actually_modified',
                    'max_along_scan_gap_m','center_error_m_max','incidence_rad_max','line_error_rad_max','standoff_m_max','standoff_m_min'):
            r[key]=first.get(key)
        r['physical_verified_path_available_s']=first.get('selected_path_available_s') if first.get('quality_completed') else None
        r['physical_initialization_s']=first.get('T_init')
        r['actual_optimized_transition_count']=len(first.get('transitions_actually_modified') or [])
    return raw,rows


def main_tables(rows):
    main=[];families=[]
    for robot in ('panda','ur5e'):
        for method in METHODS:
            group=[r for r in rows if r['robot']==robot and r['method']==method]
            d=dict(robot=robot,method=method,**summary(group))
            for key in ('quality_transition_s','actual_scan_s','actual_wait_s','selected_path_available_s','first_legal_path_s'):
                d[key+'_median']=quantile(group,key,.5)
            d['coverage_min_executed']=min((r['valid_coverage_fraction'] for r in group if r.get('physics_executed')),default=None)
            main.append(d)
            for family in ('plane','cylinder','saddle'):
                gg=[r for r in group if r['family']==family]
                families.append(dict(robot=robot,family=family,method=method,**summary(gg),
                    quality_transition_s_median=quantile(gg,'quality_transition_s',.5)))
    return main,families


def paired(rows):
    answer=[]
    for robot in ('panda','ur5e'):
        for base,method in (('B0','Proposed'),('B1','Proposed'),('B2-common','Proposed'),('A1','Proposed'),('A2','Proposed'),('B1','A2')):
            aa={r['slot']:r for r in rows if r['robot']==robot and r['method']==base}
            bb={r['slot']:r for r in rows if r['robot']==robot and r['method']==method}
            for field in ('quality_completed','quality_execution_s','quality_transition_s','total_plan_wall_s','physical_verified_path_available_s'):
                keys=sorted(aa.keys()&bb.keys())
                if field!='quality_completed':keys=[k for k in keys if aa[k].get(field) is not None and bb[k].get(field) is not None]
                if not keys:
                    answer.append(dict(robot=robot,baseline=base,method=method,metric=field,paired_scenes=0,status='not_estimable_no_common_values'));continue
                a=np.array([float(aa[k][field] is True) if field=='quality_completed' else aa[k][field] for k in keys])
                b=np.array([float(bb[k][field] is True) if field=='quality_completed' else bb[k][field] for k in keys])
                labels=np.array([aa[k]['cluster'] for k in keys]);clusters=np.unique(labels)
                rng=np.random.default_rng(2026091607);draws=rng.integers(0,len(clusters),(4000,len(clusters)))
                ac=np.array([np.mean(a[labels==c]) for c in clusters]);bc=np.array([np.mean(b[labels==c]) for c in clusters])
                effect=np.mean(bc-ac);bootstrap=np.mean((bc-ac)[draws],axis=1)
                row=dict(robot=robot,baseline=base,method=method,metric=field,paired_scenes=len(keys),clusters=len(clusters),
                         difference=float(effect),ci_lower=float(np.quantile(bootstrap,.025)) if len(clusters)>1 else None,
                         ci_upper=float(np.quantile(bootstrap,.975)) if len(clusters)>1 else None,
                         status='paired_cluster_bootstrap_95pct' if len(clusters)>1 else 'single_cluster_no_interval',
                         scope='All 12 scenes for quality; jointly observed values for cost; repeats first averaged within scene')
                if field!='quality_completed':
                    ratio=np.mean(bc[draws],axis=1)/np.mean(ac[draws],axis=1)
                    local_ratios=[(b/a)[labels==c] for c in clusters]
                    median_bootstrap=np.array([np.median(np.concatenate([local_ratios[j] for j in draw])) for draw in draws])
                    row.update(mean_ratio=float(np.mean(bc)/np.mean(ac)),median_scene_ratio=float(np.median(b/a)),
                               ratio_ci_lower=float(np.quantile(ratio,.025)) if len(clusters)>1 else None,
                               ratio_ci_upper=float(np.quantile(ratio,.975)) if len(clusters)>1 else None,
                               median_scene_ratio_ci_lower=float(np.quantile(median_bootstrap,.025)) if len(clusters)>1 else None,
                               median_scene_ratio_ci_upper=float(np.quantile(median_bootstrap,.975)) if len(clusters)>1 else None)
                row['interval_scope']='Marginal descriptive 95% cluster intervals for prespecified contrasts; not simultaneous familywise coverage'
                answer.append(row)
    return answer


def graph_tables():
    graphs=[];changes=[];checkpoints=[];sequences=[]
    for p in sorted((OUT/'graphs').glob('*/*/summary.json')):
        s=json.loads(p.read_text());slot=p.parent.parent.name;repeat=int(p.parent.name[1:]);stats=s['statistics']
        fit_status={}
        for kind in ('distance','time'):
            pp=p.parent/kind/'constrained_fit.json.gz'
            fit_status[kind+'_fit_solver_status']=read_gz(pp).get('status') if pp.exists() else 'direct_fit_no_continuous_fitting_solver'
        graphs.append(dict(slot=slot,repeat=repeat,**stats,**fit_status,
            endpoint_pruned_fraction=1-stats['endpoint_retained']/max(1,stats['endpoint_possible']),
            fine_pruned_fraction=1-stats['fine_retained']/max(1,stats['fine_possible']),
            graph_build_s=s['shared_graph_s'],candidate_validation_s=s['library_preparation_s'],
            distance_fit_feasible=s['distance']['feasible'],time_fit_feasible=s['time']['feasible'],
            distance_fit_status=s['distance']['reason'],time_fit_status=s['time']['reason']))
        a=read_gz(p.parent/'distance/selection.json.gz');b=read_gz(p.parent/'time/selection.json.gz')
        sequences.append(dict(slot=slot,repeat=repeat,both_connected=a['feasible'] and b['feasible'],
            identical_selected_sequence=a.get('indices')==b.get('indices'),
            endpoint_sequence_A1=[e['source'] for e in a.get('selected_edges',[])]+([a['selected_edges'][-1]['target']] if a.get('selected_edges') else []),
            endpoint_sequence_A2=[e['source'] for e in b.get('selected_edges',[])]+([b['selected_edges'][-1]['target']] if b.get('selected_edges') else [])))
    for p in sorted((OUT/'runs').glob('*/*/verified_history.json.gz')):
        history=read_gz(p);slot=p.parent.parent.name;method,rr=p.parent.name.rsplit('_r',1)
        task=ScanTask(**json.loads((OUT/'inputs'/slot/'identity.json').read_text())['task'])
        path=np.load(p.parent/'path_initialization.npz');basis=BSpline(path['knots'],np.eye(len(path['coeff'])),3)
        for i,h in enumerate(history):
            before=history[i-1] if i else h;changed=[]
            for j,seg in enumerate(task.segments):
                if not seg[5]:
                    ss=np.linspace(task.knots[j],task.knots[j+1],31)
                    if np.max(abs(basis(ss)@(np.array(h['coeff'])-np.array(before['coeff']))))>1e-8:changed.append(j)
            changes.append(dict(slot=slot,method=method,repeat=int(rr),update=i,available_post_graph_s=h['available_s'],
                before_real_retimed_s=before['objective'],after_real_retimed_s=h['objective'],
                reduction_s=before['objective']-h['objective'],modified_transitions=changed,
                modified_transition_count=len(changed),
                source=h['source'],physical_execution_only_final_repeat0=True))
        cp=json.loads((p.parent/'checkpoint_paths.json').read_text())
        checkpoints.extend(dict(slot=slot,method=method,repeat=int(rr),**x) for x in cp.values())
    return graphs,changes,checkpoints,sequences


def failures(raw):
    rows=[]
    for r in raw:
        reasons=[]
        if not r['planner_feasible']:reasons.append(r['status'])
        if r.get('physical_run'):
            for key,limit,comparison in (('valid_coverage_fraction',.99,'min'),('max_hole_diameter_upper_bound_m',.002,'max'),
                ('max_along_scan_gap_m',.001,'max'),('velocity_utilization_max',1.001,'max'),('acceleration_utilization_max',1.001,'max'),('collision_steps',0,'max')):
                if r.get(key) is not None and ((r[key]<limit) if comparison=='min' else (r[key]>limit)):reasons.append(key)
            if not r['execution_completed']:reasons.append('physical_execution_incomplete')
            if not r['quality_completed'] and not reasons:reasons.append('other_quality_failure')
        if reasons:rows.append(dict(slot=r['slot'],robot=r['robot'],method=r['method'],repeat=r['repeat'],
            status=r['status'],reasons=reasons,physical_run=r['physical_run'],quality_completed=r['quality_completed'],
            completion_time_s=None,elapsed_until_stop_s=r.get('elapsed_until_stop_s'),total_plan_wall_s=r.get('total_plan_wall_s')))
    return rows


def complete_checkpoints(raw, existing):
    """Keep every missing/failed condition; never substitute a late iterate."""
    lookup={(r['slot'],r['method'],r['repeat'],r['post_graph_budget_s']):r for r in existing}
    out=[]
    for r in raw:
        root=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        trace=read_gz(root/'optimizer_trace.json.gz') if r['method']=='B1' and (root/'optimizer_trace.json.gz').exists() else None
        for deadline in (1,5,10,30):
            key=(r['slot'],r['method'],r['repeat'],deadline)
            if key in lookup:
                out.append(dict(**lookup[key],checkpoint_scope='post_graph_optimization',status='verified_plan_checkpoint',common_speed_reserve=.90));continue
            value=None;scope='post_graph_optimization';status=r['status'];available=None;native=None
            if trace is not None:
                h=trace['checkpoints'].get(str(float(deadline)));scope='post_initialization_B1_smoothing'
                if h is not None:
                    native=h['timing']['duration'];reserve=h['timing'].get('speed_reserve',.98)
                    value=native*reserve/.90;available=h['available_s']
                    status='saved_geometry_incumbent_reexpressed_at_common_time_dilation_not_physically_executed'
            elif r['method']=='B0':
                scope='total_B0_planning';available=r.get('total_plan_wall_s')
                if r['planner_feasible'] and available<=deadline:value=r['predicted_execution_s'];status='returned_verified_plan'
            out.append(dict(slot=r['slot'],method=r['method'],repeat=r['repeat'],post_graph_budget_s=deadline,
                checkpoint_scope=scope,best_plan_duration_s=value,native_saved_timing_s=native,
                recorded_available_s=available,status=status,common_speed_reserve=.90,physical_quality_not_yet_verified=True))
    return out


def development_gate(rows):
    index={(r['slot'],r['method']):r for r in rows};by_robot={};comparison=[]
    for robot in ('panda','ur5e','all'):
        pp=[r for r in rows if r['method']=='Proposed' and (robot=='all' or r['robot']==robot)]
        result=dict(scheduled=len(pp),quality_complete=sum(r['quality_completed'] is True for r in pp))
        for base in ('B1','B2-common','A1','A2'):
            good=[(index[(p['slot'],base)],p) for p in pp if p['quality_completed'] and index[(p['slot'],base)]['quality_completed']]
            lost=[p['scene_uid'] for p in pp if not p['quality_completed'] and index[(p['slot'],base)]['quality_completed']]
            gained=[p['scene_uid'] for p in pp if p['quality_completed'] and not index[(p['slot'],base)]['quality_completed']]
            d=dict(robot=robot,baseline=base,common_quality_scenes=len(good),lost_uids=lost,gained_uids=gained)
            for field in ('quality_execution_s','quality_transition_s','total_plan_wall_s'):
                d[field+'_median_ratio']=float(np.median([p[field]/b[field] for b,p in good])) if good else None
            if base=='B2-common':
                certified=[]
                for b,p in good:
                    compare=certified_target_time(b,p,index[(p['slot'],'A2')])
                    if compare['time_ratio'] is not None:certified.append(compare['time_ratio'])
                d['certified_time_to_target_ratio_median']=float(np.median(certified)) if certified else None
                d['certified_scene_count']=len(certified)
            comparison.append(d);result[base]=d
        by_robot[robot]=result
    allr=by_robot['all'];b1=allr['B1'];b2=allr['B2-common']
    def atmost(value,limit):return value is not None and value<=limit
    checks=dict(panda_quality_at_least_10=by_robot['panda']['quality_complete']>=10,
        ur5e_quality_12=by_robot['ur5e']['quality_complete']==12,
        common_B1_transition_reduction_15pct=atmost(b1['quality_transition_s_median_ratio'],.85),
        common_B1_cycle_reduction_5pct=atmost(b1['quality_execution_s_median_ratio'],.95),
        common_B2_cycle_within_105pct=atmost(b2['quality_execution_s_median_ratio'],1.05),
        certified_path_availability_within_50pct_B2=atmost(b2.get('certified_time_to_target_ratio_median'),.50))
    for base in ('A1','A2'):
        d=allr[base]
        checks['increment_over_'+base]=bool(not d['lost_uids'] and (d['gained_uids'] or atmost(d['quality_execution_s_median_ratio'],1-1e-6)))
    return dict(pass_all=all(checks.values()),checks=checks,by_robot=by_robot,
        status='Development evidence only; no formal test authorized or run',
        interpretation='Availability uses own physically qualified final path or the exactly matching common A2 initial execution, credited to BOTH optimizers. Unexecuted checkpoint timing is not a physical certificate.'),comparison


def certified_target_time(b,p,a2):
    """Do not charge B2 for needless improvement after it already meets target.

    A2 physically executes the exact common initial timed path. When that path
    qualifies, BOTH optimizers own this early certificate, not just Proposed.
    Other unexecuted iterates are not labelled physically quality-complete.
    """
    out=dict(slot=p['slot'],robot=p['robot'],B2_final_quality=bool(b['quality_completed']),
             Proposed_final_quality=bool(p['quality_completed']),A2_initial_quality=bool(a2['quality_completed']),
             target_cycle_s=None,B2_certified_planning_s=None,Proposed_certified_planning_s=None,time_ratio=None)
    if not b['quality_completed']:out['status']='B2_final_not_quality_qualified';return out
    target=1.05*b['quality_execution_s'];out['target_cycle_s']=target
    times={}
    for method,row in (('B2-common',b),('Proposed',p)):
        values=[]
        if row['quality_completed'] and row['quality_execution_s']<=target and row.get('physical_verified_path_available_s') is not None:
            values.append(row['physical_verified_path_available_s'])
        same=False
        source=OUT/'runs'/p['slot']/'A2_r0/planned_path.npz'
        init=OUT/'runs'/p['slot']/f'{method}_r0/path_initialization.npz'
        if source.exists() and init.exists():
            aa=np.load(source);bb=np.load(init)
            same=all(np.array_equal(aa[k],bb[k]) for k in ('coeff','knots','s','x','u','t'))
        if same and a2['quality_completed'] and a2['quality_execution_s']<=target:
            values.append(row['physical_initialization_s'])
        out[method+'_initial_execution_exact_match']=same
        times[method]=min(values) if values else None
    out.update(B2_certified_planning_s=times['B2-common'],Proposed_certified_planning_s=times['Proposed'],
        time_ratio=times['Proposed']/times['B2-common'] if times['Proposed'] is not None and times['B2-common'] is not None else None,
        status='Only actually executed matching timed paths certify quality; both methods credited for common A2 initial witness')
    return out


def plots(main,rows,graphs,changes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
        'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    root=OUT/'reports/figures';root.mkdir(parents=True,exist_ok=True)
    write(root/'contract.json',dict(archetype='quantitative grid',backend='Python matplotlib',
        questions=['Does all-scene quality improve?','What is paired qualified execution cost?','What planning work was spent?',
                   'Where does endpoint graph closure stop before continuous feasibility?'],
        units='Scene; timing repeats averaged within scene; no inferential bars on raw-scene plots',
        source='reports CSV; unqualified durations excluded only from cost axes and retained in completion counts',
        adaptation='Style-only inheritance from Phase15; new method/paired fields; no old statistics reused',
        export='Editable SVG/PDF and 300dpi PNG; >=6pt text'))
    colors=['#999999','#6f8190','#887c9c','#ba945a','#6490a7','#378375']
    def save(fig,name):
        fig.savefig(root/f'{name}.svg',bbox_inches='tight');fig.savefig(root/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(root/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,method in enumerate(METHODS):
            d=next(r for r in main if r['robot']==robot and r['method']==method)
            axes[i,0].bar(j,d['quality_completed_scenes'],color=colors[j]);axes[i,0].text(j,d['quality_completed_scenes']+.2,str(d['quality_completed_scenes']),ha='center',fontsize=7)
            for ax,key in ((axes[i,1],'quality_execution_s'),(axes[i,2],'total_plan_wall_s')):
                values=[r[key] for r in rows if r['robot']==robot and r['method']==method and r.get(key) is not None]
                ax.scatter(np.full(len(values),j),values,color=colors[j],s=12,alpha=.7)
                if values:ax.plot([j-.2,j+.2],[np.median(values)]*2,color='black',lw=1)
        axes[i,0].set(ylim=(0,13),yticks=[0,3,6,9,12],ylabel=f'{robot.upper()} quality / 12')
        axes[i,1].set_ylabel('Qualified actual cycle (s)');axes[i,2].set_ylabel('Full charged planning work (s)')
        for ax in axes[i]:ax.set_xticks(range(6),['B0','B1','B2-C','A1','A2','Proposed'],rotation=25,ha='right',rotation_mode='anchor');ax.grid(axis='y',alpha=.15)
    for ax,title in zip(axes[0],('a  All scheduled scenes','b  Qualified scene values','c  All planning conditions')):ax.set_title(title,loc='left')
    save(fig,'phase2_quality_cost')
    fig,axes=plt.subplots(2,2,figsize=(8.5,6),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        pp=[r for r in rows if r['robot']==robot and r['method']=='Proposed']
        counts=[]
        for j,base in enumerate(('B1','B2-common','A1','A2')):
            pairs=[(next(b for b in rows if b['slot']==p['slot'] and b['method']==base),p) for p in pp]
            pairs=[(b,p) for b,p in pairs if b['quality_completed'] and p['quality_completed']]
            counts.append(len(pairs))
            for ax,key in ((axes[i,0],'quality_execution_s'),(axes[i,1],'quality_transition_s')):
                values=[100*(p[key]/b[key]-1) for b,p in pairs]
                ax.scatter(np.full(len(values),j),values,s=16,color=colors[j+1])
                if values:ax.plot([j-.2,j+.2],[np.median(values)]*2,color='black',lw=1)
        for ax in axes[i]:
            ax.axhline(0,color='black',ls='--',lw=.7)
            ax.set_xticks(range(4),[f'{name}\n(n={n})' for name,n in zip(('B1','B2-C','A1','A2'),counts)])
            ax.grid(axis='y',alpha=.15)
        axes[i,0].set_ylabel(f'{robot.upper()} cycle change (%)');axes[i,1].set_ylabel('Transition change (%)')
    axes[0,0].set_title('a  Proposed minus reference: paired quality-complete scenes',loc='left',fontsize=8)
    axes[0,1].set_title('b  Turn-time changes, same paired subset',loc='left',fontsize=8)
    save(fig,'phase2_paired_changes')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    names=sorted(set(g['slot'] for g in graphs));xx=np.arange(len(names))
    for label,offset,color in (('distance',-.15,colors[3]),('time',.15,colors[4])):
        values=[np.mean([g[label+'_fit_feasible'] for g in graphs if g['slot']==n]) for n in names]
        axes[0].bar(xx+offset,values,width=.3,color=color,label=label.capitalize()+' graph')
        failed=np.flatnonzero(np.array(values)==0)
        axes[0].plot(xx[failed]+offset,np.zeros(len(failed)),'x',color=color,ms=4,clip_on=False)
    ticks=[0,3,7,11,15,19,23]
    axes[0].set(ylim=(0,1.15),ylabel='Verified C² fit fraction (3 timing runs)',xticks=ticks,xticklabels=[str(i+1) for i in ticks],xlabel='All 24 fixed scene IDs (source-data order)');axes[0].legend(fontsize=7)
    for method,color in (('A1',colors[3]),('Proposed',colors[5]),('B2-common',colors[2])):
        for slot in names:
            c=[r for r in changes if r['slot']==slot and r['method']==method and r['repeat']==0]
            if c:axes[1].plot([r['available_post_graph_s'] for r in c],[100*(r['after_real_retimed_s']/c[0]['after_real_retimed_s']-1) for r in c],'.-',color=color,alpha=.6,lw=.6)
        axes[1].plot([],[],'.-',color=color,label=method)
    axes[1].set(xlabel='Post-graph validated availability (s)',ylabel='Real TOPPRA cycle change (%)');axes[1].legend(fontsize=7)
    axes[0].set_title('a  Connectivity is not continuous-path feasibility',loc='left',fontsize=8)
    axes[1].set_title('b  Accepted numerical updates, repeat 0',loc='left',fontsize=8)
    save(fig,'phase2_graph_to_path')
    write(root/'scene_order.json',names)


def report():
    raw,rows=collect();main,families=main_tables(rows);intervals=paired(rows)
    graphs,changes,checkpoints,sequences=graph_tables();gate,contrasts=development_gate(rows)
    checkpoints=complete_checkpoints(raw,checkpoints)
    for name,data in [('phase2_main',main),('phase2_families',families),('phase2_scenes',rows),('phase2_raw_conditions',raw),
                      ('graph_statistics',graphs),('transition_improvements',changes),('phase2_checkpoints',checkpoints),
                      ('graph_sequence_comparison',sequences),('phase2_paired_intervals',intervals),('phase2_failures',failures(raw)),('phase2_gain_loss',contrasts)]:
        csv_write(OUT/'reports'/f'{name}.csv',data)
    write(OUT/'reports/development_gate.json',gate)
    idx={(r['slot'],r['method']):r for r in rows}
    target_times=[certified_target_time(idx[(r['slot'],'B2-common')],r,idx[(r['slot'],'A2')]) for r in rows if r['method']=='Proposed']
    csv_write(OUT/'reports/time_to_B2_target.csv',target_times)
    checks=[]
    for p in sorted((OUT/'runs').glob('*/*/optimizer_trace.json.gz')):
        d=read_gz(p);method,repeat=p.parent.name.rsplit('_r',1)
        for i,c in enumerate(d.get('checks',[])):
            checks.append(dict(slot=p.parent.parent.name,method=method,repeat=int(repeat),check_index=i,**c,
                               source=str(p.relative_to(OUT))))
    csv_write(OUT/'reports/transition_candidate_checks.csv',checks)
    plots(main,rows,graphs,changes)
    documents(raw,rows,main,graphs,changes,gate)
    print(json.dumps(gate,ensure_ascii=False,indent=2))


def documents(raw,rows,main,graphs,changes,gate):
    # Narrative is rendered only from completed records, never from design aims.
    def fmt(v):return 'null' if v is None else f'{v:.3f}'
    lines=['# Process Scan Phase 2 开发结果','',
        '本轮为原24开发场景；不是正式测试。所有实际状态和回波来自共同力矩控制器下的MuJoCo仿真，不是实机。',
        f'已保存{len(raw)}个规划条件；预指定repeat0实际执行{sum(r.get("physical_run",False) for r in raw)}项。无结果筛选、替换场景或新边代价。','',
        '|机器人|设置|质量合格/12|实际周期中位/s|换行中位/s|完整规划中位/s|',
        '|---|---|---:|---:|---:|---:|']
    for r in main:lines.append(f"|{r['robot']}|{r['method']}|{r['quality_completed_scenes']}|{fmt(r['quality_execution_s_median'])}|{fmt(r['quality_transition_s_median'])}|{fmt(r['planning_s_median'])}|")
    lines+=['','上表时间是条件合格子集的边际描述，不能直接相减作收益。配对结果与全部失利UID分别在',
        '`phase2_paired_intervals.csv`、`phase2_gain_loss.csv`。场景内先平均三次规划；CAD×放置聚类、方向嵌套。',
        '物理执行仅repeat0，不能把三遍计时当三份物理证据。区间包含零不表示等效。','',
        '主表质量失败的完成时间为null。原始simulated_completion_time_s沿用旧执行器语义，',
        '只表示运动抵达并停稳；它不证明扫描质量合格。CSV另外明确区分raw_motion_completion_time_s',
        '与quality_completion_time_s，不用质量失败的短周期制造收益。95%区间为预定对比的',
        '边际描述区间，不是全部比较的同时覆盖保证。','',
        '## 图、端点闭合与连续路径','',
        f"72个共享图阶段中，距离图形成可行样条{sum(g['distance_fit_feasible'] for g in graphs)}次，时间下界图{sum(g['time_fit_feasible'] for g in graphs)}次。",
        '候选均来自独立参考IK库，首尾层强制共同关节状态。图边的内部离散合法性不能保证',
        '选中序列可由固定96控制点C²参数化表达；若约束拟合未形成合法路径，则记录失败而不',
        '回退到B1、参考最终路径或更换候选。内部选中节点与拟合值偏差另存，首尾仍必须完全闭合。',
        'Panda首个接口检查已出现“图连通但C²拟合失败”，未隐去，也没有据此追加边代价。','',
        '## 局部精修与强参照','',
        'A1/Proposed共用局部支撑限制与同一个构型—时间NLP，A2不精修；B2-common开放',
        '全部变量，逐遍复用时间图的同一可行初值。图阶段没有可行初值时三者共同缺失，',
        '这不是B2数值能力失败。工具允许域、时间模型、密集验收与物理执行未按方法改变。',
        f"记录中有{sum(c['update']>0 for c in changes)}个真正降低全路径TOPPRA时间的已验收更新；逐次幅度、时刻与实际改变的换行段见transition_improvements.csv。",
        '这些是完整重新定时后的计划改善；只有预指定最终路径做物理执行，不能把未执行检查点称为物理质量已通过。','',
        '## 完整任务与消融回答','']
    for robot in ('panda','ur5e'):
        d=gate['by_robot'][robot]
        lines.extend([f"**{robot}**：Proposed质量完成{d['quality_complete']}/12。",''])
        for base in ('B1','A1','A2','B2-common'):
            c=d[base]
            lines.append(f"- 相对{base}：恢复{len(c['gained_uids'])}个、损失{len(c['lost_uids'])}个UID；"
                f"共同成功{c['common_quality_scenes']}个场景中的周期比中位{fmt(c['quality_execution_s_median_ratio'])}，"
                f"换行比中位{fmt(c['quality_transition_s_median_ratio'])}。")
        first=[r for r in raw if r['robot']==robot and r['method']=='Proposed' and r['repeat']==0]
        legal=[r for r in first if r['planner_feasible']]
        changed=sum((r.get('adopted_updates') or 0)>0 for r in legal)
        lines.extend(['',f"局部精修在{len(legal)}个已有合法图初值的repeat0条件中，{changed}个采用了真正缩短重定时时间的更新。"])
        lines.append('')
    lines+=['距离图与时间下界图使用同一节点/边集合，但最小松弛下界不保证最短实际执行时间，',
        '也不保证选中离散序列能在固定样条表示中闭合。A1对照反映这项图选择差异，',
        'A2对照反映局部连续精修；完整任务收益仍由两者之后的共同物理验收决定。',
        '本轮结果不支持把“图连通”替代连续可行性，或把“真实计划时间下降”替代扫描质量。','',
        '## 成本与证据边界','',
        '全部成本包含候选复核、共享图费用、各自约束拟合、优化、密集验证和重定时。冻结',
        '独立多启动库的历史采集属于已有输入准备，未伪称本轮从零候选生成。图构建每遍',
        '仅执行一次，其真实费用完整计入四个消费者；所有相同图方法共享同一剪枝规则。',
        'B0/B1保持旧定义（10秒平滑检查点）；新设置取30秒最好已验证路径。每项全部',
        '实际计算仍计入规划成本，不能把检查点预算称为端到端返回时限。','',
        '首个合法路径时间对图方法指图阶段返回的完整验收路径，可能晚于约束拟合内部',
        '曾出现的合格迭代；冻结B0/B1未单独记录最早公共余量路径，故该字段保留null，',
        '另报最终合法路径返回时间，不补造更精确的首次可用时刻。','',
        '实现修正记录位于implementation_audit/coefficient_bound_fix：首轮发现合法样条',
        '的控制点可能略超B2的系数箱，局部固定不能绕过该约束。共同拟合先投影到同一',
        '系数箱后再做全部真实检查；受影响第三个场景六设置重跑，先前6项及中断图保留。',
        '前两场景36项证明该操作逐位不变，予以保留。目标、节点集、预算和质量均未变。',
        '前36条件单worker默认CPU亲和性；后续两机器人在同型独立P核并行，每worker',
        '单线程、场景内设置串行交错。绝对耗时存在资源阶段差异，配对效应在场景内计算；',
        '不把跨机器人/资源阶段边际时间差写成算法因果效果。详见资源manifest。','',
        '阶段Gate结论见PROCESS_SCAN_PHASE2_GATE.md。本轮结束后停止，不据失败再加入',
        '第三种边代价、场景或新网络，也不运行96正式场景或论文写作。','',
        '复算：`.venv-process-scan/bin/python scripts/run_process_scan_phase2.py report`。',
        '验收：同一入口`verify`。二者不运行新的规划或物理试验。']
    (ROOT/'docs/PROCESS_SCAN_PHASE2_RESULTS.md').write_text('\n'.join(lines)+'\n')
    lines=['# Phase 2 开发Gate','',f"总体：**{'达到全部开发目标' if gate['pass_all'] else '未达到全部开发目标；停止，不进入正式测试'}**。",'',
           '|目标项|结果|','|---|---|']
    lines += [f"|{k}|{'达到' if v else '未达到'}|" for k,v in gate['checks'].items()]
    for robot,d in gate['by_robot'].items():
        lines+=['',f"## {robot}",'',f"Proposed全样本质量完成：{d['quality_complete']}/{d['scheduled']}。",'']
        for base in ('B1','B2-common','A1','A2'):
            a=d[base];lines.append(f"- 相对{base}：共同质量合格{a['common_quality_scenes']}，新增{len(a['gained_uids'])}，损失{len(a['lost_uids'])}；周期比中位{fmt(a['quality_execution_s_median_ratio'])}，换行比{fmt(a['quality_transition_s_median_ratio'])}，全规划成本比{fmt(a['total_plan_wall_s_median_ratio'])}。")
    lines+=['','时间目标按共同质量合格场景的逐场景比值中位数计算，全样本完成率另列。',
        '可物理认证的路径可用时间仅使用最终实际执行路径的第一次验收时刻；未执行检查点',
        '只提供计划代理，不能据其预测时间宣布更早达到物理质量。另将A2实际执行的',
        '同一初始定时路径作为共同证据：若其已经达到105%目标，B2与Proposed都获得',
        '该初始时刻，不能强迫B2承担已无必要的后续改善费用。逐场景见time_to_B2_target.csv。',
        '相对A1/A2的数值差异',
        '不自动证明因果增量；候选相同、图序列和局部更新记录必须共同解释。','',
        '无正式测试授权。本轮无论结果如何均停止，等待用户与ChatGPT验收。']
    (ROOT/'docs/PROCESS_SCAN_PHASE2_GATE.md').write_text('\n'.join(lines)+'\n')
    p=gate['by_robot']['panda'];u=gate['by_robot']['ur5e'];a=gate['by_robot']['all']
    (ROOT/'docs/PROCESS_SCAN_PHASE2_HANDOVER.md').write_text(f'''# Phase 2 交接

当前阶段止于开发结果，不自动执行正式96场景或写论文。

## 验收结论

{len(raw)}次规划、{sum(r.get('physical_run',False) for r in raw)}次预指定MuJoCo力矩执行已经完成。
Proposed质量完成为Panda {p['quality_complete']}/12、UR5e {u['quality_complete']}/12；
Panda相对B1恢复{len(p['B1']['gained_uids'])}个UID、损失{len(p['B1']['lost_uids'])}个。
距离图A1在Panda达到{next(r['quality_completed_scenes'] for r in main if r['robot']=='panda' and r['method']=='A1')}/12。
共同成功场景中，相对B1的换行和周期中位降幅为
{100*(1-a['B1']['quality_transition_s_median_ratio']):.3f}%与{100*(1-a['B1']['quality_execution_s_median_ratio']):.3f}%。
相对B2-common的周期比中位为{a['B2-common']['quality_execution_s_median_ratio']:.3f}。
开发Gate{'通过' if gate['pass_all'] else '未通过'}；完整目标和消融结论见Gate报告。
本轮停止，正式测试不启动，不因局部计划时间确有下降而改写整体判断。

数值/输入验收见`reports/verification.json`，回归测试见
`provenance/verification_tests.xml`，图与视频复算及视觉检查见`reports/figures/QA.md`。

## 交付索引

- 方法：`docs/PROCESS_SCAN_PHASE2_METHOD.md`。
- 结果与Gate：`docs/PROCESS_SCAN_PHASE2_RESULTS.md`、`docs/PROCESS_SCAN_PHASE2_GATE.md`。
- 根目录：`outputs/process_scan/phase2_development/`。
- 输入/节点：`inputs/*/identity.json`、`candidate_library.json.gz`；冻结seal列完整hash。
- 图/全部边下界/选择：`graphs/<scene>/r<repeat>/graph.json.gz`与两代价的`selection.json.gz`。
- 共同初值：图目录中的`time/path.npz`，B2-common/Proposed/A2逐字节复用。
- 规划：`runs/<scene>/<method>_r<repeat>/`，原始优化器/密集检查/TOPPRA/可用时刻完整保留。
- 实际状态：`execution.h5`，1ms反馈；`scan_samples.npz`全部81射线、质量mask。
- 主表/全场景/家族/配对区间/失败/图/逐次更新：`reports/*.csv`。
- 图：`reports/figures/`，真实记录生成，可编辑SVG/PDF及预览PNG。
- 固定视频索引：`reports/videos/index.json`，placement0/u/全部机器人家族方法，repeat0。
- 可复算入口：`scripts/run_process_scan_phase2.py`；`report/render/verify`只读测量，不重跑物理。

历史Task Balance/GN、运动学/验收、Phase1/1.5所有原始结果与论文保持不变。
新Phase2优化内核由已验证B2方程逐行复用，只通过变量边界固定局部范围；
并没有更换损失、引入预测、学习或BoundMPC适配。

参考库获取成本是历史输入准备，不包含在本轮在线规划费用；候选复核和图构建
费用已经完整计入每个消费者。旧方法计时不直接搬进新主表。失败完成时间为null。
每个CAD×放置只有两个方向，聚类样本量有限；物理证据仅共同控制器下的仿真。

请验收全样本完成、共同成功时间、候选/图与B2共享初值、失败和成本后再决定方向。
本轮没有预授权的后续优化、场景扩展或正式测试。
''')


def verify_recorded_scan(root, task, row):
    """Recompute quality from immutable actual ray returns, without simulation."""
    from .execution import coverage
    rays=np.load(root/'scan_samples.npz')
    points=rays['points'];valid=rays['quality_valid'];raw=rays['raw_valid']
    assert points.shape[:2]==valid.shape==raw.shape
    assert points.shape[1:]==(81,3)
    assert np.all(~valid|raw)
    cover,hole,mask=coverage(task,points[valid])
    rawcover,_,_=coverage(task,points[raw])
    for value,key in ((cover,'valid_coverage_fraction'),(rawcover,'raw_coverage_fraction'),
                      (hole,'max_hole_diameter_upper_bound_m')):
        np.testing.assert_allclose(value,row[key],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(mask,rays['coverage_mask'])
    uv=task.uv(rays['s'])[0]
    same_line=np.abs(np.diff(uv[:,1 if task.direction=='u' else 0]))<1e-8
    center=points[:,40];center_ok=raw[:,40]
    pair_ok=same_line&center_ok[:-1]&center_ok[1:]
    gaps=np.linalg.norm(np.diff(center,axis=0),axis=1)[pair_ok]
    maxgap=float(np.max(gaps)) if len(gaps) else np.inf
    np.testing.assert_allclose(maxgap,row['max_along_scan_gap_m'],rtol=0,atol=1e-12)
    expected=bool(row['execution_completed'] and cover>=.99 and hole<=.002 and maxgap<=.001)
    assert expected==row['quality_completed']
    return len(rays['t'])


def verify():
    seal_path=OUT/'runtime_seal.json' if (OUT/'runtime_seal.json').exists() else OUT/'inputs/seal.json'
    seal=json.loads(seal_path.read_text());checks={}
    assert seal['code_hashes']==sources()
    for p,digest in seal['files'].items():assert sha(OUT/p)==digest,p
    old=subprocess.check_output(['git','diff','74d127c712d771b6ba03ab22432ff3ad2f3ac21b','--',
        'src/confik/task_balance_gn.py','src/confik/bounded_gn.py','src/confik/solvers/verifier.py',
        'src/confik/correction_reserve/geometry.py','src/confik/correction_reserve/native_geometry.py',
        'src/confik/process_scan/planning.py','src/confik/process_scan/execution.py','src/confik/process_scan/task.py',
        'src/confik/process_scan/phase15.py','src/confik/process_scan/models.py','src/confik/process_scan/collision.py',
        'src/confik/process_scan/geometry_adapter.py','src/confik/process_scan/study.py',
        'paper','outputs/process_scan/phase1_baselines','outputs/process_scan/phase15_baselines'],cwd=ROOT)
    assert not old;checks['old_evidence_and_kernels_unchanged']=True
    raw,rows=collect();checks['conditions']=len(raw);checks['physics']=sum(r.get('physical_run',False) for r in raw)
    original=json.loads((OUT/'inputs/seal.json').read_text())
    verified_edges=0;physics=0;shared=0;profiles=0
    for p in sorted((OUT/'graphs').glob('*/*/graph.json.gz')):
        g=read_gz(p)
        for layer in g['edges']:
            for e in layer:
                if not e['retained']:continue
                v=e['time_bound'];b=time_lower_bound(np.array(v['q0']),np.array(v['q1']),np.array(v['lower']),np.array(v['upper']),
                    np.array(v['velocity']),np.array(v['acceleration']),v['global_start_at_rest'],v['global_end_at_rest'])
                np.testing.assert_allclose(v['value_s'],b['value_s'],atol=1e-12);verified_edges+=1
    for r in raw:
        root=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        manifest=json.loads((root/'manifest.json').read_text())
        expected=original['code_hashes'] if r['slot'] in ('panda_plane_p0_u','panda_plane_p0_v') else seal['code_hashes']
        for name,digest in expected.items():
            if name in manifest['code_hashes']:assert manifest['code_hashes'][name]==digest,(r['slot'],r['method'],name)
        if r['method'] in ('B2-common','Proposed','A2') and (root/'path_initialization.npz').exists():
            source=OUT/'graphs'/r['slot']/f"r{r['repeat']}"/'time/path.npz'
            assert sha(source)==sha(root/'path_initialization.npz');shared+=1
        if not r['planner_feasible']:continue
        identity=json.loads((OUT/'inputs'/r['slot']/'identity.json').read_text());path=np.load(root/'planned_path.npz')
        np.testing.assert_allclose(path['coeff'][0],identity['common_start'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(path['coeff'][-1],identity['common_end'],atol=1e-10,rtol=0)
        task=ScanTask(**identity['task']);basis=BSpline(path['knots'],np.eye(len(path['coeff'])),3)
        if r['method'] in ('Proposed','A1'):
            initial=np.load(root/'path_initialization.npz')['coeff'];active,_=transition_controls(task,basis)
            fixed=np.setdiff1d(np.arange(len(initial)),active)
            np.testing.assert_allclose(path['coeff'][fixed],initial[fixed],rtol=0,atol=1e-12)
        if r.get('physical_run'):
            adapter,_=context(r['robot'])
            with h5py.File(root/'execution.h5') as f:
                assert not f.attrs['qpos_runtime_overwrite'];np.testing.assert_allclose(np.diff(f['t'][:]),.001,atol=1e-9)
                np.testing.assert_allclose(f['q'][0],identity['common_start'],atol=1e-10)
                np.testing.assert_allclose(np.max(abs(f['dq'][:])/(.5*adapter.public.limits.velocity)),r['velocity_utilization_max'],atol=1e-12)
                np.testing.assert_allclose(np.max(abs(f['qacc'][:]))/2,r['acceleration_utilization_max'],atol=1e-12)
                assert int(np.sum(f['ncon'][:]>0))==r['collision_steps']
            profiles+=verify_recorded_scan(root,task,r)
            physics+=1
    checks.update(recomputed_time_bound_edges=verified_edges,shared_initial_copies=shared,verified_physical_histories=physics,
                  recomputed_actual_scan_profiles=profiles,coverage_holes_and_along_scan_gaps_recomputed=True,
                  failures_not_imputed_zero=True,formal_scenes_run=False,numerical_run_manifest_hashes_checked=True,
                  manifest_note='Legacy baseline manifests also enumerate read-only reporting files that are not called during run; those incidental hashes are not numerical implementation hashes.')
    write(OUT/'reports/verification.json',checks);print(checks)
    write(OUT/'delivery_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='delivery_manifest.json'},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),verification=checks))


def render():
    from .phase2_rendering import render as run
    run()
