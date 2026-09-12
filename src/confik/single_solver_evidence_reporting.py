"""Read-only aggregation of saved commands; uses existing UID statistics."""
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .correction_reserve.study import ROOT,context,read_rows,sha,write_json,clean,utc,summarize
from .correction_reserve.reporting import csv_write,group_table,paired_intervals,LABELS
from .single_solver_evidence import outer_counts
from .types import IKQuery,Pose


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def report(out,cfg):
    folder=out/'reports';folder.mkdir(exist_ok=False)
    summaries=[];arrays={};extras={};first=[];witnesses=[];audit=Counter();completion={}
    targets=json.loads((out/'protocol/online_inputs.json').read_text())
    targetmap={(r['robot'],r['uid']):r for r in targets}
    with gzip.open(folder/'failure_and_timeout_records.jsonl.gz','xt') as events:
        for robot in cfg['robots']:
            origin=out/f'trajectories_{robot}';manifest=json.loads((origin/'completed.json').read_text())
            assert manifest['runs']==3360 and manifest['frames']==504000
            for name,digest in manifest['files'].items():assert sha(origin/name)==digest,name
            _,kin,v,_=context(robot,cfg)
            jobs=json.loads((origin/'summaries.json').read_text());uids={r['uid'] for r in targets if r['robot']==robot}
            assert Counter((s['uid'],s['method'],s['repeat']) for s in jobs)==Counter((u,m,r) for u in uids for m in cfg['methods'] for r in range(3))
            for index,s in enumerate(jobs):
                rows=read_rows(origin/s['raw_file']);assert len(rows)==150
                target=targetmap[(robot,s['uid'])];previous=np.array(target['initial_q']);counts=Counter()
                for frame,r in enumerate(rows):
                    assert r['frame']==frame and r['dt']==target['dt']==.02
                    assert np.array_equal(r['target_position'],target['target_position'][frame])
                    assert np.array_equal(r['target_rotation'],target['target_rotation'][frame])
                    assert np.array_equal(r['previous_q'],previous)
                    query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),previous,.02)
                    q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                    verdict=v.check(q,query)
                    assert verdict.accepted==r['accepted']
                    if verdict.finite_ok:
                        assert abs(verdict.position_error-r['position_error'])<=1e-12
                        assert abs(verdict.orientation_error-r['orientation_error'])<=1e-12
                    assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
                    audit['commands_reverified']+=1;audit['accepted_commands']+=verdict.accepted
                    counts['over_20ms_frames']+=r['total_latency_ns']>20_000_000
                    if not r['accepted']:counts['failure:'+r['failure_kind']]+=1
                    if r['total_latency_ns']>20_000_000 or not r['accepted']:
                        events.write(json.dumps(clean(dict(r,source=str((origin/s['raw_file']).relative_to(ROOT)))),allow_nan=False,separators=(',',':'))+'\n')
                    if s['method'].startswith('single_gn'):
                        cc=outer_counts(r)
                        assert all(value>=0 for value in cc.values()),(s['run_id'],frame,cc)
                        counts.update(cc);counts['iterations']+=r['iterations'];counts['evaluations']+=r['evaluations']
                        counts['internal:'+r['internal_status']]+=1
                        if s['method']!='single_gn_osqp':assert r['box_qp_updates']<=50*30
                    if frame==s['first_failure_frame']:first.append(dict(r,source=str((origin/s['raw_file']).relative_to(ROOT))))
                    if verdict.accepted:
                        step=kin.limits.velocity*.02+v.config.velocity_tolerance
                        physical=(np.abs(q-kin.limits.lower)<1e-8)|(np.abs(q-kin.limits.upper)<1e-8)
                        rate=np.abs(np.abs(q-previous)-step)<1e-8
                        counts['physical_boundary_frames']+=bool(physical.any());counts['rate_boundary_frames']+=bool(rate.any())
                        counts['physical_boundary_joints']+=int(physical.sum());counts['rate_boundary_joints']+=int(rate.sum())
                        previous=q.copy()
                    assert np.array_equal(r['accepted_state_q'],previous)
                calculated=summarize(rows,kin,v)
                for k in ('completion','deadline_completion','accepted_frames','deadline_frames','total_latency_ns','first_failure_frame'):
                    assert calculated[k]==s[k],(s['run_id'],k)
                assert s['accepted_contract_violations']==0
                completion.setdefault(robot,{}).setdefault(s['method'],{}).setdefault(str(s['repeat']),[])
                if s['completion']:
                    completion[robot][s['method']][str(s['repeat'])].append(s['uid'])
                    witnesses.append(dict(robot=robot,uid=s['uid'],method=s['method'],repeat=s['repeat'],frames=150,
                        initial_q=target['initial_q'],deadline_completion=s['deadline_completion'],
                        raw_file=str((origin/s['raw_file']).relative_to(ROOT)),sha256=sha(origin/s['raw_file'])))
                summaries.append(s);extras[s['run_id']]=counts
                arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                    errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
                if index%240==0:print('read-only verify',robot,index+1,'/3360',flush=True)
    assert audit['commands_reverified']==1008000
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    for table in (main,family):
        for r in table:
            ss=[s for s in summaries if s['robot']==r['robot'] and s['method']==r['method'] and (r['family']=='all' or s['family']==r['family'])]
            counts=Counter()
            for s in ss:counts.update(extras[s['run_id']])
            for k in ('backup_rate','changed_from_backup_rate'):r.pop(k,None)
            calls=sum(s['frames'] for s in ss)
            r.update(frame_calls=calls,all_frame_cumulative_ns=sum(s['total_latency_ns'] for s in ss),
                max_latency_ms=max(np.max(arrays[s['run_id']]['latency']) for s in ss)/1e6,
                over_20ms_frames=counts['over_20ms_frames'],
                failure_reasons={k[8:]:n for k,n in counts.items() if k.startswith('failure:')},
                physical_boundary_frames=counts['physical_boundary_frames'],rate_boundary_frames=counts['rate_boundary_frames'])
            if r['method'].startswith('single_gn'):
                r.update({f'mean_{k}':counts[k]/calls for k in ('iterations','evaluations','qp_calls','line_search_evaluations','extra_backtracking_evaluations')})
                r['internal_statuses']={k[9:]:n for k,n in counts.items() if k.startswith('internal:')}
    units=[];pairs=[];changes=[];jointcost=[];metrics=('completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms','successful_prefix')
    keyed=defaultdict(list)
    for s in summaries:keyed[(s['robot'],s['method'],s['uid'])].append(s)
    for robot in cfg['robots']:
        items=[r for r in targets if r['robot']==robot];items=sorted(items,key=lambda r:r['uid']);uids=[r['uid'] for r in items]
        unit={}
        for m in cfg['methods']:
            for item in items:
                ss=keyed[(robot,m,item['uid'])];assert len(ss)==3
                value={k:float(np.mean([s[k] for s in ss])) for k in metrics}
                unit[(m,item['uid'])]=value
                units.append(dict(robot=robot,method=m,uid=item['uid'],family=item['family'],site_id=item['site_id'],repeats=3,**value))
        for base in cfg['methods'][1:]:
            for familyname in ['all']+cfg['fresh']['families']:
                selected=[r for r in items if familyname=='all' or r['family']==familyname];fams=[r['family'] for r in selected]
                for metric in metrics:
                    a=np.array([unit[('single_gn_k1',r['uid'])][metric] for r in selected]);b=np.array([unit[(base,r['uid'])][metric] for r in selected])
                    pairs.append(dict(robot=robot,method='single_gn_k1',baseline=base,family=familyname,metric=metric,trajectories=len(selected),
                        **paired_intervals(a,b,fams,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
                both=[r for r in selected if unit[('single_gn_k1',r['uid'])]['completion']==unit[(base,r['uid'])]['completion']==1]
                at=sum(unit[('single_gn_k1',r['uid'])]['total_latency_ns'] for r in both)
                bt=sum(unit[(base,r['uid'])]['total_latency_ns'] for r in both)
                jointcost.append(dict(robot=robot,method='single_gn_k1',baseline=base,family=familyname,stable_joint_success_uids=len(both),
                    method_cumulative_ns=at,baseline_cumulative_ns=bt,ratio=at/bt if bt else None,uids=[r['uid'] for r in both],
                    scope='Descriptive subset, both methods complete all 3 repeats; not the primary full-sample comparison'))
            for item in items:
                u=item['uid'];a=unit[('single_gn_k1',u)]['completion'];b=unit[(base,u)]['completion']
                changes.append(dict(robot=robot,baseline=base,uid=u,site_id=item['site_id'],family=item['family'],
                    method_fraction=a,baseline_fraction=b,change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=abs(a-b)==1))
    for name,rows in [('main_table',main),('family_table',family),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('joint_success_cost',jointcost),('first_failure_inputs',first),('run_summaries',summaries)]:csv_write(folder/(name+'.csv'),rows)
    write_json(folder/'source_data.json',dict(main=main,family=family,units=units,pairs=pairs,changes=changes,joint_success_cost=jointcost))
    write_json(folder/'completion_uids.json',completion);write_json(folder/'successful_trajectory_index.json',witnesses)
    write_json(folder/'verification.json',dict(audit,accepted_contract_violations=0,solver_calls=0,successful_runs=len(witnesses)))
    qp_report(out,cfg)
    write_json(folder/'manifest.json',dict(created=utc(),reporting_source_sha256=sha(__file__),
        sources={r:sha(out/f'trajectories_{r}/completed.json') for r in cfg['robots']},
        statistics='UID average of 3 nested repeats; 4000 family-stratified paired bootstrap; descriptive unadjusted 95%; 160 independent units/robot. Full sample and all families retained.',
        errors='Only accepted commands; failed and late calls in all cost statistics; acceleration is kinematic finite difference.',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],4),r['cumulative_ms_per_sweep']/1000,flush=True)


def qp_report(out,cfg):
    selection=json.loads((out/'qp_benchmark/selection.json').read_text());rows=csv_rows(out/'qp_benchmark'/selection['source'])
    table=[];pairs=[]
    for robot in cfg['robots']:
        source={str(r['qp_index']):r for r in json.loads((out/f'qp_benchmark/{robot}_qp_metadata.json').read_text())}
        rr=[r for r in rows if r['robot']==robot]
        for r in rr:r['source_active_bounds']=source[r['qp_index']]['active_bounds']
        groups=['all','active','inactive']+cfg['fresh']['families']
        for group in groups:
            sub=[r for r in rr if group=='all' or (group=='active' and int(r['source_active_bounds'])>0) or (group=='inactive' and int(r['source_active_bounds'])==0) or r['family']==group]
            if not sub:continue
            for m in ('active_numpy','osqp_reset','osqp_warm','clip'):
                ss=[r for r in sub if r['method']==m];t=np.array([float(r['total_latency_ns']) for r in ss])
                table.append(dict(robot=robot,group=group,method=m,qp_count=len({r['qp_index'] for r in ss}),calls=len(ss),
                    p50_us=np.percentile(t,50)/1e3,p95_us=np.percentile(t,95)/1e3,p99_us=np.percentile(t,99)/1e3,
                    mean_us=t.mean()/1e3,bad_quality=sum(r['quality_pass']!='True' for r in ss),
                    native_cap_hits=sum(int(r['native_iterations'])>= (50 if m=='active_numpy' else 20000) for r in ss) if m!='clip' else None,
                    **{f'max_{k}':max(float(r[k]) for r in ss) for k in ('box_violation','projected_kkt','normalized_kkt','scaled_objective_gap','raw_projected_kkt','raw_normalized_kkt')},
                    **{f'mean_{k}':np.mean([float(r[k]) for r in ss]) for k in ('reset_ns','conversion_ns','update_ns','solve_ns','check_ns')}))
        uids=sorted({r['uid'] for r in rr});fams=[next(r['family'] for r in rr if r['uid']==u) for u in uids]
        for m in ('osqp_reset','osqp_warm','clip'):
            a=np.array([np.mean([float(r['total_latency_ns']) for r in rr if r['uid']==u and r['method']=='active_numpy']) for u in uids])
            b=np.array([np.mean([float(r['total_latency_ns']) for r in rr if r['uid']==u and r['method']==m]) for u in uids])
            pairs.append(dict(robot=robot,method='active_numpy',baseline=m,independent_source_trajectories=len(uids),
                metric='mean local call ns, equal source UID weight',**paired_intervals(a,b,fams,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
    csv_write(out/'reports/qp_quality_time.csv',table);csv_write(out/'reports/qp_paired_time.csv',pairs)


def figures(out,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':8,
        'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'svg.fonttype':'none','pdf.fonttype':42})
    folder=out/'reports/figures';folder.mkdir(exist_ok=False)
    data=json.loads((out/'reports/source_data.json').read_text());main=data['main'];pairs=data['pairs']
    short=['GN 1','GN 0','Clip','OSQP','TRAC 5','TRAC 20','Pink'];methods=cfg['methods']
    # These figures summarize fixed evidence; none claim a prespecified winner.
    def save(fig,name):
        fig.savefig(folder/f'{name}.svg')
        fig.savefig(folder/f'{name}.pdf')
        fig.savefig(folder/f'{name}.png',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,112/25.4),layout='constrained')
    for i,robot in enumerate(cfg['robots']):
        rr={r['method']:r for r in main if r['robot']==robot};x=np.arange(7)
        for col,(metric,label) in enumerate([('tsr','Complete trajectories (%)'),('dtsr20','All frames admissible within 20 ms (%)')]):
            ax=axes[i,col]
            for j,m in enumerate(methods):
                vals=rr[m]['completion_by_repeat'] if metric=='tsr' else rr[m]['deadline_completion_by_repeat']
                ax.scatter(np.full(3,j),np.array(vals)/160*100,color='#777777',s=8,alpha=.6)
            ax.plot(x,[rr[m][metric]*100 for m in methods],'o-',color='#0072B2',markersize=3,linewidth=.7)
            ax.set_xticks(x,short,rotation=45,ha='right',rotation_mode='anchor');ax.set_ylim(0,103);ax.set_ylabel(label)
        ax=axes[i,2]
        # Native full-sweep repeat range, nested timing variation only.
        for j,m in enumerate(methods):
            ss=[r for r in csv_rows(out/'reports/run_summaries.csv') if r['robot']==robot and r['method']==m]
            vals=np.array([sum(float(r['total_latency_ns']) for r in ss if int(r['repeat'])==rep)/1e9 for rep in range(3)])
            ax.plot([j,j],[min(vals),max(vals)],color='#777777');ax.scatter([j],[vals.mean()],color='#0072B2',s=12)
        ax.set_xticks(x,short,rotation=45,ha='right',rotation_mode='anchor');ax.set_ylabel('Cumulative outer time per sweep (s)');ax.set_ylim(bottom=0)
        for j in range(3):axes[i,j].set_title(f'{chr(97+i*3+j)}  {robot.capitalize()}',loc='left',fontweight='bold')
    save(fig,'task_cost')
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,108/25.4),layout='constrained')
    for i,robot in enumerate(cfg['robots']):
        for j,(metric,label,scale) in enumerate([('completion','TSR difference (percentage points)',100),('deadline_completion','DTSR20 difference (percentage points)',100),('total_latency_ns','Cumulative time ratio',1)]):
            ax=axes[i,j]
            for k,base in enumerate(methods[1:]):
                row=next(r for r in pairs if r['robot']==robot and r['baseline']==base and r['family']=='all' and r['metric']==metric)
                key='ratio' if j==2 else 'difference';c=row[key]*scale;lo,hi=np.array(row[key+'_ci'])*scale
                ax.plot([lo,hi],[k,k],color='#0072B2');ax.plot(c,k,'o',color='#0072B2',markersize=3)
            ax.axvline(1 if j==2 else 0,color='#999999',linewidth=.6,linestyle='--')
            ax.set_yticks(range(6),short[1:]);ax.set_xlabel(label);ax.set_title(f'{chr(97+i*3+j)}  {robot.capitalize()}: GN 1 vs control',loc='left',fontweight='bold')
    save(fig,'paired_contributions')
    table=csv_rows(out/'reports/qp_quality_time.csv')
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,62/25.4),layout='constrained')
    for i,robot in enumerate(cfg['robots']):
        rr={r['method']:r for r in table if r['robot']==robot and r['group']=='all'}
        for j,m in enumerate(('active_numpy','osqp_reset','osqp_warm','clip')):
            vals=[float(rr[m][f'p{p}_us']) for p in (50,95,99)]
            axes[i].plot([j,j],[vals[0],vals[2]],color='#999999');axes[i].plot(j,vals[0],'o',color='#0072B2',markersize=3);axes[i].plot(j,vals[1],'_',color='#D55E00');axes[i].plot(j,vals[2],'^',color='#777777',markersize=3)
        axes[i].set_xticks(range(4),['Active set','OSQP reset','OSQP warm','Clip']);axes[i].set_ylabel('Full local QP call (µs): P50 / P95 / P99');axes[i].set_ylim(bottom=0);axes[i].set_title(f'{chr(97+i)}  {robot.capitalize()}',loc='left',fontweight='bold')
    save(fig,'qp_timing')
    write_json(folder/'figure_contract.json',dict(backend='Python/matplotlib',size_mm=[183,112],exports=['editable SVG','editable PDF','PNG preview'],
        questions=['Task/deadline completion versus whole cost','Paired contribution of each controlled difference','Same-QP full-call time at independently checked quality'],
        evidence=['source_data.json','run_summaries.csv','qp_quality_time.csv'],
        panels='task_cost: point/range nested repeats, not independent CIs; paired_contributions: UID bootstrap95%, 4000 stratified, no multiplicity correction; qp_timing: descriptive pooled quantiles, not uncertainty.',
        selection='All seven settings, both robots; no simulated points or removed failures; family detail in complete CSVs.'))
