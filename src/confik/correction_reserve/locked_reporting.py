"""Read-only verification and full paired trajectory reporting for elastic CR-IK."""
import json
from collections import Counter,defaultdict
import numpy as np
from ..types import Pose,IKQuery
from . import study as old
from . import reporting as common
from .locked_study import configurations,verify_seal,PRIMARY,ANALYTIC
ELASTIC={ANALYTIC:0.,PRIMARY:.25}
from .elastic_runtime import improves
from .minimal_demand import PredictionDemand
from .native_geometry import NativeGeometry
from .geometry import task_scale,reserve_value

LABELS={**common.LABELS,'cr_ik_minimal':'Hard-demand Minimal',
    ANALYTIC:'Same-core analytic zero reserve cost',
    'cr_ik_elastic_mu025':'Elastic CR-IK (mu=0.25)',
    'cr_ik_elastic_mu1':'Elastic CR-IK (mu=1)',
    'cr_ik_elastic_mu4':'Elastic CR-IK (mu=4)'}
METHODS_NEW=[PRIMARY]


def avg(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def actual_pair(kin,v,query,pred,q,z,step,scale,demand,qb,zi,mu):
    c=v.check(q,query);n=v.check(z,IKQuery(pred,q,.02))
    if not c.accepted or not n.accepted:return dict(legal=False)
    gamma,mapping,slack=reserve_value(kin,pred,q,z,scale,step)
    xi=max(0.,demand-gamma)
    cost=.5*float(np.sum(((q-qb)/step)**2)+np.sum(((z-zi)/step)**2))+.5*mu*(xi/demand)**2
    met=bool(mapping.full_rank and np.all(demand*np.linalg.norm(mapping.matrix,axis=1)<=slack))
    return dict(legal=True,gamma=gamma,xi=xi,cost=cost,met=met,full_rank=mapping.full_rank)


def close(a,b):
    assert np.isclose(a,b,atol=1e-12,rtol=1e-10),(a,b)


def aggregate():
    cfg,base,root=configurations();verify_seal();out=root/'reports';out.mkdir(exist_ok=False)
    parameters=json.loads((root/'protocol/demand_parameters.json').read_text())
    summaries=[];arrays={};completion={};failures=[];audits=[];sources={};witnesses={}
    expected_jobs={(i["uid"],m,r) for i in json.loads((root/"protocol/online_inputs.json").read_text()) for m in cfg["methods"] for r in range(3)}
    observed_jobs=set()
    decisions=Counter();fallbacks=Counter();native_status=Counter();trials_count=Counter();timing=defaultdict(list)
    conditional=defaultdict(list);extras=defaultdict(lambda:defaultdict(list));focus=[];trial_examples=[]
    for robot in cfg['robots']:
        folder=root/robot
        manifest=json.loads((folder/'completed.json').read_text());assert manifest['runs']==2880 and manifest['frames']==864000
        sources[str(folder/'completed.json')]=old.sha(folder/'completed.json')
        _,kin,v,urdf=old.context(robot,base);native=NativeGeometry(kin,urdf)
        scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
        items={i['uid']:i for i in json.loads((root/'protocol/online_inputs.json').read_text())}
        counts=Counter();jobs=json.loads((folder/'summaries.json').read_text())
        for index,s in enumerate(sorted(jobs,key=lambda s:s['run_id'])):
            identity=(s['uid'],s['method'],s['repeat'])
            assert identity in expected_jobs and identity not in observed_jobs
            observed_jobs.add(identity)
            assert s['frames']==300
            raw=folder/s['raw_file'];assert old.sha(raw)==manifest['files'][s['raw_file']]
            rows=old.read_rows(raw);item=items[s['uid']];previous=np.array(item['initial_q']);method=s['method'];key=(robot,method)
            causal=PredictionDemand(scale,parameters[robot]['minimum'],parameters[robot]['initial']) if 'demand' in rows[0] else None
            run_cond=defaultdict(lambda:[0,0,0]);mu=ELASTIC.get(method)
            assert len(rows)==300
            for t,r in enumerate(rows):
                assert r['frame']==t and r['dt']==.02
                assert r['target_position']==item['target_position'][t] and r['target_rotation']==item['target_rotation'][t]
                assert np.array_equal(r['previous_q'],previous)
                target=Pose(r['target_position'],r['target_rotation']);query=IKQuery(target,previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                c=v.check(q,query);assert c.accepted==r['accepted'] and c.finite_ok==r['finite']
                assert r['accepted_within_20ms']==bool(c.accepted and r['total_latency_ns']<=20_000_000)
                counts['frames']+=1;counts['accepted_commands']+=int(c.accepted)
                if c.accepted:previous=q.copy()
                assert np.array_equal(previous,r['accepted_state_q'])
                if causal is not None:
                    pred,d=causal.observe(target)
                    assert r['demand']==d['demand'] and r['observed_eta']==d['observed_eta']
                    np.testing.assert_array_equal(pred.position,r['predicted_position']);np.testing.assert_array_equal(pred.rotation,r['predicted_rotation'])
                    counts['causal_demand_checks']+=1
                if r.get('nominal_next_verified'):
                    pred=Pose(r['predicted_position'],r['predicted_rotation'])
                    assert v.check(np.array(r['nominal_next_q']),IKQuery(pred,q,.02)).accepted
                    counts['nominal_pairs']+=1
                if method in ELASTIC:
                    assert r['conic_calls']<=2 and r['mu']==mu
                    qb=np.array(r['backup_q']) if r['backup_accepted'] else np.array(r['previous_q'])
                    zi=np.array(r['initial_nominal_q'])
                    initial=actual_pair(native,v,query,pred,qb,zi,step,scale,r['demand'],qb,zi,mu)
                    assert initial['legal']==r['initial_pair_legal']
                    if initial['legal']:
                        close(initial['cost'],r['initial_objective']);close(initial['xi'],r['initial_xi_actual'])
                    selected=None
                    if r['nominal_next_verified']:
                        selected=actual_pair(native,v,query,pred,q,np.array(r['nominal_next_q']),step,scale,r['demand'],qb,zi,mu)
                        assert selected['legal'] and selected['met']==r['demand_met']
                        close(selected['gamma'],r['predicted_gamma']);close(selected['xi'],r['xi_actual']);close(selected['cost'],r['actual_objective'])
                        assert selected['full_rank']==r['reserve_map_available']
                        if not selected['full_rank']:assert selected['gamma']==0 and not r['demand_met']
                        counts['actual_selected_objectives']+=1
                    if r['optimized_pair_selected'] and not r['recovery_mode']:
                        assert improves(selected['cost'],initial['cost']);counts['strict_actual_improvements']+=1
                    if r['direct_return']:
                        assert np.array_equal(q,r['backup_q']) and r['conic_calls']==0 and (initial['met'] or method==ANALYTIC)
                        if method==ANALYTIC:assert r['actual_objective']==0 and r['decision']=='analytic_zero_intervention'
                        counts['exact_direct_returns']+=1
                    if mu==0 and not r['recovery_mode']:assert r['command_unchanged']
                    if r['partial_correction_adopted']:
                        assert not r['demand_met'] and selected['xi']<initial['xi'] and improves(selected['cost'],initial['cost'])
                        counts['partial_adoptions']+=1
                        if len([e for e in trial_examples if e['robot']==robot and e['method']==method])<3:
                            trial_examples.append(dict(robot=robot,method=method,uid=s['uid'],repeat=s['repeat'],frame=t,
                                previous_q=r['previous_q'],backup_q=r['backup_q'],initial_z=r['initial_nominal_q'],
                                accepted_q=r['q'],accepted_z=r['nominal_next_q'],demand=r['demand'],
                                initial_gamma=initial['gamma'],actual_gamma=selected['gamma'],
                                initial_objective=initial['cost'],actual_objective=selected['cost'],
                                actual_next_success=rows[t+1]['accepted'] if t+1<len(rows) else None,
                                trajectory_completed=s['completion'],source=str(raw)))
                    for tr in r['nonlinear_trials']:
                        check=actual_pair(native,v,query,pred,np.array(tr['q']),np.array(tr['z']),step,scale,r['demand'],qb,zi,mu)
                        assert check['legal']==tr['legal'];counts['trial_geometry_checks']+=1
                        if check['legal']:
                            close(check['cost'],tr['actual_objective']);close(check['gamma'],tr['gamma_actual']);close(check['xi'],tr['xi_actual'])
                            assert check['met']==tr['demand_met']
                            if tr['objective_improved'] and tr['compared_objective'] is not None:assert improves(check['cost'],tr['compared_objective'])
                        category='illegal_pair' if not tr['legal'] else 'legal_actual_improvement' if tr['objective_improved'] else 'legal_no_actual_improvement'
                        trials_count[(robot,method,category)]+=1
                        if tr['hard_demand_would_reject']:trials_count[(robot,method,'legal_but_hard_rule_would_reject')]+=1
                    decisions[(robot,method,r['decision'])]+=1
                    if r['fallback_reason']:fallbacks[(robot,method,r['fallback_reason'])]+=1
                    for name in ('partial_correction_adopted','effective_command_adjustment','demand_met','optimization_called'):
                        extras[key][name].append(r[name])
                    if not r['recovery_mode'] and selected is not None:
                        extras[key]['xi_before'].append(initial['xi']);extras[key]['xi_after'].append(selected['xi'])
                        extras[key]['normalized_gap_reduction'].append((initial['xi']-selected['xi'])/r['demand'])
                    if t+1<len(rows):
                        scope=r['decision'];run_cond[scope][0]+=1;run_cond[scope][1]+=int(rows[t+1]['accepted']);run_cond[scope][2]+=int(rows[t+1]['accepted_within_20ms'])
                        scope='demand_met' if r['demand_met'] else 'demand_unmet'
                        run_cond[scope][0]+=1;run_cond[scope][1]+=int(rows[t+1]['accepted']);run_cond[scope][2]+=int(rows[t+1]['accepted_within_20ms'])
                metadata=r.get('native_status',[])
                stages=metadata if isinstance(metadata,list) else [metadata]
                for d in stages:
                    status=d.get('status',json.dumps(d,sort_keys=True)) if isinstance(d,dict) else str(d)
                    native_status[(robot,method,status)]+=1
                for name in ('total_latency_ns','backup_ns','demand_estimation_ns','initial_pair_ns','cone_construction_ns','cone_solve_ns','nonlinear_validation_ns','optimization_ns'):
                    if name in r:timing[(robot,method,name)].append(r[name])
                if t==s['first_failure_frame']:
                    f={k:r.get(k) for k in ('robot','uid','site_id','family','method','repeat','frame','previous_q','target_position','target_rotation','q','failure_kind',
                        'position_error','orientation_error','velocity_utilization','decision','fallback_reason','native_status','demand','demand_met','initial_pair_legal')}
                    failures.append(f)
            for scope,(n,success,timely) in run_cond.items():
                conditional[(robot,method,scope)].append(dict(uid=s['uid'],repeat=s['repeat'],frames=n,success=success,timely=timely,rate=success/n,timely_rate=timely/n))
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            summaries.append(s);completion.setdefault(robot,{}).setdefault(method,{}).setdefault(str(s['repeat']),[])
            if s['completion']:completion[robot][method][str(s['repeat'])].append(s['uid'])
            if method in ELASTIC and mu>0 and s['completion'] and key not in witnesses:
                witness=dict(robot=robot,method=method,uid=s['uid'],repeat=s['repeat'],dt=.02,initial_q=item['initial_q'],
                    source=str(raw),source_sha256=old.sha(raw),
                    frames=[{k:r[k] for k in ('frame','target_position','target_rotation','q','accepted','total_latency_ns','decision','demand','demand_met','xi_actual')} for r in rows])
                old.write_json(out/f'{robot}_{method}_successful_witness.json',witness);witnesses[key]=s['uid']
            if (index+1)%100==0:print(f'read-only nonlinear audit {robot} {index+1}/2880',flush=True)
        audits.append(dict(robot=robot,**counts,acceptance_discrepancies=0,actual_objective_discrepancies=0))
    assert observed_jobs==expected_jobs
    common.LABELS.update(LABELS)
    main=common.group_table(summaries,arrays);family=common.group_table(summaries,arrays,True)
    for r in main+family:
        group=[s for s in summaries if s['robot']==r['robot'] and s['method']==r['method'] and (r['family']=='all' or s['family']==r['family'])];e=extras[(r['robot'],r['method'])]
        for name in ('optimization_call_rate','mean_conic_calls','partial_correction_rate','effective_command_adjustment_rate','geometric_recovery_rate',
                     'intervention_normalized_mean','unchanged_given_legal_backup','accepted_near_tolerance_rate','demand_met_rate','direct_extra_p50_ms'):
            r[name]=avg(s.get(name) for s in group)
        r.update(normalized_gap_reduction_mean=avg(s.get('normalized_shortfall_reduction_mean') for s in group),
            demand_coverage=avg(s.get('demand_coverage') for s in group))
        if r['family']=='all':r.update(initial_gap_mean=avg(e['xi_before']),final_gap_mean=avg(e['xi_after']))
    metrics=['completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms','intervention_normalized_mean',
        'optimization_call_rate','partial_correction_rate','effective_command_adjustment_rate','normalized_shortfall_reduction_mean','successful_prefix','frame_deadline_success']
    units=[];pairs=[];changes=[]
    for robot in cfg['robots']:
        subs=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in subs});unit={}
        for method in cfg['methods']:
            for uid in uids:
                group=[s for s in subs if s['method']==method and s['uid']==uid]
                assert len(group)==3 and {s['repeat'] for s in group}=={0,1,2}
                row=dict(robot=robot,method=method,uid=uid,family=group[0]['family'],search_repeats=len(group),**{m:avg(s.get(m) for s in group) for m in metrics})
                units.append(row);unit[(method,uid)]=row
        families=[unit[(METHODS_NEW[0],u)]['family'] for u in uids]
        for method in METHODS_NEW:
            for baseline in [b for b in cfg['methods'] if b!=method]:
                for metric in metrics:
                    a=[unit[(method,u)][metric] for u in uids];b=[unit[(baseline,u)][metric] for u in uids]
                    if any(x is None for x in a+b):continue
                    pairs.append(dict(robot=robot,method=method,baseline=baseline,metric=metric,trajectories=160,
                        **common.paired_intervals(np.array(a),np.array(b),families,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])))
                for uid in uids:
                    a=unit[(method,uid)]['completion'];b=unit[(baseline,uid)]['completion']
                    changes.append(dict(robot=robot,method=method,baseline=baseline,uid=uid,family=unit[(method,uid)]['family'],
                        completion_fraction=a,baseline_completion_fraction=b,change='gained' if a>b else 'lost' if a<b else 'same',
                        stable_all_vs_none=(a==1 and b==0) or (a==0 and b==1)))
    conditional_rows=[]
    for (robot,method,decision),group in conditional.items():
        uids=sorted({g['uid'] for g in group});n=sum(g['frames'] for g in group)
        conditional_rows.append(dict(robot=robot,method=method,decision=decision,available_trajectories=len(uids),descriptive_frames=n,
            next_successes=sum(g['success'] for g in group),pooled_next_success=sum(g['success'] for g in group)/n,
            mean_trajectory_next_success=avg(avg(g['rate'] for g in group if g['uid']==u) for u in uids)))
    tables=dict(main_table=main,family_table=family,trajectory_units=units,paired_comparisons=pairs,gained_lost_uids=changes,
        first_failure_inputs=failures,run_summaries=summaries,conditional_next_frame=conditional_rows,
        decision_counts=[dict(robot=k[0],method=k[1],decision=k[2],frames=v) for k,v in sorted(decisions.items())],
        fallback_counts=[dict(robot=k[0],method=k[1],reason=k[2],frames=v) for k,v in sorted(fallbacks.items())],
        native_status_counts=[dict(robot=k[0],method=k[1],status=k[2],occurrences=v) for k,v in sorted(native_status.items())],
        trial_outcomes=[dict(robot=k[0],method=k[1],category=k[2],trials=v) for k,v in sorted(trials_count.items())],
        timing_phases=[dict(robot=k[0],method=k[1],phase=k[2],frames=len(v),mean_ms=float(np.mean(v)/1e6),
            p50_ms=float(np.percentile(v,50)/1e6),p95_ms=float(np.percentile(v,95)/1e6),p99_ms=float(np.percentile(v,99)/1e6)) for k,v in sorted(timing.items())])
    for name,rows in tables.items():common.csv_write(out/f'{name}.csv',rows)
    old.write_json(out/'partial_improvement_examples.json',trial_examples);old.write_json(out/'completion_uids.json',completion)
    old.write_json(out/'source_data.json',tables)
    old.write_json(out/'verification_manifest.json',dict(utc=old.utc(),results=audits,solver_calls=0,
        operation='read-only original verifier, actual nonlinear reserve/objective checks including all logged trials, causal-demand replay; no IK reruns'))
    old.write_json(out/'manifest.json',dict(utc=old.utc(),sources=sources,code_sha256=old.sha(__file__),
        scope='fixed-candidate independent evaluation',independent_unit='160 trajectory UIDs per robot; all settings have 3 nested repeats, Pink timing repeats not independent samples',
        intervals='4000 paired family-stratified UID bootstrap resamples, unadjusted descriptive 95% CIs; no significance or equivalence test',
        exclusions=0,frame_calls=1728000,runs=5760,selected_mu=.25))
    write_findings(root,tables)
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        round(r['cumulative_ms_per_sweep'],3),round(r['acceleration_rms_mean'],3))


from .locked_findings import write_findings

if __name__=='__main__':aggregate()
