"""Saved-command checks and thin tables on the existing trajectory statistics."""
from collections import Counter,defaultdict
import json
import numpy as np
from .correction_reserve import study as runner
from .correction_reserve.study import ROOT,context,read_rows,write_json,sha,utc
from .correction_reserve.reporting import csv_write,group_table,paired_intervals,LABELS
from .types import IKQuery,Pose

LABELS.update(single_gn_k0='Original GN kappa=0',single_gn_k1='Original GN kappa=1',
    task_excess_gn='Task-excess GN kappa=0',task_balance_k0='Task-balance GN kappa=0',task_balance_k1='Task-balance GN kappa=1')
PAIRS=[('task_balance_k0','single_gn_k0'),('task_balance_k1','single_gn_k1'),
       ('task_balance_k0','task_excess_gn'),('task_balance_k1','task_excess_gn')]
FLAGS=('theta_half_first_step_return','continued_dual','continued_dual_accepted','task_feasible_early')


def local_tables(out,cfg,folder):
    raw=json.loads((out/'local/same_input_results.json').read_text())
    inputs=json.loads((out/'protocol/failure_inputs.json').read_text());units=[];commands=[];groups=[];changes=[]
    contexts={r:context(r,cfg) for r in cfg['robots']};success={}
    for item in inputs:
        _,kin,v,_=contexts[item['robot']]
        p=np.array(item['previous_q']);query=IKQuery(Pose(np.array(item['target_position']),np.array(item['target_rotation'])),p,item['dt'])
        for method in cfg['methods']:
            rr=[r for r in raw if r['input_id']==item['input_id'] and r['method']==method];assert len(rr)==3
            for r in rr:
                q=np.array(r['q']);verdict=v.check(q,query);assert verdict.accepted==r['accepted']
                pn=verdict.position_error/v.config.position_tolerance;rn=verdict.orientation_error/v.config.orientation_tolerance
                commands.append(dict(input_id=item['input_id'],robot=item['robot'],method=method,repeat=r['repeat'],
                    q=q.tolist(),step=(q-p).tolist(),position_normalized=pn,orientation_normalized=rn,rho=max(pn,rn),
                    velocity_utilization=r['velocity_utilization'],accepted=r['accepted'],total_latency_ns=r['total_latency_ns'],
                    iterations=r.get('iterations'),evaluations=r.get('evaluations'),verification_calls=r.get('verification_calls'),
                    dual_updates=r.get('dual_updates'),status=r['internal_status'],subproblems=r.get('subproblems')))
            fraction=float(np.mean([r['accepted'] for r in rr]));success[item['input_id'],method]=fraction
            units.append(dict(input_id=item['input_id'],robot=item['robot'],method=method,accepted_fraction=fraction,
                source_uids=sorted({a['uid'] for a in item['aliases']}),source_methods=sorted({a['method'] for a in item['aliases']}),
                latency_mean_ms=float(np.mean([r['total_latency_ns'] for r in rr])/1e6),
                continued_dual_calls=sum(r.get('continued_dual',False) for r in rr),
                continued_dual_accepted_calls=sum(r.get('continued_dual_accepted',False) for r in rr)))
    for robot in cfg['robots']:
        allids=[i['input_id'] for i in inputs if i['robot']==robot]
        pure=[i['input_id'] for i in inputs if i['robot']==robot and any(a['method'] in ('single_gn_k0','single_gn_k1') for a in i['aliases'])]
        for label,ids in [('all52',allids),('direct_k0_k1_sources16',pure)]:
            for method in cfg['methods']:
                rr=[r for r in commands if r['input_id'] in ids and r['method']==method]
                groups.append(dict(robot=robot,source_group=label,method=method,inputs=len(ids),
                    source_uids=len({a['uid'] for i in inputs if i['input_id'] in ids for a in i['aliases']}),
                    stable_three_success=sum(success[i,method]==1 for i in ids),any_success=sum(success[i,method]>0 for i in ids),
                    accepted_calls=sum(r['accepted'] for r in rr),calls=len(rr),
                    latency_p50_p95_p99_ms=(np.percentile([r['total_latency_ns'] for r in rr],[50,95,99])/1e6).tolist()))
            for method,base in PAIRS:
                changes.append(dict(robot=robot,source_group=label,method=method,baseline=base,
                    gained_input_ids=[i for i in ids if success[i,method]>success[i,base]],
                    lost_input_ids=[i for i in ids if success[i,method]<success[i,base]]))
    csv_write(folder/'local_input_units.csv',units);csv_write(folder/'local_commands.csv',commands)
    csv_write(folder/'local_source_groups.csv',groups);csv_write(folder/'local_gained_lost.csv',changes)
    return dict(groups=groups,changes=changes,units=units)


def subproblem_tables(out,cfg,folder):
    raw=json.loads((out/'subproblems/raw_results.json').read_text());by=defaultdict(list)
    for r in raw:by[r['problem_id'],r['repeat']].append(r)
    checked=[];early=[]
    for key,rows in by.items():
        full=[r for r in rows if r['method']!='scalar_early'];assert len(full)==2
        lower=max(r['box_minorant_lower'] for r in full)
        # Scalar's trace can have a stronger lower bound at a previous theta.
        scalar=next(r for r in full if r['method']=='scalar_full')
        lower=max(lower,max(t['dual_lower'] for t in scalar['trace']))
        for r in full:
            raw_gap=r['objective']-lower;norm=max(1.,abs(r['objective']),abs(lower))
            valid=raw_gap>=-1e-8*norm and raw_gap<=1e-9+cfg['subproblems']['relative_quality_tolerance']*norm
            valid=valid and r['feasible_box_violation']<=cfg['subproblems']['box_tolerance'] and r.get('raw_box_violation',0)<=cfg['subproblems']['box_tolerance']
            checked.append(dict(**{k:v for k,v in r.items() if k not in ('trace','direction')},
                common_lower=lower,common_normalized_gap=max(0,raw_gap)/norm,common_raw_gap=raw_gap,quality_pass=bool(valid),
                objective_difference_vs_reference=r['objective']-next(x['objective'] for x in full if x['method']=='clarabel_epigraph')))
        er=next(r for r in rows if r['method']=='scalar_early')
        if er['task_feasible_early']:assert er['gap'] is None and not er['converged']
        early.append({k:v for k,v in er.items() if k not in ('trace','direction')})
    common={(r['problem_id'],r['repeat']) for r in checked if r['quality_pass']}
    common={k for k in common if all(r['quality_pass'] for r in checked if (r['problem_id'],r['repeat'])==k)}
    table=[];earlytable=[]
    for robot in cfg['robots']:
        for stratum in ('regular_first_subproblems','historical_failure_inputs'):
            for kappa in (0,1):
                for method in ('scalar_full','clarabel_epigraph'):
                    for subset in ('all','both_quality_pass'):
                        rr=[r for r in checked if r['robot']==robot and r['stratum']==stratum and r['kappa']==kappa and r['method']==method
                            and (subset=='all' or (r['problem_id'],r['repeat']) in common)]
                        if not rr:continue
                        table.append(dict(robot=robot,stratum=stratum,kappa=kappa,method=method,subset=subset,
                            problems=len({r['problem_id'] for r in rr}),calls=len(rr),quality_pass=sum(r['quality_pass'] for r in rr),
                            native_statuses=dict(Counter(r['status'] for r in rr)),
                            call_p50_p95_p99_ms=(np.percentile([r['total_ns'] for r in rr],[50,95,99])/1e6).tolist(),
                            mean_call_ms=float(np.mean([r['total_ns'] for r in rr])/1e6),
                            kernel_p50_ms=float(np.median([r['solve_ns'] for r in rr])/1e6),
                            setup_update_p50_ms=float(np.median([r.get('setup_update_ns',r.get('conversion_ns',0)) for r in rr])/1e6),
                            max_common_normalized_gap=max(r['common_normalized_gap'] for r in rr),
                            max_abs_objective_difference=max(abs(r['objective_difference_vs_reference']) for r in rr),
                            mean_updates=float(np.mean([r.get('dual_updates',r.get('iterations')) for r in rr]))))
                rr=[r for r in early if r['robot']==robot and r['stratum']==stratum and r['kappa']==kappa]
                earlytable.append(dict(robot=robot,stratum=stratum,kappa=kappa,calls=len(rr),
                    actual_full_step_accepted=sum(r['task_feasible_early'] for r in rr),
                    theta_half_early=sum(r['task_feasible_early'] and r['dual_updates']==1 for r in rr),
                    continued_dual_early=sum(r['task_feasible_early'] and r['dual_updates']>1 for r in rr),
                    call_p50_p95_p99_ms=(np.percentile([r['total_ns'] for r in rr],[50,95,99])/1e6).tolist(),
                    meaning='Early accepted commands, not matched-optimality comparison; no online 20 ms deadline in this fixed diagnostic.'))
    csv_write(folder/'subproblem_quality_calls.csv',checked);csv_write(folder/'subproblem_quality_time.csv',table)
    csv_write(folder/'subproblem_early_calls.csv',early);csv_write(folder/'subproblem_early_summary.csv',earlytable)
    return dict(table=table,early=earlytable,common_quality_calls=len(common),full_calls=len(checked),
        setup_calls=[r for r in checked if r['method']=='clarabel_epigraph' and not r['updated']])


def report(out,cfg):
    folder=out/'reports';folder.mkdir(exist_ok=False)
    summaries=[];arrays={};extra={};first=[];witnesses=[];sources={};audit=Counter();commands={}
    for robot in cfg['robots']:
        origin=out/f'development_{robot}';seal=json.loads((origin/'completed.json').read_text())
        assert seal['runs']==600 and seal['frames']==90000
        for name,digest in seal['files'].items():assert sha(origin/name)==digest
        sources[robot]=sha(origin/'completed.json');_,kin,v,_=context(robot,cfg)
        data={r['uid']:r for r in json.loads((ROOT/cfg['development'][robot+'_targets']).read_text())}
        jobs=json.loads((origin/'summaries.json').read_text())
        assert Counter((s['uid'],s['method'],s['repeat']) for s in jobs)==Counter((u,m,r) for u in data for m in cfg['methods'] for r in range(3))
        for index,s in enumerate(jobs):
            rows=read_rows(origin/s['raw_file']);assert len(rows)==150
            target=data[s['uid']];previous=np.array(target['initial_q']);counts=Counter();values=defaultdict(list)
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==target['dt']==.02
                np.testing.assert_array_equal(r['target_position'],target['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],target['target_rotation'][frame])
                np.testing.assert_array_equal(r['previous_q'],previous)
                query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),previous,.02)
                q=np.array(r['q']);verdict=v.check(q,query)
                assert bool(verdict.accepted)==r['accepted']
                assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
                np.testing.assert_allclose([r['position_error'],r['orientation_error']],
                    [verdict.position_error,verdict.orientation_error],rtol=0,atol=1e-12)
                audit['commands_checked']+=1;audit['accepted_commands']+=int(verdict.accepted)
                counts['over_20ms']+=r['total_latency_ns']>20_000_000;counts['accepted']+=int(verdict.accepted)
                if verdict.accepted:
                    pn=verdict.position_error/v.config.position_tolerance;rn=verdict.orientation_error/v.config.orientation_tolerance
                    counts['position_over90pct']+=pn>.9;counts['orientation_over90pct']+=rn>.9;counts['either_over90pct']+=max(pn,rn)>.9
                else:counts['failure:'+r['failure_kind']]+=1
                counts['status:'+str(r.get('internal_status'))]+=1
                for flag in FLAGS:counts[flag]+=int(r.get(flag,False))
                for key in ('iterations','evaluations','verification_calls','dual_updates','box_qp_updates','backtracking_evaluations'):
                    if r.get(key) is not None:values[key].append(r[key])
                for info in r.get('subproblems',[]):
                    counts['subproblem:'+info['status']]+=1
                    if info['task_feasible_early']:assert info['gap'] is None and not info['converged']
                    if info.get('gap') is not None:values['reported_gaps'].append(info['gap'])
                if frame==s['first_failure_frame']:
                    first.append({k:r.get(k) for k in ('robot','uid','site_id','family','method','repeat','frame','dt',
                        'previous_q','target_position','target_rotation','q','accepted','failure_kind','verification_reasons',
                        'position_error','orientation_error','velocity_utilization','internal_status','total_latency_ns')})
                if verdict.accepted:previous=q.copy()
                np.testing.assert_array_equal(r['accepted_state_q'],previous)
            calculated=runner.summarize(rows,kin,v)
            for key in ('completion','deadline_completion','accepted_frames','deadline_frames','total_latency_ns','first_failure_frame'):
                assert calculated[key]==s[key]
            assert s['accepted_contract_violations']==0
            if s['completion']:witnesses.append(dict(robot=robot,uid=s['uid'],site_id=s['site_id'],method=s['method'],repeat=s['repeat'],
                initial_q=target['initial_q'],frames=150,raw_file=str((origin/s['raw_file']).relative_to(ROOT)),sha256=sha(origin/s['raw_file'])))
            summaries.append(s);extra[s['run_id']]=(counts,values)
            commands[robot,s['uid'],s['method'],s['repeat']]=np.array([r['accepted_state_q'] for r in rows])
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if index%120==0:print('original verifier replay',robot,index+1,'/600',flush=True)
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    for table in (main,family):
        for row in table:
            ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
            counts=Counter();values=defaultdict(list)
            for s in ss:
                c,vals=extra[s['run_id']];counts.update(c)
                for key,value in vals.items():values[key]+=value
            for k in ('backup_rate','changed_from_backup_rate'):row.pop(k,None)
            total=sum(s['frames'] for s in ss)
            row.update(frame_calls=total,over_20ms_frames=counts['over_20ms'],all_frame_cumulative_ns=sum(s['total_latency_ns'] for s in ss),
                **{key+'_accepted':counts[key]/counts['accepted'] for key in ('position_over90pct','orientation_over90pct','either_over90pct')},
                **{flag+'_rate':counts[flag]/total for flag in FLAGS},
                **{'mean_'+k:float(np.mean(val)) for k,val in values.items()},
                counts=dict(counts),max_frame_latency_ms=max(np.max(arrays[s['run_id']]['latency']) for s in ss)/1e6)
    units=[];pairs=[];changes=[];completion={};difference=[]
    metrics=('completion','deadline_completion','total_latency_ns','frame_success','successful_prefix','acceleration_rms')
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==u) for u in uids]
        for method in cfg['methods']:
            completion.setdefault(robot,{})[method]={str(rep):sorted(s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']) for rep in range(3)}
            for uid,f in zip(uids,fam):
                rr=[s for s in ss if s['uid']==uid and s['method']==method];assert len(rr)==3
                val={k:float(np.mean([r[k] for r in rr])) for k in metrics};unit[method,uid]=val
                units.append(dict(robot=robot,uid=uid,site_id=rr[0]['site_id'],family=f,method=method,repeats=3,**val))
        for method,base in PAIRS:
            for metric in metrics:
                a=np.array([unit[method,u][metric] for u in uids]);b=np.array([unit[base,u][metric] for u in uids])
                pairs.append(dict(robot=robot,method=method,baseline=base,metric=metric,**paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,f in zip(uids,fam):
                a=unit[method,uid]['completion'];b=unit[base,uid]['completion']
                changes.append(dict(robot=robot,method=method,baseline=base,uid=uid,family=f,
                    site_id=next(s['site_id'] for s in ss if s['uid']==uid),method_fraction=a,baseline_fraction=b,
                    change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
                for rep in range(3):
                    delta=np.max(abs(commands[robot,uid,method,rep]-commands[robot,uid,base,rep]),axis=1)
                    difference.append(dict(robot=robot,uid=uid,method=method,baseline=base,repeat=rep,max_abs_rad=float(max(delta)),
                        first_difference_gt_1e_minus8_rad=int(np.flatnonzero(delta>1e-8)[0]) if np.any(delta>1e-8) else None,
                        frames_within_1e_minus8_rad=int(np.sum(delta<=1e-8)),note='Closed-loop states; after divergence these are not same-input direction comparisons.'))
    for name,rows in [('main_table',main),('family_table',family),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('first_failure_inputs',first),('run_summaries',summaries),('closed_loop_command_differences',difference)]:
        csv_write(folder/(name+'.csv'),rows)
    local=local_tables(out,cfg,folder);micro=subproblem_tables(out,cfg,folder)
    write_json(folder/'source_data.json',dict(main=main,family=family,pairs=pairs,changes=changes,units=units,local=local,micro=micro))
    write_json(folder/'completion_uids.json',completion);write_json(folder/'successful_trajectory_index.json',witnesses)
    write_json(folder/'verification.json',dict(counts=dict(audit),accepted_contract_violations=0,successful_runs=len(witnesses),solver_calls=0))
    write_json(folder/'manifest.json',dict(created=utc(),sources=sources,reporting_sha256=sha(__file__),
        independent_unit='40 trajectory UIDs/robot, three nested repeats averaged; family-stratified paired 4000-resample 95% descriptive intervals',
        timing='All full outer frame times, no exclusions. Frame quantiles descriptive, not independent samples.',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],4),'cumulative_s',r['cumulative_ms_per_sweep']/1000,flush=True)
