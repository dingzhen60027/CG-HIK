"""Thin saved-record audit and tables using the existing trajectory statistics."""
from collections import Counter
import ast
import csv
import json
import numpy as np
from .correction_reserve import study as runner
from .correction_reserve.study import ROOT,read_rows,context,write_json,sha,utc
from .correction_reserve.reporting import csv_write,group_table,paired_intervals,LABELS
from .types import IKQuery,Pose

LABELS.update(single_gn_k0='Frozen bounded GN, kappa=0',single_gn_k1='Frozen bounded GN, kappa=1',
              task_excess_gn='Task-excess bounded GN',task_excess_lbfgsb='Same-Phi L-BFGS-B')


def local_tables(out,cfg,folder):
    raw=json.loads((out/'local/same_input_results.json').read_text())
    inputs=json.loads((out/'protocol/failure_inputs.json').read_text())
    diag=list(csv.DictReader((out/'local/point_endpoint_diagnostics.csv').open()))
    units=[];main=[];comparisons=[];endpoint=[];metric=[];source_groups=[]
    ids={robot:sorted(x['input_id'] for x in inputs if x['robot']==robot) for robot in cfg['robots']}
    success={}
    for item in inputs:
        query=IKQuery(Pose(np.array(item['target_position']),np.array(item['target_rotation'])),np.array(item['previous_q']),item['dt'])
        _,kin,v,_=context(item['robot'],cfg)
        for method in cfg['local']['methods']:
            rr=[r for r in raw if r['input_id']==item['input_id'] and r['method']==method]
            assert len(rr)==3
            for r in rr:
                verdict=v.check(np.array(r['q']),query)
                assert bool(verdict.accepted)==r['accepted']
            fraction=float(np.mean([r['accepted'] for r in rr]));success[(item['input_id'],method)]=fraction
            units.append(dict(input_id=item['input_id'],robot=item['robot'],method=method,
                source_methods=sorted({a['method'] for a in item['aliases']}),
                source_trajectories=sorted({a['uid'] for a in item['aliases']}),
                accepted_fraction=fraction,within_20ms_fraction=float(np.mean([r['accepted_within_20ms'] for r in rr])),
                latency_ms_mean=float(np.mean([r['total_latency_ns'] for r in rr])/1e6),
                evaluations_mean=float(np.mean([r['evaluations'] for r in rr])),
                iterations_mean=float(np.mean([r['iterations'] for r in rr])) if method!='task_excess_lbfgsb' else None,
                statuses=dict(Counter(r['internal_status'] for r in rr))))
        dd=[d for d in diag if d['input_id']==item['input_id']]
        assert len(dd)==3
        stable_recovery=success[(item['input_id'],'task_excess_gn')]==1 and success[(item['input_id'],'single_gn_k0')]==0
        # This is a numerical diagnostic threshold, not a solver setting.
        stationary=all(float(d['point_projected_gradient_inf'])<=1e-5 for d in dd)
        endpoint.append(dict(input_id=item['input_id'],robot=item['robot'],
            point_failure_fraction=1-success[(item['input_id'],'single_gn_k0')],
            task_excess_accept_fraction=success[(item['input_id'],'task_excess_gn')],
            stable_recovery=stable_recovery,point_endpoint_near_stationary_raw_1e_5=stationary,
            near_stationary_with_independent_legal_command=stationary and stable_recovery,
            categories=sorted({d['category'] for d in dd}),
            maximum_projected_gradient_inf=max(float(d['point_projected_gradient_inf']) for d in dd),
            maximum_projected_gradient_scaled=max(float(d['point_projected_gradient_scaled']) for d in dd),
            endpoint_active_constraints=ast.literal_eval(dd[0]['actual_endpoint_kkt_active_constraints'])))
        d=json.loads((out/'local/diagnostics'/f"{item['input_id']}.json").read_text())['results']
        a,b=d['task_excess_gn']['trace'][0],d['task_excess_isotropic']['trace'][0]
        np.testing.assert_array_equal(a['q'],b['q']);np.testing.assert_array_equal(a['e'],b['e'])
        if 'direction' in a and 'direction' in b:
            metric.append(dict(input_id=item['input_id'],robot=item['robot'],q=a['q'],e=a['e'],
                full_direction=a['direction'],isotropic_direction=b['direction'],
                direction_difference_norm=float(np.linalg.norm(np.array(a['direction'])-b['direction'])),
                full_gd=a['gd'],full_dHd=a['dHd'],isotropic_gd=b['gd'],isotropic_dHd=b['dHd'],
                initial_phi=a['objective'],full_phi_after_adopted_step=a.get('new_objective'),
                isotropic_phi_after_adopted_step=b.get('new_objective'),
                full_alpha=a.get('alpha'),isotropic_alpha=b.get('alpha'),
                full_accept_fraction=success[(item['input_id'],'task_excess_gn')],
                isotropic_accept_fraction=success[(item['input_id'],'task_excess_isotropic')]))
    for robot in cfg['robots']:
        for method in cfg['local']['methods']:
            rr=[r for r in raw if r['robot']==robot and r['method']==method]
            uu=[u for u in units if u['robot']==robot and u['method']==method]
            main.append(dict(robot=robot,method=method,unique_inputs=len(uu),
                stable_all_three_success=sum(u['accepted_fraction']==1 for u in uu),
                any_success=sum(u['accepted_fraction']>0 for u in uu),
                accepted_calls=sum(r['accepted'] for r in rr),calls=len(rr),
                latency_p50_p95_p99_ms=(np.percentile([r['total_latency_ns'] for r in rr],[50,95,99])/1e6).tolist(),
                mean_evaluations=float(np.mean([r['evaluations'] for r in rr])),
                categories_point_failure=dict(Counter(d['category'] for d in diag if d['robot']==robot and d['repeat']=='0'))))
        for base in cfg['local']['methods']:
            if base=='task_excess_gn':continue
            gained=[i for i in ids[robot] if success[(i,'task_excess_gn')]>success[(i,base)]]
            lost=[i for i in ids[robot] if success[(i,'task_excess_gn')]<success[(i,base)]]
            comparisons.append(dict(robot=robot,method='task_excess_gn',baseline=base,gained_input_ids=gained,lost_input_ids=lost))
        pure=[i['input_id'] for i in inputs if i['robot']==robot and
              any(a['method'] in ('single_gn_k0','single_gn_k1') for a in i['aliases'])]
        for label,selected in [('all_gn_including_clip_osqp',ids[robot]),('frozen_k0_k1_sources',pure)]:
            for method in cfg['local']['methods']:
                source_groups.append(dict(robot=robot,source_group=label,method=method,unique_inputs=len(selected),
                    stable_three_success=sum(success[(i,method)]==1 for i in selected),
                    input_ids=selected))
    for name,rows in [('local_main',main),('local_input_units',units),('local_comparisons',comparisons),
                      ('stationarity_and_witnesses',endpoint),('same_input_first_metric_update',metric),('local_source_groups',source_groups)]:
        csv_write(folder/(name+'.csv'),rows)
    return dict(main=main,units=units,comparisons=comparisons,endpoints=endpoint,metric=metric,source_groups=source_groups)


def report(out,cfg):
    folder=out/'reports';folder.mkdir(exist_ok=False)
    summaries=[];arrays={};extra={};first=[];witnesses=[];sources={};audit=Counter()
    for robot in cfg['robots']:
        origin=out/f'development_{robot}';seal=json.loads((origin/'completed.json').read_text())
        assert seal['runs']==40*len(cfg['methods'])*3 and seal['frames']==40*len(cfg['methods'])*3*150
        for name,digest in seal['files'].items():assert sha(origin/name)==digest
        sources[robot]=dict(path=str(origin.relative_to(ROOT)),sha256=sha(origin/'completed.json'))
        _,kin,v,_=context(robot,cfg)
        data={r['uid']:r for r in json.loads((ROOT/cfg['development'][robot+'_targets']).read_text())}
        jobs=json.loads((origin/'summaries.json').read_text())
        assert Counter((s['uid'],s['method'],s['repeat']) for s in jobs)==Counter((u,m,r) for u in data for m in cfg['methods'] for r in range(3))
        for index,s in enumerate(jobs):
            rows=read_rows(origin/s['raw_file']);assert len(rows)==150
            target=data[s['uid']];previous=np.array(target['initial_q']);counts=Counter()
            iterations=[];evaluations=[]
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==target['dt']==.02
                np.testing.assert_array_equal(r['target_position'],target['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],target['target_rotation'][frame])
                np.testing.assert_array_equal(r['previous_q'],previous)
                query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                verdict=v.check(q,query)
                assert bool(verdict.accepted)==r['accepted']
                assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
                if np.isfinite(verdict.position_error):
                    np.testing.assert_allclose([r['position_error'],r['orientation_error']],
                        [verdict.position_error,verdict.orientation_error],rtol=0,atol=1e-12)
                audit['commands_checked']+=1;audit['accepted_commands']+=int(verdict.accepted)
                counts['over_20ms']+=r['total_latency_ns']>20_000_000
                counts['accepted']+=int(verdict.accepted)
                if verdict.accepted:
                    pn=verdict.position_error/v.config.position_tolerance
                    rn=verdict.orientation_error/v.config.orientation_tolerance
                    counts['position_over90pct']+=pn>.9;counts['orientation_over90pct']+=rn>.9
                    counts['either_over90pct']+=max(pn,rn)>.9
                else:counts['failure:'+r['failure_kind']]+=1
                counts['status:'+str(r.get('internal_status'))]+=1
                if r.get('evaluations') is not None:evaluations.append(r['evaluations'])
                if r.get('iterations') is not None:iterations.append(r['iterations'])
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
            if s['completion']:
                witnesses.append(dict(robot=robot,uid=s['uid'],site_id=s['site_id'],family=s['family'],method=s['method'],repeat=s['repeat'],
                    initial_q=target['initial_q'],frames=150,deadline_completion=s['deadline_completion'],
                    raw_file=str((origin/s['raw_file']).relative_to(ROOT)),sha256=sha(origin/s['raw_file'])))
            summaries.append(s);extra[s['run_id']]=dict(counts=counts,iterations=iterations,evaluations=evaluations)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if index%120==0:print('original verifier replay',robot,index+1,'/',len(jobs),flush=True)
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    for table in (main,family):
        for row in table:
            ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
            counts=Counter();iterations=[];evaluations=[]
            for s in ss:
                ex=extra[s['run_id']];counts.update(ex['counts']);iterations+=ex['iterations'];evaluations+=ex['evaluations']
            for key in ('backup_rate','changed_from_backup_rate'):row.pop(key,None)
            row.update(frame_calls=sum(s['frames'] for s in ss),over_20ms_frames=counts['over_20ms'],
                all_frame_cumulative_ns=sum(s['total_latency_ns'] for s in ss),
                position_over90pct_accepted=counts['position_over90pct']/counts['accepted'],
                orientation_over90pct_accepted=counts['orientation_over90pct']/counts['accepted'],
                either_over90pct_accepted=counts['either_over90pct']/counts['accepted'],
                internal_statuses={k.removeprefix('status:'):n for k,n in counts.items() if k.startswith('status:')},
                failure_reasons={k.removeprefix('failure:'):n for k,n in counts.items() if k.startswith('failure:')},
                mean_iterations=float(np.mean(iterations)) if iterations and row['method']!='task_excess_lbfgsb' else None,
                iteration_note='L-BFGS-B early-exit nit is unavailable (raw zero sentinel); actual function/gradient evaluations are recorded.' if row['method']=='task_excess_lbfgsb' else 'Outer checks; accepted candidate can stop within an update.',
                mean_evaluations=float(np.mean(evaluations)) if evaluations else None,
                max_frame_latency_ms=float(max(np.max(arrays[s['run_id']]['latency']) for s in ss)/1e6))
    units=[];pairs=[];changes=[];completion={}
    metrics=('completion','deadline_completion','total_latency_ns','frame_success','successful_prefix','acceleration_rms')
    comparisons=[('task_excess_gn',m) for m in cfg['methods'] if m!='task_excess_gn']
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==u) for u in uids]
        for method in cfg['methods']:
            completion.setdefault(robot,{})[method]={str(rep):sorted(s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']) for rep in range(3)}
            for uid,f in zip(uids,fam):
                rr=[s for s in ss if s['uid']==uid and s['method']==method];assert len(rr)==3
                val={k:float(np.mean([r[k] for r in rr])) for k in metrics};unit[(method,uid)]=val
                units.append(dict(robot=robot,uid=uid,site_id=rr[0]['site_id'],family=f,method=method,repeats=3,**val))
        for method,base in comparisons:
            for metric in metrics:
                a=np.array([unit[(method,u)][metric] for u in uids]);b=np.array([unit[(base,u)][metric] for u in uids])
                pairs.append(dict(robot=robot,method=method,baseline=base,metric=metric,
                    **paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,f in zip(uids,fam):
                a=unit[(method,uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,method=method,baseline=base,uid=uid,family=f,
                    site_id=next(s['site_id'] for s in ss if s['uid']==uid),method_fraction=a,baseline_fraction=b,
                    change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
    for name,rows in [('main_table',main),('family_table',family),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('first_failure_inputs',first),('run_summaries',summaries)]:
        csv_write(folder/(name+'.csv'),rows)
    local=local_tables(out,cfg,folder)
    write_json(folder/'source_data.json',dict(main=main,family=family,pairs=pairs,changes=changes,units=units,local=local))
    write_json(folder/'completion_uids.json',completion);write_json(folder/'successful_trajectory_index.json',witnesses)
    write_json(folder/'verification.json',dict(counts=dict(audit),accepted_contract_violations=0,successful_runs=len(witnesses),
        solver_calls=0,checks='All original targets, actual accepted-state feedback and original verifier; every command and all outer latencies retained.'))
    write_json(folder/'manifest.json',dict(created=utc(),sources=sources,reporting_sha256=sha(__file__),
        independent_unit='40 trajectory UIDs/robot; 3 nested repeats averaged; family-stratified 4000-resample paired 95% descriptive intervals',
        frame_quantiles='All frames pooled descriptively, not treated as independent inferential units',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],4),'cumulative_s',r['cumulative_ms_per_sweep']/1000,flush=True)
