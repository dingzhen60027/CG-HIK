#!/usr/bin/env python3
"""Current-frame recovery study, composing the existing runner and statistics."""
import argparse
import csv
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import yaml

from confik.correction_reserve import study as runner
from confik.correction_reserve.study import ROOT, context, write_json, sha, clean, read_rows, utc
from confik.correction_reserve.reporting import csv_write, group_table, paired_intervals, LABELS
from confik.types import IKQuery, Pose
from confik.current_frame_trf import CurrentFrameTRF, TracThenCurrentTRF, SETTINGS
from confik.task_contract_alignment.outcomes import ContractSolver
from confik.correction_reserve.pink_adapter import PinkAdapter


OUT=ROOT/'outputs/current_frame_recovery'
BASELINE='8e754127bb56ec6054ef197f7f1b4e2324611b91'
OLD=ROOT/'outputs/task_recourse/numerical_completion_development'
LABELS.update(current_trf='Current-frame bounded TRF',trac_then_current_trf='TRAC 5 ms then one current-frame TRF')


def code_hashes():
    names=['src/confik/current_frame_trf.py','scripts/run_current_frame_recovery.py','configs/current_frame_recovery.yaml',
        'src/confik/correction_reserve/study.py','src/confik/correction_reserve/geometry.py',
        'src/confik/correction_reserve/native_geometry.py','src/confik/correction_reserve/pink_adapter.py',
        'src/confik/task_contract_alignment/outcomes.py','src/confik/task_contract_alignment/trac_adapter.py',
        'src/confik/solvers/verifier.py','src/confik/continuation_mechanism/observation.py',
        'tmp/task_contract_build/libcontract_trac.so','configs/paper_v2.yaml']
    return {n:sha(ROOT/n) for n in names}


def factory(method,robot,cfg):
    source,kin,v,urdf=context(robot,cfg)
    if method=='current_trf':solver=CurrentFrameTRF(kin,v,urdf)
    elif method=='trac_then_current_trf':
        solver=TracThenCurrentTRF(ContractSolver('trac_task_5ms',kin,v,source,str(ROOT/cfg['native_trac_library']),urdf),CurrentFrameTRF(kin,v,urdf))
    elif method=='trac_task_5ms':solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    elif method=='pink_qp':solver=PinkAdapter(kin,v,source,urdf)
    else:raise ValueError(method)
    return solver,kin,v


def trajectory_items(cfg,robot):
    data=json.loads((ROOT/cfg['development'][robot+'_targets']).read_text())
    assert len(data)==40 and sorted(Counter(r['family'] for r in data).values())==[10]*4
    for row in data:
        assert row['dt']==.02 and len(row['target_position'])==150
    return [dict(robot=robot,**r) for r in data]


def prepare(cfg):
    folder=OUT/'protocol';folder.mkdir(exist_ok=False)
    with (ROOT/cfg['first_failures']).open() as f:source_rows=list(csv.DictReader(f))
    unique={}
    for index,src in enumerate(source_rows):
        robot=src['robot']
        raw_file=OLD/f'validated_{robot}/runs/{robot}_{src["site_id"]}_{src["method"]}_r{src["repeat"]}.jsonl.gz'
        raw=read_rows(raw_file)[int(src['frame'])];assert not raw['accepted']
        fields={k:raw[k] for k in ('previous_q','target_position','target_rotation','dt')}
        for name in ('previous_q','target_position','target_rotation'):np.testing.assert_array_equal(json.loads(src[name]),fields[name])
        digest=hashlib.sha256(robot.encode()+b''.join(np.asarray(fields[k],dtype='<f8').tobytes() for k in
            ('previous_q','target_position','target_rotation','dt'))).hexdigest()
        if digest not in unique:unique[digest]=dict(input_uid=digest,robot=robot,**fields,sources=[])
        unique[digest]['sources'].append(dict(source_csv_row=index,trajectory_uid=src['uid'],site_id=src['site_id'],
            method=src['method'],repeat=int(src['repeat']),frame=int(src['frame']),family=src['family'],raw_file=str(raw_file.relative_to(ROOT)),
            historical_failure_kind=raw['failure_kind'],historical_q=raw['q']))
    write_json(folder/'failure_inputs.json',list(unique.values()))
    write_json(folder/'protocol.json',dict(created=utc(),baseline=BASELINE,config=cfg,trf=SETTINGS,
        source_rows=len(source_rows),unique_inputs=len(unique),unique_by_robot=dict(Counter(r['robot'] for r in unique.values())),
        scope='All unique exact robot/previous_q/target/dt first failures from latest guarded development comparison; selected hard inputs, not a general workload.',
        current_only='Initial seed is actual previous_q, never supplied witness or another method state. Scaled TRF, original verifier at initial state and callbacks; 50 function evaluations, 20 ms soft wall-clock period.',
        stop='Public verifier acceptance at an iterate, then StopIteration; native status -2 retained separately. No extra 0.99999 pose margin. Deadline checked at callback, not native preemption.',
        composition='One task TRAC 5 ms. On rejection only, at most one identical TRF with the original absolute deadline; no new period and no fallback chain.',
        repeats='Three per complete UID; deterministic repeats measure timing, not independent trajectories.',
        input_files={robot:dict(path=cfg['development'][robot+'_targets'],sha256=sha(ROOT/cfg['development'][robot+'_targets'])) for robot in cfg['robots']},
        trajectories=[{k:r[k] for k in ('robot','uid','site_id','family','dt')} for robot in cfg['robots'] for r in trajectory_items(cfg,robot)],
        code_hashes=code_hashes(),package_verification_hash=sha(OUT/'package_verification/manifest.json'),
        scipy=__import__('scipy').__version__,python=platform.python_version()))
    print('Fixed failures:',len(source_rows),'rows;',len(unique),'unique inputs',Counter(r['robot'] for r in unique.values()))


def failure_study(cfg):
    folder=OUT/'failure_recovery';folder.mkdir(exist_ok=False)
    data=json.loads((OUT/'protocol/failure_inputs.json').read_text());before=code_hashes();rows=[]
    solvers={robot:factory('current_trf',robot,cfg) for robot in cfg['robots']}
    try:
        for solver,kin,_ in solvers.values():
            q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q);solver.solve(pose.position,pose.rotation,q)
        jobs=[(i,r) for i in range(len(data)) for r in range(cfg['point_repeats'])]
        order=np.random.default_rng(cfg['order_seed']).permutation(len(jobs))
        for j in order:
            i,rep=jobs[j];case=data[i];solver,kin,v=solvers[case['robot']]
            r=solver.solve(case['target_position'],case['target_rotation'],case['previous_q'],case['dt'])
            rows.append(dict(input_uid=case['input_uid'],robot=case['robot'],repeat=rep,**r))
    finally:
        for s,_,_ in solvers.values():s.close()
    assert before==code_hashes()
    with gzip.open(folder/'raw.jsonl.gz','xt') as f:
        for r in rows:f.write(json.dumps(clean(r),allow_nan=False,separators=(',',':'))+'\n')
    units=[];witnesses=[]
    for case in data:
        rr=[r for r in rows if r['input_uid']==case['input_uid']];assert len(rr)==cfg['point_repeats']
        _,kin,v=solvers[case['robot']]
        query=IKQuery(Pose(np.array(case['target_position']),np.array(case['target_rotation'])),np.array(case['previous_q']),case['dt'])
        for r in rr:assert v.check(np.array(r['q']),query).accepted==r['accepted']
        success=[r for r in rr if r['accepted']]
        chosen=success[0] if success else rr[0]
        units.append(dict(input_uid=case['input_uid'],robot=case['robot'],sources=case['sources'],
            accepted_repeats=len(success),within20_repeats=sum(r['accepted_within_20ms'] for r in rr),repeats=len(rr),
            q=chosen['q'],position_error_m=chosen['position_error'],orientation_error_rad=chosen['orientation_error'],
            max_joint_step_rad=chosen['max_joint_step_rad'],velocity_utilization=chosen['velocity_utilization'],
            latency_p50_ms=float(np.median([r['total_latency_ns'] for r in rr])/1e6),
            latency_p95_ms=float(np.percentile([r['total_latency_ns'] for r in rr],95)/1e6),
            actual_nfev=[r['native_nfev'] for r in rr],statuses=[r['internal_status'] for r in rr],
            reasons=chosen['verification_reasons']))
        if success:witnesses.append(dict(**case,q=chosen['q'],source_repeat=chosen['repeat'],verified=True))
    write_json(folder/'units.json',units);csv_write(folder/'recovery_table.csv',units)
    write_json(folder/'witnesses.json',witnesses)
    write_json(folder/'completed.json',dict(code_hashes=before,unique_inputs=len(data),repeats=cfg['point_repeats'],
        recovered_any=sum(r['accepted_repeats']>0 for r in units),recovered_all=sum(r['accepted_repeats']==r['repeats'] for r in units),
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    for robot in cfg['robots']:
        r=[u for u in units if u['robot']==robot]
        print(robot,'unique',len(r),'recovered all repeats',sum(u['accepted_repeats']==u['repeats'] for u in r),flush=True)


def trajectory_study(cfg,robot):
    frozen=json.loads((OUT/'protocol/protocol.json').read_text())
    for entry in frozen['input_files'].values():assert sha(ROOT/entry['path'])==entry['sha256']
    # Use the existing whole-trajectory job runner unchanged, with these four
    # explicit factories. No old predictive solver is instantiated or called.
    runner.factory=factory;runner.code_hashes=code_hashes
    run_cfg=dict(cfg,formal={'order_seed':cfg['order_seed']})
    runner.run(OUT/f'development_{robot}',trajectory_items(cfg,robot),cfg['methods'],run_cfg,repeats=cfg['repeats'])


def report(cfg):
    """Reuse frozen run summaries and existing paired-statistics functions."""
    folder=OUT/'reports';folder.mkdir(exist_ok=False)
    witnesses=OUT/'trajectory_witnesses';witnesses.mkdir(exist_ok=False)
    summaries=[];arrays={};audit=Counter();first_failures=[];recoveries=[];extras={};witness_index=[];sources={}
    for robot in cfg['robots']:
        origin=OUT/f'development_{robot}';seal=json.loads((origin/'completed.json').read_text())
        # Verify these new results only, not a new all-repository audit.
        for name,digest in seal['files'].items():assert sha(origin/name)==digest
        sources[robot]=sha(origin/'completed.json')
        _,kin,v,_=context(robot,cfg);data={r['uid']:r for r in trajectory_items(cfg,robot)}
        jobs=json.loads((origin/'summaries.json').read_text());assert len(jobs)==480
        for idx,s in enumerate(jobs):
            rows=read_rows(origin/s['raw_file']);assert len(rows)==150
            item=data[s['uid']];previous=np.array(item['initial_q']);states=[]
            assert s['completion']==all(r['accepted'] for r in rows)
            assert s['deadline_completion']==all(r['accepted_within_20ms'] for r in rows)
            assert s['total_latency_ns']==sum(r['total_latency_ns'] for r in rows)
            counts=Counter();trf_fevals=[]
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==.02
                np.testing.assert_array_equal(r['previous_q'],previous)
                np.testing.assert_array_equal(r['target_position'],item['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],item['target_rotation'][frame])
                query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                check=v.check(q,query);assert check.accepted==r['accepted']
                assert r['accepted_within_20ms']==bool(check.accepted and r['total_latency_ns']<=20_000_000)
                audit['current_commands_checked']+=1;audit['accepted_commands']+=check.accepted
                counts['over_20ms']+=r['total_latency_ns']>20_000_000
                if not r['accepted']:counts['failure:'+r['failure_kind']]+=1
                trf=r if s['method']=='current_trf' else r.get('trf_result')
                if trf is not None:
                    assert 0<=trf['native_nfev']<=50 and trf['counts']['residual_evaluations']<=50
                    assert trf['accounting_remainder_ns']>=0
                    assert trf['total_latency_ns']==sum(trf['phase_times_ns'].values())+trf['accounting_remainder_ns']
                    assert v.check(np.array(trf['q']),query).accepted==trf['accepted']
                    np.testing.assert_allclose(trf['step_scale'],kin.limits.velocity*.02+v.config.velocity_tolerance,rtol=0,atol=0)
                    trace=trf['iteration_trace']
                    assert not any(t['accepted'] for t in trace[:-1])
                    for t in trace:
                        cc=v.check(np.array(t['q']),query)
                        assert cc.accepted==t['accepted'];audit['trf_iterates_checked']+=1
                    counts['trf_calls']+=1;counts['optimizer_calls']+=trf['optimizer_called']
                    counts['trf_native:'+str(trf['native_return_code'])]+=1
                    counts['trf_stop:'+trf['internal_status']]+=1
                    trf_fevals.append(trf['native_nfev'])
                if s['method']=='trac_then_current_trf':
                    native=r['trac_result'];nq=np.array(native['q']) if native['q'] is not None else np.full(kin.nq,np.nan)
                    assert v.check(nq,query).accepted==native['accepted']
                    assert r['recovery_called']==(trf is not None)
                    assert r['total_latency_ns']==r['trac_outer_ns']+r['trf_outer_ns']+r['composition_remainder_ns']
                    assert r['composition_remainder_ns']>=0
                    if native['accepted']:
                        assert trf is None;np.testing.assert_array_equal(r['q'],native['q'])
                    else:
                        counts['trac_rejections']+=1
                        if trf is not None:
                            assert 0<trf['available_budget_at_entry_ns']<=r['remaining_before_trf_ns']<=20_000_000-r['trac_outer_ns']
                            np.testing.assert_array_equal(r['q'],trf['q'])
                            counts['same_frame_recoveries']+=r['accepted']
                            counts['within20_recoveries']+=r['accepted_within_20ms']
                        else:assert r['remaining_before_trf_ns']==0
                        recoveries.append({k:r.get(k) for k in ('robot','uid','site_id','method','repeat','family','frame','previous_q',
                            'target_position','target_rotation','q','accepted','accepted_within_20ms','position_error','orientation_error',
                            'total_latency_ns','remaining_before_trf_ns','trac_outer_ns','trf_outer_ns','trac_result','trf_result')})
                if s['first_failure_frame']==frame:
                    first_failures.append({k:r.get(k) for k in ('robot','uid','site_id','method','repeat','family','frame','previous_q',
                        'target_position','target_rotation','q','position_error','orientation_error','velocity_utilization','failure_kind',
                        'total_latency_ns','internal_status','native_return_code')})
                if check.accepted:previous=q.copy()
                np.testing.assert_array_equal(r['accepted_state_q'],previous)
                states.append(dict(frame=frame,dt=.02,previous_q=r['previous_q'],target_position=r['target_position'],
                    target_rotation=r['target_rotation'],q=r['q'],latency_ns=r['total_latency_ns'],verified=r['accepted']))
            summaries.append(s);extras[s['run_id']]=dict(counts=counts,trf_fevals=trf_fevals)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if s['completion']:
                path=witnesses/f'{s["run_id"]}.jsonl.gz'
                with gzip.open(path,'xt') as f:
                    for r in states:f.write(json.dumps(clean(r),allow_nan=False,separators=(',',':'))+'\n')
                witness_index.append(dict(robot=robot,method=s['method'],uid=s['uid'],repeat=s['repeat'],frames=150,
                    initial_q=item['initial_q'],deadline_completion=s['deadline_completion'],file=path.name,sha256=sha(path)))
            if idx%120==0:print('verify',robot,idx+1,'/480',flush=True)
    assert len(summaries)==960 and audit['current_commands_checked']==144000
    main=group_table(summaries,arrays);families=group_table(summaries,arrays,True)
    for table in (main,families):
        for row in table:
            ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
            c=Counter();evals=[]
            for s in ss:c.update(extras[s['run_id']]['counts']);evals+=extras[s['run_id']]['trf_fevals']
            for key in ('backup_rate','changed_from_backup_rate'):row.pop(key,None) # legacy predictive-only fields are inapplicable
            row.update(complete_frame_calls=sum(s['frames'] for s in ss),over_20ms_frames=c['over_20ms'],
                trf_calls=c['trf_calls'],optimizer_calls=c['optimizer_calls'],trac_rejections=c['trac_rejections'],
                same_frame_recoveries=c['same_frame_recoveries'],within20_recoveries=c['within20_recoveries'],
                mean_trf_nfev=float(np.mean(evals)) if evals else None,
                frame_failure_kinds={k.removeprefix('failure:'):val for k,val in c.items() if k.startswith('failure:')},
                trf_native_statuses={k.removeprefix('trf_native:'):val for k,val in c.items() if k.startswith('trf_native:')},
                trf_stop_reasons={k.removeprefix('trf_stop:'):val for k,val in c.items() if k.startswith('trf_stop:')})
    units=[];pairs=[];changes=[];completion={};metric_names=('completion','deadline_completion','total_latency_ns','frame_success','successful_prefix','acceleration_rms')
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==u) for u in uids]
        for method in cfg['methods']:
            completion.setdefault(robot,{})[method]={str(rep):sorted(s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']) for rep in range(3)}
            for uid,family in zip(uids,fam):
                rr=[s for s in ss if s['uid']==uid and s['method']==method];assert sorted(s['repeat'] for s in rr)==[0,1,2]
                val={k:float(np.mean([r[k] for r in rr])) for k in metric_names}
                unit[(method,uid)]=val;units.append(dict(robot=robot,uid=uid,site_id=rr[0]['site_id'],family=family,method=method,repeats=3,**val))
        comparisons=[('current_trf','trac_task_5ms'),('current_trf','pink_qp'),('trac_then_current_trf','trac_task_5ms'),
            ('trac_then_current_trf','pink_qp'),('trac_then_current_trf','current_trf')]
        for method,base in comparisons:
            for metric in metric_names:
                a=np.array([unit[(method,u)][metric] for u in uids]);b=np.array([unit[(base,u)][metric] for u in uids])
                pairs.append(dict(robot=robot,method=method,baseline=base,metric=metric,
                    **paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,family in zip(uids,fam):
                a=unit[(method,uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,method=method,baseline=base,uid=uid,family=family,
                    site_id=next(s['site_id'] for s in ss if s['uid']==uid),method_fraction=a,baseline_fraction=b,
                    change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
    point=json.loads((OUT/'failure_recovery/units.json').read_text());point_summary=[];source_rows=[]
    for robot in cfg['robots']:
        rr=[r for r in point if r['robot']==robot]
        point_summary.append(dict(robot=robot,unique_inputs=len(rr),recovered_all=sum(r['accepted_repeats']==3 for r in rr),
            recovered_any=sum(r['accepted_repeats']>0 for r in rr),within20_all=sum(r['within20_repeats']==3 for r in rr),
            trajectory_uids=len({s['trajectory_uid'] for r in rr for s in r['sources']}),
            recovered_input_median_latency_range_ms=[min(r['latency_p50_ms'] for r in rr if r['accepted_repeats']),max(r['latency_p50_ms'] for r in rr if r['accepted_repeats'])]))
        for method in sorted({s['method'] for r in rr for s in r['sources']}):
            sub=[r for r in rr if any(s['method']==method for s in r['sources'])]
            source_rows.append(dict(robot=robot,source_method=method,unique_inputs=len(sub),
                recovered_all=sum(r['accepted_repeats']==3 for r in sub),unrecovered=sum(r['accepted_repeats']==0 for r in sub),
                note='Source categories can overlap for identical inputs; do not sum to independent total.'))
    csv_write(folder/'main_table.csv',main);csv_write(folder/'family_table.csv',families)
    csv_write(folder/'trajectory_units.csv',units);csv_write(folder/'paired_comparisons.csv',pairs)
    csv_write(folder/'gained_lost_uids.csv',changes);csv_write(folder/'first_failure_inputs.csv',first_failures)
    csv_write(folder/'point_recovery_by_source.csv',source_rows);csv_write(folder/'point_recovery_summary.csv',point_summary)
    csv_write(folder/'run_summaries.csv',summaries);write_json(folder/'completion_uids.json',completion)
    write_json(folder/'source_data.json',dict(main=main,family=families,pairs=pairs,changes=changes,units=units,point=point_summary,point_by_source=source_rows))
    with gzip.open(folder/'same_frame_recovery_records.jsonl.gz','xt') as f:
        for r in recoveries:f.write(json.dumps(clean(r),allow_nan=False,separators=(',',':'))+'\n')
    write_json(witnesses/'index.json',witness_index)
    write_json(folder/'verification.json',dict(counts=dict(audit),accepted_contract_violations=0,full_success_witnesses=len(witness_index),
        checks='Original current verifier, actual own feedback, all 150 targets, callback stopping and 50-call limit, at most one same-input TRF after rejected TRAC with original remaining period. No solver rerun.'))
    write_json(folder/'manifest.json',dict(sources=sources,created=utc(),baseline=BASELINE,reporting_code_hash=sha(__file__),
        independent_unit='40 trajectory UIDs/robot; average 3 repeats within UID; family-stratified paired bootstrap, 4000 resamples, descriptive unadjusted 95% intervals',
        timing='Complete outer latency including failures/timeouts. For composition, trac_outer_ns + trf_outer_ns + composition_remainder_ns is the total; inherited stage fields belong to the selected inner solver, not to a second outer total.',
        point_scope='44 exact selected hard inputs from 9 trajectory UIDs, 3 timing repeats; no general incidence inference',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},witness_index_sha256=sha(witnesses/'index.json')))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],3),'seconds',round(r['cumulative_ms_per_sweep']/1000,3))


def package_check(cfg):
    folder=OUT/'package_verification';folder.mkdir(exist_ok=False)
    package=OUT/'task_package'
    cases=json.loads((package/'recorded_inputs.json').read_text())
    supplied=json.loads((package/'repeated_results.json').read_text())['summary']
    with (OLD/'reports/first_failure_inputs.csv').open() as f:sources=list(csv.DictReader(f))
    _,kin,v,_=context('panda',cfg);rows=[]
    for case in cases:
        matches=[(i,r) for i,r in enumerate(sources) if r['robot']=='panda' and r['method']==case['source'] and
                 np.array_equal(json.loads(r['previous_q']),case['previous']) and
                 np.array_equal(json.loads(r['target_position']),case['p']) and
                 np.array_equal(json.loads(r['target_rotation']),case['R'])]
        assert matches,case['id']
        index,src=matches[0]
        # Read the actual source frame for dt; the supplied transcription omits dt.
        item=json.loads((OLD/f'validated_panda/runs/panda_{src["site_id"]}_{src["method"]}_r{src["repeat"]}.summary.json').read_text())
        from confik.correction_reserve.study import read_rows
        raw=read_rows(OLD/'validated_panda'/item['raw_file'])[int(src['frame'])]
        dt=raw['dt'];assert dt==.02
        result=next(r for r in supplied if r['case']==case['id'] and r['method']=='least_squares')
        q=np.array(result['q']);previous=np.array(case['previous'])
        query=IKQuery(Pose(np.array(case['p']),np.array(case['R'])),previous,dt)
        check=v.check(q,query);step=kin.limits.velocity*dt+v.config.velocity_tolerance
        row=dict(case=case['id'],uid=src['uid'],source_method=src['method'],source_row=index,
            frame=int(src['frame']),dt=dt,previous_q=previous,target_position=case['p'],target_rotation=case['R'],q=q,
            supplied_success=bool(result['accepted_repeats']),accepted=check.accepted,
            finite=check.finite_ok,joint_limit_ok=check.joint_limit_ok,velocity_ok=check.velocity_ok,
            position_error_m=check.position_error,orientation_error_rad=check.orientation_error,
            max_joint_step_rad=float(np.max(np.abs(q-previous))),velocity_utilization=float(np.max(np.abs(q-previous)/step)),
            reasons=list(check.reasons),status='verified_witness' if check.accepted else 'not_found_no_infeasibility_claim')
        rows.append(row)
        print(case['id'],check.accepted,'position_mm',check.position_error*1000,
            'orientation_deg',np.rad2deg(check.orientation_error),'step',row['velocity_utilization'],list(check.reasons),flush=True)
    write_json(folder/'commands.json',rows);csv_write(folder/'acceptance.csv',rows)
    write_json(folder/'manifest.json',dict(baseline=BASELINE,archive_sha256=sha(ROOT/'current_ik_workbench.zip'),
        files={p.name:sha(p) for p in package.iterdir() if p.is_file()},
        source_csv_sha256=sha(OLD/'reports/first_failure_inputs.csv'),public_backend=type(kin).__name__,
        verifier_config=v.config.__dict__,joint_names=kin.joint_names,solver_calls=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['package_check','prepare','failures','trajectories','report'])
    parser.add_argument('--robot',choices=['panda','ur5e'])
    args=parser.parse_args();cfg=yaml.safe_load((ROOT/'configs/current_frame_recovery.yaml').read_text())
    if args.action=='package_check':package_check(cfg)
    elif args.action=='prepare':prepare(cfg)
    elif args.action=='failures':failure_study(cfg)
    elif args.action=='report':report(cfg)
    elif args.action=='trajectories':
        if not args.robot:parser.error('--robot required')
        trajectory_study(cfg,args.robot)


if __name__=='__main__':main()
