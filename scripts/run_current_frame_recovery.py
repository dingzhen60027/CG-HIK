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
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['package_check','prepare','failures','trajectories'])
    parser.add_argument('--robot',choices=['panda','ur5e'])
    args=parser.parse_args();cfg=yaml.safe_load((ROOT/'configs/current_frame_recovery.yaml').read_text())
    if args.action=='package_check':package_check(cfg)
    elif args.action=='prepare':prepare(cfg)
    elif args.action=='failures':failure_study(cfg)
    elif args.action=='trajectories':
        if not args.robot:parser.error('--robot required')
        trajectory_study(cfg,args.robot)


if __name__=='__main__':main()
