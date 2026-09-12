#!/usr/bin/env python3
"""Single entry point for TAR-IK mathematics, online runs and source-backed report."""
import argparse
import copy
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import types
import zipfile

import numpy as np
import yaml

from confik.task_recourse import TaskRecourseIK, scenario_nodes, common_model
from confik.correction_reserve.native_geometry import NativeGeometry
from confik.correction_reserve.study import ROOT, clean, write_json, sha, utc, context, execute_trajectory, read_rows
from confik.correction_reserve.reporting import csv_write, group_table, paired_intervals, LABELS
from confik.correction_reserve.geometry import predict_target, perturb_target, task_scale
from confik.correction_reserve.pink_adapter import PinkAdapter
from confik.task_contract_alignment.outcomes import ContractSolver
from confik.types import Pose, IKQuery
from confik.geometry import pose_error

LABELS.update(tar_free='TAR-IK free recourse',tar_fixed='Same-core fixed compensation',
              tar_nominal='Same-core nominal prediction',tar_old='Original TAR-IK',
              tar_corrected='Numerically corrected TAR-IK')
DEFAULT=ROOT/'outputs/task_recourse'
NUMERIC_BASELINE='1a9db5071e6212bc25a9cdae9fe278328e47646a'
NUMERIC_ROOT=DEFAULT/'numerical_completion_development'


def hashes():
    paths=['src/confik/task_recourse.py','configs/task_recourse.yaml','scripts/run_task_recourse.py',
           'src/confik/correction_reserve/geometry.py','src/confik/correction_reserve/native_geometry.py',
           'src/confik/correction_reserve/pink_adapter.py','src/confik/task_contract_alignment/outcomes.py',
           'src/confik/task_contract_alignment/trac_adapter.py','src/confik/solvers/verifier.py',
           'src/confik/continuation_mechanism/observation.py','tmp/task_contract_build/libcontract_trac.so']
    return {p:sha(ROOT/p) for p in paths}


def items(cfg,robot=None,integration=False):
    result=[]
    for r in cfg['robots']:
        if robot and robot!=r:continue
        data=json.loads((ROOT/cfg['development'][r+'_targets']).read_text())
        assert len(data)==40
        assert sorted(Counter(x['family'] for x in data).values())==[10]*4
        seen=Counter()
        for row in data:
            assert len(row['target_position'])==150 and row['dt']==.02
            seen[row['family']]+=1
            if not integration or seen[row['family']]<=2:result.append(dict(robot=r,**row))
    return result


def factory(method,robot,cfg):
    source,kin,v,urdf=context(robot,cfg)
    if method.startswith('tar_'):
        solver=TaskRecourseIK(kin,v,source,str(ROOT/cfg['native_trac_library']),urdf,
                              mode=method.removeprefix('tar_'),config=cfg['optimizer'])
    elif method=='pink_qp':solver=PinkAdapter(kin,v,source,urdf)
    else:solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    return solver,kin,v


def prepare(root,cfg):
    root.mkdir(parents=True,exist_ok=False)
    data=items(cfg)
    write_json(root/'input_identities.json',dict(created=utc(),configuration=cfg,
        input_files={r:dict(path=cfg['development'][r+'_targets'],sha256=sha(ROOT/cfg['development'][r+'_targets'])) for r in cfg['robots']},
        trajectories=[{k:i[k] for k in ('robot','uid','site_id','family','dt')} for i in data],
        integration_uids=[i['uid'] for i in items(cfg,integration=True)],
        mechanism_frame=cfg['mechanism']['frame'],selection='fixed before TAR outcomes; no new trajectories'))
    archive=ROOT/'codex_task_recourse_plan.zip'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe task archive')
        z.extractall(root/'task_package')
    write_json(root/'task_package_hash.json',dict(sha256=sha(archive),path=archive.name))
    write_json(root/'protocol_code_hashes.json',hashes())
    print('Prepared 80 existing development identities; no solver calls.')


def math_tests(root):
    out=root/'math';out.mkdir(exist_ok=False)
    package=next((root/'task_package').glob('*/CODEX_TASK.md')).parent
    for name in ('check_reserve_proxy.py','test_linear_recourse.py'):
        shutil.copy2(package/name,out/name)
        subprocess.run([sys.executable,str(out/name)],check=True,cwd=ROOT)
    spec=importlib.util.spec_from_file_location('recourse_regression',ROOT/'tests/test_task_recourse.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    write_json(out/'actual_joint_socp.json',mod.mathematical_regressions())
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_task_recourse.py'],
        cwd=ROOT,text=True,capture_output=True)
    write_json(out/'unit_tests.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)


def run(root,cfg,robot,integration=False):
    stage='integration' if integration else 'development'
    folder=root/f'{stage}_{robot}';folder.mkdir(exist_ok=False)
    identity=json.loads((root/'input_identities.json').read_text())
    for r,entry in identity['input_files'].items():assert sha(ROOT/entry['path'])==entry['sha256']
    data=items(cfg,robot,integration);reps=1 if integration else cfg['repeats']
    code=hashes();write_json(folder/'started.json',dict(utc=utc(),code_hashes=code,configuration=cfg,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        python=platform.python_version(),affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        stage=stage,scope='whole trajectory, nested repeats; all frames and timeouts retained'))
    jobs=[(i,m,r) for i in data for m in cfg['methods'] for r in range(reps)]
    rng=np.random.default_rng(cfg['order_seed']);order=rng.permutation(len(jobs))
    write_json(folder/'job_order.json',[dict(uid=jobs[j][0]['uid'],method=jobs[j][1],repeat=jobs[j][2]) for j in order])
    (folder/'runs').mkdir();solvers={};summaries=[];begin=time.monotonic()
    try:
        for count,j in enumerate(order):
            item,method,repeat=jobs[j]
            if method not in solvers:
                solvers[method]=factory(method,robot,cfg)
                s,k,v=solvers[method];q=np.array(item['initial_q']);p=k.forward(q)
                s.solve(p.position,p.rotation,q,item['dt'])
                write_json(folder/f'{method}_adapter.json',dict(type=type(s).__name__,warmup='one stationary call, reset before each trajectory',
                    public_backend=type(k).__name__,joint_names=k.joint_names,
                    native_metadata=s.metadata() if callable(getattr(s,'metadata',None)) else None))
            solver,kin,v=solvers[method]
            rows,summary=execute_trajectory(solver,kin,v,item,method,repeat)
            # Future diagnostics occur only AFTER the causal online trajectory.
            for t,row in enumerate(rows[:-1]):
                if 'predicted_position' in row:
                    prediction=Pose(np.array(row['predicted_position']),np.array(row['predicted_rotation']))
                    truth=Pose(np.array(item['target_position'][t+1]),np.array(item['target_rotation'][t+1]))
                    error=pose_error(truth,prediction)/task_scale(v)
                    row.update(actual_next_prediction_error=error,next_error_l1=float(np.abs(error).sum()),
                        actual_next_in_l1=bool(np.abs(error).sum()<=1),next_frame_accepted=rows[t+1]['accepted'])
            rid=f'{robot}_{item["site_id"]}_{method}_r{repeat}'
            path=folder/'runs'/f'{rid}.jsonl.gz'
            with gzip.open(path,'xt',encoding='utf8') as f:
                for row in rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
            summary.update(robot=robot,method=method,repeat=repeat,uid=item['uid'],site_id=item['site_id'],
                family=item['family'],run_id=rid,raw_file=str(path.relative_to(folder)),
                optimization_rate=float(np.mean([r.get('optimization_called',False) for r in rows])),
                command_change_rate=float(np.mean([r.get('command_changed',False) for r in rows])),
                mean_intervention=float(np.mean([r.get('intervention') or 0 for r in rows])),
                all_nodes_verified_rate=float(np.mean([r.get('all_nodes_verified',False) for r in rows])),
                fallback_decisions=dict(Counter(r.get('decision','baseline') for r in rows)))
            summaries.append(summary);write_json(folder/'runs'/f'{rid}.summary.json',summary)
            print(f'{stage} {robot} {count+1}/{len(jobs)} {method} {item["site_id"]} r{repeat} '
                f'complete={summary["completion"]} P95={summary["p95_ms"]:.3f} elapsed={(time.monotonic()-begin)/60:.1f}min',flush=True)
    finally:
        for solver,_,_ in solvers.values():solver.close()
    if hashes()!=code:raise AssertionError('experiment code changed during run')
    write_json(folder/'summaries.json',summaries)
    write_json(folder/'completed.json',dict(utc=utc(),runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))


def mechanism(root,cfg,robot):
    folder=root/f'mechanism_{robot}';folder.mkdir(exist_ok=False)
    code=hashes();data=items(cfg,robot);fixed=cfg['mechanism']['frame']
    summaries=json.loads((root/f'development_{robot}'/'summaries.json').read_text())
    solvers={m:factory(m,robot,cfg) for m in cfg['methods']}
    reference,kin,v=factory(cfg['mechanism']['reference'],robot,cfg)
    directions=[('optimized_axis' if np.any(w) else 'nominal',w) for w in scenario_nodes()]
    for radius in cfg['mechanism']['mixed_l1_radii']:
        for a,b in cfg['mechanism']['mixed_axes']:
            for sign in (-1.,1.):
                w=np.zeros(6);w[a]=radius/2;w[b]=sign*radius/2
                directions.append(('mixed_inside' if radius<=1 else 'mixed_outside',w))
    write_json(folder/'started.json',dict(code_hashes=code,frame=fixed,reference=cfg['mechanism'],
        directions=[dict(group=g,w=w) for g,w in directions],current_method_calls_per_state=1,
        current_call_note='one causal call per method; three downstream searches, not three independent states'))
    all_summaries=[]
    try:
        for index,item in enumerate(data):
            s=next(s for s in summaries if s['uid']==item['uid'] and s['method']=='trac_task_5ms' and s['repeat']==0)
            history=read_rows(root/f'development_{robot}'/s['raw_file'])
            previous=np.array(history[fixed]['previous_q'])
            current=Pose(np.array(item['target_position'][fixed]),np.array(item['target_rotation'][fixed]))
            last=Pose(np.array(item['target_position'][fixed-1]),np.array(item['target_rotation'][fixed-1]))
            prediction=predict_target(current,last)
            target_list=[(group,w,perturb_target(prediction,w,1,task_scale(v))) for group,w in directions]
            truth=Pose(np.array(item['target_position'][fixed+1]),np.array(item['target_rotation'][fixed+1]))
            target_list.append(('actual_next',pose_error(truth,prediction)/task_scale(v),truth))
            state_rows=[];inputs=[]
            for method in cfg['methods']:
                solver,_,_=solvers[method]
                if hasattr(solver,'reset'):solver.reset(previous)
                if isinstance(solver,TaskRecourseIK):solver.last_target=last
                result=solver.solve(current.position,current.rotation,previous,.02)
                inputs.append(dict(method=method,result=result))
                for node,(group,w,target) in enumerate(target_list):
                    for repeat in range(cfg['mechanism']['reference_repeats']):
                        if result['accepted']:
                            q=np.array(result['q']);r=reference.solve(target.position,target.rotation,q,.02)
                        else:r=dict(accepted=False,q=None,internal_status='current_command_unavailable',total_latency_ns=0)
                        row=dict(robot=robot,uid=item['uid'],site_id=item['site_id'],family=item['family'],
                            method=method,frame=fixed,node=node,group=group,w=w,l1=float(np.abs(w).sum()),
                            inside_l1=bool(np.abs(w).sum()<=1),repeat=repeat,
                            current_accepted=result['accepted'],current_q=result['q'],
                            target_position=target.position,target_rotation=target.rotation,
                            reference_result=r,witness=bool(r['accepted']))
                        state_rows.append(row)
            with gzip.open(folder/f'{item["site_id"]}.jsonl.gz','xt') as f:
                for row in state_rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
            write_json(folder/f'{item["site_id"]}_inputs.json',dict(uid=item['uid'],previous_q=previous,
                target_position=current.position,target_rotation=current.rotation,last_position=last.position,
                last_rotation=last.rotation,earlier_history_failures=sum(not r['accepted'] for r in history[:fixed]),
                methods=inputs))
            for method in cfg['methods']:
                for group in sorted({r['group'] for r in state_rows}):
                    rows=[r for r in state_rows if r['method']==method and r['group']==group]
                    all_summaries.append(dict(robot=robot,uid=item['uid'],method=method,family=item['family'],group=group,
                        correction_success=float(np.mean([r['witness'] for r in rows])),
                        current_accepted=rows[0]['current_accepted'],calls=len(rows),
                        total_reference_ns=sum(r['reference_result']['total_latency_ns'] for r in rows)))
            print(f'mechanism {robot} {index+1}/40',flush=True)
    finally:
        reference.close()
        for solver,_,_ in solvers.values():solver.close()
    assert code==hashes()
    write_json(folder/'summaries.json',all_summaries)
    write_json(folder/'completed.json',dict(utc=utc(),files={str(p.relative_to(folder)):sha(p) for p in folder.iterdir() if p.is_file()}))


def report(root,cfg):
    # Read-only source verification. No outcome-driven numerical reruns.
    for robot in cfg['robots']:
        for stage in ('development','mechanism'):
            folder=root/f'{stage}_{robot}'
            seal=json.loads((folder/'completed.json').read_text())
            for name,digest in seal['files'].items():assert sha(folder/name)==digest,(folder,name)
    out=root/'reports';out.mkdir(exist_ok=False)
    summaries=[];arrays={};extra=defaultdict(list);failures=[];model_rows=[]
    for robot in cfg['robots']:
        folder=root/f'development_{robot}'
        assert (folder/'completed.json').exists()
        _,kin,v,urdf=context(robot,cfg);geometry=NativeGeometry(kin,urdf)
        scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
        for s in json.loads((folder/'summaries.json').read_text()):
            rows=read_rows(folder/s['raw_file']);summaries.append(s)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if s['first_failure_frame'] is not None:
                r=rows[s['first_failure_frame']]
                failures.append({k:r.get(k) for k in ('robot','uid','method','repeat','family','frame','previous_q',
                    'target_position','target_rotation','q','position_error','orientation_error','failure_kind','total_latency_ns')})
            for r in rows:
                if s['method'].startswith('tar_'):
                    extra[(robot,s['method'])].append(dict(tau=r['tau_actual'],
                        next_inside=r.get('actual_next_in_l1'),next_success=r.get('next_frame_accepted'),
                        all_verified=r['all_nodes_verified'],phases=r['phase_times_ns'],decision=r['decision']))
                    # Compare the SAME backtracked candidate in the affine model
                    # and true FK, not a solver optimum with a different fallback.
                    anchor=np.array(r['backup']['q'] if r['backup']['accepted'] else r['previous_q'])
                    z0=anchor.copy();outer=None
                    predicted=Pose(np.array(r['predicted_position']),np.array(r['predicted_rotation']))
                    nodes=scenario_nodes(s['method']=='tar_nominal')
                    for trial in r['nonlinear_trials']:
                        if trial['outer']!=outer:
                            outer=trial['outer'];origin=z0.copy()
                            e,F,C,_=common_model(geometry,predicted,origin,scale,step)
                        zs=np.array(trial['z'])
                        affine=e+((zs-origin)/step)@F.T+nodes@C.T
                        affine_tau=max(0.,float(np.max(np.stack((np.linalg.norm(affine[:,:3],axis=1),np.linalg.norm(affine[:,3:],axis=1)))))-1.)
                        true=np.array([pose_error(perturb_target(predicted,w,1,scale),geometry.forward(z))/scale for w,z in zip(nodes,zs)])
                        actual_tau=max(0.,float(np.max(np.stack((np.linalg.norm(true[:,:3],axis=1),np.linalg.norm(true[:,3:],axis=1)))))-1.)
                        assert abs(actual_tau-trial['tau_actual'])<1e-9
                        model_rows.append(dict(robot=robot,uid=s['uid'],method=s['method'],repeat=s['repeat'],frame=r['frame'],
                            outer=outer,alpha=trial['alpha'],adopted=trial['adopted'],geometric=trial['geometric'],
                            affine_tau=affine_tau,actual_tau=actual_tau,difference=actual_tau-affine_tau))
                        if trial['adopted']:z0=zs[0].copy()
    main=group_table(summaries,arrays);families=group_table(summaries,arrays,True)
    for row in main+families:
        ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
        for metric in ('optimization_rate','command_change_rate','mean_intervention','all_nodes_verified_rate'):
            row[metric]=float(np.mean([s[metric] for s in ss]))
    units=[];pairs=[];changes=[];completion={}
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==uid) for uid in uids]
        for method in cfg['methods']:
            completion.setdefault(robot,{})[method]={str(rep):[s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']] for rep in range(3)}
            for uid,family in zip(uids,fam):
                rows=[s for s in ss if s['method']==method and s['uid']==uid];assert len(rows)==3
                val={k:float(np.mean([s[k] for s in rows])) for k in ('completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms')}
                unit[(method,uid)]=val;units.append(dict(robot=robot,method=method,uid=uid,family=family,**val))
        for base in [m for m in cfg['methods'] if m!='tar_free']:
            for metric in ('completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms'):
                a=np.array([unit[('tar_free',uid)][metric] for uid in uids]);b=np.array([unit[(base,uid)][metric] for uid in uids])
                pairs.append(dict(robot=robot,method='tar_free',baseline=base,metric=metric,
                    **paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,family in zip(uids,fam):
                a=unit[('tar_free',uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,baseline=base,uid=uid,family=family,free_fraction=a,baseline_fraction=b,
                    change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
    diagnostics=[]
    for (robot,method),rows in sorted(extra.items()):
        next_rows=[r for r in rows if r['next_inside'] is not None]
        phases={name:dict(p50_ms=float(np.median([r['phases'].get(name,0) for r in rows]))/1e6,
                          total_ms=float(sum(r['phases'].get(name,0) for r in rows))/1e6) for name in rows[0]['phases']}
        diagnostics.append(dict(robot=robot,method=method,decisions=dict(Counter(r['decision'] for r in rows)),
            actual_next_l1_coverage=float(np.mean([r['next_inside'] for r in next_rows])),phases=phases,
            mean_actual_tau=float(np.mean([r['tau'] for r in rows if r['tau'] is not None])),
            next_success_when_all_nodes_verified=float(np.mean([r['next_success'] for r in next_rows if r['all_verified']])) if any(r['all_verified'] for r in next_rows) else None))
    mechanisms=[];mechanism_units=[];mechanism_pairs=[];witness_counts=Counter()
    for robot in cfg['robots']:
        folder=root/f'mechanism_{robot}'
        if not (folder/'completed.json').exists():raise RuntimeError('mechanism incomplete')
        rows=json.loads((folder/'summaries.json').read_text())
        _,kin,v,_=context(robot,cfg)
        for item in items(cfg,robot):
            saved=json.loads((folder/f'{item["site_id"]}_inputs.json').read_text())
            query=IKQuery(Pose(np.array(saved['target_position']),np.array(saved['target_rotation'])),np.array(saved['previous_q']),.02)
            returned={x['method']:x['result'] for x in saved['methods']}
            for method,r in returned.items():
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                assert v.check(q,query).accepted==r['accepted']
                witness_counts['common_current_calls']+=1
            raw=read_rows(folder/f'{item["site_id"]}.jsonl.gz')
            for r in raw:
                result=returned[r['method']]
                assert r['current_q']==result['q'] and r['current_accepted']==result['accepted']
                witness_counts['reference_rows']+=1
                if result['accepted']:
                    command=r['reference_result']['q']
                    command=np.array(command) if command is not None else np.full(kin.nq,np.nan)
                    query_next=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),np.array(result['q']),.02)
                    check=v.check(command,query_next)
                    assert bool(check.accepted)==r['witness']
                    witness_counts['actual_reference_calls']+=1
                    witness_counts['verified_witnesses']+=bool(check.accepted)
                else:assert not r['witness']
            inside=next(r['inside_l1'] for r in raw if r['group']=='actual_next')
            for summary in [r for r in rows if r['uid']==item['uid'] and r['group']=='actual_next']:
                rows.append(dict(summary,group='actual_next_inside' if inside else 'actual_next_outside'))
        mechanism_units.extend(rows)
        for method in cfg['methods']:
            for group in sorted({r['group'] for r in rows}):
                sub=[r for r in rows if r['method']==method and r['group']==group]
                mechanisms.append(dict(robot=robot,method=method,group=group,states=len(sub),
                    current_available=sum(r['current_accepted'] for r in sub),
                    correction_success=float(np.mean([r['correction_success'] for r in sub]))))
        for group in sorted({r['group'] for r in rows}):
            free=sorted([r for r in rows if r['method']=='tar_free' and r['group']==group],key=lambda r:r['uid'])
            for base in [m for m in cfg['methods'] if m!='tar_free']:
                other=sorted([r for r in rows if r['method']==base and r['group']==group],key=lambda r:r['uid'])
                assert [r['uid'] for r in free]==[r['uid'] for r in other]
                mechanism_pairs.append(dict(robot=robot,group=group,baseline=base,states=len(free),
                    **paired_intervals(np.array([r['correction_success'] for r in free]),np.array([r['correction_success'] for r in other]),
                        [r['family'] for r in free],cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
    model_summary=[]
    for robot in cfg['robots']:
        for method in ('tar_free','tar_fixed','tar_nominal'):
            for adopted_only in (False,True):
                sub=[r for r in model_rows if r['robot']==robot and r['method']==method and (not adopted_only or r['adopted'])]
                delta=np.array([r['difference'] for r in sub])
                model_summary.append(dict(robot=robot,method=method,subset='adopted' if adopted_only else 'all_trials',trials=len(sub),
                    mean_affine_tau=float(np.mean([r['affine_tau'] for r in sub])),mean_actual_tau=float(np.mean([r['actual_tau'] for r in sub])),
                    mean_absolute_discrepancy=float(np.mean(np.abs(delta))),p95_absolute_discrepancy=float(np.percentile(np.abs(delta),95)),
                    maximum_absolute_discrepancy=float(np.max(np.abs(delta))),
                    affine_zero_actual_positive=sum(r['affine_tau']<=1e-5 and r['actual_tau']>1e-5 for r in sub)))
    for name,rows in [('main_table',main),('family_table',families),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('first_failure_inputs',failures),('run_summaries',summaries),
                      ('same_input_correction',mechanisms),('same_input_units',mechanism_units),
                      ('same_input_paired_comparisons',mechanism_pairs),('affine_nonlinear_summary',model_summary)]:csv_write(out/f'{name}.csv',rows)
    with gzip.open(out/'matched_affine_nonlinear_trials.jsonl.gz','xt') as f:
        for row in model_rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
    write_json(out/'same_input_witness_audit.json',dict(counts=dict(witness_counts),new_solver_calls=0,
        meaning='All successful corrections reverified from their own saved current command at the common input; failed searches do not prove infeasibility.'))
    write_json(out/'completion_uids.json',completion)
    write_json(out/'source_data.json',dict(main=main,family=families,pairs=pairs,changes=changes,diagnostics=diagnostics,
        mechanism=mechanisms,mechanism_pairs=mechanism_pairs,matched_model=model_summary))
    write_json(out/'manifest.json',dict(utc=utc(),code_hashes=hashes(),units=80,runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        inference='whole UID, average three repeats, paired family-stratified bootstrap, unadjusted descriptive 95% intervals',
        sources={str(p.relative_to(root)):sha(p) for p in root.glob('*/completed.json')},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    for row in main:print(row['robot'],row['method'],row['completion_by_repeat'],row['deadline_completion_by_repeat'],
        [round(row[x],3) for x in ('p50_ms','p95_ms','p99_ms')],round(row['cumulative_ms_per_sweep']/1000,3))


def audit(root,cfg,stage):
    """No new IK calls: reconstruct causal feedback, merit and saved witnesses."""
    totals=Counter();largest_tau_error=0.;largest_merit_error=0.
    for robot in cfg['robots']:
        _,kin,v,_=context(robot,cfg);scale=task_scale(v)
        byuid={i['uid']:i for i in items(cfg,robot)}
        folder=root/f'{stage}_{robot}'
        for s in json.loads((folder/'summaries.json').read_text()):
            rows=read_rows(folder/s['raw_file']);item=byuid[s['uid']];previous=np.array(item['initial_q']);last=None
            assert len(rows)==150
            for row in rows:
                totals['frames']+=1
                np.testing.assert_array_equal(previous,row['previous_q'])
                target=Pose(np.array(row['target_position']),np.array(row['target_rotation']))
                query=IKQuery(target,previous,.02)
                q=np.array(row['q']) if row['q'] is not None else np.full(kin.nq,np.nan)
                check=v.check(q,query);assert check.accepted==row['accepted']
                if row['accepted']:previous=q.copy();totals['accepted']+=1
                np.testing.assert_array_equal(previous,row['accepted_state_q'])
                assert row['accepted_within_20ms']==bool(row['accepted'] and row['total_latency_ns']<=20_000_000)
                if row['method'].startswith('tar_'):
                    prediction=predict_target(target,last)
                    np.testing.assert_array_equal(prediction.position,row['predicted_position'])
                    np.testing.assert_array_equal(prediction.rotation,row['predicted_rotation'])
                    assert len(row['native_status'])<=2
                    assert row['total_latency_ns']>=sum(row['phase_times_ns'].values())
                    nodes=scenario_nodes(row['method']=='tar_nominal')
                    anchor=np.array(row['backup']['q']) if row['backup']['accepted'] else np.array(row['previous_q'])
                    step=kin.limits.velocity*.02+v.config.velocity_tolerance
                    if row['planned_z'] is not None:
                        zs=np.array(row['planned_z']);assert len(zs)==len(nodes)
                        assert np.all(zs>=kin.limits.lower) and np.all(zs<=kin.limits.upper)
                        assert np.all(np.abs(zs-q)<=step)
                        errs=[];checks=[]
                        for z,w in zip(zs,nodes):
                            goal=perturb_target(prediction,w,1,scale)
                            c=v.check(z,IKQuery(goal,q,.02));checks.append(bool(c.accepted))
                            errs.append(max(c.position_error/scale[0],c.orientation_error/scale[3]))
                        assert checks==row['planned_node_verified']
                        assert all(checks)==row['all_nodes_verified']
                        tau=max(0.,max(errs)-1)
                        largest_tau_error=max(largest_tau_error,abs(tau-row['tau_actual']))
                        # Public acos orientation and log differ near machine zero;
                        # normalized discrepancy is checked rather than hidden.
                        assert abs(tau-row['tau_actual'])<2e-5
                        merit=.5*np.sum(((q-anchor)/step)**2)+.5*row['tau_actual']**2
                        largest_merit_error=max(largest_merit_error,abs(merit-row['objective_actual']))
                        assert abs(merit-row['objective_actual'])<1e-8
                        if row['method']=='tar_fixed':
                            np.testing.assert_allclose(zs,zs[0]+nodes@np.array(row['fixed_B_normalized']).T*step,atol=1e-12,rtol=0)
                        totals['planned_frames']+=1;totals['all_nodes_verified']+=all(checks)
                    if row['backup_used']:np.testing.assert_array_equal(row['q'],row['backup']['q'])
                    for trial in row['nonlinear_trials']:
                        if trial['adopted']:
                            assert trial['geometric'] and v.check(np.array(trial['q']),query).accepted
                            assert np.all(np.abs(np.array(trial['z'])-trial['q'])<=step)
                            totals['adopted_trials']+=1
                    if row['initial_geometric'] and row['objective_actual'] is not None:
                        assert row['objective_actual']<=row['initial_objective']+1e-12
                last=target
    write_json(root/f'{stage}_audit.json',dict(utc=utc(),counts=dict(totals),
        largest_tau_public_vs_native_error=largest_tau_error,largest_merit_error=largest_merit_error,
        tests='all current outputs/public verifier, feedback holds, causal prediction, final scenario bounds/rates/verifier, fixed-map equality, merit, iteration/timing counts',
        solver_calls=0))


def figures(root,cfg):
    """Read frozen aggregate data only; editable exports, no IK or statistics refit."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    data=json.loads((root/'reports/source_data.json').read_text())
    out=root/'figures';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,
        'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    bases=['tar_fixed','tar_nominal','trac_task_5ms','trac_task_20ms','pink_qp']
    names=['Fixed compensation','Nominal prediction','TRAC-IK 5 ms','TRAC-IK 20 ms','Pink']
    fig,axes=plt.subplots(2,3,figsize=(7.2047244094,4.5669291339),layout='constrained')
    source=[]
    for row,robot in enumerate(cfg['robots']):
        for col,metric in enumerate(('completion','deadline_completion','total_latency_ns')):
            ax=axes[row,col];color='#0072B2' if row==0 else '#D55E00'
            for y,base in enumerate(bases):
                pair=next(p for p in data['pairs'] if p['robot']==robot and p['baseline']==base and p['metric']==metric)
                value=pair['ratio'] if col==2 else 100*pair['difference']
                ci=np.array(pair['ratio_ci'] if col==2 else pair['difference_ci'])*(1 if col==2 else 100)
                ax.errorbar(value,y,xerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],fmt='o',
                    color=color,markersize=3.5,elinewidth=.9,capsize=2)
                source.append(dict(robot=robot,baseline=base,metric=metric,value=value,low=ci[0],high=ci[1]))
            ax.set_yticks(range(5),names if col==0 else ['']*5);ax.set_ylim(4.6,-.6)
            ax.axvline(1 if col==2 else 0,color='#777777',linewidth=.6,linestyle='--',zorder=0)
            ax.grid(axis='x',color='#dddddd',linewidth=.4);ax.set_axisbelow(True)
            ax.set_title(chr(65+row*3+col)+('  '+('Panda' if row==0 else 'UR5e') if col==0 else ''))
            if col==2:
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.set_xlabel(['TSR difference (pp)\npositive favors free recourse',
                           'DTSR20 difference (pp)\npositive favors free recourse',
                           'Cumulative time ratio\nlower favors free recourse'][col])
    fig.suptitle('Shared-command free recourse versus each control',fontsize=9)
    fig.savefig(out/'paired_task_cost.pdf')
    fig.savefig(out/'paired_task_cost.svg')
    fig.savefig(out/'paired_task_cost.png',dpi=360)
    plt.close(fig)
    csv_write(out/'paired_task_cost.csv',source)
    write_json(out/'manifest.json',dict(source_sha256=sha(root/'reports/source_data.json'),
        statistical_unit='40 trajectory UIDs per robot; 3 repeats averaged within UID',
        intervals='paired family-stratified 95% bootstrap; descriptive, unadjusted; not equivalence intervals',
        figure_size_mm=[183,116],minimum_intended_font_pt=6.5,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def numerical_prepare(root,cfg):
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',dict(baseline=NUMERIC_BASELINE,created=utc(),configuration=cfg,
        methods=['tar_old','tar_corrected','trac_task_5ms','pink_qp'],repeats=3,
        probe_fixed_updates=[2,4,6],probe_joint_updates=2,probe_repeats=3,
        total_soft_limit_ms=18,order_seed=2026091201,
        selection='Among fixed-update budgets with reconstructed outer P95 <=20 ms, maximize known-zero states solved in all three probes, then minimize fixed updates; if none meet 20 ms select lowest P95. No trajectory outcomes used.',
        scope='Observed 80 mechanism inputs; existing 40 trajectories/robot. No fresh data or new objective.',
        initial='One existing scaled least-squares nominal predictor step, then common-Jacobian node initial guesses; clipped to exact-anchor joint/rate interval. No node IK searches.',
        objective='0.5*||S^-1(q-anchor)||^2 + 0.5*tau^2; unchanged 13 nodes and contract',
        comparisons='Same-input numerical replay uses the identical saved TAR internal backup. Replay time and saved backup time are distinguished. Whole trajectories use actual new calls.',
        original_input_hash=sha(DEFAULT/'input_identities.json'),original_code=sha(ROOT/'src/confik/task_recourse.py')))
    print('Prepared numerical-completion protocol; no solver calls.')


def zero_references(root,cfg):
    folder=root/'zero_reference';folder.mkdir(exist_ok=False)
    table=[];checks=[];witnesses=[]
    for robot in cfg['robots']:
        _,kin,v,_=context(robot,cfg);scale=task_scale(v)
        for item in items(cfg,robot):
            origin=DEFAULT/f'mechanism_{robot}'
            data=json.loads((origin/f'{item["site_id"]}_inputs.json').read_text())
            original=next(x['result'] for x in data['methods'] if x['method']=='tar_free')
            anchor=np.array(original['backup']['q'])
            target=Pose(np.array(data['target_position']),np.array(data['target_rotation']))
            query=IKQuery(target,np.array(data['previous_q']),.02)
            anchor_check=v.check(anchor,query)
            predicted=predict_target(target,Pose(np.array(data['last_position']),np.array(data['last_rotation'])))
            np.testing.assert_array_equal(predicted.position,original['predicted_position'])
            np.testing.assert_array_equal(predicted.rotation,original['predicted_rotation'])
            raw=read_rows(origin/f'{item["site_id"]}.jsonl.gz');chosen=[]
            for node,w in enumerate(scenario_nodes()):
                goal=perturb_target(predicted,w,1,scale);pool=[]
                # Previously accepted witness configurations are candidates, not
                # evidence until rechecked from THIS exact internal backup.
                for index,r in enumerate(raw):
                    if r['witness'] and np.array_equal(r['target_position'],goal.position) and np.array_equal(r['target_rotation'],goal.rotation):
                        pool.append((r['reference_result']['q'],dict(file=str((origin/f'{item["site_id"]}.jsonl.gz').relative_to(ROOT)),row=index,
                            source_method=r['method'],source_previous_q=r['current_q'])))
                for m in data['methods']:
                    r=m['result'];plan=r.get('planned_z');flags=r.get('planned_node_verified',[])
                    if plan is not None and node<len(plan) and node<len(flags) and flags[node]:
                        # Nominal node zero matches, but does not supply axes.
                        pool.append((plan[node],dict(file=str((origin/f'{item["site_id"]}_inputs.json').relative_to(ROOT)),
                            source_method=m['method'],planned_node=node,source_previous_q=r['q'])))
                valid=[]
                for q,source in pool:
                    z=np.array(q);c=v.check(z,IKQuery(goal,anchor,.02))
                    exact_bounds=bool(np.all(z>=kin.limits.lower) and np.all(z<=kin.limits.upper))
                    checks.append(dict(robot=robot,uid=item['uid'],node=node,anchor=anchor,z=z,source=source,
                        same_source_previous=np.array_equal(source['source_previous_q'],anchor),accepted=c.accepted,
                        exact_joint_bounds=exact_bounds,reasons=list(c.reasons),position_error=c.position_error,orientation_error=c.orientation_error))
                    if c.accepted and exact_bounds:valid.append((z,source))
                chosen.append(dict(node=node,w=w,target_position=goal.position,target_rotation=goal.rotation,
                    available=len(valid),z=valid[0][0] if valid else None,source=valid[0][1] if valid else None))
            known=bool(anchor_check.accepted and original['backup']['accepted'] and all(x['available'] for x in chosen))
            current=np.array(original['q']) if original['q'] is not None else None
            step=kin.limits.velocity*.02+v.config.velocity_tolerance
            change=float(np.max(np.abs(current-anchor))) if current is not None else None
            row=dict(robot=robot,uid=item['uid'],site_id=item['site_id'],family=item['family'],
                anchor_accepted=anchor_check.accepted,known_zero=known,status='witness_confirmed_zero' if known else 'unknown',
                nodes_available=sum(bool(x['available']) for x in chosen),online_objective=original['objective_actual'],online_tau=original['tau_actual'],
                changed_exact=bool(current is not None and not np.array_equal(current,anchor)),max_change_rad=change,
                normalized_intervention=float(np.linalg.norm((current-anchor)/step)) if current is not None else None,
                zero_gap=original['objective_actual'] if known else None,
                gap_above_old_comparison_tolerance=bool(known and original['objective_actual'] is not None and original['objective_actual']>cfg['optimizer']['objective_atol']),
                source_input_hash=sha(origin/f'{item["site_id"]}_inputs.json'),source_witness_hash=sha(origin/f'{item["site_id"]}.jsonl.gz'))
            table.append(row)
            witnesses.append(dict(**row,anchor=anchor,previous_q=data['previous_q'],target_position=target.position,target_rotation=target.rotation,
                predicted_position=predicted.position,predicted_rotation=predicted.rotation,dt=.02,nodes=chosen,
                objective_lower_bound=0 if known else None,reason='Nonnegative objective, exact q=anchor and all 13 original-contract checks pass.' if known else 'No complete matched witness in saved data; no additional search.'))
    csv_write(folder/'zero_cost_comparison.csv',table)
    write_json(folder/'witnesses.json',witnesses)
    with gzip.open(folder/'reverification.jsonl.gz','xt') as f:
        for row in checks:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
    summary={r:dict(states=sum(x['robot']==r for x in table),known_zero=sum(x['robot']==r and x['known_zero'] for x in table),
        changed_exact=sum(x['robot']==r and x['known_zero'] and x['changed_exact'] for x in table),
        changed_over_1e8rad=sum(x['robot']==r and x['known_zero'] and (x['max_change_rad'] or 0)>1e-8 for x in table),
        gap_above_old_tolerance=sum(x['robot']==r and x['gap_above_old_comparison_tolerance'] for x in table)) for r in cfg['robots']}
    write_json(folder/'summary.json',summary)
    write_json(folder/'manifest.json',dict(solver_calls=0,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2))


def original_kernel():
    """Load the exact Git baseline in memory, not a maintained second runtime."""
    name='confik._task_recourse_baseline'
    if name not in sys.modules:
        source=subprocess.check_output(['git','show',f'{NUMERIC_BASELINE}:src/confik/task_recourse.py'],cwd=ROOT,text=True)
        module=types.ModuleType(name);module.__package__='confik';sys.modules[name]=module
        exec(compile(source,f'git:{NUMERIC_BASELINE}:src/confik/task_recourse.py','exec'),module.__dict__)
    return sys.modules[name].TaskRecourseIK


def numeric_factory(method,robot,cfg,fixed_updates=2):
    source,kin,v,urdf=context(robot,cfg)
    if method in ('tar_old','tar_corrected'):
        cls=original_kernel() if method=='tar_old' else TaskRecourseIK
        options=dict(cfg['optimizer'])
        if method=='tar_corrected':options.update(anchor_first=True,fixed_updates=fixed_updates)
        solver=cls(kin,v,source,str(ROOT/cfg['native_trac_library']),urdf,mode='free',config=options)
    elif method=='pink_qp':solver=PinkAdapter(kin,v,source,urdf)
    else:solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    return solver,kin,v


class RecordedBackup:
    """Offline numeric replay only: not used by any complete online run."""
    def __init__(self,data):self.data=data;self.calls=0
    def close(self):pass
    def solve(self,p,R,previous,dt):
        np.testing.assert_array_equal(p,self.data['target_position'])
        np.testing.assert_array_equal(R,self.data['target_rotation'])
        np.testing.assert_array_equal(previous,self.data['previous_q'])
        assert dt==.02;self.calls+=1
        return copy.deepcopy(next(x['result']['backup'] for x in self.data['methods'] if x['method']=='tar_free'))


def numerical_probe(root,cfg,tag='initial'):
    folder=root/('same_input' if tag=='initial' else f'same_input_{tag}');folder.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());before=hashes();summaries=[]
    known={(r['robot'],r['uid']):r['known_zero'] for r in json.loads((root/'zero_reference/witnesses.json').read_text())}
    variants=[('tar_old',0)]+[('tar_corrected',n) for n in protocol['probe_fixed_updates']]
    rng=np.random.default_rng(protocol['order_seed'])
    for robot in cfg['robots']:
        solvers={key:numeric_factory(key[0],robot,cfg,key[1] or 2) for key in variants}
        for solver,_,_ in solvers.values():solver.backup.close()
        try:
            for item in items(cfg,robot):
                data=json.loads((DEFAULT/f'mechanism_{robot}'/f'{item["site_id"]}_inputs.json').read_text())
                jobs=[(variant,rep) for variant in variants for rep in range(protocol['probe_repeats'])]
                rows=[]
                for index in rng.permutation(len(jobs)):
                    (method,updates),rep=jobs[index];solver,kin,v=solvers[(method,updates)]
                    solver.reset(np.array(data['previous_q']))
                    solver.last_target=Pose(np.array(data['last_position']),np.array(data['last_rotation']))
                    replay=RecordedBackup(data);solver.backup=replay
                    result=solver.solve(data['target_position'],data['target_rotation'],np.array(data['previous_q']),.02)
                    assert replay.calls==1
                    numeric_ns=result['total_latency_ns']-result['phase_times_ns']['backup_ns']
                    anchor=np.array(result['backup']['q'])
                    q=np.array(result['q']) if result['q'] is not None else None
                    zero=bool(q is not None and np.array_equal(q,anchor) and result['all_nodes_verified'] and result['accepted'])
                    row=dict(robot=robot,uid=item['uid'],site_id=item['site_id'],family=item['family'],method=method,
                        fixed_updates=updates,repeat=rep,known_zero=known[(robot,item['uid'])],zero_found=zero,
                        objective=result['objective_actual'],tau=result['tau_actual'],accepted=result['accepted'],
                        changed_exact=bool(q is not None and not np.array_equal(q,anchor)),
                        max_change_rad=float(np.max(np.abs(q-anchor))) if q is not None else None,
                        numerical_ns=numeric_ns,saved_backup_ns=result['backup']['total_latency_ns'],
                        reconstructed_outer_ns=numeric_ns+result['backup']['total_latency_ns'],
                        fixed_actual_updates=sum(r.get('phase')=='fixed' for r in result['native_status']),
                        joint_actual_updates=sum(r.get('phase','joint')=='joint' for r in result['native_status']),
                        zero_found_stage=result.get('zero_found_stage'),numerical_phase_times_ns=result.get('numerical_phase_times_ns'))
                    summaries.append(row);rows.append(dict(summary=row,result=result))
                with gzip.open(folder/f'{robot}_{item["site_id"]}.jsonl.gz','xt') as f:
                    for r in rows:f.write(json.dumps(clean(r),allow_nan=False,separators=(',',':'))+'\n')
                print(f'same-input {robot} {item["site_id"]}',flush=True)
        finally:
            for s,_,_ in solvers.values():s.close()
    assert hashes()==before
    write_json(folder/'summaries.json',summaries);csv_write(folder/'comparison.csv',summaries)
    write_json(folder/'completed.json',dict(code_hashes=before,utc=utc(),scope='identical frozen backup, numerical replay; reconstructed outer time is not a measured online TRAC call',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))


def numerical_select(root,cfg,tag='initial'):
    protocol=json.loads((root/'protocol.json').read_text())
    folder=root/('same_input' if tag=='initial' else f'same_input_{tag}')
    rows=json.loads((folder/'summaries.json').read_text());table=[]
    for budget in protocol['probe_fixed_updates']:
        sub=[r for r in rows if r['method']=='tar_corrected' and r['fixed_updates']==budget]
        states=sorted({(r['robot'],r['uid']) for r in sub})
        success=sum(all(r['zero_found'] for r in sub if (r['robot'],r['uid'])==key) for key in states
                    if next(r['known_zero'] for r in sub if (r['robot'],r['uid'])==key))
        p95=float(np.percentile([r['reconstructed_outer_ns'] for r in sub],95)/1e6)
        table.append(dict(fixed_updates=budget,joint_updates=2,known_zero_all_repeats=success,
            reconstructed_p95_ms=p95,eligible_timing=p95<=20))
    eligible=[r for r in table if r['eligible_timing']]
    chosen=min(eligible,key=lambda r:(-r['known_zero_all_repeats'],r['fixed_updates'])) if eligible else min(table,key=lambda r:r['reconstructed_p95_ms'])
    write_json(root/('selection.json' if tag=='initial' else f'selection_{tag}.json'),dict(created=utc(),candidates=table,selected=chosen,
        reason=protocol['selection'],code_hashes=hashes(),same_input_hash=sha(folder/'completed.json'),
        explicit_boundary='development-only numerical budget; no full-trajectory corrected outcomes opened yet'))
    print(json.dumps(chosen,indent=2))


def numerical_finalize(root):
    original=root/'selection_roundoff_corrected.json'
    selection=json.loads(original.read_text())
    folders={}
    for robot in ('panda','ur5e'):
        folder=root/f'development_{robot}'
        runs=list((folder/'runs').glob('*.summary.json'))
        folders[robot]=dict(completed_runs=len(runs),complete=(folder/'completed.json').exists(),
            files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
    jobs=json.loads((root/'development_panda/job_order.json').read_text())
    write_json(root/'interface_guard_amendment.json',dict(utc=utc(),previous_selection_hash=sha(original),
        original_runs=folders,panda_interrupted_job=jobs[folders['panda']['completed_runs']],
        cause='representable_interior called on a substantially infeasible current trial from a native unsuccessful solve; its valid-current-state precondition was not met',
        correction='Skip future roundoff reconstruction for current trials outside their allowed interval; retain true-FK/current-verifier rejection. No constraint, objective, predictor, or budget change.',
        scope='All preliminary records retained. Restart BOTH robot comparisons with the same guarded source; no outcome-driven data selection.',
        unchanged_budget=selection['selected']))
    write_json(root/'selection_final.json',dict(created=utc(),selected=selection['selected'],
        code_hashes=hashes(),previous_selection_hash=sha(original),
        reason='Inherited exactly from same-input selection; interface precondition repair only, not a new budget selection.',
        explicit_boundary='Development numerical debugging; earlier partial Panda and complete UR5e runs retained, not called unseen.'))


def numerical_run(root,cfg,robot):
    selection=json.loads((root/'selection_final.json').read_text())
    assert sha(ROOT/'src/confik/task_recourse.py')==selection['code_hashes']['src/confik/task_recourse.py']
    budget=selection['selected']['fixed_updates'];protocol=json.loads((root/'protocol.json').read_text())
    folder=root/f'validated_{robot}';folder.mkdir(exist_ok=False);(folder/'runs').mkdir()
    identity=json.loads((DEFAULT/'input_identities.json').read_text())
    for entry in identity['input_files'].values():assert sha(ROOT/entry['path'])==entry['sha256']
    before=hashes();data=items(cfg,robot)
    jobs=[(i,m,r) for i in data for m in protocol['methods'] for r in range(protocol['repeats'])]
    order=np.random.default_rng(protocol['order_seed']).permutation(len(jobs))
    write_json(folder/'started.json',dict(utc=utc(),code_hashes=before,selected_budget=selection['selected'],
        selection_hash=sha(root/'selection_final.json'),original_kernel_git=NUMERIC_BASELINE,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        affinity=sorted(os.sched_getaffinity(0)),python=platform.python_version(),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    write_json(folder/'job_order.json',[dict(uid=jobs[j][0]['uid'],method=jobs[j][1],repeat=jobs[j][2]) for j in order])
    solvers={};summaries=[];begin=time.monotonic()
    try:
        for count,j in enumerate(order):
            item,method,repeat=jobs[j]
            if method not in solvers:
                solvers[method]=numeric_factory(method,robot,cfg,budget)
                s,k,v=solvers[method];q=np.array(item['initial_q']);target=k.forward(q)
                s.solve(target.position,target.rotation,q,.02)
            solver,kin,v=solvers[method]
            assert not isinstance(getattr(solver,'backup',None),RecordedBackup)
            rows,summary=execute_trajectory(solver,kin,v,item,method,repeat)
            rid=f'{robot}_{item["site_id"]}_{method}_r{repeat}'
            path=folder/'runs'/f'{rid}.jsonl.gz'
            with gzip.open(path,'xt') as f:
                for row in rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
            summary.update(robot=robot,method=method,repeat=repeat,uid=item['uid'],site_id=item['site_id'],family=item['family'],
                run_id=rid,raw_file=str(path.relative_to(folder)),
                zero_cost_verified_rate=float(np.mean([r.get('zero_cost_verified',False) for r in rows])),
                optimization_rate=float(np.mean([r.get('optimization_called',False) for r in rows])),
                command_change_rate=float(np.mean([r.get('command_changed',False) for r in rows])),
                mean_intervention=float(np.mean([r.get('intervention') or 0 for r in rows])))
            summaries.append(summary);write_json(folder/'runs'/f'{rid}.summary.json',summary)
            print(f'numerical development {robot} {count+1}/{len(jobs)} {method} {item["site_id"]} r{repeat} '
                f'complete={summary["completion"]} P95={summary["p95_ms"]:.3f} elapsed={(time.monotonic()-begin)/60:.1f}min',flush=True)
    finally:
        for solver,_,_ in solvers.values():solver.close()
    assert hashes()==before
    write_json(folder/'summaries.json',summaries)
    write_json(folder/'completed.json',dict(utc=utc(),runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))


def numerical_report(root,cfg):
    """Read-only nonlinear verification and UID-level reporting; no solver calls."""
    out=root/'reports';out.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text())
    summaries=[];arrays={};extra=defaultdict(list);failures=[];audit=Counter();sources={}
    online_witnesses=[];max_tau_error=0.;max_objective_error=0.
    for robot in cfg['robots']:
        folder=root/f'validated_{robot}'
        seal=json.loads((folder/'completed.json').read_text())
        for name,digest in seal['files'].items():assert sha(folder/name)==digest,(folder,name)
        sources[str(folder.relative_to(ROOT))]=sha(folder/'completed.json')
        jobs=json.loads((folder/'summaries.json').read_text());assert len(jobs)==480
        _,kin,v,urdf=context(robot,cfg);geometry=NativeGeometry(kin,urdf)
        scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
        data={i['uid']:i for i in items(cfg,robot)};witness_uids=set()
        for count,s in enumerate(jobs):
            item=data[s['uid']];rows=read_rows(folder/s['raw_file']);assert len(rows)==150
            summaries.append(s)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            previous=np.array(item['initial_q']);last=None
            assert sum(r['total_latency_ns'] for r in rows)==s['total_latency_ns']
            assert all(r['accepted'] for r in rows)==s['completion']
            assert all(r['accepted_within_20ms'] for r in rows)==s['deadline_completion']
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==.02 and r['uid']==item['uid']
                np.testing.assert_array_equal(r['previous_q'],previous)
                np.testing.assert_array_equal(r['target_position'],item['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],item['target_rotation'][frame])
                target=Pose(np.array(r['target_position']),np.array(r['target_rotation']))
                query=IKQuery(target,previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                check=v.check(q,query);assert check.accepted==r['accepted']
                assert r['accepted_within_20ms']==bool(check.accepted and r['total_latency_ns']<=20_000_000)
                audit['current_commands_reverified']+=1
                audit['accepted_commands']+=check.accepted
                audit['over_20ms_frames']+=r['total_latency_ns']>20_000_000
                if s['first_failure_frame']==frame:
                    failures.append({k:r.get(k) for k in ('robot','uid','site_id','method','repeat','family','frame','previous_q',
                        'target_position','target_rotation','q','position_error','orientation_error','failure_kind',
                        'total_latency_ns','backup','native_status')})
                if s['method'].startswith('tar_'):
                    predicted=predict_target(target,last)
                    np.testing.assert_array_equal(r['predicted_position'],predicted.position)
                    np.testing.assert_array_equal(r['predicted_rotation'],predicted.rotation)
                    targets=[perturb_target(predicted,w,1,scale) for w in scenario_nodes()]
                    backup=r['backup'];anchor=np.array(backup['q'] if backup['accepted'] else previous)
                    if backup['accepted']:assert v.check(anchor,query).accepted
                    calls=r['native_status'];assert len(calls)<= (4 if s['method']=='tar_corrected' else 2)
                    if s['method']=='tar_corrected':
                        assert sum(c['phase']=='fixed' for c in calls)<=2
                        assert sum(c['phase']=='joint' for c in calls)<=2
                    assert sum(r['phase_times_ns'].values())+r['accounting_remainder_ns']==r['total_latency_ns']
                    assert r['accounting_remainder_ns']>=0
                    if r['planned_z'] is not None:
                        zs=np.array(r['planned_z']);assert zs.shape==(13,kin.nq)
                        assert np.all(zs>=kin.limits.lower) and np.all(zs<=kin.limits.upper)
                        assert np.all(np.abs(zs-q)<=step)
                        flags=[v.check(z,IKQuery(goal,q,.02)).accepted for z,goal in zip(zs,targets)]
                        assert flags==r['planned_node_verified'] and all(flags)==r['all_nodes_verified']
                        audit['planned_nodes_reverified']+=13
                        residual=np.array([pose_error(goal,geometry.forward(z))/scale for z,goal in zip(zs,targets)])
                        tau=max(0.,float(np.max([np.linalg.norm(residual[:,:3],axis=1),np.linalg.norm(residual[:,3:],axis=1)]))-1)
                        objective=.5*np.sum(((q-anchor)/step)**2)+.5*tau*tau
                        max_tau_error=max(max_tau_error,abs(tau-r['tau_actual']))
                        max_objective_error=max(max_objective_error,abs(objective-r['objective_actual']))
                        np.testing.assert_allclose(tau,r['tau_actual'],rtol=1e-9,atol=1e-8)
                        np.testing.assert_allclose(objective,r['objective_actual'],rtol=1e-9,atol=1e-8)
                    else:assert not r['all_nodes_verified']
                    if r.get('zero_cost_verified'):
                        assert check.accepted and backup['accepted'] and r['all_nodes_verified']
                        np.testing.assert_array_equal(q,anchor)
                        assert r['objective_actual']==r['tau_actual']==0
                        audit['exact_zero_returns_reverified']+=1
                        if s['uid'] not in witness_uids:
                            online_witnesses.append(dict(robot=robot,uid=s['uid'],repeat=s['repeat'],frame=frame,
                                raw_file=str((folder/s['raw_file']).relative_to(ROOT)),previous_q=previous,anchor=anchor,
                                target_position=target.position,target_rotation=target.rotation,dt=.02,objective=0,
                                nodes=[dict(w=w,z=z,target_position=g.position,target_rotation=g.rotation)
                                    for w,z,g in zip(scenario_nodes(),r['planned_z'],targets)]))
                            witness_uids.add(s['uid'])
                    incumbent=r['initial_objective'] if r['initial_geometric'] else None
                    for trial in r['nonlinear_trials']:
                        if trial.get('phase')=='fixed':np.testing.assert_array_equal(trial['q'],anchor)
                        if not trial['adopted']:continue
                        assert trial['geometric'] and trial['current_accepted'] and trial['future_bounds_ok'] and trial['future_rate_ok']
                        if incumbent is not None:assert trial['objective_actual']<incumbent
                        incumbent=trial['objective_actual'];audit['adopted_true_merit_decreases']+=1
                    if incumbent is not None:
                        np.testing.assert_allclose(r['objective_actual'],incumbent,rtol=1e-9,atol=1e-8)
                    extra[(robot,s['method'])].append({k:r.get(k) for k in
                        ('numerical_phase_times_ns','phase_times_ns','native_status','zero_cost_verified','total_latency_ns')})
                last=target
                if check.accepted:previous=q.copy()
                np.testing.assert_array_equal(r['accepted_state_q'],previous)
            if count%80==0:print(f'reverify {robot} {count+1}/480',flush=True)
    assert len(summaries)==960 and audit['current_commands_reverified']==144000
    main=group_table(summaries,arrays);families=group_table(summaries,arrays,True)
    for row in main+families:
        selected=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and
                  (row['family']=='all' or s['family']==row['family'])]
        for key in ('zero_cost_verified_rate','optimization_rate','command_change_rate','mean_intervention'):
            row[key]=float(np.mean([s[key] for s in selected]))
    units=[];pairs=[];changes=[];completion={}
    metrics=('completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms','successful_prefix')
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==uid) for uid in uids];assert len(uids)==40
        for method in protocol['methods']:
            completion.setdefault(robot,{})[method]={str(rep):sorted(s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']) for rep in range(3)}
            for uid,family in zip(uids,fam):
                rr=[s for s in ss if s['method']==method and s['uid']==uid];assert sorted(s['repeat'] for s in rr)==[0,1,2]
                val={k:float(np.mean([s[k] for s in rr])) for k in metrics}
                unit[(method,uid)]=val;units.append(dict(robot=robot,method=method,uid=uid,site_id=rr[0]['site_id'],family=family,repeats=3,**val))
        for base in [m for m in protocol['methods'] if m!='tar_corrected']:
            for metric in metrics:
                a=np.array([unit[('tar_corrected',u)][metric] for u in uids]);b=np.array([unit[(base,u)][metric] for u in uids])
                pairs.append(dict(robot=robot,method='tar_corrected',baseline=base,metric=metric,
                    **paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,family in zip(uids,fam):
                a=unit[('tar_corrected',uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,baseline=base,uid=uid,site_id=next(s['site_id'] for s in ss if s['uid']==uid),family=family,
                    corrected_fraction=a,baseline_fraction=b,change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
    phases=[]
    for (robot,method),rows in sorted(extra.items()):
        def components(r):
            if method=='tar_corrected':return r['numerical_phase_times_ns']
            p=r['phase_times_ns']
            return dict(initialization_ns=None,fixed_current_ns=0,
                joint_update_ns=sum(p[k] for k in ('linearization_ns','cone_build_ns','cone_solve_ns')),
                fk_check_ns=sum(p[k] for k in ('initial_check_ns','nonlinear_check_ns','final_verification_ns')))
        for category in ('all','exact_zero_return','not_exact_zero_return'):
            rr=[r for r in rows if category=='all' or bool(r.get('zero_cost_verified'))==(category=='exact_zero_return')]
            if not rr:continue
            p=[components(r) for r in rr];record=dict(robot=robot,method=method,subset=category,frames=len(rr))
            for key in ('initialization_ns','fixed_current_ns','joint_update_ns','fk_check_ns'):
                vals=[r[key] for r in p if r[key] is not None]
                record[key.removesuffix('_ns')+'_mean_ms']=float(np.mean(vals)/1e6) if vals else None
                record[key.removesuffix('_ns')+'_median_ms']=float(np.median(vals)/1e6) if vals else None
            record.update(backup_mean_ms=float(np.mean([r['phase_times_ns']['backup_ns'] for r in rr])/1e6),
                outer_mean_ms=float(np.mean([r['total_latency_ns'] for r in rr])/1e6),
                outer_median_ms=float(np.median([r['total_latency_ns'] for r in rr])/1e6),
                fixed_updates_mean=float(np.mean([sum(c.get('phase')=='fixed' for c in r['native_status']) for r in rr])),
                joint_updates_mean=float(np.mean([sum(c.get('phase','joint')=='joint' for c in r['native_status']) for r in rr])))
            phases.append(record)
    probe_folder=root/'same_input_roundoff_corrected'
    probe=json.loads((probe_folder/'summaries.json').read_text())
    probe=[r for r in probe if r['fixed_updates'] in (0,2)]
    same_units=[];same_table=[]
    for robot in cfg['robots']:
        for method in ('tar_old','tar_corrected'):
            rr=[r for r in probe if r['robot']==robot and r['method']==method]
            uu=[]
            for uid in sorted({r['uid'] for r in rr}):
                state=[r for r in rr if r['uid']==uid];assert len(state)==3
                row=dict(robot=robot,uid=uid,site_id=state[0]['site_id'],family=state[0]['family'],method=method,
                    known_zero=state[0]['known_zero'],zero_all_repeats=all(r['zero_found'] for r in state),
                    exact_changed_any=any(r['changed_exact'] for r in state),
                    meaningful_changed_any=any((r['max_change_rad'] or 0)>1e-8 for r in state),
                    objective_max=max((r['objective'] or 0) for r in state) if any(r['objective'] is not None for r in state) else None,
                    tau_max=max((r['tau'] or 0) for r in state) if any(r['tau'] is not None for r in state) else None,
                    max_change_rad=max((r['max_change_rad'] or 0) for r in state),
                    numerical_mean_ms=float(np.mean([r['numerical_ns'] for r in state])/1e6),
                    reconstructed_outer_mean_ms=float(np.mean([r['reconstructed_outer_ns'] for r in state])/1e6))
                same_units.append(row);uu.append(row)
            known=[r for r in uu if r['known_zero']]
            table=dict(robot=robot,method=method,states=40,known_zero=len(known),unknown=40-len(known),
                known_zero_not_found_all_repeats=sum(not r['zero_all_repeats'] for r in known),
                known_zero_but_changed_exact=sum(r['exact_changed_any'] for r in known),
                known_zero_but_changed_above_1e_minus_8=sum(r['meaningful_changed_any'] for r in known),
                known_zero_gap_above_1e_minus_10=sum((r['objective_max'] or 0)>1e-10 for r in known),
                known_zero_max_objective_gap=max(r['objective_max'] for r in known),
                known_zero_max_change_rad=max(r['max_change_rad'] for r in known))
            for key in ('numerical_ns','reconstructed_outer_ns'):
                for p in (50,95,99):table[f'{key.removesuffix("_ns")}_p{p}_ms']=float(np.percentile([r[key] for r in rr],p)/1e6)
            same_table.append(table)
    csv_write(out/'main_table.csv',main);csv_write(out/'family_table.csv',families)
    csv_write(out/'trajectory_units.csv',units);csv_write(out/'paired_comparisons.csv',pairs)
    csv_write(out/'gained_lost_uids.csv',changes);csv_write(out/'run_summaries.csv',summaries)
    csv_write(out/'numerical_phase_times.csv',phases);csv_write(out/'first_failure_inputs.csv',failures)
    csv_write(out/'same_input_units.csv',same_units);csv_write(out/'same_input_table.csv',same_table)
    write_json(out/'completion_uids.json',completion)
    write_json(out/'online_zero_witnesses.json',online_witnesses)
    write_json(out/'verification_audit.json',dict(counts=dict(audit),accepted_contract_violations=0,
        max_recomputed_tau_difference=max_tau_error,max_recomputed_objective_difference=max_objective_error,
        checking='All current commands, all saved final planned nodes, accepted-only feedback, causal prediction, complete cost, clamped-current trials and monotone accepted true merit. Numerical assertion tolerances never alter contract checks.'))
    write_json(out/'source_data.json',dict(main=main,family=families,pairs=pairs,changes=changes,units=units,phases=phases,same_input=same_table))
    write_json(out/'manifest.json',dict(utc=utc(),sources=sources,source_code_sha256=sha(__file__),
        selection_hash=sha(root/'selection_final.json'),same_input_source_sha256=sha(probe_folder/'completed.json'),
        independent_unit='40 trajectory UIDs/robot, 3 runs averaged per UID; paired family-stratified bootstrap, 4000 resamples, unadjusted descriptive 95% intervals; no equivalence claim',
        timing='Full outer solver work and acceptance, including failures and timeouts. Serialization and offline re-verification excluded. Old initialization has no separately instrumented component, recorded null.',
        exclusions=0,complete_runs=960,frames=144000,
        preliminary_scope='First partial Panda and complete UR5e attempts preserved separately; not pooled into the fixed guarded comparison.',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],3),round(r['cumulative_ms_per_sweep']/1000,3))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','test','integration','development','mechanism','report','audit','figures','numeric_prepare','zero_reference','numeric_probe','numeric_select','numeric_finalize','numeric_run','numeric_report'])
    p.add_argument('--config',default='configs/task_recourse.yaml');p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['integration','development'],default='development')
    p.add_argument('--probe-tag',choices=['initial','roundoff_corrected'],default='initial')
    p.add_argument('--robot',choices=['panda','ur5e']);args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    if args.action=='prepare':prepare(args.out,cfg)
    elif args.action=='numeric_prepare':numerical_prepare(NUMERIC_ROOT,cfg)
    elif args.action=='zero_reference':zero_references(NUMERIC_ROOT,cfg)
    elif args.action=='numeric_probe':numerical_probe(NUMERIC_ROOT,cfg,args.probe_tag)
    elif args.action=='numeric_select':numerical_select(NUMERIC_ROOT,cfg,args.probe_tag)
    elif args.action=='numeric_finalize':numerical_finalize(NUMERIC_ROOT)
    elif args.action=='numeric_report':numerical_report(NUMERIC_ROOT,cfg)
    elif args.action=='numeric_run':
        if not args.robot:p.error('--robot required')
        numerical_run(NUMERIC_ROOT,cfg,args.robot)
    elif args.action=='test':math_tests(args.out)
    elif args.action in ('integration','development'):
        if not args.robot:p.error('--robot required')
        run(args.out,cfg,args.robot,args.action=='integration')
    elif args.action=='mechanism':
        if not args.robot:p.error('--robot required')
        mechanism(args.out,cfg,args.robot)
    elif args.action=='audit':audit(args.out,cfg,args.stage)
    elif args.action=='figures':figures(args.out,cfg)
    else:report(args.out,cfg)


if __name__=='__main__':main()
