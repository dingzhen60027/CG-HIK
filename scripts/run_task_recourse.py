#!/usr/bin/env python3
"""Single entry point for TAR-IK mathematics, online runs and source-backed report."""
import argparse
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
import zipfile

import numpy as np
import yaml

from confik.task_recourse import TaskRecourseIK, scenario_nodes
from confik.correction_reserve.study import ROOT, clean, write_json, sha, utc, context, execute_trajectory, read_rows
from confik.correction_reserve.reporting import csv_write, group_table, paired_intervals, LABELS
from confik.correction_reserve.geometry import predict_target, perturb_target, task_scale
from confik.correction_reserve.pink_adapter import PinkAdapter
from confik.task_contract_alignment.outcomes import ContractSolver
from confik.types import Pose, IKQuery
from confik.geometry import pose_error

LABELS.update(tar_free='TAR-IK free recourse',tar_fixed='Same-core fixed compensation',
              tar_nominal='Same-core nominal prediction')
DEFAULT=ROOT/'outputs/task_recourse'


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
    out=root/'reports';out.mkdir(exist_ok=False)
    summaries=[];arrays={};extra=defaultdict(list);failures=[]
    for robot in cfg['robots']:
        folder=root/f'development_{robot}'
        assert (folder/'completed.json').exists()
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
                    extra[(robot,s['method'])].append(dict(tau=r['tau_actual'],affine_tau=r['native_status'][-1]['affine_tau'] if r['native_status'] else None,
                        next_inside=r.get('actual_next_in_l1'),next_success=r.get('next_frame_accepted'),
                        all_verified=r['all_nodes_verified'],phases=r['phase_times_ns'],decision=r['decision']))
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
            mean_affine_tau=float(np.mean([r['affine_tau'] for r in rows if r['affine_tau'] is not None])),
            next_success_when_all_nodes_verified=float(np.mean([r['next_success'] for r in next_rows if r['all_verified']])) if any(r['all_verified'] for r in next_rows) else None))
    mechanisms=[];mechanism_units=[]
    for robot in cfg['robots']:
        folder=root/f'mechanism_{robot}'
        if not (folder/'completed.json').exists():raise RuntimeError('mechanism incomplete')
        rows=json.loads((folder/'summaries.json').read_text());mechanism_units.extend(rows)
        for method in cfg['methods']:
            for group in sorted({r['group'] for r in rows}):
                sub=[r for r in rows if r['method']==method and r['group']==group]
                mechanisms.append(dict(robot=robot,method=method,group=group,states=len(sub),
                    current_available=sum(r['current_accepted'] for r in sub),
                    correction_success=float(np.mean([r['correction_success'] for r in sub]))))
    for name,rows in [('main_table',main),('family_table',families),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('first_failure_inputs',failures),('run_summaries',summaries),
                      ('same_input_correction',mechanisms),('same_input_units',mechanism_units)]:csv_write(out/f'{name}.csv',rows)
    write_json(out/'completion_uids.json',completion)
    write_json(out/'source_data.json',dict(main=main,family=families,pairs=pairs,changes=changes,diagnostics=diagnostics,mechanism=mechanisms))
    write_json(out/'manifest.json',dict(utc=utc(),code_hashes=hashes(),units=80,runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        inference='whole UID, average three repeats, paired family-stratified bootstrap, unadjusted descriptive 95% intervals',
        sources={str(p.relative_to(root)):sha(p) for p in root.glob('*/completed.json')},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    for row in main:print(row['robot'],row['method'],row['completion_by_repeat'],row['deadline_completion_by_repeat'],
        [round(row[x],3) for x in ('p50_ms','p95_ms','p99_ms')],round(row['cumulative_ms_per_sweep']/1000,3))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','test','integration','development','mechanism','report'])
    p.add_argument('--config',default='configs/task_recourse.yaml');p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--robot',choices=['panda','ur5e']);args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    if args.action=='prepare':prepare(args.out,cfg)
    elif args.action=='test':math_tests(args.out)
    elif args.action in ('integration','development'):
        if not args.robot:p.error('--robot required')
        run(args.out,cfg,args.robot,args.action=='integration')
    elif args.action=='mechanism':
        if not args.robot:p.error('--robot required')
        mechanism(args.out,cfg,args.robot)
    else:report(args.out,cfg)


if __name__=='__main__':main()
