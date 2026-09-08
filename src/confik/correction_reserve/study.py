"""Serial, feedback-correct trajectory experiments for the CR-IK prototype.

The runner receives targets, not reference configurations. Each job is a whole
trajectory. Result directories and completed jobs are never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import yaml

from ..config import load_config, load_robot, resolve_path
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from .runtime import CorrectionReserveIK


ROOT = Path(__file__).resolve().parents[3]
COUPLED = {"cr_ik":"reserve", "two_step_predictive":"predictive",
           "single_step_reserve":"single", "two_step_sigma":"sigma"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def code_hashes():
    code=ROOT/'src/confik/correction_reserve'
    paths=[code/name for name in ('__init__.py','geometry.py','native_geometry.py','convex.py',
           'runtime.py','study.py','pink_adapter.py','ranged_adapter.py')]
    paths+=list((code/'native').rglob('*'))
    paths += [ROOT/'configs/correction_reserve.yaml',ROOT/'docs/CRIK_ALGORITHM_PROTOCOL.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def context(robot, cfg):
    source=load_config(str(ROOT/cfg['source_config']))
    kin=load_robot(source,robot)
    verifier=SolutionVerifier(kin,VerifierConfig(**source['verifier']))
    urdf=resolve_path(source,source['robots'][robot]['urdf'])
    return source,kin,verifier,urdf


def factory(method,robot,cfg):
    source,kin,v,urdf=context(robot,cfg)
    if method in COUPLED:
        solver=CorrectionReserveIK(kin,v,source,str(ROOT/cfg['native_trac_library']),
            urdf,mode=COUPLED[method],config=cfg['optimizer'])
    elif method.startswith('trac'):
        from ..task_contract_alignment.outcomes import ContractSolver
        solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    elif method=='pink_qp':
        from .pink_adapter import PinkAdapter
        solver=PinkAdapter(kin,v,source,urdf)
    elif method in ('ranged_ik','ranged_ik_upstream','ranged_ik_positive_range'):
        from .ranged_adapter import RangedAdapter
        solver=RangedAdapter(kin,v,source,urdf,robot=robot,
            range_mode='upstream' if method=='ranged_ik_upstream' else 'positive_range',
            time_limit_ms=18.)
    else:
        raise ValueError(method)
    return solver,kin,v


def repeat_count(method,cfg):
    return cfg['trac_repeats'] if method in COUPLED or method.startswith('trac') else cfg['deterministic_repeats']


def summarize(rows,kin,verifier):
    accepted=np.array([r['accepted'] for r in rows],bool)
    timely=np.array([r['accepted_within_20ms'] for r in rows],bool)
    latency=np.array([r['total_latency_ns'] for r in rows],float)
    errors=np.array([[r['position_error'] or 0,r['orientation_error'] or 0] for r in rows])[accepted]
    previous=np.array([r['previous_q'] for r in rows]);state=np.array([r['accepted_state_q'] for r in rows])
    velocities=(state-previous)/rows[0]['dt']
    acceleration=np.diff(velocities,axis=0)/rows[0]['dt']
    failure=np.flatnonzero(~accepted)
    def quant(x,p):return float(np.percentile(x,p)) if len(x) else None
    result=dict(frames=len(rows),completion=bool(accepted.all()),deadline_completion=bool(timely.all()),
        frame_success=float(accepted.mean()),frame_deadline_success=float(timely.mean()),
        accepted_frames=int(accepted.sum()),deadline_frames=int(timely.sum()),
        total_latency_ns=int(latency.sum()),p50_ms=quant(latency,50)/1e6,
        p95_ms=quant(latency,95)/1e6,p99_ms=quant(latency,99)/1e6,
        first_failure_frame=int(failure[0]) if len(failure) else None,
        successful_prefix=int(failure[0]) if len(failure) else len(rows),
        first_failure_reason=rows[failure[0]]['failure_kind'] if len(failure) else None,
        failure_reasons=dict(Counter(r['failure_kind'] for r in rows if not r['accepted'])),
        backup_rate=float(np.mean([r.get('backup_used',False) for r in rows])),
        optimized_command_rate=float(np.mean([r.get('nominal_next_verified',False) for r in rows])),
        changed_from_backup_rate=float(np.mean([(r.get('max_difference_from_backup') or 0)>1e-8 for r in rows])),
        p95_position_m=quant(errors[:,0],95),max_position_m=quant(errors[:,0],100),
        p95_orientation_rad=quant(errors[:,1],95),max_orientation_rad=quant(errors[:,1],100),
        normalized_tolerance_p95=quant(np.maximum(errors[:,0]/verifier.config.position_tolerance,
            errors[:,1]/verifier.config.orientation_tolerance),95),
        max_accepted_step_utilization=float(np.max(np.abs(state-previous)/
            (kin.limits.velocity*rows[0]['dt']+verifier.config.velocity_tolerance))),
        acceleration_rms=float(np.sqrt(np.mean(acceleration**2))) if len(acceleration) else 0.,
        velocity_change_rms=float(np.sqrt(np.mean(np.diff(velocities,axis=0)**2))) if len(velocities)>1 else 0.,
        accepted_contract_violations=sum(r['accepted'] and (not r['finite'] or
            not r['joint_limit_ok'] or not r['velocity_ok'] or
            r['position_error']>verifier.config.position_tolerance or
            r['orientation_error']>verifier.config.orientation_tolerance) for r in rows))
    return result


def execute_trajectory(solver,kin,verifier,item,method,repeat):
    previous=np.asarray(item['initial_q'],float).copy()
    if hasattr(solver,'reset'):solver.reset(previous)
    rows=[]
    for t,(p,r) in enumerate(zip(item['target_position'],item['target_rotation'])):
        result=solver.solve(p,r,previous,item['dt'])
        result['native_method_label']=result.get('method')
        result['method']=method
        result.setdefault('failure_kind','accepted' if result['accepted'] else
                          '+'.join(result.get('verification_reasons',[])))
        result['accepted_within_20ms']=bool(result['accepted'] and result['total_latency_ns']<=20_000_000)
        row=dict(result)
        # Some native adapters also record the input. Check it, then assign the
        # authoritative runner input once instead of duplicate keyword arguments.
        if 'previous_q' in result and not np.array_equal(result['previous_q'],previous):
            raise AssertionError('adapter input does not match actual feedback')
        row.update(robot=item['robot'],uid=item['uid'],site_id=item['site_id'],family=item['family'],
                 repeat=repeat,frame=t,dt=item['dt'],previous_q=previous.tolist(),
                 target_position=p,target_rotation=r)
        # Only admissible output changes feedback, including after earlier failures.
        if result['accepted']:previous=np.asarray(result['q'],float).copy()
        row['accepted_state_q']=previous.tolist();rows.append(row)
    return rows,summarize(rows,kin,verifier)


def read_rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [json.loads(line) for line in f]


def run(folder,items,methods,cfg,*,repeats=None,resume=False):
    folder=Path(folder)
    if not resume:folder.mkdir(parents=True,exist_ok=False)
    hashes=code_hashes()
    if resume:
        before=json.loads((folder/'started.json').read_text())
        if before['code_hashes']!=hashes:raise RuntimeError('Cannot resume with changed experiment code')
    else:
        write_json(folder/'started.json',dict(utc=utc(),code_hashes=hashes,configuration=cfg,
            methods=methods,python=platform.python_version(),platform=platform.platform(),
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            inference_unit='whole trajectory; search repeats nested',
            timing_scope='solver input conversion, dynamic bounds, numerical solve, prediction/optimization if applicable, command verification; serialization excluded'))
        write_json(folder/'online_inputs.json',items)
    jobs=[(i,m,r) for i in items for m in methods for r in range(repeats if repeats is not None else repeat_count(m,cfg))]
    rng=np.random.default_rng(cfg['formal']['order_seed'])
    order=rng.permutation(len(jobs)).tolist()
    if not resume:write_json(folder/'job_order.json',[dict(uid=jobs[j][0]['uid'],method=jobs[j][1],repeat=jobs[j][2]) for j in order])
    solvers={};summaries=[];begin=time.monotonic()
    (folder/'runs').mkdir(exist_ok=True)
    try:
        for jobnum,j in enumerate(order):
            item,method,repeat=jobs[j];key=(item['robot'],method)
            rid=f"{item['robot']}_{item['site_id']}_{method}_r{repeat}"
            record=folder/'runs'/f'{rid}.jsonl.gz';summary_path=folder/'runs'/f'{rid}.summary.json'
            if summary_path.exists():
                if not resume:raise RuntimeError(f'existing job {rid}')
                summaries.append(json.loads(summary_path.read_text()));continue
            if record.exists():raise RuntimeError(f'Unfinished job requires explicit recovery, not rerun: {record}')
            if key not in solvers:
                solvers[key]=factory(method,item['robot'],cfg)
                solver,kin,verifier=solvers[key]
                # Fixed stationary startup call, not measured; then reset. No future target.
                q=np.asarray(item['initial_q']);pose=kin.forward(q)
                solver.solve(pose.position,pose.rotation,q,item['dt'])
                metadata=getattr(solver,'metadata',{})
                if callable(metadata):metadata=metadata()
                meta_path=folder/f'{item["robot"]}_{method}_adapter.json'
                if not meta_path.exists():write_json(meta_path,dict(metadata=metadata,startup_warmup_calls=1))
            solver,kin,verifier=solvers[key]
            rows,summary=execute_trajectory(solver,kin,verifier,item,method,repeat)
            with gzip.open(record,'xt',encoding='utf8') as f:
                for row in rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
            summary.update(robot=item['robot'],uid=item['uid'],site_id=item['site_id'],family=item['family'],
                           method=method,repeat=repeat,run_id=rid,raw_file=str(record.relative_to(folder)))
            write_json(summary_path,summary);summaries.append(summary)
            print(f"{folder.name} {jobnum+1}/{len(jobs)} {rid} complete={int(summary['completion'])} "
                  f"P95={summary['p95_ms']:.2f}ms optimized={summary['optimized_command_rate']:.2f} "
                  f"elapsed={(time.monotonic()-begin)/60:.1f}min",flush=True)
    finally:
        for solver,_,_ in solvers.values():solver.close()
    if hashes!=code_hashes():raise RuntimeError('Experiment code changed during a measurement run')
    write_json(folder/'summaries.json',summaries)
    write_json(folder/'completed.json',dict(utc=utc(),runs=len(summaries),frames=sum(r['frames'] for r in summaries),
        files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))


def development_items(cfg,initial):
    result=[]
    for robot in cfg['robots']:
        items=json.loads((ROOT/cfg['development'][robot+'_targets']).read_text())
        if initial:
            items=[min([i for i in items if i['family']==fam],key=lambda i:i['uid'])
                   for fam in sorted({i['family'] for i in items})]
        result.extend(dict(robot=robot,**i) for i in items)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/correction_reserve.yaml')
    p.add_argument('--folder',required=True);p.add_argument('--initial',action='store_true')
    p.add_argument('--inputs');p.add_argument('--methods',nargs='+');p.add_argument('--repeats',type=int)
    p.add_argument('--robot',choices=['panda','ur5e'])
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    items=json.loads(Path(args.inputs).read_text()) if args.inputs else development_items(cfg,args.initial)
    if args.robot:items=[i for i in items if i['robot']==args.robot]
    run(args.folder,items,args.methods or cfg['methods'],cfg,repeats=args.repeats,resume=args.resume)


if __name__=='__main__':main()
