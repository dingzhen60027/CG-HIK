"""One fixed development comparison; old runners/records remain unchanged."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import yaml

from ..types import Pose
from ..geometry import pose_error
from .geometry import predict_target, task_scale
from .minimal_demand import empirical_quantile
from .minimal_runtime import MinimalInterventionIK, MODES
from . import study as old
from .reporting import csv_write

CONFIG='configs/correction_reserve_minimal.yaml'
NEW_METHODS={v:k for k,v in MODES.items()}


def configurations():
    cfg=yaml.safe_load((old.ROOT/CONFIG).read_text())
    base=yaml.safe_load((old.ROOT/cfg['base_config']).read_text())
    return cfg,base,old.ROOT/cfg['output']


def hashes():
    cfg,base,_=configurations()
    paths=[old.ROOT/'src/confik/correction_reserve'/f'minimal_{name}.py'
           for name in ('demand','convex','runtime','study')]
    paths += [old.ROOT/CONFIG,old.ROOT/'docs/CRIK_MINIMAL_INTERVENTION_PROTOCOL.md',
              old.ROOT/cfg['base_config'],old.ROOT/base['source_config'],
              old.ROOT/'src/confik/task_contract_alignment/outcomes.py',
              old.ROOT/'src/confik/solvers/verifier.py']
    return dict(old.code_hashes(),**{str(p.relative_to(old.ROOT)):old.sha(p) for p in paths})


def load_development(base):
    # The only data inputs. Do not call the fresh generator or open fresh outputs.
    items=old.development_items(base,False)
    for robot in ('panda','ur5e'):
        group=[i for i in items if i['robot']==robot]
        assert len(group)==40 and len({i['uid'] for i in group})==40
        assert sorted(Counter(i['family'] for i in group).values())==[10]*4
        assert all(len(i['target_position'])==len(i['target_rotation'])==150 and i['dt']==.02 for i in group)
    return items


def prepare():
    cfg,base,root=configurations();protocol=root/'protocol'
    protocol.mkdir(parents=True,exist_ok=False)
    items=load_development(base);parameters={};error_rows=[]
    for robot in cfg['robots']:
        _,_,v,_=old.context(robot,base);scale=task_scale(v);errors=[]
        for item in [i for i in items if i['robot']==robot]:
            last=None;predicted=None
            for frame,(p,R) in enumerate(zip(item['target_position'],item['target_rotation'])):
                target=Pose(p,R)
                if predicted is not None:
                    vector=pose_error(target,predicted)/scale;eta=float(np.linalg.norm(vector))
                    errors.append(eta)
                    error_rows.append(dict(robot=robot,uid=item['uid'],family=item['family'],frame=frame,
                                           normalized_error=eta,normalized_vector=vector.tolist()))
                predicted=predict_target(target,last);last=target
        parameters[robot]=dict(minimum=empirical_quantile(errors,.5),initial=empirical_quantile(errors,.95),
            observed_errors=len(errors),minimum_source='target-only median',initial_source='target-only Q95',
            error_quantiles={str(p):empirical_quantile(errors,p) for p in (.5,.9,.95,.99,1.)})
    old.write_json(protocol/'online_inputs.json',items)
    old.write_json(protocol/'demand_parameters.json',parameters)
    csv_write(protocol/'target_prediction_errors.csv',error_rows)
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','clarabel','pin','pin-pink','qpsolvers','osqp')}
    assert versions['clarabel']=='0.11.1'
    old.write_json(protocol/'dependencies.json',dict(python=platform.python_version(),packages=versions,
        native_library=base['native_trac_library'],native_library_sha256=old.sha(old.ROOT/base['native_trac_library'])))
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_crik_minimal.py'],
                          cwd=old.ROOT,text=True,capture_output=True,check=True)
    old.write_json(protocol/'tests.json',dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    old.write_json(protocol/'selection_seal.json',dict(utc=old.utc(),configuration=cfg,code_hashes=hashes(),
        source_inputs={base['development'][robot+'_targets']:old.sha(old.ROOT/base['development'][robot+'_targets']) for robot in cfg['robots']},
        files={str(p.relative_to(protocol)):old.sha(p) for p in protocol.iterdir() if p.is_file()},
        state='fixed before new comparative solver outcomes; observed development paths only',
        future_data_access=False,parameter_selection_uses_solver_outcomes=False))
    print(json.dumps(parameters,indent=2))


def verify_seal(require_commit=True):
    cfg,base,root=configurations();protocol=root/'protocol'
    seal=json.loads((protocol/'selection_seal.json').read_text())
    assert seal['configuration']==cfg and seal['code_hashes']==hashes()
    for file,h in seal['files'].items():assert old.sha(protocol/file)==h,file
    for file,h in seal['source_inputs'].items():assert old.sha(old.ROOT/file)==h,file
    assert not subprocess.check_output(['git','diff',cfg['baseline_commit'],'--name-status','--diff-filter=DMRT'],cwd=old.ROOT,text=True)
    if require_commit:
        paths=list(hashes())+[str(protocol.relative_to(old.ROOT))]
        for extra in ([],['--cached']):
            assert not subprocess.check_output(['git','diff',*extra,'--name-only','--',*paths],cwd=old.ROOT,text=True)
        assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*paths],cwd=old.ROOT,text=True)
    return seal


def factory(method,robot,base,parameters):
    if method not in NEW_METHODS:return old.factory(method,robot,base)
    source,kin,v,urdf=old.context(robot,base)
    solver=MinimalInterventionIK(kin,v,source,str(old.ROOT/base['native_trac_library']),urdf,
        demand_parameters=parameters[robot],mode=NEW_METHODS[method],config=base['optimizer'])
    return solver,kin,v


def add_offline_metrics(rows,summary,kin,v):
    """Called after the complete run; none of these future joins reach the solver."""
    step=kin.limits.velocity*.02+v.config.velocity_tolerance;scale=task_scale(v)
    interventions=[];edges=[];calls=[];direct_extra=[];coverage=[];met=[]
    for i,row in enumerate(rows):
        accepted=row['accepted']
        if accepted:
            use=max(row['position_error']/scale[0],row['orientation_error']/scale[3])
            row['accepted_tolerance_use']=use;edges.append(use>.9)
        if row.get('backup_accepted',False) and accepted:
            delta=np.asarray(row['q'])-np.asarray(row['backup_q'])
            row['intervention_normalized_l2']=float(np.linalg.norm(delta/step))
            row['command_unchanged']=bool(np.array_equal(row['q'],row['backup_q']))
            interventions.append(row['intervention_normalized_l2'])
        if row['method'] in NEW_METHODS or row['method'] in ('cr_ik','two_step_predictive'):
            row.setdefault('conic_calls',sum('stage' in s for s in row.get('native_status',[])))
            row.setdefault('optimization_called',row['conic_calls']>0)
            calls.append(row['optimization_called'])
        if row.get('direct_return',False):
            row['direct_extra_overhead_ns']=row['total_latency_ns']-row['backup_ns']
            direct_extra.append(row['direct_extra_overhead_ns'])
        if i+1<len(rows):
            target=Pose(row['target_position'],row['target_rotation'])
            last=Pose(rows[i-1]['target_position'],rows[i-1]['target_rotation']) if i else None
            predicted=predict_target(target,last)
            actual=Pose(rows[i+1]['target_position'],rows[i+1]['target_rotation'])
            error=pose_error(actual,predicted)/scale
            row['offline_next_prediction_error']=float(np.linalg.norm(error))
            row['offline_next_prediction_error_vector']=error.tolist()
            row['offline_actual_next_accepted']=rows[i+1]['accepted']
            row['offline_actual_next_timely_accepted']=rows[i+1]['accepted_within_20ms']
            if 'demand' in row:
                covered=row['offline_next_prediction_error']<=row['demand']
                row['offline_demand_covered']=covered;coverage.append(covered);met.append(row['demand_met'])
    summary.update(optimization_call_rate=float(np.mean(calls)) if calls else None,
        mean_conic_calls=float(np.mean([r.get('conic_calls',0) for r in rows])) if calls else None,
        intervention_normalized_mean=float(np.mean(interventions)) if interventions else None,
        intervention_normalized_p95=float(np.percentile(interventions,95)) if interventions else None,
        unchanged_given_legal_backup=float(np.mean([r['command_unchanged'] for r in rows
            if r.get('backup_accepted',False) and r['accepted']])) if interventions else None,
        accepted_near_tolerance_rate=float(np.mean(edges)) if edges else None,
        direct_return_rate=float(np.mean([r.get('direct_return',False) for r in rows])),
        direct_extra_p50_ms=float(np.median(direct_extra)/1e6) if direct_extra else None,
        demand_coverage=float(np.mean(coverage)) if coverage else None,
        demand_met_rate=float(np.mean([r['demand_met'] for r in rows])) if 'demand_met' in rows[0] else None)


def run(robot):
    seal=verify_seal();cfg,base,root=configurations();folder=root/robot
    folder.mkdir(exist_ok=False);(folder/'runs').mkdir()
    items=[i for i in json.loads((root/'protocol/online_inputs.json').read_text()) if i['robot']==robot]
    parameters=json.loads((root/'protocol/demand_parameters.json').read_text())
    start_record=dict(utc=old.utc(),code_hashes=hashes(),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip(),
        configuration=cfg,cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    old.write_json(folder/'started.json',start_record);old.write_json(folder/'online_inputs.json',items)
    jobs=[(i,m,r) for i in items for m in cfg['methods'] for r in range(1 if m=='pink_qp' else cfg['trac_repeats'])]
    order=np.random.default_rng(cfg['order_seed']).permutation(len(jobs)).tolist()
    old.write_json(folder/'job_order.json',[dict(uid=jobs[j][0]['uid'],method=jobs[j][1],repeat=jobs[j][2]) for j in order])
    solvers={};summaries=[];begin=time.monotonic()
    try:
        for k,j in enumerate(order):
            item,method,repeat=jobs[j];rid=f'{robot}_{item["site_id"]}_{method}_r{repeat}'
            if method not in solvers:
                solvers[method]=factory(method,robot,base,parameters)
                solver,kin,v=solvers[method];q=np.asarray(item['initial_q']);pose=kin.forward(q)
                solver.solve(pose.position,pose.rotation,q,.02)  # one fixed stationary startup, then reset
            solver,kin,v=solvers[method]
            rows,summary=old.execute_trajectory(solver,kin,v,item,method,repeat)
            add_offline_metrics(rows,summary,kin,v)
            record=folder/'runs'/f'{rid}.jsonl.gz'
            with gzip.open(record,'xt',encoding='utf8') as f:
                for row in rows:f.write(json.dumps(old.clean(row),allow_nan=False,separators=(',',':'))+'\n')
            summary.update(robot=robot,uid=item['uid'],site_id=item['site_id'],family=item['family'],method=method,
                repeat=repeat,run_id=rid,raw_file=str(record.relative_to(folder)))
            old.write_json(folder/'runs'/f'{rid}.summary.json',summary);summaries.append(summary)
            print(f'{robot} {k+1}/{len(jobs)} {rid} completion={int(summary["completion"])} '
                  f'P95={summary["p95_ms"]:.3f}ms elapsed={(time.monotonic()-begin)/60:.1f}min',flush=True)
    finally:
        for solver,_,_ in solvers.values():solver.close()
    assert hashes()==seal['code_hashes']
    old.write_json(folder/'summaries.json',summaries)
    old.write_json(folder/'completed.json',dict(utc=old.utc(),runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        files={str(p.relative_to(folder)):old.sha(p) for p in folder.rglob('*') if p.is_file()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','check'])
    p.add_argument('--robot',choices=['panda','ur5e']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':verify_seal();print('Frozen minimum-intervention protocol and old evidence unchanged.')
    else:
        if args.robot is None:p.error('--robot required')
        run(args.robot)


if __name__=='__main__':main()
