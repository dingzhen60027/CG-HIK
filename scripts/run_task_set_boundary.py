#!/usr/bin/env python3
"""Fixed supplementary task-set domain study; no change to the main solver."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('existing_numpy_environment',ROOT/'scripts/run_single_solver_development.py')
environment=importlib.util.module_from_spec(spec);spec.loader.exec_module(environment)
BACKEND=environment.configure_backend()
assert BACKEND['engine']=='numpy_supplied_fallback', 'Keep the frozen NumPy execution path'
from confik.correction_reserve.study import context,sha,write_json,clean,utc
from confik.correction_reserve.reporting import csv_write
from confik.bounded_gn_adapter import SingleBoundedGN
from confik.task_balance_gn import CompletedTaskBalanceGN
from confik.task_set_controls import FixedWeightGN,ClarabelBalance
from confik.types import IKQuery,Pose

OUT=ROOT/'outputs/single_solver_evidence/task_set_boundary_evaluation'
PACKAGE=OUT/'task_package/task_set_probe'
CONFIG=ROOT/'configs/task_set_boundary.yaml'

def frozen_hashes():
    names=list(environment.code_hashes())+['src/confik/task_balance_gn.py','src/confik/task_balance_reference.py',
        'src/confik/revision_compute_allocation/data.py']
    return {n:sha(ROOT/n) for n in names}

def study_hashes():
    return {n:sha(ROOT/n) for n in ['scripts/run_task_set_boundary.py','src/confik/task_set_controls.py',
                                  'configs/task_set_boundary.yaml','tests/test_task_set_boundary.py']}

def factory(method,robot,cfg,theta=None):
    source,kin,v,urdf=context(robot,cfg)
    t=perf_counter_ns()
    if method=='gn':s=SingleBoundedGN(kin,v,urdf,1.)
    elif method in ('relative','tight'):s=CompletedTaskBalanceGN(kin,v,urdf,1.,.25 if method=='relative' else None)
    elif method.startswith('weight_') or method=='fixed_weight':
        s=FixedWeightGN(kin,v,urdf,float(method.split('_')[1]) if method.startswith('weight_') else theta)
    elif method=='clarabel':s=ClarabelBalance(kin,v,urdf)
    elif method.startswith('trac'):
        from confik.task_contract_alignment.outcomes import ContractSolver
        s=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    else:raise ValueError(method)
    initialization=perf_counter_ns()-t
    # Same synthetic, nonstationary full-call warmup for all methods, no witnesses.
    q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q+.25*kin.limits.velocity*cfg['dt'])
    for _ in range(3):s.solve(pose.position,pose.rotation,q,cfg['dt'])
    if method=='clarabel':
        # Warm/cache native conic structure, even if the easy full call exits at .5.
        n=kin.nq;G=np.eye(6,n);s.epigraph.solve(np.full(6,2.),G,-np.ones(n),np.ones(n),.01,np.zeros(n),np.ones(n)*.01,1.)
    metadata=dict(initialization_ns=initialization,warmup_calls=3,backend=BACKEND,
        additional_conic_structure_warmup=method=='clarabel',
        frozen_settings=(s.engine.settings if hasattr(s,'engine') else s.settings).__dict__ if hasattr(s,'settings') or hasattr(s,'engine') else None,
        theta=getattr(getattr(s,'engine',None),'theta',None),native_mapping=s.native.mapping if method.startswith('trac') else None,
        timing='Caller perf_counter_ns spans complete solve invocation and final verifier; initialization/warmup/serialization and independent audit are outside. All adapter timing retained separately.')
    return s,kin,v,metadata

def generate_robot(cfg,robot,split):
    from scipy.spatial.transform import Rotation
    from confik.revision_compute_allocation.data import near_singular_center
    from confik.latency_pilot_v3.benchmark import query_digest
    _,kin,v,_=context(robot,cfg);seed=cfg[split]['seeds'][robot];rng=np.random.default_rng(seed)
    records=[];span=kin.limits.upper-kin.limits.lower
    for family in cfg['anchor_families']:
        for j in range(cfg[split]['anchors_per_family']):
            previous=near_singular_center(kin,rng,64) if family=='near_singular' else kin.random_configuration(rng,.07)
            axis=None
            if family=='near_limit':
                axis=int(rng.integers(kin.nq));previous[axis]=kin.limits.upper[axis]-rng.uniform(.001,.008)*span[axis]
            anchor=hashlib.sha256(f'task_set_anchor:{robot}:{split}:{seed}:{family}:{j}'.encode()).hexdigest()
            dp=rng.normal(size=3);dp/=np.linalg.norm(dp);dr=rng.normal(size=3);dr/=np.linalg.norm(dr)
            for displacement in cfg['displacements']:
                if displacement=='regular':scale=rng.uniform(.1,.6);direction=rng.uniform(-1,1,kin.nq)
                elif displacement=='high_utilization':scale=rng.uniform(.75,.98);direction=rng.choice([-1.,1.],kin.nq)*rng.uniform(.75,1.,kin.nq)
                else:scale=.999;direction=rng.choice([-1.,1.],kin.nq)
                proposed=previous+scale*kin.limits.velocity*cfg['dt']*direction
                witness=kin.clip(proposed,margin=0.);pose=kin.forward(witness)
                for alpha in cfg['alpha']:
                    target=Pose(pose.position+alpha*v.config.position_tolerance*dp,
                        Rotation.from_rotvec(alpha*v.config.orientation_tolerance*dr).as_matrix()@pose.rotation)
                    query=IKQuery(target,previous,cfg['dt']);verdict=v.check(witness,query)
                    assert verdict.accepted,('Generator witness rejected',robot,family,j,displacement,alpha,verdict)
                    uid=hashlib.sha256(f'{anchor}:{displacement}:{alpha}'.encode()).hexdigest()
                    records.append(dict(robot=robot,split=split,seed=seed,anchor_uid=anchor,anchor_family=family,anchor_index=j,
                        uid=uid,query_hash=query_digest(query),displacement=displacement,alpha=alpha,dt=cfg['dt'],
                        previous_q=previous.tolist(),q_witness=witness.tolist(),target_position=target.position.tolist(),target_rotation=target.rotation.tolist(),
                        direction_p=dp.tolist(),direction_R=dr.tolist(),sigma_previous=float(kin.min_singular_value(previous)),near_limit_axis=axis,
                        displacement_scale=float(scale),joint_direction=direction.tolist(),physical_clipped=(proposed!=witness).tolist(),
                        witness_velocity_utilization=(abs(witness-previous)/(kin.limits.velocity*cfg['dt']+v.config.velocity_tolerance)).tolist(),
                        witness_position_error=verdict.position_error,witness_orientation_error=verdict.orientation_error,
                        exact_center='known_FK_witness' if alpha==0 else 'unknown',witness_verified=True))
    return records

def generate(cfg):
    from confik.continuation_mechanism.independent_selection import metadata_keys
    folder=OUT/'inputs';folder.mkdir(exist_ok=False)
    prior=dict(uids=set(),seeds=set(),hashes=set());sources={}
    for p in sorted((ROOT/'outputs').rglob('*identit*.json')):
        if OUT in p.parents:continue
        metadata_keys(json.loads(p.read_text()),prior);sources[str(p.relative_to(ROOT))]=sha(p)
    # Supplied observed probes have their own seeds and exact input hashes.
    from confik.latency_pilot_v3.benchmark import query_digest
    for phase in ('measured','boundary_measured'):
        prior['seeds'].add(json.loads((PACKAGE/phase/'protocol.json').read_text())['seed'])
        for i in json.loads((PACKAGE/phase/'inputs.json').read_text()):prior['hashes'].add(query_digest(query_of(i)))
    ids=[];input_files={}
    for split in ('development','validation'):
        for robot in cfg['robots']:
            assert cfg[split]['seeds'][robot] not in prior['seeds'];prior['seeds'].add(cfg[split]['seeds'][robot])
            records=generate_robot(cfg,robot,split)
            assert len(records)==cfg[split]['anchors_per_family']*27
            for r in records:
                assert r['uid'] not in prior['uids'] and r['query_hash'] not in prior['hashes']
                prior['uids'].add(r['uid']);prior['hashes'].add(r['query_hash'])
                ids.append({k:r[k] for k in ('robot','split','seed','anchor_uid','anchor_family','uid','query_hash','displacement','alpha')})
            p=folder/f'{split}_{robot}.json';write_json(p,records);input_files[p.name]=sha(p)
            print('Geometric inputs fixed; witnesses all legal:',split,robot,len(records),flush=True)
    write_json(folder/'identities.json',ids);input_files['identities.json']=sha(folder/'identities.json')
    write_json(folder/'seal.json',dict(created=utc(),config=cfg,input_files=input_files,prior_identity_sources=sources,
        frozen_hashes=frozen_hashes(),study_hashes=study_hashes(),backend=BACKEND,solver_calls_on_new_inputs=0,
        policy='All development and validation inputs fixed geometrically before any method sees either set. No success/cost filtering; center reachability unknown for nonzero offsets.'))

def check_inputs():
    seal=json.loads((OUT/'inputs/seal.json').read_text())
    assert seal['frozen_hashes']==frozen_hashes() and seal['study_hashes']==study_hashes()
    for name,h in seal['input_files'].items():assert sha(OUT/'inputs'/name)==h
    return seal

def run_points(cfg,robot,split):
    import subprocess
    check_inputs();theta=None
    if split=='development':methods=['gn','relative','tight','clarabel','trac_task_5ms','trac_task_20ms']+['weight_'+str(w) for w in cfg['weight_candidates']]
    else:
        selected=json.loads((OUT/'weight_selection.json').read_text());theta=selected['theta'];methods=cfg['methods']
        assert selected['input_seal_sha256']==sha(OUT/'inputs/seal.json')
    folder=OUT/f'{split}_{robot}';folder.mkdir(exist_ok=False)
    items=json.loads((OUT/f'inputs/{split}_{robot}.json').read_text())
    cache={m:factory(m,robot,cfg,theta) for m in methods}
    write_json(folder/'adapters.json',{m:a[3] for m,a in cache.items()})
    jobs=[];order=np.random.default_rng(cfg['order_seed']+(split=='validation')+cfg[split]['seeds'][robot])
    # Query order is fixed random; all methods/repeats interleaved within each query.
    for i in order.permutation(len(items)):
        for m,r in order.permutation([(m,r) for m in methods for r in range(cfg['repeats'])]):jobs.append((int(i),str(m),int(r)))
    write_json(folder/'job_order.json',jobs)
    with gzip.open(folder/'records.jsonl.gz','xt') as f:
        for index,(i,method,repeat) in enumerate(jobs):
            item=items[i];query=query_of(item);solver,kin,v,_=cache[method]
            if hasattr(solver,'reset'):solver.reset(query.previous_q)
            t=perf_counter_ns();result=solver.solve(query.target.position,query.target.rotation,query.previous_q,query.dt);outer=perf_counter_ns()-t
            verdict=v.check(None if result['q'] is None else np.array(result['q']),query)
            assert bool(verdict.accepted)==result['accepted']
            result['adapter_total_latency_ns']=result['total_latency_ns'];result['total_latency_ns']=outer
            result['accepted_within_20ms']=bool(verdict.accepted and outer<=20_000_000)
            record=dict(uid=item['uid'],anchor_uid=item['anchor_uid'],robot=robot,split=split,anchor_family=item['anchor_family'],
                displacement=item['displacement'],alpha=item['alpha'],repeat=repeat,input_index=i,**result)
            record['method']=method
            f.write(json.dumps(clean(record),separators=(',',':'),allow_nan=False)+'\n')
            if (index+1)%1000==0:print(split,robot,index+1,'/',len(jobs),flush=True)
    for s,_,_,_ in cache.values():s.close()
    write_json(folder/'manifest.json',dict(created=utc(),queries=len(items),calls=len(jobs),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_seal_sha256=sha(OUT/'inputs/seal.json'),frozen_hashes=frozen_hashes(),study_hashes=study_hashes(),
        selection_sha256=sha(OUT/'weight_selection.json') if split=='validation' else None,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def select_weight(cfg):
    check_inputs();scores=[]
    for weight in cfg['weight_candidates']:
        per_robot=[]
        for robot in cfg['robots']:
            folder=OUT/f'development_{robot}';manifest=json.loads((folder/'manifest.json').read_text())
            assert sha(folder/'records.jsonl.gz')==manifest['files']['records.jsonl.gz']
            with gzip.open(folder/'records.jsonl.gz','rt') as f:rows=[r for line in f if (r:=json.loads(line))['method']=='weight_'+str(weight)]
            assert len(rows)==270*3
            per_robot.append(dict(robot=robot,accepted=sum(r['accepted'] for r in rows),calls=len(rows),
                success=float(np.mean([r['accepted'] for r in rows])),mean_ms=float(np.mean([r['total_latency_ns'] for r in rows]))/1e6))
        scores.append(dict(theta=weight,robot_scores=per_robot,success=float(np.mean([r['success'] for r in per_robot])),
                           mean_ms=float(np.mean([r['mean_ms'] for r in per_robot]))))
    selected=min(scores,key=lambda r:(-r['success'],r['mean_ms'],abs(r['theta']-.5),r['theta']))
    write_json(OUT/'weight_selection.json',dict(created=utc(),theta=selected['theta'],scores=scores,
        rule='Equal-robot verified success, then equal-robot mean outer time, then distance to .5, then smaller theta for an exact remaining tie.',
        input_seal_sha256=sha(OUT/'inputs/seal.json'),validation_solver_calls=0,
        development_sources={r:sha(OUT/f'development_{r}/manifest.json') for r in cfg['robots']}))
    print('Frozen common baseline theta',selected['theta'],json.dumps(scores),flush=True)

def query_of(i):
    return IKQuery(Pose(np.array(i['target_position']),np.array(i['target_rotation'])),
                   np.array(i.get('previous_q',i.get('previous'))),float(i.get('dt',.02)))

def verdict_row(q,query,kin,v):
    q=np.array(q,float);r=v.check(q,query)
    return dict(q=q.tolist(),accepted=bool(r.accepted),position_error=r.position_error,
        orientation_error=r.orientation_error,reasons=list(r.reasons),finite=bool(r.finite_ok),
        joint_limit_ok=bool(r.joint_limit_ok),velocity_ok=bool(r.velocity_ok),
        joint_step=(q-query.previous_q).tolist(),velocity_utilization=float(np.max(abs(q-query.previous_q)/
            (kin.limits.velocity*query.dt+v.config.velocity_tolerance))))

def verify_package(cfg):
    folder=OUT/'package_verification';folder.mkdir(exist_ok=False)
    _,kin,v,_=context('panda',cfg);six=[];allrows=[]
    for i in json.loads((PACKAGE/'boundary_measured/recovery_examples.json').read_text()):
        query=query_of(i['input'])
        row=dict(anchor=i['input']['anchor'],input=i['input'],**verdict_row(i['relative']['q'],query,kin,v))
        six.append(row)
    # Save/report the existing six commands BEFORE solving any IK.
    write_json(folder/'six_original_commands.json',dict(solver_calls=0,records=six))
    print('ORIGINAL VERIFIER, NO IK:',json.dumps(clean(six)),flush=True)
    for split in ('measured','boundary_measured'):
        inputs=json.loads((PACKAGE/split/'inputs.json').read_text())
        index={(i['anchor'],i['alpha']):i for i in inputs}
        for i in inputs:
            allrows.append(dict(split=split,anchor=i['anchor'],alpha=i['alpha'],method='witness',
                **verdict_row(i['q_witness'],query_of(i),kin,v)))
        for r in json.loads((PACKAGE/split/'records.json').read_text()):
            check=verdict_row(r['q'],query_of(index[r['anchor'],r['alpha']]),kin,v)
            allrows.append(dict(split=split,anchor=r['anchor'],alpha=r['alpha'],method=r['method'],
                supplied_accept=r['accepted'],matches_supplied=check['accepted']==r['accepted'],**check))
    write_json(folder/'all_saved_commands.json',dict(solver_calls=0,records=allrows,
        package_hashes={str(p.relative_to(PACKAGE)):sha(p) for p in PACKAGE.rglob('*') if p.is_file()},
        archive_sha256=sha(ROOT/'task_set_boundary_probe.zip')))
    csv_write(folder/'six_original_commands.csv',six)
    print('All saved inputs/commands checked:',len(allrows),Counter((r['split'],r['method'],r['accepted']) for r in allrows),flush=True)

def rerun_package(cfg):
    assert (OUT/'package_verification/six_original_commands.json').exists()
    folder=OUT/'package_resolve';folder.mkdir(exist_ok=False)
    _,kin,v,urdf=context('panda',cfg)
    solvers={'gn':SingleBoundedGN(kin,v,urdf,1.),'relative':CompletedTaskBalanceGN(kin,v,urdf,1.,.25)}
    q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q+.25*kin.limits.velocity*.02)
    for s in solvers.values():
        for _ in range(3):s.solve(pose.position,pose.rotation,q,.02)
    rows=[]
    for i in json.loads((PACKAGE/'boundary_measured/recovery_examples.json').read_text()):
        query=query_of(i['input'])
        for repeat in range(3):
            for name,s in solvers.items():
                t=perf_counter_ns();r=s.solve(query.target.position,query.target.rotation,query.previous_q,query.dt);elapsed=perf_counter_ns()-t
                rows.append(dict(anchor=i['input']['anchor'],method=name,repeat=repeat,input=i['input'],outer_ns=elapsed,result=r))
    write_json(folder/'records.json',rows)
    print('Same original inputs re-solved:',Counter((r['method'],r['result']['accepted']) for r in rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','resolve','inputs','development','select','validation']);p.add_argument('--robot');args=p.parse_args()
    cfg=yaml.safe_load(CONFIG.read_text());os.sched_setaffinity(0,{cfg['cpu']})
    if args.stage in ('development','validation'):run_points(cfg,args.robot,args.stage)
    else:{'verify':verify_package,'resolve':rerun_package,'inputs':generate,'select':select_weight}[args.stage](cfg)

if __name__=='__main__':main()
