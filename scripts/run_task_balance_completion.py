#!/usr/bin/env python3
"""Finite task-balance completion work package; original runner and geometry."""
import argparse
import ast
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/single_solver_evidence/task_balance_completion'
PACKAGE=OUT/'task_package/task_balance_completion'
spec=importlib.util.spec_from_file_location('balance_previous_entry',ROOT/'scripts/run_task_balance_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OLD_FACTORY=old.factory
OLD_HASHES=old.code_hashes
from confik.correction_reserve.study import context,write_json,sha,clean,utc,read_rows
from confik.types import IKQuery,Pose
from confik.correction_reserve import study as runner
from confik.correction_reserve.reporting import csv_write
from confik.task_balance_gn import CompletedTaskBalanceGN,CompletionSettings,completion_dual_direction,box_qp,task_value
from confik.task_balance_reference import EpigraphReference,quality

CONFIG=ROOT/'configs/task_balance_completion.yaml'

def code_hashes():
    names=list(OLD_HASHES())+['scripts/run_task_balance_completion.py','configs/task_balance_completion.yaml',
        'src/confik/revision_compute_allocation/data.py']
    return {n:sha(ROOT/n) for n in names}

def factory(method,robot,cfg,trace=False):
    if method.startswith('balance_'):
        _,kin,v,urdf=context(robot,cfg)
        solver=CompletedTaskBalanceGN(kin,v,urdf,posture_weight=float(method.endswith('k1')),
            forcing=None if method.startswith('balance_cached') else .25,trace=trace)
    elif method=='trac_task_20ms':
        from confik.task_contract_alignment.outcomes import ContractSolver
        source,kin,v,urdf=context(robot,cfg)
        solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    else:return OLD_FACTORY(method,robot,cfg,trace=trace)
    q=(kin.limits.lower+kin.limits.upper)/2
    target=kin.forward(q+.25*(kin.limits.velocity*.02+v.config.velocity_tolerance))
    for _ in range(3):solver.solve(target.position,target.rotation,q,.02)
    settings=solver.settings.__dict__ if hasattr(solver,'settings') else dict(method=method,budget_ms=20,native_mapping=solver.native.mapping)
    solver.metadata=dict(backend=old.old.old.configure_backend(),settings=settings,
        warmup='3 common nonstationary midpoint calls plus one runner stationary startup',
        timing='Complete outer solve: conversion, fixed bounds, cached native residual/Jacobian, all QPs/trials and independent final verifier; no serialization.',
        local_progress='80% of optimal regularized local MODEL reduction only, conditional on numerically consistent bounds; no nonlinear/trajectory guarantee.')
    return solver,kin,v

def prepare(cfg):
    folder=OUT/'protocol';folder.mkdir(exist_ok=False)
    for name in ('failure_inputs.json','subproblems.json'):
        write_json(folder/name,json.loads((old.OUT/'protocol'/name).read_text()))
    write_json(folder/'protocol.json',dict(created=utc(),config=cfg,code_hashes=code_hashes(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [old.OUT/'protocol/failure_inputs.json',old.OUT/'protocol/subproblems.json']},
        input_files={r:dict(path=cfg['development'][r+'_targets'],sha256=sha(ROOT/cfg['development'][r+'_targets'])) for r in cfg['robots']},
        backend=old.old.old.configure_backend(),repeats='3 nested within input/UID; not independent replication',
        numerical_changes='Port supplied cached core, keep historical class intact. Finite bound validation and exact first-QP accounting; shared adapter absolute deadline; final verifier retained.',
        independent_plan=cfg['fresh'],independent_methods=cfg['independent_methods']))

def check_protocol():
    p=json.loads((OUT/'protocol/protocol.json').read_text())
    revision=OUT/'protocol/measurement_seal.json'
    expected=json.loads(revision.read_text())['code_hashes'] if revision.exists() else p['code_hashes']
    launch=OUT/'independent_inputs/launch_seal.json'
    if launch.exists():expected=json.loads(launch.read_text())['code_hashes']
    assert expected==code_hashes()
    for path,h in p['source_hashes'].items():assert sha(ROOT/path)==h
    for row in p['input_files'].values():assert sha(ROOT/row['path'])==row['sha256']

def local(cfg):
    # Reuse existing exact-input evaluator and independent verification/witness writer.
    old.OUT=OUT;old.factory=factory;old.check_protocol=check_protocol;old.code_hashes=code_hashes
    c=dict(cfg,methods=[m for m in cfg['methods'] if m!='pink_qp'])
    old.local(c)

def trajectories(cfg,robot):
    check_protocol();runner.factory=factory;runner.code_hashes=code_hashes
    runner.run(OUT/f'development_{robot}',old.old.old.items(cfg,robot),cfg['methods'],dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=3)

def cached_step(a,forcing,acceptable=lambda d:False,trace=True):
    e,G,lo,hi=(a[k] for k in ('e','G','lo','hi'));n=G.shape[1]
    lam=a['damping'];k=a['posture_weight'];c=a['posture'];w=a['posture_scale']
    H=G.T@G+lam*np.eye(n)+k*np.diag(w*w);g=G.T@e+k*w*c
    d,nit=box_qp(H,g,lo,hi)
    if acceptable(d):return d,dict(reason='task_first',theta=.5,dual_updates=1,upper=None,lower=None,gap=None)
    return completion_dual_direction(e,G,lo,hi,lam,k,c,w,box_qp,(H,g,d,nit),acceptable,
        CompletionSettings(forcing=forcing),perf_counter_ns()+10**12,trace)

def mathematics(cfg):
    from scipy.optimize import minimize
    folder=OUT/'mathematics';folder.mkdir(exist_ok=False);rng=np.random.default_rng(739102);rows=[]
    for i in range(48):
        n=6+i%2;e=rng.normal(size=6)*3.;G=rng.normal(size=(6,n));lo=-rng.uniform(.1,1.,n);hi=rng.uniform(.1,1.,n)
        a=dict(e=e,G=G,lo=lo,hi=hi,damping=.01,posture=np.zeros(n),posture_scale=np.zeros(n),posture_weight=0.)
        d,info=cached_step(a,.25)
        def obj(x):return x[-1]+.005*(x[:-1]@x[:-1])
        def jac(x):return np.r_[.01*x[:-1],1.]
        def con(x):
            r=e+G@x[:-1];return np.array([x[-1]-r[:3]@r[:3],x[-1]-r[3:]@r[3:]])
        def cj(x):
            r=e+G@x[:-1];return np.array([np.r_[-2*r[:3]@G[:3],1.],np.r_[-2*r[3:]@G[3:],1.]])
        r=minimize(obj,np.r_[np.zeros(n),task_value(e)],jac=jac,method='SLSQP',bounds=list(zip(lo,hi))+[(None,None)],
            constraints={'type':'ineq','fun':con,'jac':cj},options={'ftol':1e-11,'maxiter':1000})
        opt=obj(r.x);U=info['upper'];L=info['lower'];P0=info['pzero'];frac=(P0-U)/(P0-opt)
        assert min(con(r.x))>=-1e-7 and L<=opt+1e-6*max(1,abs(opt)) and U>=opt-1e-6*max(1,abs(opt))
        if info['reason']=='relative_progress':assert frac>=.8-1e-6
        rows.append(dict(id=i,n=n,reference_success=bool(r.success),reference_objective=opt,
            direction=d.tolist(),achieved_fraction=frac,**info))
    write_json(folder/'convex_checks.json',rows)
    print('48 supplied convex checks:',sum(r['reason']=='relative_progress' for r in rows),'relative stops',flush=True)

def subproblems(cfg):
    check_protocol();folder=OUT/'subproblems';folder.mkdir(exist_ok=False)
    problems=json.loads((OUT/'protocol/subproblems.json').read_text());records=[];refs={n:EpigraphReference(n) for n in (6,7)}
    natives={r:factory('balance_cached_k0',r,cfg) for r in cfg['robots']}
    for index,p in enumerate(problems):
        keys=('e','G','lo','hi','damping','posture','posture_scale','posture_weight')
        solver,kin,v=natives[p['robot']]
        query=IKQuery(Pose(np.array(p['target_position']),np.array(p['target_rotation'])),np.array(p['previous_q']),p['dt'])
        for method,rep in np.random.default_rng(cfg['order_seed']+index).permutation([(m,r) for m in
                ('old_tight','cached_tight','relative_progress','clarabel_epigraph','old_command','cached_command','relative_command','clarabel_command') for r in range(cfg['subproblems']['repeat'])]):
            start=perf_counter_ns();a={k:np.array(p[k],float) if isinstance(p[k],list) else p[k] for k in keys};checks=0
            def acceptable(d):
                nonlocal checks
                checks+=1;q=np.clip(np.array(p['q'])+np.array(p['step_scale'])*d,p['frame_lower'],p['frame_upper'])
                return bool(v.check(q,query).accepted)
            conversion_ns=perf_counter_ns()-start;at=perf_counter_ns()
            if method.startswith('clarabel'):d,info=refs[len(p['q'])].solve(**a)
            elif method.startswith('old_'):d,info=old.minimax_step(**a,max_dual_updates=16,acceptable=acceptable if method=='old_command' else None)
            else:d,info=cached_step(a,None if method.startswith('cached') else .25,acceptable if method.endswith('command') else lambda d:False,trace=False)
            kernel_ns=perf_counter_ns()-at
            if not method.startswith('clarabel'):info.update(quality(d,**a,theta=info['theta']))
            command_ok=acceptable(d) if method.endswith('command') else None
            records.append(dict(problem_id=p['problem_id'],robot=p['robot'],stratum=p['stratum'],kappa=p['posture_weight'],method=method,
                repeat=int(rep),direction=d.tolist(),**info,outer_ns=perf_counter_ns()-start,kernel_ns=kernel_ns,conversion_ns=conversion_ns,verifier_calls=checks,command_accepted=command_ok))
        if index%80==0:print('fixed real problem',index+1,'/',len(problems),flush=True)
    write_json(folder/'raw_results.json',records)
    write_json(folder/'manifest.json',dict(code_hashes=code_hashes(),problems=len(problems),calls=len(records),
        structure_initialization_ns={n:r.initialization_ns for n,r in refs.items()},source_sha256=sha(OUT/'protocol/subproblems.json')))

def inputs(cfg):
    from confik.continuation_mechanism.independent_selection import metadata_keys
    from confik.revision_compute_allocation.data import points,trajectories as generate
    check_protocol();folder=OUT/'independent_inputs';folder.mkdir(exist_ok=False)
    prior=dict(uids=set(),seeds=set(),hashes=set());sources={}
    for p in sorted((ROOT/'outputs').rglob('*identit*.json')):
        if OUT in p.parents:continue
        metadata_keys(json.loads(p.read_text()),prior);sources[str(p.relative_to(ROOT))]=sha(p)
    for robot in cfg['robots']:metadata_keys(old.old.old.items(cfg,robot),prior)
    identities=[];online=[];point_ids=[]
    families=['smooth','near_singular','joint_limit_return','high_curvature']
    for robot in cfg['robots']:
        source,kin,v,_=context(robot,cfg);seed=cfg['fresh']['point_seeds'][robot]
        assert seed not in prior['seeds'],'Frozen fresh seed collision; stop, no outcome-based replacement'
        gen=dict(cfg['fresh'],_source=source,oracle_counts={f:0 for f in cfg['fresh']['point_counts']})
        ds,ids,_=points(kin,gen,robot);assert len(ids)==2000
        for identity in ids:
            assert identity['uid'] not in prior['uids'] and identity['query_hash'] not in prior['hashes']
            identity['robot']=robot;metadata_keys(identity,prior)
        ds.save(folder/f'{robot}_points.npz');point_ids+=ids
        refs=[];base=cfg['fresh']['trajectory_seeds'][robot]
        for i,family in enumerate(f for f in families for _ in range(20)):
            # Preserve the prior independent generator's per-trajectory seed recipe.
            seed=base+i*2;assert seed not in prior['seeds']
            gen=dict(_source=source,trajectory_seeds={robot:seed},trajectories_per_family=1,
                trajectory_frames=150,dt=.02,trajectory_families=[family])
            ds,ids=generate(kin,gen,robot);identity=ids[0]
            assert identity['uid'] not in prior['uids'] and not(set(identity['query_hashes'])&prior['hashes'])
            identity.update(robot=robot,site_id=f'trajectory_{i:03d}');metadata_keys(identity,prior);identities.append(identity)
            refs.append(np.vstack([ds.previous_q[0],ds.reference_q]))
            online.append(dict(robot=robot,site_id=identity['site_id'],uid=identity['uid'],family=family,seed=seed,
                initial_q=ds.previous_q[0].tolist(),target_position=ds.target_position.tolist(),target_rotation=ds.target_rotation.tolist(),dt=.02))
            if i%20==0:print('fresh reference verified',robot,i+1,flush=True)
        with (folder/f'{robot}_trajectory_witnesses.npz').open('xb') as f:np.savez_compressed(f,q=np.array(refs))
    write_json(folder/'point_identities.json',point_ids);write_json(folder/'trajectory_identities.json',identities);write_json(folder/'online_inputs.json',online)
    write_json(folder/'execution_seal.json',dict(created=utc(),code_hashes=code_hashes(),config=cfg,
        old_identity_sources=sources,fresh_solver_calls=0,points=4000,trajectories=160,reference_frames_verified=24000,
        input_selection='Only new seeds; original geometry, amplitudes, utilization and verifier. No solver outcome has been generated.',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def check_seal():
    check_protocol();p=json.loads((OUT/'independent_inputs/execution_seal.json').read_text())
    launch=json.loads((OUT/'independent_inputs/launch_seal.json').read_text())
    assert launch['code_hashes']==code_hashes() and launch['input_seal_sha256']==sha(OUT/'independent_inputs/execution_seal.json')
    for name,h in p['files'].items():assert sha(OUT/'independent_inputs'/name)==h
    return p

def points_run(cfg,robot):
    from confik.data.datasets import QueryDataset
    check_seal();folder=OUT/f'independent_points_{robot}';folder.mkdir(exist_ok=False)
    data=QueryDataset.load(OUT/f'independent_inputs/{robot}_points.npz')
    ids=[r for r in json.loads((OUT/'independent_inputs/point_identities.json').read_text()) if r['robot']==robot]
    cache={m:factory(m,robot,cfg) for m in cfg['independent_methods']}
    with gzip.open(folder/'records.jsonl.gz','xt') as f:
        for i,identity in enumerate(ids):
            p,R,previous=data.target_position[i],data.target_rotation[i],data.previous_q[i]
            for method,rep in np.random.default_rng(cfg['order_seed']+i).permutation([(m,r) for m in cfg['independent_methods'] for r in range(3)]):
                solver,kin,v=cache[method]
                if hasattr(solver,'reset'):solver.reset(previous)
                result=solver.solve(p,R,previous,.02)
                # Audit check is outside solver time; each method already includes final verification.
                q=None if result['q'] is None else np.array(result['q'])
                check=v.check(q,IKQuery(Pose(p,R),previous,.02));assert bool(check.accepted)==result['accepted']
                row=dict(identity,repeat=int(rep),**result);row['method']=str(method)
                row.update(previous_q=previous.tolist(),target_position=p.tolist(),target_rotation=R.tolist(),dt=.02,
                    witness_confirmed_miss=not result['accepted'])
                f.write(json.dumps(clean(row),separators=(',',':'),allow_nan=False)+'\n')
            if i%200==0:print('independent point',robot,i+1,'/2000',flush=True)
    for solver,_,_ in cache.values():solver.close()
    write_json(folder/'manifest.json',dict(created=utc(),code_hashes=code_hashes(),query_count=2000,calls=36000,
        source_seal_sha256=sha(OUT/'independent_inputs/execution_seal.json'),records_sha256=sha(folder/'records.jsonl.gz')))

def evaluate(cfg,robot):
    check_seal();runner.factory=factory;runner.code_hashes=code_hashes
    data=[r for r in json.loads((OUT/'independent_inputs/online_inputs.json').read_text()) if r['robot']==robot]
    runner.run(OUT/f'independent_trajectories_{robot}',data,cfg['independent_methods'],dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=3)

def package_arrays(cfg):
    data=json.loads((PACKAGE/'panda_observed160.json').read_text())
    original=[r for r in json.loads((ROOT/'outputs/single_solver_evidence/protocol/online_inputs.json').read_text()) if r['robot']=='panda']
    lookup={r['uid']:r for r in original};assert {r['uid'] for r in data}==set(lookup);rows=[]
    for r in data:
        source=lookup[r['uid']];diff={k:float(np.max(abs(np.array(r[k])-np.array(source[k])))) for k in ('initial_q','target_position','target_rotation')}
        rows.append(dict(uid=r['uid'],site_id=r['site_id'],dt_equal=r['dt']==source['dt'],max_abs_difference=diff,
            within_1e12=all(x<=1e-12 for x in diff.values())))
    write_json(OUT/'package_verification/observed160_all_array_comparison.json',dict(records=rows,
        exact_uid_identity=True,source_sha256=sha(ROOT/'outputs/single_solver_evidence/protocol/online_inputs.json'),
        note='Observed historical input reconstruction only; floating FK differences retained; not new independent evidence.'))

def verify(cfg):
    folder=OUT/'package_verification';folder.mkdir(parents=True,exist_ok=False)
    data=json.loads((PACKAGE/'results/selected.json').read_text());i=data['input']
    with (ROOT/i['source']).open() as f:source=list(csv.DictReader(f))
    matches=[r for r in source if r['robot']==i['robot'] and r['uid']==i['uid'] and int(r['frame'])==i['frame'] and r['method']==i['source_method']]
    assert any(float(r['dt'])==i['dt'] and all(np.array_equal(ast.literal_eval(r[k]),i[k]) for k in ('previous_q','target_position','target_rotation')) for r in matches)
    _,kin,v,_=context(i['robot'],cfg)
    query=IKQuery(Pose(np.array(i['target_position']),np.array(i['target_rotation'])),np.array(i['previous_q']),i['dt'])
    records=[]
    for row in data['results']:
        q=np.array(row['q']);verdict=v.check(q,query)
        records.append(dict(method=row['method'],repeat=row['repeat'],q=q.tolist(),accepted=bool(verdict.accepted),
            finite=bool(verdict.finite_ok),joint_limit_ok=bool(verdict.joint_limit_ok),velocity_ok=bool(verdict.velocity_ok),
            position_m=verdict.position_error,orientation_rad=verdict.orientation_error,orientation_deg=float(np.rad2deg(verdict.orientation_error)),
            reasons=list(verdict.reasons),step=(q-query.previous_q).tolist(),
            velocity_utilization=float(np.max(abs(q-query.previous_q)/(kin.limits.velocity*i['dt']+v.config.velocity_tolerance)))))
    write_json(folder/'original_verifier.json',dict(input=i,solver_calls=0,records=records,
        source_sha256=sha(ROOT/i['source']),archive_sha256=sha(ROOT/'task_balance_completion_package.zip'),
        package_files={str(p.relative_to(PACKAGE)):sha(p) for p in PACKAGE.rglob('*') if p.is_file()}))
    csv_write(folder/'commands.csv',records)
    print(json.dumps(clean(records),indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['verify','prepare','math','local','subproblems','trajectories','inputs','points','evaluate','package_arrays','seal','launch_seal']);parser.add_argument('--robot');args=parser.parse_args()
    os.sched_setaffinity(0,{4});cfg=yaml.safe_load(CONFIG.read_text())
    if args.stage=='verify':verify(cfg)
    elif args.stage=='prepare':prepare(cfg)
    elif args.stage=='math':mathematics(cfg)
    elif args.stage=='local':local(cfg)
    elif args.stage=='subproblems':subproblems(cfg)
    elif args.stage=='trajectories':trajectories(cfg,args.robot)
    elif args.stage=='inputs':inputs(cfg)
    elif args.stage=='points':points_run(cfg,args.robot)
    elif args.stage=='evaluate':evaluate(cfg,args.robot)
    elif args.stage=='package_arrays':package_arrays(cfg)
    elif args.stage=='seal':
        p=json.loads((OUT/'protocol/protocol.json').read_text())
        oldh=p['code_hashes'];newh=code_hashes()
        assert {k for k in newh if oldh.get(k)!=newh[k]}=={'scripts/run_task_balance_completion.py'}
        write_json(OUT/'protocol/measurement_seal.json',dict(created=utc(),code_hashes=newh,
            previous_protocol_sha256=sha(OUT/'protocol/protocol.json'),reason='Before any micro/fresh results: avoid duplicate Clarabel quality checks; add all four public command-acceptance cost controls; archive supplied package under result root. No numerical runtime/config changes; completed development records untouched.'))
    elif args.stage=='launch_seal':
        source=OUT/'independent_inputs/execution_seal.json';p=json.loads(source.read_text());h=code_hashes()
        assert {k for k in h if p['code_hashes'].get(k)!=h[k]}=={'scripts/run_task_balance_completion.py'}
        adapters=[]
        for robot in cfg['robots']:
            for method in ('trac_task_5ms','trac_task_20ms'):
                solver,kin,v=factory(method,robot,cfg)
                adapters.append(dict(robot=robot,method=method,native_mapping=solver.native.mapping));solver.close()
        write_json(OUT/'independent_inputs/launch_seal.json',dict(created=utc(),code_hashes=h,input_seal_sha256=sha(source),
            fresh_solver_calls=0,smoke='Synthetic midpoint warmups only, no independent input opened by any solver',adapters=adapters,
            change='Metadata-only correction: TRAC20 has native bounds/budget metadata, not inapplicable GN default settings. Inputs, numerical runtime, settings, previous seals and all development measurements unchanged.'))

if __name__=='__main__':main()
