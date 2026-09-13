#!/usr/bin/env python3
"""Single entry: supplied-command acceptance and fixed task-balance development."""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/single_solver_evidence/task_balance_development'
PACKAGE=OUT/'task_package/task_balance_gn'
spec=importlib.util.spec_from_file_location('old_excess_entry',ROOT/'scripts/run_task_excess_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from confik.correction_reserve.study import context,write_json,sha,clean,utc,read_rows
from confik.types import IKQuery,Pose
from confik.correction_reserve import study as runner
from confik.correction_reserve.reporting import csv_write
from confik.correction_reserve.geometry import residual_linearization
from confik.task_balance_gn import TaskBalanceGN,minimax_step
from confik.task_balance_reference import EpigraphReference,quality
from confik.task_excess_gn import dynamic_interval

CONFIG=ROOT/'configs/task_balance_development.yaml'


def code_hashes():
    names=list(old.code_hashes())+['src/confik/task_balance_gn.py','src/confik/task_balance_reference.py',
        'scripts/run_task_balance_development.py','configs/task_balance_development.yaml']
    return {n:sha(ROOT/n) for n in names}


def factory(method,robot,cfg,trace=False):
    if not method.startswith('task_balance'):return old.factory(method,robot,cfg,trace=trace)
    _,kin,v,urdf=context(robot,cfg)
    solver=TaskBalanceGN(kin,v,urdf,posture_weight=float(method.endswith('k1')),trace=trace)
    assert solver.settings.max_dual_updates==cfg['task_balance']['max_dual_updates']==16
    q=(kin.limits.lower+kin.limits.upper)/2
    target=kin.forward(q+.25*(kin.limits.velocity*.02+v.config.velocity_tolerance))
    for _ in range(3):solver.solve(target.position,target.rotation,q,.02)
    solver.metadata=dict(backend=old.old.configure_backend(),settings=solver.settings.__dict__,
        timing='Complete solve including input conversion, bounds, residuals/Jacobians, dual and box QPs, true-FK checks and original verifier; no serialization.',
        warmup='3 common nonstationary midpoint calls plus one runner stationary startup',
        theta0=.5,gap_atol=1e-9,gap_rtol=1e-7,
        task_feasible_early='Actual full-step accepted command, not convex optimality; gap is null.',
        deadline='One absolute outer 20 ms clock shared by outer/dual/backtracking; no preemption; late results retained.')
    return solver,kin,v


def first_problem(item,solver,kin,v,kappa):
    previous=np.asarray(item['previous_q']);target=Pose(np.array(item['target_position']),np.array(item['target_rotation']))
    S,lower,upper=dynamic_interval(previous,item['dt'],kin.limits,v.config.velocity_tolerance)
    q=np.clip(previous,lower,upper)
    e,J,_=residual_linearization(solver.native,target,q,solver.scale)
    return dict(e=e.tolist(),G=(J*S).tolist(),lo=np.maximum((lower-q)/S,-1).tolist(),
        hi=np.minimum((upper-q)/S,1).tolist(),damping=.01,
        posture=((q-(kin.limits.lower+kin.limits.upper)/2)/(kin.limits.upper-kin.limits.lower)).tolist(),
        posture_scale=(S/(kin.limits.upper-kin.limits.lower)).tolist(),posture_weight=kappa,
        q=q.tolist(),step_scale=S.tolist(),frame_lower=lower.tolist(),frame_upper=upper.tolist(),
        previous_q=previous.tolist(),target_position=target.position.tolist(),target_rotation=target.rotation.tolist(),dt=item['dt'])


def prepare(cfg):
    folder=OUT/'protocol';folder.mkdir(exist_ok=False)
    source=old.OUT/'protocol/failure_inputs.json';inputs=json.loads(source.read_text());assert len(inputs)==52
    write_json(folder/'failure_inputs.json',inputs)
    matrices=[];inventory=[];sources={str(source.relative_to(ROOT)):sha(source)}
    for robot in cfg['robots']:
        solver,kin,v=factory('task_balance_k0',robot,cfg)
        trajectories=old.old.items(cfg,robot)
        for family in sorted({x['family'] for x in trajectories}):
            selected=sorted([x for x in trajectories if x['family']==family],key=lambda x:x['site_id'])[:2]
            for t in selected:
                for kappa in (0,1):
                    path=old.OUT/f'development_{robot}/runs/{robot}_{t["site_id"]}_single_gn_k{kappa}_r0.jsonl.gz'
                    sources[str(path.relative_to(ROOT))]=sha(path);eligible=[]
                    for r in read_rows(path):
                        problem=first_problem(r,solver,kin,v,kappa)
                        e=np.array(problem['e'])
                        if max(np.linalg.norm(e[:3]),np.linalg.norm(e[3:]))>1:eligible.append((r,problem))
                    chosen=np.unique(np.linspace(0,len(eligible)-1,min(len(eligible),cfg['subproblems']['per_uid_kappa']),dtype=int)) if eligible else []
                    inventory.append(dict(robot=robot,uid=t['uid'],site_id=t['site_id'],family=family,kappa=kappa,
                        eligible_first_subproblems=len(eligible),selected=len(chosen)))
                    for j in chosen:
                        r,problem=eligible[j]
                        matrices.append(dict(problem_id=f'{robot}_{t["site_id"]}_k{kappa}_f{r["frame"]}',robot=robot,
                            stratum='regular_first_subproblems',uid=t['uid'],site_id=t['site_id'],family=family,frame=r['frame'],
                            source=str(path.relative_to(ROOT)),**problem))
        for item in inputs:
            if item['robot']!=robot:continue
            for kappa in (0,1):
                matrices.append(dict(problem_id=item['input_id']+f'_k{kappa}',robot=robot,stratum='historical_failure_inputs',
                    input_id=item['input_id'],aliases=item['aliases'],**first_problem(item,solver,kin,v,kappa)))
    write_json(folder/'subproblems.json',matrices);csv_write(folder/'subproblem_sampling.csv',inventory)
    write_json(folder/'protocol.json',dict(created=utc(),config=cfg,code_hashes=code_hashes(),source_hashes=sources,
        subproblems_sha256=sha(folder/'subproblems.json'),backend=old.old.configure_backend(),
        input_files={r:dict(path=cfg['development'][r+'_targets'],sha256=sha(ROOT/cfg['development'][r+'_targets'])) for r in cfg['robots']},
        identities=[{k:t[k] for k in ('robot','uid','site_id','family','dt','seed')} for r in cfg['robots'] for t in old.old.items(cfg,r)],
        statistics='40 UID/robot, three nested repeats averaged, family-stratified paired 4000 bootstrap; 95% descriptive unadjusted intervals, no equivalence inference.',
        local='Exact previously frozen 52 inputs including clip/OSQP source aliases; direct original k0/k1-source 16 also reported. No new inventory search.',
        subproblem_sampling='First two site IDs/family, old GN k0/k1 repeat0 actual previous states. Eight time-uniform eligible first outer problems/UID/kappa; all 52 failure inputs at both kappa, separate stratum. No solver-success or speed selection.',
        quality='Full-dual (early acceptance OFF, fixed cap16) vs cached Clarabel same epigraph; common box and objective gap checks. Both strong-convex minorant lower bounds are floating diagnostics, not formal certificates. Early acceptance separately reported.',
        controls='Frozen GN k0/k1, frozen excess k0, new balance k0/k1. Same-kappa attribution only; no IK fallback or other objective.',
        timing='CPU4, single BLAS/OpenMP thread; all late/failure frames kept; conversion and verification included; setup and cached reference call times separate.'))
    print('Frozen 52 inputs, 80 trajectories,',len(matrices),'real first-subproblems',flush=True)


def check_protocol():
    p=json.loads((OUT/'protocol/protocol.json').read_text());assert p['code_hashes']==code_hashes()
    for path,digest in p['source_hashes'].items():assert sha(ROOT/path)==digest,path
    for r in p['input_files'].values():assert sha(ROOT/r['path'])==r['sha256']
    assert sha(OUT/'protocol/subproblems.json')==p['subproblems_sha256']


def local(cfg):
    check_protocol();folder=OUT/'local';folder.mkdir(exist_ok=False);(folder/'diagnostics').mkdir()
    inputs=json.loads((OUT/'protocol/failure_inputs.json').read_text());records=[];witnesses=[]
    cache={(r,m):factory(m,r,cfg) for r in cfg['robots'] for m in cfg['methods']}
    for index,item in enumerate(inputs):
        p=np.array(item['previous_q']);query=IKQuery(Pose(np.array(item['target_position']),np.array(item['target_rotation'])),p,item['dt'])
        order=np.random.default_rng(cfg['order_seed']+index).permutation([(m,r) for m in cfg['methods'] for r in range(3)])
        for method,rep in order:
            solver,kin,v=cache[(item['robot'],method)]
            result=solver.solve(query.target.position,query.target.rotation,p,item['dt'])
            verdict=v.check(np.array(result['q']),query);assert verdict.accepted==result['accepted']
            records.append(dict(input_id=item['input_id'],robot=item['robot'],repeat=int(rep),method=method,**{k:v for k,v in result.items() if k!='method'}))
            if verdict.accepted:witnesses.append(dict(input_id=item['input_id'],robot=item['robot'],method=method,repeat=int(rep),
                previous_q=p.tolist(),target_position=item['target_position'],target_rotation=item['target_rotation'],dt=item['dt'],q=result['q'],
                position_error=verdict.position_error,orientation_error=verdict.orientation_error,velocity_utilization=result['velocity_utilization']))
        diagnostics={}
        for method in ('task_balance_k0','task_balance_k1'):
            solver,_,_=factory(method,item['robot'],cfg,trace=True)
            diagnostics[method]=solver.solve(query.target.position,query.target.rotation,p,item['dt'])
        write_json(folder/'diagnostics'/f'{item["input_id"]}.json',dict(input=item,results=diagnostics,timing_use='Separate diagnostic calls, not performance table'))
        if index%10==0:print('same input',index+1,'/52',flush=True)
    write_json(folder/'same_input_results.json',records);csv_write(folder/'same_input_results.csv',records)
    write_json(folder/'legal_witnesses.json',witnesses)
    write_json(folder/'manifest.json',dict(code_hashes=code_hashes(),calls=len(records),witnesses=len(witnesses),
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))
    for robot in cfg['robots']:
        for method in cfg['methods']:
            rr=[r for r in records if r['robot']==robot and r['method']==method]
            print(robot,method,sum(r['accepted'] for r in rr),'/',len(rr),flush=True)


def subproblems(cfg):
    check_protocol();folder=OUT/'subproblems';folder.mkdir(exist_ok=False)
    problems=json.loads((OUT/'protocol/subproblems.json').read_text());records=[];init=[]
    refs={n:EpigraphReference(n) for n in (6,7)}
    init=[dict(n=n,structure_initialization_ns=r.initialization_ns) for n,r in refs.items()]
    natives={robot:factory('task_balance_k0',robot,cfg) for robot in cfg['robots']}
    for index,p in enumerate(problems):
        keys=('e','G','lo','hi','damping','posture','posture_scale','posture_weight')
        raw={k:p[k] for k in keys};robot=p['robot'];solver,kin,v=natives[robot]
        query=IKQuery(Pose(np.array(p['target_position']),np.array(p['target_rotation'])),np.array(p['previous_q']),p['dt'])
        order=np.random.default_rng(cfg['order_seed']+1000+index).permutation([(m,r) for m in ('scalar_full','clarabel_epigraph','scalar_early') for r in range(cfg['subproblems']['repeat'])])
        for method,rep in order:
            start=perf_counter_ns();a={k:np.array(raw[k],float) if isinstance(raw[k],list) else raw[k] for k in keys}
            conversion_ns=perf_counter_ns()-start;native_start=perf_counter_ns()
            if method=='clarabel_epigraph':
                d,info=refs[len(p['q'])].solve(**a);info['total_ns']=perf_counter_ns()-start
            else:
                checks=0
                def acceptable(d):
                    nonlocal checks
                    checks+=1;q=np.clip(np.array(p['q'])+np.array(p['step_scale'])*d,p['frame_lower'],p['frame_upper'])
                    return bool(v.check(q,query).accepted)
                d,info=minimax_step(**a,max_dual_updates=16,acceptable=acceptable if method=='scalar_early' else None)
                kernel_ns=perf_counter_ns()-native_start
                info.update(quality(d,**a,theta=info['theta']))
                info.update(total_ns=perf_counter_ns()-start,solve_ns=kernel_ns,conversion_ns=conversion_ns,
                    actual_verifier_calls=checks,updated=True)
            records.append(dict(problem_id=p['problem_id'],robot=robot,stratum=p['stratum'],kappa=p['posture_weight'],
                method=method,repeat=int(rep),direction=d.tolist(),**info))
        if index%80==0:print('fixed real subproblem',index+1,'/',len(problems),flush=True)
    write_json(folder/'raw_results.json',records);write_json(folder/'structure_initialization.json',init)
    write_json(folder/'manifest.json',dict(created=utc(),code_hashes=code_hashes(),problems=len(problems),calls=len(records),
        problem_file_sha256=sha(OUT/'protocol/subproblems.json'),files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))


def trajectories(cfg,robot):
    check_protocol();runner.factory=factory;runner.code_hashes=code_hashes
    runner.run(OUT/f'development_{robot}',old.old.items(cfg,robot),cfg['methods'],dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=3)


def verify(cfg):
    folder=OUT/'package_verification';folder.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((PACKAGE/'manifest.json').read_text())
    for name,digest in hashes.items():assert sha(PACKAGE/name)==digest,name
    data=json.loads((PACKAGE/'results/prototype_checks.json').read_text())
    i=data['selected_input']['input'];q=np.array(data['robot']['q'])
    with (ROOT/i['source']).open() as f:rows=list(csv.DictReader(f))
    matches=[r for r in rows if r['robot']==i['robot'] and r['uid']==i['uid'] and int(r['frame'])==i['frame'] and r['method']==i['source_method']]
    assert any(float(r['dt'])==i['dt'] and all(np.array_equal(ast.literal_eval(r[k]),i[k]) for k in
        ('previous_q','target_position','target_rotation')) for r in matches)
    _,kin,v,_=context('panda',cfg)
    query=IKQuery(Pose(np.array(i['target_position']),np.array(i['target_rotation'])),np.array(i['previous_q']),i['dt'])
    verdict=v.check(q,query)
    result=dict(input=i,q=q.tolist(),solver_calls=0,accepted=bool(verdict.accepted),
        finite=bool(verdict.finite_ok),joint_limit_ok=bool(verdict.joint_limit_ok),velocity_ok=bool(verdict.velocity_ok),
        position_m=verdict.position_error,orientation_rad=verdict.orientation_error,
        orientation_deg=float(np.rad2deg(verdict.orientation_error)),reasons=list(verdict.reasons),
        step=(q-query.previous_q).tolist(),velocity_utilization=float(np.max(abs(q-query.previous_q)/(kin.limits.velocity*i['dt']+v.config.velocity_tolerance))),
        archive_sha256=sha(ROOT/'task_balance_gn_prototype.zip'),package_hashes=hashes,source_sha256=sha(ROOT/i['source']))
    write_json(folder/'original_verifier.json',result)
    print(json.dumps(clean(result),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','prepare','local','subproblems','trajectories','report'])
    p.add_argument('--robot',choices=['panda','ur5e']);args=p.parse_args()
    os.sched_setaffinity(0,{4})
    cfg=yaml.safe_load(CONFIG.read_text())
    if args.stage=='verify':verify(cfg)
    elif args.stage=='prepare':prepare(cfg)
    elif args.stage=='local':local(cfg)
    elif args.stage=='subproblems':subproblems(cfg)
    elif args.stage=='trajectories':
        if args.robot is None:p.error('--robot is required')
        trajectories(cfg,args.robot)
    elif args.stage=='report':
        from confik.task_balance_reporting import report
        report(OUT,cfg)


if __name__=='__main__':main()
