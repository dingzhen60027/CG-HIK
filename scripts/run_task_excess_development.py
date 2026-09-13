#!/usr/bin/env python3
"""One development entry for the supplied task-excess Bounded-GN mechanism."""
import argparse
import ast
import csv
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/single_solver_evidence/task_excess_development'
PACKAGE=OUT/'task_package/task_excess_gn'
spec=importlib.util.spec_from_file_location('existing_single_gn',ROOT/'scripts/run_single_solver_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.configure_backend()
from confik.correction_reserve import study as runner
from confik.correction_reserve.study import context,write_json,sha,clean,read_rows,utc
from confik.types import IKQuery,Pose
from confik.correction_reserve.reporting import csv_write
from confik.task_excess_gn import TaskExcessGN,Settings,projected_task_model,dynamic_interval
from confik.correction_reserve.geometry import residual_linearization
from confik.bounded_gn_adapter import SingleBoundedGN

CONFIG=ROOT/'configs/task_excess_development.yaml'
FIXED={'src/confik/bounded_gn.py':'efb81e50a998505511660bdf22733858789d90409474f703fff4c06dd87b6a7a',
       'src/confik/bounded_gn_adapter.py':'ab6212f2f8f70dee4c0eef639f8bb768739376796341df6482de2b915819651a'}


def code_hashes():
    for name,digest in FIXED.items():assert sha(ROOT/name)==digest,name
    paths=list(old.code_hashes())+['src/confik/task_excess_gn.py',
        'scripts/run_task_excess_development.py','configs/task_excess_development.yaml']
    return {n:sha(ROOT/n) for n in paths}


def factory(method,robot,cfg,trace=False):
    source,kin,v,urdf=context(robot,cfg)
    if method in ('single_gn_k0','single_gn_k1'):
        solver=SingleBoundedGN(kin,v,urdf,posture_weight=float(method.endswith('k1')))
    elif method.startswith('task_excess') or method=='point_loop_control':
        mode={'task_excess_gn':'task','task_excess_lbfgsb':'lbfgsb',
              'task_excess_isotropic':'isotropic','point_loop_control':'point'}[method]
        solver=TaskExcessGN(kin,v,urdf,mode,Settings(**cfg['task_excess']),trace=trace)
    elif method=='trac_task_5ms':
        from confik.task_contract_alignment.outcomes import ContractSolver
        solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    elif method=='pink_qp':
        from confik.correction_reserve.pink_adapter import PinkAdapter
        solver=PinkAdapter(kin,v,source,urdf)
    else:raise ValueError(method)
    # Same warmup inputs/count for every method; outside all measured calls.
    q=(kin.limits.lower+kin.limits.upper)/2
    target=kin.forward(q+.25*(kin.limits.velocity*.02+v.config.velocity_tolerance))
    for _ in range(3):solver.solve(target.position,target.rotation,q,.02)
    solver.metadata=dict(backend=old.configure_backend(),fixed_settings=cfg['task_excess'] if method.startswith('task_excess') else None,
        standard_reference_options=cfg['lbfgsb'] if method=='task_excess_lbfgsb' else None,
        warmup='3 common nonstationary midpoint calls, plus runner stationary startup',
        timing='Complete adapter outer time, including conversion, bounds, all numerical work, task checks and final original verifier; no trace during trajectory timing.')
    return solver,kin,v


def failure_inventory():
    """Exact numerical input hashes; source aliases are not independent units."""
    inputs={};sources={};source_rows=[]
    def add(r,source):
        if not str(r.get('method','')).startswith('single_gn'):return
        raw={k:r[k] for k in ('robot','previous_q','target_position','target_rotation','dt')}
        for key in ('previous_q','target_position','target_rotation'):
            if isinstance(raw[key],str):raw[key]=ast.literal_eval(raw[key])
            raw[key]=np.asarray(raw[key],float).tolist()
        raw['dt']=float(raw['dt'])
        digest=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        alias={k:r.get(k) for k in ('uid','site_id','family','method','repeat','frame','failure_kind','q','position_error','orientation_error','internal_status')}
        alias['source']=source
        for key in ('q',):
            if isinstance(alias[key],str):alias[key]=ast.literal_eval(alias[key])
        if digest not in inputs:inputs[digest]=dict(input_id=digest,**raw,aliases=[])
        inputs[digest]['aliases'].append(alias)
        source_rows.append(dict(input_id=digest,**alias))
    # Exhaust all existing first-failure CSV inventories, not only the selected
    # successful prototype input. This run's outputs are never ingested.
    for path in sorted((ROOT/'outputs').rglob('first_failure_inputs.csv')):
        if OUT in path.parents:continue
        rows=list(csv.DictReader(path.open()))
        gn=[r for r in rows if str(r.get('method','')).startswith('single_gn')]
        if gn:
            relative=str(path.relative_to(ROOT));sources[relative]=sha(path)
            for r in gn:add(r,relative)
    # Prior lightweight integration failures may not have a CSV inventory.
    folder=ROOT/'outputs/single_solver_evidence/integration'
    for path in sorted((folder/'runs').glob('*.summary.json')):
        s=json.loads(path.read_text())
        if s['method'].startswith('single_gn') and s['first_failure_frame'] is not None:
            raw=folder/s['raw_file'];relative=str(raw.relative_to(ROOT));sources[relative]=sha(raw)
            add(read_rows(raw)[s['first_failure_frame']],relative)
    return [inputs[k] for k in sorted(inputs)],sources,source_rows


def prepare(cfg):
    folder=OUT/'protocol';folder.mkdir(exist_ok=False)
    assert old.configure_backend()['engine']=='numpy_supplied_fallback'
    inputs,sources,aliases=failure_inventory()
    write_json(folder/'failure_inputs.json',inputs);csv_write(folder/'failure_source_aliases.csv',aliases)
    write_json(folder/'protocol.json',dict(created=utc(),config=cfg,code_hashes=code_hashes(),
        original_sources=FIXED,backend=old.configure_backend(),source_hashes=sources,
        exact_unique_failure_inputs=len(inputs),source_failure_rows=len(aliases),
        input_files={robot:dict(path=cfg['development'][robot+'_targets'],sha256=sha(ROOT/cfg['development'][robot+'_targets'])) for robot in cfg['robots']},
        trajectory_identities=[{k:r[k] for k in ('robot','uid','site_id','family','dt','seed')} for robot in cfg['robots'] for r in old.items(cfg,robot)],
        input_policy='All observed GN failure inputs, exact target/previous/dt deduplication; no failed endpoint seed. All 80 existing development trajectories; no new identities.',
        loop_differences='Supplied task prototype uses true Armijo merit, fixed 1e-6 interior pose radius, and original verifier at iterates/trials. Frozen GN has its original point-merit backtracking and final verification. The same-loop point control isolates this non-objective difference in same-input diagnostics.',
        generic_optimizer='L-BFGS-B same Phi analytic gradient in normalized joint variables; original verifier checked every function evaluation; 20 ms soft outer deadline, maxiter30/maxfun240/maxls20/gtol1e-8/ftol0. ftol disabled to avoid squared tiny positive excess relative-f termination; no continued solve after admissibility.',
        statistics='Trajectory UID is the unit, 3 nested repeats averaged before family-stratified paired bootstrap (4000); 95% unadjusted descriptive intervals; no equivalence claim from intervals including zero.',
        timing='CPU4, one BLAS/OpenMP thread, common warmup; complete outer invocation including verification; all late and failed frames retained. Detailed local traces are separate calls, excluded from primary timing.',
        selection='No parameter search or result gate. Fixed supplied radius, damping, QP and iteration count; isotropic control only on existing failure inputs.'))
    print('Fixed',len(inputs),'unique inputs from',len(aliases),'failure aliases; 80 existing trajectory UIDs',flush=True)


def check_protocol():
    p=json.loads((OUT/'protocol/protocol.json').read_text())
    assert p['code_hashes']==code_hashes()
    for row in p['input_files'].values():assert sha(ROOT/row['path'])==row['sha256']
    for path,digest in p['source_hashes'].items():assert sha(ROOT/path)==digest
    return p


def local(cfg):
    check_protocol();folder=OUT/'local';folder.mkdir(exist_ok=False)
    (folder/'diagnostics').mkdir()
    inputs=json.loads((OUT/'protocol/failure_inputs.json').read_text())
    methods=cfg['local']['methods'];records=[];witnesses=[];diagnostics=[]
    cache={(r,m):factory(m,r,cfg) for r in cfg['robots'] for m in methods}
    for index,item in enumerate(inputs):
        robot=item['robot'];p=np.array(item['previous_q']);dt=item['dt']
        query=IKQuery(Pose(np.array(item['target_position']),np.array(item['target_rotation'])),p,dt)
        order=np.random.default_rng(cfg['order_seed']+index).permutation([(m,r) for m in methods for r in range(3)])
        for method,rep in order:
            solver,kin,v=cache[(robot,method)]
            result=solver.solve(query.target.position,query.target.rotation,p,dt)
            verdict=v.check(np.array(result['q']),query)
            assert verdict.accepted==result['accepted']
            row=dict(input_id=item['input_id'],robot=robot,method=method,repeat=int(rep),**{k:v for k,v in result.items() if k!='method'})
            records.append(row)
            if verdict.accepted:
                witnesses.append(dict(input_id=item['input_id'],robot=robot,method=method,repeat=int(rep),
                    previous_q=p.tolist(),target_position=item['target_position'],target_rotation=item['target_rotation'],dt=dt,
                    q=result['q'],position_error=verdict.position_error,orientation_error=verdict.orientation_error,
                    velocity_utilization=result['velocity_utilization']))
        # Trace is a separate same-input diagnostic, never timed as main calls.
        traced={}
        for method in ('task_excess_gn','task_excess_isotropic','point_loop_control'):
            solver,kin,v=factory(method,robot,cfg,trace=True)
            result=solver.solve(query.target.position,query.target.rotation,p,dt)
            traced[method]=result
        write_json(folder/'diagnostics'/f"{item['input_id']}.json",dict(input=item,results=traced,timing_use='diagnostic only'))
        # Diagnose each actual frozen k0 returned endpoint, not use it as seed.
        for row in [x for x in records if x['input_id']==item['input_id'] and x['method']=='single_gn_k0']:
            solver,kin,v=cache[(robot,'task_excess_gn')];q=np.array(row['q'])
            S,lo,hi=dynamic_interval(p,dt,kin.limits,v.config.velocity_tolerance)
            e,J,_=residual_linearization(solver.native,query.target,q,solver.scale)
            g=(J*S).T@e;y=(q-p)/S;yl=(lo-p)/S;yu=(hi-p)/S
            projected=g.copy();atlo=np.abs(y-yl)<=1e-7;athi=np.abs(y-yu)<=1e-7
            projected[atlo & (g>=0)]=0;projected[athi & (g<=0)]=0
            active=[]
            for j in range(kin.nq):
                if (atlo[j] and g[j]>1e-8) or (athi[j] and g[j]<-1e-8):
                    lower=bool(atlo[j]);physical=(lo[j]==kin.limits.lower[j]+1e-12) if lower else (hi[j]==kin.limits.upper[j]-1e-12)
                    active.append(dict(joint=j,side='lower' if lower else 'upper',kind='physical' if physical else 'frame_displacement',gradient=float(g[j])))
            pn,rn=np.linalg.norm(e[:3]),np.linalg.norm(e[3:])
            category='both_fail' if pn>1 and rn>1 else 'position_fail' if pn>1 else 'orientation_fail' if rn>1 else 'both_pose_pass'
            diagnostics.append(dict(input_id=item['input_id'],robot=robot,repeat=row['repeat'],
                accepted=row['accepted'],status=row['internal_status'],q=q.tolist(),e=e.tolist(),
                position_normalized=pn,orientation_normalized=rn,category=category,
                point_objective=.5*float(e@e),point_projected_gradient_inf=float(np.max(abs(projected))),
                point_projected_gradient_scaled=float(np.max(abs(projected))/(1+np.max(abs(g)))),
                actual_endpoint_kkt_active_constraints=active))
        if index%10==0:print('same input',index+1,'/',len(inputs),flush=True)
    csv_write(folder/'same_input_results.csv',records);write_json(folder/'same_input_results.json',records)
    write_json(folder/'legal_witnesses.json',witnesses);csv_write(folder/'point_endpoint_diagnostics.csv',diagnostics)
    write_json(folder/'manifest.json',dict(created=utc(),code_hashes=code_hashes(),inputs=len(inputs),
        measured_calls=len(records),legal_witnesses=len(witnesses),files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))
    for robot in cfg['robots']:
        for method in methods:
            rr=[r for r in records if r['robot']==robot and r['method']==method]
            print(robot,method,'accepted',sum(r['accepted'] for r in rr),'/',len(rr),flush=True)


def trajectories(cfg,robot):
    check_protocol();runner.factory=factory;runner.code_hashes=code_hashes
    runner.run(OUT/f'development_{robot}',old.items(cfg,robot),cfg['methods'],
               dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=cfg['repeats'])


def verify_package(cfg):
    folder=OUT/'package_verification';folder.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((PACKAGE/'manifest.json').read_text())
    for name,digest in hashes.items():assert sha(PACKAGE/name)==digest,name
    result=json.loads((PACKAGE/'selected_input_results.json').read_text());i=result['input']
    with (ROOT/i['source']).open() as f:
        rows=[r for r in csv.DictReader(f) if r['robot']==i['robot'] and r['uid']==i['uid'] and
              int(r['frame'])==i['frame'] and r['method']==i['source_method']]
    assert rows
    assert any(all(np.array_equal(ast.literal_eval(r[key]),i[key])
                   for key in ['previous_q','target_position','target_rotation']) for r in rows)
    assert all(float(r['dt'])==i['dt'] for r in rows)
    _,kin,v,_=context('panda',cfg)
    query=IKQuery(Pose(np.array(i['target_position']),np.array(i['target_rotation'])),np.array(i['previous_q']),i['dt'])
    records=[]
    for record in result['records']:
        q=np.array(record['q']);verdict=v.check(q,query)
        records.append(dict(method=record['method'],q=q.tolist(),accepted=bool(verdict.accepted),
            position_m=verdict.position_error,orientation_rad=verdict.orientation_error,
            orientation_deg=float(np.rad2deg(verdict.orientation_error)),
            finite=bool(verdict.finite_ok),joint_limit_ok=bool(verdict.joint_limit_ok),
            velocity_ok=bool(verdict.velocity_ok),reasons=list(verdict.reasons),
            velocity_utilization=float(np.max(abs(q-query.previous_q)/(kin.limits.velocity*i['dt']+v.config.velocity_tolerance)))))
    write_json(folder/'original_verifier.json',dict(input=i,solver_calls=0,package_hashes=hashes,
        archive_sha256=sha(ROOT/'task_excess_gn_prototype.zip'),source_sha256=sha(ROOT/i['source']),records=records))
    print(json.dumps(clean(records),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','prepare','local','trajectories','report'])
    p.add_argument('--robot',choices=['panda','ur5e']);args=p.parse_args()
    os.sched_setaffinity(0,{4})
    cfg=yaml.safe_load(CONFIG.read_text())
    if args.stage=='verify':verify_package(cfg)
    elif args.stage=='prepare':prepare(cfg)
    elif args.stage=='local':local(cfg)
    elif args.stage=='trajectories':
        if args.robot is None:p.error('--robot required')
        trajectories(cfg,args.robot)
    elif args.stage=='report':
        from confik.task_excess_reporting import report
        report(OUT,cfg)


if __name__=='__main__':main()
