#!/usr/bin/env python3
"""Thin experiment entry: unchanged feedback runner and trajectory statistics."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns
import numpy as np
import yaml

from confik.correction_reserve import study as runner
from confik.correction_reserve.study import ROOT,context,sha,write_json,clean,utc,read_rows
from confik.correction_reserve.reporting import csv_write,LABELS

spec=importlib.util.spec_from_file_location('supplied_entry',ROOT/'scripts/run_single_solver_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert old.configure_backend()['engine']=='numpy_supplied_fallback'
from confik.bounded_gn import box_qp
from confik.bounded_gn_adapter import SingleBoundedGN
from confik.single_solver_evidence import bind_qp,clipped_qp,CachedOSQP,quality

OUT=ROOT/'outputs/single_solver_evidence'
LABELS.update(single_gn_k1='Bounded GN kappa=1',single_gn_k0='Bounded GN kappa=0',
              single_gn_clip='GN kappa=1, clipped step',single_gn_osqp='GN kappa=1, OSQP')
FIXED={'src/confik/bounded_gn.py':'efb81e50a998505511660bdf22733858789d90409474f703fff4c06dd87b6a7a',
       'src/confik/bounded_gn_adapter.py':'ab6212f2f8f70dee4c0eef639f8bb768739376796341df6482de2b915819651a'}


def code_hashes():
    hashes=old.code_hashes()
    for p in ['scripts/run_single_solver_evidence.py','src/confik/single_solver_evidence.py',
              'configs/single_solver_evidence.yaml','src/confik/revision_compute_allocation/data.py']:
        hashes[p]=sha(ROOT/p)
    return hashes


def fixed_sources():
    for p,h in FIXED.items():assert sha(ROOT/p)==h,p


def eps_setting():
    return json.loads((OUT/'qp_benchmark/selection.json').read_text())['eps_abs']


def factory(method,robot,cfg):
    source,kin,v,urdf=context(robot,cfg)
    if method.startswith('single_gn'):
        solver=SingleBoundedGN(kin,v,urdf,posture_weight=0. if method=='single_gn_k0' else 1.)
        if method=='single_gn_clip':bind_qp(solver.engine,clipped_qp)
        if method=='single_gn_osqp':
            backend=CachedOSQP(kin.nq,eps_setting());bind_qp(solver.engine,backend)
            solver.reset=backend.reset
            solver.metadata=dict(qp_settings=backend.settings,qp_initialization_ns=backend.initialization_ns,
                qp_iteration_unit='OSQP ADMM iterations; not active-set iterations',
                warm_start='previous local QP x/y; reset to zero at trajectory boundary',
                roundoff_projection='raw d projected only when violation <=1e-8; otherwise zero update; task verifier unchanged')
        else:solver.metadata=dict(qp_iteration_unit='linear solves' if method=='single_gn_clip' else 'active-set updates')
        solver.metadata.update(settings=solver.engine.settings.__dict__,backend=old.configure_backend())
    elif method.startswith('trac'):
        from confik.task_contract_alignment.outcomes import ContractSolver
        solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    elif method=='pink_qp':
        from confik.correction_reserve.pink_adapter import PinkAdapter
        solver=PinkAdapter(kin,v,source,urdf)
    else:raise ValueError(method)
    # Same three nonstationary synthetic warmups for every method, outside time.
    q=(kin.limits.lower+kin.limits.upper)/2
    target=kin.forward(q+.25*(kin.limits.velocity*.02+v.config.velocity_tolerance))
    for _ in range(3):solver.solve(target.position,target.rotation,q,.02)
    return solver,kin,v


def development(cfg,robot):
    rows=old.items(cfg,robot)
    return [r for f in cfg['fresh']['families'] for r in [x for x in rows if x['family']==f][:2]]


def prepare(cfg):
    fixed_sources();folder=OUT/'protocol';folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'development_protocol.json',dict(created=utc(),config=cfg,frozen_sources=FIXED,
        backend=old.configure_backend(),scope='Only per-instance local QP callbacks; original core, adapter, generator, verifier unchanged.',
        qp_quality='Independent raw projected KKT <=1e-5, normalized <=1e-8, box <=1e-8; scaled objective difference <=1e-8 on quality-qualified active-set references. Start OSQP eps_abs=1e-8, eps_rel=0, polishing, cap=20000; tighten only for quality. Failed quality samples retained.',
        qp_timing='Five fixed repeated passes, full conversion/reset/update/solve/result check timed; one-time cached setup separately. Warm starts reset at UID changes. No matrix capture in production timing.',
        primary_comparisons='M1 against all six controls; trajectory is unit; three repeats averaged within UID; 4000 family-stratified paired bootstrap, descriptive unadjusted 95% intervals.',
        cpu=cfg['cpu'],development_inputs=[{k:r[k] for k in ('robot','uid','family','site_id','seed')} for robot in cfg['robots'] for r in development(cfg,robot)],
        input_hashes={r:sha(ROOT/cfg['development'][r+'_targets']) for r in cfg['robots']},
        freshness='Unchanged generator, 160/robot, 150frames; seeds increment on identity collision only; input verification and identities sealed before fresh outcomes.'))


def collect(cfg):
    folder=OUT/'qp_benchmark';folder.mkdir(exist_ok=False)
    for robot in cfg['robots']:
        matrices=[];meta=[];frames=[]
        solver,kin,v=factory('single_gn_k1',robot,cfg)
        current={}
        def capture(H,g,lo,hi):
            local=sys._getframe(1).f_locals
            d,n=box_qp(H,g,lo,hi)
            q=local['q'];step=local['step'];previous=local['previous']
            raw=np.linalg.solve(H,-g);clip=np.clip(raw,lo,hi)
            constrained=(raw<lo)|(raw>hi)
            active=(np.abs(d-lo)<1e-8)|(np.abs(d-hi)<1e-8)
            proposal=q+step*d
            physical=active & ((np.abs(proposal-kin.limits.lower)<1e-8)|(np.abs(proposal-kin.limits.upper)<1e-8))
            rate=active & (np.abs(np.abs(proposal-previous)-step)<1e-8)
            matrices.append((H.copy(),g.copy(),lo.copy(),hi.copy()))
            meta.append(dict(current,qp_index=len(meta),outer_iteration=local['iteration'],
                dimension=len(g),damping=local['lam'],active_bounds=int(active.sum()),
                physical_active=int(physical.sum()),frame_rate_active=int(rate.sum()),
                constrained_joints=int(constrained.sum()),
                free_joint_redistribution=float(np.linalg.norm((d-clip)[~constrained])),
                original_updates=n))
            return d,n
        bind_qp(solver.engine,capture)
        for item in development(cfg,robot):
            previous=np.asarray(item['initial_q']);solver.reset(previous)
            for frame,(p,r) in enumerate(zip(item['target_position'],item['target_rotation'])):
                current.update(robot=robot,uid=item['uid'],family=item['family'],site_id=item['site_id'],frame=frame)
                before=len(meta);row=solver.solve(p,r,previous,item['dt'])
                frames.append(dict(current,qp_calls=len(meta)-before,iterations=row['iterations'],evaluations=row['evaluations'],internal_status=row['internal_status']))
                if row['accepted']:previous=np.array(row['q'])
        solver.close()
        rng=np.random.default_rng(cfg['qp']['sample_seed']);indices=[]
        for uid in dict.fromkeys(r['uid'] for r in meta):
            ids=[i for i,r in enumerate(meta) if r['uid']==uid]
            indices.extend(rng.choice(ids,min(len(ids),cfg['qp']['maximum_per_uid']),replace=False).tolist())
        indices=sorted(indices);assert len(indices)<=1000
        with (folder/f'{robot}_qp_matrices.npz').open('xb') as f:
            np.savez_compressed(f,H=np.array([matrices[i][0] for i in indices]),
                g=np.array([matrices[i][1] for i in indices]),lo=np.array([matrices[i][2] for i in indices]),hi=np.array([matrices[i][3] for i in indices]))
        write_json(folder/f'{robot}_qp_metadata.json',[meta[i] for i in indices])
        write_json(folder/f'{robot}_capture_sources.json',dict(instrumented_not_online_timing=True,
            collected=len(meta),selected=len(indices),all_metadata=meta,frame_counters=frames,
            source_distribution=dict(Counter(r['uid'] for r in meta)),sample_distribution=dict(Counter(meta[i]['uid'] for i in indices))))
        print('QP capture',robot,len(meta),'sample',len(indices),flush=True)


def qualified(q,cfg):
    c=cfg['qp']
    return q['box_violation']<=c['box_tolerance'] and q['projected_kkt']<=c['raw_kkt_tolerance'] and q['normalized_kkt']<=c['normalized_kkt_tolerance']


def micro(cfg):
    folder=OUT/'qp_benchmark';selection=None
    for eps in cfg['qp']['eps_abs_candidates']:
        rows=[];setups=[];bad=0
        for robot in cfg['robots']:
            bank=np.load(folder/f'{robot}_qp_matrices.npz');meta=json.loads((folder/f'{robot}_qp_metadata.json').read_text())
            n=bank['g'].shape[1]
            solvers={m:CachedOSQP(n,eps) for m in ('osqp_reset','osqp_warm')}
            setups.extend(dict(robot=robot,method=m,initialization_ns=s.initialization_ns,settings=s.settings) for m,s in solvers.items())
            for repeat in range(cfg['qp']['repeats']):
                order=np.random.default_rng(cfg['order_seed']+repeat).permutation(['active_numpy','osqp_reset','osqp_warm','clip'])
                for method in order:
                    last_uid=None
                    for i,info in enumerate(meta):
                        H,g,lo,hi=(bank[k][i] for k in ('H','g','lo','hi'))
                        # Independent reference/checks are deliberately outside micro timing.
                        reference,ni=box_qp(H,g,lo,hi);ref=quality(H,g,lo,hi,reference)
                        start=perf_counter_ns();reset_end=start
                        if method.startswith('osqp'):
                            s=solvers[method]
                            if method=='osqp_reset' or info['uid']!=last_uid:s.reset()
                            reset_end=perf_counter_ns();x,it=s(H,g,lo,hi);extra=dict(s.last);raw=extra.pop('raw_x')
                            extra.pop('native_iterations')
                        else:
                            solve_start=perf_counter_ns()
                            x,it=(box_qp if method=='active_numpy' else clipped_qp)(H,g,lo,hi)
                            solve_end=perf_counter_ns();raw=x
                            assert np.isfinite(x).all() and np.all(x>=lo) and np.all(x<=hi)
                            extra=dict(conversion_ns=0,update_ns=0,solve_ns=solve_end-solve_start,
                                check_ns=perf_counter_ns()-solve_end,status='returned',status_val=None,usable=True,projection_delta=0.)
                        total=perf_counter_ns()-start;last_uid=info['uid']
                        check=quality(H,g,lo,hi,x);rawcheck=quality(H,g,lo,hi,raw)
                        gap=check['objective']-ref['objective'];scaled=abs(gap)/max(1.,abs(ref['objective']))
                        ok=qualified(check,cfg) and qualified(rawcheck,cfg)
                        if qualified(ref,cfg):ok=ok and scaled<=cfg['qp']['scaled_objective_tolerance']
                        if method.startswith('osqp') and not ok:bad+=1
                        rows.append(dict(info,method=str(method),repeat=repeat,eps_abs=eps,
                            total_latency_ns=total,reset_ns=reset_end-start,native_iterations=int(it),
                            quality_pass=ok,reference_quality_pass=qualified(ref,cfg),
                            objective_gap=gap,scaled_objective_gap=scaled,raw_projected_kkt=rawcheck['projected_kkt'],
                            raw_normalized_kkt=rawcheck['normalized_kkt'],**check,**extra))
            bank.close()
        tag=f'{eps:.0e}';csv_write(folder/f'micro_{tag}.csv',rows)
        write_json(folder/f'quality_{tag}.json',dict(eps_abs=eps,bad_osqp_calls=bad,setups=setups,
            method_counts={m:dict(calls=sum(r['method']==m for r in rows),bad=sum(r['method']==m and not r['quality_pass'] for r in rows)) for m in set(r['method'] for r in rows)}))
        print('QP quality',eps,'bad OSQP',bad,flush=True)
        if bad==0:
            selection=dict(eps_abs=eps,source=f'micro_{tag}.csv',reason='First accuracy setting passing all development QP quality checks, not trajectory outcomes.',settings=solvers['osqp_warm'].settings)
            break
    if selection is None:raise RuntimeError('OSQP quality unresolved; do not run independent outcomes')
    write_json(folder/'selection.json',selection)


def integration(cfg):
    runner.factory=factory;runner.code_hashes=code_hashes
    data=[r for robot in cfg['robots'] for r in development(cfg,robot)]
    runner.run(OUT/'integration',data,cfg['methods'][:4],dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=1)


def inputs(cfg):
    from confik.continuation_mechanism.independent_selection import metadata_keys
    from confik.revision_compute_allocation.data import trajectories
    folder=OUT/'protocol';prior=dict(uids=set(),seeds=set(),hashes=set());sources={}
    for p in sorted((ROOT/'outputs').rglob('*identit*.json')):
        if OUT in p.parents:continue
        metadata_keys(json.loads(p.read_text()),prior);sources[str(p.relative_to(ROOT))]=sha(p)
    # Original development online files may carry identities omitted by old manifests.
    for robot in cfg['robots']:metadata_keys(old.items(cfg,robot),prior)
    collisions=[];identities=[];online=[]
    for robot in cfg['robots']:
        source,kin,v,_=context(robot,cfg);refs=[];nextseed=cfg['fresh']['seeds'][robot]
        for i,family in enumerate(f for f in cfg['fresh']['families'] for _ in range(cfg['fresh']['per_family'])):
            while True:
                seed=nextseed;nextseed+=1
                if seed in prior['seeds']:
                    collisions.append(dict(robot=robot,seed=seed,kind='seed'));continue
                gen=dict(_source=source,trajectory_seeds={robot:seed},trajectories_per_family=1,
                    trajectory_frames=150,dt=.02,trajectory_families=[family])
                ds,ids=trajectories(kin,gen,robot);identity=ids[0]
                if identity['uid'] in prior['uids'] or set(identity['query_hashes'])&prior['hashes']:
                    collisions.append(dict(robot=robot,seed=seed,kind='uid_or_query'));continue
                break
            identity.update(robot=robot,site_id=f'trajectory_{i:03d}')
            metadata_keys(identity,prior);identities.append(identity)
            refs.append(np.vstack([ds.previous_q[0],ds.reference_q]))
            online.append(dict(robot=robot,site_id=identity['site_id'],uid=identity['uid'],family=family,seed=seed,
                initial_q=ds.previous_q[0].tolist(),target_position=ds.target_position.tolist(),
                target_rotation=ds.target_rotation.tolist(),dt=.02))
            if i%20==0:print('reference verified',robot,i+1,flush=True)
        with (folder/f'{robot}_reference_witnesses.npz').open('xb') as f:np.savez_compressed(f,q=np.array(refs))
    write_json(folder/'identities.json',identities);write_json(folder/'online_inputs.json',online)
    write_json(folder/'execution_seal.json',dict(created=utc(),code_hashes=code_hashes(),
        config=cfg,qp_selection=json.loads((OUT/'qp_benchmark/selection.json').read_text()),
        old_identity_sources=sources,collisions=collisions,reference_frames_verified=48000,
        fresh_solver_calls=0,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))


def evaluate(cfg,robot,resume):
    fixed_sources();seal=json.loads((OUT/'protocol/execution_seal.json').read_text())
    assert seal['code_hashes']==code_hashes()
    for name,digest in seal['files'].items():assert sha(OUT/'protocol'/name)==digest
    assert seal['qp_selection']==json.loads((OUT/'qp_benchmark/selection.json').read_text())
    data=[r for r in json.loads((OUT/'protocol/online_inputs.json').read_text()) if r['robot']==robot]
    assert len(data)==160
    runner.factory=factory;runner.code_hashes=code_hashes
    runner.run(OUT/f'trajectories_{robot}',data,cfg['methods'],dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=3,resume=resume)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect','micro','integration','inputs','evaluate','report','figures'])
    p.add_argument('--robot',choices=['panda','ur5e']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/single_solver_evidence.yaml').read_text());fixed_sources()
    os.sched_setaffinity(0,{cfg['cpu']})
    if args.action=='evaluate':
        if not args.robot:p.error('--robot required')
        evaluate(cfg,args.robot,args.resume)
    elif args.action in ('report','figures'):
        from confik.single_solver_evidence_reporting import report,figures
        (report if args.action=='report' else figures)(OUT,cfg)
    else:globals()[args.action](cfg)


if __name__=='__main__':main()
