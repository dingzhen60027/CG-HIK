from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import numpy as np
import torch
from ..config import load_config,load_robot,resolve_path
from ..data.datasets import QueryDataset
from ..latency_pilot_v3.benchmark import query_from_dataset
from ..solvers.verifier import SolutionVerifier,VerifierConfig
from . import data
from .benchmark import benchmark_points,benchmark_trajectories,measured_call
from .common import csv_write,digest,json_write,load_npz
from .decomposition import development_labels
from .policies import build_internal,select_geometry,INTERNAL_METHODS
from .trac_ik import TracIK


def trac_methods(root,cfg,robot,kin,budgets=None):
    urdf=resolve_path(cfg['_source'],cfg['_source']['robots'][robot]['urdf'])
    methods={}
    for budget in budgets or cfg['trac_ik']['budgets_ms']:
        v=SolutionVerifier(kin,VerifierConfig(**cfg['_source']['verifier']))
        methods[f'trac_ik_{budget}ms']=TracIK(root/'tmp/revision_dependencies/build/librevision_trac.so',urdf,kin,v,budget,cfg['trac_ik']['internal_epsilon'])
    return methods


def warmup(methods,dataset,frames=24):
    for i in range(frames):
        q=query_from_dataset(dataset,i%len(dataset))
        for m in methods.values():m.solve(q)
    torch.cuda.synchronize()


def calibration_queries(root,robot):
    labels,_=development_labels(root,robot,'calibration_queries')
    source=QueryDataset.load(root/f'outputs/paper_v2_seed17/{robot}/datasets/calibration_queries.npz')
    indices=labels['source_indices']
    return QueryDataset(**{k:getattr(source,k)[indices] for k in source.__dataclass_fields__}),labels


def prepare(root,cfg,out):
    if (out/'selection_and_identity_seal.json').exists():
        raise FileExistsError('selection and identities already sealed')
    geometry=select_geometry(root,cfg,out/'02_point_mechanism_benchmark')
    trac_selected={};selection=[];native_checks={};artifacts={}
    for robot in cfg['robots']:
        kin=load_robot(cfg['_source'],robot)
        calibration,labels=calibration_queries(root,robot)
        external=trac_methods(root,cfg,robot,kin)
        first=next(iter(external.values()))
        rng=np.random.default_rng(cfg['smoke_seeds'][robot])
        errs=[]
        for _ in range(100):
            q=kin.random_configuration(rng,.001);p,r=first.forward(q);pose=kin.forward(q)
            errs.append(max(float(np.max(np.abs(p-pose.position))),float(np.max(np.abs(r-pose.rotation)))))
        if max(errs)>1e-10:raise AssertionError('KDL/public FK mismatch')
        native_checks[robot]=dict(fk_samples=100,max_abs_fk_difference=max(errs),joint_names=list(kin.joint_names),continuous_mask=kin.continuous_mask.tolist())
        # Fixed stratified subset of existing calibration, unrelated to B outcomes.
        families=np.unique(calibration.category)
        ix=[]
        for family in families:
            pool=np.flatnonzero(calibration.category==family)
            ix.extend(rng.permutation(pool)[:cfg['trac_ik']['trajectory_selection_count']//len(families)])
        rest=[i for i in rng.permutation(len(calibration)) if i not in ix]
        ix+=rest[:cfg['trac_ik']['trajectory_selection_count']-len(ix)]
        warmup(external,calibration,4)
        sample_rows=[]
        for j,i in enumerate(ix):
            for name in rng.permutation(list(external)):
                _,row=measured_call(external[name],query_from_dataset(calibration,int(i)),name)
                row.update(robot=robot,source_index=int(labels['source_indices'][i]),query_index=int(i))
                sample_rows.append(row)
            if (j+1)%30==0:print(f'{robot} TRAC-IK calibration {j+1}/{len(ix)}',flush=True)
        ranking=[]
        for name,m in external.items():
            subset=[r for r in sample_rows if r['method']==name]
            successes=sum(r['accepted'] for r in subset);total=sum(r['latency_ns'] for r in subset)
            selection.append(dict(robot=robot,budget_ms=m.budget_ms,query_count=len(subset),successes=successes,total_ns=total))
            ranking.append((-successes,total,m.budget_ms))
        trac_selected[robot]=min(ranking)[2]
        json_write(out/'02_point_mechanism_benchmark'/f'{robot}_trac_calibration_records.json',sample_rows)
        for m in external.values():m.close()
        point,ids,oracle=data.points(kin,cfg,robot)
        traj,tids=data.trajectories(kin,cfg,robot)
        # All new identities are committed to disk before benchmark outcomes.
        paths={f'{robot}_points':out/'02_point_mechanism_benchmark'/f'{robot}_queries.npz',
               f'{robot}_trajectories':out/'03_feasible_trajectory_benchmark'/f'{robot}_trajectories.npz'}
        for key,dataset in [(f'{robot}_points',point),(f'{robot}_trajectories',traj)]:
            p=paths[key];p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():raise FileExistsError(p)
            dataset.save(p);artifacts[str(p.relative_to(root))]=digest(p)
        p=out/'02_point_mechanism_benchmark'/f'{robot}_identities.json'
        json_write(p,dict(queries=ids,oracle_query_indices=oracle.tolist()));artifacts[str(p.relative_to(root))]=digest(p)
        p=out/'03_feasible_trajectory_benchmark'/f'{robot}_identities.json'
        json_write(p,tids);artifacts[str(p.relative_to(root))]=digest(p)
        ph=[r['query_hash'] for r in ids];th=[h for r in tids for h in r['query_hashes']]
        if len(set(ph+th))!=len(ph+th):raise AssertionError('new point/trajectory query overlap')
        old=load_npz(root/f'outputs/fresh_transition_v4_test/{robot}_raw_records.npz')
        prior=set(old['source_query_hash'])|set(old['executed_query_hash'].ravel())
        for role in ['risk_train_queries','calibration_queries','policy_validation_queries']:
            labels,_=development_labels(root,robot,role);prior.update(labels['query_sha256'])
        if set(ph+th)&prior:raise AssertionError('new query overlaps used evidence/development')
        native_checks[robot].update(new_point_queries=len(point),new_trajectory_frames=len(traj),point_witness_count=int(point.continuity_feasible.sum()),trajectory_witness_count=len(traj),prior_query_hashes_checked=len(prior),overlap=0)
    csv_write(out/'02_point_mechanism_benchmark'/'trac_trajectory_calibration.csv',selection)
    sources={str(p.relative_to(root)):digest(p) for p in (root/'src/confik/revision_compute_allocation').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    release_files=list((root/'outputs/release_v4_locked').rglob('*'))+list((root/'outputs/release_v3_locked/panda/seed17').rglob('*'))+list((root/'outputs/release_v3_locked/ur5e/seed17').rglob('*'))
    frozen={str(p.relative_to(root)):digest(p) for p in release_files if p.is_file()}
    json_write(out/'selection_and_identity_seal.json',dict(protocol=cfg['protocol'],geometry=geometry,trac_trajectory_budget_ms=trac_selected,
                new_test_outcomes_opened=False,datasets=artifacts,implementation=sources,frozen_release=frozen,
                config_sha256=digest(root/'configs/revision_compute_allocation.yaml'),native_checks=native_checks,
                trac_ik=dict(version='2.2.0',commit='90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb',mode='Speed',parallel_workers=2,
                             source_unmodified=True,compatibility='header spelling urdf/model.hpp -> Humble urdf/model.h',
                             tolerance='epsilon=1e-5 for each Cartesian component; stricter than 1mm Euclidean and 0.00872664626rad norm public bounds; zero extra Twist bounds',
                             runtime_library_sha256=digest(root/'tmp/revision_dependencies/build/librevision_trac.so')),
                environment=dict(cpu=platform.processor(),platform=platform.platform(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                                 torch_intra_threads=torch.get_num_threads(),torch_inter_threads=torch.get_num_interop_threads(),
                                 affinity=sorted(os.sched_getaffinity(0)),blas_threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})))


def smoke(root,cfg):
    result={}
    for robot in cfg['robots']:
        kin=load_robot(cfg['_source'],robot)
        points,ids,_=data.points(kin,cfg,robot,smoke=True)
        traj,tids=data.trajectories(kin,cfg,robot,smoke=True)
        methods=build_internal(root,cfg['_source'],robot,kin,(.01,.05))
        methods.update(trac_methods(root,cfg,robot,kin,[5]))
        warmup(methods,points,3)
        successes={name:0 for name in methods}
        for i in range(len(points)):
            query=query_from_dataset(points,i)
            for name,method in methods.items():
                _,row=measured_call(method,query,name)
                successes[name]+=row['accepted']
        # Original hard shell and streamlined fixed baseline must return the same
        # command, stage trace, FEV and fallback on identical inputs.
        fixed=methods['always_hard']
        for i in range(min(8,len(points))):
            query=query_from_dataset(points,i)
            a=fixed.runtime.base.solve(query);b=fixed.solve(query)
            assert (a.accepted,a.function_evaluations,a.fallback_used,a.executed_stages)==(b.accepted,b.function_evaluations,b.fallback_used,b.executed_stages)
            if a.accepted:np.testing.assert_array_equal(a.q,b.q)
        result[robot]=dict(point_count=len(points),witnesses=int(points.continuity_feasible.sum()),trajectory_frames=len(traj),successes=successes,fixed_hard_semantic_equivalence=True)
        for name,m in methods.items():
            if name.startswith('trac_ik'):m.close()
    json_write(root/'tmp/revision_smoke_pass.json',result)
    print(json.dumps(result),flush=True)


def evaluate(root,cfg,out,stage):
    seal=json.loads((out/'selection_and_identity_seal.json').read_text())
    execution=json.loads((out/'final_execution_seal.json').read_text())
    for p,h in execution['measurement_sources'].items():
        if digest(root/p)!=h:raise RuntimeError(f'measurement adapter changed: {p}')
    for p,h in {**seal['datasets'],**seal['frozen_release']}.items():
        if digest(root/p)!=h:raise RuntimeError(f'sealed input changed: {p}')
    for robot in cfg['robots']:
        kin=load_robot(cfg['_source'],robot)
        if stage=='points':
            folder=out/'02_point_mechanism_benchmark'
            dataset=QueryDataset.load(folder/f'{robot}_queries.npz')
            ids=json.loads((folder/f'{robot}_identities.json').read_text())
            methods=build_internal(root,cfg['_source'],robot,kin,seal['geometry'][robot])
            methods.update(trac_methods(root,cfg,robot,kin))
            cal,_=calibration_queries(root,robot);warmup(methods,cal)
            benchmark_points(methods,dataset,ids['queries'],folder/f'{robot}_raw_records.jsonl.gz',repeats=cfg['point_repeats'])
            for name,m in methods.items():
                if name.startswith('trac_ik'):m.close()
            oracle=build_internal(root,cfg['_source'],robot,kin,None,['forced_easy','forced_medium','forced_hard'])
            warmup(oracle,cal)
            benchmark_points(oracle,dataset,ids['queries'],folder/f'{robot}_oracle_records.jsonl.gz',repeats=cfg['oracle_repeats'],indices=ids['oracle_query_indices'],seed=960641)
        else:
            folder=out/'03_feasible_trajectory_benchmark'
            dataset=QueryDataset.load(folder/f'{robot}_trajectories.npz')
            ids=json.loads((folder/f'{robot}_identities.json').read_text())
            methods=build_internal(root,cfg['_source'],robot,kin,seal['geometry'][robot],['always_hard','routing_only','full_cghik'])
            methods.update(trac_methods(root,cfg,robot,kin,[seal['trac_trajectory_budget_ms'][robot]]))
            cal,_=calibration_queries(root,robot);warmup(methods,cal)
            benchmark_trajectories(methods,dataset,ids,folder/f'{robot}_raw_records.jsonl.gz')
            for name,m in methods.items():
                if name.startswith('trac_ik'):m.close()
        json_write(folder/f'{robot}_completed.json',dict(stage=stage,robot=robot,complete=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['a','smoke','prepare','points','trajectories','report'])
    args=parser.parse_args();root=Path.cwd();cfg=load_config(root/'configs/revision_compute_allocation.yaml');cfg['_source']=load_config(root/cfg['source_config'])
    out=root/cfg['output'];torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.stage=='a':
        from .decomposition import run
        run(root,out)
    elif args.stage=='smoke':smoke(root,cfg)
    elif args.stage=='prepare':prepare(root,cfg,out)
    elif args.stage in ['points','trajectories']:evaluate(root,cfg,out,args.stage)
    else:
        from .reporting import run
        run(root,cfg,out)


if __name__=='__main__':main()
