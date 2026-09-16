#!/usr/bin/env python3
"""One entry: prepare / official / development / seal / points / application.

Reuses frozen point factories and trajectory execution/statistics, never rewrites
their outputs. Protocol and inputs are fixed before comparison calls.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = module('frozen_task_set_runner', 'scripts/run_task_set_boundary.py')
from confik.correction_reserve.study import context, sha, write_json, clean, utc, execute_trajectory
from confik.task_balance_comparison import FixedLocalBudget, CurrentOptimizer

OUT = ROOT/'outputs/task_balance_comparison'
CONFIG = ROOT/'configs/task_balance_comparison.yaml'

def code_hashes():
    return {n:sha(ROOT/n) for n in ('src/confik/task_balance_comparison.py',
        'scripts/run_task_balance_comparison.py','configs/task_balance_comparison.yaml')}

def factory(method, robot, cfg):
    if method in ('relative','gn','tight','clarabel'):
        return old.factory(method, robot, cfg)
    _, kin, v, urdf = context(robot, cfg); start = perf_counter_ns()
    solver = (FixedLocalBudget(kin,v,urdf,int(method[-1])) if method.startswith('fixed_qp')
              else CurrentOptimizer(kin,v,urdf,method))
    elapsed = perf_counter_ns()-start
    q = (kin.limits.lower+kin.limits.upper)/2
    pose = kin.forward(q+.25*kin.limits.velocity*cfg['dt'])
    for _ in range(3): solver.solve(pose.position,pose.rotation,q,cfg['dt'])
    return solver, kin, v, dict(initialization_ns=elapsed,warmup_calls=3,
        settings=solver.settings if isinstance(solver.settings,dict) else solver.settings.__dict__,
        backend=old.BACKEND)

def prepare(cfg):
    paths = [old.OUT/f'inputs/{split}_{robot}.json' for split in ('development','validation') for robot in cfg['robots']]
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    write_json(OUT/'protocol/frozen_before.json',dict(utc=utc(),baseline=cfg['baseline'],
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_hashes=old.frozen_hashes(),old_study_hashes=old.study_hashes(),
        input_hashes=inputs,backend=old.BACKEND,affinity=sorted(os.sched_getaffinity(0))))
    for robot in cfg['robots']:
        _,kin,v,_ = context(robot,cfg)
        q = np.array(cfg['application']['initial_q'][robot]); pose = kin.forward(q)
        assert v.check(q,old.query_of(dict(previous_q=q,target_position=pose.position,target_rotation=pose.rotation,dt=cfg['dt']))).accepted
        items=[]
        for amplitude in cfg['application']['amplitudes_m']:
            for angle in cfg['application']['plane_angles_rad']:
                u=pose.rotation@np.array([np.cos(angle),np.sin(angle),0.])
                vv=pose.rotation@np.array([-np.sin(angle),np.cos(angle),0.])
                s=np.linspace(0,1,cfg['application']['frames'])
                p=pose.position+amplitude*np.sin(4*np.pi*s)[:,None]*u+.5*amplitude*np.sin(2*np.pi*s)[:,None]*vv
                identity=f'task_balance_review:cartesian_scan:{robot}:{amplitude}:{angle}:20260916'
                uid=hashlib.sha256(identity.encode()).hexdigest()
                items.append(dict(robot=robot,uid=uid,site_id=f'scan_{len(items):02d}',family='planar_observation_scan',
                    dt=cfg['dt'],initial_q=q.tolist(),amplitude_m=amplitude,plane_angle_rad=angle,
                    scan_origin=pose.position.tolist(),scan_u=u.tolist(),scan_v=vv.tolist(),
                    target_position=p.tolist(),target_rotation=[pose.rotation.tolist() for _ in s],
                    source='application-shaped synthetic Cartesian geometry; not a random FK joint trajectory',
                    witness_availability='common starting pose only; later method-specific feasibility unknown'))
        write_json(OUT/f'application/inputs_{robot}.json',items)
    for phase in ('development','points','application'):
        for robot in cfg['robots']:
            methods=cfg['development_methods'] if phase=='development' else cfg['methods'] if phase=='points' else cfg['application_methods']
            n=270 if phase=='development' else 2160 if phase=='points' else 12
            rng=np.random.default_rng(cfg['order_seed']+(robot=='ur5e')+100*('development','points','application').index(phase))
            jobs=[]
            for i in rng.permutation(n):
                jobs.extend((int(i),str(m),int(r)) for m,r in rng.permutation([(m,r) for m in methods for r in range(3)]))
            write_json(OUT/f'protocol/{phase}_{robot}_order.json',jobs)

def verify_frozen():
    b=json.loads((OUT/'protocol/frozen_before.json').read_text())
    assert b['frozen_hashes']==old.frozen_hashes()
    assert b['old_study_hashes']==old.study_hashes()
    for name,h in b['input_hashes'].items(): assert sha(ROOT/name)==h
    return b

def official():
    folder=ROOT/'tmp/crik_dependencies/ranged_ik'
    env=os.environ.copy()
    toolchain=ROOT/'tmp/crik_dependencies/rust_home/toolchains/1.85.1-x86_64-unknown-linux-gnu/bin'
    env.update(CARGO_HOME=str(ROOT/'tmp/crik_dependencies/cargo_home'),
               RUSTUP_HOME=str(ROOT/'tmp/crik_dependencies/rust_home'),PATH=str(toolchain)+':'+env['PATH'])
    cmd=[str(toolchain/'cargo'),'run','--release','--offline','--bin','relaxed_ik_bin',
         '--target-dir',str(ROOT/'tmp/task_balance_review_ranged_build')]
    result=subprocess.run(cmd,cwd=folder,env=env,text=True,capture_output=True)
    write_json(OUT/'references/official_demo.json',dict(command=cmd,cwd=str(folder),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip(),
        worktree=subprocess.check_output(['git','status','--porcelain'],cwd=folder,text=True),
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        native_task='unmodified supplied Sawyer example; dependency smoke, not a third robot evaluation',
        build='release opt-level 3; isolated target dir and existing offline Rust dependencies'))
    print('OFFICIAL DEMO',result.returncode,result.stdout[-1500:],result.stderr[-1500:],flush=True)

def seal(cfg):
    verify_frozen()
    files={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'protocol').glob('*order.json')}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in (OUT/'application').glob('inputs_*.json')})
    for robot in cfg['robots']:
        path=OUT/f'points/development_{robot}/manifest.json'
        files[str(path.relative_to(ROOT))]=sha(path)
        path=OUT/f'points/development_sqp_{robot}/manifest.json'
        files[str(path.relative_to(ROOT))]=sha(path)
    write_json(OUT/'protocol/comparison_seal.json',dict(utc=utc(),code_hashes=code_hashes(),files=files,
        official_demo_hash=sha(OUT/'references/official_demo.json'),
        parameters_selected='Loss defaults and fixed controls unchanged; SLSQP 1e-7 normalized squared-constraint numerical interior guard after unguarded development endpoint rejection. No main-method tuning.',
        study='supplementary common-input comparison of already observed 2160 queries/robot',
        config=cfg,main_calls_started=False))

class OuterTimer:
    def __init__(self,solver): self.solver=solver
    def reset(self,q): self.solver.reset(q)
    def solve(self,*args):
        t=perf_counter_ns();r=self.solver.solve(*args);elapsed=perf_counter_ns()-t
        r['adapter_total_latency_ns']=r['total_latency_ns'];r['total_latency_ns']=elapsed
        r['accepted_within_20ms']=bool(r['accepted'] and elapsed<=20_000_000)
        return r

def run(cfg, phase, robot):
    verify_frozen()
    if not phase.startswith('development'):
        sealdata=json.loads((OUT/'protocol/comparison_seal.json').read_text())
        assert sealdata['code_hashes']==code_hashes()
        for name,h in sealdata['files'].items(): assert sha(ROOT/name)==h
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        assert head!=cfg['baseline'],'Commit protocol/code/identities before main comparison'
    app=phase=='application'
    folder=OUT/('application' if app else 'points')/f'{phase}_{robot}';folder.mkdir(exist_ok=False)
    ip=OUT/f'application/inputs_{robot}.json' if app else old.OUT/f'inputs/{"development" if phase.startswith("development") else "validation"}_{robot}.json'
    orderphase='development' if phase=='development_sqp' else phase
    items=json.loads(ip.read_text());jobs=json.loads((OUT/f'protocol/{orderphase}_{robot}_order.json').read_text())
    if phase=='development_sqp': jobs=[j for j in jobs if j[1]=='direct_sqp']
    methods=list(dict.fromkeys(j[1] for j in jobs));cache={m:factory(m,robot,cfg) for m in methods}
    write_json(folder/'adapters.json',{m:a[3] for m,a in cache.items()});summaries=[]
    with gzip.open(folder/'records.jsonl.gz','xt') as f:
        for index,(i,m,rep) in enumerate(jobs):
            solver,kin,v,_=cache[m];item=items[i]
            if app:
                rows,s=execute_trajectory(OuterTimer(solver),kin,v,item,m,rep)
                s.update(robot=robot,uid=item['uid'],method=m,repeat=rep,family=item['family'],
                         run_id=f'{robot}_{i}_{m}_{rep}')
                summaries.append(s)
            else:
                query=old.query_of(item);solver.reset(query.previous_q)
                r=OuterTimer(solver).solve(query.target.position,query.target.rotation,query.previous_q,query.dt)
                assert bool(v.check(np.array(r['q']),query).accepted)==r['accepted']
                r.update(uid=item['uid'],anchor_uid=item['anchor_uid'],robot=robot,method=m,
                    repeat=rep,input_index=i,anchor_family=item['anchor_family'],displacement=item['displacement'],alpha=item['alpha'])
                rows=[r]
            for r in rows: f.write(json.dumps(clean(r),separators=(',',':'),allow_nan=False)+'\n')
            if (index+1)%(12 if app else 1000)==0: print(phase,robot,index+1,'/',len(jobs),flush=True)
    for solver,_,_,_ in cache.values(): solver.close()
    if app: write_json(folder/'summaries.json',summaries)
    verify_frozen()
    write_json(folder/'manifest.json',dict(utc=utc(),phase=phase,robot=robot,jobs=len(jobs),
        calls=len(jobs)*(cfg['application']['frames'] if app else 1),input_sha256=sha(ip),code_hashes=code_hashes(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),backend=old.BACKEND,
        affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','official','development','development_sqp','seal','points','application']);p.add_argument('--robot',choices=['panda','ur5e'])
    a=p.parse_args();cfg=yaml.safe_load(CONFIG.read_text());os.sched_setaffinity(0,{cfg['cpu']})
    if a.phase=='prepare':prepare(cfg)
    elif a.phase=='official':official()
    elif a.phase=='seal':seal(cfg)
    else:
        if a.robot is None:p.error('--robot required')
        run(cfg,a.phase,a.robot)

if __name__=='__main__':main()
