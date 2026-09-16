#!/usr/bin/env python3
"""Thin frozen-method data replay and five-condition development sensitivity."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = module('final_frozen_comparison', 'scripts/run_task_balance_comparison.py')
from confik.correction_reserve.study import context, sha, write_json, clean, utc
from confik.solvers.verifier import SolutionVerifier
from confik.task_balance_gn import CompletedTaskBalanceGN
from confik.bounded_gn_adapter import SingleBoundedGN
from confik.task_balance_comparison import CurrentOptimizer, FixedLocalBudget
from confik.task_set_controls import ClarabelBalance
from confik.task_balance_replay import fetch_object, identity_index, raw_schema

OUT = ROOT/'outputs/task_balance_final_evidence'
CONFIG = ROOT/'configs/task_balance_final_evidence.yaml'

def frozen():
    paths = set(old.old.frozen_hashes()) | set(old.old.study_hashes()) | set(old.code_hashes())
    paths.update(['src/confik/solvers/verifier.py', 'paper/main.tex', 'paper/main.pdf'])
    return {p:sha(ROOT/p) for p in sorted(paths) if (ROOT/p).exists()}

def study_hashes():
    return {p:sha(ROOT/p) for p in ['scripts/run_task_balance_final_evidence.py',
        'src/confik/task_balance_replay.py','configs/task_balance_final_evidence.yaml']}

def configured_context(robot, cfg, condition):
    source, kin, original, urdf = context(robot, cfg)
    setting = cfg['sensitivity'][condition]
    v = SolutionVerifier(kin, replace(original.config, position_tolerance=setting['position_m'],
        orientation_tolerance=np.deg2rad(setting['orientation_deg'])))
    return source, kin, v, urdf

def factory(method, robot, cfg, condition='nominal', dt=.02):
    _, kin, v, urdf = configured_context(robot,cfg,condition)
    start = perf_counter_ns()
    if method in cfg['forcing']:
        solver = CompletedTaskBalanceGN(kin,v,urdf,1.,cfg['forcing'][method])
    elif method == 'gn': solver = SingleBoundedGN(kin,v,urdf,1.)
    elif method == 'direct_sqp': solver = CurrentOptimizer(kin,v,urdf,method)
    elif method == 'fixed_qp2': solver = FixedLocalBudget(kin,v,urdf,2)
    elif method == 'clarabel': solver = ClarabelBalance(kin,v,urdf)
    else: raise ValueError(method)
    init = perf_counter_ns()-start
    q = (kin.limits.lower+kin.limits.upper)/2
    pose = kin.forward(q+.25*kin.limits.velocity*dt)
    for _ in range(3): solver.solve(pose.position,pose.rotation,q,dt)
    if method == 'clarabel':
        n=kin.nq;solver.epigraph.solve(np.full(6,2.),np.eye(6,n),-np.ones(n),np.ones(n),.01,np.zeros(n),np.ones(n)*.01,1.)
    settings = solver.engine.settings if hasattr(solver,'engine') else solver.settings
    return solver,kin,v,dict(initialization_ns=init,warmup_calls=3,
        settings=settings if isinstance(settings,dict) else settings.__dict__, verifier=v.config.__dict__,
        conic_structure_warmup=method=='clarabel',backend=old.old.BACKEND)

def reconstruct(item, kin, v, original_v, condition):
    result = dict(item, parent_query_uid=item['uid'], condition=condition)
    if condition != 'nominal':
        witness = kin.forward(np.array(item['q_witness']))
        result['target_position'] = (witness.position + v.config.position_tolerance/original_v.config.position_tolerance *
            (np.array(item['target_position'])-witness.position)).tolist()
        omega = Rotation.from_matrix(np.array(item['target_rotation'])@witness.rotation.T).as_rotvec()
        result['target_rotation'] = (Rotation.from_rotvec(omega*v.config.orientation_tolerance/original_v.config.orientation_tolerance).as_matrix()@witness.rotation).tolist()
        result['uid'] = hashlib.sha256((item['uid']+':'+condition).encode()).hexdigest()
        result['target_identity'] = 'paired reconstruction, not same absolute target'
    verdict = v.check(np.array(result['q_witness']),old.old.query_of(result))
    assert verdict.accepted, (condition, result['uid'],verdict)
    result['configuration_witness_check'] = clean(verdict.__dict__)
    return result

def prepare(cfg):
    root = OUT/'sensitivity'
    original_inputs = {}
    for robot in cfg['robots']:
        ip = old.old.OUT/f'inputs/development_{robot}.json'
        original_inputs[str(ip.relative_to(ROOT))] = sha(ip)
        items = json.loads(ip.read_text());assert len(items)==270
        _,_,original_v,_ = context(robot,cfg)
        for condition in cfg['sensitivity']:
            _,kin,v,_ = configured_context(robot,cfg,condition)
            inputs = [reconstruct(i,kin,v,original_v,condition) for i in items]
            write_json(root/f'inputs_{condition}_{robot}.json',inputs)
        rng = np.random.default_rng(cfg['order_seed']+(robot=='ur5e'))
        jobs=[]
        for i in rng.permutation(len(items)):
            tuples=[(c,m,r) for c,s in cfg['sensitivity'].items() for m in s['methods'] for r in range(3)]
            jobs.extend([int(i),str(c),str(m),int(r)] for c,m,r in rng.permutation(tuples))
        assert len(jobs)==8910
        write_json(root/f'order_{robot}.json',jobs)
    write_json(root/'protocol.json',dict(utc=utc(),config=cfg,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen=frozen(),original_inputs=original_inputs,code=study_hashes(),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in root.glob('*.json')},
        statistics='query repeat means; 30 anchor clusters/robot stratified by 3 geometric classes; descriptive unadjusted 95% intervals',
        selection='No selection: eta=.25 main unchanged. Existing development anchors, not fresh.'))

def check():
    p=json.loads((OUT/'sensitivity/protocol.json').read_text())
    assert p['frozen']==frozen()
    assert p['code']==study_hashes()
    for name,h in dict(p['original_inputs'],**p['input_hashes']).items():assert sha(ROOT/name)==h
    return p

def run_sensitivity(cfg,robot):
    check()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head!=cfg['baseline'], 'Commit protocol/input identities first'
    folder=OUT/f'sensitivity/run_{robot}';folder.mkdir(exist_ok=False)
    items={c:json.loads((OUT/f'sensitivity/inputs_{c}_{robot}.json').read_text()) for c in cfg['sensitivity']}
    jobs=json.loads((OUT/f'sensitivity/order_{robot}.json').read_text())
    cache={(c,m):factory(m,robot,cfg,c) for c,s in cfg['sensitivity'].items() for m in s['methods']}
    write_json(folder/'adapters.json',{c+':'+m:a[3] for (c,m),a in cache.items()})
    with gzip.open(folder/'records.jsonl.gz','xt') as f:
        for index,(i,c,m,rep) in enumerate(jobs):
            solver,kin,v,_=cache[c,m];item=items[c][i];query=old.old.query_of(item)
            solver.reset(query.previous_q)
            row=old.OuterTimer(solver).solve(query.target.position,query.target.rotation,query.previous_q,query.dt)
            row.update(robot=robot,uid=item['uid'],parent_query_uid=item['parent_query_uid'],condition=c,
                anchor_uid=item['anchor_uid'],anchor_family=item['anchor_family'],displacement=item['displacement'],
                alpha=item['alpha'],method=m,repeat=rep,input_index=i)
            f.write(json.dumps(clean(row),separators=(',',':'),allow_nan=False)+'\n')
            if (index+1)%1000==0:print(robot,index+1,'/',len(jobs),flush=True)
    for a in cache.values():a[0].close()
    check()
    write_json(folder/'manifest.json',dict(utc=utc(),head=head,calls=len(jobs),frozen=frozen(),code=study_hashes(),
        affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def fetch_index(cfg):
    objects=json.loads((OUT/'source_replay/source/droid100_objects.json').read_text())['items']
    objects=[o for o in objects if '.tfrecord-' in o['name']]
    assert len(objects)==31 and all(o['name'].startswith(cfg['droid']['bucket_prefix']) for o in objects)
    directory=ROOT/'tmp/task_balance_droid100'
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=[];identities=[]
        for result in pool.map(lambda o:fetch_object(o,directory),objects):
            results.append(result);identities.extend(identity_index(result))
            print('DROID index',len(results),'/31',len(identities),'episodes',flush=True)
    assert len(identities)==100
    write_json(OUT/'source_replay/shards.json',results)
    write_json(OUT/'source_replay/episode_index.json',identities)

def fetch_raw(cfg):
    import requests
    from urllib.parse import quote
    index=json.loads((OUT/'source_replay/episode_index.json').read_text())
    candidates=[]
    for ep in index:
        path=ep['metadata']['episode_metadata/file_path'][0].split('/r2d2-data-full/')[-1]
        uid=hashlib.sha256(path.encode()).hexdigest()
        parts=path.split('/')
        candidates.append(dict(ep,canonical_path=path,uid=uid,
            session=parts[0]+'/'+parts[2],selection='not_reached_by_hash_order'))
    candidates.sort(key=lambda x:x['uid'])
    # Identity order is saved BEFORE downloading fields or checking FK.
    write_json(OUT/'source_replay/candidate_order.json',candidates)
    chosen=[];interface_sessions=set();attempts=[]
    for item in candidates:
        if len(chosen)>=30:break
        if len(chosen)>=6 and item['session'] in interface_sessions:
            item['selection']='excluded_interface_session';continue
        folder=OUT/'source_replay/raw'/item['uid']
        found=False
        for version,prefix in [('1.0.1','robotics/droid_raw/1.0.1/'),
                               ('1.0.0','robotics/droid_raw/1.0.0/r2d2-all-blurred-final/')]:
            name=prefix+item['canonical_path']
            url='https://storage.googleapis.com/storage/v1/b/gresearch/o/'+quote(name,safe='')
            try:
                response=requests.get(url,timeout=30)
                attempts.append(dict(uid=item['uid'],url=url,http_status=response.status_code))
                if response.status_code==404:continue
                response.raise_for_status();obj=response.json()
                # Low-dimensional file only: never trajectory_im128.h5 or videos.
                if int(obj['size'])>20_000_000:raise ValueError('unexpected low-dimensional object size')
                source=fetch_object(obj,folder);schema=raw_schema(source['path'])
                item.update(source=source,object_metadata=obj,schema=schema,raw_version=version)
                found=True;break
            except requests.RequestException as exc:
                attempts.append(dict(uid=item['uid'],url=url,error=str(exc)))
        if not found:
            item['selection']='raw_access_unavailable';continue
        if not schema['eligible']:
            item['selection']='schema_or_time_excluded';continue
        role='interface' if len(chosen)<6 else 'evaluation'
        item['selection']=role
        if role=='interface':interface_sessions.add(item['session'])
        chosen.append(item)
        print('Selected',len(chosen),role,item['canonical_path'],schema['frames'],flush=True)
    write_json(OUT/'source_replay/selection.json',dict(utc=utc(),candidates=candidates,selected=chosen,
        access_attempts=attempts,interface_sessions=sorted(interface_sessions),
        session_definition='lab/date conservative recording-day cluster; no claim of independent frames',
        rule='ascending SHA256 raw identity; first 6 schema/time-usable interface, next 24 not sharing lab/date; no IK, geometry proximity or success filters',
        complete=len(chosen)==30))

def prepare_replay(cfg):
    import h5py
    selected=json.loads((OUT/'source_replay/selection.json').read_text())
    assert selected['complete'], 'A incomplete: fewer than 30 schema/time-usable episodes'
    _,kin,v,_=context('panda',cfg);all_inputs=[];audit=[];observations=[]
    for episode in selected['selected']:
        with h5py.File(episode['source']['path'],'r') as f:
            q=f['action/robot_state/joint_positions'][()]
            state=f['action/robot_state/cartesian_position'][()]
            obsq=f['observation/robot_state/joint_positions'][()]
            target=f['action/cartesian_position'][()]
            command=f['action/joint_position'][()]
            times=f['observation/timestamp/control/step_start'][()]
            control=f['observation/timestamp/control/control_start'][()]
            skip=f['observation/timestamp/skip_action'][()] if 'observation/timestamp/skip_action' in f else np.zeros(len(q),bool)
            dt=1./cfg['droid']['source_control_hz']
            fkp=[];fkr=[];witness_count=0
            for frame in range(len(q)):
                pose=kin.forward(q[frame]);rotation=Rotation.from_euler('xyz',target[frame,3:]).as_matrix()
                source_rotation=Rotation.from_euler('xyz',state[frame,3:]).as_matrix()
                ep=float(np.linalg.norm(pose.position-state[frame,:3]))
                er=float(np.linalg.norm(Rotation.from_matrix(source_rotation@pose.rotation.T).as_rotvec()))
                fkp.append(ep);fkr.append(er)
                item=dict(robot='panda',uid=hashlib.sha256(f"droid:{episode['uid']}:{frame}".encode()).hexdigest(),
                    episode_uid=episode['uid'],session=episode['session'],source_frame=frame,
                    previous_q=q[frame].tolist(),target_position=target[frame,:3].tolist(),target_rotation=rotation.tolist(),dt=dt,
                    source_target_xyz_euler=target[frame].tolist(),source_state_xyz_euler=state[frame].tolist(),
                    source_command_q=command[frame].tolist(),observation_q=obsq[frame].tolist(),
                    source_step_start_ms=int(times[frame]),source_control_start_ms=int(control[frame]),
                    source_inter_record_ms=None if frame==0 else int(times[frame]-times[frame-1]),
                    source_skip_action=bool(skip[frame]),role=episode['selection'])
                query=old.old.query_of(item);w=v.check(command[frame],query);now=v.check(q[frame],query)
                item.update(witness_available=bool(w.accepted),source_witness_check=clean(w.__dict__),
                    target_normalized_from_state=[now.position_error/v.config.position_tolerance,now.orientation_error/v.config.orientation_tolerance],
                    source_command_step_utilization=(abs(command[frame]-q[frame])/(kin.limits.velocity*dt+v.config.velocity_tolerance)).tolist(),
                    physical_joint_margin=np.minimum(q[frame]-kin.limits.lower,kin.limits.upper-q[frame]).tolist(),
                    fk_source_position_difference=ep,fk_source_orientation_difference=er)
                witness_count+=w.accepted
                all_inputs.append(item)
            audit.append(dict(episode_uid=episode['uid'],canonical_path=episode['canonical_path'],role=episode['selection'],
                session=episode['session'],frames=len(q),source_version=episode['schema']['source_version'],
                position_difference_quantiles_m=np.percentile(fkp,[0,50,95,100]).tolist(),
                orientation_difference_quantiles_rad=np.percentile(fkr,[0,50,95,100]).tolist(),
                observation_action_joint_difference_max=float(np.max(abs(q-obsq))),
                interval_ms_quantiles=np.percentile(np.diff(times),[0,50,95,100]).tolist(),
                gaps_over_two_configured_periods=int(np.sum(np.diff(times)>2000/15)),
                nonmonotonic_steps=int(np.sum(np.diff(times)<=0)),source_skipped_action_records=int(np.sum(skip)),
                observation_to_control_ms_quantiles=np.percentile(control-times,[0,50,95,100]).tolist(),
                source_command_witnesses=int(witness_count),
                fk_matches=bool(max(fkp)<=cfg['droid']['fk_interface_max_position_m'] and max(fkr)<=cfg['droid']['fk_interface_max_orientation_rad'])))
    write_json(OUT/'source_replay/interface_audit.json',dict(episodes=audit,
        interface_passed=all(a['fk_matches'] for a in audit if a['role']=='interface'),
        all_source_fk_consistent=all(a['fk_matches'] for a in audit),
        mapping='identity panda_link0->panda_link8; no fit or tool transformation; xyz extrinsic Euler, radians/metres',
        timing='fixed 1/15 s, configuration-reconstructed; record interval diagnostics not enlarged movement budget',
        missing_provenance='HDF5 version_number is not a git SHA; exact collection checkout not recorded. Current pinned official control code and timestamp cadence are corroborating evidence.',
        thresholds=cfg['droid']))
    # If any frame semantics/model mismatch remains, preserve evidence and stop A.
    assert all(a['fk_matches'] for a in audit), 'Unresolved model/source mismatch: do not run replay'
    for role in ('interface','evaluation'):
        write_json(OUT/f'source_replay/inputs_{role}.json',[i for i in all_inputs if i['role']==role])
    items=[i for i in all_inputs if i['role']=='evaluation']
    rng=np.random.default_rng(cfg['order_seed']+10);jobs=[]
    for i in rng.permutation(len(items)):
        jobs.extend([int(i),str(m),int(r)] for m,r in rng.permutation([(m,r) for m in cfg['droid']['methods'] for r in range(3)]))
    write_json(OUT/'source_replay/order.json',jobs)
    write_json(OUT/'source_replay/replay_protocol.json',dict(utc=utc(),config=cfg,code=study_hashes(),frozen=frozen(),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'source_replay/selection.json',
            OUT/'source_replay/interface_audit.json',OUT/'source_replay/inputs_evaluation.json',OUT/'source_replay/order.json']},
        queries=len(items),episodes=24,dt=1/15,deadline_ms=20,online_state='same action-time logged q for every method/query; no feedback',
        statistics='episode units, queries and repeats nested; session lab/date clusters; no acceleration from unrelated offline outputs'))

def run_replay(cfg):
    check()
    seal=json.loads((OUT/'source_replay/replay_protocol.json').read_text())
    assert seal['code']==study_hashes() and seal['frozen']==frozen()
    for name,h in seal['input_hashes'].items():assert sha(ROOT/name)==h
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head!=cfg['baseline']
    folder=OUT/'source_replay/run';folder.mkdir(exist_ok=False)
    items=json.loads((OUT/'source_replay/inputs_evaluation.json').read_text())
    jobs=json.loads((OUT/'source_replay/order.json').read_text())
    cache={m:factory(m,'panda',cfg,dt=1/15) for m in cfg['droid']['methods']}
    write_json(folder/'adapters.json',{m:x[3] for m,x in cache.items()})
    with gzip.open(folder/'records.jsonl.gz','xt') as f:
        for index,(i,m,rep) in enumerate(jobs):
            solver,kin,v,_=cache[m];item=items[i];query=old.old.query_of(item)
            solver.reset(query.previous_q)
            row=old.OuterTimer(solver).solve(query.target.position,query.target.rotation,query.previous_q,query.dt)
            row.update(robot='panda',uid=item['uid'],episode_uid=item['episode_uid'],session=item['session'],
                source_frame=item['source_frame'],input_index=i,method=m,repeat=rep,witness_available=item['witness_available'])
            f.write(json.dumps(clean(row),separators=(',',':'),allow_nan=False)+'\n')
            if (index+1)%5000==0:print('DROID replay',index+1,'/',len(jobs),flush=True)
    for a in cache.values():a[0].close()
    check()
    write_json(folder/'manifest.json',dict(utc=utc(),head=head,calls=len(jobs),code=study_hashes(),frozen=frozen(),
        affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch_index','fetch_raw','prepare_replay','replay','prepare','sensitivity']);p.add_argument('--robot',choices=['panda','ur5e'])
    a=p.parse_args();cfg=yaml.safe_load(CONFIG.read_text());os.sched_setaffinity(0,{cfg['cpu']})
    if a.phase=='fetch_index':fetch_index(cfg)
    elif a.phase=='fetch_raw':fetch_raw(cfg)
    elif a.phase=='prepare_replay':prepare_replay(cfg)
    elif a.phase=='replay':run_replay(cfg)
    elif a.phase=='prepare':prepare(cfg)
    else:
        if a.robot is None:p.error('--robot required')
        run_sensitivity(cfg,a.robot)

if __name__=='__main__':main()
