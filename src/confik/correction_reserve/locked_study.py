"""Fixed-candidate fresh evaluation; no numerical changes to Elastic mu=0.25."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml
from ..types import IKQuery
from ..latency_pilot_v3.benchmark import query_digest
from . import study as old
from . import elastic_study as elastic
from . import minimal_study as minimal
from .data import reference_path,preserve_inventory
from .mu0_analytic import MuZeroAnalyticIK

CONFIG='configs/correction_reserve_elastic_locked.yaml'
ANALYTIC='cr_ik_mu0_analytic'
PRIMARY='cr_ik_elastic_mu025'


def configurations():
    cfg=yaml.safe_load((old.ROOT/CONFIG).read_text())
    return cfg,yaml.safe_load((old.ROOT/cfg['base_config']).read_text()),old.ROOT/cfg['output']


def hashes():
    paths=[CONFIG,'docs/CRIK_ELASTIC_LOCKED_PROTOCOL.md','tests/test_crik_elastic_locked.py',
        'scripts/run_crik_elastic_locked.sh']
    paths += ['src/confik/correction_reserve/'+n+'.py' for n in ('mu0_analytic','locked_study','data','reporting')]
    return dict(elastic.hashes(),**{p:old.sha(old.ROOT/p) for p in paths})


def target_hash(item):
    keys=('initial_q','target_position','target_rotation','dt')
    if not all(k in item for k in keys):return None
    return hashlib.sha256(json.dumps({k:item[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def historical_identities():
    """Only identity/online-input registries, never method outcome records."""
    names={'identities.json','trajectory_identities.json','ur5e_trajectory_identities.json',
        'trajectory_split_manifest.json','online_targets.json','online_inputs.json','ur5e_online_targets.json'}
    paths=subprocess.check_output(['git','ls-files','outputs'],cwd=old.ROOT,text=True).splitlines()
    paths=[p for p in paths if Path(p).name in names or Path(p).name.endswith('_identity_manifest.json')]
    uids=set();seeds=set();queries=set();sequences=set();sources={}
    def walk(x,key=''):
        if isinstance(x,dict):
            if (h:=target_hash(x)) is not None:sequences.add(h)
            for k,v in x.items():walk(v,k)
        elif isinstance(x,list):
            for v in x:walk(v,key)
        elif key in ('uid','trajectory_uid','trajectory_uids') and isinstance(x,str):uids.add(x)
        elif key in ('seed','seeds','trajectory_seed','trajectory_seeds') and isinstance(x,int):seeds.add(x)
        elif key in ('query_hash','query_hashes','query_sha256') and isinstance(x,str):queries.add(x)
    for p in paths:
        # New protocol is excluded when checking the committed registry later.
        if '/elastic_locked_evaluation/' in p:continue
        sources[p]=old.sha(old.ROOT/p);walk(json.loads((old.ROOT/p).read_text()))
    return uids,seeds,queries,sequences,sources


def prepare():
    cfg,base,root=configurations();protocol=root/'protocol'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip()==cfg['baseline_commit']
    assert not subprocess.check_output(['git','diff',cfg['baseline_commit'],'--name-status','--diff-filter=DMRT'],cwd=old.ROOT,text=True)
    protocol.mkdir(parents=True,exist_ok=False)
    prior=old.ROOT/cfg['previous_output']/'protocol'
    for name in ('demand_parameters.json','dependencies.json'):
        value=json.loads((prior/name).read_text());old.write_json(protocol/name,value)
        assert old.sha(protocol/name)==old.sha(prior/name)
    deps=json.loads((protocol/'dependencies.json').read_text())
    assert all(importlib.metadata.version(k)==v for k,v in deps['packages'].items())
    assert old.sha(old.ROOT/deps['native_library'])==deps['native_library_sha256']
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_crik_elastic_locked.py',
        'tests/test_crik_elastic.py','tests/test_crik_minimal.py'],cwd=old.ROOT,text=True,capture_output=True,check=True)
    old.write_json(protocol/'tests.json',dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    old_uids,old_seeds,old_queries,old_sequences,sources=historical_identities()
    targets=[];witnesses=[];identities=[];new_hashes=set();new_sequences=set()
    for ri,robot in enumerate(cfg['robots']):
        _,kin,v,_=old.context(robot,base)
        for fi,family in enumerate(cfg['families']):
            for j in range(cfg['trajectories_per_family']):
                seed=cfg['seed_base']+ri*10000+fi*100+j
                assert seed not in old_seeds,(robot,seed,'old seed')
                q,info=reference_path(kin,family,seed,cfg['frames'],cfg['dt'],j)
                uid=hashlib.sha256(f'{cfg["uid_namespace"]}:{robot}:{family}:{seed}'.encode()).hexdigest()
                assert uid not in old_uids
                poses=[kin.forward(x) for x in q];queries=[]
                for t,pose in enumerate(poses):
                    query=IKQuery(pose,q[max(0,t-1)],cfg['dt'])
                    check=v.check(q[t],query)
                    assert check.accepted,(robot,seed,t,check)
                    digest=query_digest(query)
                    assert digest not in old_queries and digest not in new_hashes,(robot,seed,t,'query collision')
                    new_hashes.add(digest);queries.append(digest)
                common=dict(robot=robot,uid=uid,site_id=f'trajectory_{fi*cfg["trajectories_per_family"]+j:03d}',
                    family=family,seed=seed,dt=cfg['dt'])
                item=dict(**common,initial_q=q[0].tolist(),target_position=[p.position.tolist() for p in poses],
                    target_rotation=[p.rotation.tolist() for p in poses])
                digest=target_hash(item)
                assert digest not in old_sequences and digest not in new_sequences
                new_sequences.add(digest);targets.append(item)
                witnesses.append(dict(**common,reference_q=q.tolist(),verified=[True]*len(q),
                    meaning='reference-state feasible path, not method-specific next-state feasibility'))
                identities.append(dict(**common,frames=len(q),query_hashes=queries,target_sequence_hash=digest,**info,
                    min_limit_margin=float(np.min(np.minimum(q-kin.limits.lower,kin.limits.upper-q)))))
            print(f'Fixed {robot} {family}: {len(targets)} total paths; no comparison calls',flush=True)
    assert len(targets)==320 and len({i['uid'] for i in targets})==320
    old.write_json(protocol/'online_inputs.json',targets)
    old.write_json(protocol/'reference_witnesses.json',witnesses)
    old.write_json(protocol/'identities.json',identities)
    old.write_json(protocol/'freshness.json',dict(sources=sources,old_uid_count=len(old_uids),old_seed_count=len(old_seeds),
        old_query_hash_count=len(old_queries),old_target_sequence_count=len(old_sequences),
        uid_collisions=0,seed_collisions=0,known_query_hash_collisions=0,target_sequence_collisions=0,
        reference_frames_verified=len(new_hashes),solver_outcome_screening=False,resampled_trajectories=0))
    old.write_json(protocol/'baseline_git_inventory.json',preserve_inventory())
    old.write_json(protocol/'selection_seal.json',dict(utc=old.utc(),configuration=cfg,code_hashes=hashes(),
        files={p.name:old.sha(p) for p in protocol.iterdir() if p.is_file()},
        prior_manifest_sha256=old.sha(old.ROOT/cfg['previous_output']/'delivery_manifest.json'),
        state='candidate and all 320 reference-verified input identities fixed before any new comparative outcomes',
        comparison_calls=0,selected_from_observed_development=True))


def verify_seal(require_commit=True):
    cfg,_,root=configurations();protocol=root/'protocol'
    seal=json.loads((protocol/'selection_seal.json').read_text())
    assert cfg==seal['configuration'] and hashes()==seal['code_hashes']
    for p,h in seal['files'].items():assert old.sha(protocol/p)==h,p
    assert old.sha(old.ROOT/cfg['previous_output']/'delivery_manifest.json')==seal['prior_manifest_sha256']
    assert not subprocess.check_output(['git','diff',cfg['baseline_commit'],'--name-status','--diff-filter=DMRT'],cwd=old.ROOT,text=True)
    if require_commit:
        paths=list(hashes())+[str(protocol.relative_to(old.ROOT))]
        for extra in ([],['--cached']):
            assert not subprocess.check_output(['git','diff',*extra,'--name-only','--',*paths],cwd=old.ROOT,text=True)
        for p in paths:
            # Force-added output paths are ignored by normal untracked checks.
            assert subprocess.check_output(['git','ls-files','--',p],cwd=old.ROOT,text=True).strip(),p
    return seal


def factory(method,robot,base,parameters):
    if method!=ANALYTIC:return elastic.factory(method,robot,base,parameters)
    source,kin,v,urdf=old.context(robot,base)
    return MuZeroAnalyticIK(kin,v,source,str(old.ROOT/base['native_trac_library']),urdf,
        demand_parameters=parameters[robot],config=base['optimizer']),kin,v


def jobs_for(items,cfg):
    return [(i,m,r) for i in items for m in cfg['methods'] for r in range(cfg['repeats'])]


def add_metrics(rows,s,kin,v):
    # Future joins run only after all 300 calls; they cannot enter solver state.
    minimal.add_offline_metrics(rows,s,kin,v)
    if rows[0]['method'] not in (ANALYTIC,PRIMARY):return
    for key,field in [('optimization_called','optimization_call_rate'),('conic_calls','mean_conic_calls'),
        ('partial_correction_adopted','partial_correction_rate'),('effective_command_adjustment','effective_command_adjustment_rate')]:
        s[field]=float(np.mean([r[key] for r in rows]))
    pairs=[r for r in rows if r['initial_pair_legal'] and r['actual_shortfall_reduction'] is not None]
    s.update(actual_shortfall_reduction_mean=float(np.mean([r['actual_shortfall_reduction'] for r in pairs])) if pairs else None,
        normalized_shortfall_reduction_mean=float(np.mean([r['actual_shortfall_reduction']/r['demand'] for r in pairs])) if pairs else None,
        decision_counts=dict(Counter(r['decision'] for r in rows)),
        fallback_counts=dict(Counter(r['fallback_reason'] for r in rows if r['fallback_reason'])))


def run(robot):
    seal=verify_seal();cfg,base,root=configurations();folder=root/robot
    assert sorted(os.sched_getaffinity(0))==cfg['cpu_affinity'][robot]
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    folder.mkdir(exist_ok=False);(folder/'runs').mkdir()
    items=[i for i in json.loads((root/'protocol/online_inputs.json').read_text()) if i['robot']==robot]
    assert len(items)==160 and all(len(i['target_position'])==300 and 'reference_q' not in i for i in items)
    parameters=json.loads((root/'protocol/demand_parameters.json').read_text())
    old.write_json(folder/'started.json',dict(utc=old.utc(),code_hashes=hashes(),configuration=cfg,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        timing_scope='all command-ready computation, not serialization or post-trajectory diagnostic joins',
        startup_warmup_calls_per_method=1))
    jobs=jobs_for(items,cfg);order=np.random.default_rng(cfg['order_seed']).permutation(len(jobs)).tolist()
    old.write_json(folder/'job_order.json',[dict(uid=jobs[j][0]['uid'],method=jobs[j][1],repeat=jobs[j][2]) for j in order])
    solvers={};summaries=[];begin=time.monotonic()
    try:
        for k,j in enumerate(order):
            item,method,repeat=jobs[j];rid=f'{robot}_{item["site_id"]}_{method}_r{repeat}'
            if method not in solvers:
                solvers[method]=factory(method,robot,base,parameters)
                solver,kin,v=solvers[method];q=np.asarray(item['initial_q']);pose=kin.forward(q)
                solver.solve(pose.position,pose.rotation,q,.02)
            solver,kin,v=solvers[method]
            rows,s=old.execute_trajectory(solver,kin,v,item,method,repeat)
            add_metrics(rows,s,kin,v);record=folder/'runs'/f'{rid}.jsonl.gz'
            with gzip.open(record,'xt',encoding='utf8',compresslevel=6) as f:
                for row in rows:f.write(json.dumps(old.clean(row),allow_nan=False,separators=(',',':'))+'\n')
            s.update(robot=robot,uid=item['uid'],site_id=item['site_id'],family=item['family'],method=method,
                repeat=repeat,run_id=rid,raw_file=str(record.relative_to(folder)))
            old.write_json(folder/'runs'/f'{rid}.summary.json',s);summaries.append(s)
            if (k+1)%10==0:print(f'{robot} {k+1}/{len(jobs)} elapsed={(time.monotonic()-begin)/60:.1f}min',flush=True)
    finally:
        for solver,_,_ in solvers.values():solver.close()
    assert hashes()==seal['code_hashes']
    old.write_json(folder/'summaries.json',summaries)
    old.write_json(folder/'completed.json',dict(utc=old.utc(),runs=len(summaries),frames=sum(s['frames'] for s in summaries),
        files={str(p.relative_to(folder)):old.sha(p) for p in folder.rglob('*') if p.is_file()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','run']);p.add_argument('--robot',choices=['panda','ur5e'])
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':verify_seal();print('Candidate, identities and frozen evidence verified.')
    elif a.robot is None:p.error('--robot required')
    else:run(a.robot)


if __name__=='__main__':main()
