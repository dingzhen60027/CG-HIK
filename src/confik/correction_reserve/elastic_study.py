"""Independent elastic-demand entry point, one complete development comparison."""
import argparse
from collections import Counter
import csv
import gzip
import json
import os
import subprocess
import sys
import time
import numpy as np
import yaml
from . import study as old
from . import minimal_study as previous
from .elastic_runtime import ElasticInterventionIK, MU_METHODS, OBJECTIVE_ATOL, OBJECTIVE_RTOL
from .reporting import csv_write

CONFIG='configs/correction_reserve_elastic.yaml'
ELASTIC={method:mu for mu,method in MU_METHODS.items()}
FOCUS_UID='bf166cd6db08076d8c85cfd7e7364d2446c3c1ec592ea82609b812c0faa204ea'


def configurations():
    cfg=yaml.safe_load((old.ROOT/CONFIG).read_text());base=yaml.safe_load((old.ROOT/cfg['base_config']).read_text())
    return cfg,base,old.ROOT/cfg['output']


def hashes():
    cfg,_,_=configurations()
    paths=[old.ROOT/'src/confik/correction_reserve'/f'elastic_{name}.py' for name in ('convex','runtime','study')]
    paths += [old.ROOT/CONFIG,old.ROOT/'docs/CRIK_ELASTIC_DEMAND_PROTOCOL.md',old.ROOT/'tests/test_crik_elastic.py']
    return dict(previous.hashes(),**{str(p.relative_to(old.ROOT)):old.sha(p) for p in paths})


def preflight(root,prior):
    """Read-only existing-log localization; no solver calls or trajectory replay."""
    result=[];focus=[];source={};raw_hashes={}
    failures=prior/'reports/first_failure_inputs.csv'
    with failures.open(newline='') as f:
        old_failure_rows=[r for r in csv.DictReader(f) if r['uid']==FOCUS_UID and r['method']=='cr_ik_minimal']
    source[str(failures.relative_to(old.ROOT))]=old.sha(failures)
    for robot in ('panda','ur5e'):
        folder=prior/robot;c=Counter();states=Counter()
        source[str((folder/'completed.json').relative_to(old.ROOT))]=old.sha(folder/'completed.json')
        for s in json.loads((folder/'summaries.json').read_text()):
            if s['method']!='cr_ik_minimal':continue
            file=folder/s['raw_file'];raw_hashes[str(file.relative_to(old.ROOT))]=old.sha(file)
            for r in old.read_rows(file):
                c['frames']+=1;fallback=r['backup_used'] and not r['direct_return']
                if fallback:
                    st=[d['status'] for d in r['native_status']]
                    c['non_direct_backup_return']+=1;c['backup_not_accepted']+=not r['backup_accepted']
                    c['initial_pair_illegal']+=not r['initial_pair_legal']
                    c['rank_condition_insufficient']+=any('rank_deficient' in v for v in st)
                    c['convex_unsuccessful']+=any(d.get('solver_called') and d['status'] not in ('Solved','AlmostSolved') for d in r['native_status'])
                    c['solved_but_not_selected_nonlinear_or_demand_ambiguous']+=any(v in ('Solved','AlmostSolved') for v in st)
                    c['returned_initial_pair_legal_demand_unmet']+=r['initial_pair_legal'] and not r['demand_met']
                    states.update(st)
                if s['uid']==FOCUS_UID and r['frame']==s['first_failure_frame']:
                    focus.append({k:r.get(k) for k in ('robot','uid','site_id','repeat','frame','previous_q','q','failure_kind',
                        'position_error','orientation_error','initial_pair_legal','initial_pair_gamma','demand','native_status','backup_accepted')})
        result.append(dict(robot=robot,**c,native_status_occurrences=dict(states),
            nonlinear_rejection_count=None,geometrically_legal_improving_partial_trial_rejected_count=None))
    data=dict(utc=old.utc(),counts=result,focus=focus,existing_first_failure_rows=old_failure_rows,
        sources=source,raw_sources=raw_hashes,solver_calls=0,
        interpretation='Counts are overlapping observed flags, not a mutually exclusive causal partition. Old logs omit rejected trial q/z and trial costs: nonlinear geometry failure cannot be split from unmet demand, nor can useful partial objective improvement be retrospectively counted. No replay or new search used.')
    old.write_json(root/'protocol/preflight.json',data)
    csv_write(root/'protocol/old_fallback_counts.csv',result)
    print(json.dumps(result,indent=2))


def prepare():
    previous.verify_seal();cfg,base,root=configurations();prior=old.ROOT/cfg['previous_output']
    (root/'protocol').mkdir(parents=True,exist_ok=False);preflight(root,prior)
    assert cfg['objective_comparison_atol']==OBJECTIVE_ATOL and cfg['objective_comparison_rtol']==OBJECTIVE_RTOL
    for name in ('online_inputs.json','demand_parameters.json','dependencies.json'):
        # Re-serialization is deterministic; demand values and target identities are unchanged.
        value=json.loads((prior/'protocol'/name).read_text());old.write_json(root/'protocol'/name,value)
        assert old.sha(root/'protocol'/name)==old.sha(prior/'protocol'/name)
    items=json.loads((root/'protocol/online_inputs.json').read_text())
    assert items==previous.load_development(base)
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_crik_elastic.py','tests/test_crik_minimal.py'],
                          cwd=old.ROOT,text=True,capture_output=True,check=True)
    old.write_json(root/'protocol/tests.json',dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    protocol=root/'protocol'
    old.write_json(protocol/'selection_seal.json',dict(utc=old.utc(),configuration=cfg,code_hashes=hashes(),
        prior_manifest_sha256=old.sha(prior/'delivery_manifest.json'),
        files={p.name:old.sha(p) for p in protocol.iterdir() if p.is_file()},
        state='fixed before comparative outcomes; all mu values retained; no formal targets opened'))


def verify_seal(require_commit=True):
    previous.verify_seal();cfg,_,root=configurations();protocol=root/'protocol'
    seal=json.loads((protocol/'selection_seal.json').read_text())
    assert seal['configuration']==cfg and seal['code_hashes']==hashes()
    assert seal['prior_manifest_sha256']==old.sha(old.ROOT/cfg['previous_output']/'delivery_manifest.json')
    for name,h in seal['files'].items():assert old.sha(protocol/name)==h,name
    assert not subprocess.check_output(['git','diff',cfg['baseline_commit'],'--name-status','--diff-filter=DMRT'],cwd=old.ROOT,text=True)
    if require_commit:
        paths=list(hashes())+[str(protocol.relative_to(old.ROOT))]
        for extra in ([],['--cached']):
            assert not subprocess.check_output(['git','diff',*extra,'--name-only','--',*paths],cwd=old.ROOT,text=True)
        assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*paths],cwd=old.ROOT,text=True)
    return seal


def factory(method,robot,base,parameters):
    if method not in ELASTIC:return previous.factory(method,robot,base,parameters)
    source,kin,v,urdf=old.context(robot,base)
    solver=ElasticInterventionIK(kin,v,source,str(old.ROOT/base['native_trac_library']),urdf,
        demand_parameters=parameters[robot],mu=ELASTIC[method],config=base['optimizer'])
    return solver,kin,v


def add_offline_metrics(rows,summary,kin,v):
    previous.add_offline_metrics(rows,summary,kin,v)
    if rows[0]['method'] not in ELASTIC:return
    def avg(key):return float(np.mean([r[key] for r in rows]))
    legal_pairs=[r for r in rows if r['initial_pair_legal'] and r['initial_xi_actual'] is not None and r['xi_actual'] is not None]
    partial=[r for r in rows if r['partial_correction_adopted']]
    summary.update(optimization_call_rate=avg('optimization_called'),mean_conic_calls=avg('conic_calls'),
        partial_correction_rate=avg('partial_correction_adopted'),effective_command_adjustment_rate=avg('effective_command_adjustment'),
        geometric_recovery_rate=float(np.mean([r['decision']=='geometric_recovery' for r in rows])),
        actual_shortfall_reduction_mean=float(np.mean([r['actual_shortfall_reduction'] for r in legal_pairs])) if legal_pairs else None,
        normalized_shortfall_reduction_mean=float(np.mean([r['actual_shortfall_reduction']/r['demand'] for r in legal_pairs])) if legal_pairs else None,
        partial_shortfall_reduction_mean=float(np.mean([r['actual_shortfall_reduction'] for r in partial])) if partial else None,
        fallback_counts=dict(Counter(r['fallback_reason'] for r in rows if r['fallback_reason'])),
        decision_counts=dict(Counter(r['decision'] for r in rows)))


def run(robot):
    seal=verify_seal();cfg,base,root=configurations();folder=root/robot
    folder.mkdir(exist_ok=False);(folder/'runs').mkdir()
    items=[i for i in json.loads((root/'protocol/online_inputs.json').read_text()) if i['robot']==robot]
    parameters=json.loads((root/'protocol/demand_parameters.json').read_text())
    old.write_json(folder/'started.json',dict(utc=old.utc(),code_hashes=hashes(),configuration=cfg,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    old.write_json(folder/'online_inputs.json',items)
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
                solver.solve(pose.position,pose.rotation,q,.02)
            solver,kin,v=solvers[method];rows,summary=old.execute_trajectory(solver,kin,v,item,method,repeat)
            add_offline_metrics(rows,summary,kin,v);record=folder/'runs'/f'{rid}.jsonl.gz'
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
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','check']);p.add_argument('--robot',choices=['panda','ur5e'])
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':verify_seal();print('Elastic protocol and preserved evidence verified.')
    else:
        if args.robot is None:p.error('--robot required')
        run(args.robot)


if __name__=='__main__':main()
