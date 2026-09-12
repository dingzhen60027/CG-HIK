#!/usr/bin/env python3
"""Package-command verification and fixed single-solver development comparison.

Reuses the repository whole-trajectory runner and statistics, without invoking
historical predictive policies or adding a recovery chain.
"""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from confik.correction_reserve import study as runner
from confik.correction_reserve.study import ROOT, context, sha, write_json, clean, utc, read_rows
from confik.correction_reserve.reporting import csv_write, group_table, paired_intervals, LABELS
from confik.types import IKQuery, Pose

OUT=ROOT/'outputs/single_solver_development'
PACKAGE=OUT/'task_package/single_solver'
BACKEND=None
LABELS.update(single_gn_k0='Single bounded GN, kappa=0',single_gn_k1='Single bounded GN, kappa=1')


def configure_backend():
    """Make an unavailable optional accelerator use the supplied NumPy path.

    No installed package is changed. The current environment has an installed
    Numba whose import fails against its coverage dependency. Marking that
    optional import unavailable lets the byte-identical core's ImportError
    branch operate; this choice is fixed before any development measurement.
    """
    global BACKEND
    if BACKEND is None:
        versions={}
        for name in ('numpy','scipy','numba','coverage','pin','pink'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        try:
            import numba
            BACKEND=dict(engine='numba',versions=versions,import_error=None)
        except (ImportError,AttributeError) as error:
            sys.modules['numba']=None
            BACKEND=dict(engine='numpy_supplied_fallback',versions=versions,
                         import_error=type(error).__name__+': '+str(error))
    return BACKEND


def code_hashes():
    names=['src/confik/bounded_gn.py','src/confik/bounded_gn_adapter.py',
        'scripts/run_single_solver_development.py','configs/single_solver_development.yaml',
        'src/confik/correction_reserve/study.py','src/confik/correction_reserve/reporting.py',
        'src/confik/correction_reserve/geometry.py','src/confik/correction_reserve/native_geometry.py',
        'src/confik/correction_reserve/pink_adapter.py','src/confik/task_contract_alignment/outcomes.py',
        'src/confik/task_contract_alignment/trac_adapter.py','src/confik/solvers/verifier.py',
        'tmp/task_contract_build/libcontract_trac.so','configs/paper_v2.yaml']
    return {n:sha(ROOT/n) for n in names}


def check_supplied_sources():
    assert sha(ROOT/'src/confik/bounded_gn.py')==sha(PACKAGE/'bounded_gn.py')
    assert sha(ROOT/'src/confik/bounded_gn_adapter.py')==sha(PACKAGE/'repo_adapter.py')


def factory(method,robot,cfg):
    source,kin,v,urdf=context(robot,cfg)
    if method in ('single_gn_k0','single_gn_k1'):
        backend=configure_backend()
        from confik.bounded_gn import box_qp
        from confik.bounded_gn_adapter import SingleBoundedGN
        weight=0. if method=='single_gn_k0' else 1.
        solver=SingleBoundedGN(kin,v,urdf,posture_weight=weight)
        s=solver.engine.settings
        assert (s.damping,s.posture_weight,s.task_stop,s.maximum_iterations,s.deadline_ms)==(.01,weight,1.,30,20.)
        # Explicitly warm a nonzero box update and the actual full adapter path.
        # Synthetic midpoint targets only; no benchmark result or witness seed.
        n=kin.nq;box_qp(np.eye(n),np.full(n,2.),np.full(n,-.5),np.full(n,.5))
        q=(kin.limits.lower+kin.limits.upper)/2
        target=kin.forward(q+.25*(kin.limits.velocity*.02+v.config.velocity_tolerance))
        warm=[solver.solve(target.position,target.rotation,q,.02) for _ in range(3)]
        solver.metadata=dict(optional_backend=backend,settings=s.__dict__,byte_identical_package_sources=True,
            additional_pre_measurement_warmup=dict(box_qp_calls=1,full_adapter_calls=3,
                evaluation_counts=[r['evaluations'] for r in warm]),
            timing='Unchanged supplied adapter outer time includes conversion, solve and final original verifier; no serialization or warmup. Core 20 ms soft check starts inside adapter, so outer overruns remain possible.')
    elif method=='trac_task_5ms':
        from confik.task_contract_alignment.outcomes import ContractSolver
        solver=ContractSolver(method,kin,v,source,str(ROOT/cfg['native_trac_library']),urdf)
    elif method=='pink_qp':
        from confik.correction_reserve.pink_adapter import PinkAdapter
        solver=PinkAdapter(kin,v,source,urdf)
    else:raise ValueError(method)
    return solver,kin,v


def items(cfg,robot):
    rows=json.loads((ROOT/cfg['development'][robot+'_targets']).read_text())
    assert len(rows)==len({r['uid'] for r in rows})==40
    assert sorted(Counter(r['family'] for r in rows).values())==[10]*4
    for r in rows:assert r['dt']==.02 and len(r['target_position'])==len(r['target_rotation'])==150
    return [dict(robot=robot,**r) for r in rows]


def prepare(cfg):
    check_supplied_sources();folder=OUT/'protocol';folder.mkdir(exist_ok=False)
    verification=json.loads((OUT/'package_verification/summary.json').read_text())
    assert verification['solver_calls']==0 and verification['commands']==18000
    write_json(folder/'protocol.json',dict(created=utc(),config=cfg,code_hashes=code_hashes(),
        optional_backend=configure_backend(),archive_sha256=sha(ROOT/'single_solver_completed_results.zip'),
        public_verification_summary=verification,
        input_files={robot:dict(path=cfg['development'][robot+'_targets'],sha256=sha(ROOT/cfg['development'][robot+'_targets'])) for robot in cfg['robots']},
        identities=[{k:r[k] for k in ('robot','uid','site_id','family','dt','seed')} for robot in cfg['robots'] for r in items(cfg,robot)],
        source_policy='Core and adapter are copied byte for byte. Fixed supplied settings, only kappa0/1 comparison. No fallback and no future inputs.',
        timing='New server measurements only; supplied historical command replay has no server solver timing. Warmup outside timing, final verifier inside timing; no trimming of over-deadline calls.',
        statistics='40 complete UIDs per robot, three nested repeats, average within UID then paired family-stratified bootstrap; 4000 descriptive unadjusted 95% intervals; no equivalence or generalization claim.'))
    print('Fixed 80 original trajectory identities and byte-identical numerical sources.',configure_backend(),flush=True)


def trajectories(cfg,robot):
    check_supplied_sources();fixed=json.loads((OUT/'protocol/protocol.json').read_text())
    assert configure_backend()==fixed['optional_backend']
    for row in fixed['input_files'].values():assert sha(ROOT/row['path'])==row['sha256']
    runner.factory=factory;runner.code_hashes=code_hashes
    runner.run(OUT/f'development_{robot}',items(cfg,robot),cfg['methods'],
               dict(cfg,formal={'order_seed':cfg['order_seed']}),repeats=3)


def report(cfg):
    """Saved-record verification and reuse of the existing statistics functions."""
    check_supplied_sources();folder=OUT/'reports';folder.mkdir(exist_ok=False)
    summaries=[];arrays={};extra={};first=[];witnesses=[];sources={};audit=Counter()
    for robot in cfg['robots']:
        origin=OUT/f'development_{robot}';seal=json.loads((origin/'completed.json').read_text())
        assert seal['runs']==480 and seal['frames']==72000
        for name,digest in seal['files'].items():assert sha(origin/name)==digest
        sources[robot]=dict(path=str(origin.relative_to(ROOT)),completed_sha256=sha(origin/'completed.json'))
        _,kin,v,_=context(robot,cfg);data={r['uid']:r for r in items(cfg,robot)}
        jobs=json.loads((origin/'summaries.json').read_text())
        assert Counter((s['uid'],s['method'],s['repeat']) for s in jobs)==Counter((u,m,r) for u in data for m in cfg['methods'] for r in range(3))
        for index,s in enumerate(jobs):
            rows=read_rows(origin/s['raw_file']);assert len(rows)==150
            target=data[s['uid']];previous=np.array(target['initial_q']);counts=Counter();iterations=[];evaluations=[]
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==target['dt']==.02
                np.testing.assert_array_equal(r['target_position'],target['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],target['target_rotation'][frame])
                np.testing.assert_array_equal(r['previous_q'],previous)
                query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                verdict=v.check(q,query)
                assert verdict.accepted==r['accepted']
                assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
                if np.isfinite(verdict.position_error):
                    np.testing.assert_allclose([r['position_error'],r['orientation_error']],
                        [verdict.position_error,verdict.orientation_error],rtol=0,atol=1e-12)
                audit['commands_checked']+=1;audit['accepted_commands']+=verdict.accepted
                counts['over_20ms_frames']+=r['total_latency_ns']>20_000_000
                if not r['accepted']:counts['failure:'+r['failure_kind']]+=1
                if s['method'].startswith('single_gn'):
                    assert 1<=r['iterations']<=30 and r['evaluations']>=1
                    assert r['box_qp_updates']<=50*30
                    assert not any(k in r for k in ('trac_result','trf_result','nominal_next_verified','backup_q'))
                    counts['internal:'+r['internal_status']]+=1
                    counts['native_task_candidate']+=r['internal_ok']
                    iterations.append(r['iterations']);evaluations.append(r['evaluations'])
                if frame==s['first_failure_frame']:
                    first.append({k:r.get(k) for k in ('robot','uid','site_id','family','method','repeat','frame','dt',
                        'previous_q','target_position','target_rotation','q','accepted','failure_kind','verification_reasons',
                        'position_error','orientation_error','velocity_utilization','internal_status','total_latency_ns')})
                if verdict.accepted:previous=q.copy()
                np.testing.assert_array_equal(r['accepted_state_q'],previous)
            calculated=runner.summarize(rows,kin,v)
            for key in ('completion','deadline_completion','accepted_frames','deadline_frames','total_latency_ns','first_failure_frame'):
                assert calculated[key]==s[key]
            assert s['accepted_contract_violations']==0
            if s['completion']:
                witnesses.append(dict(robot=robot,uid=s['uid'],site_id=s['site_id'],family=s['family'],method=s['method'],repeat=s['repeat'],
                    initial_q=target['initial_q'],frames=150,deadline_completion=s['deadline_completion'],
                    raw_file=str((origin/s['raw_file']).relative_to(ROOT)),sha256=sha(origin/s['raw_file'])))
            summaries.append(s);extra[s['run_id']]=dict(counts=counts,iterations=iterations,evaluations=evaluations)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if index%120==0:print('verify saved commands',robot,index+1,'/480',flush=True)
    assert len(summaries)==960 and audit['commands_checked']==144000
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    for table in (main,family):
        for row in table:
            ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
            counts=Counter();iterations=[];evaluations=[]
            for s in ss:
                ex=extra[s['run_id']];counts.update(ex['counts']);iterations+=ex['iterations'];evaluations+=ex['evaluations']
            for key in ('backup_rate','changed_from_backup_rate'):row.pop(key,None)
            row.update(frame_calls=sum(s['frames'] for s in ss),over_20ms_frames=counts['over_20ms_frames'],
                all_frame_cumulative_ns=sum(s['total_latency_ns'] for s in ss),
                internal_statuses={k.removeprefix('internal:'):n for k,n in counts.items() if k.startswith('internal:')},
                failure_reasons={k.removeprefix('failure:'):n for k,n in counts.items() if k.startswith('failure:')},
                mean_iterations=float(np.mean(iterations)) if iterations else None,
                mean_evaluations=float(np.mean(evaluations)) if evaluations else None,
                max_frame_latency_ms=float(max(np.max(arrays[s['run_id']]['latency']) for s in ss)/1e6))
    units=[];pairs=[];changes=[];completion={}
    metrics=('completion','deadline_completion','total_latency_ns','frame_success','successful_prefix','acceleration_rms')
    comparisons=[('single_gn_k1','single_gn_k0'),('single_gn_k1','trac_task_5ms'),('single_gn_k1','pink_qp'),
                 ('single_gn_k0','trac_task_5ms'),('single_gn_k0','pink_qp')]
    for robot in cfg['robots']:
        ss=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in ss});unit={}
        fam=[next(s['family'] for s in ss if s['uid']==u) for u in uids]
        for method in cfg['methods']:
            completion.setdefault(robot,{})[method]={str(rep):sorted(s['uid'] for s in ss if s['method']==method and s['repeat']==rep and s['completion']) for rep in range(3)}
            for uid,f in zip(uids,fam):
                rr=[s for s in ss if s['uid']==uid and s['method']==method];assert len(rr)==3
                val={k:float(np.mean([r[k] for r in rr])) for k in metrics};unit[(method,uid)]=val
                units.append(dict(robot=robot,uid=uid,site_id=rr[0]['site_id'],family=f,method=method,repeats=3,**val))
        for method,base in comparisons:
            for metric in metrics:
                a=np.array([unit[(method,u)][metric] for u in uids]);b=np.array([unit[(base,u)][metric] for u in uids])
                pairs.append(dict(robot=robot,method=method,baseline=base,metric=metric,
                    **paired_intervals(a,b,fam,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            for uid,f in zip(uids,fam):
                a=unit[(method,uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,method=method,baseline=base,uid=uid,family=f,
                    site_id=next(s['site_id'] for s in ss if s['uid']==uid),method_fraction=a,baseline_fraction=b,
                    change='gained' if a>b else 'lost' if a<b else 'same',stable_all_vs_none=bool(abs(a-b)==1)))
    for name,rows in [('main_table',main),('family_table',family),('trajectory_units',units),('paired_comparisons',pairs),
                      ('gained_lost_uids',changes),('first_failure_inputs',first),('run_summaries',summaries)]:
        csv_write(folder/(name+'.csv'),rows)
    write_json(folder/'source_data.json',dict(main=main,family=family,pairs=pairs,changes=changes,units=units))
    write_json(folder/'completion_uids.json',completion);write_json(folder/'successful_trajectory_index.json',witnesses)
    write_json(folder/'verification.json',dict(counts=dict(audit),accepted_contract_violations=0,successful_runs=len(witnesses),
        solver_calls=0,checks='Original target and actual accepted-state feedback on every frame; original verifier and residuals; complete outer time and all failures retained; 30-update bound; no recovery calls.'))
    write_json(folder/'manifest.json',dict(created=utc(),sources=sources,reporting_sha256=sha(__file__),
        source_policy='All statistics are new original-server calls; historical portable timings are not mixed in.',
        independent_unit='40 trajectory UIDs/robot; average 3 nested repeats, then paired family-stratified bootstrap; descriptive unadjusted 95% intervals',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],4),'cumulative_s',r['cumulative_ms_per_sweep']/1000,flush=True)


def package_verify(cfg):
    """No IK calls: check supplied commands using original targets and feedback."""
    folder=OUT/'package_verification';folder.mkdir(exist_ok=False)
    original=json.loads((ROOT/cfg['development']['panda_targets']).read_text())
    reconstructed=json.loads((PACKAGE/'data/panda_reconstructed.json').read_text())
    originals={r['uid']:r for r in original};rebuilt={r['uid']:r for r in reconstructed}
    assert len(original)==len(originals)==len(reconstructed)==len(rebuilt)==40
    assert set(originals)==set(rebuilt)
    matches=[]
    for uid,o in originals.items():
        r=rebuilt[uid]
        assert (o['dt'],o['site_id'],o['family'],o['seed'])==(r['dt'],r['site_id'],r['family'],r['seed'])
        row=dict(uid=uid,site_id=o['site_id'],frames=150)
        for key in ('initial_q','target_position','target_rotation'):
            a,b=np.array(o[key]),np.array(r[key]);assert a.shape==b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            row[key+'_max_absolute_difference']=float(np.max(np.abs(a-b)))
        assert np.array(o['target_position']).shape==(150,3)
        assert np.array(o['target_rotation']).shape==(150,3,3)
        matches.append(row)
    csv_write(folder/'identity_comparison.csv',matches)
    maximum={key:max(r[key] for r in matches) for key in matches[0] if key.endswith('_difference')}
    # A fixed roundoff-scale identity criterion, not a pose-contract change.
    if max(maximum.values())>1e-12:
        write_json(folder/'identity_mismatch.json',dict(maximum=maximum,solver_calls=0))
        raise RuntimeError('Reconstruction differs materially: no result substitution')
    supplied=json.loads((PACKAGE/'measured/portable_repeated.json').read_text())
    assert len(supplied['runs'])==120
    assert Counter((r['uid'],r['repeat']) for r in supplied['runs'])==Counter((u,i) for u in originals for i in range(3))
    _,kin,v,_=context('panda',cfg)
    runs=[];mismatches=[];commands=0;accepted=0;max_previous_difference=0.
    with gzip.open(folder/'verified_commands.jsonl.gz','xt') as f:
        for record in supplied['runs']:
            o=originals[record['uid']];previous=np.array(o['initial_q']);ok=[]
            assert record['site_id']==o['site_id'] and len(record['frames'])==150
            for frame,row in enumerate(record['frames']):
                assert row['frame']==frame
                difference=float(np.max(np.abs(np.array(row['previous_q'])-previous)))
                max_previous_difference=max(max_previous_difference,difference)
                query=IKQuery(Pose(np.array(o['target_position'][frame]),np.array(o['target_rotation'][frame])),previous,o['dt'])
                q=np.array(row['q']);verdict=v.check(q,query)
                out=dict(uid=o['uid'],site_id=o['site_id'],repeat=record['repeat'],frame=frame,dt=o['dt'],
                    previous_q=previous.copy(),supplied_previous_q=row['previous_q'],previous_difference=difference,
                    target_position=query.target.position,target_rotation=query.target.rotation,q=q,
                    accepted=verdict.accepted,supplied_accepted=row['accepted'],position_error=verdict.position_error,
                    orientation_error=verdict.orientation_error,finite=verdict.finite_ok,joint_limit_ok=verdict.joint_limit_ok,
                    velocity_ok=verdict.velocity_ok,verification_reasons=list(verdict.reasons),
                    velocity_utilization=float(np.max(np.abs(q-previous)/(kin.limits.velocity*o['dt']+v.config.velocity_tolerance))),
                    supplied_local_latency_ns=row['latency_ns'])
                if verdict.accepted!=row['accepted'] or difference>1e-12:mismatches.append(out)
                f.write(json.dumps(clean(out),allow_nan=False,separators=(',',':'))+'\n')
                commands+=1;accepted+=verdict.accepted;ok.append(verdict.accepted)
                if verdict.accepted:previous=q.copy()
            runs.append(dict(uid=o['uid'],site_id=o['site_id'],repeat=record['repeat'],completion=all(ok),
                accepted_frames=sum(ok),first_failure=next((i for i,a in enumerate(ok) if not a),None)))
    assert commands==18000
    summary=dict(solver_calls=0,commands=commands,accepted_commands=accepted,
        completion_by_repeat=[sum(r['completion'] for r in runs if r['repeat']==rep) for rep in range(3)],
        completion_uids={str(rep):[r['uid'] for r in runs if r['repeat']==rep and r['completion']] for rep in range(3)},
        target_identity_maxima=maximum,max_previous_difference=max_previous_difference,mismatches=len(mismatches),
        timing='Supplied latency is local historical time, not measured repository solver time. Original-environment DTSR20 is not inferred from command-only replay.',
        public_backend=type(kin).__name__,verifier_config=v.config.__dict__)
    csv_write(folder/'runs.csv',runs);write_json(folder/'mismatches.json',mismatches)
    write_json(folder/'summary.json',summary)
    write_json(folder/'manifest.json',dict(created=utc(),source_files={str(p.relative_to(ROOT)):sha(p) for p in
        [ROOT/cfg['development']['panda_targets'],PACKAGE/'data/panda_reconstructed.json',PACKAGE/'measured/portable_repeated.json']},
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},solver_calls=0))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify-package','prepare','trajectories','report'])
    p.add_argument('--robot',choices=['panda','ur5e'])
    args=p.parse_args();cfg=yaml.safe_load((ROOT/'configs/single_solver_development.yaml').read_text())
    if args.action=='verify-package':package_verify(cfg)
    elif args.action=='prepare':prepare(cfg)
    elif args.action=='report':report(cfg)
    elif args.action=='trajectories':
        if not args.robot:p.error('--robot required')
        trajectories(cfg,args.robot)


if __name__=='__main__':main()
