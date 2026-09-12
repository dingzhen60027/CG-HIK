#!/usr/bin/env python3
"""Bounded follow-up to single_solver_evidence; no new online method/evaluation."""
import argparse
from collections import Counter,defaultdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('existing_evidence',ROOT/'scripts/run_single_solver_evidence.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from confik.correction_reserve.study import context,read_rows,sha,write_json,clean,utc
from confik.correction_reserve.reporting import csv_write
from confik.bounded_gn import box_qp
from confik.bounded_gn_adapter import SingleBoundedGN
from confik.single_solver_evidence import bind_qp,clipped_qp,CachedOSQP,quality
from confik.constrained_qp_reference import QPOases,activity,qualified,active_set_trace

BASE=ROOT/'outputs/single_solver_evidence'
OUT=BASE/'constrained_qp_mechanism'
LIB=ROOT/'tmp/single_solver_qpoases/build/libbox_reference.so'
CASES={'panda':['trajectory_050','trajectory_052','trajectory_094','trajectory_096'],
       'ur5e':['trajectory_109','trajectory_131']}


def dump_gz(path,rows):
    with gzip.open(path,'xt') as f:
        for row in rows:f.write(json.dumps(clean(row),allow_nan=False)+'\n')


def save_bank(path,arrays):
    with path.open('xb') as f:np.savez_compressed(f,**arrays)


def provenance():
    old.fixed_sources()
    write_json(OUT/'provenance.json',dict(created=utc(),baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        official_source='https://github.com/coin-or/qpOASES',tag='releases/3.2.1',
        upstream_sha=subprocess.check_output(['git','-C','tmp/single_solver_qpoases','rev-parse','HEAD'],text=True).strip(),
        version_note='Release tag 3.2.1; official CMake PACKAGE_VERSION string remains 3.2.0.',
        compiler=subprocess.check_output(['c++','--version'],text=True),
        build_flags=(ROOT/'tmp/single_solver_qpoases/build/CMakeFiles/qpOASES.dir/flags.make').read_text(),
        bridge_build='c++ -O3 -DNDEBUG -std=c++11 -fPIC -shared -I tmp/single_solver_qpoases/include scripts/native/qpoases_box_reference.cpp tmp/single_solver_qpoases/build/libs/libqpOASES.a -o tmp/single_solver_qpoases/build/libbox_reference.so',
        library_sha256=sha(LIB),bridge_sha256=sha(ROOT/'scripts/native/qpoases_box_reference.cpp'),
        native_options='Options::setToDefault(), printLevel=PL_NONE; double precision; nWSR=1000; SQProblem(n,0,HST_POSDEF); no retry',
        old_frozen_sources=old.FIXED,cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu=subprocess.check_output(['lscpu'],text=True),
        numerical_backend=old.old.configure_backend(),numpy=np.__version__,
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        official_api='https://github.com/coin-or/qpOASES/blob/51d3fbea30142d3acbf40cf7a1c519efc27ea67b/include/qpOASES/SQProblem.hpp',
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/confik/constrained_qp_reference.py',Path(__file__)]}))


def capture(cfg,robot):
    folder=OUT/'capture';folder.mkdir(exist_ok=True)
    source,kin,v,urdf=context(robot,cfg);solver=SingleBoundedGN(kin,v,urdf)
    matrices=[];meta=[];frame_log=[];pending=[];source_hashes={}
    def callback(H,g,lo,hi):
        local=sys._getframe(1).f_locals
        d,it=box_qp(H,g,lo,hi)
        pending.append(dict(H=H.copy(),g=g.copy(),lo=lo.copy(),hi=hi.copy(),d=d.copy(),
            q=local['q'].copy(),step=local['step'].copy(),e=local['e'].copy(),
            outer_iteration=local['iteration'],damping=local['lam'],updates=it))
        return d,it
    bind_qp(solver.engine,callback)
    paths=sorted((BASE/f'trajectories_{robot}/runs').glob(f'{robot}_*_single_gn_k1_r0.jsonl.gz'))
    assert len(paths)==160
    for pi,path in enumerate(paths):
        source_hashes[str(path.relative_to(ROOT))]=sha(path)
        last_state=None
        for original in read_rows(path):
            pending.clear()
            result=solver.solve(original['target_position'],original['target_rotation'],original['previous_q'],original['dt'])
            diff=float(np.max(np.abs(np.array(result['q'])-original['q'])))
            frame_log.append({k:original[k] for k in ['robot','uid','site_id','family','frame'] }|
                dict(qp_calls=len(pending),q_max_difference=diff,original_status=original['internal_status'],
                     replay_status=result['internal_status'],original_accepted=original['accepted'],
                     replay_accepted=result['accepted'],diagnostic_only_latency_ns=result['total_latency_ns']))
            # All classification and counterfactual work is outside the unchanged solve.
            for p in pending:
                H,g,lo,hi,d=(p[k] for k in ['H','g','lo','hi','d'])
                a=activity(H,g,lo,hi,d);state=np.array(a['state']);active=state!=0
                end=p['q']+p['step']*d
                physical=active & ((abs(end-kin.limits.lower)<1e-8)|(abs(end-kin.limits.upper)<1e-8))
                rate=active & (abs(abs(end-original['previous_q'])-p['step'])<1e-8)
                trust=active & (abs(abs(d)-1)<1e-8) & ~physical & ~rate
                changed=(state!=last_state) if last_state is not None else np.zeros_like(state,bool)
                valid_change=last_state is not None and a['qualified']
                info={k:original[k] for k in ['robot','uid','site_id','family','frame','dt']}
                info.update(qp_index=len(meta),outer_iteration=p['outer_iteration'],damping=p['damping'],
                    updates=p['updates'],**a,physical_joints=np.where(physical)[0].tolist(),
                    rate_joints=np.where(rate)[0].tolist(),trust_joints=np.where(trust)[0].tolist(),
                    sequence_additions=int(np.sum(changed & active)) if valid_change else 0,
                    sequence_releases=int(np.sum(changed & (last_state!=0))) if valid_change else 0,
                    original_frame_q_difference=diff)
                matrices.append({k:p[k] for k in ['H','g','lo','hi','d','q','step','e']})
                meta.append(info);last_state=state if a['qualified'] else None
        if (pi+1)%20==0:print('capture',robot,pi+1,'UIDs',len(meta),'QP',flush=True)
    keys=matrices[0].keys()
    save_bank(folder/f'{robot}_all_qp.npz',{k:np.array([m[k] for m in matrices]) for k in keys})
    dump_gz(folder/f'{robot}_chronological_metadata.jsonl.gz',meta)
    dump_gz(folder/f'{robot}_frame_replay.jsonl.gz',frame_log)
    candidates=[i for i,m in enumerate(meta) if m['active_count']>0 or m['updates']>1 or not m['qualified']]
    rng=np.random.default_rng(2026091501);byuid=defaultdict(list)
    for i in candidates:byuid[meta[i]['uid']].append(i)
    for uid in byuid:rng.shuffle(byuid[uid])
    selected=[]
    while len(selected)<min(2000,len(candidates)):
        for uid in sorted(byuid):
            if byuid[uid] and len(selected)<2000:selected.append(byuid[uid].pop())
    selected=sorted(selected)
    write_json(folder/f'{robot}_selection.json',dict(indices=selected,candidate_count=len(candidates),
        selected_count=len(selected),all_qp=len(meta),source_hashes=source_hashes,
        census_counts=dict(Counter('unqualified' if not m['qualified'] else str(m['active_count']) for m in meta)),
        changed_output_frames=sum(r['q_max_difference']>1e-8 for r in frame_log),
        changed_acceptance_frames=sum(r['original_accepted']!=r['replay_accepted'] for r in frame_log),
        full_replay_not_performance_evaluation=True))
    print('captured',robot,len(meta),'targeted',len(selected),'of',len(candidates),flush=True)


def bank(robot,cohort):
    if cohort=='natural':
        with np.load(BASE/f'qp_benchmark/{robot}_qp_matrices.npz') as b:arrays={k:b[k] for k in b.files}
        meta=json.loads((BASE/f'qp_benchmark/{robot}_qp_metadata.json').read_text())
    else:
        ids=json.loads((OUT/f'capture/{robot}_selection.json').read_text())['indices']
        with np.load(OUT/f'capture/{robot}_all_qp.npz') as b:arrays={k:b[k][ids] for k in b.files}
        full=read_rows(OUT/f'capture/{robot}_chronological_metadata.jsonl.gz')
        meta=[full[i] for i in ids]
    return arrays,meta


def benchmark(tag='benchmark',osqp_cap=20000):
    folder=OUT/tag;folder.mkdir(exist_ok=False)
    write_json(folder/'settings.json',dict(osqp_max_iter=osqp_cap,eps_abs=1e-8,eps_rel=0.,
        rationale='Only increase iteration cap to reach the unchanged common QP quality; no online solver change.'))
    setups=[];classification=[]
    for robot in ['panda','ur5e']:
        for cohort in ['natural','targeted']:
            arrays,meta=bank(robot,cohort);n=arrays['g'].shape[1]
            references=[]
            for i,m in enumerate(meta):
                args=[arrays[k][i] for k in ['H','g','lo','hi']]
                x,it=box_qp(*args);a=activity(*args,x)
                xt,itt,events=active_set_trace(box_qp,*args)
                assert np.array_equal(x,xt) and it==itt
                references.append(a['quality'])
                classification.append(dict(m,cohort=cohort,index=i,
                    reference_activity=a,internal_events=events,
                    internal_additions=sum(e['iteration']>=0 and e['after']!=0 for e in events),
                    internal_releases=sum(e['iteration']>=0 and e['after']==0 for e in events)))
            # Warm-up: actual first ten QPs; no timed input is dropped afterwards.
            qs=QPOases(n,LIB);osqp=CachedOSQP(n)
            osqp.solver.update_settings(max_iter=osqp_cap)
            for i in range(min(10,len(meta))):
                args=[arrays[k][i] for k in ['H','g','lo','hi']];qs(*args);osqp(*args)
            qs.close()
            records=[]
            for repeat in range(5):
                methods=['active_numpy','osqp_cached','qpoases','clip']
                order=methods[repeat%4:]+methods[:repeat%4]
                for method in order:
                    s=QPOases(n,LIB) if method=='qpoases' else CachedOSQP(n) if method=='osqp_cached' else None
                    if method=='osqp_cached':s.solver.update_settings(max_iter=osqp_cap)
                    setups.append(dict(robot=robot,cohort=cohort,repeat=repeat,method=method,
                                       initialization_ns=s.initialization_ns if s else 0))
                    last_uid=None
                    for i,m in enumerate(meta):
                        args=[arrays[k][i] for k in ['H','g','lo','hi']]
                        H,g,lo,hi=args
                        start=perf_counter_ns();initial=m['uid']!=last_uid
                        if initial and s is not None:s.reset()
                        reset_end=perf_counter_ns()
                        if s is None:
                            x,it=(box_qp if method=='active_numpy' else clipped_qp)(*args)
                            raw=x;extra={};native=perf_counter_ns()-reset_end
                        else:
                            x,it=s(*args);extra=dict(s.last)
                            raw=extra.pop('raw_x',x)
                            native=extra.get('native_solve_ns',extra.get('solve_ns'))
                        returned=perf_counter_ns()
                        check=quality(*args,x);rawcheck=quality(*args,raw)
                        gap=check['objective']-references[i]['objective']
                        scaled=abs(gap)/max(1,abs(references[i]['objective']))
                        ok=qualified(check) and qualified(rawcheck) and scaled<=1e-8
                        checked=perf_counter_ns();last_uid=m['uid']
                        records.append(dict(robot=robot,cohort=cohort,index=i,uid=m['uid'],site_id=m['site_id'],
                            frame=m['frame'],outer_iteration=m['outer_iteration'],repeat=repeat,method=method,
                            first_in_uid=initial,reset_ns=reset_end-start,kernel_ns=native,
                            return_ns=returned-start,common_check_ns=checked-returned,total_ns=checked-start,
                            **check,raw_projected_kkt=rawcheck['projected_kkt'],
                            raw_normalized_kkt=rawcheck['normalized_kkt'],raw_box_violation=rawcheck['box_violation'],
                            objective_gap=gap,scaled_objective_gap=scaled,quality_pass=ok,
                            reference_qualified=qualified(references[i]),x=x.tolist(),raw_x=raw.tolist(),
                            native_iterations=it,native=extra))
                    if method=='qpoases':s.close()
                print('micro',robot,cohort,'pass',repeat+1,flush=True)
            dump_gz(folder/f'{robot}_{cohort}_calls.jsonl.gz',records)
    write_json(folder/'setups.json',setups);dump_gz(folder/'classification.jsonl.gz',classification)


def mechanism(cfg):
    from confik.types import Pose,IKQuery
    from confik.correction_reserve.geometry import residual_linearization
    folder=OUT/'mechanism';folder.mkdir(exist_ok=False)
    cases=[]
    for robot,sites in CASES.items():
        source,kin,v,urdf=context(robot,cfg);solver=SingleBoundedGN(kin,v,urdf)
        with np.load(OUT/f'capture/{robot}_all_qp.npz') as b:a={k:b[k] for k in b.files}
        meta=read_rows(OUT/f'capture/{robot}_chronological_metadata.jsonl.gz')
        for site in sites:
            historical={m:read_rows(BASE/f'trajectories_{robot}/runs/{robot}_{site}_{m}_r0.jsonl.gz') for m in ['single_gn_k1','single_gn_clip']}
            record=None
            for i,m in enumerate(meta):
                if m['site_id']!=site:continue
                H,g,lo,hi,d,q,step=[a[k][i] for k in ['H','g','lo','hi','d','q','step']]
                clip,_=clipped_qp(H,g,lo,hi)
                if np.max(abs(d-clip))<=1e-6:continue
                row=historical['single_gn_k1'][m['frame']]
                other=historical['single_gn_clip'][m['frame']]
                target=Pose(np.array(row['target_position']),np.array(row['target_rotation']))
                query=IKQuery(target,np.array(row['previous_q']),row['dt'])
                e,A,_=residual_linearization(solver.native,target,q,solver.scale)
                cost=.5*e@e;trials={}
                for label,direction in [('bounded',d),('clip',clip)]:
                    trials[label]=[]
                    for k in range(8):
                        qq=q+(2.**(-k))*step*direction
                        er,_,_=residual_linearization(solver.native,target,qq,solver.scale)
                        verdict=v.check(qq,query)
                        trials[label].append(dict(alpha=2.**(-k),q=qq.tolist(),normalized_residual=er.tolist(),
                            task_cost=.5*float(er@er),decrease=float(cost-.5*er@er),
                            line_search_accept=bool(.5*er@er<cost-1e-12),
                            task_verified=bool(verdict.accepted),position_error=verdict.position_error,
                            orientation_error=verdict.orientation_error,verifier_reasons=list(verdict.reasons)))
                unconstrained=np.linalg.solve(H,-g);free=(unconstrained>=lo)&(unconstrained<=hi)
                outcomes={}
                for method in historical:
                    outcomes[method]=[]
                    for r in range(3):
                        rows=historical[method] if r==0 else read_rows(BASE/f'trajectories_{robot}/runs/{robot}_{site}_{method}_r{r}.jsonl.gz')
                        outcomes[method].append(dict(repeat=r,complete=all(x['accepted'] for x in rows),
                            first_failure=next((x['frame'] for x in rows if not x['accepted']),None)))
                record=dict(**m,H=H.tolist(),g=g.tolist(),lo=lo.tolist(),hi=hi.tolist(),
                    linearization_q=q.tolist(),step=step.tolist(),bounded_direction=d.tolist(),
                    clipped_direction=clip.tolist(),unconstrained_direction=unconstrained.tolist(),
                    free_joint_redistribution=(d-clip)[free].tolist(),free_joints=np.where(free)[0].tolist(),
                    before_normalized_residual=e.tolist(),before_cost=float(cost),trials=trials,
                    previous_q=row['previous_q'],target_position=row['target_position'],target_rotation=row['target_rotation'],
                    other_method_previous_q=other['previous_q'],
                    previous_q_max_difference=float(np.max(abs(np.array(row['previous_q'])-other['previous_q']))),
                    historical_outcomes=outcomes,
                    interpretation='The two directions share this exact model and linearization. Other method closed-loop histories are not substituted; their later outcomes are association, not a single-cause proof.')
                break
            assert record is not None,(robot,site)
            cases.append(record)
            dump_gz(folder/f'{robot}_{site}_historical_rows.jsonl.gz',[dict(x) for method in historical for x in historical[method]])
    write_json(folder/'six_same_input_cases.json',cases)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['provenance','capture','benchmark','mechanism','report']);p.add_argument('--robot',choices=['panda','ur5e'])
    p.add_argument('--tag',default='benchmark');p.add_argument('--osqp-max-iter',type=int,default=20000)
    args=p.parse_args();os.sched_setaffinity(0,{4});old.fixed_sources()
    cfg=yaml.safe_load((ROOT/'configs/single_solver_evidence.yaml').read_text())
    if args.stage=='provenance':provenance()
    elif args.stage=='capture':capture(cfg,args.robot)
    elif args.stage=='benchmark':benchmark(args.tag,args.osqp_max_iter)
    elif args.stage=='mechanism':mechanism(cfg)
    elif args.stage=='report':
        from confik.constrained_qp_reporting import report
        report(OUT)


if __name__=='__main__':main()
