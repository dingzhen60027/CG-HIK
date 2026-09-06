"""Paired timing with explicit offline-witness/online-input separation."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import torch
from ..latency_pilot_v3.benchmark import query_digest, query_from_dataset


def measured_call(method, query, name, *, audit=True):
    gpu = not name.startswith('trac_ik')
    if gpu: torch.cuda.synchronize()
    start = perf_counter_ns()
    out = method.solve(query)
    if gpu: torch.cuda.synchronize()
    elapsed = perf_counter_ns()-start
    verifier = method.verifier
    check = verifier.check(out.q,query) if out.accepted else None
    if out.accepted and (out.q is None or not check.accepted):
        raise AssertionError('accepted command violates unchanged verifier')
    kin = method.kinematics
    decision = getattr(getattr(method,'runtime',method),'last_decision',None)
    row = dict(method=name,latency_ns=elapsed,accepted=bool(out.accepted),accepted_within_20ms=bool(out.accepted and elapsed<=20_000_000),
               accepted_contract_violation=False,fev=None if name.startswith('trac_ik') else int(out.function_evaluations),
               fallback=bool(out.fallback_used),route=out.entry_action,reject_reason=out.reject_reason,
               verification_reasons=list(out.verification_reasons),stages=list(out.executed_stages),stage_latency_ns=out.timings_ns,
               query_hash=query_digest(query),previous_q=query.previous_q.tolist(),command_q=out.q.tolist() if out.q is not None else None,
               position_error=float(check.position_error) if check else None,orientation_error=float(check.orientation_error) if check else None,
               max_joint_step=float(np.max(np.abs(kin.difference(out.q,query.previous_q)))) if out.q is not None else None,
               max_velocity_utilization=float(np.max(np.abs(kin.difference(out.q,query.previous_q))/(kin.limits.velocity*query.dt))) if out.q is not None else None)
    if decision is not None:
        row.update(decision_reason=decision.reason,eligible=list(decision.eligible_actions),p50_prediction=list(decision.predicted_p50_ms),p95_prediction=list(decision.predicted_p95_ms))
    if name.startswith('trac_ik'):
        row['solver_return_code']=method.last_solver_return_code
    return out,row


def benchmark_points(methods,dataset,identities,path,*,repeats=3,seed=960640,indices=None):
    names=list(methods)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(dataset) if indices is None else np.asarray(indices))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'xt',encoding='utf8') as f:
        for qi,i in enumerate(order):
            query=query_from_dataset(dataset,int(i))
            base=list(rng.permutation(names))
            for repeat in range(repeats):
                # Cyclic, randomized per-query interleaving; no method batching.
                shift=(qi+repeat)%len(names)
                sequence=base[shift:]+base[:shift]
                for oi,name in enumerate(sequence):
                    _,row=measured_call(methods[name],query,name)
                    row.update(query_index=int(i),uid=identities[i]['uid'],family=str(dataset.category[i]),witness=bool(dataset.continuity_feasible[i]),repeat=repeat,order_position=oi)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
            if (qi+1)%25==0 or qi+1==len(order):
                f.flush();print(f'{path.name}: queries {qi+1}/{len(order)}',flush=True)


def benchmark_trajectories(methods,dataset,identities,path,*,seed=960650):
    names=list(methods);rng=np.random.default_rng(seed)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'xt',encoding='utf8') as f:
        for ti in rng.permutation(len(identities)):
            ix=np.flatnonzero(dataset.trajectory_id==ti)
            ix=ix[np.argsort(dataset.time_index[ix])]
            states={name:dataset.previous_q[ix[0]].copy() for name in names}
            order=list(rng.permutation(names))
            for t,i in enumerate(ix):
                shift=t%len(names)
                for oi,name in enumerate(order[shift:]+order[:shift]):
                    query=query_from_dataset(dataset,int(i),previous_q=states[name])
                    out,row=measured_call(methods[name],query,name)
                    if out.accepted:states[name]=out.q.copy()
                    row.update(query_index=int(i),trajectory_index=int(ti),uid=identities[ti]['uid'],family=str(dataset.category[i]),frame=int(t),repeat=0,order_position=oi)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush();print(f'{path.name}: trajectory {int(ti)} complete ({len(ix)} targets, all methods)',flush=True)


def read_records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [json.loads(line) for line in f]
