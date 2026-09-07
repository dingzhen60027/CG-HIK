"""Trajectory-unit reporting and independent verification; no solver calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json

import numpy as np

from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import csv_write,digest,json_write
from ..types import IKQuery,Pose
from .tolerance_comparison import ToleranceComparison,METHODS,FAMILIES,FRAMES,DT,now,run_summary
from .tolerance_solvers import feedback


def dist(values, scale=1.):
    a=np.asarray([v for v in values if v is not None],float)/scale
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),
        p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()))


def summarize(method, runs, rows):
    assert runs and rows
    repeats=sorted({r['repeat'] for r in runs});uids=sorted({r['uid'] for r in runs})
    assert len(runs)==len(repeats)*len(uids)
    complete={uid:sum(r['complete'] for r in runs if r['uid']==uid) for uid in uids}
    counts=[sum(r['complete'] for r in runs if r['repeat']==rep) for rep in repeats]
    deadline=[sum(r['complete_within_20ms'] for r in runs if r['repeat']==rep) for rep in repeats]
    per_sweep=[sum(r['total_latency_ns'] for r in runs if r['repeat']==rep)/1e6 for rep in repeats]
    accepted=[r for r in rows if r['accepted']]
    return dict(method=method,trajectories=len(uids),repeats=len(repeats),
        completion_counts_by_repeat=counts,mean_completion_count=float(np.mean(counts)),
        trajectory_completion_rate=float(np.mean([r['complete'] for r in runs])),
        deadline_completion_counts_by_repeat=deadline,
        mean_deadline_completion_count=float(np.mean(deadline)),
        all_repeat_completed_uids=[u for u in uids if complete[u]==len(repeats)],
        any_repeat_completed_uids=[u for u in uids if complete[u]>0],
        zero_repeat_completed_uids=[u for u in uids if complete[u]==0],
        completion_repeats_per_uid=complete,
        verified_frame_rate=len(accepted)/len(rows),frame_count=len(rows),
        frame_latency_ms=dist([r['total_latency_ns'] for r in rows],1e6),
        within_5ms_rate=float(np.mean([r['returned_within_5ms'] for r in rows])),
        within_20ms_rate=float(np.mean([r['returned_within_20ms'] for r in rows])),
        accepted_within_20ms_rate=float(np.mean([r['accepted'] and r['returned_within_20ms'] for r in rows])),
        cumulative_ms_by_repeat=per_sweep,mean_cumulative_ms_per_sweep=float(np.mean(per_sweep)),
        cumulative_latency_per_trajectory_ms=dist([r['total_latency_ns'] for r in runs],1e6),
        accepted_position_error_m=dist([r['position_error'] for r in accepted]),
        accepted_orientation_error_rad=dist([r['orientation_error'] for r in accepted]),
        all_returned_position_error_m=dist([r['position_error'] for r in rows]),
        all_returned_orientation_error_rad=dist([r['orientation_error'] for r in rows]),
        velocity_utilization=dist([r['velocity_utilization'] for r in rows]),
        accepted_velocity_utilization=dist([r['velocity_utilization'] for r in accepted]),
        native_or_internal_nonconvergence=sum(not r['internal_ok'] for r in rows),
        public_accept_despite_internal_nonconvergence=sum(r['accepted'] and not r['internal_ok'] for r in rows),
        native_return_codes=dict(Counter(str(r['native_return_code']) for r in rows)),
        rejection_reasons=dict(Counter(reason for r in rows if not r['accepted'] for reason in r['verification_reasons'])),
        first_failure_kinds=dict(Counter(r['first_failure_kind'] for r in runs if not r['complete'])),
        dls_solver_fev=dist([r['solver_function_evaluations'] for r in rows]),
        native_fev=None,
        statistics='whole trajectory is independent unit; repeats nested; pooled frame quantiles descriptive only')


def compare_completion(base, task, label):
    assert base['trajectories']==task['trajectories']
    a=base['completion_repeats_per_uid'];b=task['completion_repeats_per_uid']
    assert set(a)==set(b)
    na,nb=base['repeats'],task['repeats']
    return dict(comparison=label,base=base['method'],task=task['method'],
        recovered_uids=[u for u in a if b[u]*na>a[u]*nb],
        lost_uids=[u for u in a if b[u]*na<a[u]*nb],
        stable_recovered_uids=[u for u in a if a[u]==0 and b[u]==nb],
        stable_lost_uids=[u for u in a if a[u]==na and b[u]==0],
        mean_completion_count_difference=task['mean_completion_count']-base['mean_completion_count'],
        cumulative_latency_ratio=task['mean_cumulative_ms_per_sweep']/base['mean_cumulative_ms_per_sweep'],
        definition='compare per-trajectory completion fractions, not paired stochastic seeds; stable means all planned repeats, not probability-one guarantee')


def verify_row(study, row, previous, position, rotation):
    assert np.array_equal(row['previous_q'],previous)
    assert np.array_equal(row['target_position'],position) and np.array_equal(row['target_rotation'],rotation)
    q=IKQuery(Pose(position,rotation),np.asarray(previous),row['dt'])
    raw=np.asarray(row['q']) if row['q'] is not None else np.full(study.kin.nq,np.nan)
    check=study.verifier.check(raw,q)
    assert check.accepted==row['accepted'] and check.finite_ok==row['finite']
    assert list(check.reasons)==row['verification_reasons']
    if check.finite_ok:
        assert check.position_error==row['position_error'] and check.orientation_error==row['orientation_error']
        step=study.kin.limits.velocity*q.dt+study.verifier.config.velocity_tolerance
        assert float(np.max(np.abs(study.kin.difference(raw,previous))/step))==row['velocity_utilization']
    assert row['returned_within_5ms']==(row['total_latency_ns']<=5_000_000)
    assert row['returned_within_20ms']==(row['total_latency_ns']<=20_000_000)
    assert row['total_latency_ns']>=row['conversion_ns']+row['solve_ns']+row['verification_ns']
    return feedback(previous,row)


def build(root='.'):
    study=ToleranceComparison(root);study.sealed();out=study.out
    assert json.loads((out/'online_completed.json').read_text())['runs']==560
    online={i['uid']:i for i in json.loads((out/'online_targets.json').read_text())}
    fixed={i['input_id']:i for i in json.loads((out/'fixed_inputs.json').read_text())}
    fixed_rows=json.loads((out/'fixed_records.json').read_text())
    assert len(fixed_rows)==748
    for row in fixed_rows:
        i=fixed[row['input_id']]
        verify_row(study,row,i['previous_q'],i['target_position'],i['target_rotation'])
        assert query_digest(IKQuery(Pose(i['target_position'],i['target_rotation']),np.asarray(i['previous_q']),DT))==row['query_hash']
    fixed_table=[]
    for input_id,i in fixed.items():
        for method in METHODS:
            rr=sorted([r for r in fixed_rows if r['input_id']==input_id and r['method']==method],key=lambda r:r['repeat'])
            assert len(rr)==(5 if method.startswith('trac') else 1)
            fixed_table.append(dict(input_id=input_id,site_id=i['site_id'],uid=i['uid'],method=method,
                repeats=len(rr),accepted_count=sum(r['accepted'] for r in rr),
                internal_success_count=sum(r['internal_ok'] for r in rr),
                native_return_codes=[r['native_return_code'] for r in rr],
                position_error_m=dist([r['position_error'] for r in rr]),orientation_error_rad=dist([r['orientation_error'] for r in rr]),
                velocity_utilization=dist([r['velocity_utilization'] for r in rr]),
                latency_ms=dist([r['total_latency_ns'] for r in rr],1e6),
                failure_reasons=[r['failure_kind'] for r in rr],
                known_legal_command_witness=i['known_legal_command_witness']))
    fixed_summary=[]
    for m in METHODS:
        rr=[r for r in fixed_table if r['method']==m]
        fixed_summary.append(dict(method=m,inputs=len(rr),
            all_repeat_passed=sum(r['accepted_count']==r['repeats'] for r in rr),
            all_repeat_failed=sum(r['accepted_count']==0 for r in rr),
            mixed=sum(0<r['accepted_count']<r['repeats'] for r in rr),
            accepted_calls=sum(r['accepted_count'] for r in rr),calls=sum(r['repeats'] for r in rr)))
    runs=json.loads((out/'online_run_summaries.json').read_text())
    assert len(runs)==len({r['run_id'] for r in runs})==560
    assert len(list((out/'online_runs').glob('*.jsonl.gz')))==560
    metric_rows=defaultdict(list);failure_table=[];witness_index=[];checked=0
    keep=['accepted','total_latency_ns','returned_within_5ms','returned_within_20ms','position_error',
          'orientation_error','velocity_utilization','internal_ok','native_return_code','verification_reasons','solver_function_evaluations']
    for run in runs:
        with gzip.open(out/'online_runs'/f'{run["run_id"]}.jsonl.gz','rt',encoding='utf8') as f:
            rows=[json.loads(line) for line in f]
        assert len(rows)==FRAMES
        i=online[run['uid']];previous=np.asarray(i['initial_q'])
        assert np.array_equal(run['initial_q'],previous)
        for t,row in enumerate(rows):
            assert row['frame']==t and row['method']==run['method'] and row['repeat']==run['repeat']
            previous=verify_row(study,row,previous,i['target_position'][t],i['target_rotation'][t])
            assert np.array_equal(previous,row['accepted_state_q'])
            metric_rows[run['method']].append(dict(family=run['family'],**{k:row[k] for k in keep}))
            checked+=1
        computed=run_summary(rows)
        assert all(computed[k]==run[k] for k in computed)
        if run['complete']:
            witness_index.append(dict(run_id=run['run_id'],uid=run['uid'],method=run['method'],repeat=run['repeat'],
                source=f'online_runs/{run["run_id"]}.jsonl.gz',verified_frames=FRAMES,initial_q=run['initial_q']))
        else:
            failure_table.append(dict(run['first_failure']))
    assert checked==84000
    reference_checks=0
    ids=json.loads((out/'trajectory_identities.json').read_text())
    with np.load(out/'reference_trajectories.npz',allow_pickle=False) as ds:
        for si,identity in enumerate(ids):
            for t in range(FRAMES):
                j=si*FRAMES+t
                q=IKQuery(Pose(ds['target_position'][j],ds['target_rotation'][j]),ds['previous_q'][j],DT)
                assert study.verifier.check(ds['reference_q'][j],q).accepted
                assert query_digest(q)==identity['query_hashes'][t]
                assert np.array_equal(online[identity['uid']]['target_position'][t],q.target.position)
                assert np.array_equal(online[identity['uid']]['target_rotation'][t],q.target.rotation)
                if t==0: assert np.array_equal(online[identity['uid']]['initial_q'],q.previous_q)
                reference_checks+=1
    main=[summarize(m,[r for r in runs if r['method']==m],metric_rows[m]) for m in METHODS]
    family=[dict(family=f,**summarize(m,[r for r in runs if r['method']==m and r['family']==f],
        [r for r in metric_rows[m] if r['family']==f])) for f in FAMILIES for m in METHODS]
    lookup={r['method']:r for r in main}
    pairs=[('trac_strict_5ms','trac_task_5ms'),('trac_strict_20ms','trac_task_20ms'),
           ('dls_strict','dls_task'),('trac_task_5ms','dls_task'),('trac_task_20ms','dls_task')]
    comparisons=[compare_completion(lookup[a],lookup[b],f'{b}_vs_{a}') for a,b in pairs]
    repetition=[]
    for m in METHODS:
        for repeat in range(3 if m.startswith('trac') else 1):
            rr=[r for r in runs if r['method']==m and r['repeat']==repeat]
            repetition.append(dict(method=m,repeat=repeat,completion_count=sum(r['complete'] for r in rr),
                deadline_completion_count=sum(r['complete_within_20ms'] for r in rr),
                accepted_frames=sum(r['accepted_frames'] for r in rr),
                cumulative_latency_ms=sum(r['total_latency_ns'] for r in rr)/1e6,
                completed_uids=[r['uid'] for r in rr if r['complete']]))
    for name,data in [('fixed_input_table',fixed_table),('fixed_input_summary',fixed_summary),
                      ('main_table',main),('family_table',family),('completion_comparisons',comparisons),
                      ('repeat_table',repetition),('first_failure_table',failure_table),('successful_witness_index',witness_index)]:
        json_write(out/f'{name}.json',data);csv_write(out/f'{name}.csv',data)
    json_write(out/'completion_uid_sets.json',[{k:r[k] for k in ['method','completion_repeats_per_uid',
        'all_repeat_completed_uids','any_repeat_completed_uids','zero_repeat_completed_uids']} for r in main])
    study.sealed()
    json_write(out/'verification.json',dict(utc=now(),online_frames_checked=checked,fixed_input_calls_checked=len(fixed_rows),
        reference_frames_checked=reference_checks,successful_full_witnesses=len(witness_index),
        accepted_contract_violations=0,all_feedback_and_targets_match=True,no_reporting_solver_calls=True,
        protected_files_unchanged=len(study.check()['protected_files']),execution_seal_sha256=digest(out/'execution_seal.json')))
    print(json.dumps(dict(fixed=fixed_summary,main=[{k:r[k] for k in ['method','completion_counts_by_repeat',
        'deadline_completion_counts_by_repeat','verified_frame_rate','frame_latency_ms','mean_cumulative_ms_per_sweep']} for r in main],
        comparison=comparisons),ensure_ascii=False,indent=2))


def seal(root):
    study=ToleranceComparison(root);study.sealed();out=study.out
    verification=json.loads((out/'verification.json').read_text())
    assert verification['accepted_contract_violations']==0
    paths=sorted(p for p in out.rglob('*') if p.is_file())
    paths += [study.root/n for n in ['src/confik/continuation_mechanism/tolerance_comparison.py',
        'src/confik/continuation_mechanism/tolerance_solvers.py','src/confik/continuation_mechanism/tolerance_reporting.py',
        'src/confik/continuation_mechanism/tolerance_native/CMakeLists.txt',
        'src/confik/continuation_mechanism/tolerance_native/tolerance_adapter.cpp',
        'tests/test_tolerance_solver_comparison.py','tests/test_tolerance_reporting.py',
        'docs/TOLERANCE_MATCHED_SOLVER_FINDINGS.md']]
    json_write(out/'delivery_manifest.json',dict(utc=now(),baseline=study.check()['baseline'],
        files={str(p.relative_to(study.root)):digest(p) for p in paths},
        execution_seal_sha256=digest(out/'execution_seal.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--seal',action='store_true')
    a=p.parse_args();(seal if a.seal else build)(a.root)


if __name__=='__main__':main()
