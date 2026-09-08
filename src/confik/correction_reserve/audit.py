"""Replay FK/acceptance only: no new IK searches or updated evidence records."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
import yaml
from ..types import Pose,IKQuery
from ..geometry import pose_error
from .geometry import predict_target,task_scale
from .study import ROOT,context,read_rows,write_json,clean,sha,utc,code_hashes
from .reporting import csv_write


def audit(folders,out,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'outputs/correction_reserve_ik/formal_protocol'
    seal=json.loads((protocol/'selection_seal.json').read_text())
    assert seal['code_hashes']==code_hashes()
    identities=json.loads((protocol/'identities.json').read_text())
    by_uid={r['uid']:r for r in identities}
    modified=subprocess.check_output(['git','diff',cfg['baseline_commit'],'--name-only','--diff-filter=DMRT'],cwd=ROOT,text=True)
    assert not modified,'old tracked evidence or source was changed'
    failures=[];prediction_rows=[];abrupt=[];phase=defaultdict(list);job_counts=[];witnesses=[]
    for folder in map(Path,folders):
        assert (folder/'completed.json').exists()
        inputs=json.loads((folder/'online_inputs.json').read_text());robot=inputs[0]['robot']
        source,kin,v,_=context(robot,cfg);scale=task_scale(v)
        summaries=json.loads((folder/'summaries.json').read_text())
        targets={i['uid']:i for i in inputs}
        assert len(targets)==80
        for item in inputs:
            assert 'reference_q' not in item
            last=None;forecast=[]
            for t,(p,r) in enumerate(zip(item['target_position'],item['target_rotation'])):
                target=Pose(p,r);pred=predict_target(target,last);last=target
                if t+1<len(item['target_position']):
                    actual=Pose(item['target_position'][t+1],item['target_rotation'][t+1])
                    residual=pose_error(actual,pred)/scale
                    forecast.append([float(np.linalg.norm(residual[:3])),float(np.linalg.norm(residual[3:]))])
            forecast=np.array(forecast)
            prediction_rows.append(dict(robot=robot,uid=item['uid'],family=item['family'],
                abruptness_level=by_uid[item['uid']]['abruptness_level'],
                position_prediction_p95_tolerance_units=float(np.percentile(forecast[:,0],95)),
                orientation_prediction_p95_tolerance_units=float(np.percentile(forecast[:,1],95)),
                prediction_p95_combined=float(np.percentile(np.linalg.norm(forecast,axis=1),95)),
                prediction_max_combined=float(np.max(np.linalg.norm(forecast,axis=1))),
                forecast_outside_nominal_contract_rate=float(np.mean(np.max(forecast,axis=1)>1))))
        counts=dict(robot=robot,runs=0,frames=0,accepted=0,nominal_pairs_checked=0,
                    acceptance_discrepancies=0,feedback_discrepancies=0,deadline_discrepancies=0)
        witness_done=False
        for s in sorted(summaries,key=lambda r:(r['uid'],r['repeat'],r['method'])):
            item=targets[s['uid']];rows=read_rows(folder/s['raw_file'])
            assert len(rows)==300
            previous=np.array(item['initial_q']);last=None;first_failure=None
            for t,r in enumerate(rows):
                assert r['frame']==t and r['dt']==.02
                assert np.array_equal(previous,r['previous_q'])
                assert np.array_equal(item['target_position'][t],r['target_position'])
                assert np.array_equal(item['target_rotation'][t],r['target_rotation'])
                target=Pose(r['target_position'],r['target_rotation']);query=IKQuery(target,previous,.02)
                raw=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                check=v.check(raw,query)
                assert check.accepted==r['accepted'],(s['run_id'],t,'acceptance mismatch')
                if check.finite_ok:
                    assert abs(check.position_error-r['position_error'])<1e-12
                    assert abs(check.orientation_error-r['orientation_error'])<1e-12
                assert r['accepted_within_20ms']==bool(check.accepted and r['total_latency_ns']<=20_000_000)
                if r.get('nominal_next_verified') and r['method']!='single_step_reserve':
                    prediction=predict_target(target,last)
                    assert np.allclose(prediction.position,r['predicted_position'],atol=1e-12,rtol=0)
                    assert np.allclose(prediction.rotation,r['predicted_rotation'],atol=1e-12,rtol=0)
                    assert v.check(np.array(r['nominal_next_q']),IKQuery(prediction,raw,.02)).accepted
                    counts['nominal_pairs_checked']+=1
                last=target
                if check.accepted:previous=raw.copy();counts['accepted']+=1
                elif first_failure is None:first_failure=r
                assert np.array_equal(previous,r['accepted_state_q'])
                counts['frames']+=1
                for key in ('total_latency_ns','backup_ns','prediction_ns','optimization_ns','verification_ns'):
                    if key in r:phase[(robot,s['method'],key)].append(r[key])
            counts['runs']+=1
            if first_failure is not None:
                r=first_failure
                failures.append(dict(robot=robot,uid=s['uid'],family=s['family'],method=s['method'],repeat=s['repeat'],
                    first_failure_frame=r['frame'],successful_prefix=r['frame'],internal_ok=r['internal_ok'],
                    native_status=r.get('native_status'),native_return_code=r.get('native_return_code'),
                    reasons=r['verification_reasons'],position_error_m=r['position_error'],
                    orientation_error_rad=r['orientation_error'],velocity_utilization=r.get('velocity_utilization'),
                    previous_q=r['previous_q'],returned_q=r['q'],target_position=r['target_position'],
                    target_rotation=r['target_rotation'],total_latency_ns=r['total_latency_ns'],
                    interpretation='not found in this method-specific state; not proof of infeasibility'))
            if s['family']=='high_curvature':
                abrupt.append(dict(robot=robot,method=s['method'],uid=s['uid'],repeat=s['repeat'],
                    abruptness_level=by_uid[s['uid']]['abruptness_level'],completion=s['completion'],
                    deadline_completion=s['deadline_completion'],frame_success=s['frame_success'],
                    cumulative_ms=s['total_latency_ns']/1e6))
            if s['method']=='cr_ik' and s['completion'] and not witness_done:
                witness=dict(robot=robot,uid=s['uid'],family=s['family'],repeat=s['repeat'],
                    source_run=str(folder/s['raw_file']),source_sha256=sha(folder/s['raw_file']),dt=.02,
                    selection='lexicographically first completed CR run; illustrative witness, not another experiment',
                    initial_q=item['initial_q'],target_position=item['target_position'],target_rotation=item['target_rotation'],
                    accepted_q=[r['q'] for r in rows],total_latency_ns=[r['total_latency_ns'] for r in rows],
                    all_frames_reverified=True,deadline_completion=s['deadline_completion'])
                write_json(out/f'{robot}_crik_success_witness.json',witness)
                witnesses.append(dict(robot=robot,uid=s['uid'],frames=len(rows)));witness_done=True
            if counts['runs']%200==0:print(f'FK-only replay {robot}: {counts["runs"]}/{len(summaries)}',flush=True)
        job_counts.append(counts)
    csv_write(out/'first_failure_inputs.csv',failures)
    csv_write(out/'forecast_error_per_trajectory.csv',prediction_rows)
    csv_write(out/'abruptness_run_results.csv',abrupt)
    aggregate=[]
    for robot in cfg['robots']:
        for method in cfg['methods']:
            for level in range(4):
                rows=[r for r in abrupt if r['robot']==robot and r['method']==method and r['abruptness_level']==level]
                if not rows:continue
                aggregate.append(dict(robot=robot,method=method,abruptness_level=level,
                    trajectories=len({r['uid'] for r in rows}),mean_completed=float(np.mean([r['completion'] for r in rows])*5),
                    tsr=float(np.mean([r['completion'] for r in rows])),dtsr20=float(np.mean([r['deadline_completion'] for r in rows])),
                    mean_cumulative_ms=float(np.mean([r['cumulative_ms'] for r in rows]))))
    csv_write(out/'abruptness_table.csv',aggregate)
    csv_write(out/'timing_phases.csv',[dict(robot=robot,method=method,phase=key,mean_ms=float(np.mean(values)/1e6),
        p50_ms=float(np.percentile(values,50)/1e6),p95_ms=float(np.percentile(values,95)/1e6),
        p99_ms=float(np.percentile(values,99)/1e6),max_ms=float(np.max(values)/1e6),calls=len(values),
        all_call_returned_within_20ms=float(np.mean(np.array(values)<=20_000_000)) if key=='total_latency_ns' else None)
        for (robot,method,key),values in sorted(phase.items())])
    write_json(out/'verification_manifest.json',dict(utc=utc(),results=job_counts,source_sha256=sha(__file__),
        old_tracked_modifications=modified,old_base_commit=cfg['baseline_commit'],
        numerical_solver_calls=0,operation='replay original FK verifier and recorded feedback; no new experiment',
        witnesses=witnesses,unchanged_runtime_hashes=code_hashes()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',default='configs/correction_reserve.yaml');a=p.parse_args()
    audit(a.folders,a.out,yaml.safe_load(Path(a.config).read_text()))


if __name__=='__main__':main()
