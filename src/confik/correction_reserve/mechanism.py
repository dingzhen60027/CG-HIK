"""Prespecified common-input, nonlinear next-target perturbation checks.

This is an offline diagnostic, not another online predictor. The original next
target is read only after the current command is produced. All probe continuations
use the same TRAC-IK 20 ms adapter and the unchanged verifier.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import yaml

from ..types import Pose,IKQuery
from ..geometry import pose_error
from .geometry import predict_target,perturb_target,reserve_value,task_scale
from .study import factory,read_rows,write_json,clean,sha,utc,code_hashes
from .reporting import csv_write


def fixed_sites(folders,cfg):
    sites=[]
    for folder in map(Path,folders):
        inputs=json.loads((folder/'online_inputs.json').read_text())
        summaries=json.loads((folder/'summaries.json').read_text())
        for family in cfg['formal']['families']:
            item=min([i for i in inputs if i['family']==family],key=lambda i:i['uid'])
            run=next(r for r in summaries if r['uid']==item['uid'] and r['method']=='trac_task_5ms' and r['repeat']==0)
            rows=read_rows(folder/run['raw_file'])
            for frame in cfg['mechanism']['frames']:
                row=rows[frame]
                history=[rows[max(0,frame-j)]['previous_q'] for j in range(4)]
                sites.append(dict(robot=item['robot'],uid=item['uid'],family=family,frame=frame,dt=item['dt'],
                    site_id=f'{item["robot"]}_{item["site_id"]}_f{frame}',previous_q=row['previous_q'],
                    history_q=history,position=row['target_position'],rotation=row['target_rotation'],
                    last_position=rows[frame-1]['target_position'],last_rotation=rows[frame-1]['target_rotation'],
                    actual_next_position=rows[frame+1]['target_position'],actual_next_rotation=rows[frame+1]['target_rotation'],
                    source_run=str(folder/run['raw_file']),source_sha256=sha(folder/run['raw_file'])))
    return sites


def main():
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--config',default='configs/correction_reserve.yaml')
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    sites=fixed_sites(args.folders,cfg)
    write_json(out/'site_seal.json',dict(utc=utc(),sites=sites,configuration=cfg['mechanism'],
        source_code_sha256=sha(__file__),runtime_hashes=code_hashes(),current_searches_per_method=1,
        continuation_searches_per_perturbed_input=cfg['mechanism']['repeats'],
        status='observed development states; not independent fresh trajectories; no search-seed control claimed'))
    directions=[s*e for e in np.eye(6) for s in (-1.,1.)]
    probes=[(0.,0,np.zeros(6))]+[(a,j,d) for a in cfg['mechanism']['amplitudes'] if a>0 for j,d in enumerate(directions)]
    solvers={};summary=[];candidate_rows=[]
    (out/'raw').mkdir()
    try:
        for site in sites:
            robot=site['robot']
            if robot not in solvers:
                solvers[robot]={m:factory(m,robot,cfg) for m in cfg['methods']}
            continuation,kin,v=solvers[robot]['trac_task_20ms'];scale=task_scale(v)
            target=Pose(site['position'],site['rotation']);last=Pose(site['last_position'],site['last_rotation'])
            predicted=predict_target(target,last);previous=np.array(site['previous_q'])
            # These current candidates are generated before reading actual next.
            current={}
            for method in cfg['methods']:
                solver,_,_=solvers[robot][method]
                if hasattr(solver,'reset'):solver.reset(previous)
                if hasattr(solver,'_history'):solver._history=np.asarray(site['history_q'],float)
                if hasattr(solver,'last_target'):solver.last_target=last
                current[method]=solver.solve(target.position,target.rotation,previous,site['dt'])
            actual_next=Pose(site['actual_next_position'],site['actual_next_rotation'])
            forecast_error=pose_error(actual_next,predicted)/scale
            for method,result in current.items():
                record=dict(site_id=site['site_id'],robot=robot,uid=site['uid'],family=site['family'],frame=site['frame'],
                    method=method,current_result=result,prediction_error_vector=forecast_error.tolist(),
                    prediction_error_norm=float(np.linalg.norm(forecast_error)),
                    prediction_position=predicted.position.tolist(),prediction_rotation=predicted.rotation.tolist())
                if not result['accepted']:
                    candidate_rows.append(record)
                    summary.append(dict(site_id=site['site_id'],robot=robot,uid=site['uid'],family=site['family'],method=method,
                        current_accepted=False,predicted_gamma=None,probe_count=0,probe_success=None,all_directions_radius=None))
                    continue
                q=np.array(result['q']);query=IKQuery(target,previous,site['dt'])
                assert v.check(q,query).accepted
                nominal=continuation.solve(predicted.position,predicted.rotation,q,site['dt'])
                z=(result.get('nominal_next_q') if method!='single_step_reserve' else None)
                gamma_origin='optimized nominal next' if z is not None else 'common TRAC20 nominal next'
                if z is None and nominal['accepted']:z=nominal['q']
                gamma=None;rank=None
                if z is not None and v.check(np.array(z),IKQuery(predicted,q,site['dt'])).accepted:
                    gamma,mapping,_=reserve_value(kin,predicted,q,np.array(z),scale,
                        kin.limits.velocity*site['dt']+v.config.velocity_tolerance)
                    rank=mapping.rank
                record.update(nominal_result=nominal,predicted_gamma=gamma,gamma_origin=gamma_origin,rank=rank)
                candidate_rows.append(record);rows=[];success={}
                for amplitude,direction_id,direction in probes:
                    disturbed=perturb_target(predicted,direction,amplitude,scale)
                    for rep in range(cfg['mechanism']['repeats']):
                        r=continuation.solve(disturbed.position,disturbed.rotation,q,site['dt'])
                        if r['accepted']:assert v.check(np.array(r['q']),IKQuery(disturbed,q,site['dt'])).accepted
                        rows.append(dict(site_id=site['site_id'],robot=robot,uid=site['uid'],method=method,
                            amplitude=amplitude,direction=direction.tolist(),direction_id=direction_id,repeat=rep,
                            common_previous_q=previous.tolist(),current_q=q.tolist(),
                            current_position=target.position.tolist(),current_rotation=target.rotation.tolist(),
                            probe_position=disturbed.position.tolist(),probe_rotation=disturbed.rotation.tolist(),
                            dt=site['dt'],continuation=r))
                        success.setdefault(amplitude,[]).append(bool(r['accepted']))
                radii=[]
                for direction_id in range(12):
                    radius=0. if all(success[0.]) else None
                    if radius is not None:
                        for amplitude in cfg['mechanism']['amplitudes'][1:]:
                            outcomes=[r['continuation']['accepted'] for r in rows
                                if r['amplitude']==amplitude and r['direction_id']==direction_id]
                            if not all(outcomes):break
                            radius=amplitude
                    radii.append(radius)
                summary.append(dict(site_id=site['site_id'],robot=robot,uid=site['uid'],family=site['family'],method=method,
                    current_accepted=True,predicted_gamma=gamma,gamma_origin=gamma_origin,probe_count=len(rows),
                    probe_success=float(np.mean([r['continuation']['accepted'] for r in rows])),
                    success_by_amplitude={a:float(np.mean(s)) for a,s in success.items()},
                    all_directions_radius=min(radii) if all(r is not None for r in radii) else None,
                    direction_radii=radii,actual_prediction_error=float(np.linalg.norm(forecast_error))))
                with gzip.open(out/'raw'/f'{site["site_id"]}_{method}.jsonl.gz','xt') as f:
                    for row in rows:f.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n')
            print(f'mechanism {site["site_id"]}: {len(candidate_rows)}/{len(sites)*len(cfg["methods"])} commands',flush=True)
    finally:
        for group in solvers.values():
            for solver,_,_ in group.values():solver.close()
    write_json(out/'current_commands.json',candidate_rows);write_json(out/'summaries.json',summary)
    csv_write(out/'perturbation_table.csv',summary)
    write_json(out/'manifest.json',dict(utc=utc(),sites=len(sites),independent_trajectories=len({s['uid'] for s in sites}),
        mode='offline common-input next-target searches; not a nonlinear infeasibility proof',
        witness='each successful raw row stores common previous, admissible current q and admissible next q; verify both targets',
        amplitude_ceiling=max(cfg['mechanism']['amplitudes']),ceiling_is_censoring=True,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
