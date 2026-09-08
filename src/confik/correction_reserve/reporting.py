"""Source-backed trajectory summaries and paired, trajectory-cluster intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from .study import clean,read_rows,sha,write_json,utc


LABELS={'trac_task_5ms':'TRAC-IK task 5 ms','trac_task_20ms':'TRAC-IK task 20 ms',
    'pink_qp':'Pink QP-IK','ranged_ik_upstream':'RangedIK (original cutoff)',
    'ranged_ik_positive_range':'RangedIK (positive-range adapter)',
    'two_step_predictive':'Two-step predictive','cr_ik':'CR-IK',
    'single_step_reserve':'Single-step reserve','two_step_sigma':'Two-step sigma-min'}


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('empty table')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(list,dict)) else clean(v)
                             for k,v in row.items()})


def paired_intervals(a,b,families,seed,resamples):
    """Keep whole UIDs paired; average search repeats before bootstrapping.

    Stratification retains the prespecified equal family composition. Intervals
    describe between-trajectory variation conditional on observed nested repeats.
    They are descriptive, unadjusted 95% intervals, not multiple-testing claims.
    """
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(np.array(families)==f) for f in sorted(set(families))]
    idx=np.concatenate([rng.choice(g,size=(resamples,len(g)),replace=True) for g in groups],axis=1)
    difference=np.mean(a[idx]-b[idx],axis=1)
    denominator=np.sum(b[idx],axis=1)
    ratio=np.divide(np.sum(a[idx],axis=1),denominator,out=np.full(resamples,np.nan),where=denominator!=0)
    return dict(difference=float(np.mean(a-b)),difference_ci=np.percentile(difference,[2.5,97.5]).tolist(),
        ratio=float(np.sum(a)/np.sum(b)) if np.sum(b)!=0 else None,
        ratio_ci=np.nanpercentile(ratio,[2.5,97.5]).tolist() if np.isfinite(ratio).any() else None)


def group_table(summaries,raw_arrays,group_family=False):
    grouped=defaultdict(list)
    for row in summaries:
        grouped[(row['robot'],row['method'],row['family'] if group_family else 'all')].append(row)
    table=[]
    for (robot,method,family),rows in sorted(grouped.items()):
        repeats=sorted({r['repeat'] for r in rows});uids=sorted({r['uid'] for r in rows})
        perrep=[sum(r['completion'] for r in rows if r['repeat']==rep) for rep in repeats]
        deadlines=[sum(r['deadline_completion'] for r in rows if r['repeat']==rep) for rep in repeats]
        arrays=[raw_arrays[r['run_id']] for r in rows]
        lat=np.concatenate([a['latency'] for a in arrays]);errors=np.concatenate([a['errors'] for a in arrays])
        duration_by_uid=[np.mean([r['total_latency_ns'] for r in rows if r['uid']==uid])/1e6 for uid in uids]
        qstats=np.percentile(lat,[50,95,99])/1e6
        table.append(dict(robot=robot,method=method,label=LABELS[method],family=family,
            trajectories=len(uids),repeats=len(repeats),completion_by_repeat=perrep,
            mean_completed_trajectories=float(np.mean(perrep)),tsr=float(np.mean(perrep)/len(uids)),
            deadline_completion_by_repeat=deadlines,dtsr20=float(np.mean(deadlines)/len(uids)),
            frame_success=float(np.mean([r['frame_success'] for r in rows])),
            frame_deadline_success=float(np.mean([r['frame_deadline_success'] for r in rows])),
            p50_ms=float(qstats[0]),p95_ms=float(qstats[1]),p99_ms=float(qstats[2]),
            cumulative_ms_per_sweep=float(np.sum(duration_by_uid)),
            trajectory_time_mean_ms=float(np.mean(duration_by_uid)),
            trajectory_time_median_ms=float(np.median(duration_by_uid)),
            trajectory_time_p95_ms=float(np.percentile(duration_by_uid,95)),
            backup_rate=float(np.mean([r['backup_rate'] for r in rows])),
            changed_from_backup_rate=float(np.mean([r['changed_from_backup_rate'] for r in rows])),
            p95_position_m=float(np.percentile(errors[:,0],95)) if len(errors) else None,
            max_position_m=float(np.max(errors[:,0])) if len(errors) else None,
            p95_orientation_rad=float(np.percentile(errors[:,1],95)) if len(errors) else None,
            max_orientation_rad=float(np.max(errors[:,1])) if len(errors) else None,
            max_accepted_step_utilization=float(max(r['max_accepted_step_utilization'] for r in rows)),
            acceleration_rms_mean=float(np.mean([r['acceleration_rms'] for r in rows])),
            velocity_change_rms_mean=float(np.mean([r['velocity_change_rms'] for r in rows])),
            successful_prefix_mean=float(np.mean([r['successful_prefix'] for r in rows])),
            accepted_contract_violations=sum(r['accepted_contract_violations'] for r in rows)))
    return table


def aggregate(folders,out,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summaries=[];arrays={};source_hashes={};completion={}
    for folder in map(Path,folders):
        if not (folder/'completed.json').exists():raise RuntimeError(f'incomplete study {folder}')
        source_hashes[str(folder/'completed.json')]=sha(folder/'completed.json')
        jobs=json.loads((folder/'summaries.json').read_text())
        for s in jobs:
            rows=read_rows(folder/s['raw_file']);summaries.append(s)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows],float),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']],float).reshape(-1,2))
            completion.setdefault(s['robot'],{}).setdefault(s['method'],{}).setdefault(str(s['repeat']),[])
            if s['completion']:completion[s['robot']][s['method']][str(s['repeat'])].append(s['uid'])
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    pairs=[];changes=[];unit_rows=[]
    for robot in cfg['robots']:
        sub=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in sub})
        methods=sorted({s['method'] for s in sub})
        unit={}
        for method in methods:
            for uid in uids:
                rows=[s for s in sub if s['method']==method and s['uid']==uid]
                if not rows:raise AssertionError((robot,method,uid,'missing independent unit'))
                unit[(method,uid)]={k:float(np.mean([s[k] for s in rows])) for k in
                    ['completion','deadline_completion','total_latency_ns','frame_success','successful_prefix']}
                unit_rows.append(dict(robot=robot,method=method,uid=uid,family=rows[0]['family'],
                    search_repeats=len(rows),**unit[(method,uid)]))
        families=[next(s['family'] for s in sub if s['uid']==uid) for uid in uids]
        for base in [m for m in methods if m!='cr_ik']:
            for metric in ('completion','deadline_completion','total_latency_ns','frame_success'):
                a=np.array([unit[('cr_ik',uid)][metric] for uid in uids]);b=np.array([unit[(base,uid)][metric] for uid in uids])
                pairs.append(dict(robot=robot,method='cr_ik',baseline=base,metric=metric,trajectories=len(uids),
                    **paired_intervals(a,b,families,cfg['formal']['bootstrap_seed'],cfg['formal']['bootstrap_resamples'])))
            for uid in uids:
                a=unit[('cr_ik',uid)]['completion'];b=unit[(base,uid)]['completion']
                changes.append(dict(robot=robot,baseline=base,uid=uid,
                    family=next(s['family'] for s in sub if s['uid']==uid),cr_completion_fraction=a,
                    baseline_completion_fraction=b,change='gained' if a>b else 'lost' if a<b else 'same',
                    stable_all_vs_none=(a==1 and b==0) or (a==0 and b==1)))
    csv_write(out/'main_table.csv',main);csv_write(out/'family_table.csv',family)
    csv_write(out/'trajectory_units.csv',unit_rows);csv_write(out/'paired_comparisons.csv',pairs)
    csv_write(out/'gained_lost_uids.csv',changes);csv_write(out/'run_summaries.csv',summaries)
    write_json(out/'completion_uids.json',completion)
    write_json(out/'source_data.json',dict(main=main,family=family,pairs=pairs,changes=changes,units=unit_rows))
    write_json(out/'manifest.json',dict(utc=utc(),sources=source_hashes,source_code_sha256=sha(__file__),
        independent_unit='trajectory UID, not frames or search repetitions',
        confidence_intervals='paired family-stratified trajectory bootstrap, 4000 resamples; average nested searches first; conditional on observed repeats; unadjusted descriptive 95% intervals',
        frame_quantiles='descriptive pooled full-frame quantiles, no failed or over-deadline calls excluded',
        accepted_errors='only truly accepted commands; acceptance counts and rejected frames reported separately',
        exclusions=0,multiplicity='no significance tests, no selection by interval; all prespecified comparisons shown',
        complete_trajectory_runs=len(summaries),frame_calls=sum(s['frames'] for s in summaries)))
    return main


def main():
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--config',default='configs/correction_reserve.yaml')
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    rows=aggregate(args.folders,args.out,cfg)
    for r in rows:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
                      np.round([r['p50_ms'],r['p95_ms'],r['p99_ms']],3))


if __name__=='__main__':main()
