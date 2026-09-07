"""Read-only aggregation, preserving the independent query/trajectory unit."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from ..revision_compute_allocation.common import json_write,csv_write,paired_interval,digest
from .study import Study,now
from .outcomes import taxonomy


def records(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def distribution(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(a):return dict(p50=None,p95=None,p99=None,mean=None,max=None)
    return dict(zip(['p50','p95','p99','mean','max'],map(float,[*np.quantile(a,[.5,.95,.99]),a.mean(),a.max()])))


def summarize(rows):
    n=len(rows);acc=[r for r in rows if r['accepted']]
    uids={r['uid'] for r in rows};nr=n/len(uids)
    latency=distribution([r['total_latency_ns']/1e6 for r in rows])
    matrix={taxonomy(i,a):sum(bool(r['internal_ok'])==i and bool(r['accepted'])==a for r in rows)
            for i in [False,True] for a in [False,True]}
    out=dict(calls=n,units=len(uids),verified_success=float(len(acc)/n),
        internal_success=float(sum(r['internal_ok'] for r in rows)/n),
        accepted_within_20ms=float(sum(r['accepted'] and r['returned_within_20ms'] for r in rows)/n),
        returned_within_5ms=float(sum(r['returned_within_5ms'] for r in rows)/n),
        returned_within_20ms=float(sum(r['returned_within_20ms'] for r in rows)/n),
        total_latency_ns=sum(r['total_latency_ns'] for r in rows),
        **{'latency_'+k+'_ms':v for k,v in latency.items()},**matrix,
        accepted_position=distribution([r['position_error'] for r in acc]),
        accepted_orientation=distribution([r['orientation_error'] for r in acc]),
        all_returned_position=distribution([r['position_error'] for r in rows]),
        all_returned_orientation=distribution([r['orientation_error'] for r in rows]),
        joint_rejections=sum('joint_limit' in r['verification_reasons'] for r in rows),
        velocity_rejections=sum('velocity_limit' in r['verification_reasons'] for r in rows),
        native_failure_task_accept=sum(not r['internal_ok'] and r['accepted'] for r in rows))
    if 'witness_available' in rows[0]:
        feasible=rows[0]['witness_available']
        miss=[r for r in rows if not r['accepted'] and r['witness_available']]
        counts=defaultdict(int)
        for r in miss:counts[r['uid']]+=1
        out.update(witness_feasible=feasible,missed_calls=len(miss),misses_per_sweep=len(miss)/nr,
            any_repeat_missed_uids=len(counts),all_repeats_missed_uids=sum(v==nr for v in counts.values()),
            false_acceptances=sum(r['accepted'] and not r['witness_available'] for r in rows))
    return out


def unit_table(rows,trajectory=False):
    groups=defaultdict(list)
    for r in rows:groups[r['uid']].append(r)
    result={}
    for uid,rs in groups.items():
        repeats=defaultdict(list)
        for r in rs:repeats[r['repeat']].append(r)
        result[uid]=dict(uid=uid,family=rs[0]['family'],
            success=float(np.mean([all(r['accepted'] for r in rr) for rr in repeats.values()])) if trajectory else float(np.mean([r['accepted'] for r in rs])),
            time=float(np.mean([sum(r['total_latency_ns'] for r in rr) for rr in repeats.values()])) if trajectory else float(np.mean([r['total_latency_ns'] for r in rs])))
    return result


def compare(a,b,cfg,trajectory=False):
    aa,bb=unit_table(a,trajectory),unit_table(b,trajectory)
    assert set(aa)==set(bb)
    ids=sorted(aa);fam=[aa[i]['family'] for i in ids]
    kwargs=dict(repeats=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    gained=[i for i in ids if aa[i]['success']>bb[i]['success']]
    lost=[i for i in ids if aa[i]['success']<bb[i]['success']]
    return dict(units=len(ids),gained_uids=gained,lost_uids=lost,
        stable_gained_uids=[i for i in ids if aa[i]['success']==1 and bb[i]['success']==0],
        stable_lost_uids=[i for i in ids if aa[i]['success']==0 and bb[i]['success']==1],
        success_difference=paired_interval([aa[i]['success'] for i in ids],[bb[i]['success'] for i in ids],fam,ratio=False,**kwargs),
        latency_ratio=paired_interval([aa[i]['time'] for i in ids],[bb[i]['time'] for i in ids],fam,**kwargs),
        latency_difference_ns=paired_interval([aa[i]['time'] for i in ids],[bb[i]['time'] for i in ids],fam,ratio=False,**kwargs))


def point_tables(s,subdir):
    groups=defaultdict(list);traces=[]
    for p in sorted((s.out/subdir).glob('*_records.jsonl.gz')):
        for r in records(p):
            trace=r.pop('dls_trace',None)
            if trace:
                traces.append(dict(robot=r['robot'],uid=r['uid'],family=r['family'],method=r['method'],
                    final_position_ratio=r['position_error']/s.source['verifier']['position_tolerance'],
                    final_orientation_ratio=r['orientation_error']/s.source['verifier']['orientation_tolerance'],
                    **{k:v for k,v in trace.items() if k!='evaluations'}))
            groups[(r['robot'],r['scale'],r['method'],r['witness_available'])].append(r)
    main=[];families=[];comparisons=[];units=[]
    for (robot,scale,method,witness),rs in groups.items():
        identity=dict(robot=robot,scale=scale,method=method,witness_feasible=witness)
        main.append(dict(**identity,**{k:v for k,v in summarize(rs).items() if k!='witness_feasible'}))
        for family in sorted({r['family'] for r in rs}):
            families.append(dict(**identity,family=family,**{k:v for k,v in summarize([r for r in rs if r['family']==family]).items() if k!='witness_feasible'}))
        units.extend(dict(**identity,**u) for u in unit_table(rs).values())
        base='dls_strict' if method=='dls_task' else 'trac_strict_20ms' if method=='trac_task_20ms' else 'trac_strict_5ms'
        if method!=base and (robot,scale,base,witness) in groups:
            comparisons.append(dict(**identity,baseline=base,**compare(rs,groups[(robot,scale,base,witness)],s.cfg)))
    prefix='point' if subdir=='02_point_study' else 'sensitivity'
    for name,rows in [('main',main),('families',families),('paired',comparisons),('units',units)]:
        json_write(s.out/'05_aggregate'/f'{prefix}_{name}.json',rows)
        csv_write(s.out/'05_aggregate'/f'{prefix}_{name}.csv',rows)
    if traces:
        json_write(s.out/'05_aggregate/dls_excess_iterations.json',traces)
        csv_write(s.out/'05_aggregate/dls_excess_iterations.csv',traces)


def trajectory_tables(s):
    main=[];families=[];comparisons=[];units=[];runs=[];first=[];completion_sets=[]
    for robot,folder in [('panda',s.root/s.cfg['panda_authority']),('ur5e',s.out/'04_trajectory_study')]:
        groups=defaultdict(list)
        for p in sorted((folder/'online_runs').glob('*.jsonl.gz')):
            rs=list(records(p));r=rs[0];groups[r['method']].extend(rs)
            bad=next((row for row in rs if not row['accepted']),None)
            run=dict(robot=robot,run_id=r['run_id'],uid=r['uid'],family=r['family'],method=r['method'],repeat=r['repeat'],
                complete=bad is None,complete_within_20ms=all(row['accepted'] and row['returned_within_20ms'] for row in rs),
                total_latency_ns=sum(row['total_latency_ns'] for row in rs),source=str(p.relative_to(s.root)))
            runs.append(run)
            if bad:first.append(dict(robot=robot,source=run['source'],**bad))
        def rowset(rs):
            out=summarize(rs);uu=unit_table(rs,True)
            relevant=[r for r in runs if r['robot']==robot and r['method']==rs[0]['method'] and r['uid'] in uu]
            reps=sorted({r['repeat'] for r in relevant});nr=len(reps)
            out.update(completion_counts=[sum(r['complete'] for r in relevant if r['repeat']==rep) for rep in reps],
                deadline_completion_counts=[sum(r['complete_within_20ms'] for r in relevant if r['repeat']==rep) for rep in reps],
                completion_rate=float(np.mean([u['success'] for u in uu.values()])),
                cumulative_latency_ns_per_sweep=out['total_latency_ns']/nr,
                per_trajectory_cumulative_ms=distribution([r['total_latency_ns']/1e6 for r in relevant]),
                paired_unit_cumulative_ms=distribution([r['time']/1e6 for r in uu.values()]))
            return out
        for method,rs in groups.items():
            main.append(dict(robot=robot,method=method,authority=str(folder.relative_to(s.root)),**rowset(rs)))
            for family in sorted({r['family'] for r in rs}):
                families.append(dict(robot=robot,method=method,family=family,**rowset([r for r in rs if r['family']==family])))
            units.extend(dict(robot=robot,method=method,**r) for r in unit_table(rs,True).values())
            for rep in sorted({r['repeat'] for r in rs}):
                completion_sets.append(dict(robot=robot,method=method,repeat=rep,
                    completed_uids=[r['uid'] for r in runs if r['robot']==robot and r['method']==method and r['repeat']==rep and r['complete']]))
            if method in ['trac_task_5ms','trac_task_20ms','dls_task']:
                base=method.replace('task','strict')
                comparisons.append(dict(robot=robot,method=method,baseline=base,**compare(rs,groups[base],s.cfg,True)))
                for fam in sorted({r['family'] for r in rs}):
                    comparisons.append(dict(robot=robot,method=method,baseline=base,family=fam,
                        **compare([r for r in rs if r['family']==fam],[r for r in groups[base] if r['family']==fam],s.cfg,True)))
    for name,rows in [('trajectory_main',main),('trajectory_families',families),('trajectory_paired',comparisons),
                      ('trajectory_units',units),('trajectory_runs',runs),('first_failures',first),('completion_uid_sets',completion_sets)]:
        json_write(s.out/'05_aggregate'/f'{name}.json',rows);csv_write(s.out/'05_aggregate'/f'{name}.csv',rows)


def main():
    s=Study();s.check();assert (s.out/'04_trajectory_study/completed.json').exists()
    json_write(s.out/'05_aggregate/started.json',dict(utc=now()))
    point_tables(s,'02_point_study');point_tables(s,'03_contract_sensitivity');trajectory_tables(s)
    authority=s.root/'outputs/revision_compute_allocation/reports'
    import csv
    tables={p.name:list(csv.DictReader(p.open())) for p in [authority/'point_paired_comparisons.csv',
        authority/'trajectory_paired_comparisons.csv',authority/'oracle_crossfit_summary.csv',
        s.root/'outputs/revision_compute_allocation/01_existing_result_decomposition/trajectory_savings_decomposition.csv']}
    json_write(s.out/'05_aggregate/allocation_boundary.json',dict(
        interpretation='read-only historical case study; NOT newly measured routing on aligned solvers',tables=tables))
    json_write(s.out/'05_aggregate/completed.json',dict(utc=now()))


if __name__=='__main__':main()
