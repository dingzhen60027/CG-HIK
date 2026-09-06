"""Supplementary results with query/trajectory-level paired uncertainty."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .benchmark import read_records
from .common import csv_write,json_write,paired_interval,digest
from .decomposition import GROUPS
from .policies import INTERNAL_METHODS


def summary(rows):
    latency=np.array([r['latency_ns'] for r in rows],float)/1e6
    fev=[r['fev'] for r in rows if r['fev'] is not None]
    result = dict(calls=len(rows),verified_success_rate=float(np.mean([r['accepted'] for r in rows])),
                accepted_within_20ms_rate=float(np.mean([r['accepted_within_20ms'] for r in rows])),
                accepted_contract_violations=sum(r['accepted_contract_violation'] for r in rows),
                total_latency_ns=sum(r['latency_ns'] for r in rows),p50_ms=float(np.quantile(latency,.5)),
                p95_ms=float(np.quantile(latency,.95)),p99_ms=float(np.quantile(latency,.99)),
                mean_fev=float(np.mean(fev)) if fev else '',
                reject_count=sum(r['route']=='reject' for r in rows),defer_count=sum(r['route']=='defer' for r in rows),
                fallback_count=sum(r['fallback'] for r in rows),fallback_rate=float(np.mean([r['fallback'] for r in rows])),
                route_counts=json.dumps(dict(Counter(r['route'] for r in rows))))
    for metric in ['position_error','orientation_error','max_joint_step','max_velocity_utilization']:
        values=[r[metric] for r in rows if r.get(metric) is not None]
        result[f'accepted_{metric}_median']=float(np.median(values)) if values else ''
        result[f'accepted_{metric}_max']=float(np.max(values)) if values else ''
    return result


def point_tables(records,robot,cfg):
    groups=defaultdict(list)
    for r in records:
        # The infeasible subset and its family share one name; do not double count.
        for subset in set(['all','feasible' if r['witness'] else 'constructed_inexecutable',r['family']]):
            groups[subset,r['method']].append(r)
    main=[];unit_rows=[];unit={}
    for (subset,method),rows in sorted(groups.items()):
        row=dict(robot=robot,subset=subset,method=method,query_count=len(set(r['uid'] for r in rows)),**summary(rows))
        witness=[r for r in rows if r['witness']]
        row['witness_false_reject_rate']=float(np.mean([r['route']=='reject' for r in witness])) if witness else ''
        byquery=defaultdict(list)
        for r in rows:byquery[r['uid']].append(r)
        units={}
        for uid,rr in sorted(byquery.items()):
            if len(rr)!=cfg['point_repeats']:raise AssertionError('missing/duplicate query repetitions')
            units[uid]=dict(uid=uid,family=rr[0]['family'],latency=float(np.mean([r['latency_ns'] for r in rr])),
                           latency_samples_ns=[r['latency_ns'] for r in sorted(rr,key=lambda r:r['repeat'])],
                           success=float(np.mean([r['accepted'] for r in rr])),fev=float(np.mean([r['fev'] for r in rr])) if rr[0]['fev'] is not None else None,
                           deadline=float(np.mean([r['accepted_within_20ms'] for r in rr])),all_success=all(r['accepted'] for r in rr))
        row['all_repeats_success_query_count']=sum(u['all_success'] for u in units.values())
        unit[subset,method]=units;main.append(row)
        if subset=='all':unit_rows.extend(dict(robot=robot,method=method,**u) for u in units.values())
    paired=[]
    comparisons=[(m,'always_hard') for m in INTERNAL_METHODS if m!='always_hard']+[
        ('full_cghik','reject_only_hard'),('full_cghik','p50_selection'),('full_cghik','geometry_threshold'),('full_cghik','routing_only')]
    comparisons += [('full_cghik',m) for m in sorted({r['method'] for r in records}) if m.startswith('trac_ik')]
    for subset in ['all','feasible',*cfg['point_counts']]:
        if subset in ['all','feasible'] or subset in cfg['point_counts']:
            for a,b in comparisons:
                aa,bb=unit[subset,a],unit[subset,b]
                assert set(aa)==set(bb)
                ids=sorted(aa);families=[aa[x]['family'] for x in ids]
                row=dict(robot=robot,subset=subset,method=a,reference=b,query_count=len(ids))
                for metric,ratio in [('latency',True),('fev',True),('success',False),('deadline',False)]:
                    if metric=='fev' and b.startswith('trac_ik'):continue
                    values=paired_interval([aa[x][metric] for x in ids],[bb[x][metric] for x in ids],families,
                                          ratio=ratio,repeats=cfg['statistics']['bootstrap_repeats'])
                    label=metric+('_ratio' if ratio else '_difference')
                    row.update(zip([label,label+'_ci_low',label+'_ci_high'],values))
                row['paired_latency_difference_ms']=float(np.mean([aa[x]['latency']-bb[x]['latency'] for x in ids])/1e6)
                delta=paired_interval([aa[x]['latency']/1e6 for x in ids],[bb[x]['latency']/1e6 for x in ids],families,
                                      ratio=False,repeats=cfg['statistics']['bootstrap_repeats'])
                row['paired_latency_difference_ms_ci_low'],row['paired_latency_difference_ms_ci_high']=delta[1:]
                if subset=='feasible' and a=='full_cghik' and b in ['always_hard','p50_selection','geometry_threshold']:
                    qa=np.array([aa[x]['latency_samples_ns'] for x in ids],float)/1e6
                    qb=np.array([bb[x]['latency_samples_ns'] for x in ids],float)/1e6
                    rng=np.random.default_rng(cfg['statistics']['seed'])
                    strata=[np.flatnonzero(np.asarray(families)==f) for f in np.unique(families)]
                    estimates=[]
                    for _ in range(cfg['statistics']['bootstrap_repeats']):
                        index=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
                        estimates.append(np.quantile(qa[index],[.5,.95,.99])/np.quantile(qb[index],[.5,.95,.99]))
                    qs=np.quantile(qa,[.5,.95,.99])/np.quantile(qb,[.5,.95,.99])
                    bounds=np.quantile(estimates,[.025,.975],axis=0)
                    for j,label in enumerate(['p50_ratio','p95_ratio','p99_ratio']):
                        row.update({label:float(qs[j]),label+'_ci_low':float(bounds[0,j]),label+'_ci_high':float(bounds[1,j])})
                paired.append(row)
    # Deduplicate repeated constructed-inexecutable subset requested as a family.
    paired=list({(r['robot'],r['subset'],r['method'],r['reference']):r for r in paired}.values())
    curves=[]
    for (subset,method),rows in sorted(groups.items()):
        if subset not in ['all','feasible','constructed_inexecutable']:continue
        for ms in [.1,.25,.5,1,2,5,10,20,50,100,200,400,800]:
            curves.append(dict(robot=robot,subset=subset,method=method,elapsed_limit_ms=ms,
                               verified_within_rate=float(np.mean([r['accepted'] and r['latency_ns']<=ms*1e6 for r in rows]))))
    return main,paired,unit_rows,curves


def oracle_tables(records,robot):
    byquery=defaultdict(dict)
    for r in records:byquery[r['uid']][r['method'],r['repeat']]=r
    details=[];entries=['forced_easy','forced_medium','forced_hard']
    for uid,rr in sorted(byquery.items()):
        assert len(rr)==30
        latency=np.array([[rr[e,k]['latency_ns']/1e6 for k in range(10)] for e in entries])
        success=np.array([[rr[e,k]['accepted'] for k in range(10)] for e in entries])
        folds=[np.arange(0,10,2),np.arange(1,10,2)]
        winners=[]
        for fold,train in enumerate(folds):
            costs=np.quantile(latency[:,train],.95,axis=1)
            eligible=success[:,train].all(axis=1)
            winners.append(int(np.argmin(np.where(eligible,costs,np.inf))) if eligible.any() else -1)
        for fold,chosen in enumerate(winners):
            test=folds[1-fold]
            details.append(dict(robot=robot,uid=uid,family=next(iter(rr.values()))['family'],selection_fold=fold,
                                preferred_entry=entries[chosen] if chosen>=0 else 'no_successful_entry',
                                preference_stable=winners[0]==winners[1],
                                heldout_selected_mean_ms=float(latency[chosen,test].mean()) if chosen>=0 else '',
                                heldout_selected_p95_ms=float(np.quantile(latency[chosen,test],.95)) if chosen>=0 else '',
                                heldout_selected_success=float(success[chosen,test].mean()) if chosen>=0 else '',
                                heldout_hard_mean_ms=float(latency[2,test].mean()),
                                heldout_hard_p95_ms=float(np.quantile(latency[2,test],.95)),
                                heldout_hard_success=float(success[2,test].mean()),
                                full_measurement_entry_p95_range_ms=float(np.ptp(np.quantile(latency,.95,axis=1))),
                                full_measurement_entry_mean_range_ms=float(np.ptp(latency.mean(axis=1))),
                                within_entry_median_absolute_deviation_ms=float(np.median(np.abs(latency-np.median(latency,axis=1)[:,None])))))
    valid=[r for r in details if r['preferred_entry']!='no_successful_entry']
    mean_byquery={}
    for r in valid:mean_byquery.setdefault(r['uid'],[]).append(r)
    two_valid=[rows for rows in mean_byquery.values() if len(rows)==2]
    aa=[np.mean([r['heldout_selected_mean_ms'] for r in rows]) for rows in mean_byquery.values()]
    bb=[np.mean([r['heldout_hard_mean_ms'] for r in rows]) for rows in mean_byquery.values()]
    family=[rows[0]['family'] for rows in mean_byquery.values()]
    ratios=paired_interval(aa,bb,family)
    result=dict(robot=robot,query_count=len(byquery),selection_without_success_count=sum(r['preferred_entry']=='no_successful_entry' for r in details),
                preference_stability=float(np.mean([rows[0]['preference_stable'] for rows in two_valid])) if two_valid else None,
                preference_stability_query_count=len(two_valid),
                all_query_preference_stability_including_no_solution=float(np.mean([r['preference_stable'] for r in details])),
                entry_mean_range_at_most_015ms_rate=float(np.mean([r['full_measurement_entry_mean_range_ms']<=.15 for r in details])),
                entry_p95_range_at_most_015ms_rate=float(np.mean([r['full_measurement_entry_p95_range_ms']<=.15 for r in details])),
                crossfit_mean_latency_ratio=ratios[0],crossfit_mean_latency_ratio_ci_low=ratios[1],crossfit_mean_latency_ratio_ci_high=ratios[2],
                crossfit_selected_success=float(np.mean([r['heldout_selected_success'] for r in valid])),
                crossfit_hard_success=float(np.mean([r['heldout_hard_success'] for r in valid])),
                notes='P95 selects on five even/odd interleaved repeats; other five evaluate, then swap; units remain queries; no training/selection use')
    return result,details


def trajectory_tables(records,robot,cfg):
    byunit=defaultdict(list)
    for r in records:byunit[r['method'],r['uid']].append(r)
    units=[]
    for (method,uid),rr in sorted(byunit.items()):
        rr=sorted(rr,key=lambda r:r['frame'])
        if [r['frame'] for r in rr]!=list(range(cfg['trajectory_frames'])):raise AssertionError('trajectory target sequence is incomplete')
        failed=[r for r in rr if not r['accepted']]
        row=dict(robot=robot,method=method,uid=uid,family=rr[0]['family'],complete=not failed,
                 deadline_complete=all(r['accepted_within_20ms'] for r in rr),
                 first_failure_frame=failed[0]['frame'] if failed else '',
                 first_failure_reason=(failed[0]['reject_reason']+';'+','.join(failed[0]['verification_reasons'])) if failed else '',
                 total_fev=sum(r['fev'] for r in rr) if rr[0]['fev'] is not None else '',**summary(rr))
        units.append(row)
    summary_rows=[]
    for method in sorted(set(r['method'] for r in records)):
        for family in ['all',*cfg['trajectory_families']]:
            selected=[r for r in units if r['method']==method and (family=='all' or r['family']==family)]
            rr=[r for r in records if r['method']==method and (family=='all' or r['family']==family)]
            sums=[r['total_latency_ns'] for r in selected]
            summary_rows.append(dict(robot=robot,method=method,family=family,trajectory_count=len(selected),completed=sum(r['complete'] for r in selected),
                                     completion_rate=float(np.mean([r['complete'] for r in selected])),deadline_complete=sum(r['deadline_complete'] for r in selected),
                                     completion_uids=json.dumps([r['uid'] for r in selected if r['complete']]),
                                     trajectory_cumulative_mean_ms=float(np.mean(sums)/1e6),trajectory_cumulative_median_ms=float(np.median(sums)/1e6),
                                     trajectory_cumulative_p95_ms=float(np.quantile(sums,.95)/1e6),**summary(rr)))
    comparisons=[('full_cghik','always_hard'),('routing_only','always_hard'),('full_cghik','routing_only')]
    external=next(r['method'] for r in units if r['method'].startswith('trac_ik'))
    comparisons.append(('full_cghik',external))
    pairs=[];decomposition=[]
    for a,b in comparisons:
        for family in ['all',*cfg['trajectory_families']]:
            aa={r['uid']:r for r in units if r['method']==a and (family=='all' or r['family']==family)}
            bb={r['uid']:r for r in units if r['method']==b and (family=='all' or r['family']==family)}
            ids=sorted(aa);assert set(aa)==set(bb)
            row=dict(robot=robot,method=a,reference=b,family=family,trajectory_count=len(ids),
                     lost_uids=json.dumps([x for x in ids if bb[x]['complete'] and not aa[x]['complete']]),
                     gained_uids=json.dumps([x for x in ids if aa[x]['complete'] and not bb[x]['complete']]))
            for metric,source,ratio in [('latency','total_latency_ns',True),('completion','complete',False)]:
                values=paired_interval([float(aa[x][source]) for x in ids],[float(bb[x][source]) for x in ids],
                                      [aa[x]['family'] for x in ids],ratio=ratio,repeats=cfg['statistics']['bootstrap_repeats'])
                label=metric+('_ratio' if ratio else '_difference')
                row.update(zip([label,label+'_ci_low',label+'_ci_high'],values))
            if not b.startswith('trac_ik'):
                values=paired_interval([aa[x]['total_fev'] for x in ids],[bb[x]['total_fev'] for x in ids],[aa[x]['family'] for x in ids],repeats=cfg['statistics']['bootstrap_repeats'])
                row.update(zip(['fev_ratio','fev_ratio_ci_low','fev_ratio_ci_high'],values))
            pairs.append(row)
            total_saved=sum(bb[x]['total_latency_ns']-aa[x]['total_latency_ns'] for x in ids)
            for group in GROUPS:
                members=[]
                for x in ids:
                    ac,bc=aa[x]['complete'],bb[x]['complete']
                    g=GROUPS[0 if ac and bc else 1 if ac else 2 if bc else 3]
                    if g==group:members.append(x)
                method_time=sum(aa[x]['total_latency_ns'] for x in members);reference_time=sum(bb[x]['total_latency_ns'] for x in members)
                gr=dict(robot=robot,method=a,reference=b,family=family,group=group,trajectory_count=len(members),
                        method_total_ns=method_time,reference_total_ns=reference_time,saved_ns=reference_time-method_time,
                        saved_fraction_of_total=(reference_time-method_time)/total_saved if total_saved else '',
                        method_total_fev=sum(aa[x]['total_fev'] for x in members) if not a.startswith('trac_ik') else '',
                        reference_total_fev=sum(bb[x]['total_fev'] for x in members) if not b.startswith('trac_ik') else '')
                interval=paired_interval([aa[x]['total_latency_ns'] for x in members],[bb[x]['total_latency_ns'] for x in members],[aa[x]['family'] for x in members],repeats=cfg['statistics']['bootstrap_repeats'])
                gr.update(zip(['latency_ratio','latency_ratio_ci_low','latency_ratio_ci_high'],interval));decomposition.append(gr)
    return summary_rows,units,pairs,decomposition


def rejection_events(records,robot,workload):
    """Only identical complete inputs can witness a rejected query's rescue.

    A different closed-loop state is explicitly not evidence of a false reject.
    This uses already-recorded commands, never another solver invocation.
    """
    bykey={(r['method'],r['uid'],r.get('frame',0),r['repeat']):r for r in records}
    events=[]
    for r in records:
        if r['method']!='full_cghik' or r['route']!='reject':continue
        other=bykey['routing_only',r['uid'],r.get('frame',0),r['repeat']]
        same=r['query_hash']==other['query_hash']
        events.append(dict(robot=robot,workload=workload,uid=r['uid'],family=r['family'],frame=r.get('frame',0),repeat=r['repeat'],
                           same_complete_query=same,routing_only_accepted=other['accepted'],
                           same_query_verified_rescue=bool(same and other['accepted']),
                           original_point_witness=r.get('witness','not applicable after closed-loop branch divergence'),
                           full_latency_ns=r['latency_ns'],routing_only_latency_ns=other['latency_ns']))
    return events


def run(root,cfg,out):
    dest=out/'reports'
    if (dest/'report_manifest.json').exists():raise FileExistsError('report already finalized')
    points=[];paired=[];units=[];curves=[];oracles=[];oracle_details=[];trajectories=[];tunits=[];tpairs=[];decomposition=[];reject_events=[]
    for robot in cfg['robots']:
        print(f'report: {robot} points',flush=True)
        records=read_records(out/'02_point_mechanism_benchmark'/f'{robot}_raw_records.jsonl.gz')
        a,b,c,d=point_tables(records,robot,cfg);points+=a;paired+=b;units+=c;curves+=d
        reject_events+=rejection_events(records,robot,'points')
        del records
        records=read_records(out/'02_point_mechanism_benchmark'/f'{robot}_oracle_records.jsonl.gz')
        a,b=oracle_tables(records,robot);oracles.append(a);oracle_details+=b
        del records
        print(f'report: {robot} trajectories',flush=True)
        records=read_records(out/'03_feasible_trajectory_benchmark'/f'{robot}_raw_records.jsonl.gz')
        a,b,c,d=trajectory_tables(records,robot,cfg);trajectories+=a;tunits+=b;tpairs+=c;decomposition+=d
        reject_events+=rejection_events(records,robot,'reference_trajectories')
        del records
    tables={'point_main_table':points,'point_paired_comparisons':paired,'point_query_units':units,'external_success_time_curve':curves,
            'oracle_crossfit_summary':oracles,'oracle_crossfit_query_details':oracle_details,'trajectory_main_table':trajectories,
            'trajectory_units':tunits,'trajectory_paired_comparisons':tpairs,'feasible_trajectory_savings_decomposition':decomposition,
            'early_rejection_events':reject_events}
    for name,rows in tables.items():csv_write(dest/f'{name}.csv',rows)
    json_write(dest/'supplementary_tables.json',tables)
    from .figures import draw
    draw(root,out,tables)
    json_write(dest/'report_manifest.json',dict(protocol=cfg['protocol'],role='supplementary evaluation',old_decomposition='post-hoc descriptive analysis',
                independent_units='point queries (3 technical repeats) and complete trajectories (150 dependent frames)',
                confidence_intervals='paired family-stratified percentile bootstrap 95%; descriptive, not multiplicity-adjusted',
                files={str(p.relative_to(out)):digest(p) for p in dest.iterdir() if p.is_file()}))
