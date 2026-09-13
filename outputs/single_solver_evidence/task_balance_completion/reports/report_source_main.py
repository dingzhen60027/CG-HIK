#!/usr/bin/env python3
"""Read-only command replay and source-backed tables; no IK calls."""
import importlib.util
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completion',ROOT/'scripts/run_task_balance_completion.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
OUT=entry.OUT
from confik.correction_reserve.study import context,read_rows,write_json,sha,utc
from confik.correction_reserve.reporting import csv_write,group_table,paired_intervals,LABELS
from confik.types import IKQuery,Pose
from confik.task_balance_gn import relative_stop
from confik import task_balance_reporting as previous_reporting

LABELS.update({m:m for m in ('single_gn_k0','single_gn_k1','task_balance_k0','task_balance_k1',
    'balance_cached_k0','balance_cached_k1','balance_progress_k0','balance_progress_k1')})

def comparisons(methods):
    p=[]
    for k in (0,1):
        for a,b in [(f'balance_cached_k{k}',f'task_balance_k{k}'),(f'balance_progress_k{k}',f'balance_cached_k{k}'),
                    (f'balance_progress_k{k}',f'single_gn_k{k}'),(f'balance_progress_k{k}',f'task_balance_k{k}')]:
            if a in methods and b in methods:p.append((a,b))
    for b in ('pink_qp','trac_task_5ms','trac_task_20ms'):
        if b in methods:p.append(('balance_progress_k1',b))
    return p

def paired(units,methods,metrics,cfg):
    pairs=[];changes=[]
    for robot in cfg['robots']:
        rows=[r for r in units if r['robot']==robot];uids=sorted({r['uid'] for r in rows})
        lookup={(r['method'],r['uid']):r for r in rows};families=[lookup[methods[0],u]['family'] for u in uids]
        for a,b in comparisons(methods):
            for metric in metrics:
                av=np.array([lookup[a,u][metric] for u in uids]);bv=np.array([lookup[b,u][metric] for u in uids])
                pairs.append(dict(robot=robot,method=a,baseline=b,metric=metric,units=len(uids),
                    **paired_intervals(av,bv,families,cfg['bootstrap_seed'],cfg['bootstrap_samples'])))
            metric=metrics[0]
            for u in uids:
                av=lookup[a,u][metric];bv=lookup[b,u][metric]
                changes.append(dict(robot=robot,method=a,baseline=b,uid=u,family=lookup[a,u]['family'],
                    method_fraction=av,baseline_fraction=bv,change='gained' if av>bv else 'lost' if av<bv else 'same',
                    stable_all_vs_none=abs(av-bv)==1))
    return pairs,changes

def trajectory_tables(stage,cfg,folder):
    methods=cfg['methods'] if stage=='development' else cfg['independent_methods']
    summaries=[];arrays={};extras={};first=[];late=[];witnesses=[];audit=Counter();sources={}
    for robot in cfg['robots']:
        origin=OUT/(f'development_{robot}' if stage=='development' else f'independent_trajectories_{robot}')
        seal=json.loads((origin/'completed.json').read_text());sources[robot]=sha(origin/'completed.json')
        for name,h in seal['files'].items():assert sha(origin/name)==h
        targets={r['uid']:r for r in json.loads((origin/'online_inputs.json').read_text())}
        jobs=json.loads((origin/'summaries.json').read_text());_,kin,v,_=context(robot,cfg)
        assert Counter((s['uid'],s['method'],s['repeat']) for s in jobs)==Counter((u,m,r) for u in targets for m in methods for r in range(3))
        for index,s in enumerate(jobs):
            rows=read_rows(origin/s['raw_file']);assert len(rows)==150
            t=targets[s['uid']];prev=np.array(t['initial_q']);counts=Counter();values=defaultdict(list)
            for frame,r in enumerate(rows):
                assert r['frame']==frame and r['dt']==t['dt']==.02
                for key in ('target_position','target_rotation'):np.testing.assert_array_equal(r[key],t[key][frame])
                np.testing.assert_array_equal(r['previous_q'],prev)
                query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),prev,.02)
                q=None if r['q'] is None else np.array(r['q']);verdict=v.check(q,query)
                assert bool(verdict.accepted)==r['accepted']
                assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
                audit['commands_checked']+=1;audit['accepted_commands']+=int(verdict.accepted)
                if verdict.accepted:
                    np.testing.assert_allclose([r['position_error'],r['orientation_error']],[verdict.position_error,verdict.orientation_error],atol=1e-12,rtol=0)
                    counts['accepted']+=1
                    counts['edge']+=max(verdict.position_error/v.config.position_tolerance,verdict.orientation_error/v.config.orientation_tolerance)>.9
                counts.update(r.get('reason_counts',{}));counts['late']+=r['total_latency_ns']>20_000_000
                counts['status:'+str(r.get('internal_status'))]+=1
                for info in r.get('subproblems',[]):
                    if info.get('reason')=='relative_progress':
                        p,u,l=info['pzero'],info['upper'],info['lower'];tol=64*np.finfo(float).eps*max(1,abs(p),abs(u),abs(l))
                        assert info['bounds_valid'] and relative_stop(p,u,l,.25,tol)
                        audit['consistent_relative_stops']+=1
                for key in ('iterations','evaluations','dual_updates','box_qp_updates','verification_calls','backtracking_evaluations'):
                    if r.get(key) is not None:values[key].append(r[key])
                if frame==s['first_failure_frame'] or r['total_latency_ns']>20_000_000:
                    small={k:r.get(k) for k in ('robot','uid','site_id','family','method','repeat','frame','dt','previous_q','target_position',
                        'target_rotation','q','accepted','failure_kind','verification_reasons','position_error','orientation_error','velocity_utilization','internal_status','total_latency_ns')}
                    if frame==s['first_failure_frame']:first.append(small)
                    if r['total_latency_ns']>20_000_000:late.append(small)
                if verdict.accepted:prev=q.copy()
                np.testing.assert_array_equal(r['accepted_state_q'],prev)
            calc=entry.runner.summarize(rows,kin,v)
            for key in ('completion','deadline_completion','accepted_frames','deadline_frames','total_latency_ns','first_failure_frame'):assert calc[key]==s[key]
            if s['completion']:witnesses.append(dict(robot=robot,uid=s['uid'],method=s['method'],repeat=s['repeat'],initial_q=t['initial_q'],
                raw_file=str((origin/s['raw_file']).relative_to(ROOT)),sha256=sha(origin/s['raw_file'])))
            summaries.append(s);extras[s['run_id']]=(counts,values)
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            if index%240==0:print('verifier replay',stage,robot,index+1,'/',len(jobs),flush=True)
    main=group_table(summaries,arrays);family=group_table(summaries,arrays,True)
    for table in (main,family):
        for row in table:
            ss=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method'] and (row['family']=='all' or s['family']==row['family'])]
            c=Counter();val=defaultdict(list)
            for s in ss:
                cc,vv=extras[s['run_id']];c.update(cc)
                for k,v in vv.items():val[k]+=v
            row.pop('backup_rate',None);row.pop('changed_from_backup_rate',None)
            row.update(edge_fraction=c['edge']/max(1,c['accepted']),counts=dict(c),
                all_frames=len(ss)*150,all_frame_cumulative_ns=sum(s['total_latency_ns'] for s in ss),
                **{'mean_'+k:float(np.mean(v)) for k,v in val.items()})
    by=defaultdict(list)
    for s in summaries:by[s['robot'],s['method'],s['uid']].append(s)
    metrics=('completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms','successful_prefix')
    units=[dict(robot=key[0],method=key[1],uid=key[2],family=ss[0]['family'],site_id=ss[0]['site_id'],
        **{k:float(np.mean([s[k] for s in ss])) for k in metrics}) for key,ss in by.items()]
    pairs,changes=paired(units,methods,metrics,cfg)
    completion={r:{m:{rep:sorted(s['uid'] for s in summaries if s['robot']==r and s['method']==m and s['repeat']==rep and s['completion']) for rep in range(3)} for m in methods} for r in cfg['robots']}
    for name,rows in [('main',main),('family',family),('units',units),('paired',pairs),('gained_lost_uids',changes),('first_failure_inputs',first),('late_commands',late),('runs',summaries)]:
        if rows:csv_write(folder/f'{stage}_{name}.csv',rows)
    write_json(folder/f'{stage}_completion_uids.json',completion);write_json(folder/f'{stage}_successful_witness_index.json',witnesses)
    return dict(main=main,family=family,pairs=pairs,changes=changes,verification=dict(audit),sources=sources)

def point_tables(cfg,folder):
    main=[];units=[];sources={};audit=Counter()
    for robot in cfg['robots']:
        origin=OUT/f'independent_points_{robot}';sources[robot]=sha(origin/'manifest.json')
        _,kin,v,_=context(robot,cfg);raw=read_rows(origin/'records.jsonl.gz');by=defaultdict(list)
        for r in raw:
            query=IKQuery(Pose(np.array(r['target_position']),np.array(r['target_rotation'])),np.array(r['previous_q']),.02)
            q=None if r['q'] is None else np.array(r['q']);check=v.check(q,query)
            assert r['accepted']==bool(check.accepted) and r['witness_confirmed_miss']==(not check.accepted)
            assert r['accepted_within_20ms']==bool(check.accepted and r['total_latency_ns']<=20_000_000)
            audit['commands_checked']+=1;audit['accepted']+=check.accepted;by[r['method'],r['uid']].append(r)
        for (method,uid),rr in by.items():
            assert len(rr)==3
            units.append(dict(robot=robot,method=method,uid=uid,family=rr[0]['family'],
                accepted=float(np.mean([r['accepted'] for r in rr])),deadline=float(np.mean([r['accepted_within_20ms'] for r in rr])),
                total_latency_ns=float(np.mean([r['total_latency_ns'] for r in rr])),
                accepted_by_repeat=[r['accepted'] for r in sorted(rr,key=lambda r:r['repeat'])]))
        for method in cfg['independent_methods']:
            for family in ['all']+list(cfg['fresh']['point_counts']):
                rr=[r for r in raw if r['method']==method and (family=='all' or r['family']==family)]
                uu=[r for r in units if r['robot']==robot and r['method']==method and (family=='all' or r['family']==family)]
                accepted=[r for r in rr if r['accepted']];times=np.array([r['total_latency_ns'] for r in rr])
                errors=np.array([[r['position_error'],r['orientation_error']] for r in accepted])
                main.append(dict(robot=robot,method=method,family=family,queries=len(uu),calls=len(rr),
                    success_by_repeat=[sum(r['accepted'] for r in rr if r['repeat']==rep) for rep in range(3)],
                    verified_success=float(np.mean([u['accepted'] for u in uu])),within20=float(np.mean([u['deadline'] for u in uu])),
                    missed_per_repeat=[sum(not r['accepted'] for r in rr if r['repeat']==rep) for rep in range(3)],
                    stable_missed_uids=sum(u['accepted']==0 for u in uu),mixed_uids=sum(0<u['accepted']<1 for u in uu),
                    p50_p95_p99_ms=(np.percentile(times,[50,95,99])/1e6).tolist(),mean_latency_ms=float(times.mean()/1e6),
                    cumulative_ms_per_sweep=float(times.sum()/3e6),late_calls=int(sum(times>20e6)),
                    accepted_position_p50_p95_p99_max_m=np.r_[np.percentile(errors[:,0],[50,95,99]),max(errors[:,0])].tolist(),
                    accepted_orientation_p50_p95_p99_max_rad=np.r_[np.percentile(errors[:,1],[50,95,99]),max(errors[:,1])].tolist(),
                    edge_fraction=float(np.mean(np.max(errors/np.array([v.config.position_tolerance,v.config.orientation_tolerance]),axis=1)>.9))))
        print('point verifier replay',robot,len(raw),flush=True)
    pairs,changes=paired(units,cfg['independent_methods'],('accepted','deadline','total_latency_ns'),cfg)
    for name,rows in [('main',main),('units',units),('paired',pairs),('gained_lost_uids',changes)]:csv_write(folder/f'points_{name}.csv',rows)
    write_json(folder/'point_witness_confirmed_misses.json',[u for u in units if u['accepted']<1])
    return dict(main=main,pairs=pairs,changes=changes,verification=dict(audit),sources=sources)

def micro_tables(cfg,folder):
    raw=json.loads((OUT/'subproblems/raw_results.json').read_text());by=defaultdict(list);checks=[]
    for r in raw:by[r['problem_id'],r['repeat']].append(r)
    for rows in by.values():
        lower=max(r['box_minorant_lower'] for r in rows)
        lower=max([lower]+[r['lower'] for r in rows if r.get('lower') is not None])
        ref=next(r for r in rows if r['method']=='clarabel_epigraph')
        for r in rows:
            gap=r['objective']-lower;scale=max(1,abs(r['objective']),abs(lower))
            good=gap>=-1e-8*scale and gap<=1e-9+1e-7*scale and r['feasible_box_violation']<=1e-8 and r.get('raw_box_violation',0)<=1e-8
            frac=None
            if r.get('pzero') is not None and r['pzero']>ref['objective']:
                frac=(r['pzero']-r['objective'])/(r['pzero']-ref['objective'])
            checks.append(dict(**{k:v for k,v in r.items() if k not in ('trace','direction')},common_gap=gap,
                tight_quality_pass=good,common_normalized_gap=gap/scale,model_fraction_vs_reference=frac,
                objective_difference_vs_clarabel=r['objective']-ref['objective']))
    table=[];groups=defaultdict(list)
    for r in checks:groups[r['robot'],r['stratum'],r['kappa'],r['method']].append(r)
    for key,rr in sorted(groups.items()):
        fractions=[r['model_fraction_vs_reference'] for r in rr if r['model_fraction_vs_reference'] is not None and r.get('reason')=='relative_progress']
        table.append(dict(zip(('robot','stratum','kappa','method'),key),calls=len(rr),tight_quality_calls=sum(r['tight_quality_pass'] for r in rr),
            p50_p95_p99_ms=(np.percentile([r['outer_ns'] for r in rr],[50,95,99])/1e6).tolist(),
            kernel_p50_ms=float(np.median([r['kernel_ns'] for r in rr])/1e6),mean_qps=float(np.mean([r.get('dual_updates',r.get('iterations',0)) for r in rr])),
            statuses=dict(Counter(r.get('reason',r.get('status')) for r in rr)),relative_stops=len(fractions),
            achieved_at_least80_calls=sum(f>=.8-1e-6 for f in fractions),minimum_achieved_fraction=min(fractions) if fractions else None,
            max_normalized_gap=max(r['common_normalized_gap'] for r in rr)))
    csv_write(folder/'subproblem_calls.csv',checks);csv_write(folder/'subproblem_quality_time.csv',table)
    return table

def main():
    cfg=yaml.safe_load(entry.CONFIG.read_text());folder=OUT/'reports';folder.mkdir(exist_ok=False)
    previous_reporting.PAIRS=comparisons([m for m in cfg['methods'] if m!='pink_qp'])
    local=previous_reporting.local_tables(OUT,dict(cfg,methods=[m for m in cfg['methods'] if m!='pink_qp']),folder)
    result=dict(local=local,micro=micro_tables(cfg,folder))
    for stage in ('development','independent'):result[stage]=trajectory_tables(stage,cfg,folder)
    result['points']=point_tables(cfg,folder)
    write_json(folder/'source_data.json',result)
    lines=['# Complete source-generated result tables','',
        'All three repeats are nested within UID. Latency quantiles include failures and late calls. See CSV/source_data for family results and paired intervals.','']
    for stage in ('development','independent'):
        lines += ['## '+stage+' trajectories','',
            '| Robot | Method | TSR counts, repeats | DTSR20 counts | P50/P95/P99 ms | Cumulative ms/sweep | Edge % | Acceleration RMS |',
            '|---|---|---|---|---|---:|---:|---:|']
        for r in result[stage]['main']:
            lines.append('| '+str(r['robot'])+' | '+r['method']+' | '+str(r['completion_by_repeat'])+' | '+str(r['deadline_completion_by_repeat'])+
                ' | '+('/'.join(f'{r[k]:.4f}' for k in ('p50_ms','p95_ms','p99_ms')))+f" | {r['cumulative_ms_per_sweep']:.3f} | {100*r['edge_fraction']:.3f} | {r['acceleration_rms_mean']:.5f} |")
        lines+=['']
    lines+=['## Independent witness-feasible points','',
        '| Robot | Method | Accepted /2000, repeats | Missed, repeats | P50/P95/P99 ms | Mean ms | Within20 % | Edge % |',
        '|---|---|---|---|---|---:|---:|---:|']
    for r in result['points']['main']:
        if r['family']!='all':continue
        lines.append('| '+r['robot']+' | '+r['method']+' | '+str(r['success_by_repeat'])+' | '+str(r['missed_per_repeat'])+' | '+
            '/'.join(f'{x:.4f}' for x in r['p50_p95_p99_ms'])+f" | {r['mean_latency_ms']:.4f} | {100*r['within20']:.3f} | {100*r['edge_fraction']:.3f} |")
    (folder/'TABLES.md').write_text('\n'.join(lines)+'\n')
    write_json(folder/'manifest.json',dict(created=utc(),reporting_sha256=sha(__file__),solver_calls=0,
        statistical_unit='Query or complete trajectory UID, three within-unit repeats averaged. Paired family-stratified 4000-resample descriptive unadjusted 95% intervals. No equivalence claim.',
        frame_quantiles='All calls pooled for descriptive latency quantiles; failure and late frames retained.',
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

if __name__=='__main__':main()
