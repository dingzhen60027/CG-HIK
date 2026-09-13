#!/usr/bin/env python3
"""Read-only command replay, anchor-cluster inference and source-backed tables.

No solve call occurs in this file. All reported commands are saved measured
outputs. The stationarity calculation is an offline residual/gradient check.
"""
from collections import Counter,defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task_set_frozen_entry',ROOT/'scripts/run_task_set_boundary.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from confik.correction_reserve.study import write_json,sha,clean,utc
from confik.correction_reserve.reporting import csv_write
from confik.correction_reserve.native_geometry import NativeGeometry
from confik.correction_reserve.geometry import residual_linearization,task_scale
from confik.task_balance_gn import relative_stop

OUT=entry.OUT;REPORT=OUT/'reports'

def read_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]

def quant(values,p):return float(np.percentile(values,p)) if len(values) else None

def mean_field(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None

def replay(split,robot,cfg):
    folder=OUT/f'{split}_{robot}';seal=json.loads((folder/'manifest.json').read_text())
    for name,h in seal['files'].items():assert sha(folder/name)==h
    items=json.loads((OUT/f'inputs/{split}_{robot}.json').read_text());index={i['uid']:i for i in items}
    _,kin,v,_=entry.context(robot,cfg);rows=read_rows(folder/'records.jsonl.gz');methods=sorted({r['method'] for r in rows})
    assert Counter((r['uid'],r['method'],r['repeat']) for r in rows)==Counter((i['uid'],m,k) for i in items for m in methods for k in range(3))
    audit=Counter()
    for i in items:
        assert v.check(np.array(i['q_witness']),entry.query_of(i)).accepted;audit['witnesses']+=1
    for r in rows:
        i=index[r['uid']];query=entry.query_of(i);q=None if r['q'] is None else np.array(r['q'])
        verdict=v.check(q,query);assert bool(verdict.accepted)==r['accepted']
        assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
        assert r['total_latency_ns']>=r['adapter_total_latency_ns']
        if verdict.finite_ok:
            np.testing.assert_allclose([r['position_error'],r['orientation_error']],
                [verdict.position_error,verdict.orientation_error],rtol=0,atol=1e-12)
        for info in r.get('subproblems',[]):
            if info['reason']=='relative_progress':
                p,u,l=[info[k] for k in ('pzero','upper','lower')]
                rounding=64*np.finfo(float).eps*max(1,abs(p),abs(u),abs(l))
                assert relative_stop(p,u,l,.25,rounding);audit['relative_bound_checks']+=1
        audit['calls']+=1;audit['accepted']+=r['accepted'];audit['late_calls']+=r['total_latency_ns']>20_000_000
    return items,rows,dict(audit)

def summarize(rows,items,cfg,robot,label,methods):
    result=[];n=len(items);anchor_count=len({i['anchor_uid'] for i in items})
    _,_,v,_=entry.context(robot,cfg)
    for method in methods:
        a=[r for r in rows if r['method']==method];accepted=[r for r in a if r['accepted']]
        assert len(a)==n*3
        lat=np.array([r['total_latency_ns']/1e6 for r in a]);ep=[r['position_error'] for r in accepted];er=[r['orientation_error'] for r in accepted]
        success_counts=Counter(r['uid'] for r in accepted)
        first=[r['first_half_accepted'] for r in a if 'first_half_accepted' in r]
        continued=[bool(r['subproblems']) for r in a if 'subproblems' in r]
        multiple=[any(i['dual_updates']>1 for i in r['subproblems']) for r in a if 'subproblems' in r]
        forcing=[any(i['reason']=='relative_progress' for i in r['subproblems']) for r in a if 'subproblems' in r]
        infos=[i for r in a for i in r.get('subproblems',[])]
        result.append(dict(robot=robot,group=label,method=method,queries=n,anchors=anchor_count,repeats=3,
            success_by_repeat=[sum(r['accepted'] for r in a if r['repeat']==k) for k in range(3)],
            within20_by_repeat=[sum(r['accepted_within_20ms'] for r in a if r['repeat']==k) for k in range(3)],
            mean_success_count=sum(r['accepted'] for r in a)/3,verified_success=sum(r['accepted'] for r in a)/len(a),
            within20_success=sum(r['accepted_within_20ms'] for r in a)/len(a),
            all_repeat_success_queries=sum(success_counts[i['uid']]==3 for i in items),
            witness_missed_unique_queries=len({r['uid'] for r in a if not r['accepted']}),
            p50_ms=quant(lat,50),p95_ms=quant(lat,95),p99_ms=quant(lat,99),mean_ms=float(lat.mean()),
            cumulative_ms_per_repeat=float(lat.sum()/3),late_calls=sum(lat>20),
            accepted_position_p95_m=quant(ep,95),accepted_position_max_m=quant(ep,100),
            accepted_orientation_p95_rad=quant(er,95),accepted_orientation_max_rad=quant(er,100),
            accepted_pose_utilization_p95=quant([max(p/v.config.position_tolerance,r/v.config.orientation_tolerance) for p,r in zip(ep,er)],95),
            accepted_edge_fraction=float(np.mean([max(p/v.config.position_tolerance,r/v.config.orientation_tolerance)>.9 for p,r in zip(ep,er)])) if accepted else None,
            accepted_position_margin_p05_m=quant([v.config.position_tolerance-p for p in ep],5),
            accepted_orientation_margin_p05_rad=quant([v.config.orientation_tolerance-r for r in er],5),
            max_accepted_step_utilization=max([r['velocity_utilization'] for r in accepted],default=None),
            mean_iterations=mean_field(a,'iterations'),
            mean_evaluations=mean_field(a,'evaluations'),
            first_half_accepted_call_fraction=float(np.mean(first)) if first else None,
            continued_after_first_call_fraction=float(np.mean(continued)) if continued else None,
            multiple_local_qp_call_fraction=float(np.mean(multiple)) if multiple else None,
            relative_stop_call_fraction=float(np.mean(forcing)) if forcing else None,
            relative_stop_subproblem_fraction=sum(i['reason']=='relative_progress' for i in infos)/len(infos) if infos else None,
            mean_dual_updates=mean_field(a,'dual_updates'),
            internal_status_counts=dict(Counter(str(r.get('internal_status')) for r in a)),
            failure_counts=dict(Counter(r['failure_kind'] for r in a if not r['accepted']))))
    return result

def units(rows,items):
    grouped=defaultdict(list)
    for r in rows:grouped[r['uid'],r['method']].append(r)
    index={i['uid']:i for i in items};output=[]
    for (uid,method),a in sorted(grouped.items()):
        i=index[uid];output.append(dict(uid=uid,method=method,robot=i['robot'],anchor_uid=i['anchor_uid'],anchor_family=i['anchor_family'],
            displacement=i['displacement'],alpha=i['alpha'],accepted=float(np.mean([r['accepted'] for r in a])),
            timely=float(np.mean([r['accepted_within_20ms'] for r in a])),latency_ns=float(np.mean([r['total_latency_ns'] for r in a])),
            success_by_repeat=[next(r['accepted'] for r in a if r['repeat']==k) for k in range(3)]))
    return output

def groups(items,cfg):
    yield 'all',items
    for d in cfg['displacements']:
        for a in cfg['alpha']:
            yield f'{d}/alpha={a}',[i for i in items if i['displacement']==d and i['alpha']==a]
    for f in cfg['anchor_families']:
        for d in cfg['displacements']:
            for a in cfg['alpha']:
                yield f'{f}/{d}/alpha={a}',[i for i in items if i['anchor_family']==f and i['displacement']==d and i['alpha']==a]

def paired(unit_rows,selected,label,cfg,robot,methods):
    ids={i['uid'] for i in selected};lookup={(r['uid'],r['method']):r for r in unit_rows if r['uid'] in ids}
    anchor_ids=sorted({i['anchor_uid'] for i in selected});anchor_queries={a:[i['uid'] for i in selected if i['anchor_uid']==a] for a in anchor_ids}
    family={i['anchor_uid']:i['anchor_family'] for i in selected};rng=np.random.default_rng(cfg['bootstrap_seed'])
    indices=np.concatenate([rng.choice([j for j,a in enumerate(anchor_ids) if family[a]==f],
        (cfg['bootstrap_samples'],sum(family[a]==f for a in anchor_ids)),replace=True)
        for f in cfg['anchor_families'] if any(family[a]==f for a in anchor_ids)],axis=1)
    output=[];changes=[]
    for baseline in methods:
        if baseline=='relative':continue
        # Query repeat-mean vectors -> equal-sized anchor blocks -> paired bootstrap.
        a=np.array([[np.mean([lookup[u,'relative'][k] for u in anchor_queries[x]]) for k in ('accepted','timely','latency_ns')] for x in anchor_ids])
        b=np.array([[np.mean([lookup[u,baseline][k] for u in anchor_queries[x]]) for k in ('accepted','timely','latency_ns')] for x in anchor_ids])
        ba=a[indices].mean(axis=1);bb=b[indices].mean(axis=1)
        dif=np.array([lookup[u,'relative']['accepted']-lookup[u,baseline]['accepted'] for u in sorted(ids)])
        success=ba[:,0]-bb[:,0];timely=ba[:,1]-bb[:,1];ratio=ba[:,2]/bb[:,2]
        output.append(dict(robot=robot,group=label,method='relative',baseline=baseline,queries=len(ids),anchors=len(anchor_ids),
            success_difference=float(a[:,0].mean()-b[:,0].mean()),success_difference_ci=np.percentile(success,[2.5,97.5]).tolist(),
            timely_difference=float(a[:,1].mean()-b[:,1].mean()),timely_difference_ci=np.percentile(timely,[2.5,97.5]).tolist(),
            latency_ratio=float(a[:,2].mean()/b[:,2].mean()),latency_ratio_ci=np.percentile(ratio,[2.5,97.5]).tolist(),
            gained_query_units=int(sum(dif>0)),lost_query_units=int(sum(dif<0)),net_mean_success_count=float(dif.sum()),
            stable_gains=sum(lookup[u,'relative']['accepted']==1 and lookup[u,baseline]['accepted']==0 for u in ids),
            stable_losses=sum(lookup[u,'relative']['accepted']==0 and lookup[u,baseline]['accepted']==1 for u in ids),
            interval='4000 paired bootstrap resamples of anchors within geometric class; repeat means inside query; descriptive unadjusted95%'))
        if label=='all':
            for u in sorted(ids):
                x,y=lookup[u,'relative'],lookup[u,baseline]
                if x['accepted']!=y['accepted']:
                    changes.append(dict(robot=robot,uid=u,anchor_uid=x['anchor_uid'],anchor_family=x['anchor_family'],
                        displacement=x['displacement'],alpha=x['alpha'],baseline=baseline,
                        relative_success=x['accepted'],baseline_success=y['accepted'],
                        relative_by_repeat=x['success_by_repeat'],baseline_by_repeat=y['success_by_repeat']))
    return output,changes

def stationarity(q,query,kin,v,native):
    e,J,_=residual_linearization(native,query.target,q,task_scale(v));S=kin.limits.velocity*query.dt+v.config.velocity_tolerance
    span=kin.limits.upper-kin.limits.lower;mid=(kin.limits.lower+kin.limits.upper)/2
    lo=np.maximum(kin.limits.lower+1e-12,query.previous_q-S*(1-1e-12));hi=np.minimum(kin.limits.upper-1e-12,query.previous_q+S*(1-1e-12))
    G=J*S;posture=(q-mid)/span;g=G.T@e+(S/span)*posture;dl=(lo-q)/S;du=(hi-q)/S
    projected=np.clip(-g,dl,du)
    return dict(normalized_error_vector=e.tolist(),normalized_block_norms=[float(np.linalg.norm(e[:3])),float(np.linalg.norm(e[3:]))],
        actual_error_vector=(e*task_scale(v)).tolist(),task_cost=.5*float(e@e),worst_utilization=float(max(np.linalg.norm(e[:3]),np.linalg.norm(e[3:]))),
        gradient_scaled_task_plus_kappa_posture=g.tolist(),projected_gradient_inf=float(np.linalg.norm(projected,np.inf)),
        zero_box_violation=float(max(0,np.max(dl),np.max(-du))),dynamic_lower=lo.tolist(),dynamic_upper=hi.tolist(),
        closest_physical_margin=float(np.min(np.minimum(q-kin.limits.lower,kin.limits.upper-q))),
        closest_dynamic_margin=float(np.min(np.minimum(q-lo,hi-q))),
        scope='Offline diagnostic at saved final q: zero-damping-step gradient includes original kappa1 posture. No optimizer call, global-minimum or infeasibility claim.')

def recovered_commands(items,rows,cfg,robot):
    lookup=defaultdict(list)
    for r in rows:lookup[r['uid'],r['method']].append(r)
    _,kin,v,urdf=entry.context(robot,cfg);native=NativeGeometry(kin,urdf);records=[]
    methods=cfg['methods']
    for i in items:
        a=lookup[i['uid'],'gn'];b=lookup[i['uid'],'relative']
        if any(not r['accepted'] for r in a) and any(r['accepted'] for r in b):
            commands={m:lookup[i['uid'],m] for m in methods};diagnostics=[]
            for r in a:
                if not r['accepted'] and r['q'] is not None:
                    diagnostics.append(dict(repeat=r['repeat'],native_status=r['internal_status'],
                        **stationarity(np.array(r['q']),entry.query_of(i),kin,v,native)))
            records.append(dict(input=i,commands=commands,gn_stationarity=diagnostics,
                witness='Every relative accepted q is a saved real command on this exact input; the geometric witness is separately identified.'))
    return records

def main():
    cfg=yaml.safe_load(entry.CONFIG.read_text());entry.check_inputs();REPORT.mkdir(exist_ok=True)
    assert not any(REPORT.iterdir()), 'Never overwrite an existing report; empty failed render only'
    summary=[];comparisons=[];changes=[];allunits=[];allrecoveries=[];audits={};inputgroups=[]
    for split in ('development','validation'):
        for robot in cfg['robots']:
            items,rows,audit=replay(split,robot,cfg);audits[f'{split}_{robot}']=audit
            methods=sorted({r['method'] for r in rows});u=units(rows,items)
            for r in u:r['split']=split
            allunits+=u
            for label,selected in groups(items,cfg):
                ids={i['uid'] for i in selected};filtered=[r for r in rows if r['uid'] in ids]
                s=summarize(filtered,selected,cfg,robot,label,methods)
                for r in s:r['split']=split
                summary+=s
                if split=='validation':
                    comp,c=paired(u,selected,label,cfg,robot,methods);comparisons+=comp;changes+=c
                inputgroups.append(dict(split=split,robot=robot,group=label,queries=len(selected),
                    min_witness_utilization=float(np.min([i['witness_velocity_utilization'] for i in selected])),
                    max_witness_utilization=float(np.max([i['witness_velocity_utilization'] for i in selected])),
                    queries_with_physical_clipping=sum(any(i['physical_clipped']) for i in selected)))
            if split=='validation':allrecoveries+=recovered_commands(items,rows,cfg,robot)
            print('Replayed and aggregated',split,robot,audit,flush=True)
    csv_write(REPORT/'all_cells.csv',summary);csv_write(REPORT/'query_units.csv',allunits)
    csv_write(REPORT/'paired_cluster_intervals.csv',comparisons);csv_write(REPORT/'gained_lost_uids.csv',changes)
    csv_write(REPORT/'geometric_load.csv',inputgroups)
    write_json(REPORT/'recovery_commands.json',allrecoveries)
    write_json(REPORT/'source_data.json',dict(summary=summary,comparisons=comparisons,changes=changes,verification=audits,
        selected_weight=json.loads((OUT/'weight_selection.json').read_text())))
    with (REPORT/'TABLES.md').open('x') as f:
        f.write('# Task-set boundary: all cells and comparators\n\nCounts list three nested repeats, not independent samples. Each validation cell has 240 queries/240 geometric anchors; all-cell analyses cluster nine queries per anchor. Costs are original-server whole-call measurements.\n\n')
        for split in ('development','validation'):
            for robot in cfg['robots']:
                f.write(f'## {split}: {robot}\n\n| Cell | Method | Verified counts (3 repeats) | Within20 counts | P50/P95/P99 ms | Mean ms | Accepted edge fraction |\n|---|---|---|---|---|---|---|\n')
                for r in summary:
                    if r['split']!=split or r['robot']!=robot or r['group'].count('/')>1:continue
                    f.write(f"| {r['group']} | {r['method']} | {r['success_by_repeat']} | {r['within20_by_repeat']} | {r['p50_ms']:.4f}/{r['p95_ms']:.4f}/{r['p99_ms']:.4f} | {r['mean_ms']:.4f} | {r['accepted_edge_fraction']:.5f} |\n")
        f.write('\n## Relative-progress paired comparisons (validation)\n\n| Robot | Cell | Baseline | Gain/loss query units | Success difference [95% CI], pp | Mean time ratio [95% CI] |\n|---|---|---|---|---|---|\n')
        for r in comparisons:
            if r['group'].count('/')>1:continue
            ci=r['success_difference_ci'];ti=r['latency_ratio_ci']
            f.write(f"| {r['robot']} | {r['group']} | {r['baseline']} | {r['gained_query_units']}/{r['lost_query_units']} | {100*r['success_difference']:.3f} [{100*ci[0]:.3f},{100*ci[1]:.3f}] | {r['latency_ratio']:.4f} [{ti[0]:.4f},{ti[1]:.4f}] |\n")
    write_json(REPORT/'manifest.json',dict(created=utc(),reporting_sha256=sha(Path(__file__)),solver_calls=0,verification=audits,
        sources={f'{s}_{r}':sha(OUT/f'{s}_{r}/manifest.json') for s in ('development','validation') for r in cfg['robots']},
        files={p.name:sha(p) for p in REPORT.iterdir() if p.is_file()},
        independent_units='240 anchors per robot, nine derived queries and three nested calls; no trajectory analysis or global infeasibility claims.'))

if __name__=='__main__':main()
