"""Read-only measurements to delivery tables and branch diagnostics; no IK calls."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .common import csv_write,json_write,digest
from .figures import draw,DISPLAY,ROBOT


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])+'\n'


def run(root):
    out=root/'outputs/revision_compute_allocation';dest=out/'reports'
    tables=json.loads((dest/'supplementary_tables.json').read_text())
    text=['# Supplementary computation-allocation tables\n',
          'New supplementary evaluation; the original trajectory grouping is post-hoc descriptive. '
          'Ratios and 95% percentile intervals use paired, family-stratified resampling of queries '
          '(three nested calls) or entire trajectories (150 dependent frames), with 4,000 resamples. '
          'All fixed-length calls are retained. Quantiles below are whole-call empirical quantiles, '
          'not sums of stage quantiles. No composite pass/fail gate is used.\n']
    name=lambda s:DISPLAY.get(s,s.replace('trac_ik_','TRAC-IK '))
    for subset,title in [('feasible','Witness-feasible points: 2,500 queries per robot'),
                         ('constructed_inexecutable','Constructed inexecutable points: 500 queries per robot')]:
        text += ['## '+title+'\n',md_table(['Robot','Method','Verified %','Within 20 ms %','Mean ms','P50','P95','P99','Mean FEV','Reject / defer / fallback calls'],[
            [ROBOT[r['robot']],name(r['method']),f"{100*r['verified_success_rate']:.3f}",f"{100*r['accepted_within_20ms_rate']:.3f}",
             f"{r['total_latency_ns']/r['calls']/1e6:.3f}",*[f"{r[k]:.3f}" for k in ['p50_ms','p95_ms','p99_ms']],
             f"{r['mean_fev']:.3f}" if r['mean_fev']!='' else 'not comparable',f"{r['reject_count']} / {r['defer_count']} / {r['fallback_count']}"]
            for r in tables['point_main_table'] if r['subset']==subset])]
    text += ['## Paired feasible-point effects\n',md_table(['Robot','Comparison','Mean time ratio [95% CI]','FEV ratio [95% CI]','Success difference (pp)'],[
        [ROBOT[r['robot']],name(r['method'])+' / '+name(r['reference']),
         f"{r['latency_ratio']:.4f} [{r['latency_ratio_ci_low']:.4f}, {r['latency_ratio_ci_high']:.4f}]",
         f"{r['fev_ratio']:.4f} [{r['fev_ratio_ci_low']:.4f}, {r['fev_ratio_ci_high']:.4f}]" if 'fev_ratio' in r else 'not comparable',
         f"{100*r['success_difference']:.3f} [{100*r['success_difference_ci_low']:.3f}, {100*r['success_difference_ci_high']:.3f}]"]
        for r in tables['point_paired_comparisons'] if r['subset']=='feasible' and (r['method']=='full_cghik' or r['method']=='routing_only' and r['reference']=='always_hard')])]
    text += ['## Feasible reference trajectories: 40 per robot\n',md_table(['Robot','Method','Complete / 40','All frames within 20 ms','Frame verified %','Frame within 20 ms %','Total s','P50 / P95 / P99 ms','Mean FEV'],[
        [ROBOT[r['robot']],name(r['method']),r['completed'],r['deadline_complete'],f"{100*r['verified_success_rate']:.3f}",f"{100*r['accepted_within_20ms_rate']:.3f}",
         f"{r['total_latency_ns']/1e9:.3f}",' / '.join(f"{r[k]:.3f}" for k in ['p50_ms','p95_ms','p99_ms']),
         f"{r['mean_fev']:.3f}" if r['mean_fev']!='' else 'not comparable'] for r in tables['trajectory_main_table'] if r['family']=='all'])]
    text += ['## Trajectory families: ten per robot and family\n',md_table(['Robot','Family','Method','Complete / 10','Total s','Mean FEV'],[
        [ROBOT[r['robot']],r['family'],name(r['method']),r['completed'],f"{r['total_latency_ns']/1e9:.3f}",f"{r['mean_fev']:.3f}" if r['mean_fev']!='' else 'not comparable']
        for r in tables['trajectory_main_table'] if r['family']!='all'])]
    text += ['## Reference-trajectory cost decomposition: CG-HIK vs hard\n',
             'Outcome-conditioned groups describe where the measured difference resides; they are not randomized causal subgroups.\n',
             md_table(['Robot','Group','Trajectories','Hard s','CG-HIK s','Hard minus CG-HIK s','Hard FEV','CG-HIK FEV'],[
        [ROBOT[r['robot']],r['group'],r['trajectory_count'],f"{r['reference_total_ns']/1e9:.3f}",f"{r['method_total_ns']/1e9:.3f}",f"{r['saved_ns']/1e9:.3f}",r['reference_total_fev'],r['method_total_fev']]
        for r in tables['feasible_trajectory_savings_decomposition'] if r['family']=='all' and r['method']=='full_cghik' and r['reference']=='always_hard'])]
    text += ['## Paired trajectory effects\n',md_table(['Robot','Comparison','Total time ratio [95% CI]','Completion difference (pp) [95% CI]','Lost / gained UIDs'],[
        [ROBOT[r['robot']],name(r['method'])+' / '+name(r['reference']),f"{r['latency_ratio']:.4f} [{r['latency_ratio_ci_low']:.4f}, {r['latency_ratio_ci_high']:.4f}]",
         f"{100*r['completion_difference']:.1f} [{100*r['completion_difference_ci_low']:.1f}, {100*r['completion_difference_ci_high']:.1f}]",f"{len(json.loads(r['lost_uids']))} / {len(json.loads(r['gained_uids']))}"]
        for r in tables['trajectory_paired_comparisons'] if r['family']=='all'])]
    checks={};diagnostics=[];external_rejections=[]
    for robot in ['panda','ur5e']:
        lost=json.loads(next(r['lost_uids'] for r in tables['trajectory_paired_comparisons'] if r['robot']==robot and r['family']=='all' and r['method']=='full_cghik' and r['reference']=='always_hard'))
        byuid=defaultdict(dict);counter=Counter()
        with gzip.open(out/f'03_feasible_trajectory_benchmark/{robot}_raw_records.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);counter[r['method']]+=1
                if r['uid'] in lost:byuid[r['uid']][r['method'],r['frame']]=r
        assert set(counter.values())=={6000}
        checks[robot]={'trajectory_calls_per_method':dict(counter)}
        for uid,rr in byuid.items():
            diff=next(t for t in range(150) if rr['full_cghik',t]['accepted'] and rr['always_hard',t]['accepted'] and np.max(np.abs(np.array(rr['full_cghik',t]['command_q'])-rr['always_hard',t]['command_q']))>1e-10)
            fail=next(t for t in range(150) if not rr['full_cghik',t]['accepted'])
            a,b=rr['full_cghik',diff],rr['always_hard',diff];fa,fb=rr['full_cghik',fail],rr['always_hard',fail]
            diagnostics.append(dict(robot=robot,uid=uid,family=a['family'],first_command_divergence_frame=diff,
                divergence_same_input=a['query_hash']==b['query_hash'],first_command_difference_max_rad=float(np.max(np.abs(np.array(a['command_q'])-b['command_q']))),
                first_failure_frame=fail,first_failure_route=fa['route'],first_failure_reason=fa['reject_reason'],
                first_failure_stages=json.dumps(fa['stages']),hard_accepted_same_frame=fb['accepted'],
                previous_state_difference_at_failure_max_rad=float(np.max(np.abs(np.array(fa['previous_q'])-fb['previous_q'])))))
        rc=Counter();pc=Counter();violations=0
        with gzip.open(out/f'02_point_mechanism_benchmark/{robot}_raw_records.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);pc[r['method']]+=1;violations+=int(r['accepted_contract_violation'])
                if r['method'].startswith('trac_ik') and r['witness'] and not r['accepted']:
                    rc[r['method'],r['solver_return_code'],r['reject_reason'],','.join(r['verification_reasons'])]+=1
        assert set(pc.values())=={9000} and len(pc)==10 and violations==0
        checks[robot]['point_calls_per_method']=dict(pc)
        for (method,code,reason,vr),count in sorted(rc.items()):
            external_rejections.append(dict(robot=robot,method=method,return_code=code,reason=reason,verification_reasons=vr,calls=count))
        for method in ['full_cghik','routing_only']:
            groups=[r for r in tables['feasible_trajectory_savings_decomposition'] if r['robot']==robot and r['method']==method and r['reference']=='always_hard' and r['family']=='all']
            for actual,field in [(method,'method'),('always_hard','reference')]:
                total=next(r for r in tables['trajectory_main_table'] if r['robot']==robot and r['family']=='all' and r['method']==actual)
                assert sum(r[field+'_total_ns'] for r in groups)==total['total_latency_ns']
                assert sum(r[field+'_total_fev'] for r in groups)==round(total['mean_fev']*6000)
        checks[robot]['trajectory_decomposition_additive']=True
    csv_write(dest/'branch_divergence_diagnostics.csv',diagnostics)
    csv_write(dest/'external_verifier_rejections.csv',external_rejections)
    text += ['## Supporting records and figures\n',
             'Full completion UIDs, first-failure frames/reasons and trajectory cumulative mean/median/P95 appear in '
             '`trajectory_main_table.csv`, `trajectory_units.csv` and `trajectory_paired_comparisons.csv`. '
             'Accepted pose errors and joint steps are retained in the main tables and raw records. '
             'All point witnesses had zero learned false rejection. External nonnegative solver returns '
             'that fail the public velocity check remain failures; they are itemized in `external_verifier_rejections.csv`.\n',
             'The first partial adapter run was discarded in full after a code-inspection timing correction; '
             'its raw file and seal remain in `diagnostic_adapter_attempt_01`. Neither its outcomes nor the '
             'complete supplemental outcomes were used to tune the method. See the measurement notes.\n']
    publication=dest/'publication_figures'
    draw(root,out,tables,dest=publication)
    for filename,title in [('cost_sources_decomposition','Original trajectory savings decomposition'),('internal_strategy_mechanisms','Six internal strategies, feasible points'),('trac_ik_success_time','External verified-success versus actual time'),('feasible_reference_trajectories','Reference-path completion and all-frame cost')]:
        text += [f'### {title}\n',f'![{title}]({(publication/(filename+".png")).resolve()})\n']
    with (dest/'SUPPLEMENTARY_MAIN_TABLES.md').open('x') as f:f.write('\n'.join(text))
    json_write(dest/'analysis_delivery_manifest.json',dict(role='supplementary evaluation; no additional solver calls',checks=checks,
         source_tables_sha256=digest(dest/'supplementary_tables.json'),
         figures={p.name:digest(p) for p in publication.iterdir()},
         interpretation_unit='query or trajectory, never independent repeats/frames',
         initial_render_preserved='Initial report_manifest and render outputs are unchanged; publication figures use ordinary numeric log tick labels to keep every glyph above 5 pt'))


if __name__=='__main__':run(Path(__file__).resolve().parents[3])
