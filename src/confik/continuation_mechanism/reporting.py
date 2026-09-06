"""Read-only analysis of completed windows, plus two source-backed figures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from ..geometry import pose_error
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.benchmark import read_records
from ..revision_compute_allocation.common import csv_write, digest, json_write
from .observation import metrics
from .study import Study, METHODS


def compact_rate(rows):
    return f'{sum(r["complete"] for r in rows)}/{len(rows)}' if rows else 'not run'


def analyse(study):
    out=study.out;protocol=study.check_seal()
    runs=json.loads((out/'continuation_runs.json').read_text())
    blocks=json.loads((out/'candidate_pools.json').read_text())
    replays=json.loads((out/'fresh_failure_replays.json').read_text())
    internal=json.loads((out/'internal_stage_traces.json').read_text())
    raw=read_records(out/'continuation_raw.jsonl.gz')
    rg=defaultdict(list)
    for r in raw:rg[r['run_id']].append(r)
    lookup=defaultdict(list)
    for r in runs:lookup[(r['site_id'],r['candidate_id'],r['variant'])].append(r)
    table=[];sites=[];pairs=[];failure_table=[]
    for case in protocol['cases']:
        f=case['first_failure'];uid=case['uid']
        if f is None:continue
        old=study.old[(uid,case['focal'])][f]
        row=dict(case_id=case['case_id'],uid=uid,family=case['family'],focal=case['focal'],first_failure=f,
                 original_solver_return_code=old.get('solver_return_code'),original_failure=old['reject_reason'],
                 original_verifier_reasons=old['verification_reasons'],original_candidate_unavailable=old['command_q'] is None)
        rr=[x for x in replays if x['case_id']==case['case_id'] and x['frame']==f]
        for budget,interior,label in [(5,False,'5ms'),(100,False,'100ms'),(5,True,'5ms_interior')]:
            subset=[x for x in rr if x['budget_ms']==budget and x['interior_bounds']==interior]
            row[label+'_verified_success']=f'{sum(x["accepted"] for x in subset)}/{len(subset)}'
            row[label+'_failure_kinds']=dict(Counter(x['failure_kind'] for x in subset))
            row[label+'_position_error_range']=[min(x['position_error'] for x in subset),max(x['position_error'] for x in subset)]
            row[label+'_orientation_error_range']=[min(x['orientation_error'] for x in subset),max(x['orientation_error'] for x in subset)]
            row[label+'_joint_margin_min']=min(x['joint_limit_margin'] for x in subset)
            row[label+'_velocity_excess_max']=max(x['velocity_excess_rad'] for x in subset)
        traces=next((x['traces'] for x in internal if x['case_id']==case['case_id'] and x['frame']==f),[])
        row['internal_stage_failure_kinds']=dict(Counter(t['failure_kind'] for t in traces))
        row['internal_stage_statuses']=dict(Counter(t['solver_status'] for t in traces))
        row['internal_all_candidates_finite']=all(t['finite'] for t in traces) if traces else None
        row['internal_min_position_error']=min((t['position_error'] for t in traces),default=None)
        row['internal_min_converged_velocity_excess']=min((t['velocity_excess_rad'] for t in traces if t['solver_ok']),default=None)
        row['previous_q']=old['previous_q']
        query=study.query(uid,f,old['previous_q'])
        row.update(target_position=query.target.position.tolist(),target_rotation=query.target.rotation.tolist(),query_hash=old['query_hash'])
        failure_table.append(row)
    for block in blocks:
        site=block['site'];uid=site['uid'];frame=site['frame'];candidate_rows={}
        srow=dict(site_id=site['site_id'],uid=uid,role=site['role'],family=site['family'],frame=frame,
                  horizon=site['continuation_frames'],current_query_hash=site['query_hash'],legal_candidates=len(block['candidates']))
        counts=[]
        for c in block['candidates']:
            q=np.asarray(c['q']);im=c['initial_metrics'];j=study.kin.jacobian(q)
            nextquery=study.query(uid,frame+1,q)
            need=np.linalg.pinv(j)@pose_error(nextquery.target,study.kin.forward(q))
            diagnostic_util=float(np.max(np.abs(need)/(study.kin.limits.velocity*.02+study.verifier.config.velocity_tolerance)))
            for variant in ['trac_5ms','trac_100ms','trac_5ms_interior']:
                rr=lookup[(site['site_id'],c['candidate_id'],variant)]
                if not rr:continue
                assert len(rr)==5
                row=dict(site_id=site['site_id'],uid=uid,role=site['role'],family=site['family'],frame=frame,
                    current_query_hash=site['query_hash'],candidate_id=c['candidate_id'],aliases=c['aliases'],variant=variant,
                    current_verifier_accepted=True,current_q=c['q'],initial_position_error=im['position_error'],
                    initial_orientation_error=im['orientation_error'],initial_velocity_margin_rad=im['velocity_margin_rad'],
                    initial_joint_limit_margin=im['joint_limit_margin'],initial_sigma_min=im['sigma_min'],
                    initial_reference_distance=im['to_reference_max_rad'],distance_to_previous=c['distance_to_previous'],
                    next_target_linearized_pseudoinverse_velocity_utilization=diagnostic_util,
                    nearest=c['candidate_id']==block['nearest_candidate_id'],repeats=len(rr),
                    completed=sum(r['complete'] for r in rr),horizon=site['continuation_frames'],
                    first_failure_frames=[r['first_failure'] for r in sorted(rr,key=lambda x:x['repeat'])],
                    contiguous_success_frames=[r['contiguous_success_frames'] for r in sorted(rr,key=lambda x:x['repeat'])],
                    minimum_velocity_margins=[r['minimum_accepted_velocity_margin_rad'] for r in rr],
                    first_failure_kinds=dict(Counter(r['first_failure_kind'] for r in rr if not r['complete'])))
                table.append(row)
                if variant=='trac_5ms':candidate_rows[c['candidate_id']]=row;counts.append(row['completed'])
                tags=[]
                if 'original_focal' in c['aliases']:tags.append('focal')
                if 'refined_focal' in c['aliases']:tags.append('refined_focal')
                if 'previous_first_dls' in c['aliases']:tags.append('previous_first_dls')
                if c['candidate_id']==block['nearest_candidate_id']:tags.append('nearest')
                for tag in tags:srow[tag+'_'+variant]=compact_rate(rr)
            for r in lookup[(site['site_id'],c['candidate_id'],'trac_5ms')]:
                if r['complete']:
                    witness=json.loads((out/r['witness']).read_text())
                    previous=np.asarray(site['previous_q'])
                    for row in [witness['initial'],*witness['frames']]:
                        assert np.array_equal(previous,np.asarray(row['previous_q']))
                        query=study.query(uid,row['frame'],previous)
                        assert study.verifier.check(np.asarray(row['q']),query).accepted
                        previous=np.asarray(row['q'])
        for a,b in itertools.combinations(block['candidates'],2):
            ra=candidate_rows[a['candidate_id']];rb=candidate_rows[b['candidate_id']]
            pairs.append(dict(site_id=site['site_id'],role=site['role'],uid=uid,current_query_hash=site['query_hash'],
                 a=a['candidate_id'],a_aliases=a['aliases'],b=b['candidate_id'],b_aliases=b['aliases'],
                 a_complete=ra['completed'],b_complete=rb['completed'],repeats=5,
                 completion_count_difference=ra['completed']-rb['completed'],
                 a_prefix_median=float(np.median(ra['contiguous_success_frames'])),
                 b_prefix_median=float(np.median(rb['contiguous_success_frames'])),
                 current_q_difference_max_rad=float(np.max(np.abs(np.asarray(a['q'])-b['q'])))))
        srow.update(minimum_completed=min(counts),maximum_completed=max(counts),
                    completion_spread=max(counts)-min(counts))
        sites.append(srow)
    csv_write(out/'failure_reason_table.csv',failure_table)
    csv_write(out/'same_input_continuation_table.csv',table)
    csv_write(out/'site_comparison_table.csv',sites)
    csv_write(out/'all_candidate_pairs.csv',pairs)
    velocities=[r for r in raw if r['failure_kind']=='velocity_violation']
    summary=dict(cases=len(protocol['cases']),sites=len(sites),candidate_configurations=sum(len(b['candidates']) for b in blocks),
      candidate_attempts=len(json.loads((out/'candidate_attempts.json').read_text())),
      repeated_windows=len(runs),continuation_calls=len(raw),complete_windows=sum(r['complete'] for r in runs),
      primary_complete_windows=sum(r['complete'] for r in runs if r['variant']=='trac_5ms'),
      sites_with_completion_spread=[r for r in sites if r['completion_spread']>0],
      sites_with_5_vs_0=[r for r in sites if r['maximum_completed']==5 and r['minimum_completed']==0],
      continuation_failure_counts={v:dict(Counter(r['failure_kind'] for r in raw if r['variant']==v)) for v in sorted(set(r['variant'] for r in raw))},
      velocity_rejection_max_excess=max((r['velocity_excess_rad'] for r in velocities),default=None),
      velocity_rejections_inside_library_bounds=sum(r['bound_excess_rad']<=0 for r in velocities),
      verified_witness_files=sum('witness' in r for r in runs),
      primary_witness_replay_passed=True,old_inputs_unchanged=True,
      scope='30-frame suffixes (25 at frame 124); selected cases, no full-path or population significance inference')
    json_write(out/'analysis_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def figures(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8,'axes.titlesize':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=study.out;study.check_seal()
    pools=json.loads((out/'candidate_pools.json').read_text())
    runs=json.loads((out/'continuation_runs.json').read_text())
    raw=read_records(out/'continuation_raw.jsonl.gz')
    byrun=defaultdict(list)
    for r in raw:byrun[r['run_id']].append(r)
    # Figure selection is descriptive: strongest within-site completion/prefix
    # contrast within each focal class, not exclusion from the complete tables.
    selections=[]
    for role in ['cghik_lost','trac_failed']:
        choices=[]
        for block in pools:
            s=block['site']
            if s['role']!=role or s['frame']==0:continue
            scores=[]
            for c in block['candidates']:
                rr=[r for r in runs if r['site_id']==s['site_id'] and r['candidate_id']==c['candidate_id'] and r['variant']=='trac_5ms']
                scores.append((sum(r['complete'] for r in rr),float(np.median([r['contiguous_success_frames'] for r in rr])),c['candidate_id']))
            lo=min(scores);hi=max(scores)
            choices.append(((hi[0]-lo[0],hi[1]-lo[1],s['site_id']),block,lo[2],hi[2]))
        _,block,aid,bid=max(choices,key=lambda x:x[0])
        if aid==bid:
            focal=next(c['candidate_id'] for c in block['candidates'] if 'original_focal' in c['aliases'])
            refined=next(c['candidate_id'] for c in block['candidates'] if 'refined_focal' in c['aliases'])
            aid,bid=focal,refined
        s=block['site'];uid=s['uid'];candidate={c['candidate_id']:c for c in block['candidates']}
        qa=np.asarray(candidate[aid]['q']);qb=np.asarray(candidate[bid]['q'])
        joint=int(np.argmax(np.abs(qa-qb)))
        width_in=183/25.4
        height_in=185/25.4
        fig,axs=plt.subplots(4,2,figsize=(width_in,height_in),sharex='col',layout='constrained')
        colors={'full_cghik':'#30638E','always_hard':'#757575','trac_ik_5ms':'#458A78'}
        names={'full_cghik':'CG-HIK','always_hard':'Always-hard','trac_ik_5ms':'TRAC-IK'}
        source=[]
        for method in METHODS:
            series=study.old[(uid,method)];qs=[];pe=[];oe=[];margin=[]
            for t in range(150):
                r=series[t];query=study.query(uid,t,r['previous_q'])
                state=r['command_q'] if r['accepted'] else r['previous_q']
                m=metrics(study.kin,study.verifier,query,state)
                qs.append(state[joint]);pe.append(m['position_error']*1000);oe.append(m['orientation_error']*1000)
                margin.append(m['velocity_margin_rad']*1000 if r['accepted'] else np.nan)
                source.append(dict(panel_column='original',method=method,frame=t,repeat=0,accepted=r['accepted'],
                     selected_joint=joint+1,state_q=state,position_error=m['position_error'],orientation_error=m['orientation_error'],
                     velocity_margin_rad=m['velocity_margin_rad'] if r['accepted'] else None))
            for ax,vals in zip(axs[:,0],[qs,pe,oe,margin]):ax.plot(range(150),vals,color=colors[method],lw=1,label=names[method])
            failures=[t for t,r in series.items() if not r['accepted']]
            if failures:
                f=min(failures)
                for ax in axs[:,0]:ax.axvline(f,color=colors[method],lw=.8,ls=':')
        ref=[study.reference(uid,t)[joint] for t in range(150)]
        axs[0,0].plot(range(150),ref,color='#000000',ls='--',lw=.8,label='Reference')
        source.extend(dict(panel_column='original',method='reference',frame=t,repeat=0,
                           selected_joint=joint+1,state_q=study.reference(uid,t).tolist()) for t in range(150))
        for label,cid,col in [('Candidate A',aid,'#6E6E6E'),('Candidate B',bid,'#8056A6')]:
            rr=[r for r in runs if r['site_id']==s['site_id'] and r['candidate_id']==cid and r['variant']=='trac_5ms']
            for run in rr:
                seq=sorted(byrun[run['run_id']],key=lambda r:r['frame']);q0=candidate[cid]['q'];im=candidate[cid]['initial_metrics']
                ts=[s['frame']]+[r['frame'] for r in seq]
                vals=[[q0[joint]]+[r['accepted_state_q'][joint] for r in seq],
                      [im['position_error']*1000]+[r['state_position_error']*1000 for r in seq],
                      [im['orientation_error']*1000]+[r['state_orientation_error']*1000 for r in seq],
                      [im['velocity_margin_rad']*1000]+[r['velocity_margin_rad']*1000 if r['accepted'] else np.nan for r in seq]]
                for ax,v in zip(axs[:,1],vals):ax.plot(ts,v,color=col,alpha=.5,lw=.9)
                axs[3,1].plot(ts[0],vals[3][0],marker='o',ms=2,color=col)
                if run['first_failure'] is not None:
                    f=run['first_failure'];index=ts.index(f)
                    for ax,v in zip(axs[:,1],vals):ax.plot(f,v[index],marker='x',ms=4,color=col)
                source.append(dict(panel_column='continuation',method=label,candidate_id=cid,
                     frame=s['frame'],repeat=run['repeat'],accepted=True,selected_joint=joint+1,
                     state_q=q0,position_error=im['position_error'],orientation_error=im['orientation_error'],
                     velocity_margin_rad=im['velocity_margin_rad'],failure_kind='initial_verified_candidate'))
                for r in seq:
                    source.append(dict(panel_column='continuation',method=label,candidate_id=cid,frame=r['frame'],repeat=r['repeat'],
                         accepted=r['accepted'],selected_joint=joint+1,state_q=r['accepted_state_q'],
                         position_error=r['state_position_error'],orientation_error=r['state_orientation_error'],
                         velocity_margin_rad=r['velocity_margin_rad'] if r['accepted'] else None,
                         raw_return_velocity_margin_rad=r['velocity_margin_rad'],failure_kind=r['failure_kind']))
            axs[0,1].plot([],[],color=col,label=f'{label}: {compact_rate(rr)} complete')
        axs[0,0].set_title(f'Original path: {uid[:8]}',loc='left')
        axs[0,1].set_title(f'Same input at frame {s["frame"]}; TRAC-IK suffix',loc='left')
        labels=[f'Joint {joint+1} (rad)','Position residual (mm)',
                'Orientation residual (mrad)','Verified step margin (mrad)']
        for row,label in enumerate(labels):
            for col in range(2):
                ax=axs[row,col];ax.set_ylabel(label);ax.grid(alpha=.15)
                ax.text(-.13,1.03,chr(97+row*2+col),transform=ax.transAxes,fontweight='bold',fontsize=9)
        for ax in axs[1,:]:ax.axhline(1,color='black',lw=.7,ls='--')
        for ax in axs[2,:]:ax.axhline(study.verifier.config.orientation_tolerance*1000,color='black',lw=.7,ls='--')
        for ax in axs[3,:]:ax.axhline(0,color='black',lw=.7,ls='--')
        for ax in axs[:,0]:ax.axvspan(s['frame'],s['frame']+s['continuation_frames'],color='#8056A6',alpha=.07)
        axs[0,0].legend(loc='best',fontsize=7,ncol=2)
        axs[0,1].legend(loc='best',fontsize=7)
        axs[3,0].set_xlabel('Original target frame');axs[3,1].set_xlabel('Target frame (five raw repeated suffixes)')
        stem='cghik_continuation' if role=='cghik_lost' else 'trac_ik_continuation'
        folder=out/'final_figures';folder.mkdir(exist_ok=True)
        for extension in ['svg','pdf','png']:
            target=folder/f'{stem}.{extension}'
            if target.exists():raise FileExistsError(target)
        fig.savefig(str(folder/stem)+'.svg')
        fig.savefig(str(folder/stem)+'.pdf')
        fig.savefig(str(folder/stem)+'.png',dpi=600)
        plt.close(fig)
        csv_write(folder/f'{stem}_source.csv',source)
        selections.append(dict(figure=stem,site_id=s['site_id'],candidate_a=candidate[aid],candidate_b=candidate[bid],
          selected_joint=joint+1,selection='largest observed completion then prefix contrast within focal class; all sites/pairs retained in tables',
          caption='Left: original recorded accepted/held states; dotted lines mark each first failure and shading identifies the tested suffix. Right: two currently verified candidates from identical input, five raw 5 ms TRAC-IK suffixes each; crosses mark first failures. Position/orientation residuals use the accepted/held state, not a rejected numerical proposal. Step margin is shown only for verified commands, with gaps on failures; initial candidate margins have circular markers. Raw rejected-return diagnostics remain in the source tables. All five repetitions are drawn individually and can overlap. No smoothing, p-value, or independent-frame inference. These are suffix witnesses, not full original path completions.'))
    json_write(out/'final_figures/figure_manifest.json',dict(selections=selections,
        files={str(p.relative_to(out)):digest(p) for p in (out/'final_figures').iterdir() if p.is_file()}))


def verify(study):
    """Replay stored commands through verification only; never call an IK solver."""
    out=study.out;protocol=study.check_seal()
    for path,h in protocol['measurement_sources'].items():
        assert digest(study.root/path)==h, path
    pools=json.loads((out/'candidate_pools.json').read_text())
    runs=json.loads((out/'continuation_runs.json').read_text())
    raw=read_records(out/'continuation_raw.jsonl.gz')
    grouped=defaultdict(list)
    for row in raw:grouped[row['run_id']].append(row)
    blocks={b['site']['site_id']:b for b in pools}
    accepted_count=0;witness_count=0;witness_transitions=0
    for run in runs:
        block=blocks[run['site_id']];site=block['site'];uid=site['uid']
        candidate=next(c for c in block['candidates'] if c['candidate_id']==run['candidate_id'])
        previous=np.asarray(candidate['q'])
        assert study.verifier.check(previous,study.query(uid,site['frame'],site['previous_q'])).accepted
        seq=sorted(grouped[run['run_id']],key=lambda r:r['frame'])
        assert len(seq)==site['continuation_frames']==run['frames']
        for t,row in enumerate(seq,site['frame']+1):
            assert row['frame']==t
            assert np.array_equal(previous,row['previous_q'])
            query=study.query(uid,t,previous)
            assert query_digest(query)==row['query_hash']
            assert np.array_equal(query.target.position,row['target_position'])
            assert np.array_equal(query.target.rotation,row['target_rotation'])
            if row['accepted']:
                assert row['solver_ok']
                assert study.verifier.check(np.asarray(row['q']),query).accepted
                previous=np.asarray(row['q']);accepted_count+=1
            assert np.array_equal(previous,row['accepted_state_q'])
        assert run['complete']==all(r['accepted'] for r in seq)
        if run['complete']:
            witness=json.loads((out/run['witness']).read_text())
            assert witness['frames']==seq
            previous=np.asarray(site['previous_q'])
            for row in [witness['initial'],*witness['frames']]:
                assert np.array_equal(previous,row['previous_q'])
                query=study.query(uid,row['frame'],previous)
                assert query_digest(query)==row['query_hash']
                assert study.verifier.check(np.asarray(row['q']),query).accepted
                previous=np.asarray(row['q']);witness_transitions+=1
            witness_count+=1
    assert set(grouped)=={r['run_id'] for r in runs}
    assert witness_count==len(list((out/'successful_witnesses').glob('*.json')))
    tracked_diff=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=study.root,text=True).strip()
    assert not tracked_diff, tracked_diff
    result=dict(solver_calls_in_verification=0,windows=len(runs),recorded_calls=len(raw),
                accepted_continuation_commands_reverified=accepted_count,
                successful_witness_files_reverified=witness_count,witness_transitions_reverified=witness_transitions,
                exact_previous_state_and_query_hash_checks=True,all_accepted_commands_pass=True,
                frozen_inputs_unchanged=True,measurement_sources_unchanged=True,tracked_baseline_files_unchanged=True)
    json_write(out/'verification.json',result)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    files.extend((study.root/'src/confik/continuation_mechanism').glob('*.py'))
    files.extend([study.root/'tests/test_continuation_mechanism.py',study.root/'docs/CONTINUATION_MECHANISM_FINDINGS.md'])
    json_write(out/'delivery_manifest.json',dict(baseline=protocol['baseline'],scope=protocol['role'],
        final_figures_directory='final_figures',initial_figure_exports_preserved=True,
        files={str(p.relative_to(study.root)):digest(p) for p in sorted(files)}))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['analyse','figures','verify']);p.add_argument('--root',default='.')
    args=p.parse_args();study=Study(args.root)
    globals()[args.stage](study)


if __name__=='__main__':main()
