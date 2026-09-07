"""Descriptive reporting and stored-command verification, with no solver calls."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.benchmark import read_records
from ..revision_compute_allocation.common import csv_write, json_write, digest
from .disentangle import Disentanglement
from .disentangle_math import MATCH_POSITION, MATCH_ORIENTATION


def summarize(rr):
    assert len(rr)==5
    ordered=sorted(rr,key=lambda r:r['repeat'])
    return dict(completed=sum(r['complete'] for r in rr),repeats=5,
        first_failures=[r['first_failure'] for r in ordered],
        prefixes=[r['contiguous_success_frames'] for r in ordered],
        prefix_median=float(np.median([r['contiguous_success_frames'] for r in rr])),
        cumulative_latency_ms=[r['total_latency_ns']/1e6 for r in ordered],
        cumulative_mean_ms=float(np.mean([r['total_latency_ns']/1e6 for r in rr])),
        cumulative_median_ms=float(np.median([r['total_latency_ns']/1e6 for r in rr])),
        failure_kinds=dict(Counter(r['first_failure_kind'] for r in rr if not r['complete'])))


def analyse(study):
    study.protect();out=study.out
    runs=json.loads((out/'new_continuation_runs.json').read_text())
    blocks=json.loads((out/'matched_candidates.json').read_text())
    diagnostic=json.loads((out/'candidate_diagnostics.json').read_text())
    features={(d['site_id'],d['candidate_id']):d for d in diagnostic}
    newlookup=defaultdict(list);oldlookup=defaultdict(list)
    for r in runs:newlookup[(r['site_id'],r['candidate_id'],r['variant'])].append(r)
    for r in study.oldruns:oldlookup[(r['site_id'],r['candidate_id'],r['variant'])].append(r)
    uniform=[];matched=[];ordinary={};candidate_table=[];availability=[]
    for block in study.pools:
        site=block['site']
        for c in block['candidates']:
            sid=site['site_id'];cid=c['candidate_id'];f=features[(sid,cid)]
            r=summarize(oldlookup[(sid,cid,'trac_5ms')])
            candidate_table.append(dict(site_id=sid,candidate_id=cid,role=site['role'],
                population='reused_old_context',boundary='historical_original_bounds',
                **f['current'],next_demand=f['next_demand'],**r))
            rr=oldlookup[(sid,cid,'trac_5ms_interior')]
            if rr:ordinary[(sid,cid)]=summarize(rr)
    for block in blocks:
        site=block['site'];sid=site['site_id'];old=next(b for b in study.sites if b['site']['site_id']==sid)
        focal=next(c for c in old['candidates'] if 'original_focal' in c['aliases'])
        ref=next(c for c in old['candidates'] if 'refined_focal' in c['aliases'])
        a=summarize(oldlookup[(sid,focal['candidate_id'],'trac_5ms_interior')])
        current=summarize(oldlookup[(sid,ref['candidate_id'],'trac_5ms_interior')])
        br=newlookup[(sid,focal['candidate_id'],'uniform_refinement')];b=summarize(br)
        uniform.append(dict(site_id=sid,uid=site['uid'],role=site['role'],frame=site['frame'],horizon=site['continuation_frames'],
            original_id=focal['candidate_id'],refined_id=ref['candidate_id'],A=a,current_only=current,B=b,
            B_refinement_residual_calls=[r['refinement_residual_calls'] for r in br],
            B_refinement_residual_calls_mean=float(np.mean([r['refinement_residual_calls'] for r in br])),
            B_refinement_scipy_nfev_mean=float(np.mean([r['refinement_scipy_nfev'] for r in br])),
            B_initial_refinement_ms=[r['initial_refinement_latency_ns']/1e6 for r in br],
            B_initial_residual_calls=[r['initial_refinement_residual_calls'] for r in br],
            total_FEV=None,native_TRAC_FEV=None,
            timing_note='A/current-only historical interior-bound data; B new solve+refine+verify; initial prescribed-candidate work separately reported'))
        for center in block['centers']:
            children=[c for c in block['candidates'] if c['center_source']==center['source']]
            availability.append(dict(site_id=sid,role=site['role'],center_source=center['source'],center_id=center['candidate_id'],
                matched_count=len(children),available=bool(children)))
        for c in block['candidates']:
            cid=c['candidate_id'];f=features[(sid,cid)];cf=features[(sid,c['center_id'])]
            r=summarize(newlookup[(sid,cid,'ordinary_matched')]);ordinary[(sid,cid)]=r
            base=ordinary[(sid,c['center_id'])]
            matched.append(dict(site_id=sid,uid=site['uid'],role=site['role'],candidate_id=cid,
                center_id=c['center_id'],center_source=c['center_source'],projection=c['projection'],
                center=base,candidate=r,center_features=cf['current'],candidate_features=f['current'],
                center_demand=cf['next_demand'],candidate_demand=f['next_demand']))
            candidate_table.append(dict(site_id=sid,candidate_id=cid,role=site['role'],
                population='new_pose_matched',boundary='representable_interior',
                **f['current'],next_demand=f['next_demand'],**r))
    raw=read_records(out/'new_continuation_raw.jsonl.gz')
    oldraw=read_records(study.oldout/'continuation_raw.jsonl.gz')
    first_frames=[]
    for row in raw:
        site=next(b['site'] for b in study.sites if b['site']['site_id']==row['site_id'])
        if row['frame']!=site['frame']+1:continue
        obs=row['solver_observation']
        first_frames.append(dict(site_id=row['site_id'],candidate_id=row['candidate_id'],variant=row['variant'],repeat=row['repeat'],
            accepted=row['accepted'],failure_kind=row['failure_kind'],position_error=obs['position_error'],
            orientation_error=obs['orientation_error'],velocity_margin=obs['velocity_margin_rad'],q=row['q']))
    for row in oldraw:
        if row['variant']!='trac_5ms_interior':continue
        site=next(b['site'] for b in study.sites if b['site']['site_id']==row['site_id'])
        if row['frame']!=site['frame']+1:continue
        first_frames.append(dict(site_id=row['site_id'],candidate_id=row['candidate_id'],variant='reused_ordinary',repeat=row['repeat'],
            accepted=row['accepted'],failure_kind=row['failure_kind'],position_error=row['position_error'],
            orientation_error=row['orientation_error'],velocity_margin=row['velocity_margin_rad'],q=row['q']))
    pairs=[]
    residual_tie=2*np.hypot(MATCH_POSITION/study.study.verifier.config.position_tolerance,
                          MATCH_ORIENTATION/study.study.verifier.config.orientation_tolerance)
    for block in blocks:
        sid=block['site']['site_id'];keys=[k for k in ordinary if k[0]==sid]
        pose_group={c['candidate_id']:c['center_id'] for c in block['candidates']}
        for a,b in itertools.combinations(keys,2):
            ra,rb=ordinary[a],ordinary[b]
            ya=(ra['completed'],ra['prefix_median']);yb=(rb['completed'],rb['prefix_median'])
            if ya==yb:continue
            fa,fb=features[a],features[b]
            sign=1 if ya>yb else -1
            ga=pose_group.get(a[1],a[1]);gb=pose_group.get(b[1],b[1])
            row=dict(site_id=sid,role=block['site']['role'],a=a[1],b=b[1],a_completed=ra['completed'],
                b_completed=rb['completed'],a_prefix=ra['prefix_median'],b_prefix=rb['prefix_median'],
                same_pose_group=ga==gb,completion_differs=ra['completed']!=rb['completed'])
            predictors={'residual':(-fa['current']['normalized_residual_norm'],-fb['current']['normalized_residual_norm'],residual_tie),
                'nearest':(-fa['current']['distance_to_previous'],-fb['current']['distance_to_previous'],1e-10),
                'sigma_min':(fa['current']['scaled_sigma_min'],fb['current']['scaled_sigma_min'],1e-10)}
            da,db=fa['next_demand'],fb['next_demand']
            if da['optimum_reported'] and db['optimum_reported']:
                predictors['next_demand']=(-da['demand'],-db['demand'],1e-6)
            for name in ['residual','nearest','sigma_min','next_demand']:
                if name not in predictors:row[name+'_concordance']=None;continue
                x,y,tol=predictors[name];difference=x-y
                row[name+'_concordance']=.5 if abs(difference)<=tol else float((difference>0)==(sign>0))
            pairs.append(row)
    explanatory=[]
    for scope in ['all_informative','completion_only','pose_matched_informative']:
        selected=[r for r in pairs if scope=='all_informative' or
                  (scope=='completion_only' and r['completion_differs']) or
                  (scope=='pose_matched_informative' and r['same_pose_group'])]
        for predictor in ['residual','nearest','sigma_min','next_demand']:
            values=[r[predictor+'_concordance'] for r in selected if r[predictor+'_concordance'] is not None]
            explanatory.append(dict(scope=scope,predictor=predictor,informative_pairs=len(selected),
                comparable_pairs=len(values),unknown_pairs=len(selected)-len(values),
                concordant=sum(v==1 for v in values),discordant=sum(v==0 for v in values),ties=sum(v==.5 for v in values),
                score=float(np.mean(values)) if values else None,
                contributing_sites=sorted(set(r['site_id'] for r in selected)),
                note='descriptive dependent within-site pairs, not accuracy on independent held-out observations'))
    summary=dict(reused_sites=18,reused_candidates=95,new_candidates=sum(len(b['candidates']) for b in blocks),
        new_windows=len(runs),new_continuation_calls=len(raw),new_successful_witnesses=sum(r['complete'] for r in runs),
        uniform_A_complete_windows=sum(r['A']['completed'] for r in uniform),
        uniform_B_complete_windows=sum(r['B']['completed'] for r in uniform),
        current_only_complete_windows=sum(r['current_only']['completed'] for r in uniform),
        sites_with_uniform_gain=[r['site_id'] for r in uniform if r['B']['completed']>r['A']['completed']],
        sites_with_uniform_gain_over_current_only=[r['site_id'] for r in uniform if r['B']['completed']>r['current_only']['completed']],
        matched_completion_differences=[dict(site_id=r['site_id'],candidate_id=r['candidate_id'],center_id=r['center_id'],
            center_completed=r['center']['completed'],candidate_completed=r['candidate']['completed'])
            for r in matched if r['center']['completed']!=r['candidate']['completed']],
        matched_prefix_differences=[dict(site_id=r['site_id'],candidate_id=r['candidate_id'],center_id=r['center_id'],
            center_prefix=r['center']['prefix_median'],candidate_prefix=r['candidate']['prefix_median'])
            for r in matched if r['center']['prefix_median']!=r['candidate']['prefix_median']],
        unavailable_centers=[r for r in availability if not r['available']],
        demand_optimization_statuses=dict(Counter(str(d['next_demand']['solver_status']) for d in diagnostic)),
        demand_nonlinear_verified=sum(d['next_demand']['nonlinear_verification']['verifier_accepted'] for d in diagnostic),
        diagnostic_candidates=len(diagnostic),
        new_failure_counts=dict(Counter(r['failure_kind'] for r in raw if not r['accepted'])),
        statistics='Selected sites; no population rates, no independent-repeat inference, no total pass/fail gate')
    for name,data in [('uniform_refinement_comparison',uniform),('pose_matched_comparison',matched),
                      ('projection_availability',availability),('explanatory_concordance',explanatory),
                      ('informative_pairs',pairs),('candidate_context',candidate_table),('next_frame_outcomes',first_frames)]:
        json_write(out/f'{name}.json',data);csv_write(out/f'{name}.csv',data)
    json_write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def figures(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    study.protect();out=study.out;folder=out/'final_figures';folder.mkdir(exist_ok=True)
    uniform=json.loads((out/'uniform_refinement_comparison.json').read_text())
    matched=json.loads((out/'pose_matched_comparison.json').read_text())
    concordance=json.loads((out/'explanatory_concordance.json').read_text())
    blocks=json.loads((out/'matched_candidates.json').read_text())
    source=[]
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,170/25.4),layout='constrained',
                         gridspec_kw={'height_ratios':[1.3,1]})
    labels=[r['site_id'].split('_')[1] for r in uniform]
    matrix=np.array([[r[k]['completed'] for k in ['A','current_only','B']] for r in uniform])
    ax=axs[0,0];ax.imshow(matrix,cmap='Purples',vmin=0,vmax=5,aspect='auto')
    for i,r in enumerate(uniform):
        for j,k in enumerate(['A','current_only','B']):
            n=r[k]['completed'];ax.text(j,i,f'{n}/5',ha='center',va='center',color='white' if n>=4 else '#111111')
            source.append(dict(panel='a',site_id=r['site_id'],condition=k,completed=n,repeats=5))
    ax.set(xticks=range(3),xticklabels=['Ordinary A','Current only','Uniform B'],yticks=range(9),yticklabels=labels,
           ylabel='Panda state',title='Uniform refinement: no extra completion')
    ax.tick_params(length=0)
    ax=axs[0,1];yticks=[];ylabels=[]
    for i,block in enumerate(blocks):
        u=uniform[i]
        for j,center in enumerate(block['centers']):
            y=i*2+j;yticks.append(y);ylabels.append(labels[i]+(' O' if j==0 else ' R'))
            key='A' if center['source']=='original_focal' else 'current_only'
            center_count=u[key]['completed']
            ax.scatter(center_count,y,s=19,facecolors='none',edgecolors='#707070',zorder=4)
            children=[r for r in matched if r['site_id']==u['site_id'] and r['center_source']==center['source']]
            if not children:ax.text(2.5,y,'Unavailable',ha='center',va='center',fontsize=7,color='#707070')
            for offset,r in zip(np.linspace(-.22,.22,len(children)),children):
                n=r['candidate']['completed']
                ax.plot([center_count,n],[y,y+offset],color='#C2B6D4',lw=.5,zorder=1)
                ax.scatter(n,y+offset,s=12,color='#8056A6',zorder=3)
                source.append(dict(panel='b',site_id=u['site_id'],center_id=center['candidate_id'],
                    candidate_id=r['candidate_id'],center_completed=center_count,completed=n,
                    position_match_error=r['projection']['position_match_error'],
                    orientation_match_error=r['projection']['orientation_match_error'],repeats=5))
    ax.set(yticks=yticks,yticklabels=ylabels,xticks=[0,1,2,3,4,5],xlim=(-.4,5.4),ylim=(17.7,-.7),
           xlabel='Completed suffixes / 5',title='Same achieved pose; different configuration')
    ax.grid(axis='x',alpha=.12)
    ax.legend(handles=[Line2D([],[],marker='o',mfc='none',mec='#707070',ls='',label='Center'),
                       Line2D([],[],marker='o',color='#8056A6',ls='',label='Pose-matched')],
              loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
    ax=axs[1,0]
    display={'residual':'Residual','nearest':'Nearest','sigma_min':'Scaled σmin','next_demand':'Next demand'}
    for index,predictor in enumerate(display):
        for offset,scope,color,marker in [(-.11,'all_informative','#666666','o'),(.11,'pose_matched_informative','#8056A6','s')]:
            r=next(x for x in concordance if x['scope']==scope and x['predictor']==predictor)
            ax.scatter(r['score'],index+offset,color=color,marker=marker,s=21)
            source.append(dict(panel='c',**r))
    ax.axvline(.5,color='#BBBBBB',ls=':',lw=.7)
    ax.set(yticks=range(4),yticklabels=list(display.values()),ylim=(3.6,-.6),xlim=(0,1.03),xticks=[0,.25,.5,.75,1],
           xlabel='Concordance (ties = 0.5)',title='Candidate ordering (descriptive)')
    ax.legend(handles=[Line2D([],[],marker='o',color='#666666',ls='',label='All 82 pairs'),
                       Line2D([],[],marker='s',color='#8056A6',ls='',label='Matched 25 pairs')],
              loc='upper center',bbox_to_anchor=(.5,-.19),ncol=1)
    ax=axs[1,1]
    colors={'A':'#777777','current_only':'#30638E','B':'#8056A6'}
    names={'A':'Ordinary A*','current_only':'Current only*','B':'Uniform B'}
    for i,r in enumerate(uniform):
        for offset,k in [(-.18,'A'),(0,'current_only'),(.18,'B')]:
            values=np.asarray(r[k]['cumulative_latency_ms']);assert np.all(values>0)
            median=float(np.median(values))
            ax.errorbar(median,i+offset,xerr=[[median-float(values.min())],[float(values.max())-median]],
                fmt='o',ms=2.5,color=colors[k],lw=.7,capsize=1.5,label=names[k] if i==0 else None)
            for rep,t in enumerate(values):source.append(dict(panel='d',site_id=r['site_id'],condition=k,repeat=rep,cumulative_ms=t))
    ax.set_xscale('log')
    ax.set_xticks([10,100,1000,5000],labels=['10','100','1000','5000'])
    ax.set(yticks=range(9),yticklabels=labels,ylim=(8.7,-.7),xlabel='Measured suffix time (ms; log scale)',
           title='Uniform refinement incurs added work')
    ax.grid(axis='x',alpha=.12)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    for label,ax in zip('abcd',axs.flat):
        ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    stem=folder/'residual_configuration_mechanism'
    for ext in ['pdf','svg','png']:
        if stem.with_suffix('.'+ext).exists():raise FileExistsError(stem.with_suffix('.'+ext))
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    plt.close(fig)
    csv_write(folder/'source_data.csv',source)
    json_write(folder/'figure_manifest.json',dict(width_mm=183,height_mm=170,
        caption='a: completion counts, five numerical searches per selected state. b: each point is one currently legal candidate; O/R denote original/refined centers, not robot branches; all 43 new configurations shown, unavailable centers explicit. c: dependent informative within-state pair ordering, not independent-sample predictive accuracy; residual ties respect matching precision. d: medians and full min-max ranges over five searches; *historical reused measurements, not contemporaneous timing trials. No p-values, no population claims, no native TRAC-IK FEV imputation.',
        files={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))


def verify(study):
    study.protect();out=study.out
    raw=read_records(out/'new_continuation_raw.jsonl.gz')
    runs=json.loads((out/'new_continuation_runs.json').read_text())
    byrun=defaultdict(list)
    for r in raw:byrun[r['run_id']].append(r)
    commands=0;witnesses=0;transitions=0;matching_checks=0
    from ..geometry import pose_error
    matched=json.loads((out/'matched_candidates.json').read_text())
    for block in matched:
        assert len(block['candidates'])<=6
        site=block['site'];query=study.study.query(site['uid'],site['frame'],site['previous_q'])
        for c in block['candidates']:
            center=next(x for x in block['centers'] if x['candidate_id']==c['center_id'])
            error=pose_error(study.study.kin.forward(np.asarray(center['q'])),study.study.kin.forward(np.asarray(c['q'])))
            assert np.linalg.norm(error[:3])<=MATCH_POSITION and np.linalg.norm(error[3:])<=MATCH_ORIENTATION
            assert study.study.verifier.check(np.asarray(c['q']),query).accepted
            matching_checks+=1
    for run in runs:
        initial=run['initial'];q=np.asarray(initial['q'])
        query=study.study.query(run['uid'],initial['frame'],initial['previous_q'])
        assert study.study.verifier.check(q,query).accepted
        rows=sorted(byrun[run['run_id']],key=lambda r:r['frame'])
        assert len(rows)==run['frames']
        for t,r in enumerate(rows,initial['frame']+1):
            assert r['frame']==t and np.array_equal(q,r['previous_q'])
            query=study.study.query(run['uid'],t,q)
            assert query_digest(query)==r['query_hash']
            assert np.array_equal(query.target.position,r['target_position'])
            assert np.array_equal(query.target.rotation,r['target_rotation'])
            assert r['solver_observation']['interior_bounds']
            if r['accepted']:
                assert study.study.verifier.check(np.asarray(r['q']),query).accepted
                q=np.asarray(r['q']);commands+=1
            else:assert r['q'] is None
            assert np.array_equal(q,r['accepted_state_q'])
        assert run['complete']==all(r['accepted'] for r in rows)
        if run['complete']:
            witness=json.loads((out/run['witness']).read_text())
            assert witness['frames']==rows and witness['initial']==initial
            witnesses+=1;transitions+=1+len(rows)
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=study.root,text=True).strip()
    result=dict(solver_calls=0,accepted_suffix_commands_reverified=commands,pose_matched_candidates_reverified=matching_checks,
        successful_witnesses_reverified=witnesses,successful_witness_transitions=transitions,
        unchanged_previous_state_feedback=True,unchanged_old_artifacts=True,
        new_windows=len(runs),new_calls=len(raw))
    json_write(out/'verification.json',result)
    paths=[p for p in out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    paths.extend([study.root/'src/confik/continuation_mechanism'/name for name in
                  ['disentangle.py','disentangle_math.py','disentangle_reporting.py']])
    paths.extend([study.root/'tests/test_residual_configuration.py',study.root/'docs/RESIDUAL_CONFIGURATION_FINDINGS.md'])
    json_write(out/'delivery_manifest.json',dict(files={str(p.relative_to(study.root)):digest(p) for p in sorted(paths)},
        parent_manifest_sha256=digest(study.oldout/'delivery_manifest.json')))
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyse','figures','verify']);parser.add_argument('--root',default='.')
    args=parser.parse_args();globals()[args.stage](Disentanglement(args.root))


if __name__=='__main__':main()
