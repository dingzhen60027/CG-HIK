#!/usr/bin/env python3
"""Read-only numerical replay and source-backed report; never runs an IK solver."""
import argparse
from collections import Counter,defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
entry=module('comparison_run','scripts/run_task_balance_comparison.py')
stats=module('frozen_anchor_statistics','scripts/report_task_set_boundary.py')
from confik.correction_reserve.study import sha,write_json,utc,clean,context,read_rows
from confik.correction_reserve.reporting import csv_write,paired_intervals,group_table,LABELS
OUT=entry.OUT
LABEL={'relative':'Progress','gn':'Original GN','tight':'Tight','clarabel':'Clarabel',
       'range_loss_matched':'Range loss','direct_sqp':'Direct SQP','fixed_qp1':'Fixed 1','fixed_qp2':'Fixed 2'}

def check_manifest(folder):
    m=json.loads((folder/'manifest.json').read_text())
    for name,h in m['files'].items():assert sha(folder/name)==h
    return m

def audit_points(items,rows,cfg,robot):
    _,kin,v,_=context(robot,cfg);index={i['uid']:i for i in items};counts=Counter()
    assert Counter((r['uid'],r['method'],r['repeat']) for r in rows)==Counter(
        (i['uid'],m,k) for i in items for m in cfg['methods'] for k in range(3))
    for i in items:
        assert v.check(np.array(i['q_witness']),entry.old.query_of(i)).accepted
    for r in rows:
        vv=v.check(np.array(r['q']),entry.old.query_of(index[r['uid']]))
        assert bool(vv.accepted)==r['accepted']
        assert r['accepted_within_20ms']==bool(vv.accepted and r['total_latency_ns']<=20_000_000)
        np.testing.assert_allclose([vv.position_error,vv.orientation_error],
                                  [r['position_error'],r['orientation_error']],atol=1e-12,rtol=0)
        assert r['total_latency_ns']>=r['adapter_total_latency_ns']
        for info in r.get('subproblems',[]):
            if info['reason']=='relative_progress':
                p,u,l=[info[k] for k in ('pzero','upper','lower')]
                tol=64*np.finfo(float).eps*max(1,abs(p),abs(u),abs(l))
                assert stats.relative_stop(p,u,l,.25,tol)
            if r['method'].startswith('fixed_qp'):
                assert info['dual_updates']<=int(r['method'][-1]) and info['lower'] is None
        counts['calls']+=1;counts['accepted']+=r['accepted'];counts['late']+=r['total_latency_ns']>20_000_000
    counts['witnesses']=len(items)
    return dict(counts)

def point_results(cfg,out):
    main=[];cells=[];pairs=[];changes=[];units=[];stop=[];audits={};commands=[];prior=[];sources={};development=[]
    work_units=[];failure_index=[];matrices=[]
    for robot in cfg['robots']:
        folder=OUT/f'points/points_{robot}';check_manifest(folder);sources[str(folder.relative_to(ROOT))]=sha(folder/'manifest.json')
        items=json.loads((entry.old.OUT/f'inputs/validation_{robot}.json').read_text())
        rows=read_rows(folder/'records.jsonl.gz');audits[robot]=audit_points(items,rows,cfg,robot)
        unit=stats.units(rows,items);units.extend(unit)
        for label,selected in stats.groups(items,cfg):
            ids={i['uid'] for i in selected};subset=[r for r in rows if r['uid'] in ids]
            table=stats.summarize(subset,selected,cfg,robot,label,cfg['methods'])
            (main if label=='all' else cells).extend(table)
            pp,cc=stats.paired(unit,selected,label,cfg,robot,cfg['methods']);pairs.extend(pp);changes.extend(cc)
            if label in ('all','multijoint_boundary/alpha=0.95'):
                for t in table:
                    if t['method'] in ('relative','fixed_qp1','fixed_qp2','tight','clarabel'):
                        a=[r for r in subset if r['method']==t['method']]
                        theta=[i['theta'] for r in a for i in r.get('subproblems',[])]
                        stop.append(dict(t,theta_min=min(theta,default=None),theta_max=max(theta,default=None),
                            mean_box_active_updates=stats.mean_field(a,'box_qp_updates'),
                            mean_backtracking=stats.mean_field(a,'backtracking_evaluations')))
        lookup={(r['uid'],r['method'],r['repeat']):(j+1,r) for j,r in enumerate(rows)}
        input_index={i['uid']:i for i in items}
        for m in cfg['methods']:
            a=[r for r in rows if r['method']==m]
            for native in (False,True,None):
                for accepted in (False,True):
                    matrices.append(dict(robot=robot,method=m,internal_ok=native,task_accepted=accepted,
                        calls=sum(r.get('internal_ok')==native and r['accepted']==accepted for r in a)))
        for line,r in enumerate(rows):
            if not r['accepted'] or r['total_latency_ns']>20_000_000:
                failure_index.append(dict(robot=robot,uid=r['uid'],method=r['method'],repeat=r['repeat'],
                    accepted=r['accepted'],latency_ns=r['total_latency_ns'],reason=r['failure_kind'],
                    raw_file=str((folder/'records.jsonl.gz').relative_to(ROOT)),line=line+1))
        for i in items:
            for m in ('relative','fixed_qp1','fixed_qp2','tight','clarabel'):
                a=[lookup[i['uid'],m,k][1] for k in range(3)]
                infos=[sub for r in a for sub in r.get('subproblems',[])]
                work_units.append(dict(robot=robot,uid=i['uid'],method=m,anchor_uid=i['anchor_uid'],
                    displacement=i['displacement'],alpha=i['alpha'],accepted=np.mean([r['accepted'] for r in a]),
                    mean_ms=np.mean([r['total_latency_ns'] for r in a])/1e6,
                    mean_total_local_qp_calls=np.mean([r.get('dual_updates',0) for r in a]),
                    max_qps_one_local_model=max([s['dual_updates'] for s in infos],default=1),
                    exit_reasons=dict(Counter(s['reason'] for s in infos))))
        for c in [c for c in changes if c['robot']==robot]:
            i=input_index[c['uid']]
            for k in range(3):
                la,a=lookup[i['uid'],'relative',k];lb,b=lookup[i['uid'],c['baseline'],k]
                commands.append(dict(robot=robot,uid=i['uid'],baseline=c['baseline'],repeat=k,
                    change='gained' if c['relative_success']>c['baseline_success'] else 'lost',
                    previous_q=i['previous_q'],target_position=i['target_position'],target_rotation=i['target_rotation'],dt=i['dt'],
                    relative_q=a['q'],baseline_q=b['q'],relative_accepted=a['accepted'],baseline_accepted=b['accepted'],
                    relative_reasons=a['verification_reasons'],baseline_reasons=b['verification_reasons'],
                    raw_file=str((folder/'records.jsonl.gz').relative_to(ROOT)),relative_line=la,baseline_line=lb))
        oldrows=read_rows(entry.old.OUT/f'validation_{robot}/records.jsonl.gz')
        oldmiss={r['uid'] for r in oldrows if r['method']=='gn' and not r['accepted']}
        assert len(oldmiss)==(10 if robot=='panda' else 12)
        for u in sorted(oldmiss):
            for m in cfg['methods']:
                a=[lookup[u,m,k][1] for k in range(3)]
                prior.append(dict(robot=robot,uid=u,method=m,accepted_by_repeat=[r['accepted'] for r in a],
                                  mean_accepted=float(np.mean([r['accepted'] for r in a]))))
        for phase in ('development','development_sqp'):
            f=OUT/f'points/{phase}_{robot}';check_manifest(f);dev=read_rows(f/'records.jsonl.gz')
            di=json.loads((entry.old.OUT/f'inputs/development_{robot}.json').read_text())
            mm=sorted({r['method'] for r in dev})
            development.extend(stats.summarize(dev,di,cfg,robot,phase,mm))
    for name,data in [('points_main',main),('points_cells',cells),('paired_effects',pairs),('query_units',units),
                      ('gained_lost_queries',changes),('stopping_comparison',stop),('command_difference_index',commands),
                      ('prior_22_misses',prior),('development_comparison',development)]:
        csv_write(out/f'{name}.csv',data)
    csv_write(out/'stopping_queries.csv',work_units)
    csv_write(out/'point_failures_timeouts.csv',failure_index)
    csv_write(out/'native_task_matrix.csv',matrices)
    write_json(out/'points_data.json',dict(main=main,cells=cells,pairs=pairs,changes=changes,stopping=stop,prior=prior))
    write_json(out/'point_replay.json',dict(audits=audits,sources=sources,accepted_contract_violations=0,solver_calls=0))
    return main,cells,pairs,stop

def application_results(cfg,out):
    summaries=[];arrays={};changes=[];pairs=[];command_index=[];sources={};audits=Counter();plot=[]
    LABELS.update(LABEL)
    for robot in cfg['robots']:
        folder=OUT/f'application/application_{robot}';check_manifest(folder)
        sources[str(folder.relative_to(ROOT))]=sha(folder/'manifest.json')
        items=json.loads((OUT/f'application/inputs_{robot}.json').read_text());index={i['uid']:i for i in items}
        _,kin,v,_=context(robot,cfg);rows=read_rows(folder/'records.jsonl.gz');grouped=defaultdict(list)
        for line,r in enumerate(rows):grouped[r['uid'],r['method'],r['repeat']].append((line+1,r))
        assert set(grouped)=={(i['uid'],m,k) for i in items for m in cfg['application_methods'] for k in range(3)}
        sr=json.loads((folder/'summaries.json').read_text());summaries.extend(sr)
        for s in sr:
            a=grouped[s['uid'],s['method'],s['repeat']];item=index[s['uid']]
            assert len(a)==150
            previous=np.array(item['initial_q'])
            for frame,(line,r) in enumerate(a):
                assert r['frame']==frame
                np.testing.assert_array_equal(previous,r['previous_q'])
                np.testing.assert_array_equal(r['target_position'],item['target_position'][frame])
                np.testing.assert_array_equal(r['target_rotation'],item['target_rotation'][frame])
                vv=v.check(np.array(r['q']),entry.old.query_of(r));assert bool(vv.accepted)==r['accepted']
                if vv.accepted:previous=np.array(r['q'])
                np.testing.assert_array_equal(previous,r['accepted_state_q'])
                assert r['accepted_within_20ms']==bool(vv.accepted and r['total_latency_ns']<=20_000_000)
                audits['calls']+=1;audits['accepted']+=vv.accepted;audits['late']+=r['total_latency_ns']>20_000_000
                if not r['accepted'] or r['total_latency_ns']>20_000_000:
                    command_index.append(dict(robot=robot,uid=s['uid'],method=s['method'],repeat=s['repeat'],frame=frame,
                        accepted=r['accepted'],latency_ns=r['total_latency_ns'],reasons=r['failure_kind'],
                        raw_file=str((folder/'records.jsonl.gz').relative_to(ROOT)),line=line))
                if s['uid']==items[8]['uid'] and s['repeat']==0:
                    # Predefined largest-amplitude, zero-angle scan, not selected by failure.
                    pose=kin.forward(np.array(r['accepted_state_q']));p=np.array(pose.position)-item['scan_origin']
                    target=np.array(r['target_position'])-item['scan_origin']
                    plot.append(dict(robot=robot,method=s['method'],frame=frame,uid=s['uid'],
                        x_mm=1000*float(p@item['scan_u']),y_mm=1000*float(p@item['scan_v']),
                        target_x_mm=1000*float(target@item['scan_u']),target_y_mm=1000*float(target@item['scan_v']),
                        position_utilization=r['position_error']/v.config.position_tolerance,
                        orientation_utilization=r['orientation_error']/v.config.orientation_tolerance,
                        accepted=r['accepted']))
            values=[r for _,r in a]
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in values]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in values if r['accepted']]).reshape(-1,2))
        uids=sorted(index);units={}
        metrics=('completion','deadline_completion','frame_success','total_latency_ns')
        for m in cfg['application_methods']:
            for u in uids:
                rr=[s for s in sr if s['method']==m and s['uid']==u]
                units[m,u]={k:float(np.mean([s[k] for s in rr])) for k in metrics}
        for m in cfg['application_methods']:
            if m=='relative':continue
            for metric in metrics:
                a=np.array([units['relative',u][metric] for u in uids]);b=np.array([units[m,u][metric] for u in uids])
                pairs.append(dict(robot=robot,baseline=m,metric=metric,independent_units=12,
                    **paired_intervals(a,b,['scan']*12,cfg['bootstrap_seed'],4000)))
            for u in uids:
                a=units['relative',u]['completion'];b=units[m,u]['completion']
                changes.append(dict(robot=robot,uid=u,baseline=m,relative_completion=a,baseline_completion=b,
                    change='gained' if a>b else 'lost' if a<b else 'same'))
    main=group_table(summaries,arrays)
    for row in main:
        rr=[s for s in summaries if s['robot']==row['robot'] and s['method']==row['method']]
        er=np.concatenate([arrays[s['run_id']]['errors'] for s in rr])
        row['edge_fraction']=float(np.mean(np.maximum(er[:,0]/.001,er[:,1]/np.deg2rad(.5))>.9))
    for name,data in [('application_main',main),('application_sequences',summaries),('application_paired_effects',pairs),
                      ('application_gained_lost',changes),('application_plot_data',plot)]:csv_write(out/f'{name}.csv',data)
    # Empty failures are meaningful; keep JSON even if no CSV rows exist.
    write_json(out/'application_failures_timeouts.json',command_index)
    write_json(out/'application_data.json',dict(main=main,pairs=pairs,changes=changes,plot=plot))
    write_json(out/'application_replay.json',dict(counts=dict(audits),sources=sources,solver_calls=0,
        accepted_contract_violations=0,application_scope='synthetic geometric task, no real-world effectiveness estimate'))
    return main,plot

def figures(out,main,pairs,stop,plot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42})
    colors={'relative':'#007C83','gn':'#777777','tight':'#B78300','clarabel':'#905EA8',
            'direct_sqp':'#3467B0','range_loss_matched':'#B75B35','fixed_qp1':'#999999','fixed_qp2':'#566B2F'}
    def save(fig,name):
        fig.savefig(out/f'{name}.svg')
        fig.savefig(out/f'{name}.pdf')
        fig.savefig(out/f'{name}.png',dpi=300)
        plt.close(fig)
    # Paired cluster uncertainty, not overlapping marginal call distributions.
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.6),layout='constrained')
    bases=['gn','fixed_qp1','fixed_qp2','range_loss_matched','direct_sqp','tight','clarabel']
    for row,robot in enumerate(('panda','ur5e')):
        selected={p['baseline']:p for p in pairs if p['robot']==robot and p['group']=='all'}
        for j,m in enumerate(bases):
            p=selected[m];v=100*p['success_difference'];ci=100*np.array(p['success_difference_ci'])
            axes[row,0].errorbar(v,j,xerr=[[v-ci[0]],[ci[1]-v]],fmt='o',color=colors[m],capsize=2)
            v=p['latency_ratio'];ci=p['latency_ratio_ci']
            axes[row,1].errorbar(v,j,xerr=[[v-ci[0]],[ci[1]-v]],fmt='o',color=colors[m],capsize=2)
        for col,ax in enumerate(axes[row]):
            ax.set_yticks(range(len(bases)),[LABEL[m] for m in bases]);ax.invert_yaxis()
            ax.axvline(0 if col==0 else 1,color='#BBBBBB',lw=.8,ls='--')
            ax.set_title(f'{"abcd"[row*2+col]}  {robot.upper()} — all 2160 queries',loc='left')
            ax.set_xlabel('Progress − baseline coverage (pp)' if col==0 else 'Progress / baseline mean time')
            ax.spines[['top','right']].set_visible(False)
    save(fig,'coverage_time')
    # Same three methods and metrics in every panel. Each dot is one full sample
    # mean, repeat nesting handled in tables; descriptive work counts, not inference.
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),layout='constrained')
    methods=['fixed_qp1','fixed_qp2','relative']
    for row,robot in enumerate(('panda','ur5e')):
        for col,group in enumerate(('all','multijoint_boundary/alpha=0.95')):
            a={s['method']:s for s in stop if s['robot']==robot and s['group']==group}
            ax=axes[row,col]
            for j,m in enumerate(methods):
                t=a[m];ax.scatter(t['mean_dual_updates'],t['mean_ms'],color=colors[m],s=35,marker=['s','^','o'][j])
                # Labels in fixed figure gutter, avoiding point/interval masking.
            ax.set_title(f'{"abcd"[row*2+col]}  {robot.upper()} — '+('all' if col==0 else 'boundary / offset 0.95'),loc='left')
            ax.set_xlabel('Mean local QP calls / query');ax.set_ylabel('Mean complete call (ms)')
            ax.spines[['top','right']].set_visible(False)
            ax.legend([f'{LABEL[m]}: {a[m]["mean_success_count"]:.0f}/{a[m]["queries"]}' for m in methods],loc='best',frameon=False)
    save(fig,'stopping_work')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
    methods=['relative','gn','direct_sqp','range_loss_matched','fixed_qp1','fixed_qp2']
    for row,robot in enumerate(('panda','ur5e')):
        target=sorted([r for r in plot if r['robot']==robot and r['method']=='relative'],key=lambda r:r['frame'])
        axes[row,0].plot([r['target_x_mm'] for r in target],[r['target_y_mm'] for r in target],color='black',ls=':',lw=1,label='Target')
        for j,m in enumerate(methods):
            a=sorted([r for r in plot if r['robot']==robot and r['method']==m],key=lambda r:r['frame'])
            axes[row,0].plot([r['x_mm'] for r in a],[r['y_mm'] for r in a],color=colors[m],lw=.8,ls='-' if j<3 else '--',label=LABEL[m])
            axes[row,1].plot([r['frame'] for r in a],[max(r['position_utilization'],r['orientation_utilization']) for r in a],color=colors[m],lw=.8,ls='-' if j<3 else '--')
        axes[row,0].set_aspect('equal',adjustable='datalim');axes[row,0].set_xlabel('Scan-plane u (mm)');axes[row,0].set_ylabel('Scan-plane v (mm)')
        axes[row,1].axhline(1,color='black',ls=':',lw=.8);axes[row,1].set_xlabel('Frame');axes[row,1].set_ylabel('Worst pose tolerance utilization')
        for col,ax in enumerate(axes[row]):
            ax.set_title(f'{"abcd"[row*2+col]}  {robot.upper()} — scan 08, repeat 0',loc='left');ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    save(fig,'application_actual')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT/'reports'));p.add_argument('--figures-only',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(entry.CONFIG.read_text());out=Path(a.out)
    if a.figures_only:
        pts=json.loads((out/'points_data.json').read_text());app=json.loads((out/'application_data.json').read_text())
        figures(out,pts['main'],pts['pairs'],pts['stopping'],app['plot']);return
    # Directory is provisioned, but tables are exclusive-created and never overwritten.
    out.mkdir(parents=True,exist_ok=True);entry.verify_frozen()
    main,cells,pairs,stop=point_results(cfg,out);app,plot=application_results(cfg,out)
    figures(out,main,pairs,stop,plot)
    write_json(out/'frozen_after.json',dict(frozen_hashes=entry.old.frozen_hashes(),old_study_hashes=entry.old.study_hashes(),
        unchanged=True,main_source_diff='none',verifier_diff='none'))
    write_json(out/'manifest.json',dict(utc=utc(),report_script_sha256=sha(__file__),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        inference='query-repeat means, anchor cluster and geometric-family stratification; application whole UID; 4000 paired bootstrap; unadjusted descriptive 95% intervals',
        total_new_main_point_calls=103680,total_application_calls=64800,solver_calls_in_reporting=0,
        timing='all complete call stages; failed and late calls retained; no historical latency pooling'))
    for r in main: print(r['robot'],r['method'],r['success_by_repeat'],round(r['mean_ms'],4),flush=True)
    for r in app: print('APPLICATION',r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],flush=True)

if __name__=='__main__':main()
