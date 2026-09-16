#!/usr/bin/env python3
"""Recompute the final evidence tables/figures from saved commands, no IK calls."""
import argparse
from collections import Counter, defaultdict
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
entry=module('final_evidence','scripts/run_task_balance_final_evidence.py')
stats=module('anchor_stats','scripts/report_task_set_boundary.py')
from confik.correction_reserve.study import read_rows,write_json,sha,clean,utc
from confik.correction_reserve.reporting import csv_write
from confik.task_balance_gn import relative_stop
OUT=entry.OUT

def quant(a,qs):return np.percentile(a,qs).tolist() if len(a) else None

def summary(rows,items,methods,v,label):
    result=[];n=len(items);known={i['uid'] for i in items if i.get('witness_available',True)}
    for m in methods:
        a=[r for r in rows if r['method']==m];assert len(a)==3*n
        ok=[r for r in a if r['accepted']]
        lat=np.array([r['total_latency_ns']/1e6 for r in a])
        ep=[r['position_error'] for r in ok];er=[r['orientation_error'] for r in ok]
        # Only minimax adapters expose this count. GN/SQP exception placeholders
        # are not observations of zero work; keep their uninstrumented count null.
        qp=[r['dual_updates'] for r in a if 'dual_updates' in r] if m in ('relative','eta010','eta050','fixed_qp2','clarabel') else []
        w=[r for r in a if r['uid'] in known]
        result.append(dict(group=label,method=m,queries=n,repeats=3,
            success_by_repeat=[sum(r['accepted'] for r in a if r['repeat']==k) for k in range(3)],
            within20_by_repeat=[sum(r['accepted_within_20ms'] for r in a if r['repeat']==k) for k in range(3)],
            verified_success=float(np.mean([r['accepted'] for r in a])),
            within20_success=float(np.mean([r['accepted_within_20ms'] for r in a])),
            mean_ms=float(lat.mean()),p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),
            total_ms_per_repeat=float(lat.sum()/3),late_calls=int(sum(lat>20)),
            witness_queries=len(known),witness_success_by_repeat=[sum(r['accepted'] for r in w if r['repeat']==k) for k in range(3)],
            witness_confirmed_missed_uids=sorted({r['uid'] for r in w if not r['accepted']}),
            accepted_position_p50_p95_max_m=quant(ep,[50,95,100]),accepted_orientation_p50_p95_max_rad=quant(er,[50,95,100]),
            edge_fraction=float(np.mean([max(p/v.config.position_tolerance,r/v.config.orientation_tolerance)>.9 for p,r in zip(ep,er)])) if ok else None,
            command_step_utilization_p50_p95_max=quant([r['velocity_utilization'] for r in ok],[50,95,100]),
            mean_local_qps=float(np.mean(qp)) if qp else None,
            first_half_accepted_fraction=sum(r.get('first_half_accepted',False) for r in a)/len(a) if m in ('relative','eta010','eta050','fixed_qp2','clarabel') else None,
            no_qp_fraction=sum(r.get('dual_updates',1)==0 and r['accepted'] for r in a)/len(a) if m=='relative' else None,
            continued_after_first_fraction=sum(bool(r.get('subproblems')) for r in a)/len(a) if m in ('relative','eta010','eta050','fixed_qp2','clarabel') else None,
            extra_weighted_qp_fraction=sum(any(s['dual_updates']>1 for s in r.get('subproblems',[])) for r in a)/len(a) if m in ('relative','eta010','eta050','fixed_qp2') else None,
            relative_stop_fraction=sum(any(s['reason']=='relative_progress' for s in r.get('subproblems',[])) for r in a)/len(a) if m in ('relative','eta010','eta050') else None,
            mean_iterations=float(np.mean([r.get('iterations',0) for r in a])),
            failure_counts=dict(Counter(r['failure_kind'] for r in a if not r['accepted'])),
            native_status_counts=dict(Counter(r['internal_status'] for r in a))))
    return result

def audit(rows,items,methods,v,cfg):
    index={i['uid']:i for i in items};counts=Counter()
    assert Counter((r['uid'],r['method'],r['repeat']) for r in rows)==Counter((i['uid'],m,k) for i in items for m in methods for k in range(3))
    for r in rows:
        q=None if r['q'] is None else np.array(r['q'])
        verdict=v.check(q,entry.old.old.query_of(index[r['uid']]))
        assert bool(verdict.accepted)==r['accepted']
        assert r['accepted_within_20ms']==bool(verdict.accepted and r['total_latency_ns']<=20_000_000)
        assert r['total_latency_ns']>=r['adapter_total_latency_ns']
        if r['q'] is not None:
            np.testing.assert_allclose([verdict.position_error,verdict.orientation_error],
                [r['position_error'],r['orientation_error']],rtol=0,atol=1e-12)
        for sub in r.get('subproblems',[]):
            if sub['reason']=='relative_progress':
                p,u,l=[sub[k] for k in ('pzero','upper','lower')]
                tol=64*np.finfo(float).eps*max(1,abs(p),abs(u),abs(l))
                assert relative_stop(p,u,l,cfg['forcing'][r['method']],tol)
                counts['relative_conditions_checked']+=1
        counts['calls']+=1;counts['accepted']+=r['accepted'];counts['late']+=r['total_latency_ns']>20_000_000
    counts['accepted_contract_violations']=0
    return dict(counts)

def differences(rows,items,methods,kind):
    lookup=defaultdict(list);index={i['uid']:i for i in items};changes=[];commands=[]
    for line,r in enumerate(rows):lookup[r['uid'],r['method']].append((line+1,r))
    for uid,i in index.items():
        aa=lookup[uid,'relative'];sa=np.mean([r['accepted'] for _,r in aa])
        for m in methods:
            if m=='relative':continue
            bb=lookup[uid,m];sb=np.mean([r['accepted'] for _,r in bb])
            if sa==sb:continue
            c=dict(kind=kind,uid=uid,baseline=m,main_success=float(sa),baseline_success=float(sb),
                change='gained' if sa>sb else 'lost',unit=i.get('anchor_uid',i.get('episode_uid')))
            changes.append(c)
            commands.append(dict(c,input=i,main=[dict(line=line,**r) for line,r in aa],baseline_records=[dict(line=line,**r) for line,r in bb]))
    return changes,commands

def check_manifest(folder):
    m=json.loads((folder/'manifest.json').read_text())
    for name,h in m['files'].items():assert sha(folder/name)==h
    return m

def sensitivity(cfg,out):
    main=[];cells=[];pairs=[];units=[];changes=[];commands=[];audits={};bad=[]
    for robot in cfg['robots']:
        folder=OUT/f'sensitivity/run_{robot}';check_manifest(folder)
        full=read_rows(folder/'records.jsonl.gz')
        for c,s in cfg['sensitivity'].items():
            items=json.loads((OUT/f'sensitivity/inputs_{c}_{robot}.json').read_text())
            _,kin,v,_=entry.configured_context(robot,cfg,c)
            rows=[r for r in full if r['condition']==c]
            for i in items:assert v.check(np.array(i['q_witness']),entry.old.old.query_of(i)).accepted
            audits[robot+':'+c]=audit(rows,items,s['methods'],v,cfg)
            unit=stats.units(rows,items);units.extend(dict(u,condition=c) for u in unit)
            for label,selected in stats.groups(items,cfg):
                # Retain all nine cells plus existing geometric subdivisions.
                ids={i['uid'] for i in selected};a=[r for r in rows if r['uid'] in ids]
                table=[dict(t,robot=robot,condition=c) for t in summary(a,selected,s['methods'],v,label)]
                (main if label=='all' else cells).extend(table)
                pp,_=stats.paired(unit,selected,label,cfg,robot,s['methods'])
                pairs.extend(dict(p,condition=c) for p in pp)
                if label=='all' and c=='nominal':
                    for variant in ['eta010','eta050']:
                        uu=[dict(u,method='relative') if u['method']==variant else u for u in unit if u['method'] in (variant,'gn')]
                        pp,_=stats.paired(uu,selected,label,cfg,robot,['relative','gn'])
                        pairs.extend(dict(p,condition=c,method=variant) for p in pp)
            cc,cmd=differences(rows,items,s['methods'],robot+':'+c);changes.extend(cc);commands.extend(cmd)
        for line,r in enumerate(full):
            if not r['accepted'] or r['total_latency_ns']>20_000_000:
                bad.append(dict(robot=robot,uid=r['uid'],condition=r['condition'],method=r['method'],repeat=r['repeat'],
                    accepted=r['accepted'],reason=r['failure_kind'],latency_ns=r['total_latency_ns'],line=line+1,raw_file=str(folder.relative_to(ROOT)/'records.jsonl.gz')))
    for name,data in [('sensitivity_main',main),('sensitivity_cells',cells),('sensitivity_paired',pairs),('sensitivity_units',units),('sensitivity_changes',changes),('sensitivity_failures_timeouts',bad)]:csv_write(out/f'{name}.csv',data)
    write_json(out/'sensitivity_data.json',dict(main=main,cells=cells,pairs=pairs,changes=changes,audits=audits))
    write_json(out/'sensitivity_changed_commands.json',commands)
    return main,pairs

def replay(cfg,out):
    folder=OUT/'source_replay/run_complete';check_manifest(folder)
    items=json.loads((OUT/'source_replay/inputs_evaluation.json').read_text());rows=read_rows(folder/'records.jsonl.gz')
    _,kin,v,_=entry.context('panda',cfg);methods=cfg['droid']['methods']
    counts=audit(rows,items,methods,v,cfg)
    main=summary(rows,items,methods,v,'all_queries');episodes=[];inputs=[];subsets=[];bad=[]
    for ep in sorted({i['episode_uid'] for i in items}):
        selected=[i for i in items if i['episode_uid']==ep];a=[r for r in rows if r['episode_uid']==ep]
        episodes.extend(dict(t,episode_uid=ep,session=selected[0]['session']) for t in summary(a,selected,methods,v,ep))
        norm=np.array([i['target_normalized_from_state'] for i in selected])
        margins=np.array([i['physical_joint_margin'] for i in selected])
        command=np.array([i['source_command_step_utilization'] for i in selected])
        inputs.append(dict(episode_uid=ep,session=selected[0]['session'],queries=len(selected),witness_count=sum(i['witness_available'] for i in selected),
            target_position_utilization_p50_p95_max=quant(norm[:,0],[50,95,100]),target_orientation_utilization_p50_p95_max=quant(norm[:,1],[50,95,100]),
            both_tasks_already_legal=int(np.sum(np.max(norm,axis=1)<=1)),position_excess=int(sum(norm[:,0]>1)),orientation_excess=int(sum(norm[:,1]>1)),
            physical_joint_margin_min_p05_median_rad=quant(margins.min(axis=1),[0,5,50]),
            commanded_step_utilization_p50_p95_max=quant(command.max(axis=1),[50,95,100]),
            logged_states_outside_project_bounds=int(sum(margins.min(axis=1)<0)),
            observation_action_joint_difference_max_rad=max(np.max(abs(np.array(i['observation_q'])-i['previous_q'])) for i in selected)))
    # Descriptive all-query pooled estimates AND equal-episode estimates.
    for t in main:
        a=[e for e in episodes if e['method']==t['method']]
        t.update(episode_mean_success=float(np.mean([e['verified_success'] for e in a])),
                 episode_mean_within20=float(np.mean([e['within20_success'] for e in a])),
                 episode_mean_ms=float(np.mean([e['mean_ms'] for e in a])))
    first={r['uid']:r for r in rows if r['method']=='relative' and r['repeat']==0}
    for label,keep in [('witness_confirmed',lambda i:i['witness_available']),
                       ('logged_state_within_joint_range',lambda i:not first[i['uid']]['logged_state_outside_project_joint_range']),
                       ('empty_dynamic_box',lambda i:first[i['uid']]['empty_dynamic_box'])]:
        selected=[i for i in items if keep(i)];ids={i['uid'] for i in selected}
        if ids:subsets.extend(summary([r for r in rows if r['uid'] in ids],selected,methods,v,label))
    for line,r in enumerate(rows):
        if not r['accepted'] or r['total_latency_ns']>20_000_000:
            bad.append(dict(uid=r['uid'],episode_uid=r['episode_uid'],frame=r['source_frame'],method=r['method'],repeat=r['repeat'],
                accepted=r['accepted'],latency_ns=r['total_latency_ns'],reason=r['failure_kind'],empty_dynamic_box=r['empty_dynamic_box'],
                witness=r['witness_available'],line=line+1))
    lookup={(e['episode_uid'],e['method']):e for e in episodes}
    epids=sorted({e['episode_uid'] for e in episodes});sessions=sorted({e['session'] for e in episodes})
    session_eps={s:[e for e in epids if lookup[e,'relative']['session']==s] for s in sessions}
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot=rng.integers(0,len(sessions),(cfg['bootstrap_samples'],len(sessions)))
    ns=np.array([len(session_eps[s]) for s in sessions]);denom=ns[boot].sum(axis=1)
    pairs=[]
    for m in methods:
        if m=='relative':continue
        keys=['verified_success','within20_success','mean_ms']
        x=np.array([[sum(lookup[e,'relative'][k] for e in session_eps[s]) for k in keys] for s in sessions])
        y=np.array([[sum(lookup[e,m][k] for e in session_eps[s]) for k in keys] for s in sessions])
        bx=x[boot].sum(axis=1)/denom[:,None];by=y[boot].sum(axis=1)/denom[:,None]
        pairs.append(dict(method='relative',baseline=m,episodes=len(epids),session_clusters=len(sessions),
            success_difference=float((x[:,0]-y[:,0]).sum()/len(epids)),success_difference_ci=quant(bx[:,0]-by[:,0],[2.5,97.5]),
            timely_difference=float((x[:,1]-y[:,1]).sum()/len(epids)),timely_difference_ci=quant(bx[:,1]-by[:,1],[2.5,97.5]),
            latency_ratio=float(x[:,2].sum()/y[:,2].sum()),latency_ratio_ci=quant(bx[:,2]/by[:,2],[2.5,97.5]),
            interval='equal-episode estimate; repeat means within query, episode aggregates within lab/date session; 4000 paired session bootstrap, unadjusted95%'))
    changes,commands=differences(rows,items,methods,'DROID offline')
    for name,data in [('source_main',main),('source_episodes',episodes),('source_inputs',inputs),('source_subsets',subsets),('source_paired',pairs),('source_changes',changes),('source_failures_timeouts',bad)]:csv_write(out/f'{name}.csv',data)
    write_json(out/'source_data.json',dict(main=main,episodes=episodes,inputs=inputs,subsets=subsets,pairs=pairs,changes=changes,audit=counts))
    write_json(out/'source_changed_commands.json',commands)
    return main,episodes,pairs

def figures(out):
    # Quantitative grid: coverage and cost can change without universal benefit.
    # Source: all 270 queries/robot, 30 anchors; CIs only paired comparisons.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    data=json.loads((out/'sensitivity_data.json').read_text())
    settings=[('nominal','eta010'),('nominal','relative'),('nominal','eta050'),('position_half','relative'),('orientation_half','relative')]
    labels=['η=.10','η=.25','η=.50','Position /2','Rotation /2']
    fig,axes=plt.subplots(2,2,figsize=(7.204724,4.409449),layout='constrained')
    for col,robot in enumerate(['panda','ur5e']):
        selected=[next(t for t in data['main'] if t['robot']==robot and t['condition']==c and t['method']==m) for c,m in settings]
        axes[0,col].plot(range(5),[270*t['verified_success'] for t in selected],'o-',color='#32648a',label='Task Balance')
        for m,color,marker in [('gn','#6c7177','s'),('direct_sqp','#b18758','^')]:
            a=[next(t for t in data['main'] if t['robot']==robot and t['condition']==c and t['method']==m) for c,_ in settings]
            axes[0,col].plot(range(5),[270*t['verified_success'] for t in a],marker+'--',color=color,label='GN' if m=='gn' else 'SQP')
        axes[0,col].set(title=robot.upper() if robot=='ur5e' else 'Panda',ylabel='Verified queries / 270',xticks=range(5),xticklabels=labels)
        axes[0,col].legend(fontsize=7)
        for j,(c,m) in enumerate(settings):
            p=next(p for p in data['pairs'] if p['robot']==robot and p['condition']==c and p['group']=='all' and p['baseline']=='gn' and p['method']==m)
            ratio=p['latency_ratio'];ci=p['latency_ratio_ci']
            axes[1,col].errorbar(j,ratio,yerr=[[ratio-ci[0]],[ci[1]-ratio]],fmt='o',capsize=3,color='#32648a')
        axes[1,col].axhline(1,color='#9e9e9e',linestyle=':',linewidth=.8)
        axes[1,col].set(ylabel='Mean time / GN',xticks=range(5),xticklabels=labels)
    for ax,letter in zip(axes.flat,'abcd'):ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight='bold')
    fig.savefig(out/'sensitivity.svg');fig.savefig(out/'sensitivity.pdf');fig.savefig(out/'sensitivity.png',dpi=600)
    plt.close(fig)
    data=json.loads((out/'source_data.json').read_text());pairs=data['pairs']
    fig,axes=plt.subplots(1,2,figsize=(7.204724,2.755906),layout='constrained')
    for j,p in enumerate(pairs):
        for ax,key,factor in [(axes[0],'success_difference',100),(axes[1],'latency_ratio',1)]:
            value=p[key]*factor;ci=np.array(p[key+'_ci'])*factor
            ax.errorbar(j,value,yerr=[[value-ci[0]],[ci[1]-value]],fmt='o',capsize=3,color='#32648a')
    for ax in axes:ax.set(xticks=range(len(pairs)),xticklabels=[p['baseline'].replace('direct_sqp','SQP').replace('fixed_qp2','Fixed 2').replace('clarabel','Clarabel').replace('gn','GN') for p in pairs])
    axes[0].axhline(0,color='#9e9e9e',linestyle=':',linewidth=.8);axes[0].set_ylabel('Equal-episode coverage difference (pp)')
    axes[1].axhline(1,color='#9e9e9e',linestyle=':',linewidth=.8);axes[1].set_ylabel('Equal-episode mean time ratio')
    for ax,letter in zip(axes,'ab'):ax.text(-.12,1.04,letter,transform=ax.transAxes,fontweight='bold')
    fig.savefig(out/'source_replay.svg');fig.savefig(out/'source_replay.pdf');fig.savefig(out/'source_replay.png',dpi=600)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'reports');p.add_argument('--figures-only',action='store_true')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if args.figures_only:figures(out);return
    cfg=yaml.safe_load(entry.CONFIG.read_text());entry.check()
    sensitivity(cfg,out);replay(cfg,out);figures(out)
    write_json(out/'manifest.json',dict(utc=utc(),solver_calls_in_reporting=0,frozen_after=entry.frozen(),
        input_manifests={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('manifest.json') if 'reports' not in p.parts},
        report_script_sha=sha(Path(__file__)),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('Saved final evidence tables and figures:',out)

if __name__=='__main__':main()
