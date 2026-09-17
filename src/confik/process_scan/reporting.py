"""Read-only Phase-1 reduction; scene repeats are never independent samples."""
import csv
import gzip
import json
import shutil
import importlib.metadata
import sys
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import h5py
from ..experiments.statistics import paired_cluster_bootstrap_difference
from .study import OUT,ROOT,write,sha,hashes
from .task import ScanTask
from .models import context,physical_model
from .planning import spline_basis,ray_center_function

METHODS=('B0','B1','B2','B3','G')


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def avg(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def quantile(rows,key,p):
    v=[r[key] for r in rows if r.get(key) is not None]
    return float(np.quantile(v,p)) if v else None


def actual_state_derivatives(raw):
    """Postprocess immutable feedback; no IK, optimization or dynamics rerun."""
    from .collision import Clearance
    cache={};extra=[]
    for r in raw:
        if not r.get('physical_run'):continue
        root=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        if r['slot'] not in cache:
            task=ScanTask(r['family'],r['placement'],r['direction'],r['robot'])
            model,_,_,_=physical_model(r['robot'],task);cache[r['slot']]=(task,Clearance(model))
        task,clearance=cache[r['slot']]
        with h5py.File(root/'execution.h5') as f:h={k:f[k][:] for k in ('t','s','q','qr','tcp','rotation')}
        rays=np.load(root/'scan_samples.npz');idx=np.minimum(np.searchsorted(h['t'],rays['t']),len(h['t'])-1)
        assert np.max(abs(h['t'][idx]-rays['t']))<1e-9
        values=np.array([[v[k] for k in ('standoff_m','center_error_m','incidence_rad','line_error_rad')]
            for v in (task.process_values(h['tcp'][i],h['rotation'][i].reshape(3,3),h['s'][i]) for i in idx)])
        d=np.array([clearance.numeric(q) for q in h['q']]);tracking=np.max(abs(h['q']-h['qr']),axis=1)
        derived=dict(slot=r['slot'],method=r['method'],repeat=r['repeat'],
            standoff_error_p95_m=float(np.quantile(abs(values[:,0]-.15),.95)),
            standoff_error_max_m=float(np.max(abs(values[:,0]-.15))),
            incidence_p95_deg=float(np.rad2deg(np.quantile(values[:,2],.95))),
            incidence_max_deg=float(np.rad2deg(np.max(values[:,2]))),
            center_ray_path_error_max_m=float(np.max(values[:,1])),
            line_orientation_error_max_deg=float(np.rad2deg(np.max(values[:,3]))),
            valid_profile_count=int(np.any(rays['quality_valid'],axis=1).sum()),profile_count=len(rays['t']),
            tracking_joint_error_p95_rad=float(np.quantile(tracking,.95)),
            actual_clearance_sampled_5ms_min_m=float(np.min(d)),
            actual_feedback_source='execution.h5 and actual scan timestamps; no new solve or simulation')
        r.update(derived);extra.append(derived)
        dest=OUT/'reports/process_traces'/f"{r['slot']}_{r['method']}_r{r['repeat']}.npz"
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,t=rays['t'],process=values,actual_history_t=h['t'],clearance=d,tracking_error=tracking)
    return extra


def scene_rows(raw):
    rows=[]
    for slot in sorted(set(r['slot'] for r in raw)):
        for method in METHODS:
            group=[r for r in raw if r['slot']==slot and r['method']==method]
            first=next(r for r in group if r['repeat']==0)
            row={k:first[k] for k in ('scene_uid','slot','cluster','robot','family','placement','direction','method')}
            row.update(repeats=len(group),implementation_ready=all(r['implementation_ready'] for r in group),
                       statuses=[r['status'] for r in group],
                       planner_feasible_repeat_mean=float(np.mean([r['planner_feasible'] for r in group])),
                       physics_executed=bool(first.get('physical_run')),quality_completed=first.get('quality_completed'),
                       execution_completed=first.get('execution_completed'),
                       initialization_success_repeat_mean=avg(group,'initialization_success'))
            for key in ('T_init','T_smooth','T_opt','T_retime','T_validate','total_plan_wall_s','predicted_execution_s',
                        'initialization_residual_evaluations','initialization_nodes','tb_dual_continuation_calls',
                        'smoothing_displacement_rms_rad','initialization_node_process_legal'):
                row[key]=avg(group,key)
            for key in ('simulated_completion_time_s','elapsed_until_stop_s','valid_coverage_fraction','raw_coverage_fraction',
                        'max_hole_diameter_upper_bound_m','max_along_scan_gap_m','velocity_utilization_max',
                        'acceleration_utilization_max','tracking_error_max_rad','torque_saturation_fraction',
                        'scan_duration_s','transition_duration_s','settling_duration_s','acceleration_rms_rad_s2',
                        'profile_process_violation_duration_s','standoff_error_p95_m','standoff_error_max_m',
                        'incidence_p95_deg','incidence_max_deg','center_ray_path_error_max_m','line_orientation_error_max_deg',
                        'valid_profile_count','tracking_joint_error_p95_rad','actual_clearance_sampled_5ms_min_m'):
                row[key]=first.get(key)
            # Quality-qualified duration is distinct from merely reaching the end.
            row['quality_execution_s']=row['simulated_completion_time_s'] if row['quality_completed'] else None
            for N in (1,10,100):
                row[f'amortized_per_job_N{N}_s']=(row['total_plan_wall_s']/N+row['quality_execution_s']) if row['quality_execution_s'] is not None else None
                row[f'total_batch_N{N}_s']=N*row[f'amortized_per_job_N{N}_s'] if row[f'amortized_per_job_N{N}_s'] is not None else None
            rows.append(row)
    return rows


def summary(rows):
    ready=any(r['implementation_ready'] for r in rows)
    return dict(scheduled_scenes=len(rows),implementation_ready=ready,
        timed_scenes=sum(r['total_plan_wall_s'] is not None for r in rows),
        reference_missing_scenes=sum('common_reference_unavailable' in r['statuses'] for r in rows),
        shared_b1_missing_scenes=sum('common_b1_initialization_missing' in r['statuses'] for r in rows),
        planner_feasible_scene_equivalents=sum(r['planner_feasible_repeat_mean'] for r in rows) if ready else None,
        initialization_attempted_scenes=sum(r['initialization_success_repeat_mean'] is not None for r in rows),
        initialization_success_scene_equivalents=sum(r['initialization_success_repeat_mean'] or 0 for r in rows) if ready else None,
        physics_executed_scenes=sum(r['physics_executed'] for r in rows),
        execution_completed_scenes=sum(r['execution_completed'] is True for r in rows) if ready else None,
        quality_completed_scenes=sum(r['quality_completed'] is True for r in rows) if ready else None,
        quality_rate_all_scheduled=float(np.mean([r['quality_completed'] is True for r in rows])) if ready else None,
        quality_execution_s_mean=avg(rows,'quality_execution_s'),quality_execution_s_median=quantile(rows,'quality_execution_s',.5),
        planning_s_mean=avg(rows,'total_plan_wall_s'),planning_s_median=quantile(rows,'total_plan_wall_s',.5),
        planning_s_p95=quantile(rows,'total_plan_wall_s',.95),
        predicted_execution_s_mean=avg(rows,'predicted_execution_s'),
        initial_s_mean=avg(rows,'T_init'),smoothing_s_mean=avg(rows,'T_smooth'),optimization_s_mean=avg(rows,'T_opt'),
        retime_s_mean=avg(rows,'T_retime'),validation_s_mean=avg(rows,'T_validate'),
        valid_coverage_mean_executed=avg(rows,'valid_coverage_fraction'),
        acceleration_max_mean_executed=avg(rows,'acceleration_utilization_max'))


def paired(rows):
    out=[]
    for robot in ('panda','ur5e'):
        for base,method in (('G','B1'),('B0','B1'),('B1','B2')):
            a={r['slot']:r for r in rows if r['robot']==robot and r['method']==base}
            b={r['slot']:r for r in rows if r['robot']==robot and r['method']==method}
            for field in ('quality_completed','quality_execution_s','total_plan_wall_s','T_init','predicted_execution_s'):
                keys=sorted(set(a)&set(b))
                if field!='quality_completed':
                    keys=[k for k in keys if a[k].get(field) is not None and b[k].get(field) is not None]
                if not keys:continue
                av=np.array([float(a[k][field] is True) if field=='quality_completed' else a[k][field] for k in keys])
                bv=np.array([float(b[k][field] is True) if field=='quality_completed' else b[k][field] for k in keys])
                clusters=np.array([a[k]['cluster'] for k in keys])
                result=paired_cluster_bootstrap_difference(av,bv,clusters,samples=4000,seed=2026091607)
                if len(np.unique(clusters))<2:
                    result.update(ci_lower=None,ci_upper=None,interval_status='not_estimable_single_cluster')
                out.append(dict(robot=robot,baseline=base,method=method,metric=field,paired_scenes=len(keys),
                    interpretation='quality durations only on jointly quality-complete scenes; missing non-time outcomes count in all-scene yield, not as algorithm infeasibility',**result))
                if field not in ('quality_completed','T_init'):
                    unique=np.unique(clusters);rng=np.random.default_rng(2026091607);ratios=[]
                    # Mean within cluster first; preserve both directions together.
                    ac=np.array([np.mean(av[clusters==c]) for c in unique]);bc=np.array([np.mean(bv[clusters==c]) for c in unique])
                    ids=rng.integers(0,len(unique),size=(4000,len(unique)))
                    ratios=np.mean(bc[ids],axis=1)/np.mean(ac[ids],axis=1)
                    out.append(dict(robot=robot,baseline=base,method=method,metric=field+'_ratio',paired_scenes=len(keys),
                                    mean_ratio=float(np.mean(bc)/np.mean(ac)),ci_lower=float(np.quantile(ratios,.025)),
                                    ci_upper=float(np.quantile(ratios,.975)),cluster_count=len(unique)))
                    if len(unique)<2:out[-1].update(ci_lower=None,ci_upper=None,interval_status='not_estimable_single_cluster')
    return out


def bridge(rows):
    out=[]
    for a in [r for r in rows if r['method']=='B1']:
        b=next(r for r in rows if r['slot']==a['slot'] and r['method']=='G')
        row=dict(slot=a['slot'],scene_uid=a['scene_uid'],robot=a['robot'],family=a['family'])
        for prefix,r in (('tb',a),('gn',b)):
            for field in ('statuses','T_init','initialization_success_repeat_mean','initialization_nodes',
                          'initialization_node_process_legal','initialization_residual_evaluations','tb_dual_continuation_calls',
                          'smoothing_displacement_rms_rad','planner_feasible_repeat_mean','quality_completed',
                          'quality_execution_s','T_retime','total_plan_wall_s'):
                row[prefix+'_'+field]=r.get(field)
        left=OUT/'runs'/a['slot']/'B1_r0/path_initialization.npz';right=left.parent.parent/'G_r0/path_initialization.npz'
        if left.exists() and right.exists():
            l=np.load(left);r=np.load(right)
            if l['q'].shape==r['q'].shape:row['initial_q_difference_max_rad']=float(np.max(abs(l['q']-r['q'])))
        left=left.parent/'planned_path.npz';right=right.parent/'planned_path.npz'
        if left.exists() and right.exists():
            row['final_control_difference_max_rad']=float(np.max(abs(np.load(left)['coeff']-np.load(right)['coeff'])))
        out.append(row)
    return out


def bottlenecks(rows):
    out=[];segments=[]
    for r in rows:
        path=OUT/'runs'/r['slot']/f"{r['method']}_r0/planned_path.npz"
        if not path.exists():continue
        task=ScanTask(r['family'],r['placement'],r['direction'],r['robot']);adapter,_=context(r['robot']);basis=spline_basis(task)
        data=np.load(path);s=(data['s'][1:]+data['s'][:-1])/2;x=(data['x'][1:]+data['x'][:-1])/2;dt=np.diff(data['t'])
        q=basis(s)@data['coeff'];qs=basis(s,nu=1)@data['coeff'];qss=basis(s,nu=2)@data['coeff']
        vr=np.max(abs(qs)*np.sqrt(x)[:,None]/(.5*adapter.public.limits.velocity),axis=1)
        ar=np.max(abs(qss*x[:,None]+qs*data['u'][:,None]),axis=1)/2
        ray=ray_center_function(task,adapter);ray_speed=np.array([np.linalg.norm(np.array(ray(a)[1])@b)*np.sqrt(c) for a,b,c in zip(q,qs,x)])
        tags=task.references(s)[4];sr=ray_speed/.198
        trace=OUT/'reports/planned_derivatives'/f"{r['slot']}_{r['method']}_r0.npz"
        trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,s=s,interval_time_s=dt,speed=np.sqrt(x),path_acceleration=data['u'],
            q=q,q_s=qs,q_ss=qss,q_dot=qs*np.sqrt(x)[:,None],
            q_ddot=qss*x[:,None]+qs*data['u'][:,None],scan=tags,center_ray_speed=ray_speed,
            joint_velocity_limits=.5*adapter.public.limits.velocity,joint_acceleration_limits=np.full(q.shape[1],2.))
        duration=float(np.sum(dt));sample_lower=task.rows*max(0,task.long-2*.001/np.cos(np.deg2rad(10)))/.2
        item=dict(slot=r['slot'],robot=r['robot'],family=r['family'],method=r['method'],
            predicted_execution_s=duration,quality_completed=r['quality_completed'],
            sampling_proxy_lower_bound_s=sample_lower,
            lower_bound_assumption='Continuous center-ray speed cap 0.2m/s on straight raster chords, endpoint tube allowance 1mm/cos10deg; ignores transitions and acceleration.',
            scan_time_s=float(np.sum(dt[tags])),turn_time_s=float(np.sum(dt[~tags])),
            time_fraction_near_sampling_cap=float(np.sum(dt[(sr>=.95)&tags])/duration),
            time_fraction_near_joint_speed=float(np.sum(dt[vr>=.95])/duration),
            time_fraction_near_acceleration=float(np.sum(dt[ar>=.95])/duration),
            classification='Nonexclusive occupancy of local constraints; not a global time derivative',
            actual_acceleration_max=r.get('acceleration_utilization_max'),
            actual_tracking_max_rad=r.get('tracking_error_max_rad'),
            actual_profile_gap_m=r.get('max_along_scan_gap_m'))
        out.append(item)
        for k in range(len(s)):
            segments.append(dict(slot=r['slot'],method=r['method'],s=float(s[k]),dt_s=float(dt[k]),scan=bool(tags[k]),
                                 sample_utilization=float(sr[k]),speed_utilization=float(vr[k]),acceleration_utilization=float(ar[k])))
    return out,segments


def numerical_work(raw):
    """Describe actually validated incumbents, not just optimizer return codes."""
    result=[]
    for r in raw:
        if r['method'] not in ('B1','B2','G'):continue
        path=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"/'optimizer_trace.json.gz'
        row={k:r[k] for k in ('slot','robot','family','method','repeat','status')}
        if path.exists():
            with gzip.open(path,'rt') as f:a=json.load(f)
            for key in ('build_time','solver_time','elapsed','iterations','status','sample_constraint_violation','solver_feasible'):
                row['optimizer_'+key]=a.get(key)
            history=a['incumbents'];initial=next((v for v in history if v['source']=='verified_initial_path'),None)
            row['verified_improved_incumbents']=sum(v['source']!='verified_initial_path' for v in history)
            for checkpoint,v in a['checkpoints'].items():
                prefix='checkpoint_'+checkpoint
                row[prefix+'_available_s']=None if v is None else v['available_s']
                row[prefix+'_source']=None if v is None else v['source']
                row[prefix+'_objective']=None if v is None else v['objective']
                row[prefix+'_relative_to_initial']=None if v is None or initial is None else v['objective']/initial['objective']
            initial_path=path.parent/'path_initialization.npz';planned=path.parent/'planned_path.npz'
            if r['method']=='B2' and planned.exists():
                row['primary_control_change_max_rad']=float(np.max(abs(np.load(planned)['coeff']-np.load(initial_path)['coeff'])))
        result.append(row)
    return result


def failure_breakdown(rows):
    result=[]
    for r in rows:
        raw=json.loads((OUT/'runs'/r['slot']/f"{r['method']}_r0"/'metrics.json').read_text())
        reasons=[]
        if not r['physics_executed']:reasons=[raw['status']]
        else:
            for flag,condition in (
                ('physical_contact',raw['collision_steps']>0),
                ('physical_velocity_exceeded',raw['velocity_utilization_max']>1.001),
                ('physical_acceleration_exceeded',raw['acceleration_utilization_max']>1.001),
                ('coverage_below_99_percent',raw['valid_coverage_fraction']<.99),
                ('connected_hole_bound_over_2mm',raw['max_hole_diameter_upper_bound_m']>.002),
                ('profile_gap_unavailable',raw['max_along_scan_gap_m'] is None),
                ('profile_gap_over_1mm',raw['max_along_scan_gap_m'] is not None and raw['max_along_scan_gap_m']>.001)):
                if condition:reasons.append(flag)
            if not raw['execution_completed'] and not any(s.startswith('physical_') for s in reasons):
                reasons.append('other_motion_or_settling_condition_not_satisfied')
        result.append(dict(slot=r['slot'],scene_uid=r['scene_uid'],robot=r['robot'],method=r['method'],
                           quality_completed=r['quality_completed'],physics_executed=r['physics_executed'],
                           nonexclusive_reasons=reasons))
    return result


def plots(main,rows,bottleneck):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    root=OUT/'reports/figures';root.mkdir(parents=True,exist_ok=True)
    write(root/'figure_contract.json',dict(question='Do initialized paths yield physically quality-complete scans, and at what measured cost?',
        sources=['scene_results.csv','baseline_main.csv','bottlenecks.csv'],independent_unit='robot/family/placement cluster; 6 per robot',
        timing_repeat_reduction='mean within scene',uncertainty='paired cluster intervals in paired_intervals.csv; plotted points are descriptive scene values',
        negative_results='missing input, implementation incomplete and physical/quality failure retained separately',
        exports=['PNG 300dpi','editable SVG','vector PDF'],target='Phase-1 review, not a journal submission'))
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    colors={'B0':'#0072B2','B1':'#009E73','B2':'#D55E00','B3':'#777777','G':'#CC79A7'}
    for i,robot in enumerate(('panda','ur5e')):
        sub=[r for r in main if r['robot']==robot]
        for j,method in enumerate(METHODS):
            r=next(v for v in sub if v['method']==method)
            if r['quality_completed_scenes'] is not None:axes[i,0].bar(j,r['quality_completed_scenes'],color=colors[method])
            else:axes[i,0].text(j,1,'N/E',ha='center',color=colors[method])
            for col,key in ((1,'quality_execution_s'),(2,'total_plan_wall_s')):
                vals=[v[key] for v in rows if v['robot']==robot and v['method']==method and v.get(key) is not None]
                axes[i,col].scatter(np.full(len(vals),j),vals,s=13,color=colors[method],alpha=.7)
                if vals:axes[i,col].plot([j-.22,j+.22],[np.median(vals)]*2,color='black',lw=1)
        axes[i,0].set(ylim=(0,12.8),yticks=[0,3,6,9,12],ylabel=f'{robot.upper()}\nQuality-complete / 12')
        axes[i,1].set_ylabel('Quality-complete execution (s)')
        axes[i,2].set_ylabel('Full measured planning work (s)')
        for ax in axes[i]:ax.set_xticks(range(5),METHODS);ax.grid(axis='y',alpha=.15)
    for ax,title in zip(axes[0],('a  All scheduled scenes','b  Qualified scenes only','c  All timed scenes')):ax.set_title(title,loc='left')
    export_figure(fig,root/'phase1_overview')
    plt.close(fig)


def markdown_report(audit,main,rows,bridges,bottleneck,work,comparisons):
    def f(x):return 'N/E' if x is None else f'{x:.3f}'
    lines=['# Process-scan Phase 1：开发基线与物理执行结果','',
        '本轮仅复用旧局部数学并实施共同扫描评价，没有提出新路径优化方法，也没有运行正式测试。',
        '所有数字来自同根目录的原始记录；不是历史在线IK数据或标定尝试的拼接。','',
        '## 1. 完成范围','',
        f"保留全部24个开发场景、360个计划条件记录；实际数值调用{audit['numerical_invocations']}次，实际物理执行{audit['actual_physics_executions']}次。",
        f"B3的{audit['not_evaluable']}条条件记录为not_evaluable。共同参考缺失涉及{audit['common_reference_missing']}条条件，不能解释为各方法已证实求解失败。",
        '三次重复是场景内计时重复；离线方法仅第0遍作物理执行。每台机器人有6个CAD×放置聚类、12个含方向子场景。',
        'Panda的3个柱面参考构造没有完成，原场景、失败结点及部分路径全部保留，不补换场景，不声称任务不可行。',
        '官方BoundMPC数学例程完成3次更新；其七关节IIWA状态布局及误差框没有完成Panda/UR5e扫描合同适配。该项没有性能排名。','',
        '## 2. 全场景质量与实测成本','',
        'B0=标称精确位姿IK+TOPPRA；B1=TB初始化+工艺域约束平滑+TOPPRA；',
        'G=同容差GN初始化、其余与B1相同；B2=共享B1起点的全变量构型—时间NLP；B3=尚未完成扫描适配的BoundMPC。','',
        '|机器人|设置|有规划计时的场景|物理执行|运动完成|扫描质量完成/12|合格作业时间中位数(s)|完整规划工作中位数(s)|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in main:
        lines.append('|'+ '|'.join([r['robot'],r['method'],str(r['timed_scenes']),str(r['physics_executed_scenes']),
            str(r['execution_completed_scenes']) if r['implementation_ready'] else 'N/E',
            str(r['quality_completed_scenes']) if r['implementation_ready'] else 'N/E',
            f(r['quality_execution_s_median']),f(r['planning_s_median'])])+'|')
    lines+=['','规划工作包含完整30 s检查点运行、构建、回调及独立验证；主路径只取10 s时已经验收的最佳可行解。',
        '因此表中不是“10 s必定返回”的规划器时延。B2的总成本还包括取得共同B1路径的原始成本；不把初值当免费。',
        '完成时间仅来自实际模拟反馈；不合格／未执行条件为null。上表各方法的合格时间子集可能不同，',
        '不能据这些边际中位数直接宣布加速；共同合格场景的配对差与比值另见`paired_intervals.csv`。',
        '一次及10/100次作业的完整／摊销成本在`scene_results.csv`，不把CPU耗时再次计入模拟运动时间。','',
        '|机器人|比较|共同质量合格场景|作业时间比（后者/前者）|聚类95%区间|',
        '|---|---|---:|---:|---|']
    for r in comparisons:
        if r['metric']=='quality_execution_s_ratio':
            ci='不可估计' if r['ci_lower'] is None else f"[{r['ci_lower']:.5f}, {r['ci_upper']:.5f}]"
            lines.append(f"|{r['robot']}|{r['baseline']} → {r['method']}|{r['paired_scenes']}|{r['mean_ratio']:.5f}|{ci}|")
    lines+=['','共同合格子集上的时间变化不能抵消全场景完成的损失；两类指标一起解释。','',
            '## 3. 旧Task Balance贡献在哪里','']
    known=[b for b in bridges if b.get('initial_q_difference_max_rad') is not None]
    changed=sum(b['initial_q_difference_max_rad']>1e-10 for b in known)
    calls=sum(r.get('tb_dual_continuation_calls') or 0 for r in rows if r['method']=='B1')
    lines += [f"第0遍可共同核对的初始化有{len(known)}个场景，其中{changed}个的TB/G关节结点差超过1e−10 rad。",
        f"场景内重复取平均后，TB局部对偶继续调用数之和为{calls:.1f}；详细trace保留实际调用，不为体现复用而额外调用。",
        '若第一QP已经合格，直接验收是原Task Balance数值设计的一部分；这种情况下TB与GN共享方向。',
        '桥接表分别列出初始化耗时、FEV、平滑位移、最终控制点差、规划和执行结果。公共平滑、重定时和控制器不归功于TB。','',
        '## 4. 全路径参照的有效范围','',
        'B1/G确实构造并调用了带完整工艺允许域、关节范围和碰撞间隙的约束平滑；不是无约束插值后只验原结点。',
        '但最终返回的可能是仍未被改善的可行初始拟合。每条optimizer trace记录更新数、末次约束残差、',
        '已验证incumbent和可用时间。没有改进不能隐藏，也不能只凭写出了NLP就断言它是足够强的最优参照。',
        'B2所有控制点与时间变量参与优化，起点是共同保存的B1第0遍可行路径；随后用同一完整重定时与密集验收。',
        '预算外的低目标值或不合格末次迭代不回填早期检查点。B2是否带来实际时间改进，应看B2—B1配对记录，而非只看求解器status。','',
        '']
    # Explain observed bridge differences by the actual saved same-input
    # optimizer histories, not by attributing every final outcome to the seed.
    insert_at=lines.index('## 4. 全路径参照的有效范围')
    explanation=[]
    for b in known:
        if b['initial_q_difference_max_rad']>1e-10 or b.get('tb_quality_completed')==b.get('gn_quality_completed'):continue
        histories={}
        for method in ('B1','G'):
            p=OUT/'runs'/b['slot']/f'{method}_r0/optimizer_trace.json.gz'
            with gzip.open(p,'rt') as file:histories[method]=json.load(file)['incumbents']
        updates={m:[v for v in h if v['source']=='nonlinear_validated_iterate'] for m,h in histories.items()}
        if all(updates.values()):
            a,c=updates['B1'][0],updates['G'][0]
            same=np.array_equal(np.asarray(a['coeff']),np.asarray(c['coeff']))
            explanation += [f"`{b['slot']}`的初值相同，但最终质量结果不同：首个平滑更新分别在{a['available_s']:.6f} s（B1）和{c['available_s']:.6f} s（G）可用。",
                f"该更新的控制点数组相同：{same}；10 s截点是否包含它不同。这是有限墙钟预算的截点差异，不是TB局部更新取得了不同构型。"]
    explanation+=['本组数据支持旧内核可复用；没有检验出TB对偶机制相对同容差GN的路径初始化增量。','']
    lines[insert_at:insert_at]=explanation
    for method in ('B1','G','B2'):
        group=[r for r in work if r['method']==method and 'optimizer_elapsed' in r]
        changed=sum(r.get('checkpoint_10.0_source')=='nonlinear_validated_iterate' for r in group)
        lines += [f"{method}实际完成{len(group)}次约束优化调用，其中{changed}次在10 s检查点采用了非线性验证后的更新，",
                  f"而非初始路径；全部检查点、真实可用时刻及末次约束残差见`numerical_work.csv`。"]
    lines += ['','## 5. 慢在哪里，下一阶段应改善什么','']
    for robot in ('panda','ur5e'):
        sub=[r for r in bottleneck if r['robot']==robot and r['method']=='B1']
        if sub:
            lines += [f"{robot}的可定时B1路径：平均计划时间{np.mean([r['predicted_execution_s'] for r in sub]):.3f} s；",
                      f"采样速度代理下界平均{np.mean([r['sampling_proxy_lower_bound_s'] for r in sub]):.3f} s；换行占计划时间平均{100*np.mean([r['turn_time_s']/r['predicted_execution_s'] for r in sub]):.1f}%。",
                      f"接近采样、关节速度、加速度限制的非互斥时间占比分别为{100*np.mean([r['time_fraction_near_sampling_cap'] for r in sub]):.1f}%、{100*np.mean([r['time_fraction_near_joint_speed'] for r in sub]):.1f}%、{100*np.mean([r['time_fraction_near_acceleration'] for r in sub]):.1f}%。"]
    lines += ['这些是实际计划中的约束接近程度，不是全程时间梯度或因果分解。说明性采样界假定中心射线具有连续速度上限，',
        '采用直扫描弦长减端点管道允许量，不含换行和加速度；不能称已证明的整个仿真任务最优值。',
        '实际控制跟踪还可能使计划合格的路径发生加速度或扫描质量失败。后续若获授权，应先面对共同初始化覆盖、',
        '全变量参照求解完成度、换行／加速度传播和真实采样质量，而不是再次修改单点IK损失。',
        '本轮没有自动启动该后续研究。','',
        '## 6. 数据、统计与边界','',
        '- `baseline_families.csv`保留全部曲面类别；`scene_results.csv`保留全部场景与缺失状态。',
        '- `failure_breakdown.csv`逐场景列出非互斥拒绝原因；`numerical_work.csv`分开返回码、实际迭代和已验证可行改进。',
        '- `paired_intervals.csv`复用旧聚类统计函数，先在场景内平均计时重复，再按机器人×CAD×放置做4000次配对聚类bootstrap（seed2026091607）。',
        '- 6聚类/机器人只能给出有限开发证据；区间含零不称等效，观测相同不称普遍无退化。',
        '- 质量完成要求原工艺覆盖／孔洞／采样阈值及物理运动条件同时成立；速度与加速度极值按每个1 ms步检查。',
        '- 回波是实际MuJoCo状态射线，不是理想轨迹；无硬件、真实反射、激光噪声、缺陷检测或硬实时保证。',
        '- 视频固定选择两机器人×三曲面×放置0×u方向×四离线设置，不按成功挑片；无物理记录者明确缺失。',
        '- 旧代码／论文／冻结证据不覆盖，正式场景只保留设计规则。',
        '', '更多实现差异见`PROCESS_SCAN_REUSE_MAP.md`；来源和BoundMPC适配边界见`PROCESS_SCAN_PRIOR_ART.md`；复算及停止点见`PROCESS_SCAN_PHASE1_HANDOVER.md`。','']
    (ROOT/'docs/PROCESS_SCAN_BASELINE_REPORT.md').write_text('\n'.join(lines))


def status_plot(rows):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    root=OUT/'reports/figures'
    # Every scene remains visible; missing is not colored as a solver failure.
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,robot in zip(axes,('panda','ur5e')):
        slots=sorted(set(r['slot'] for r in rows if r['robot']==robot));matrix=[]
        for slot in slots:
            vals=[]
            for method in METHODS:
                r=next(r for r in rows if r['slot']==slot and r['method']==method)
                vals.append(2 if r['quality_completed'] is True else 1 if r['physics_executed'] else 0)
            matrix.append(vals)
        ax.imshow(matrix,cmap=ListedColormap(['#D9D9D9','#E69F00','#009E73']),vmin=0,vmax=2,aspect='auto')
        ax.set_xticks(range(5),METHODS);ax.set_yticks(range(12),[s.replace(robot+'_','') for s in slots]);ax.set_title(robot.upper())
    fig.suptitle('Green: quality complete; orange: executed but rejected; grey: not executed',fontsize=9)
    export_figure(fig,root/'all_scene_status')
    plt.close(fig)


def export_figure(fig,stem):
    fig.savefig(str(stem)+'.svg',bbox_inches='tight')
    fig.savefig(str(stem)+'.pdf',bbox_inches='tight')
    fig.savefig(str(stem)+'.png',dpi=300,bbox_inches='tight')


def paired_plot(comparisons):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(7.2,5),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,metric in enumerate(('quality_completed','quality_execution_s')):
            ax=axes[i,j]
            for y,(base,method) in enumerate((('G','B1'),('B0','B1'),('B1','B2'))):
                hits=[r for r in comparisons if r['robot']==robot and r['metric']==metric and r['baseline']==base and r['method']==method]
                if not hits:
                    ax.text(.5,y,'No common measured pair',transform=ax.get_yaxis_transform(),ha='center',fontsize=7);continue
                r=hits[0];v=r['mean_difference']
                if r.get('ci_lower') is not None:
                    ax.errorbar(v,y,xerr=np.array([[v-r['ci_lower']],[r['ci_upper']-v]]),fmt='o',color='#0072B2',capsize=3,ms=4)
                else:ax.plot(v,y,'o',color='#777777',ms=4)
                ax.annotate(f"n={int(r['cluster_count'])}",(v,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6)
            ax.axvline(0,color='#555555',lw=.7,ls='--');ax.set_yticks(range(3),['B1 − G','B1 − B0','B2 − B1'])
            ax.set(ylim=(-.4,2.6),xlabel='Quality-yield difference' if j==0 else 'Common-quality execution difference (s)')
            ax.set_title(robot.upper(),loc='left')
    fig.suptitle('Paired cluster means and 95% bootstrap intervals; n = CAD–placement clusters',fontsize=8)
    export_figure(fig,OUT/'reports/figures/paired_effects');plt.close(fig)


def physical_detail_plots(rows,segments):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    index=[]
    for robot in ('panda','ur5e'):
        for family in ('plane','cylinder','saddle'):
            slot=f'{robot}_{family}_p0_u';task=ScanTask(family,0,'u',robot)
            exists=any((OUT/'runs'/slot/f'{m}_r0/scan_samples.npz').exists() for m in ('B0','B1','B2','G'))
            if not exists:
                index.append(dict(slot=slot,status='no_physical_record',files=None));continue
            fig,axes=plt.subplots(4,4,figsize=(11.8,10),layout='constrained')
            adapter,_=context(robot)
            for j,method in enumerate(('B0','B1','B2','G')):
                path=OUT/'runs'/slot/f'{method}_r0/scan_samples.npz'
                info=next(r for r in rows if r['slot']==slot and r['method']==method)
                axes[0,j].set_title(method+' | quality '+str(info['quality_completed']),fontsize=8)
                if path.exists():
                    rays=np.load(path);trace=np.load(OUT/'reports/process_traces'/f'{slot}_{method}_r0.npz')
                    axes[0,j].imshow(rays['coverage_mask'],origin='lower',extent=(-120,120,-80,80),
                        cmap=ListedColormap(['#F4A582','#D9F0D3']),vmin=0,vmax=1,aspect='equal')
                    target=task.references(np.linspace(0,1,1501))[0];uv=(target-task.center)@task.world_R
                    axes[0,j].plot(uv[:,0]*1000,uv[:,1]*1000,color='#777777',lw=.5,label='Prescribed center')
                    mask=rays['raw_valid'][:,40];uv=(rays['points'][mask,40]-task.center)@task.world_R
                    axes[0,j].plot(uv[:,0]*1000,uv[:,1]*1000,'.',color='#0072B2',ms=.8,label='Actual center rays')
                    axes[0,j].set(xlabel='CAD u (mm)',ylabel='CAD v (mm)',xlim=(-145,145),ylim=(-90,90))
                    values=trace['process'];values=values.copy();values[:,0]=abs(values[:,0]-.15)/.005
                    values[:,1]/=.001;values[:,2]/=np.deg2rad(10);values[:,3]/=np.deg2rad(5)
                    for k,(name,color) in enumerate(zip(('Standoff','Center ray','Incidence','Line angle'),('#0072B2','#D55E00','#009E73','#CC79A7'))):
                        axes[1,j].plot(trace['t'],values[:,k],label=name,color=color,lw=.6)
                    axes[1,j].axhline(1,color='black',ls='--',lw=.6)
                    axes[1,j].set(xlabel='Actual scan time (s)',ylabel='Process-limit utilization',ylim=(0,max(1.1,float(values.max())*1.05)))
                    with h5py.File(path.parent/'execution.h5') as history:
                        t=history['t'][:];velocity=np.max(abs(history['dq'][:])/(.5*adapter.public.limits.velocity),axis=1)
                        acceleration=np.max(abs(history['qacc'][:]),axis=1)/2
                    axes[3,j].plot(t,velocity,label='Actual velocity / limit',color='#009E73',lw=.65)
                    axes[3,j].plot(t,acceleration,label='Actual acceleration / limit',color='#D55E00',lw=.65)
                    axes[3,j].axhline(1,color='black',ls='--',lw=.6)
                    top=max(1.1,info['acceleration_utilization_max']*1.2,info['velocity_utilization_max']*1.2)
                    axes[3,j].set(xlabel='Actual execution time (s)',ylabel='Actual limits (5 ms samples)',ylim=(0,top))
                    axes[3,j].text(.02,.98,f"1 ms maxima: v={info['velocity_utilization_max']:.3f}, a={info['acceleration_utilization_max']:.3f}",
                                  transform=axes[3,j].transAxes,va='top',fontsize=6)
                else:
                    for ax in axes[[0,1,3],j]:ax.text(.5,.5,'Not executed',transform=ax.transAxes,ha='center')
                group=[r for r in segments if r['slot']==slot and r['method']==method]
                if group:
                    for key,name,color in (('sample_utilization','Sampling cap','#0072B2'),('speed_utilization','Joint speed','#009E73'),('acceleration_utilization','Joint acceleration','#D55E00')):
                        axes[2,j].plot([r['s'] for r in group],[r[key] for r in group],label=name,color=color,lw=.7)
                    axes[2,j].axhline(1,color='black',ls='--',lw=.6)
                    axes[2,j].set(xlabel='Full-path progress s',ylabel='Planned-limit utilization')
                else:axes[2,j].text(.5,.5,'No verified timed path',transform=axes[2,j].transAxes,ha='center',fontsize=7)
            for i in range(4):
                handles,labels=axes[i,0].get_legend_handles_labels()
                if handles:axes[i,3].legend(handles,labels,fontsize=6,loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
            fig.suptitle(f'{robot.upper()} / {family} / placement0 / u: actual coverage, process quality, planning and physical motion',fontsize=9)
            stem=OUT/'reports/figures'/f'physical_details_{slot}';export_figure(fig,stem);plt.close(fig)
            index.append(dict(slot=slot,status='rendered',selection='fixed placement0/u for every robot/family',
                              files=[str(Path(str(stem)+'.'+e).relative_to(ROOT)) for e in ('pdf','svg','png')]))
    write(OUT/'reports/figures/physical_detail_index.json',index)


def provenance():
    target=OUT/'provenance';target.mkdir(exist_ok=True)
    package=ROOT/'tmp/process_scan_v2_package/process_scan_continuity_v2'
    copied={}
    for name in ('MASTER_PLAN_V2.md','CODEX_PHASE1_REUSE_BASELINES.md','protocol_draft.yaml',
                 'metrics_schema.json','scene_design.json','REFERENCE_LINES.md','previous_reference_notes.md'):
        dest=target/'provided_design'/name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(package/name,dest)
        copied[name]=sha(dest)
    versions={}
    for name in ('numpy','scipy','pin','coal','mujoco','casadi','toppra','matplotlib','h5py'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='distribution_metadata_unavailable'
    models={};assets={};partial=[]
    for p in sorted((OUT/'inputs').glob('*/scene.xml')):
        for node in ET.fromstring(p.read_text()).findall('./asset/mesh'):
            source=Path(node.get('file'));assets[str(source)]=dict(sha256=sha(source),bytes=source.stat().st_size)
        identity=json.loads((p.parent/'identity.json').read_text());task=ScanTask(identity['family'],identity['placement'],identity['direction'],identity['robot'])
        u,v=np.meshgrid(np.linspace(-.18,.18,181),np.linspace(-.18,.18,181))
        xyz,_=task.surface(np.column_stack([u.ravel(),v.ravel()]));heights=xyz[:,2]-task.center[2]
        dest=target/'heightfields'/f"{identity['slot']}.npz";dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,height_m=heights.reshape(181,181),normalized=(heights-heights.min())/max(float(np.ptp(heights)),1e-4),center=task.center,rotation=task.world_R)
        if not identity['reference_geometry_feasible']:
            saved=np.load(p.parent/'reference_geometry.npz')
            correction=target/'partial_reference_indices'/f"{identity['slot']}.npz";correction.parent.mkdir(exist_ok=True)
            # The initial constructor's early-return record stored linspace over
            # the prefix. These metadata were NEVER used by an evaluated method.
            # Preserve it and supply the true original 401-node indices separately.
            np.savez_compressed(correction,q=saved['q'],s=np.arange(len(saved['q']))/400)
            partial.append(dict(slot=identity['slot'],count=len(saved['q']),
                erratum='Unavailable reference only: original saved s stretched partial prefix to 1; use corrected index/400 file. Placeholder common endpoints were not executed.',
                original_preserved=True,used_by_any_baseline=False,corrected=str(correction.relative_to(ROOT))))
    for robot in ('panda','ur5e'):
        adapter,path=context(robot)
        models[robot]=dict(urdf=str(path),urdf_sha256=sha(path),joint_names=list(adapter.public.joint_names),
                          lower=adapter.public.limits.lower,upper=adapter.public.limits.upper,
                          original_velocity=adapter.public.limits.velocity,
                          optimizer_backend='Pinocchio NativeGeometry',geometry_acceptance_backend='frozen NumPy URDFKinematics',
                          nonlinear_program_backend='CasADi/Ipopt/MUMPS',physics_backend='MuJoCo 3.10.0')
    write(target/'environment_and_assets.json',dict(python=sys.version,executable=sys.executable,versions=versions,
         packaging_time_utc=datetime.now(timezone.utc).isoformat(),os_release=Path('/etc/os-release').read_text(),
         provided_design_files=copied,models=models,mesh_assets=assets,
         asset_hash_capture_scope='Captured during result packaging; XML/FK equivalence and numerical-source input seal were captured before the comparison.',
         old_environment_upgraded=False,external_asset_paths_required=True))
    write(target/'partial_reference_metadata_errata.json',partial)
    license_target=target/'third_party_licenses';license_target.mkdir(exist_ok=True)
    upstream=ROOT/'tmp/process_scan_dependencies/BoundMPC/LICENSE'
    shutil.copyfile(upstream,license_target/'BoundMPC_MIT.txt')


def validate_artifacts(raw):
    """Check lineage and saved measurements without re-solving any condition."""
    seal=json.loads((OUT/'inputs/numerical_seal_grid_repair.json').read_text())
    assert hashes()==seal['code_hashes'],'Frozen numerical source changed'
    for relative,digest in seal['files'].items():assert sha(OUT/relative)==digest,relative
    identities={r['slot']:r for r in seal['scenes']};seen=set();physics=0;shared=0;start_differences=[]
    assert len(identities)==24 and all(r['split']=='development' for r in identities.values())
    for r in raw:
        key=r['slot'],r['method'],r['repeat'];assert key not in seen;seen.add(key)
        assert r['scene_uid']==identities[r['slot']]['scene_uid']
        root=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['code_hashes']==seal['code_hashes']
        assert manifest['old_online_interface_called'] is False
        assert manifest['physical_rate_in_geometry'] is False
        if r.get('quality_completed'):
            assert r['planner_feasible'] and r['execution_completed'] and r['physical_run']
            assert r['valid_coverage_fraction']>=.99 and r['max_hole_diameter_upper_bound_m']<=.002
            assert r['max_along_scan_gap_m']<=.001 and r['simulated_completion_time_s']>0
        if r.get('physical_run'):
            assert r['repeat']==0,'Offline physics must use the predetermined first output'
            physics+=1
            with h5py.File(root/'execution.h5') as f:
                assert not bool(f.attrs['qpos_runtime_overwrite'])
                assert f.attrs['source']=='torque_actuators_mj_step'
                t=f['t'][:];q=f['q'][:]
                assert np.isfinite(q).all() and np.isfinite(t).all()
                assert t[0]==0 and np.allclose(np.diff(t),.005,rtol=1e-8,atol=1e-12)
                start_differences.append(float(np.max(abs(q[0]-np.asarray(identities[r['slot']]['common_start'])))))
                np.testing.assert_allclose(q[0],identities[r['slot']]['common_start'],atol=1e-12)
            plan=np.load(root/'planned_path.npz')
            assert np.all(np.diff(plan['t'])>0) and np.all(np.diff(plan['s'])>1e-12)
            assert np.isfinite(plan['coeff']).all()
            if not r['execution_completed']:assert r['simulated_completion_time_s'] is None
        if r['method']=='B2' and (root/'path_initialization.npz').exists():
            initial=OUT/'shared_b1'/r['slot']/'path.npz'
            assert manifest['shared_initial_sha256']==sha(initial)
            a=np.load(initial);b=np.load(root/'path_initialization.npz')
            for field in ('coeff','s','x','u','t'):np.testing.assert_array_equal(a[field],b[field])
            shared+=1
        if r['status'] in ('not_evaluable','common_reference_unavailable','common_b1_initialization_missing'):
            assert not r['physical_run'] and r['total_plan_wall_s'] is None
            assert r['quality_completed'] is None
    expected={(slot,method,repeat) for slot in identities for method in METHODS for repeat in range(3)}
    assert seen==expected
    return dict(status='passed',unique_conditions=len(seen),physical_records_checked=physics,
                identical_B1_snapshot_checks=shared,numerical_and_input_hashes_unchanged=True,
                max_common_start_displacement_rad=max(start_differences,default=None),
                new_solver_calls=0,new_physics_calls=0,
                scope='saved lineage, timestamps, common starts, time grids, source hashes and recorded acceptance invariants; not a continuous-time proof')


def report():
    paths=sorted((OUT/'runs').glob('*/*/metrics.json'));raw=[json.loads(p.read_text()) for p in paths]
    if len(raw)!=360:raise RuntimeError(f'{len(raw)}/360 condition records; not a final report')
    write(OUT/'reports/artifact_verification.json',validate_artifacts(raw))
    derived=actual_state_derivatives(raw)
    rows=scene_rows(raw);root=OUT/'reports';csv_write(root/'all_runs.csv',raw);csv_write(root/'scene_results.csv',rows)
    csv_write(root/'actual_quality_and_motion.csv',derived)
    main=[];families=[]
    for robot in ('panda','ur5e'):
        for method in METHODS:
            sub=[r for r in rows if r['robot']==robot and r['method']==method]
            main.append(dict(robot=robot,method=method,**summary(sub)))
            for family in ('plane','cylinder','saddle'):
                families.append(dict(robot=robot,method=method,family=family,**summary([r for r in sub if r['family']==family])))
    csv_write(root/'baseline_main.csv',main);csv_write(root/'baseline_families.csv',families)
    comparisons=paired(rows);csv_write(root/'paired_intervals.csv',comparisons)
    bridges=bridge(rows);csv_write(root/'bridge_tb_vs_gn.csv',bridges)
    bottleneck,segments=bottlenecks(rows);csv_write(root/'bottlenecks.csv',bottleneck);csv_write(root/'bottleneck_segments.csv',segments)
    work=numerical_work(raw);csv_write(root/'numerical_work.csv',work)
    csv_write(root/'failure_breakdown.csv',failure_breakdown(rows))
    ledger=[]
    for r in raw:
        folder=OUT/'runs'/r['slot']/f"{r['method']}_r{r['repeat']}"
        ledger.append(dict(slot=r['slot'],method=r['method'],repeat=r['repeat'],status=r['status'],
            files={p.name:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()},
            absent_not_fabricated=[name for name in ('path_initialization.npz','planned_path.npz','execution.h5','scan_samples.npz','optimizer_trace.json.gz') if not (folder/name).exists()]))
    write(root/'raw_record_index.json',ledger)
    audit=dict(scheduled_conditions=360,condition_records=len(raw),
        numerical_invocations=sum(r.get('total_plan_wall_s') is not None for r in raw),
        actual_physics_executions=sum(bool(r.get('physical_run')) for r in raw),
        not_evaluable=sum(r['status']=='not_evaluable' for r in raw),
        common_reference_missing=sum(r['status']=='common_reference_unavailable' for r in raw),
        implementation_errors=[dict(slot=r['slot'],method=r['method'],repeat=r['repeat']) for r in raw if r['status']=='implementation_error'],
        independent_clusters_per_robot=6,timing_repeats_per_scene=3,formal_test_outcomes=0,
        B3='official IIWA example checked; Panda/UR5e process mapping NOT EVALUABLE',
        statuses={s:sum(r['status']==s for r in raw) for s in sorted(set(r['status'] for r in raw))})
    write(root/'summary.json',dict(audit=audit,main=main,paired=comparisons,bridge=bridges))
    plots(main,rows,bottleneck)
    status_plot(rows)
    paired_plot(comparisons)
    physical_detail_plots(rows,segments)
    markdown_report(audit,main,rows,bridges,bottleneck,work,comparisons)
    provenance()
    print(json.dumps(audit,indent=2))
