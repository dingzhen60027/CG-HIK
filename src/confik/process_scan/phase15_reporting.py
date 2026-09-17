"""Read-only reduction of Phase 1.5; all four methods/all scenes retained."""
import json
import gzip
from pathlib import Path
import numpy as np
import h5py
from scipy.interpolate import BSpline
from .phase15 import OUT,OLD,ROOT,code_hashes
from .study import write,sha
from .reporting import csv_write,scene_rows,summary,paired,bridge,numerical_work,failure_breakdown
from .models import context
from .task import ScanTask
from .planning import ray_center_function

METHODS=('B0','B1','G','B2')


def bottlenecks(rows):
    answer=[]
    for row in rows:
        path=OUT/'runs'/row['slot']/f"{row['method']}_r0/planned_path.npz"
        if not path.exists():continue
        identity=json.loads((OUT/'inputs'/row['slot']/'identity.json').read_text())
        task=ScanTask(**identity['task']);adapter,_=context(row['robot']);raw=np.load(path)
        C=raw['coeff'];basis=BSpline(raw['knots'],np.eye(len(C)),3)
        s=(raw['s'][:-1]+raw['s'][1:])/2;x=(raw['x'][:-1]+raw['x'][1:])/2;dt=np.diff(raw['t'])
        q=basis(s)@C;qs=basis(s,nu=1)@C;qss=basis(s,nu=2)@C
        vr=np.max(abs(qs)*np.sqrt(x)[:,None]/(.5*adapter.public.limits.velocity),axis=1)
        ar=np.max(abs(qss*x[:,None]+qs*raw['u'][:,None]),axis=1)/2
        ray=ray_center_function(task,adapter)
        sr=np.array([np.linalg.norm(np.asarray(ray(qq)[1])@dd)*np.sqrt(xx)/.198 for qq,dd,xx in zip(q,qs,x)])
        scan=task.references(s)[4];total=float(sum(dt))
        answer.append(dict(slot=row['slot'],robot=row['robot'],family=row['family'],method=row['method'],
            planned_total_s=total,planned_scan_s=float(sum(dt[scan])),planned_turn_s=float(sum(dt[~scan])),
            actual_scan_s=row.get('scan_duration_s'),actual_turn_s=row.get('transition_duration_s'),actual_settle_s=row.get('settling_duration_s'),
            near_public_joint_speed_fraction=float(sum(dt[vr>=.95])/total),
            near_reserved_joint_speed_fraction=float(sum(dt[vr/.90>=.95])/total),
            near_public_acceleration_fraction=float(sum(dt[ar>=.95])/total),
            near_reserved_acceleration_fraction=float(sum(dt[ar/.90**2>=.95])/total),
            near_reserved_sampling_fraction=float(sum(dt[(sr/.90>=.95)&scan])/total),
            actual_acceleration_max=row.get('acceleration_utilization_max'),actual_tracking_max_rad=row.get('tracking_error_max_rad'),
            torque_saturation_fraction=row.get('torque_saturation_fraction'),
            constraint_occupancy_is_nonexclusive=True,quality_completed=row['quality_completed']))
        dest=OUT/'reports/planned_derivatives'/f"{row['slot']}_{row['method']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,s=s,dt=dt,q=q,q_dot=qs*np.sqrt(x)[:,None],q_ddot=qss*x[:,None]+qs*raw['u'][:,None],
                            scan=scan,sample_utilization=sr,speed_utilization=vr,acceleration_utilization=ar)
    return answer


def calibration_tables():
    margin=[];b2=[];reference=[]
    for p in sorted((OUT/'execution_calibration').glob('*/*/execution_metrics.json')):
        m=json.loads(p.read_text());t=json.loads((p.parent/'timing.json').read_text())
        margin.append(dict(path=p.parent.parent.name,setting=p.parent.name,planned_s=t['duration'],**m))
    for p in sorted((OUT/'b2_calibration').glob('*/summary.json')):
        d=json.loads(p.read_text());b2.append(dict(problem=p.parent.name,status=d['status'],initial_s=d['initial_s'],
            known_faster_s=d['known_faster_s'],adopted_s=d['adopted_s'],
            initial_actual_s=d['metrics']['initial']['simulated_completion_time_s'],
            adopted_actual_s=d['metrics']['adopted']['simulated_completion_time_s'],
            known_faster_actual_s=d['metrics']['known_faster']['simulated_completion_time_s'],
            **d['derivative_check']))
    for p in sorted((OUT/'inputs').glob('*/identity.json')):
        d=json.loads(p.read_text());reference.append(dict(slot=d['slot'],scene_uid=d['scene_uid'],
            available=d['reference_geometry_feasible'],selected_tilt_deg=d['selected_tilt_deg'],reference_source=d['reference_source']))
    csv_write(OUT/'reports/execution_calibration.csv',margin);csv_write(OUT/'reports/b2_calibration.csv',b2)
    csv_write(OUT/'reports/reference_audit.csv',reference)
    return margin,b2,reference


def figures(main,rows,bottle,margin,b2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
        'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    root=OUT/'reports/figures';root.mkdir(parents=True,exist_ok=True)
    write(root/'contract.json',dict(conclusion='Distinguish verified baseline capability, realized quality, and execution bottlenecks.',
        archetype='quantitative grid',backend='Python matplotlib',unit='scene; timing repeats averaged within scene',
        paired_intervals='six CAD-placement clusters per robot; directions nested',
        plots='all scene values or explicitly qualified paired subsets; no plotted inferential bars',
        limits='Missing/failing paths count against all-scene yield; no formal or proposed-method result',
        export='editable SVG, vector PDF, 300 dpi PNG; minimum 6 pt',source='reports CSV and immutable recorded execution'))
    colors=['#467a9b','#49a191','#a88eb8','#c7884f']
    def save(fig,name):
        fig.savefig(root/f'{name}.svg',bbox_inches='tight')
        fig.savefig(root/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(root/f'{name}.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,5.8),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,method in enumerate(METHODS):
            a=next(r for r in main if r['robot']==robot and r['method']==method)
            axes[i,0].bar(j,a['quality_completed_scenes'],color=colors[j])
            axes[i,0].text(j,a['quality_completed_scenes']+.15,str(a['quality_completed_scenes']),ha='center',fontsize=7)
            for ax,key in ((axes[i,1],'quality_execution_s'),(axes[i,2],'total_plan_wall_s')):
                values=[r[key] for r in rows if r['robot']==robot and r['method']==method and r.get(key) is not None]
                ax.scatter(np.full(len(values),j),values,s=12,color=colors[j],alpha=.7)
                if values:ax.plot([j-.2,j+.2],[np.median(values)]*2,color='black',lw=1)
        axes[i,0].set(ylim=(0,13),yticks=[0,3,6,9,12],ylabel=f'{robot.upper()}\nQuality-complete / 12')
        axes[i,1].set_ylabel('Qualified actual cycle time (s)');axes[i,2].set_ylabel('All measured planning work (s)')
        for ax in axes[i]:ax.set_xticks(range(4),METHODS);ax.grid(axis='y',alpha=.15)
    for ax,title in zip(axes[0],('a  All scheduled scenes','b  Qualified scenes only','c  All timed scenes')):ax.set_title(title,loc='left')
    save(fig,'core_baselines')
    fig,axes=plt.subplots(1,3,figsize=(9,3),layout='constrained')
    names=sorted(set(r['path'] for r in margin))
    for label,color in (('before','#888888'),('after','#467a9b')):
        vals=[next(r for r in margin if r['path']==name and r['setting']==label) for name in names]
        axes[0].plot(range(len(names)),[r['acceleration_utilization_max'] for r in vals],'o-',label=label,color=color)
        axes[1].plot(range(len(names)),[1000*r['max_hole_diameter_upper_bound_m'] for r in vals],'o-',label=label,color=color)
    axes[0].axhline(1,color='black',ls='--',lw=.7);axes[1].axhline(2,color='black',ls='--',lw=.7)
    for ax in axes[:2]:ax.set_xticks(range(len(names)),[n.replace('panda','P').replace('ur5e','U').replace('_','\n') for n in names],fontsize=6)
    axes[0].set(title='a  Isolated margin calibration',ylabel='Actual acceleration / public limit');axes[0].legend(fontsize=7)
    axes[1].set(title='b  Coverage holes',ylabel='Hole bound (mm)')
    for k,r in enumerate(b2):
        axes[2].plot([0,1,2],[r['initial_actual_s'],r['adopted_actual_s'],r['known_faster_actual_s']],'o-',label=r['problem'])
    axes[2].set_xticks([0,1,2],['Slow input','B2 adopted','Known faster'],fontsize=7)
    axes[2].set(title='c  Independent B2 validation',ylabel='Quality-qualified actual cycle (s)');axes[2].legend(fontsize=7)
    save(fig,'calibration_closure')
    fig,axes=plt.subplots(3,2,figsize=(8,7),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        for setting,color in (('before','#888888'),('after','#467a9b')):
            source=OUT/'execution_calibration'/name/setting/'execution.h5'
            with h5py.File(source) as f:
                t=f['t'][:];actual=np.max(abs(f['qacc'][:]),axis=1)/2
                planned=np.max(abs(f['ddqr'][:]),axis=1)/2
            ax.plot(t,actual,color=color,lw=.55,label=f'{setting}: actual')
            ax.plot(t,planned,color=color,ls='--',lw=.5,label=f'{setting}: planned')
        ax.axhline(1,color='black',ls=':',lw=.7)
        ax.set(title=name.replace('_',' '),xlabel='Actual time (s)',ylabel='Acceleration / public limit')
    axes[0,0].legend(fontsize=6,ncol=2)
    save(fig,'one_ms_calibration_traces')
    fig,axes=plt.subplots(2,2,figsize=(8,5.5),layout='constrained')
    for i,robot in enumerate(('panda','ur5e')):
        for j,method in enumerate(METHODS):
            rr=[r for r in bottle if r['robot']==robot and r['method']==method]
            if rr:
                turn=[r['planned_turn_s']/r['planned_total_s'] for r in rr]
                axes[i,0].scatter(np.full(len(rr),j),turn,color=colors[j],s=14)
        a={r['slot']:r for r in rows if r['robot']==robot and r['method']=='B1'}
        b={r['slot']:r for r in rows if r['robot']==robot and r['method']=='B2'}
        keys=[k for k in a if a[k]['quality_execution_s'] is not None and b[k]['quality_execution_s'] is not None]
        axes[i,1].scatter(range(len(keys)),[b[k]['quality_execution_s']-a[k]['quality_execution_s'] for k in keys],color=colors[3],s=18)
        axes[i,1].axhline(0,color='black',lw=.6);axes[i,1].set(xlabel=f'Jointly quality-complete scenes (n={len(keys)})',ylabel='Actual B2 − B1 cycle time (s)')
        axes[i,0].set_xticks(range(4),METHODS);axes[i,0].set_ylabel(f'{robot.upper()}\nPlanned transition-time fraction')
    axes[0,0].set_title('a  All available paths',loc='left');axes[0,1].set_title('b  Paired physical comparison',loc='left')
    save(fig,'bottlenecks_and_paired_times')


def markdown_report(main,rows,bottle,work,failures,bridges,refs,b2):
    available=sum(r['available'] for r in refs);physical=sum(r['physics_executed'] for r in rows)
    motion=sum(r['execution_completed'] is True for r in rows);quality=sum(r['quality_completed'] is True for r in rows)
    fmt=lambda v:'N/E' if v is None else f'{v:.3f}'
    lines=['# Process Scan Phase 1.5：基线收口结果','',
        '本轮仅为原24开发场景的基线收口。没有新方法、正式96场景或论文改写。',
        '旧证据全部保留，新结果根目录为 `outputs/process_scan/phase15_baselines/`。','',
        '## 1. 共同参考与公共余量','',
        f'全部 **{available}/24** 场景取得独立共同参考。三个原Panda cylinder缺失场景均已补齐；',
        '详情见 `PROCESS_SCAN_REFERENCE_AUDIT.md`。可行的是完整工艺允许域，不能据此称每个标称姿态中心精确可达。','',
        '隔离的六条校准路径统一确定：扫描端点延伸3 mm、轮廓重叠余量1 mm、',
        '路径速度系数0.90；共同控制器保持400/40，质量阈值不变。新增轮廓使u/v分别为',
        '10/14条，所有方法共同使用96个C²样条控制点。该公共路线变更不是同一旧路径上的',
        '纯控制干预，不能把前后总体差异全部归因为一个余量。校准前后12次执行均保存1ms状态。','',
        f'核心比较共288次规划条件，预指定第一遍中有{physical}/96项产生可执行计划；',
        f'实际运动合格{motion}项，完整扫描质量合格{quality}项。未生成计划的条件不假填物理时间。',
        '所有控制饱和、速度/加速度超限、采样空缺、覆盖不足及孔洞都保留，不能由校准成功推断全部场景可靠通过。','',
        '## 2. 四核心基线','',
        '|机器人|方法|质量合格/12|执行次数|质量合格实际周期中位/s|全部规划成本中位/s|',
        '|---|---|---:|---:|---:|---:|']
    for r in main:
        lines.append(f"|{r['robot']}|{r['method']}|{r['quality_completed_scenes']}|{r['physics_executed_scenes']}|{fmt(r['quality_execution_s_median'])}|{fmt(r['planning_s_median'])}|")
    lines += ['', '每个场景的三遍是规划计时重复，先在场景内平均。实际执行只有预指定r0，不把',
        '三遍当成三份独立物理证据。表内条件成功集不同，不能用边际周期中位数直接宣称加速。',
        '配对结果在 `paired_intervals.csv`：同机器人6个CAD×放置聚类，两个方向嵌套，',
        '4000次预定聚类bootstrap、95%区间；时间仅比较共同合格场景，并列全样本质量差异。',
        '区间包含零不表示等效。BoundMPC保留官方例程已核对、扫描适配not_evaluable的范围说明，不进入性能表。','',
        '## 3. B2是否成为可信全变量参照','',
        '三个独立冗余慢路径校准均采用了非初始、密集验证且实际执行更快的路径。',
        '目标及约束的有限差分和检查点复算通过；完整物理扫描通过。',
        '因此它已不是只会返回初值的无效实现，但仍是有限预算局部参照，不是全局最优解。',
        '详见 `PROCESS_SCAN_B2_VALIDATION.md`。','']
    for robot in ('panda','ur5e'):
        ww=[r for r in work if r['robot']==robot and r['method']=='B2' and 'optimizer_status' in r]
        improved=sum(r.get('checkpoint_10.0_source')=='nonlinear_validated_iterate' for r in ww)
        later=sum(r.get('checkpoint_30.0_source')=='nonlinear_validated_iterate' for r in ww)
        lines.append(f'- {robot}：{len(ww)}次实际B2调用；10秒采用新可行路径{improved}次，30秒内找到新可行路径{later}次。')
    lines += ['', '主路径使用10秒已验证incumbent，费用包含完整30秒检查点实验以及构图、回调和验收；',
        'B2还计入获取共同B1初值的成本，不把它称为10秒端到端返回保证。','',
        '## 4. 时间与失败来自哪里','',
        '|机器人|方法|计划扫描占比|计划换行占比|近预留加速度占比|近预留关节速度占比|实际等待均值/s|',
        '|---|---|---:|---:|---:|---:|---:|']
    for robot in ('panda','ur5e'):
        for method in METHODS:
            bb=[r for r in bottle if r['robot']==robot and r['method']==method]
            if not bb:continue
            total=sum(r['planned_total_s'] for r in bb);wait=[r['actual_settle_s'] for r in bb if r['actual_settle_s'] is not None]
            acc=sum(r['near_reserved_acceleration_fraction']*r['planned_total_s'] for r in bb)/total
            vel=sum(r['near_reserved_joint_speed_fraction']*r['planned_total_s'] for r in bb)/total
            lines.append(f"|{robot}|{method}|{sum(r['planned_scan_s'] for r in bb)/total:.1%}|{sum(r['planned_turn_s'] for r in bb)/total:.1%}|{acc:.1%}|{vel:.1%}|{fmt(float(np.mean(wait)) if wait else None)}|")
    lines += ['', '扫描/换行/稳定等待按预定义段累加。加速度和速度是非互斥约束占用，不能与',
        '扫描时间再相加当成独立时间份额；“控制耗时”这里仅指实际稳定等待和跟踪/饱和记录，',
        '不虚构控制器导致的反事实时间贡献。近预留加速度按公共0.90²额度计，近预留速度按0.90额度计。','']
    counts={}
    for r in failures:
        for reason in r['nonexclusive_reasons']:counts[reason]=counts.get(reason,0)+1
    lines += ['失败原因（非互斥）：','']+[f'- {k}：{v}项。' for k,v in sorted(counts.items())]
    lines += ['', '全部场景与每类结果、原始失败、时限、孔洞均保留在CSV及r0实际记录中。',
        '下面的TB/G桥接只用于分离初始化，不把墙钟截断导致的平滑路径差异归因于IK损失。','']
    paired_init=[r for r in bridges if 'initial_q_difference_max_rad' in r]
    equal=sum(r['initial_q_difference_max_rad']==0 for r in paired_init)
    lines.append(f'TB/G可比较初始化{len(paired_init)}对，其中逐数值完全一致{equal}对；全部差异见 `bridge_tb_vs_gn.csv`。')
    lines += ['', '## 5. 后续应该优化什么','',
        '后续只定义为固定扫描直线段、优化换行过渡及相邻边界层，在工艺域、碰撞和物理',
        '执行质量约束下联合调整工具姿态、关节构型与时间。时间敏感性选块、均匀选块、',
        '曲率选块的具体算法尚未实现；B2为共同初值下的全变量参照。',
        '这面对的是整条路径与物理执行的瓶颈，不需要继续修改单点TB损失。',
        '冗余/任务容差覆盖规划和路径—时间双层优化的已有先例见 `PROCESS_SCAN_METHOD_TARGET.md`。','',
        '## 复算与证据范围','',
        '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv-process-scan/bin/python -m confik.process_scan.phase15_reporting',
        'OPENBLAS_NUM_THREADS=1 .venv-process-scan/bin/python scripts/verify_process_scan_phase15.py',
        '```','',
        '报告与校验是只读原始记录，不启动规划或物理新试验。代码和输入seal在核心比较前固定；',
        '源文件一致性、B2共同初值逐位一致、实际1ms峰值和覆盖重算另存QA。旧TB/GN、',
        '旧几何与verifier、旧论文及全部Phase1输出均不覆盖。图为matplotlib源数据绘图，',
        '统计按完整场景聚类；先例仅核对指定两篇，不用新文献宣称新颖性。',
        '停止在本报告，等待用户与ChatGPT验收。']
    (ROOT/'docs/PROCESS_SCAN_PHASE15_REPORT.md').write_text('\n'.join(lines)+'\n')


def report():
    raw=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/*/metrics.json'))]
    if len(raw)!=288:raise RuntimeError(f'Expected 288 core conditions, found {len(raw)}')
    rows=scene_rows(raw,METHODS);main=[];families=[]
    for robot in ('panda','ur5e'):
        for method in METHODS:
            sub=[r for r in rows if r['robot']==robot and r['method']==method]
            main.append(dict(robot=robot,method=method,**summary(sub)))
            for family in ('plane','cylinder','saddle'):
                families.append(dict(robot=robot,method=method,family=family,**summary([r for r in sub if r['family']==family])))
    bridges=bridge(rows,OUT);work=numerical_work(raw,OUT);fails=failure_breakdown(rows,OUT);bottle=bottlenecks(rows);intervals=paired(rows)
    for name,values in (('all_conditions',raw),('scene_results',rows),('baseline_main',main),('baseline_families',families),
                        ('bridge_tb_vs_gn',bridges),('bottlenecks',bottle),('failure_breakdown',fails),('numerical_work',work),('paired_intervals',intervals)):
        csv_write(OUT/'reports'/f'{name}.csv',values)
    margin,b2,refs=calibration_tables();figures(main,rows,bottle,margin,b2)
    markdown_report(main,rows,bottle,work,fails,bridges,refs,b2)
    hashes=json.loads((OUT/'inputs/seal.json').read_text())['code_hashes']
    write(OUT/'reports/integrity.json',dict(conditions=len(raw),expected_conditions=288,
        source_unchanged=hashes==code_hashes(),common_references=sum(r['available'] for r in refs),
        physical_runs=sum(r.get('physical_run',False) for r in raw),formal_scenes_run=False,
        raw_record_index=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((OUT/'runs').rglob('*')) if p.is_file()]))
    write(OUT/'reports/summary.json',dict(main=main,b2_calibration=b2,reference_count=sum(r['available'] for r in refs),
        failure_rows=fails,paired_intervals=intervals))
    print(json.dumps(main,indent=2),flush=True)


if __name__=='__main__':report()
