"""Write the fixed-evaluation readout directly from validated source tables."""
from . import study as old
from .locked_study import PRIMARY,ANALYTIC,configurations


def write_findings(root,tables,output_path=None):
    cfg,_,_=configurations()
    main={(r['robot'],r['method']):r for r in tables['main_table']}
    pairs={(r['robot'],r['baseline'],r['metric']):r for r in tables['paired_comparisons']}
    def delta(robot,baseline,metric):
        p=pairs[(robot,baseline,metric)];lo,hi=p['difference_ci']
        return f"{100*p['difference']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]"
    def ratio(robot,baseline):
        p=pairs[(robot,baseline,'total_latency_ns')];lo,hi=p['ratio_ci']
        return f"{p['ratio']:.3f}× [{lo:.3f}, {hi:.3f}]"
    def direction(robot):
        p=pairs[(robot,ANALYTIC,'completion')];d=p['difference'];lo,hi=p['difference_ci']
        word='观察到完成增量' if d>0 else '观察到完成退化' if d<0 else '观察均值相同'
        uncertainty='区间跨越或包含零，未建立稳定增益、等效性或无退化。' if lo<=0<=hi else '描述性配对区间未跨零；这不是多重检验后的显著性或通用保证。'
        return word+'；'+uncertainty
    lines=['# Elastic CR-IK μ=0.25：固定候选独立评估', '',
        '**本轮不再选参数。统一μ=0.25来自已观察开发结果，本报告只评估其在新轨迹上的表现。**', '',
        '每台机器人160条新轨迹，四类各40条，每条300帧。六个设置均运行三遍，共5760次整轨迹运行、1728000帧。'
        '未筛掉失败或超时，未用新结果改变输入、需求参数或数值预算。旧论文及旧证据不变。', '',
        'μ=0解析对照在初始合法对上直接保留完全相同的备用q和名义z；其他情况复用冻结的几何恢复。'
        '其目标没有改变；旧低效μ=0的时间不是本报告的比较分母。共享备用结果的决策等价测试不意味着随机TRAC整轨迹逐遍一致。', '',
        '## 完整主结果', '',
        '完成数与按时完成数按三遍分别列出（每遍分母160）；TSR/DTSR20先在UID内平均。'
        'P50/P95/P99为全部帧的描述性时延，包含失败与超时；累计时间为三遍均值。', '',
        '| 机器人 | 方法 | 完成数 | 全帧≤20ms完成数 | TSR / DTSR20 | P50/P95/P99 ms | 每160条累计s | 加速度RMS rad/s² |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for robot in cfg['robots']:
        for method in cfg['methods']:
            r=main[(robot,method)]
            lines.append(f"| {robot} | {r['label']} | {'/'.join(map(str,r['completion_by_repeat']))} | "
                f"{'/'.join(map(str,r['deadline_completion_by_repeat']))} | {r['tsr']:.2%} / {r['dtsr20']:.2%} | "
                f"{r['p50_ms']:.3f}/{r['p95_ms']:.3f}/{r['p99_ms']:.3f} | {r['cumulative_ms_per_sweep']/1000:.3f} | {r['acceleration_rms_mean']:.3f} |")
    lines += ['', '## 1. UR5e开发增量是否重现？', '',
        f"相对公平μ=0：ΔTSR={delta('ur5e',ANALYTIC,'completion')}；ΔDTSR20={delta('ur5e',ANALYTIC,'deadline_completion')}。{direction('ur5e')}", '',
        '这是新UID、300帧轨迹上的独立检验，不是对旧开发trajectory_15的重复，也不据此更换μ。', '',
        '## 2. Panda是增量、持平还是退化？', '',
        f"相对公平μ=0：ΔTSR={delta('panda',ANALYTIC,'completion')}；ΔDTSR20={delta('panda',ANALYTIC,'deadline_completion')}。{direction('panda')}", '',
        '两机器人分别分析；不以合并平均掩盖方向差异。', '',
        '## 3. 修正缺口代价相对公平μ=0增加了什么？', '',
        '| 机器人 | ΔTSR [95%区间] | ΔDTSR20 [95%区间] | 累计成本比 [95%区间] | 部分采用率 | 优化调用率 | 平均归一化干预 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for robot in cfg['robots']:
        r=main[(robot,PRIMARY)]
        lines.append(f"| {robot} | {delta(robot,ANALYTIC,'completion')} | {delta(robot,ANALYTIC,'deadline_completion')} | "
            f"{ratio(robot,ANALYTIC)} | {r['partial_correction_rate']:.2%} | {r['optimization_call_rate']:.2%} | {r['intervention_normalized_mean']:.4f} |")
    lines += ['', '这些增量比较把不必要的μ=0锥求解去掉了。已采用部分修正的q/z均按真实FK重新检查，'
        '实际目标与缺口用更新构型复算。部分采用率、γ或缺口下降本身不等于完成收益；以本节任务对照为准。', '',
        '| 机器人 | 对照 | UID均值改善数 | UID均值损失数 | 稳定3/3 vs 0/3改善 | 稳定损失 |',
        '| --- | --- | --- | --- | --- | --- |']
    for robot in cfg['robots']:
        for base in [m for m in cfg['methods'] if m!=PRIMARY]:
            group=[c for c in tables['gained_lost_uids'] if c['robot']==robot and c['baseline']==base]
            vals=[sum(c['change']==which and (not stable or c['stable_all_vs_none']) for c in group)
                for which,stable in [('gained',False),('lost',False),('gained',True),('lost',True)]]
            lines.append(f"| {robot} | {main[(robot,base)]['label']} | "+' | '.join(map(str,vals))+' |')
    stable=[c for c in tables['gained_lost_uids'] if c['baseline']==ANALYTIC and c['change']=='gained' and c['stable_all_vs_none']]
    for c in stable:
        lines += ['', f"局部成功实例：{c['robot']} / {c['family']}，UID `{c['uid']}`，Elastic为3/3完成、解析μ=0为0/3。"
            '这是一个UID内的三次搜索重复，不是三条独立轨迹；它支持存在局部成功实例，不抵消其他UID的损失，也不建立总体收益。']
    lines += ['', '全部UID、三遍完成率和首次失败输入见source-data，不只展示恢复实例。', '',
        '## 4. 相对TRAC/Pink的完成—时限—成本折中', '',
        '| 机器人 | 对照 | ΔTSR [95%区间] | ΔDTSR20 [95%区间] | 累计成本比 [95%区间] |',
        '| --- | --- | --- | --- | --- |']
    for robot in cfg['robots']:
        for base in ('trac_task_5ms','trac_task_20ms','pink_qp','two_step_predictive'):
            lines.append(f"| {robot} | {main[(robot,base)]['label']} | {delta(robot,base,'completion')} | "
                f"{delta(robot,base,'deadline_completion')} | {ratio(robot,base)} |")
    lines += ['', '本次尚未建立新增修正缺口代价的实用必要性。UR5e相对Pink和原普通双步预测有明确的样本收益，'
        '但公平解析μ=0保留了近似的完成均值，因此这些对照不能单独归因于修正目标。'
        '相对TRAC-IK 20ms，UR5e的完成与DTSR20均值更低、累计成本均值更高；'
        'Panda相对Pink的完成与时限点估计也不利。各区间和尾时延仍完整保留，不据此宣称全指标支配或等效。', '',
        '是否值得增加计算必须同时看本表的任务与时限差异，而不是仅与昂贵旧实现比速度。'
        '20ms是软件评价deadline，不是硬实时证明；算法总时间包含备用求解、预测/需求、初始对、锥构造/求解和非线性验收。', '',
        '## 全部类别与实际误差', '',
        '| 机器人 | 类别 | 方法 | TSR / DTSR20 | 累计s | P95 ms | 加速度RMS |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in tables['family_table']:
        lines.append(f"| {r['robot']} | {cfg.get('family_display',{}).get(r['family'],r['family'])} | {r['label']} | "
            f"{r['tsr']:.2%} / {r['dtsr20']:.2%} | {r['cumulative_ms_per_sweep']/1000:.3f} | {r['p95_ms']:.3f} | {r['acceleration_rms_mean']:.3f} |")
    lines += ['', '| 机器人 | 方法 | 接受位置误差P95/max mm | 接受姿态误差P95/max deg | 接近>90%容差比例 |',
        '| --- | --- | --- | --- | --- |']
    import math
    for r in tables['main_table']:
        lines.append(f"| {r['robot']} | {r['label']} | {r['p95_position_m']*1000:.4f}/{r['max_position_m']*1000:.4f} | "
            f"{math.degrees(r['p95_orientation_rad']):.4f}/{math.degrees(r['max_orientation_rad']):.4f} | {r['accepted_near_tolerance_rate']:.2%} |")
    violations=sum(r['accepted_contract_violations'] for r in tables['main_table'])
    lines += ['', f'接受命令合同违约：{violations}。误差统计以接受帧为分母；失败帧仍进入成功率和全部成本。', '',
        '## 5. 已建立与尚未建立的主张', '',
        '已建立的是固定候选在这两组完整新轨迹上的实际命令、完成、deadline与成本比较，'
        '以及每个记录的解析直接返回或部分改善决策。哪些机器人有正向均值，按上面的配对结果逐一判断。'
        '尚未建立μ最优、普遍优越、无退化、非线性未来可行性保证或硬实时能力。'
        '局部修正需求是经验估计，不是概率保证；单帧合法也不保证下一帧一定存在合法延拓。', '',
        '## 统计、来源与停止点', '',
        '独立单位是每机器人160个trajectory UID，先平均三遍，再在四个family内配对bootstrap 4000次。'
        '95%区间是描述性、未多重校正的百分位区间；不把帧或搜索重复当新增独立样本，'
        '也不因区间含零声称等效。主比较预先固定，未延长采样或删除不利类别。', '',
        '候选、配置、数值代码与全部身份在运行前提交于`67645112`；每次运行的实际Git SHA、'
        'CPU分配、依赖与全部原始记录hash均保存。只读复核覆盖所有帧、反馈状态、因果需求、'
        '名义q/z以及全部已记录非线性trial，不重新运行IK。', '',
        '- [完整source-data](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/source_data.json)',
        '- [配对区间](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/paired_comparisons.csv)',
        '- [恢复/损失UID](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/gained_lost_uids.csv)',
        '- [首次失败输入](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/first_failure_inputs.csv)',
        '- [原始记录与协议](../outputs/correction_reserve_ik/elastic_locked_evaluation/)', '',
        '**本轮结束：保留固定结果，不自动搜索μ、增加模块、启动另一轮评估或重写论文。**', '']
    lines[2:2]=['**结论：本次未确认修正缺口代价的独立完成收益。** '
        f"相对公平解析μ=0，Panda的ΔTSR为{delta('panda',ANALYTIC,'completion')}，"
        f"UR5e为{delta('ur5e',ANALYTIC,'completion')}；累计成本比分别为"
        f"{ratio('panda',ANALYTIC)}、{ratio('ur5e',ANALYTIC)}。"
        'UR5e开发期完成增量未得到清楚的独立复现；Panda点估计略降。区间含零不是等效或无退化证明。','']
    path=output_path if output_path is not None else old.ROOT/'docs/CRIK_ELASTIC_LOCKED_EVALUATION.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))


def finalize_delivery():
    """Package an already completed read-only report; never invoke an IK solver."""
    import importlib.metadata
    import json
    import os
    import subprocess
    from .locked_study import verify_seal
    cfg,_,root=configurations();seal=verify_seal()
    doc=old.ROOT/'docs/CRIK_ELASTIC_LOCKED_EVALUATION.md'
    assert doc.is_file() and (root/'reports/verification_manifest.json').is_file()
    versions=json.loads((root/'protocol/dependencies.json').read_text())
    assert all(importlib.metadata.version(k)==v for k,v in versions['packages'].items())
    assert old.sha(old.ROOT/versions['native_library'])==versions['native_library_sha256']
    hardware=json.loads(subprocess.check_output(['lscpu','-J'],text=True,env=dict(os.environ,LC_ALL='C')))
    old.write_json(root/'delivery_hardware.json',hardware)
    readme='''# Elastic CR-IK locked evaluation

Candidate and identities were committed before evaluation. The only primary is
Elastic mu=0.25, shared by Panda and UR5e; all six settings have three repetitions.
160 independent UIDs/robot, four equal families, 300 frames each: 5760 runs and
1,728,000 measured calls. No new mu search, old evidence replacement or paper edit.

`protocol/` contains the pre-outcome seal, online inputs, disjoint identities,
separate verified reference paths and unchanged dependency/demand parameters.
`panda/runs/` and `ur5e/runs/` contain every frame and run summary, including
failures/timeouts. Each robot's completed manifest hashes every raw record.
`reports/` contains full/family tables, UID-averaged paired comparisons, completion
UIDs, first-failure inputs, decision/timing tables and a read-only nonlinear audit.
Successful raw trajectories retain their actual accepted q sequence, not q_ref.

The six-method run is complete. Do not rerun it to replace these results.
The independent numerical entry point is
`python -m confik.correction_reserve.locked_study`; its shell wrapper requires an
explicit prepare/check/panda/ur5e action and refuses occupied result directories.
Read-only reporting is `python -m confik.correction_reserve.locked_reporting`;
it also refuses to overwrite an existing report. See the frozen protocol for
the environment and the main document `docs/CRIK_ELASTIC_LOCKED_EVALUATION.md`.

Outer latency includes all command-ready operations. Offline future-error joins,
serialization and this audit are not online compute. dt=20 ms is not a hard
real-time guarantee. All statistical intervals use trajectory UIDs, with the
three repeats averaged first; Pink timing repeats are not independent samples.
'''
    with (root/'README.md').open('x',encoding='utf8') as f:f.write(readme)
    paths=['src/confik/correction_reserve/locked_reporting.py','src/confik/correction_reserve/locked_findings.py']
    old.write_json(root/'delivery_manifest.json',dict(utc=old.utc(),baseline=cfg['baseline_commit'],
        selected_mu=.25,robots=cfg['robots'],trajectories_per_robot=160,runs=5760,frames=1728000,
        evaluation_git_sha='48b58546a26251225ac450294c4913efc6a16d7f',
        numerical_seal_sha256=old.sha(root/'protocol/selection_seal.json'),
        report_sha256=old.sha(doc),analysis_code={p:old.sha(old.ROOT/p) for p in paths},
        files={str(p.relative_to(root)):old.sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        old_tracked_files_changed=[],numerical_configuration_changed_after_outcomes=False,
        additional_solver_experiments_after_evaluation=0,stop='evaluate fixed results; no tuning, new module or paper rewrite'))


if __name__=='__main__':finalize_delivery()
