"""Write the five-question conclusion from completed frozen tables, without solves."""
from collections import Counter
import json
import numpy as np
from .study import Study,now
from .reporting import read,table_text,LABELS
from ..revision_compute_allocation.common import json_write,csv_write,digest


def interval(x,scale=1.):
    return f'{x[0]*scale:.3f} [{x[1]*scale:.3f}, {x[2]*scale:.3f}]'


def main():
    s=Study();p=[r for r in read(s,'point_main') if r['witness_feasible']]
    t=read(s,'trajectory_main');sens=read(s,'sensitivity_main');paired=read(s,'point_paired')
    tp=read(s,'trajectory_paired');traces=read(s,'dls_excess_iterations')
    verification=read(s,'verification')
    key={}
    for robot in ['panda','ur5e']:
        pp={r['method']:r for r in p if r['robot']==robot}
        tt={r['method']:r for r in t if r['robot']==robot}
        pt={r['method']:r for r in paired if r['robot']==robot and r['witness_feasible']}
        tr={r['method']:r for r in tp if r['robot']==robot and 'family' not in r}
        obs=[r for r in traces if r['robot']==robot and r['method']=='dls_strict' and r['first_admissible_iteration'] is not None]
        key[robot]=dict(point=pp,trajectory=tt,point_paired=pt,trajectory_paired=tr,
            dls_observation=dict(admissible_queries=len(obs),positive_excess_queries=sum(r['excess_iterations_after_task_admissibility']>0 for r in obs),
                median_excess_iterations=float(np.median([r['excess_iterations_after_task_admissibility'] for r in obs])),
                p95_excess_iterations=float(np.quantile([r['excess_iterations_after_task_admissibility'] for r in obs],.95)),
                median_excess_evaluations=float(np.median([r['excess_fk_residual_evaluations'] for r in obs])),
                p95_excess_evaluations=float(np.quantile([r['excess_fk_residual_evaluations'] for r in obs],.95))))
    json_write(s.out/'05_aggregate/key_findings.json',key)
    lines=['# Task-contract alignment：最终实验结论',
        '本报告只回答预定的五个问题。Panda 历史轨迹数据直接读取；UR5e 和点查询的新测量已经结束。没有全局通过门槛，也不追加实验。数字由 `confik.task_contract_alignment.findings` 读取冻结汇总自动写出。',
        '## 1. Solver-internal success 与 task-level admissibility 的失配有多常见？',
        '失配明显依赖 solver，而不是在所有配置中普遍存在。每台机器人有 2,500 个具有同输入合法 witness 的点查询；TRAC 的三次搜索是 query 内重复，DLS 只运行一次。以下矩阵是原始调用计数，不能将三次搜索当成三个独立查询。',
        '| Robot | Setting | Internal+/Task+ | Internal+/Task− | Internal−/Task+ | Internal−/Task− | 任一重复漏解 UID |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(p,key=lambda r:(r['robot'],list(LABELS).index(r['method']))):
        lines.append(f'| {r["robot"]} | {LABELS[r["method"]]} | {r["internal_success__task_accept"]} | {r["internal_success__task_reject"]} | {r["internal_failure__task_accept"]} | {r["internal_failure__task_reject"]} | {r["any_repeat_missed_uids"]} |')
    for robot,k in key.items():
        a=k['point']['dls_strict'];o=k['dls_observation']
        lines.append(f'{robot}: strict DLS 的 {a["native_failure_task_accept"]}/2500 ({a["native_failure_task_accept"]/25:.2f}%) 个查询内部未收敛，但返回命令通过公共合同。{o["admissible_queries"]} 个查询出现合法主迭代状态，其中 {o["positive_excess_queries"]} 个随后仍继续迭代；额外迭代 median/P95={o["median_excess_iterations"]:.0f}/{o["p95_excess_iterations"]:.0f}，额外 FK/residual evaluations median/P95={o["median_excess_evaluations"]:.0f}/{o["p95_excess_evaluations"]:.0f}。术语为 excess iterations after task admissibility；不推测 TRAC 未暴露的内部迭代，也不估算额外阶段时间。')
    lines+=['strict 与 fully aligned TRAC 在两个机器人的全部 nominal 可行点查询上均成功，因此其主比较中不存在 witness-confirmed miss 可供恢复。UR5e orientation-only 有一个 UID 的一次搜索失败；这是同输入 witness-confirmed miss，不是不可行证明。DLS 的实际漏解数量并未因 task stop 而减少，已在矩阵中完整保留。',
        f'独立回放检查了 {sum(verification["counts"].values()):,} 条新旧调用记录，accepted contract violations={verification["accepted_contract_violations"]}，{verification["protected_files_unchanged"]} 个旧受保护文件未变化。成功完整轨迹 witness 索引包含 {verification["successful_trajectory_witnesses"]} 个实际运行序列。所有点查询 witness 位于固定源 NPZ，其 UID/hash 链接和返回配置均保存在 raw records 中。',
        '500 个明确构造不可执行查询/robot 单列，不混入以上可行点指标：所有方法均 public rejection，全部 internal failure，false acceptance=0。TRAC 5/20 ms 消耗接近各自搜索预算；合同对齐没有避开真正不可执行输入的搜索成本。见 `point_main.csv` 中 witness_feasible=False 的完整时延。',
        '## 2. Position tolerance 与 orientation tolerance 分别贡献了多少恢复？',
        'nominal 点查询主比较的 strict 已全成功，所以 position-only、orientation-only、fully aligned 的恢复数量均为零；不能从这一 ceiling workload 归因轨迹恢复由哪种容差驱动。orientation-only 在 UR5e 反而出现上文的一次搜索损失。分量对齐的计算变化如下，ratio 为方法/strict 5 ms，区间按 query UID 配对分层重采样：',
        '| Robot | Setting | Success difference pp [95% CI] | Mean cost ratio [95% CI] |',
        '|---|---|---|---|']
    for robot,k in key.items():
        for m in ['trac_position_5ms','trac_orientation_5ms','trac_task_5ms']:
            r=k['point_paired'][m];lines.append(f'| {robot} | {LABELS[m]} | {interval(r["success_difference"],100)} | {interval(r["latency_ratio"])} |')
    lines+=['单独对齐一种误差并不保证降低开销。fully aligned 的联合映射降低了这里的平均开销，但不等于两种单独作用可线性相加。5→20 ms 的 strict 配置在可行点上没有增加成功，也没有显示需要更大预算。',
        '## 3. 对齐是否在两机器人上提高 verified success 或降低计算成本？',
        '点查询：TRAC fully aligned 5 ms 保持全部成功并小幅减少平均时间；DLS task stop 保持与 strict 相同的 verified success，大幅减少计算。',table_text(p),
        'DLS 点查询两臂均包含相同的轻量 trace 记录成本；轨迹使用与历史 Panda 完全相同、未开启 trace 的 wrapper。DLS 的 25 iterations 不是与 TRAC 等价的 5/20 ms 时间预算。所有耗时包含必要转换、动态边界、求解和最终 verifier，不含离线序列化、统计和 trace 回放。',
        '完整轨迹：以下 completion 为每次完整 sweep 的条数（每 sweep 40 条），累计时延是三次 TRAC sweep 的平均，不是三倍样本量。所有 150 帧均保留，失败只保持自身 previous_q，目标继续推进。',table_text(t,True),
        '| Robot | Aligned / strict | Completion difference pp [95% CI] | Cumulative cost ratio [95% CI] | Gained / lost UID |',
        '|---|---|---|---|---|']
    for robot,k in key.items():
        for m in ['trac_task_5ms','trac_task_20ms','dls_task']:
            r=k['trajectory_paired'][m];lines.append(f'| {robot} | {LABELS[m]} | {interval(r["success_difference"],100)} | {interval(r["latency_ratio"])} | {len(r["gained_uids"])}/{len(r["lost_uids"])} |')
    for robot,k in key.items():
        r=k['trajectory_paired']['trac_task_5ms'];ds=k['trajectory']['dls_strict'];da=k['trajectory']['dls_task']
        lines.append(f'{robot}: fully aligned TRAC 5 ms 累计时延降低 {100*(1-r["latency_ratio"][0]):.2f}%。DLS 从 {ds["completion_counts"][0]}/40 到 {da["completion_counts"][0]}/40，累计时间降低 {100*(1-k["trajectory_paired"]["dls_task"]["latency_ratio"][0]):.2f}%。')
    lines+=['重要负结果：UR5e DLS task stop 丢失 7 条 strict 已完成轨迹（smooth 3、near-singular 2、joint-limit-return 2）。节省时间不能抵消这一完成损失。当前结果证明接受同一位姿合同下不同的数值停止过程可以产生不同的闭环路径，但不能仅凭这些结果证明具体的分支漂移因果，或保证每个失败输入从自身 previous_q 仍存在合法下一步。',
        'TRAC 5 ms 在 Panda 的恢复集中于 near-singular，UR5e 的恢复位于 smooth/high-curvature。UR5e near-singular 没有增加整轨迹完成；joint-limit-return 平均耗时略升。所有 family 结果保留，不按显著性删选。',
        '| Robot | Family | Strict→aligned completion /10 | Cumulative ratio |',
        '|---|---|---|---:|']
    families=read(s,'trajectory_families')
    for r in tp:
        if r['method']=='trac_task_5ms' and 'family' in r:
            pair={q['method']:q for q in families if q['robot']==r['robot'] and q['family']==r['family']}
            a=pair['trac_task_5ms'];b=pair['trac_strict_5ms']
            lines.append(f'| {r["robot"]} | {r["family"]} | {b["completion_counts"]} → {a["completion_counts"]} | {r["latency_ratio"][0]:.3f} |')
    lines+=['截止时间结局也单独保留；几何合法但迟到的帧仍按预定几何反馈规则更新，不把这一软件流程解释为 deadline-triggered controller 或硬实时控制。',
        '| Robot | Setting | All-frame ≤20 ms completion /40 | Frame verified % | Frame accepted ≤20 ms % |',
        '|---|---|---|---:|---:|']
    for r in t:lines.append(f'| {r["robot"]} | {LABELS[r["method"]]} | {r["deadline_completion_counts"]} | {100*r["verified_success"]:.3f} | {100*r["accepted_within_20ms"]:.3f} |')
    lines+=['合法误差确实变大，不能隐藏。以下为 accepted P95，完整 median/P95/P99/max 分布见 main JSON：',
        '| Population | Robot | Setting | Position P95 mm | Orientation P95 deg |',
        '|---|---|---|---:|---:|']
    for pop,rs in [('point',p),('trajectory',t)]:
        for r in rs:
            if r['method'] in ['trac_strict_5ms','trac_task_5ms','dls_strict','dls_task']:
                lines.append(f'| {pop} | {r["robot"]} | {LABELS[r["method"]]} | {r["accepted_position"]["p95"]*1000:.5f} | {np.rad2deg(r["accepted_orientation"]["p95"]):.5f} |')
    lines+=['## 4. 0.5×、1×、2× 合同下是否呈现可解释变化？',
        '三个尺度采用相同的 500 个已在最严格尺度验证 witness 的查询/robot。TRAC strict 和 fully aligned 在每个尺度都全部成功，不能声称随尺度恢复更多 TRAC 漏解。Panda DLS 在 0.5× 漏 2 个，到 1×/2× 均成功；UR5e DLS 三尺度均成功。',
        '| Robot | Scale | Setting | Verified % | P50/P95/P99 ms | Mean ms | Accepted position P95 / tolerance |',
        '|---|---:|---|---:|---|---:|---:|']
    for r in sens:
        lines.append(f'| {r["robot"]} | {r["scale"]} | {LABELS[r["method"]]} | {100*r["verified_success"]:.2f} | {r["latency_p50_ms"]:.3f}/{r["latency_p95_ms"]:.3f}/{r["latency_p99_ms"]:.3f} | {r["latency_mean_ms"]:.3f} | {r["accepted_position"]["p95"]/(.001*r["scale"]):.3f} |')
    lines+=['合同放宽时 fully aligned TRAC 的相对平均开销收益总体增大，DLS task 的平均时间下降；strict 内部停止条件不随尺度变化。单个分位点并非严格单调，因此结论是可解释的总体计算趋势，不是每条查询的单调改进定理。nominal 始终是主合同。敏感性改变的是明确列出的任务要求，不能将更宽公共合同的成功直接当作同合同算法收益。',
        '## 5. 合同对齐之后，learned adaptive computation 还剩什么实际价值？',
        '本轮不训练、不运行路由器；这里是旧 allocation 证据的只读边界分析，不是将路由器重新接到 aligned TRAC/DLS 后的新增直接比较。已有 witness-feasible points 中 Full 与 hard 成功相同，但平均耗时增加；routing-only 无稳定净收益，geometry rule 与 full 接近，P95 selection 的独立尾延迟优势较弱。原轨迹总体节省主要来自共同失败组，reject 能在失败密集负载减少徒劳计算。旧 external TRAC 比较具有明显速度优势，但旧浮点端点拒绝是实现问题，原记录与说明均保留，不计入新对齐恢复。详见 `allocation_boundary.json`、原 `REVISION_EXPERIMENT_FINDINGS.md` 和原 `REVISION_MEASUREMENT_NOTES.md`。',
        '因此值得保留的是明确的失败成本管理，而非预设需要学习入口选择。当前证据支持：Align first; allocate only when residual failure cost justifies the allocation overhead。这不是证明任何 learned allocation 在 aligned solver 上必然无效。阶段一至此结束，不因 UR5e DLS 负结果调整容差、预算、数据或增加方法。',
        '交付索引：`01_protocol/` 固定身份/映射/环境；`02_point_study/`、`03_contract_sensitivity/`、`04_trajectory_study/` 原始记录；`05_aggregate/` CSV/JSON、completion UID、first-failure、witness index、paired CI；`reports/` 五张主图和 DLS 补图的 PDF/SVG/PNG、主表。全部输入转换/边界/solve/verifier 的时间分项留在点查询 raw records。',
        '只读汇总曾因 UR5e first-failure 行已有 robot 字段而发生重复关键字异常；修复仅为字典合并，原 execution seal 保留，已写出的点/敏感性表未覆盖，未重跑任何 solver。`analysis_revision.json` 保存逐字节允许变更与哈希。该报告生成在实验全部结束、论文正文开始修改之前。']
    document='\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n'
    with (s.root/'docs/TASK_CONTRACT_ALIGNMENT_FINDINGS.md').open('x') as f:f.write(document)
    json_write(s.out/'reports/findings_provenance.json',dict(utc=now(),
        report_sha256=digest(s.root/'docs/TASK_CONTRACT_ALIGNMENT_FINDINGS.md'),
        sources={str(p.relative_to(s.root)):digest(p) for p in (s.out/'05_aggregate').glob('*.json')}))


if __name__=='__main__':main()
