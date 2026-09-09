"""Generate the bounded development findings from completed source-data tables."""
import json
from . import study as old
from .elastic_study import configurations,FOCUS_UID
from .elastic_reporting import LABELS

MU_LABELS={'cr_ik_elastic_mu0':'0','cr_ik_elastic_mu025':'0.25','cr_ik_elastic_mu1':'1','cr_ik_elastic_mu4':'4'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    cfg,_,root=configurations();out=root/'reports'
    data=json.loads((out/'source_data.json').read_text())
    pre=json.loads((root/'protocol/preflight.json').read_text())
    audit=json.loads((out/'verification_manifest.json').read_text())
    def row(robot,method):return next(r for r in data['main_table'] if r['robot']==robot and r['method']==method)
    def pair(robot,method,baseline,metric):
        return next(r for r in data['paired_comparisons'] if r['robot']==robot and r['method']==method and r['baseline']==baseline and r['metric']==metric)
    def p(v):return '—' if v is None else f'{100*v:.2f}%'
    def number(v):return '—' if v is None else f'{v:.4f}'
    def trip(v):return '/'.join('—' if x is None else str(x) for x in v)
    def count(robot,method,decision):
        return sum(r['frames'] for r in data['decision_counts'] if r['robot']==robot and r['method']==method and r['decision']==decision)
    def reason(robot,method,reason):
        return sum(r['frames'] for r in data['fallback_counts'] if r['robot']==robot and r['method']==method and r['reason']==reason)
    positive=['cr_ik_elastic_mu025','cr_ik_elastic_mu1','cr_ik_elastic_mu4'];mu0='cr_ik_elastic_mu0'
    lines=['# CR-IK Elastic Demand：完整开发比较','',
        '**结论：允许有代价的缺口确实产生了合法的部分改善；在已观察UR5e开发轨迹上，μ=0.25/1将其转化为完成增量，'
        '但Panda没有相对μ=0的完成增量，μ=4也没有一致收益。成本仍接近硬需求Minimal，而非退回原CR-IK。**',
        '本轮保留全部μ结果，不据此自动选定设置、启动正式评估、增加模块或改写论文。','',
        '## 范围与实现','',
        '独立 `ElasticInterventionIK` 保留预测器、W=20需求历史、原r_min/启动值、TRAC备用解、两步结构、'
        '真实SO(3)导数和全部任务合同。仅增加xi的代价与相应实际目标验收；采用h=xi/r数值缩放。'
        '当前q和名义z的位姿、关节和单帧速度不松弛。最多两次单阶段锥求解，固定结构缓存，不重新最大化gamma。'
        '初始对满足完整需求时逐元素不变返回备用解。合法初始对的替换必须通过真实FK及原verifier，且实际目标下降'
        '超过 `1e-10 + 1e-9*abs(reference_objective)`；不是只看线性目标，也不要求demand_met。','',
        '初始对不合法或备用失败时，所有μ复用原同核普通双步可行性恢复；此类返回单独标为geometric_recovery，'
        '不作为修正目标的独立增量。秩不足时xi=r、六维余量为零，不作正余量声明。','',
        '数值代码、协议、输入和参数在运行前提交并push：`e26215f3b5cb8422b8d51dec1b5bb4e43cafa0d2`。'
        '两台机器人各40条既有开发轨迹、四类各10条、每条150帧。八种含TRAC配置各3遍，Pink一遍，'
        '共2000次整轨迹运行、300000帧。每种方法均从第0帧开始，失败保持上一接受状态，目标索引正常前进。'
        '未读取新正式数据，没有轨迹筛选、未来q_ref输入或失败前人工接入。','',
        '统计单位为完整trajectory UID，搜索重复嵌套；先在UID内平均，再作4000次按family分层的配对bootstrap。'
        '95%区间是描述性、未作多重比较校正的区间，不是非劣效性、等效性或全局gate。'
        '原生TRAC随机种子不可控，不声称相同种子。时延包含全部command-ready计算，失败和超时帧均保留。','']
    for robot in cfg['robots']:
        lines += [f'### {robot}：每遍40条的完整主表','',
            table(['方法','完成数','全帧≤20ms完成数','TSR / DTSR20','P50/P95/P99 ms','每40条累计s','加速度RMS rad/s²'],[
                [LABELS[m],trip(r['completion_by_repeat']),trip(r['deadline_completion_by_repeat']),f'{p(r["tsr"])} / {p(r["dtsr20"])}',
                 '/'.join(f'{r[k]:.3f}' for k in ('p50_ms','p95_ms','p99_ms')),f'{r["cumulative_ms_per_sweep"]/1000:.3f}',f'{r["acceleration_rms_mean"]:.3f}']
                for m in cfg['methods'] for r in [row(robot,m)]]),'']
    lines += ['## 1. 原硬需求是否拒绝过几何合法但仅部分改善的候选？','',
        '旧实现的规则会拒绝几何合法但demand_met=False的非恢复候选；然而旧日志没有保存这些被拒绝的trial q/z和实际目标值，'
        '所以**不能从旧日志补造“有用部分改善被拒绝”的精确数量**。以下仅为要求的只读定位，列之间可重叠：','',
        table(['机器人','非直接备用返回','初始对非法','秩不足','凸求解未成功','已解但未选：原因不可拆分','返回初始合法对但需求不足'],[
            [r['robot'],r['non_direct_backup_return'],r['initial_pair_illegal'],r['rank_condition_insufficient'],r['convex_unsuccessful'],
             r['solved_but_not_selected_nonlinear_or_demand_ambiguous'],r['returned_initial_pair_legal_demand_unmet']] for r in pre['counts']]),'',
        '非线性失败与需求不足不能从上述“已解但未选”中分开；preflight.json将不可辨识计数保留为null。'
        '这些是回退相关状态，不是互斥因果归类。本轮未重跑这些旧输入。','',
        '新的逐trial记录则能直接确认：已采用的部分候选q/z通过真实几何合同、实际目标下降、xi减少且仍大于零。'
        '这些**相同新候选若应用旧硬需求验收规则会被拒绝**；这不等于重现旧方法全部闭环反事实。','',
        table(['机器人','μ','部分改善采用帧/18000','采用率','初始合法对平均xi：前→后','平均归一化缺口下降'],[
            [robot,MU_LABELS[m],count(robot,m,'partial_effective_correction'),p(r['partial_correction_rate']),
             f'{number(r["initial_gap_mean"])} → {number(r["final_gap_mean"])}',number(r['normalized_gap_reduction_mean'])]
            for robot in cfg['robots'] for m in positive for r in [row(robot,m)]]),'',
        '表中的缺口均值以有合法初始对且有合法返回名义对的帧为分母，包含未调整帧，是描述性帧统计。'
        '保存的partial_improvement_examples.json仅是可检查例子；全部trial留在原始记录，不用几个例子替代全样本。','',
        '## 2. 新模式是否将部分改善转化为更好的完整续接？','',
        '有局部开发证据，但不是每台机器人、每个μ都提高。μ=0.25在Panda保持38/40的三遍完成，'
        '相对μ=0没有几何完成增量；UR5e的μ=0.25和μ=1均达到40/39/40或39/40/40，'
        '高于同核μ=0的39/39/38。μ=4没有相同的增量，增大惩罚不等于单调改善。','',
        '## 3. 相比同核μ=0，修正余量是否确有额外价值？','',
        table(['机器人','μ','ΔTSR vs μ=0，pp [95%区间]','ΔDTSR20，pp [95%区间]','累计时间比 vs μ=0'],[
            [robot,MU_LABELS[m],
             f'{100*a["difference"]:+.2f} [{100*a["difference_ci"][0]:+.2f}, {100*a["difference_ci"][1]:+.2f}]',
             f'{100*b["difference"]:+.2f} [{100*b["difference_ci"][0]:+.2f}, {100*b["difference_ci"][1]:+.2f}]',f'{c["ratio"]:.3f}']
            for robot in cfg['robots'] for m in positive for a in [pair(robot,m,mu0,'completion')]
            for b in [pair(robot,m,mu0,'deadline_completion')] for c in [pair(robot,m,mu0,'total_latency_ns')]]),'',
        '这一对照支持修正目标在部分UR5e轨迹上的增量，不支持普遍收益；区间、损失UID和μ=4结果应共同阅读。'
        '不同方法进入后续帧时的previous_q不同，不能将整轨迹差异改写成固定输入的不可行性证明。','',
        'μ=0保留同一预测、变量、硬约束、结构和预算，只将缺口惩罚置零。初始对合法时，它已经是零干预目标的全局最小，'
        '因此实际目标验收保持备用命令不变。为隔离目标，控制组仍沿用相同的“完整需求满足才直接返回”流程；'
        '其未改善帧可能耗尽两次更新，而正μ首个实际改善即可停止。**相对该控制的节省不是优于一个专门简化后的μ=0实现的证明。**'
        '因此成本仍以Hard Minimal、TRAC和Pink为实用参照。','',
        '## 4. 是否恢复trajectory_15，是否引入其他损失？','',f'UR5e完整UID：`{FOCUS_UID}`。','',
        '上一轮硬Minimal三遍均在125帧首次失败，previous_q第一关节接近有限上界。'
        '本轮各方法重新完整运行，原CR-IK与硬Minimal均有搜索波动：','',
        table(['方法','各遍是否完整完成','各遍首次失败帧'],[
            [LABELS[m],trip([int(s['completion']) for s in g]),trip([s['first_failure_frame'] for s in g])]
            for m in cfg['methods'] for g in [sorted([s for s in data['focus_trajectory_15'] if s['method']==m],key=lambda s:s['repeat'])]]),'',
        'μ=0.25和μ=1恢复为3/3；μ=4仅1/3，μ=0为0/3。该结果不单独证明奇异性、某个分支或shortcut是唯一原因。'
        '本轮原CR-IK的1/3与前一轮3/3均按各自原记录保留，不改写历史值。','',
        table(['机器人','μ','对照','轨迹UID（前12位）','类别','对照完成比例→新完成比例','变化'],[
            [r['robot'],MU_LABELS[r['method']],LABELS[r['baseline']],r['uid'][:12],r['family'],
             f'{r["baseline_completion_fraction"]:.3f} → {r["completion_fraction"]:.3f}',r['change']]
            for r in data['gained_lost_uids'] if r['method'] in positive and r['baseline'] in (mu0,'cr_ik_minimal') and r['change']!='same']),'',
        '完整UID、与全部基线的恢复/损失、逐遍完成集合及首次失败真实输入见gained_lost_uids.csv、completion_uids.json和first_failure_inputs.csv。'
        '恢复焦点轨迹不抵消其他轨迹的损失。','']
    lines += cost_and_accounting(data,cfg,row,pair,p,number,count,reason,positive,mu0)
    lines += ['## 交付与停止','',
        f'28项新旧测试通过。只读复核 {sum(r["frames"] for r in audit["results"]):,} 帧、'
        f'{sum(r["accepted_commands"] for r in audit["results"]):,} 个接受命令、'
        f'{sum(r["trial_geometry_checks"] for r in audit["results"]):,} 个trial及'
        f'{sum(r["strict_actual_improvements"] for r in audit["results"]):,} 个实际目标改善，未发现合同/反馈/目标值不一致。'
        '首次只读汇总遇到基线native_status格式差异；已修复报告解析器并保留reports_readonly_attempt_01。'
        '没有重跑IK，没有修改数值代码、原始结果或旧证据。','',
        '- 原始逐帧记录及trial：`elastic_demand_development/{panda,ur5e}/runs/`。\n'
        '- 主表、全部family表、配对区间、UID、失败、阶段计时与完整成功witness：`reports/`。\n'
        '- 配置、输入、需求参数、旧日志定位、测试与哈希：`protocol/`。\n'
        '- 数值入口：`scripts/run_crik_elastic.sh`；拒绝覆盖现有完整结果。\n'
        '- source-data：`reports/source_data.json`；本报告由elastic_findings.py生成。','',
        '结论边界：40个完整轨迹UID/机器人是统计单位，3次搜索不是3倍独立样本；这是已观察开发数据。'
        'xi/gamma仅描述局部模型修正能力，不是非线性全局保证；20ms是测得的评价deadline，不是硬实时证明。'
        '目前存在值得保留的UR5e开发增量，但还不能据此宣称统一优于成熟求解器。全部μ同时冻结，不自动选型、'
        '新增正式评估、改论文或继续开发。','']
    path=old.ROOT/'docs/CRIK_ELASTIC_DEMAND_FINDINGS.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    old.write_json(root/'findings_manifest.json',dict(utc=old.utc(),report=str(path.relative_to(old.ROOT)),report_sha256=old.sha(path),
        generator_sha256=old.sha(__file__),sources={str(p):old.sha(p) for p in (out/'source_data.json',out/'verification_manifest.json',root/'protocol/preflight.json')}))


def cost_and_accounting(data,cfg,row,pair,p,number,count,reason,positive,mu0):
    lines=['## 5. 成本是否接近Minimal，而非回到昂贵原版？','',
        table(['机器人','μ','累计成本/Hard Minimal [95%区间]','/原CR-IK','/TRAC5','/Pink','加速度相对Hard Minimal变化'],[
            [robot,MU_LABELS[m],f'{a["ratio"]:.3f} [{a["ratio_ci"][0]:.3f}, {a["ratio_ci"][1]:.3f}]',
             f'{row(robot,m)["cumulative_ms_per_sweep"]/row(robot,"cr_ik")["cumulative_ms_per_sweep"]:.3f}',
             f'{row(robot,m)["cumulative_ms_per_sweep"]/row(robot,"trac_task_5ms")["cumulative_ms_per_sweep"]:.3f}',
             f'{row(robot,m)["cumulative_ms_per_sweep"]/row(robot,"pink_qp")["cumulative_ms_per_sweep"]:.3f}',
             f'{100*(row(robot,m)["acceleration_rms_mean"]/row(robot,"cr_ik_minimal")["acceleration_rms_mean"]-1):+.2f}%']
            for robot in cfg['robots'] for m in positive for a in [pair(robot,m,'cr_ik_minimal','total_latency_ns')]]),'',
        '正μ配置的累计成本均值保持在Hard Minimal的1.25倍以内，没有回到原CR-IK的成本级别。'
        '但仍明显慢于TRAC，且未普遍优于Pink；这个描述性目标不是总gate或置信区间上界要求。'
        'μ=0.25未靠相对Hard Minimal增大的加速度RMS换取收益，μ=1在UR5e则有运动波动增加。'
        'UR5e所有新模式的加速度仍高于原CR-IK，原版数值也保留在主表中。','',
        table(['机器人','模式','优化调用率','部分采用率','接受命令改变率','平均归一化干预L2','已接受误差>90%容差','P95位置mm / 姿态deg'],[
            [robot,LABELS[m],p(r['optimization_call_rate']),p(r['partial_correction_rate']),p(r['effective_command_adjustment_rate']),
             number(r['intervention_normalized_mean']),p(r['accepted_near_tolerance_rate']),
             f'{1000*r["p95_position_m"]:.4f} / {r["p95_orientation_rad"]*180/3.141592653589793:.4f}']
            for robot in cfg['robots'] for m in [mu0]+positive for r in [row(robot,m)]]),'',
        '“接受命令改变率”严格表示当前被接受q与合法备用q逐元素不同；干预幅度另报，不能仅凭浮点不相等宣称有物理意义的改善。'
        '返回误差均经过原合同，较大但合法的误差没有隐藏；P95、最大残差和速度利用率完整保留在主表CSV。','',
        '### 决策与回退分类','',
        table(['机器人','μ','完整需求满足','部分有效修正','几何恢复','无改善备用','没有合法命令','求解未成功回退','非线性拒绝回退','实际目标未改善回退'],[
            [robot,MU_LABELS[m]]+[count(robot,m,d) for d in ('full_demand_satisfied','partial_effective_correction','geometric_recovery','backup_without_improvement','no_legal_command')]
             +[reason(robot,m,d) for d in ('convex_solve_unsuccessful','nonlinear_pair_rejected','no_actual_objective_improvement')]
            for robot in cfg['robots'] for m in [mu0]+positive]),'',
        '前五列是运行决策，后三列是未采用新对的回退原因，不应相加。外层软限时另在fallback_counts.csv保留。'
        '“求解未成功”不表示数学不可行；“几何恢复”不要求完整需求，不被标成余量保证。','',
        '### 全部四类开发轨迹','',
        '以下每格为每遍平均完成数/10与每条轨迹平均累计毫秒；全部九方法的完整family指标见family_table.csv。','',
        table(['机器人','类别','Hard Minimal','μ=0','μ=0.25','μ=1','μ=4'],[
            [robot,family]+[f'{r["mean_completed_trajectories"]:.2f}; {r["trajectory_time_mean_ms"]:.1f}ms'
                for m in ['cr_ik_minimal',mu0]+positive for r in data['family_table'] if r['robot']==robot and r['family']==family and r['method']==m]
            for robot in cfg['robots'] for family in ('smooth','near_singular','joint_limit_return','high_curvature')]),'']
    return lines


if __name__=='__main__':main()
