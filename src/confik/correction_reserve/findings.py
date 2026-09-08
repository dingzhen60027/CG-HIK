"""Generate the new algorithm report from finished source-data, not hand numbers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .study import ROOT,sha,write_json,utc
from .reporting import LABELS
from .figures import ORDER,FOCUS,FAMILIES,FAMILY_LABELS,ROBOT_LABEL


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def nums(values):return '/'.join(str(v) for v in values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/correction_reserve_ik')
    p.add_argument('--out',default='docs/CRIK_FINDINGS.md');p.add_argument('--conclusion',required=True)
    a=p.parse_args();root=Path(a.root)
    data=json.loads((root/'formal_reports/source_data.json').read_text())
    development=json.loads((root/'development_reports/source_data.json').read_text())
    audit=json.loads((root/'formal_verification/verification_manifest.json').read_text())
    mechanism=json.loads((root/'formal_mechanism/summaries.json').read_text())
    contrasts=json.loads((root/'formal_mechanism_comparison/source_data.json').read_text())
    with (root/'formal_verification/timing_phases.csv').open(newline='') as f:
        timings=list(csv.DictReader(f))
    with (root/'formal_verification/abruptness_table.csv').open(newline='') as f:
        abruptness=list(csv.DictReader(f))
    report_manifest=json.loads((root/'formal_reports/manifest.json').read_text())
    featured_directions=[r for r in contrasts['directions']
        if r['site_id']=='panda_trajectory_079_f80' and r['amplitude']==4 and r['direction_id'] in (0,1)]
    assert len(featured_directions)==2
    featured_text='；'.join(f'{"负" if r["direction_id"]==0 else "正"} x：普通双步 '
        f'{r["predictive_accepted"]}/{r["searches"]}，CR-IK {r["crik_accepted"]}/{r["searches"]}'
        for r in featured_directions)
    lines=['# Correction-Reserve IK: fixed two-robot evaluation','',a.conclusion,'',
        '## 1. 实现、范围和实验单位','',
        '已实现能够从第 0 帧闭环运行的 CR-IK；不是只改变返回码或路由。当前 q、名义下一 z 和 γ 联合优化，'
        '最多三次序列凸化，使用已观察目标作固定外推。当前命令与名义下一配置由原 verifier 分别验收，'
        '没有真实未来目标、参考 q、候选池、训练网络或周期锚定。旧论文、旧 solver/verifier 和旧证据未改。','',
        '参数和新轨迹身份在代码提交 `e13f32f53a451eb19879292dbfa8e382edb4bc39` 中固定并 push，随后执行正式比较。'
        '每机器人 80 条新轨迹，四类各 20 条、每条 300 帧；dt=20 ms，合同仍为 1 mm / 0.5° 和原关节/速度界限。'
        'TRAC-IK 及包含它的方法各 3 次完整搜索重复；Pink 和 RangedIK 各 1 次。统计单位是 trajectory UID，'
        '重复先在 UID 内平均。区间为按 family 分层的配对 trajectory bootstrap（4000 次、描述性 95% 区间），'
        '不把帧或搜索重复当独立轨迹，不作多重比较后的显著性宣称。','',
        f'正式主比较共 {report_manifest["complete_trajectory_runs"]} 个完整 trajectory runs、'
        f'{report_manifest["frame_calls"]:,} 次在线帧调用。所有失败与超时均保留。','',
        '## 2. 完整轨迹结果','',
        '完成数和按时完成数按各次搜索分别列出；分位数为全部帧的外层时延，包括失败。'
        'DTSR20 要求同一整条轨迹每帧既通过合同又在 20 ms 内返回。','']
    for robot in ('panda','ur5e'):
        rows=[]
        for method in ORDER:
            r=next(x for x in data['main'] if x['robot']==robot and x['method']==method)
            rows.append([LABELS[method],nums(r['completion_by_repeat']),nums(r['deadline_completion_by_repeat']),
                f'{100*r["tsr"]:.2f}%',f'{100*r["dtsr20"]:.2f}%',
                f'{r["p50_ms"]:.3f}/{r["p95_ms"]:.3f}/{r["p99_ms"]:.3f}',
                f'{r["cumulative_ms_per_sweep"]/1000:.3f}'])
        lines += [f'### {ROBOT_LABEL[robot]}（每遍 80 条）','',table(
            ['方法','完成数','全帧按时完成数','TSR','DTSR20','P50/P95/P99 ms','80 条累计时间 s（重复平均）'],rows),'']
    lines += ['### 已观察开发集（不作为 fresh 证据）','',
        '开发集每机器人 40 条、每条 150 帧。完整记录同样保留；尤其 UR5e 的开发结果没有表现出'
        '相对普通双步的优势，不能只展示 fresh 结果而把这一点抹去。','',
        table(['机器人','TRAC 5 ms 完成数','普通双步完成数','CR-IK 完成数'],[
            [ROBOT_LABEL[robot]]+[nums(next(r['completion_by_repeat'] for r in development['main']
                if r['robot']==robot and r['method']==method))
                for method in ('trac_task_5ms','two_step_predictive','cr_ik')]
            for robot in ('panda','ur5e')]),'',
        '## 3. 相对成熟 TRAC-IK 与同核预测的增量','']
    comparison=[]
    for robot in ('panda','ur5e'):
        for baseline in ('trac_task_5ms','two_step_predictive'):
            completion=next(r for r in data['pairs'] if r['robot']==robot and r['baseline']==baseline and r['metric']=='completion')
            deadline=next(r for r in data['pairs'] if r['robot']==robot and r['baseline']==baseline and r['metric']=='deadline_completion')
            cost=next(r for r in data['pairs'] if r['robot']==robot and r['baseline']==baseline and r['metric']=='total_latency_ns')
            changes=[r for r in data['changes'] if r['robot']==robot and r['baseline']==baseline]
            comparison.append([ROBOT_LABEL[robot],LABELS[baseline],
                f'{100*completion["difference"]:+.2f} [{100*completion["difference_ci"][0]:+.2f}, {100*completion["difference_ci"][1]:+.2f}]',
                f'{100*deadline["difference"]:+.2f} [{100*deadline["difference_ci"][0]:+.2f}, {100*deadline["difference_ci"][1]:+.2f}]',
                f'{cost["ratio"]:.3f} [{cost["ratio_ci"][0]:.3f}, {cost["ratio_ci"][1]:.3f}]',
                f'{sum(r["change"]=="gained" for r in changes)}/{sum(r["change"]=="lost" for r in changes)}',
                f'{sum(r["change"]=="gained" and r["stable_all_vs_none"] for r in changes)}/'
                f'{sum(r["change"]=="lost" and r["stable_all_vs_none"] for r in changes)}'])
    lines += [table(['机器人','比较对象','TSR 差 pp [95% CI]','DTSR20 差 pp [95% CI]',
        '累计时延比 [95% CI]','恢复/损失 UID（重复均值）','全重复稳定恢复/损失'],comparison),'',
        '恢复/损失表示该 UID 在三次搜索中的完成比例增减，不等于有相同随机种子的配对实验。'
        '“全重复稳定”限定为 3/3 对 0/3。完整 UID 列表在 `formal_reports/gained_lost_uids.csv`，'
        '每遍完成集合在 `completion_uids.json`。累计成本比不用于掩盖 TSR 或 DTSR20。','',
        '相对普通双步，两个机器人的平均 TSR 均提高；UR5e 的描述性配对区间未跨零，Panda 跨零。'
        '这是修正余量目标的正向但尚不对称的增量证据，不能等同于已证明普遍优势。'
        'Panda 的 Pink 在 TSR/DTSR20 上略高于 CR-IK，UR5e 的 TRAC-IK 20 ms 也略高，且两者累计成本明显更低。'
        '因此“胜过成熟方法且值得额外计算”尚未建立；低于 20 ms 的大多数调用也不意味着没有整轨迹时限损失。','',
        '## 4. 四类轨迹与消融','']
    rows=[]
    for robot in ('panda','ur5e'):
        for family,label in zip(FAMILIES,FAMILY_LABELS):
            group={r['method']:r for r in data['family'] if r['robot']==robot and r['family']==family}
            rows.append([ROBOT_LABEL[robot],label]+[
                f'{group[m]["mean_completed_trajectories"]:.2f}/20; {group[m]["trajectory_time_mean_ms"]:.1f} ms'
                for m in ('trac_task_5ms','two_step_predictive','cr_ik','single_step_reserve','two_step_sigma')])
    lines += [table(['机器人','类别','TRAC 5','普通双步','CR-IK','单帧余量','双步 sigma'],rows),'',
        '每格为平均完成数及每条轨迹累计时间；不是只统计成功轨迹。单帧、普通双步与 sigma 消融'
        '使用相同接受条件；单帧模式不优化预测配置，普通双步不最大化余量，sigma 模式用经典缩放几何 Jacobian。','',
        '## 5. 实际误差、运动和备用解','']
    rows=[]
    for robot in ('panda','ur5e'):
        for method in FOCUS:
            r=next(x for x in data['main'] if x['robot']==robot and x['method']==method)
            rows.append([ROBOT_LABEL[robot],LABELS[method],
                f'{r["p95_position_m"]*1000:.4f}/{r["max_position_m"]*1000:.4f}',
                f'{np.rad2deg(r["p95_orientation_rad"]):.4f}/{np.rad2deg(r["max_orientation_rad"]):.4f}',
                f'{r["max_accepted_step_utilization"]:.6f}',f'{r["acceleration_rms_mean"]:.3f}',
                f'{100*r["backup_rate"]:.2f}%',f'{100*r["changed_from_backup_rate"]:.2f}%'])
    lines += [table(['机器人','方法','位置 P95/max mm','姿态 P95/max °','最大已接受步长利用率',
        '加速度 RMS rad/s²（轨迹均值）','备用解率','相对备用解改变命令率'],rows),'',
        '误差只在真正接受的命令上统计，拒绝与整体完成率另报。关节运动由实际接受/保持状态计算；'
        '失败后保持不会被记为完成。CR-IK 较大的合法残差是主动使用容差的一部分，不能隐藏或称为更高精度。','',
        table(['机器人','方法','所有调用 ≤20 ms','超过 20 ms 次数','最大外层时延 ms'],[
            [ROBOT_LABEL[r['robot']],LABELS[r['method']],
             f'{100*float(r["all_call_returned_within_20ms"]):.4f}%',
             round((1-float(r['all_call_returned_within_20ms']))*int(r['calls'])),
             f'{float(r["max_ms"]):.3f}'] for r in timings
            if r['method'] in FOCUS and r['phase']=='total_latency_ns']),'',
        '## 6. 固定扰动验证：γ 与真实能力','',
        '正式轨迹使用预先提交的 first-UID-per-family 与 frame 40/80/110 规则，得到每机器人 12 个状态、'
        '4 条轨迹。所有当前候选共享实际 previous_q、当前及历史目标；不按续接效果挑状态。'
        '当前返回接受后，下一目标按 ±6 个笛卡尔轴和 0/0.5/1/2/4 倍容差扰动，以共同 TRAC-IK 20 ms '
        '续接，各 3 次搜索。未找到不是不可行证明。','']
    rows=[]
    for robot in ('panda','ur5e'):
        for method in ORDER:
            group=[r for r in mechanism if r['robot']==robot and r['method']==method]
            gamma=[r['predicted_gamma'] for r in group if r['predicted_gamma'] is not None]
            conditional=[r['probe_success'] for r in group if r['probe_success'] is not None]
            all_input=[r['probe_success'] if r['probe_success'] is not None else 0. for r in group]
            rows.append([ROBOT_LABEL[robot],LABELS[method],f'{sum(r["current_accepted"] for r in group)}/{len(group)}',
                f'{np.mean(gamma):.3f}' if gamma else '不可用',
                f'{100*np.mean(conditional):.2f}%' if conditional else '不可用',f'{100*np.mean(all_input):.2f}%'])
    lines += [table(['机器人','方法','当前合法状态','平均 γ（有合法名义下一步）','条件扰动成功率','含当前失败的成功率'],rows),'',
        '主表、原始两步命令 witness 与所有不可用状态一并保留。γ 是一个指定右逆修正策略的局部线性量，'
        '不是非线性全方向保证；测得扰动范围还受搜索能力、固定离散尺度和 4 倍容差上限影响。'
        '相邻探针、方向与重复不是新的独立轨迹。已观察开发集上的同一实验也保留在 `development_mechanism/`。','',
        '### 相对同核预测的逐状态差异','',
        '下表保留测得全方向离散范围不同的所有状态；全部状态（含不可用与无差异）以及全部方向计数另存'
        ' `formal_mechanism_comparison/`。这些是固定样本内的描述性实例，不是按结果增加的新测试。','',
        table(['状态','预测基线 γ → CR-IK γ','测得范围：预测 → CR-IK','扰动成功比例：预测 → CR-IK'],[
            [r['site_id'],f'{r["predictive_predicted_gamma"]:.4f} → {r["crik_predicted_gamma"]:.4f}',
             f'{r["predictive_all_directions_radius"]:g} → {r["crik_all_directions_radius"]:g}',
             f'{100*r["predictive_probe_success"]:.2f}% → {100*r["crik_probe_success"]:.2f}%']
            for r in contrasts['sites'] if r['all_directions_radius_difference'] not in (None,0)]),'',
        'Panda `trajectory_079_f80` 提供了清楚的同输入实例：在 4 倍容差的正、负 x 方向，'
        f'找到通过原合同的下一命令次数为 {featured_text}。'
        '原始记录同时保存共同 previous_q、各自实际当前 q 和全部下一命令，未借用其他构型的历史。'
        '同一轨迹 frame 110 的范围改善只涉及一次搜索差异，而且 γ 反而略降；UR5e 的范围变化也只涉及一次搜索差异。'
        '因此少量真实恢复为该目标提供有限机制支持，但总体扰动成功比例接近，且 γ 增量不等于实测能力的单调增量。'
        '这些扰动以外推目标为中心；真实下一目标仅用于离线计算预测误差，从未提供给在线算法。','',
        '## 7. 预测失准与失败行为','',
        '四类中 high-curvature 预置四种速度/反向突变等级，每机器人每等级 5 条，几何限幅不看 solver outcome。'
        '`formal_verification/abruptness_table.csv` 保留所有等级的 TSR/DTSR20 和累计成本；'
        '`forecast_error_per_trajectory.csv` 逐轨迹报告预测误差（按任务容差归一化），'
        '`first_failure_inputs.csv` 保存首次失败的真实 previous_q、目标、返回码、残差和速度利用率。'
        '代表时间序列按 first UID 选取，不因哪个方法成功而替换。','',
        table(['机器人','预定突变等级（每级 5 条）','TRAC 5 TSR / DTSR20','普通双步 TSR / DTSR20','CR-IK TSR / DTSR20'],[
            [ROBOT_LABEL[robot],level]+[
                f'{100*float(r["tsr"]):.1f}% / {100*float(r["dtsr20"]):.1f}%'
                for method in FOCUS for r in abruptness
                if r['robot']==robot and r['method']==method and int(r['abruptness_level'])==level]
            for robot in ('panda','ur5e') for level in range(4)]),'',
        '各等级是不同的固定轨迹，不是对同一轨迹只改变外推误差的随机干预；不能把等级间波动解释为单独的因果效应。'
        '全样本、所有类别和计算成本仍为主结果。','',
        '## 8. 复核、时限和已有方法边界','']
    checked=sum(r['frames'] for r in audit['results']);accepted=sum(r['accepted'] for r in audit['results'])
    nominal=sum(r['nominal_pairs_checked'] for r in audit['results'])
    lines += [f'只读 FK 复核检查了 {checked:,} 个记录帧，其中 {accepted:,} 个接受命令、{nominal:,} 个名义下一配置；'
        '验收、状态反馈和 deadline 标记未发现不一致，旧已跟踪文件未修改。该复核不运行 IK 搜索。'
        '两个明确的 CR-IK 全轨迹成功 witness 保存在 `formal_verification/`。','',
        '外层时间包含输入转换、备用求解、预测、Jacobian、凸问题构造/求解、选择及最终验收；'
        '记录序列化不在命令就绪时间内。18 ms 是停止新增优化工作的软界限，不是 20 ms 最坏执行时间证明。'
        '完整命令时延和时限失败照实保留，内部 Clarabel 耗时不能替代外层时延。','',
        'RangedIK 已利用目标范围，已有预测 IK 已使用视域，Pink 已处理 QP 与速度/关节约束。'
        '本实现不把这些作为原创。原始 RangedIK 在本任务小容差下不激活 ranged loss；正范围版显式修改两处判断，'
        '两版均去除共同合同之外的碰撞目标，其他默认权重与目标不变。开发及正式集中两者整条完成率均为零，'
        '因此它们是透明的接口/工作负载参照，而不是已经充分调优的强 RangedIK 复现。'
        '胜过这些行本身不能支撑优越性；关键证据仍是 TRAC-IK 与同核普通双步。详见 `CRIK_IMPLEMENTATION.md` '
        '的官方来源对照和两个 adapter 文档。','',
        '## 9. 交付与停止','',
        '- 算法、同核基线及适配器：`src/confik/correction_reserve/`。\n'
        '- 固定协议、配置、版本及输入身份：`formal_protocol/`，原始逐帧记录：`formal/{panda,ur5e}/runs/`。\n'
        '- 全方法主表、family 表、配对区间、完成 UID、原始 source-data：`formal_reports/`。\n'
        '- 固定扰动和两步 witness：`formal_mechanism/`；完整轨迹 witness 与逐帧验收复核：`formal_verification/`。\n'
        '- 可编辑 SVG/PDF 与 PNG 最终图：`formal_figures_final/`，含完成/时延、配对 family、实际修正范围和真实轨迹时间序列；较早的图形排版草稿保留。','',
        '这里评估的是两种精确运动学模型下的软件在线命令生成，没有碰撞、动力学、接触、力矩或实机结论。'
        '参考路径只证明其自身状态下存在连续合法序列。实验结束后不调参数、不新增模型/指标/视域，'
        '不自动进入另一个算法分支，也不修改旧论文。','']
    destination=Path(a.out)
    with destination.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    write_json(root/'findings_manifest.json',dict(utc=utc(),report=str(destination),report_sha256=sha(destination),
        generator_sha256=sha(__file__),sources={str(p):sha(p) for p in [root/'formal_reports/source_data.json',
            root/'formal_verification/verification_manifest.json',root/'formal_mechanism/summaries.json',
            root/'formal_mechanism_comparison/source_data.json',root/'formal_verification/timing_phases.csv',
            root/'formal_verification/abruptness_table.csv']}))


if __name__=='__main__':main()
