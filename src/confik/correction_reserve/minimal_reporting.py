"""Complete-trajectory analysis and read-only verification of the new mode."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from ..types import Pose, IKQuery
from . import study as old
from . import reporting as common
from .minimal_study import configurations, verify_seal, NEW_METHODS
from .minimal_demand import PredictionDemand
from .geometry import task_scale, reserve_value
from .native_geometry import NativeGeometry

LABELS={**common.LABELS,'cr_ik_minimal':'Minimal CR-IK',
    'cr_ik_minimal_fixed':'Minimal CR-IK: fixed demand',
    'cr_ik_minimal_no_shortcut':'Minimal CR-IK: no shortcut'}
PRIMARY='cr_ik_minimal'


def mean(values):
    a=[v for v in values if v is not None]
    return float(np.mean(a)) if a else None


def aggregate():
    cfg,base,root=configurations();verify_seal();out=root/'reports';out.mkdir(exist_ok=False)
    parameters=json.loads((root/'protocol/demand_parameters.json').read_text())
    summaries=[];arrays={};extras=defaultdict(lambda:defaultdict(list));conditions=defaultdict(list)
    timing=defaultdict(list);failures=[];audit=[];witnesses=[];sources={};completion={}
    for robot in cfg['robots']:
        folder=root/robot
        if not (folder/'completed.json').exists():raise RuntimeError(f'{robot} run not finished')
        source,kin,v,urdf=old.context(robot,base);native=NativeGeometry(kin,urdf)
        scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
        items={i['uid']:i for i in json.loads((folder/'online_inputs.json').read_text())}
        jobs=json.loads((folder/'summaries.json').read_text())
        assert len(jobs)==880
        source_manifest=json.loads((folder/'completed.json').read_text())
        sources[str(folder/'completed.json')]=old.sha(folder/'completed.json')
        counts=dict(robot=robot,frames=0,accepted_commands=0,verified_nominal_pairs=0,
                    met_demand_checked=0,direct_returns_checked=0,causal_demands_checked=0,
                    acceptance_discrepancies=0,feedback_discrepancies=0,demand_discrepancies=0,
                    minimum_recomputed_demand_slack=None)
        found_witness=False
        for j,s in enumerate(sorted(jobs,key=lambda r:r['run_id'])):
            file=folder/s['raw_file'];assert old.sha(file)==source_manifest['files'][s['raw_file']]
            rows=old.read_rows(file);summaries.append(s);item=items[s['uid']]
            previous=np.asarray(item['initial_q']);causal=None
            if s['method'] in NEW_METHODS:
                causal=PredictionDemand(scale,parameters[robot]['minimum'],parameters[robot]['initial'],
                                        fixed=s['method']=='cr_ik_minimal_fixed')
            group=(robot,s['method']);conditional=defaultdict(lambda:[0,0,0])
            for t,r in enumerate(rows):
                assert r['frame']==t and r['dt']==.02
                assert r['target_position']==item['target_position'][t] and r['target_rotation']==item['target_rotation'][t]
                assert np.array_equal(previous,r['previous_q'])
                target=Pose(r['target_position'],r['target_rotation']);query=IKQuery(target,previous,.02)
                q=np.array(r['q']) if r['q'] is not None else np.full(kin.nq,np.nan)
                check=v.check(q,query)
                assert check.accepted==r['accepted'] and check.finite_ok==r['finite']
                assert r['accepted_within_20ms']==bool(check.accepted and r['total_latency_ns']<=20_000_000)
                counts['frames']+=1;counts['accepted_commands']+=int(check.accepted)
                if check.accepted:
                    previous=q.copy()
                    extras[group]['near_tolerance'].append(max(check.position_error/scale[0],check.orientation_error/scale[3])>.9)
                assert np.array_equal(previous,r['accepted_state_q'])
                if r.get('nominal_next_verified'):
                    predicted=Pose(r['predicted_position'],r['predicted_rotation']);z=np.asarray(r['nominal_next_q'])
                    assert v.check(z,IKQuery(predicted,q,.02)).accepted
                    counts['verified_nominal_pairs']+=1
                if causal is not None:
                    prediction,d=causal.observe(target)
                    assert r['demand']==d['demand'] and r['observed_eta']==d['observed_eta']
                    np.testing.assert_allclose(prediction.position,r['predicted_position'],atol=0,rtol=0)
                    np.testing.assert_allclose(prediction.rotation,r['predicted_rotation'],atol=0,rtol=0)
                    counts['causal_demands_checked']+=1
                    assert r['conic_calls']<=2
                    if r['demand_met']:
                        assert check.accepted and r['nominal_next_verified']
                        gamma,mapping,slack=reserve_value(native,prediction,q,np.asarray(r['nominal_next_q']),scale,step)
                        assert mapping.full_rank
                        margin=float(np.min(slack-r['demand']*np.linalg.norm(mapping.matrix,axis=1)))
                        assert margin>=0, (s['run_id'],t,margin)
                        counts['met_demand_checked']+=1
                        prior=counts['minimum_recomputed_demand_slack']
                        counts['minimum_recomputed_demand_slack']=margin if prior is None else min(prior,margin)
                    if r['direct_return']:
                        assert not r['optimization_called'] and r['initial_demand_met'] and r['demand_met']
                        assert np.array_equal(r['q'],r['backup_q'])
                        counts['direct_returns_checked']+=1
                        extras[group]['direct_extra'].append(r['total_latency_ns']-r['backup_ns'])
                    extras[group]['demand'].append(r['demand']);extras[group]['demand_met'].append(r['demand_met'])
                    if r.get('offline_next_prediction_error') is not None:
                        extras[group]['true_next_error'].append(r['offline_next_prediction_error'])
                        extras[group]['coverage'].append(r['offline_demand_covered'])
                        for scope in ('all_current','accepted_current'):
                            if scope=='accepted_current' and not r['accepted']:continue
                            key=(scope,'met' if r['demand_met'] else 'unmet')
                            conditional[key][0]+=1
                            conditional[key][1]+=int(rows[t+1]['accepted'])
                            conditional[key][2]+=int(rows[t+1]['accepted_within_20ms'])
                if r.get('backup_accepted',False) and r['accepted']:
                    extras[group]['intervention'].append(r['intervention_normalized_l2'])
                    extras[group]['unchanged'].append(r['command_unchanged'])
                if 'conic_calls' in r:
                    extras[group]['calls'].append(r['conic_calls'])
                for key in ('total_latency_ns','backup_ns','demand_estimation_ns','initial_pair_ns',
                    'cone_construction_ns','cone_solve_ns','nonlinear_validation_ns','prediction_ns','optimization_ns'):
                    if key in r:timing[(robot,s['method'],key)].append(r[key])
                if t==s['first_failure_frame']:
                    failures.append({k:r.get(k) for k in ('robot','uid','site_id','family','method','repeat','frame',
                        'previous_q','target_position','target_rotation','q','failure_kind','position_error',
                        'orientation_error','velocity_utilization','demand','demand_met','native_return_code')})
            for (scope,demand_status),(n,success,timely) in conditional.items():
                conditions[(robot,s['method'],scope,demand_status)].append(dict(uid=s['uid'],repeat=s['repeat'],
                    frames=n,next_successes=success,next_timely=timely,rate=success/n,timely_rate=timely/n))
            arrays[s['run_id']]=dict(latency=np.array([r['total_latency_ns'] for r in rows]),
                errors=np.array([[r['position_error'],r['orientation_error']] for r in rows if r['accepted']]).reshape(-1,2))
            completion.setdefault(robot,{}).setdefault(s['method'],{}).setdefault(str(s['repeat']),[])
            if s['completion']:completion[robot][s['method']][str(s['repeat'])].append(s['uid'])
            if not found_witness and s['method']==PRIMARY and s['completion']:
                witness={k:s[k] for k in ('robot','uid','method','repeat','raw_file')}
                witness.update(initial_q=item['initial_q'],dt=.02,
                    frames=[{k:r[k] for k in ('frame','target_position','target_rotation','q','accepted','total_latency_ns',
                        'demand','demand_met')} for r in rows])
                old.write_json(out/f'{robot}_successful_witness.json',witness);witnesses.append(s['uid']);found_witness=True
            if (j+1)%100==0:print(f'read-only audit {robot}: {j+1}/{len(jobs)} runs',flush=True)
        audit.append(counts)
    # Add display labels in this analysis process only; no old source or result is edited.
    common.LABELS.update(LABELS)
    main=common.group_table(summaries,arrays);family=common.group_table(summaries,arrays,True)
    def percentile(x,p):return float(np.percentile(x,p)) if len(x) else None
    for r in main:
        e=extras[(r['robot'],r['method'])]
        r.update(optimization_call_rate=mean([v>0 for v in e['calls']]),mean_conic_calls=mean(e['calls']),
            intervention_normalized_mean=mean(e['intervention']),intervention_normalized_p95=percentile(e['intervention'],95),
            unchanged_given_legal_backup=mean(e['unchanged']),accepted_near_tolerance_rate=mean(e['near_tolerance']),
            direct_return_rate=len(e['direct_extra'])/sum(s['frames'] for s in summaries if s['robot']==r['robot'] and s['method']==r['method']),
            direct_extra_p50_ms=percentile(e['direct_extra'],50)/1e6 if e['direct_extra'] else None,
            direct_extra_p95_ms=percentile(e['direct_extra'],95)/1e6 if e['direct_extra'] else None,
            demand_coverage=mean(e['coverage']),demand_met_rate=mean(e['demand_met']),
            demand_p50=percentile(e['demand'],50),demand_p95=percentile(e['demand'],95),demand_max=percentile(e['demand'],100),
            true_next_error_p95=percentile(e['true_next_error'],95))
    conditional_table=[]
    for (robot,method,scope,status),rows in sorted(conditions.items()):
        uids=sorted({r['uid'] for r in rows});total=sum(r['frames'] for r in rows)
        conditional_table.append(dict(robot=robot,method=method,current_scope=scope,demand_status=status,
            available_trajectories=len(uids),descriptive_frames=total,next_successes=sum(r['next_successes'] for r in rows),
            pooled_next_success=sum(r['next_successes'] for r in rows)/total,
            mean_trajectory_next_success=float(np.mean([np.mean([r['rate'] for r in rows if r['uid']==u]) for u in uids])),
            mean_trajectory_next_timely=float(np.mean([np.mean([r['timely_rate'] for r in rows if r['uid']==u]) for u in uids]))))
    units=[];pairs=[];changes=[]
    metrics=['completion','deadline_completion','total_latency_ns','frame_success','acceleration_rms',
             'accepted_near_tolerance_rate','optimization_call_rate','intervention_normalized_mean','unchanged_given_legal_backup']
    for robot in cfg['robots']:
        robot_s=[s for s in summaries if s['robot']==robot];uids=sorted({s['uid'] for s in robot_s});unit={}
        for method in cfg['methods']:
            for uid in uids:
                group=[s for s in robot_s if s['method']==method and s['uid']==uid]
                assert len(group)==(1 if method=='pink_qp' else 3)
                row=dict(robot=robot,method=method,uid=uid,family=group[0]['family'],search_repeats=len(group),
                         **{m:mean([s.get(m) for s in group]) for m in metrics})
                unit[(method,uid)]=row;units.append(row)
        families=[unit[(PRIMARY,u)]['family'] for u in uids]
        comparisons=[(PRIMARY,b) for b in cfg['methods'] if b!=PRIMARY]+[('cr_ik_minimal_no_shortcut','cr_ik')]
        for method,base_method in comparisons:
            for metric in metrics:
                a=[unit[(method,u)][metric] for u in uids];b=[unit[(base_method,u)][metric] for u in uids]
                if any(v is None for v in a+b):continue
                pairs.append(dict(robot=robot,method=method,baseline=base_method,metric=metric,trajectories=len(uids),
                    **common.paired_intervals(np.array(a),np.array(b),families,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])))
            for u in uids:
                a,b=unit[(method,u)]['completion'],unit[(base_method,u)]['completion']
                changes.append(dict(robot=robot,method=method,baseline=base_method,uid=u,family=unit[(method,u)]['family'],
                    completion_fraction=a,baseline_completion_fraction=b,change='gained' if a>b else 'lost' if a<b else 'same',
                    stable_all_vs_none=(a==1 and b==0) or (a==0 and b==1)))
    time_table=[dict(robot=k[0],method=k[1],phase=k[2],frames=len(v),mean_ms=float(np.mean(v)/1e6),
        p50_ms=float(np.percentile(v,50)/1e6),p95_ms=float(np.percentile(v,95)/1e6),p99_ms=float(np.percentile(v,99)/1e6),
        max_ms=float(np.max(v)/1e6)) for k,v in sorted(timing.items())]
    tables={'main_table':main,'family_table':family,'trajectory_units':units,'paired_comparisons':pairs,
        'gained_lost_uids':changes,'conditional_next_frame':conditional_table,'timing_phases':time_table,
        'first_failure_inputs':failures,'run_summaries':summaries}
    for name,rows in tables.items():common.csv_write(out/f'{name}.csv',rows)
    old.write_json(out/'completion_uids.json',completion)
    old.write_json(out/'source_data.json',tables)
    old.write_json(out/'verification_manifest.json',dict(utc=old.utc(),results=audit,numerical_solver_calls=0,
        operation='read-only original verifier, causal-demand replay and nonlinear local-map recomputation',witness_uids=witnesses))
    old.write_json(out/'manifest.json',dict(utc=old.utc(),sources=sources,code_sha256=old.sha(__file__),
        independent_unit='40 development trajectory UIDs per robot; searches nested; not a fresh test',
        intervals='paired family-stratified UID bootstrap, 4000 resamples, unadjusted descriptive 95% intervals',
        exclusions=0,frame_calls=sum(s['frames'] for s in summaries),runs=len(summaries),
        demand_conditioning='descriptive after-run join; no causal or probabilistic guarantee',
        timing='all command-ready computation; serialization and offline analysis excluded'))
    for r in main:print(r['robot'],r['method'],r['completion_by_repeat'],r['deadline_completion_by_repeat'],
        round(r['cumulative_ms_per_sweep'],3),round(r['p50_ms'],3),round(r['p95_ms'],3))


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def write_findings():
    cfg,_,root=configurations();folder=root/'reports'
    data=json.loads((folder/'source_data.json').read_text())
    audit=json.loads((folder/'verification_manifest.json').read_text())
    params=json.loads((root/'protocol/demand_parameters.json').read_text())
    def mainrow(robot,method):return next(r for r in data['main_table'] if r['robot']==robot and r['method']==method)
    def pair(robot,baseline,metric,method=PRIMARY):
        return next(r for r in data['paired_comparisons'] if r['robot']==robot and r['method']==method and
                    r['baseline']==baseline and r['metric']==metric)
    def percent(v):return '—' if v is None else f'{100*v:.2f}%'
    def triplet(v):return '/'.join(map(str,v))
    lines=['# CR-IK 最小命令干预：完整开发比较','',
        '**结论：计算和无谓命令修改大幅减少，但没有完整保留原 CR-IK 的完成表现。** '
        '本实现可以冻结为可追溯的开发快照；目前不建议启动新的正式轨迹评估。'
        '不将本轮更便宜的实现解释为已经优于成熟求解器，也不根据这轮结果改选消融或追加模块。','',
        '## 范围与实现','',
        '已实现独立的 `MinimalInterventionIK`，原 `runtime.py`、`convex.py`、`geometry.py`、'
        'TRAC/Pink、verifier、旧论文和旧结果未改。主模式仅使用当前及历史目标估计需求，'
        '没有真实下一目标或 q_ref 输入；最多两次单阶段 SOCP，lambda=1。'
        '初始对足够时直接返回逐元素相同的备用命令；需求不满足时明确标记而不截小需求。','',
        '代码、协议、目标身份和需求参数在比较前提交为 `cc73024384349f74b5938a975849ebd0cb3a6a79` 并 push。'
        '本轮为每台 40 条**已观察开发轨迹**、每条 150 帧、四类各 10 条；不是新测试，'
        '未读取最新 160 条正式轨迹来调参。七种含 TRAC 的配置各三次整轨迹重复，Pink 一次；'
        '共 1760 个 trajectory runs、264000 次在线帧调用。顺序按固定种子打乱，所有失败和超时保留。'
        'TRAC 搜索随机性未被声称为可控种子。','',
        '统计以完整 trajectory UID 为单位，先平均轨迹内搜索重复，再按 family 分层配对 bootstrap '
        '4000 次。下列区间为描述性、未作多重比较校正的 95% 区间，不是等效性或非劣效性检验。','',
        markdown_table(['机器人','r_min（目标误差中位数）','启动值 / 固定-r（目标误差Q95）','W'],[
            [robot,f'{p["minimum"]:.6f}',f'{p["initial"]:.6f}',20] for robot,p in params.items()]),'',
        '## 1. 是否保留了原修正余量目标的完成收益？','',
        '没有完整保留。Panda 整体完成数略降、按时完成数提高；UR5e 整体与按时完成数均略降。'
        '更重要的是，UR5e 有一条 near-singular 轨迹从原 CR-IK 的三次全部完成变为新模式三次全部失败。'
        '这不是仅从总体平均值推测的损失。区间较宽或包含零也不能证明性能已经等效。','']
    for robot in cfg['robots']:
        rows=[]
        for method in cfg['methods']:
            r=mainrow(robot,method)
            rows.append([LABELS[method],triplet(r['completion_by_repeat']),triplet(r['deadline_completion_by_repeat']),
                percent(r['tsr']),percent(r['dtsr20']),f'{r["p50_ms"]:.3f}/{r["p95_ms"]:.3f}/{r["p99_ms"]:.3f}',
                f'{r["cumulative_ms_per_sweep"]/1000:.3f}'])
        lines += [f'### {robot}（每遍 40 条）','',markdown_table(
            ['方法','完成数','全帧≤20ms完成数','TSR','DTSR20','P50/P95/P99 ms','40条累计时间s（重复均值）'],rows),'']
    rows=[]
    for robot in cfg['robots']:
        for metric in ('completion','deadline_completion'):
            p=pair(robot,'cr_ik',metric)
            rows.append([robot,metric,f'{100*p["difference"]:+.2f}',
                         f'[{100*p["difference_ci"][0]:+.2f}, {100*p["difference_ci"][1]:+.2f}]'])
    lines += [markdown_table(['机器人','Minimal − 原 CR-IK','差值 pp','95%区间'],rows),'',
        markdown_table(['机器人','类别','完整 UID','原 CR-IK完成比例','Minimal完成比例','变化'],[
            [r['robot'],r['family'],r['uid'],f'{r["baseline_completion_fraction"]:.3f}',
             f'{r["completion_fraction"]:.3f}',r['change']]
            for r in data['gained_lost_uids'] if r['method']==PRIMARY and r['baseline']=='cr_ik' and r['change']!='same']),'',
        '对全部其他方法的恢复/损失 UID、逐遍完成集合和首次失败真实输入，见 '
        '`reports/gained_lost_uids.csv`、`completion_uids.json`、`first_failure_inputs.csv`。'
        '不同方法闭环 previous_q 可以不同；这些损失不被解释成同输入的数学不可行性。','',
        '## 2. 计算降低来自目标变化，还是避免无必要求解？','',
        '主要来自新的单阶段数值流程，而不是仅来自直接返回。禁用 shortcut 的消融已经消除了原来的 '
        'primary+secondary 双阶段以及三轮更新，仍使用固定结构缓存；它本身就获得了绝大部分成本下降。'
        '因此不能把这部分收益单独归因于目标函数，或单独归因于缓存。','']
    rows=[]
    for robot in cfg['robots']:
        oldr=mainrow(robot,'cr_ik');new=mainrow(robot,PRIMARY);forced=mainrow(robot,'cr_ik_minimal_no_shortcut')
        cost=pair(robot,'cr_ik','total_latency_ns');shortcut=pair(robot,'cr_ik_minimal_no_shortcut','total_latency_ns')
        rows.append([robot,f'{100*(1-cost["ratio"]):.2f}%',
            f'{100*(1-forced["cumulative_ms_per_sweep"]/oldr["cumulative_ms_per_sweep"]):.2f}%',
            f'{100*(1-shortcut["ratio"]):.2f}%',f'[{shortcut["ratio_ci"][0]:.3f}, {shortcut["ratio_ci"][1]:.3f}]',
            f'{oldr["mean_conic_calls"]:.3f} → {forced["mean_conic_calls"]:.3f} → {new["mean_conic_calls"]:.3f}'])
    lines += [markdown_table(['机器人','主模式累计降幅 vs原','禁shortcut降幅 vs原','shortcut附加降幅 vs禁用',
        '主模式/禁用成本比95%区间','平均SOCP数：原→禁用→主'],rows),'',
        '直接返回的额外成本收益在这轮均值上为正，但其配对成本比区间均包含 1。'
        '各配置闭环状态和随机搜索也有差异，以上是端到端消融对照，不是严格可加的因果耗时分解。'
        '禁用 shortcut 的完成表现反而略好；本轮不据此重选主算法。固定-r 的主结果也完整保留，'
        '没有因为其需求经常不满足而将它删除。','',
        '## 3. 原始命令已经足够时，是否真正保持不变？','',
        '是。所有标为直接返回的帧都在只读复核中确认与备用 q 完全相同、没有 SOCP 调用，'
        '并且真实当前/名义下一合同及需求均满足。下表“不变”以合法备用解且当前被接受的帧为分母；'
        '它也包括尝试优化未获合格改进而保留备用解的情况，不等于需求全部满足。','']
    rows=[]
    for robot in cfg['robots']:
        for method in ('cr_ik',PRIMARY,'cr_ik_minimal_fixed','cr_ik_minimal_no_shortcut'):
            r=mainrow(robot,method)
            rows.append([robot,LABELS[method],percent(r['optimization_call_rate']),percent(r['direct_return_rate']),
                percent(r['unchanged_given_legal_backup']),f'{r["intervention_normalized_mean"]:.6f}',
                percent(r['accepted_near_tolerance_rate']),f'{r["acceleration_rms_mean"]:.3f}'])
    lines += [markdown_table(['机器人','方法','SOCP调用率','直接返回率','备用命令完全不变','平均归一化干预L2',
        '已接受命令>90%容差','加速度RMS rad/s²（轨迹均值）'],rows),'',
        markdown_table(['机器人','直接返回新增开销 P50/P95 ms','原→新加速度RMS变化'],[
            [robot,f'{mainrow(robot,PRIMARY)["direct_extra_p50_ms"]:.3f}/{mainrow(robot,PRIMARY)["direct_extra_p95_ms"]:.3f}',
             f'{100*(mainrow(robot,PRIMARY)["acceleration_rms_mean"]/mainrow(robot,"cr_ik")["acceleration_rms_mean"]-1):+.2f}%']
            for robot in cfg['robots']]),'',
        '新增开销是同帧完整外层时间减去完整备用调用时间，需求估计、初始对检查和选择没有移出计时。'
        '两机器人该中位数低于 0.5 ms，但 Panda 的对应 P95 略高于 0.5 ms。'
        '命令干预和贴边误差明显减少，运动波动却不是一致下降：UR5e 的加速度 RMS 反而增加。','',
        '### 需求、覆盖与真实下一帧','',
        markdown_table(['机器人','模式','需求满足率','经验需求覆盖率','运行中最大 r'],[
            [robot,LABELS[method],percent(mainrow(robot,method)['demand_met_rate']),
             percent(mainrow(robot,method)['demand_coverage']),f'{mainrow(robot,method)["demand_max"]:.3f}']
            for robot in cfg['robots'] for method in (PRIMARY,'cr_ik_minimal_fixed','cr_ik_minimal_no_shortcut')]),'',
        '覆盖率比较 r_t 与**完成在线运行之后**连接的真实下一目标预测误差；最后一帧没有下一目标，'
        '不进入该分母。滚动经验Q95不是95%概率保证，实测覆盖低于95%。固定-r的95%覆盖来自'
        '同一已观察开发目标样本上的分位数构造，不能当作独立泛化证据。需求没有上限裁剪。','',
        markdown_table(['机器人','当前合法时需求状态','有该状态的轨迹数','描述性帧数','下一帧成功/帧数','轨迹均值成功率'],[
            [r['robot'],r['demand_status'],r['available_trajectories'],r['descriptive_frames'],
             f'{r["next_successes"]}/{r["descriptive_frames"]}',percent(r['mean_trajectory_next_success'])]
            for r in data['conditional_next_frame'] if r['method']==PRIMARY and r['current_scope']=='accepted_current']),'',
        '需求满足后本样本下一帧均成功，但未满足组的下一帧也绝大多数成功。'
        '两组输入困难程度不同，这不是需求判定的因果试验，也不说明该余量是下一帧成功的必要条件。'
        '全体当前帧及包含当前失败的分母另存 `conditional_next_frame.csv`。','',
        '## 4. 相比 TRAC-IK 和 Pink，是否缩小了实际成本差距？','',
        '是，差距大幅缩小，但没有形成普遍净优势。Panda 已接近 Pink 的累计成本；UR5e 仍较高。'
        '两台的 TRAC-IK 5 ms 仍显著更便宜，UR5e 的完成率也更高。只看新模式相对原昂贵 CR-IK 的降幅会夸大实用收益。','',
        markdown_table(['机器人','比较对象','原CR-IK/对照累计成本','Minimal/对照累计成本','Minimal − 对照TSR pp'],[
            [robot,LABELS[method],f'{mainrow(robot,"cr_ik")["cumulative_ms_per_sweep"]/mainrow(robot,method)["cumulative_ms_per_sweep"]:.3f}',
             f'{pair(robot,method,"total_latency_ns")["ratio"]:.3f}',f'{100*pair(robot,method,"completion")["difference"]:+.2f}']
            for robot in cfg['robots'] for method in ('trac_task_5ms','trac_task_20ms','pink_qp')]),'',
        '### 全部四类开发轨迹','',
        '每格为平均完成数/10及每条轨迹的平均累计毫秒数；包括失败轨迹的全部150帧。'
        '完整 family-wise TSR、DTSR20、P50/P95/P99、残差等在 `family_table.csv`。','',
        markdown_table(['机器人','类别','TRAC5','TRAC20','Pink','原CR','普通双步','Minimal','固定r','禁shortcut'],[
            [robot,family]+[f'{r["mean_completed_trajectories"]:.2f}; {r["trajectory_time_mean_ms"]:.1f}ms'
                for method in cfg['methods'] for r in data['family_table']
                if r['robot']==robot and r['family']==family and r['method']==method]
            for robot in cfg['robots'] for family in ('smooth','near_singular','joint_limit_return','high_curvature')]),'',
        '## 5. 下一步是否值得固定算法并做新轨迹评估？','',
        '建议**固定这份开发实现与结果，但暂不启动新的正式轨迹评估**。'
        '已证明这里不需要每帧六次锥求解才能获得低成本、少干预的合法命令，'
        '但未完整保留原方法的完成收益，UR5e 存在稳定丢失轨迹、加速度增加，'
        '且普通双步在本轮 UR5e 上三遍均完整完成所有轨迹。'
        '目前还不足以把本主模式定位为更可靠的连续 IK 方法。'
        '这个判断没有将设计目标合并成结果驱动总gate，也没有把区间覆盖零当作等效证明。','',
        '## 可复现交付与停止','',
        f'只读复核检查 {sum(r["frames"] for r in audit["results"]):,} 个帧记录、'
        f'{sum(r["accepted_commands"] for r in audit["results"]):,} 个已接受命令、'
        f'{sum(r["met_demand_checked"] for r in audit["results"]):,} 个需求满足声明。'
        '原 verifier、实际状态反馈、因果需求计算和重新计算的非线性局部余量未发现不一致。'
        '复核不调用 IK；所有数值设置与比较前提交保持一致，12 项新测试通过。','',
        '- 代码：`src/confik/correction_reserve/minimal_{demand,convex,runtime,study,reporting,figures}.py`。\n'
        '- 入口：`scripts/run_crik_minimal.sh`；已有输出目录拒绝重跑/覆盖。\n'
        '- 参数、输入及依赖：`minimal_intervention_development/protocol/`。\n'
        '- 原始逐帧记录：`minimal_intervention_development/{panda,ur5e}/runs/`。\n'
        '- 主表、family表、UID、配对区间、阶段计时、成功witness：`reports/`。\n'
        '- 可编辑SVG/PDF与PNG：`figures/`，全样本配对权衡和固定UID真实需求/工作量时间序列。','',
        '统计和制图按完整轨迹进行，保持全部类别、未满足需求和时限失败可见。'
        '这是精确运动学软件开发研究；局部右逆余量不是非线性全局保证，20ms指标不是硬实时证明。'
        '旧论文、旧实现和原正式结果保持原样。本轮到此停止，不自动训练、换题、扩展数据或增加模块。','']
    path=old.ROOT/'docs/CRIK_MINIMAL_INTERVENTION_FINDINGS.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    old.write_json(root/'findings_manifest.json',dict(report=str(path.relative_to(old.ROOT)),report_sha256=old.sha(path),
        generator_sha256=old.sha(__file__),sources={str(p):old.sha(p) for p in
            (folder/'source_data.json',folder/'verification_manifest.json',root/'protocol/demand_parameters.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-findings',action='store_true')
    args=parser.parse_args()
    if args.write_findings:write_findings()
    else:aggregate()


if __name__=='__main__':main()
