# Task Balance GN：数值完成度与独立验证结论

基线：`23f8555bd214eb4ac45b164a536ead55e142e44c`；同一分支 `codex/hierarchical-v5`。
独立结果出现前的代码／开发证据／输入锁定提交：`97be2df6513b84d83dc3eaeaf90de5fa7c65619c`。
统一结果根：[task_balance_completion](../outputs/single_solver_evidence/task_balance_completion/)。

## 最终判断

**本轮支持数值降耗，不支持相对原 GN 新增独立任务能力。** 相对同缓存紧精度版，固定相对进展规则在新点查询上减少了 Panda/UR5e 的平均计算成本 **17.83%/25.16%**，同时保留已观察局部恢复。但新点查询上原 GN 本来就全部成功；新轨迹上相同 κ 的完成 UID 集合也没有增加。不能把缓存降耗、80% 局部模型性质或 UR5e 优于某个 TRAC 设置的结果，单独包装为新目标带来的任务能力。

完整主表由原始记录自动生成：[TABLES.md](../outputs/single_solver_evidence/task_balance_completion/reports/TABLES.md)。[source_data.json](../outputs/single_solver_evidence/task_balance_completion/reports/source_data.json) 同时保留全部类别、配对区间与 UID 变化。以下均为原服务器结果，不拼接包内本地时间。

## 1. 实现与现成命令验收

在原 `task_balance_gn.py` 中追加一个缓存数值核心，旧类和原 GN 未变。紧精度／相对进展共用该核心，只有 `forcing=None/.25` 不同。θ=.5 第一 QP 使用同 κ 原 GN 的 H/g；公共 1 mm、0.5°、dt=.02 s、原速度和限位合同不变。没有 IK 兜底、未来目标、候选池、新 loss 或权重搜索。

先核对 Panda `trajectory_094/frame53` 的准确 UID、target、previous_q 和 dt，再将包内全部 12 个现成向量送入原 verifier，**未调用 IK 替换命令**：新相对进展版三条均通过，位置 **0.973170585 mm**、姿态 **0.460786628°**；旧本地均衡版、缓存紧精度版各三条也通过。原 GN 三条均因 **0.545854361°** 姿态误差被拒绝，非有限性、限位、速度均无违约。[原验收记录](../outputs/single_solver_evidence/task_balance_completion/package_verification/original_verifier.json) 保存完整命令、步长和源文件 hash。

48 个包内凸问题移植至原 NumPy box-QP，并与独立冷启动 SLSQP epigraph 结果核对；全部达到所检查的上下界关系，实际模型下降比例最小 **0.830951**。接口回归覆盖同 κ 第一方向、固定动态边界、外部拒绝、绝对期限和异常浮点界。数学及接口通过不是性能或创新结论。[数学记录](../outputs/single_solver_evidence/task_balance_completion/mathematics/convex_checks.json)、[限定性质与协议](TASK_BALANCE_COMPLETION_METHOD.md)。

## 2. 全部历史输入与开发轨迹

52 个历史唯一输入、三遍嵌套重复；“直接 GN 来源”是固定的 16 个子集，不把来源别名当新增输入。

| 范围 | 机器人 | 原 GN κ0/κ1 | 旧均衡 κ0/κ1 | 缓存紧精度 κ0/κ1 | 相对进展 κ0/κ1 |
|---|---|---:|---:|---:|---:|
| 全部历史输入 | Panda，41 | 4/4 | 21/21 | 21/21 | 21/21 |
| 全部历史输入 | UR5e，11 | 1/1 | 6/6 | 6/6 | 6/6 |
| 直接 GN 来源 | Panda，13 | 0/0 | 5/5 | 5/5 | 5/5 |
| 直接 GN 来源 | UR5e，3 | 0/0 | 1/1 | 1/1 | 1/1 |

表中为三遍均接受的输入数，未出现成功／失败混合。新规则保留恢复，没有新增局部恢复或损失；剩余未解决输入仍是“未找到命令”，不判数学不可行。[逐输入命令](../outputs/single_solver_evidence/task_balance_completion/reports/local_commands.csv)、[合法 witness](../outputs/single_solver_evidence/task_balance_completion/local/legal_witnesses.json)、[直接来源分组](../outputs/single_solver_evidence/task_balance_completion/reports/local_source_groups.csv)。

原 40 条开发轨迹／机器人、150 帧、三遍：原 GN、缓存紧精度和相对进展在 Panda κ0 均为 **39/40**，κ1 均 **40/40**；UR5e 两 κ 均 **40/40**。UR5e `trajectory_30` 两 κ 均保持完成，但原 GN 也完成，不能算新的任务增量。Pink 为 Panda **37/40**、UR5e **38/40**，本轮重新实测而非借用旧时间。

κ1 相对进展的三遍 DTSR20 均为两机器人 40/40；κ0 各机器人各出现一次合法晚到，第三遍按时完成少一条。旧均衡版 Panda κ0 为 39/38/39、κ1 为 39/40/40，单次 deadline 失败原样保留，未用更好重复替换。主表包含每个设置、类别和全部晚到记录。

开发累计时间（每个完整 sweep，三遍平均）：Panda κ1 原 GN **1603.706 ms**、旧均衡 **2943.489 ms**、缓存紧精度 **1465.348 ms**、相对进展 **1459.599 ms**；UR5e 分别 **1547.349/2764.717/1395.528/1395.663 ms**。这里相对进展的额外收益很小，降耗主要已由缓存紧精度版获得。κ1 合格命令贴边比例分别 **0.933%/0.667%**，不是通过收紧公共位姿容差得到。[全开发表及配对结果](../outputs/single_solver_evidence/task_balance_completion/reports/development_main.csv)。

## 3. 同目标真实局部问题：分开质量、进展与命令验收

原先固定的 **360 个真实 first-outer 问题**全部复用：256 个常规问题和 104 个历史失败输入×κ 问题。每问题每数值模式五次重复，原／缓存紧精度／相对进展／现有缓存 Clarabel，以及四者命令验收诊断，共 **14,400 次**。这不是自然在线内迭代频率样本。

- 紧精度共同检查保留箱违反、同目标值和原箱强凸下界。原、缓存紧精度各 **1775/1800** 调用达标，Clarabel **1795/1800**；三者共同达标对应 **355/360** 问题。其余未隐藏，也未增加迭代以补齐。
- 关闭任务提前接受时，相对进展版在这 360 个固定首问题上均用一个 θ=.5 QP满足进展规则；对独立高精度参照的实际模型下降比例最小 **0.900477**。这不表示一步就能返回任务合格命令。
- 以历史 Panda κ1 子问题为例，紧精度缓存／Clarabel／相对进展的完整调用 P50 为 **0.449739/0.158772/0.154753 ms**。前两者在该组都满足紧精度；最后一个只要求相对进展，**不是同精度击败 Clarabel**。UR5e 对应 **0.437620/0.148089/0.136365 ms**。
- 命令验收诊断揭示不同终止语义：历史 Panda 每 κ 的 205 次调用中，缓存紧精度和 Clarabel 各有 **80** 个真实全步命令合法，相对进展仅 **20** 个。后者其余调用只是可用于后续外层更新的局部方向。完整在线循环最终保留前述 21/41 恢复，不能混为“一次局部进展即成功”。

公平计时说明：[质量—时间表](../outputs/single_solver_evidence/task_balance_completion/reports/subproblem_quality_time.csv) 的 `kernel_p50_ms` 是子程序调用段，**不是各实现匹配的裸内核时间**；Clarabel 该段包含内部 setup/update/质量复算。其真正原生锥求解 `solve_ns`、setup/update、共同全调用时间和验收计数单列于 [timing_scopes_and_commands](../outputs/single_solver_evidence/task_balance_completion/reports/subproblem_timing_scopes_and_commands.csv)。例如 Panda 历史 κ0 的原生 Clarabel P50 约 **0.0635 ms**。命令诊断采用公共 verifier 回调及最后检查，其耗时不冒充在线缓存 native-FK 回调的耗时。未赋予 Clarabel 不具备的同样任务提前中断能力。

## 4. 独立点查询：数值效率有支持，新增漏解恢复没有

每机器人 2,000 个新 query UID，四类各 500；全体保存并通过原 verifier 的 q_ref witness。三遍在 query 内平均后做固定类别分层配对，不反馈 previous_q，不按被测方法失败筛选。

| 机器人 | 原 GN | 旧均衡／缓存紧精度／相对进展 | TRAC 5 ms | TRAC 20 ms |
|---|---|---|---|---|
| Panda | 2000/2000，三遍 | 各 2000/2000，三遍 | 2000/2000，三遍 | 2000/2000，三遍 |
| UR5e | 2000/2000，三遍 | 各 2000/2000，三遍 | 1999/1999/2000 | 2000/2000，三遍 |

因此不存在新方法相对原 GN 减少 witness 确认漏解的独立证据。UR5e TRAC 5 ms 的两次漏解来自**同一个** high-utilization query，第三次成功；不是两个独立困难输入。其完整 UID、三次返回与合法参考见 [miss index](../outputs/single_solver_evidence/task_balance_completion/reports/point_witness_confirmed_misses.json)。所有成功调用均在 20 ms 内；原始状态和失败仍保留。

| 比较：相对进展／基线 | Panda 平均时间比 [95% 区间] | UR5e 平均时间比 [95% 区间] |
|---|---|---|
| 同缓存紧精度 | **0.8217 [0.8031, 0.8396]** | **0.7484 [0.7306, 0.7675]** |
| 原 GN | 0.9337 [0.9311, 0.9364] | 0.9383 [0.9358, 0.9408] |
| TRAC 5 ms | 1.3947 [1.3548, 1.4264] | 1.3643 [1.3048, 1.4223] |

相对进展版 P50/P95/P99：Panda **0.3010/0.5083/0.5705 ms**，UR5e **0.2929/0.4832/0.5328 ms**；同缓存紧精度为 **0.3008/0.9786/1.8795**、**0.2931/1.0857/2.2632 ms**。这支持避免常规可行查询中过度求解局部 minimax 的计算增量，不支持新的可解集合。原 GN P95/P99 仍略低于相对进展；不能只以中位数宣称所有分位数更优。

误差未隐藏：相对进展的接受命令贴边比例为 Panda **2.05%**、UR5e **1.15%**，与原 GN 相同；缓存紧精度为 **2.20%/1.25%**，TRAC 为 **0/0.10%**。所有四类的误差分布、原始命令和逐 UID 配对保留在 [points_main.csv](../outputs/single_solver_evidence/task_balance_completion/reports/points_main.csv) 及两个 `independent_points_*` 目录。

## 5. 独立完整轨迹：同 κ 原 GN 的完成集合没有扩大

每机器人 80 条全新轨迹，四类各 20、150 帧；六设置三遍，共 432,000 帧。不读未来目标／q_ref、不重置分支，失败保持真实上一接受状态且所有目标继续推进。下表累计时间包含全部失败及超时帧，并对三遍取平均。

| 机器人／设置 | TSR 完成数，三遍 /80 | DTSR20 完成数，三遍 /80 | P50/P95/P99 ms | 累计 ms/sweep |
|---|---|---|---|---:|
| Panda 原 GN | 78/78/78 | 78/77/78 | .2695/.2875/.5080 | 3638.656 |
| Panda 旧均衡 | 77/77/78 | 77/77/78 | .4925/.5159/4.5121 | 8149.120 |
| Panda 缓存紧精度 | 78/78/78 | 78/78/78 | .2444/.2630/3.2784 | 4402.458 |
| Panda 相对进展 | **78/78/78** | **77/78/77** | **.2446/.2631/1.4241** | **4128.581** |
| Panda TRAC 5 ms | 79/79/79 | 79/78/79 | .1693/.2241/.4241 | 2555.423 |
| Panda TRAC 20 ms | 79/79/79 | 79/79/79 | .1695/.2245/.4141 | 3553.692 |
| UR5e 原 GN | 80/80/80 | 80/80/80 | .2558/.2694/.3497 | 3061.359 |
| UR5e 旧均衡 | 80/79/80 | 80/79/80 | .4607/.4795/.5532 | 5488.419 |
| UR5e 缓存紧精度 | 80/80/80 | 80/80/80 | .2305/.2439/.3212 | 2763.305 |
| UR5e 相对进展 | **80/80/80** | **80/80/80** | **.2308/.2444/.3204** | **2765.696** |
| UR5e TRAC 5 ms | 78/78/78 | 78/78/78 | .1610/.2237/5.1792 | 2712.467 |
| UR5e TRAC 20 ms | 77/79/77 | 77/78/75 | .1615/.2259/20.2029 | 4805.337 |

**缓存与规则分开判断。** 缓存紧精度／旧均衡累计时间比为 Panda **0.5402 [0.4957,0.5828]**、UR5e **0.5035 [0.5001,0.5057]**。这是缓存、重复检查消除和规定数值界复算的实现对照，不把其中每一项都单独归为原创数学收益。

相对进展／同缓存紧精度：Panda **0.9378 [0.8758,1.0177]**，描述性降幅 **6.22%**；UR5e **1.0009 [0.9992,1.0027]**，没有测得有意义的降幅。Panda P99 从 3.2784 降至 1.4241 ms，但相对原 GN 仍有更高 P99，累计成本也 **高13.46%**（比值区间 **[0.9080,1.4037]**）。UR5e 相对原 GN 成本 **低9.66%**，然而缓存紧精度已经获得这一收益，不能归因于相对进展规则。

**全部类别：**相对进展 Panda 的 smooth/near-singular 各三遍 **19/20**，joint-limit-return/high-curvature 各 **20/20**；累计时间分别 **1362.201/1260.806/743.152/762.423 ms**。UR5e 四类三遍均 **20/20**，累计 **702.954/656.244/701.960/704.538 ms**。Panda smooth 和 high-curvature 各一次合法晚到影响按时完成，未从表中删除。[所有设置的分类表](../outputs/single_solver_evidence/task_balance_completion/reports/independent_family.csv)。

**实际误差与运动：**相对进展 Panda/UR5e 的合格命令贴边率 **0.480%/0.708%**，原 GN 为 **0.472%/0.708%**；加速度 RMS 为 **3.67408/4.48392 rad/s²**，原 GN **3.67321/4.48392**。位置 P95/max 为 Panda **0.387915/0.996167 mm**、UR5e **0.382262/0.999909 mm**；姿态 P95/max、最大步长与所有残差均在主 CSV。没有通过收紧容差隐藏误差，亦不将这些软件量等同实机安全。

**UID 与重复波动：**新／缓存紧精度／原 GN 的几何完成 UID 集合逐遍相同。相对 TRAC，Panda 稳定丢失 `2c543db9…`（smooth）和 `2ec05e99…`（near-singular），稳定获得 `4b594826…`（near-singular），净少一条。UR5e 相对 TRAC 5 ms 有一条稳定恢复和两条部分重复恢复，无损失；这也是原 GN 已有的完成能力。完整 64 位 UID、每遍成功集及变化分数见 [completion_uids](../outputs/single_solver_evidence/task_balance_completion/reports/independent_completion_uids.json)、[gained_lost_uids](../outputs/single_solver_evidence/task_balance_completion/reports/independent_gained_lost_uids.csv)。

旧均衡相对新实现多失去的三个 UID 分别发生一次约 49–50 ms 的 `deadline` 返回；其余重复完成。记录保留，不认定为 objective 的固有缺陷。相对进展 Panda 也有合法晚到，DTSR20 不优于缓存紧精度，不能挑选一次重复来宣称时限收益。没有等效性检验；零差和跨零区间均不证明普遍持平。

## 6. 一页贡献判断与停止

| 待检验增量 | 当前证据判断 |
|---|---|
| 原 minimax 目标能处理平方和 GN 的部分冲突输入 | 历史同输入 witness 支持，直接原 GN 来源 16 个中恢复 6 个；新点查询上原 GN 无漏解，因此未建立独立漏解优势。 |
| 缓存和重复工作消除降低实际成本 | 两机器人开发与独立结果支持，属于可测量工程改善；不是新优化原理。 |
| 相对进展规则避免不必要的紧精度子问题求解 | 同缓存核心、新点查询成本及尾部结果支持，数学性质仅限正则局部模型；真实固定问题的 80% 目标通过核对。 |
| 相对进展提高完整跟踪能力或提供跨机器人稳定时限收益 | 未建立。与同 κ 原 GN 完成集合相同；Panda 存在成本和 DTSR20 不利结果，UR5e 的完成优势已由原 GN 达到。 |
| 标量对偶专用内核普遍优于成熟同目标优化器 | 未建立。困难紧精度问题上 Clarabel 更快，低精度相对进展与紧精度锥解不能混比。 |
| 已经具备新算法论文的独立任务能力证据 | 仍不足。可主张的是此实现中的数值工作控制与选定历史冲突恢复，不能声称普遍扩大可解集合、全局收敛、未来可行性或普遍优于 TRAC/Pink。 |

RangedIK、凸对偶、非精确 GN 与 gap 停止均有成熟先例，任务书四项文献已按 nature-academic-search 流程核验，具体边界见[方法说明](TASK_BALANCE_COMPLETION_METHOD.md)。本轮统计按 nature-statistics 的完整 UID 与嵌套重复规则：固定分层配对、4,000 次 bootstrap、描述性未校正 95% 区间，不把三遍当三倍样本。

包内本地旧 160 条仍 **154/160**、相对进展比原 GN 贵约 **30%**、无新增完整完成，未隐去。逐 UID/所有输入数组与原记录核对，最大差 **4.44e−16**；这是已观察数据的本地重建，不是本轮新独立证据。

本轮复验了 **828,000** 条已保存的点／轨迹调用，其中 **824,427** 个接受命令，接受违约 **0**；另外复验 4,000 个点 witness 和 24,000 个参考路径步。全部命令、失败、超时、当前真实反馈、成功轨迹索引及文件 hash 已保存。复验没有运行 IK 替换命令。旧模型、旧 solver/verifier、旧证据和论文不变；新数值核心／固定配置在独立运行中未修改。

本轮有限任务完成后停止：不追加第三种目标、参数网格、正式样本或论文写作。

## 复现与文件入口

唯一实验入口：[run_task_balance_completion.py](../scripts/run_task_balance_completion.py)，配置：[task_balance_completion.yaml](../configs/task_balance_completion.yaml)。新结果目录均独占创建，不覆盖既有运行。阶段顺序为 `verify → prepare → math → local → trajectories --robot panda/ur5e → seal → subproblems → inputs → launch_seal → points --robot panda/ur5e → evaluate --robot panda/ur5e`；现存 `seal` 文件同时记录本次包装器修正的具体历史，并非要求复现者再制造同样编辑。

环境为原 `/home/eric/anaconda3/envs/isaaclab_3/bin/python`，`PYTHONPATH=tmp/crik_dependencies/python:src`，`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，执行器固定 CPU 4。原 NumPy 路径和 Pinocchio 几何适配保留，未使用包内 Numba Panda 模型替代。

报告使用 [report_task_balance_completion.py](../scripts/report_task_balance_completion.py)，其 `--supplement` 只复验保存 witness、界和计时范围，不运行 IK。主报告实际执行源码快照与 manifest hash 匹配，补充检查另有 source hash。原始数据分别位于 `local/`、`subproblems/`、`development_*/runs/`、`independent_points_*/records.jsonl.gz`、`independent_trajectories_*/runs/`；新输入和合法参考位于 `independent_inputs/`。全部成功轨迹均有逐帧真实命令和验收索引，不以成功数代替 witness。
