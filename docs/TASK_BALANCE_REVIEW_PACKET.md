# Task Balance GN：最近方法比较与贡献确认验收包

## 结论先行

本轮补充共同输入比较中，冻结主方法在Panda/UR5e均返回 **2160/2160** 个合格命令，
直接同合同SQP也达到2160/2160。主方法的平均完整调用时间为 **0.3963/0.3874 ms**，
SQP为 **2.0712/2.0592 ms**。固定1次局部QP仍漏10/12个查询，固定2次仍漏3/4个；
相对进展方法在这些已观察输入上同时降低了相对固定2次的平均成本。

这不是完整应用成功率的普遍优势：本轮应用型合成扫描的所有方法、两机器人、
每次重复均完成12/12条序列。主方法在该应用未进入额外对偶求解，其命令与原GN
逐帧完全相同。SQP耗时较高，但关节加速度RMS明显更低。范围损失参照是如实标注
的移植，不是完整RangedIK；相对gap原则有明确先例。本包据此保留点查询中的
实际覆盖—成本增量，同时限定其方法归属和应用证据。

研究基线 `6a7336f64bd48e4e293f2c43a1ec0833e529ed04`；先提交代码/协议/输入的
实际运行提交 **`8e1257e16`**。当前分支 `codex/hierarchical-v5`。
最终结果提交SHA在交付消息报告，不在提交内部递归写入自身SHA。

## 1. 方法归属：哪些是基础，哪些是本实现的设计

完整说明：[TASK_BALANCE_PRIOR_ART_AND_SCOPE.md](TASK_BALANCE_PRIOR_ART_AND_SCOPE.md)。

| 已核对对象 | 归属与本实现区别 | 本轮证据入口 |
|---|---|---|
| RangedIK Eq.(5)、(10)–(13)，官方ranged-ik源码 | 范围任务和范围内偏好已有；其分量范围/多任务损失与两个公共范数球不同。本实现的局部minimax不是Swamp Groove累加 | 官方Release例程、FK复验；同合同 `range_loss_matched` |
| Zheng、Ma、Xue，TSP2024 Eq.(23) | H−D≤ρ(H0−H)与U−L≤η(P0−U)同型；80%是有效界下的代数推论，**不再单列原创规则** | 同缓存tight、Clarabel、fixed_qp1/2 |
| 集合任务、epigraph/标量对偶、非精确GN | 成熟基础；θ是两块局部模型的对偶变量，不是机器人专属手调权重 | 同κ原GN与直接约束SQP |
| 当前完整数值实现 | 同一动态箱内协调两块容差利用率，首个θ=.5方向可验收即返回，否则继续同一个局部模型；真实下降和最终验收仍独立 | 全样本、全部格、实际命令与应用反馈 |

两个关键论文的作者、DOI、卷页由Crossref与作者原文核对；没有将附件30条线索
全部冒称为已核验。来源与公式定位见
[verified_sources.json](../outputs/task_balance_comparison/references/verified_sources.json)。
原 `METHOD_AND_CONTRIBUTIONS.md` 保留为历史，不删真实结果；本文档明确更正停止规则的归属。

## 2. 对照身份与公平接口

完整公式、API、初始化、停止、计时和映射见
[baseline_definitions.md](../outputs/task_balance_comparison/protocol/baseline_definitions.md)。

| 设置 | 实际求解内容 | 与主方法的区别 |
|---|---|---|
| relative | 冻结Task Balance GN，κ=1、forcing=.25，30/16/8预算、NumPy活动集 | 主方法无改动 |
| gn | 冻结同κ有界平方和GN | 目标与原有实现 |
| tight | 同缓存、同minimax外层，取消relative停止，仅紧gap | 局部精度工作量 |
| clarabel | 既有同minimax外层epigraph锥求解，保留首QP验收机会 | 局部数值后端/紧解 |
| fixed_qp1/2 | 每局部最多1/2次加权QP，第二θ沿用同一保障Newton/括区；返回最好方向继续外层 | 不计算不需要的forcing下界；非整个IK只做1/2步 |
| direct_sqp | SciPy SLSQP，单previous_q初始化；min .5||S⁻¹(q−previous)||²；两个独立位姿不等式及动态箱；解析导数/同点缓存 | 独立直接约束方法；一旦原verifier通过就返回，不强迫位移最优 |
| range_loss_matched | 官方Swamp Groove作用于两归一化块范数，保留内部向0偏好；独立L-BFGS-B、解析梯度和硬动态箱 | 不是完整RangedIK，不使用minimax/forcing；不保留其其他运动/碰撞软任务 |

所有方法使用同一公共1mm/.5°验收、实际previous_q与dt=.02s。学习、未来目标、
witness初值、候选池和IK兜底均未引入。点查询每次重置；应用接受才反馈、失败保持，
目标索引继续。原生终止状态和任务验收分别保存。

**SQP数值端点修订公开保留。** 原始SLSQP的 `1−||e_b||²≥0, ftol=1e−8`
开发记录大多内部成功、却在公共边界外微小距离。未将此当成零覆盖对手。
主比较前，仅在270-query开发集固定内部裕量 `1−||e_b||²≥1e−7`，仍maxiter100、
ftol1e−8、20ms软期限。它比公共合同略严格（位置半径约少5e−11m），公共容差和
verifier不变；不是完全相同的内部可行集，也不是新任务要求。没有参数网格，
两个机器人同设置。原零裕量记录、修訂后记录及源码快照全部保留。
修订后开发两机器人均270/270，各三遍；详见
[development_comparison.csv](../outputs/task_balance_comparison/reports/development_comparison.csv)。

**RangedIK身份核验。** 官方commit `1c48d2ae408b4e024ee037641aac1e728267984e`，
原环境隔离Release编译，未改官方源码，例程正常返回10个有限关节向量。
返回顺序 `right_j0…right_j6`、base_link→right_hand；保存向量在作者FK与本项目
URDF FK间最大位置差5.56e−16m、旋转差6.56e−16rad。
[官方例程日志](../outputs/task_balance_comparison/references/official_demo.json)、
[只读FK复验](../outputs/task_balance_comparison/references/official_fk_audit.json)均保留。
Sawyer仅为官方依赖smoke，未加入机器人性能研究。官方原生分量范围及小范围
Groove分支与当前合同不同；本轮**没有完整RangedIK的同合同性能结论**，也没有以
其历史适配失败填零成功。范围对照完成的是明确命名的损失移植。

## 3. 补充共同输入查询：全样本结果

每机器人2160个已观察查询，240个anchor、每anchor九格；三遍交错，成功数三遍
完全一致。不是fresh test。下表成功数分母均2160；均值与分位数含全部失败和晚到调用。
全部主查询调用均在20ms内，因此本批within20成功数与合格成功数相同。

| 机器人 | 方法 | 合格数 | 平均ms | P50/P95/P99 ms | 合格命令>90%容差 |
|---|---|---:|---:|---|---:|
| Panda | relative | 2160 | 0.3963 | 0.331 / 0.590 / 0.658 | 6.67% |
| Panda | gn | 2150 | 0.4210 | 0.358 / 0.581 / 0.652 | 6.28% |
| Panda | tight | 2160 | 0.8119 | 0.330 / 2.499 / 3.119 | 7.55% |
| Panda | clarabel | 2160 | 0.4840 | 0.332 / 0.813 / 0.874 | 7.36% |
| Panda | range_loss_matched | 2108 | 2.3625 | 2.081 / 4.845 / 6.532 | 38.47% |
| Panda | direct_sqp | 2160 | 2.0712 | 2.180 / 2.760 / 2.974 | 99.95% |
| Panda | fixed_qp1 | 2150 | 0.3972 | 0.331 / 0.563 / 0.631 | 6.28% |
| Panda | fixed_qp2 | 2157 | 0.4650 | 0.331 / 0.766 / 0.848 | 6.81% |
| UR5e | relative | 2160 | 0.3874 | 0.400 / 0.541 / 0.616 | 6.57% |
| UR5e | gn | 2148 | 0.4068 | 0.395 / 0.529 / 0.592 | 6.05% |
| UR5e | tight | 2160 | 0.8374 | 0.543 / 2.263 / 2.605 | 7.69% |
| UR5e | clarabel | 2160 | 0.4928 | 0.593 / 0.760 / 0.890 | 7.13% |
| UR5e | range_loss_matched | 2126 | 2.1611 | 1.874 / 4.578 / 6.180 | 36.83% |
| UR5e | direct_sqp | 2160 | 2.0592 | 2.151 / 2.762 / 2.968 | 100.00% |
| UR5e | fixed_qp1 | 2148 | 0.3810 | 0.373 / 0.511 / 0.580 | 6.05% |
| UR5e | fixed_qp2 | 2156 | 0.4612 | 0.503 / 0.695 / 0.792 | 6.96% |

[完整主表](../outputs/task_balance_comparison/reports/points_main.csv)另含实际两块误差分位数/
最大值、位移利用率、原生状态、失败原因、外层更新和QP数。
主方法并非所有分位数都比原GN或fixed1小；对SQP的优势是本实现成本，不是扩大
其覆盖。SQP贴边是最小位移可行解目标的可解释结果，仍然合法，不能以此叫作违约。

### 全部九格，不能只展示压力格

每格每机器人240查询。relative、tight、clarabel、SQP在**所有格均240/240**。
以下GN和fixed1覆盖相同；数字为成功数，括号为主方法/SQP平均ms。

| 位移 / α | Panda GN/F1 | F2 | 范围损失 | Panda 主/SQP ms | UR5e GN/F1 | F2 | 范围损失 | UR5e 主/SQP ms |
|---|---:|---:|---:|---|---:|---:|---:|---|
| regular / 0 | 240 | 240 | 240 | .279 / 1.456 | 240 | 240 | 240 | .268 / 1.460 |
| regular / .5 | 240 | 240 | 240 | .285 / 1.490 | 240 | 240 | 238 | .271 / 1.489 |
| regular / .95 | 240 | 240 | 238 | .290 / 1.557 | 240 | 240 | 238 | .276 / 1.539 |
| high / 0 | 240 | 240 | 238 | .392 / 2.311 | 240 | 240 | 239 | .401 / 2.326 |
| high / .5 | 240 | 240 | 235 | .392 / 2.322 | 240 | 240 | 237 | .405 / 2.337 |
| high / .95 | 240 | 240 | 234 | .404 / 2.364 | 239 | 240 | 235 | .414 / 2.369 |
| boundary / 0 | 240 | 240 | 235 | .508 / 2.375 | 240 | 240 | 240 | .474 / 2.333 |
| boundary / .5 | 240 | 240 | 236 | .482 / 2.365 | 240 | 240 | 235 | .462 / 2.332 |
| boundary / .95 | 230 | 237 | 212 | .534 / 2.403 | 229 | 236 | 224 | .515 / 2.349 |

每方法在九格和27个几何细分的完整覆盖、成本、误差、失败在
[points_cells.csv](../outputs/task_balance_comparison/reports/points_cells.csv)，无类别删除。
主方法相对GN在压力格为+4.17/+4.58个百分点，**全样本仅+0.463/+0.556个百分点**。
此前22个GN漏解逐UID分解见 [prior_22_misses.csv](../outputs/task_balance_comparison/reports/prior_22_misses.csv)。

### 聚类配对区间

先在query UID内平均三遍，再以anchor聚类、按三类几何分层，4000次bootstrap。
下表是主方法/对照平均时延比及描述性未校正95%区间；不把重复当独立query。

| 对照 | Panda 比值[95%区间] | UR5e 比值[95%区间] |
|---|---|---|
| GN | .941 [.923,.957] | .952 [.935,.967] |
| 紧精度 | .488 [.469,.507] | .463 [.446,.480] |
| Clarabel | .819 [.809,.830] | .786 [.778,.795] |
| 范围损失 | .168 [.162,.174] | .179 [.173,.186] |
| SQP | .191 [.188,.195] | .188 [.185,.191] |
| 固定1次 | .998 [.970,1.022] | 1.017 [.995,1.036] |
| 固定2次 | .852 [.842,.862] | .840 [.828,.850] |

相对固定1次的覆盖差为Panda **+0.463 pp [.185,.787]**、UR5e **+0.556 pp [.278,.880]**；
相对固定2次为 **+0.139 pp [0,.324] / +0.185 pp [.046,.370]**。
固定1次的成本区间跨1不是等效证明；Panda固定2次覆盖事件少，区间含0不是无增量证明。
与SQP、紧精度、Clarabel的观测覆盖差为0；经验bootstrap的零宽区间也不证明普遍等效。
[所有格配对区间](../outputs/task_balance_comparison/reports/paired_effects.csv)保留全量。

## 4. 固定工作量是否已足够

| 机器人 | 方法 | 平均局部QP调用/查询 | 平均外层检查 | 成功数 |
|---|---|---:|---:|---:|
| Panda | relative | 1.457 | 1.429 | 2160 |
| Panda | fixed1 | 1.487 | 1.488 | 2150 |
| Panda | fixed2 | 1.894 | 1.442 | 2157 |
| Panda | tight | 3.921 | 1.410 | 2160 |
| UR5e | relative | 1.562 | 1.534 | 2160 |
| UR5e | fixed1 | 1.573 | 1.574 | 2148 |
| UR5e | fixed2 | 2.102 | 1.544 | 2156 |
| UR5e | tight | 4.699 | 1.504 | 2160 |

主方法分别有42.50%/52.69%的调用使用了相对进展出口。与fixed2相比，本批平均
时间少14.77%/16.01%，并多恢复3/4个查询；不是所有输入都多算。
在这7个fixed2未恢复而主方法恢复的输入上，主方法某个局部问题使用了3–13个
加权QP。UID、调用数和退出原因见
[stopping_queries.csv](../outputs/task_balance_comparison/reports/stopping_queries.csv)；
它说明**这些执行中**更多局部工作有用，不是证明任何替代算法都至少需要该次数。

同缓存tight以及Clarabel也全部恢复，说明任务恢复不是相对gap独有；其作用是
在本实现中减少求局部紧解的额外工作。相对fixed1覆盖提高而成本比接近1，不能
宣称稳定提速。以上归因不混入TRAC原生集合差异。

## 5. 一个应用来源：合成平面观测扫描

仓库未找到可用的真实Cartesian日志/规划器路径，采用明确标注的**应用型合成验证**。
12条/机器人、150帧/条，A=10/20/30mm×四个平面方向；固定姿态，Cartesian
Lissajous扫描曲线见协议。初始FK只定义工位，不从随机FK轨迹或成功输出生成后续目标。
输入在任何方法结果前固定并随协议提交。1mm/.5°是软件假设，不是工业公差。
没有相机、碰撞、接触或硬件实验，也不估计现场收益。

输入身份和准确目标：[Panda](../outputs/task_balance_comparison/application/inputs_panda.json)、
[UR5e](../outputs/task_balance_comparison/application/inputs_ur5e.json)。
24条全部保留。后续可行性并非预先保证；本批没有几何失败，但不外推为普遍可行。

各方法两机器人每遍均 **TSR=12/12**；下表累计时间为12条全部帧、三遍平均，
加速度单位rad/s²。DTSR列按三遍完整完成数列出。

| 机器人 | 方法 | DTSR20各遍/12 | 累计ms | P50/P95/P99 ms | 加速度RMS | >90%容差 |
|---|---|---|---:|---|---:|---:|
| Panda | relative | 12/12/12 | 390.57 | .241/.256/.298 | 2.846 | 3.22% |
| Panda | GN | 12/12/12 | 428.23 | .266/.280/.314 | 2.846 | 3.22% |
| Panda | fixed1 | 12/12/12 | 388.65 | .241/.255/.295 | 2.846 | 3.22% |
| Panda | fixed2 | 12/11/12 | 406.32 | .241/.256/.295 | 2.846 | 3.22% |
| Panda | 范围损失 | 12/12/12 | 1257.48 | .655/1.235/1.649 | 4.266 | 31.89% |
| Panda | SQP | 12/12/12 | 1943.89 | 1.084/1.503/1.637 | 0.523 | 99.11% |
| UR5e | relative | 12/12/12 | 381.00 | .236/.249/.286 | 3.068 | 3.11% |
| UR5e | GN | 12/12/12 | 421.83 | .264/.275/.298 | 3.068 | 3.11% |
| UR5e | fixed1 | 12/12/11 | 396.88 | .236/.249/.284 | 3.068 | 3.11% |
| UR5e | fixed2 | 12/12/12 | 380.40 | .236/.249/.287 | 3.068 | 3.11% |
| UR5e | 范围损失 | 12/12/12 | 1260.51 | .661/1.216/1.619 | 4.669 | 31.33% |
| UR5e | SQP | 12/12/12 | 2003.97 | 1.119/1.580/1.729 | 0.550 | 99.11% |

全部帧均合格。两个约49–50ms的合法晚到调用未删除：Panda fixed2 的
`67fc7835…` repeat1/frame106，UR5e fixed1 的 `bc0747a7…` repeat2/frame106。
完整UID、实际时间和原始行号在
[application_failures_timeouts.json](../outputs/task_balance_comparison/reports/application_failures_timeouts.json)。
这两次运行扰动不能归因于算法稳定时限劣势。

**应用没有激活需要进一步平衡的局部模型。** 两机器人relative全部帧的
`subproblems`为空，与GN命令最大差为0；新增minimax/进展逻辑没有带来不同命令。
相对原GN的计时差不能因此归为双块目标收益。与同缓存fixed1/2也没有完成或命令
优势。另一方面，SQP最小位移命令的加速度明显较低，同时更贴近允许边界、成本
更高；主方法不是运动平滑性赢家。

每UID三遍平均后统计，12个完整序列为独立单位。全部指标、首次失败/成功前缀、
实际误差最大值和配对区间见
[application_main.csv](../outputs/task_balance_comparison/reports/application_main.csv)、
[application_sequences.csv](../outputs/task_balance_comparison/reports/application_sequences.csv)、
[application_paired_effects.csv](../outputs/task_balance_comparison/reports/application_paired_effects.csv)。
几何完成恢复/损失UID均为空；DTSR扰动如上。没有把150帧当150个独立样本。

## 6. 四项验收判断

| 问题 | 判断 | 证据与具体范围 |
|---|---|---|
| 相对范围感知/直接约束对手还有覆盖—成本增量？ | **支持本批匹配参照下的增量；完整RangedIK结论证据不足** | 比SQP同覆盖但约少81%平均成本；比移植范围损失恢复52/34查询且更快。损失/优化器/软任务均与完整RangedIK不同，不能将此改写为全面胜出 |
| 相对fixed1/fixed2，进展判断有实际价值？ | **支持本批受限查询的数值工作控制价值** | fixed1仍漏10/12，fixed2漏3/4；relative在全样本减少固定2次和紧精度的平均工作。Panda额外3个事件的不确定性保留。已有gap原则本身不原创 |
| 应用来源上是否证明新增跟踪收益？ | **不支持新增TSR/命令收益；真实现场价值证据不足** | 所有方法12/12，主方法不进入额外平衡且与GN命令相同；只有应用型合成来源，SQP更平滑 |
| 去掉成熟原则后还有什么具体设计由数据支持？ | **支持这套任务化组合的可测量实现效果；新优化理论不成立** | 两公共容差块在硬动态箱内协调、首命令任务早停、相同局部问题按已有界控制继续工作，在少量冲突输入避免固定预算不足，又不普遍付紧解成本。实验支持这项组合价值，不证明先例不存在、全局收敛或闭环保证 |

旧普通点查询原GN2000/2000、旧完整轨迹无新增完成、旧压力格增益集中的事实
不变。本轮没有总pass/fail gate，没有扩大样本、添加模块或先写正面论文摘要。

## 7. 原始记录、图表和可复算入口

结果根目录：[outputs/task_balance_comparison](../outputs/task_balance_comparison/)。
主入口 [run_task_balance_comparison.py](../scripts/run_task_balance_comparison.py)，
薄对照模块 [task_balance_comparison.py](../src/confik/task_balance_comparison.py)，
固定配置 [task_balance_comparison.yaml](../configs/task_balance_comparison.yaml)。

- 运行协议/输入hash/顺序：`protocol/frozen_before.json`、`comparison_seal.json`、六个order文件；原任务包完整保存。
- 点原始记录：`points/points_{panda,ur5e}/records.jsonl.gz`；实际q及原生状态均保留。
- 应用原始记录：`application/application_{panda,ur5e}/records.jsonl.gz`；每帧包含真实previous、target、返回q及反馈接受状态。
- 恢复/损失命令：[command_difference_index.csv](../outputs/task_balance_comparison/reports/command_difference_index.csv)，包含双方q、准确输入、验收及JSONL行号；不是借用别的方法previous。
- 全失败/超时：`point_failures_timeouts.csv`、`application_failures_timeouts.json`；原生/任务四象限：`native_task_matrix.csv`。
- 停止与工作量：`stopping_comparison.csv`、`stopping_queries.csv`；所有九格和27细分：`points_cells.csv`。
- 命令只读复验：`point_replay.json`、`application_replay.json`，共168480个主比较/应用调用，合格命令违约0；复验不运行IK。4320个原query witness亦重新验证。
- 主方法/verifier前后hash：`protocol/frozen_before.json`及`reports/frozen_after.json`完全一致，主源码、旧证据、旧论文无改动。

三张图均由保存记录生成，含可编辑SVG/PDF与PNG预览：

1. [覆盖—时间](../outputs/task_balance_comparison/reports/coverage_time.pdf)：全样本的anchor聚类配对差/比及95%区间。
2. [固定工作与进展规则](../outputs/task_balance_comparison/reports/stopping_work.pdf)：全样本及预定义压力格的覆盖、QP工作量、完整平均时间。
3. [实际应用轨迹](../outputs/task_balance_comparison/reports/application_actual.pdf)：预定scan08/repeat0的原始反馈FK与真实误差，未按失败选图；重合轨迹不意味着隐藏方法。

只重建表图、不调用IK：

```bash
PYTHONPATH=tmp/crik_dependencies/python:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/report_task_balance_comparison.py \
  --out /tmp/task_balance_review_rebuild_unique
```

输出以独占创建保护，复算请使用新目录。实际运行命令、源hash、清单及阶段提交在
`outputs/task_balance_comparison/review_manifest.json`；不要对冻结目录重新prepare或覆盖。
数值代码与协议在结果前提交；报表脚本为只读后处理，不改变求解结果。
文献核验、嵌套统计和source-backed图表分别遵循nature-academic-search、
nature-statistics与nature-figure技能；这些约束用于避免归属误判、伪重复与图表脱离数据。

**本轮已停止：不宣布创新性充分或期刊可录用，等待用户与ChatGPT验收。**
