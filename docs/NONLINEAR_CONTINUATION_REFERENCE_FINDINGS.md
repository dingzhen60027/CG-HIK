# 固定当前候选的非线性有限视域续接参照

## 判断：先区分求解器漏解，不应先增加构型选择准则

**原 TRAC-IK 失败的候选中，确实存在已经找到并逐帧验证的合法续接。** 固定原来的当前配置，不更换候选、不移动目标、不减速：`state_14` 六个候选全部找到 30 帧 witness；`state_18` 六个原 0/5 候选中找到三个；TRAC-IK 来源 `case_01` 的两个原/精修中心与六个既有等位姿候选全部找到。其中原 0/5 的 11 个候选有 8 个获得 witness，但这些候选属于三个已观察状态，不是 11 个独立样本，也不是新方法的 8/11 成功率估计。

最直接的证据来自 **state_14 candidate_04 真实第 101 帧接受状态 → 原第 102 帧目标**：原 TRAC-IK 返回 -3，而非线性单帧参照找到合法命令，实际最大允许步长比为 **0.048256**。这里没有借用 candidate_01 的 previous_q。更重要的是，预先规定的 DLS 初始化本身也能从该失败输入找到合法命令；因此这个具体失败没有显示“必须提前更换候选”或“必须长视域预览”。

**下一步若继续，应优先审视逐帧求解的搜索过程及其内部终止精度与公共合同的匹配，再决定是否需要候选选择。** 本轮没有修改该过程。`case_01` 显示联合优化能为普通续接与 DLS 初始化均失败的候选构造延拓，但并未证明预览是唯一恢复途径。已有预测 IK 已经使用移动视域和轨迹优化；本轮建立的是具体输入上的能力参照与漏解证据，不是新的预测 IK 算法。

## 1. 样本、固定问题与执行范围

证据基线：`9f00d6160b3e556a2201a401975e2e37447edbc0`。全部数据已被观察，明确属于 **mechanism development，不是 fresh/formal test**。旧论文、模型、阈值、runtime、verifier 和旧结果均未修改。本轮不调用 TRAC-IK，不生成新轨迹、候选或评分指标。

| 状态 | 固定候选 | 固定当前帧 / 后续帧 | 用途 |
| --- | --- | --- | --- |
| state_14 | candidate_00–05 | 74 / 75–104 | 原结果混合，经典指标选中的 04 为 0/5 |
| state_18 | candidate_00–05 | 74 / 75–104 | 原全部 0/5 |
| case_01_pre_failure | candidate_00、05；matched_00–05 | 33 / 34–63 | 原 TRAC-IK 来源等位姿成功反转 |
| state_13 | candidate_00–05 | 74 / 75–104 | 全成功对照，UID 排序第一 |
| state_08 | candidate_00–05 | 74 / 75–104 | 全成功对照，UID 排序第二 |
| state_14 第 102 帧专项 | 04、01 各自 repeat 0 的真实 q_101 | 101 / 102 | 同目标、不相同 previous_q 的输入检查 |

两个对照按上一轮全部候选均成功状态的完整 UID 升序选取，未按本轮优化结果筛选。UID 分别为 `0041eb55dd40437ac17fe39ad8e73df25d7db92ccc7ab532775994d081047948` 和 `06ca0c5d8d0cb31f50338aee261f6c596974205eb08300b4526361bb949a0c56`。`case_01` 仅复用成功反转涉及的两个中心及六个匹配候选，没有重新生成候选。全部 UID、当前目标、实际 previous_q、固定 q_t、后续目标与旧 run ID 见 [inputs.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/inputs.json)。

每个主候选独立求解 L∈{1,5,10,30}，两个专项输入仅求 L=1。共 **32 个主候选、34 个固定输入、130 个候选×视域问题**。各问题严格使用三种初始化，各执行一次，共 **390 次搜索**；不是搜索随机性的五次重复。表中的旧 0/5–5/5 仅是已经存在的 TRAC-IK 状态内重复，未重新运行，不作配对随机种子或统计独立性声明。

### 固定的非线性问题

变量为 q_(t+1), …, q_(t+L), rho≥0，最小化 rho。**q_t 是只读常量，不在优化变量中**。每一步要求：

- 真实 FK 的位置误差范数 ≤ 0.001 m；姿态误差范数 ≤ 0.00872664626 rad；
- 原 Panda URDF 关节范围；
- 每关节相邻帧差的绝对值 ≤ rho × (v_i × 0.02 + 0.0001) rad。

固定目标序列、dt=0.02 s、原速度限值和公共 verifier。rho 是允许步长的参照倍率，不用于修改执行时间；只有 **真实序列 rho≤1 且所有帧通过原 verifier** 才保存合法 witness。rho>1、未满足位姿约束或局部搜索未收敛，只记“本次搜索未找到”，不解释为数学不可行。本文所有 rho 都是找到的可行序列给出的上界，不是全局最优值。

实际后端是项目的 **URDFKinematics**，不是根据环境名称推断的 Pinocchio。位置残差为世界坐标 p_target−p(q)，姿态残差为同一世界坐标轴下 Log(R_target R(q)^T)。姿态导数使用 SO(3) 右 Jacobian 的逆与世界几何 Jacobian；位置、姿态各按原容差缩放。关节变量按 v_i dt+velocity_tolerance 缩放，所有 7L 个自由度同时参与优化。解析残差与完整约束 Jacobian 均已通过有限差分测试。

采用 SciPy 1.17.0 SLSQP，每个初始化固定最多 100 次迭代、15 秒主要迭代回调检查限额、ftol=1e−10。三个初始化为：当前 q_t 保持；**该候选 repeat 0** 的既有成功前缀后保持末状态；从该候选出发的既有 DLS 逐目标预测（每目标最多 25 次迭代）。DLS 初始化可能违反速度或位姿约束，只是初值，必须重新验收。不同视域间没有 warm start，没有使用未来 q_ref，也没有移植其他候选的起点。三个初始化对所有候选规则和预算相同，运行顺序用固定 seed 打乱，结果出现后未追加搜索。

运行前将平方归一化位姿球固定内缩 1e−8，仅作浮点边界构造；没有放宽公共容差。最终一律用原非线性 FK 与 verifier 验收。优化器声明的 rho、对应步长约束缺陷、最后迭代序列以及真正找到的最小实际步长比均分别保留；不会用低 rho 的非法位姿序列充当可行参照。

## 2. 候选×视域结果

以下数值是三个预定初始化中找到的最小**实际** rho；所有数值单元均有完整、逐帧合法 witness。**NF = 该固定搜索没有找到位姿/关节范围可行的全序列，绝不是不可行证明**。小数仅为展示舍入；全精度、状态、约束残差、调用数与各次耗时保存在 [候选×视域表](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/candidate_horizon_table.csv)、[搜索明细](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/horizon_search_details.json)及 [原始 runs](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/runs/)。

| 状态 / 候选 | 旧普通 TRAC-IK 完成 | L=1 | L=5 | L=10 | L=30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_14 / 00 | 4/5 | 0.063885 | 0.093168 | 0.098803 | 0.113050 |
| state_14 / 01 | 5/5 | 0.064116 | 0.093513 | 0.099251 | 0.115431 |
| state_14 / 02 | 1/5 | 0.063661 | 0.092827 | 0.098387 | 0.109569 |
| state_14 / 03 | 5/5 | 0.064342 | 0.093861 | 0.099735 | 0.117755 |
| state_14 / 04 | 0/5 | 0.063447 | 0.092489 | 0.097999 | 0.107145 |
| state_14 / 05 | 5/5 | 0.063977 | 0.093306 | 0.098978 | 0.113999 |
| state_18 / 00 | 0/5 | 0.102498 | 0.120233 | 0.126819 | NF |
| state_18 / 01 | 0/5 | 0.102756 | 0.120715 | 0.127369 | 0.192232 |
| state_18 / 02 | 0/5 | 0.102230 | 0.119798 | 0.126266 | NF |
| state_18 / 03 | 0/5 | 0.103002 | 0.121243 | 0.127916 | 0.189530 |
| state_18 / 04 | 0/5 | 0.101951 | 0.119399 | 0.125709 | NF |
| state_18 / 05 | 0/5 | 0.102602 | 0.120419 | 0.127040 | 0.203211 |
| state_13 / 00 | 5/5 | 0.109270 | 0.136012 | 0.138586 | 0.185391 |
| state_13 / 01 | 5/5 | 0.109647 | 0.136953 | 0.139478 | 0.184509 |
| state_13 / 02 | 5/5 | 0.108914 | 0.135378 | 0.138143 | 0.186324 |
| state_13 / 03 | 5/5 | 0.110045 | 0.137958 | 0.140514 | 0.183683 |
| state_13 / 04 | 5/5 | 0.108582 | 0.134959 | 0.138000 | 0.187304 |
| state_13 / 05 | 5/5 | 0.109418 | 0.136351 | 0.138877 | 0.185032 |
| state_08 / 00 | 5/5 | 0.167224 | 0.187548 | 0.196934 | 0.222911 |
| state_08 / 01 | 5/5 | 0.167344 | 0.187624 | 0.197014 | 0.223094 |
| state_08 / 02 | 5/5 | 0.167110 | 0.187487 | 0.196926 | 0.222728 |
| state_08 / 03 | 5/5 | 0.167470 | 0.187713 | 0.197156 | 0.223276 |
| state_08 / 04 | 5/5 | 0.167004 | 0.187439 | 0.196929 | 0.222545 |
| state_08 / 05 | 5/5 | 0.167271 | 0.187577 | 0.196955 | 0.222984 |
| case_01 / candidate_00 | 0/5 | 0.131082 | 0.444596 | 0.444596 | 0.444596 |
| case_01 / candidate_05 | 0/5 | 0.131080 | 0.444596 | 0.444596 | 0.444596 |
| case_01 / matched_00 | 0/5 | 0.129569 | 0.348594 | 0.348900 | 0.348900 |
| case_01 / matched_01 | 5/5 | 0.132193 | 0.407184 | 0.410332 | 0.410332 |
| case_01 / matched_02 | 5/5 | 0.127641 | 0.318123 | 0.323509 | 0.323509 |
| case_01 / matched_03 | 0/5 | 0.129568 | 0.348594 | 0.348900 | 0.348900 |
| case_01 / matched_04 | 5/5 | 0.132191 | 0.407184 | 0.410332 | 0.410332 |
| case_01 / matched_05 | 5/5 | 0.127639 | 0.318123 | 0.323509 | 0.323509 |

### 初始化与优化的贡献必须分开

**state_14：**所有六个 DLS 初始化本身已经是合法 30 帧序列，随后优化仅进一步降低实际步长比。04 的 DLS 初始 rho 为 0.126333，搜索后为 0.107145；旧 TRAC-IK 为 0/5，但“构型在公共合同下无法续接”的解释已被合法序列否定。00、02 也有旧失败重复，其固定当前配置同样存在延拓。

**state_18：**01、03、05 的 DLS 初始化已经合法，搜索保存的最好 rho 分别为 0.192232、0.189530、0.203211；00、02、04 的三个初始化均未给出合法 30 帧序列，SLSQP 均达到 100 次迭代。其 DLS 启动的最后迭代位置超限分别约 18.56、17.95、7.50 µm，姿态也仍超限，因此即使实际步长比很低也不能验收。该现象反映**当前数值搜索的完成差异**，没有形成候选固有可行性差异的证明。

**case_01：**四个原 0/5 候选为 candidate_00、candidate_05、matched_00、matched_03。三种原始初始化均没有给出合法全窗口；DLS 启动的联合优化为四者找到合法 30 帧序列。它们不需要改换当前接受配置即可延拓。其余四个原 5/5 匹配候选的合法初始化和既有 witness 已知，不当作新恢复。完整初始化检查见 [initialization_checks.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/initialization_checks.json)。

**视域结论：**全部 32 个候选在 L=1、5、10 均找到合法序列；state_18 到 L=30 才出现“找到 / 未找到”的搜索结果分化，但短视域成功不能保证整个窗口完成，长视域 NF 也不能证明当前构型不可续接。state_14 和 case_01 在 L=30 仍全部存在延拓，不支持用这批结果宣称“坏构型必须提前淘汰”。更不能把不同上界的小幅排序当作已经建立的新评分指标。本轮没有识别出一个能够证明“必须使用长视域才能辨别物理可行性”的候选对。

## 3. state_14 第 102 帧：同目标、各自真实 previous_q

专项固定 repeat 0，不挑选最有利重复。candidate_04 第 101 帧真实命令为：

```text
[2.3971239410544025, 0.07041267205320365, -1.616130595673609,
 -0.3758544732182841, -1.573361785194308, 0.32722790204966923,
 0.2610436886855599]
```

candidate_01 在第 101 帧的真实 previous_q **不同**；其完整值和两个新命令均存于 [首次失败输入对照表](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/first_failure_input_table.json)。两者使用完全相同的第 102 帧目标，不能称为“同输入候选对”，也没有用 01 的成功命令替代 04 的命令。

| 第 101 帧来源 | 原第 102 帧结果 | 原返回位置 / 姿态误差 | 非线性 L=1 新命令误差 | 新实际 rho | 原 verifier |
| --- | --- | --- | --- | ---: | --- |
| candidate_04，自身 previous_q | native -3；有限返回值；position_tolerance 拒绝 | 3.122493 mm / 0.005133004 rad | 0.999999995 mm / 0.002768903 rad | 0.048256 | 通过 |
| candidate_01，自身 previous_q | native +1；接受 | 0.002100 mm / 0.000004065 rad | 0.999999995 mm / 0.004687388 rad | 0.048410 | 通过 |

新命令的位置误差接近原 1 mm 边界，是最小步长目标利用公共误差球的结果，不是更精确的解，也没有修改验收容差。完整残差向量、URDF 限位余量、逐关节速度比和精确命令保存在 witness。

04 三个启动全部找到合法下一帧命令，整个三启动成本 **11.730 ms**；01 为 **13.049 ms**。这些是离线三启动搜索成本，不是与旧单次 5 ms TRAC-IK 公平等预算竞赛。04 的 DLS 初始化在优化前已经合法：位置误差约 **0.170185 mm**、姿态误差约 **0.000596178 rad**、实际 rho **0.087938**。所以该输入上连 L=1 的联合最优化也不是合法性发现的必要条件。

**可以下的确定结论是：此固定输入存在合法解，但原求解器未返回可接受命令。** 不能据此推断整个 TRAC-IK 库一般失效。旧适配的内部 epsilon=1e−5/笛卡尔分量、零额外 Twist bounds，原本就比公共位置/姿态范数合同严格；该事实已在旧代码和测量说明记录。本轮未调整内部精度、搜索域、初始化或预算作单因素消融，因而尚不能把这次漏解唯一归因为“精度过严”、某个求解器内核或闭环分支漂移。值得首先核查的是**固定实际状态下，当前合同允许的解如何被现有搜索与内部成功判据遗漏**。

## 4. Witness、数值状态与真实成本

共 130 个候选×视域问题中 127 个找到合法序列；390 次初始化搜索产生 **339 份逐帧合法 witness**。51 次搜索虽未返回优化成功状态，却已经找到并保存合法序列；反之，不会因优化器返回小 rho 就忽略未满足的位姿约束。返回码为 0：288 次，9（迭代上限）：88 次，8（线搜索方向问题）：12 次，3（LSQ 子问题迭代上限）：2 次。没有触发 15 秒限额。所有返回消息、最后迭代和最好序列均保留，不只保留成功试验。

代表新增 witness：

- [state_14 candidate_04，固定第 74 帧后 30 帧](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/successful_witnesses/state_14_candidate_04_L30_dls_prediction.json)：旧 0/5，DLS 初始化已合法；
- [state_18 candidate_01，固定第 74 帧后 30 帧](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/successful_witnesses/state_18_candidate_01_L30_dls_prediction.json)：旧 0/5，DLS 初始化已合法；
- [case_01 原中心 candidate_00，固定第 33 帧后 30 帧](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/successful_witnesses/case_01_candidate_00_L30_dls_prediction.json)：原初始化未通过，联合优化后合法；
- [case_01 matched_00 的 30 帧延拓](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/successful_witnesses/case_01_matched_00_L30_dls_prediction.json)：原 0/5 匹配候选的合法延拓；
- [state_14 candidate_04，真实第 101→102 帧](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/successful_witnesses/failure_input_candidate_04_L1_dls_prediction.json)：确切失败输入上的合法命令。

每份 witness 包含固定 q_t、完整 q_(t+1:t+L)、每帧原目标、dt、实际 previous_q、真实残差向量和原 verifier 结果。计算步长、FK 和 verifier 时沿**所保存序列自身**逐帧推进；没有在拒绝节点换用另一条路径、保持旧状态修补序列或更换 q_t。

| L | 问题数 | 三启动总时间中位数 / P95（ms/问题） | 单启动总时间中位数（ms） | 实际 FK 方法调用 | 几何 Jacobian 方法调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 34 | 17.258 / 61.022 | 5.442 | 4,107 | 1,547 |
| 5 | 32 | 131.700 / 299.733 | 45.581 | 25,052 | 16,257 |
| 10 | 32 | 499.386 / 949.368 | 179.848 | 72,942 | 55,622 |
| 30 | 32 | 5,169.095 / 6,848.484 | 1,630.922 | 358,510 | 269,763 |

总计算 **182.711 秒、460,611 次 FK 方法调用、343,189 次几何 Jacobian 方法调用**。另外独立记录目标函数 25,450 次、目标梯度 19,102 次、位姿约束 26,642 次及其 Jacobian 19,102 次、步长约束 25,862 次及其 Jacobian 19,102 次。缓存命中不冒充新 FK；Jacobian 本身内部的几何运算不能再当作可观察的 TRAC-IK FEV。原生 TRAC-IK 内部 FEV 仍未知，本文不作二者 FEV 比值。

单启动总计时包含初始化生成/复制、问题构造及初始检查、SLSQP、搜索过程中的候选验证、最终最好与最后序列的完整 FK/验证；各部分另存。读取旧数据、写 JSON、事后独立复核不计入 solver 时间。历史成功前缀初始化只测复制既有数据，不把过去 TRAC-IK 求解当作免费在线获取方式。测量使用共享工作站、三个数值库线程环境均为 1，并非隔离延迟基准。**L=30 的秒级离线计算是能力参照，没有展示可直接用于 20 ms 控制周期的低成本方法。** [reference_summary.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/reference_summary.json)保留完整分阶段分布。

## 5. 与三项最近工作的关系

只核查用户指定的三项，不扩展文献范围，也不把“使用未来信息”当作创新点。

| 工作与核查来源 | 已解决的问题 | 本轮对应现象与具体区别 |
| --- | --- | --- |
| Schuetz 等，*Predictive online inverse kinematics for redundant manipulators*，ICRA 2014；[TUM 作者机构条目与摘要](https://portal.fis.tum.de/en/publications/predictive-online-inverse-kinematics-for-redundant-manipulators/)，DOI 10.1109/ICRA.2014.6907600 | 针对瞬时 IK 的局部性与高关节速度，采用移动视域最优控制，并展示冗余机械臂实时计算。 | 预览处理关节运动已有直接先例。本轮区别是固定已接受候选，用原 timed command 合同构造 witness，检验一次具体 TRAC-IK 失败是否遗漏合法解；不是提出移动视域 IK。摘要不足以判断其精确约束是否覆盖本轮全部合同。 |
| Origanti、Danzglock、Kirchner，*Look Ahead Optimization for Managing Nullspace in Cartesian Impedance Control of Dual-Arm Robots*，SII 2025；[DFKI 作者机构摘要](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15449) | 双臂 KUKA IIWA 笛卡尔阻抗控制中，以序列 QP 的视域优化处理零空间、奇异性、关节限位与双臂间碰撞。 | 零空间预览和约束管理不是本轮首次提出。本轮仅为单 Panda、固定当前 q、有限速度合同的可找到延拓参照，不评价阻抗控制或碰撞性能；没有证据宣称该工作无法处理当前问题。 |
| Morgan、Millard、Sukhatme，*CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning*，ICRA 2024；[作者全文 §II–III](https://arxiv.org/html/2309.09102v2) | 生成整条候选路径，离散全局搜索后作数值轨迹精修，处理位姿、关节范围、碰撞和相邻配置不连续性。 | 原文明确把时间参数化交给后续模块；本轮固定 dt 与目标时刻，禁止通过重新定时恢复，且当前 q_t 不可改。这个合同差异具体成立，但不构成对 CppFlow 能力的否定或新算法证明。 |

核查日期为 2026-09-07。文献检索工具不可用、OpenAlex 返回 429 后，按学术检索技能转用官方机构页面及作者原文；前两篇 IEEE 全文访问受限，仅完成题录/作者摘要核查，未作全文级“缺失某约束”的判断。CppFlow 的时间参数化说明已在原文问题定义中核对。统计规范在本报告中用于分开状态、候选、历史搜索重复与优化初始化；不把这些计数当作独立轨迹样本或普遍发生率。

**当前识别出的具体问题是“公共容差与速度合同下存在延拓，但冻结逐帧搜索没有输出”，不是“缺少未来可行性这一概念”。** 可讨论的研究问题是如何在当前实际配置固定、目标不能重新定时的条件下，使求解器更可靠地搜索公共合法集合，同时控制计算成本；本轮未证明这已超出已有预测 IK 的能力，也未证明需要预览、学习模型或新的构型评分。state_18 的三个 NF 仍是数值参照的未解部分，不自动追加预算或数据。

## 6. 复核、复现入口与停止点

初值、目标和测量代码在优化前封存于 [input_seal.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/input_seal.json)，SHA-256 为 `6da7fde031002d05c1c1c9d3b7860cec17ac973e126c686367698d950b90d17c`。使用历史成功 witness 完成 **84 项候选×视域问题自检**，没有优化调用；这只验证实现能识别已知合法序列，不作为新恢复结果。

独立只读复核检查了 390 次运行的固定 q_t、目标和计数一致性；对最好与最后迭代共 **8,844 帧**重新计算 FK/verifier，对全部 390 条初始化重新验收，核对 **339 份 witness、2,972 个 witness 帧**。接受合同违约为 **0**。全部 **647 个受保护旧文件**的 hash 复核通过，原始输入 seal 与测量代码 hash 不变。[verification.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/verification.json)记录完整数量与边界；测试涵盖数学实现、汇总逻辑及先前机制模块，共 **21 项通过**。

独立入口为 `confik.continuation_mechanism.nonlinear_reference`，阶段为 `prepare`、`selfcheck`、`run`；只读汇总入口为 `confik.continuation_mechanism.nonlinear_reference_reporting`。环境为 Python 3.12.13 / NumPy 2.4.4 / SciPy 1.17.0，运行时 `PYTHONPATH=src`，`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`。写入采用 exclusive create，已有结果存在时拒绝覆盖；当前已完成，不应再次执行测量入口。交付清单在 [delivery_manifest.json](../outputs/continuation_mechanism_study/nonlinear_continuation_reference/delivery_manifest.json)。

本轮停止于参照和结论：**确有漏解证据；不能把普通 TRAC-IK 的候选排序等同于物理可续接性排序；应先审视逐帧求解过程，而非自动增加构型选择器。** 未训练、未部署新策略、未重跑旧包、未扩大数据规模，也未修改论文。
