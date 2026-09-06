# Panda 同输入合法候选的续接机制研究

证据基线：`bea96172a955f5140d796d3615349c77f9d28e51`。本轮是限定状态的事后机制诊断，不是新方法、正式测试或整体性能评估。旧稿件、模型、阈值、solver/verifier、旧结果均保持不变；未重新运行补充实验包。数值测量完成后只进行验证和报告，没有按结果增加实验条件。

## 结论先行

1. **存在可重复的当前合法、后续表现不同。** 两个 CG-HIK 来源的失败前状态，在相同当前输入上，原候选与有界残差精修候选均通过原 verifier；随后统一使用 TRAC-IK，30 帧完成分别为 **0/5 与 5/5**。这是两个选定状态的续接证据，不是普遍发生率，也不是三条原轨迹均已恢复。
2. **TRAC-IK 作为统一后续求解器时，差异仍存在；但其自身失败轨迹上没有同样的完成反转。** 四条 TRAC-IK 来源的失败前状态中，所测合法候选均未完成整个窗口。部分候选只推迟一帧失败，不能称为成熟基线也已复现同等程度的候选效应。
3. **简单数值处理已解释并解决两处正面实例，但没有解决所有失败。** 有界精修救回上述两个窗口；预算从 5 ms 增至 100 ms、previous-state 优先初始化、最近合法解选择均未救回原候选。另发现浮点边界与 verifier 运算之间的实现问题；收紧到可表示的内部边界消除了这类拒绝，但未恢复其对应的整段续接。
4. **更值得刻画的是候选相关的下一目标步长需求，以及它与残差、冗余姿态和速度约束的耦合。** 不是先验的“分支漂移”标签。精修同时改变残差与姿态，当前实验没有分离二者的独立因果贡献，也没有证明需要学习模型。

## 1. 对象、输入和比较合同

只研究 Panda：三条 CG-HIK 相对 hard 丢失的 near-singular 轨迹、全部四条 TRAC-IK 未完成轨迹，以及 smooth/near-singular 各一条三个方法均成功的对照。对照按该类最小 UID 选取，不按新续接结果选取。完整 UID、目标、previous_q 和输入 hash 保存在 [protocol.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/protocol.json)。

每条轨迹两个位置：第一处分歧（均为第 0 帧），以及首次失败前一帧；成功对照使用同类失败位置的中位数匹配。共 **9 条轨迹、18 个状态、234 次候选尝试、95 个去重后的当前合法候选**。主比较每候选五次 5 ms TRAC-IK 续接；失败前/匹配位置的原候选、最近候选、精修候选另做 100 ms 和可表示内部边界对照。合计 **735 个重复窗口、21,800 次续接调用**。另有首次失败附近 ±2 帧的 **675 次 TRAC-IK 固定输入诊断调用**及 25 次冻结内部方法的阶段记录。不是新数据集训练或全包重跑。

每个窗口通常为后续 30 帧；第 124 帧状态仅剩 25 帧，按全部剩余帧评价。dt 固定 0.02 s，目标序列不改、不减速。完成要求窗口每帧得到被原 verifier 接受的命令。失败后保持上一接受状态，继续记录后续目标。这里评价几何命令合同，不将 100 ms 预算或求解后返回的合法命令称为满足 20 ms 实时截止期。

当前候选来源包括：原方法状态、其他方法状态、离线参考、同输入冻结 hard/CG-HIK、五次 TRAC-IK、previous-state 初始化的 25 步 DLS、原候选的有界残差精修。所有候选必须先通过同一个当前 verifier；近邻选择只用到共同 previous_q 的欧氏距离，不使用续接结果。共同输入包括相同目标、previous_q、dt、约束和模型。去重容差为最大关节差 1e-10 rad，原始尝试及别名全部保留。

TRAC-IK 使用原适配器所调用的官方 2.2.0 库、Speed 模式和并行搜索；所有后续调用均以其实际上一接受状态初始化。试验顺序交错，五次搜索均保留。库未提供本调用链中的随机种子控制，**没有声称 A/B 使用相同随机种子**；调度 seed 仅决定调用顺序。五次重复是数值搜索重复，不是五条独立轨迹；仅报告描述性计数，不做帧级显著性推断。

实际公共运动学后端为仓库 **URDFKinematics** 的 FK/几何 Jacobian；TRAC-IK 内部为独立 KDL 链，不能根据环境名称称其为 Pinocchio。原适配核对的关节顺序及 FK 一致性记录保留在 [measurement notes](/home/eric/wjg/btry/docs/REVISION_MEASUREMENT_NOTES.md)。本轮没有发现新的关节排列、角度绕回或 FK 不一致证据。

## 2. 首次失败的原因，而非统一“分支漂移”标签

所有帧号从 0 起算。旧记录没有保存拒绝命令的原始 q，因此旧失败的残差、有限性和速度超额不能倒推填补。下表把**历史已知状态**与**同输入新诊断**分开；阶段残差与边界值来自本轮实际返回。详细目标、previous_q、求解返回码、每关节利用率、joint-limit margin、参考/其他方法距离见 [失败原因表](/home/eric/wjg/btry/outputs/continuation_mechanism_study/failure_reason_table.csv)、[旧窗口](/home/eric/wjg/btry/outputs/continuation_mechanism_study/old_failure_windows.json)、[新固定输入记录](/home/eric/wjg/btry/outputs/continuation_mechanism_study/fresh_failure_replays.json)与[内部阶段记录](/home/eric/wjg/btry/outputs/continuation_mechanism_study/internal_stage_traces.json)。

| Case / UID 前缀 | 来源；首次失败 | 历史已知 | 同输入诊断与分类 |
| --- | --- | --- | --- |
| 00 / 05fed739 | CG-HIK；120 | 全级联失败 | 8 个数值阶段均未成功，返回有限。最接近的位置残差约 1.001011 mm，仍超 1 mm 合同。TRAC-IK 5/100 ms 均 0/5。属于有限预算求解失败，伴随残差未满足；不是非有限输出或已证明不可行。 |
| 04 / c27ce734 | CG-HIK；125 | 全级联失败 | 4 个阶段未成功；另外 4 个阶段得到精确位姿，但被速度约束拒绝，最小收敛解速度步长超额约 0.21955 rad。TRAC-IK 5/100 ms 均 0/5。残差未满足与真实速度拒绝并存。 |
| 06 / f69a8f1f | CG-HIK；96 | 全级联失败 | 6 个收敛阶段被速度拒绝，最小超额约 0.03096 rad；另 2 个阶段未成功。部分 DLS 位姿已在合同内（位置约 0.59–0.61 mm），但关节步长过大。TRAC-IK 5/100 ms 均 0/5。 |
| 01 / 80071244 | TRAC-IK；34；smooth | 返回码 −3，未找到解 | 5/100 ms 固定输入均 0/5，均为 solver failure。有限的返回缓冲区不等于有收敛候选，不能据此证明数学不可行。 |
| 02 / a5b1761b | TRAC-IK；87；near-singular | 返回码 −3 | 5/100 ms 均 0/5 solver failure。邻近关节限位是状态特征，但并未观测到收敛命令被 joint-limit verifier 拒绝。 |
| 03 / b82acfb0 | TRAC-IK；87；near-singular | 返回码 −3 | 5 ms 0/5；100 ms 1/5 接受、3/5 浮点速度边界拒绝、1/5 求解失败。5 ms 内部边界对照为 1/5 接受。表明搜索与实现边界均影响此单帧，未恢复整个后续窗口。 |
| 05 / d13cc08f | TRAC-IK；72；near-singular | 返回码 1，但 velocity verifier 拒绝 | 新同输入 5 ms 为 1/5 接受、4/5 浮点速度边界拒绝；100 ms 为 2/5 接受、3/5 此类拒绝；内部边界对照 5/5 接受。旧拒绝 q 缺失，不能逐位证明历史那一次亦由舍入造成；本轮复现证实适配边界确有实现问题。 |

分类先区分 solver 是否报告成功，再区分返回是否有限以及 position/orientation、joint-limit、velocity 拒绝；原始 verifier reasons 始终同时保存。三个 CG-HIK 首次失败的本轮阶段返回均有限，没有 joint-limit 拒绝。不能用“solver failure”掩盖已经收敛但超速的命令，也不能把失败时保留的 previous_q 当作收敛结果。

### 浮点边界问题：实现诊断，不是科研贡献

公共速度合同检查 `abs(q - previous_q) <= velocity * dt + tolerance`。传给 TRAC-IK 的上/下界按加减法构造；浮点加减并非精确互逆，因此返回 q 可以在传入区间内，却在公共减法检查时超出约 **1.39e-17–8.33e-17 rad**。675 次固定输入调用中出现 11 次此类拒绝；续接记录中出现 26 次，均未超出传给库的区间。此量级与 case 04/06 的真实速度超额不同。

独立诊断对照把区间端点向内部的可表示值移动，并再次按原减法合同检查端点；不放宽 verifier，也不修改旧适配器或重计旧结果。本轮该对照没有再产生速度边界拒绝。case 05 原候选的首次失败从 72 帧推迟到 73–75 帧，但五次完整窗口仍全失败。因此该实现问题可解释某些单帧拒绝，不能单独解释所有连续跟踪失败。

## 3. 同输入、不同合法候选的后续表现

表中分子是完成整个固定窗口的次数，分母均为五次。长预算、内部边界列展示“原候选 → 精修候选”；另外测试的最近候选结果见完整表。各行的输入相同只指**行内候选之间**，不同 case 不共享 previous_q。

| Case；当前帧；后续帧数 | 合法候选数 | 原候选，5 ms | 精修候选，5 ms | 最近候选，5 ms | 100 ms，原→精修 | 内部边界，原→精修 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 00；119；30 | 2 | 0/5 | **5/5** | 0/5 | 0/5 → 5/5 | 0/5 → 5/5 |
| 01；33；30 | 6 | 0/5 | 0/5 | 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |
| 02；86；30 | 5 | 0/5 | 0/5 | 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |
| 03；86；30 | 10 | 0/5 | 0/5 | 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |
| 04；124；25 | 4 | 0/5 | 0/5 | 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |
| 05；71；30 | 5 | 0/5 | 0/5 | 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |
| 06；95；30 | 3 | 0/5 | **5/5** | 0/5 | 0/5 → 5/5 | 0/5 → 5/5 |
| 07；90；30；成功对照 | 7 | 5/5 | 5/5 | 5/5 | 5/5 → 5/5 | 5/5 → 5/5 |
| 08；33；30；成功对照 | 5 | 5/5 | 5/5 | 5/5 | 5/5 → 5/5 | 5/5 → 5/5 |

所有九个第 0 帧位置，全部合法候选都完成了接下来 30 帧。因此本轮没有证明第一帧的微小差别会在该短窗口造成失败，更没有通过只观察 30 帧证明它是后来 72–125 帧失败的起因。两个成功对照在初始和匹配窗口也未出现完成差异。

case 03 的精修候选在 5 ms 条件下三次先成功一帧，再于 88 帧失败，另两次在 87 帧失败；原候选五次均在 87 帧失败。这只是有限的一帧推进，不是完成恢复。case 01/02/04 的所测候选均在后续第一帧失败。完整 [同输入候选表](/home/eric/wjg/btry/outputs/continuation_mechanism_study/same_input_continuation_table.csv)、[状态比较表](/home/eric/wjg/btry/outputs/continuation_mechanism_study/site_comparison_table.csv)及[全部候选对](/home/eric/wjg/btry/outputs/continuation_mechanism_study/all_candidate_pairs.csv)保留了负面结果，而非只选择有改善的一对。

### 两个 0/5 对 5/5 实例究竟改变了什么？

| 位置 / 候选 | 当前位置残差 mm | 姿态残差 mrad | 当前最小步长余量 mrad | 到 previous_q 的欧氏距离 rad | 后续完成 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 / 原候选 | 0.996423 | 0.252372 | 38.5644 | 0.007925 | 0/5 |
| 00 / 有界精修 | 0.635290 | 0.019277 | ≈0，仍合法 | 0.075143 | 5/5 |
| 06 / 原候选 | 0.982912 | 0.374966 | 36.6380 | 0.008374 | 0/5 |
| 06 / 有界精修 | 0.223060 | 0.010111 | ≈0，仍合法 | 0.083092 | 5/5 |

精修仅优化**当前**目标残差，使用共同输入的单帧 bounds，无未来目标、q_ref 或学习预测。它采用有界 least-squares，最多 200 次函数评价，优化停止容差 1e-12，额外报告是否达到 1e-7 m/rad 的严格残差目标。上表两个精修结果**均未达到这一严格目标**；准确结论是“残差显著降低的另一个当前合法候选有更好的续接”，不是“精确 IK 已解决”。

更小当前位移不是有利代理：两处改善候选都更远离 previous_q，且几乎用尽本帧允许步长。case 06 的 previous-state DLS 候选更近，但依旧 0/5。更接近参考也不是一致解释：case 00 精修后的最大参考距离从 0.453924 降至 0.412451 rad；case 06 反而从 0.527102 增至 0.579253 rad。

两个精修候选的 Jacobian 最小奇异值均增加；下一目标误差的线性伪逆步长利用率分别从 4.535 降至 1.199、从 2.007 降至 0.457。这个量只是局部线性诊断：没有考虑完整零空间优化、有限步非线性和可接受残差，**大于 1 并不证明不可行**。case 04 也降低了残差并增大最小奇异值，却未恢复。因此仅以当前残差、奇异值、当前余量或最近解任一单值解释全部失败都不充分。

TRAC-IK 本身以比公共 1 mm/0.5° 合同更严格的内部 Cartesian 条件搜索；未达到其内部目标不等于公共合同不存在合法解。剩余负面窗口仍可能涉及内部精度要求、搜索局部性、姿态和方向相关步长需求，当前结果不把它们直接归为数学不可行或离散分支转换。

### 不可达参考没有被用于“恢复”

九个第 0 帧 q_ref 均合法；**全部九个失败前/匹配位置的 q_ref 均因单帧速度约束被过滤**。其他方法的原状态也须从共同 previous_q 重新检查，不能沿用它在自身闭环中的合法性。被拒候选的完整 q 和原因保存在 [candidate_attempts.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/candidate_attempts.json)，没有向不可达参考状态跳转。

## 4. 真实轨迹图与成功 witness

两图左列是原记录的 150 帧闭环状态，右列是在共同输入处接入两个当前合法候选后的五次真实 TRAC-IK 窗口。代表实例按来源类别中最大完成/连续成功前缀差选择，是展示性选择；所有状态的结果均在前述表中。

![CG-HIK 来源状态的同输入续接对照](/home/eric/wjg/btry/outputs/continuation_mechanism_study/final_figures/cghik_continuation.png)

图 1：case 06，第 95 帧原候选与有界精修候选；后续分别 0/5、5/5 完成。状态图显示当前差别最大的第 5 关节，完整七关节在 source CSV 和 raw records 中。

![TRAC-IK 来源状态的同输入续接对照](/home/eric/wjg/btry/outputs/continuation_mechanism_study/final_figures/trac_ik_continuation.png)

图 2：case 03，第 86 帧原候选与有界精修候选；均为 0/5 完成，精修后三次仅多推进一帧。它不是一个被救回的成熟基线实例。

两图虚线分别给出公共残差/余量阈值，左列竖虚线是原首次失败，浅色区域是所测窗口，右列叉号标记首次失败。位置/姿态残差是**接受或保持的实际状态**对当前目标的误差；失败后的大误差不是已接受的越约束命令。步长余量只画 verified command，失败处留空；初始候选用圆点。五次重复逐条绘出，重合的线并非遗漏；无平滑、平均或伪造插值。首次导出的图仍保留，最终图只改显示与源数据完整性，没有改测量。

共保存 **390 个完整续接 witness**，其中主 5 ms 条件 310 个，包含初始当前合法转移及真实逐帧反馈 q。每个 witness 均独立重新通过原 verifier 检查；[verification.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/verification.json)记录核查数量，不调用 IK 求解器。

- [case 00：第 119 帧合法接入，续接 120–149 帧](/home/eric/wjg/btry/outputs/continuation_mechanism_study/successful_witnesses/case_00_pre_failure_candidate_01_trac_5ms_r0.json)：覆盖该位置之后全部剩余目标，但不是原算法从第 0 帧自动生成的新完整运行。
- [case 06：第 95 帧合法接入，续接 96–125 帧](/home/eric/wjg/btry/outputs/continuation_mechanism_study/successful_witnesses/case_06_pre_failure_candidate_02_trac_5ms_r0.json)：仅证明这 30 帧，不外推到 149 帧。

## 5. 与三类指定工作的边界

这是限定三篇工作的问题对照，不是穷尽文献检索或新颖性证明。学术检索工具在本会话不可调用，采用作者原文/出版社/机构记录核对。UR 论文机构详情页与出版社动态页的直接访问受限，相关范围以可检索机构摘要及出版社索引内容为据，不声称完整阅读全文或复现实验。

| 已有工作 | 已经解决什么 | 本轮失败属于什么 | 尚未由本轮证据回答的具体差别 |
| --- | --- | --- | --- |
| [CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning](https://arxiv.org/html/2309.09102v2)，Morgan 等，2024 | 生成整条候选轨迹，做全局候选路径搜索与 LM 精修；明确研究冗余机器人。时间参数化留给外部模块。 | 当前合法候选影响固定后续目标的跟踪结果，与整条路径选择问题相关，不是首次发现未来路径重要。 | 本轮固定 20 ms 时序及逐帧速度合同，只干预一个可达当前候选；没有比较整轨迹搜索能否解决这些窗口。 |
| [Handling Transitions Across Singularities for UR-Like Serial Robots](https://doi.org/10.1109/LRA.2026.3653295)，Boschi 等，2026；[机构记录](https://cris.unibo.it/handle/11585/1039115) | 针对 UR 类非冗余串联机器人，处理经过奇异位置的 IK 分支转换，以保持可容许路径的关节连续性与光滑性。 | Panda 的七自由度冗余姿态、残差与步长约束耦合；本轮未识别出 UR 式离散分支切换事件。 | 需先分离冗余姿态与残差的作用；不能将 Panda 失败直接归入非冗余离散分支转换，也没有与该算法做对照。 |
| [Ensuring Viability: A QP-based Inverse Kinematics for Handling Joint Range, Velocity and Acceleration Limits, as Well as Whole-body Collision Avoidance](https://link.springer.com/article/10.1007/s10846-025-02335-z)，Zhang 与 Kikuuwe，2026 | 通过离线计算附加在线 QP 约束，保持存在后续满足物理/碰撞等约束的控制序列。 | 本轮是对固定、带时间的目标序列进行 solver-specific 有限窗口跟踪；不包含加速度与碰撞研究。 | 保持物理约束下的 viability 不等于保证任意给定目标每帧达到位姿容差。本轮既没有构造 viability 证书，也未证明剩余窗口不可行。 |

因此不使用“首次考虑未来可行性”“已经证明分支漂移”或“需要新的学习 gate”等结论。本轮最明确的正面证据已被简单有界精修获得；未解决部分应保持为未解决的诊断结果。

## 6. 交付及停止点

独立模块为 [continuation_mechanism](/home/eric/wjg/btry/src/confik/continuation_mechanism/)；原 runtime 没有修改。诊断代码只在新目录记录输出，已有测量文件使用排他写入防止覆盖。分析中的一处窗口长度文字注释由“25 或 29”订正为实际的“25”（17 个位置为 30 帧，1 个为 25 帧）；未改任何数值记录。

原始续接：[continuation_raw.jsonl.gz](/home/eric/wjg/btry/outputs/continuation_mechanism_study/continuation_raw.jsonl.gz)；逐重复摘要：[continuation_runs.csv](/home/eric/wjg/btry/outputs/continuation_mechanism_study/continuation_runs.csv)；交付 hash：[delivery_manifest.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/delivery_manifest.json)。四项定向单元测试覆盖合同检查、失败分类、浮点内部边界和不可达参考过滤。统计技能约束了重复单位和描述性表述；制图技能要求逐条真实轨迹及源数据，不按重复帧扩大样本量。

研究在此停止：不扩展机器人、模式、周期参数、模型或 gate；不修改论文，不把未完成搜索变成数学不可行证明，也不由这两个窗口直接启动新算法。
