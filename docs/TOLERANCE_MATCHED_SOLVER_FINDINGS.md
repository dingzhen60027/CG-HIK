# 对齐公共任务合同后的 TRAC-IK 与 DLS 连续跟踪比较

## 结论：已有容差能力解释了重要失败，尚不支持开发新方法

**合理启用 TRAC-IK 的官方 Cartesian bounds 足以恢复本轮全部五个同输入 strict 失败案例。** 这包括 `state_14 candidate_04` 自身第 101 帧接受状态到第 102 帧目标，以及 `case_01` 四个既有失败候选的下一帧输入。strict 在 5 ms、20 ms 下均为这五个输入各 0/5，task-tolerance 均为各 5/5。公共 verifier、目标、实际 previous_q 和速度合同不变。这是正确配置成熟库带来的恢复，不是新算法，也没有证明 case_01 的整个后续窗口都已被单帧容差配置解决。

在新生成、预先固定的 40 条完整轨迹上，TRAC-IK strict 两个预算均每次完成 **34/40**；task-tolerance 在 5 ms 下完成 **36、37、37/40**，20 ms 下完成 **37、37、38/40**。两种预算均未丢失 strict 已完成的轨迹，累计时延分别降低 **32.53% / 58.40%**。仅延长 strict 预算没有增加完成数，说明原内部精度要求是本批失效的重要因素之一，但不是全部原因。

普通 task-tolerance DLS 以真实上一接受状态反馈，完成 **37/40**，全部 6,000 次外层调用均小于 20 ms；然而累计求解时间 **4.814 秒**，是 task-tolerance TRAC-IK 5 ms 的 **2.24 倍**、20 ms 的 **1.34 倍**。它在大多数本批轨迹中能够续接，但并非零失败，也没有显示比合理配置 TRAC-IK 更低的整体计算成本。

**当前不必提出新的构型评分、学习模型或预览方法。建议采用成熟方法的合理配置，并结束本支线的新算法开发；保留明确失败作为局限。** 尚存近奇异和关节范围返回轨迹上的数值不收敛、残差超限与搜索重复波动，不能把它们直接升级为数学不可行或新方法必要性的证据。本轮不自动追加求解器、候选池、模型或实验。

## 1. 固定配置与公共接受条件

基线为 `b247641bb98807339c98723dcdefb4d29fc8f103`。所有新记录仅写入 [tolerance_matched_solver_comparison](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/)，旧论文、配置、solver/verifier 源码及冻结证据不变。本轮是机制开发和独立轨迹比较，不称新的 formal test，不设总 gate。

四种配置形成六个实际设置：TRAC-IK 的两种精度各比较 5 ms 与 20 ms，DLS 的两种精度各只运行一次确定性求解。未搜索“最佳预算”或按结果调整参数。

| 配置 | 内部停止 / 容差 | 计算预算 | 其他设置 |
| --- | --- | --- | --- |
| TRAC-IK strict | epsilon=1e−5；额外 Cartesian bounds 全零 | 固定 5、20 ms | 原 2.2.0、Speed、实际 previous_q 初始化 |
| TRAC-IK task-tolerance | 同一 epsilon；保守任务分量框 | 固定 5、20 ms | 与 strict 相同，仅传入官方 bounds |
| DLS strict | 位置范数 1e−5 m、姿态范数 1e−5 rad | 最多 25 次迭代 | 原 AdaptiveDLS 阻尼、方向、线搜索、步长参数 |
| DLS task-tolerance | 原公共位置 / 姿态范数阈值 | 最多 25 次迭代 | 与 DLS strict 相同，仅内部停止精度不同 |

公共验收仍为位置误差 ≤ **0.001 m**、姿态误差 ≤ **0.00872664626 rad**（原 0.5°数值）、原 URDF 关节范围，以及逐关节 |q−previous_q| ≤ v_i×**0.02 s**+**0.0001 rad**。所有返回配置均送入同一 `SolutionVerifier`。**公共通过才接受，内部未收敛不自动等于公共非法；内部成功也不免除 verifier。** 原生返回码、内部停止原因和公共验收结果分列，不能混成一个成功标志。没有最后一步额外精修、重试或输出后关节裁剪。

### 官方 bounds 的坐标系与保守范围

核查对象是实际使用的 TRAC-IK upstream commit `90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb`，其三份 solver 实现未改动。通过新增独立 C ABI 将官方 `CartToJnt(..., KDL::Twist bounds)` 参数暴露出来，不重写求解器。实际源码的 `diffRelative(target, actual)` 使用**目标坐标系下的位置差和轴角旋转向量**，不是 RPY 分量，也不是实际角速度；这些三维向量的范数分别与公共位置距离、旋转测地角一致。[官方固定版本头文件](https://github.com/traclabs/trac_ik/blob/90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb/trac_ik_lib/include/trac_ik/kdl_tl.hpp)

设公共位置与姿态容差为 ε_p、ε_R，三个位置分量的 bounds 均取 `nextafter(ε_p/√3, 0)`，三个姿态分量均取 `nextafter(ε_R/√3, 0)`。实际数值为 **0.0005773502691896257 m** 和 **0.00503833156733364 rad**。这是两个范数球的内接分量框，**并不与完整公共误差球完全等价**：沿某一坐标轴仍在公共球内的点可能被框排除。DLS task-tolerance 的停止条件则直接使用两个范数球，因此两种内部搜索区域也不完全相同。

源码会把框内误差分量置零，再检查内部 epsilon；不是把 epsilon 加到 bounds 上。当前每个 bound 都大于 epsilon，因此不会因为内部 epsilon 而放宽这个框。最终公共 verifier 仍作独立非线性检查。真实 KDL 残差函数的 100 组坐标变换 / 框内样本测试确认上述参数化和范围；20 组 URDF 配置的原生 KDL FK 与公共模型误差低于 1e−12，关节顺序一致。[官方 KDL 求解器源码](https://github.com/traclabs/trac_ik/blob/90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb/trac_ik_lib/src/kdl_tl.cpp)、[官方 NLOpt 求解器源码](https://github.com/traclabs/trac_ik/blob/90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb/trac_ik_lib/src/nlopt_ik.cpp)

TRAC-IK strict 的逐分量 epsilon 和 DLS strict 的范数阈值同样不是完全等价的内部停止集合。这里对齐的是**公共任务接受条件**，并透明保留现有 API 的参数化差异，不将更宽的公共任务要求计作算法收益。

### DLS 的在线边界适配不同于旧离线初始化

旧 `make_initialization` 在预测序列中直接推进 `result.q`，没有逐帧速度合同验收，不能直接称为在线策略。本轮没有调用它来运行在线轨迹。

六个设置每帧均由实际 previous_q 构造相同的 `representable_interior` 关节区间：URDF 范围与单帧允许步长相交，并处理已有的浮点端点问题。TRAC-IK 通过原有 `setKDLLimits` 使用此区间。两种 DLS 共用一个 kinematics adapter，将原有 `clip` 运算的范围限制在该区间内；不改变其阻尼、方向、线搜索或 `max_joint_step` 参数。这是将原 DLS 接入同一在线约束合同的边界适配，**不是额外构型搜索或新优化算法**。相对于旧离线 DLS，不能隐去这项明确的可行区间处理。

公共运动学与 DLS Jacobian 的实际后端为项目 **URDFKinematics**；TRAC-IK 内部为同一 URDF 对应的 KDL 链，没有调用 Pinocchio。Panda 根链接惯量警告只涉及 KDL 不支持根惯量，不影响这里的运动学链；未修改 URDF。

## 2. 同输入检查：五个 strict 失败输入全部恢复

复用上一轮 34 个固定输入：32 个既有当前候选作为下一帧的真实 previous_q，加上 state_14 两个候选各自的第 101→102 帧专项输入。每个输入的 target、previous_q、dt 和 query hash 完全固定。既有 L=1 witness 仅标记“该确切输入存在合法命令”，不传入任何在线初始化。

四个 TRAC-IK 设置各对每个输入做五次搜索；两个数值确定的 DLS 各一次，共 **748 次调用**。重复不是独立轨迹，不声称 TRAC-IK 配对随机种子。完整表见 [fixed_input_table.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/fixed_input_table.json)，所有原始调用见 [fixed_records.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/fixed_records.json)。

| 设置 | 全部重复公共通过的输入 | 全部失败输入 | 混合输入 | 公共通过调用 / 全调用 |
| --- | ---: | ---: | ---: | ---: |
| TRAC strict 5 ms | 29/34 | 5 | 0 | 145/170 |
| TRAC task 5 ms | 34/34 | 0 | 0 | 170/170 |
| TRAC strict 20 ms | 29/34 | 5 | 0 | 145/170 |
| TRAC task 20 ms | 34/34 | 0 | 0 | 170/170 |
| DLS strict | 34/34 | 0 | 0 | 34/34 |
| DLS task | 34/34 | 0 | 0 | 34/34 |

strict 的五个失败输入为 `case_01_candidate_00`、`case_01_candidate_05`、`case_01_matched_00`、`case_01_matched_03` 和 `failure_input_candidate_04`。task-tolerance 没有损失任何原 strict 成功输入。四个 case_01 输入的恢复只说明其**下一帧**可以通过合理容差求解，不能据此把原联合优化的 30 帧恢复归功于本轮尚未执行的 case_01 全窗口测试。

### state_14 candidate_04 的确切失败输入

仍固定其自己真实的 q_101，不借用 candidate_01 的状态。各设置的返回误差与速度比分别如下；同一 TRAC 配置五次的解在此例一致。

| 设置 | 内部状态 / 公共验收 | 位置误差（mm） | 姿态误差（rad） | 最大合同步长比 | 实测单次时间（ms） |
| --- | --- | ---: | ---: | ---: | --- |
| TRAC strict 5 ms | native -3；0/5 | 3.122493 | 0.005133004 | 0（返回初始化状态） | 5.253–5.315 |
| TRAC strict 20 ms | native -3；0/5 | 3.122493 | 0.005133004 | 0 | 20.341–20.701 |
| TRAC task 5 ms | native +1；5/5 | 0.712264 | 0.005540472 | 0.096794 | 0.190–0.460 |
| TRAC task 20 ms | native +1；5/5 | 0.712264 | 0.005540472 | 0.096794 | 0.192–0.556 |
| DLS strict | 25 次迭代未达内部精度；公共通过 | 0.018033 | 0.000018495 | 0.088867 | 8.416 |
| DLS task | 内部 / 公共均通过 | 0.170185 | 0.000596178 | 0.087938 | 0.603 |

因此，在这个固定输入上，**开启原库现成容差即可恢复；单独延长严格预算不能恢复**。DLS strict 虽然公共合法，但没有达到 1e−5 的内部停止精度，不能把它写成严格精度成功。

## 3. 完整独立轨迹：全样本结果

采用 40 个新 seeds **970909000–970909039**，四类各 10 条、每条 150 帧。原参考可行轨迹生成器未修改；生成只依赖几何条件，不接触任何本轮求解结果。全部 6,000 个参考转移通过原 verifier，且与扫描到的旧身份元数据按 UID、seed、query hash 无交集；覆盖范围记录在 [execution_seal.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/execution_seal.json)，不冒称穷尽未记录身份的全库证明。

所有配置从每条轨迹相同的已知初始 q 开始。在线入口只收到当前目标、实际 previous_q 和 dt；载入的在线目标文件已剥离未来 q_ref。通过 verifier 才更新 previous_q；失败则保持上一接受状态，但目标索引正常推进至第 149 帧。没有失败后重置、跳到参考路径或丢弃该轨迹。未来目标虽然存在于试验驱动器的数据文件中，但不进入单帧 solver API，也不用于求解器选择。

每个 TRAC 配置进行三次完整重复；DLS 各一次，共 **560 条运行记录、84,000 次在线调用**。独立单位仍是 **40 条轨迹**，而非 560 次独立轨迹抽样或 84,000 个独立帧。TRAC 重复顺序和设置顺序预先随机打乱，不控制或宣称相同原生搜索随机种子；所有重复保留。帧分位数仅作描述性时延分布，不作为独立样本显著性检验。

### 主表

完成数的斜线表示 repeat 0/1/2，**每次分母均为 40**；DLS 只有一次。累计时间按完成一遍固定 40×150 帧目标的总时间报告，TRAC 取三遍均值，不将三倍执行量直接与一次 DLS 相比。成功与失败调用全部计时。

| 设置 | 几何完成数 | 几何且全帧≤20 ms 完成数 | frame verified success | 帧 P50 / P95 / P99（ms） | 每遍累计时延（ms） |
| --- | ---: | ---: | ---: | --- | ---: |
| trac_strict_5ms | 34/34/34 | 34/34/33 | 93.983% | 0.190 / 5.249 / 5.434 | 3187.230 |
| trac_task_5ms | 36/37/37 | 36/36/37 | 97.278% | 0.179 / 0.484 / 5.305 | 2150.281 |
| trac_strict_20ms | 34/34/34 | 34/34/34 | 93.994% | 0.192 / 20.352 / 20.559 | 8653.782 |
| trac_task_20ms | 37/37/38 | 37/37/38 | 98.161% | 0.180 / 0.462 / 20.392 | 3600.127 |
| dls_strict | 37 | 37 | 96.800% | 0.943 / 9.119 / 10.624 | 14254.820 |
| dls_task | 37 | 37 | 96.800% | 0.550 / 1.451 / 6.830 | 4814.011 |

TRAC task 5 ms 的逐遍累计时间为 **2336.241、2057.953、2056.648 ms**；task 20 ms 为 **4237.323、3825.170、2737.887 ms**。搜索重复带来的完成和成本波动均保留，不能挑选其中一遍代表方法。[repeat_table.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/repeat_table.json)同时提供 strict 各遍时间和完整 completion UID。

| 设置 | 单条轨迹累计时间均值 / 中位数 / P95（ms） | 外层≤5 ms 返回比例 | 外层≤20 ms 返回比例 |
| --- | --- | ---: | ---: |
| TRAC strict 5 ms | 79.681 / 34.097 / 417.526 | 93.978% | 99.994% |
| TRAC task 5 ms | 53.757 / 32.348 / 212.622 | 97.261% | 99.994% |
| TRAC strict 20 ms | 216.345 / 34.719 / 1541.861 | 93.978% | 93.994% |
| TRAC task 20 ms | 90.003 / 32.919 / 664.989 | 98.083% | 98.161% |
| DLS strict | 356.371 / 167.645 / 1034.216 | 82.233% | 100% |
| DLS task | 120.350 / 90.314 / 450.507 | 96.783% | 100% |

DLS 的 25 次迭代没有被冒充为 5 ms 或 20 ms 时间预算。其 task 配置最大实测调用为 13.201 ms，但这只是在当前样本/工作站上的观察，不是实时最坏情况保证。原生 TRAC 的预算只覆盖其内部搜索；外层转换、`setKDLLimits` 重建内部对象、线程相关开销和最终验证另占时间，因而 20 ms 设置的失败外层调用通常超过 20 ms。5 ms 设置也各出现一次超过 20 ms 的外层调用，导致一条几何完成运行未满足全帧 deadline；本轮未单独识别这些尾部异常的系统原因。

### 每类轨迹的完成与计算量

下表每类分母为 **10 条**。累计时间是该类固定 1,500 帧的一遍总时间，TRAC 对三遍取均值。

| 类别 | TRAC strict 5 ms | TRAC task 5 ms | TRAC strict 20 ms | TRAC task 20 ms | DLS strict | DLS task |
| --- | --- | --- | --- | --- | --- | --- |
| smooth 完成 | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 | 10 | 10 |
| near-singular 完成 | 5/5/5 | 7/8/8 | 5/5/5 | 8/7/8 | 8 | 8 |
| joint-limit-return 完成 | 9/9/9 | 9/9/9 | 9/9/9 | 9/10/10 | 9 | 9 |
| high-curvature 完成 | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 | 10 | 10 |
| smooth 累计 ms | 351.866 | 328.689 | 348.259 | 319.862 | 1475.052 | 891.931 |
| near-singular 累计 ms | 1757.452 | 765.246 | 5973.605 | 2087.075 | 8353.853 | 1617.580 |
| joint-limit-return 累计 ms | 754.074 | 718.902 | 1997.008 | 869.255 | 2725.842 | 1353.854 |
| high-curvature 累计 ms | 323.838 | 337.444 | 334.911 | 323.935 | 1700.073 | 950.646 |

计算收益主要来自困难轨迹中减少严格求解失败和预算耗尽；不能因 smooth/high-curvature 全成功就推断整个工作区都已解决。完整分类别误差、FEV（仅 DLS 可观测）、时延和成功分布见 [family_table.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/family_table.json)。

## 4. 精度变化、恢复/损失与剩余具体问题

### 成功命令的精度没有被隐藏

task-tolerance 返回更大的残差，但上限仍是原公共合同。以下均为**实际接受命令**的位置 / 姿态误差分布。

| 设置 | 位置 P50 / P95 / P99 / max（mm） | 姿态 P50 / P95 / P99 / max（rad） |
| --- | --- | --- |
| trac_strict_5ms | 0.000029 / 0.003483 / 0.007985 / 0.014633 | 0.00000005 / 0.00000536 / 0.00001087 / 0.00001400 |
| trac_task_5ms | 0.049329 / 0.391503 / 0.688605 / 0.967956 | 0.00006385 / 0.00066183 / 0.00498002 / 0.00813920 |
| trac_strict_20ms | 0.000029 / 0.003489 / 0.007982 / 0.014633 | 0.00000005 / 0.00000533 / 0.00001085 / 0.00001400 |
| trac_task_20ms | 0.049772 / 0.396400 / 0.699377 / 0.995192 | 0.00006357 / 0.00068467 / 0.00510782 / 0.00800543 |
| dls_strict | 0.000055 / 0.135670 / 0.421015 / 0.952818 | 0.00000011 / 0.00008898 / 0.00031446 / 0.00639267 |
| dls_task | 0.077957 / 0.736310 / 0.977397 / 0.999967 | 0.00009592 / 0.00091358 / 0.00333964 / 0.00866450 |

DLS strict 的 952 次内部未收敛中，**760 次返回配置仍通过公共合同**；这些合法命令被正常接受。因此它完成 37 条并不表示每帧都满足严格内部精度。DLS task 的 192 次内部未收敛均被公共 verifier 拒绝。TRAC strict 5/20 ms 与 task 5/20 ms 的内部失败数分别为 **1083 / 1081 / 490 / 331**，本次没有内部失败但公共通过的 TRAC 返回值。

[main_table.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/main_table.json)也完整报告**全部返回值**而非只报告成功子集的误差分布。失败后目标继续推进、上一命令保持，故全部返回值最大位置误差可达 0.3948–0.4691 m；这些值均被拒绝，不能误称为实际接受命令误差。六个设置的 joint-limit / velocity verifier 拒绝数均为 **0**；剩余公共拒绝均涉及位置或姿态残差，而不是浮点边界越界。

### trajectory identity：容差恢复了什么

按同一 UID 的完成重复比例比较，strict→task 的恢复与损失如下。这里“稳定”仅指所有三次计划重复均一致，不是概率为一的保证。

| 比较 | 完成比例提高的轨迹 | 其中 strict 0/3 → task 3/3 | 损失轨迹 |
| --- | --- | --- | --- |
| TRAC task 5 ms vs strict 5 ms | trajectory_10、19、12 | 10、19 | 无 |
| TRAC task 20 ms vs strict 20 ms | trajectory_10、19、12、26 | 10、19 | 无 |
| DLS task vs DLS strict | 无；同为 37/40，完成 UID 相同 | 不适用，DLS 单次 | 无 |

`trajectory_12` 在两种 task TRAC 预算中均为 2/3 完成；`trajectory_26` 在 task 20 ms 为 2/3，task 5 ms 为 0/3。不能把这些部分恢复写成稳定全部恢复。完整 UID、逐次 completion 和恢复/损失定义保存在 [completion_uid_sets.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/completion_uid_sets.json)与 [completion_comparisons.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/completion_comparisons.json)。

### 合理配置后仍存在的可重复差异

| 轨迹 | task TRAC 5 ms 首次失败帧（各重复；—为完成） | task TRAC 20 ms | DLS task 首次失败 | 已观察的具体类型 |
| --- | --- | --- | --- | --- |
| trajectory_14，near-singular | 113 / 115 / 115 | 115 / 116 / 115 | — | 两种 TRAC 预算均 0/3；native -3 后位置残差约 2.72–3.80 mm；DLS 完成 |
| trajectory_17，near-singular | 119 / 118 / 119 | 118 / 118 / 119 | 97 | 两种 TRAC 均 0/3；DLS 位置误差 1.035435 mm 被拒绝；TRAC 首次失败同时位置/姿态超限 |
| trajectory_12，near-singular | 94 / — / — | — / 93 / — | 91 | TRAC 搜索重复有波动；DLS 位置误差 1.076437 mm，未达公共精度 |
| trajectory_26，joint-limit-return | 72 / 71 / 73 | 74 / — / — | 70 | task 20 ms 部分恢复；DLS 姿态误差 0.009481344 rad，超过原上限 |

`trajectory_17` 是两种 task TRAC 在全部重复及确定性 DLS 中均未完成的共同实例，UID 为 `ca9065ec04392f64d1cce7cea6e6ae7ba224495be6a41fa0902e34ddeb925465`。`trajectory_14` 是普通 DLS 完成而两种 task TRAC 全部重复失败的实例，UID 为 `4e626ba28da6a8d45d22f023c3732e2d796c83a0e74df2aa41168fe26f6dccab`。全部首次失败输入、原生状态、实际 previous_q、返回 q、误差、速度比和时间见 [first_failure_table.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/first_failure_table.json)。

这几条轨迹在不同设置下具有各自的实际闭环历史，不能把同一目标帧当成相同输入。参考整条路径可行也**不保证求解器已经到达的某个失败 previous_q 仍有合法下一帧命令**。本轮未在这些新失败输入上追加非线性参照，也未证明数学不可行或特定“分支漂移”原因。当前能确认的是：合理容差未消除所有数值/闭环跟踪失败，且 TRAC 与 DLS 没有形成逐轨迹的完全支配关系。

DLS task 相对其 strict 配置累计时延降低 **66.23%**、完成 UID 不变，说明严格精度在该实现上也有明显额外成本。但它相对 task TRAC 5 ms 的中位数、P95、P99 和累计成本均更高；相对 task TRAC 20 ms 的 P99 更低，累计成本仍更高。这里比较的是现有 C++ TRAC-IK 与项目 Python/NumPy AdaptiveDLS 实现，不能外推成抽象算法复杂度排序。

## 5. 计时、保护、复现与停止点

所有必要输入转换、当前单帧允许区间构造、native `setKDLLimits`/solve 或 DLS、最终公共 verifier 均计入外层 `total_latency_ns`。同时分别保留转换、求解、验证时间。模型/URDF 载入和初始构造不计入单帧；日志序列化、辅助误差统计、事后复核不参与求解或接受，也不计入单帧。没有把 DLS 迭代数等同于 TRAC 时间预算，也没有把原生不可观测 FEV 填为零。deadline 是单独指标；几何合法但迟到的返回仍按预定几何反馈规则更新，不把该运行称为全帧实时完成。

环境为共享工作站，Python 3.12.13 / NumPy 2.4.4，BLAS/OpenMP 环境线程均为 1，TRAC 原生保留两个并行求解 worker。测量阶段未并行运行其他基准或单元测试。每个设置仅执行预定样本和重复，没有结果驱动重跑。初始化与边界 adapter 是任务接入实现，不作为科研贡献。

协议、40 个 seeds、轨迹、剥离 q_ref 的在线输入及 34 个同输入案例，均在本轮 solver outcomes 产生前固定。execution seal 的 SHA-256 为 **`5808dd276f8588ce34cb2882f5412aee0c163f736ccdcfe94474eb83c74318c1`**。独立只读复核重算了 **84,000 个在线返回、748 个固定输入返回、6,000 个参考命令**，检查每次失败后的真实状态反馈、目标顺序、验收理由和时间阈值一致性。**接受合同违约为 0**。500 份成功完整运行的全序列索引见 [successful_witness_index.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/successful_witness_index.json)；其 source 指向真实 150 帧原始记录，不是插值或参考路径替代。

**29 项测试通过，1,404 个受保护旧文件 hash 不变**，测量入口、边界配置、旧 solver/verifier 与原生库 hash 均通过封存检查。检索/统计技能分别用于优先核对实际官方固定版本源码、明确内接近似，以及按完整轨迹区分嵌套重复；没有扩展文献综述或做伪独立帧统计。[verification.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/verification.json)和 [delivery_manifest.json](../outputs/continuation_mechanism_study/tolerance_matched_solver_comparison/delivery_manifest.json)提供复核与交付清单。

实现位于原 `continuation_mechanism` 模块，不建立版本 runtime。独立入口为 `python -m confik.continuation_mechanism.tolerance_comparison {prepare,sample,fixed,online}`，只读报告入口为 `python -m confik.continuation_mechanism.tolerance_reporting`。已有输出采用 exclusive create，当前已完成，不应再次运行。原生扩展可用下列固定依赖路径重建，输出独立于旧库：

```bash
cmake -S src/confik/continuation_mechanism/tolerance_native -B tmp/tolerance_solver_build \
  -DTRAC_SOURCE=/home/eric/wjg/btry/tmp/revision_dependencies/trac_ik \
  '-DCMAKE_PREFIX_PATH=/home/eric/wjg/btry/tmp/revision_dependencies/prefix/usr;/opt/ros/humble' \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build tmp/tolerance_solver_build -j 2
```

**停止判断：已有容差配置已解释同输入漏解并带来完整轨迹收益；DLS 是有用的成熟数值对照，但本轮未显示需要包装新方法。** 剩余问题以具体 trajectory UID、首次失败输入和重复差异保留；不自动转入新 gate、学习网络、构型指标、预览模块或论文修改。
