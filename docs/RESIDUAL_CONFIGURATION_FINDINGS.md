# 残差与冗余构型解耦：Panda 续接机制研究

证据基线：`e192176f3182f86a4356a3d0003957ce170a736f`。本轮未修改旧 runtime、verifier、模型、论文或旧结果；没有重跑旧实验包。新结果仅写入 `outputs/continuation_mechanism_study/residual_configuration_disentanglement/`。

## 一页判断：是否需要超出统一残差精修的方法？

**统一残差精修不足以解释全部已观察到的续接差异；构型信息值得继续研究，但本轮尚不支持提出某个新的续接准则。**

统一精修 B 在七个失败前状态中恢复两个 30 帧窗口，均为上一轮已被“仅当前帧精修”恢复的 CG-HIK 来源状态。逐帧继续精修没有额外恢复窗口，两个成功对照也没有增加完成收益。完整成功的 B 窗口耗时约 5.2 秒/30 帧，并增加 48,000 次精修残差函数调用；这不是已证明的低成本处理。

更关键的是 TRAC-IK 来源的 smooth 状态 **case 01，第 33 帧**。原候选与其精修候选均为后续 **0/5** 完成，统一逐帧精修仍为 **0/5**；但分别以两者实际 FK 位姿为目标得到的四个冗余构型，使用普通 TRAC-IK 均为 **5/5** 完成。它们与各自中心的完整末端位置/姿态差在约 1e-16 的数值精度内，且从共同 previous_q 可在原单帧合同内到达。因此这一实例不能用“只是当前位置或姿态残差更小”解释，且不局限于 CG-HIK 来源状态。

不过，下一目标的受约束线性运动需求并未成为更好的解释量。在 82 个有结局/前缀差异的同状态候选对中，其排序一致度为 **74.4%**，当前残差为 **75.0%**，最近解为 **17.1%**，一致缩放后的最小奇异值为 **91.5%**。这些是依赖于少量选定状态的描述性排序，不是独立测试集上的预测准确率。138 个候选的线性最小需求均小于等于一个允许步长，但只有 78 个对应线性解在非线性 FK 后通过原 verifier，说明不能把这个局部线性量当作续接证明。

下一步真正值得分清的是：**在实际末端位姿已匹配时，冗余构型如何改变后续任务方向的运动学条件，以及固定 TRAC-IK 搜索/内部精度要求下的可求解性。** 目前证据区分了残差与构型，却没有进一步区分“物理上不可续接”和“该求解器在此构型下没有找到解”。三个 CG-HIK 来源状态的精修中心均未找到符合本轮有限搜索范围的等位姿替代构型，其残差与构型作用仍未完全解耦。

结论是：保留这个明确的构型相关反例，拒绝把统一精修包装成普适低成本解法；也不把下一目标线性需求或单一奇异值立即升级为新算法。本轮停止，不训练模型、不加模式或周期参数、不改论文。

## 1. 固定设计与数据复用

沿用上一轮 18 个状态、95 个合法候选及所有旧续接记录，仅为这些候选增加新的解释量计算。新增干预覆盖原九条轨迹的失败前/匹配位置，不对第 0 帧状态重做续接。原目标、dt=0.02 s、关节范围、速度限值及公共 1 mm/0.5° 位姿容差不变。

运行前写入的 [protocol.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/protocol.json)固定了匹配精度、有限扰动、投影、精修、诊断、预算和源文件 hash。79 次投影尝试中保留 43 个新合法候选，每个状态 3–6 个；未按续接结果选择或增加候选。新增 260 个五次重复组成的窗口，共 7,700 次后续 TRAC-IK 调用。每个窗口为 30 帧；case 04 当前为第 124 帧，只剩 25 帧，不补帧。

主对照为：

- **A，普通续接：** 复用旧原候选的 5 ms、可表示内部边界 TRAC-IK 记录，五次，不重跑。
- **Current only：** 复用旧精修候选的相同 TRAC-IK 续接记录，后续不再精修。
- **B，统一精修：** 当前原候选精修后提交；此后每帧 TRAC-IK 返回的合法候选都按同一规则精修。未产生更好合法候选时保留原合法候选；失败的求解返回不作为合法候选交给精修器。
- **Pose-matched：** 新匹配候选统一使用普通 5 ms TRAC-IK，不在后续加入精修，从而只改变当前接受构型。

新调用统一采用已记录的 `representable_interior` 边界构造，原 verifier 不放宽。A 和 Current only 是历史同边界测量，不能与本轮 B 的计时冒充同期交错速度试验。旧普通边界的其他候选保留在背景表中，不混入本轮同边界的主要候选排序比较。

TRAC-IK 未做随机种子配对控制；调度 seed 只控制新试验顺序。五次搜索不是五条独立轨迹，窗口中的帧也不是独立样本。不估计总体发生率，不做 p 值检验，不设总 gate。实际后端仍是 **URDFKinematics**，TRAC-IK 内部是 KDL，未以环境名推断为 Pinocchio。

## 2. 统一精修：恢复数与实际代价

精修函数只读取当前目标、候选和实际 previous_q。目标保持上一轮的 `sum(translation_m²) + sum(rotation_vector_rad²)`，没有换为新加权目标。边界始终取当前 previous_q 的允许单帧区间；算法为 bounded TRF least-squares，`max_nfev=200`，`ftol=xtol=gtol=1e-12`，不搜索新阈值。只在返回候选通过原 verifier 且同一目标函数严格下降时替换原候选。

初始 B 的精修执行用于测量新增处理成本，并检查其结果与已保存的原/精修配置相差小于 1e-10 rad；没有把它重新生成一份“新候选池”。历史初始求解不重跑。位置与姿态残差由 q 和固定 FK 唯一决定，本实验不声称可以在固定 q 上独立改变残差。

| 状态；来源；当前帧 | A 完成 | 仅当前精修完成 | B 完成 | A / 仅当前精修 / B：平均后续累计 ms | B 后续精修残差调用均值 |
| --- | ---: | ---: | ---: | --- | ---: |
| 00；CG-HIK；119 | 0/5 | 5/5 | 5/5 | 162.61 / 11.72 / 5225.09 | 48,000 |
| 01；TRAC-IK smooth；33 | 0/5 | 0/5 | 0/5 | 163.35 / 163.19 / 163.38 | 0 |
| 02；TRAC-IK；86 | 0/5 | 0/5 | 0/5 | 162.90 / 162.76 / 164.09 | 0 |
| 03；TRAC-IK；86 | 0/5 | 0/5 | 0/5 | 162.37 / 161.60 / 290.19 | 1,185.4 |
| 04；CG-HIK；124（25 帧） | 0/5 | 0/5 | 0/5 | 136.00 / 136.11 / 136.35 | 0 |
| 05；TRAC-IK；71 | 0/5 | 0/5 | 0/5 | 155.14 / 153.90 / 407.11 | 2,317.8 |
| 06；CG-HIK；95 | 0/5 | 5/5 | 5/5 | 162.81 / 12.15 / 5221.81 | 48,000 |
| 07；near-singular 成功对照；90 | 5/5 | 5/5 | 5/5 | 11.65 / 11.78 / 5208.84 | 48,000 |
| 08；smooth 成功对照；33 | 5/5 | 5/5 | 5/5 | 11.84 / 11.93 / 5206.02 | 48,000 |

计数以整个固定窗口每帧均接受为完成，不是原 150 帧轨迹重新完成。B 的五次首次失败：case 01 均为 34；case 02 均为 87；case 03 为 87、88、88、88、88；case 04 均为 125；case 05 均为 75。增加连续前缀不等于完成恢复。表中精修次数为零的失败窗口没有合法后续求解结果可精修，但**仍有初始精修成本**，并非 B 总计算为零。

### 计时和函数评价的边界

新后续调用的外层计时包括：TRAC-IK 的 per-query 边界设置与求解、原适配观测/验证、精修（如适用）及最终验证。独立解释量、下一目标离线优化、序列化和事后 witness 复核在计时外。普通新候选的投影是离线生成成本，单独记录，不隐匿成免费在线候选生成。

当前给定候选的精修成本与后续窗口分开列出：B 的初始精修平均约 **48.79–175.72 ms/状态**，45 次共 6,196.63 ms、56,670 次残差调用，不包含历史当前候选的求解成本。后续共有 619 次合法候选精修，均降低原目标函数；其中 444 次达到额外报告的 1e-7 m/rad 严格残差目标。后续总残差回调数为 **977,516**，SciPy 报告的 `nfev` 合计为 122,221，二者不混为一谈。

精确计数的是每次残差函数实际调用，包括有限差分 Jacobian 的额外调用。`max_nfev=200` 沿用旧预算，并不表示含有限差分的总回调数最多 200；全成功 B 窗口达到约 1,600 次回调/帧。实际环境为 Python 3.12.13、SciPy 1.17.0、NumPy 2.4.4；计数字段及优化定义参见 [SciPy least_squares 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。

**原生 TRAC-IK 不暴露其内部 FEV，因此 native FEV 和完整 pipeline 总 FEV 均保留为空，而非记为零。** 已交付所有可观测的精修回调数、优化器 nfev、精修内 verifier 调用数与实际总时间。不能据此报告完整总 FEV 比，更不能把恢复包装成低成本方法或 50 Hz 实时结果。

完整五次结果及初始/后续成本见 [uniform_refinement_comparison.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/uniform_refinement_comparison.json) 和 [逐次运行表](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/new_continuation_runs.csv)。

## 3. 实际末端位姿匹配后的构型效应

每个原候选和旧精修候选分别作为中心。先用一致缩放 Jacobian 的零空间方向做固定有限扰动，再固定扰动中最大的零空间关节坐标，对其余六个关节做非线性投影。投影目标是**中心实际达到的完整 FK 位置和旋转矩阵**，不是用户目标，也不是相同误差范数。共同 previous_q 的速度区间与 joint range 始终不变。

运行前固定的匹配标准是位置范数 ≤1e-8 m、姿态旋转向量范数 ≤1e-8 rad；构型最大关节差至少 1e-5 rad。最多保留每中心三个、每状态六个候选；有限尝试失败即记不可用，不放宽精度或约束。43 个保留候选实际 FK 最大匹配差为 **2.61e-16 m / 6.09e-16 rad**。全部从共同 previous_q 通过原 verifier；未使用未来 q_ref，也没有向不可达状态跳转。

| 状态 / 来源 | 原中心 / 精修中心新增匹配数 | 匹配候选的整个窗口结果 | 解释 |
| --- | --- | --- | --- |
| 00 / CG-HIK | 3 / 0 | 原中心及三个匹配候选均 0/5 | 精修中心无可用匹配构型，不能完全分离其恢复中的残差与构型作用。 |
| 01 / TRAC-IK smooth | 3 / 3 | 每个中心均 0/5；各有两个匹配候选 5/5、一个 0/5 | 四个同位姿替代构型稳定完成，统一精修 B 仍 0/5。 |
| 02 / TRAC-IK | 3 / 3 | 全部 0/5 | 四个候选的中位连续前缀增加一帧，但未完成。 |
| 03 / TRAC-IK | 3 / 1 | 全部 0/5 | 一个原中心匹配候选的中位前缀从 0 增至 1。 |
| 04 / CG-HIK | 3 / 0 | 全部 0/5 | 精修中心无可用匹配构型。 |
| 05 / TRAC-IK | 3 / 3 | 全部 0/5 | 部分匹配候选把中位前缀从 1 增至 2，或从 2 增至 3。 |
| 06 / CG-HIK | 3 / 0 | 原中心及三个匹配候选均 0/5 | 精修中心无可用匹配构型，尚不能单独归因为残差。 |
| 07 / 成功对照 | 3 / 3 | 中心及全部匹配候选均 5/5 | 未损失完成。 |
| 08 / 成功对照 | 3 / 3 | 中心及全部匹配候选均 5/5 | 未损失完成。 |

“无可用匹配构型”仅指固定的有限零空间扰动/投影没有找到，不代表证明这样的构型不存在。

### 关键实例：case 01，原候选中心

UID：`8007124479c3c7bc724a1818507bb840030dd31b60be2b3793048463dfa14ffc`。共同输入为第 33 帧目标及同一 previous_q；后续普通 TRAC-IK 跟踪第 34–63 帧。

| 候选 | 相对中心最大关节差 rad | 当前归一化残差范数 | 一致缩放 σmin | 下一目标线性需求 | 后续完成 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原中心 candidate_00 | 0 | 4.96593e-5 | 0.497972 | 0.130138 | 0/5 |
| matched_00 | 0.0109 | 4.96593e-5 | 0.491284 | 0.128863 | 0/5 |
| matched_01 | 0.0109 | 4.96593e-5 | 0.537114 | 0.130932 | 5/5 |
| matched_02 | 0.0218 | 4.96593e-5 | 0.532779 | 0.127111 | 5/5 |

中心的位置残差约 4.79706e-8 m、姿态残差约 1.12050e-7 rad；匹配候选保持的不只是表中范数，而是完整实际末端位姿。matched_01 到 previous_q 的欧氏距离为 0.045811 rad，比原中心的 0.044709 rad 更远；其下一目标线性需求还略高于原中心，却完成了全部五次窗口。因此“最近解”和“较小线性需求”均不能完整解释这个成功对照。精修中心 candidate_05 的 matched_04/05 重现相同 0/5→5/5 现象。

所有投影尝试、失败原因、中心实际位姿、匹配向量及合法性见 [projection_attempts.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/projection_attempts.json)。完整续接对照见 [pose_matched_comparison.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/pose_matched_comparison.json)。

## 4. 下一目标需求是否提供了更好的解释？

当前解释量为：完整平移/旋转残差向量、分别按公共位姿容差归一化后的向量及范数、逐关节步长与速度利用率、上下限位余量，以及一致无量纲化 Jacobian 的奇异值与条件数。138 个候选的这些量全部保存；没有把上一轮未缩放的 σmin 与本轮缩放值直接比较。

令 `D = diag(position_tolerance × 3, orientation_tolerance × 3)`，`S = diag(velocity_i × dt + velocity_tolerance)`，`A = D⁻¹ J(q) S`，`e = D⁻¹ pose_error(next_target, FK(q))`。在全部七个关节变量 z 上数值求解：

```text
minimize u
subject to -u <= z_i <= u,  u >= 0
           q_min <= q + S z <= q_max
           ||e_position - A_position z||₂ <= 1
           ||e_rotation - A_rotation z||₂ <= 1
```

这允许利用公共位置/姿态容差，不强迫精确命中目标；逐关节速度归一化和关节范围均在问题中。使用 bounded linear least-squares 初始化，再以固定 SLSQP 参数求解；全部 138 次报告数值成功。它是局部几何 Jacobian 线性模型的解，不是由单个伪逆向量代表冗余解集，也不是非线性全局可行性证书。优化器状态、约束余量和返回值均保留，失败分支定义为未知而非数学不可行。[SLSQP 定义](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。

每个线性解 `q + S z` 均再次经过真实非线性 FK 和原 verifier，不将超速解裁剪后冒充原结果。**138 个线性需求均 ≤1，但只有 78 个非线性检查通过；其余 60 个因位置和/或姿态误差被拒绝。** 例如 case 01 原中心对应线性解的位置误差为 1.01880 mm、姿态误差为 0.00879951 rad，均略超公共边界。这反映线性最小步长解常落在容差边缘，不能直接当作真实可接受命令；这些诊断解从未提交到续接实验。

对有完成数差异、或完成数相同但中位连续前缀不同的同状态候选对，比较各指标排序是否与结果排序一致。匹配精度范围内的残差差别计为平局，平局记 0.5。只纳入相同内部边界下的普通续接，不混入 B 的额外逐帧预算。

| 解释量（预定方向） | 全部 82 个有差异候选对 | 其中 29 个完成数有差异对 | 25 个等位姿组内有差异对 |
| --- | ---: | ---: | ---: |
| 当前残差更小 | 75.0% | 72.4% | 50.0%（全部平局） |
| 离 previous_q 更近 | 17.1% | 13.8% | 24.0% |
| 一致缩放 σmin 更大 | 91.5% | 86.2% | 96.0% |
| 下一目标最小需求更小 | 74.4% | 72.4% | 68.0% |

这是所选状态上的依赖候选对排序摘要，不是训练/测试分类准确率，也不能据此估计泛化性能。缩放 σmin 在本批样本表现最好，但仍有反例且没有独立验证。下一目标需求在等位姿组内比不区分构型的残差有更多区分能力，却未优于单一缩放奇异值。不能据此宣称需要更复杂的下一目标准则。

下一目标诊断允许读下一目标，是离线解释量，不进入投影生成、统一精修或 TRAC-IK 在线调用策略。实际 TRAC-IK 下一帧的五次非线性求解结果另存于 [next_frame_outcomes.csv](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/next_frame_outcomes.csv)，与线性诊断解的 FK 检查明确分开。完整特征、诊断、全部有差异候选对见 [candidate_diagnostics.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/candidate_diagnostics.json) 和 [informative_pairs.csv](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/informative_pairs.csv)。

## 5. 机制图、witness 与验证

![残差、冗余构型、下一目标需求与续接结果](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/final_figures/residual_configuration_mechanism.png)

图中 a 为全部九状态的五次窗口完成；b 为原/精修中心及全部 43 个位姿匹配构型（O/R 仅指中心来源，不是机器人分支编号）；c 为依赖候选对的描述性排序；d 为五次测量的中位数与 min–max 范围，星号标明历史复用计时。没有 p 值或独立帧推断。源数据和可编辑矢量文件位于同目录。首次图形导出保留；最终导出只改对数刻度显示，以保证所有 PDF 字形不小于 5 pt，不更改实验数值。

新增 **100 份完整成功 witness**，包括：TRAC-IK 来源匹配候选 20 份、CG-HIK 来源统一精修 10 份、成功对照 70 份。每份保存当前合法接入和整个后续窗口的实际关节序列、目标及 previous_q；不是通过跳到 q_ref 得到的“恢复”。

- [case 01 / matched_01 / repeat 0：合法接入 33，验证续接 34–63](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/successful_witnesses/case_01_pre_failure_matched_01_ordinary_matched_r0.json)。
- [case 01 / 精修中心 matched_04 / repeat 0](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/successful_witnesses/case_01_pre_failure_matched_04_ordinary_matched_r0.json)。
- [case 00 / 统一精修 / repeat 0](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/successful_witnesses/case_00_pre_failure_candidate_00_uniform_refinement_r0.json)。

独立复核覆盖 43 个匹配候选的实际 FK 和当前合法性，以及所有新增接受命令、真实状态反馈和成功 witness，不再调用求解器。[verification.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/verification.json)记录数量，[delivery_manifest.json](/home/eric/wjg/btry/outputs/continuation_mechanism_study/residual_configuration_disentanglement/delivery_manifest.json)绑定交付文件。八项定向测试通过，包括冗余分配、公共容差利用、关节范围、旧精修目标/预算一致性、fallback、匹配 FK 和原 verifier。

独立入口为 [disentangle.py](/home/eric/wjg/btry/src/confik/continuation_mechanism/disentangle.py)，数学干预及只读汇总位于同一现有模块；没有建立版本化 runtime。统计技能用于限定独立单位和解释力度，科学制图及 PDF 技能用于完整源数据、可读字体和最终实图核查。本轮交付后停止，不自动提出或开发新的续接算法。
