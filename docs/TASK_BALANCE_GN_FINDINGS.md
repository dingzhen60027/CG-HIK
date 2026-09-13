# 位置—姿态容差均衡 GN：原环境开发结果

基线：`d345fdbfe24ce2d0dedbf3bfc2c09ee7cb62089b`；分支 `codex/hierarchical-v5`。
结果根目录：[task_balance_development](../outputs/single_solver_evidence/task_balance_development/)。
本轮只实现任务书的两误差块 minimax／标量对偶机制。没有新增 IK 兜底、未来信息、目标、权重网格或正式轨迹；原 GN、纯超差实现、历史结果和论文未改。

## 结论

1. **保留了这批同输入恢复。** 52 个历史唯一输入中，新 κ=0/1 均接受 Panda 21/41、UR5e 6/11；与纯超差的成功输入集合完全相同。同 κ 原 GN 为 4/41、1/11；新增恢复 17、5 个，无局部损失。直接原 GN κ0/κ1 来源的 16 个子集中，新方法为 5/13、1/3，原 GN 均为零。三遍结果在这些输入上没有成功与失败的混合。
2. **避免了纯超差普遍贴边，但主要保留的是原 GN 正常行为。** 新方法的合格命令贴边比例为 Panda κ0/κ1 的 0.988%/0.933%，UR5e 均为 0.667%；纯超差约 99.27%/99.26%。同 κ 原 GN 的这些比例相同。Panda 加速度 RMS 几乎与原 GN 相同，UR5e 有约 0.00046 rad/s² 的小幅增加；不是通过固定收紧公共容差取得。
3. **恢复了 UR5e trajectory_30，无其他 UR5e 几何完成损失。** 新 κ0/κ1 三遍均为 40/40，纯超差三遍为 39/40。但原 GN 本来也是 40/40，因此这是相对纯超差的恢复，不是相对原 GN 的新轨迹能力。
4. **同 κ 下没有新增完整轨迹完成。** Panda 新 κ0 为 39/40、新 κ1 为 40/40，分别与原 κ0、κ1 的 UID 集合相同；κ0 的 trajectory_14 仍失败。不能把 κ1 的 40/40 归因于 minimax。
5. **额外成本尚未换来完整任务增量。** 相对同 κ 原 GN，新方法累计外层时间为 Panda 1.941/1.829 倍、UR5e 1.802/1.792 倍。相对纯超差更便宜、误差与运动更正常，但困难固定子问题充分求解时，现有 Clarabel 通常比当前 NumPy 标量对偶快。支持“局部竞争输入可恢复、正常帧接近原 GN”的机制结论，未建立更好的整体完成—成本折中或专用对偶内核的普遍效率优势。

## 1. 原环境验收、实现和计量范围

先对包内现成命令验收，未运行 IK 替换它。Panda `trajectory_094`、第 53 帧、原始 UID
`c2ebae0d24e306f3744e9ac2a38add8502705448e7eda234f19e3082f56fa52c`：

- 输入的 target、previous_q、dt 与原始首次失败 CSV 完全一致；不是其他方法的有利状态。
- 原 verifier 接受：位置 **0.963749970 mm**、姿态 **0.471467712°**；有限性、关节范围和速度均通过，最大速度利用率 **0.999999999999**。
- 完整关节向量、原输入和包文件 hash 见 [原环境验收记录](../outputs/single_solver_evidence/task_balance_development/package_verification/original_verifier.json)。这不是安全裕量或长期稳定性证据。

新文件 [task_balance_gn.py](../src/confik/task_balance_gn.py) 直接组合原 `box_qp`、NativeGeometry、SO(3) 导数和 verifier，不继承历史预测算法。维持 λ=.01、最多 30 外层／16 对偶／8 回溯，κ=0/1，θ 每次从 .5 开始；使用共享的绝对 20 ms deadline，并保留无法抢占的原生调用及其晚到结果。

主合同不变：1 mm、0.5°、dt=.02 s、原逐关节速度与 `velocity_tolerance=1e-4`。动态关节区间在实际 previous_q 上形成并在本帧固定；采用原可表示内部端点，不缩小公共位姿容差。真正更新非合格中间点时检查非线性 M 下降。当前状态或全步已合格即停止，不额外精修。

实测仍为原服务器 NumPy 执行路径：NumPy 2.4.4、SciPy 1.17.0、Pinocchio 3.9.0；未升级环境或启用新加速器。原可选 Numba 因现有 coverage 接口问题不可用，沿用已有明确的 NumPy fallback 设置。这是数值执行后端选择，不是 IK fallback。

五设置均使用共同的 3 次非零中点 warmup 和运行器 1 次 stationary startup；计时外 warmup 与输入相同。逻辑 CPU 4，BLAS/OpenMP 单线程，按整条轨迹交错方法。外层时间含转换、区间、残差/Jacobian、箱 QP/标量对偶、回溯与原验收；文件序列化不计入。全部 150 帧、晚到及失败保留。

固定协议和输入在比较前写入 [protocol.json](../outputs/single_solver_evidence/task_balance_development/protocol/protocol.json)。52 个输入直接读取上一阶段冻结 inventory，未重新搜集以混入本轮失败。实际执行 780 次同输入测量、1200 次完整轨迹运行（180,000 帧）、360 个固定子问题的 5400 次数值对照。详细 trace 为额外同输入诊断，不混入主计时。

## 2. 全部历史输入，不只展示成功原型

下表为“三遍全部通过”的唯一输入数。搜索/计时重复不是额外独立输入；52 个状态也不是 52 条独立轨迹。

| 来源范围 | 机器人 | 原 GN κ0 | 原 GN κ1 | 纯超差 κ0 | 新均衡 κ0 | 新均衡 κ1 |
|---|---|---:|---:|---:|---:|---:|
| 全部历史输入 | Panda（41） | 4 | 4 | 21 | 21 | 21 |
| 全部历史输入 | UR5e（11） | 1 | 1 | 6 | 6 | 6 |
| 直接 κ0/κ1 来源 | Panda（13） | 0 | 0 | 5 | 5 | 5 |
| 直接 κ0/κ1 来源 | UR5e（3） | 0 | 0 | 1 | 1 | 1 |

全组来自 Panda 13 个、UR5e 3 个来源 UID；直接子组涉及 10、2 个来源 UID。逐 UID 的输入数与恢复数见 [local_source_uid.csv](../outputs/single_solver_evidence/task_balance_development/reports/local_source_uid.csv)，由 `source_data.local.units` 按 robot/source_uids/method 展开汇总，输入在各 UID 内只计一次；不把来源别名当重复实验。

局部命令、两块归一化残差、rho、真实关节步长、速度利用率、耗时与 θ 轨迹见 [local_commands.csv](../outputs/single_solver_evidence/task_balance_development/reports/local_commands.csv) 和 [完整诊断](../outputs/single_solver_evidence/task_balance_development/local/diagnostics/)；[合法 witness](../outputs/single_solver_evidence/task_balance_development/local/legal_witnesses.json) 带自身 previous_q/target/dt。

新方法并未恢复剩余 20 个 Panda、5 个 UR5e 输入。全输入延迟 P50/P95/P99（ms）为：Panda κ0 **2.936/20.241/20.276**、κ1 **2.940/20.269/20.292**；UR5e κ0 **1.359/20.236/20.256**、κ1 **1.346/20.224/20.255**。恢复数不代表全部请求更快；未找到合格命令不等于数学不可行。

## 3. 完整开发轨迹：同 κ 归因

每机器人原 40 条、四类各 10、每条 150 帧，三遍完整运行。TSR 是全帧几何验收通过；DTSR20 还要求每帧外层时间 ≤20 ms。下面的“累计”是 40 条全部目标的一遍总时间，在三遍间取平均，未剔除失败或超时。

| 机器人／设置 | 完成数，三遍（/40） | 按时完成数，三遍（/40） | P50/P95/P99 ms | 累计 s/遍 | 合格命令贴边 % | 加速度 RMS rad/s² |
|---|---|---|---|---:|---:|---:|
| Panda 原 κ0 | 39/39/39 | 39/39/39 | .2655/.2850/.4412 | 1.7047 | .988 | 3.09334 |
| Panda 原 κ1 | 40/40/40 | 40/40/40 | .2657/.2813/.3735 | 1.5991 | .933 | 3.21435 |
| Panda 纯超差 κ0 | 40/40/39 | 40/40/39 | .7565/1.0911/1.3730 | 4.5220 | 99.267 | 3.83291 |
| Panda 新 κ0 | 39/39/39 | 39/39/39 | .4855/.5104/.8560 | 3.3082 | .988 | 3.09334 |
| Panda 新 κ1 | 40/40/40 | 40/40/40 | .4862/.5087/.7235 | 2.9240 | .933 | 3.21435 |
| UR5e 原 κ0 | 40/40/40 | 40/40/40 | .2595/.2754/.3374 | 1.5606 | .667 | 5.74385 |
| UR5e 原 κ1 | 40/40/40 | 39/40/40 | .2593/.2752/.3330 | 1.5760 | .667 | 5.79604 |
| UR5e 纯超差 κ0 | 39/39/39 | 39/39/39 | .7297/1.0729/2.3287 | 5.0852 | 99.256 | 9.45742 |
| UR5e 新 κ0 | 40/40/40 | 40/40/40 | .4703/.4914/.5574 | 2.8116 | .667 | 5.74432 |
| UR5e 新 κ1 | 40/40/40 | 40/40/40 | .4709/.4951/.6426 | 2.8244 | .667 | 5.79650 |

贴边定义为接受命令 `max(position/epsilon_p, orientation/epsilon_R)>.9`。例如新 Panda κ1 接受误差的位置 P95/max 为 .433419/.998420 mm、姿态 P95/max 为 .031602/.476619°；UR5e κ1 为 .368595/.998079 mm、.029775/.495179°。相同 κ 原 GN 数字按这一级精度相同。所有误差分布、真实步长、帧成功率、分位数与最大延迟见 [主表](../outputs/single_solver_evidence/task_balance_development/reports/main_table.csv)。不将低贴边率解释为实机安全或闭环保证。

### 全部类别与 UID

- **Panda smooth/high-curvature**：五设置三遍均 10/10；**near-singular**：原 κ0、新 κ0 均 9/10，其余 10/10；**joint-limit-return**：除纯超差第三遍 9/10，其余 10/10。
- **UR5e smooth/near-singular/joint-limit-return**：五设置几何均 10/10；**high-curvature**：纯超差 9/10，其余 10/10。原 κ1 的 joint-limit-return 第一遍按时完成 9/10，其余完整方法均 10/10；纯超差的几何失利仍算按时未完成。
- 逐类成本、误差与运动全部保留于 [family_table.csv](../outputs/single_solver_evidence/task_balance_development/reports/family_table.csv)，不以全样本均值遮盖类别。

UR5e `trajectory_30`（UID `382307fdf25e31f8b1ae5c14a7545e81bf0c54489223c4626934c3dbe31a195e`）纯超差仍从第 32 帧失败，新 κ0/κ1 三遍全部完成。相对纯超差恢复这一 UID，无 UR5e 新损失；同 κ 原 GN 无新增恢复或损失。

Panda `trajectory_14`（UID `4e626ba28da6a8d45d22f023c3732e2d796c83a0e74df2aa41168fe26f6dccab`）原 κ0、新 κ0 都从第 122 帧失败。新 κ0 在该帧返回约 1.086575 mm/.543288°，仍不合格；原 κ0 为约 1.315487 mm/.242491°。误差更均衡并未转化为合法命令。κ0 相对纯超差在该 UID 上仍有稳定损失。

**保留测量波动，不重新运行替换：** Panda 纯超差 `trajectory_20`、repeat2/frame140 的 deadline 返回不合格位置 1.008526 mm，外层 46.663544 ms，因此本轮该方法为 40/40/39，而非把历史 40/40 搬进新表。UR5e 原 κ1 存在一个 53.228097 ms 的合法晚到命令，使一次 DTSR20 减一；没有证据将单次晚到归因为算法固有缺陷或某个确定系统原因。新 Panda κ0 的 24 个 >20 ms 调用都保留，不能因为其 TSR 已失败而删除成本。

### 配对统计与常规帧行为

每个 UID 先平均三遍，再按固定四类配对 bootstrap 4000 次，给出未作多重性校正的描述性 95% 区间。时间比新／同 κ 原 GN：

| 机器人 | κ0 时间比 [95% 区间] | κ1 时间比 [95% 区间] |
|---|---|---|
| Panda | 1.941 [1.813, 2.143] | 1.829 [1.820, 1.838] |
| UR5e | 1.802 [1.797, 1.806] | 1.792 [1.749, 1.818] |

相同 κ 的 TSR 差在这批 UID 上都为零；相同成功向量产生退化的 [0,0] bootstrap 区间，不是等效性试验、普遍不退化保证或新样本预测。UR5e κ1 的 DTSR20 平均差为 .833 百分点 [0,2.5]，来自前述单次晚到，不能据此建立稳定时限优势。[完整配对结果](../outputs/single_solver_evidence/task_balance_development/reports/paired_comparisons.csv) 未删除零差或不利区间。

| 新方法 | θ=.5 首外层全步即返回／全部帧 | 继续对偶的帧数／18000 | 继续对偶且最终合法的帧数 | 平均残差/Jacobian 评价 | 平均原 verifier 调用 |
|---|---:|---:|---:|---:|---:|
| Panda κ0 | 95.667% | 156 | 72 | 4.522 | 3.283 |
| Panda κ1 | 95.983% | 102 | 102 | 3.993 | 2.977 |
| UR5e κ0 | 96.683% | 9 | 9 | 3.970 | 2.968 |
| UR5e κ1 | 96.683% | 9 | 9 | 3.970 | 2.968 |

这些帧数含三遍，同一帧不当作三个独立恢复。原 GN 的平均残差/Jacobian 调用为 Panda κ0/κ1 4.421/3.942、UR5e 3.936/3.936；原代码每次最后调用一次 verifier，但旧 raw 未独立暴露该计数字段，主 CSV 保留空值，不伪造在线计数。新实现真实计数含全步试验及末次检查；单纯数外层迭代会漏掉这部分成本。

正常帧大多用同 κ 原 GN 对应的 θ=.5 方向。新／原完整反馈状态的最大差为 Panda κ0/κ1 9.69e-7/6.14e-7 rad，UR5e 均约 8.78e-5 rad；差异后的状态不是相同输入，见 [完整状态差表](../outputs/single_solver_evidence/task_balance_development/reports/closed_loop_command_differences.csv)。不能把“继续对偶后合法”的所有帧称为原 GN 必败帧；可归因的局部增量来自前述严格同输入比较。

## 4. 固定真实子问题：质量与完整调用成本

在结果出现前固定 **360 个问题**：256 个常规 first-outer 问题（每机器人每类前两条开发 UID，各 κ 从原 GN repeat0 的实际 previous_q 取 8 个时间均匀的需更新帧），以及全部 52 个历史输入的两个 κ、共 104 个问题。矩阵、输入、来源 hash 与选择计数在 [subproblems.json](../outputs/single_solver_evidence/task_balance_development/protocol/subproblems.json)。这里只代表这些真实初始局部问题，不冒充全部内迭代的自然频率。

关闭提前验收，比较固定 16 次上限标量对偶与既有 Clarabel 的同一 epigraph SOCP（两平方范数锥、同一 R、同一箱）。Clarabel 缓存 CSC 稀疏结构并更新数据，未加人为 setup 惩罚，`max_iter=100`、feas/gap 1e-9、单线程。质量先检查箱违反≤1e-8，并用共同数值下界检查 gap≤1e-9+1e-7×max(1,|upper|,|lower|)。下界是内 QP 法锥残差修正与箱内强凸 minorant 的较强者；均为浮点诊断，不是形式证书。不以 `Solved`/`AlmostSolved` 字样代替质量复算。

下表保留**全部调用**，P50 是包括转换、构造/更新、求解、复算检查的调用时间（ms）。质量通过数按 5 次重复列出，括号给问题数；不合格调用不算“更快胜出”。

| 机器人／来源 | κ | 对偶合格／调用 | Clarabel 合格／调用 | 对偶 P50 / P95 | Clarabel P50 / P95 |
|---|---:|---:|---:|---|---|
| Panda 常规（64） | 0 | 320/320 | 320/320 | .1216/.2021 | .1456/.1703 |
| Panda 常规（64） | 1 | 320/320 | 320/320 | .1838/.3205 | .1453/.1699 |
| Panda 失败输入（41） | 0 | 205/205 | 205/205 | .4417/1.1939 | .1536/.1809 |
| Panda 失败输入（41） | 1 | 205/205 | 205/205 | .4548/1.3291 | .1534/.1794 |
| UR5e 常规（64） | 0 | 300/320 | 320/320 | .1232/.9792 | .1449/.1712 |
| UR5e 常规（64） | 1 | 315/320 | 315/320 | .1759/.4658 | .1411/.1651 |
| UR5e 失败输入（11） | 0 | 55/55 | 55/55 | .4347/.4662 | .1433/.1578 |
| UR5e 失败输入（11） | 1 | 55/55 | 55/55 | .4331/.4608 | .1445/.1576 |

共同质量通过的是 355/360 个问题（1775/1800 对嵌套重复）。标量对偶有 5 个 UR5e 常规问题达到 16 次后仍 `inexact`，最大归一化 gap=1.426e-5；未追加更新来美化结果。Clarabel 一个常规 κ1 问题也未达到共同质量阈值，最大 gap=1.118e-7，完整保留。共同通过子集、各自原生状态、原始 gap、上/下界、目标差、P99 与 kernel/update 分段成本均见 [质量—时间表](../outputs/single_solver_evidence/task_balance_development/reports/subproblem_quality_time.csv) 与 [逐调用质量](../outputs/single_solver_evidence/task_balance_development/reports/subproblem_quality_calls.csv)。

Clarabel 6/7 维结构初始化为 .3715/.1244 ms，第一次 setup+更新约 .0754/.1042 ms，之后常规更新中位约 .022 ms；第一次完整调用 .2173/.2369 ms。原始 setup 调用单列于 source_data，未把重复 setup 叠加到每次调用。困难输入充分求解时，Clarabel 内核中位约 .056–.062 ms，对偶约 .393–.414 ms，差别不只是 Python 输入转换。常规 κ0 对偶有中位调用优势，但不具备全面的均值/尾部或困难问题优势。

**提前返回另列：** 常规各组 64 个问题里 63 个在 θ=.5 首全步已合格；历史输入每个 κ 下，Panda 有 16/41 在本次局部全步过程中变合法（4 个在 θ=.5、12 个在继续对偶后），UR5e 为 5/11（1+4）。这不是完整 IK 的恢复数，也不是同精度 minimax 性能比较。[early 表](../outputs/single_solver_evidence/task_balance_development/reports/subproblem_early_summary.csv) 包括真实原 verifier 成本；提前接受 gap=null、converged=false，没有填造最优性。

## 5. 一页数学与已有工作边界

令两块残差按原位姿容差归一化，`M=max(||e_p||²,||e_R||²)`。因此 **M≤1 当且仅当两项位姿条件都通过**；关节范围、从实际 previous_q 出发的速度及有限性仍需独立满足。M 在合法域不恒为零，但本算法找到真正合法命令即可停止，不承诺使它最小。

在本帧固定动态区间中，以 S=diag(v_max dt+epsilon_v)、G=J_e S、c=(q−q_mid)/joint_span、W=diag(S/joint_span) 形成

`min_{lo≤d≤hi} max{f_p(d),f_R(d)} + R(d)`，
`f_b=||e_b+G_b d||²`，`R=λ||d||²/2+κ||c+Wd||²/2`。

epigraph 约束为 f_p≤t、f_R≤t。非空箱上可将 t 取充分大，使用精炼 Slater 条件（必要时消去固定坐标）得到强对偶；λ>0 保证每个加权内问题强凸。消去 t 后两非负乘子之和为 1，得到

`max_{0≤θ≤1} min_{lo≤d≤hi} θ f_p+(1−θ) f_R+R`。

固定 θ 的箱 QP 有
`H=2θ G_pᵀG_p+2(1−θ)G_RᵀG_R+λI+κW²`，
`g=2θ G_pᵀe_p+2(1−θ)G_Rᵀe_R+κWc`。
θ=.5 时 H/g 对应原 GN 同 κ 的 H/g；不是再运行一个原 GN 求解器。内解准确时，值函数导数为 f_p−f_R；在活动面固定且自由变量集合 F 不变时，二阶导数为 `−z_Fᵀ H_FF⁻¹ z_F`，`z=∇f_p−∇f_R`。活动集切换时保守括区，退回二分；有限内解不自动具有该准确曲率。

可行 d 给原始上界 U。对给定 θ，设加权梯度为 g_theta，选择合法箱法锥向量 n，使 stationarity residual r=g_theta+n；强凸性给 `L=h_theta(d)−||r||²/(2λ)`。共同质量检查还在原箱上最小化 `h_theta(d)+g_thetaᵀδ+λ||δ||²/2`，避免把“几乎在边界”的点误当成法锥成员。两者只是数值下界诊断。16 次达到上限为 inexact，不是最优或不可行结论。

真实全步通过原 verifier 即可 `task_feasible_early` 返回，**不等于 minimax 最优**。否则只有真实非线性 M 下降的中间更新才被采用。局部正则项参与形成方向，不推出完整非线性 M+posture 的全局下降定理，更不推出闭环稳定性。

数学检查：包内一维平衡解 d≈14/15、gap≈1.454e-13；包内 36 个随机凸问题的独立 SLSQP 对照最大相对目标差 6.423e-8、gap 3.918e-7。包内该随机诊断使用 32 次上限，不冒充在线 16 次已充分；实际 16 次的不足在上述真实子问题完整报告。新增与原超差回归共 21 项测试通过，包括 θ=.5 同 κ H/g 方向、有限差分对偶导数/曲率、Clarabel 同目标对照、deadline、inexact 和初始已合法命令不变。测试通过不是任务贡献证据。

文献核验采用 nature-academic-search 的引文核验流程；学术 MCP 未挂载、web 的 Crossref API 打开失败后，用 Crossref 直接 HTTP 核对两个 DOI，再读作者/出版社全文。结果是概念边界，不是未执行的成熟方法性能对比。

| 已有工作／核验来源 | 已有内容 | 本轮应如何定位 |
|---|---|---|
| Wang, Praveena, Rakita, Gleicher，RangedIK，ICRA 2023，9700–9706；[DOI](https://doi.org/10.1109/ICRA48891.2023.10161311)、[作者全文](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/2023_ICRA_RangedIK.pdf)，§III–IV | 对点目标、允许范围和偏好目标进行带参数损失的加权多目标优化，利用范围内自由度改善运动。 | 不能称首次利用任务容差。这里是两块最差利用率与可提前验收的箱 GN 对偶实现，未证明优于 RangedIK。 |
| Moe, Antonelli, Teel, Pettersen, Schrimpf，Frontiers in Robotics and AI 3:16，2016；[出版社全文及 DOI](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00016/full)，§3.2–4 | 集合任务、任务优先级、激活/解除以及切锥条件；达标任务可不参与更新。 | 不把“达标后停止”或允许域本身称为首次。本轮没有复现其优先级控制或不变性定理。 |
| Boyd & Vandenberghe，Convex Optimization，Cambridge 2004；[作者书页](https://web.stanford.edu/~boyd/cvxbook/)、[全文](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)，§4–5 | epigraph、凸强对偶和约束资格提供现成数学基础。 | 标量乘子消元及活动面计算是这些工具的具体应用，不是新的 minimax 或全局收敛理论。 |

## 6. 交付、复现与停止

唯一实验入口：[run_task_balance_development.py](../scripts/run_task_balance_development.py)，固定配置：[task_balance_development.yaml](../configs/task_balance_development.yaml)。各阶段目录独占创建，不覆盖已完成运行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=tmp/crik_dependencies/python:src
# 在无本轮结果的新工作副本中，先将包解压到该结果根的 task_package/。
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py verify
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py prepare
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py local
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py subproblems
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py trajectories --robot panda
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py trajectories --robot ur5e
/home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/run_task_balance_development.py report
```

原型、本地 κ0=39/40 的负结果、初始 full-dual 开发历史原样保存于结果根 `task_package/`。它们不是本服务器表。原环境 raw 命令在两个 `development_*/runs/`；[成功轨迹索引](../outputs/single_solver_evidence/task_balance_development/reports/successful_trajectory_index.json) 提供真实初始状态与可追踪逐帧 witness。全量复验 180,000 帧，179,567 个接受命令，接受违约为 0；复验未运行新 IK。逐输入、逐 UID、分类表、配对区间、所有失败与晚到记录均已保存。

接入期间仅修正了验收文件嵌套字段读取和报告代码括号错误，未对已有向量、方法、设置或任何测量结果进行替换；后者修复仅重启尚未开始写表的报告阶段。统计采用 nature-statistics 的完整轨迹单位与嵌套重复规则，未把帧或重复扩大为样本量。

本轮到此停止。支持的是可执行的同内核容差均衡和选定失败输入上的恢复；**尚无相对同 κ 原 GN 的完整轨迹新增能力、整体成本优势、独立泛化或投稿创新成立的证据**。不自动增加第三种目标、正式评估或论文写作。
