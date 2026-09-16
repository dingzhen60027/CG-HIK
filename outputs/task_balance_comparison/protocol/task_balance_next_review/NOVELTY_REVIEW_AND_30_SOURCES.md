# Task Balance GN：近年文献、新颖性与有限下一步方案

日期：2026-09-16。研究对象为 `CG-HIK/codex/hierarchical-v5` 当前的 Task Balance GN，重点依据 `docs/METHOD_AND_CONTRIBUTIONS.md` 和 `docs/TASK_SET_BOUNDARY_FINDINGS.md`；不是旧的学习级联入口CG-HIK稿件。

## 1. 范围与阅读深度

本表整理30项文献/基础来源，其中23项使用2023—2026年的论文或作者预印本版本，7项为理解创新边界不可省略的经典论文/教材。按主题、问题、机制及实验类型组织，不把全部条目称为同等贴近的基线。

关键冲突文献核对了方法/公式；邻近的生成式、解析式、运动规划论文主要核对作者摘要、出版记录和方法范围。阅读深度单列。不保证穷尽所有出版物，也不将未找到相同标题理解为新颖性证明。预印本与后来的期刊/会议版本不重复计数；“使用预印本版本”不声称其截至今日一定没有后续发表。

检索范围包括：task/ranged-goal IK，set-based/tolerance IK，minimax/Chebyshev scalarization，position-orientation tradeoff，constrained/differential IK，inexact Gauss–Newton/prox-linear，adaptive stopping，primal-dual gap relative reduction。使用作者、出版社、会议和机构来源。旧论文实验数字与当前方法不混用。

## 2. 最重要的新颖性发现

Zheng、Ma、Xue，IEEE TSP 2024，Eq.(23) 的LACC-FISTA：

`H_k(z_k(lambda)) - D_k(lambda) <= rho_l * [H_k(0) - H_k(z_k(lambda))]`。

当前方法：

`U - L <= eta * [P(0) - U]`。

二者是同型的“对偶间隙相对于已实现模型下降”的停止准则。当前的IK模型、箱约束、两块标量对偶、下界计算和命令验收可以不同，不能据此认定整个算法相同；但不能把停止不等式和80%局部下降的代数推论当作独立原创规则。原文§2.3及§3还讨论了更早的相关非精确方法。

RangedIK已经用范围损失，而不仅是固定权重点误差，因此当前fixed-weight GN不是完整RangedIK的替代。能否比现代容差感知方法更值得用，尚未由该基线排除。

## 3. 当前判断

- 已有两机器人、实际合法命令、同目标计算对照，实验劳动量充足。
- 成功与效率结果是真实证据，不因已有数学先例而失效。
- 两项“独立新原理”的定位不成立；更合理的候选贡献是一个面向双块验收的具体有界数值求解方法。
- 范围任务、minimax、标量乘子、相对gap准则、GN及活动集不是新概念。
- 当前最需要回答的是：相对最近的范围感知/直接约束方法，以及简单低预算同核求解，是否仍有值得采用的覆盖—时间折中。
- “逻辑闭环”“实验量”“算法新颖性”“录用可能”是四个不同判断，不互相替代。

## 4. 只执行一个有限的贡献确认阶段

唯一目标：让现有Task Balance GN面对最接近的可替代方法，确定应用型数值增量，不修改主方法。

A. 方法定位：引用TSP2024 Eq.23，把停止准则定位为已有原则在当前任务中的具体实现；将独立创新主张集中为一个完整求解结构。保留所有实际数值结果，不重写成负面结论或删除历史不利结果。

B. 最近对手与同核控制：
1. 范围感知对手：RangedIK原实现先通过作者例程；改到原范数球/速度合同的任何任务映射均单列。不能将其原坐标范围语义直接说成与范数球相同，也不能用适配失败的0成功证明优势。
2. 直接同合同SQP参照：以最小关节位移为目标，位置和姿态各自作为不等式、关节/速率为固定当前帧边界，使用现成可靠SQP与解析导数。这是标准优化参照，不作为主方法兜底，也不包装创新。
3. 固定局部QP预算1/2次：与现有minimax外层、初始化和任务验收完全相同；到预算返回最好箱内方向，继续外层而非强制宣告失败；不承担原本不需要的界计算开销。此组只判断相对进展控制是否优于简单少算，不发展新算法。

既有GN/固定权重/紧精度/Clarabel资料按原身份保留。新比较同批交错计时，不能与历史时间拼接。主算法参数不再调。现有2160点数据已被观察，新对照在其上的结果标为补充同输入比较；不能冒充新的独立泛化证据。

C. 工程意义：先明确拟定应用的容差和单帧运动限制来源。仅在现有材料不能解释适用场景时，复用同一机械臂和运动学接口，采集或采用一个真实任务来源的Cartesian目标序列；不再搭接触、碰撞或VLA系统。若只能合成，明确仿真身份，不声称部署收益。不得调整任务使基线失败。

阶段结束以证据确定结论：相对最近方法保留覆盖且节省时间，可以把整体方法作为应用型算法增量；若仅胜过固定点目标而不胜过合理范围方法，贡献限于这一基线差别；若固定1/2次更好，停止准则不作为实用优势。不得自动追加第三种loss、兜底或新模型去挽救不利结果。

本文件不是让Codex立即修改仓库的写授权，也不是已执行新实验的结果。

## 5. 30项来源与关联矩阵
### [01] RangedIK: An Optimization-based Robot Motion Generation Method for Ranged-Goal Tasks

**作者：** Wang et al.  
**年份/身份：** 2023；ICRA 2023  
**关系：** 直接相关：范围任务  
**核对层次：** 全文方法/公式及实验页

已有内容：Swamp/Swamp Groove等范围损失与加权多目标优化；任务范围不等于简单固定权重点误差。

对本项目的意义：比较范围内外损失与偏好，而非把普通加权GN称作RangedIK。

来源：https://graphics.cs.wisc.edu/Papers/2023/WPRG23/
全文或第二原始来源：https://graphics.cs.wisc.edu/Papers/2023/WPRG23/2023_ICRA_RangedIK.pdf

### [02] A New Inexact Proximal Linear Algorithm With Adaptive Stopping Criteria for Robust Phase Retrieval

**作者：** Zhong Zheng; Shiqian Ma; Lingzhou Xue  
**年份/身份：** 2024；IEEE Transactions on Signal Processing 72:1081–1093; DOI 10.1109/TSP.2024.3365933  
**关系：** 直接相关：停止准则  
**核对层次：** 全文：重点§2–3、Eq.18/Eq.23

已有内容：Eq.23以原始—对偶gap相对已实现模型下降终止内迭代，与当前U−L≤η(P(0)−U)同型。

对本项目的意义：停止不等式及其80%代数推论不能作为独立原创；任务、子问题和具体内核不相同。

来源：https://pure.psu.edu/en/publications/a-new-inexact-proximal-linear-algorithm-with-adaptive-stopping-cr/
全文或第二原始来源：https://arxiv.org/html/2304.12522v2

### [03] A New Inexact Manifold Proximal Linear Algorithm with Adaptive Stopping Criteria

**作者：** Zheng; Yu; Ma; Xue  
**年份/身份：** 2025；本次使用 arXiv:2508.19234v1（2025预印本版本）  
**关系：** 直接相关：非精确复合优化  
**核对层次：** 方法部分与摘要

已有内容：流形上的非光滑复合问题、自适应内层精度及收敛分析。

对本项目的意义：进一步说明任务几何与非精确求解的结合已有广泛数学基础。

来源：https://arxiv.org/abs/2508.19234
全文或第二原始来源：https://arxiv.org/html/2508.19234v1

### [04] An inexact variable metric proximal linearization method for composite optimization on manifolds

**作者：** Hao He; Ruyu Liu; Yitian Qian; Shaohua Pan  
**年份/身份：** 2025；2025首稿；本次版本页为arXiv:2508.12003v2（2026-05更新）  
**关系：** 相关：变量度量与非精确性  
**核对层次：** 作者摘要/版本页筛查

已有内容：变量度量、流形复合优化和非精确子问题。

对本项目的意义：引用背景；未据摘要推断其公式与Task Balance完全相同。

来源：https://arxiv.org/abs/2508.12003

### [05] Optimal inexactness schedules for Tunable Oracle based Methods

**作者：** Guillaume Van Dessel; François Glineur  
**年份/身份：** 2024；Optimization Methods and Software 39(3):664–698; DOI 10.1080/10556788.2023.2296982  
**关系：** 直接相关：计算—精度分配  
**核对层次：** 出版记录/作者摘要

已有内容：研究可调精度计算的总成本与求解精度安排。

对本项目的意义：少算内层步骤不是独有动机；需要与简单低预算及已有准则对照。

来源：https://www.tandfonline.com/doi/abs/10.1080/10556788.2023.2296982
全文或第二原始来源：https://arxiv.org/abs/2309.07787

### [06] Linear-time Differential Inverse Kinematics: an Augmented Lagrangian Perspective

**作者：** Bruce Wingo et al.  
**年份/身份：** 2024；RSS 2024; DOI 10.15607/RSS.2024.XX.110  
**关系：** 相关：数值结构（LoIK）  
**核对层次：** 会议作者摘要/出版页

已有内容：利用运动学结构形成增广拉格朗日求解和线性复杂度子问题。

对本项目的意义：以新的计算结构说明数值IK贡献；不是当前6/7维标量对偶的直接复现对象。

来源：https://www.roboticsproceedings.org/rss20/p110.html

### [07] Propagative Distance Optimization for Constrained Inverse Kinematics

**作者：** Yu Chen et al.  
**年份/身份：** 2024；本次使用 arXiv:2406.11572v1（2024预印本版本）  
**关系：** 相关：结构化约束IK（PDO-IK）  
**核对层次：** 全文方法筛查

已有内容：通过距离表述及传播结构处理受约束IK。

对本项目的意义：约束处理已有结构化替代方法，不能只把裁剪GN当传统方法代表。

来源：https://arxiv.org/html/2406.11572v1

### [08] Variable Step Sizes for Iterative Jacobian-Based Inverse Kinematics of Robotic Manipulators

**作者：** Jacinto Colan; Ana Davila; Yasuhisa Hasegawa  
**年份/身份：** 2024；IEEE Access 12:87909–87922; DOI 10.1109/ACCESS.2024.3418206  
**关系：** 相关：迭代工作控制  
**核对层次：** 出版/摘要核查；IEEE全文访问受限

已有内容：研究Jacobian迭代中步长选择对性能的作用。

对本项目的意义：当前方法不能把一般步长/阻尼调整重新命名为新原理。

来源：https://ieeexplore.ieee.org/document/10568941

### [09] Real-time inverse kinematics for robotic manipulation under remote center-of-motion constraint using memetic evolution

**作者：** Ana Davila; Jacinto Colan; Yasuhisa Hasegawa  
**年份/身份：** 2024；Journal of Computational Design and Engineering 11(3):248–264; DOI 10.1093/jcde/qwae047  
**关系：** 相关：明确应用约束（PivotIK）  
**核对层次：** 出版方全文方法与实验筛查

已有内容：围绕RCM约束结合数值搜索，并以同类方法和具体任务验证。

对本项目的意义：借鉴问题—方法—应用闭环，不将其硬件结果作为我们已有结果。

来源：https://doi.org/10.1093/jcde/qwae047

### [10] IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition

**作者：** Alexander J. Elias; John T. Wen  
**年份/身份：** 2025；Mechanism and Machine Theory 209:105971; DOI 10.1016/j.mechmachtheory.2025.105971  
**关系：** 邻近：解析几何IK  
**核对层次：** 作者摘要/出版元数据

已有内容：以几何子问题分解统一处理多类机械臂逆解。

对本项目的意义：主要解决几何根问题；不要求本项目增加不同结构的解析求解器。

来源：https://arxiv.org/abs/2211.05737

### [11] Automatic Geometric Decomposition for Analytical Inverse Kinematics

**作者：** Ostermeier; Külz; Althoff  
**年份/身份：** 2025；IEEE RA-L 10(10):9964–9971; DOI 10.1109/LRA.2025.3597897  
**关系：** 邻近：自动解析分解（EAIK）  
**核对层次：** 作者机构记录/摘要

已有内容：自动化几何分解与解析解生成。

对本项目的意义：新颖性边界来自问题结构与自动化，不是仅有更好成功数。

来源：https://portal.fis.tum.de/en/publications/automatic-geometric-decomposition-for-analytical-inverse-kinemati/
全文或第二原始来源：https://arxiv.org/abs/2409.14815

### [12] IKSel: Selecting Good Seed Joint Values for Fast Numerical Inverse Kinematics Iterations

**作者：** Xinyi Yuan; Weiwei Wan; Kensuke Harada  
**年份/身份：** 2025；本次使用 arXiv:2503.22234（2025预印本版本）  
**关系：** 邻近：初值选择  
**核对层次：** 作者摘要/版本页

已有内容：通过检索、排序和重新选择种子改善数值IK。

对本项目的意义：与当前无额外种子的单求解器不同；不回到旧学习路由分支。

来源：https://arxiv.org/abs/2503.22234

### [13] Inverse Kinematics with Vision-Based Constraints

**作者：** Liangting Wu; Roberto Tron  
**年份/身份：** 2024；本次使用 arXiv:2406.10682（2024预印本版本）  
**关系：** 邻近：视觉约束IK  
**核对层次：** 作者摘要/版本页

已有内容：将视觉条件纳入IK的几何/优化表述。

对本项目的意义：说明具体任务约束可产生明确方法问题；不是本轮新增视觉模块依据。

来源：https://arxiv.org/abs/2406.10682

### [14] Ensuring Viability: A QP-Based Inverse Kinematics for Handling Joint Range, Velocity and Acceleration Limits, as Well as Whole-body Collision Avoidance

**作者：** Yachen Zhang; Ryo Kikuuwe  
**年份/身份：** 2025；Jxiv预印本; DOI 10.51094/jxiv.1053  
**关系：** 相关：可持续可行性  
**核对层次：** 作者摘要/预印本说明

已有内容：联合关节范围、速度、加速度及碰撞约束。

对本项目的意义：当前单帧验收不等于递归可行/闭环安全，不借用其保证。

来源：https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/1053

### [15] Fast Functionally Redundant Inverse Kinematics for Robotic Toolpath Optimisation in Manufacturing Tasks

**作者：** Andrew Razjigaev et al.  
**年份/身份：** 2025；ACRA 2025；作者全文 arXiv:2512.10116v1  
**关系：** 相关：任务冗余与数值结构  
**核对层次：** 全文方法与应用筛查

已有内容：利用工具任务的功能冗余及数值更新处理制造路径。

对本项目的意义：容差或自由度必须由具体任务解释，不能只用有利随机压力输入讲应用。

来源：https://arxiv.org/html/2512.10116v1

### [16] Actuator-Aware Inverse Kinematics with Joint-Limit Admissibility for Torque-Controlled Redundant Robots

**作者：** Mohammad Dastranj; Mahdi Hejrati; Jouni Mattila  
**年份/身份：** 2026；本次使用 arXiv:2605.31436v1（2026预印本版本）  
**关系：** 相关：执行器与约束可执行性  
**核对层次：** 全文方法筛查

已有内容：把执行器特征、任务松弛及关节限制纳入速度求解。

对本项目的意义：当前URDF软件命令验收不能宣称已经验证真实执行器性能。

来源：https://arxiv.org/html/2605.31436v1

### [17] CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning

**作者：** Jeremy Morgan; David Millard; Gaurav S. Sukhatme  
**年份/身份：** 2024；ICRA 2024; DOI 10.1109/ICRA57147.2024.10611724  
**关系：** 邻近：路径级生成  
**核对层次：** 作者摘要/出版元数据

已有内容：沿Cartesian路径生成候选，并进行路径搜索与数值优化。

对本项目的意义：轨迹规划问题与当前帧命令生成不同；不能要求当前方法解释全部路径选择。

来源：https://arxiv.org/abs/2309.09102

### [18] Generative Graphical Inverse Kinematics

**作者：** Oliver Limoyo et al.  
**年份/身份：** 2025；IEEE Transactions on Robotics 41:1002–1018; DOI 10.1109/TRO.2024.3521862  
**关系：** 邻近：生成式IK  
**核对层次：** 作者摘要/出版信息交叉核对

已有内容：利用距离几何图结构生成多解并研究跨机器人表征。

对本项目的意义：与固定已知模型下的容差可行性问题不同；2026摘要转载不重复计为新论文。

来源：https://arxiv.org/abs/2209.08812

### [19] CycleIK: Neuro-inspired Inverse Kinematics

**作者：** Jan-Gerrit Habekost et al.  
**年份/身份：** 2023；ICANN 2023:457–470; DOI 10.1007/978-3-031-44207-0_38  
**关系：** 邻近：神经逆解  
**核对层次：** 出版方方法摘要

已有内容：神经预测与几何一致性/数值处理。

对本项目的意义：无需为赶热点把学习模块加回当前数值方法。

来源：https://link.springer.com/chapter/10.1007/978-3-031-44207-0_38

### [20] Inverse Kinematics for Neuro-Robotic Grasping with Humanoid Embodied Agents

**作者：** Jan-Gerrit Habekost et al.  
**年份/身份：** 2024；IROS 2024；作者预印本 arXiv:2404.08825  
**关系：** 邻近：具身抓取应用  
**核对层次：** 作者摘要及机构会议记录

已有内容：把神经IK用于类人机器人抓取。

对本项目的意义：使用场景与训练依赖不同；仅作为近年领域背景。

来源：https://arxiv.org/abs/2404.08825

### [21] cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation

**作者：** Balakumar Sundaralingam et al.  
**年份/身份：** 2023；本次使用2023作者预印本 arXiv:2310.17274  
**关系：** 邻近：并行运动生成  
**核对层次：** 作者摘要/官方项目说明

已有内容：GPU并行的IK、优化与碰撞约束运动生成。

对本项目的意义：不能把其GPU/整路径时间和本项目单核每查询直接排快慢。

来源：https://arxiv.org/abs/2310.17274

### [22] cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots

**作者：** Balakumar Sundaralingam; Adithyavairavan Murali; Stan Birchfield  
**年份/身份：** 2026；技术报告 arXiv:2603.05493v2（2026-04更新）  
**关系：** 邻近：动力学与感知规划  
**核对层次：** 作者摘要/版本页

已有内容：面向更丰富动力学/感知约束的运动生成。

对本项目的意义：当前研究不承担其问题，相关工作简述即可。

来源：https://arxiv.org/abs/2603.05493

### [23] JPSP-IK: A Fast Reduced-Space Inverse Kinematics Framework for Industrial Redundant Manipulators

**作者：** Tianle Yang; Yuanlin Yi; Haolong Chen; Zhijie Li; Qin Zhou  
**年份/身份：** 2026；Machines 14(8):866; DOI 10.3390/machines14080866  
**关系：** 目标期刊实例：结构化数值增量  
**核对层次：** 出版方方法/实验全文可检索内容

已有内容：用关节参数化解析重建降低维数，再用驻点求解安排冗余变量。

对本项目的意义：说明应用型数值创新可发表；其接受不能担保Task Balance录用。

来源：https://www.mdpi.com/2075-1702/14/8/866

### [24] Control of Redundant Robots Under Hard Joint Constraints: Saturation in the Null Space

**作者：** Fabrizio Flacco; Alessandro De Luca; Oussama Khatib  
**年份/身份：** 2015；IEEE Transactions on Robotics 31:637–654; DOI 10.1109/TRO.2015.2418582  
**关系：** 基础：SNS  
**核对层次：** 作者机构出版记录与方法内容

已有内容：饱和关节处理、其他关节重新分配、任务缩放。

对本项目的意义：边界后重新分配运动不是本项目新原理。

来源：https://www.diag.uniroma1.it/en/publication/18357

### [25] Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results

**作者：** Moe; Antonelli; Teel; Pettersen; Schrimpf  
**年份/身份：** 2016；Frontiers in Robotics and AI 3:16; DOI 10.3389/frobt.2016.00016  
**关系：** 基础：集合任务  
**核对层次：** 全文方法/定义/假设

已有内容：任务允许集、分层控制及集合激活。

对本项目的意义：范围允许域观念已有，当前不具备其全部控制系统保证。

来源：https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00016/full

### [26] RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion

**作者：** Daniel Rakita; Bilge Mutlu; Michael Gleicher  
**年份/身份：** 2018；RSS 2018; DOI 10.15607/RSS.2018.XIV.043  
**关系：** 基础：优化运动生成  
**核对层次：** 会议作者摘要/文献回溯

已有内容：位姿、平滑性、碰撞与奇异性等目标协同。

对本项目的意义：不能把现代IK简单描述成纯精确求根或无约束裁剪。

来源：https://www.roboticsproceedings.org/rss14/p43.html

### [27] TRAC-IK: An Open-Source Library for Improved Solving of Generic Inverse Kinematics

**作者：** Patrick Beeson; Barrett Ames  
**年份/身份：** 2015；Humanoids 2015:928–935; DOI 10.1109/HUMANOIDS.2015.7363472  
**关系：** 基础：成熟外部IK  
**核对层次：** 出版信息/官方接口；原PDF访问受限

已有内容：Jacobian/SQP求解与原生Cartesian bounds。

对本项目的意义：任务球与内部容差框差异须披露，不把内部集合差异当算法胜利。

来源：https://ieeexplore.ieee.org/document/7363472
全文或第二原始来源：https://docs.ros.org/en/latest-available/api/trac_ik_lib/html/classTRAC__IK_1_1TRAC__IK.html

### [28] On the convergence of an inexact Gauss–Newton trust-region method for nonlinear least-squares problems with simple bounds

**作者：** Margherita Porcelli  
**年份/身份：** 2013；Optimization Letters 7:447–465; DOI 10.1007/s11590-011-0430-z；online 2011  
**关系：** 基础：有界非精确GN  
**核对层次：** 出版摘要/方法条件核查

已有内容：简单边界下的非精确GN与信赖域收敛条件。

对本项目的意义：按进展控制内层精度已有理论；不能将该收敛保证直接贴到当前有限预算循环。

来源：https://doi.org/10.1007/s11590-011-0430-z

### [29] Hierarchical quadratic programming: Fast online humanoid-robot motion generation

**作者：** Adrien Escande; Nicolas Mansard; Pierre-Brice Wieber  
**年份/身份：** 2014；International Journal of Robotics Research 33(7); DOI 10.1177/0278364914521306  
**关系：** 基础：任务优先级QP  
**核对层次：** 出版记录/摘要

已有内容：分层等式/不等式任务的在线处理。

对本项目的意义：当前同优先级minimax不同，但并非首次联合处理任务约束。

来源：https://journals.sagepub.com/doi/10.1177/0278364914521306

### [30] Convex Optimization

**作者：** Stephen Boyd; Lieven Vandenberghe  
**年份/身份：** 2004；Cambridge University Press；教材/基础来源  
**关系：** 基础：数学工具  
**核对层次：** 相关凸对偶章节

已有内容：最大值上图、凸强对偶、乘子及二次问题。

对本项目的意义：两个上图乘子和为1及80%代数推论不独立算创新。

来源：https://web.stanford.edu/~boyd/cvxbook/
全文或第二原始来源：https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

## 6. 期刊判断的依据与限制

MDPI Article类型要求科学可靠的实验及足够的新信息，不存在“三区所以成熟组件拼接一定可发”的通用门槛。来源：https://www.mdpi.com/about/article_types 。

Machines 2026 JPSP-IK可作为论文组织实例：具体降维结构＋求解更新＋同结构基线的证据。但一篇已发表论文不是本稿录用担保。若学校要求“中科院三区”，应另外核对学校采用的年度版本，不能用JCR Q2/Q3代替。本文不报告未经核验的中科院分区。

结论：当前不是工作量不够，而是原创来源标注、最近对手和任务适用价值需要准确落地。不要为增加工作量再生成更多同分布轨迹；也不要把已有公式重新命名作为新增创新。
