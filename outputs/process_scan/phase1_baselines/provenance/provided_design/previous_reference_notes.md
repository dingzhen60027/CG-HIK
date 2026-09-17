# 相关工作与本协议采用的具体依据

检索日期：2026-09-16。以下为原论文、作者页面及官方实现；本清单用于设计，不宣称穷尽整个领域。未核实作者代码时明确写“未确认”，不能凭二手网站的Code按钮认定开源。

## 直接相邻的方法

### R1. Pham & Pham. A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis. IEEE T-RO 34(3), 645–659, 2018.
- 原论文：https://arxiv.org/abs/1707.07239
- 官方文档：https://hungpham2511.github.io/toppra/index.html
- 官方代码：https://github.com/hungpham2511/toppra
- 用途：给定几何路径的强时间分配基线、速度/加速度约束、rest-to-rest条件。
- 不承担：优化原始关节几何路径；不因它只做时间分配就故意提供劣质路径。
- 实施注意：本次读取的develop README提示Python支持会调整。使用已通过本机例程的固定发布/提交；不要依赖未锁定latest，必要时用官方C++绑定，不升级旧IK环境。

### R2. Weingartshofer et al. Optimization-based path planning framework for industrial manufacturing processes with complex continuous paths. RCIM 82, 102516, 2023.
- DOI：https://doi.org/10.1016/j.rcim.2022.102516
- 作者项目：https://www.acin.tuwien.ac.at/en/4adf/
- 开放全文：https://repositum.tuwien.at/handle/20.500.12708/153895
- 已有：工艺容差、工艺自由度、机器人冗余和碰撞约束的联合路径规划，描画实机及喷涂仿真。
- 采用：先定义任务允许域；不把固定姿态当唯一合理基线；质量和路径可行性共同评价。
- 边界：不是声称本研究首次利用工艺容差或允许经过奇异区域。

### R3. Fried & Paternain. A Bi-Level Optimization Approach to Joint Trajectory Optimization for Redundant Manipulators. arXiv:2412.07859, 2024-12-10作者预印本。
- 原文：https://arxiv.org/abs/2412.07859
- PDF：https://arxiv.org/pdf/2412.07859
- 作者页：https://jonathanfried.faculty.bio/
- 已有：关节路径与路径速度双层优化，下层凸速度问题，上层利用值函数方向导数；讨论常速和变速，并报告UR10e实验。
- 采用：B2必须是同目标全变量强参照；构型收益与只重定时的收益分开；不把初值重复当多条独立工件；实际反馈误差单列。
- 关键新颖性约束：双层+下层时间敏感性不是我们的原创。作者代码在本次核查中未确认，不得承诺已开源或称本地匹配实现为官方复现。

### R4. Oelerich et al. BoundMPC: Cartesian path following with error bounds based on model predictive control in the joint space. IJRR, 2025.
- DOI：https://doi.org/10.1177/02783649241309354
- 官方代码：https://github.com/TU-Wien-ACIN-CDS/BoundMPC
- 作者视频：https://www.acin.tuwien.ac.at/42d0/
- 已有：路径进度与关节轨迹，位置/姿态误差范围，多个实体任务及重规划。
- 采用：B3系统级参照；区分路径跟随和固定时间轨迹；任务完成时间、规划时间、误差和成功率联合展示。
- 实施注意：README为ROS2 Humble/CasADi/Ipopt；公开例程未配置其实际实时实验所需HSL MA57。必须记实际后端与是否实时，不据开源依赖差异作因果归因。

### R5. Yoon, Baek & Park. Reactive Model Predictive Contouring Control for Robot Manipulators. arXiv:2508.09502, 2025作者预印本。
- 原文：https://arxiv.org/abs/2508.09502
- HTML：https://arxiv.org/html/2508.09502v1
- 已有：路径进度、Jacobian/GN局部化、碰撞/奇异性等处理，作者报告100Hz与实机实验。
- 采用：说明实时控制与离线规划是不同评估位置；不用理想计算时钟伪造实时。
- 不纳入首轮算法复现：本任务不研究动态障碍，已有BoundMPC作为唯一在线系统参照，避免又开一条CBF支线。

## 扫描任务与物理验证的依据

### R6. Roos-Hoefgeest et al. Reinforcement Learning Approach to Optimizing Profilometric Sensor Trajectories for Surface Inspection. Sensors 25(7), 2271, 2025.
- DOI：https://doi.org/10.3390/s25072271
- 作者预印本：https://arxiv.org/abs/2409.03429
- 作者页面：https://sararht.github.io/publication/preprint/
- 已有：CAD上的线轮廓扫描、工作距离/相对朝向/轮廓间距，多个形状仿真与UR3e执行。
- 采用：扫描合格不只是TCP走过；须保留采样几何及覆盖。线扫描旋转会影响采样排列，不能随意假设全滚转自由。
- 不纳入方法主基线：其核心是传感器路径自适应与RL，本研究固定覆盖路线、比较构型—时间数值规划；不为了引用它再训练PPO。
- 本协议150mm等值是明确虚拟研究工况，并未声称来自该论文的具体设备参数。

### R7. Ma & Xu. Dual-Objective Optimization of G3-Continuous Quintic B-Spline Trajectories for Robotic Ultrasonic Testing. Sensors 25(18), 5693, 2025.
- DOI：https://doi.org/10.3390/s25185693
- 用途：认识曲率、运动平顺性和表面误差需要同时评价。
- 边界：本研究不做超声耦合/接触，不把其样条连续性名称直接移植；C²路径不自动保证时间轨迹jerk有界。

### R8. An Optimal Control Approach to the Minimum-Time Trajectory Planning of Robotic Manipulators. Robotics 12(3),64, 2023.
- 原文：https://www.mdpi.com/2218-6581/12/3/64
- DOI：https://doi.org/10.3390/robotics12030064
- 用途：直接最优控制/全量非线性优化是成熟强对照类型，不应该只与逐点IK比较。
- 边界：其不预设几何路径的任务不同于本固定覆盖路线，不将其任务结果数值直接搬来做排行榜。

## 追踪到但不扩展为当前实验的方向

### R9. Fried & Paternain. A Bi-Level Optimization Method for Redundant Dual-Arm Minimum Time Problems. IEEE Control Systems Letters, 2025（作者页列出）。
- 来源：https://jonathanfried.faculty.bio/
- 相关：说明值函数/冗余轨迹最短时间研究持续发展。
- 不纳入首轮：用户任务单臂，不将双臂问题作为弱参照，不复述为本研究原创。

### R10. Chen, Fried & Paternain. Diffusion-Based Optimization for Accelerated Convergence of Redundant Dual-Arm Minimum Time Problems. arXiv:2604.16670, 2026作者预印本。
- 原文：https://arxiv.org/abs/2604.16670
- 相关：作者针对其前序双层方法的优化成本提出采样式扩展。
- 不纳入首轮：不因它是新论文就加入扩散、双臂或训练任务；只用于划清“已有工作也在研究计算效率”。

## 官方工具来源

- MuJoCo物理与驱动：https://mujoco.readthedocs.io/en/stable/computation/index.html
- Panda模型：https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
- UR5e模型：https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e （实施时必须核验实际目录/提交及与本项目轴系的一致性，不因路径可写就认定模型可用）
- CasADi：https://web.casadi.org/
- Ipopt：https://coin-or.github.io/Ipopt/

## 已核对深度

R1：官方论文摘要、文档和仓库README；R2：作者项目及开放全文文本；R3：摘要与PDF文本中的方法、常速/变速实验部分；R4：出版方全文与官方README；R5：作者预印本HTML；R6：出版方方法段/摘要及作者预印本；R7/R8：出版方页面；R9：作者页；R10：作者预印本摘要。

本次网络PDF截图读取失败，因此没有声称已逐图复核R2/R3的数值。实施引用精确图表数字前应读取原图。此清单不包含未阅读论文的虚构实验次数，也不承诺最小规划时间改进。
