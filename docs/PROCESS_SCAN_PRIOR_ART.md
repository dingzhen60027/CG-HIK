# Process-scan Phase 1：方法归属与比较边界

本页是方法实现的来源说明，不是新算法或论文新颖性结论。所引用任务参数均为
本项目的虚拟研究设置，不是从某型号扫描仪规格推得。核查日期2026-09-16。

| 来源 | 已有方法／本轮采用内容 | 本轮实现的具体边界 |
|---|---|---|
| Pham & Pham, *A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis*, IEEE TRO 34(3), 645–659, 2018 ([作者稿](https://arxiv.org/abs/1707.07239)，[官方接口](https://hungpham2511.github.io/toppra/index.html)) | 固定几何路径上的可达集时间参数化；速度、加速度与rest-to-rest条件 | B0/B1/G及更新后的B2路径调用官方TOPPRA 0.6.3、seidel后端。额外加入实际计划中心射线速度约束。密集导数复核后共同时间膨胀，保留原始解；不将修正后的结果称为连续时间全局最优 |
| Weingartshofer, Bischof, Meiringer, Hartl-Nesic & Kugi, RCIM 82, 102516, 2023 ([开放记录与全文](https://repositum.tuwien.at/handle/20.500.12708/153895)，[DOI](https://doi.org/10.1016/j.rcim.2022.102516)) | 工艺自由度、允许偏差、机器人冗余及碰撞的联合几何规划 | B1/G是本地匹配的受约束样条平滑，不是作者完整软件复现。允许域由扫描距离、入射、中心射线和线方向共同定义，不只是标称姿态插值 |
| Fried & Paternain, *A Bi-Level Optimization Approach to Joint Trajectory Optimization for Redundant Manipulators*, arXiv:2412.07859, 2024 ([方法全文](https://arxiv.org/html/2412.07859v1)) | 路径参数与速度的联合优化；固定路径凸时间子问题、值函数方向导数及上层更新 | 本轮B2用全变量CasADi/Ipopt NLP，不实现作者的双层原算法，不声称复现其性能。所有64个样条控制点均可变；共同B1初值、同几何与运动约束、相同最终重定时。不存在本轮新设计的选块算法 |
| Oelerich et al., *BoundMPC*, IJRR, 2025 ([DOI](https://doi.org/10.1177/02783649241309354)，[官方源码](https://github.com/TU-Wien-ACIN-CDS/BoundMPC)) | 带位置／姿态误差边界的关节空间MPC和路径进度控制 | 在官方实验1的机器人、路径与权重下完成前三次无硬件数学更新。使用MUMPS，不是论文实时配置的MA57。Panda/UR5e扫描任务适配尚未验证，因此B3为not_evaluable，而不是算法失败或可比较的性能数字 |
| Roos-Hoefgeest et al., *Reinforcement Learning Approach to Optimizing Profilometric Sensor Trajectories for Surface Inspection*, Sensors 25(7), 2271, 2025 ([作者方法稿](https://arxiv.org/html/2409.03429v1)，[DOI](https://doi.org/10.3390/s25072271)) | 轮廓扫描的距离、入射朝向、轮廓间距与覆盖质量；扫描并非只让TCP经过路径 | 本轮不复现RL、不优化扫描顺序。固定CAD往复扫描，真实模拟反馈射线生成回波；没有激光反射、散斑噪声、实物标定或缺陷识别性能的主张 |

## 公共数值问题

几何路径为64控制点的三次C²样条。B1/G的共同平滑目标是归一化一阶及二阶
路径导数能量的共同网格离散平均，β=1；数值上乘共同的1e−4，不改变两项相对权重。
工作距离、光轴到指定中心的距离、入射角、线方向、物理关节范围、端点和
静态碰撞间隙作为约束。中间段及换行均密集复核，不接受“结点通过所以路径通过”。

B2使用 `x=s_dot²`、`u=s_ddot`，满足
`x[i+1]−x[i]=2 Δs[i] u[i]`，并用
`q_dot=q_s sqrt(x)`、`q_ddot=q_ss x+q_s u`约束运动。
目标是区间时间之和 `Σ 2Δs/(sqrt(x[i])+sqrt(x[i+1]))`。
这来自成熟时间参数化／联合优化，不以局部可操作度代替作业时间。
采样约束来自当前几何路径的中心射线—CAD交点速度，最后仍由200 Hz真实回波检查。

样条设计矩阵以其固有局部支撑存储结构零，只减少自动微分构建开销；所有控制点
仍参与优化。这与后续可能研究的稀疏变量选择不同，本轮没有实现后者。
Ipopt返回码不是验收；1/5/10/30 s仅记录当时已完成真实非线性复核的可行解。
额外计算、回调及晚结束时间全部保留。主路径取10 s检查点，实际记录的总计算
包含完整30 s检查点运行，因此不能宣称该实现总在10 s内返回。

## BoundMPC适配阻塞的精确范围

官方提交 `b286df18641145bb97b31f29387227e4089a1ba4` 已安装ROS消息依赖并运行。
`BoundMPC.py`固定`nr_joints=7, nr_x=44`，初始化与解码含`8:15`、`15:22`切片；
`RobotModel`内嵌IIWA几何及限制，积分函数也重新实例化该模型。
官方分段位置／姿态误差框与本项目射线距离—入射—线方向的耦合允许域并不相同。
仅替换URDF或把这些条件当独立姿态框，不能证明共同任务合同成立。
本轮没有用未验证的映射、零计算延迟或自写的另一MPC替代它。
因此该项只有官方接口记录，没有Panda/UR5e扫描排名；这是适配未完成的限制，
不是关于BoundMPC原方法能力或速度的负面结论。

## 本阶段能够回答什么

旧TB局部工具的作用只由B1—G分离；B0比较还包含初始化精度差异。
B2—B1才考察完整时间目标是否改善路径。质量合格率与共同合格场景的执行时间
必须并列，不用失败场景的短停止耗时充当快作业。
扫描下界是明确假设下的说明性界，不把TOPP-RA某个LP乘子当全程时间梯度。
没有本阶段数据能够证明新稀疏方法、全局最优、实机安全或论文创新性。
