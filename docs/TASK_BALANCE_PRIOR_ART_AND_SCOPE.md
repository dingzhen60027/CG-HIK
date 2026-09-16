# Task Balance GN：方法归属与本轮检验边界

本说明基于研究基线 `6a7336f64bd48e4e293f2c43a1ec0833e529ed04`。
旧方法说明和实验记录保留。这里更正贡献的表述归属，不改动已测方法，也不因
数学先例而抹去真实的合格命令恢复。结果判断见 [验收包](TASK_BALANCE_REVIEW_PACKET.md)。

## 1. 从任务合同出发，而不是从新颖性口号出发

输入为当前目标位姿、实际上一接受配置和采样间隔。机器人需要位置和姿态分别
达标，并满足关节范围及单帧位移限制；较低的加权总误差不自动满足这个合取条件。
令 e_p、e_R 分别除以公共位置和旋转容差，动态箱为 B_t。
对有限且属于 B_t 的 q，max(||e_p(q)||²,||e_R(q)||²)≤1恰对应两项位姿验收。
这是任务定义与最大值的直接对应关系，不是新的可行性定理。

当前方法在同一有界数值循环中，以 G=J_e S 形成局部模型：

    P(d)=max(||e_p+G_p d||²,||e_R+G_R d||²)
         + λ||d||²/2 + κ||c+w⊙d||²/2,  l≤d≤u.

κ=1、阻尼、动态范围、初始θ=.5、缓存与预算保持冻结。第一加权QP与同κ原GN
一致；若其真实命令通过任务验收就返回，否则仍在这个局部模型中调整权衡。
这不是GN失败后调用第二个IK，也不读取未来目标。后续方向必须通过真实非线性
下降检查，最终由原verifier决定接受。

## 2. 相对gap不是本方法独创

Zhong Zheng、Shiqian Ma、Lingzhou Xue 的 *A New Inexact Proximal Linear
Algorithm With Adaptive Stopping Criteria for Robust Phase Retrieval*，
IEEE Transactions on Signal Processing **72**, 1081–1093 (2024)，
[DOI](https://doi.org/10.1109/TSP.2024.3365933)，作者全文
[§2.3、§3、Eq.(23)](https://arxiv.org/html/2304.12522v2)已逐式核对；作者、题名、卷页与Crossref相符。

| 原文Eq.(23) | 本实现 |
|---|---|
| H_k(z)−D_k(λ) ≤ ρ_l[H_k(0)−H_k(z)] | U−L ≤ η[P(0)−U] |
| 当前原始模型值H、对偶下界D | 箱内方向的最好原始上界U、强凸加权QP下界L |
| 相对实际已取得的模型下降控制子问题误差 | 同型条件，冻结η=.25；另有任务合格早停 |

这是同型的相对原始—对偶间隙原则。原文同时说明LACC与更早非精确近端梯度
条件的联系，不能把历史起点一概归到2024年。该文求相位恢复的近端线性子问题，
用FISTA处理其多维对偶；本实现求两误差块的有界二次模型并使用标量对偶。
相同停止原理不等于相同完整算法，更不意味着其收敛定理可以移植到当前IK。

在有效L≤P*≤U且P(0)>U时，U−L≤η(P(0)−U)蕴含
P(0)−U≥[P(0)−P*]/(1+η)。η=.25给出80%的**局部正则模型下降**。
它是上述界关系的代数推论，不另列原创规则，也不是80%成功率、非线性收敛率
或轨迹保证。浮点实现若上下界不一致不能宣称该条件成立。

## 3. RangedIK已经把允许范围放入运动生成

Yeping Wang、Pragathi Praveena、Daniel Rakita、Michael Gleicher，
*RangedIK: An Optimization-based Robot Motion Generation Method for Ranged-Goal
Tasks*，ICRA 2023，9700–9706，[DOI](https://doi.org/10.1109/ICRA48891.2023.10161311)。
[作者全文](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/2023_ICRA_RangedIK.pdf)
Eq.(5)、(10)–(13)及
[官方ranged-ik源码](https://github.com/uwgraphics/relaxed_ik_core/tree/ranged-ik)已核对。
其任务包含范围内等价目标和带偏好的范围目标；Swamp和Swamp Groove分别服务这些语义。
通过多任务加权目标结合精度与运动需求，不能把“利用容差”作为这里的独有思想。

本轮使用已有官方commit `1c48d2ae408b4e024ee037641aac1e728267984e`，
独立Release目录编译并运行原例程，不升级环境、不改作者源码。
公式与实际接口需要分开：原生任务是目标坐标系位置、scaled-axis旋转的分量范围；
官方小范围分支使用Groove，且包含其他软任务。当前任务则是两个范数球与硬动态箱。
因此核心比较采用明确标为 `range_loss_matched` 的损失移植：将相同范围损失
作用于归一化块范数，保留范围内偏好，解析导数与独立L-BFGS-B求解。
完整配置和移植边界在 [baseline_definitions](../outputs/task_balance_comparison/protocol/baseline_definitions.md)。
其结果只能说明这个同合同范围损失参照，不代表完整RangedIK系统优劣。

Task Balance GN与该参照的实际区别，是以当前最差任务利用率决定两块的局部权衡，
而非累加固定形状的范围损失；两者都具有已有的任务容差基础。单独比较默认原生
分量盒与本项目球合同不能隔离这一建模差别，所以不承担核心归因。

## 4. 集合任务、对偶与非精确GN的基础归属

[Moe等，2016，Set-Based Tasks within the Singularity-Robust Multiple Task-Priority
Inverse Kinematics Framework](https://doi.org/10.3389/frobt.2016.00016)
给出了按允许集合组织机器人任务的基础。这里没有高低任务优先级的零空间切换；
位置和姿态以共同动态箱内的minimax模型协调。区别应落实到更新方程和对照，而
不是声称第一次让达标任务停止追求精度。

两块epigraph写成 min t+正则，约束f_p(d)≤t、f_R(d)≤t。对t的驻定性使两个
非负乘子和为1，因此用θ∈[0,1]表示；固定θ就是一个加权箱QP。
在连续凸模型、非空紧箱及适当相对内点/可行性条件下使用标准凸对偶；
λ>0确保各加权问题的唯一解。θ是局部对偶变量，不是按机器人手调的策略权重。
这一推导属于 [Boyd与Vandenberghe，Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
中的epigraph/对偶基础，活动集QP、缓存与解析几何Jacobian亦是成熟组件。

[Porcelli，Optimization Letters 7, 447–465 (2013)](https://doi.org/10.1007/s11590-011-0430-z)
研究简单边界下非精确Gauss–Newton信赖域。当前实现不是其信赖域算法复现，
不继承其全局结论。这里需要测量的是：当第一方向尚不能交付命令时，继续求局部
权衡是否有任务收益，以及相对精度停止是否比固定少量工作更有价值。

## 5. 本轮如何确认具体增量

| 待判断内容 | 成熟基础 | 本实现具体设计 | 直接检验 |
|---|---|---|---|
| 有限运动范围内处理不合格权衡 | 集合任务、minimax | 两个公共容差块、同κ首QP、动态箱联合更新 | 原GN、同合同范围损失、直接约束SQP |
| 控制局部额外计算 | 非精确优化、相对原始—对偶gap | 优先交付合格命令，否则按已有gap原则继续；真实下降独立检查 | 同缓存紧精度、同外层Clarabel、fixed_qp1/2 |
| 完整在线应用价值 | 当前帧反馈与公共验收 | 已冻结方法在同一几何任务下的实际命令序列 | 不筛路径的应用型合成扫描、TSR/DTSR20与成本 |

历史事实仍成立：普通点查询中原GN已2000/2000；旧完整轨迹没有新增完成；任务集
边界全样本新增10/12个恢复集中在预定义压力格。它们限定适用范围，不被本轮
改写成普遍收益。相对gap来源更正不会使这些真实命令失效；反之，真实恢复也
不能证明新优化理论或先例不存在。

本轮结果将区分三件事：任务建模是否有用，已有准则的任务化实现是否值得，
完整系统的覆盖—成本是否有实际增量。不以测试数量或文献数量代替判断。
