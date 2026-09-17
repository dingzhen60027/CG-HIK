# Process-scan Phase 1：旧实现的实际复用

本阶段基于 `bf9ccc4e0314c0d289101d89ff4e40577de392f9`，仍在
`codex/hierarchical-v5`。旧算法、在线接口、verifier、论文与旧输出不改。
新代码位于 `src/confik/process_scan/`，入口为 `scripts/run_process_scan.py`。
这不是旧 IK 恢复链的继承或串联。

实际后端分工：局部残差/Jacobian使用已安装Pinocchio的`NativeGeometry`；
几何候选验收使用原`URDFKinematics`的NumPy FK；样条NLP使用由同一URDF
链构建并核对的CasADi FK；物理执行使用MuJoCo。这里按真实调用说明后端，
不从Conda环境名称推断运动学实现。

## 调用与语义

| 旧函数／资产 | 实际新调用点 | 保持的内容 | 明确改变／未继承的内容 |
|---|---|---|---|
| `bounded_gn.box_qp` | `GeometryAdapter.solve` | NumPy 箱 QP、阻尼初值、κ=1、第一加权方向 | 外层为无时间的几何求解；不套用旧20 ms动态箱 |
| `task_balance_gn.completion_dual_direction`、`CompletionSettings`、`task_value` | `geometry_adapter.py` 的 `method='tb'` | 双块 minimax、forcing=.25、局部验收／对偶继续 | 第一QP已经合格时不额外强制调用对偶；实际继续调用次数由原始 trace 和桥接表统计 |
| `correction_reserve.geometry.residual_linearization` | 几何 GN/TB 和共同参考构造 | 位置与 SO(3) 残差、正确姿态导数、显式尺度 | 不调用 CR-IK、Elastic、TAR 的运行时、预测器或备用链 |
| `NativeGeometry`、`URDFKinematics` | `models.context` | 原 URDF 链、关节命名、范围及解析几何 | 增加明确的80 mm法兰—光学中心刚性工具变换；原模型文件不改 |
| `geometry.pose_distance` | `GeometryAdapter.verify` | 公共位置范数、姿态距离定义 | 新建几何验收器检查有限性和几何箱；它不是关闭了速度检查的旧在线 verifier |
| `solvers/verifier.py` | 只读对照其语义和源文件 hash | 原在线 verifier 原样保留 | **没有调用**旧 `verify` 来接受无时间路径；真实速度／加速度在重定时和物理执行后独立验收 |
| `experiments.statistics.paired_cluster_bootstrap_difference` | `reporting.paired` | 成对聚类重采样实现 | cluster 改为机器人×曲面×放置，方向嵌套；三次计时先在场景内平均 |
| 已冻结在线、DROID、失败输入和独立测试 | 仅历史证据 | 全部保留 | 不作为扫描输入、物理完成或新方法有效性的证据 |

几何接口的参数固定为：`target_pose, seed_q, physical_bounds,
continuity_box, step_scale, pose_tolerances`。`seed_q` 是数值初值，不是一个
假定20 ms前执行的命令。连续性半径0.35 rad、更新尺度0.15 rad与后续物理
速度无关。几何外层最多30次、8次回溯、1 s/结点；这些不是实时预算。
TB用minimax非线性下降，GN用平方和下降；其余初始化、箱、尺度和预算共同。

TB/G的初始化位姿球为1 mm/0.5°；B0为50 μm/0.05°。初始化位姿球不等于
线扫描工艺允许域。后者另行检查工作距离、中心光轴、入射角和投影线方向，
并约束共同的C²样条，包括换行。Task Balance不求最近点，也不提供整路径保证。

## 新物理接口不是旧结果的自动外推

`models.py`从实际冻结URDF读取轴、限位、质量和惯量，转换为MuJoCo模型；
Panda使用原资产碰撞网格，UR5e使用固定官方`ur_description`网格。
检查过Menagerie，但**没有采用**其不同运动学模型。三个配置的旧FK、
CasADi FK与MuJoCo光学TCP核对，以及Jacobian有限差分测试，均在开发比较前完成。
工具质量0.15 kg、80 mm安装偏移、附加armature、阻尼及加速度上限是明示的仿真假设。

规划碰撞距离使用已安装的Coal 3.0.2及同一凸网格。工件使用保守包围体作
规划间隙检查；物理接触与射线使用实际CAD高度场。执行器使用力矩电机和
共同的惯量缩放前馈PD。只有reset写入执行状态的qpos；规划器自己的几何
查询数据和离线视频渲染数据不冒充执行状态。

`execution.h5`来自`mj_step`，`scan_samples.npz`来自该时刻真实TCP射线。
物理步1 ms，控制2 ms，轮廓5 ms；速度、加速度及接触极值按**每个物理步**
累计，较稀疏的记录仅用于可视化与RMS。实际跟踪仍可能超限，不能用规划合格覆盖。

## 复用是否带来增量

以 `reports/bridge_tb_vs_gn.csv` 为准，分别看初始化、平滑后路径、重定时和
真实扫描。若第一QP已足够、TB对偶未启动，或TB与GN给出相同路径，直接报告。
不把公共几何平滑、TOPP-RA、控制器的收益算成TB自身收益。

原始数值与输入hash在`inputs/seal.json`；时间网格浮点去重修正后的完整批次使用
`inputs/numerical_seal_grid_repair.json`。输入身份未变，首批部分结果另存、不混入主表。
每个run保存其实际源文件hash；修正范围见交接文档。
只读报告和渲染脚本可在求解期间完成，不进入数值方法的冻结范围。
