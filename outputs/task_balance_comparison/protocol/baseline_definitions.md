# 固定比较协议与对照身份（结果产生前）

研究基线 `6a7336f64bd48e4e293f2c43a1ec0833e529ed04`，同分支。
本轮是已观察输入的补充共同输入比较，不是 fresh independent test。
主方法、原 GN、紧精度、Clarabel、公共 verifier、旧数据/代码文件由
`frozen_before.json` 及既有源文件 hash 保护。绝不以新对照替换主方法。

## 共同输入、合同与时间

每机器人原开发270查询仅用于接口检查；原验证2160查询全部保留，三遍交错。
1 mm位置范数球、0.5° SO(3)测地球、dt=.02 s、原URDF范围和速度容差。
令 S=diag(velocity*dt+velocity_tolerance)，动态箱为物理范围与上一实际状态
可达区间之交，采用原可表示内部边界。所有方法从各自输入 previous_q 开始；
点查询不反馈。只验收真实 FK 后通过原 verifier 的命令。
公共验收、solver原生终止、20 ms内验收分开记录。

CPU4，NumPy冻结路径，OMP/MKL/OpenBLAS各1线程，原环境版本不升级。
每实例加载后3次相同非静止预热；Clarabel按既有代码额外预热固定锥结构。
初始化/预热另记，不进入逐查询；完整调用外层 perf_counter_ns 包括输入转换、
求解、在线检查、最终 verifier。序列化与独立离线复验不计入。
保留超时及其实际时间，软期限不等于抢占式硬实时保证。

## 八个方法

| 名称 | 实际接口与目标 | 设置与停止 |
|---|---|---|
| relative | 冻结 CompletedTaskBalanceGN；局部 max(||ep+Gp d||²,||eR+GR d||²)+正则，标量对偶 | κ=1、forcing=.25、原30/16/8预算；任务合格或有效相对gap/紧gap；原真实下降与验收 |
| gn | 冻结 SingleBoundedGN κ=1，平方和方向 | 原缓存、动态箱、停止与20ms预算 |
| tight | 与relative同缓存/同外层，forcing=None | 紧gap=1e−9+1e−7尺度；其他完全同relative |
| clarabel | 冻结ClarabelBalance同minimax外层、同首QP机会，Clarabel epigraph | 既有数值精度和20ms预算，不调参 |
| fixed_qp1/2 | 实例级回调，仅替换局部对偶方向函数，外层函数bytecode完全相同 | 每局部1/2加权QP；θ=.5开始；第二步沿用保障Newton/括区；最好原始目标方向继续外层；不计算forcing界 |
| direct_sqp | SciPy SLSQP：min .5||y||²，1−||ep||²≥0、1−||eR||²≥0，y=S⁻¹(q−previous)动态箱 | 单启动、解析目标/约束Jacobian、同点缓存，maxiter100 ftol1e−8；首个公共验收命令立即返回（非最优声明）；20ms软期限 |
| range_loss_matched | SciPy L-BFGS-B；将官方Swamp Groove标量函数作用于两个归一化误差块范数，等权相加，动态箱直接入求解 | 单启动、解析导数、maxiter100 ftol1e−12 gtol1e−8 maxls20；首个合格命令早停，20ms软期限 |

后两项的内部成功仅指优化器正常返回success；`task_accepted_early`是提前找到
任务命令，不伪称位移/范围损失已最优。内部失败的最好实际候选仍送公共验收。
范围对照与SQP均不用minimax、forcing、witness、别的方法seed或额外IK。
本轮只采用一套预声明配置，不扫描或按机器人选参数。

### 开发阶段数值修订（主比较前）

未加裕量SLSQP在Panda/UR5e开发集仅0/1个query等效接受，绝大多数原生成功
返回在范数球边界外微小距离，不能当作可信的SQP零覆盖结论。
旧记录及源码快照保留（development_v1_source.py）；只复跑新SQP开发对照。
第二且最终设置预声明：约束内部使用1−||e_b||²≥1e−7，ftol仍1e−8。
该数值裕量对应半径缩小约5e−8的相对量（位置约5e−11m、姿态约4.36e−10rad），
比公共合同略严格，绝不放宽最终验收，不改变主方法。它处理通用优化器可行性
停止误差；不是新任务合同、参数网格或额外IK。原版失败记录不删除。
只根据接口数值问题作此一次修正，之后无论结果好坏固定，不再搜索第三设置。

## RangedIK移植差异（不是完整RangedIK）

官方仓库 `uwgraphics/relaxed_ik_core` ranged-ik，源码
`1c48d2ae408b4e024ee037641aac1e728267984e`。先跑未改动的官方
`relaxed_ik_bin` Release示例，原Sawyer仅为依赖smoke，不是第三机器人研究。
完整命令、编译告警、输出和版本在 `references/official_demo.json`。

官方位置误差在目标坐标系，旋转为目标相对实际的scaled-axis分量（代码取绝对值）；
原生范围是分量区间，不是当前两范数球。官方源码bound≤.01走Groove而非Swamp
Groove，这包含本任务的微小分量容差。不能把该接口差异当成主方法的优势。
原生PANOC采用关节矩形约束，多任务还含速度/加速度/jerk、操纵性与碰撞软项。
既有原生适配的动态箱、链顺序和FK一致性核对保留于 `docs/CRIK_RANGED_ADAPTER.md`；
不把其旧失败计入本轮成功表。

同合同移植将官方目标分量改为 r=||ep|| 或 ||eR||，范围对称[-1,1]，优选0：

    l(r)=-exp(-r²/8)+.01r²+100[1-exp(-(r/b)^20)],
    b=(-1/log(.05))^(1/20).

参数来自官方 `swamp_groove_loss` 调用（g0,c2,f1=1,f2=.01,f3=100,p20），
保留范围内向0的偏好；不是纯超差零平台。范数球与硬速度箱匹配公共合同。
使用现有解析SO(3)残差导数与L-BFGS-B替代PANOC，两个无量纲块等权，
不保留官方50/10分量权重、历史平滑/碰撞/操纵性任务。因此名称只能是
`range_loss_matched`，其输赢不能代表完整RangedIK输赢。点查询历史重置为实际previous，
该移植没有另外的历史状态。无额外投影或求解器兜底。

## 应用定义（结果前固定）

仓库文件检索没有找到独立采集的Cartesian任务/规划器日志；既有FK轨迹不改名应用。
采用应用型合成“非接触平面观测扫描”，没有实际相机、工件接触或现场效益证据。
每机器人12条、150帧；初始关节配置写入配置，FK仅定义一次工位原点和工具坐标系。
后续目标完全由Cartesian曲线生成：s=t/149，
p(s)=p0+A sin(4πs)u+(A/2)sin(2πs)v，姿态固定起始工具姿态。
A∈{10,20,30}mm，工具平面旋角∈{0,π/4,π/2,3π/4}；总时间2.98s。
这是一种Lissajous扫描路径，不是随机FK路径；最大Cartesian速度由公式决定，
不按算法失败推高关节利用率。所有24条输入在任何方法运行前保存并hash。
1mm/.5°是沿用的软件假设，不冒称工业公差。
六方法 relative/gn/range_loss_matched/direct_sqp/fixed_qp1/fixed_qp2，各三遍；
接受才反馈、失败保持、时间索引继续，所有150帧计入。无q_ref或未来target在线读取。
只证明共同起点合法，后续没有通用可行性保证；失败不称数学不可行或witness确认漏解。

## 统计与停点

点重复先在query UID内平均；anchor为聚类单位，几何类别分层4000次配对bootstrap，
报告覆盖差/时延比及描述性未校正95%区间。全2160、九格、27细分全保留。
22个旧GN漏解只是预定义全集内分解，不据此选样。应用以完整序列UID统计，重复嵌套。
零差/含零区间不证明等效。新增命令与当前输入保存，不从别的闭环状态推单因果。
测试/开发后冻结代码、输入及运行次序，先提交再运行主比较；结果不触发参数更换。
完成报告及push后停止，等待验收，不写论文或追加实验。
