# Task Balance GN：最后一轮证据补强

本轮完成A（DROID公开Panda离线共同输入回放）和B（五个固定配置的开发敏感性），不修改算法或论文。基线为`79d3d9324647a51876dca2b2ec612d94599b6559`；输入、协议与代码在结果出现前提交为`1345598c9`。主方法始终为κ=1、forcing=.25。

**结论：真实来源中确有少量额外合格命令，但不是总体效率提升。** 7301个DROID请求中，主方法相对原GN增加15个、无损失，平均调用时间却为4.123 versus 1.800 ms；同外层Clarabel和直接SQP分别比主方法多1个、4个。902个来源命令预先确认可行的请求，五种方法均全部通过。五配置敏感性在两个机器人上均270/270，但只说明这组已观察开发输入上的有限稳定性，不证明`.25`最优。

方法归属与数学假设见[FINAL_SCOPE](TASK_BALANCE_FINAL_SCOPE.md)，结果驱动提纲见[PAPER_PLAN](TASK_BALANCE_PAPER_PLAN.md)。以下数字来自[可复算报告目录](../outputs/task_balance_final_evidence/reports/)，不是纸面预期。

## A. 来源、请求语义与输入身份

使用[官方DROID数据说明](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/docs/the-droid-dataset.md)所列的公开`droid_100/1.0.0`调试子集。31个TFRecord分片只用于100个episode身份索引，不解码图像；按原始路径SHA256升序，读取同一episode对应的低维HDF5，取得6个接口核验episode和24个评价episode。评价集为22个lab/date记录日簇，和接口集无记录日重叠。它不是整个DROID的概率样本。

100个候选中：6个接口、24个评价、6个raw对象访问不可用、1个与接口记录日重合、63个按顺序未取到。访问失败及相同episode的1.0.1/1.0.0对象尝试完整保存在[selection.json](../outputs/task_balance_final_evidence/source_replay/selection.json)。没有依据IK结果、边界距离或动作幅度筛选。最终数据访问完成，没有换数据源；原始数据自身success/failure标签未用于排除。

| 查询量 | 实际字段与核对结果 |
|---|---|
| 请求目标 | `action/cartesian_position`：官方`create_action_dict`产生的请求；不是观测FK，也不是另一个高层`target_cartesian_position`字段 |
| 配对上一状态 | `action/robot_state/joint_positions`，即生成此action时读取的实际状态；保留较早的observation作时序核对，不拿它替换 |
| 位姿表达 | 米、弧度、extrinsic `xyz` Euler；按官方等价转换得到旋转矩阵，不当旋转向量用 |
| 工具与基座 | `panda_link0 → panda_link8`，项目映射为恒等，无拟合；官方Polymetis配置与记录FK共同核对 |
| 来源关节命令 | `action/joint_position`仅作独立witness检查，不作求解seed |
| 运动周期 | **1/15 s，配置重建**；官方环境与IK控制配置均为15 Hz。实际记录间隔约69–77 ms含主机开销，不据此扩大位移额度 |
| 计算期限 | 仍为20 ms，与66.7 ms运动周期分开；不是硬实时或DROID任务成功定义 |

官方源码固定在`33ae6a67274f36d2e29525b86f23a56616ef43a7`，Polymetis指针为`0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2`。**准确的历史采集Git checkout没有保存在日志中**，因此这里没有声称证明了该checkout与当前官方源码完全相同。30个HDF5记录的版本标签为1.3×21、1.1×7、1.2×1、unknown×1；协议中提到的1.1是这种标签的例子，不代表所有记录或Git版本。现有配置、时间戳与软件FK是一致性证据，来源周期的“配置重建”身份须随论文保留。

接口6个episode共1348帧；评价24个episode共7301帧。全部30个episode的配对状态FK最大差为4.20×10⁻⁸ m、8.24×10⁻⁸ rad，小于事前固定的1e−5 m/rad接口阈值。没有非单调时间或超过两个配置周期的间隔。588条来源skip记录及原始终止记录仍保留，不因容易或困难删除。模型核对结果在[interface_audit.json](../outputs/task_balance_final_evidence/source_replay/interface_audit.json)。

仅做只读文件操作，从未导入或启动DROID硬件环境、机器人服务器或控制器。软件验收使用项目原URDFKinematics/verifier；解析运动学沿用现有Pinocchio接口，数值求解仍为原NumPy路径。

### 一个必须单列的来源—研究合同不相容项

`AUTOLab/success/2023-11-30/Thu_Nov_30_14:10:48_2023/trajectory.h5`（episode UID `4cdb89faa14ccdba127d52c7c586226cbd80c2f17ac1f36a3e1c7754173d7d83`）有215个记录状态超出项目冻结关节范围；第6关节最高4.124761 rad，项目上界3.7525 rad。其中119个请求的原关节范围与1/15 s速度箱交集为空。

没有改角度表示、放宽边界、替换episode或将其算作算法漏解收益。保留全部输入，同时单列7086个记录状态处于项目关节范围内的请求，以及902个来源命令通过同一合同的可行子集。这里不能从一个数值差异推断硬件型号、奇异性或唯一实现原因；原研究合同与部分真实日志的关节范围不相容本身就是外推限制。

初次回放在495次已完成调用后因空动态箱抛出异常。保留[原部分记录](../outputs/task_balance_final_evidence/source_replay/run/)，仅在运行器中把两种明确的空区间异常记为未返回命令，没有更改任何solver。修订提交`bcb9d0196`后对同一固定顺序完整重跑A，B不重跑。完整性能表仅使用[run_complete](../outputs/task_balance_final_evidence/source_replay/run_complete/)，异常及重跑范围见[runner_amendment.json](../outputs/task_balance_final_evidence/source_replay/runner_amendment.json)。没有将部分记录拼入性能表。

## A的结果：触发频度、覆盖与代价

每个请求均从其日志q独立求解，方法输出不反馈到下一个请求；不报告TSR或由不同查询解差分得到的加速度。7301个请求中654个已经同时满足研究的位姿条件；位置、姿态分别超差6644、6584个。归一化初始位置误差中位/P95为22.59/63.65，姿态为5.50/17.02。来源关节命令的最大逐关节位移利用率中位/P95为.432/1.182。它们描述真实请求相对**本研究1 mm/.5°合同**，不是DROID原任务的要求或现场失效率。

主方法在全部21,903次调用中：初始无需QP而接受8.96%，第一轮θ=.5直接接受36.83%，继续首步后的局部处理52.58%；真正至少一次用了超过一个局部加权QP的调用为25.82%，触发相对停止的调用52.58%。三者不是互斥分类，不能相加：相对停止也能在第一QP后发生。每episode触发比例、输入误差和边界余量见`source_episodes.csv`、`source_inputs.csv`。

下表分母都是全部7301请求；计时保留失败，三遍成功数相同。延迟分位数描述全部调用，推断单位不是调用。

| 方法 | 合格命令/7301（每遍） | 合格率 | 20 ms内合格/7301 | 平均ms | P50/P95/P99 ms | 每遍累计s | >20 ms调用/21903 |
|---|---:|---:|---:|---:|---|---:|---:|
| 主方法η=.25 | 5289 | 72.442% | 5289 | 4.123 | .410 / 20.159 / 20.212 | 30.099 | 1589 |
| 同κ原GN | 5274 | 72.237% | 5274 | 1.800 | .403 / 7.526 / 8.099 | 13.141 | 0 |
| fixed_qp2 | 5285 | 72.387% | 5285 | 3.087 | .515 / 12.317 / 13.571 | 22.536 | 5 |
| 同外层Clarabel | 5290 | 72.456% | 5290 | 3.750 | .605 / 14.789 / 17.675 | 27.377 | 13 |
| 直接同合同SQP | 5293 | 72.497% | 5293 | 6.279 | 2.865 / 20.165 / 20.186 | 45.841 | 3210 |

所有晚到调用均未取得合格命令，不因此把失败时间排除。未找到命令的请求不自动等于数学不可行。

统计先在query内平均三遍，再取24个完整episode的等权均值；以22个lab/date簇作4000次配对bootstrap，保留簇内所有episode。下面95%区间为未作多重比较调整的描述性区间；与上表按query汇总的比例不同。

| 主方法相对 | 获益/损失query UID | 等episode覆盖差pp [95%区间] | 等episode平均时延比 [95%区间] |
|---|---:|---|---|
| 原GN | 15 / 0 | +.219 [.102, .348] | 2.269 [2.014, 2.531] |
| fixed_qp2 | 4 / 0 | +.057 [.004, .124] | 1.336 [1.211, 1.477] |
| Clarabel | 0 / 1 | −.0039 [−.0122, 0] | 1.109 [1.021, 1.211] |
| SQP | 0 / 4 | −.0156 [−.0489, 0] | .566 [.456, .662] |

GN增量分布在9个记录日簇，不是一个时刻重复调用算15个独立episode。主方法相对Clarabel/SQP的损失集中于上述AUTOLab记录，但**损失输入本身仍有合法命令**；不能用该episode另一些输入的空箱解释这些损失。全部增减UID与实际关节向量、准确输入、原始行号在[source_changed_commands.json](../outputs/task_balance_final_evidence/reports/source_changed_commands.json)。

两条具体的同输入记录示例（第一遍，另外两遍同向）：

| query UID前缀 / 原帧 | 比较 | 主方法位置mm / 姿态° | 对照位置mm / 姿态° | 解释 |
|---|---|---|---|---|
| `e91159391fd738a3` / 218 | 原GN | .992417 / .495680 | .224025 / .570993 | 主方法允许位置误差增加，获得两项共同验收；保存真实命令即证明此输入有可接受解 |
| `3e55fe778eb9f57a` / 904 | Clarabel | .993191 / .512442 | .986379 / .499863 | 同目标通用局部求解找到主方法未找到的合格命令，保留数值完成度不足的反例 |

**预先来源witness子集**：902/902，五方法三遍均全通过、均20 ms内，主方法没有恢复增量；平均时间主方法/GN/fixed2/Clarabel/SQP为.207/.225/.206/.206/.842 ms。主方法在其中没有触发额外加权QP，不能将这个子集的细小接口成本差写成新平衡机制收益。其余输入没有来源witness不等于不可行；上面的新恢复命令是另外保存的事后可行见证。

**记录状态处于项目关节范围内的子集**：主方法5273/7086，GN5258，fixed2 5269，Clarabel5274，SQP5277；平均时间依次4.044、1.786、3.059、3.706、6.271 ms。方向与全样本一致。119个空箱请求所有方法均无合格命令，独立列出，不作方法贡献。

合格命令的误差并未隐藏：主方法位置P95 .878 mm、姿态P95 .187°，GN .867 mm/.183°，SQP接近1 mm/.5°；最大归一化误差>.9的比例为4.18%、3.92%、87.66%。主方法最大关节步长利用率P95为.987，GN为.983；实际分布、每episode差异和其他对照完整保存在表中。

**A回答：额外任务平衡在该来源中会发生，也确实产生15个原GN漏掉的合格命令；但收益小、成本明显增加，同目标Clarabel还以较低平均成本多找到一个命令。真实来源支持有限的命令覆盖作用，不支持总体速度或闭环任务优势。**

## B. 五配置敏感性，不作参数选择

每机器人保留全部270个已观察开发查询、30个anchor及九个位移×偏移格。三种η使用同一原目标；仅位置/姿态收紧时，按任务包公式相对原witness FK缩放对应目标偏移，保持归一化α、previous_q与运动边界，逐个验证原witness。后两项是同anchor的配对重构，不是同一绝对目标突然要求更精确。

共17,820次调用，三遍成功状态相同，均无>20 ms调用。每个配置主方法与SQP均270/270。下表列出五个主实例，完整所有11方法—合同组合及P50/P95/P99在[sensitivity_main.csv](../outputs/task_balance_final_evidence/reports/sensitivity_main.csv)。

| 机器人 | 配置（位置mm/姿态°） | 主/原GN/SQP成功数 | 主方法平均ms | GN / SQP平均ms | 主方法平均局部QP | 主方法贴边>.9 |
|---|---|---|---:|---|---:|---:|
| Panda | η=.10，1/.5 | 270 / 268 / 270 | .3964 | .4292 / 2.0647 | 1.430 | 8.15% |
| Panda | **η=.25，1/.5** | 270 / 268 / 270 | .3964 | .4292 / 2.0647 | 1.430 | 8.15% |
| Panda | η=.50，1/.5 | 270 / 268 / 270 | .3954 | .4292 / 2.0647 | 1.430 | 8.15% |
| Panda | η=.25，.5/.5 | 270 / 269 / 270 | .4382 | .4414 / 2.3331 | 1.659 | 4.81% |
| Panda | η=.25，1/.25 | 270 / 269 / 270 | .3956 | .4190 / 2.2152 | 1.430 | 5.93% |
| UR5e | η=.10，1/.5 | 270 / 267 / 270 | .3992 | .4290 / 2.1173 | 1.600 | 5.93% |
| UR5e | **η=.25，1/.5** | 270 / 267 / 270 | .3981 | .4290 / 2.1173 | 1.604 | 5.93% |
| UR5e | η=.50，1/.5 | 270 / 267 / 270 | .3965 | .4290 / 2.1173 | 1.604 | 5.93% |
| UR5e | η=.25，.5/.5 | 270 / 269 / 270 | .4145 | .4315 / 2.4022 | 1.659 | 6.30% |
| UR5e | η=.25，1/.25 | 270 / 265 / 270 | .4016 | .4580 / 2.2860 | 1.630 | 7.78% |

同一nominal条件的GN/SQP只各运行一次三遍，在前三行复用，不伪装额外样本。主方法相对GN在nominal/位置半/姿态半条件分别恢复Panda 2/1/1个、UR5e 3/1/5个，无损失。所有条件SQP也全部通过，这不是其他合理优化器不能取得的覆盖。

每个九格及三个几何子类的覆盖、误差、时间和增减UID均在`sensitivity_cells.csv`、`sensitivity_paired.csv`和`sensitivity_changed_commands.json`中，包括普通格，不以压力格替代全样本。统计在query内平均重复，以30个anchor按三类几何分层bootstrap4000次。nominal主/GN时间比Panda .924 [.854,.980]、UR5e .928 [.864,.985]；主相对η=.10/.50时间比的区间均跨1。全通过形成零宽成功率差区间只是这批样本的bootstrap退化，不是等效检验。

也保留尾部与贴边的非单调结果：UR5e位置半条件主方法P99 .676 ms，高于GN的.616 ms；Panda nominal主方法P95 .581 ms，高于GN .574 ms。主方法并不在每个分位数都更快。SQP在这些构造输入上100%的合格命令至少一项误差>.9；主方法为4.81–8.15%，但较小误差不等同于更好的下一帧跟踪。

**B回答：五个预声明条件下覆盖保持，η三点没有改变成功集合；收紧某一任务后的收益与成本随机器人和合同变化。保留`.25`，没有选新值，也没有从已观察开发集推断普遍鲁棒性。**

## 与原冻结结果合起来，应写什么

| 证据身份 | 现有发现 | 方法主张的边界 |
|---|---|---|
| 已观察2160/机器人最近方法共同输入比较 | 主方法/紧解/Clarabel/SQP均2160；GN2150/2148，fixed2 2157/2156；同批主方法均值.396/.387 ms | 支持少数受限请求的覆盖与该负载中的数值工作控制；全样本增量仅.463/.556 pp，不能冒充全域显著突破 |
| 预定义压力格 | multijoint-boundary、α=.95格中GN差10/240、11/240；UR另一个格差1 | 压力格有解释作用，不能把其比率当总体收益；原全部九格继续保留 |
| 旧普通点和独立完整轨迹 | 原GN已经2000/2000；主/GN完整UID集合相同，Panda78/80、UR5e80/80 | 无新增完成；Panda累计成本比1.135，区间[.908,1.404]，保留不利方向和时限波动，不据区间跨1称等效 |
| 旧合成扫描 | 每机器人12条全部方法完成，未触发额外平衡 | 不是实机；不能作为额外任务收益。SQP较低加速度的事实不删除 |
| 本轮真实来源 | 原GN基础上增加15/7301，但更贵；Clarabel/SQP增加更多；902来源witness无增量 | 限定离线命令能力与成本外推边界，不构成实机任务/轨迹优势 |
| 本轮有限敏感性 | 两机器人五配置均270/270，η无覆盖选择理由 | 有限开发条件稳定，不是主参数优化或独立发生率估计 |

旧同批数据源是[TASK_BALANCE_REVIEW_PACKET](TASK_BALANCE_REVIEW_PACKET.md)，旧独立与数学来源在[方法归属文件](TASK_BALANCE_FINAL_SCOPE.md)中链接。不把不同时段时延混成一张排行榜，不借TRAC内部容差集合差异制造本方法的核心优势，不将范围损失移植视为完整RangedIK。

## 复算、计时与完整性

- [唯一运行入口](../scripts/run_task_balance_final_evidence.py)：`prepare`、`sensitivity`、`fetch_index`、`fetch_raw`、`prepare_replay`、`replay`阶段分别保存排程、输入、记录及manifest；本轮已执行完毕，不要求再运行IK。
- [只读表图脚本](../scripts/report_task_balance_final_evidence.py)：读取已保存命令，原verifier复核并复算统计；**不调用IK**。使用`--out /tmp/task_balance_final_recomputed`写新位置，不覆盖冻结报告。
- 环境`/home/eric/anaconda3/envs/isaaclab_3/bin/python`；`PYTHONPATH=tmp/crik_dependencies/python:src`，`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。性能运行CPU affinity=4，未升级包、未用新加速后端；运行中交错方法，预热同协议。
- 外层时间包括输入转换、实际数值调用及最终验证；不含记录序列化、冷启动/预热、下载及离线复核。源接口异常也在实际计时内处理。不是实机周期耗时。
- A完整109,515条、B17,820条，合计127,335条输出均重新核对验收和20 ms标记，**已接受命令违约0**；所有失利与4817次A晚到调用保留。源码、原verifier、旧配置及论文hash前后一致。
- 原始命令：`source_replay/run_complete/records.jsonl.gz`、`sensitivity/run_{panda,ur5e}/records.jsonl.gz`；实际输入与来源HDF5均同根保存。更改UID表为`source_changes.csv`、`sensitivity_changes.csv`，失败/超时索引为对应`*_failures_timeouts.csv`。来源JSON含目标、配对q、dt、source object hash和frame，输出含实际返回q，不用虚拟命令代替。
- 报告目录`manifest.json`记录输入manifest、报告脚本、表图hash及冻结文件hash。原始部分A运行与其完整重跑身份单独保留。

复算示例（只读既有实验）：

```bash
env MPLCONFIGDIR=/tmp/task_balance_final_mpl PYTHONPATH=tmp/crik_dependencies/python:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/report_task_balance_final_evidence.py --out /tmp/task_balance_final_recomputed
```

### 图说明与检查

[真实回放图](../outputs/task_balance_final_evidence/reports/source_replay.pdf)：a为主方法减各对照的等episode合格率差，b为等episode平均时延比；24episode/22记录日簇，95%配对cluster bootstrap，完整查询含失败，无新增采样。图中包含对主方法不利的Clarabel/SQP覆盖与GN/fixed2/Clarabel成本方向。

[敏感性图](../outputs/task_balance_final_evidence/reports/sensitivity.pdf)：a/b为两机器人五条件覆盖（各270，三遍状态相同），c/d为对应主方法/GN均值时间比及按anchor分层配对区间。前三列GN/SQP相同参照不算三个独立对照；连线仅连接分类条件，不表示连续插值。覆盖轴局部放大，所有分母完整标注。

两图均183 mm宽、可编辑SVG/PDF与600 dpi PNG；Python生成。已逐panel和整图检查，无文字遮挡；PDF最小字体7/8 pt。静态检查18通过、0失败：PNG非TIFF是报告预览格式而非期刊栅格提交，随机数只用于从真实观测bootstrap，并未生成合成结果。自动检查不替代来源与统计核对。

**停止点：A/B和方法归属、写作计划已完成；不因DROID成本劣势换源、调forcing或继续实验。本报告不判断原创性充分或期刊录用，等待用户与ChatGPT验收。**
