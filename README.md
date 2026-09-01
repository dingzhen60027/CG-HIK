# CG-HIK

CG-HIK 研究连续机械臂控制中的混合逆运动学。它不让神经网络直接替代数值求解器，而是把在线 IK 看成一个资源分配问题：学习模型选择求解入口或拒绝动作，数值级联生成关节解，确定性验证器决定命令能否接受。

## 项目结论

v2、v3 和 v4 的实验链已经完成。最终 `test_v4` 覆盖 Panda、UR5e 和三个训练 seed，共 744 个 checkpoint、650,000 条 method-query records。正式结果表明：

- v4 在两台机器人上保持了与 fixed robust cascade 相同的可行查询成功表现和轨迹完成率；
- feasible P95 latency 相对 fixed 降低约 24.6%（Panda）和 25.7%（UR5e）；
- 对已知不可行查询，零求解拒绝避免了约 95.7% 和 93.9% 的 FEV；
- OOD 点检测表现较弱，Panda 的冻结 OOD 改善门未通过；
- 因此 joint Holm gate 通过、UR5e gate 通过，但 Panda gate 和整体 paper gate 未通过。

这不是“实验失败后无论文可写”。论文最有价值的结论是：学习模型适合做经过验证的 IK 资源分配，但当前 OOD 判别不足以支撑广泛的安全拒绝主张。

## 从这里开始

项目只保留三个活动入口：

1. [研究主线与全部结果](docs/RESEARCH.md)：论文故事、方法演化、实验设计、正式结果和写作边界。
2. [运行与复核手册](docs/RUNBOOK.md)：环境、只读复核命令、证据位置和后续工作规则。
3. [论文包](paper/README.md)：稿件状态、重写任务和构建方式。

## 目录

| 路径 | 内容 |
|---|---|
| `src/confik/` | 数据、模型、solver、runtime 和版本化实验代码 |
| `configs/`, `scripts/`, `tests/` | 配置、运行入口和测试 |
| `outputs/` | v2 到 v4 的冻结实验结果；保持原路径以维持 manifest 与哈希链 |
| `czy/` | Panda 闭环补充实验原始记录；路径参与 test_v4 身份审计 |
| `paper/` | LaTeX、PDF、图、表和投稿材料 |

## 当前工作

正式实验已结束，目前没有训练或测试任务需要继续运行。下一步是依据冻结的 v3/v4 结果重写论文，而不是继续调参。现有 `main.tex` 和 `main.pdf` 仍是正式测试前的旧稿，不能直接投稿。
