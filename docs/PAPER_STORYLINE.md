# CG-HIK 论文主线

更新日期：2026-09-02

## 一句话主线

> **CG-HIK treats online inverse kinematics as query-adaptive resource allocation over a shared numerical solver portfolio: learning selects where computation should begin, numerical geometry generates joint commands, and deterministic verification alone decides whether a command is admissible.**

中文：CG-HIK 将在线逆运动学视为共享数值求解组合上的逐查询资源分配问题：学习模型决定从哪里开始计算，数值几何生成关节命令，确定性验证器独占命令接受权。

全文短句：**Compute adaptively; accept deterministically.**

## Problem--Gap--Hypothesis--Method--Evidence--Conclusion

| 环节 | 论文中的唯一表述 |
|---|---|
| Problem | 连续在线 IK 必须在有限控制周期内，把任务空间目标转成满足位姿、关节限位和单帧连续性合同的关节命令。多解、奇异性、初值和可达性使不同查询的求解成本差异显著。 |
| Gap | 解析、数值、学习和混合 IK 已经改善了精度、候选覆盖和鲁棒性，但所审阅的混合系统通常预先固定候选数量、迭代预算或求解顺序。即使级联会在成功后短路，固定从同一入口开始仍可能支付可预测地冗余的计算前缀。 |
| Hypothesis | **Robust coverage does not require every query to pay the worst-case computational cost.** 在保持相同候选、数值阶段、回退和验证合同的条件下，可以根据查询状态选择成本更低但仍充分的求解入口。 |
| Method | 对每个开发查询实际运行 easy、medium、hard 三个级联入口，记录五次原始延迟、FEV、fallback、失败原因和 verified success；轻量 MLP 预测共享语义成功概率、各入口 P50/P95 延迟和 fail-all 概率；策略在满足成功风险与 20 ms 约束的入口中选择预测 P95 最低者。 |
| Safety boundary | 高置信、分布内的 portfolio failure 才触发零求解 command reject；OOD 或不确定查询 defer 到完整 fixed robust cascade；所有非 reject 输出必须通过相同 deterministic verifier。 |
| Evidence | 开发集 40,000 queries 的反事实路径矩阵验证计算异质性与可预测性；冻结的 fresh test 在 Panda、UR5e 上验证 verified success、FEV、P50/P95/P99、deadline、reject、defer 和 trajectory behavior。 |
| Conclusion | 当前证据支持“共享鲁棒覆盖下的查询自适应计算”：相对 fixed cascade，两台机器人保持 feasible success 与 trajectory completion，同时降低 feasible P95 和 FEV；但 P50 增加、Panda P99 几乎不变、点查询 OOD 区分很弱，故不能主张全面加速或已解决 OOD。 |

## 因果闭环

```text
连续任务空间目标
    ↓
查询间存在不同的初始化、奇异性、限位和可达性条件
    ↓
共享鲁棒级联可覆盖困难查询，但固定从第一阶段开始会产生冗余计算
    ↓
对同一开发查询执行所有入口，得到 action-complete latency/solver outcomes
    ↓
学习满足共同语义合同的入口成本，而不是人为“难度类别”
    ↓
选择最低预测 P95 的 eligible entry；reject 和 defer 承担不同语义
    ↓
共享数值级联生成命令；同一 verifier 决定接受
    ↓
用开发集回答“异质性与可预测性”，用冻结测试回答“系统收益”
```

## 术语账本

| 术语 | 精确定义 | 不应写成 |
|---|---|---|
| online IK query | 目标位姿、上一关节状态和控制步长构成的单帧查询 | 孤立的 pose-to-joint 回归样本 |
| decision/cascade entry | `easy`、`medium`、`hard` 是进入同一鲁棒级联的位置 | 查询的真实难度类别 |
| fixed robust cascade | 总是从 `easy` 开始并允许完整升级的共享审计基线 | 不同 solver 或更宽松 verifier |
| counterfactual pathway supervision | 对同一开发查询实际执行全部三个入口形成 action-complete 结果矩阵 | 因果推断意义上的未观测反事实 |
| shared semantic success | 三入口最终使用相同 robust fallback，故终端 verified-success 目标共享并广播 | 三个动作特定成功分类头 |
| action-specific latency | 三入口分别预测 P50/P95；以五个 raw repeats 的 pinball loss 训练 | 单次 wall-clock 回归或人工难度标签 |
| command reject | 非 OOD、无 eligible entry 且 fail-all 高置信时，零数值求解拒绝 | 对所有困难或 OOD 查询直接拒绝 |
| OOD/uncertainty defer | 回到从 easy 开始的完整 fixed robust cascade | reject 或已证明的恢复增益 |
| deterministic verifier | 检查数值有限性、位姿误差、URDF 关节限位和单帧速度合同；唯一接受命令的模块 | 学习模型的置信度阈值 |
| FEV | 当前 portfolio 内部的数值函数评估代理 | FLOPs、总能耗或通用计算复杂度 |
| tail latency | batch-one、warmup 后用 `perf_counter_ns` 得到的 P95/P99 | 硬实时上界 |

## 三个研究问题

- **RQ1 — Computation heterogeneity:** 同一查询在不同级联入口下的 verified outcome、FEV 和尾延迟是否存在足以利用的差异？
- **RQ2 — Predictability:** compact router 能否从查询特征预测入口尾延迟，并以有限 regret 逼近 development-only empirical oracle？
- **RQ3 — System gain:** 冻结路由能否在保持相同 verifier、feasible success 和 trajectory completion 的条件下，降低端到端 P95 与 FEV，并可靠执行 reject/defer 语义？

## 正式结论边界

正式证据支持：

- Panda/UR5e feasible P95 ratio 为 0.7538/0.7427；
- feasible mean FEV 降低 16.14%/36.27%；
- known-infeasible reject recall 为 95.65%/93.95%，避免 95.69%/93.92% 的 fixed FEV；
- trajectory completion 与 fixed 相同；
- joint Holm confirmatory gate 通过。

正式证据同时否定更强说法：

- feasible P50 约增加 16.5%；Panda P99 ratio 为 0.9931；
- point OOD AUROC 仅 0.430/0.504，不能声称强 OOD detector；
- defer semantic match 为 100%，但 defer recovery success 为 0；
- Panda operational gate 失败，因冻结的 OOD feasible false-reject improvement 为 0；因此 overall paper gate 为 false；
- 实验是精确 URDF/Pinocchio 的离线运动学 benchmark，不是 Isaac Lab 物理、碰撞、动力学、实机或硬实时安全验证。
