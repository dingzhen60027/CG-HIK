# CG-HIK 研究主线与项目总览

更新日期：2026-08-31

## 1. 核心问题

连续机器人控制中的逆运动学不是一次性的“求一个关节角”。控制器连续接收末端目标，必须在有限周期内选择一个平滑、可执行并满足关节约束的关节命令。不同查询需要的数值成本差异很大：局部连续请求可能只需一次 DLS 更新，近奇异或错误收敛盆地的请求需要更强初始化和回退，明确不可执行的请求继续迭代只会浪费预算。

CG-HIK 的核心假设是：

> 学习模型最适合预测解盆地、失败风险和求解成本；数值几何负责生成解；确定性验证器负责最终接受。

因此，论文讲的不是“神经网络取代数值 IK”，而是如何在不交出命令接受权的前提下，用学习模型分配数值求解资源。

## 2. 方法主线

### 2.1 基础架构

```text
目标位姿 + 上一关节状态
          ↓
历史条件候选模型
          ↓
入口路由：easy / medium / hard / reject
          ↓
共享的 DLS / TRF 数值级联与回退
          ↓
确定性运动学验证器
          ↓
接受命令或拒绝
```

四个入口的含义是：

- `easy`：从上一关节状态执行轻量 DLS，失败后继续升级；
- `medium`：从最佳学习候选开始轻量精修；
- `hard`：使用更完整的候选、上一状态和 TRF/KDTree 回退；
- `reject`：对高置信、分布内的不可执行请求零数值求解拒绝。

`easy/medium/hard` 是级联入口，不是真实难度标签。所有被接受的命令都由同一数值级联产生，并通过同一验证合同：位姿误差、数值有限性、URDF 关节限位和单帧速度约束。

### 2.2 主对照

主因果比较是 `fixed_robust_cascade` 与 CG-HIK。两者共享候选模型、DLS、TRF、回退、预算和验证器；fixed 始终从 easy 开始，CG-HIK 只改变入口或执行拒绝。因此差异主要反映资源路由，而不是不同 solver 或不同接受标准。

### 2.3 v4 扩展

v4 将固定四分类升级为反事实延迟路由和双弃权：

1. 对开发集中的同一查询分别执行 `easy/medium/hard`，记录 raw latency repeats、FEV、fallback 和失败原因；
2. 训练 compact MLP，预测共享的语义可行性以及各入口的 P50/P95 latency；
3. 在满足冻结成功风险约束的动作中，选择预测 P95 最低的入口；
4. 高置信不可执行请求执行 `reject`；
5. OOD 或模型不确定请求执行 `defer`，进入完整 fixed robust cascade，而不是直接拒绝。

开发数据表明三个入口在终端 robust fallback 后的 verified-success 标签没有差异，因此模型使用共享语义成功头和动作特定延迟头。论文不能声称模型学到了三个不同的终端成功概率。

## 3. 研究演化

### v2：算法收益与 eager 部署矛盾

`paper_v2` 证明入口路由可以保持可达查询成功表现、减少 FEV，并对构造的不可执行查询执行零求解拒绝。但 eager 风险模型带来约 2 ms 的固定单样本开销，正式 feasible P95 latency gate 未通过。

这个负结果把研究从“数值工作量是否减少”推进到“节省能否转化为端到端延迟”。

### v3：定位开销并做精确部署

validation-only `latency_pilot_v3` 将主要固定开销定位到 sklearn HistGradientBoosting 的逐树单样本推理。随后用 exact TorchScript 封装冻结种子模型、风险模型和策略，在 Panda、UR5e × seeds 17/29/43 上完成输出、路由、命令、FEV 和轨迹行为的数值等价检查。

fresh `test_v3` 证实了 P95 和 FEV 收益，也揭示 P50 上升、P99 改善有限和部分主机负载偏差。v3 的意义是把算法节省与部署实现分开验证，而不是把 TorchScript 当作新的 IK 算法贡献。

### v4：学习动作成本，并区分 reject 与 defer

v4 development 完成了 validation pilot、40,000-query bulk、Panda/UR5e 独立模型训练与校准、策略选择和 `release_v4_locked`。bulk 包含 160,000 条 query-action rows 和 600,000 次实际计时 solver executions。模型和阈值只使用 training、calibration 和 policy-validation 数据。

`test_v4` 在冻结 release 后使用全新数据一次性执行。原 runner 在六组测量全部完成后的聚合阶段遇到 JSON key-order 缺陷；没有重跑查询。独立 aggregation-only repair 只重新读取封存指标并修正成员检查，随后通过独立 attestation 和标准库复算审计。

## 4. 正式实验设计

每台机器人使用 25,000 条查询或轨迹帧：

| Population | 数量 |
|---|---:|
| ID feasible points | 10,000 |
| Known-infeasible points | 2,000 |
| OOD feasible points | 4,000 |
| 40 条 ID trajectories | 6,000 frames |
| 20 条 OOD trajectories | 3,000 frames |

primary confirmatory run 使用 seed17，比较七种方法；seeds 29/43 只运行 fixed、v3 和 v4，作为相同 test queries 上的模型敏感性分析，不能当作三个独立测试集。六个 robot×seed 组合共生成 744 个 checkpoint 和 650,000 条 method-query records。

正式比较覆盖 fixed robust cascade、v3 CG-HIK、Cartesian threshold guard、learned seed + fixed refinement、previous-state DLS、SciPy trust-region reflective baseline 和 v4 counterfactual router。`trf_previous` 不是 TRAC-IK；当前环境没有可复现的 TRAC-IK 依赖，因此论文必须使用准确名称。

## 5. 核心结果

### 5.1 v4 与 fixed robust cascade

| Robot | Feasible success gap | P95 ratio v4/fixed | P99 ratio | Feasible mean FEV reduction | Infeasible reject recall | Infeasible FEV avoided | Trajectory completion v4/fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Panda | 0 | 0.7538 | 0.9931 | 16.14% | 95.65% | 95.69% | 95.0% / 95.0% |
| UR5e | 0 | 0.7427 | 0.8808 | 36.27% | 93.95% | 93.92% | 78.33% / 78.33% |

这组结果支持三个结论：

1. 在当前查询分布上，资源路由没有牺牲相对 fixed 的 verified feasible success 或 whole-trajectory completion；
2. v4 明显降低 feasible P95 和数值工作量；
3. 对已知不可行请求，command reject 能在高召回下避免大部分徒劳求解。

同时，v4 的 feasible P50 比 fixed 高约 16.5%，Panda P99 几乎不变。v4 不是所有延迟分位数上的全面加速。

### 5.2 OOD 与 defer

| Robot | Point OOD AUROC | Point OOD AUPRC | Defer semantic match | Defer recovery success |
|---|---:|---:|---:|---:|
| Panda | 0.430 | 0.210 | 100% | 0 |
| UR5e | 0.504 | 0.230 | 100% | 0 |

`defer` 的语义实现正确：它与完整 fixed cascade 匹配。但冻结 OOD score 在点查询上接近或低于随机，本次测试也没有观察到 defer 恢复成功。因此 v4 证明了安全的回退机制，而没有证明强 OOD 检测器。

### 5.3 正式 gate

```text
joint Holm confirmatory gate = true
Panda robot gate             = false
UR5e robot gate              = true
overall paper gate           = false
```

Panda 唯一失败项是冻结的 `ood_feasible_false_reject_improvement`：v4 和 v3 在 Panda OOD feasible points 上都没有 false reject，改善估计为 0，而预注册门要求严格大于 0。这个结果不能在测试后通过修改阈值或改写 gate 消除。

## 6. 论文应如何表述

### 可以作为核心贡献

1. 将连续在线 IK 表述为共享数值级联上的 verified resource allocation；
2. 用历史条件候选和入口路由减少不必要的数值工作，同时让 verifier 保留唯一接受权；
3. 用反事实 raw latency samples 学习动作成本，而不是把入口当作人工难度类别；
4. 明确分离高置信不可执行的 `reject` 与不确定/OOD 的 `defer`；
5. 在两个机器人、三组模型和 fresh formal test 上同时报告效率收益与预注册负结果。

### 不能写成已经证明

- 神经网络替代数值 IK；
- 两台机器人所有正式 gate 均通过；
- 已解决 OOD 检测或 OOD 鲁棒性；
- defer 在本次测试中提高恢复成功率；
- 所有延迟分位数都优于 fixed 或 v3；
- collision、torque、contact、动力学控制器或实机安全；
- Isaac Lab 闭环物理验证。

当前实验是基于 Pinocchio 和精确 URDF 的离线运动学 benchmark。Python 环境名为 `isaaclab_3`，但这不等于实验在 Isaac Lab 物理仿真中执行。

## 7. 论文结构

目标稿件可按以下顺序重建：

1. **Introduction**：从单点求根转向连续控制中的可信资源分配；
2. **Related Work**：数值稳定、多解/历史连续性、生成候选和混合求解；
3. **Method**：历史条件候选、共享数值级联、反事实 cost heads、reject/defer 和 verifier；
4. **Experimental Design**：数据角色隔离、机器人/seed、基线、指标、预注册 gate 和 Holm family；
5. **Results**：v2 eager 负结果、v3 exact 部署、v4 P95/FEV/reject/trajectory、弱 OOD 和 Panda gate 失败；
6. **Discussion**：资源分配解释、P50/P99 权衡、OOD 限制、外部有效性；
7. **Conclusion**：学习分配预算，数值几何生成解，验证器接受命令。

目前最合适的论文贡献不是继续加模块，而是把这条从算法收益、部署瓶颈到双弃权负结果的链条写清楚。

## 8. 证据位置

| 内容 | 位置 |
|---|---|
| v2 正式结果 | `outputs/paper_v2_seed*`, `outputs/paper_v2_aggregate` |
| v3 profiling 与 release | `outputs/latency_pilot_v3`, `outputs/release_v3_locked` |
| v3 fresh test | `outputs/test_v3_seed*`, `outputs/test_v3_aggregate` |
| v4 development | `outputs/counterfactual_v4_*`, `outputs/release_v4_candidate` |
| v4 frozen release | `outputs/release_v4_locked` |
| v4 raw formal measurements | `outputs/.test_v4_seed*.incomplete` |
| v4 aggregate | `outputs/test_v4_aggregate_repair_v1` |
| repair attestation | `outputs/test_v4_aggregate_repair_v1_attestation_v1` |
| v3 audit implementation | `scripts/audit_test_v3_locked.py` |
| v4 composite audit implementation | `scripts/audit_test_v4_aggregation_repair.py` |

`.test_v4_*.incomplete` 的名称来自原聚合失败，不表示六组科学测量未完成。日常写作优先使用本文件；需要核查数字时再进入相应 aggregate 或运行只读审计脚本。

## 9. 当前任务

实验阶段已经结束。接下来只有两项工作：

1. 用冻结的 v3/v4 证据重写 `paper_mdpi_machines_v3/main.tex`；
2. 补齐作者、单位、基金、代码/数据归档和投稿信息。

如果未来继续研究 OOD、碰撞或物理闭环，应建立新的 v5 数据和预注册测试，不应回到 `test_v4` 调整当前方法。
