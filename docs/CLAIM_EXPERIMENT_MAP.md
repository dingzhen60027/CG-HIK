# Claim--Experiment Map

更新日期：2026-09-02

本文件把 Introduction 的科学问题映射到已有证据。development-only 诊断与冻结 formal test 必须分开报告。

## RQ1：查询间计算异质性是否存在？

**Claim.** 同一 verified solver portfolio 的不同入口对不同查询具有不同尾延迟和数值工作量；固定入口不能逐查询达到最低代价。

**Evidence source.** `outputs/counterfactual_v4_bulk/`，共 40,000 个 training/calibration/policy-validation 查询。对每个查询实际执行 `easy/medium/hard`，每入口保存五个 raw latency samples；`fixed_robust` 是 `easy` 的逐查询语义 alias。`bulk_summary.json` 明确记录 `test_data_loaded=false`。

**Development-only readout.** empirical oracle 定义为：在终端 verified-success 查询上，选择五次样本经验 P95 最低的入口。

| 指标 | Panda | UR5e |
|---|---:|---:|
| verified-success development queries | 17,507 | 17,584 |
| oracle chooses easy | 26.69% | 23.41% |
| oracle chooses medium | 27.77% | 25.36% |
| oracle chooses hard | 45.54% | 51.22% |
| easy minus oracle, mean empirical P95 | 0.391 ms | 0.478 ms |
| easy minus oracle, P95 gap | 1.399 ms | 1.415 ms |
| queries with positive easy-to-oracle gap | 73.31% | 76.59% |
| global-best fixed entry | hard | hard |
| best-fixed minus oracle, mean P95 gap | 0.137 ms | 0.154 ms |
| easy minus oracle, mean FEV | 1.60 | 2.30 |

`hard_valid` 最能激发入口差异：Panda/UR5e 的 empirical oracle 分别在 94.69%/94.05% 的该类查询上选择 hard，easy-to-oracle 平均 P95 差为 1.238/1.118 ms。

**Interpretation.** 异质性不仅是“hard 总体更快”：即使 hard 是全局最佳固定入口，逐查询 oracle 仍有正的剩余 gap；这为 query-adaptive entry selection 提供动机。

**Limits.** 每查询只有五个 timing repeats，empirical P95 接近样本最大值，因而 oracle 是噪声诊断，不是训练标签，也不是正式测试 claim。模型直接对全部 raw repeats 做 quantile loss。

## RQ2：入口代价能否被预测？

**Claim.** compact predictor 能在共享 semantic-success 约束下预测动作特定 P95，并以有限 regret 路由。

**Evidence source.** `outputs/release_v4_candidate/` 的 training/calibration/policy-selection artifacts，以及 `outputs/release_v4_locked/` 的冻结 exact predictor/policy。训练、calibration 和 policy-validation query sets 按角色隔离。

| Policy-validation 指标 | Panda | UR5e |
|---|---:|---:|
| route counts (easy/medium/hard/reject/defer) | 572/9/1454/442/23 | 118/0/1931/418/33 |
| successful non-abstained queries | 2,031 | 2,049 |
| empirical-oracle agreement | 47.42% | 53.88% |
| $\leq 0.15$ ms empirical-oracle regret | 64.65% | 69.35% |
| routing regret, mean empirical P95 | 0.135 ms | 0.133 ms |
| routing regret, median | 0.016 ms | 0.000 ms |
| routing regret, P95 | 0.467 ms | 0.605 ms |
| raw-sample P95 coverage, easy | 94.66% | 94.16% |
| raw-sample P95 coverage, medium | 94.24% | 94.42% |
| raw-sample P95 coverage, hard | 94.71% | 94.03% |
| successful queries incorrectly rejected | 0 | 0 |
| successful queries incorrectly deferred | 0 | 0 |

**Interpretation.** exact oracle agreement不是唯一目标：五样本 empirical winner 本身不稳定。Panda/UR5e 的 P50 winner 与 empirical-P95 winner 仅一致 54.79%/59.24%；单个 repeat winner 的平均一致率也有限。更合适的证据是 raw-repeat quantile coverage、routing regret 与后续冻结系统结果。

这里的 policy-validation 是 **development policy-selection split**：它与 model fitting 和 calibration 分离，但参与冻结策略选择，因而不是 confirmatory test。

**Comparator boundary.** 现有 frozen bulk 没有保存旧 categorical router 与 Cartesian threshold guard 在同一个 action-complete matrix 上的逐查询预测，所以不能伪造它们的 oracle regret。它们只在 formal system-level benchmark 中比较 success、FEV 和 latency。

## RQ3：预测是否转化为冻结系统收益？

**Claim.** 相对从 easy 开始的 fixed robust cascade，query-adaptive routing 在共享数值组合和 verifier 下保持 feasible success/trajectory completion，同时降低 feasible P95 与 FEV；command reject 避免大部分 known-infeasible 求解。

**Evidence source.** `outputs/test_v4_aggregate_repair_v1/`；primary confirmatory seed 使用 fresh 25,000-query/robot workload，另两个训练 seed 在相同测试查询上做 sensitivity analysis。六个 robot×seed 组合、744 checkpoints、650,000 records 均已封存并独立复核。

**Estimand boundary.** RQ1/RQ2 的经验 P95 是每个开发查询五次 repeat 的 query-level statistic；formal headline P95 则把每个方法在 fresh point queries 上的三次 raw solve calls 汇总后取总体分位数，并按 query 做 paired bootstrap。两者回答不同问题，不能把 development oracle gap 直接解释成 formal P95 的数值预测。

| Formal metric: proposed / fixed | Panda | UR5e |
|---|---:|---:|
| feasible success gap | 0 | 0 |
| feasible P95 ratio (95% paired CI) | 0.7538 [0.7504, 0.7578] | 0.7427 [0.7401, 0.7454] |
| feasible P99 ratio (95% paired CI) | 0.9931 [0.9846, 0.9984] | 0.8808 [0.8393, 0.8940] |
| feasible mean FEV reduction | 16.14% | 36.27% |
| feasible P50 change | +16.47% | +16.45% |
| known-infeasible reject recall | 95.65% | 93.95% |
| known-infeasible FEV avoided | 95.69% | 93.92% |
| trajectory completion | 95.0% / 95.0% | 78.33% / 78.33% |
| point OOD AUROC / AUPRC | 0.430 / 0.210 | 0.504 / 0.230 |
| defer semantic match | 100% | 100% |
| defer recovery success | 0 | 0 |

Confirmatory success, P95, P99 and trajectory margin tests组成预注册的 8-hypothesis family，joint Holm gate 通过。Panda operational gate 仍失败，UR5e 通过，overall paper gate 为 false。

## Claim discipline

| 可写 | 不可写 |
|---|---|
| query-adaptive routing reduces feasible P95 relative to the shared fixed cascade on both evaluated robots | uniformly faster at every quantile |
| verified feasible success and whole-trajectory completion match the fixed comparator in this benchmark | guaranteed reliability or hard real-time safety |
| command reject avoids most numerical work on the constructed known-infeasible population | universal reachability or infeasibility certification |
| defer is semantically identical to the fixed cascade | defer improved recovery; observed recovery count was zero |
| a frozen OOD score triggers conservative defer | strong OOD detection; point AUROC was weak |
| two robot models and three model seeds were evaluated | six independent test datasets |
| exact URDF/Pinocchio offline sequential kinematics | Isaac Lab physics, collision, dynamics, torque/contact, or real-robot validation |

## Reproducible read-only sources

- Development data: `outputs/counterfactual_v4_bulk/`
- Frozen predictor and policy: `outputs/release_v4_locked/`
- Formal aggregate: `outputs/test_v4_aggregate_repair_v1/`
- Formal aggregate audit: `python scripts/audit_test_v4_aggregation_repair.py`
- Current narrative facts: `docs/RESEARCH.md`
