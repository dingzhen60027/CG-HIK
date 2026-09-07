# Task-contract alignment：最终实验结论

本报告只回答预定的五个问题。Panda 历史轨迹数据直接读取；UR5e 和点查询的新测量已经结束。没有全局通过门槛，也不追加实验。数字由 `confik.task_contract_alignment.findings` 读取冻结汇总自动写出。

## 1. Solver-internal success 与 task-level admissibility 的失配有多常见？

失配明显依赖 solver，而不是在所有配置中普遍存在。每台机器人有 2,500 个具有同输入合法 witness 的点查询；TRAC 的三次搜索是 query 内重复，DLS 只运行一次。以下矩阵是原始调用计数，不能将三次搜索当成三个独立查询。

| Robot | Setting | Internal+/Task+ | Internal+/Task− | Internal−/Task+ | Internal−/Task− | 任一重复漏解 UID |
|---|---|---:|---:|---:|---:|---:|
| panda | Strict 5 | 7500 | 0 | 0 | 0 | 0 |
| panda | Position 5 | 7500 | 0 | 0 | 0 | 0 |
| panda | Orientation 5 | 7500 | 0 | 0 | 0 | 0 |
| panda | Aligned 5 | 7500 | 0 | 0 | 0 | 0 |
| panda | Strict 20 | 7500 | 0 | 0 | 0 | 0 |
| panda | Aligned 20 | 7500 | 0 | 0 | 0 | 0 |
| panda | DLS strict | 1939 | 0 | 548 | 13 | 13 |
| panda | DLS aligned | 2487 | 0 | 0 | 13 | 13 |
| ur5e | Strict 5 | 7500 | 0 | 0 | 0 | 0 |
| ur5e | Position 5 | 7500 | 0 | 0 | 0 | 0 |
| ur5e | Orientation 5 | 7499 | 0 | 0 | 1 | 1 |
| ur5e | Aligned 5 | 7500 | 0 | 0 | 0 | 0 |
| ur5e | Strict 20 | 7500 | 0 | 0 | 0 | 0 |
| ur5e | Aligned 20 | 7500 | 0 | 0 | 0 | 0 |
| ur5e | DLS strict | 1968 | 0 | 531 | 1 | 1 |
| ur5e | DLS aligned | 2499 | 0 | 0 | 1 | 1 |

panda: strict DLS 的 548/2500 (21.92%) 个查询内部未收敛，但返回命令通过公共合同。2487 个查询出现合法主迭代状态，其中 2397 个随后仍继续迭代；额外迭代 median/P95=1/24，额外 FK/residual evaluations median/P95=2/49。术语为 excess iterations after task admissibility；不推测 TRAC 未暴露的内部迭代，也不估算额外阶段时间。

ur5e: strict DLS 的 531/2500 (21.24%) 个查询内部未收敛，但返回命令通过公共合同。2499 个查询出现合法主迭代状态，其中 2346 个随后仍继续迭代；额外迭代 median/P95=1/24，额外 FK/residual evaluations median/P95=2/49。术语为 excess iterations after task admissibility；不推测 TRAC 未暴露的内部迭代，也不估算额外阶段时间。

strict 与 fully aligned TRAC 在两个机器人的全部 nominal 可行点查询上均成功，因此其主比较中不存在 witness-confirmed miss 可供恢复。UR5e orientation-only 有一个 UID 的一次搜索失败；这是同输入 witness-confirmed miss，不是不可行证明。DLS 的实际漏解数量并未因 task stop 而减少，已在矩阵中完整保留。

独立回放检查了 309,000 条新旧调用记录，accepted contract violations=0，2534 个旧受保护文件未变化。成功完整轨迹 witness 索引包含 1019 个实际运行序列。所有点查询 witness 位于固定源 NPZ，其 UID/hash 链接和返回配置均保存在 raw records 中。

500 个明确构造不可执行查询/robot 单列，不混入以上可行点指标：所有方法均 public rejection，全部 internal failure，false acceptance=0。TRAC 5/20 ms 消耗接近各自搜索预算；合同对齐没有避开真正不可执行输入的搜索成本。见 `point_main.csv` 中 witness_feasible=False 的完整时延。

## 2. Position tolerance 与 orientation tolerance 分别贡献了多少恢复？

nominal 点查询主比较的 strict 已全成功，所以 position-only、orientation-only、fully aligned 的恢复数量均为零；不能从这一 ceiling workload 归因轨迹恢复由哪种容差驱动。orientation-only 在 UR5e 反而出现上文的一次搜索损失。分量对齐的计算变化如下，ratio 为方法/strict 5 ms，区间按 query UID 配对分层重采样：

| Robot | Setting | Success difference pp [95% CI] | Mean cost ratio [95% CI] |
|---|---|---|---|
| panda | Position 5 | 0.000 [0.000, 0.000] | 1.022 [1.011, 1.032] |
| panda | Orientation 5 | 0.000 [0.000, 0.000] | 1.043 [1.025, 1.064] |
| panda | Aligned 5 | 0.000 [0.000, 0.000] | 0.936 [0.928, 0.945] |
| ur5e | Position 5 | 0.000 [0.000, 0.000] | 0.989 [0.979, 0.998] |
| ur5e | Orientation 5 | -0.013 [-0.040, 0.000] | 1.006 [0.993, 1.019] |
| ur5e | Aligned 5 | 0.000 [0.000, 0.000] | 0.960 [0.949, 0.970] |

单独对齐一种误差并不保证降低开销。fully aligned 的联合映射降低了这里的平均开销，但不等于两种单独作用可线性相加。5→20 ms 的 strict 配置在可行点上没有增加成功，也没有显示需要更大预算。

## 3. 对齐是否在两机器人上提高 verified success 或降低计算成本？

点查询：TRAC fully aligned 5 ms 保持全部成功并小幅减少平均时间；DLS task stop 保持与 strict 相同的 verified success，大幅减少计算。

| Robot | Setting | Verified % | P50 / P95 / P99 ms | Mean ms |
|---|---|---:|---|---:|
| panda | Position 5 | 100.000 | 0.211 / 0.379 / 0.681 | 0.241 |
| panda | DLS strict | 99.480 | 1.038 / 8.944 / 9.230 | 3.015 |
| panda | Aligned 5 | 100.000 | 0.199 / 0.332 / 0.478 | 0.221 |
| panda | Strict 5 | 100.000 | 0.211 / 0.367 / 0.487 | 0.236 |
| panda | Aligned 20 | 100.000 | 0.199 / 0.331 / 0.467 | 0.220 |
| panda | Strict 20 | 100.000 | 0.211 / 0.362 / 0.485 | 0.236 |
| panda | Orientation 5 | 100.000 | 0.210 / 0.382 / 0.799 | 0.247 |
| panda | DLS aligned | 99.480 | 0.625 / 1.042 / 4.479 | 0.762 |
| ur5e | Position 5 | 100.000 | 0.199 / 0.317 / 0.414 | 0.216 |
| ur5e | DLS strict | 99.960 | 1.185 / 8.113 / 8.427 | 2.757 |
| ur5e | Aligned 5 | 100.000 | 0.191 / 0.312 / 0.415 | 0.209 |
| ur5e | Strict 5 | 100.000 | 0.201 / 0.318 / 0.409 | 0.218 |
| ur5e | Aligned 20 | 100.000 | 0.192 / 0.313 / 0.410 | 0.210 |
| ur5e | Strict 20 | 100.000 | 0.200 / 0.320 / 0.424 | 0.222 |
| ur5e | Orientation 5 | 99.987 | 0.200 / 0.324 / 0.445 | 0.219 |
| ur5e | DLS aligned | 99.960 | 0.578 / 0.926 / 1.458 | 0.656 |

DLS 点查询两臂均包含相同的轻量 trace 记录成本；轨迹使用与历史 Panda 完全相同、未开启 trace 的 wrapper。DLS 的 25 iterations 不是与 TRAC 等价的 5/20 ms 时间预算。所有耗时包含必要转换、动态边界、求解和最终 verifier，不含离线序列化、统计和 trace 回放。

完整轨迹：以下 completion 为每次完整 sweep 的条数（每 sweep 40 条），累计时延是三次 TRAC sweep 的平均，不是三倍样本量。所有 150 帧均保留，失败只保持自身 previous_q，目标继续推进。

| Robot | Setting | Complete / 40 | P50 / P95 / P99 ms | Cumulative s |
|---|---|---:|---|---:|
| panda | DLS strict | 37 | 0.943 / 9.119 / 10.624 | 14.255 |
| panda | DLS aligned | 37 | 0.550 / 1.451 / 6.830 | 4.814 |
| panda | Strict 20 | 34/34/34 | 0.192 / 20.352 / 20.559 | 8.654 |
| panda | Strict 5 | 34/34/34 | 0.190 / 5.249 / 5.434 | 3.187 |
| panda | Aligned 20 | 37/37/38 | 0.180 / 0.462 / 20.392 | 3.600 |
| panda | Aligned 5 | 36/37/37 | 0.179 / 0.484 / 5.305 | 2.150 |
| ur5e | DLS strict | 35 | 0.816 / 7.813 / 7.955 | 16.965 |
| ur5e | DLS aligned | 28 | 0.495 / 5.066 / 7.769 | 6.586 |
| ur5e | Strict 20 | 37/37/37 | 0.169 / 0.303 / 20.357 | 4.932 |
| ur5e | Strict 5 | 37/37/37 | 0.169 / 0.300 / 5.256 | 2.032 |
| ur5e | Aligned 20 | 39/39/39 | 0.161 / 0.229 / 0.412 | 1.568 |
| ur5e | Aligned 5 | 39/39/39 | 0.161 / 0.223 / 0.405 | 1.167 |
| Robot | Aligned / strict | Completion difference pp [95% CI] | Cumulative cost ratio [95% CI] | Gained / lost UID |
|---|---|---|---|---|
| panda | Aligned 5 | 6.667 [1.667, 13.333] | 0.675 [0.482, 0.890] | 3/0 |
| panda | Aligned 20 | 8.333 [1.667, 16.667] | 0.416 [0.231, 0.720] | 4/0 |
| panda | DLS aligned | 0.000 [0.000, 0.000] | 0.338 [0.269, 0.415] | 0/0 |
| ur5e | Aligned 5 | 5.000 [0.000, 12.500] | 0.574 [0.330, 0.979] | 2/0 |
| ur5e | Aligned 20 | 5.000 [0.000, 12.500] | 0.318 [0.125, 0.985] | 2/0 |
| ur5e | DLS aligned | -17.500 [-30.000, -7.500] | 0.388 [0.287, 0.499] | 0/7 |

panda: fully aligned TRAC 5 ms 累计时延降低 32.53%。DLS 从 37/40 到 37/40，累计时间降低 66.23%。

ur5e: fully aligned TRAC 5 ms 累计时延降低 42.59%。DLS 从 35/40 到 28/40，累计时间降低 61.18%。

重要负结果：UR5e DLS task stop 丢失 7 条 strict 已完成轨迹（smooth 3、near-singular 2、joint-limit-return 2）。节省时间不能抵消这一完成损失。当前结果证明接受同一位姿合同下不同的数值停止过程可以产生不同的闭环路径，但不能仅凭这些结果证明具体的分支漂移因果，或保证每个失败输入从自身 previous_q 仍存在合法下一步。

TRAC 5 ms 在 Panda 的恢复集中于 near-singular，UR5e 的恢复位于 smooth/high-curvature。UR5e near-singular 没有增加整轨迹完成；joint-limit-return 平均耗时略升。所有 family 结果保留，不按显著性删选。

| Robot | Family | Strict→aligned completion /10 | Cumulative ratio |
|---|---|---|---:|
| panda | high_curvature | [10, 10, 10] → [10, 10, 10] | 1.042 |
| panda | joint_limit_return | [9, 9, 9] → [9, 9, 9] | 0.953 |
| panda | near_singular | [5, 5, 5] → [7, 8, 8] | 0.435 |
| panda | smooth | [10, 10, 10] → [10, 10, 10] | 0.934 |
| ur5e | high_curvature | [9, 9, 9] → [10, 10, 10] | 0.514 |
| ur5e | joint_limit_return | [10, 10, 10] → [10, 10, 10] | 1.004 |
| ur5e | near_singular | [9, 9, 9] → [9, 9, 9] | 0.981 |
| ur5e | smooth | [9, 9, 9] → [10, 10, 10] | 0.290 |

截止时间结局也单独保留；几何合法但迟到的帧仍按预定几何反馈规则更新，不把这一软件流程解释为 deadline-triggered controller 或硬实时控制。

| Robot | Setting | All-frame ≤20 ms completion /40 | Frame verified % | Frame accepted ≤20 ms % |
|---|---|---|---:|---:|
| panda | DLS strict | [37] | 96.800 | 96.800 |
| panda | DLS aligned | [37] | 96.800 | 96.800 |
| panda | Strict 20 | [34, 34, 34] | 93.994 | 93.994 |
| panda | Strict 5 | [34, 34, 33] | 93.983 | 93.978 |
| panda | Aligned 20 | [37, 37, 38] | 98.161 | 98.161 |
| panda | Aligned 5 | [36, 36, 37] | 97.278 | 97.272 |
| ur5e | DLS strict | [35] | 95.317 | 95.317 |
| ur5e | DLS aligned | [28] | 88.367 | 88.367 |
| ur5e | Strict 20 | [37, 36, 37] | 96.817 | 96.811 |
| ur5e | Strict 5 | [37, 37, 37] | 96.817 | 96.817 |
| ur5e | Aligned 20 | [39, 39, 38] | 99.583 | 99.578 |
| ur5e | Aligned 5 | [39, 39, 39] | 99.572 | 99.572 |

合法误差确实变大，不能隐藏。以下为 accepted P95，完整 median/P95/P99/max 分布见 main JSON：

| Population | Robot | Setting | Position P95 mm | Orientation P95 deg |
|---|---|---|---:|---:|
| point | panda | DLS strict | 0.05718 | 0.00364 |
| point | panda | Aligned 5 | 0.57600 | 0.13970 |
| point | panda | Strict 5 | 0.00733 | 0.00050 |
| point | panda | DLS aligned | 0.80043 | 0.14469 |
| point | ur5e | DLS strict | 0.05951 | 0.00358 |
| point | ur5e | Aligned 5 | 0.50289 | 0.06943 |
| point | ur5e | Strict 5 | 0.00745 | 0.00036 |
| point | ur5e | DLS aligned | 0.79077 | 0.13149 |
| trajectory | panda | DLS strict | 0.13567 | 0.00510 |
| trajectory | panda | DLS aligned | 0.73631 | 0.05234 |
| trajectory | panda | Strict 5 | 0.00348 | 0.00031 |
| trajectory | panda | Aligned 5 | 0.39150 | 0.03792 |
| trajectory | ur5e | DLS strict | 0.23420 | 0.00881 |
| trajectory | ur5e | DLS aligned | 0.81253 | 0.09000 |
| trajectory | ur5e | Strict 5 | 0.00660 | 0.00037 |
| trajectory | ur5e | Aligned 5 | 0.38533 | 0.03572 |

## 4. 0.5×、1×、2× 合同下是否呈现可解释变化？

三个尺度采用相同的 500 个已在最严格尺度验证 witness 的查询/robot。TRAC strict 和 fully aligned 在每个尺度都全部成功，不能声称随尺度恢复更多 TRAC 漏解。Panda DLS 在 0.5× 漏 2 个，到 1×/2× 均成功；UR5e DLS 三尺度均成功。

| Robot | Scale | Setting | Verified % | P50/P95/P99 ms | Mean ms | Accepted position P95 / tolerance |
|---|---:|---|---:|---|---:|---:|
| panda | 0.5 | Strict 5 | 100.00 | 0.207/0.334/0.434 | 0.222 | 0.014 |
| panda | 0.5 | DLS aligned | 99.60 | 0.589/1.098/7.122 | 0.819 | 0.893 |
| panda | 0.5 | Aligned 5 | 100.00 | 0.196/0.289/0.486 | 0.211 | 0.614 |
| panda | 1.0 | DLS aligned | 100.00 | 0.589/0.941/4.167 | 0.711 | 0.838 |
| panda | 1.0 | Aligned 5 | 100.00 | 0.194/0.262/0.389 | 0.203 | 0.507 |
| panda | 1.0 | Strict 5 | 100.00 | 0.206/0.321/0.407 | 0.219 | 0.007 |
| panda | 2.0 | DLS aligned | 100.00 | 0.586/0.927/1.941 | 0.630 | 0.803 |
| panda | 2.0 | Strict 5 | 100.00 | 0.207/0.320/0.395 | 0.221 | 0.004 |
| panda | 2.0 | Aligned 5 | 100.00 | 0.194/0.253/0.351 | 0.203 | 0.580 |
| ur5e | 0.5 | Strict 5 | 100.00 | 0.195/0.254/0.326 | 0.203 | 0.015 |
| ur5e | 0.5 | DLS aligned | 100.00 | 0.547/0.989/2.033 | 0.695 | 0.903 |
| ur5e | 0.5 | Aligned 5 | 100.00 | 0.187/0.237/0.316 | 0.195 | 0.555 |
| ur5e | 1.0 | DLS aligned | 100.00 | 0.543/0.864/1.150 | 0.614 | 0.820 |
| ur5e | 1.0 | Aligned 5 | 100.00 | 0.185/0.238/0.310 | 0.193 | 0.453 |
| ur5e | 1.0 | Strict 5 | 100.00 | 0.194/0.253/0.370 | 0.203 | 0.007 |
| ur5e | 2.0 | DLS aligned | 100.00 | 0.539/0.844/0.875 | 0.553 | 0.771 |
| ur5e | 2.0 | Strict 5 | 100.00 | 0.193/0.252/0.330 | 0.201 | 0.004 |
| ur5e | 2.0 | Aligned 5 | 100.00 | 0.183/0.223/0.301 | 0.189 | 0.534 |

合同放宽时 fully aligned TRAC 的相对平均开销收益总体增大，DLS task 的平均时间下降；strict 内部停止条件不随尺度变化。单个分位点并非严格单调，因此结论是可解释的总体计算趋势，不是每条查询的单调改进定理。nominal 始终是主合同。敏感性改变的是明确列出的任务要求，不能将更宽公共合同的成功直接当作同合同算法收益。

## 5. 合同对齐之后，learned adaptive computation 还剩什么实际价值？

本轮不训练、不运行路由器；这里是旧 allocation 证据的只读边界分析，不是将路由器重新接到 aligned TRAC/DLS 后的新增直接比较。已有 witness-feasible points 中 Full 与 hard 成功相同，但平均耗时增加；routing-only 无稳定净收益，geometry rule 与 full 接近，P95 selection 的独立尾延迟优势较弱。原轨迹总体节省主要来自共同失败组，reject 能在失败密集负载减少徒劳计算。旧 external TRAC 比较具有明显速度优势，但旧浮点端点拒绝是实现问题，原记录与说明均保留，不计入新对齐恢复。详见 `allocation_boundary.json`、原 `REVISION_EXPERIMENT_FINDINGS.md` 和原 `REVISION_MEASUREMENT_NOTES.md`。

因此值得保留的是明确的失败成本管理，而非预设需要学习入口选择。当前证据支持：Align first; allocate only when residual failure cost justifies the allocation overhead。这不是证明任何 learned allocation 在 aligned solver 上必然无效。阶段一至此结束，不因 UR5e DLS 负结果调整容差、预算、数据或增加方法。

交付索引：`01_protocol/` 固定身份/映射/环境；`02_point_study/`、`03_contract_sensitivity/`、`04_trajectory_study/` 原始记录；`05_aggregate/` CSV/JSON、completion UID、first-failure、witness index、paired CI；`reports/` 五张主图和 DLS 补图的 PDF/SVG/PNG、主表。全部输入转换/边界/solve/verifier 的时间分项留在点查询 raw records。

只读汇总曾因 UR5e first-failure 行已有 robot 字段而发生重复关键字异常；修复仅为字典合并，原 execution seal 保留，已写出的点/敏感性表未覆盖，未重跑任何 solver。`analysis_revision.json` 保存逐字节允许变更与哈希。该报告生成在实验全部结束、论文正文开始修改之前。
