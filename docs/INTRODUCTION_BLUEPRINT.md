# Introduction Blueprint

目标标题：**Query-Adaptive Hybrid Inverse Kinematics through Counterfactual Tail-Latency Routing and Deterministic Verification**

核心问题：**How can learned priors, numerical solvers, and deterministic verification be coordinated to enable query-adaptive computation for online inverse kinematics under latency and command-admissibility constraints?**

核心假设：**Robust coverage does not require every query to pay the worst-case computational cost.** 对当前 fixed cascade 的精确解释是：它会在成功后短路，但固定从同一入口开始，仍可能执行对当前查询可预测地冗余的前缀。

## 八段结构

| 段落 | 唯一任务 | 关键内容 | 外部引用簇 | 内部证据 |
|---|---|---|---|---|
| 1 | 建立连续在线 IK 问题 | 任务空间到关节空间；$q_t=\Pi(x_t^d,q_{t-1},\mathcal C_t,B_t)$；分支、限位、单帧连续性和时限 | RelaxedIK, CppFlow, sequential IK, MimicIK | `docs/RESEARCH.md` §1；test config 的 20 ms/trajectory contract |
| 2 | 肯定解析、数值与约束 IK | 集合值逆；DLS；solver portfolio；几何分解；约束与奇异跨越 | Wampler, Nakamura, Chiaverini, TRAC-IK, variable step, IK-Geo, EAIK, GeoFIK, IKSPARK, viability, singularity transition, HJCD-IK | shared DLS/TRF/verifier architecture |
| 3 | 肯定学习型多解与历史条件 IK | 解分布、图结构、无序集合、reference/history state、diffusion；学习擅长 proposal 而非 command certification | IKFlow, GGIK, set IK, learning-by-example, sequential IK, MimicIK, IKDiffuser, KineNN, EMIKNet | candidate model never accepts commands |
| 4 | 建立混合范式 | learning proposes, numerical optimization refines, constraints verify；指出 hybrid 不等于 compute-adaptive | IKFlow, CycleIK, CppFlow, ETA-IK, DiffusionSeeder, MPD, Machines hybrid, XGNN, IKSel | fixed/proposed 共享 portfolio 与 verifier |
| 5 | 定义缺口与 RQ1 | 从 algorithm selection 视角定义 query heterogeneity；entry 不是 difficulty class；提出核心假设 | Rice, ASlib | 40k action-complete development matrix；oracle entry/gap/FEV |
| 6 | 给出方法与 RQ2 | 五次 raw timings；shared terminal-success head；action P50/P95 heads；20 ms eligibility；min-P95 + tie margin | ASlib 仅作一般监督框架背景 | `collector.py`, `model.py`, `policy.py`; policy-selection regret/coverage |
| 7 | 命令合同与 RQ3 | verifier 独占接受；non-OOD high-confidence failure 才 reject；OOD/uncertain defer 到 fixed | runtime failure detection 仅作邻域背景 | frozen runtime spec；reject zero FEV/stage；defer semantic match |
| 8 | 三项贡献与证据闭环 | 问题表述、方法、冻结评估；同时报告 P50/P99/OOD/gate 负结果 | 无需新增引用 | fresh formal aggregate 与 joint Holm |

## 段间过渡

```text
continuous command interface
→ set-valued geometry and constrained solvers
→ learned global priors
→ hybrid division of labor
→ fixed-entry computation gap
→ counterfactual P95 routing
→ deterministic acceptance plus two-sided abstention
→ three contributions and bounded evidence
```

## 三项贡献的固定写法

1. **Problem formulation.** Continuous online IK is formulated as verified resource allocation over a shared numerical portfolio; learned proposal/cost prediction/abstention are separated from numerical refinement and command acceptance.
2. **Method.** Action-complete development executions supervise a compact shared-success/action-latency predictor; the policy selects the eligible minimum-P95 entry and distinguishes command reject from OOD/uncertainty defer without relaxing the verifier.
3. **Evidence.** A frozen one-shot Panda/UR5e evaluation reports success, FEV, P50/P95/P99, deadline, reject/defer, trajectory behavior, sensitivity across training seeds, and prespecified adverse results.

## RQ 与论文主体的接口

| RQ | Development / test role | 应在后续正文出现的位置 |
|---|---|---|
| RQ1 heterogeneity | development-only bulk；5-repeat query-level empirical P95 oracle | Experimental Design + development diagnostics |
| RQ2 predictability | development policy-selection split；不属于 confirmatory test | Model/Policy Validation |
| RQ3 system gain | fresh frozen formal test；raw-call pooled P95/P99 + query-cluster bootstrap | Formal Results |

RQ1/RQ2 的五次 empirical-P95 与 RQ3 的 formal pooled raw-call P95 是不同 estimand。Introduction 只用它们分别回答“是否存在/能否预测”和“是否产生系统收益”，不进行数值等同。

## Canonical English text

完整、严格八段的英文 Introduction 已写入 [`paper/main.tex`](../paper/main.tex) 的 `\section{Introduction}`。该文本是当前唯一 canonical Introduction；本文件不维护第二份副本，以免后续出现漂移。

## 写作禁区

- 不在 Introduction 使用研发版本号、release 名或内部输出目录名；
- 不把 entry 写成 ground-truth difficulty class；
- 不宣称 action-specific terminal success heads；
- 不把 fixed cascade 写成每条查询一定跑满全部预算；
- 不把 OOD 当 reject，也不宣称 OOD detector 已解决；
- 不把三个 training seeds 当三个独立 test sets；
- 不写 uniformly faster、hard real-time、collision safe、physics validated 或 real-robot reliable；
- 不把 SciPy `trf_previous` 称为 TRAC-IK。
