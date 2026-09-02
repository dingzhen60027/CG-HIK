# Literature Map for the CG-HIK Introduction

检索与核验截止：2026-09-02。去重后参考文献总数为 **40**。正式出版条目优先按 DOI/Crossref 与出版社页面核验；只有 arXiv 记录的条目明确标为 preprint。本文不以 ResearchGate 作为唯一书目来源。

## 分布

| 组别 | 数量 | 说明 |
|---|---:|---|
| 2024 | 4 | CppFlow、variable-step IK、IKSPARK、learning-by-example IK |
| 2025 | 11 | 几何/解析、生成/结构化学习、执行代价与运动生成 |
| 2026 | 9 | viability、seed selection、set/sequential/generative IK、混合求解与奇异跨越 |
| **2024--2026** | **24** | 占总文献 60%，满足近期文献主导要求 |
| 奠基/成熟文献 | 10 | DLS、冗余控制、solver portfolio、motion synthesis、algorithm selection、calibration/uncertainty |
| 其他过渡与软件文献 | 6 | IKFlow/CycleIK 及可复现软件栈 |
| **总计** | **40** | DOI 优先去重 |

四条综述链并非互斥：一篇论文可以同时支撑两条链，但在 Introduction 中只承担必要的局部论断。

## A. 数值、解析与约束 IK

| Key | 文献与状态 | 主要贡献 | Introduction 作用与边界 | 核验 |
|---|---|---|---|---|
| `wampler1986dls` | Wampler, 1986, IEEE TSMC | DLS 形式化 | P2：局部病态下的数值稳定基础；不是本文创新 | [DOI](https://doi.org/10.1109/TSMC.1986.289285) |
| `nakamura1986singularity` | Nakamura & Hanafusa, 1986, ASME JDSMC | singularity-robust inverse | P2：奇异鲁棒性是独立数值要求 | [DOI](https://doi.org/10.1115/1.3143764) |
| `siciliano1990tutorial` | Siciliano, 1990, JIRS | 冗余机械臂运动学控制教程 | P2：Jacobian、null-space 与冗余解析框架 | [DOI](https://doi.org/10.1007/BF00126069) |
| `chiaverini1994review` | Chiaverini et al., 1994, IEEE TCST | 加权/自适应 DLS 与工业实验 | P2：阻尼是成熟基础，不作为新贡献 | [DOI](https://doi.org/10.1109/87.294335) |
| `beeson2015tracik` | Beeson & Ames, 2015, Humanoids | 并发 Jacobian/SQP portfolio | P2/P5：互补路线减少 false failure；并发会让查询支付多路线成本 | [DOI](https://doi.org/10.1109/HUMANOIDS.2015.7363472) |
| `rakita2018relaxedik` | Rakita et al., 2018, RSS | 多目标实时 motion synthesis | P2：位姿、连续性、奇异与碰撞可联合建模 | [DOI](https://doi.org/10.15607/RSS.2018.XIV.043) |
| `colan2024variablestep` | Colan et al., 2024, IEEE Access | variable steps 与重启 | P2/P5：求解率和成本依赖数值策略 | [DOI](https://doi.org/10.1109/ACCESS.2024.3418206) |
| `wu2024ikspark` | Wu & Tron, first posted 2024, arXiv preprint; v2 2026 | SDP relaxation、rank minimization、obstacle-aware IK | P2：约束 IK 的凸松弛路线；不得写成同行评审定论 | [arXiv](https://arxiv.org/abs/2403.12235) |
| `elias2025ikgeo` | Elias & Wen, 2025, Mechanism and Machine Theory | 几何子问题分解 | P2：可利用结构下的快速、精确解析/几何 IK | [DOI](https://doi.org/10.1016/j.mechmachtheory.2025.105971) |
| `ostermeier2025eaik` | Ostermeier et al., 2025, IEEE RA-L | 自动几何分解生成解析 IK | P2：现代解析 IK 仍然强，但依赖可分解结构 | [DOI](https://doi.org/10.1109/LRA.2025.3597897) |
| `lopezcustodio2025geofik` | Lopez-Custodio et al., first posted 2025, arXiv preprint | Franka screw-theoretic geometric IK | P2：冗余参数与奇异结构的专用解析处理；仅 preprint | [arXiv](https://arxiv.org/abs/2503.03992) |
| `zhang2026viability` | Zhang & Kikuuwe, 2026, JIRS | joint/velocity/acceleration/collision viability QP | P2/P7：command admissibility 可显式检查；本文实验不包含 collision | [DOI](https://doi.org/10.1007/s10846-025-02335-z) |
| `boschi2026singularity` | Boschi et al., 2026, IEEE RA-L | uniqueness-domain branch transition | P1/P2：阻尼稳定与拓扑分支跨越不是同一问题 | [DOI](https://doi.org/10.1109/LRA.2026.3653295) |
| `yasutake2026hjcdik` | Yasutake et al., accepted IROS 2026; arXiv preprint | batched coordinate-descent initialization + Jacobian polishing | P2/P4：现代 GPU hybrid numerical portfolio；会议页码/DOI 待 proceedings | [arXiv](https://arxiv.org/abs/2510.07514) |

## B. 学习型多解、结构与历史条件 IK

| Key | 文献与状态 | 主要贡献 | Introduction 作用与边界 | 核验 |
|---|---|---|---|---|
| `ames2022ikflow` | Ames et al., 2022, IEEE RA-L | conditional flow 生成多样 IK 解 | P3/P4：从单点回归转向解分布；可再数值精修 | [DOI](https://doi.org/10.1109/LRA.2022.3181374) |
| `habekost2023cycleik` | Habekost et al., 2023, ICANN | FK-consistent neural IK 与优化组合 | P3/P4：物理一致性损失和 downstream refinement | [DOI](https://doi.org/10.1007/978-3-031-44207-0_38) |
| `limoyo2025ggik` | Limoyo et al., 2025, IEEE T-RO | graphical/equivariant generative IK | P3：多解与跨运动链结构泛化 | [DOI](https://doi.org/10.1109/TRO.2024.3521862) |
| `demby2024learning` | Demby's et al., 2024, IROS | learning-by-example pose/joint conditioning | P3：reference state 提供 pose-only 映射缺失的局部上下文 | [DOI](https://doi.org/10.1109/IROS58592.2024.10802048) |
| `diprasetya2025kinenn` | Diprasetya et al., 2025, RCIM | kinematics-informed invertible modular network | P3：结构参数共享与迁移；不是 verified command generator | [DOI](https://doi.org/10.1016/j.rcim.2024.102945) |
| `go2025emiknet` | Go & Moon, 2025, IEEE Access | multiple-instance, multi-end-effector, multi-solution IK | P3：多解/多末端输出扩展 | [DOI](https://doi.org/10.1109/ACCESS.2025.3539022) |
| `zhang2025ikdiffuser` | Zhang & Jiao, first posted 2025, arXiv preprint | kinematic-tree conditional diffusion | P3/P4：任意树、多末端与 optimization warm start；under review | [arXiv](https://arxiv.org/abs/2506.13087) |
| `nguyen2026setik` | Nguyen et al., 2026, IEEE RA-L | permutation-invariant set prediction | P3：离散 IK 分支是无序集合 | [DOI](https://doi.org/10.1109/LRA.2026.3703280) |
| `nguyen2026sequential` | Nguyen et al., 2026, RAS, online/issue forthcoming | previous-state sequential joint update | P1/P3：从 pose-only 映射到历史条件连续更新 | [DOI](https://doi.org/10.1016/j.robot.2026.105669) |
| `yang2026mimik` | Yang et al., 2026, arXiv preprint | teleoperation prior + current-state joint increments | P3：IK 作为局部动作解码器；不得写成正式同行评审结论 | [arXiv](https://arxiv.org/abs/2606.15148) |

## C. Learned proposal + numerical refinement

| Key | 文献与状态 | 主要贡献 | Introduction 作用与边界 | 核验 |
|---|---|---|---|---|
| `morgan2024cppflow` | Morgan et al., 2024, ICRA | generative path proposals + optimization/validity checks | P3/P4：路径级 proposal/refinement 分工 | [DOI](https://doi.org/10.1109/ICRA57147.2024.10611724) |
| `huang2025diffusionseeder` | Huang et al., CoRL 2024 proceedings published 2025 | diffusion trajectory seeds + motion optimization | P4：学习 proposal 可降低下游 optimization 成本；属于 motion planning 邻域 | [PMLR](https://proceedings.mlr.press/v270/huang25f.html) |
| `carvalho2025mpd` | Carvalho et al., 2025, IEEE T-RO | diffusion trajectory prior + cost guidance | P4：生成先验与可微代价结合；不是 IK 专用 router | [DOI](https://doi.org/10.1109/TRO.2025.3593109) |
| `tang2025etaik` | Tang et al., 2025, IROS | execution-time-aware redundancy selection | P4/P5：IK 解的价值可由 downstream execution cost 定义 | [DOI](https://doi.org/10.1109/IROS60139.2025.11247583) |
| `jayabalan2026hybrid` | Jayabalan et al., 2026, Machines | ANFIS approximation + numerical refinement | P4：软计算提议、经典方法收尾 | [DOI](https://doi.org/10.3390/machines14030292) |
| `jlidi2026xgnn` | Jlidi et al., 2026, Electronics | graph warm start + DLS | P4：learned initializer 应以 downstream convergence 衡量；不支持 reject/defer claim | [DOI](https://doi.org/10.3390/electronics15143071) |

## D. 自适应计算、算法选择与不确定性

| Key | 文献与状态 | 主要贡献 | Introduction 作用与边界 | 核验 |
|---|---|---|---|---|
| `rice1976selection` | Rice, 1976, Advances in Computers | instance-to-algorithm selection framework | P5/P6：将逐查询入口选择放入 algorithm-selection 视角 | [DOI](https://doi.org/10.1016/S0065-2458(08)60520-3) |
| `bischl2016aslib` | Bischl et al., 2016, Artificial Intelligence | action-complete instance--algorithm performance matrices | P5/P6：同一 query 的多入口实际结果是训练 router 的充分监督形式 | [DOI](https://doi.org/10.1016/j.artint.2016.04.003) |
| `yuan2025iksel` | Yuan et al., 2026, IEEE T-ASE（BibTeX key 为历史命名） | seed retrieval, ranking, reselection | P5：进入哪个收敛盆地与何时重选决定时间--成功率权衡 | [DOI](https://doi.org/10.1109/TASE.2026.3659225) |
| `xu2025faildetect` | Xu et al., 2025, RSS | uncertainty-aware runtime failure detection | P7：不确定性可触发 conservative fallback；不是本文 OOD 性能证据 | [DOI](https://doi.org/10.15607/RSS.2025.XXI.073) |
| `guo2017calibration` | Guo et al., 2017, ICML | neural probability calibration | 方法背景：冻结 Platt calibration；不用于宣称 OOD detection | [PMLR](https://proceedings.mlr.press/v70/guo17a.html) |
| `lakshminarayanan2017ensembles` | Lakshminarayanan et al., 2017, NeurIPS | deep ensembles uncertainty | 方法背景：历史候选 ensemble；不等同确定性保证 | [NeurIPS](https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) |

## E. 可复现软件栈

| Key | 软件/论文 | 本项目中的作用 | 核验 |
|---|---|---|---|
| `pedregosa2011sklearn` | scikit-learn | calibration/历史模型实现 | [JMLR](https://www.jmlr.org/papers/v12/pedregosa11a.html) |
| `paszke2019pytorch` | PyTorch | compact predictor 与 exact TorchScript backend | [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| `virtanen2020scipy` | SciPy | TRF comparator 与数值工具 | [DOI](https://doi.org/10.1038/s41592-019-0686-2) |
| `carpentier2019pinocchio` | Pinocchio | 精确 URDF 运动学、Jacobian 与 verifier 基础 | [DOI](https://doi.org/10.1109/SII.2019.8700380) |

## 引用纪律

1. IKSPARK、GeoFIK、IKDiffuser、MimicIK 是 arXiv-only；HJCD-IK 为 arXiv + accepted IROS 2026。正文必须保留状态限定。
2. `nguyen2026sequential` 的 DOI 已激活，但期刊 issue 日期为 2026-11；当前写作中标为 online / forthcoming issue。
3. `yuan2025iksel` 已正式发表于 2026 T-ASE；保留旧 key 仅为避免破坏既有正文引用。
4. CppFlow 的 DOI、题名和会议状态已核验，但不同数据库页码冲突，BibTeX 暂不写 pages，投稿前从 IEEE PDF 首页人工核验。
5. 在本次 40 篇集合中，没有识别到同时采用 action-complete pathway supervision、verified-success constraint、P95 entry selection、command reject、OOD/uncertainty defer 和 deterministic verifier 的既有 IK 系统。正文只能写 “Among the reviewed systems, we did not identify ...”，不能写无边界的 “the first”。
