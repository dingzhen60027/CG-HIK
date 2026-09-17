# 本轮已核对来源与方法边界

本包是2026-09-16的规划修订，不是实验结果。

1. 本项目分支HEAD：bf9ccc4e0314c0d289101d89ff4e40577de392f9。读取了task_balance_gn.py、TASK_BALANCE_FINAL_SCOPE.md及当前分支API。
   https://github.com/dingzhen60027/CG-HIK/tree/codex/hierarchical-v5
2. TOPP-RA：Pham & Pham, A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis. 作者预印本 https://arxiv.org/abs/1707.07239 。本轮核对摘要；给定几何路径的时间参数化，不是关节构型路径优化。官方实现使用固定版本，先运行示例。
3. Fried & Paternain, A Bi-Level Optimization Approach to Joint Trajectory Optimization for Redundant Manipulators, 2024作者预印本 https://arxiv.org/abs/2412.07859 。本轮核对作者摘要：联合路径/速度、下层凸问题、上层值函数方向导数已有。B2是同任务匹配实现而非已确认的官方复现。
4. BoundMPC：Oelerich et al., IJRR 2025. https://doi.org/10.1177/02783649241309354 。本轮核对出版页的任务和方法说明、以及官方 https://github.com/TU-Wien-ACIN-CDS/BoundMPC 的README。开源示例实时性能依赖差异不能当作方法失败。
5. 原MASTER_PLAN.md、CODEX_PHASE1_BASELINES.md是用户给定的前一规划文件。本包保留其任务、场景、强基线、未来测试框架，并显式加入TB初始化/G桥接以及几何接口修订；不将先前计划当作已有性能。

上一版较广参考清单保存在previous_reference_notes.md，状态仍是此前整理记录；本轮没有声称全部重新阅读全文。Codex实现前按其中R1–R4/R6读取所用方法与接口。
