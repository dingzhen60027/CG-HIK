# Elastic CR-IK μ=0.25：固定候选独立评估

**结论：本次未确认修正缺口代价的独立完成收益。** 相对公平解析μ=0，Panda的ΔTSR为-0.21 pp [-2.08, +1.46]，UR5e为+0.21 pp [-1.67, +2.50]；累计成本比分别为1.095× [1.054, 1.140]、1.126× [1.036, 1.211]。UR5e开发期完成增量未得到清楚的独立复现；Panda点估计略降。区间含零不是等效或无退化证明。

**本轮不再选参数。统一μ=0.25来自已观察开发结果，本报告只评估其在新轨迹上的表现。**

每台机器人160条新轨迹，四类各40条，每条300帧。六个设置均运行三遍，共5760次整轨迹运行、1728000帧。未筛掉失败或超时，未用新结果改变输入、需求参数或数值预算。旧论文及旧证据不变。

μ=0解析对照在初始合法对上直接保留完全相同的备用q和名义z；其他情况复用冻结的几何恢复。其目标没有改变；旧低效μ=0的时间不是本报告的比较分母。共享备用结果的决策等价测试不意味着随机TRAC整轨迹逐遍一致。

## 完整主结果

完成数与按时完成数按三遍分别列出（每遍分母160）；TSR/DTSR20先在UID内平均。P50/P95/P99为全部帧的描述性时延，包含失败与超时；累计时间为三遍均值。

| 机器人 | 方法 | 完成数 | 全帧≤20ms完成数 | TSR / DTSR20 | P50/P95/P99 ms | 每160条累计s | 加速度RMS rad/s² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| panda | Elastic CR-IK (mu=0.25) | 118/117/120 | 108/111/117 | 73.96% / 70.00% | 0.810/6.081/6.139 | 68.933 | 6.622 |
| panda | Same-core analytic zero reserve cost | 118/121/117 | 114/114/112 | 74.17% / 70.83% | 0.564/6.082/6.143 | 62.932 | 6.557 |
| panda | Two-step predictive | 121/122/123 | 118/116/120 | 76.25% / 73.75% | 3.761/6.266/6.349 | 196.676 | 5.999 |
| panda | TRAC-IK task 5 ms | 119/118/116 | 117/113/114 | 73.54% / 71.67% | 0.166/5.230/5.274 | 38.616 | 6.286 |
| panda | TRAC-IK task 20 ms | 121/119/120 | 118/116/119 | 75.00% / 73.54% | 0.166/20.262/20.301 | 121.891 | 6.477 |
| panda | Pink QP-IK | 124/124/124 | 123/121/123 | 77.50% / 76.46% | 0.997/1.050/1.154 | 44.315 | 6.803 |
| ur5e | Elastic CR-IK (mu=0.25) | 147/147/152 | 137/143/147 | 92.92% / 88.96% | 0.574/2.568/6.028 | 42.910 | 14.437 |
| ur5e | Same-core analytic zero reserve cost | 150/146/149 | 144/138/139 | 92.71% / 87.71% | 0.537/2.600/6.027 | 38.102 | 14.277 |
| ur5e | Two-step predictive | 130/133/129 | 124/131/127 | 81.67% / 79.58% | 3.543/6.173/6.241 | 180.638 | 8.710 |
| ur5e | TRAC-IK task 5 ms | 143/144/148 | 140/141/143 | 90.62% / 88.33% | 0.152/0.296/5.224 | 17.004 | 13.683 |
| ur5e | TRAC-IK task 20 ms | 151/152/150 | 146/150/144 | 94.38% / 91.67% | 0.152/0.245/20.244 | 31.597 | 13.848 |
| ur5e | Pink QP-IK | 110/110/110 | 107/108/109 | 68.75% / 67.50% | 0.627/0.755/0.868 | 31.171 | 17.550 |

## 1. UR5e开发增量是否重现？

相对公平μ=0：ΔTSR=+0.21 pp [-1.67, +2.50]；ΔDTSR20=+1.25 pp [-1.88, +4.58]。观察到完成增量；区间跨越或包含零，未建立稳定增益、等效性或无退化。

这是新UID、300帧轨迹上的独立检验，不是对旧开发trajectory_15的重复，也不据此更换μ。

## 2. Panda是增量、持平还是退化？

相对公平μ=0：ΔTSR=-0.21 pp [-2.08, +1.46]；ΔDTSR20=-0.83 pp [-3.75, +1.88]。观察到完成退化；区间跨越或包含零，未建立稳定增益、等效性或无退化。

两机器人分别分析；不以合并平均掩盖方向差异。

## 3. 修正缺口代价相对公平μ=0增加了什么？

| 机器人 | ΔTSR [95%区间] | ΔDTSR20 [95%区间] | 累计成本比 [95%区间] | 部分采用率 | 优化调用率 | 平均归一化干预 |
| --- | --- | --- | --- | --- | --- | --- |
| panda | -0.21 pp [-2.08, +1.46] | -0.83 pp [-3.75, +1.88] | 1.095× [1.054, 1.140] | 3.25% | 55.99% | 0.0108 |
| ur5e | +0.21 pp [-1.67, +2.50] | +1.25 pp [-1.88, +4.58] | 1.126× [1.036, 1.211] | 3.52% | 43.06% | 0.0101 |

这些增量比较把不必要的μ=0锥求解去掉了。已采用部分修正的q/z均按真实FK重新检查，实际目标与缺口用更新构型复算。部分采用率、γ或缺口下降本身不等于完成收益；以本节任务对照为准。

| 机器人 | 对照 | UID均值改善数 | UID均值损失数 | 稳定3/3 vs 0/3改善 | 稳定损失 |
| --- | --- | --- | --- | --- | --- |
| panda | Same-core analytic zero reserve cost | 5 | 5 | 0 | 0 |
| panda | Two-step predictive | 9 | 16 | 5 | 7 |
| panda | TRAC-IK task 5 ms | 9 | 7 | 2 | 1 |
| panda | TRAC-IK task 20 ms | 5 | 8 | 2 | 2 |
| panda | Pink QP-IK | 14 | 22 | 12 | 15 |
| ur5e | Same-core analytic zero reserve cost | 4 | 6 | 1 | 0 |
| ur5e | Two-step predictive | 32 | 5 | 12 | 1 |
| ur5e | TRAC-IK task 5 ms | 19 | 9 | 2 | 3 |
| ur5e | TRAC-IK task 20 ms | 10 | 12 | 0 | 3 |
| ur5e | Pink QP-IK | 45 | 7 | 39 | 1 |

局部成功实例：ur5e / smooth，UID `445202081cd300ea8c1d75bf7c20a0e1b35d41e13a7fcb0ffc9833f72f43c29b`，Elastic为3/3完成、解析μ=0为0/3。这是一个UID内的三次搜索重复，不是三条独立轨迹；它支持存在局部成功实例，不抵消其他UID的损失，也不建立总体收益。

全部UID、三遍完成率和首次失败输入见source-data，不只展示恢复实例。

## 4. 相对TRAC/Pink的完成—时限—成本折中

| 机器人 | 对照 | ΔTSR [95%区间] | ΔDTSR20 [95%区间] | 累计成本比 [95%区间] |
| --- | --- | --- | --- | --- |
| panda | TRAC-IK task 5 ms | +0.42 pp [-2.50, +3.33] | -1.67 pp [-5.21, +2.08] | 1.785× [1.625, 2.021] |
| panda | TRAC-IK task 20 ms | -1.04 pp [-3.96, +2.08] | -3.54 pp [-7.08, +0.21] | 0.566× [0.494, 0.681] |
| panda | Pink QP-IK | -3.54 pp [-10.21, +3.12] | -6.46 pp [-13.12, +0.62] | 1.556× [1.350, 1.785] |
| panda | Two-step predictive | -2.29 pp [-6.88, +2.29] | -3.75 pp [-8.54, +1.04] | 0.350× [0.311, 0.393] |
| ur5e | TRAC-IK task 5 ms | +2.29 pp [-1.46, +5.62] | +0.62 pp [-3.54, +4.58] | 2.523× [2.108, 3.112] |
| ur5e | TRAC-IK task 20 ms | -1.46 pp [-4.79, +1.46] | -2.71 pp [-6.46, +0.83] | 1.358× [0.995, 2.091] |
| ur5e | Pink QP-IK | +24.17 pp [+17.49, +30.83] | +21.46 pp [+14.58, +28.33] | 1.377× [1.261, 1.508] |
| ur5e | Two-step predictive | +11.25 pp [+6.46, +16.25] | +9.38 pp [+4.38, +14.79] | 0.238× [0.219, 0.258] |

本次尚未建立新增修正缺口代价的实用必要性。UR5e相对Pink和原普通双步预测有明确的样本收益，但公平解析μ=0保留了近似的完成均值，因此这些对照不能单独归因于修正目标。相对TRAC-IK 20ms，UR5e的完成与DTSR20均值更低、累计成本均值更高；Panda相对Pink的完成与时限点估计也不利。各区间和尾时延仍完整保留，不据此宣称全指标支配或等效。

是否值得增加计算必须同时看本表的任务与时限差异，而不是仅与昂贵旧实现比速度。20ms是软件评价deadline，不是硬实时证明；算法总时间包含备用求解、预测/需求、初始对、锥构造/求解和非线性验收。

## 全部类别与实际误差

| 机器人 | 类别 | 方法 | TSR / DTSR20 | 累计s | P95 ms | 加速度RMS |
| --- | --- | --- | --- | --- | --- | --- |
| panda | curvature-speed variation | Elastic CR-IK (mu=0.25) | 91.67% / 88.33% | 11.814 | 1.725 | 9.909 |
| panda | joint_limit_return | Elastic CR-IK (mu=0.25) | 67.50% / 64.17% | 22.014 | 6.093 | 4.900 |
| panda | near_singular | Elastic CR-IK (mu=0.25) | 65.83% / 60.83% | 17.378 | 6.087 | 7.713 |
| panda | smooth | Elastic CR-IK (mu=0.25) | 70.83% / 66.67% | 17.727 | 6.082 | 3.967 |
| panda | curvature-speed variation | Same-core analytic zero reserve cost | 92.50% / 89.17% | 9.682 | 1.728 | 9.821 |
| panda | joint_limit_return | Same-core analytic zero reserve cost | 65.00% / 60.83% | 21.898 | 6.100 | 4.555 |
| panda | near_singular | Same-core analytic zero reserve cost | 68.33% / 64.17% | 14.893 | 6.083 | 7.781 |
| panda | smooth | Same-core analytic zero reserve cost | 70.83% / 69.17% | 16.459 | 6.082 | 4.070 |
| panda | curvature-speed variation | Pink QP-IK | 90.00% / 89.17% | 10.958 | 1.048 | 9.787 |
| panda | joint_limit_return | Pink QP-IK | 65.00% / 64.17% | 11.048 | 1.050 | 4.940 |
| panda | near_singular | Pink QP-IK | 77.50% / 76.67% | 11.035 | 1.048 | 8.495 |
| panda | smooth | Pink QP-IK | 77.50% / 75.83% | 11.274 | 1.053 | 3.990 |
| panda | curvature-speed variation | TRAC-IK task 20 ms | 94.17% / 91.67% | 8.507 | 0.264 | 9.605 |
| panda | joint_limit_return | TRAC-IK task 20 ms | 71.67% / 71.67% | 45.465 | 20.273 | 4.470 |
| panda | near_singular | TRAC-IK task 20 ms | 64.17% / 61.67% | 32.347 | 20.264 | 7.854 |
| panda | smooth | TRAC-IK task 20 ms | 70.00% / 69.17% | 35.572 | 20.265 | 3.979 |
| panda | curvature-speed variation | TRAC-IK task 5 ms | 91.67% / 89.17% | 4.345 | 0.292 | 9.557 |
| panda | joint_limit_return | TRAC-IK task 5 ms | 69.17% / 67.50% | 13.942 | 5.243 | 4.272 |
| panda | near_singular | TRAC-IK task 5 ms | 63.33% / 63.33% | 9.696 | 5.229 | 7.365 |
| panda | smooth | TRAC-IK task 5 ms | 70.00% / 66.67% | 10.633 | 5.233 | 3.950 |
| panda | curvature-speed variation | Two-step predictive | 95.00% / 94.17% | 45.889 | 4.280 | 9.370 |
| panda | joint_limit_return | Two-step predictive | 67.50% / 65.00% | 52.340 | 6.289 | 3.819 |
| panda | near_singular | Two-step predictive | 71.67% / 70.00% | 48.037 | 6.259 | 6.886 |
| panda | smooth | Two-step predictive | 70.83% / 65.83% | 50.410 | 6.271 | 3.919 |
| ur5e | curvature-speed variation | Elastic CR-IK (mu=0.25) | 94.17% / 90.83% | 11.915 | 2.527 | 18.623 |
| ur5e | joint_limit_return | Elastic CR-IK (mu=0.25) | 89.17% / 85.00% | 11.450 | 2.884 | 12.062 |
| ur5e | near_singular | Elastic CR-IK (mu=0.25) | 93.33% / 90.83% | 9.414 | 1.510 | 13.415 |
| ur5e | smooth | Elastic CR-IK (mu=0.25) | 95.00% / 89.17% | 10.130 | 1.668 | 13.648 |
| ur5e | curvature-speed variation | Same-core analytic zero reserve cost | 94.17% / 89.17% | 9.761 | 2.550 | 18.252 |
| ur5e | joint_limit_return | Same-core analytic zero reserve cost | 90.00% / 83.33% | 10.683 | 3.495 | 11.886 |
| ur5e | near_singular | Same-core analytic zero reserve cost | 92.50% / 89.17% | 9.005 | 1.603 | 13.373 |
| ur5e | smooth | Same-core analytic zero reserve cost | 94.17% / 89.17% | 8.653 | 1.470 | 13.597 |
| ur5e | curvature-speed variation | Pink QP-IK | 52.50% / 52.50% | 7.913 | 0.766 | 19.220 |
| ur5e | joint_limit_return | Pink QP-IK | 82.50% / 80.83% | 7.744 | 0.746 | 16.637 |
| ur5e | near_singular | Pink QP-IK | 82.50% / 81.67% | 7.633 | 0.689 | 16.619 |
| ur5e | smooth | Pink QP-IK | 57.50% / 55.00% | 7.881 | 0.757 | 17.724 |
| ur5e | curvature-speed variation | TRAC-IK task 20 ms | 95.83% / 92.50% | 3.490 | 0.218 | 18.403 |
| ur5e | joint_limit_return | TRAC-IK task 20 ms | 95.83% / 90.83% | 9.633 | 0.256 | 11.307 |
| ur5e | near_singular | TRAC-IK task 20 ms | 94.17% / 93.33% | 6.417 | 0.243 | 12.659 |
| ur5e | smooth | TRAC-IK task 20 ms | 91.67% / 90.00% | 12.057 | 0.413 | 13.022 |
| ur5e | curvature-speed variation | TRAC-IK task 5 ms | 95.00% / 90.83% | 2.638 | 0.220 | 18.440 |
| ur5e | joint_limit_return | TRAC-IK task 5 ms | 95.00% / 93.33% | 4.116 | 0.261 | 11.144 |
| ur5e | near_singular | TRAC-IK task 5 ms | 88.33% / 86.67% | 4.467 | 0.356 | 12.357 |
| ur5e | smooth | TRAC-IK task 5 ms | 84.17% / 82.50% | 5.783 | 5.194 | 12.791 |
| ur5e | curvature-speed variation | Two-step predictive | 83.33% / 82.50% | 43.588 | 3.994 | 14.932 |
| ur5e | joint_limit_return | Two-step predictive | 73.33% / 73.33% | 47.407 | 6.195 | 5.757 |
| ur5e | near_singular | Two-step predictive | 89.17% / 84.17% | 43.337 | 3.960 | 6.503 |
| ur5e | smooth | Two-step predictive | 80.83% / 78.33% | 46.307 | 6.185 | 7.649 |

| 机器人 | 方法 | 接受位置误差P95/max mm | 接受姿态误差P95/max deg | 接近>90%容差比例 |
| --- | --- | --- | --- | --- |
| panda | Elastic CR-IK (mu=0.25) | 0.5021/1.0000 | 0.0911/0.4986 | 0.28% |
| panda | Same-core analytic zero reserve cost | 0.4416/0.9797 | 0.0476/0.4988 | 0.20% |
| panda | Pink QP-IK | 0.2789/0.9996 | 0.0353/0.4948 | 0.15% |
| panda | TRAC-IK task 20 ms | 0.4815/0.9976 | 0.0446/0.4900 | 0.22% |
| panda | TRAC-IK task 5 ms | 0.4701/0.9946 | 0.0419/0.4911 | 0.15% |
| panda | Two-step predictive | 0.2275/1.0000 | 0.1263/0.5000 | 0.23% |
| ur5e | Elastic CR-IK (mu=0.25) | 0.5067/1.0000 | 0.1266/0.5000 | 0.20% |
| ur5e | Same-core analytic zero reserve cost | 0.4576/0.9897 | 0.0575/0.4906 | 0.07% |
| ur5e | Pink QP-IK | 0.2874/0.9948 | 0.0323/0.4776 | 0.15% |
| ur5e | TRAC-IK task 20 ms | 0.4541/0.9953 | 0.0420/0.4791 | 0.10% |
| ur5e | TRAC-IK task 5 ms | 0.4490/0.9940 | 0.0398/0.4759 | 0.09% |
| ur5e | Two-step predictive | 0.2019/1.0000 | 0.1514/0.5000 | 0.14% |

接受命令合同违约：0。误差统计以接受帧为分母；失败帧仍进入成功率和全部成本。

## 5. 已建立与尚未建立的主张

已建立的是固定候选在这两组完整新轨迹上的实际命令、完成、deadline与成本比较，以及每个记录的解析直接返回或部分改善决策。哪些机器人有正向均值，按上面的配对结果逐一判断。尚未建立μ最优、普遍优越、无退化、非线性未来可行性保证或硬实时能力。局部修正需求是经验估计，不是概率保证；单帧合法也不保证下一帧一定存在合法延拓。

## 统计、来源与停止点

独立单位是每机器人160个trajectory UID，先平均三遍，再在四个family内配对bootstrap 4000次。95%区间是描述性、未多重校正的百分位区间；不把帧或搜索重复当新增独立样本，也不因区间含零声称等效。主比较预先固定，未延长采样或删除不利类别。

候选、配置、数值代码与全部身份在运行前提交于`67645112`；每次运行的实际Git SHA、CPU分配、依赖与全部原始记录hash均保存。只读复核覆盖所有帧、反馈状态、因果需求、名义q/z以及全部已记录非线性trial，不重新运行IK。

- [完整source-data](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/source_data.json)
- [配对区间](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/paired_comparisons.csv)
- [恢复/损失UID](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/gained_lost_uids.csv)
- [首次失败输入](../outputs/correction_reserve_ik/elastic_locked_evaluation/reports/first_failure_inputs.csv)
- [原始记录与协议](../outputs/correction_reserve_ik/elastic_locked_evaluation/)

**本轮结束：保留固定结果，不自动搜索μ、增加模块、启动另一轮评估或重写论文。**
