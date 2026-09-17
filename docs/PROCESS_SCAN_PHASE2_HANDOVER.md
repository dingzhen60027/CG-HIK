# Phase 2 交接

当前阶段止于开发结果，不自动执行正式96场景或写论文。

## 验收结论

432次规划、121次预指定MuJoCo力矩执行已经完成。
Proposed质量完成为Panda 9/12、UR5e 12/12；
Panda相对B1恢复4个UID、损失1个。
距离图A1在Panda达到10/12。
共同成功场景中，相对B1的换行和周期中位降幅为
0.059%与0.117%。
相对B2-common的周期比中位为1.068。
开发Gate未通过；完整目标和消融结论见Gate报告。
本轮停止，正式测试不启动，不因局部计划时间确有下降而改写整体判断。

数值/输入验收见`reports/verification.json`，回归测试见
`provenance/verification_tests.xml`，图与视频复算及视觉检查见`reports/figures/QA.md`。

## 交付索引

- 方法：`docs/PROCESS_SCAN_PHASE2_METHOD.md`。
- 结果与Gate：`docs/PROCESS_SCAN_PHASE2_RESULTS.md`、`docs/PROCESS_SCAN_PHASE2_GATE.md`。
- 根目录：`outputs/process_scan/phase2_development/`。
- 输入/节点：`inputs/*/identity.json`、`candidate_library.json.gz`；冻结seal列完整hash。
- 图/全部边下界/选择：`graphs/<scene>/r<repeat>/graph.json.gz`与两代价的`selection.json.gz`。
- 共同初值：图目录中的`time/path.npz`，B2-common/Proposed/A2逐字节复用。
- 规划：`runs/<scene>/<method>_r<repeat>/`，原始优化器/密集检查/TOPPRA/可用时刻完整保留。
- 实际状态：`execution.h5`，1ms反馈；`scan_samples.npz`全部81射线、质量mask。
- 主表/全场景/家族/配对区间/失败/图/逐次更新：`reports/*.csv`。
- 图：`reports/figures/`，真实记录生成，可编辑SVG/PDF及预览PNG。
- 固定视频索引：`reports/videos/index.json`，placement0/u/全部机器人家族方法，repeat0。
- 可复算入口：`scripts/run_process_scan_phase2.py`；`report/render/verify`只读测量，不重跑物理。

历史Task Balance/GN、运动学/验收、Phase1/1.5所有原始结果与论文保持不变。
新Phase2优化内核由已验证B2方程逐行复用，只通过变量边界固定局部范围；
并没有更换损失、引入预测、学习或BoundMPC适配。

参考库获取成本是历史输入准备，不包含在本轮在线规划费用；候选复核和图构建
费用已经完整计入每个消费者。旧方法计时不直接搬进新主表。失败完成时间为null。
每个CAD×放置只有两个方向，聚类样本量有限；物理证据仅共同控制器下的仿真。

请验收全样本完成、共同成功时间、候选/图与B2共享初值、失败和成本后再决定方向。
本轮没有预授权的后续优化、场景扩展或正式测试。
