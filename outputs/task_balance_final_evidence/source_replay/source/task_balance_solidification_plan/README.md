# Task Balance GN：有限补强与写作计划

建立日期：2026-09-16。
基线：dingzhen60027/CG-HIK，codex/hierarchical-v5，79d3d9324647a51876dca2b2ec612d94599b6559。

## 当前交付身份

本包只包含已研究来源后的执行计划，尚未下载DROID实际episode、运行新IK、执行新敏感性或修改GitHub仓库。

- CODEX_TASK.md：两项有限证据补强及验收交付。
- PAPER_WRITING_PLAN.md：验收后的完整写作安排。

现有方法与对照无需重做。新增工作仅回答：
1. 真实来源的请求是否会触发所研究的约束冲突？
2. 固定算法在小范围精度控制与任务要求变化下是否表现一致？

不是寻找更有利样本，不改变目标流速度制造优势，不将未测到的应用增量预写进论文。

## 本次核实的官方来源

DROID官方文档：
https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md
- 官方100-episode调试子集；
- action_dict的commanded Cartesian/joint字段与observation字段；
- raw trajectory.h5为低维记录。

DROID机器人接口：
https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py
- commanded Cartesian目标、robot_state与实际更新关系；
- 状态和命令姿态表示需按同版本变换处理。

DROID环境：
https://github.com/droid-dataset/droid/blob/main/droid/robot_env.py
- 本次读取的代码配置control_hz=15。
- 这不是宣称所有历史episode都用同一版本，执行前必须核实实际数据身份。

原始研究来源：
RangedIK, ICRA 2023:
https://graphics.cs.wisc.edu/Papers/2023/WPRG23/

Zheng, Ma, Xue (2024 author manuscript), relative inexactness precedent:
https://arxiv.org/html/2304.12522v2

DROID project:
https://droid-dataset.github.io/

所有主方法数值事实以基线仓库报告和原始记录为准；此包不产生新性能结论。
