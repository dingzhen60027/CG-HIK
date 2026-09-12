# 当前帧 IK 恢复：已执行的五输入检查

## 实际结果

本包不是新 IK 算法，也不是新论文实验。它直接使用仓库报告中可核对的失败输入，
运行两种标准当前帧数值求解，保存真正算出的关节配置。

- 输入来源：CG-HIK `8e754127bb56ec6054ef197f7f1b4e2324611b91` 的
  `outputs/task_recourse/numerical_completion_development/reports/first_failure_inputs.csv`。
- 选择：所读取 CSV 的前五条 Panda 输入，来自三条轨迹。不是随机测试集。
- 保持每行自己的 `previous_q`、目标、20 ms 名义间隔、关节范围与速度上限。
- 只优化当前七维关节变量；没有后续目标、轨迹 witness seed 或场景分支。
- 标准有界 TRF 和标准 SLSQP minimax 都在五个输入中的四个找到合法当前配置。
- TRF 在四个恢复输入上的本地中位耗时为 1.25–3.61 ms。
- minimax 没有增加恢复数，且大多更慢。因此不建议仅凭此次检查采用新目标。
- 剩余输入未找到合格解；没有全局不可行证明。

## 不得过度解读

1. **没有运行仓库原始 verifier、TRAC-IK/Pink 二进制或完整轨迹。**
2. 采用公开 Panda URDF 的运动学参数重建链，五条历史位置/姿态残差复算差异不超过
   约 1.1e-16（分别按米和弧度）。这是对这五条记录的核对，不是所有配置的等价证明。
3. 新解经过第二个齐次变换 FK 实现和相同数值接受条件复核。Codex 仍需在原仓库模型和
   原 verifier 下核验关节向量，才能将其称为项目内已验收命令。
4. 11遍是确定性数值计算的计时重复，不是55个独立输入。
5. 本地硬件/依赖与用户服务器不同。禁止与历史时延直接相减宣称算法提速。
6. 恢复当前一帧，不等于恢复整条轨迹，不等于证明科研创新。
7. 两个方法的早停检查使用0.99999倍位姿容差，最终验收使用完整原数值容差。
   关节范围和速度没有放宽。单例脚本没有严格墙钟中断，50次函数评价不等于20 ms保证。

## 运行

需要 NumPy 和支持 least_squares callback 的 SciPy（1.16或更新版本）。
不需要 ROS、Pinocchio、模型权重或联网。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python repeat_probe.py
```

- `probe.py`：可运行求解代码、输入和公开模型参数。
- `recorded_inputs.json`：五个实际输入和对应历史残差。
- `probe_results.json`：首次执行结果、模型残差核对和解析Jacobian数值检查。
- `repeated_results.json`：交错的11遍计时、全部关节结果、第二实现验收和环境。
- `CODEX_NEXT_TASK.md`：下一步执行要求。原证据不变，不创建新研究分支。

## 来源

```text
输入：https://github.com/dingzhen60027/CG-HIK/blob/8e754127bb56ec6054ef197f7f1b4e2324611b91/outputs/task_recourse/numerical_completion_development/reports/first_failure_inputs.csv
模型：https://github.com/justagist/franka_panda_description/blob/master/robots/panda_arm.urdf
模型源blob：407642f8156a754b7e9c74bc4bab4209fd330d06
标准TRF：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
```

运行日期：2026-09-12。容器无法联网下载或运行用户的原环境；来源通过GitHub connector读取后
转录。未对远端仓库执行写操作。
