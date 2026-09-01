# CG-HIK 运行与复核手册

本文只回答三个问题：当前能做什么、结果在哪里、怎样复核。研究故事和结果解释见 [RESEARCH.md](RESEARCH.md)。

## 1. 当前状态

- v2、v3、v4 的训练、开发和正式测试均已结束；
- 当前没有需要自动继续的训练进程；
- 论文主稿仍是旧版本，下一步是写作；
- 冻结 outputs 保持原路径，因为配置、manifest 和审计哈希引用这些路径。

## 2. 环境

项目根目录：`/home/eric/wjg/btry`

推荐 Python：

```bash
export CONFIK_PYTHON=/home/eric/anaconda3/envs/isaaclab_3/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

环境名虽然是 `isaaclab_3`，当前正式实验使用 Pinocchio/URDF 运动学链路，并不是 Isaac Lab 动力学仿真。

## 3. 日常检查

运行测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$CONFIK_PYTHON" -m pytest -q
```

复核 v3 冻结测试：

```bash
"$CONFIK_PYTHON" scripts/audit_test_v3_locked.py --output /tmp/cghik_test_v3_audit.json
```

复核 v4 六组测量、aggregation repair 和 attestation：

```bash
"$CONFIK_PYTHON" scripts/audit_test_v4_aggregation_repair.py
```

v4 审计应以退出码 0 和顶层 `verdict: PASS` 结束。该命令是只读复核，不会重新执行 query、solver、model 或 bootstrap。

## 4. 结果导航

| 想确认的内容 | 首选入口 |
|---|---|
| 项目结论和论文故事 | `docs/RESEARCH.md` |
| v3 正式汇总 | `outputs/test_v3_aggregate/aggregate_summary_v3.json` |
| v4 正式汇总 | `outputs/test_v4_aggregate_repair_v1/aggregate_summary_v4.json` |
| v4 正式 gate | `outputs/test_v4_aggregate_repair_v1/paper_gate_v4.json` |
| v4 Holm 结果 | `outputs/test_v4_aggregate_repair_v1/joint_holm_v4.json` |
| 论文源文件 | `paper_mdpi_machines_v3/` |

先看 `RESEARCH.md`。只有核对具体数字、split 或哈希时才进入 outputs 或运行对应审计脚本。

## 5. 写论文的工作流

1. 以 `docs/RESEARCH.md` 的主线和 claim 边界为大纲；
2. 从冻结的 v3/v4 aggregate 提取数字；
3. 保留 v2 eager 失败、v3 exact 部署和 v4 Panda gate 失败；
4. 更新 Abstract、Methods、Experimental Design、Results、Discussion 和 Conclusion；
5. 重新生成图表并记录来源；
6. 编译 PDF，逐页检查；
7. 投稿前再核验 MDPI 模板、期刊分区、费用和政策。

论文包的具体命令见 `paper_mdpi_machines_v3/README.md`。

## 6. 后续研究规则

正式 `test_v4` 已经看过结果，因此不能再用于模型、阈值、solver budget 或 claim gate 的选择。若要继续优化 OOD、defer、碰撞或物理闭环，请创建 v5：

1. 使用新的 training/validation 数据开发；
2. 在开发结束后冻结模型、配置和门；
3. 再生成新的独立 test；
4. 将新结果写入新的 output namespace。

这条规则是为了维持测试集的一次性解释，不意味着现有代码或证据无法使用。v2–v4 可继续作为固定基线和历史比较。

## 7. 路径约定

- `outputs/` 不做美化搬迁，避免破坏 manifest 和 provenance；
- 根 README、`docs/RESEARCH.md`、本文和论文包 README 是全部活动文档；
- 冻结证据保留在 `outputs/`，复核逻辑保留在 `scripts/audit_*.py`；
- 新说明应合并到现有入口，不再新建平行的总览、交接或警告文档。
