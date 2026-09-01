# CG-HIK 论文包

本目录保存 CG-HIK 的 LaTeX 稿件、图表数据和构建脚本。`Machines` 是当前候选期刊，正式投稿前仍需重新核验最新分区、栏目、版面费和模板。

## 稿件状态

`main.tex` 和 `main.pdf` 是正式 test_v3/test_v4 之前形成的结构草稿，数值和结论已经过期，不能直接投稿。旧稿保留的价值是章节结构、方法描述和参考文献基础；当前事实以仓库的 [研究主线](../docs/RESEARCH.md) 为准。

下一版必须写清楚：

- v2 先减少 FEV，但 eager 推理开销使延迟门失败；
- v3 通过 exact TorchScript 把实现问题与算法问题分开；
- v4 使用反事实动作成本、command reject 和 OOD/uncertainty defer；
- 两台机器人都改善 feasible P95，并保持相对 fixed 的成功率和轨迹完成率；
- Panda OOD 改善门失败，point OOD AUROC 很弱，整体 paper gate 为 false。

论文的中心句是：

> Learning allocates IK computation; numerical geometry generates joint commands; a deterministic verifier retains acceptance authority.

## 当前写作顺序

1. 用冻结结果重写 Abstract 和 Introduction 的贡献表述；
2. 将 v4 counterfactual router、shared semantic head、action latency heads 和 reject/defer 写入 Method；
3. 重建 Experimental Design，明确数据角色、seed 解释、七个基线和预注册 gate；
4. 用正式 v4 aggregate 重写 Results；
5. 在 Discussion 中同时解释 P95/FEV 收益、P50/P99 权衡、弱 OOD 和 Panda gate 失败；
6. 重新生成 source data、图和 PDF；
7. 补齐作者、单位、基金、代码/数据 DOI 和利益冲突信息；
8. 投稿前重新核验期刊分区、费用、政策和最新模板。

## 目录

| 路径 | 用途 |
|---|---|
| `main.tex`, `main.pdf` | 待重写的主稿和当前预览 |
| `references.bib` | 参考文献数据库 |
| `figures/` | 论文图件 |
| `source_data/` | 图表源数据 |
| `generated/` | 自动生成的数字宏和 evidence snapshot |
| `scripts/` | 证据提取与绘图 |

目录名中的 `v3` 为历史名称，暂不移动，以免破坏脚本和记录中的路径。论文标题和正文不使用该版本号。

## 构建

先更新正文和数字来源，再运行：

```bash
/home/eric/anaconda3/envs/isaaclab_3/bin/python paper_mdpi_machines_v3/scripts/build_evidence.py
/home/eric/anaconda3/envs/isaaclab_3/bin/python paper_mdpi_machines_v3/scripts/make_figures.py
cd paper_mdpi_machines_v3
/home/eric/.local/bin/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

正式投稿前应把修订后的正文迁移到当时从 MDPI author portal 下载的最新官方 LaTeX 模板，并逐页检查生成 PDF。
