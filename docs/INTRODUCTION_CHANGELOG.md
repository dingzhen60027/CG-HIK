# Introduction Changelog

日期：2026-09-02

## 本轮范围

本轮只确立论文主线与重构 Introduction。没有修改训练数据、冻结测试、模型、阈值、solver budget、fallback、verifier、结果文件、图件、Method、Experimental Design、Results、Discussion 或 Conclusion。

## 修改内容

| 文件 | 修改 |
|---|---|
| `paper/main.tex` | 替换标题；将 Introduction 重写为严格八段；将旧四分类/部署补丁式四项贡献改为三项科学贡献；增加必要引用 |
| `paper/references.bib` | 保留既有 key 兼容性；将 IKSel 更新为 2026 正式 T-ASE 元数据；新增 17 条经 DOI/官方 arXiv/PMLR 核验的文献，使总数达到 40 |
| `docs/PAPER_STORYLINE.md` | 固化一句话主线、Problem--Gap--Hypothesis--Method--Evidence--Conclusion 与术语边界 |
| `docs/INTRODUCTION_BLUEPRINT.md` | 固化八段逻辑、段间过渡、三项贡献和 RQ 接口 |
| `docs/LITERATURE_MAP.md` | 40 条文献按四条科学链和软件栈映射，记录出版状态与引用边界 |
| `docs/CLAIM_EXPERIMENT_MAP.md` | 将 RQ1/RQ2 development evidence 与 RQ3 formal evidence 分离，记录可复核数字与 estimand 边界 |
| `docs/INTRODUCTION_CHANGELOG.md` | 记录本轮变更、未修改范围和人工检查项 |

## 旧稿冲突说明

当前 `paper/main.tex` 的标题和 Introduction 已按现有反事实尾延迟路由与 fresh formal evidence 更新；但本轮用户明确禁止重写全文。因此 Abstract、页眉状态通知、Related Work、Method、Experiments、Results、Discussion 与 Conclusion 仍含较早阶段叙事，整稿当前仍不具备投稿一致性。这个事实不通过顺手扩大修改范围解决，下一轮全文重写时再统一。

## 文献变更

- 总条目：40；
- 2024--2026：24；
- 奠基/成熟文献：10；
- arXiv-only 或 accepted-but-not-yet-in-proceedings 条目均在 BibTeX `note` 和 literature map 中标明；
- IKSPARK 使用当前 v2 题名，同时保留首次公开年份 2024；
- IKDiffuser 使用当前 v4 题名并标记 under review；
- HJCD-IK 使用 v2 四位作者并标记 accepted to IROS 2026；
- Sequential IK 的 DOI 已激活，但 2026-11 issue 尚未到期，标为 in press；
- IKSel 保留历史 BibTeX key `yuan2025iksel`，元数据更新为 2026 T-ASE 正式论文。

## 证据修正记录

1. 三入口在共享 terminal fallback 后的 verified-success 标签完全一致，所以正文改为 shared semantic-success head + action-specific latency heads。
2. fixed cascade 会在成功后短路；正文不再写成每条查询都执行完整最坏路径，而写成固定入口可能产生可预测的冗余前缀。
3. policy-validation split 参与策略选择；正文不称其为 confirmatory held-out test。
4. development 的五次 query-level empirical P95 与 formal test 的 pooled raw-call P95 是不同 estimand，claim map 明确隔离。
5. formal negative evidence被放入贡献闭环：P50 增加、Panda P99 几乎不变、弱 point-OOD、defer recovery 为零、overall paper gate 为 false。

## 自动校验结果

- Introduction：8 个逻辑段落，约 1,558 个英文词；
- Introduction 内部研发版本标签：0；
- BibTeX：40 个唯一 key、40 个均在全文被引用、重复 DOI 为 0；
- 2024--2026 文献：24；
- LaTeX：在独立临时输出目录完成 `latexmk + BibTeX` 全流程，17 页 PDF，无 undefined citation、undefined reference、LaTeX error 或 overfull box；
- 视觉检查：Introduction 所在第 1--3 页及参考文献页未发现裁切、重叠、不可读字符或异常链接换行；
- `paper/main.pdf` 未覆盖，因为本轮允许修改范围只包括 LaTeX 源和参考文献库。

## 提交前人工检查

- [ ] 用 IEEE PDF 首页核验 CppFlow 页码；当前 BibTeX 有意省略争议页码。
- [ ] Sequential IK 正式 issue 发布后，移除 `in press` note 并复核卷/文章号。
- [ ] HJCD-IK proceedings 上线后补 DOI、页码和正式 venue metadata。
- [ ] 在全文重写阶段消除 Abstract/Method/Results 与新版 Introduction 的版本和方法冲突。
- [ ] 补作者、单位、ORCID、基金、利益冲突、数据/代码 DOI。
- [ ] 正式投稿前从 MDPI portal 获取最新模板，重新核验期刊分区、栏目、APC 与政策。
- [ ] 全文引用逐条做最终 DOI、题名、作者、卷期页复核，不依赖搜索摘要。
