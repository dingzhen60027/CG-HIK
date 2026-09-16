# 图表QA与统计图注

后端：Python/Matplotlib，复用已有环境。SVG文本可编辑，PDF嵌入TrueType；
PNG为300dpi报告预览，不是投稿TIFF。宽度182.9mm；三个PDF文本审计最小字形7pt。
source preflight无FAIL；两个WARN为未输出投稿TIFF及未采用600dpi，因本轮只交验收包，
已有矢量图和300dpi预览，接受这两项。没有把它们标为Nature正式投稿完成。
渲染时Matplotlib采用/tmp临时字体缓存（用户配置目录只读），未升级环境。

| 图/面板 | 研究问题与数据 | 单位/统计 | 视觉核查 |
|---|---|---|---|
| coverage_time a/c | 两机器人主方法相对七对照覆盖差 | query内三遍均值，240 anchor，几何分层4000 bootstrap，未校正95%区间 | 标签、零线、区间均可读，无遮挡；两机器人各自轴数值明确 |
| coverage_time b/d | 对应完整调用平均成本比 | 同anchor配对bootstrap；含失败，无截尾 | 比值1参考线可见，未使用不利样本筛选 |
| stopping_work a/c | 全样本工作量与成本 | 同批全样本均值，图是描述性工作计数；推断区间见coverage图/CSV | 三方法使用不同点形及颜色，图例列成功分母；不以此无区间图作显著性推断 |
| stopping_work b/d | 预定义压力格，与全样本同时呈现 | 每格240query，三遍嵌套；均值 | 未把压力格效果当总体；标签未遮数据 |
| application_actual a/c | 固定scan08/repeat0实际FK轨迹 | 保存的真实接受/保持状态，无插值、无独立重复推断 | 等轴比例，目标/反馈区分；GN/fixed/progress相同命令曲线重合为真实事实 |
| application_actual b/d | 同一序列真实误差 | 两块范数利用率的最大值，公共上限1 | SQP贴近边界、range曲线波动与主方法精度均完整保留 |

人工逐面板检查三个PNG，未发现裁切、文字/图例遮挡或单位误标。
点图的零覆盖差/零经验区间不证明总体等效；应用图为单条预定说明图，
不代替全部12条/机器人主表。原始数值文件为points_data.json、
stopping_comparison.csv和application_plot_data.csv。

可重复QA：

```
python /home/eric/.codex/skills/nature-figure/scripts/validate_figure.py scripts/report_task_balance_comparison.py --json
python /home/eric/.codex/skills/nature-figure/scripts/audit_pdf_text.py outputs/task_balance_comparison/reports/coverage_time.pdf --min-pt 5 --json
python /home/eric/.codex/skills/nature-figure/scripts/audit_pdf_text.py outputs/task_balance_comparison/reports/stopping_work.pdf --min-pt 5 --json
python /home/eric/.codex/skills/nature-figure/scripts/audit_pdf_text.py outputs/task_balance_comparison/reports/application_actual.pdf --min-pt 5 --json
```
