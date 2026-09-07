# 最终论文说明（单页提要）

## 一句话主线

在线 IK 应先对齐 solver 内部目标与任务命令合同，再判断剩余失败成本是否足以
支持复杂计算分配；内部收敛、命令合法和按时返回是三个不同结局。

## 三项贡献与三个 RQ

| 贡献 | 对应问题 | 主要证据 |
|---|---|---|
| 三层结局与同输入 witness 漏解定义 | RQ1：内部收敛与任务接受何时不一致？ | 两机器人冻结点查询、完整状态矩阵、DLS 逐迭代记录；图 1/2、表 2/5 |
| 保守容差映射、动态关节/速度边界、独立 verifier 的统一接口 | RQ2：对齐是否改善相同合同下的成功和计算？ | 严格/分量/完整对齐、固定搜索预算、合同尺度与接受误差；图 2/3、表 1/2 |
| 两机器人闭环证据及分配收益边界 | RQ3：效果能否延续到轨迹，何时值得分配计算？ | 旧 Panda 与对称 UR5e 全轨迹、各 family、deadline、历史 allocation；图 4/5、表 3/4/6 |

## 结论与证据闭环

严格 DLS 存在较多“内部失败但返回命令合法”，其 trace 直接显示合法之后的额外
迭代。TRAC 严格设置已完成全部主点查询，因此这里证明的是计算差异，不是普遍
恢复漏解。完整对齐的 TRAC 改善两机器人的轨迹完成与累计耗时，但收益依 family
而异。DLS 提前接受降低计算，却在 UR5e 丢失完整轨迹，说明局部合法不等于更好的
闭环路径。历史 learned routing 未普遍加速成功求解，reject 的价值主要在失败
密集请求中。精确数字统一见
[TASK_CONTRACT_ALIGNMENT_FINDINGS.md](TASK_CONTRACT_ALIGNMENT_FINDINGS.md)
及论文自动生成 source-data，不在此另维护一套数值。

## 明确边界

仅精确软件运动学，无碰撞、动力学、接触、扭矩或实机证据；deadline 不是硬实时
证明。TRAC 内接分量框不等价于公共范数球。只测量既有 TRAC-IK 和 DLS，延迟依赖
当前实现与硬件；参考路径不保证 method-specific 状态可续接。历史 allocation
是边界案例，不是对齐 solver 后新接路由器的直接实验。没有提出新 solver，也不
把启用原生容差功能包装成新算法。

## 投稿适配与停止点

定位为机器人软件接口、任务定义与系统比较的工程实证研究，适合
[Machines 的 Robotics, Mechatronics and Intelligent Machines 栏目](https://www.mdpi.com/journal/machines/sections/Robotics-Mechatronics-Intelligent-Machines)
或 [Applied Sciences 的 Robotics and Automation 栏目](https://www.mdpi.com/journal/applsci/sections/robotics_automation)。
这是基于栏目范围的适配判断，不是录用预测。创新点在统一问题、接口语义及完整
比较，而非新数值算法；审稿仍可能质疑增量创新和仅软件证据。

科学稿、图表、引用与数据链已完成。正式投稿前需作者提供真实署名、单位、贡献、
资助和利益冲突声明，并确定期刊模板；未虚构这些信息或数据 DOI。
本研究到此停止，不追加实验、模型、solver 或其他算法分支。
