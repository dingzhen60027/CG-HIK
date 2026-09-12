# 直接接入已经实现的单求解器，不再增加兜底

仓库dingzhen60027/CG-HIK，当前分支codex/hierarchical-v5。
起点f72f89d76019f7887ae5105dd87050cfa699a0e5。

本包已经有实际代码与本地Panda整轨迹40/40/40结果，不是让你重新发明公式。
完整阅读README，严格区分本地重建开发数据、原服务器结果和独立测试。

## 先验收现成命令，不运行新IK

读取measured/portable_repeated.json和原仓库Panda40条目标。
按UID、帧索引匹配，比较本包data/panda_reconstructed.json中的initial_q、目标位姿
与原online_targets.json。保存最大数值差，不因名称相同就视为相同输入。
若位姿差在数值舍入尺度内，使用**原始目标**、每帧真实前一接受配置和原verifier
复核18000个已生成命令，输出完整通过/失败数及UID。
存在实质差异时定位模型/生成差异，不把本地40/40冒充原仓库完成率。

## 原样接入，不套壳

复制bounded_gn.py到src/confik/bounded_gn.py；
复制repo_adapter.py到src/confik/bounded_gn_adapter.py。
直接复用原NativeGeometry、原verifier与当前运行器。
保留交付的lambda=.01、kappa=1、30次更新、任务停止值1，不重新扫描参数。
可选Numba只是数值加速；原环境没有它时先跑同一NumPy算法，不盲目升级环境。
计时前预热所有实际使用的数值路径，完整计入原verifier和外层工作。

SingleBoundedGN.solve只求当前q，不允许：
- 先调用TRAC再修复；
- 失败后调用TRF；
- 调用TAR/Elastic/候选网络；
- 新增未来预测、13场景或新的评分；
- 接收q_ref或真实未来目标。

## 同一任务完成实际对照

在原有40条Panda与40条UR5e开发轨迹上，从第0帧运行三遍：
1. 已有task-aligned TRAC-IK5ms；
2. 已有Pink；
3. 单求解器kappa=0，同核消融；
4. 单求解器kappa=1，唯一主候选。

使用同机同环境交错运行；实际所有帧都记录，失败保持真实状态且目标正常推进。
不重新生成数据，不改变速度/容差，不为不同机器人选择不同参数。
提交TSR、DTSR20、P50/P95/P99、累计时间、误差、关节运动和全部命令。
若接口错误影响输入或计时，只修接口并保留运行记录，不偷偷改数值设计。

## 交付

先给出现成18000命令在原verifier的验收结果，再完成80条轨迹实测。
复用原运行器，新增单一入口或factory项即可，不新建框架或Git分支。
将代码、原始结果和简洁结论提交并push，确认SHA。
不要以“测试通过”“数学公式成立”代替实际轨迹结果。
不要在未比较相关研究之前把经典活动集、阻尼或居中项写成新发明。
