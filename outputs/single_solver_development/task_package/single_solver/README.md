# 单一有界 Gauss–Newton IK：已运行的 Panda 开发结果

## 已完成，而非计划

本包包含实际执行的数值代码、完整轨迹命令、同环境 TRF 对照和独立验收记录。
没有 TRAC→TRF 兜底、学习路由、未来目标、场景变量或多初值搜索。

**可移植实现：Panda 40条 × 150帧，三次完整运行均为40/40；全轨迹20ms成功数亦为40/40。**
可移植版本的本地完整调用 P50/P95/P99 为 0.253503 / 0.3486265 / 0.5049234 ms，
最大3.080869 ms。包含最终独立Python FK验收，运行前预热JIT，不含磁盘写盘。
这不是原服务器、实机或硬实时测试。

## 数据身份与证据边界

- 仓库基线：dingzhen60027/CG-HIK，f72f89d76019f7887ae5105dd87050cfa699a0e5。
- 源生成器：src/confik/revision_compute_allocation/data.py::trajectories。
- Panda调度：continuation_mechanism/tolerance_comparison.py，seed=970909000+i，i=0…39。
- 当前容器不能直接下载大型GitHub JSON，因此使用公开的同一Panda链参数、生成器和种子重建轨迹，
  不是逐字节复制原数据。5条已有目标记录的最大位置差1.12e-16 m、旋转矩阵元素差2.23e-16；
  对应UID一致。尚未在原服务器逐帧比对全部6000个目标。
- 本地运动学来自此前已经对照过残差的Panda URDF参数；32个随机状态与另一FK实现相符。
- **这是已观察数据上的开发结果。** 本轮进行了14组小范围开发配置比较；所有配置及结果
  保存在results/与DEVELOPMENT_LOG.csv。选择姿态系数1、无历史外推的配置发生在观察这些结果后。
  不能将40/40称为独立泛化验证，三次运行也不是120条独立轨迹。
- 尚未本地执行UR5e，也没有执行原生TRAC-IK/Pink。不得把本包耗时除以服务器历史耗时声称提速。

## 单一数值更新

在每个当前目标的求解中，用固定的真实上一接受配置构造关节范围与单帧速度区间。
令S=diag(qdot_max*dt+epsilon_v)，L=diag(q_max-q_min)，q_c=(q_max+q_min)/2。
归一化位姿残差为e(q)，残差Jacobian为J_r。
每次迭代直接求一个带上下界的二次问题：

    min_d 0.5 ||e(q)+J_r S d||^2
          +0.5 lambda ||d||^2
          +0.5 kappa ||L^-1(q+S d-q_c)||^2

约束为q+S*d处于本帧真实允许的关节区间，且|d_i|<=1。
活动集算法在变量触及边界时重新求解剩余自由变量；KKT符号允许变量释放。
使用真实当前残差回溯、阻尼更新；当前命令通过原任务条件才允许执行。
整个求解只优化当前关节变量，不向任何其他IK库请求候选。

固定交付设置：lambda_initial=.01，kappa=1，最多30次外层迭代；
每次活动集最多50次更新，回溯8档。位置1mm、姿态0.00872664626rad，dt=.02s。
本包原型没有宣称全局最优、完整轨迹可行保证或新发明了活动集/Gauss–Newton。
本轮实际增加的是一个已实现、在本地同源开发轨迹上得到整轨迹改善的单求解器。

## 本地同环境主比较

以下比较使用同一Numba FK/Jacobian和同一直接验收、同一组目标，均预热后执行。
| 实现 | 完成数（三遍，每遍40） | DTSR20数 | P50/P95/P99 ms | 每40条累计ms |
|---|---|---|---|---:|
| 标准SciPy有界TRF参考 | 36/36/36 |36/36/36|.950469/6.090916/10.262417|10177.770|
| 单一Box-GN，无姿态项 |39/39/39|39/39/39|.028622/.044138/.085039|197.353|
| 单一Box-GN，姿态项kappa=1 |40/40/40|40/40/40|.028922/.042636/.067489|184.049|

TRF设置遵循已有current_frame_trf的归一化、动态范围、最多50次评估和迭代任务验收；
本地TRF外围实现与原服务器不是同一个二进制/计时路径，36/40不覆盖服务器的28/30/29。
速度差同时包含数值方法、调用次数和实现路径，不能称为纯算法复杂度提升。
同核去掉姿态项的对照只差1条开发轨迹；不把这一差异夸大成已经证明的广泛方法创新。

另有**通用Python求解循环＋独立Python FK终验**的可移植运行：40/40/40，
.253503/.3486265/.5049234 ms。它较JIT原型慢，但更接近要接到现有Pinocchio接口的形式。
这两种实现必须分开报告，不把JIT原型速度贴到尚未运行的仓库适配器上。

## 检查与结果文件

- measured/summary.json：完整360次轨迹运行的环境、主表及文件索引。
- measured/independent_check.json：54000帧独立FK复核，53793个接受命令，0处接受分歧。
- measured/portable_repeated.json：通用实现三遍的18000个命令、每帧耗时和完整结果。
- measured/portable_check.json：100个随机QP与SciPy对照，最大目标差3.87e-12、投影KKT残差3.56e-15。
- measured/derivative_checks.json：32个随机状态；归一化残差导数有限差分最大差1.42e-7。
- data/reconstruction_checks.json：5条仓库目标的重建数值比较。
- measured/*_r*_*.json：所有方法、全部失败和全部超时的原始记录。
- results/：14组开发探索的完整轨迹结果，首个探索运行可能包含JIT加载，不用于正式时延对比。

## 代码

- bounded_gn.py：通用单一数值求解器，NumPy；可选Numba加速活动集。
- repo_adapter.py：接到现有confik的适配类，不创建任何TRAC/TRF对象。
- single_core.py、panda_model.py：本地JIT实测原型和Panda链。
- reconstruct_development.py：同源生成器的本地重建版本。
- final_comparison.py：本地同环境三方法完整比较。
- repeat_portable.py：显式预热后的通用版本运行。
- check_core.py：QP正确性与可移植运行检查。

安装/运行本地复现需要numpy、scipy、numba（本次已安装版本见环境文件）。
不要在原项目盲目升级依赖。

    OPENBLAS_NUM_THREADS=1 python reconstruct_development.py
    OPENBLAS_NUM_THREADS=1 python final_comparison.py
    OPENBLAS_NUM_THREADS=1 python repeat_portable.py

原测量目录已经有结果，请复制到新的工作目录再复现，以免覆盖本包证据。
**CODEX_NEXT_TASK.md只要求直接接入和原环境复核，不要求设计下一套框架。**
