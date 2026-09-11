# Codex任务包

先读 `CODEX_TASK.md`。本包是下一轮任务书和数学检查，不含已完成的Panda/UR5e新算法或性能结果，也未push远程仓库。

本包包含：
- 单一直接修正算法的数学定义、代码结构、基线与开发实验；
- 前一轮固定右逆代理排序反例；
- 新的单程序线性修正检查与实际运行结果。

运行：
```bash
python test_linear_recourse.py --output linear_recourse_results.json
python check_reserve_proxy.py
```
依赖：NumPy、SciPy。`checks_passed`仅代表线性数学检查，不表示机器人任务已改善。

原在线任务和旧证据保持不变；不以本包预先保证论文新颖性或录用。
