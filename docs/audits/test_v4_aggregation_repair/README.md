# test_v4 aggregation-only repair：独立审计契约

状态：`POST_REPAIR_NOT_RUN`。本目录只定义独立审计方法；它不表示正式修复已经通过。

## 审计边界

正式六组合测量已经结束，聚合阶段因 JSON 对象键顺序检查失败。本审计只允许验证一次独立的 aggregation-only repair，禁止重新生成 query、调用 solver、加载模型、重写 checkpoint、重做 bootstrap、修改阈值或重跑正式方法。

审计器是 [audit_test_v4_aggregation_repair.py](../../../scripts/audit_test_v4_aggregation_repair.py)。它只使用 Python 标准库，不导入 `confik`、NumPy、Torch、SciPy、Pinocchio 或修复模块；只向 stdout 输出，不写任何 evidence 目录。

## 修复前冻结事实

| 项目 | 冻结值 |
|---|---:|
| 机器人 × seed | 2 × 3 = 6 |
| checkpoint | 744 |
| method-query records | 650,000 |
| seed17 每机器人 records | 175,000 |
| seed29/43 每机器人 records | 75,000 |
| 原 failure manifests | 42 |
| resume events | 41 |
| 原失败树文件 | 44 |
| 原失败树 digest | `bd294001b46cd6ebf96f55163030d329d4df582c2fdbf45818465f052aeb6bcd` |
| preregistration SHA-256 | `7808206ac9b76684f724523123bc461b7eedca4f4b759462eb142100385ad56d` |
| dataset manifest SHA-256 | `77fa670e5de1bf40301f0bcd0292ebcc7860f412ac8bd946a485dc782458d51e` |
| control-plane seal SHA-256 | `e97d13c190842451a20184a5279f3b0fceaadf1c501bcb2d5930817bd4d2b26a` |
| measurement fingerprint | `e0dda714ff8d6cb28fcaf8c8c9bf1ed2db0fffa66b476b0d1785f3a30cbabc38` |

六个 `combination_complete.json` 的哈希也直接冻结在审计器中。审计不会以修复程序新生成的摘要替代这些修复前常量。

## 独立检查链

1. 重新哈希 preregistration、dataset manifest、control seal、8 个正式 dataset artifact 和 frozen fingerprint 中的源文件/资产。
2. 对原 `.test_v4_aggregate.incomplete` 的 42 个 failure manifest、`latest_failure_manifest.json` 和 `resume_history.json` 重建 canonical tree digest，要求与修复前值完全相同。
3. 对六个组合逐一验证冻结的 completion marker；验证其 artifact 清单与磁盘文件集合严格相等。
4. 逐个检查 744 个 checkpoint manifest 和 `records.jsonl.gz`；验证哈希、大小、robot、seed、role、source identity，并构造完整的 `source query × methods` 集合，拒绝重复或缺失 pair。
5. 只读取 seed17 已存储的 8 个 unadjusted p-values，以 query-independent 的标准 Holm step-down 算法重算。`metrics` JSON 键顺序不参与成员身份判断。
6. 将重算结果与 `joint_holm_v4.json` 逐 hypothesis 对比；同时要求 aggregate summary 是六个已封存 summary/claim/OOD 文件的纯复制。
7. 结果必须仍为 `Panda=false`、`UR5e=true`、`joint Holm=true`、`paper=false`。Panda 唯一失败项必须仍是 `ood_feasible_false_reject_improvement`。
8. 校验 repair manifest 的零活动账本、原始输入哈希、纯键顺序 bug 分类、独立 staging 路径和原子目录提升声明；静态解析修复工具，拒绝科学运行时 import 和 solver/query/model 调用。
9. 校验 `test_v4_final_manifest.json` 列出的文件集合与最终三个 seed 目录和 aggregate 目录完全一致，并逐文件重算 SHA-256/size。

## repair manifest 最小契约

最终 aggregate 必须含 `aggregation_repair_manifest.json`，至少包括：

- `protocol = test_v4_aggregation_only_repair_v1`；
- `status = completed`，`repair_scope = aggregation_only`；
- `scientific_activity`：`query_generation_calls`、`solver_calls`、`model_inference_calls`、`checkpoint_record_writes`、`bootstrap_resamples`、`threshold_changes`、`gate_definition_changes` 均为 0；
- `input_evidence`：三个 control-plane hash、measurement fingerprint、原失败树 digest、六个 completion hash、744 checkpoint 和 650,000 records；
- `bug_classification.class = json_mapping_key_order_only`，且 stored metrics 未变、stored unadjusted p-values 被复用；
- `atomic_promotion`：独立 repair staging、同文件系统、原子目录 rename、最终路径；
- `repair_tool`：脚本相对路径和 SHA-256。

`test_v4_final_manifest.json` 必须排除自身后封存最终全部文件；`aggregation_repair_manifest.json` 应先生成，随后被 final manifest 纳入哈希清单，避免循环哈希。

## 正式运行

修复完成并完成原子提升后才运行：

```bash
python scripts/audit_test_v4_aggregation_repair.py
```

最终审计禁止使用 `--skip-fingerprint-file-rehash`。退出码 0 且 stdout 的 `verdict` 为 `PASS` 才表示审计通过。

## 证据限度

事后文件审计不能像系统调用追踪一样绝对证明一个进程从未执行某个函数。因此“0 query/solver/model”采用三层证据：修复脚本的静态 AST 审计、repair manifest 的显式零活动账本、以及六个 completion marker、全部 checkpoint/record 和源资产哈希保持不变。任何一层不通过都判定阻断。
