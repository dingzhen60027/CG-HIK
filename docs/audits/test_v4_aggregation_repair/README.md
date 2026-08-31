# test_v4 aggregation-only repair：独立审计契约

状态：`V1_SCHEMA_ADAPTED_ATTESTATION_PENDING`。本目录只定义独立审计方法；在外部 execution attestation 接入并完成正式只读运行前，不表示 repair v1 已通过最终审计。

## 审计边界

正式六组合测量已经结束，原 runner 在聚合阶段因把 JSON 对象键顺序误当作 confirmatory metric membership 而失败。保留的修复是独立 namespace：

```text
outputs/test_v4_aggregate_repair_v1/
├── aggregate_summary_v4.json
├── aggregation_repair_input_manifest.json
├── aggregation_repair_integrity.json
├── aggregation_repair_preregistration.json
├── joint_holm_v4.json
├── paper_gate_v4.json
└── test_v4_repair_final_manifest.json
```

原 `.test_v4_aggregate.incomplete` 和三个 `.test_v4_seed*.incomplete` 没有被提升、改名或覆盖。repair 只能重建 aggregate；禁止重新生成 query、调用 solver、加载模型、重写 checkpoint、重做 bootstrap、修改阈值或重跑正式方法。

审计器是 [audit_test_v4_aggregation_repair.py](../../../scripts/audit_test_v4_aggregation_repair.py)。它只使用 Python 标准库，不导入 `confik`、NumPy、Torch、SciPy 或 Pinocchio；只向 stdout 输出，不写 evidence 目录，也不导入 repair 实现。

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
| repair final manifest SHA-256 | `4f3b5024bdb6aa4ec283be4b4a0a8d3438e7a3ba03467154567a418817ba7ccc` |
| repair commit | `63e2ed6cbd14bbce0db869a247a9fb84e1f6911f` |
| repair tree | `254024def41e34f9ac0d83aacd7feca4947c8a82` |
| permanent ref | `refs/heads/codex/v4-aggregation-repair-exec` |

六个 `combination_complete.json` 和全部七个 repair v1 文件的 SHA-256 也直接冻结在审计器中。审计不会以 repair 新生成的摘要替代这些独立常量。

## 独立检查链

1. 重新哈希原 preregistration、dataset manifest、control seal、8 个正式 dataset artifact 和 frozen fingerprint 中的源文件/资产。唯一允许改变的原 fingerprint 文件是 aggregation repair 修改的 `reporting.py`；其原始字节必须仍可从 measurement commit `e22fe91...` 恢复并匹配原 descriptor。
2. 对原失败树的 42 个 failure manifest、`latest_failure_manifest.json` 和 `resume_history.json` 重建 canonical digest，要求修复前后完全相同。
3. 对六个组合逐一验证冻结 completion marker；验证其 artifact 清单与磁盘文件集合严格相等。
4. 逐个检查 744 个 checkpoint 和 gzip records；验证 hash、size、robot、seed、role、source identity，并重建完整 `source query × methods`，拒绝重复或缺失 pair。总数必须为 650,000。
5. 独立复核 repair input manifest 的四棵输入树：声明文件集合必须与磁盘严格相等，每个文件重新计算 SHA-256/size，before/after/integrity 三份结构必须完全一致。
6. 只读取 seed17 已存储的 8 个 unadjusted p-values，以标准 Holm step-down 重算。`metrics` JSON 键顺序不参与成员身份判断；不运行 bootstrap。
7. 将独立 Holm、robot claim gates 和 paper gate 与 v1 输出逐项比较。审计条件是“独立重算相等”，不是预设 Panda、UR5e 或 paper 必须为某个方向；方向只进入报告。
8. 要求 aggregate summary 是六个已封存 summary/claim/OOD 文件的纯复制；禁止重新读取 raw records 进行 aggregation。
9. 核对 repair preregistration/integrity/final 三层均声明 0 query rerun、0 solver invocation、0 model inference，输入树和 protected tree 不变，原失败分类不变。
10. 校验 permanent Git ref、commit、tree、parent、六个 repair source descriptor 与当前磁盘和 Git object 字节一致，但不导入或执行 repair 源码。
11. 校验 authoritative namespace 恰好七个文件；逐文件匹配独立冻结 hash，复算 final `hash_chain_digest` 和 `manifest_payload_digest`。

## Shadow Git provenance 的处理

v1 执行时使用：

```text
GIT_DIR=/tmp/confik-v4-repair-lineage.ZJRjoy/.git
GIT_WORK_TREE=/home/eric/wjg/btry
```

v1 七文件没有记录这两个环境变量。因此不能把 v1 内的 `scope_clean=true` 描述成“主物理工作树全局 clean”。它只表示 repair lineage/source scope 与 commit `63e2ed6...` 一致；主物理树当时存在与 repair 无关的 dirty docs，之后还增加了独立 auditor commit。

项目决定保留 v1，不做 v2。最终审计还必须读取一个输出目录之外、独立封存的 retrospective execution attestation。它需明确披露 shadow Git 环境、主物理树并非全局 clean、证据来源和这一事后补证的限制，并绑定永久 ref、source bytes、七个 v1 hash 及 final manifest。attestation 生成器 schema 落地后，审计器将增加该层；在此之前状态保持 `ATTESTATION_PENDING`。

## 正式运行

attestation 层接入后推荐执行：

```bash
python scripts/audit_test_v4_aggregation_repair.py
```

最终审计禁止使用 `--skip-fingerprint-file-rehash`。退出码 0 且 stdout 的 `verdict` 为 `PASS` 才表示审计通过；正式运行前不得修改任何 `outputs/.test_v4_*` 或 `outputs/test_v4_aggregate_repair_v1`。

## 证据限度

事后文件审计不能像同时代系统调用追踪一样绝对证明进程从未执行某个函数。因此“0 query/solver/model”采用多层证据：三份 v1 manifest 的一致声明、四棵输入树 before/after 逐文件不变、六个 completion marker 与全部 checkpoint/record 不变、repair source 的 content-addressed lineage，以及外部 attestation 的执行披露。任何一层不通过都应阻断最终接受。
