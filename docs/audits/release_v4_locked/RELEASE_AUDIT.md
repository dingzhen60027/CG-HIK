# release_v4_locked 独立审计

**结论：PASS。**

本审计只读取冻结发布包、冻结候选、开发/validation 数据与上游 v3 发布件；未解析任何旧 test_v3 性能内容，也未修改 `outputs/`。

## 核心结果

- 发布 payload：10/10，大小与 SHA-256 全部匹配。
- 发布 digest：独立复算 一致。
- 源码冻结：before/after 一致；记录 commit/tree 与 release runner/artifacts blob 均复算通过。
- protected tree：825 个文件，当前摘要与冻结 after 一致。
- 数值等价：Panda 与 UR5e 均在 2,500 条 policy-validation 查询上通过 batch-one eager↔TorchScript 复算。
- runtime 等价：6/6 个 robot×seed 组合通过，point 4,200 条、trajectory 2,400 条。
- test 边界：所有显式 test/load/start/authorize 标志通过；旧 test_v3 文件仅作为 opaque bytes 参与 protected-tree 哈希。

## 六个 runtime 组合

| 组合 | Point 接受命令 | Point reject/defer | Trajectory 接受命令 | Trajectory reject/defer | route/accepted/FEV/fallback/stages | reject零solver | defer从easy进入 |
|---|---:|---:|---:|---:|---|---:|---:|
| panda/seed17 | 499/700 | 184/16 | 400/400 | 0/0 | 1.0 / 通过 | 1.0 | 1.0 |
| panda/seed29 | 499/700 | 192/8 | 400/400 | 0/0 | 1.0 / 通过 | 1.0 | 1.0 |
| panda/seed43 | 500/700 | 191/9 | 400/400 | 0/0 | 1.0 / 通过 | 1.0 | 1.0 |
| ur5e/seed17 | 500/700 | 190/10 | 400/400 | 0/0 | 1.0 / 通过 | 1.0 | 1.0 |
| ur5e/seed29 | 500/700 | 188/12 | 386/400 | 1/2 | 1.0 / 通过 | 1.0 | 1.0 |
| ur5e/seed43 | 500/700 | 189/11 | 400/400 | 0/0 | 1.0 / 通过 | 1.0 | 1.0 |

## 证据边界

发布包保存了逐字段的 paired agreement、连续输出最大误差、接受命令误差、reject/defer 计数及语义率，但没有保存逐查询 runtime rows，也没有保存 FEV、fallback 和 executed-stage 的性能总量。因此，本审计能够确认 eager 与 locked runtime 在这些字段上逐条一致，却不能仅由 seal 独立重建这些字段的绝对性能总量。这不影响部署等价性 PASS，但建议未来发布包附带压缩的、去标识化 paired semantic rows。

## 非阻断加固项

- `release_manifest.json` 不在 10 项 payload manifest 内，也不进入 release digest；本次审计记录的控制文件 SHA-256 为 `816f6fa7fba2a015eb0f3dd146f76d469dd3ef13d8e7440b8381fcdcc0efee10`。建议在仓库外或签名清单中 pin 该摘要。
- 当前发布目录/文件权限为 `0o775` / `['0o664']`；sealed 是协议与哈希级冻结，不是 OS immutable。

## 阻断项

- 无。发布包满足锁定与进入全新、预注册 test_v4 的前置条件。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/audit_release_v4_locked.py
```

脚本只写入 `docs/audits/release_v4_locked/`，不会写入或修改 `outputs/`。
