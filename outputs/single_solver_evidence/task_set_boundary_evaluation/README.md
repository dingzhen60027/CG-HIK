# Task-set boundary supplementary evaluation

This is a fixed current-query domain study, not a new algorithm or trajectory evaluation.

- Frozen baseline: `8e4aa63975d465f3de8a1eef2f7773889252c91d`.
- Input/code/development/weight selection commit before validation: `00e28a91102e6d71abe03896dbf0d77eb1b0eadc`.
- Main method unchanged: CompletedTaskBalanceGN, κ1, forcing .25, original NumPy and verifier.
- Fixed-weight comparator: common theta .35, selected on 30 independent development anchors per robot.
- Validation: 240 new anchors per robot ×9 queries, seven settings ×three nested calls.

See [findings](../../../docs/TASK_SET_BOUNDARY_FINDINGS.md), [protocol](../../../docs/TASK_SET_BOUNDARY_PROTOCOL.md), and [all tables](reports/TABLES.md). Raw commands are in `development_*` and `validation_*/records.jsonl.gz`; exact inputs/witnesses in `inputs/`; saved legal recoveries and all-method comparisons in `reports/recovery_commands.json`.

Original supplied six commands were verified before any replacement solve. Both full local probes remain under `task_package/`. Their timings are not server timings. No old outputs, old trajectory results or paper were changed.

All folders are exclusive-write experiment artifacts. Do not rerun into these paths. The read-only integrity check is `python verify_delivery.py delivery_manifest.json` from this directory; it performs no IK calls. It uses repository Git history and verifies source/data hashes.

The report's first empty render attempt encountered unavailable (`null`) TRAC iteration counts. The renderer was corrected to retain these as unknown; no measured solver calls or input were rerun or overwritten.
