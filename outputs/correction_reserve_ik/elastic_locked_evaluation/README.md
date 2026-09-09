# Elastic CR-IK locked evaluation

Candidate and identities were committed before evaluation. The only primary is
Elastic mu=0.25, shared by Panda and UR5e; all six settings have three repetitions.
160 independent UIDs/robot, four equal families, 300 frames each: 5760 runs and
1,728,000 measured calls. No new mu search, old evidence replacement or paper edit.

`protocol/` contains the pre-outcome seal, online inputs, disjoint identities,
separate verified reference paths and unchanged dependency/demand parameters.
`panda/runs/` and `ur5e/runs/` contain every frame and run summary, including
failures/timeouts. Each robot's completed manifest hashes every raw record.
`reports/` contains full/family tables, UID-averaged paired comparisons, completion
UIDs, first-failure inputs, decision/timing tables and a read-only nonlinear audit.
Successful raw trajectories retain their actual accepted q sequence, not q_ref.

The six-method run is complete. Do not rerun it to replace these results.
The independent numerical entry point is
`python -m confik.correction_reserve.locked_study`; its shell wrapper requires an
explicit prepare/check/panda/ur5e action and refuses occupied result directories.
Read-only reporting is `python -m confik.correction_reserve.locked_reporting`;
it also refuses to overwrite an existing report. See the frozen protocol for
the environment and the main document `docs/CRIK_ELASTIC_LOCKED_EVALUATION.md`.

Outer latency includes all command-ready operations. Offline future-error joins,
serialization and this audit are not online compute. dt=20 ms is not a hard
real-time guarantee. All statistical intervals use trajectory UIDs, with the
three repeats averaged first; Pink timing repeats are not independent samples.
