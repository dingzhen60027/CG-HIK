# Supplied single bounded Gauss–Newton solver: original-environment verification

Baseline `f72f89d76019f7887ae5105dd87050cfa699a0e5`, branch
`codex/hierarchical-v5`. This work imports the supplied implementation, not a
redesigned solver. No old paper, solver, verifier, trajectory or result is
modified. All new evidence is under `outputs/single_solver_development/`.

## Existing-command acceptance, before any new IK measurement

The supplied README and CODEX_NEXT_TASK.md were read in full. All 40 Panda UIDs,
site IDs, families, seeds, dt values and 150 frame indices match the original
`online_targets.json`. Across all 6,000 targets, the maximum reconstructed
position-coordinate difference is 4.440892098500626e−16 m; the maximum rotation
matrix element difference is 3.885780586188048e−16. Initial q is identical.
The fixed comparison criterion was 1e−12 in each stored coordinate, not a
change to the command contract.

Using **original** targets, the actual preceding accepted command and the
original `URDFKinematics` verifier, all **18,000 supplied commands** pass.
The three complete-trajectory counts are **40/40, 40/40, 40/40**. The supplied
previous-q records and the reconstructed accepted-state feedback agree exactly.
No IK was run to repair or replace a command. UID sets, frame checks and
hashes are in `package_verification/`.

This confirms geometric command acceptance in the original environment.
It does not establish original-server DTSR20 or solver latency: the supplied
`latency_ns` values remain historical **local** measurements, explicitly saved
as `supplied_local_latency_ns`. Replay verification time is not IK solve time.

## Fixed integration and measurement protocol

`src/confik/bounded_gn.py` and `src/confik/bounded_gn_adapter.py` are byte-for-byte
copies of package `bounded_gn.py` and `repo_adapter.py`; their SHA256s are
`efb81e50a998505511660bdf22733858789d90409474f703fff4c06dd87b6a7a` and
`ab6212f2f8f70dee4c0eef639f8bb768739376796341df6482de2b915819651a`.
The core is not changed for either robot. Fixed settings are initial damping
0.01, kappa0/1, task stop1, 30 outer updates, 50 active-set updates per box QP,
eight backtracking scales, and the supplied internal20 ms soft time check.
Kappa weights joint-posture centering; it does not alter the common orientation
tolerance. The supplied damping adaptation and residual-descent logic remain
unchanged. No TRAC/TRF fallback, TAR/Elastic calls, future targets, q_ref,
candidate pool, new search or robot-specific tuning is added.

The installed optional Numba cannot import because the installed coverage
module lacks `coverage.types.Tracer`. Before measurement, the experiment entry
marks that optional import unavailable so the **supplied NumPy fallback** is
used. Neither installed dependencies nor the supplied numerical files are
patched. This environment compatibility choice is not selected by performance.
Versions and the actual import error are retained in the protocol and adapter
metadata. No package is upgraded.

Each single-solver factory explicitly warms the actual dimension's box QP and
three complete nonstationary adapter calls on fixed synthetic midpoint inputs,
then the unchanged runner performs its normal stationary warmup. Warmup never
uses a benchmark success or witness seed. Every measured call includes input
conversion, the numerical loop and the original final verifier. Disk writing
and one-time construction/warmup are outside per-call latency. Since the core
time check starts inside the adapter and cannot preempt an iteration, late
outer returns remain possible and are counted, not trimmed.

The sole entry `scripts/run_single_solver_development.py` reuses unchanged
`correction_reserve.study.run`, `execute_trajectory`, `group_table` and paired
statistics; only the method factory and study-specific reporting are supplied.
The four methods are task-aligned TRAC5, the fixed Pink adapter, single GN
kappa0 and single GN kappa1. Each processes the original40 trajectories per
robot, four families of ten, all150 frames, in three complete repeats. Commands
update feedback only after original-verifier acceptance; failure holds the
previous accepted command and the external target index advances normally.

Method jobs are interleaved with fixed seed2026091205. Inferential units are40
trajectory UIDs per robot, with the three repeats averaged inside each UID
before paired family-stratified bootstrap (4,000 resamples, seed2026091206).
Intervals are descriptive unadjusted95% intervals, conditional on the observed
repeats, not equivalence tests. Complete and deadline-complete trajectory
counts, all frame times/errors/motion, failed calls, late calls, first failures
and gained/lost UIDs are retained. No trajectories or calls are excluded.

The supplied package contains prior development parameter exploration and
locally reconstructed Panda results. Those remain development history; they
are neither fresh data nor original-server timing. This run uses the fixed
delivered settings and the original two-robot targets without a new scan.

## Implementation checks

Seven tests pass: unchanged package source hashes, small positive-definite box
QP KKT checks, both kappas on both robots with original-verifier acceptance and
analytic-derivative checks, and final-verifier rejection despite a native task
candidate. These are implementation checks, not trajectory performance results.
The sole warning is the existing hppfcl-to-coal import deprecation; it does not
trigger an environment upgrade.

## Run commands

Use the unchanged `isaaclab_3` Python, one OpenMP/BLAS thread and
`PYTHONPATH=tmp/crik_dependencies/python:src`. Actions, in order:
`verify-package`, `prepare`, `trajectories --robot panda`,
`trajectories --robot ur5e`, `report`.
New result directories refuse overwrite. The provided local-run scripts are
not rerun in place and their historical files are preserved under `task_package/`.
