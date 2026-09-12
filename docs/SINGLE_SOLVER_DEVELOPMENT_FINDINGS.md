# Supplied single bounded Gauss–Newton solver: original-environment verification

Baseline `f72f89d76019f7887ae5105dd87050cfa699a0e5`, branch
`codex/hierarchical-v5`. This work imports the supplied implementation, not a
redesigned solver. No old paper, solver, verifier, trajectory or result is
modified. All new evidence is under `outputs/single_solver_development/`.

## Conclusion

The delivered kappa1 solver works as a single current-frame solver in the
original environment: **both robots complete40/40 in every repeat, and all
three repeats are also40/40 under DTSR20**. This is a full-run result, not
replacement of stored commands by later IK. Kappa0 completes39/40 on Panda
and40/40 on UR5e; the centering term's observed geometric increment is one
Panda development trajectory, not a demonstrated general benefit. Relative to
TRAC5, kappa1's cumulative time decreases9.27% on Panda but increases38.41%
on UR5e. Its median and P95 remain higher than TRAC5 on both robots. It is
faster and completes more trajectories than the fixed Pink implementation in
this comparison. No new optimization principle or independent generalization
claim follows from these already observed development data.

All results below are backed by
[`source_data.json`](../outputs/single_solver_development/reports/source_data.json)
and the [one-page conclusion](../outputs/single_solver_development/reports/CONCLUSION.md).

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

## Complete original-server comparison

Code, fixed settings, exact target identities and the18,000-command acceptance
were committed before full measurement as
`bad8b0abf5f7215b7548a5fcef2bebcea88b519b`. Later edits add saved-record
aggregation and this report only. The two supplied numerical files retain
their original hashes. Panda used CPU cores0,2; UR5e used4,6, with one
OpenMP/BLAS thread and within-robot interleaving. The original environment
versions are NumPy2.4.4, SciPy1.17.0, Pinocchio3.9.0, Pink3.3.0,
qpsolvers4.13.0 and OSQP1.0.5; the original TRAC binary is unchanged and hashed.

Each slash-separated entry is a separate full sweep, denominator40. TSR and
DTSR20 average the three runs within UID. Frame quantiles describe all calls,
including failed and late calls; cumulative seconds are the mean40-trajectory
sweep, not just successful trajectories. All table times below are measured
in this run, not imported from package timing or earlier repository runs.

| Robot / method | Complete /40 by repeat | Deadline-complete /40 | TSR (%) | DTSR20 (%) | P50/P95/P99 (ms) | Cumulative (s/sweep) |
|---|---|---|---:|---:|---|---:|
| Panda / TRAC5 | 36/37/37 | 36/36/37 | 91.67 | 90.83 | 0.1637 / 0.2627 / 5.2285 | 1.7937 |
| Panda / Pink | 37/37/37 | 37/37/37 | 92.50 | 92.50 | 0.9904 / 1.0348 / 1.1935 | 5.4525 |
| Panda / kappa0 | 39/39/39 | 39/39/39 | 97.50 | 97.50 | 0.2690 / 0.2915 / 0.4528 | 1.7258 |
| Panda / kappa1 | 40/40/40 | 40/40/40 | 100.00 | 100.00 | 0.2693 / 0.2931 / 0.3995 | 1.6275 |
| UR5e / TRAC5 | 38/39/39 | 38/39/39 | 96.67 | 96.67 | 0.1542 / 0.2156 / 0.3654 | 1.1244 |
| UR5e / Pink | 38/38/38 | 38/38/37 | 95.00 | 94.17 | 0.6302 / 0.6728 / 0.7928 | 3.8304 |
| UR5e / kappa0 | 40/40/40 | 40/40/39 | 100.00 | 99.17 | 0.2599 / 0.2740 / 0.3392 | 1.5760 |
| UR5e / kappa1 | 40/40/40 | 40/40/40 | 100.00 | 100.00 | 0.2594 / 0.2730 / 0.3299 | 1.5563 |

Kappa1 has zero late frames; its maximum full-call latency is0.655490 ms on
Panda and0.556722 ms on UR5e. All36,000 kappa1 commands pass the original
contract. This observed outcome is not a hard-real-time or population
completion guarantee. UR5e kappa0 has one48.509805 ms admissible call; its
internal status is `deadline`, while final task acceptance is true. UR5e Pink
has one48.802535 ms late call, and Panda TRAC has one48.145123 ms late call.
These large observations are retained. The one-call UR5e DTSR difference does
not establish that posture centering removes timing jitter; no causal source
of these isolated delays was instrumented.

### Paired comparisons and scope of the increment

| Robot / kappa1 comparator | TSR difference (pp),95% interval | Cumulative ratio,95% interval |
|---|---|---|
| Panda / kappa0 | +2.50 [0.00,7.50] | 0.943 [0.841,1.006] |
| Panda / TRAC5 | +8.33 [1.67,16.67] | 0.907 [0.641,1.333] |
| Panda / Pink | +7.50 [0.00,15.00] | 0.298 [0.290,0.309] |
| UR5e / kappa0 | 0.00 [0.00,0.00] | 0.987 [0.967,0.999] |
| UR5e / TRAC5 | +3.33 [0.00,9.17] | 1.384 [1.115,1.598] |
| UR5e / Pink | +5.00 [0.00,12.50] | 0.406 [0.402,0.410] |

These are the predeclared descriptive trajectory-stratified intervals; no
p-values, equivalence claim or post-result selection is used. The degenerate
UR5e kappa0 completion interval reflects identical observed complete outcomes,
not proof of equal population performance or zero failure risk. Full paired
DTSR20, frame success, motion and latency comparisons are in
[`paired_comparisons.csv`](../outputs/single_solver_development/reports/paired_comparisons.csv).

Compared with Pink, observed cumulative cost is lower70.15% on Panda and59.37%
on UR5e. Compared with TRAC, the cost direction differs between robots; TRAC
also remains faster at the median and P95 on both. Thus the delivered solver
provides a useful completion–cost tradeoff here, not uniform speed dominance.
No ratio uses the package's historical local timings.

For kappa1 versus kappa0, the only geometric gain is Panda trajectory14,
UID `4e626ba28da6a8d45d22f023c3732e2d796c83a0e74df2aa41168fe26f6dccab`:
0/3 → 3/3 full completions, with no lost UID. Kappa0 first fails at frame122
in every repeat. Its84 rejected frames terminate `stationary` above the task
position and/or orientation tolerance. This is an observed difference between
closed-loop executions, not proof that a particular earlier state was
mathematically infeasible or that one geometric mechanism is the sole cause.
Both kappas complete all UR5e trajectories; that set shows no geometric
completion increment from centering.

Against TRAC, kappa1 gains Panda trajectories12,14,17,26 and UR5e15,22, with
some gains only in a subset of TRAC repeats; against Pink it gains Panda14,17,38
and UR5e17,38. No kappa1 trajectory is lost because all are completed. Exact
UIDs, within-UID success fractions and stable3/3-versus0/3 distinctions are in
[`gained_lost_uids.csv`](../outputs/single_solver_development/reports/gained_lost_uids.csv).
[`completion_uids.json`](../outputs/single_solver_development/reports/completion_uids.json)
keeps the separate per-repeat sets rather than reporting a union as completion.

### Every family is retained

Counts are per-repeat geometric completions out of ten. Kappa1 also meets
DTSR20 for all ten in every family/repeat. Detailed family cost, error and
deadline results are in
[`family_table.csv`](../outputs/single_solver_development/reports/family_table.csv).

| Robot / family | TRAC5 | Pink | kappa0 | kappa1 |
|---|---|---|---|---|
| Panda / smooth | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 |
| Panda / near singular | 7/8/8 | 8/8/8 | 9/9/9 | 10/10/10 |
| Panda / joint-limit return | 9/9/9 | 10/10/10 | 10/10/10 | 10/10/10 |
| Panda / high curvature | 10/10/10 | 9/9/9 | 10/10/10 | 10/10/10 |
| UR5e / smooth | 10/10/10 | 10/10/10 | 10/10/10 | 10/10/10 |
| UR5e / near singular | 9/9/9 | 9/9/9 | 10/10/10 | 10/10/10 |
| UR5e / joint-limit return | 9/10/10 | 10/10/10 | 10/10/10 | 10/10/10 |
| UR5e / high curvature | 10/10/10 | 9/9/9 | 10/10/10 | 10/10/10 |

### Accepted precision and joint motion

Errors below are measured at actually accepted commands; rejected commands
remain in the complete raw records. No public tolerance is relaxed.

| Robot / method | Position P95 / max (mm) | Orientation P95 / max (rad) | Mean per-run acceleration RMS (rad/s²) |
|---|---|---|---:|
| Panda / TRAC5 | 0.3957 / 0.9915 | 0.0006413 / 0.0081062 | 4.0662 |
| Panda / Pink | 0.1965 / 0.9855 | 0.0003939 / 0.0083049 | 4.2044 |
| Panda / kappa0 | 0.4259 / 0.9911 | 0.0005491 / 0.0083748 | 3.0933 |
| Panda / kappa1 | 0.4334 / 0.9984 | 0.0005516 / 0.0083186 | 3.2144 |
| UR5e / TRAC5 | 0.3912 / 0.9558 | 0.0006362 / 0.0078383 | 10.2584 |
| UR5e / Pink | 0.1299 / 0.9992 | 0.0003204 / 0.0052590 | 12.1056 |
| UR5e / kappa0 | 0.3696 / 0.9981 | 0.0004946 / 0.0086425 | 5.7439 |
| UR5e / kappa1 | 0.3686 / 0.9981 | 0.0005197 / 0.0086425 | 5.7960 |

Kappa1's joint acceleration RMS is slightly higher than kappa0, but below
both mature baselines in these runs; it is not reported as an across-the-board
motion improvement. Some accepted commands approach the original velocity
boundary (maximum normalized step approximately1), which remains admissible
under the unchanged contract. Kappa1 averages1.971/1.968 outer iterations and
3.942/3.936 residual/Jacobian evaluations per call on Panda/UR5e. These counts
are execution diagnostics, not evidence of a new complexity bound.

## Actual commands and final verification

`development_panda/` and `development_ur5e/` preserve all960 full runs and
144,000 frames. Saved-record replay rechecked every target, actual previous
accepted state, returned command, residual, acceptance flag, deadline flag
and summary. It found142,896 accepted commands and zero accepted-contract
violations; no solver was called during this verification.

There are928 successful full runs, including all240 kappa1 runs. The
[`successful_trajectory_index.json`](../outputs/single_solver_development/reports/successful_trajectory_index.json)
links each to its actual150-frame command file with SHA256 and distinguishes
deadline completion. These are runs of80 independent trajectory UIDs, not928
independent trajectories. Every failure input and all actual commands remain
available, including kappa0's Panda failures and all timing outliers.

The supplied implementation has now been accepted and exercised end to end in
the original environment, with fixed settings and no recovery chain. That
practical delivery is established. A new algorithmic principle, broad benefit
of the centering term, independent-test generalization and hard-real-time
guarantees are not established by this development comparison. Work stops
here without modifying the paper, scanning parameters or adding a module.
