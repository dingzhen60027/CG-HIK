# Current-frame bounded TRF: recovery and complete online execution

Baseline `8e754127bb56ec6054ef197f7f1b4e2324611b91`, branch
`codex/hierarchical-v5`. This is a standard numerical reference and practical
two-stage baseline, not a newly named IK algorithm. No paper, old solver,
verifier, target identity or frozen result is modified.

## Outcome in brief

The supplied four commands are genuinely admissible. Standard current-frame
bounded TRF also recovers 36 of the 44 exact selected first-failure inputs,
without future information or a witness seed. This establishes actual recovery
capability, not a new algorithm. The complete online comparison is less
favourable: standalone TRF substantially reduces Panda trajectory completion;
one remaining-period TRF after TRAC provides a small Panda geometric-completion
increase, no Panda deadline-completion increase, and no UR5e geometric increase.
Its observed cumulative cost is higher than TRAC alone on both robots.

All numbers below come from
[`source_data.json`](../outputs/current_frame_recovery/reports/source_data.json).
The [one-page conclusion](../outputs/current_frame_recovery/reports/ONE_PAGE_CONCLUSION.md)
summarizes the practical result. No parameters or inputs were changed after
the recorded outcomes.

## Before execution: fixed scope and semantics

The supplied `current_ik_workbench.zip` README, task specification and both
scripts were read in full. Its files are preserved in
`outputs/current_frame_recovery/task_package/`; its scripts are not rerun in
place, which would overwrite their evidence. The four successful TRF vectors
all pass the original project verifier at their own exact previous q, target
and source-frame dt. The fifth, originally unresolved input (Pink,
trajectory14, frame122) is rejected for position error 1.315486 mm. The four
accepted position errors are 0.971728, 0.881976, 0.340628 and 0.922141 mm.
These are verified current-command witnesses, not trajectory completions.

The only new bottom-level adapter is `src/confik/current_frame_trf.py`.
The public verifier retains `URDFKinematics`; optimizer FK and analytic
geometric Jacobians use the existing checked `NativeGeometry` Pinocchio
adapter with the same URDF, frames and scalar joint ordering. The SO(3)
residual derivative is the existing world-frame right-Jacobian-inverse
expression. This is not inferred from an environment name.

Optimization variable u satisfies q = previous_q + S u, where
S_i = velocity_i * dt + velocity_tolerance. Residual components are scaled by
the original position and orientation tolerances, with Jacobian D⁻¹ J_residual S.
Bounds are the intersection of URDF limits and this actual previous state's
single-frame interval, using the existing representable-interior construction.
Reconstruction into q clips only floating-point endpoint error ≤1e−12 rad;
the original verifier is unchanged. The previous command is checked first.
If legal, it is returned unchanged without a numerical search.

The standard SciPy `least_squares(method='trf')` uses analytic dense Jacobians,
the exact dense trust-region backend, linear least-squares loss, explicit
x_scale=1 in normalized coordinates, max_nfev=50 and ftol=xtol=gtol=1e−10.
The already installed SciPy is 1.17.0; no dependency upgrade. At each iteration
the original verifier checks the actual configuration. Task acceptance or
expiration of the remaining wall-clock period stops through the supported
callback. Native status −2 is retained separately from task acceptance; it is
not mislabeled native convergence. See the
[SciPy 1.17 API](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html).

The package's extra 0.99999 early-stop margin is not needed here: the direct
public verifier determines stopping at its unchanged full tolerances, as the
task requests. This is stated before results, not selected on solver success.
No witness vector seeds optimization. Native work is not asynchronously
preempted, so 50 evaluations and callback checks do not guarantee 20 ms.
All conversion, bounds, solve, callbacks, final verification and return-record
assembly remain in measured outer latency; disk serialization does not.

The composition first calls unchanged task-aligned TRAC 5 ms. If its returned
command passes the contract, it returns immediately. Only on rejection, and
only with positive remaining time, one identical TRF runs from the same actual
previous q with the original absolute 20 ms deadline. It neither starts a new
20 ms period nor uses the rejected TRAC configuration as an advantageous seed.
Late admissible returns count as geometric success but not deadline success.
Failure retains the preceding accepted command while targets keep advancing.

## Difference from existing repository TRF

| Property | Original `TRFFallbackSolver` | Earlier offline bounded refinement | This current-frame reference |
|---|---|---|---|
| Variables | q in radians; default scale | q in radians | u=(q−previous)/S; x_scale=1 |
| Residual | metres; orientation ×0.35 by default | unscaled metres and radians | separate physical task-tolerance normalization |
| Bounds | global URDF only; no previous/dt input | dynamic interval at supplied previous q | exact actual previous-state dynamic interval |
| Derivative | default two-point finite difference | default finite difference | original analytic geometry with SO(3) derivative, then D⁻¹ J S |
| Seed | caller / bank candidate | supplied candidate (often already legal) | actual previous q only |
| Stop | native convergence, then pose test | tight residual minimization, then verifier | first public-admissible iterate, native termination, or callback deadline |
| Evaluations | default100, caller-configurable | max200 | max50, no extra searches |
| Online acceptance | external verifier required | offline diagnosis, not deployed | original full verifier at entry/callback/final return |

Sources: `src/confik/solvers/fallback.py`,
`continuation_mechanism/observation.py::refine_candidate`, and
`continuation_mechanism/disentangle_math.py::uniform_refine`.
The configuration and integration differences are implementation choices,
not evidence of a new numerical algorithm or of a new scientific contribution.

## Fixed measurement plan

- Deduplicate all latest first-failure rows by exact binary robot, target pose,
  previous q and dt, preserving every source method/UID/repeat/frame link.
  Three TRF timing repeats per unique input, no future targets or cross-method
  previous-state replacement. The selected hard set does not estimate a
  general query recovery rate. Not found never means mathematically infeasible.
- Reuse both original 40-trajectory development files, four families ×10,
  150 frames, dt=20 ms. Do not regenerate or filter data.
- Compare task TRAC 5 ms, fixed Pink, current TRF alone and TRAC followed by
  one remaining-period TRF. All four run three full repeats from frame0.
  The unchanged existing `correction_reserve.study.run` and
  `execute_trajectory` perform the job/feedback logic; this entry supplies only
  these four factories, not TAR/Elastic. Method jobs are interleaved with a
  fixed order seed; robot runs use separate equivalent CPU pairs.
- TSR, DTSR20, complete outer cost/quantiles, accepted residuals, acceleration,
  first failures and gained/lost UIDs are retained. Complete successful command
  sequences are saved as witnesses only after actual full verification.
- UID is the independent trajectory unit (40/robot). Average the three repeats
  inside each UID, then reuse paired family-stratified 4,000-resample bootstrap
  intervals. Report all comparisons as descriptive unadjusted 95% intervals;
  no equivalence inference from zero-containing intervals, no pooled-robot
  success claim, and no claim of identical random TRAC seeds.

Protocol and exact identities are written before recovery or full-trajectory
outcomes. There is no weight/budget/seed search and no overall pass/fail gate.
The results below separate current-input recovery from complete tracking and
identify remaining observed failures without inventing their cause.

## Exact-input recovery

The latest baseline first-failure CSV contains 55 source rows and 44 unique
robot/previous-q/target/dt inputs. The selected inputs originate from five
Panda and four UR5e trajectory UIDs. Every recovered input passes in all three
calls, including the complete outer 20 ms criterion; none has a mixed geometric
result across these three calls.

| Robot | Unique selected inputs | Recovered in 3/3 calls | Still not found | Range of recovered-input median latency (ms) |
|---|---:|---:|---:|---:|
| Panda | 33 | 31 | 2 | 0.833–4.189 |
| UR5e | 11 | 5 | 6 | 0.724–3.571 |

Sources, exact inputs, returned joint vectors, residuals, maximum joint steps,
velocity utilization, native status, evaluations and timing are in
[`recovery_table.csv`](../outputs/current_frame_recovery/failure_recovery/recovery_table.csv)
and its `raw.jsonl.gz` companion; the 36 exact successful commands are in
[`witnesses.json`](../outputs/current_frame_recovery/failure_recovery/witnesses.json).
Source-method breakdowns are in
[`point_recovery_by_source.csv`](../outputs/current_frame_recovery/reports/point_recovery_by_source.csv).
Identical inputs can have multiple source methods, so these source categories
must not be summed as independent samples. Three calls measure within-input
repeatability/timing, not three independent states or trajectories.

The supplied unresolved Panda Pink input at trajectory14/frame122 remains
unresolved: position error 1.315486 mm, median 7.410 ms, 30 native function
evaluations. The other unrecovered Panda input (Pink trajectory38) has position
and orientation rejection and reaches 50 evaluations. The six UR5e inputs have
position rejection; they terminate after 6–21 evaluations, with median times
1.693–5.131 ms. Thus the remaining selected failures are not all explained by
exhausting the time budget. These are unsuccessful local numerical searches,
not infeasibility certificates. The report does not assign them a singularity,
branch or prediction cause.

## Complete online results

Every method processes all 150 targets from frame0 with its own accepted-state
feedback, on all 40 original trajectories per robot, three times. Slash-separated
counts below are repeats 0/1/2, each out of 40; TSR and DTSR20 average repeats
inside each UID. Cumulative time is the mean full 40-trajectory sweep, including
failed and late frames. Frame quantiles pool the recorded calls descriptively;
frames are not treated as independent inferential samples.

| Robot / method | Completed /40, by repeat | All frames accepted within20 ms /40 | TSR (%) | DTSR20 (%) | P50/P95/P99 (ms) | Cumulative time (s/sweep) |
|---|---|---|---:|---:|---|---:|
| Panda / TRAC5 | 37/37/36 | 37/36/36 | 91.67 | 90.83 | 0.299 / 1.263 / 5.614 | 3.126 |
| Panda / Pink | 37/37/37 | 37/37/37 | 92.50 | 92.50 | 1.187 / 2.136 / 2.473 | 8.214 |
| Panda / TRF | 28/30/29 | 28/29/29 | 72.50 | 71.67 | 1.845 / 10.123 / 20.486 | 18.337 |
| Panda / TRAC5 → one TRF | 38/38/36 | 37/38/34 | 93.33 | 90.83 | 0.329 / 1.319 / 10.494 | 4.138 |
| UR5e / TRAC5 | 38/39/38 | 36/38/38 | 95.83 | 93.33 | 0.295 / 0.850 / 5.493 | 2.920 |
| UR5e / Pink | 38/38/38 | 38/38/38 | 95.00 | 95.00 | 1.007 / 1.464 / 1.799 | 6.064 |
| UR5e / TRF | 38/38/38 | 38/38/37 | 95.00 | 94.17 | 1.129 / 2.031 / 3.521 | 7.494 |
| UR5e / TRAC5 → one TRF | 38/38/38 | 37/38/38 | 95.00 | 94.17 | 0.314 / 1.044 / 11.205 | 4.021 |

The composition versus TRAC5 has a Panda TSR difference of +1.67 percentage
points (descriptive paired95% interval 0.00 to +4.17), and DTSR20 difference
0.00 pp (−3.33 to +3.33). Its cumulative ratio is 1.324 (1.047–1.649), an
observed 32.37% increase, not a speedup. On UR5e, TSR changes −0.83 pp
(−2.50 to 0.00), DTSR20 +0.83 pp (−2.50 to +4.17), and cumulative ratio is
1.377 (0.976–1.792), an observed 37.70% increase. These intervals use 40 UIDs
per robot, averaging within UID before family-stratified resampling. They do
not establish equivalence, general benefit, or a causal explanation of changed
closed-loop trajectories. All requested comparisons, including Pink and
standalone TRF, remain in
[`paired_comparisons.csv`](../outputs/current_frame_recovery/reports/paired_comparisons.csv).

### Genuine same-frame recoveries versus whole-trajectory changes

Within the composition itself, TRAC rejects 306 Panda frames and 434 UR5e
frames across three sweeps. Exactly one TRF is then called per rejected frame.
It returns 46 Panda and three UR5e admissible commands, all within the original
20 ms period. These paired internal records prove recovery at the exact same
input; they are in
[`same_frame_recovery_records.jsonl.gz`](../outputs/current_frame_recovery/reports/same_frame_recovery_records.jsonl.gz).
They do not prove that the whole-trajectory difference between independently
run stochastic TRAC methods is entirely caused by those recoveries.

For the composition versus standalone TRAC, UID-averaged geometric completion
increases on Panda trajectory12 and trajectory26 (each 1/3 → 2/3), with no
UID-averaged loss; UR5e trajectory02 changes 1/3 → 0/3. None is an all-three
failures to all-three successes reversal. Full UIDs are:

- Panda trajectory12: `ea5503530085a4b7fba1457389f090742de31c9aa6ebe8907aa867289b032370`.
- Panda trajectory26: `b47f91198895baa3999a406c9c32516cd3e78dd7e42a8bccccc4cd058a03020a`.
- UR5e trajectory02: `f87cbb198e6aa95e16ea15160dd1d4f545d70b0d61622238ff991b3a2d27786a`.

[`gained_lost_uids.csv`](../outputs/current_frame_recovery/reports/gained_lost_uids.csv)
contains every UID and comparison, including unchanged and adverse outcomes.
[`completion_uids.json`](../outputs/current_frame_recovery/reports/completion_uids.json)
retains the actual per-repeat completion sets; they must not be replaced by
the union of successes across repeats.

### Families, errors and motion

Each family has ten trajectories. This table retains each repeat's geometric
count; family costs and all other measures are in
[`family_table.csv`](../outputs/current_frame_recovery/reports/family_table.csv).

| Robot / family | TRAC5 | Pink | TRF | TRAC5 → one TRF |
|---|---|---|---|---|
| Panda / smooth | 10/10/10 | 10/10/10 | 9/9/9 | 10/10/10 |
| Panda / near singular | 7/8/7 | 8/8/8 | 6/6/6 | 8/8/7 |
| Panda / joint-limit return | 10/9/9 | 10/10/10 | 6/8/7 | 10/10/9 |
| Panda / high curvature | 10/10/10 | 9/9/9 | 7/7/7 | 10/10/10 |
| UR5e / smooth | 9/10/9 | 10/10/10 | 10/10/10 | 9/9/9 |
| UR5e / near singular | 9/9/9 | 9/9/9 | 9/9/9 | 9/9/9 |
| UR5e / joint-limit return | 10/10/10 | 10/10/10 | 9/9/9 | 10/10/10 |
| UR5e / high curvature | 10/10/10 | 9/9/9 | 10/10/10 | 10/10/10 |

Standalone TRF's accepted position-error P95 is 0.932 mm on Panda and
0.923 mm on UR5e; orientation-error P95 is 0.008470 and 0.007949 rad,
respectively. These are larger but legal errors under the unchanged contract,
not tighter-precision commands. The mean per-run joint-acceleration RMS on
Panda is 28.30 rad/s² for TRF, versus 4.24 for TRAC, 4.20 for Pink and 4.89
for the composition; on UR5e the corresponding values are 10.29, 10.56,
12.11 and 10.38. Large Panda motion fluctuation is an adverse observation,
not hidden behind the selected-input recovery count or assigned an untested
cause. Position/orientation maxima and accepted step utilization are preserved
in the main table.

Standalone TRF has 339 Panda and one UR5e frames over20 ms. The composition
has 146 Panda and one UR5e such frames. Native callback checks cannot enforce
a hard real-time bound, and a legal late command is not deadline success.
New first-failure inputs, exact own previous states, returned configurations
and reasons are in
[`first_failure_inputs.csv`](../outputs/current_frame_recovery/reports/first_failure_inputs.csv).
All full-run TRF task rejections are position and/or orientation failures;
no accepted command violates the original pose, joint or rate contract.

## Delivered evidence and reproducibility

The implementation and fixed protocol were committed before full execution in
`c993ef39120894e0a7bf57175d351045beadcb62`. The later reporting-only addition
does not change the adapter, numerical settings, verifier or recorded runs.
Output stages refuse an existing directory rather than overwrite it.

- `development_panda/` and `development_ur5e/`: 960 complete run records,
  144,000 target frames, native statuses and TRF iteration traces.
- `trajectory_witnesses/`: 877 actually complete 150-frame joint sequences,
  each with exact inputs and measured times, plus a hashed index. These are
  successful **runs**, not 877 independent trajectories. Late-complete runs
  are explicitly distinguished from deadline-complete runs.
- `reports/verification.json`: 144,000 current returned-command checks and
  131,026 TRF callback-iterate checks; 141,141 accepted commands and zero
  accepted-contract violations. Feedback, original targets, the 50-evaluation
  limit and the composition's remaining-period semantics were also checked
  from saved records without rerunning solvers.
- `reports/main_table.csv`, `family_table.csv`, `trajectory_units.csv`,
  `paired_comparisons.csv`, `source_data.json`, manifests and completion sets
  are the machine-readable evidence. The numerical acceptance tests are
  implementation checks, not performance results.

The sole entry is `scripts/run_current_frame_recovery.py`. Its actions, in
order, are `package_check`, `prepare`, `failures`, `trajectories --robot panda`,
`trajectories --robot ur5e`, and `report`. Reproduction requires a separate
output location; do not delete this run to repeat it. Use the unchanged
`isaaclab_3` environment, `PYTHONPATH=tmp/crik_dependencies/python:src`, and
one BLAS/OpenMP thread. The two robot jobs used CPU pairs0,2 and4,6. Tests:
`python -m pytest -q tests/test_current_frame_trf.py` with third-party pytest
plugin autoload disabled.
All nine implementation tests passed, including scaled analytic derivatives
on both robots, the supplied recoverable inputs without witness initialization,
unchanged legal-command return, and the composition's single-call/shared-period
semantics. The only emitted warning is the existing hppfcl import deprecation;
no dependency was changed in response.

## What remains unresolved

Standard bounded TRF genuinely repairs many selected current-frame misses;
that practical result does not require a future-reserve score. It does not
eliminate every selected failure, and starting it from frame0 is not a reliable
universal replacement for the mature baselines. Remaining observed issues are
specific: local searches can terminate above the task tolerance, standalone
Panda tracking has lower completion and larger motion variation, and rare
recovery calls increase tail cost without a demonstrated two-robot deadline
benefit. Whether any failed exact input lacks a legal command remains unknown
unless a witness is available. No new objective, scenario set, model, paper
claim or further experiment is inferred from these observations. This task
ends with the implemented reference and the complete recorded comparison.
