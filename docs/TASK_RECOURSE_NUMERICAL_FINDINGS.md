# TAR-IK: numerical completion of the unchanged joint objective

Baseline: `1a9db5071e6212bc25a9cdae9fe278328e47646a`. Development only.
No new algorithm name, objective, scenario, predictor, task contract or solver.
All new evidence is under
`outputs/task_recourse/numerical_completion_development/`; old outputs and paper
are not overwritten. The exact old kernel is loaded from the baseline Git blob
in memory for comparisons, rather than maintaining another versioned runtime.

**Conclusion:** the corrected kernel finds the verified zero-cost solution for
all 79 known-zero mechanism inputs and modestly reduces complete-run cost, but
does not improve aggregate trajectory completion. Panda retains the observed
UID-average completion vector; UR5e loses net completion and deadline success
in the observed repeats. This supports a numerical repair, not an additional
task-performance claim. See the [one-page conclusion](../outputs/task_recourse/numerical_completion_development/reports/ONE_PAGE_CONCLUSION.md).

## Protocol, before complete trajectory outcomes

1. Recheck all saved witness commands at each of the 80 mechanism states using
   the **actual internal backup of that state's old TAR call**. All 13 scenario
   targets must match exactly. A witness originally obtained from another
   previous state is only evidence after complete re-verification at this
   anchor; its source and previous-state equality are recorded. Also allow
   previously verified planned nodes at the same saved targets. No new witness
   searches are performed. Missing complete witnesses remain `unknown`.
2. A legal anchor plus legal commands for all nodes establishes J=0: q=anchor
   makes the command term zero and all future normalized pose excesses are zero.
   Nonnegativity of the unchanged objective supplies a global lower bound for
   this finite-node input only. No optimizer return code proves this property.
3. Improve initialization with the existing scaled least-squares nominal
   prediction step, then a common-Jacobian node correction. Clip guesses to
   the exact-anchor joint/rate interval. The map initializes z but never
   constrains the final free configurations. It is not thirteen IK solves.
4. Use the same sparse joint program with q's lower and upper bounds clamped to
   anchor; only after remaining real excess release q for joint updates.
   Recompute real FK, keep the lowest legal true objective, and return the exact
   unchanged anchor whenever its complete zero-cost plan is verified. A native
   current-pose interior margin must not exclude an already publicly legal
   fixed anchor; its bound uses the public unit radius in the fixed phase.
   This does not relax the public verifier or future joint/rate constraints.
5. Compare fixed-current update caps 2, 4, 6 on all 80 existing mechanism inputs,
   with two released updates and the existing 18 ms soft outer / 2 ms native
   limit. Three timing repeats per input. Select the budget with most known-zero
   states solved in all three repeats among those with reconstructed outer P95
   ≤20 ms, then the smallest cap; if none qualify select the lowest P95.
   This is a development numerical-budget choice, not a performance gate.
6. Same-input replay substitutes exactly the same saved backup in old and new
   kernels. Its numerical time is measured; adding the saved backup duration is
   labeled **reconstructed**, not a newly measured online TRAC call. Saved
   witnesses and actual next targets never enter online optimization.
7. Freeze the chosen budget before complete online runs. Compare old TAR,
   corrected TAR, task TRAC 5 ms and fixed Pink on all 40 existing trajectories
   per robot, three full repeats each, 150 frames. Interleave method jobs on the
   same resources. Accept before updating own state, hold after failure, advance
   every target and count every frame/time/timeout. No reference reset.

## Measurement and interpretation rules

The two success endpoints are TSR and DTSR20. Retain full outer latency,
per-trajectory cumulative time, accepted pose errors, commanded acceleration,
and all gained/lost UIDs. The additions concern only zero-cost completeness and
the initialization/fixed-current/joint-update/FK-check costs. Those timing
components are disjoint: fixed and joint categories contain linearization,
cone construction and solve; initial and nonlinear/final acceptance checks are
reported in FK checks. Backup, prediction and accounting remainder also remain
inside outer latency. Stage medians are not added to estimate a total median.

For complete trajectories the independent unit is UID (40/robot), with three
runs averaged inside UID before paired family-stratified bootstrap intervals.
Intervals are descriptive, unadjusted 95%, using the existing 4,000 resample
procedure. A zero-crossing interval does not establish equality. Same-input
comparisons use the same actual backup; across different closed-loop states,
aggregate percentages do not establish causal explanations.

Exact nonzero joint changes are distinguished from changes >1e−8 rad (the
existing reporting threshold); objective gaps are also shown against the old
1e−10 absolute comparison tolerance. Neither reporting threshold relaxes any
witness check. A tiny scalar objective does not by itself establish exact zero.

## Initial read-only finding

Saved commands establish zero-cost references for 39/40 Panda and 40/40 UR5e
inputs, with one Panda unknown. The old call changed all 79 anchors in exact
floating-point comparison, but no Panda change exceeds 1e−8 rad and none has an
objective gap above 1e−10. Only one UR5e input exceeds both reporting levels.
Thus most exact changes in these mechanism states are rounding-scale, not
evidence of a large unresolved optimization gap. Full raw configurations,
targets and re-verification results are saved; no unmatched source previous q
was silently reused.

The final conclusion will distinguish: (A) the old numerical implementation
did not solve the input's original objective well, (B) solving it better still
does not add task value, and (C) numerical completion yields actual task benefit.
None is assumed from the initial reference count.

## Numerical development log

The first same-input implementation used caps 2/4/6 and solved 78 of the 79
known-zero inputs exactly in all three repeats. Its provisional selection is
preserved as `selection.json`; no complete trajectory run used it. On UR5e
trajectory_22 the new initialized plan touched the sixth joint upper bound.
Although the cone solver returned `Solved`, reconstructed future commands
exceeded that bound by 1.2e−13 to 5.5e−13 rad. Every backtracked update was
therefore rejected, leaving true objective 8.10826 instead of the known zero.
This is an implementation-level numerical boundary issue, not evidence that
the recourse objective needs a different algorithm.

The existing q reconstruction rule (only clip violations ≤1e−8 rad, then check
the original nonlinear contract) is now applied to future z as well, using the
same representable interior bounds from the candidate current q. Larger
violations are not projected. This fixes arithmetic reconstruction, not the
verifier. The initial probe files remain in `same_input/`; the corrected probe
uses `same_input_roundoff_corrected/`, with a separate final selection file.
No objective/weight/trust-region/predictor change was made. A regression test
retains this exact state. Eleven tests pass after this correction.

Final same-input selection: **at most two fixed-current and two released joint
updates**, with the existing 18 ms soft limit and 2 ms native call limit.
All three candidate caps (2/4/6) solve all 79 known-zero inputs exactly in all
three repetitions after the roundoff correction; the smallest cap wins under
the recorded rule. Of the 79 states, 45 are zero at initialization, 33 after
one fixed-current update, and one after two. No released update is needed for
these known-zero inputs. One Panda state remains unknown/unavailable. These
results do not establish that the budget suffices on every online input.
`selection_roundoff_corrected.json` fixes that budget before the first complete
run attempt; `selection_final.json` inherits it unchanged after the interface
repair described below.

The first full comparison attempt completed UR5e but stopped Panda after 217
trajectory runs. A substantially infeasible current trial from an unsuccessful
native solve was passed to `representable_interior`, whose valid-current-state
precondition did not hold. The helper correctly asserted. The repaired caller
skips future roundoff reconstruction for such current trials and retains their
true-FK/verifier rejection; it does not clip a substantive violation or relax a
constraint. A regression injects a native infeasible trial and checks rejection
and exact backup retention. All 12 tests pass. Both robots are rerun under the
same guarded source in `validated_panda/` and `validated_ur5e/`; all preliminary
records remain in `development_panda/` and `development_ur5e/`, with hashes and
the interrupted job in `interface_guard_amendment.json`. This is development
debugging, not a fresh evaluation or outcome-driven numerical-budget change.

## Same-input zero-cost comparison

All entries use the old call's exact internal backup, not a new random TRAC
return. Saved witnesses are evaluated offline; online initialization never
reads them. The one unavailable Panda input remains in the timing denominator
and is not relabeled mathematically infeasible.

| Robot | Known J=0 / all states | Original exact-zero misses | Original changes >1e−8 rad among known-zero | Corrected exact-zero misses (all three repeats) |
|---|---:|---:|---:|---:|
| Panda | 39/40 | 39 | 0 | 0 |
| UR5e | 40/40 | 40 | 1 | 0 |

The largest old gap on these inputs is only 4.640891e−10 (UR5e
trajectory_23); its largest joint change is 1.917283e−6 rad. The corresponding
Panda maxima are 3.002647e−16 and 8.153139e−10 rad. The corrected selected
budget gives J=tau=0 and bitwise-unchanged commands for all 79 known-zero
inputs in each repeat. This establishes better numerical completion on these
exact inputs, **not a large previous objective failure or a task benefit**.

| Robot | Kernel | Measured numerical P50 / P95 / P99 (ms) | Reconstructed outer P50 / P95 / P99 (ms) |
|---|---|---|---|
| Panda | Original | 5.204 / 5.481 / 5.705 | 5.431 / 5.934 / 7.936 |
| Panda | Corrected | 3.334 / 4.880 / 5.865 | 3.594 / 5.639 / 8.351 |
| UR5e | Original | 4.848 / 5.055 / 5.248 | 5.049 / 5.312 / 7.163 |
| UR5e | Corrected | 4.454 / 4.691 / 5.781 | 4.628 / 5.060 / 6.691 |

Numerical timing excludes the replay stub; reconstructed timing adds the old
backup's saved measured duration. It is not a newly timed native search, and
the two columns must not be presented as complete online execution benchmarks.
Tail improvements are not uniform even on the same-input replay. Three timing
repeats are nested inside each of 40 states, not 120 independent states.

## Complete development comparison

Source: [main table](../outputs/task_recourse/numerical_completion_development/reports/main_table.csv).
Each completion cell lists the three runs, each out of the same 40 UIDs.
Cumulative seconds are per 40-trajectory sweep, averaged over repeats; frame
quantiles include every processed frame, including failures and timeouts.

| Robot | Method | Completed /40, by repeat | All frames accepted and ≤20 ms /40 | Outer P50 / P95 / P99 (ms) | Cumulative seconds |
|---|---|---|---|---|---:|
| Panda | Original TAR | 36 / 37 / 37 | 35 / 37 / 34 | 4.392 / 5.742 / 7.128 | 25.744 |
| Panda | Corrected TAR | 36 / 37 / 37 | 35 / 35 / 36 | 3.593 / 5.320 / 8.224 | 23.894 |
| Panda | Task TRAC 5 ms | 37 / 37 / 36 | 37 / 36 / 36 | 0.166 / 0.292 / 5.232 | 1.937 |
| Panda | Pink | 37 / 37 / 37 | 37 / 37 / 37 | 1.003 / 1.053 / 1.273 | 5.539 |
| UR5e | Original TAR | 38 / 40 / 39 | 38 / 39 / 39 | 4.083 / 5.227 / 6.967 | 23.902 |
| UR5e | Corrected TAR | 38 / 39 / 39 | 36 / 38 / 38 | 3.330 / 4.733 / 7.250 | 22.409 |
| UR5e | Task TRAC 5 ms | 39 / 39 / 39 | 39 / 37 / 39 | 0.154 / 0.223 / 0.389 | 1.150 |
| UR5e | Pink | 38 / 38 / 38 | 38 / 37 / 38 | 0.638 / 0.697 / 0.835 | 3.893 |

Compared with original TAR, cumulative time falls **7.19% on Panda** (paired
ratio 0.928, descriptive 95% interval 0.901–0.955) and **6.25% on UR5e**
(0.938, 0.911–0.967). Median and P95 decrease, but P99 increases on both
robots. Corrected cost remains **12.33× / 19.48× TRAC** and **4.31× / 5.76×
Pink** for Panda / UR5e. A useful reduction relative to TAR is not online
competitiveness against the mature baselines.

Panda TSR and mean DTSR20 remain 91.67% and 88.33%. Its observed UID-average
completion vector is identical, producing a degenerate conditional bootstrap
interval; this is not a population equivalence claim. Its deadline difference
interval is −4.17 to +4.17 pp. UR5e TSR changes from 97.50% to 96.67%
(−0.83 pp; interval −3.33 to +1.67 pp), and DTSR20 from 96.67% to 93.33%
(−3.33 pp; interval −7.50 to +0.83 pp). Intervals crossing zero do not establish
non-degradation. Native search repeats are not claimed to share a controllable
random seed; inferential pairing is by UID after averaging the three runs.

### All trajectory families and UID changes

Each family has ten trajectories. Cells give mean completed trajectories /10
and cumulative seconds per sweep, old → corrected. All four methods' full
family results remain in the [family table](../outputs/task_recourse/numerical_completion_development/reports/family_table.csv).

| Robot | Family | Completion old → corrected | Cumulative seconds old → corrected |
|---|---|---:|---:|
| Panda | smooth | 10 → 10 | 6.647 → 5.451 |
| Panda | near-singular | 7.667 → 7.667 | 6.748 → 6.888 |
| Panda | joint-limit return | 9 → 9 | 6.529 → 6.022 |
| Panda | high curvature | 10 → 10 | 5.820 → 5.532 |
| UR5e | smooth | 9.667 → 9.333 | 6.070 → 5.768 |
| UR5e | near-singular | 9.667 → 9.333 | 5.966 → 5.638 |
| UR5e | joint-limit return | 10 → 10 | 6.172 → 5.902 |
| UR5e | high curvature | 9.667 → 10 | 5.695 → 5.101 |

There are no Panda UID-average completion changes relative to old TAR. UR5e
trajectory_38 (`5547871d…`) changes from 2/3 to 3/3 completed runs, while
trajectory_15 (`bf166cd6…`) and trajectory_02 (`f87cbb19…`) each change from
2/3 to 1/3. These are nested-repeat changes, not stable all-success versus
all-failure reversals. In particular, trajectory_15 is **not reliably
recovered**. Full unshortened UIDs, first-failure inputs, per-repeat completion
sets, and comparisons with TRAC/Pink are saved. Different methods' evolved
previous states are not treated as identical inputs.

### Accuracy, motion and measured numerical work

Accepted position P95 improves from 0.972 to 0.468 mm on Panda and 0.978 to
0.568 mm on UR5e. Accepted orientation P95 changes from 0.005754 to 0.001588
rad and 0.007332 to 0.003644 rad, respectively. All accepted maxima remain
inside the original contract; maxima and baseline distributions are in the
main table. Frame success changes 97.19%→97.03% and 98.97%→98.39%, so smaller
accepted residuals do not imply better tracking coverage.

Trajectory-averaged acceleration RMS is 5.406→4.892 rad/s² on Panda and
12.194→12.320 rad/s² on UR5e. The motion benefit is not uniform. Changes from
backup greater than 1e−8 rad fall from 58.81% to 0.106% of Panda frames and
67.06% to 0.200% of UR5e frames. Corrected online calls verify exact zero at
their own anchors in 96.27% / 96.78% of frames. These whole-run proportions
do **not** establish that the old method could have retained its own anchor on
the same proportion of inputs: its closed-loop states and native backups differ.
Only the 80 matched mechanism inputs permit the exact-input comparison above.

Mean numerical phase times below include all frames of each method. Old TAR
has no separately measured initialization column, not a fabricated zero.

| Robot / method | Initialization ms | Fixed-current program ms | Released program ms | FK / acceptance checks ms | Total outer ms |
|---|---:|---:|---:|---:|---:|
| Panda old | — | 0 | 0.711 | 3.004 | 4.291 |
| Panda corrected | 0.215 | 0.126 | 0.030 | 3.053 | 3.982 |
| UR5e old | — | 0 | 0.720 | 2.801 | 3.984 |
| UR5e corrected | 0.207 | 0.188 | 0.014 | 2.851 | 3.735 |

The program columns include linearization, construction and native solve.
Backup, prediction, recording/accounting work are also inside total outer
time and explain the remaining components. Fewer program calls reduce that
component, but added initialization and roughly 3 ms of FK/acceptance work
limit the total saving. This is a measured cost decomposition, not evidence
from matching different closed-loop states. No stage is omitted from latency.

## A / B / C: final interpretation and stop

**A — A numerical-completion issue is established, with limited magnitude on
the matched set.** All 79 verified zero-cost inputs now return the exact anchor
and J=0. The old implementation's largest observed gap there was only
4.64e−10; most changes were rounding-scale. The future-bound reconstruction
and invalid-trial helper defects are explicitly implementation issues. This
does not support describing every previous command adjustment as necessary,
or as a large numerical failure.

**B — Better numerical completion has not added aggregate task value here.**
The matched zero-cost inputs are solved to their certified lower bound; the
complete development comparison shows no Panda completion increment and a
negative observed UR5e completion/deadline difference. We have not proved
global optimality on every remaining online input, and a failed search is not
an infeasibility proof. The result is bounded to this implementation and these
observed development trajectories, not a theorem that the objective can never
help.

**C — No additional complete-tracking benefit from this repair is established.**
There is modest compute reduction, fewer command modifications and smaller
accepted residuals, but higher P99 on both robots and substantial remaining
cost against TRAC/Pink. Isolated repeat recoveries do not justify a positive
aggregate task claim. Stop here: no new objective, network, horizon, weight
scan, formal evaluation or paper rewrite is triggered.

## Reproducibility and delivered evidence

- [Zero-cost witness table](../outputs/task_recourse/numerical_completion_development/zero_reference/zero_cost_comparison.csv), exact anchors / thirteen-node witnesses and every recheck in `zero_reference/`.
- [Selected same-input comparison](../outputs/task_recourse/numerical_completion_development/reports/same_input_table.csv), UID-level rows and full old/new output in `same_input_roundoff_corrected/`. The later guard is only reached by invalid current trials; all 79 successful fixed-anchor probe paths satisfy its precondition.
- `selection.json` and `selection_roundoff_corrected.json` retain both numerical probes; `selection_final.json` inherits the selected budget without a new search.
- Complete raw records: `validated_panda/runs/` and `validated_ur5e/runs/`; previous debug runs and their failure are retained separately, not pooled into the final comparison.
- Main/family tables, paired intervals, first failures, completion UIDs, numerical phase timings, and representative online zero witnesses are in `reports/`; `source_data.json` supplies every report number.
- The read-only audit reverified all **144,000 current commands** and **916,318 final planned nodes**, with zero accepted contract violations. All **34,750 exact-zero returns** retain their exact own backup. Recorded and recomputed true tau / objective differences are zero. All 17 outer-time exceedances remain in the metrics.
- Twelve regression tests pass, including the actual SOCP mathematics, both robot derivatives, exact-anchor/roundoff checks and rejection of an infeasible native trial. No environment upgrade.

Single entry: `scripts/run_task_recourse.py`. Actions `zero_reference`,
`numeric_probe`, `numeric_select`, `numeric_finalize`, `numeric_run --robot ...`
and `numeric_report` retain exclusive output creation, so invoking a completed
stage does not silently overwrite it. Full guarded runs used code commit
`af07c76b`; exact source hashes, CPU affinity, thread settings, input hashes and
baseline Git blob are recorded in each `started.json`. Raw records and manifests
predating this task, the verifier, historical adapters/configurations, and the
paper remain unchanged. Final reporting code was added only after both timed
runs completed and was not part of solver time.
