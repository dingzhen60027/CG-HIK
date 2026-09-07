# Task-contract alignment: fixed two-robot supplementary protocol

Evidence baseline: `02aa287b65d4f1b7bb5317d8e83fcf972559103f`.
This protocol precedes all new measured outcomes. It is a supplementary engineering
study, not a new learned method, parameter-selection exercise, or global pass/fail
test. Point inputs have previously been observed; new UR5e trajectories are fixed
before solver execution. The earlier Panda results remain authoritative, unchanged.
Paper rewriting starts only after this study's report and results have been pushed.

## Question and endpoints

**Solver convergence is not command admissibility.** For the online input
`xi=(desired pose, actual previous accepted q, dt)`, distinguish:

1. Internal success: the solver's original return code/convergence flag.
2. Task admissibility: the unchanged independent `SolutionVerifier` accepts q.
3. Within-deadline admissibility: task acceptance AND outer latency <=20 ms.

The four internal/task cells are recorded separately. A witness-confirmed missed
admissible query requires an independently retained configuration for the EXACT
target, previous_q, dt and contract, accepted by the same verifier, while the tested
call returns no accepted command. Native failure is never mathematical infeasibility.
The unique-query number with any miss, with all repeats missed, and the average
miss count per sweep are different reported quantities.

## Frozen contract and implementation

Nominal position norm <=0.001 m; orientation geodesic norm <=0.00872664626 rad;
dt=0.02 s; original URDF joint ranges and velocities; velocity allowance
`velocity_i*dt+0.0001 rad`. The existing verifier also has a 1e-9 rad joint-bound
floating-point allowance: it is retained, not silently rewritten as exactly zero.
All values must be finite. Dynamic solver domains use the strict URDF intersection
with the single-frame interval and the existing `representable_interior` construction.
This handles already documented floating endpoint issues without loosening acceptance.
Both supplied robots have bounded revolute, not continuous, active joints.

Actual public FK/Jacobian backend: repository `URDFKinematics`. TRAC-IK uses the
same URDF and joint ordering through KDL. No Pinocchio inference from environment
names; no dynamics, collisions or hardware control. The native 2.2.0 sources at
`90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb` remain unchanged. Speed mode,
previous_q initialization, epsilon=1e-5, same two native search workers.

### Native tolerance mapping

For every aligned position component use `nextafter(epsilon_p/sqrt(3),0)` metres;
for every aligned rotation component use `nextafter(epsilon_R/sqrt(3),0)` radians.
Unaligned components get zero additional bounds. `KDL::diffRelative(target,actual)`
uses target-frame translation `R_d.T*(p-p_d)` and rotation vector
`Log(R_d.T*R)`, not RPY. Native code zeroes bounded components before its epsilon
test. Its effective per-component limit is max(bound,epsilon), not their sum.
All tested scales keep aligned bounds above epsilon. The effective 3-vectors of
limits must have norms no greater than the public limits.

This is an **inscribed component box, not equality with the public norm balls**.
DLS stops directly on position/orientation norms. Even the two strict solvers'
internal acceptance sets are not identical. Public verification is identical.
All formulae, frames, units, native bounds and their effective limits are saved.

| Setting | Position | Orientation | Budget |
|---|---|---|---|
| TRAC strict | zero additional bound | zero additional bound | 5 and 20 ms |
| TRAC position-aligned | inscribed position box | strict | 5 ms |
| TRAC orientation-aligned | strict | inscribed rotation box | 5 ms |
| TRAC fully aligned | inscribed position box | inscribed rotation box | 5 and 20 ms |
| DLS strict | 1e-5 m norm stop | 1e-5 rad norm stop | 25 iterations |
| DLS task-aligned | public position norm | public orientation norm | 25 iterations |

DLS reuses `AdaptiveDLS` unchanged, including damping, sigma threshold, orientation
weight, line search, max joint step, minimum improvement and stagnation. A common
kinematics wrapper restricts clipping to the actual query's allowable domain.
Internal step clipping does not replace final total-step verification. Every raw
return is verified, including internal failures; no extra seed, retry or refinement.

## A. Fixed point queries

Read exactly the frozen `revision_compute_allocation/02_point_mechanism_benchmark`
NPZ and identities: per robot 1,000 local, 500 near-singular, 500 near-limit,
500 hard-feasible, 500 constructed inexecutable. All 2,500 witnesses are replayed
before evaluation. Inexecutable targets exceed a sum-of-URDF-link-length global
position reach bound; failed solver calls are not their label source.

All eight settings above use identical point input; previous_q never updates.
TRAC has three within-query search repeats; DLS runs once. Per query, each repeat
interleaves a fixed-RNG permutation of eligible methods (DLS only in repeat 0).
Query order is also fixed-RNG shuffled. Native random search seeds are not controlled
or claimed to be paired. Total: 120,000 new calls across 6,000 independent queries.
The primary contrast is fully aligned / strict TRAC at 5 ms; 20 ms distinguishes
search-budget effects, while position/orientation arms separate component effects.

Raw output retains UID, family, witness link, repeat, internal status/native code,
q, finiteness, residuals, joint/rate results, task/deadline results, native bounds,
actual interval, complete timing and four-cell classification. Report feasible and
inexecutable populations separately; never let the latter's time savings hide the
former. Error distributions include accepted command P50/P95/P99/max and normalized
public-tolerance ratios, with all returned values preserved in raw records.

### Direct DLS iteration observation

For both DLS point settings (equal instrumentation), a wrapper records actual FK
evaluation q/pose and identifies the main-iteration call site in the unchanged
`AdaptiveDLS.solve`. It does not change solver arithmetic or rerun solves. After
timing, replay the original verifier on those evaluations and retain complete
world-frame residual vectors and the main-iterate flags. The first main iterate
meeting the contract defines the first admissible iteration; any earlier legal
line-search evaluation is also recorded separately. Compare actual stopping iteration
and evaluation count. Term: **excess iterations after task admissibility**.
No unobserved TRAC internal process is labeled over-solving. Extra time is not
estimated by subtracting separate runs. Point DLS timings include lightweight logging;
trace processing/reverification is excluded. The unchanged uninstrumented DLS wrapper
is used for trajectories to match the authoritative Panda measurement scope.

## B. Contract scale sensitivity

Before any method runs, select 500 per robot by ascending UID within fixed families:
200 local and 100 in each other feasible family, requiring the witness to pass the
0.5x pose contract. Preserve selection identities; no solver-result screening.
At 0.5x, 1x, 2x scale only BOTH pose tolerances change, never target, previous_q,
joint/rate limits or dt. Recompute native bounds using the same formula. Run strict
TRAC 5 ms and aligned TRAC 5 ms three times each, DLS task once, at each scale:
21,000 calls total. Report all scales; nominal remains primary. A missing eligible
family is a protocol blocker, not permission to replace inputs or loosen witnesses.

## C. Complete online trajectories

Panda: directly read all 560 runs / 84,000 calls from the frozen tolerance comparison.
No remeasurement, output overwrite, reinterpretation of original counts or replacement
of its timing values.

UR5e: exactly 40 new seeds 971008000--971008039, ten per predefined family:
smooth / near-singular / joint-limit-return / high-curvature. Same geometry-only
generator as Panda, 150 target frames, dt=0.02. Verify all 6,000 q_ref transitions.
New UID/seed/query-hash isolation is checked against stored identity metadata; save
the scan coverage and do not claim coverage of unrecorded identities.

All six trajectory settings (exclude position/orientation-only) reuse the exact
Panda `ToleranceSolver`, old frozen native library, parameters and timing. Three
complete TRAC repeats, one deterministic DLS run; 560 runs / 84,000 new calls.
Initialize all at q_ref(0); target frame 0 corresponds to the first transition to
q_ref(1), exactly as the Panda generator. The online file strips future reference
joints. Each solver sees only current target, its actual previous accepted q, dt.
Only public acceptance updates previous_q; otherwise hold it and advance the target
index normally through all 150 frames. No witness seeds or future lookahead.

Reference feasibility proves a path from reference states, not from every
method-specific state. Geometrically legal late commands update the geometric loop,
but count as late; this does not simulate deadline-triggered control or hard 50 Hz.
All runs remain, including failures, and use complete trajectories as statistical units.

## Timing, statistics and preservation

Outer latency starts before necessary input conversion and includes dynamic bounds,
native setKDLLimits setup, numerical search, final independent verifier and intervening
Python/ABI costs. Point records split conversion, bounds/setup, solve, verification
and accounting remainder without dropping overhead. Native timing instrumentation
only wraps the original calls. Model/chain initial loading, offline statistics and
serialization are excluded. No warmup query from either measured set is discarded;
smoke uses separate static geometric fixtures. DLS iterations are not an equivalent
5/20 ms budget: report measured within-5/20-ms proportions.

Same recorded Python/numerical environment as Panda, BLAS/OpenMP threads=1,
TRAC native workers=2, serial measured runs on the shared workstation. No concurrent
tests/benchmarks during measurements. Record hardware/software and library hashes.

Primary summaries include all returns and full fixed target sequences. Query UID
or trajectory UID is the independent unit. Average search repeats within each unit
for paired success differences and cumulative/mean cost ratios. Use 4,000 paired,
family-stratified percentile bootstrap resamples, seed 971008902, 95% intervals.
Whole-call frame quantiles are descriptive, not independent-frame inference. Preserve
per-repeat totals/completion and per-UID all/any repeat success. No pseudo-independent
replicates, outlier exclusion, winsorization, multiplicity-based significance claim,
sample replacement, tuning or global gate. Cost ratios and success are always reported
together. Family comparisons are complete descriptive analyses, not selected winners.

Strict-to-aligned recovery/loss counts compare per-UID fractions across all fixed
repeats, not imagined common random seeds. Report stable 0/3->3/3 separately from
partial recovery. Point matrix counts are raw calls plus per-sweep averages, not
inflated unique query counts. Trajectory cumulative cost is normalized to one full
40-path sweep (TRAC mean of three sweeps), preserving old Panda published values.

## D. Allocation boundary, read only

Read the existing direct-hard, geometry, reject-only, routing-only, P50-selection,
full CG-HIK, external TRAC and witnessed-trajectory result tables. No training or
new allocation call. This is a secondary case study and **not a new randomized
comparison of a retrained router on aligned solvers**. Preserve the findings that
entry routing does not generally accelerate successful solving, similar geometry
rules exist, P95-selection tail evidence is weak, aggregate savings concentrate in
joint failures, and rejection can avoid unsuccessful numerical work. The older
TRAC comparison's endpoint representation issue remains disclosed, not overwritten
or counted as newly recovered numerical mismatch in the current corrected study.

## Execution and stopping order

1. Read old evidence; prepare this protocol and code without editing paper.
2. Freeze point/sensitivity/UR5e identities and mappings; commit protocol + code.
3. Build independent adapter; run smoke and unit tests; save execution seal.
4. Once only: point study, sensitivity, UR5e online runs, in that order.
5. Read-only aggregation with authoritative Panda and allocation tables; independent
   verifier replay; write findings, source data, figures/tables and manifest.
6. Commit and push complete stage-one evidence; confirm remote SHA.
7. Only then reconstruct manuscript from frozen evidence; no further experiments.

All measurement sources, old tracked evidence and input files are hashed. Output
creation is exclusive; started markers block accidental repeats. If an implementation
error prevents execution, stop and disclose it rather than choosing results. Old
papers, solver/config files, outputs/manifests and identities remain untouched.
After these two stages: no new models, solvers, gates, preview, OOD or robot domains.
