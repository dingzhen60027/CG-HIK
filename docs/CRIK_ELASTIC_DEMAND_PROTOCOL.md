# Elastic correction demand: fixed development protocol

Baseline: `2a24dd37604e7c3e9f82f48eb3f782f8623fd509`, branch `codex/hierarchical-v5`.
This is CR-IK development on already observed trajectories, not a fresh test.
Old algorithms, verifier, paper, configurations, manifests and evidence are not edited.

## Question and fixed change

Does permitting a penalized correction-demand shortfall yield useful partial
command improvements that improve complete online continuation?
The only new algorithmic objective is

`0.5 ||S^-1(q-q_backup)||² + 0.5 ||S^-1(z-z_initial)||² + 0.5 mu (xi/r)²`.

Prediction, causal W=20 demand, its median floor and startup Q95 values are reused
byte-for-byte from the preceding development protocol, without refitting. There
is no demand cap. No actual future target or reference joint sequence is given
to a solver. The actual next-error join is offline, after each entire run.

Current q and predicted-next z retain the original 1 mm / 0.5 degree pose norms,
URDF joint bounds and actual single-frame velocity contract with dt=0.02 s and
the original velocity tolerance. Only the additional correction demand is
elastic. The original deterministic verifier has exclusive acceptance authority.

## Numerical definition

The native variables are `(q-previous)/S`, `(z-previous)/S`, and `h=xi/r`, with
`0<=h<=1`. Here `S=diag(velocity*dt+velocity_tolerance)`, unchanged. The fixed
SO(3) residual derivative and right-inverse map B are reused. For each joint,
`r*(1-h)*||B_i||` must fit all three original remaining step/range margins.
All four pose norm cones remain hard. Relinearization uses the existing trust
step, pose-interior value, native iteration/time limits and 18 ms outer soft
limit. These soft solver limits do not prove a 20 ms end-to-end deadline.

There is one quadratic-objective SOCP, at most one further linearization, and
the existing `1, 0.5, 0.25` first-improving nonlinear backtracking rule. No gamma
maximization, primary/secondary reserve solves, new candidates, horizon, learning
model or acceleration objective is added. Explicit CSC sparsity is cached and
updated under installed Clarabel 0.11.1 restrictions, as in Minimal. The zero-mu
quadratic coefficient is structurally stored; variable and constraint patterns
are identical across mu. No unsupported native primal/dual warm-start claim is
made. Previous nominal z is only an initialization, never an executable command.

If the map lacks full six-dimensional rank/compensation accuracy, the SOCP
forces h=1 and checks only the unchanged hard geometry at that linearization;
the actual reserve is zero until a recomputed full-rank map exists. No clipped
singular-value substitute or positive six-dimensional reserve is claimed.

## Runtime acceptance and recovery

1. Obtain the existing task-aligned TRAC 5 ms backup and unchanged nominal initial
   prediction. A legal initial pair with full demand met returns the exact backup
   without building any cone, including in the mu=0 control.
2. Otherwise, evaluate every trial with true FK and the original q/z verifier.
   Recompute B, actual gamma, `xi_actual=max(0,r-gamma)`, and the actual objective.
   Linear cost or linear xi cannot authorize a return.
3. When the initial pair is legal, replacement requires objective reduction
   strictly greater than `1e-10 + 1e-9*abs(reference_objective)`. This is a
   numerical comparison tolerance only, not relaxed pose/rate acceptance.
   At most two SOCP linearizations are attempted; stop after the first genuine
   nonlinear improvement. A legal pair's actual objective, rather than demand_met,
   governs improvement. Full, partial, no-improvement and recovery are distinct.
4. When the initial pair is unavailable or the backup fails, all mu settings use
   the unchanged ordinary predictive feasibility subproblem from the existing
   core, within the same two-update budget. A true legal pair may restore
   feasibility without meeting correction demand; it is labelled geometric
   recovery, not reserve-objective evidence. If none exists, retain any verified
   backup or report failure. Failed output never updates feedback.
5. All trial q/z, actual costs, shortfalls, rank, task rejection reasons and native
   statuses are logged. This addresses the old records' missing rejected-trial
   detail without rerunning old queries or rewriting old evidence.

Timing includes input conversion, backup solve/verification, demand estimation,
initial-pair creation/check, cone construction/update/solve, true nonlinear
evaluation, selection, and final verification. Serialization and offline
statistics are outside command-ready timing. No phase is omitted for speed claims.

## Existing-log localization

Only the previous hard-Minimal 240 raw trajectory runs and corresponding first
failure table are inspected. Fallback flags may overlap. Old native-status
records do not preserve rejected trial q/z or costs: solved-but-not-selected
cannot be separated into nonlinear geometry rejection versus demand-only
rejection or claimed to be useful objective improvement. Those unavailable
counts stay null rather than being inferred. The full UID of trajectory_15 is
retained, and its first-joint upper-bound proximity is an observation, not a
unique causal attribution to singularity, branch choice or shortcut.

## Complete comparison and statistics

Exactly the existing 40 Panda and 40 UR5e target trajectories are reused, four
families of ten, 150 frames each. Every method starts at frame 0, has identical
targets and initial state, updates only on verified output and otherwise holds
the last accepted state while the external target index advances. No favourable
mid-trajectory handoff. All failures, categories and frames remain in results.

Nine configurations: task TRAC 5/20 ms, fixed Pink, original CR-IK, hard Minimal,
elastic mu=0 (same-core no-reserve-cost control), and mu=0.25/1/4. The control is
not a reproduction of a predictive-IK paper. Eight TRAC-containing configurations
have three whole-trajectory search repeats; Pink has one. Native TRAC seeds are
not controllable and are not claimed matched. Job order is fixed at 982009201;
one stationary startup call per solver is followed by trajectory reset. Total:
2000 trajectory runs and 300000 online frame calls. Robots run on disjoint CPU
sets 0,2 and 4,6; OMP/OpenBLAS/MKL thread counts are one. Installed dependencies
and native-library hashes are retained, not upgraded.

TSR and DTSR20 are primary. Complete UID is the independent unit. Average the
three searches within each UID before paired family-stratified bootstrap (4000
resamples, seed 982009202). Report all mu comparisons with mu=0, hard Minimal,
original CR and mature baselines. Intervals are descriptive unadjusted 95%
intervals, not multiplicity-corrected hypothesis tests or equivalence tests.
No global gate, no excluded failures, no favourable-mu selection this round.

Full main/family tables include whole-trajectory completion, deadline completion,
frame success, pooled descriptive latency quantiles, aggregate and per-trajectory
latency, legal residuals, acceleration, command intervention and gained/lost UIDs.
Additional records distinguish partial adoption, actual shortfall decrease,
solver failure, nonlinear rejection, no objective improvement, optimization calls
and effective command changes. Conditional next-frame outcomes are descriptive,
not independent frame samples or causal tests. Report trajectory_15 alongside all
other losses; it cannot determine a robot-specific mu.

The <=1.25 hard-Minimal cost ratio and preservation of completion are development
aims, not a result-driven combined gate. Whatever the result, stop after this
one comparison, report every mu, commit and push. No formal evaluation, paper
rewrite, retuning or additional module follows automatically.
