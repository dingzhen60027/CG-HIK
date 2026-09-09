# CR-IK: demand-satisfying minimum intervention — development protocol

Baseline: `156da7b773fa2d8708babac45f8e8f0e338abd30`. This is the user's explicitly
authorized next CR-IK experiment, not a change to the earlier paper or evidence.
Only original development targets are used. The latest 160 formal trajectories
and their outcomes are not read by parameter preparation, runtime or this study.

## Fixed demand definition

Keep the original extrapolation and world-frame split position/SO(3) residual.
Let eta_t be the Euclidean norm of `pose_error(actual_target_t,
prediction_made_at_t_minus_1) / D`. The first target has no eta; its forecast holds
that first target. Subsequent targets, including those after command failures,
advance observed history. W=20. Empirical quantiles use the inverse empirical CDF:
the sorted observation at `ceil(p*n)-1`, without interpolation.

Before any comparative solver outcome, calculate each robot's **minimum demand
as the median** of all eta values in its old development target sequences. Use
their **95% quantile for initialization** until 20 errors have been observed.
The same initialization value is the constant r in the fixed-demand ablation.
These are target-only engineering conventions, not values chosen by solver
completion. They are reported with their source hashes and never searched.

After startup, r_t=max(r_min, empirical_Q95(last 20 eta)). No upper clipping or
probabilistic coverage guarantee is used. A large unmet demand remains visible.

## New numerical mode

Do not edit the old `runtime.py`, `convex.py`, `geometry.py`, public verifier,
TRAC-IK or Pink. `MinimalInterventionIK` reuses their interfaces/geometry in a
separate mode. Current q and nominal next z remain the only optimization nodes;
gamma is no longer an optimization variable. Minimize

`||S^-1(q-q_backup)||^2 + ||S^-1(z-z_initial)||^2` (lambda=1).

Retain both true task contracts and the original SO(3)-consistent right inverse.
With its row norms fixed in each convex subproblem, require r_t times each row
norm to fit the nominal next step's rate and joint-range slack. A full-rank right
inverse is required for a positive six-dimensional demand claim. After solving,
recompute true FK, both original verifier checks and the map/slack at the returned
pair. A solved linear cone alone never sets `demand_met=True`.

Get one task-aligned TRAC-IK 5 ms backup. Construct initial z by the existing
single linearized prediction step and representable interval clipping. When a
verified previous nominal exists, it is only the starting point for that step;
it is clipped to the **actual current backup's** next-step interval. It is never
executed. Reset nominal and demand history at every trajectory.

If backup/current and nominal-next contracts plus demand already hold, return the
identical backup array without constructing or solving a cone. Its original
verifier result is reused, not redefined. Otherwise solve one minimum-intervention
SOCP, allowing at most one additional relinearization. Keep the old first-feasible
backtracking factors 1, 1/2, 1/4. Accept an improved pair only after true demand
and both contracts hold. Otherwise return the existing legal backup and explicitly
report demand unmet. When backup fails, use the unchanged same-core ordinary
predictive feasibility subproblem for at most two updates. A recovered legal
pair may be returned with demand unmet; if none is found, report failure and hold
the actual accepted state. No demand failure is called mathematical infeasibility.

All original trust, interior, rank, native iteration/time and outer soft-limit
values are retained. There is no primary gamma solve, lexicographic secondary
solve, extra residual objective, candidate pool, extra horizon, trained module
or temporal anchor. Only the demand, objective and associated avoidable-work
shortcut change. Two control modes: fixed r; and disabling the initial shortcut.
The latter still retains an already feasible zero-objective pair after its forced
solve, so numerical noise does not create a gratuitous command intervention.

## Caching and timing

Keep installed Clarabel 0.11.1, Pinocchio 3.9.0 and the frozen TRAC/Pink versions.
The new SOCP has fixed dimension 2*n, linear-cone rows and four pose norm cones.
Cache its CSC pattern, including structural zeros, and update numeric data. Disable
presolve, chordal decomposition and zero-dropping as required for updates; no
unsupported native primal/dual warm-start is claimed. This follows the
[official data-update API](https://clarabel.org/stable/user_guide_data_updating/),
checked against the installed version. Cache construction is lazy and included
in the first frame that actually needs it.

Outer timing includes conversion, complete backup call, demand estimation and
prediction, initial-pair construction/check, cone construction/update/solve,
nonlinear checks, command selection and state-bookkeeping. Only result-record
serialization is excluded, as in the old runner. Log these phases separately.
Backup subcall fields remain explicitly prefixed. The retained 18 ms soft stop
is not a hard 20 ms execution guarantee. No timing outlier is removed.

## One complete development comparison

Reuse exactly the old 40 Panda and 40 UR5e target sequences, 150 frames each,
10 per original family. No new sequence, favorable intervention or failed-path
replacement. Methods: task-aligned TRAC 5/20 ms, frozen Pink, old CR-IK, old
ordinary two-step prediction, new minimal mode, fixed-r and no-shortcut controls.
Do not run RangedIK, old single-step or sigma ablations. All TRAC-containing
methods get three complete nested repeats; Pink gets one.

Reuse the old feedback-correct `execute_trajectory`: every target advances,
only accepted q updates actual previous_q, and all 150 frames are processed.
q_ref and true next targets are not solver inputs. Each robot's jobs are shuffled
with a fixed seed and run serially on its old CPU affinity (Panda 0/2, UR5e 4/6),
allowing the robots to run concurrently on disjoint cores. BLAS/OpenMP=1.

Before launching the complete comparison, save target identities, target-only
demand parameters, code/config hashes, tests and the protocol in a commit.
Short stationary/synthetic adapter unit tests may debug implementation errors;
they do not select demand or optimization parameters. Run one complete development
comparison afterward, preserving any interrupted records instead of overwriting.

## Outcomes and analysis

TSR and DTSR20 remain primary; retain every completion UID, gained/lost UID,
first failure, frame result, full outer latency, residual, velocity and acceleration.
Report all four families and both robots. New outcomes:

- fraction with a sequential-convex SOCP call and number of calls;
- exact backup equality, normalized joint intervention and >1e-8 rad command
  change rate (the old reporting resolution, not an algorithm decision threshold);
- accepted commands using >90% of either public pose tolerance;
- demand, observed eta and **offline-only** true next-target prediction error;
- empirical coverage of that error by r_t, including startup;
- actual next-frame success conditioned on met/unmet demand, with current-failure
  and accepted-current denominators distinguished;
- shortcut-only overhead = total outer time minus the timed complete backup call.

The offline next-frame join is computed only after each complete online run. It
does not feed back into demand, initialization or solving. Conditional outcomes
are descriptive, not proof that demand status caused the difference.

Whole trajectory UID is the independent unit. Average searches within UID, then
use paired family-stratified bootstrap differences/ratios (4000 resamples,
unadjusted descriptive 95% intervals). Do not count frames/searches as independent
trajectories. Compare minimal to old CR-IK, ordinary prediction and mature solvers;
the fixed-r/no-shortcut contrasts separate the two specified changes. No total
pass/fail gate, post-outcome tuning, further algorithm, or new evaluation is
authorized by an interesting result. Stop, report and push after this comparison.
