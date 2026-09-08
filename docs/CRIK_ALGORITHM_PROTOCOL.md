# Correction-Reserve IK: algorithm and evaluation plan

Working hypothesis, not a performance result. This study supersedes the earlier
stop-development instruction at the user's explicit request; it does not revise
the frozen task-contract manuscript or any earlier evidence.
Incoming commit: 276f3d311aafc67786c916c5fec596dd860cc5dc.

## Decision and information

CR-IK optimizes the current executed q and a nominal next z, using only actual
previous accepted q, current target, and last observed target. Translation uses
constant-increment extrapolation; orientation uses R_t Exp(Log(R_{t-1}^T R_t)).
First-frame prediction holds the current target. A failed command does not stop
target history from advancing. The nominal z is never executed or treated as a
witness for the actual unknown future target.

The original nominal contract remains 1 mm, 0.5 degrees, dt=20 ms, original URDF
limits, per-joint rates and velocity allowance. The existing verifier is unchanged.
Current and nominal next configurations both undergo nonlinear FK and that verifier.
The current query has a task-aligned TRAC-IK 5 ms backup obtained first. A coupled
search can recover an input for which that backup failed, but only a verified q
can be submitted. An unsuccessful optimization returns the backup, not an
unverified trial iterate.

## Local correction map and scope of the bound

Let e=[p_target-p(z); Log(R_target R(z)^T)], D=diag(epsilon_p I,
epsilon_R I), and S=diag(b), b_i=velocity_i dt+epsilon_v. All pose derivatives use
world axes. Perturb the predicted target by p'=p+D_p dy_p and
R'=Exp(D_R dy_R) R. Define

A = -D^-1 partial(e)/partial(z),
C = diag(I, J_l^-1(e_R)),
B = S (A S)^right_inverse C.

The q-side rotation derivative is -J_r^-1(e_R) J_angular and the target-side
derivative is J_l^-1(e_R). Thus A B=C, including at nonzero nominal orientation
residual. Ignoring C is generally not exact. The weighted right inverse is formed
by an untruncated full-row-rank SVD. A relative rank test or failed right-inverse
residual check yields no positive six-dimensional correction certificate.

For ||dy||_2<=gamma, each correction satisfies
|B_i dy|<=gamma ||B_i||_2. If gamma ||B_i||_2 <=
min(b_i-|z_i-q_i|,z_i-q_min_i,q_max_i-z_i), then z+B dy stays inside the
linearized step/range box. The first-order pose-residual change cancels because
C dy-A B dy=0. This is a bound for one specified local linear correction policy,
not a globally best redundancy resolution, a nonlinear robustness certificate,
or a guarantee about all future targets. Actual perturbation testing uses FK and
the original verifier; a larger gamma is not itself a robustness result.

## Native numerical implementation

Use Clarabel 0.11.1 directly on small SOCPs, no CVXPY, multistart or candidate pool.
Scale q/z variables by S. Freeze Jacobians and B within each subproblem.
The norm-ball constraints are affine-linearized SOCs; absolute-step/reserve/range
inequalities are linear. At most three relinearizations are allowed.
Current accepted commands are selected only after true nonlinear checks.
Optimization uses cached Pinocchio analytic geometry from the identical URDF,
with construction-time FK/Jacobian equivalence checks; the final verifier still
uses the original URDFKinematics backend. Backtracking tries factors 1, 1/2, 1/4
and stops at the first nonlinear-feasible trial. These are not a candidate pool.

The first solve maximizes gamma. A second convex solve minimizes the stated
normalized joint movement plus current residual within 0.1% (minimum 1e-8)
of its gamma optimum. This is a numerical lexicographic tolerance, not a fitted
weighted gamma score. A fixed small pose interior is numerical conservatism,
not a relaxed public tolerance. Every setting is in correction_reserve.yaml.
The 18 ms outer soft limit stops new optimization work, but does not establish
a 20 ms worst-case execution bound. All preprocessing, backup, prediction,
linearization, conic setup/solve, selection and final verification are timed.

## Comparators and ablations

- B1: unchanged task-aligned official TRAC-IK, 5 ms and prescribed 20 ms reference.
- B2: official installed Pink, one constrained differential QP, same final verifier.
  Its body SE(3) objective is not identical to the public split-norm contract.
- B3: official RangedIK original-cutoff implementation; also retain a separately
  named positive-range adapter as a range-activation sensitivity comparison.
  Both retain the exact published losses, non-collision weights and optimizer.
  The original does not activate its ranged loss at our small tolerances; the
  adapter does, but is not an unmodified reproduction. Neither beating the
  original with disabled ranged terms nor beating a poorly behaving adaptation
  alone establishes superiority over RangedIK. See CRIK_RANGED_ADAPTER.md.
  Self-collision objective terms are disabled in both: they impose an extra task
  absent from every other method and the shared contract.
- B4: same-core two-step predictive IK. Identical prediction, feasible constraints,
  trust region and outer budget; minimize ordinary movement/current residual
  without reserve maximization. Not a reproduction of the 2014 predictive paper.
- A1: single-step reserve. Optimize the current pose set and current correction
  allowance from previous q, with no predicted target or z.
- A2 is B4, not another duplicate method.
- A3: same two-step constraints, maximize classical sigma_min(D^-1 J_geometric S)
  using a finite-difference first-order objective. It does not use the reserve map
  as its objective. All derivative overhead is timed.

Library choice and ability to change q do not establish novelty or performance.
RangedIK already exploits task ranges; predictive IK already uses look-ahead.
The claimed increment remains unproven until reserve beats both adequate existing
methods and the same-core predictive control.

## Stages and measurements

Development uses existing observed Panda/UR5e trajectories and failure records.
Initial integration uses first UID per family; full development includes every
old trajectory. Parameters may be debugged here, with every run retained, before
formal freezing. No old record or manuscript is overwritten.

After implementation and development, fix code, settings and all new identities.
The planned formal set is 80 new trajectories per robot, 20 per family, 300 frames.
Families: smooth, near-singular, joint-limit-return, high-curvature/turn/speed
changes. The last family includes prespecified abruptness levels, not only
constant-velocity motion. Geometry establishes feasibility before outcomes.
No outcome-dependent screening or source q_ref enters an online method.
All targets advance; only accepted q updates feedback. TSR and DTSR20 are primary.

TRAC and methods containing TRAC use three complete search repeats. Deterministic
Pink/Ranged runs are not repeated as independent evidence. Independent units are
whole trajectory UIDs; search repeats remain nested. Report paired completion
differences and cumulative latency ratios with trajectory-level 95% intervals,
all families and ordinary-motion regressions. No total gate or all-metrics-win
requirement. Every returned frame, rejection and over-deadline call is retained.
Whole-robot runs may execute concurrently on disjoint physical cores: Panda
CPUs 0/2, UR5e CPUs 4/6. Every method for a robot uses the same affinity;
BLAS/OpenMP are single-threaded and TRAC retains its native search workers.
Analysis work uses other cores. This is software timing, not hard-real-time proof.

Auxiliary outcomes: whole-call P50/P95/P99; cumulative time; fallback fraction;
position/orientation P95/max and tolerance use; normalized joint steps; velocity
change/acceleration RMS; first failure and successful prefix; gained/lost UIDs.

Mechanism probe positions are fixed before continuation outcomes. At common
previous/target/history inputs, evaluate each actual returned q against fixed
signed-axis perturbations of the predicted next pose, using the same TRAC-IK
20 ms continuation search and three nested repeats. The actual next target's
prediction error is separately recorded offline. Failure to find a perturbed
command is not proof of nonexistence. Compare empirical perturbation success
with gamma; do not equate them.

## Preservation and stopping

New code, records and reports live in correction_reserve paths. Existing models,
runtimes, identities, results and paper remain unchanged. No training, learned
routing, six-candidate projection pool, periodic anchors, 30-frame online solver,
new robot, collision or contact task is part of this work. Formal outcomes will
not trigger parameter changes or another automatic algorithm branch.
