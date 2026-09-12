# Current-frame bounded TRF: recovery and complete online execution

Baseline `8e754127bb56ec6054ef197f7f1b4e2324611b91`, branch
`codex/hierarchical-v5`. This is a standard numerical reference and practical
two-stage baseline, not a newly named IK algorithm. No paper, old solver,
verifier, target identity or frozen result is modified.

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
The report will separate current-input recovery from complete tracking and
identify remaining observed failures without inventing their cause.
