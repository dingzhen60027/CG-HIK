# B2: official Pink constrained differential-IK adapter

This is a current-target-only baseline for the CR-IK development study. It is
not a new IK algorithm and not yet evidence of whole-trajectory performance.
No frozen solver, verifier, configuration, output, or paper was modified.

## Implementation and provenance

The adapter calls the installed **Pink `build_ik`** once, and passes its QP to
the installed **qpsolvers OSQP** interface once per query. It uses the public
`FrameTask`, `PostureTask`, `ConfigurationLimit`, and `VelocityLimit` classes;
none of their optimization or geometry code is reimplemented.

| Dependency | Installed version |
|---|---:|
| `pin-pink` | 3.3.0 |
| `pin` / Pinocchio | 3.9.0 |
| `qpsolvers` | 4.13.0 |
| `osqp` | 1.0.5 |

The checked environment is
`/home/eric/anaconda3/envs/isaaclab_3/bin/python`. Nothing was installed,
upgraded, or patched in that environment.

The live documentation presently describes Pink 4.3.0. The adapter therefore
uses the **installed 3.3.0 source** for exact signatures, also checked against
the official version-tagged source:

- [Pink 3.3.0 FrameTask](https://github.com/stephane-caron/pink/blob/v3.3.0/pink/tasks/frame_task.py).
- [Pink 3.3.0 configuration limit](https://github.com/stephane-caron/pink/blob/v3.3.0/pink/limits/configuration_limit.py).
- [Pink 3.3.0 velocity limit](https://github.com/stephane-caron/pink/blob/v3.3.0/pink/limits/velocity_limit.py).
- [Official IK interface overview](https://stephane-caron.github.io/pink/inverse-kinematics.html).
- [Official OSQP settings](https://osqp.org/docs/interfaces/solver_settings.html).

In particular, Pink 3.3.0 limits receive a configuration **array**, not the
newer documented `Configuration` object. Its `VelocityLimit` reads the model's
velocity vector; there is no newer-version constructor override assumed here.

## Fixed baseline definition

Let the incoming query supply the actual previous accepted configuration,
current target pose, and `dt`. The single QP variable is a joint displacement
`delta_q`. The nominal command is obtained by Pinocchio integration from that
same supplied previous configuration.

| Item | Fixed setting |
|---|---|
| Frame task | End link from the frozen robot configuration |
| Position cost | `1 / public position tolerance` |
| Orientation cost | `1 / public orientation tolerance` |
| Frame gain | 1.0 |
| Frame LM damping | 0.0 |
| Posture target | Actual previous accepted `q`, updated every query |
| Posture cost / gain | `1e-3` / 1.0 |
| Global Tikhonov damping | `1e-12` |
| Configuration limit gain | 1.0, not Pink's default 0.5 |
| QP solver | OSQP |
| Absolute / relative QP tolerance | `1e-9` / 0.0 |
| Maximum QP iterations | 2000 |
| Native QP time limit | 18 ms |
| QP polishing | Enabled |
| Adaptive-rho update interval | Fixed at 25 iterations |
| QPs per query | Exactly one |

These settings are declared in `PINK_SETTINGS`, not selected by trajectory
outcomes. The task costs normalize metres and radians by the unchanged task
tolerances. Pink squares these costs in its weighted quadratic objective. The
small previous-state posture term makes the redundant problem regularized
without giving the baseline a reference path or future posture.

The 18 ms native limit was synchronized with the CR-IK/RangedIK comparison
before any measured trajectory comparison, leaving a nominal 2 ms portion of
the 20 ms task deadline for outer work. It is not a measured guarantee that
this portion is sufficient. Initial adapter-only smoke calls used 5 ms; no
trajectory result or success-rate comparison was used to choose the limit.

**The Pink frame objective is not the final task constraint.** It is a local
SE(3) logarithm, with linear then angular coordinates in the end-effector body
frame. Its translational log coordinate is coupled to orientation; it is not
identical to the public Euclidean position difference. The official `Jlog6`
derivative is used. This is a soft weighted differential objective, not a hard
nonlinear pose-tolerance guarantee. A successful convex QP may still produce
a command rejected by the independent original verifier.

There is no extra candidate, TRAC-IK call, future-target prediction, nonlinear
iteration, residual refinement, or fallback. This baseline measures a standard
one-step constrained differential QP, not every possible tuning or iterative
extension of Pink.

## Bounds and endpoint handling

For each call the model-owned configuration interval is set to the original
URDF interval intersected with the original single-frame allowance:

`[q_previous - (v_max * dt + epsilon_v), q_previous + (v_max * dt + epsilon_v)]`.

The unchanged study helper `representable_interior` supplies the documented
one-ULP interior endpoints and verifies subtraction-based rate compliance.
The official configuration limit uses gain 1 so it does not introduce an
additional fraction-of-distance constraint. The official velocity limit is
also supplied with `v_max + epsilon_v / dt`. Neither the public pose nor rate
tolerance is changed, and internal QP accuracy does not override them.

After integration, a numerical box overshoot of **at most `1e-8` rad** is
clipped to the already constructed interior interval. Larger overshoots are
not clipped into apparent success. This is a disclosed adapter correction
for finite-precision bound reconstruction, not a pose-refinement step. The
raw command, overshoot, clip size/flag, and **pre/post verifier results** are
recorded. If clipping occurs, both verifier calls are included in outer
timing. Otherwise their identical outcome is reused. No additional FK is
needed for clipping itself.

The final command is accepted if and only if the original public verifier
accepts it, independently of OSQP's native return status. On rejection, the
runner must hold its previous accepted `q` while advancing the target index.
The adapter never substitutes its stored state for the supplied actual
`previous` input.

## URDF, joint order, and geometry checks

Pink necessarily uses its Pinocchio backend. Public acceptance continues to
use the existing **`URDFKinematics`** implementation; the environment name is
not used as evidence that this old verifier is Pinocchio-based.

The constructor checks the exact active-joint names/order, scalar-joint
representation, joint/rate limits, base/end frames, and fixed-base transform.
It compares FK and geometric Jacobians at five deterministic configurations,
with a `1e-11` component discrepancy ceiling. This is startup interface QA,
not solver training or a candidate pool. Targets in the public base frame are
converted through the checked fixed base-to-world transform for Pink.

Panda's order is `panda_joint1` through `panda_joint7`; its base/end links are
`panda_link0` / `panda_link8`. UR5e's order is `shoulder_pan_joint`,
`shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`,
`wrist_3_joint`; its base/end links are `base_link` / `tool0`.

The observed maximum FK translation discrepancy was `1.11e-16` m for both
robots; maximum rotation-matrix component discrepancy was `2.78e-16` for
Panda and `3.33e-16` for UR5e. Maximum Jacobian component discrepancy was
`2.22e-16` and `1.84e-16`, respectively. The exact values remain available
from `metadata()` and are not a performance result.

## API and timing

```python
adapter = PinkAdapter(kin, verifier, source_configuration, urdf_path)
adapter.reset(initial_q)
record = adapter.solve(position, rotation, actual_previous_q, dt=0.02,
                       last_target=None)
adapter.close()
```

`last_target` is accepted only for the shared runner signature and is never
read. `source_configuration` identifies the original contract; it does not
silently override baseline settings. Use one adapter instance per serial
runner. The optional OSQP stdout capture is process-global, so the adapter is
not intended for concurrent calls from multiple threads in the same process.
OSQP's unsolicited polishing message is retained as `native_stdout` rather
than flooding an evaluation log.

Outer time starts before input conversion. It includes target transformation,
query bounds, Pinocchio FK/Jacobians, Pink task/QP assembly, CSC conversion,
native QP construction and solution, integration, any endpoint correction,
and final verification. Breakdown fields are `conversion_ns`,
`bounds_setup_ns`, `qp_build_ns`, `solve_ns`, `verification_ns`, and a measured
accounting remainder. Setup once per adapter and later JSON serialization or
offline metrics are not query execution time. **An 18 ms OSQP limit is not a
20 ms outer deadline guarantee.** Both actual 5/20 ms return flags and accepted
within 20 ms are returned separately.

OSQP's own setup, solve, polishing, and total run durations are additionally
retained as `native_setup_ns`, `native_solve_ns`, `native_polish_ns`, and
`native_run_ns`. They are nested backend timings, not extra disjoint terms to
add to the measured outer phase sum.

The records include native code/message/iterations and primal/dual residuals,
returned/raw configurations, public errors/reasons, per-joint rate
utilization, limits, and clip outcomes. `solver_function_evaluations` is
`None`, not an invented FEV conversion from QP iterations. Separate counters
identify the single joint-Jacobian/FK update, frame error/Jacobian requests,
and actual verifier calls.

## Verification scope

`tests/test_crik_pink.py` contains eight passing smoke/unit cases across both
robots: geometry and ordering, exact/small-step targets, current-only input,
actual-state precedence, rate endpoints, native-failure/task-accept semantics,
rejection feedback, and bounded raw/post-correction verification.

Only small adapter smoke calls have been run here. They demonstrate that the
baseline is callable and returns real verified joint commands; they do not
establish trajectory success, tail latency, or superiority. Complete-method
evaluation belongs to the main CR-IK runner and must preserve all failures.

A final eight-call smoke batch per robot, using the 18 ms native limit, used the deterministic center plus
`linspace(-0.08, 0.08, nq)` as previous state and joint-space perturbations
`0.003 * sin(arange(nq) + 0.1*j)` to construct current FK targets. It produced
8/8 native and public successes for each robot, with no endpoint clipping.
These are independent adapter calls, not a tracking trajectory and not eight
independent experimental trajectories. The observed outer times, including
the first cold QP call, were:

| Robot | Eight-call median (ms) | Maximum (ms) |
|---|---:|---:|
| Panda | 1.364429 | 2.227227 |
| UR5e | 1.497931 | 1.735703 |

The full measured vectors in milliseconds were
`[2.227227, 1.536332, 1.443884, 1.372759, 1.353062, 1.336775, 1.356099, 1.305706]`
and
`[1.702895, 1.549407, 1.486452, 1.452930, 1.457229, 1.417446, 1.735703, 1.509409]`.
Native solve times in the same calls ranged from 0.029657 to 0.037569 ms for
Panda and from 0.122757 to 0.430952 ms for UR5e. The much larger outer times
illustrate why native QP time alone is not the online cost metric.
These smoke numbers are disclosed to identify the implementation check only;
they must not be copied into a paper's main latency comparison.

Run the checks without changing the environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/eric/anaconda3/envs/isaaclab_3/bin/python -m pytest -q tests/test_crik_pink.py
```

Files added by this bounded adapter task are
`src/confik/correction_reserve/pink_adapter.py`, `tests/test_crik_pink.py`, and
this document. No trajectory experiment, commit, or push is performed by the
adapter task itself.
