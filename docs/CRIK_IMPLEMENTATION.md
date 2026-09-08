# Correction-Reserve IK implementation

This is an executable online algorithm and a hypothesis test, not a claim of
superiority. The earlier manuscript and all earlier evidence remain unchanged.

## What actually runs

`src/confik/correction_reserve/runtime.py` produces a current joint command on
every target frame. It first obtains an independently verified task-aligned
TRAC-IK backup, extrapolates only already observed targets, and solves up to
three small two-configuration convex subproblems through native Clarabel.
True FK and the original verifier govern both the current and nominal next
configuration. Only the current verified q is fed back. Rejected outputs never
update feedback; target indices and observed-target history still advance.

```python
from confik.correction_reserve.runtime import CorrectionReserveIK

solver = CorrectionReserveIK(kin, verifier, source_config,
    trac_library_path, urdf_path, mode="reserve")
solver.reset(initial_q)
previous = initial_q.copy()
for target in current_target_stream:
    result = solver.solve(target.position, target.rotation, previous, dt=0.02)
    if result["accepted"]:
        previous = np.asarray(result["q"])
solver.close()
```

Modes `predictive`, `single` and `sigma` are the fixed same-core controls, not
additional learned policies. Detailed equations, local-bound proof and parameter
values are in `CRIK_ALGORITHM_PROTOCOL.md` and `configs/correction_reserve.yaml`.
The optimizer's native Pinocchio FK/Jacobian is checked against the unchanged
public URDF backend. The latter still performs final verification.

## Dependencies and reproducibility

Measured environment: Python 3.12; Clarabel 0.11.1; Pinocchio 3.9.0; Pink 3.3.0;
qpsolvers 4.13.0; OSQP 1.0.5; the frozen TRAC-IK 2.2.0 bridge and original URDFs.
Clarabel was installed into a task-local directory with `--no-deps`; the existing
numerical environment was not upgraded. RangedIK source, Rust lockfile, exact
range-activation patch and reproducible bridge builder are retained under
`src/confik/correction_reserve/native/`. No experiment compiles or downloads a
dependency during a measured solve.

Run `bash scripts/run_correction_reserve.sh test` in that environment, setting
`CRIK_PYTHON` to its Python executable if necessary. `prepare` fixes new reference
paths without invoking any online solver. Commit code/configuration before
`formal`, which checks the seal and refuses an existing output directory.
`report` reads finished raw jobs and does not solve IK. CPU affinities in the
script reflect the measured host; changing them is a new measurement environment,
not a reproduction of its exact latency distribution.

## Strong-comparator boundary

Pink is an official one-step differential QP with configuration/rate limits,
not a reimplemented DLS. Its SE(3) soft cost is not an exact nonlinear pose-set
constraint. The RangedIK original-cutoff and positive-range adapter are both
reported. The latter changes only two hard-coded activation tests; both disable
the extra self-collision objective outside this task. No other native weight or
loss was fitted to these trajectories.

On the observed development paths neither Ranged configuration completed an
entire trajectory at this narrow command contract. That is an adapter/workload
limitation, not a satisfactory demonstration that CR-IK improves the best
RangedIK configuration. This evidence must not carry an algorithm-superiority
claim. Task-aligned TRAC-IK and the same-core predictive method are the decisive
comparators; their results and all negative baseline results remain visible.

## Closest work and the remaining hypothesis

| Existing work | What is already established | What this prototype tests |
|---|---|---|
| [RangedIK, ICRA 2023](https://graphics.cs.wisc.edu/Papers/2023/WPRG23/) | Weighted task ranges provide freedom for feasible, smooth motion. | Whether arranging a current/next pair around an explicit local correction policy improves actual continuation. Using tolerance itself is not new. |
| [Predictive Online IK, ICRA 2014](https://portal.fis.tum.de/en/publications/predictive-online-inverse-kinematics-for-redundant-manipulators/) | Moving-horizon optimization addresses limitations of instantaneous redundancy resolution. | Reserve versus ordinary prediction using the same numerical core. B4 is not a full reproduction of the original method. |
| [Look-Ahead Optimization, SII 2025](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15449) | Horizon optimization manages nullspace and kinematic constraints in dual-arm impedance control. | A two-configuration local correction objective in this existing pose/rate task, without true future-target access. Preview itself is not new. |
| [Pink official interface](https://stephane-caron.github.io/pink/inverse-kinematics.html) and [limits](https://stephane-caron.github.io/pink/limits.html) | Differential task QPs with configuration and velocity limits. | Whether additional nonlinear two-step computation is useful beyond one constrained QP. Constraint handling itself is not new. |
| [Clarabel](https://doi.org/10.1007/s12532-026-00320-7) | An existing conic optimizer supporting quadratic objectives. | Used as a numerical dependency, not an algorithmic novelty claim. |

Official sources were checked on 2026-09-08. Current Pink web documentation is
4.3.0, whereas the measured installed code is explicitly 3.3.0; the adapter notes
identify its actual API and checks. Literature-search MCP connectors were not
available, so verification used primary publication/author and official project
pages. No claim of first using tolerance, prediction, QP or solver bounds is made.

## Measurement and interpretation

TSR and DTSR20 are the two primary outcomes. Every frame, including failures and
late returns, contributes to cost. Whole trajectories are independent units;
three TRAC searches are nested repeats, not three independent trajectories.
Paired confidence intervals resample complete UIDs within the four fixed
families, after averaging repeated searches. They are descriptive unadjusted
intervals, not multiplicity-corrected significance claims.

The perturbation study fixes common input states from first-UID development
trajectories before testing continuations. Its accepted current commands and
successful next commands are saved as nonlinear two-frame witnesses. Its
amplitude ceiling and failed searches do not establish a global robustness radius
or mathematical infeasibility. A larger internal gamma is not itself success.

The initial full-development launcher stopped at a duplicate metadata-key error
in job 10. Its completed nine jobs per robot and explicit abort records remain
under `development_full/`. The record-merging fix did not change a solver;
`development_full_002/` contains the complete comparison. Formal evaluation is
separate and is not tuned after outcomes.
