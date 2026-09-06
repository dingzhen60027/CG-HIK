# Supplementary measurement notes

The old formal outputs, model artifacts and original paper source data are unchanged.
New artifacts live only in `outputs/revision_compute_allocation/`.

## Hardware and implementation

- CPU: Intel Core i7-13700K (24 logical CPUs available; no affinity restriction).
- GPU: NVIDIA Quadro RTX 8000, used by the frozen learned proposal.
- PyTorch: 2.11.0+cu128; eight intra-op and one inter-op thread, matching release.
- BLAS/OpenMP thread environment: eight. Numerical and neural calls are serial
  between methods; no second benchmark is scheduled concurrently.
- Public kinematics: repository URDFKinematics chain, FK and geometric Jacobian.
  Pinocchio is installed but is not the implementation called by this runner.
- Official TRAC-IK: 2.2.0, upstream commit
  `90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb`, Speed mode, two concurrent solver
  workers (KDL and nonlinear optimization). Three official library implementation
  files are built unmodified with a C ABI adapter. ROS Humble compatibility only
  maps the upstream `urdf/model.hpp` header spelling to `urdf/model.h`.
- Compiler: GNU C++ 9.5.0, Release build; NLopt 2.7.1 from Ubuntu packages;
  Orocos KDL 1.5.1 and ROS Humble kdl_parser 2.6.4. No system packages are replaced.

## Adapter boundary

TRAC-IK reads the identical URDF base/tool chain and joint ordering. On 100 random
configurations per robot, its KDL FK and the public model agree to below 7e-16.
The Panda parser notes an unsupported root-link inertia; inertia has no role in
these kinematic transforms, and the source URDF is not edited.

The internal epsilon is 1e-5 per Cartesian component, with zero additional Twist
tolerance. This is stricter than the common 1 mm position norm and 0.00872664626 rad
orientation norm criteria. The public verifier is still the final acceptance test.
Each solve uses `setKDLLimits` to intersect finite URDF bounds with the allowed
single-frame interval. This setup recreates internal solver objects and its cost is
included in every outer-call timing. Model loading, chain parsing and initial
construction are outside per-query timing.

Both supplied chains have only bounded revolute active joints (including UR5e
joints with wide finite bounds). Their differences must not be wrapped as if they
were continuous. The adapter separately handles genuine continuous variables by
searching a local unwrapped interval and normalizing before the public verifier;
a unit test covers an interval crossing the representation boundary.

## Timing scope and comparisons

Full CG-HIK is loaded through the existing sealed-release runtime factory. Policy
variants change only the decision after one frozen predictor invocation. P50 is a
median-selection ablation with unchanged P95 eligibility, never a mean-trained
model. Geometry routing retains abstention and changes only non-abstained entries.
Always-hard runs no unused risk features or predictor, but retains the same learned
candidate preparation, numerical hard stage, fallback and verifier. The streamlined
wrapper's command, FEV, executed stages and fallback matched the old hard wrapper in
smoke checks. No runtime uses offline reference joints as an extra seed.

Measured outer calls include all input conversion, required proposal/diagnostic/
prediction/decision work, numerical work, fallback and in-runtime verification.
GPU methods synchronize at call boundaries. Independent acceptance replay and
serialization are outside the interval. A call that returns after 20 ms is not
counted as accepted within 20 ms, even when the geometric command is accepted.

The learned models and internal numerical stages are deterministic under the frozen
settings. TRAC-IK uses its official parallel stochastic search; all three repetitions
are kept, including outcome variation. Its FEV is reported as unavailable rather
than equated with the repository's residual-evaluation definition.

## Selection completed before test outcomes

Geometry thresholds selected on old calibration labels:
Panda 0.02 m / 0.01 rad; UR5e 0.005 m / 0.025 rad. Threshold candidates and selection
criteria are retained. Calibration label costs are not mislabeled as deployment
end-to-end costs.

TRAC-IK trajectory configuration: independent, stratified 120-query calibration
subset per robot; maximize verified success, then minimize total measured time,
then choose smaller budget. Both selected 5 ms. Point testing still includes all
four fixed budgets (5, 20, 100 and 400 ms).

Each robot has 3,000 new point identities, 500 fixed feasible oracle-diagnostic
identities and 40 new 150-frame reference trajectories. All 2,500 point witnesses
and all 6,000 reference transitions per robot pass the unchanged verifier before
method evaluation. No geometric sample is screened by an online method's success.

Statistical units are the query or whole trajectory. Repeats and frames remain
nested. Bootstrap intervals are paired and family-stratified. Undefined cost ratios
(zero reference FEV) are left blank, not presented as zero.

## Interrupted implementation attempt

The first point run was interrupted after the Panda progress counter reached
1,425/3,000, before UR5e, oracle diagnostics or trajectory measurements. Static
inspection found an avoidable P95 ranking in the policy-override adapter: it first
computed the full decision before replacing the ranking/entry for a control.
This was a timing-confound correction, not an outcome-driven parameter change.
The incomplete raw file and its original seal/smoke record are preserved separately
and excluded in full. No aggregate outcomes from it were inspected or used.

The corrected controls perform one frozen inference and only their required
decision: no entry sorting for geometry or reject-only, one P50 sorting for the
P50 control, and one P95 sorting for routing-only. Tests explicitly count ranking
calls. Existing calibration choices, fresh identities, frozen main method, numerical
budgets and verifier are unchanged. A final execution seal binds the corrected
measurement sources before the complete evaluation. This is disclosed rather than
described as an uninterrupted one-shot run.
