# Terminology ledger

This ledger is locked for the current `paper_v2` manuscript revision. Terms
marked **future only** must not be described as implemented or evaluated in the
current paper.

| Canonical term | First-use definition | Variants to avoid or restrict | Decision |
|---|---|---|---|
| confidence-gated hybrid inverse kinematics (CG-HIK) | The current four-action learned entry policy over a shared numerical cascade | confidence-based IK, neural IK controller | Use `CG-HIK` after first definition |
| fixed robust cascade | The controlled baseline that always enters at the easy stage and shares all downstream components with CG-HIK | baseline solver, fixed method | Use the full term when causal attribution matters |
| history-conditioned seed ensemble | Five frozen MLP members that propose joint increments from the previous state and target pose | neural IK, candidate model | Never imply that this module accepts commands |
| calibrated action gate | The four-class risk model and locked thresholds that select easy, medium, hard, or reject | uncertainty gate, confidence model | `risk model` may denote the probability estimator alone |
| easy, medium, hard, reject | Locked entry actions of the current cascade | difficulty classes | Use lowercase action names in prose and monospace in implementation contexts |
| adaptive damped least squares (DLS) | The shared local numerical refinement stage | adaptive solver | DLS is a numerical foundation, not the claimed novelty |
| trust-region reflective (TRF) fallback | The shared bounded robust fallback stage | optimizer, robust solver | Use `TRF fallback` after first definition |
| deterministic kinematic verifier | The shared acceptance layer for pose, joint-limit, and one-frame velocity conditions | safety verifier, verified safety | Never abbreviate to `safety verifier` |
| kinematically verified command | A command accepted under the stated pose, joint-limit, and one-frame velocity contract | safe command, executable proof, unqualified verified command | Does not imply collision, dynamic, controller, or hardware safety |
| kinematically verified success | Fraction of queries returning a kinematically verified command under the locked contract | IK accuracy, solve rate, unqualified verified success | State the query class and denominator |
| feasible point query | A generated point query with a known admissible target under the study protocol | reachable query | Use `reachable` only when geometric reachability alone is intended |
| rejectable point query | A generated radially unreachable or constructed one-frame-inadmissible request expected to be refused | impossible query, unsafe query, provably discontinuous query | Rejection is portfolio- and contract-specific, not proof of global unreachability |
| function evaluations (FEV) | DLS FK/current/trial/terminal calls plus SciPy TRF `nfev`, summed over executed solver attempts | FK calls, iterations | Candidate scoring, risk inference, final verification, and Jacobian/SVD work are excluded; do not equate FEV with latency |
| command spike | The same one-frame joint-velocity predicate used by the verifier | smoothness violation | Zero accepted-command spike is an acceptance invariant, not independent smoothness evidence |
| frozen formal test | The six `paper_v2` robot/seed evaluations | final test, optimized test | The original eager latency claim gate failed |
| validation-only latency pilot | `latency_pilot_v3`, used for backend selection and implementation profiling | optimized test, deployment result | Never present as independent test evidence |
| exact TorchScript backend (`torchscript_exact`) | Frozen batch-one deployment backend selected on validation | optimized backend, v3 model | Current all-seed release is incomplete |
| training-seed sensitivity run | One of seeds 17, 29, and 43 evaluated on the same robot-level formal query set | independent replicate | Never treat three seeds as three independent test datasets |
| command rejection | Zero-solve refusal of a request judged rejectable by the current gate | abstention | Current implementation supports this action |
| model deferral | **Future only.** Escalation of an OOD or uncertain request to the fixed robust cascade | OOD rejection, defer | Must not enter current Methods or Results as an implemented component |
| two-sided abstention | **Future only.** Separation of command rejection from model deferral | dual gate, dual rejection | Reserve for the proposed v4 study |
| counterfactual action outcome | **Future only.** Outcome obtained by executing every candidate entry action for one training query | first-success label | Reserve for the proposed v4 study |

## One-sentence current-paper argument

In online kinematic IK under a locked pose, joint-limit, and one-frame velocity
contract, the study shows that a calibrated learned gate can reduce numerical
work and reject constructed futile requests without reducing kinematically verified success when compared
with an otherwise identical fixed-entry cascade, while the failed eager latency
gate and validation-only remediation bound the deployment claim.

## Current evidence boundary

- Supported: controlled shared-cascade attribution, kinematically verified success,
  FEV, rejectable-query latency, descriptive trajectory completion, the accepted-command
  velocity invariant, the failed eager
  feasible-latency gate, and validation-only exact-backend equivalence.
- Not yet supported: independent optimized `test_v3` latency, OOD model
  deferral, counterfactual latency routing, acceleration or jerk verification,
  collision safety, controller tracking, and physical-robot reliability.
