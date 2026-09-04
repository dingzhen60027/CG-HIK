# CG-HIK research record

Updated: 2026-09-04
Final evidence commit entering paper preparation:
244a216caed4367bf90c5d1eba5a0e1a88997019

## Research decision

The final method is frozen CG-HIK: a query-adaptive tail-latency router over a shared,
deterministically verified IK solver portfolio. Development of per-frame and temporal
shortcuts has stopped. No additional model, threshold, solver budget, fallback,
verifier, or test query is to be selected from the final results.

> **Learning allocates solver effort per query; numerical geometry generates joint
> commands; deterministic verification governs acceptance.**

## Scientific problem

An online IK query is composed of desired pose, previous accepted joint state, and
control interval. Queries differ in branch history, candidate quality, singularity,
joint-limit proximity, and motion scale. A robust fixed cascade covers many of these
conditions, but always entering at the same stage can spend a redundant numerical
prefix.

CG-HIK keeps the candidate model, DLS, trust-region fallback, seed bank, budgets, and
verifier fixed. During development, every easy/medium/hard entry is actually executed
for the same queries. A compact predictor learns shared terminal verified success and
entry-specific P50/P95 latency, then selects the minimum predicted-P95 entry among
those satisfying frozen success and deadline criteria.

Easy, medium, and hard are entry actions, not true difficulty labels. Reject is
permitted only for high-confidence portfolio failure and emits no command. Uncertain
or OOD queries defer to the complete Fixed robust cascade. Every numerical output must
pass the same deterministic verifier.

## Implemented command contract

The verifier checks:

- finite joint commands;
- position residual;
- orientation residual;
- URDF joint limits;
- one-step velocity continuity relative to the previous accepted state and dt.

The contract does not include collision, contact, acceleration, jerk, torque, dynamics,
controller tracking, or hardware uncertainty. “Verified” always means verified against
this specific kinematic contract.

## Frozen method

The shared portfolio contains:

- **easy entry:** one previous-state DLS update, then escalation if needed;
- **medium entry:** one DLS refinement from the best learned candidate, then hard
  escalation if needed;
- **hard entry:** longer DLS attempts from learned and previous seeds, followed by
  bounded trust-region-reflective fallback from heterogeneous seeds.

The learned proposal is a five-member, history-conditioned seed ensemble. The router
uses nine proposal/history/geometric diagnostics and a two-layer 32×32 SiLU MLP. It
predicts one shared semantic-success probability, entry-specific P50/P95 latency, and
fail-all probability. The release is an exact batch-one TorchScript artifact with
frozen calibration, Mahalanobis OOD transform, and routing policy.

## Evidence hierarchy

### 1. Action-complete development evidence

Source: outputs/counterfactual_v4_bulk/

There are 40,000 development queries across Panda and UR5e, with every entry executed
and five raw timing repeats retained. Among successful queries, empirical oracle
choices were:

| Robot | Easy | Medium | Hard |
|---|---:|---:|---:|
| Panda | 26.69% | 27.77% | 45.54% |
| UR5e | 23.41% | 25.36% | 51.22% |

Hard is the strongest fixed entry, but its mean empirical-P95 gap to the per-query
oracle remains 0.137 ms for Panda and 0.154 ms for UR5e. Frozen policy-selection
regret is 0.135/0.133 ms mean and 0.467/0.605 ms at P95. These are development
diagnostics, not fresh-test claims.

### 2. Fresh point-query evidence

Source: outputs/test_v4_aggregate_repair_v1/

The primary released system uses training seed 17. Seeds 29 and 43 repeat the same
test identities to assess model sensitivity and are not independent test datasets.
The primary point subset includes 14,000 feasible points and 2,000 known-infeasible or
one-frame-inadmissible points per robot.

| Metric, CG-HIK vs Fixed robust cascade | Panda | UR5e |
|---|---:|---:|
| Feasible success gap | 0 | 0 |
| Mean FEV reduction | 16.14% | 36.27% |
| P50 change | +16.47% | +16.45% |
| P95 ratio | 0.7538 | 0.7427 |
| P99 ratio | 0.9931 | 0.8808 |
| Known-infeasible reject recall | 95.65% | 93.95% |
| Known-infeasible FEV avoided | 95.69% | 93.92% |
| Point OOD AUROC / AUPRC | 0.430 / 0.210 | 0.504 / 0.230 |
| Defer semantic match / recovery | 100% / 0% | 100% / 0% |

The legacy frozen suite's eight hypotheses---six point margins and two
trajectory-completion margins---passed joint Holm correction. Its trajectory arm is
distinct from the final transition-rich evaluation below. The separate Panda
operational gate required strictly positive improvement in OOD-feasible false
rejection, whereas both methods had zero false rejects. Panda and the overall paper
gate for that suite were therefore false. This result is retained explicitly.

### 3. Final one-shot transition-rich evidence

Source: outputs/fresh_transition_v4_test/

Each robot has 80 unseen trajectories, 20 per family and 150 frames per trajectory.
The only evaluated methods are Fixed robust cascade, Fixed hard-entry cascade, and
frozen CG-HIK. Fresh identities were sealed before solver calls, each method was called
once per frame, and results were not used for tuning. The final gate passed for both
robots.

| Robot | Completion, CG-HIK vs hard | Cumulative time change | Mean FEV change | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 41/80 vs 40/80 | −43.37% | −45.16% | +11.66% | −12.04% | +0.03% |
| UR5e | 34/80 vs 34/80 | −54.95% | −65.12% | +11.09% | −21.94% | −23.09% |

Accepted contract violations were zero for all three methods. P50 was report-only.
The correct interpretation is that a fixed prediction cost raises the median while
avoiding expensive numerical work lowers aggregate latency and the upper tail.

Family outcomes show where the effect comes from:

- regular–near-singular–regular: −59.26% Panda and −59.60% UR5e cumulative latency;
- slow–high-curvature/high-speed–slow: −44.09% and −20.47%;
- smooth–fast-orientation–smooth: −29.62% and −9.44%;
- central–joint-limit-skim–return: Panda +0.68% (slightly worse), UR5e −79.26%.

Panda high-curvature/high-speed completion is only 1/20 for both hard and CG-HIK.
Lower computation does not mean this regime has been solved.

## Historical mechanism analyses

Later development-only studies tested lightweight per-frame gates and temporal local
shortcuts. They showed that one-step local DLS can sustain long successful runs and
reduce median latency, but they also produced robot-specific closed-loop branch drift,
long robust episodes, and completion losses near transitions and singularities.
Sparse hard re-anchoring did not consistently remove the problem. Those artifacts are
preserved as negative mechanism evidence, not as main methods or formal comparators.

## Final paper claim

CG-HIK provides query-specific tail-latency routing over a shared, deterministically
verified IK portfolio using action-complete pathway observations. It preserves the
numerical solver as command generator and the verifier as acceptance authority.

It does **not** claim:

- universally lower latency or P50;
- strong OOD detection;
- collision or physical safety;
- dynamics, contact, torque, or controller validation;
- hard real-time guarantees;
- cross-robot transfer;
- real-robot performance.

## Reproducible paper inputs

- Development evidence: outputs/counterfactual_v4_bulk/
- Sealed model and policy: outputs/release_v4_locked/
- Point aggregate: outputs/test_v4_aggregate_repair_v1/
- Final trajectory evidence: outputs/fresh_transition_v4_test/
- Paper evidence generator: paper/scripts/build_evidence.py
- Figure generator: paper/scripts/make_figures.py
- Machine-readable snapshot: paper/generated/evidence_snapshot.json
- Final manuscript: paper/main.tex and paper/main.pdf
