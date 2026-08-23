# Executable v4 research plan

Working title: **Two-Sided Abstention for Risk-Controlled Adaptive-Compute
Inverse Kinematics**

Status: prospective plan only. No method, threshold, label, or result described
here is part of the frozen `paper_v2` or `latency_pilot_v3` evidence.

## Research questions

1. Can action-level success and tail-latency prediction reduce feasible-query
   P95/P99 latency and 20 ms deadline misses relative to an otherwise identical
   fixed robust cascade?
2. Under prespecified distribution shifts, does separating command rejection
   from model deferral reduce false rejection of feasible OOD requests?
3. Does the exact batch-one implementation preserve point- and trajectory-level
   gains on Panda and UR5e, and, if hardware is available, under controller
   execution?

## Phase 0. Close v3 before starting v4

1. Preserve the failed `ur5e/seed43` release diagnostic and all frozen outputs.
2. Resolve only the exporter-equivalence defect under a separately authorized
   change. Do not change the backend, tolerances, thresholds, labels, solver, or
   claim gates.
3. Re-run all six release equivalence checks from a clean committed tree.
4. Only if all six pass, freeze the v3 preregistration and a fresh, disjoint
   test dataset, then run the one-shot six-combination `test_v3`.
5. Freeze and archive v3 before generating any v4 action labels.

**Stop condition.** If the exact exporter cannot satisfy the locked tolerance,
v3 remains a negative deployment result. Do not repair the paper by relaxing the
gate.

## Phase 1. Counterfactual action dataset

For each training-only query, execute every non-reject entry action under the
same downstream cascade and verifier:

- `entry_easy`;
- `entry_medium`;
- `entry_hard`;
- `fixed_robust` as an audit reference.

Record per query and action:

- verifier acceptance and reason;
- core end-to-end latency using `perf_counter_ns`;
- FEV and fallback use;
- deadline success at 20 ms;
- accepted joint command;
- maximum joint step, velocity, acceleration, and jerk under the frozen
  discrete-time definitions.

Proposed initial scale per robot, subject to a pilot cost audit:

- 120,000 action-training queries;
- 20,000 model-validation queries;
- 10,000 calibration queries;
- 10,000 policy-validation queries;
- independent ID and OOD runtime tests generated only after model and rules are
  frozen.

The first 2,000 queries constitute a cost and label-quality pilot only. Scale is
confirmed before bulk generation and is not selected from test performance.

## Phase 2. Compact multi-head predictor

Use a small two- or three-layer MLP rather than a new high-capacity sequence
model. The candidate input set contains the existing nine features plus only
features justified by an ablation hook:

- Jacobian condition diagnostics;
- velocity, acceleration, and jerk margins;
- prior action, FEV, and verifier reason;
- candidate residual distribution and candidate diversity.

For each action, predict:

1. probability of verifier acceptance before the 20 ms deadline;
2. conditional latency quantiles at P50 and P95;
3. probability that all non-reject actions fail under the locked portfolio.

Train latency heads with pinball loss. Calibrate success and fail-all outputs on
the dedicated calibration split. Record calibration error, Brier score, NLL,
quantile coverage, and quantile crossing.

## Phase 3. Two-sided abstention rule

Define two semantically different actions:

- **command rejection** for an in-distribution query for which every numerical
  entry is predicted to fail and fail-all confidence passes a frozen threshold;
- **model deferral** for an OOD or uncertain query, which enters the full fixed
  robust cascade and is never converted directly into reject.

Use the gate embedding to fit a regularized Mahalanobis detector on training
only. Freeze the OOD threshold on calibration, initially at its 0.99 ID
quantile. Ensemble disagreement may be a second signal only if its validation
ablation changes OOD recovery or false rejection.

For an ID query, choose the action with the smallest predicted P95 latency among
actions satisfying the frozen lower success-confidence bound. If no action is
eligible and fail-all confidence is insufficient, defer rather than reject.

## Phase 4. Temporal rule and verifier extension

Evaluate a latency-margin hysteresis rule before considering a recurrent model.
An action switch is allowed only when the new action improves predicted P95 by
a preregistered margin. Compare no hysteresis, a fixed hold rule, and the latency
margin at the complete-trajectory level.

Acceleration and jerk can enter the verifier only after robot-specific limits,
finite-difference conventions, initialization at trajectory start, and units are
frozen. Adding these constraints requires regenerating labels for every method.
Collision remains an optional, separately scoped subset.

## Phase 5. ID and OOD evaluation

### ID query families

- local IID;
- near singularity;
- near joint limit;
- workspace boundary;
- hard-valid;
- discontinuous large step;
- unreachable;
- ambiguous boundary queries where early stages fail but robust fallback
  succeeds.

### Prespecified OOD families

- held-out workspace sectors;
- held-out trajectory frequency and curvature ranges;
- target-pose perturbations;
- previous-state perturbation and one- to three-frame delay;
- URDF link or zero-offset perturbation;
- held-out fast-motion and trajectory families.

Each OOD family needs fixed parameter ranges, generation seeds, sample counts,
and zero-overlap audits. Calibration remains ID-only unless the research
question is explicitly changed before preregistration.

## Baselines and causal comparisons

Required internal baselines:

- previous-state adaptive DLS;
- previous-state TRF;
- fixed robust cascade;
- Cartesian-step threshold guard;
- learned seed with fixed refinement;
- frozen v3 CG-HIK;
- v4 two-sided abstention method.

Target two or three external baselines selected before test from implementations
that can share the command contract and batch-one timing environment. Candidate
families include TRAC-IK, IKSel, IK-Geo, and a viability/QP implementation.
Unavailable or semantically incomparable methods must be reported as such rather
than approximated under their names.

## Required ablations

1. first-success labels versus counterfactual success/latency labels;
2. no OOD gate;
3. OOD to reject versus OOD to defer;
4. no latency-quantile head;
5. no hysteresis;
6. single member or no disagreement;
7. eager versus exact deployment implementation.

If ensemble disagreement remains null on the frozen OOD tests, simplify the
final method instead of claiming an unsupported uncertainty benefit.

## Estimands and metrics

Point-query estimands use query ID as the unit. Trajectory estimands use the
complete trajectory. Hardware estimands, if authorized, use trajectory by
repetition.

Primary metrics:

- feasible verified-success difference;
- feasible P95 and P99 latency ratios;
- 20 ms deadline-miss difference;
- ID false-rejection rate;
- OOD feasible false-rejection rate;
- trajectory-completion difference.

Secondary metrics:

- mean FEV and fallback rate;
- reject precision and recall;
- P50, P95, P99, maximum latency;
- AUROC, AUPRC, ECE, Brier score, NLL, risk-coverage curve;
- OOD escalation rate and recovered-OOD rate;
- route switches, branch switches, joint step, velocity, acceleration, jerk,
  and command spikes.

Use paired cluster bootstrap with complete queries or trajectories as clusters.
Use paired quantile bootstrap for P95/P99. Treat training seeds as sensitivity
runs, not independent datasets. Apply Holm correction to the frozen family of
primary hypotheses.

## Prospective paper gates

These are draft v4 gates and must be finalized before v4 test generation. They
do not replace any v3 gate.

- feasible success 95% CI lower bound at least -1 percentage point;
- zero accepted kinematic-contract violations;
- ID false rejection at most 1%;
- feasible P95 ratio below 1.0;
- feasible P99 ratio at most 1.05;
- trajectory completion non-inferior to fixed;
- OOD feasible false rejection lower than frozen v3 CG-HIK;
- hardware deadline misses no higher than fixed, if hardware is in scope;
- no increase in command-spike rate.

## Eight-week deliverables

| Week | Deliverable | Exit criterion |
|---|---|---|
| 1 | Close or formally stop v3 | Six release artifacts plus one-shot test, or documented locked failure |
| 2 | Action-label pilot and cost audit | Label schema, hashes, agreement audit, projected bulk cost |
| 3 | Frozen counterfactual dataset and compact model | Split audit and validation metrics |
| 4 | OOD detector and two-sided decision rule | Frozen calibration and policy thresholds |
| 5 | Internal and selected external baselines | Same verifier, query order, and timing contract |
| 6 | ID/OOD and trajectory tests; hardware only if authorized | All logs complete, no threshold edits |
| 7 | Statistics and figures | Frozen tables, intervals, and claim gates |
| 8 | Manuscript revision and release package | Claim-evidence audit and reproducible archive |

## Go/no-go checkpoints

- Do not start v4 test if v4 training, calibration, or policy-selection rules are
  still changing.
- Do not claim a benefit for uncertainty, OOD, hysteresis, acceleration/jerk, or
  hardware unless its dedicated gate passes.
- Do not call the verifier a safety proof.
- Do not use test results to select architecture, OOD threshold, latency margin,
  external baseline settings, or paper gates.
