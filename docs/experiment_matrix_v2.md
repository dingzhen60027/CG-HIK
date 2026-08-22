# Executable evidence matrix for the paper

## Central claim

The learned module predicts a verified cascade entry action and rejection risk;
it does not certify IK. Numerical stages produce candidate configurations and a
common verifier alone decides whether a command is admissible.

## Claim-to-experiment contract

| ID | Falsifiable claim | Required comparison | Independent unit | Pass/fail evidence |
|---|---|---|---|---|
| H1 | History-conditioned seeds help enter a useful local basin | learned seed vs previous-state DLS; no-history ablation | generating point-query cluster or whole trajectory | success-budget curves and trajectory completion |
| H2 | The action gate predicts useful solver effort, not only unreachable targets | proposed vs identical fixed cascade on known-feasible point queries | point-query cluster | success non-inferiority and paired FEV reduction |
| H3 | Learned routing adds value beyond a Cartesian-distance rule | proposed vs validation-constrained threshold guard | point-query cluster | matched false-reject/reject-recall constraints, then success and FEV |
| H4 | Explicit rejection avoids futile computation without emitting invalid commands | proposed vs fixed cascade on unreachable/discontinuous points | point query | rejection, FEV, P95 latency, false rejection |
| H5 | Calibration/uncertainty/fallback choices matter | uncalibrated, no-uncertainty, single-member, no-fallback ablations | point-query cluster / trajectory | held-out calibration and solver outcomes |
| H6 | The verifier is operationally necessary | counterfactual convergence-only audit | point query | converged-but-rejected interception count and reason |
| H7 | The behavior persists in sequential control | proposed, fixed cascade, no-history on four reference path families | whole trajectory | completion, frame acceptance, hold-on-failure, command spike rate |
| H8 | Conclusions are not a one-robot/one-seed accident | UR5e and Panda, seeds 17/29/43 | locked query cluster; seed is sensitivity replicate | same effect direction in all six runs and per-run cluster intervals |

## Six primary methods

1. `dls_previous_1x50`: conventional previous-state adaptive DLS.
2. `learned_1x25`: history-conditioned learned seed followed by DLS.
3. `fixed_robust_cascade`: identical portfolio that always starts at `easy`.
4. `threshold_guard_cascade`: validation-constrained Cartesian-step guard plus the identical cascade.
5. `trf_previous`: optimization fallback from the previous state.
6. `proposed_v2`: calibrated learned entry action plus verified escalation/fallback.

These are the main result table. The apparent “13 methods” in a full run is six
main methods plus seven diagnostic ablations; it is not 13 unrelated models.

## Seven diagnostic ablations

| Ablation | Question answered | Manuscript status |
|---|---|---|
| `ablation_no_history` | Does current/previous state improve seeds and paths? | contribution test |
| `ablation_single_member` | Does the ensemble add anything? | diagnostic; no diversity claim if null |
| `ablation_no_uncertainty` | Do disagreement features improve routing? | contribution test |
| `ablation_uncalibrated` | Does calibration improve rejection/routing? | contribution test |
| `ablation_no_reject` | Is explicit refusal responsible for rejectable-query savings? | mechanism test |
| `ablation_no_fallback` | Does bounded fallback recover hard-valid queries? | reliability test |
| `ablation_fixed_damping` | Does adaptive DLS remain a useful numerical foundation? | control, not novelty claim |

## Data separation

| Split | Sole permitted use |
|---|---|
| `seed_train` | fit seed ensemble |
| `seed_validation` | report seed accuracy/diversity |
| `risk_train` | fit risk classifier and distance thresholds |
| `risk_validation` | select MLP versus gradient boosting |
| `calibration` | fit isotonic calibration |
| `policy_validation` | choose action thresholds/guard quantile under locked constraints |
| `risk_test` | one-time held-out action and calibration metrics |
| runtime test | one-time solver, timing, rejection, and trajectory outcomes |

The manifest hashes the complete config and `src/confik` tree. Reusing an output
directory after either changes is refused; a new `experiment_name` is required.

## Full-run sample plan

- Seed training: 200,000 transitions per robot and seed.
- Risk labels: 60,000 train; 15,000 model-validation; 15,000 calibration;
  15,000 policy-validation; 20,000 classifier-test.
- Runtime points: 5,000 ID; 1,000 each near-singular, near-limit,
  workspace-boundary, large-step, and unreachable; 2,000 hard-valid.
- Runtime paths: 10 paths × 150 frames for each of four path families.
- Robots: UR5e and Panda.
- Training seeds: 17, 29, 43, sharing a locked runtime test set per robot.

## Statistical reporting

- Report absolute rates and paired differences with 95% cluster-bootstrap CIs.
- Use generating group IDs for point clusters and whole-path IDs for trajectories.
- Apply Holm correction to feasible success, feasible FEV, and rejectable acceptance.
- Do not pool trajectory frames with point-query success.
- Do not treat three training seeds as three independent datasets.
- Report P50/P95/P99 batch-one latency, method-order randomization, CUDA
  synchronization, CPU-thread count, software versions, and GPU model.
- Report all locked categories, including negative and null ablations.

## Stop rules

1. The full run is blocked unless the fresh pilot claim gate is true.
2. A failed full run is reported; test thresholds are not retuned.
3. Efficiency language is removed if feasible success is inferior or FEV effect
   direction changes on either robot.
4. Ensemble/multi-solution language is removed unless held-out diversity and its
   ablation support it.
5. Kinematic evidence is not described as Isaac Lab physics, collision safety,
   controller tracking, or hardware real-time validation.
