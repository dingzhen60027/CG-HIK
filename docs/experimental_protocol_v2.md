# Frozen experimental protocol v2

## Falsifiable research question

Can a calibrated action gate enter a shared, verified IK cascade at the cheapest appropriate stage—or reject a query—while preserving feasible-query success and reducing unproductive computation relative to the identical fixed robust cascade?

The neural model does not certify an IK result. Numerical solvers generate/refine configurations and a common verifier makes the acceptance decision.

## Locked action contract

The four oracle labels and runtime entry actions are:

| Action | First stage | Budget | Escalation after failure |
|---|---|---:|---|
| easy | previous-state DLS | 1 iteration | medium, then hard |
| medium | best history-conditioned learned seed | 1 iteration | hard |
| hard | learned and previous DLS, then heterogeneous TRF seeds | 25 DLS iterations per seed | bounded fallback only |
| reject | no numerical solve | 0 | none |

The fixed robust cascade always enters at easy and follows the same escalation path. The proposed method differs only in its calibrated entry action. This makes skipped work observable and keeps solvers, learned model, seed bank, verifier, and fallback identical between the primary pair.

The probability thresholds are selected once on a dedicated `policy_validation`
split, after model selection and calibration but before either classifier-test or
runtime-test evaluation. The grid is frozen in the configuration. Selection is
constrained to at most 1% false rejection on non-reject oracle actions and at
least 95% recall on reject actions; among admissible policies, non-reject routing
macro-F1 is maximized. No threshold may be changed after inspecting test outcomes.

The data roles are disjoint: `risk_train` fits the classifier,
`risk_validation` selects its model family, `calibration` fits probability
calibration, `policy_validation` selects runtime thresholds, `risk_test` reports
held-out action/calibration metrics, and the independently generated runtime
benchmark evaluates solver behavior.

## Oracle action labels

Each query first receives candidate diagnostics. For a query known to be reachable and one-step continuous, the label is the first stage—easy, medium, or hard—that returns a solution accepted by the full runtime verifier. If all locked stages fail, the label is reject. Known unreachable or deliberately discontinuous queries are reject without spending solver calls during labelling.

Thus labels represent portfolio-specific actions, not physical reachability alone and not arbitrary iteration bins.

## Acceptance contract

- position error no greater than 1 mm;
- orientation error no greater than 0.5 degrees;
- finite configuration inside URDF joint limits;
- per-joint displacement no greater than the URDF velocity limit multiplied by 0.02 s, plus numerical tolerance.

No rejected configuration is sent downstream as a valid command.

## Locked query families

Point-query sets contain local ID, near-singular, near-limit, sampled workspace-boundary, discontinuous large-step, provably unreachable, and hard-valid queries. Each point query has a unique generating ID; correlated frames are reserved for the trajectory experiment. Hard-valid queries have a known reference configuration inside the one-frame velocity contract, fail the one-iteration previous-state DLS screen, and succeed with the locked longer previous-state DLS. Screening uses no learned model.

Trajectory targets are obtained by applying FK to independently generated, velocity-feasible joint references. Tested solvers are not used to filter trajectories. Four trajectory families emphasize smooth multi-joint motion, wrist/orientation motion, low-manipulability regions, and joint-limit skimming.

The benchmark is deliberately stratified and is not claimed to estimate natural deployment prevalence. Category-specific outcomes accompany every mixture-level result.

## Primary comparison and outcomes

Primary comparison: `proposed_v2` versus `fixed_robust_cascade`. A prespecified
interpretable baseline, `threshold_guard_cascade`, rejects using only Cartesian
position/orientation step and otherwise enters the identical fixed cascade. Its
quantile is selected from the frozen grid on `policy_validation` under exactly
the same false-reject and reject-recall constraints as the learned gate.

Primary outcomes:

1. verified success on known-feasible **point queries**;
2. mean FK/function evaluations on those same feasible point queries;
3. correct rejection and avoided computation on unreachable/discontinuous point queries;
4. trajectory-level completion on independent closed-loop paths.

These are separate estimands. Trajectory frames after a tracking failure are not
pooled into point-query success, and a method cannot claim routing efficiency by
incorrectly rejecting feasible queries. End-to-end P95 latency at batch size one
is reported separately for feasible and rejectable point strata.

Secondary outcomes include hard-valid success, P99 latency, stage-entry distribution, fallback frequency, trajectory completion, joint spikes, and accepted pose error.

## Pilot go/no-go gate

The full two-robot experiment is prohibited unless the UR5e one-seed pilot satisfies all conditions:

- feasible-point success difference versus fixed cascade is at least -1 percentage point;
- correct rejection on rejectable point queries is at least 95%;
- reject AUROC is at least 0.75, reject ECE is at most 0.10, held-out false rejection is at most 2%, reject recall is at least 95%, and non-reject routing macro-F1 is at least 0.30;
- feasible-point function evaluations fall by at least 10%, with feasible-point P95 latency no more than 25% worse;
- rejectable-point function evaluations and P95 latency each fall by at least 50%;
- trajectory completion differs from the fixed cascade by no less than -10 percentage points and no verifier-accepted command violates the velocity bound;
- relative to the validation-tuned threshold guard, feasible-point success differs by no less than -1 percentage point, feasible-point evaluations fall by at least 5%, and feasible-point P95 latency is not more than 25% worse.

Failed thresholds are reported and are not repaired by tuning on the pilot test set. Model or policy changes require a new versioned pilot test split.

## Replication and statistics

The full run uses UR5e and Panda with training seeds 17, 29, and 43 on an identical locked test set per robot. A training seed is a model-run replicate, not a new test dataset.

Paired point comparisons retain their generating trajectory/group ID. Trajectory
frames are clustered by trajectory ID and trajectory completion is computed once
per path. Confidence intervals use paired cluster bootstrap resampling; sign-flip
tests operate on cluster-level paired effects. Holm correction covers the three
prespecified primary point comparisons: feasible success, feasible function
evaluations, and rejectable acceptance. Paper-level claims require the same
effect direction on both robots and across locked seeds.

Method execution order is deterministically randomized per query. CUDA is synchronized around end-to-end timing. Timing runs must be performed without unrelated compute workloads.

## Scope boundary

This protocol establishes kinematic solver behavior under exact URDF models. Isaac Lab physics, controller tracking, acceleration/jerk limits, collision checking, and hardware experiments are external-validation layers and must not be inferred from the kinematic benchmark.
