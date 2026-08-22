# Artifact schema

## `query_results_v2.jsonl`

One JSON object is emitted for every `(query, method)` pair.

| Field | Meaning |
|---|---|
| `method` | Stable baseline, ablation, or proposed-method name |
| `query_index` | Index shared across methods for paired statistics |
| `category` | ID, stress type, or Cartesian trajectory type |
| `trajectory_id`, `time_index` | Sequence identity and frame index |
| `closed_loop` | Whether the method consumes its own previous accepted output |
| `closed_loop_previous_from_method` | Previous joint state came from this method's own trajectory state |
| `expected_reachable` | True only when a reference joint configuration exists |
| `continuity_feasible` | Whether the query was constructed to satisfy the 50 Hz velocity contract |
| `accepted` | True only after all verifier checks pass |
| `solver_converged` | At least one solver met pose tolerances, independently of velocity acceptance |
| `fallback_used` | KDTree/TRF stage was entered |
| `entry_action` / `risk_level` | Runtime easy, medium, hard, or reject decision |
| `executed_stages` | Ordered stages actually entered after escalation |
| `candidate_count` | Number of learned candidates made available to the cascade |
| `p_*` | Calibrated class probabilities |
| `latency_seconds` | End-to-end wall time measured around `solve` |
| `timing_repeats` | Number of repeated solves used for the per-query median |
| `deterministic_acceptance` | Whether all timing repetitions made the same acceptance decision |
| `iterations` | Sum of iteration counters over all attempted seeds |
| `function_evaluations` | Numerical-solver FK/residual evaluations over all attempts; feature/model work is represented in end-to-end latency instead |
| `position_error` | Best attempted solution position error in metres |
| `orientation_error` | Best attempted solution geodesic error in radians |
| `joint_step_max` | Largest displacement of an accepted command; NaN on rejection |
| `trajectory_spike` | An accepted command exceeded the velocity contract (must remain false under the verifier) |
| `candidate_joint_step_max` | Diagnostic displacement of the best attempted candidate, including rejected candidates |
| `candidate_velocity_violation` | Diagnostic candidate would violate the velocity contract |
| `verification_reasons` | Failed hard checks for the returned candidate |
| `reject_reason` | Structured reason when no candidate is accepted |

## Risk class indices

The stored integer labels are stable:

```text
0 easy
1 medium
2 hard
3 reject
```

## Reproducibility metadata

`solver_metadata.json` stores the robot-specific tenth-percentile singular-value threshold and the joint ordering. Model artifacts also store joint names and reject loading against a different robot ordering.

`protocol_manifest_v2.json` stores SHA-256 hashes of the resolved configuration
and `src/confik` tree. `results/environment_v2.json` stores Python, package,
CUDA, GPU, CPU, and platform metadata. `results/policy_selection_v2.json`
contains the policy-validation search and one-time test metrics.

`results/claim_gate_v2.json` keeps feasible-point routing, rejectable-point
handling, and trajectory completion as separate estimands. The paper-level
six-run decision is written to `outputs/paper_v2_aggregate/paper_gate_v2.json`.
