from __future__ import annotations

import numpy as np

from confik.test_v4_locked.reporting import (
    CONFIRMATORY_INFERENCE_METRICS,
    claim_gate_report,
    joint_holm_confirmatory,
    method_metrics,
    ood_and_abstention_metrics,
    paired_confirmatory_intervals,
)


GATES = {
    "feasible_success_gap_ci_lower_min": -0.01,
    "contract_violation_count_max": 0,
    "id_feasible_false_reject_rate_max": 0.01,
    "feasible_p95_ratio_strict_max": 1.0,
    "feasible_p99_ratio_max": 1.05,
    "trajectory_completion_gap_min": -0.10,
    "trajectory_command_spike_increase_max": 0.0,
    "ood_feasible_false_reject_improvement_vs_v3_min": 0.0,
    "defer_fixed_semantic_match_rate_min": 1.0,
    "reject_function_evaluations_max": 0,
    "reject_executed_stage_count_max": 0,
}


def _row(
    method: str,
    role: str,
    index: int,
    *,
    latency: float,
    accepted: bool = True,
    decision: str | None = None,
    expected: bool = True,
    trajectory_id: int | None = None,
    ood_score: float | None = None,
    fev: int = 1,
    latency_repeats: tuple[float, ...] | None = None,
) -> dict[str, object]:
    trajectory = role.endswith("trajectories")
    action = decision or "easy"
    stages = [] if action == "reject" else ["easy"]
    raw_repeats = latency_repeats or (latency, latency, latency)
    return {
        "robot": "panda",
        "training_seed": 17,
        "method": method,
        "role": role,
        "domain": "ood" if role.startswith("ood_") else "id",
        "is_trajectory": trajectory,
        "source_query_sha256": f"{role}-{index}",
        "category": role,
        "trajectory_id": index // 2 if trajectory_id is None else trajectory_id,
        "time_index": index % 2 if trajectory else 0,
        "expected_reachable": expected,
        "continuity_feasible": expected,
        "verified_success": accepted,
        "accepted": accepted,
        "function_evaluations": 0 if action == "reject" else fev,
        "iterations": 0 if action == "reject" else fev,
        "fallback_used": False,
        "latency_ms": latency,
        "latency_repeats_ns": [int(value * 1.0e6) for value in raw_repeats],
        "latency_repeat_count": len(raw_repeats),
        "trajectory_command_spike": False,
        "entry_action": action,
        "decision_action": decision,
        "executed_stages": stages,
        "verification_reasons": [],
        "reject_reason": "" if accepted else ("command_reject" if action == "reject" else "failed"),
        "command_q": [float(index)] if accepted else None,
        "contract_violations": [],
        "ood_score": ood_score,
    }


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = ("fixed_robust_cascade", "proposed_v2", "proposed_v4")
    for index in range(4):
        for method in methods:
            rows.append(
                _row(
                    method,
                    "id_points",
                    index,
                    latency=2.0 if method != "proposed_v4" else 1.0,
                    decision=("defer" if method == "proposed_v4" and index == 0 else None),
                    ood_score=(0.1 if method == "proposed_v4" else None),
                )
            )
    for index in range(4):
        for method in methods:
            # v3 false-rejects feasible OOD; v4 defers and recovers.
            decision = None
            accepted = True
            if method == "proposed_v2":
                decision = "reject"
                accepted = False
            elif method == "proposed_v4":
                decision = "defer"
            rows.append(
                _row(
                    method,
                    "ood_points",
                    index,
                    latency=2.0 if method != "proposed_v4" else 1.0,
                    accepted=accepted,
                    decision=decision,
                    ood_score=(3.0 if method == "proposed_v4" else None),
                )
            )
    # One infeasible OOD command reject proves the zero-budget accounting.
    for method in methods:
        rows.append(
            _row(
                method,
                "ood_points",
                99,
                latency=0.2 if method == "proposed_v4" else 2.0,
                accepted=False,
                decision="reject" if method == "proposed_v4" else None,
                expected=False,
                ood_score=(0.2 if method == "proposed_v4" else None),
            )
        )
    for role in ("id_trajectories", "ood_trajectories"):
        for index in range(4):
            for method in methods:
                rows.append(
                    _row(
                        method,
                        role,
                        index,
                        trajectory_id=index // 2,
                        latency=2.0 if method != "proposed_v4" else 1.0,
                        decision=("defer" if method == "proposed_v4" else None),
                        ood_score=(3.0 if role.startswith("ood") else 0.1)
                        if method == "proposed_v4"
                        else None,
                    )
                )
    return rows


def test_metrics_include_tail_latency_fev_deadline_and_route_switch() -> None:
    metrics = method_metrics(_records())
    v4 = metrics["proposed_v4"]
    assert v4["id_points_feasible"]["verified_success_rate"] == 1.0
    assert v4["id_points_feasible"]["latency_ms"]["p99"] == 1.0
    assert v4["id_points_feasible"]["function_evaluations"]["mean"] == 1.0
    assert v4["id_points_feasible"]["latency_estimand"] == "raw_solve_calls"
    assert v4["id_points_feasible"]["latency_raw_call_count"] == 12
    assert v4["id_trajectories"]["trajectory_completion_rate"] == 1.0
    assert v4["id_trajectories"]["route_switch_rate"] == 0.0


def test_raw_call_latency_and_deadline_are_not_query_median_metrics() -> None:
    rows = [
        _row(
            "proposed_v4",
            "id_points",
            0,
            latency=2.0,
            latency_repeats=(1.0, 2.0, 30.0),
            ood_score=0.1,
        )
    ]
    metrics = method_metrics(rows)["proposed_v4"]["id_points_feasible"]
    assert metrics["latency_ms"]["count"] == 3
    assert metrics["latency_per_query_median_ms"]["p50"] == 2.0
    assert metrics["deadline_miss_rate_20ms"] == 1.0 / 3.0
    assert metrics["deadline_any_repeat_miss_rate_20ms"] == 1.0
    assert metrics["deadline_query_median_miss_rate_20ms"] == 0.0


def test_ood_and_abstention_use_raw_score_and_exact_defer_semantics() -> None:
    metrics = ood_and_abstention_metrics(_records())
    assert metrics["ood_auroc"] > 0.9
    assert metrics["ood_auprc"] > 0.9
    assert metrics["ood_feasible_command_false_reject_rate"] == 0.0
    assert metrics["defer_fixed_semantic_match_rate"] == 1.0
    assert metrics["defer_support_status"] == "estimated"
    assert metrics["defer_fixed_semantic_mismatch_fields"] == {}
    assert metrics["defer_recovery_success_rate"] == 1.0
    assert metrics["command_reject_max_fev"] == 0
    assert metrics["command_reject_max_executed_stage_count"] == 0
    assert metrics["ood_headline_scope"] == "independent_point_queries"
    assert metrics["ood_point_detection"]["unit"] == "independent_point_query"
    assert metrics["ood_trajectory_cluster_detection"]["sample_count"] == 4
    assert metrics["infeasible_command_reject_recall"] == 1.0
    assert metrics["infeasible_function_evaluations_avoided_vs_fixed"] == 1


def test_defer_full_equivalence_reports_field_level_mismatches() -> None:
    records = _records()
    deferred = next(
        row
        for row in records
        if row["method"] == "proposed_v4"
        and row["role"] == "id_points"
        and row["decision_action"] == "defer"
    )
    deferred["command_q"] = [1.0e-6]
    deferred["iterations"] = 9
    deferred["fallback_used"] = True
    deferred["verification_reasons"] = ["changed"]
    metrics = ood_and_abstention_metrics(records)
    assert metrics["defer_fixed_semantic_match_rate"] < 1.0
    assert metrics["defer_fixed_semantic_mismatch_fields"] == {
        "command_q": 1,
        "fallback_used": 1,
        "iterations": 1,
        "verification_reasons": 1,
    }
    assert metrics["defer_command_q_max_abs_difference"] == 1.0e-6


def test_paired_bootstrap_clusters_queries_and_retains_raw_latency_calls() -> None:
    intervals = paired_confirmatory_intervals(
        _records(), samples=100, seed=31, gates=GATES
    )
    assert intervals["resampling"]["point_unit"] == "source_query_sha256"
    assert intervals["resampling"]["trajectory_unit"] == "(explicit_role, trajectory_id)"
    assert intervals["resampling"]["latency_within_query_unit"] == "raw_solve_calls"
    assert intervals["inference_family"]["joint_robot_holm_required"]
    metrics = intervals["metrics"]
    assert tuple(metrics) == CONFIRMATORY_INFERENCE_METRICS
    assert metrics["feasible_p95_latency_ratio"]["estimate"] == 0.5
    assert metrics["feasible_p95_latency_ratio"]["raw_call_count_per_method"] == 24
    assert metrics["feasible_p95_latency_ratio"]["repeat_count_per_query"] == 3
    assert metrics["feasible_success_gap"]["estimate"] == 0.0
    operational = intervals["operational_finite_test_metrics"]
    assert operational["ood_feasible_false_reject_improvement_vs_v3"]["estimate"] == 1.0
    assert all(
        0.0 <= value["one_sided_unadjusted_p"] <= 1.0
        for value in metrics.values()
    )


def test_joint_holm_is_applied_once_across_both_robots_and_four_claims() -> None:
    def interval(pvalues: tuple[float, ...]) -> dict[str, object]:
        return {
            "inference_family": {"members": list(CONFIRMATORY_INFERENCE_METRICS)},
            "metrics": {
                name: {"one_sided_unadjusted_p": value}
                for name, value in zip(
                    CONFIRMATORY_INFERENCE_METRICS, pvalues, strict=True
                )
            },
        }

    report = joint_holm_confirmatory(
        {
            "panda": interval((0.001, 0.002, 0.003, 0.004)),
            "ur5e": interval((0.005, 0.006, 0.007, 0.008)),
        },
        alpha=0.05,
    )
    assert report["hypothesis_count"] == 8
    assert report["all_confirmatory_nulls_rejected"]
    assert all(
        0.0 <= payload["holm_adjusted_p"] <= 1.0
        for payload in report["hypotheses"].values()
    )


def test_claim_gate_combines_exact_contracts_and_preregistered_margins() -> None:
    records = _records()
    intervals = paired_confirmatory_intervals(
        records, samples=100, seed=32, gates=GATES
    )
    claim = claim_gate_report(records, gates=GATES, intervals=intervals)
    assert claim["feasible_p95_latency_ratio"] == 0.5
    assert claim["feasible_p99_latency_ratio"] == 0.5
    assert claim["checks"]["reject_zero_fev"]
    assert claim["checks"]["defer_fixed_semantic_match"]
    assert claim["checks"]["reject_support"]
    assert claim["checks"]["defer_support"]
    assert claim["feasible_latency_ms"]["estimand"] == "raw_solve_calls"
    assert claim["confirmatory_inference"]["joint_panda_ur5e_holm_required"]
    assert claim["formal_gate_pass"]
    assert not claim["test_set_retuning_performed"]


def test_empty_reject_and_defer_support_are_not_vacuous_passes() -> None:
    records = _records()
    for row in records:
        if row["method"] != "proposed_v4":
            continue
        row["decision_action"] = None
        row["entry_action"] = "easy"
        row["executed_stages"] = ["easy"]
        row["function_evaluations"] = 1
        row["iterations"] = 1
        if not row["accepted"]:
            row["reject_reason"] = "all_cascade_stages_failed"
    abstention = ood_and_abstention_metrics(records)
    assert abstention["defer_support_status"] == "not_estimated"
    assert abstention["defer_fixed_semantic_match_rate"] is None
    assert abstention["command_reject_support_status"] == "not_estimated"
    assert abstention["command_reject_zero_fev_rate"] is None
    intervals = paired_confirmatory_intervals(
        records, samples=100, seed=33, gates=GATES
    )
    claim = claim_gate_report(records, gates=GATES, intervals=intervals)
    assert not claim["checks"]["reject_support"]
    assert not claim["checks"]["defer_support"]
    assert not claim["checks"]["reject_zero_fev"]
    assert not claim["checks"]["defer_fixed_semantic_match"]
    assert not claim["formal_gate_pass"]
