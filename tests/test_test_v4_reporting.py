from __future__ import annotations

import numpy as np

from confik.test_v4_locked.reporting import (
    claim_gate_report,
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
) -> dict[str, object]:
    trajectory = role.endswith("trajectories")
    action = decision or "easy"
    stages = [] if action == "reject" else ["easy"]
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
        "latency_ms": latency,
        "trajectory_command_spike": False,
        "entry_action": action,
        "decision_action": decision,
        "executed_stages": stages,
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
    assert v4["id_trajectories"]["trajectory_completion_rate"] == 1.0
    assert v4["id_trajectories"]["route_switch_rate"] == 0.0


def test_ood_and_abstention_use_raw_score_and_exact_defer_semantics() -> None:
    metrics = ood_and_abstention_metrics(_records())
    assert metrics["ood_auroc"] > 0.9
    assert metrics["ood_auprc"] > 0.9
    assert metrics["ood_feasible_command_false_reject_rate"] == 0.0
    assert metrics["defer_fixed_semantic_match_rate"] == 1.0
    assert metrics["defer_recovery_success_rate"] == 1.0
    assert metrics["command_reject_max_fev"] == 0
    assert metrics["command_reject_max_executed_stage_count"] == 0


def test_paired_bootstrap_is_query_or_trajectory_clustered_and_holm_adjusted() -> None:
    intervals = paired_confirmatory_intervals(
        _records(), samples=100, seed=31, gates=GATES
    )
    assert intervals["resampling"]["point_unit"] == "source_query_sha256"
    assert intervals["resampling"]["trajectory_unit"] == "(explicit_role, trajectory_id)"
    assert intervals["resampling"]["multiplicity_correction"] == "Holm"
    metrics = intervals["metrics"]
    assert metrics["feasible_p95_latency_ratio"]["estimate"] == 0.5
    assert metrics["feasible_success_gap"]["estimate"] == 0.0
    assert metrics["ood_feasible_false_reject_improvement_vs_v3"]["estimate"] == 1.0
    assert all(0.0 <= value["holm_adjusted_p"] <= 1.0 for value in metrics.values())


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
    assert claim["formal_gate_pass"]
    assert not claim["test_set_retuning_performed"]
