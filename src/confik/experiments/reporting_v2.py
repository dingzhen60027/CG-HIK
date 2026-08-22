from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _rows(records: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [record for record in records if record["method"] == method]


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else float("nan")


def _p95(rows: list[dict[str, object]], field: str) -> float:
    return float(np.percentile([float(row[field]) for row in rows], 95)) if rows else float("nan")


def _trajectory_completion(rows: list[dict[str, object]]) -> float:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[int(row["trajectory_id"])].append(bool(row["accepted"]))
    return float(np.mean([all(values) for values in grouped.values()])) if grouped else float("nan")


def _method_metrics(records: list[dict[str, object]], method: str) -> dict[str, Any]:
    rows = _rows(records, method)
    points = [row for row in rows if not bool(row["closed_loop"])]
    point_feasible = [
        row for row in points
        if bool(row["expected_reachable"]) and bool(row["continuity_feasible"])
    ]
    point_rejectable = [
        row for row in points
        if not (bool(row["expected_reachable"]) and bool(row["continuity_feasible"]))
    ]
    trajectories = [row for row in rows if bool(row["closed_loop"])]
    hard_valid = [row for row in point_feasible if row["category"] == "hard_valid"]
    converged_but_rejected = [
        row for row in points if bool(row["solver_converged"]) and not bool(row["accepted"])
    ]
    return {
        "query_count": len(rows),
        "point_count": len(points),
        "point_feasible_count": len(point_feasible),
        "point_rejectable_count": len(point_rejectable),
        "point_feasible_success": _mean(point_feasible, "accepted"),
        "point_rejectable_rejection": 1.0 - _mean(point_rejectable, "accepted"),
        "hard_valid_success": _mean(hard_valid, "accepted"),
        "point_feasible_mean_function_evaluations": _mean(
            point_feasible, "function_evaluations"
        ),
        "point_rejectable_mean_function_evaluations": _mean(
            point_rejectable, "function_evaluations"
        ),
        "point_feasible_p95_latency_seconds": _p95(point_feasible, "latency_seconds"),
        "point_rejectable_p95_latency_seconds": _p95(point_rejectable, "latency_seconds"),
        "trajectory_count": len({int(row["trajectory_id"]) for row in trajectories}),
        "trajectory_completion_rate": _trajectory_completion(trajectories),
        "trajectory_frame_acceptance": _mean(trajectories, "accepted"),
        "trajectory_command_spike_rate": _mean(trajectories, "trajectory_spike"),
        "verification_interception_count": len(converged_but_rejected),
        "entry_action_counts": dict(Counter(str(row["entry_action"]) for row in rows)),
    }


def _reduction(proposed: float, baseline: float) -> float:
    return 1.0 - proposed / max(baseline, 1e-12)


def write_claim_gate_v2(
    records: list[dict[str, object]],
    risk_payload: dict[str, Any],
    output_dir: str | Path,
    *,
    policy_report: dict[str, Any],
) -> dict[str, Any]:
    """Write the preregistered pilot decision without pooling trajectory frames.

    Feasible-point routing, rejectable-point handling, and trajectory tracking
    are separate estimands. This prevents cheap false rejection or post-failure
    trajectory frames from manufacturing an apparent compute advantage.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline = _method_metrics(records, "fixed_robust_cascade")
    proposed = _method_metrics(records, "proposed_v2")
    threshold = _method_metrics(records, "threshold_guard_cascade")

    success_gap = proposed["point_feasible_success"] - baseline["point_feasible_success"]
    feasible_evaluation_reduction = _reduction(
        proposed["point_feasible_mean_function_evaluations"],
        baseline["point_feasible_mean_function_evaluations"],
    )
    feasible_latency_reduction = _reduction(
        proposed["point_feasible_p95_latency_seconds"],
        baseline["point_feasible_p95_latency_seconds"],
    )
    reject_evaluation_reduction = _reduction(
        proposed["point_rejectable_mean_function_evaluations"],
        baseline["point_rejectable_mean_function_evaluations"],
    )
    reject_latency_reduction = _reduction(
        proposed["point_rejectable_p95_latency_seconds"],
        baseline["point_rejectable_p95_latency_seconds"],
    )
    trajectory_completion_gap = (
        proposed["trajectory_completion_rate"] - baseline["trajectory_completion_rate"]
    )

    threshold_success_gap = (
        proposed["point_feasible_success"] - threshold["point_feasible_success"]
    )
    threshold_evaluation_reduction = _reduction(
        proposed["point_feasible_mean_function_evaluations"],
        threshold["point_feasible_mean_function_evaluations"],
    )
    threshold_latency_reduction = _reduction(
        proposed["point_feasible_p95_latency_seconds"],
        threshold["point_feasible_p95_latency_seconds"],
    )

    learned_policy_test = policy_report["learned_gate"]["test_metrics"]
    risk = risk_payload["test_metrics"]
    payload = {
        "comparison": "proposed_v2 vs identical fixed robust cascade",
        "estimands": {
            "routing": "known-feasible non-trajectory point queries",
            "rejection": "unreachable/discontinuous non-trajectory point queries",
            "sequence": "trajectory-level completion; frames are not pooled into routing success",
        },
        "baseline": baseline,
        "proposed": proposed,
        "threshold_guard": threshold,
        "point_feasible_success_gap": success_gap,
        "point_feasible_evaluation_reduction": feasible_evaluation_reduction,
        "point_feasible_p95_latency_reduction": feasible_latency_reduction,
        "point_rejectable_evaluation_reduction": reject_evaluation_reduction,
        "point_rejectable_p95_latency_reduction": reject_latency_reduction,
        "trajectory_completion_gap": trajectory_completion_gap,
        "threshold_guard_point_success_gap": threshold_success_gap,
        "threshold_guard_point_evaluation_reduction": threshold_evaluation_reduction,
        "threshold_guard_point_p95_latency_reduction": threshold_latency_reduction,
        "risk_reject_auroc": risk.get("fail_auroc", float("nan")),
        "risk_reject_ece": risk.get("fail_ece", float("nan")),
        "policy_test_metrics": learned_policy_test,
    }
    payload["success_gate_pass"] = bool(success_gap >= -0.01)
    payload["rejection_gate_pass"] = bool(proposed["point_rejectable_rejection"] >= 0.95)
    payload["routing_efficiency_gate_pass"] = bool(
        feasible_evaluation_reduction >= 0.10 and feasible_latency_reduction >= -0.25
    )
    payload["reject_efficiency_gate_pass"] = bool(
        reject_evaluation_reduction >= 0.50 and reject_latency_reduction >= 0.50
    )
    payload["risk_gate_pass"] = bool(
        risk.get("fail_auroc", float("nan")) >= 0.75
        and risk.get("fail_ece", float("inf")) <= 0.10
        and learned_policy_test["false_reject_rate"] <= 0.02
        and learned_policy_test["reject_recall"] >= 0.95
        and learned_policy_test["nonreject_macro_f1"] >= 0.30
    )
    payload["trajectory_gate_pass"] = bool(
        trajectory_completion_gap >= -0.10
        and proposed["trajectory_command_spike_rate"] <= 1e-12
    )
    payload["nontriviality_gate_pass"] = bool(
        threshold_success_gap >= -0.01
        and threshold_evaluation_reduction >= 0.05
        and threshold_latency_reduction >= -0.25
    )
    payload["pilot_gate_pass"] = bool(
        payload["success_gate_pass"]
        and payload["rejection_gate_pass"]
        and payload["routing_efficiency_gate_pass"]
        and payload["reject_efficiency_gate_pass"]
        and payload["risk_gate_pass"]
        and payload["trajectory_gate_pass"]
        and payload["nontriviality_gate_pass"]
    )
    payload["note"] = (
        "Per-run pilot gate only. A paper-level claim additionally requires locked-seed "
        "replication on both robots. Failed thresholds are reported without test-set retuning."
    )
    (output_path / "claim_gate_v2.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    return payload
