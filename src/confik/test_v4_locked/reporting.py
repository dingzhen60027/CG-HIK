"""Confirmatory summaries and preregistered inference for formal test_v4."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .benchmark import distribution


def _feasible(row: Mapping[str, Any]) -> bool:
    return bool(row["expected_reachable"]) and bool(row["continuity_feasible"])


def _command_reject(row: Mapping[str, Any]) -> bool:
    action = row.get("decision_action")
    if action is not None:
        return str(action) == "reject"
    return str(row.get("entry_action")) == "reject"


def _rate(rows: list[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return float(np.mean([predicate(row) for row in rows])) if rows else float("nan")


def _trajectory_groups(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["role"]), int(row["trajectory_id"]))].append(row)
    return grouped


def method_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the full metric contract without converting seeds to replicates."""

    result: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in records}):
        rows = [row for row in records if row["method"] == method]
        payload: dict[str, Any] = {
            "record_count": len(rows),
            "route_counts": dict(
                Counter(
                    str(row.get("decision_action") or row["entry_action"])
                    for row in rows
                )
            ),
        }
        subsets: dict[str, list[dict[str, Any]]] = {
            "all": rows,
            "id_points_feasible": [
                row for row in rows
                if row["role"] == "id_points" and _feasible(row)
            ],
            "id_points_infeasible": [
                row for row in rows
                if row["role"] == "id_points" and not _feasible(row)
            ],
            "ood_points_feasible": [
                row for row in rows
                if row["role"] == "ood_points" and _feasible(row)
            ],
            "ood_points_infeasible": [
                row for row in rows
                if row["role"] == "ood_points" and not _feasible(row)
            ],
            "id_trajectories": [row for row in rows if row["role"] == "id_trajectories"],
            "ood_trajectories": [row for row in rows if row["role"] == "ood_trajectories"],
        }
        for name, selected in subsets.items():
            trajectory_groups = _trajectory_groups(
                row for row in selected if bool(row["is_trajectory"])
            )
            transitions = 0
            switches = 0
            for trajectory_rows in trajectory_groups.values():
                ordered = sorted(trajectory_rows, key=lambda row: int(row["time_index"]))
                actions = [
                    str(row.get("decision_action") or row["entry_action"])
                    for row in ordered
                ]
                transitions += max(len(actions) - 1, 0)
                switches += sum(left != right for left, right in zip(actions[:-1], actions[1:]))
            payload[name] = {
                "count": len(selected),
                "verified_success_rate": _rate(selected, lambda row: bool(row["verified_success"])),
                "verified_failure_rate": _rate(selected, lambda row: not bool(row["verified_success"])),
                "command_false_reject_rate": _rate(selected, _command_reject),
                "function_evaluations": distribution(
                    [float(row["function_evaluations"]) for row in selected]
                ),
                "latency_ms": distribution([float(row["latency_ms"]) for row in selected]),
                "deadline_miss_rate_20ms": _rate(
                    selected, lambda row: float(row["latency_ms"]) > 20.0
                ),
                "trajectory_count": len(trajectory_groups),
                "trajectory_completion_rate": (
                    float(
                        np.mean(
                            [
                                all(bool(row["verified_success"]) for row in group)
                                for group in trajectory_groups.values()
                            ]
                        )
                    )
                    if trajectory_groups
                    else float("nan")
                ),
                "trajectory_command_spike_rate": _rate(
                    selected, lambda row: bool(row["trajectory_command_spike"])
                ),
                "route_switch_count": switches if trajectory_groups else None,
                "route_switch_rate": (
                    switches / transitions if trajectory_groups and transitions else None
                ),
            }
        result[method] = payload
    return result


def ood_and_abstention_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate OOD detection, command rejection, and fixed-cascade deferral."""

    v4 = [row for row in records if row["method"] == "proposed_v4"]
    if not v4:
        return {"available": False}
    if any(row.get("ood_score") is None for row in v4):
        raise RuntimeError("formal v4 record is missing raw OOD scores")
    labels = np.asarray([int(row["domain"] == "ood") for row in v4], dtype=np.int64)
    scores = np.asarray([float(row["ood_score"]) for row in v4], dtype=np.float64)
    ood_auroc = (
        float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    )
    ood_auprc = (
        float(average_precision_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )

    fixed_index = {
        (str(row["role"]), str(row["source_query_sha256"])): row
        for row in records
        if row["method"] == "fixed_robust_cascade" and not row["is_trajectory"]
    }
    deferred_points = [
        row for row in v4
        if row["decision_action"] == "defer" and not row["is_trajectory"]
    ]
    semantic_matches = 0
    recovered = 0
    for row in deferred_points:
        key = (str(row["role"]), str(row["source_query_sha256"]))
        fixed = fixed_index.get(key)
        if fixed is None:
            raise RuntimeError("deferred v4 point lacks its same-query fixed comparator")
        recovered += int(bool(row["verified_success"]))
        semantic_matches += int(
            bool(row["verified_success"]) == bool(fixed["verified_success"])
            and int(row["function_evaluations"]) == int(fixed["function_evaluations"])
            and list(row["executed_stages"]) == list(fixed["executed_stages"])
        )

    rejected = [row for row in v4 if row["decision_action"] == "reject"]
    zero_fev = [row for row in rejected if int(row["function_evaluations"]) == 0]
    zero_stages = [row for row in rejected if not row["executed_stages"]]
    feasible_ood = [
        row for row in v4
        if row["role"] == "ood_points" and _feasible(row)
    ]
    feasible_id = [
        row for row in v4
        if row["role"] == "id_points" and _feasible(row)
    ]
    return {
        "available": True,
        "sample_count": len(v4),
        "id_count": int(np.sum(labels == 0)),
        "ood_count": int(np.sum(labels == 1)),
        "ood_auroc": ood_auroc,
        "ood_auprc": ood_auprc,
        "ood_score_definition": "raw frozen Mahalanobis score from V4Decision",
        "id_feasible_command_false_reject_rate": _rate(feasible_id, _command_reject),
        "ood_feasible_command_false_reject_rate": _rate(feasible_ood, _command_reject),
        "defer_count_points": len(deferred_points),
        "defer_recovery_success_rate": (
            recovered / len(deferred_points) if deferred_points else 1.0
        ),
        "defer_fixed_semantic_match_rate": (
            semantic_matches / len(deferred_points) if deferred_points else 1.0
        ),
        "defer_semantic_scope": "point queries only; closed-loop histories are method-specific",
        "command_reject_count": len(rejected),
        "command_reject_zero_fev_rate": len(zero_fev) / len(rejected) if rejected else 1.0,
        "command_reject_zero_stage_rate": len(zero_stages) / len(rejected) if rejected else 1.0,
        "command_reject_max_fev": max(
            (int(row["function_evaluations"]) for row in rejected), default=0
        ),
        "command_reject_max_executed_stage_count": max(
            (len(row["executed_stages"]) for row in rejected), default=0
        ),
    }


def _paired_rows(
    records: list[dict[str, Any]],
    left_method: str,
    right_method: str,
    include: Callable[[Mapping[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in records if include(row)]
    left_index = {
        (str(row["role"]), str(row["source_query_sha256"])): row
        for row in selected if row["method"] == left_method
    }
    right_index = {
        (str(row["role"]), str(row["source_query_sha256"])): row
        for row in selected if row["method"] == right_method
    }
    if set(left_index) != set(right_index):
        raise RuntimeError(
            f"paired records differ for {left_method} and {right_method}"
        )
    keys = sorted(left_index)
    return [left_index[key] for key in keys], [right_index[key] for key in keys]


def _bootstrap_rows(
    left: np.ndarray,
    right: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap inputs must be non-empty equal vectors")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(left), size=len(left))
        estimates[index] = statistic(left[sampled], right[sampled])
    estimate = statistic(left, right)
    return {
        "pair_count": int(len(left)),
        "estimate": float(estimate),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
        "bootstrap_samples": int(samples),
        "bootstrap_values": estimates,
    }


def _trajectory_metric_vectors(
    records: list[dict[str, Any]],
    left_method: str,
    right_method: str,
    value: Callable[[list[dict[str, Any]]], float],
) -> tuple[np.ndarray, np.ndarray]:
    left_groups = _trajectory_groups(
        row for row in records
        if row["method"] == left_method and bool(row["is_trajectory"])
    )
    right_groups = _trajectory_groups(
        row for row in records
        if row["method"] == right_method and bool(row["is_trajectory"])
    )
    if set(left_groups) != set(right_groups):
        raise RuntimeError("trajectory cluster keys differ between paired methods")
    keys = sorted(left_groups)
    return (
        np.asarray([value(left_groups[key]) for key in keys], dtype=np.float64),
        np.asarray([value(right_groups[key]) for key in keys], dtype=np.float64),
    )


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted[name] = running
    return adjusted


def paired_confirmatory_intervals(
    records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    """Run 10k query-paired or trajectory-cluster paired bootstraps."""

    point_include = lambda row: (
        row["role"] in {"id_points", "ood_points"} and _feasible(row)
    )
    fixed, v4 = _paired_rows(
        records, "fixed_robust_cascade", "proposed_v4", point_include
    )
    fixed_success = np.asarray([float(row["verified_success"]) for row in fixed])
    v4_success = np.asarray([float(row["verified_success"]) for row in v4])
    fixed_latency = np.asarray([float(row["latency_ms"]) for row in fixed])
    v4_latency = np.asarray([float(row["latency_ms"]) for row in v4])

    metrics: dict[str, dict[str, Any]] = {
        "feasible_success_gap": _bootstrap_rows(
            fixed_success,
            v4_success,
            lambda left, right: float(np.mean(right - left)),
            samples=samples,
            seed=seed,
        ),
        "feasible_p95_latency_ratio": _bootstrap_rows(
            fixed_latency,
            v4_latency,
            lambda left, right: float(np.percentile(right, 95) / max(np.percentile(left, 95), 1e-12)),
            samples=samples,
            seed=seed + 1,
        ),
        "feasible_p99_latency_ratio": _bootstrap_rows(
            fixed_latency,
            v4_latency,
            lambda left, right: float(np.percentile(right, 99) / max(np.percentile(left, 99), 1e-12)),
            samples=samples,
            seed=seed + 2,
        ),
    }

    fixed_completion, v4_completion = _trajectory_metric_vectors(
        records,
        "fixed_robust_cascade",
        "proposed_v4",
        lambda rows: float(all(bool(row["verified_success"]) for row in rows)),
    )
    metrics["trajectory_completion_gap"] = _bootstrap_rows(
        fixed_completion,
        v4_completion,
        lambda left, right: float(np.mean(right - left)),
        samples=samples,
        seed=seed + 3,
    )
    fixed_spike, v4_spike = _trajectory_metric_vectors(
        records,
        "fixed_robust_cascade",
        "proposed_v4",
        lambda rows: float(np.mean([bool(row["trajectory_command_spike"]) for row in rows])),
    )
    metrics["trajectory_command_spike_increase"] = _bootstrap_rows(
        fixed_spike,
        v4_spike,
        lambda left, right: float(np.mean(right - left)),
        samples=samples,
        seed=seed + 4,
    )

    v3_ood, v4_ood = _paired_rows(
        records,
        "proposed_v2",
        "proposed_v4",
        lambda row: row["role"] == "ood_points" and _feasible(row),
    )
    v3_reject = np.asarray([float(_command_reject(row)) for row in v3_ood])
    v4_reject = np.asarray([float(_command_reject(row)) for row in v4_ood])
    metrics["ood_feasible_false_reject_improvement_vs_v3"] = _bootstrap_rows(
        v3_reject,
        v4_reject,
        lambda left, right: float(np.mean(left - right)),
        samples=samples,
        seed=seed + 5,
    )

    nulls = {
        "feasible_success_gap": float(gates["feasible_success_gap_ci_lower_min"]),
        "feasible_p95_latency_ratio": float(gates["feasible_p95_ratio_strict_max"]),
        "feasible_p99_latency_ratio": float(gates["feasible_p99_ratio_max"]),
        "trajectory_completion_gap": float(gates["trajectory_completion_gap_min"]),
        "trajectory_command_spike_increase": float(gates["trajectory_command_spike_increase_max"]),
        "ood_feasible_false_reject_improvement_vs_v3": float(
            gates["ood_feasible_false_reject_improvement_vs_v3_min"]
        ),
    }
    upper_is_good = {
        "feasible_p95_latency_ratio",
        "feasible_p99_latency_ratio",
        "trajectory_command_spike_increase",
    }
    pvalues: dict[str, float] = {}
    for name, payload in metrics.items():
        values = np.asarray(payload.pop("bootstrap_values"), dtype=np.float64)
        null = nulls[name]
        pvalues[name] = float(
            (np.sum(values >= null) + 1) / (len(values) + 1)
            if name in upper_is_good
            else (np.sum(values <= null) + 1) / (len(values) + 1)
        )
        payload["preregistered_margin"] = null
        payload["one_sided_unadjusted_p"] = pvalues[name]
    adjusted = _holm(pvalues)
    for name, value in adjusted.items():
        metrics[name]["holm_adjusted_p"] = value
    return {
        "resampling": {
            "point_unit": "source_query_sha256",
            "trajectory_unit": "(explicit_role, trajectory_id)",
            "paired": True,
            "samples": int(samples),
            "confidence_level": 0.95,
            "multiplicity_correction": "Holm",
        },
        "metrics": metrics,
    }


def claim_gate_report(
    records: list[dict[str, Any]],
    *,
    gates: Mapping[str, float],
    intervals: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = method_metrics(records)
    abstention = ood_and_abstention_metrics(records)
    bootstrap = intervals["metrics"]
    fixed = metrics["fixed_robust_cascade"]
    v4 = metrics["proposed_v4"]
    feasible_fixed = [
        row for row in records
        if row["method"] == "fixed_robust_cascade"
        and row["role"] in {"id_points", "ood_points"}
        and _feasible(row)
    ]
    feasible_v4 = [
        row for row in records
        if row["method"] == "proposed_v4"
        and row["role"] in {"id_points", "ood_points"}
        and _feasible(row)
    ]
    fixed_latency = distribution([float(row["latency_ms"]) for row in feasible_fixed])
    v4_latency = distribution([float(row["latency_ms"]) for row in feasible_v4])
    p95_ratio = float(v4_latency["p95"]) / max(float(fixed_latency["p95"]), 1e-12)
    p99_ratio = float(v4_latency["p99"]) / max(float(fixed_latency["p99"]), 1e-12)
    contract_count = sum(len(row["contract_violations"]) for row in records)
    checks = {
        "feasible_success_noninferiority": (
            float(bootstrap["feasible_success_gap"]["ci_lower"])
            >= float(gates["feasible_success_gap_ci_lower_min"])
        ),
        "zero_contract_violations": contract_count <= int(gates["contract_violation_count_max"]),
        "id_feasible_false_reject": (
            float(abstention["id_feasible_command_false_reject_rate"])
            <= float(gates["id_feasible_false_reject_rate_max"])
        ),
        "feasible_p95_latency": p95_ratio < float(gates["feasible_p95_ratio_strict_max"]),
        "feasible_p99_latency": p99_ratio <= float(gates["feasible_p99_ratio_max"]),
        "trajectory_completion_noninferiority": (
            float(bootstrap["trajectory_completion_gap"]["ci_lower"])
            >= float(gates["trajectory_completion_gap_min"])
        ),
        "trajectory_command_spike": (
            float(bootstrap["trajectory_command_spike_increase"]["estimate"])
            <= float(gates["trajectory_command_spike_increase_max"])
        ),
        "ood_feasible_false_reject_improvement": (
            float(
                bootstrap["ood_feasible_false_reject_improvement_vs_v3"]["estimate"]
            )
            >= float(gates["ood_feasible_false_reject_improvement_vs_v3_min"])
        ),
        "defer_fixed_semantic_match": (
            float(abstention["defer_fixed_semantic_match_rate"])
            >= float(gates["defer_fixed_semantic_match_rate_min"])
        ),
        "reject_zero_fev": (
            int(abstention["command_reject_max_fev"])
            <= int(gates["reject_function_evaluations_max"])
        ),
        "reject_zero_stages": (
            int(abstention["command_reject_max_executed_stage_count"])
            <= int(gates["reject_executed_stage_count_max"])
        ),
    }
    return {
        "comparison": "proposed_v4 vs fixed robust cascade; OOD false reject also vs frozen proposed_v2",
        "primary_training_seed": 17,
        "fixed_metrics": fixed,
        "proposed_v4_metrics": v4,
        "ood_and_abstention": abstention,
        "feasible_latency_ms": {"fixed": fixed_latency, "proposed_v4": v4_latency},
        "feasible_p95_latency_ratio": p95_ratio,
        "feasible_p99_latency_ratio": p99_ratio,
        "contract_violation_count": contract_count,
        "checks": checks,
        "formal_gate_pass": all(checks.values()),
        "test_set_retuning_performed": False,
        "threshold_or_gate_changes_after_test": False,
    }


__all__ = [
    "claim_gate_report",
    "method_metrics",
    "ood_and_abstention_metrics",
    "paired_confirmatory_intervals",
]
