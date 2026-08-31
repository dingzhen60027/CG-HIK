"""Confirmatory summaries and preregistered inference for formal test_v4."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .benchmark import distribution


CONFIRMATORY_INFERENCE_METRICS = (
    "feasible_success_gap",
    "feasible_p95_latency_ratio",
    "feasible_p99_latency_ratio",
    "trajectory_completion_gap",
)
DEFER_Q_ATOL = 1.0e-12


def _feasible(row: Mapping[str, Any]) -> bool:
    return bool(row["expected_reachable"]) and bool(row["continuity_feasible"])


def _command_reject(row: Mapping[str, Any]) -> bool:
    action = row.get("decision_action")
    if action is not None:
        return str(action) == "reject"
    return str(row.get("entry_action")) == "reject"


def _rate(rows: list[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return float(np.mean([predicate(row) for row in rows])) if rows else float("nan")


def _latency_repeats_ms(row: Mapping[str, Any]) -> np.ndarray:
    """Return the raw solve-call latency samples retained for one query.

    Formal records always contain ``latency_repeats_ns``.  The scalar fallback
    keeps this reporting helper usable for old development fixtures, but formal
    manifests explicitly identify whether raw repeats were present.
    """

    values = row.get("latency_repeats_ns")
    if values is None:
        return np.asarray([float(row["latency_ms"])], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("latency_repeats_ns must be a non-empty finite vector")
    expected = row.get("latency_repeat_count")
    if expected is not None and int(expected) != len(array):
        raise ValueError("latency repeat count differs from retained samples")
    return array / 1.0e6


def _raw_latency_values(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    return [
        float(value)
        for row in rows
        for value in _latency_repeats_ms(row).tolist()
    ]


def _query_median_latency_values(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    return [float(np.median(_latency_repeats_ms(row))) for row in rows]


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
            raw_latency_ms = _raw_latency_values(selected)
            query_median_latency_ms = _query_median_latency_values(selected)
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
                # Headline latency is the distribution of actual solve calls,
                # not the distribution of per-query timing medians.
                "latency_ms": distribution(raw_latency_ms),
                "latency_estimand": "raw_solve_calls",
                "latency_raw_call_count": len(raw_latency_ms),
                "latency_per_query_median_ms": distribution(
                    query_median_latency_ms
                ),
                "deadline_miss_rate_20ms": _rate(
                    [
                        {"miss": value > 20.0}
                        for value in raw_latency_ms
                    ],
                    lambda row: bool(row["miss"]),
                ),
                "deadline_any_repeat_miss_rate_20ms": _rate(
                    selected,
                    lambda row: bool(np.any(_latency_repeats_ms(row) > 20.0)),
                ),
                "deadline_query_median_miss_rate_20ms": _rate(
                    selected,
                    lambda row: float(np.median(_latency_repeats_ms(row))) > 20.0,
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


def _binary_score_metrics(
    labels: np.ndarray, scores: np.ndarray, *, unit: str
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not np.all(np.isfinite(scores)):
        raise ValueError("OOD labels and scores must be equal finite vectors")
    estimable = bool(len(labels) and np.unique(labels).size == 2)
    return {
        "status": "estimated" if estimable else "not_estimated",
        "unit": unit,
        "sample_count": int(len(labels)),
        "id_count": int(np.sum(labels == 0)),
        "ood_count": int(np.sum(labels == 1)),
        "auroc": float(roc_auc_score(labels, scores)) if estimable else None,
        "auprc": float(average_precision_score(labels, scores)) if estimable else None,
    }


def _command_q_mismatch(
    left: Mapping[str, Any], right: Mapping[str, Any], *, tolerance: float
) -> tuple[bool, float | None]:
    left_q = left.get("command_q")
    right_q = right.get("command_q")
    if left_q is None or right_q is None:
        return left_q is not None or right_q is not None, None
    left_array = np.asarray(left_q, dtype=np.float64)
    right_array = np.asarray(right_q, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return True, None
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        return True, None
    maximum = float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0
    return maximum > tolerance, maximum


def _defer_mismatch_fields(
    deferred: Mapping[str, Any], fixed: Mapping[str, Any], *, q_atol: float
) -> tuple[list[str], float | None]:
    mismatches: list[str] = []
    scalar_fields = (
        "accepted",
        "function_evaluations",
        "iterations",
        "fallback_used",
        "reject_reason",
    )
    sequence_fields = ("executed_stages", "verification_reasons")
    for field in scalar_fields:
        if deferred.get(field) != fixed.get(field):
            mismatches.append(field)
    for field in sequence_fields:
        if list(deferred.get(field, ())) != list(fixed.get(field, ())):
            mismatches.append(field)
    q_mismatch, q_difference = _command_q_mismatch(
        deferred, fixed, tolerance=q_atol
    )
    if q_mismatch:
        mismatches.append("command_q")
    return mismatches, q_difference


def ood_and_abstention_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate OOD detection, command rejection, and fixed-cascade deferral."""

    v4 = [row for row in records if row["method"] == "proposed_v4"]
    if not v4:
        return {"available": False}
    if any(row.get("ood_score") is None for row in v4):
        raise RuntimeError("formal v4 record is missing raw OOD scores")

    point_rows = [row for row in v4 if not bool(row["is_trajectory"])]
    point_labels = np.asarray(
        [int(row["domain"] == "ood") for row in point_rows], dtype=np.int64
    )
    point_scores = np.asarray(
        [float(row["ood_score"]) for row in point_rows], dtype=np.float64
    )
    point_detection = _binary_score_metrics(
        point_labels, point_scores, unit="independent_point_query"
    )

    trajectory_groups = _trajectory_groups(
        row for row in v4 if bool(row["is_trajectory"])
    )
    trajectory_labels: list[int] = []
    trajectory_scores: list[float] = []
    for group in trajectory_groups.values():
        domains = {str(row["domain"]) for row in group}
        if len(domains) != 1:
            raise RuntimeError("one trajectory contains mixed ID/OOD domain labels")
        trajectory_labels.append(int(next(iter(domains)) == "ood"))
        trajectory_scores.append(float(np.mean([float(row["ood_score"]) for row in group])))
    trajectory_detection = _binary_score_metrics(
        np.asarray(trajectory_labels, dtype=np.int64),
        np.asarray(trajectory_scores, dtype=np.float64),
        unit="whole_trajectory_mean_frame_score",
    )

    fixed_points = [
        row
        for row in records
        if row["method"] == "fixed_robust_cascade" and not row["is_trajectory"]
    ]
    fixed_index = {
        (str(row["role"]), str(row["source_query_sha256"])): row
        for row in fixed_points
    }
    deferred_points = [
        row
        for row in point_rows
        if row.get("decision_action") == "defer"
    ]
    recovered = 0
    semantic_matches = 0
    mismatch_counts: Counter[str] = Counter()
    maximum_q_difference = 0.0
    q_comparison_count = 0
    for row in deferred_points:
        key = (str(row["role"]), str(row["source_query_sha256"]))
        fixed = fixed_index.get(key)
        if fixed is None:
            raise RuntimeError("deferred v4 point lacks its same-query fixed comparator")
        recovered += int(bool(row["verified_success"]))
        mismatches, q_difference = _defer_mismatch_fields(
            row, fixed, q_atol=DEFER_Q_ATOL
        )
        mismatch_counts.update(mismatches)
        semantic_matches += int(not mismatches)
        if q_difference is not None:
            q_comparison_count += 1
            maximum_q_difference = max(maximum_q_difference, q_difference)

    rejected = [row for row in v4 if row.get("decision_action") == "reject"]
    rejected_points = [row for row in rejected if not row["is_trajectory"]]
    zero_fev = [row for row in rejected if int(row["function_evaluations"]) == 0]
    zero_stages = [row for row in rejected if not row["executed_stages"]]
    feasible_ood = [
        row for row in point_rows if row["role"] == "ood_points" and _feasible(row)
    ]
    feasible_id = [
        row for row in point_rows if row["role"] == "id_points" and _feasible(row)
    ]
    infeasible_v4 = [row for row in point_rows if not _feasible(row)]
    infeasible_fixed = []
    for row in infeasible_v4:
        key = (str(row["role"]), str(row["source_query_sha256"]))
        fixed = fixed_index.get(key)
        if fixed is None:
            raise RuntimeError("infeasible v4 point lacks its same-query fixed comparator")
        infeasible_fixed.append(fixed)
    fixed_fev_total = int(
        sum(int(row["function_evaluations"]) for row in infeasible_fixed)
    )
    v4_fev_total = int(sum(int(row["function_evaluations"]) for row in infeasible_v4))
    defer_status = "estimated" if deferred_points else "not_estimated"
    reject_status = "estimated" if rejected else "not_estimated"
    return {
        "available": True,
        "sample_count": len(v4),
        "ood_score_definition": "raw frozen Mahalanobis score from V4Decision",
        "ood_headline_scope": "independent_point_queries",
        "ood_point_detection": point_detection,
        "ood_trajectory_cluster_detection": trajectory_detection,
        # Backward-compatible headline aliases are deliberately point-level.
        "ood_auroc": point_detection["auroc"],
        "ood_auprc": point_detection["auprc"],
        "id_feasible_command_false_reject_rate": _rate(feasible_id, _command_reject),
        "ood_feasible_command_false_reject_rate": _rate(feasible_ood, _command_reject),
        "defer_support_status": defer_status,
        "defer_count_points": len(deferred_points),
        "defer_recovery_success_rate": (
            recovered / len(deferred_points) if deferred_points else None
        ),
        "defer_fixed_semantic_match_rate": (
            semantic_matches / len(deferred_points) if deferred_points else None
        ),
        "defer_fixed_semantic_mismatch_count": len(deferred_points) - semantic_matches,
        "defer_fixed_semantic_mismatch_fields": dict(sorted(mismatch_counts.items())),
        "defer_command_q_tolerance": DEFER_Q_ATOL,
        "defer_command_q_comparison_count": q_comparison_count,
        "defer_command_q_max_abs_difference": (
            maximum_q_difference if q_comparison_count else None
        ),
        "defer_semantic_scope": "point queries only; closed-loop histories are method-specific",
        "command_reject_support_status": reject_status,
        "command_reject_count": len(rejected),
        "command_reject_count_points": len(rejected_points),
        "command_reject_zero_fev_rate": (
            len(zero_fev) / len(rejected) if rejected else None
        ),
        "command_reject_zero_stage_rate": (
            len(zero_stages) / len(rejected) if rejected else None
        ),
        "command_reject_max_fev": (
            max(int(row["function_evaluations"]) for row in rejected)
            if rejected
            else None
        ),
        "command_reject_max_executed_stage_count": (
            max(len(row["executed_stages"]) for row in rejected)
            if rejected
            else None
        ),
        "infeasible_point_count": len(infeasible_v4),
        "infeasible_command_reject_count": sum(
            int(_command_reject(row)) for row in infeasible_v4
        ),
        "infeasible_reject_support_status": (
            "estimated"
            if any(_command_reject(row) for row in infeasible_v4)
            else "not_estimated"
        ),
        "infeasible_command_reject_recall": (
            _rate(infeasible_v4, _command_reject) if infeasible_v4 else None
        ),
        "infeasible_fixed_function_evaluations_total": fixed_fev_total,
        "infeasible_v4_function_evaluations_total": v4_fev_total,
        "infeasible_function_evaluations_avoided_vs_fixed": (
            fixed_fev_total - v4_fev_total if infeasible_v4 else None
        ),
        "infeasible_function_evaluations_avoided_fraction_vs_fixed": (
            (fixed_fev_total - v4_fev_total) / fixed_fev_total
            if infeasible_v4 and fixed_fev_total > 0
            else None
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


def _latency_repeat_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    repeats = [_latency_repeats_ms(row) for row in rows]
    widths = {len(values) for values in repeats}
    if len(widths) != 1:
        raise RuntimeError("formal paired latency rows have unequal repeat counts")
    return np.stack(repeats)


def _bootstrap_query_clustered_latency_ratio(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    percentile: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(left_rows) != len(right_rows) or not left_rows:
        raise ValueError("paired latency bootstrap requires equal non-empty query rows")
    left = _latency_repeat_matrix(left_rows)
    right = _latency_repeat_matrix(right_rows)
    if left.shape != right.shape:
        raise RuntimeError("fixed and v4 formal latency repeat matrices differ")

    def ratio(left_values: np.ndarray, right_values: np.ndarray) -> float:
        denominator = max(float(np.percentile(left_values, percentile)), 1.0e-12)
        return float(np.percentile(right_values, percentile)) / denominator

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_queries = rng.integers(0, len(left), size=len(left))
        estimates[index] = ratio(
            left[sampled_queries].reshape(-1),
            right[sampled_queries].reshape(-1),
        )
    return {
        "pair_count": int(len(left)),
        "raw_call_count_per_method": int(left.size),
        "repeat_count_per_query": int(left.shape[1]),
        "estimand": f"raw_solve_call_p{int(percentile)}_ratio",
        "resampling_unit": "source_query_sha256",
        "estimate": ratio(left.reshape(-1), right.reshape(-1)),
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
    """Run the four prespecified query/trajectory-cluster inference claims.

    Latency queries are resampled as clusters and all raw solve-call repeats in
    each sampled query are then flattened for the P95/P99 statistic.  Holm is
    intentionally not applied per robot; :func:`joint_holm_confirmatory`
    applies it once across Panda and UR5e after both runs are sealed.
    """

    point_include = lambda row: (
        row["role"] in {"id_points", "ood_points"} and _feasible(row)
    )
    fixed, v4 = _paired_rows(
        records, "fixed_robust_cascade", "proposed_v4", point_include
    )
    fixed_success = np.asarray([float(row["verified_success"]) for row in fixed])
    v4_success = np.asarray([float(row["verified_success"]) for row in v4])

    metrics: dict[str, dict[str, Any]] = {
        "feasible_success_gap": _bootstrap_rows(
            fixed_success,
            v4_success,
            lambda left, right: float(np.mean(right - left)),
            samples=samples,
            seed=seed,
        ),
        "feasible_p95_latency_ratio": _bootstrap_query_clustered_latency_ratio(
            fixed,
            v4,
            percentile=95,
            samples=samples,
            seed=seed + 1,
        ),
        "feasible_p99_latency_ratio": _bootstrap_query_clustered_latency_ratio(
            fixed,
            v4,
            percentile=99,
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
    v3_ood, v4_ood = _paired_rows(
        records,
        "proposed_v2",
        "proposed_v4",
        lambda row: row["role"] == "ood_points" and _feasible(row),
    )
    v3_reject = np.asarray([float(_command_reject(row)) for row in v3_ood])
    v4_reject = np.asarray([float(_command_reject(row)) for row in v4_ood])
    operational = {
        "trajectory_command_spike_increase": {
            "pair_count": int(len(fixed_spike)),
            "unit": "whole_trajectory",
            "estimate": float(np.mean(v4_spike - fixed_spike)),
            "gate_type": "frozen_operational_finite_test",
            "included_in_holm_family": False,
        },
        "ood_feasible_false_reject_improvement_vs_v3": {
            "pair_count": int(len(v3_reject)),
            "unit": "independent_ood_point_query",
            "estimate": float(np.mean(v3_reject - v4_reject)),
            "gate_type": "frozen_operational_finite_test",
            "included_in_holm_family": False,
        },
    }

    nulls = {
        "feasible_success_gap": float(gates["feasible_success_gap_ci_lower_min"]),
        "feasible_p95_latency_ratio": float(gates["feasible_p95_ratio_strict_max"]),
        "feasible_p99_latency_ratio": float(gates["feasible_p99_ratio_max"]),
        "trajectory_completion_gap": float(gates["trajectory_completion_gap_min"]),
    }
    upper_is_good = {
        "feasible_p95_latency_ratio",
        "feasible_p99_latency_ratio",
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
    if tuple(metrics) != CONFIRMATORY_INFERENCE_METRICS:
        raise RuntimeError("formal confirmatory inference family changed")
    return {
        "resampling": {
            "point_unit": "source_query_sha256",
            "trajectory_unit": "(explicit_role, trajectory_id)",
            "paired": True,
            "samples": int(samples),
            "confidence_level": 0.95,
            "latency_within_query_unit": "raw_solve_calls",
            "multiplicity_correction": "joint Holm after both robots are sealed",
        },
        "inference_family": {
            "members": list(CONFIRMATORY_INFERENCE_METRICS),
            "member_count_per_robot": len(CONFIRMATORY_INFERENCE_METRICS),
            "per_robot_holm_applied": False,
            "joint_robot_holm_required": True,
        },
        "metrics": metrics,
        "operational_finite_test_metrics": operational,
    }


def joint_holm_confirmatory(
    robot_intervals: Mapping[str, Mapping[str, Any]], *, alpha: float = 0.05
) -> dict[str, Any]:
    """Apply one Holm correction across both robots and all four claims."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Holm alpha must lie strictly between zero and one")
    if set(robot_intervals) != {"panda", "ur5e"}:
        raise ValueError("joint formal Holm requires exactly Panda and UR5e")
    unadjusted: dict[str, float] = {}
    for robot in ("panda", "ur5e"):
        payload = robot_intervals[robot]
        family = payload.get("inference_family", {})
        if tuple(family.get("members", ())) != CONFIRMATORY_INFERENCE_METRICS:
            raise RuntimeError(f"{robot} confirmatory inference family changed")
        metrics = payload.get("metrics", {})
        # JSON objects are unordered. The formal writer uses ``sort_keys=True``,
        # so a round trip cannot preserve the construction-time insertion
        # order even though the exact prespecified member set is unchanged.
        if (
            not isinstance(metrics, Mapping)
            or len(metrics) != len(CONFIRMATORY_INFERENCE_METRICS)
            or set(metrics) != set(CONFIRMATORY_INFERENCE_METRICS)
        ):
            raise RuntimeError(f"{robot} confirmatory metrics changed")
        for name in CONFIRMATORY_INFERENCE_METRICS:
            value = float(metrics[name]["one_sided_unadjusted_p"])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid unadjusted p-value for {robot}/{name}")
            unadjusted[f"{robot}/{name}"] = value
    adjusted = _holm(unadjusted)
    hypotheses = {
        name: {
            "robot": name.split("/", 1)[0],
            "metric": name.split("/", 1)[1],
            "one_sided_unadjusted_p": unadjusted[name],
            "holm_adjusted_p": adjusted[name],
            "reject_margin_null": adjusted[name] <= float(alpha),
        }
        for name in unadjusted
    }
    return {
        "method": "Holm",
        "scope": "Panda and UR5e x four prespecified confirmatory claims",
        "alpha": float(alpha),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "all_confirmatory_nulls_rejected": all(
            bool(payload["reject_margin_null"]) for payload in hypotheses.values()
        ),
        "operational_finite_test_gates_included": False,
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
    operational = intervals["operational_finite_test_metrics"]
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
    fixed_latency = distribution(_raw_latency_values(feasible_fixed))
    v4_latency = distribution(_raw_latency_values(feasible_v4))
    fixed_query_median_latency = distribution(
        _query_median_latency_values(feasible_fixed)
    )
    v4_query_median_latency = distribution(
        _query_median_latency_values(feasible_v4)
    )
    p95_ratio = float(v4_latency["p95"]) / max(float(fixed_latency["p95"]), 1e-12)
    p99_ratio = float(v4_latency["p99"]) / max(float(fixed_latency["p99"]), 1e-12)
    contract_count = sum(len(row["contract_violations"]) for row in records)
    reject_support_min = int(gates.get("reject_support_count_min", 1))
    defer_support_min = int(gates.get("defer_support_count_min", 1))
    reject_count = int(abstention["infeasible_command_reject_count"])
    defer_count = int(abstention["defer_count_points"])
    reject_max_fev = abstention["command_reject_max_fev"]
    reject_max_stages = abstention["command_reject_max_executed_stage_count"]
    defer_match = abstention["defer_fixed_semantic_match_rate"]
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
            float(operational["trajectory_command_spike_increase"]["estimate"])
            <= float(gates["trajectory_command_spike_increase_max"])
        ),
        "ood_feasible_false_reject_improvement": (
            float(
                operational[
                    "ood_feasible_false_reject_improvement_vs_v3"
                ]["estimate"]
            )
            >= float(gates["ood_feasible_false_reject_improvement_vs_v3_min"])
        ),
        "defer_support": defer_count >= defer_support_min,
        "defer_fixed_semantic_match": (
            defer_match is not None
            and float(defer_match)
            >= float(gates["defer_fixed_semantic_match_rate_min"])
        ),
        "reject_support": reject_count >= reject_support_min,
        "reject_zero_fev": (
            reject_max_fev is not None
            and int(reject_max_fev)
            <= int(gates["reject_function_evaluations_max"])
        ),
        "reject_zero_stages": (
            reject_max_stages is not None
            and int(reject_max_stages)
            <= int(gates["reject_executed_stage_count_max"])
        ),
    }
    return {
        "comparison": "proposed_v4 vs fixed robust cascade; OOD false reject also vs frozen proposed_v2",
        "primary_training_seed": 17,
        "fixed_metrics": fixed,
        "proposed_v4_metrics": v4,
        "ood_and_abstention": abstention,
        "feasible_latency_ms": {
            "estimand": "raw_solve_calls",
            "fixed": fixed_latency,
            "proposed_v4": v4_latency,
        },
        "feasible_latency_per_query_median_ms": {
            "estimand": "one_median_per_source_query_sha256",
            "fixed": fixed_query_median_latency,
            "proposed_v4": v4_query_median_latency,
        },
        "feasible_p95_latency_ratio": p95_ratio,
        "feasible_p99_latency_ratio": p99_ratio,
        "contract_violation_count": contract_count,
        "support_requirements": {
            "reject_support_count_min": reject_support_min,
            "defer_support_count_min": defer_support_min,
            "reject_support_definition": "known_infeasible_point_command_rejects",
            "reject_support_count_observed": reject_count,
            "defer_support_count_observed": defer_count,
        },
        "confirmatory_inference": {
            "family_members": list(CONFIRMATORY_INFERENCE_METRICS),
            "joint_panda_ur5e_holm_required": True,
            "joint_holm_applied_in_this_robot_report": False,
        },
        "operational_finite_test_metrics": operational,
        "checks": checks,
        "formal_gate_pass": all(checks.values()),
        "test_set_retuning_performed": False,
        "threshold_or_gate_changes_after_test": False,
    }


__all__ = [
    "CONFIRMATORY_INFERENCE_METRICS",
    "claim_gate_report",
    "joint_holm_confirmatory",
    "method_metrics",
    "ood_and_abstention_metrics",
    "paired_confirmatory_intervals",
]
