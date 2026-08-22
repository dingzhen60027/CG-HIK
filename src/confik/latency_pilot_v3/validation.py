from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..data.datasets import QueryDataset, TransitionDataset
from ..experiments.metrics import binary_calibration_error, expected_calibration_error
from ..experiments.policy_selection import action_predictions, action_policy_metrics
from ..runtime.cascade import ActionGateConfig


def subset_query_dataset(dataset: QueryDataset, indices: np.ndarray) -> QueryDataset:
    selected = np.asarray(indices, dtype=np.int64)
    return QueryDataset(
        previous_q=dataset.previous_q[selected],
        target_position=dataset.target_position[selected],
        target_rotation=dataset.target_rotation[selected],
        reference_q=dataset.reference_q[selected],
        category=dataset.category[selected],
        expected_reachable=dataset.expected_reachable[selected],
        continuity_feasible=dataset.continuity_feasible[selected],
        trajectory_id=dataset.trajectory_id[selected],
        time_index=dataset.time_index[selected],
    )


def stratified_point_subset(
    dataset: QueryDataset,
    *,
    per_category: int,
    seed: int,
) -> tuple[QueryDataset, list[int]]:
    if per_category <= 0:
        raise ValueError("point queries per category must be positive")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for category in sorted(np.unique(dataset.category)):
        available = np.flatnonzero(dataset.category == category)
        count = min(per_category, len(available))
        chosen.extend(rng.choice(available, size=count, replace=False).astype(int).tolist())
    selected = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(selected)
    return subset_query_dataset(dataset, selected), selected.tolist()


def trajectory_validation_subset(
    dataset: TransitionDataset,
    *,
    trajectory_count: int,
    seed: int,
) -> tuple[QueryDataset, list[int]]:
    if trajectory_count <= 0:
        raise ValueError("trajectory count must be positive")
    rng = np.random.default_rng(seed)
    ids = np.unique(dataset.trajectory_id)
    selected_ids = np.sort(
        rng.choice(ids, size=min(trajectory_count, len(ids)), replace=False).astype(np.int64)
    )
    mask = np.isin(dataset.trajectory_id, selected_ids)
    indices = np.flatnonzero(mask)
    time_index = np.empty(len(indices), dtype=np.int64)
    counters: dict[int, int] = defaultdict(int)
    for local_index, source_index in enumerate(indices):
        trajectory_id = int(dataset.trajectory_id[source_index])
        time_index[local_index] = counters[trajectory_id]
        counters[trajectory_id] += 1
    count = len(indices)
    result = QueryDataset(
        previous_q=dataset.previous_q[indices],
        target_position=dataset.target_position[indices],
        target_rotation=dataset.target_rotation[indices],
        reference_q=dataset.target_q[indices],
        category=np.full(count, "trajectory_validation", dtype=str),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=dataset.trajectory_id[indices],
        time_index=time_index,
    )
    return result, selected_ids.astype(int).tolist()


def risk_probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    gate_config: ActionGateConfig,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    actions = action_predictions(probs, gate_config)
    fail_targets = (truth == 3).astype(np.int64)
    p_fail = probs[:, 3]
    payload: dict[str, Any] = {
        "sample_count": int(len(truth)),
        "multiclass_ece": expected_calibration_error(probs, truth),
        "fail_ece": binary_calibration_error(p_fail, fail_targets),
        "fail_auroc": float(roc_auc_score(fail_targets, p_fail))
        if np.unique(fail_targets).size == 2
        else float("nan"),
        "argmax_macro_f1": float(
            f1_score(truth, np.argmax(probs, axis=1), labels=[0, 1, 2, 3], average="macro", zero_division=0)
        ),
        "action_policy": action_policy_metrics(truth, actions),
    }
    return payload


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else float("nan")


def method_validation_metrics(
    records: list[dict[str, object]],
    robot: str,
    backend: str,
    method: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in records
        if row["robot"] == robot and row["backend"] == backend and row["method"] == method
    ]
    points = [row for row in rows if not bool(row["closed_loop"])]
    feasible = [
        row
        for row in points
        if bool(row["expected_reachable"]) and bool(row["continuity_feasible"])
    ]
    rejectable = [
        row
        for row in points
        if not (bool(row["expected_reachable"]) and bool(row["continuity_feasible"]))
    ]
    trajectories = [row for row in rows if bool(row["closed_loop"])]
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in trajectories:
        grouped[int(row["trajectory_id"])].append(bool(row["accepted"]))
    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in points}):
        category_rows = [row for row in points if str(row["category"]) == category]
        category_feasible_rows = [
            row
            for row in category_rows
            if bool(row["expected_reachable"]) and bool(row["continuity_feasible"])
        ]
        category_rejectable_rows = [
            row
            for row in category_rows
            if not (bool(row["expected_reachable"]) and bool(row["continuity_feasible"]))
        ]
        by_category[category] = {
            "count": len(category_rows),
            "feasible_count": len(category_feasible_rows),
            "rejectable_count": len(category_rejectable_rows),
            "acceptance_rate": _mean(category_rows, "accepted"),
            "rejection_rate": 1.0 - _mean(category_rows, "accepted"),
            "feasible_success": _mean(category_feasible_rows, "accepted"),
            "rejectable_rejection": 1.0 - _mean(category_rejectable_rows, "accepted"),
            "mean_function_evaluations": _mean(category_rows, "function_evaluations"),
        }
    return {
        "point_feasible_count": len(feasible),
        "point_rejectable_count": len(rejectable),
        "point_feasible_success": _mean(feasible, "accepted"),
        "point_rejectable_rejection": 1.0 - _mean(rejectable, "accepted"),
        "point_feasible_mean_function_evaluations": _mean(feasible, "function_evaluations"),
        "point_rejectable_mean_function_evaluations": _mean(rejectable, "function_evaluations"),
        "trajectory_count": len(grouped),
        "trajectory_completion": float(np.mean([all(values) for values in grouped.values()]))
        if grouped
        else float("nan"),
        "trajectory_mean_function_evaluations": _mean(trajectories, "function_evaluations"),
        "trajectory_command_spike": _mean(trajectories, "trajectory_command_spike"),
        "by_category": by_category,
    }


def record_equivalence(
    records: list[dict[str, object]],
    *,
    robot: str,
    reference_backend: str,
    candidate_backend: str,
) -> dict[str, Any]:
    reference_rows = {
        (str(row["method"]), str(row["split"]), int(row["query_index"])): row
        for row in records
        if row["robot"] == robot and row["backend"] == reference_backend
    }
    candidate_rows = {
        (str(row["method"]), str(row["split"]), int(row["query_index"])): row
        for row in records
        if row["robot"] == robot and row["backend"] == candidate_backend
    }
    keys = sorted(set(reference_rows) & set(candidate_rows))
    accepted_matches = 0
    fev_matches = 0
    point_fev_matches = 0
    point_fev_total = 0
    trajectory_fev_matches = 0
    trajectory_fev_total = 0
    point_route_matches = 0
    point_route_total = 0
    trajectory_route_matches = 0
    trajectory_route_total = 0
    max_q_error = 0.0
    paired_q = 0
    for key in keys:
        reference = reference_rows[key]
        candidate = candidate_rows[key]
        accepted_matches += int(bool(reference["accepted"]) == bool(candidate["accepted"]))
        fev_match = int(
            int(reference["function_evaluations"]) == int(candidate["function_evaluations"])
        )
        fev_matches += fev_match
        if bool(reference["closed_loop"]):
            trajectory_fev_total += 1
            trajectory_fev_matches += fev_match
        else:
            point_fev_total += 1
            point_fev_matches += fev_match
        if key[0] == "proposed":
            route_match = int(reference["entry_action"] == candidate["entry_action"])
            if bool(reference["closed_loop"]):
                trajectory_route_total += 1
                trajectory_route_matches += route_match
            else:
                point_route_total += 1
                point_route_matches += route_match
        reference_q = reference["command_q"]
        candidate_q = candidate["command_q"]
        if reference_q is not None and candidate_q is not None:
            paired_q += 1
            max_q_error = max(
                max_q_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(reference_q, dtype=np.float64)
                            - np.asarray(candidate_q, dtype=np.float64)
                        )
                    )
                ),
            )
    return {
        "paired_record_count": len(keys),
        "accepted_agreement": accepted_matches / len(keys) if keys else float("nan"),
        "function_evaluations_agreement": fev_matches / len(keys) if keys else float("nan"),
        "point_function_evaluations_agreement": point_fev_matches / point_fev_total
        if point_fev_total
        else float("nan"),
        "trajectory_function_evaluations_agreement": trajectory_fev_matches / trajectory_fev_total
        if trajectory_fev_total
        else float("nan"),
        "point_route_action_agreement": point_route_matches / point_route_total
        if point_route_total
        else float("nan"),
        "point_route_action_comparison_count": point_route_total,
        "trajectory_route_action_agreement": trajectory_route_matches / trajectory_route_total
        if trajectory_route_total
        else float("nan"),
        "trajectory_route_action_comparison_count": trajectory_route_total,
        "all_route_action_agreement": (
            (point_route_matches + trajectory_route_matches)
            / (point_route_total + trajectory_route_total)
        )
        if point_route_total + trajectory_route_total
        else float("nan"),
        "accepted_command_max_abs_error_rad": max_q_error,
        "accepted_command_pair_count": paired_q,
    }
