from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..experiments.metrics import binary_calibration_error, expected_calibration_error
from ..experiments.policy_selection import action_policy_metrics
from ..experiments.statistics import paired_cluster_bootstrap_difference
from .benchmark import distribution


PRIMARY_BACKEND = "torchscript_exact"


def primary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["backend"] == PRIMARY_BACKEND
        or row["backend"] == "production_comparator"
    ]


def _rows(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    preferred = [
        row for row in records
        if row["method"] == method and row["backend"] == PRIMARY_BACKEND
    ]
    if preferred:
        return preferred
    return [
        row for row in records
        if row["method"] == method and row["backend"] == "production_comparator"
    ]


def _subsets(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    points = [row for row in rows if not row["closed_loop"]]
    feasible = [
        row for row in points
        if row["expected_reachable"] and row["continuity_feasible"]
    ]
    rejectable = [
        row for row in points
        if not (row["expected_reachable"] and row["continuity_feasible"])
    ]
    trajectories = [row for row in rows if row["closed_loop"]]
    return feasible, rejectable, trajectories


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else float("nan")


def _p95_latency(rows: list[dict[str, Any]]) -> float:
    return float(np.percentile([float(row["latency_ns"]) for row in rows], 95)) / 1e9


def _trajectory_completion(rows: list[dict[str, Any]]) -> float:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[int(row["trajectory_id"])].append(bool(row["accepted"]))
    return float(np.mean([all(values) for values in grouped.values()])) if grouped else float("nan")


def gate_method_metrics(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = _rows(records, method)
    feasible, rejectable, trajectories = _subsets(rows)
    hard = [row for row in feasible if row["category"] == "hard_valid"]
    return {
        "query_count": len(rows),
        "point_count": len(feasible) + len(rejectable),
        "point_feasible_count": len(feasible),
        "point_rejectable_count": len(rejectable),
        "point_feasible_success": _mean(feasible, "accepted"),
        "point_rejectable_rejection": 1.0 - _mean(rejectable, "accepted"),
        "hard_valid_success": _mean(hard, "accepted"),
        "point_feasible_mean_function_evaluations": _mean(feasible, "function_evaluations"),
        "point_rejectable_mean_function_evaluations": _mean(rejectable, "function_evaluations"),
        "point_feasible_p95_latency_seconds": _p95_latency(feasible),
        "point_rejectable_p95_latency_seconds": _p95_latency(rejectable),
        "trajectory_count": len({int(row["trajectory_id"]) for row in trajectories}),
        "trajectory_completion_rate": _trajectory_completion(trajectories),
        "trajectory_frame_acceptance": _mean(trajectories, "accepted"),
        "trajectory_command_spike_rate": _mean(trajectories, "trajectory_command_spike"),
    }


def risk_metrics_from_formal_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        (str(row["split"]), int(row["query_index"])): row
        for row in _rows(records, "fixed_robust_cascade")
        if not row["closed_loop"]
    }
    proposed = {
        (str(row["split"]), int(row["query_index"])): row
        for row in _rows(records, "proposed_v2")
        if not row["closed_loop"]
    }
    if set(baseline) != set(proposed):
        raise RuntimeError("formal risk metric pairing is incomplete")
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    name_to_action = {"easy": 0, "medium": 1, "hard": 2, "reject": 3}
    for key in sorted(baseline):
        fixed = baseline[key]
        learned = proposed[key]
        if fixed["query_sha256"] != learned["query_sha256"]:
            raise RuntimeError("formal risk metric query hashes differ")
        if not (fixed["expected_reachable"] and fixed["continuity_feasible"]):
            label = 3
        elif fixed["accepted"]:
            stages = list(fixed["executed_stages"])
            label = name_to_action[str(stages[-1])] if stages else 3
        else:
            label = 3
        labels.append(label)
        predictions.append(name_to_action[str(learned["entry_action"])])
        probabilities.append([float(value) for value in learned["risk_probabilities"]])
    truth = np.asarray(labels, dtype=np.int64)
    actions = np.asarray(predictions, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    fail = (truth == 3).astype(np.int64)
    return {
        "label_definition": (
            "ground-truth reject for unreachable/discontinuous points; otherwise the first "
            "verified stage of the identical fixed robust cascade, or reject if all stages fail"
        ),
        "sample_count": int(len(truth)),
        "label_counts": np.bincount(truth, minlength=4).tolist(),
        "fail_auroc": float(roc_auc_score(fail, probs[:, 3])) if np.unique(fail).size == 2 else float("nan"),
        "fail_ece": binary_calibration_error(probs[:, 3], fail),
        "multiclass_ece": expected_calibration_error(probs, truth),
        "policy_test_metrics": action_policy_metrics(truth, actions),
    }


def write_locked_claim_gate(
    records: list[dict[str, Any]],
    gate: dict[str, float],
) -> dict[str, Any]:
    baseline = gate_method_metrics(records, "fixed_robust_cascade")
    proposed = gate_method_metrics(records, "proposed_v2")
    threshold = gate_method_metrics(records, "threshold_guard_cascade")
    risk = risk_metrics_from_formal_records(records)

    def reduction(candidate: float, reference: float) -> float:
        return 1.0 - candidate / max(reference, 1e-12)

    success_gap = proposed["point_feasible_success"] - baseline["point_feasible_success"]
    feasible_fev = reduction(
        proposed["point_feasible_mean_function_evaluations"],
        baseline["point_feasible_mean_function_evaluations"],
    )
    feasible_latency = reduction(
        proposed["point_feasible_p95_latency_seconds"],
        baseline["point_feasible_p95_latency_seconds"],
    )
    reject_fev = reduction(
        proposed["point_rejectable_mean_function_evaluations"],
        baseline["point_rejectable_mean_function_evaluations"],
    )
    reject_latency = reduction(
        proposed["point_rejectable_p95_latency_seconds"],
        baseline["point_rejectable_p95_latency_seconds"],
    )
    trajectory_gap = proposed["trajectory_completion_rate"] - baseline["trajectory_completion_rate"]
    threshold_success_gap = proposed["point_feasible_success"] - threshold["point_feasible_success"]
    threshold_fev = reduction(
        proposed["point_feasible_mean_function_evaluations"],
        threshold["point_feasible_mean_function_evaluations"],
    )
    threshold_latency = reduction(
        proposed["point_feasible_p95_latency_seconds"],
        threshold["point_feasible_p95_latency_seconds"],
    )
    policy = risk["policy_test_metrics"]
    payload: dict[str, Any] = {
        "comparison": "proposed_v2 exact TorchScript vs identical fixed robust cascade",
        "backend": PRIMARY_BACKEND,
        "baseline": baseline,
        "proposed": proposed,
        "threshold_guard": threshold,
        "risk": risk,
        "point_feasible_success_gap": success_gap,
        "point_feasible_evaluation_reduction": feasible_fev,
        "point_feasible_p95_latency_reduction": feasible_latency,
        "point_feasible_p95_latency_ratio": 1.0 - feasible_latency,
        "point_rejectable_evaluation_reduction": reject_fev,
        "point_rejectable_p95_latency_reduction": reject_latency,
        "trajectory_completion_gap": trajectory_gap,
        "threshold_guard_point_success_gap": threshold_success_gap,
        "threshold_guard_point_evaluation_reduction": threshold_fev,
        "threshold_guard_point_p95_latency_reduction": threshold_latency,
        "locked_thresholds": gate,
    }
    payload["success_gate_pass"] = success_gap >= gate["feasible_success_gap_min"]
    payload["rejection_gate_pass"] = proposed["point_rejectable_rejection"] >= gate["rejectable_rejection_min"]
    payload["routing_efficiency_gate_pass"] = (
        feasible_fev >= gate["feasible_fev_reduction_min"]
        and 1.0 - feasible_latency <= gate["feasible_p95_ratio_max"]
    )
    payload["reject_efficiency_gate_pass"] = (
        reject_fev >= gate["rejectable_fev_reduction_min"]
        and reject_latency >= gate["rejectable_latency_reduction_min"]
    )
    payload["risk_gate_pass"] = (
        risk["fail_auroc"] >= gate["fail_auroc_min"]
        and risk["fail_ece"] <= gate["fail_ece_max"]
        and policy["false_reject_rate"] <= gate["false_reject_rate_max"]
        and policy["reject_recall"] >= gate["reject_recall_min"]
        and policy["nonreject_macro_f1"] >= gate["nonreject_macro_f1_min"]
    )
    payload["trajectory_gate_pass"] = (
        trajectory_gap >= gate["trajectory_completion_gap_min"]
        and proposed["trajectory_command_spike_rate"] <= gate["trajectory_command_spike_max"]
    )
    payload["nontriviality_gate_pass"] = (
        threshold_success_gap >= gate["threshold_success_gap_min"]
        and threshold_fev >= gate["threshold_fev_reduction_min"]
        and 1.0 - threshold_latency <= gate["threshold_p95_ratio_max"]
    )
    payload["pilot_gate_pass"] = all(
        payload[key]
        for key in (
            "success_gate_pass",
            "rejection_gate_pass",
            "routing_efficiency_gate_pass",
            "reject_efficiency_gate_pass",
            "risk_gate_pass",
            "trajectory_gate_pass",
            "nontriviality_gate_pass",
        )
    )
    payload["test_set_retuning_performed"] = False
    return payload

def latency_report(
    records: list[dict[str, Any]],
    *,
    tail_thresholds_ms: list[float],
) -> dict[str, Any]:
    profiled = [row for row in records if row["backend"] in {PRIMARY_BACKEND, "eager_reference"}]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profiled:
        feasible = row["expected_reachable"] and row["continuity_feasible"]
        subset = "trajectory" if row["closed_loop"] else ("point_feasible" if feasible else "point_rejectable")
        groups[f"{row['backend']}/{row['method']}/{subset}"].append(row)
    marginal: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        marginal[key] = {
            "core_end_to_end_ms": distribution([row["latency_ns"] / 1e6 for row in rows]),
            "function_evaluations": distribution([float(row["function_evaluations"]) for row in rows]),
            "tail_rates": {
                f"over_{threshold:g}ms": float(np.mean([row["latency_ns"] / 1e6 > threshold for row in rows]))
                for threshold in tail_thresholds_ms
            },
        }
        timing_keys = sorted(set().union(*(row["timings_ns"] for row in rows)))
        marginal[key]["stages_ms"] = {
            timing.removesuffix("_ns"): {
                name: value
                for name, value in distribution(
                    [float(row["timings_ns"].get(timing, 0)) / 1e6 for row in rows]
                ).items()
                if name in {"p95", "p99", "max"}
            }
            for timing in timing_keys
        }

    paired: dict[str, Any] = {}
    for backend in (PRIMARY_BACKEND, "eager_reference"):
        for subset in ("point_feasible", "point_rejectable"):
            selected = [
                row for row in profiled
                if row["backend"] == backend and not row["closed_loop"]
                and ((row["expected_reachable"] and row["continuity_feasible"]) == (subset == "point_feasible"))
            ]
            index: dict[tuple[str, int], dict[str, Any]] = {
                (str(row["method"]), int(row["query_index"])): row for row in selected
            }
            ids = sorted({query for method, query in index if method == "fixed_robust_cascade"})
            differences: list[float] = []
            baseline: list[float] = []
            proposed: list[float] = []
            for query in ids:
                left = index[("fixed_robust_cascade", query)]
                right = index[("proposed_v2", query)]
                if left["query_sha256"] != right["query_sha256"]:
                    raise RuntimeError("paired formal latency query hash mismatch")
                b = left["latency_ns"] / 1e6
                p = right["latency_ns"] / 1e6
                baseline.append(b)
                proposed.append(p)
                differences.append(p - b)
            b_summary = distribution(baseline)
            p_summary = distribution(proposed)
            paired[f"{backend}/{subset}"] = {
                "baseline_ms": b_summary,
                "proposed_ms": p_summary,
                "p95_ratio_proposed_over_baseline": p_summary["p95"] / max(b_summary["p95"], 1e-12),
                "paired_mean_difference_ms": float(np.mean(differences)),
                "paired_median_difference_ms": float(np.median(differences)),
                "paired_p95_difference_ms": float(np.percentile(differences, 95)),
                "paired_difference_ms": distribution(differences),
                "query_hash_mismatch_count": 0,
            }

    eager_reduction: dict[str, Any] = {}
    for method in ("fixed_robust_cascade", "proposed_v2"):
        for subset in ("point_feasible", "point_rejectable"):
            key_exact = f"{PRIMARY_BACKEND}/{method}/{subset}"
            key_eager = f"eager_reference/{method}/{subset}"
            exact = marginal[key_exact]["core_end_to_end_ms"]
            eager = marginal[key_eager]["core_end_to_end_ms"]
            eager_reduction[f"{method}/{subset}"] = {
                "p95_reduction_exact_vs_eager": 1.0 - exact["p95"] / max(eager["p95"], 1e-12),
                "diagnostic_only": True,
                "eligible_for_backend_selection": False,
            }
    return {"marginal": marginal, "paired": paired, "reduction_vs_eager": eager_reduction}


def cluster_intervals(
    records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    exact = [row for row in records if row["backend"] == PRIMARY_BACKEND]
    comparisons = {
        "point_feasible_success": ("accepted", "point_feasible"),
        "point_feasible_function_evaluations": ("function_evaluations", "point_feasible"),
        "point_feasible_latency_seconds": ("latency_seconds", "point_feasible"),
        "point_rejectable_acceptance": ("accepted", "point_rejectable"),
        "point_rejectable_function_evaluations": ("function_evaluations", "point_rejectable"),
        "point_rejectable_latency_seconds": ("latency_seconds", "point_rejectable"),
        "trajectory_frame_acceptance": ("accepted", "trajectory"),
    }
    payload: dict[str, Any] = {}
    for index, (name, (field, subset)) in enumerate(comparisons.items()):
        def include(row: dict[str, Any]) -> bool:
            feasible = row["expected_reachable"] and row["continuity_feasible"]
            if subset == "point_feasible":
                return not row["closed_loop"] and feasible
            if subset == "point_rejectable":
                return not row["closed_loop"] and not feasible
            return bool(row["closed_loop"])

        rows = [row for row in exact if include(row)]
        by_key = {(str(row["method"]), int(row["query_index"])): row for row in rows}
        ids = sorted({query for method, query in by_key if method == "fixed_robust_cascade"})
        baseline = np.asarray([float(by_key[("fixed_robust_cascade", query)][field]) for query in ids])
        proposed = np.asarray([float(by_key[("proposed_v2", query)][field]) for query in ids])
        clusters = np.asarray([int(by_key[("fixed_robust_cascade", query)]["trajectory_id"]) for query in ids])
        payload[name] = paired_cluster_bootstrap_difference(
            baseline,
            proposed,
            clusters,
            samples=samples,
            seed=seed + index,
        )
    return payload
