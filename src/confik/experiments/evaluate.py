from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Protocol

import numpy as np

from ..data.datasets import QueryDataset
from ..types import IKQuery, IKResult, Pose


class IKMethod(Protocol):
    def solve(self, query: IKQuery) -> IKResult: ...


def method_kinematics(method: IKMethod) -> object | None:
    kinematics = getattr(method, "kinematics", None)
    if kinematics is None:
        solver = getattr(method, "solver", None)
        kinematics = getattr(solver, "kinematics", None)
    return kinematics


def method_kinematics_difference(method: IKMethod, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    kinematics = method_kinematics(method)
    if kinematics is None:
        return np.asarray(q) - np.asarray(reference)
    return kinematics.difference(q, reference)  # type: ignore[no-any-return,attr-defined]


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def evaluate_methods(
    methods: Mapping[str, IKMethod],
    dataset: QueryDataset,
    *,
    dt: float = 0.02,
    output_jsonl: str | Path | None = None,
    warmup_iterations: int = 0,
    timing_repeats: int = 1,
    method_order_seed: int | None = None,
    synchronize_cuda: bool = False,
) -> list[dict[str, object]]:
    if timing_repeats <= 0 or warmup_iterations < 0:
        raise ValueError("timing_repeats must be positive and warmup_iterations non-negative")

    def make_query(index: int, previous_q: np.ndarray | None = None) -> IKQuery:
        return IKQuery(
            Pose(dataset.target_position[index], dataset.target_rotation[index]),
            dataset.previous_q[index] if previous_q is None else previous_q,
            dt=dt,
        )

    def synchronize() -> None:
        if not synchronize_cuda:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:  # pragma: no cover
            return

    for _ in range(warmup_iterations):
        query = make_query(0)
        for method in methods.values():
            method.solve(query)

    records: list[dict[str, object]] = []
    closed_loop_states: dict[tuple[str, int], np.ndarray] = {}
    output_handle = Path(output_jsonl).open("w", encoding="utf-8") if output_jsonl else None
    try:
        for query_index in range(len(dataset)):
            category = str(dataset.category[query_index])
            trajectory_id = int(dataset.trajectory_id[query_index])
            is_trajectory = category.startswith("trajectory_")
            method_items = list(methods.items())
            if method_order_seed is not None:
                order_rng = np.random.default_rng(method_order_seed + query_index)
                order_rng.shuffle(method_items)
            for order_index, (method_name, method) in enumerate(method_items):
                state_key = (method_name, trajectory_id)
                previous_from_method = is_trajectory and state_key in closed_loop_states
                previous = closed_loop_states.get(state_key) if is_trajectory else None
                query = make_query(query_index, previous)
                repeated_results: list[IKResult] = []
                repeated_latencies: list[float] = []
                for _ in range(timing_repeats):
                    synchronize()
                    started = perf_counter()
                    repeated_results.append(method.solve(query))
                    synchronize()
                    repeated_latencies.append(perf_counter() - started)
                result = repeated_results[-1]
                elapsed = float(np.median(repeated_latencies))
                deterministic_acceptance = len({item.accepted for item in repeated_results}) == 1
                traces = result.traces
                best_trace = min(
                    traces,
                    key=lambda trace: trace.position_error + 0.35 * trace.orientation_error,
                    default=None,
                )
                if is_trajectory:
                    if result.accepted and result.q is not None:
                        closed_loop_states[state_key] = result.q.copy()
                    elif state_key not in closed_loop_states:
                        closed_loop_states[state_key] = query.previous_q.copy()
                command_q = result.q if result.accepted else None
                diagnostic_q = result.q if result.q is not None else (best_trace.q if best_trace else None)
                joint_step = (
                    float(np.max(np.abs(method_kinematics_difference(method, command_q, query.previous_q))))
                    if command_q is not None
                    else float("nan")
                )
                kinematics = method_kinematics(method)
                if command_q is not None and kinematics is not None:
                    joint_delta = np.abs(method_kinematics_difference(method, command_q, query.previous_q))
                    spike = bool(np.any(joint_delta > kinematics.limits.velocity * dt + 1e-4))  # type: ignore[attr-defined]
                else:
                    spike = False
                if diagnostic_q is not None and kinematics is not None:
                    diagnostic_delta = np.abs(
                        method_kinematics_difference(method, diagnostic_q, query.previous_q)
                    )
                    diagnostic_joint_step = float(np.max(diagnostic_delta))
                    diagnostic_velocity_violation = bool(
                        np.any(diagnostic_delta > kinematics.limits.velocity * dt + 1e-4)  # type: ignore[attr-defined]
                    )
                else:
                    diagnostic_joint_step = float("nan")
                    diagnostic_velocity_violation = False
                record: dict[str, object] = {
                    "method": method_name,
                    "method_order_index": order_index,
                    "query_index": query_index,
                    "category": category,
                    "trajectory_id": trajectory_id,
                    "time_index": int(dataset.time_index[query_index]),
                    "closed_loop": is_trajectory,
                    "closed_loop_previous_from_method": previous_from_method,
                    "expected_reachable": bool(dataset.expected_reachable[query_index]),
                    "continuity_feasible": bool(dataset.continuity_feasible[query_index]),
                    "accepted": result.accepted,
                    "solver_converged": any(trace.converged for trace in traces),
                    "fallback_used": result.fallback_used,
                    "risk_level": result.policy.level.value,
                    "risk_score": result.risk.score,
                    "p_easy": result.risk.probability("easy"),
                    "p_medium": result.risk.probability("medium"),
                    "p_hard": result.risk.probability("hard"),
                    "p_fail": result.risk.probability("fail"),
                    "latency_seconds": elapsed,
                    "timing_repeats": timing_repeats,
                    "deterministic_acceptance": deterministic_acceptance,
                    "iterations": int(sum(trace.iterations for trace in traces)),
                    "function_evaluations": int(sum(trace.function_evaluations for trace in traces)),
                    "position_error": best_trace.position_error if best_trace else float("inf"),
                    "orientation_error": best_trace.orientation_error if best_trace else float("inf"),
                    "joint_step_max": joint_step,
                    "trajectory_spike": spike,
                    "candidate_joint_step_max": diagnostic_joint_step,
                    "candidate_velocity_violation": diagnostic_velocity_violation,
                    "verification_reasons": list(result.verification.reasons) if result.verification else [],
                    "reject_reason": result.reject_reason,
                    "entry_action": result.metadata.get("entry_action", result.policy.level.value),
                    "executed_stages": result.metadata.get("executed_stages", []),
                    "candidate_count": result.metadata.get("candidate_count", 0),
                }
                records.append(record)
                if output_handle:
                    output_handle.write(json.dumps(record, allow_nan=True) + "\n")
    finally:
        if output_handle:
            output_handle.close()
    return records


def summarize_records(records: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), str(record["category"]))].append(record)
        grouped[(str(record["method"]), "all")].append(record)
    summary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, category), rows in grouped.items():
        latency = [float(row["latency_seconds"]) for row in rows]
        evaluations = [float(row["function_evaluations"]) for row in rows]
        finite_joint_steps = [
            float(row["joint_step_max"])
            for row in rows
            if np.isfinite(float(row["joint_step_max"]))
        ]
        accepted = np.array([bool(row["accepted"]) for row in rows], dtype=np.float64)
        reachable = np.array([bool(row["expected_reachable"]) for row in rows])
        accepted_errors = [float(row["position_error"]) for row in rows if bool(row["accepted"])]
        solver_converged = [row for row in rows if bool(row["solver_converged"])]
        summary[method][category] = {
            "count": float(len(rows)),
            "acceptance_rate": float(np.mean(accepted)),
            "reachable_acceptance_rate": float(np.mean(accepted[reachable])) if np.any(reachable) else float("nan"),
            "unreachable_rejection_rate": float(np.mean(1.0 - accepted[~reachable])) if np.any(~reachable) else float("nan"),
            "mean_latency_ms": 1000.0 * float(np.mean(latency)),
            "p95_latency_ms": 1000.0 * _percentile(latency, 95),
            "p99_latency_ms": 1000.0 * _percentile(latency, 99),
            "mean_function_evaluations": float(np.mean(evaluations)),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in rows])),
            "query_spike_rate": float(np.mean([bool(row["trajectory_spike"]) for row in rows])),
            "trajectory_spike_rate": _trajectory_spike_rate(rows),
            "max_joint_step": max(finite_joint_steps) if finite_joint_steps else float("nan"),
            "candidate_velocity_violation_rate": float(
                np.mean([bool(row.get("candidate_velocity_violation", False)) for row in rows])
            ),
            "verification_interception_count": float(
                sum(not bool(row["accepted"]) for row in solver_converged)
            ),
            "invalid_if_convergence_only_rate": float(
                np.mean([not bool(row["accepted"]) for row in solver_converged])
            ) if solver_converged else 0.0,
            "trajectory_completion_rate": _trajectory_completion_rate(rows),
            "mean_accepted_position_error_mm": 1000.0 * float(np.mean(accepted_errors))
            if accepted_errors
            else float("nan"),
        }
    return dict(summary)


def _trajectory_completion_rate(rows: list[dict[str, object]]) -> float:
    trajectory_rows = [row for row in rows if bool(row.get("closed_loop", False))]
    if not trajectory_rows:
        return float("nan")
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[int(row["trajectory_id"])].append(bool(row["accepted"]))
    return float(np.mean([all(values) for values in grouped.values()]))


def _trajectory_spike_rate(rows: list[dict[str, object]]) -> float:
    trajectory_rows = [row for row in rows if bool(row.get("closed_loop", False))]
    if not trajectory_rows:
        return float("nan")
    return float(np.mean([bool(row["trajectory_spike"]) for row in trajectory_rows]))


def write_summary(summary: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
