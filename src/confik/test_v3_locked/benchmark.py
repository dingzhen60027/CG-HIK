from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from ..data.datasets import QueryDataset
from ..experiments.evaluate import method_kinematics, method_kinematics_difference
from ..latency_pilot_v3.benchmark import (
    CORE_STAGE_KEYS,
    ProfiledCascadeRuntime,
    ProfiledOutcome,
    query_digest,
    query_from_dataset,
)
from ..types import IKQuery, IKResult


def _sync_cuda(enabled: bool) -> None:
    if not enabled:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_profiled(outcomes: list[ProfiledOutcome]) -> ProfiledOutcome:
    reference = outcomes[-1]
    semantic = (
        reference.accepted,
        reference.entry_action,
        reference.function_evaluations,
        reference.verification_reasons,
    )
    if any(
        (
            item.accepted,
            item.entry_action,
            item.function_evaluations,
            item.verification_reasons,
        )
        != semantic
        for item in outcomes
    ):
        raise RuntimeError("repeated primary timing changed solver semantics")
    keys = set().union(*(item.timings_ns for item in outcomes))
    timings = {
        key: int(np.median([item.timings_ns.get(key, 0) for item in outcomes]))
        for key in keys
    }
    return replace(reference, timings_ns=timings)


def _profiled_record(
    *,
    robot: str,
    backend: str,
    method: str,
    split: str,
    query_index: int,
    dataset: QueryDataset,
    query: IKQuery,
    outcome: ProfiledOutcome,
    order_index: int,
    kinematics: object,
) -> dict[str, Any]:
    if outcome.accepted and outcome.q is not None:
        joint_delta = np.abs(kinematics.difference(outcome.q, query.previous_q))  # type: ignore[attr-defined]
        joint_step = float(np.max(joint_delta))
        spike = bool(
            np.any(joint_delta > kinematics.limits.velocity * query.dt + 1e-4)  # type: ignore[attr-defined]
        )
    else:
        joint_step = float("nan")
        spike = False
    probabilities = np.asarray(outcome.risk_probabilities, dtype=np.float64)
    latency_ns = int(outcome.timings_ns["total_end_to_end_ns"])
    return {
        "robot": robot,
        "backend": backend,
        "method": method,
        "split": split,
        "query_index": int(query_index),
        "query_sha256": query_digest(query),
        "category": str(dataset.category[query_index]),
        "trajectory_id": int(dataset.trajectory_id[query_index]),
        "time_index": int(dataset.time_index[query_index]),
        "closed_loop": split == "trajectory",
        "method_order_index": int(order_index),
        "expected_reachable": bool(dataset.expected_reachable[query_index]),
        "continuity_feasible": bool(dataset.continuity_feasible[query_index]),
        "accepted": bool(outcome.accepted),
        "solver_converged": bool(outcome.accepted),
        "fallback_used": bool(outcome.fallback_used),
        "risk_level": outcome.entry_action,
        "risk_score": float(outcome.risk_score),
        "p_easy": float(probabilities[0]),
        "p_medium": float(probabilities[1]),
        "p_hard": float(probabilities[2]),
        "p_fail": float(probabilities[3]),
        "risk_probabilities": probabilities.tolist(),
        "latency_ns": latency_ns,
        "latency_seconds": latency_ns / 1e9,
        "iterations": int(outcome.iterations),
        "function_evaluations": int(outcome.function_evaluations),
        "joint_step_max": joint_step,
        "trajectory_spike": spike,
        "trajectory_command_spike": spike,
        "verification_reasons": list(outcome.verification_reasons),
        "reject_reason": outcome.reject_reason,
        "entry_action": outcome.entry_action,
        "executed_stages": list(outcome.executed_stages),
        "candidate_count": int(outcome.candidate_count),
        "command_q": None if outcome.q is None else outcome.q.tolist(),
        "timings_ns": dict(outcome.timings_ns),
    }


def warmup_profiled(
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    iterations: int,
    dt: float,
) -> None:
    if not len(dataset):
        return
    categories = sorted(np.unique(dataset.category).tolist())
    available = {category: np.flatnonzero(dataset.category == category) for category in categories}
    for index in range(iterations):
        category = categories[index % len(categories)]
        source = int(available[category][(index // len(categories)) % len(available[category])])
        query = query_from_dataset(dataset, source, dt=dt)
        for baseline, proposed in runtimes.values():
            pair = (proposed, baseline) if index % 2 else (baseline, proposed)
            for runtime in pair:
                runtime.solve(query)


def benchmark_profiled_points(
    robot: str,
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    repeats: int,
    dt: float,
    order_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = {"baseline": "fixed_robust_cascade", "proposed": "proposed_v2"}
    for query_index in range(len(dataset)):
        query = query_from_dataset(dataset, query_index, dt=dt)
        backend_order = list(runtimes)
        np.random.default_rng(order_seed + query_index).shuffle(backend_order)
        for backend in backend_order:
            baseline, proposed = runtimes[backend]
            collected: dict[str, list[ProfiledOutcome]] = {"baseline": [], "proposed": []}
            first_order: dict[str, int] = {}
            for repeat in range(repeats):
                pair = (
                    (("proposed", proposed), ("baseline", baseline))
                    if (query_index + repeat) % 2
                    else (("baseline", baseline), ("proposed", proposed))
                )
                if not first_order:
                    first_order = {name: index for index, (name, _) in enumerate(pair)}
                for name, runtime in pair:
                    started = perf_counter_ns()
                    outcome = runtime.solve(query)
                    elapsed = perf_counter_ns() - started
                    outcome.timings_ns["total_end_to_end_ns"] = elapsed
                    outcome.timings_ns["unattributed_framework_ns"] = elapsed - sum(
                        outcome.timings_ns.get(key, 0) for key in CORE_STAGE_KEYS
                    )
                    collected[name].append(outcome)
            for name, runtime in (("baseline", baseline), ("proposed", proposed)):
                records.append(
                    _profiled_record(
                        robot=robot,
                        backend=backend,
                        method=names[name],
                        split="point",
                        query_index=query_index,
                        dataset=dataset,
                        query=query,
                        outcome=_median_profiled(collected[name]),
                        order_index=first_order[name],
                        kinematics=runtime.kinematics,
                    )
                )
    return records


def benchmark_profiled_trajectories(
    robot: str,
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    dt: float,
    order_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = {"baseline": "fixed_robust_cascade", "proposed": "proposed_v2"}
    states: dict[tuple[str, str, int], np.ndarray] = {}
    for query_index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[query_index])
        backend_order = list(runtimes)
        np.random.default_rng(order_seed + 100_000 + query_index).shuffle(backend_order)
        for backend in backend_order:
            baseline, proposed = runtimes[backend]
            pair = (
                (("proposed", proposed), ("baseline", baseline))
                if query_index % 2
                else (("baseline", baseline), ("proposed", proposed))
            )
            for order_index, (name, runtime) in enumerate(pair):
                key = (backend, name, trajectory_id)
                previous = states.get(key, dataset.previous_q[query_index])
                query = query_from_dataset(dataset, query_index, previous_q=previous, dt=dt)
                started = perf_counter_ns()
                outcome = runtime.solve(query)
                elapsed = perf_counter_ns() - started
                outcome.timings_ns["total_end_to_end_ns"] = elapsed
                outcome.timings_ns["unattributed_framework_ns"] = elapsed - sum(
                    outcome.timings_ns.get(key, 0) for key in CORE_STAGE_KEYS
                )
                if outcome.accepted and outcome.q is not None:
                    states[key] = outcome.q.copy()
                else:
                    states[key] = np.asarray(previous, dtype=np.float64).copy()
                records.append(
                    _profiled_record(
                        robot=robot,
                        backend=backend,
                        method=names[name],
                        split="trajectory",
                        query_index=query_index,
                        dataset=dataset,
                        query=query,
                        outcome=outcome,
                        order_index=order_index,
                        kinematics=runtime.kinematics,
                    )
                )
    return records


def benchmark_profiled_single(
    robot: str,
    *,
    backend: str,
    method: str,
    runtime: ProfiledCascadeRuntime,
    dataset: QueryDataset,
    split: str,
    repeats: int,
    dt: float,
) -> list[dict[str, Any]]:
    """Benchmark one profiled comparator with the same core timing boundary."""

    records: list[dict[str, Any]] = []
    closed_loop = split == "trajectory"
    states: dict[int, np.ndarray] = {}
    for query_index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[query_index])
        previous = states.get(trajectory_id) if closed_loop else None
        query = query_from_dataset(dataset, query_index, previous_q=previous, dt=dt)
        outcomes: list[ProfiledOutcome] = []
        for _ in range(1 if closed_loop else repeats):
            started = perf_counter_ns()
            outcome = runtime.solve(query)
            elapsed = perf_counter_ns() - started
            outcome.timings_ns["total_end_to_end_ns"] = elapsed
            outcome.timings_ns["unattributed_framework_ns"] = elapsed - sum(
                outcome.timings_ns.get(key, 0) for key in CORE_STAGE_KEYS
            )
            outcomes.append(outcome)
        selected = _median_profiled(outcomes)
        if closed_loop:
            if selected.accepted and selected.q is not None:
                states[trajectory_id] = selected.q.copy()
            else:
                states[trajectory_id] = np.asarray(query.previous_q, dtype=np.float64).copy()
        records.append(
            _profiled_record(
                robot=robot,
                backend=backend,
                method=method,
                split=split,
                query_index=query_index,
                dataset=dataset,
                query=query,
                outcome=selected,
                order_index=0,
                kinematics=runtime.kinematics,
            )
        )
    return records


def _generic_record(
    *,
    robot: str,
    method_name: str,
    method: object,
    split: str,
    query_index: int,
    dataset: QueryDataset,
    query: IKQuery,
    result: IKResult,
    latency_ns: int,
    order_index: int,
) -> dict[str, Any]:
    kinematics = method_kinematics(method)  # type: ignore[arg-type]
    traces = result.traces
    if result.accepted and result.q is not None:
        delta = np.abs(method_kinematics_difference(method, result.q, query.previous_q))  # type: ignore[arg-type]
        joint_step = float(np.max(delta))
        spike = bool(
            kinematics is not None
            and np.any(delta > kinematics.limits.velocity * query.dt + 1e-4)  # type: ignore[attr-defined]
        )
    else:
        joint_step = float("nan")
        spike = False
    probabilities = np.asarray(result.risk.probabilities, dtype=np.float64)
    return {
        "robot": robot,
        "backend": "production_comparator",
        "method": method_name,
        "split": split,
        "query_index": int(query_index),
        "query_sha256": query_digest(query),
        "category": str(dataset.category[query_index]),
        "trajectory_id": int(dataset.trajectory_id[query_index]),
        "time_index": int(dataset.time_index[query_index]),
        "closed_loop": split == "trajectory",
        "method_order_index": int(order_index),
        "expected_reachable": bool(dataset.expected_reachable[query_index]),
        "continuity_feasible": bool(dataset.continuity_feasible[query_index]),
        "accepted": bool(result.accepted),
        "solver_converged": any(trace.converged for trace in traces),
        "fallback_used": bool(result.fallback_used),
        "risk_level": result.policy.level.value,
        "risk_score": float(result.risk.score),
        "p_easy": float(probabilities[0]),
        "p_medium": float(probabilities[1]),
        "p_hard": float(probabilities[2]),
        "p_fail": float(probabilities[3]),
        "risk_probabilities": probabilities.tolist(),
        "latency_ns": int(latency_ns),
        "latency_seconds": latency_ns / 1e9,
        "iterations": int(sum(trace.iterations for trace in traces)),
        "function_evaluations": int(sum(trace.function_evaluations for trace in traces)),
        "joint_step_max": joint_step,
        "trajectory_spike": spike,
        "trajectory_command_spike": spike,
        "verification_reasons": (
            list(result.verification.reasons) if result.verification is not None else []
        ),
        "reject_reason": result.reject_reason,
        "entry_action": str(result.metadata.get("entry_action", result.policy.level.value)),
        "executed_stages": list(result.metadata.get("executed_stages", [])),
        "candidate_count": int(result.metadata.get("candidate_count", 0)),
        "command_q": None if result.q is None else np.asarray(result.q).tolist(),
        "timings_ns": {"total_end_to_end_ns": int(latency_ns)},
    }


def warmup_methods(
    methods: Mapping[str, object],
    dataset: QueryDataset,
    *,
    iterations: int,
    dt: float,
) -> None:
    if not len(dataset):
        return
    for index in range(iterations):
        query = query_from_dataset(dataset, index % len(dataset), dt=dt)
        items = list(methods.values())
        if index % 2:
            items.reverse()
        for method in items:
            method.solve(query)  # type: ignore[attr-defined]


def benchmark_methods(
    robot: str,
    methods: Mapping[str, object],
    dataset: QueryDataset,
    *,
    split: str,
    point_repeats: int,
    dt: float,
    order_seed: int,
    synchronize_cuda: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    states: dict[tuple[str, int], np.ndarray] = {}
    closed_loop = split == "trajectory"
    for query_index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[query_index])
        items = list(methods.items())
        np.random.default_rng(order_seed + query_index).shuffle(items)
        for order_index, (method_name, method) in enumerate(items):
            state_key = (method_name, trajectory_id)
            previous = states.get(state_key) if closed_loop else None
            query = query_from_dataset(dataset, query_index, previous_q=previous, dt=dt)
            repeats = 1 if closed_loop else point_repeats
            results: list[IKResult] = []
            elapsed: list[int] = []
            for _ in range(repeats):
                _sync_cuda(synchronize_cuda)
                started = perf_counter_ns()
                result = method.solve(query)  # type: ignore[attr-defined]
                _sync_cuda(synchronize_cuda)
                elapsed.append(perf_counter_ns() - started)
                results.append(result)
            semantic = (
                results[-1].accepted,
                int(sum(trace.function_evaluations for trace in results[-1].traces)),
                results[-1].reject_reason,
            )
            if any(
                (
                    result.accepted,
                    int(sum(trace.function_evaluations for trace in result.traces)),
                    result.reject_reason,
                )
                != semantic
                for result in results
            ):
                raise RuntimeError(f"repeated comparator timing changed {method_name} semantics")
            result = results[-1]
            if closed_loop:
                if result.accepted and result.q is not None:
                    states[state_key] = np.asarray(result.q, dtype=np.float64).copy()
                else:
                    states[state_key] = np.asarray(query.previous_q, dtype=np.float64).copy()
            records.append(
                _generic_record(
                    robot=robot,
                    method_name=method_name,
                    method=method,
                    split=split,
                    query_index=query_index,
                    dataset=dataset,
                    query=query,
                    result=result,
                    latency_ns=int(np.median(elapsed)),
                    order_index=order_index,
                )
            )
    return records


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: float("nan") for key in ("p50", "p90", "p95", "p99", "p99_9", "mean", "max")} | {"count": 0}
    return {
        "count": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in records}):
        method_rows = [row for row in records if row["method"] == method]
        subsets = {
            "point_feasible": [
                row for row in method_rows
                if not row["closed_loop"] and row["expected_reachable"] and row["continuity_feasible"]
            ],
            "point_rejectable": [
                row for row in method_rows
                if not row["closed_loop"] and not (row["expected_reachable"] and row["continuity_feasible"])
            ],
            "trajectory": [row for row in method_rows if row["closed_loop"]],
        }
        payload: dict[str, Any] = {}
        for subset, rows in subsets.items():
            grouped: dict[int, list[bool]] = defaultdict(list)
            route_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
            for row in rows:
                if row["closed_loop"]:
                    grouped[int(row["trajectory_id"])].append(bool(row["accepted"]))
                    route_groups[int(row["trajectory_id"])].append(
                        (int(row["time_index"]), str(row["entry_action"]))
                    )
            route_switches = 0
            route_transitions = 0
            for values in route_groups.values():
                actions = [action for _, action in sorted(values)]
                route_switches += sum(left != right for left, right in zip(actions[:-1], actions[1:]))
                route_transitions += max(len(actions) - 1, 0)
            payload[subset] = {
                "count": len(rows),
                "acceptance_rate": float(np.mean([row["accepted"] for row in rows])) if rows else float("nan"),
                "rejection_rate": float(np.mean([not row["accepted"] for row in rows])) if rows else float("nan"),
                "mean_function_evaluations": float(np.mean([row["function_evaluations"] for row in rows])) if rows else float("nan"),
                "function_evaluations": distribution([float(row["function_evaluations"]) for row in rows]),
                "latency_ms": distribution([float(row["latency_ns"]) / 1e6 for row in rows]),
                "deadline_miss_rate_20ms": float(np.mean([row["latency_ns"] > 20_000_000 for row in rows])) if rows else float("nan"),
                "trajectory_completion": float(np.mean([all(values) for values in grouped.values()])) if grouped else float("nan"),
                "trajectory_command_spike": float(np.mean([row["trajectory_command_spike"] for row in rows])) if rows else float("nan"),
                "route_switch_count": route_switches if route_groups else None,
                "route_switch_rate": route_switches / route_transitions if route_transitions else None,
            }
        payload["entry_action_counts"] = dict(Counter(str(row["entry_action"]) for row in method_rows))
        result[method] = payload
    return result
