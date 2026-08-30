"""Same-query, interleaved timing primitives for the locked v4 test.

The formal runner keeps every query and every method result in memory until a
role has finished.  In particular, this module performs no file I/O.  Point
queries retain all raw timing repeats; trajectory queries are executed once in
closed loop because a repeated command would change the controller state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from ..data.datasets import QueryDataset
from ..experiments.evaluate import method_kinematics, method_kinematics_difference
from ..latency_pilot_v3.benchmark import (
    CORE_STAGE_KEYS,
    ProfiledOutcome,
    query_digest,
    query_from_dataset,
)
from ..release_v4_locked.artifacts import decision_record
from ..types import IKQuery, IKResult
from .data import ROLE_DOMAIN, TEST_V4_ROLES, dataset_query_hashes


PRIMARY_METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "threshold_guard_cascade",
    "learned_1x25",
    "dls_previous_1x50",
    "trf_previous",
    "proposed_v4",
)
SENSITIVITY_METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "proposed_v4",
)


def _sync_cuda(enabled: bool) -> None:
    if not enabled:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class _NormalizedOutcome:
    q: np.ndarray | None
    accepted: bool
    entry_action: str
    executed_stages: tuple[str, ...]
    function_evaluations: int
    iterations: int
    fallback_used: bool
    verification_reasons: tuple[str, ...]
    reject_reason: str
    candidate_count: int
    timings_ns: dict[str, int]
    raw_decision: dict[str, Any] | None


def _raw_v4_decision(method: object) -> dict[str, Any] | None:
    decision = getattr(method, "last_decision", None)
    if decision is None:
        return None
    # These are the independently calibrated V4Decision values.  The legacy
    # CalibratedRisk field normalizes its four inputs and must never be used as
    # the v4 formal probability record.
    return decision_record(decision)


def _normalize(method: object, result: object) -> _NormalizedOutcome:
    raw_decision = _raw_v4_decision(method)
    if isinstance(result, ProfiledOutcome):
        return _NormalizedOutcome(
            q=None if result.q is None else np.asarray(result.q, dtype=np.float64),
            accepted=bool(result.accepted),
            entry_action=str(result.entry_action),
            executed_stages=tuple(str(value) for value in result.executed_stages),
            function_evaluations=int(result.function_evaluations),
            iterations=int(result.iterations),
            fallback_used=bool(result.fallback_used),
            verification_reasons=tuple(str(value) for value in result.verification_reasons),
            reject_reason=str(result.reject_reason),
            candidate_count=int(result.candidate_count),
            timings_ns={str(key): int(value) for key, value in result.timings_ns.items()},
            raw_decision=raw_decision,
        )
    if not isinstance(result, IKResult):
        raise TypeError(f"unsupported IK result type: {type(result)!r}")
    traces = list(result.traces)
    metadata = dict(result.metadata)
    return _NormalizedOutcome(
        q=None if result.q is None else np.asarray(result.q, dtype=np.float64),
        accepted=bool(result.accepted),
        entry_action=str(metadata.get("entry_action", result.policy.level.value)),
        executed_stages=tuple(str(value) for value in metadata.get("executed_stages", ())),
        function_evaluations=int(sum(trace.function_evaluations for trace in traces)),
        iterations=int(sum(trace.iterations for trace in traces)),
        fallback_used=bool(result.fallback_used),
        verification_reasons=(
            tuple(str(value) for value in result.verification.reasons)
            if result.verification is not None
            else ()
        ),
        reject_reason=str(result.reject_reason),
        candidate_count=int(metadata.get("candidate_count", 0)),
        timings_ns={},
        raw_decision=raw_decision,
    )


def _solve_once(
    method: object,
    query: IKQuery,
    *,
    synchronize_cuda: bool,
) -> tuple[_NormalizedOutcome, int]:
    _sync_cuda(synchronize_cuda)
    started = perf_counter_ns()
    result = method.solve(query)  # type: ignore[attr-defined]
    _sync_cuda(synchronize_cuda)
    elapsed = perf_counter_ns() - started
    outcome = _normalize(method, result)
    timings = dict(outcome.timings_ns)
    timings["total_end_to_end_ns"] = int(elapsed)
    if timings:
        timings["unattributed_framework_ns"] = int(elapsed) - sum(
            int(timings.get(key, 0)) for key in CORE_STAGE_KEYS
        )
    return _NormalizedOutcome(**{**outcome.__dict__, "timings_ns": timings}), int(elapsed)


def _semantic_signature(outcome: _NormalizedOutcome) -> tuple[Any, ...]:
    decision = outcome.raw_decision
    return (
        outcome.accepted,
        outcome.entry_action,
        outcome.executed_stages,
        outcome.function_evaluations,
        outcome.iterations,
        outcome.fallback_used,
        outcome.verification_reasons,
        outcome.reject_reason,
        None if decision is None else decision["decision_action"],
        None if decision is None else decision["decision_reason"],
    )


def _select_median_outcome(
    outcomes: list[_NormalizedOutcome], latencies_ns: list[int]
) -> _NormalizedOutcome:
    if not outcomes or len(outcomes) != len(latencies_ns):
        raise ValueError("timed outcomes and latency repeats must be non-empty and paired")
    reference = outcomes[-1]
    if any(_semantic_signature(item) != _semantic_signature(reference) for item in outcomes):
        raise RuntimeError("timing repeats changed method semantics")
    order = np.argsort(np.asarray(latencies_ns, dtype=np.int64), kind="stable")
    return outcomes[int(order[len(order) // 2])]


def _record(
    *,
    robot: str,
    training_seed: int,
    method_name: str,
    role: str,
    query_index: int,
    dataset: QueryDataset,
    source_query_sha256: str,
    query: IKQuery,
    method: object,
    outcome: _NormalizedOutcome,
    repeated_outcomes: list[_NormalizedOutcome],
    raw_latency_ns: list[int],
    first_order_index: int,
    order_indices_by_repeat: list[int],
) -> dict[str, Any]:
    if role not in ROLE_DOMAIN:
        raise ValueError(f"unknown explicit test role: {role}")
    kinematics = getattr(method, "kinematics", None) or method_kinematics(method)  # type: ignore[arg-type]
    if outcome.accepted and outcome.q is not None and kinematics is not None:
        difference = np.abs(
            method_kinematics_difference(method, outcome.q, query.previous_q)  # type: ignore[arg-type]
        )
        joint_step = float(np.max(difference))
        spike = bool(
            np.any(difference > kinematics.limits.velocity * query.dt + 1e-4)  # type: ignore[attr-defined]
        )
    else:
        joint_step = float("nan")
        spike = False

    raw_decision = outcome.raw_decision
    decision_action = None if raw_decision is None else str(raw_decision["decision_action"])
    decision_reason = None if raw_decision is None else str(raw_decision["decision_reason"])
    contract_violations: list[str] = []
    if outcome.accepted != (outcome.q is not None):
        contract_violations.append("accepted_command_presence_mismatch")
    if decision_action == "reject":
        if outcome.function_evaluations != 0:
            contract_violations.append("command_reject_nonzero_fev")
        if outcome.executed_stages:
            contract_violations.append("command_reject_executed_solver_stage")
        if outcome.accepted or outcome.q is not None:
            contract_violations.append("command_reject_emitted_command")
    if decision_action == "defer" and outcome.executed_stages:
        if outcome.executed_stages[0] != "easy":
            contract_violations.append("defer_did_not_enter_fixed_easy_stage")

    median_latency = int(np.median(np.asarray(raw_latency_ns, dtype=np.int64)))
    return {
        "robot": robot,
        "training_seed": int(training_seed),
        "backend": "torchscript_exact_v4" if method_name == "proposed_v4" else "formal_comparator",
        "method": method_name,
        "role": role,
        "domain": ROLE_DOMAIN[role],
        "is_trajectory": role.endswith("trajectories"),
        "query_index": int(query_index),
        "source_query_sha256": source_query_sha256,
        "executed_query_sha256": query_digest(query),
        "category": str(dataset.category[query_index]),
        "trajectory_id": int(dataset.trajectory_id[query_index]),
        "time_index": int(dataset.time_index[query_index]),
        "method_order_index_first_repeat": int(first_order_index),
        "method_order_indices_by_repeat": [
            int(value) for value in order_indices_by_repeat
        ],
        "expected_reachable": bool(dataset.expected_reachable[query_index]),
        "continuity_feasible": bool(dataset.continuity_feasible[query_index]),
        "verified_success": bool(outcome.accepted),
        "accepted": bool(outcome.accepted),
        "command_q": None if outcome.q is None else outcome.q.tolist(),
        "function_evaluations": int(outcome.function_evaluations),
        "iterations": int(outcome.iterations),
        "fallback_used": bool(outcome.fallback_used),
        "entry_action": str(outcome.entry_action),
        "executed_stages": list(outcome.executed_stages),
        "verification_reasons": list(outcome.verification_reasons),
        "reject_reason": str(outcome.reject_reason),
        "candidate_count": int(outcome.candidate_count),
        "joint_step_max": joint_step,
        "trajectory_command_spike": spike,
        "latency_ns": median_latency,
        "latency_ms": median_latency / 1e6,
        "latency_repeats_ns": [int(value) for value in raw_latency_ns],
        "latency_repeat_count": len(raw_latency_ns),
        "timings_ns": dict(outcome.timings_ns),
        "timing_repeats_ns": [dict(item.timings_ns) for item in repeated_outcomes],
        "decision_action": decision_action,
        "decision_reason": decision_reason,
        "eligible_actions": None if raw_decision is None else list(raw_decision["eligible_actions"]),
        "predicted_success": None if raw_decision is None else list(raw_decision["predicted_success"]),
        "predicted_p50_ms": None if raw_decision is None else list(raw_decision["predicted_p50_ms"]),
        "predicted_p95_ms": None if raw_decision is None else list(raw_decision["predicted_p95_ms"]),
        "fail_all_probability": None if raw_decision is None else float(raw_decision["fail_all_probability"]),
        "ood_score": None if raw_decision is None else float(raw_decision["ood_score"]),
        "is_ood": None if raw_decision is None else bool(raw_decision["is_ood"]),
        "contract_violations": contract_violations,
    }


def warmup_methods(
    methods: Mapping[str, object],
    dataset: QueryDataset,
    *,
    iterations: int,
    dt: float,
    synchronize_cuda: bool = True,
) -> None:
    """Warm every formal method on the same rotating set of queries."""

    if iterations <= 0 or not len(dataset):
        return
    names = list(methods)
    for warmup_index in range(iterations):
        query = query_from_dataset(dataset, warmup_index % len(dataset), dt=dt)
        order = names if warmup_index % 2 == 0 else list(reversed(names))
        for name in order:
            _solve_once(methods[name], query, synchronize_cuda=synchronize_cuda)


def benchmark_role(
    *,
    robot: str,
    training_seed: int,
    role: str,
    methods: Mapping[str, object],
    dataset: QueryDataset,
    repeats_by_method: Mapping[str, int],
    dt: float,
    order_seed: int,
    synchronize_cuda: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark one explicit role with randomized same-query interleaving."""

    if role not in TEST_V4_ROLES:
        raise ValueError(f"unknown test role: {role}")
    if not methods:
        raise ValueError("at least one formal method is required")
    unknown = set(repeats_by_method) - set(methods)
    if unknown:
        raise ValueError(f"repeat configuration contains unknown methods: {sorted(unknown)}")
    trajectory = role.endswith("trajectories")
    repeat_counts = {
        name: (1 if trajectory else int(repeats_by_method.get(name, 1)))
        for name in methods
    }
    if any(value <= 0 for value in repeat_counts.values()):
        raise ValueError("every method needs a positive timing repeat count")

    records: list[dict[str, Any]] = []
    source_hashes = dataset_query_hashes(dataset, dt=dt).astype(str)
    states: dict[tuple[str, int], np.ndarray] = {}
    method_names = list(methods)
    for query_index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[query_index])
        outcomes: dict[str, list[_NormalizedOutcome]] = defaultdict(list)
        latencies: dict[str, list[int]] = defaultdict(list)
        executed_queries: dict[str, IKQuery] = {}
        first_order: dict[str, int] = {}
        repeat_order: dict[str, list[int]] = defaultdict(list)
        maximum_repeats = max(repeat_counts.values())
        for repeat in range(maximum_repeats):
            active = [name for name in method_names if repeat < repeat_counts[name]]
            rng = np.random.default_rng(
                np.random.SeedSequence([int(order_seed), query_index, repeat])
            )
            rng.shuffle(active)
            for order_index, name in enumerate(active):
                method = methods[name]
                previous = (
                    states.get((name, trajectory_id), dataset.previous_q[query_index])
                    if trajectory
                    else dataset.previous_q[query_index]
                )
                query = query_from_dataset(dataset, query_index, previous_q=previous, dt=dt)
                if name not in first_order:
                    first_order[name] = order_index
                    executed_queries[name] = query
                repeat_order[name].append(order_index)
                outcome, elapsed = _solve_once(
                    method, query, synchronize_cuda=synchronize_cuda
                )
                outcomes[name].append(outcome)
                latencies[name].append(elapsed)
                if trajectory:
                    states[(name, trajectory_id)] = (
                        outcome.q.copy()
                        if outcome.accepted and outcome.q is not None
                        else np.asarray(previous, dtype=np.float64).copy()
                    )

        for name in method_names:
            selected = _select_median_outcome(outcomes[name], latencies[name])
            record = _record(
                robot=robot,
                training_seed=training_seed,
                method_name=name,
                role=role,
                query_index=query_index,
                dataset=dataset,
                source_query_sha256=str(source_hashes[query_index]),
                query=executed_queries[name],
                method=methods[name],
                outcome=selected,
                repeated_outcomes=outcomes[name],
                raw_latency_ns=latencies[name],
                first_order_index=first_order[name],
                order_indices_by_repeat=repeat_order[name],
            )
            if record["contract_violations"]:
                raise RuntimeError(
                    f"formal command contract violation for {robot}/{name}/{role}/"
                    f"{query_index}: {record['contract_violations']}"
                )
            records.append(record)
    return records


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


__all__ = [
    "PRIMARY_METHODS",
    "SENSITIVITY_METHODS",
    "benchmark_role",
    "distribution",
    "warmup_methods",
]
