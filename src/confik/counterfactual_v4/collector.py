from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from ..data.datasets import QueryDataset
from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime, ProfiledOutcome
from ..latency_pilot_v3.optimized_inference import (
    SeedEngine,
    cached_risk_features,
)
from ..types import IKQuery


DECISION_ACTIONS = ("easy", "medium", "hard")
AUDIT_ACTIONS = ("fixed_robust",)
# Keep ACTIONS as the decision portfolio for callers that build prediction
# heads.  COLLECTED_ACTIONS additionally contains the fixed-cascade audit arm.
ACTIONS = DECISION_ACTIONS
COLLECTED_ACTIONS = DECISION_ACTIONS + AUDIT_ACTIONS
ALLOWED_SOURCE_ROLES = {
    "risk_train_queries",
    "risk_validation_queries",
    "calibration_queries",
    "policy_validation_queries",
}
STAGE_TIMING_KEYS = (
    "feature_preparation_ns",
    "numpy_torch_conversion_ns",
    "learned_seed_inference_ns",
    "uncertainty_risk_inference_ns",
    "routing_decision_ns",
    "numerical_solver_ns",
    "verification_ns",
    "unattributed_framework_ns",
    "total_end_to_end_ns",
)


def validate_source_role(role: str) -> str:
    if role not in ALLOWED_SOURCE_ROLES:
        raise ValueError(
            f"counterfactual v4 accepts only training/validation roles; got {role!r}"
        )
    if "test" in role.lower():
        raise ValueError("test data are forbidden during counterfactual label collection")
    return role


def select_pilot_indices(
    dataset: QueryDataset,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic random pilot across the complete source split."""

    if count <= 0 or count > len(dataset):
        raise ValueError(f"pilot count must be in [1, {len(dataset)}], got {count}")
    # Preserve the randomized evaluation order. Sorting would re-create any
    # category blocks present in the source file and confound latency with
    # thermal/time drift.
    indices = np.random.default_rng(seed).choice(len(dataset), size=count, replace=False)
    return np.asarray(indices, dtype=np.int64)


def _timed_call(runtime: ProfiledCascadeRuntime, query: IKQuery) -> ProfiledOutcome:
    started = perf_counter_ns()
    outcome = runtime.solve(query)
    elapsed = perf_counter_ns() - started
    timings = dict(outcome.timings_ns)
    timings["total_end_to_end_ns"] = int(elapsed)
    core = sum(
        int(timings.get(key, 0))
        for key in STAGE_TIMING_KEYS
        if key not in {"total_end_to_end_ns", "unattributed_framework_ns"}
    )
    timings["unattributed_framework_ns"] = int(elapsed) - core
    return replace(outcome, timings_ns=timings)


def _semantic_signature(outcome: ProfiledOutcome) -> tuple[Any, ...]:
    command = None if outcome.q is None else np.asarray(outcome.q, dtype=np.float64).tobytes()
    return (
        outcome.accepted,
        outcome.entry_action,
        outcome.executed_stages,
        outcome.function_evaluations,
        outcome.iterations,
        outcome.fallback_used,
        outcome.verification_reasons,
        outcome.reject_reason,
        command,
    )


def _failure_reason(outcome: ProfiledOutcome) -> str:
    if outcome.accepted:
        return ""
    components = [outcome.reject_reason]
    components.extend(outcome.verification_reasons)
    return "|".join(dict.fromkeys(item for item in components if item)) or "unverified_failure"


def collect_query_actions(
    *,
    query: IKQuery,
    query_index: int,
    source_index: int,
    dataset: QueryDataset,
    runtimes: Mapping[str, ProfiledCascadeRuntime],
    seed_engine: SeedEngine,
    repeats: int,
    deadline_ms: float,
    order_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Execute all entry actions and retain every timing repeat.

    Each action keeps the frozen escalation semantics: easy may execute
    easy->medium->hard, medium may execute medium->hard, and hard executes the
    robust hard stage.  The deterministic verifier remains the sole command
    acceptance authority.
    """

    if set(runtimes) != set(COLLECTED_ACTIONS):
        raise ValueError(f"runtimes must contain exactly {COLLECTED_ACTIONS}")
    if repeats < len(COLLECTED_ACTIONS):
        raise ValueError(
            f"at least {len(COLLECTED_ACTIONS)} repeats are required for balanced action order"
        )
    collected: dict[str, list[ProfiledOutcome]] = {
        action: [] for action in COLLECTED_ACTIONS
    }
    rng = np.random.default_rng(order_seed)
    base_order = list(COLLECTED_ACTIONS)
    rng.shuffle(base_order)
    for repeat in range(repeats):
        # A randomized Latin rotation gives every action each order position
        # once in every block of four repeats.
        offset = repeat % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        for action in order:
            collected[action].append(_timed_call(runtimes[action], query))

    records: list[dict[str, Any]] = []
    if _semantic_signature(collected["easy"][-1]) != _semantic_signature(
        collected["fixed_robust"][-1]
    ):
        raise RuntimeError("fixed_robust audit arm is not semantically equal to entry_easy")

    for action in COLLECTED_ACTIONS:
        outcomes = collected[action]
        reference = outcomes[-1]
        signature = _semantic_signature(reference)
        if any(_semantic_signature(outcome) != signature for outcome in outcomes):
            raise RuntimeError(
                f"counterfactual action {action} changed semantics across timing repeats"
            )
        latencies = np.asarray(
            [outcome.timings_ns["total_end_to_end_ns"] for outcome in outcomes],
            dtype=np.int64,
        )
        timing_samples = {
            key: [int(outcome.timings_ns.get(key, 0)) for outcome in outcomes]
            for key in STAGE_TIMING_KEYS
        }
        accepted = bool(reference.accepted)
        deadline_samples = latencies <= int(deadline_ms * 1e6)
        if reference.accepted and reference.q is not None:
            kinematics = runtimes[action].kinematics
            joint_delta = np.abs(
                kinematics.difference(reference.q, query.previous_q)  # type: ignore[attr-defined]
            )
            joint_step_max = float(np.max(joint_delta))
            joint_velocity_max = float(np.max(joint_delta / query.dt))
            velocity_utilization_max = float(
                np.max(joint_delta / (kinematics.limits.velocity * query.dt))  # type: ignore[attr-defined]
            )
        else:
            joint_step_max = None
            joint_velocity_max = None
            velocity_utilization_max = None
        records.append(
            {
                "query_index": int(query_index),
                "source_index": int(source_index),
                "category": str(dataset.category[source_index]),
                "trajectory_id": int(dataset.trajectory_id[source_index]),
                "time_index": int(dataset.time_index[source_index]),
                "expected_reachable": bool(dataset.expected_reachable[source_index]),
                "continuity_feasible": bool(dataset.continuity_feasible[source_index]),
                "entry_action": action,
                "verified_success": accepted,
                "verified_success_before_deadline": bool(
                    accepted and np.percentile(latencies, 95) <= deadline_ms * 1e6
                ),
                "deadline_success_rate": float(np.mean(deadline_samples)) if accepted else 0.0,
                "latency_samples_ns": latencies.tolist(),
                "latency_p50_ns": float(np.percentile(latencies, 50)),
                "latency_p95_ns": float(np.percentile(latencies, 95)),
                "function_evaluations": int(reference.function_evaluations),
                "iterations": int(reference.iterations),
                "fallback_used": bool(reference.fallback_used),
                "executed_stages": list(reference.executed_stages),
                "failure_reason": _failure_reason(reference),
                "verification_reasons": list(reference.verification_reasons),
                "command_q": (
                    None
                    if reference.q is None
                    else np.asarray(reference.q, dtype=np.float64).tolist()
                ),
                "max_joint_step_rad": joint_step_max,
                "max_joint_velocity_rad_s": joint_velocity_max,
                "max_velocity_limit_utilization": velocity_utilization_max,
                # Independent point-query records contain q_(t-1) but not
                # q_(t-2) or q_(t-3).  Acceleration and jerk therefore have no
                # identifiable finite-difference value and are intentionally
                # null rather than being fabricated from a zero-velocity
                # assumption.  The trajectory benchmark computes both from
                # sequential accepted commands.
                "max_joint_acceleration_rad_s2": None,
                "max_joint_jerk_rad_s3": None,
                "dynamic_history_available": False,
                "fixed_robust_matches_easy": True,
                "timing_samples_ns": timing_samples,
            }
        )

    # Feature extraction is deliberately outside every action timing interval.
    # It produces the training input and does not alter the frozen action labels.
    prepared = seed_engine.prepare(query)
    features = cached_risk_features(query, prepared, reuse_best_pose=True)
    return np.asarray(features, dtype=np.float64).copy(), records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"record_count": len(records), "actions": {}}
    for action in COLLECTED_ACTIONS:
        rows = [row for row in records if row["entry_action"] == action]
        p50 = np.asarray([row["latency_p50_ns"] for row in rows], dtype=np.float64) / 1e6
        p95 = np.asarray([row["latency_p95_ns"] for row in rows], dtype=np.float64) / 1e6
        summary["actions"][action] = {
            "query_count": len(rows),
            "verified_success_rate": float(np.mean([row["verified_success"] for row in rows])),
            "verified_success_before_deadline_rate": float(
                np.mean([row["verified_success_before_deadline"] for row in rows])
            ),
            "mean_function_evaluations": float(
                np.mean([row["function_evaluations"] for row in rows])
            ),
            "fallback_rate": float(np.mean([row["fallback_used"] for row in rows])),
            "per_query_latency_p50_ms": {
                "median": float(np.median(p50)),
                "p95": float(np.percentile(p50, 95)),
                "max": float(np.max(p50)),
            },
            "per_query_latency_p95_ms": {
                "median": float(np.median(p95)),
                "p95": float(np.percentile(p95, 95)),
                "max": float(np.max(p95)),
            },
            "failure_reason_counts": dict(
                Counter(str(row["failure_reason"]) for row in rows if row["failure_reason"])
            ),
        }

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["query_index"])].append(row)
    oracle_counts = Counter()
    fail_all = 0
    action_success_disagreement = 0
    for rows in grouped.values():
        decision_rows = [
            row for row in rows if str(row["entry_action"]) in DECISION_ACTIONS
        ]
        accepted = [
            row for row in decision_rows if row["verified_success_before_deadline"]
        ]
        if accepted:
            selected = min(accepted, key=lambda row: row["latency_p95_ns"])
            oracle_counts[str(selected["entry_action"])] += 1
        else:
            oracle_counts["fail_all"] += 1
            fail_all += 1
        if len({bool(row["verified_success"]) for row in decision_rows}) > 1:
            action_success_disagreement += 1
    summary["oracle_min_p95_action_counts"] = dict(oracle_counts)
    summary["fail_all_rate"] = fail_all / max(len(grouped), 1)
    summary["action_success_disagreement_rate"] = (
        action_success_disagreement / max(len(grouped), 1)
    )
    return summary
