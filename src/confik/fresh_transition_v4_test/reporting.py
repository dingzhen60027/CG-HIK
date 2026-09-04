"""Deterministic reporting for the final fresh transition-rich V4 test.

Every value in the compact tables is recomputed from the per-frame raw arrays.
The trajectory, rather than a frame, is the independent closed-loop unit; frame
quantiles are nevertheless retained as the preregistered operational summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .benchmark import BenchmarkData, METHODS


FAMILIES = (
    "smooth_fast_orientation_smooth",
    "regular_near_singular_regular",
    "central_joint_limit_skim_return",
    "slow_high_curvature_high_speed_slow",
)
ROBOTS = ("panda", "ur5e")
TRAJECTORIES_PER_ROBOT = 80
TRAJECTORIES_PER_FAMILY = 20
FRAMES_PER_TRAJECTORY = 150
FRAMES_PER_ROBOT = TRAJECTORIES_PER_ROBOT * FRAMES_PER_TRAJECTORY


def _groups(data: BenchmarkData) -> tuple[tuple[str, np.ndarray], ...]:
    uid = np.asarray(data.trajectory_uid).astype(str)
    groups: list[tuple[str, np.ndarray]] = []
    for raw_uid in np.asarray(data.trajectory_order).astype(str):
        indices = np.flatnonzero(uid == raw_uid).astype(np.int64)
        if indices.shape != (FRAMES_PER_TRAJECTORY,):
            raise ValueError(f"trajectory {raw_uid} does not have 150 frames")
        if not np.array_equal(
            np.asarray(data.time_index)[indices],
            np.arange(FRAMES_PER_TRAJECTORY, dtype=np.int64),
        ):
            raise ValueError(f"trajectory {raw_uid} is not in exact frame order")
        groups.append((raw_uid, indices))
    flattened = np.concatenate([indices for _, indices in groups])
    if not np.array_equal(np.sort(flattened), np.arange(FRAMES_PER_ROBOT)):
        raise ValueError("trajectory groups do not partition the raw frame rows")
    return tuple(groups)


def validate_benchmark(data: BenchmarkData) -> None:
    """Validate the final fixed-size raw contract before any aggregation."""

    if data.robot not in ROBOTS:
        raise ValueError(f"unsupported robot {data.robot!r}")
    if tuple(data.method_names) != METHODS:
        raise ValueError("the final benchmark must contain exactly three methods")
    if len(data.source_query_hash) != FRAMES_PER_ROBOT:
        raise ValueError("the final benchmark must contain exactly 80x150 frames")
    if len(data.trajectory_order) != TRAJECTORIES_PER_ROBOT:
        raise ValueError("the final benchmark must contain exactly 80 trajectories")
    category = np.asarray(data.category).astype(str)
    if set(category.tolist()) != set(FAMILIES):
        raise ValueError("fresh transition family set changed")
    for family in FAMILIES:
        mask = category == family
        if int(np.sum(mask)) != TRAJECTORIES_PER_FAMILY * FRAMES_PER_TRAJECTORY:
            raise ValueError(f"family {family} does not contain exactly 20x150 frames")
        family_uids = set(np.asarray(data.trajectory_uid).astype(str)[mask].tolist())
        if len(family_uids) != TRAJECTORIES_PER_FAMILY:
            raise ValueError(f"family {family} does not contain exactly 20 trajectories")
    if len(set(np.asarray(data.source_query_hash).astype(str).tolist())) != FRAMES_PER_ROBOT:
        raise ValueError("fresh source query hashes are not unique")
    if not np.all(np.asarray(data.expected_reachable, dtype=bool)):
        raise ValueError("fresh FK target suite unexpectedly contains unreachable labels")
    if not np.all(np.asarray(data.continuity_feasible, dtype=bool)):
        raise ValueError("fresh reference suite violates its continuity contract")
    accepted = np.asarray(data.accepted, dtype=bool)
    command_finite = np.all(np.isfinite(np.asarray(data.command_q, dtype=np.float64)), axis=2)
    if not np.array_equal(command_finite, accepted):
        raise ValueError("accepted/command sentinel contract is inconsistent")
    expected_violation = accepted & ~(
        np.asarray(data.verifier_checked, dtype=bool)
        & np.asarray(data.verifier_accepted, dtype=bool)
    )
    if not np.array_equal(
        expected_violation,
        np.asarray(data.accepted_contract_violation, dtype=bool),
    ):
        raise ValueError("accepted contract-violation telemetry is inconsistent")
    _groups(data)


def _verified(data: BenchmarkData) -> np.ndarray:
    """Return the independently reverified command-acceptance mask.

    Runtime ``accepted`` telemetry is intentionally not sufficient for a
    scientific success claim.  A frame counts as verified only when the
    benchmark's independent invocation of the same deterministic verifier was
    performed and also accepted the materialized command.
    """

    return (
        np.asarray(data.accepted, dtype=bool)
        & np.asarray(data.verifier_checked, dtype=bool)
        & np.asarray(data.verifier_accepted, dtype=bool)
    )


def _completion(data: BenchmarkData) -> np.ndarray:
    groups = _groups(data)
    accepted = _verified(data)
    return np.asarray(
        [[bool(np.all(accepted[indices, column])) for _, indices in groups]
         for column in range(len(METHODS))],
        dtype=bool,
    )


def trajectory_rows(data: BenchmarkData) -> list[dict[str, Any]]:
    """Return one auditable row per robot, method, and complete trajectory."""

    validate_benchmark(data)
    completion = _completion(data)
    verified = _verified(data)
    rows: list[dict[str, Any]] = []
    for column, method in enumerate(METHODS):
        for ordinal, (uid, indices) in enumerate(_groups(data)):
            families = set(np.asarray(data.category).astype(str)[indices].tolist())
            if len(families) != 1:
                raise ValueError(f"trajectory {uid} spans multiple families")
            total_ns = int(np.sum(data.latency_ns[indices, column], dtype=np.int64))
            rows.append(
                {
                    "robot": data.robot,
                    "method": method,
                    "trajectory_ordinal": ordinal,
                    "trajectory_uid": uid,
                    "family": next(iter(families)),
                    "frame_count": FRAMES_PER_TRAJECTORY,
                    "completed": bool(completion[column, ordinal]),
                    "verified_frame_count": int(np.sum(verified[indices, column])),
                    "cumulative_latency_ns": total_ns,
                    "cumulative_latency_ms": total_ns / 1e6,
                    "mean_fev": float(np.mean(data.function_evaluations[indices, column])),
                    "fallback_rate": float(np.mean(data.fallback_used[indices, column])),
                    "accepted_contract_violation_count": int(
                        np.sum(data.accepted_contract_violation[indices, column])
                    ),
                }
            )
    return rows


def main_rows(data: BenchmarkData) -> list[dict[str, Any]]:
    """Build the preregistered main table from raw per-frame records."""

    validate_benchmark(data)
    groups = _groups(data)
    completion = _completion(data)
    verified = _verified(data)
    rows: list[dict[str, Any]] = []
    for column, method in enumerate(METHODS):
        latency_ns = np.asarray(data.latency_ns[:, column], dtype=np.int64)
        trajectory_latency_ns = np.asarray(
            [np.sum(latency_ns[indices], dtype=np.int64) for _, indices in groups],
            dtype=np.int64,
        )
        completed_uids = [
            uid for (uid, _), completed in zip(groups, completion[column], strict=True)
            if completed
        ]
        rows.append(
            {
                "robot": data.robot,
                "method": method,
                "trajectory_count": TRAJECTORIES_PER_ROBOT,
                "frame_count": FRAMES_PER_ROBOT,
                "whole_trajectory_completion_count": int(np.sum(completion[column])),
                "whole_trajectory_completion_rate": float(np.mean(completion[column])),
                "completion_trajectory_uids": completed_uids,
                "frame_verified_success_count": int(np.sum(verified[:, column])),
                "frame_verified_success_rate": float(np.mean(verified[:, column])),
                "accepted_contract_violation_count": int(
                    np.sum(data.accepted_contract_violation[:, column])
                ),
                "total_cumulative_latency_ns": int(
                    np.sum(latency_ns, dtype=np.int64)
                ),
                "total_cumulative_latency_seconds": float(
                    np.sum(latency_ns, dtype=np.int64) / 1e9
                ),
                "trajectory_cumulative_latency_mean_ms": float(
                    np.mean(trajectory_latency_ns) / 1e6
                ),
                "trajectory_cumulative_latency_median_ms": float(
                    np.median(trajectory_latency_ns) / 1e6
                ),
                "trajectory_cumulative_latency_p95_ms": float(
                    np.quantile(trajectory_latency_ns, 0.95) / 1e6
                ),
                "frame_p50_latency_ms": float(np.quantile(latency_ns, 0.50) / 1e6),
                "frame_p95_latency_ms": float(np.quantile(latency_ns, 0.95) / 1e6),
                "frame_p99_latency_ms": float(np.quantile(latency_ns, 0.99) / 1e6),
                "mean_fev": float(np.mean(data.function_evaluations[:, column])),
                "fallback_rate": float(np.mean(data.fallback_used[:, column])),
                "learned_seed_invocation_rate": float(
                    np.mean(data.learned_seed_invoked[:, column])
                ),
            }
        )
    return rows


def family_rows(data: BenchmarkData) -> list[dict[str, Any]]:
    """Aggregate completion, exact cumulative latency, and FEV by family."""

    validate_benchmark(data)
    all_groups = _groups(data)
    completion = _completion(data)
    verified = _verified(data)
    categories = np.asarray(data.category).astype(str)
    rows: list[dict[str, Any]] = []
    for column, method in enumerate(METHODS):
        for family in FAMILIES:
            family_ordinal = [
                ordinal
                for ordinal, (_, indices) in enumerate(all_groups)
                if str(categories[int(indices[0])]) == family
            ]
            indices = np.concatenate([all_groups[ordinal][1] for ordinal in family_ordinal])
            trajectory_latency = np.asarray(
                [
                    np.sum(data.latency_ns[all_groups[ordinal][1], column], dtype=np.int64)
                    for ordinal in family_ordinal
                ],
                dtype=np.int64,
            )
            rows.append(
                {
                    "robot": data.robot,
                    "method": method,
                    "family": family,
                    "trajectory_count": TRAJECTORIES_PER_FAMILY,
                    "frame_count": TRAJECTORIES_PER_FAMILY * FRAMES_PER_TRAJECTORY,
                    "whole_trajectory_completion_count": int(
                        np.sum(completion[column, family_ordinal])
                    ),
                    "whole_trajectory_completion_rate": float(
                        np.mean(completion[column, family_ordinal])
                    ),
                    "frame_verified_success_rate": float(
                        np.mean(verified[indices, column])
                    ),
                    "total_cumulative_latency_ns": int(
                        np.sum(trajectory_latency, dtype=np.int64)
                    ),
                    "total_cumulative_latency_seconds": float(
                        np.sum(trajectory_latency, dtype=np.int64) / 1e9
                    ),
                    "trajectory_cumulative_latency_mean_ms": float(
                        np.mean(trajectory_latency) / 1e6
                    ),
                    "trajectory_cumulative_latency_median_ms": float(
                        np.median(trajectory_latency) / 1e6
                    ),
                    "trajectory_cumulative_latency_p95_ms": float(
                        np.quantile(trajectory_latency, 0.95) / 1e6
                    ),
                    "mean_fev": float(np.mean(data.function_evaluations[indices, column])),
                    "fallback_rate": float(np.mean(data.fallback_used[indices, column])),
                    "accepted_contract_violation_count": int(
                        np.sum(data.accepted_contract_violation[indices, column])
                    ),
                }
            )
    return rows


def completion_identity(data: BenchmarkData) -> dict[str, Any]:
    """Persist ordered completion vectors and V4 lost/gained trajectory UIDs."""

    validate_benchmark(data)
    completion = _completion(data)
    order = np.asarray(data.trajectory_order).astype(str).tolist()
    by_method = {
        method: {
            "completion_vector": completion[column].tolist(),
            "completion_count": int(np.sum(completion[column])),
            "completed_trajectory_uids": [
                uid for uid, ok in zip(order, completion[column], strict=True) if ok
            ],
        }
        for column, method in enumerate(METHODS)
    }
    hard = completion[METHODS.index("always_hard")]
    v4 = completion[METHODS.index("counterfactual_cghik_v4")]
    return {
        "robot": data.robot,
        "trajectory_uid_order": order,
        "methods": by_method,
        "primary_comparison": "counterfactual_cghik_v4_vs_always_hard",
        "v4_lost_vs_always_hard_trajectory_uids": [
            uid for uid, hard_ok, v4_ok in zip(order, hard, v4, strict=True)
            if hard_ok and not v4_ok
        ],
        "v4_gained_vs_always_hard_trajectory_uids": [
            uid for uid, hard_ok, v4_ok in zip(order, hard, v4, strict=True)
            if v4_ok and not hard_ok
        ],
        "v4_completion_count_not_below_always_hard": bool(
            np.sum(v4) >= np.sum(hard)
        ),
    }


def final_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the six frozen V4-vs-HARD gates, separately per robot."""

    index = {
        (str(row["robot"]), str(row["method"])): row
        for row in rows
    }
    expected = {(robot, method) for robot in ROBOTS for method in METHODS}
    if set(index) != expected:
        raise ValueError("final gate requires one main row per robot and method")
    robots: dict[str, Any] = {}
    for robot in ROBOTS:
        hard = index[(robot, "always_hard")]
        v4 = index[(robot, "counterfactual_cghik_v4")]
        hard_latency = int(hard["total_cumulative_latency_ns"])
        hard_fev = float(hard["mean_fev"])
        if hard_latency <= 0 or hard_fev <= 0:
            raise ValueError("always-hard denominator must be positive")
        ratios = {
            "aggregate_cumulative_latency": int(v4["total_cumulative_latency_ns"])
            / hard_latency,
            "mean_fev": float(v4["mean_fev"]) / hard_fev,
            "p50_latency": float(v4["frame_p50_latency_ms"])
            / float(hard["frame_p50_latency_ms"]),
            "p95_latency": float(v4["frame_p95_latency_ms"])
            / float(hard["frame_p95_latency_ms"]),
            "p99_latency": float(v4["frame_p99_latency_ms"])
            / float(hard["frame_p99_latency_ms"]),
        }
        checks = {
            "v4_completion_count_not_below_always_hard": int(
                v4["whole_trajectory_completion_count"]
            ) >= int(hard["whole_trajectory_completion_count"]),
            "aggregate_cumulative_latency_ratio_at_most_0_85": ratios[
                "aggregate_cumulative_latency"
            ] <= 0.85,
            "mean_fev_ratio_at_most_0_85": ratios["mean_fev"] <= 0.85,
            "p95_latency_ratio_at_most_1_0": ratios["p95_latency"] <= 1.0,
            "p99_latency_ratio_at_most_1_05": ratios["p99_latency"] <= 1.05,
            "accepted_contract_violation_count_zero": int(
                sum(
                    int(index[(robot, method)]["accepted_contract_violation_count"])
                    for method in METHODS
                )
            ) == 0,
        }
        robots[robot] = {
            "pass": bool(all(checks.values())),
            "checks": checks,
            "ratios_vs_always_hard": ratios,
            "accepted_contract_violation_counts_by_method": {
                method: int(
                    index[(robot, method)]["accepted_contract_violation_count"]
                )
                for method in METHODS
            },
            "accepted_contract_violation_gate_scope": "all_three_methods",
            "p50_is_report_only": True,
        }
    passed = bool(all(value["pass"] for value in robots.values()))
    return {
        "protocol": "fresh_transition_v4_final_gate_v1",
        "status": "pass" if passed else "fail",
        "all_robots_pass": passed,
        "robots": robots,
        "method_frozen_before_results": True,
        "results_used_for_tuning": False,
        "next_stage_is_paper_regardless_of_gate": True,
    }


def main_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Robot | Method | Completion | Frame success | Total s | P50 ms | P95 ms | P99 ms | Mean FEV | Fallback | Violations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {robot} | {method} | {whole_trajectory_completion_count}/80 | "
            "{frame_verified_success_rate:.6f} | {total_cumulative_latency_seconds:.6f} | "
            "{frame_p50_latency_ms:.6f} | {frame_p95_latency_ms:.6f} | "
            "{frame_p99_latency_ms:.6f} | {mean_fev:.6f} | {fallback_rate:.6f} | "
            "{accepted_contract_violation_count} |".format(**row)
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "FAMILIES",
    "ROBOTS",
    "completion_identity",
    "family_rows",
    "final_gate",
    "main_markdown",
    "main_rows",
    "trajectory_rows",
    "validate_benchmark",
]
