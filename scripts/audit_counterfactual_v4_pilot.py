#!/usr/bin/env python3
"""Independent, read-only audit of the validation-only v4 label pilot.

The script reads ``outputs/counterfactual_v4_pilot`` and the explicitly named
training/validation source artifacts, then writes one audit JSON under
``docs/audits/counterfactual_v4_pilot``.  It never opens a v2/v3 formal test
artifact and never writes below ``outputs``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


ROBOTS = ("panda", "ur5e")
SEED = 17
DECISION_ACTIONS = ("easy", "medium", "hard")
COLLECTED_ACTIONS = DECISION_ACTIONS + ("fixed_robust",)
FEATURE_NAMES = (
    "learned_seed_position_error",
    "learned_seed_orientation_error",
    "ensemble_uncertainty_mean",
    "ensemble_uncertainty_max",
    "learned_seed_min_singular_value",
    "learned_seed_joint_limit_margin",
    "learned_seed_joint_step_l2",
    "current_pose_position_step",
    "current_pose_orientation_step",
)
TIMING_KEYS = (
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
RECORD_KEYS = {
    "category",
    "command_q",
    "continuity_feasible",
    "deadline_success_rate",
    "dynamic_history_available",
    "entry_action",
    "executed_stages",
    "expected_reachable",
    "failure_reason",
    "fallback_used",
    "fixed_robust_matches_easy",
    "function_evaluations",
    "iterations",
    "latency_p50_ns",
    "latency_p95_ns",
    "latency_samples_ns",
    "max_joint_acceleration_rad_s2",
    "max_joint_jerk_rad_s3",
    "max_joint_step_rad",
    "max_joint_velocity_rad_s",
    "max_velocity_limit_utilization",
    "query_index",
    "query_sha256",
    "risk_features",
    "robot",
    "source_index",
    "source_query_sha256",
    "source_role",
    "time_index",
    "timing_samples_ns",
    "training_seed",
    "trajectory_id",
    "verification_reasons",
    "verified_success",
    "verified_success_before_deadline",
}
NPZ_KEYS = {
    "feature_names",
    "action_names",
    "decision_action_names",
    "features",
    "source_indices",
    "query_sha256",
    "category",
    "expected_reachable",
    "continuity_feasible",
    "verified_success",
    "verified_success_before_deadline",
    "latency_p50_ms",
    "latency_p95_ms",
    "function_evaluations",
    "fallback_used",
    "failure_reason",
    "command_q",
    "max_joint_step_rad",
    "max_joint_velocity_rad_s",
    "max_velocity_limit_utilization",
    "max_joint_acceleration_rad_s2",
    "max_joint_jerk_rad_s3",
    "dynamic_history_available",
}
SEMANTIC_FIELDS = (
    "verified_success",
    "function_evaluations",
    "iterations",
    "fallback_used",
    "executed_stages",
    "failure_reason",
    "verification_reasons",
    "command_q",
    "max_joint_step_rad",
    "max_joint_velocity_rad_s",
    "max_velocity_limit_utilization",
    "max_joint_acceleration_rad_s2",
    "max_joint_jerk_rad_s3",
    "dynamic_history_available",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/audits/counterfactual_v4_pilot/pilot_audit.json"),
    )
    return parser.parse_args()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def strict_json_line(line: str, path: Path, number: int) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(
            f"non-finite JSON constant {value!r} in {path}:{number}"
        )

    value = json.loads(line, parse_constant=reject)
    if not isinstance(value, dict):
        raise TypeError(f"JSONL row is not an object in {path}:{number}")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        try:
            return bool(
                np.array_equal(
                    np.asarray(left, dtype=np.float64),
                    np.asarray(right, dtype=np.float64),
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return left == right
    return left == right


def query_digest(source: Any, index: int, dt: float = 0.02) -> str:
    digest = sha256()
    for array in (
        source["previous_q"][index],
        source["target_position"][index],
        source["target_rotation"][index],
    ):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    digest.update(np.asarray([dt], dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            records.append(strict_json_line(line, path, number))
    return records


def hash_audit(
    workspace: Path,
    output_root: Path,
    combination_root: Path,
    run_manifest: dict[str, Any],
    selection: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[str] = []
    for name, expected in artifact["generated_files"].items():
        path = combination_root / name
        if not path.is_file():
            mismatches.append(f"missing generated file: {path}")
            continue
        if path.stat().st_size != int(expected["size"]):
            mismatches.append(f"size mismatch: {path}")
        if file_sha256(path) != str(expected["sha256"]):
            mismatches.append(f"sha256 mismatch: {path}")
    release_mismatches: list[str] = []
    for name, expected in artifact["release_artifacts"].items():
        path = Path(expected["path"])
        if not path.is_file() or file_sha256(path) != str(expected["sha256"]):
            release_mismatches.append(name)

    source_path = Path(selection["source_path"])
    source_ok = source_path.is_file() and file_sha256(source_path) == selection["source_sha256"]
    config_path = Path(run_manifest["config_path"])
    runner_path = workspace / "src/confik/counterfactual_v4/runner.py"
    release_root = Path(run_manifest["release_root"])
    return {
        "generated_artifact_mismatches": mismatches,
        "release_artifact_mismatches": release_mismatches,
        "source_sha256_matches": bool(source_ok),
        "config_sha256_matches_current_file": bool(
            config_path.is_file()
            and file_sha256(config_path) == run_manifest["config_sha256"]
        ),
        "runner_sha256_matches_current_file": bool(
            runner_path.is_file()
            and file_sha256(runner_path) == run_manifest["runner_sha256"]
        ),
        "release_manifest_sha256_matches": bool(
            file_sha256(release_root / "release_manifest.json")
            == run_manifest["release_manifest_sha256"]
        ),
        "release_equivalence_sha256_matches": bool(
            file_sha256(release_root / "release_equivalence.json")
            == run_manifest["release_equivalence_sha256"]
        ),
        "collector_sha256_not_explicitly_recorded": True,
        "collector_recoverable_from_clean_collection_commit": bool(
            not any(
                "src/confik/counterfactual_v4/collector.py" in line
                for line in run_manifest["git_status_at_collection"]
            )
        ),
        "output_root_read_only_target": str(output_root),
    }


def source_and_npz_audit(
    records: list[dict[str, Any]],
    grouped: dict[int, dict[str, dict[str, Any]]],
    label_path: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    mismatches: Counter[str] = Counter()
    source_path = Path(selection["source_path"])
    with np.load(label_path, allow_pickle=False) as labels, np.load(
        source_path, allow_pickle=False
    ) as source:
        # NpzFile.__getitem__ decompresses an array on every access. Cache each
        # member once before the row-wise cross-check so the audit stays O(n).
        label_arrays = {name: labels[name] for name in labels.files}
        source_arrays = {name: source[name] for name in source.files}
        if set(labels.files) != NPZ_KEYS:
            mismatches["npz_key_set"] += 1
        if tuple(label_arrays["feature_names"].astype(str)) != FEATURE_NAMES:
            mismatches["feature_names"] += 1
        if tuple(label_arrays["action_names"].astype(str)) != COLLECTED_ACTIONS:
            mismatches["action_names"] += 1
        if tuple(label_arrays["decision_action_names"].astype(str)) != DECISION_ACTIONS:
            mismatches["decision_action_names"] += 1
        count = len(grouped)
        if label_arrays["features"].shape != (count, len(FEATURE_NAMES)):
            mismatches["feature_shape"] += 1
        selected = np.asarray(label_arrays["source_indices"], dtype=np.int64)
        if len(np.unique(selected)) != len(selected):
            mismatches["duplicate_source_indices"] += 1
        selection_hash = sha256(
            np.ascontiguousarray(selected, dtype=np.int64).tobytes()
        ).hexdigest()
        if selection_hash != selection["selection_indices_sha256"]:
            mismatches["selection_indices_sha256"] += 1
        source_query_hashes: list[str] = []
        for query_index in range(count):
            rows = grouped[query_index]
            reference = rows["easy"]
            source_index = int(selected[query_index])
            digest = query_digest(source_arrays, source_index)
            source_query_hashes.append(digest)
            checks = {
                "source_index": int(reference["source_index"]) == source_index,
                "source_category": str(reference["category"])
                == str(source_arrays["category"][source_index]),
                "source_expected_reachable": bool(reference["expected_reachable"])
                == bool(source_arrays["expected_reachable"][source_index]),
                "source_continuity_feasible": bool(reference["continuity_feasible"])
                == bool(source_arrays["continuity_feasible"][source_index]),
                "source_trajectory_id": int(reference["trajectory_id"])
                == int(source_arrays["trajectory_id"][source_index]),
                "source_time_index": int(reference["time_index"])
                == int(source_arrays["time_index"][source_index]),
                "source_query_sha256": reference["query_sha256"] == digest,
                "npz_query_sha256": str(label_arrays["query_sha256"][query_index]) == digest,
                "npz_category": str(label_arrays["category"][query_index])
                == str(reference["category"]),
                "npz_expected_reachable": bool(
                    label_arrays["expected_reachable"][query_index]
                )
                == bool(reference["expected_reachable"]),
                "npz_continuity_feasible": bool(
                    label_arrays["continuity_feasible"][query_index]
                )
                == bool(reference["continuity_feasible"]),
                "npz_features": np.array_equal(
                    label_arrays["features"][query_index],
                    np.asarray(reference["risk_features"], dtype=np.float64),
                ),
            }
            for name, passed in checks.items():
                if not passed:
                    mismatches[name] += 1
            for action_index, action in enumerate(COLLECTED_ACTIONS):
                row = rows[action]
                scalar_checks = {
                    "npz_verified_success": bool(
                        label_arrays["verified_success"][query_index, action_index]
                    )
                    == bool(row["verified_success"]),
                    "npz_verified_success_before_deadline": bool(
                        label_arrays["verified_success_before_deadline"][
                            query_index, action_index
                        ]
                    )
                    == bool(row["verified_success_before_deadline"]),
                    "npz_latency_p50": np.isclose(
                        label_arrays["latency_p50_ms"][query_index, action_index],
                        float(row["latency_p50_ns"]) / 1e6,
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    "npz_latency_p95": np.isclose(
                        label_arrays["latency_p95_ms"][query_index, action_index],
                        float(row["latency_p95_ns"]) / 1e6,
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    "npz_fev": int(label_arrays["function_evaluations"][query_index, action_index])
                    == int(row["function_evaluations"]),
                    "npz_fallback": bool(label_arrays["fallback_used"][query_index, action_index])
                    == bool(row["fallback_used"]),
                    "npz_failure_reason": str(
                        label_arrays["failure_reason"][query_index, action_index]
                    )
                    == str(row["failure_reason"]),
                }
                for name, passed in scalar_checks.items():
                    if not passed:
                        mismatches[name] += 1
                expected_command = (
                    np.full(label_arrays["command_q"].shape[2], np.nan)
                    if row["command_q"] is None
                    else np.asarray(row["command_q"], dtype=np.float64)
                )
                if not np.array_equal(
                    label_arrays["command_q"][query_index, action_index],
                    expected_command,
                    equal_nan=True,
                ):
                    mismatches["npz_command"] += 1
        source_counts = Counter(source_arrays["category"].astype(str).tolist())
        selected_counts = Counter(label_arrays["category"].astype(str).tolist())
        max_share_difference = max(
            abs(
                selected_counts[name] / count
                - source_counts[name] / len(source_arrays["category"])
            )
            for name in source_counts
        )
    return {
        "mismatch_counts": dict(sorted(mismatches.items())),
        "all_raw_npz_source_checks_pass": not mismatches,
        "selected_source_indices_unique": len(set(int(r["source_index"]) for r in records))
        == len(grouped),
        "query_hashes_unique": len(set(source_query_hashes)) == len(source_query_hashes),
        "selected_category_counts": dict(sorted(selected_counts.items())),
        "source_category_counts": dict(sorted(source_counts.items())),
        "max_absolute_selected_vs_source_category_share_difference": float(
            max_share_difference
        ),
    }


def timing_audit(
    records: list[dict[str, Any]], grouped: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    problems: Counter[str] = Counter()
    action_cv: dict[str, list[float]] = defaultdict(list)
    action_range_ms: dict[str, list[float]] = defaultdict(list)
    action_total_seconds: Counter[str] = Counter()
    deadline: dict[str, Counter[str]] = {
        action: Counter() for action in COLLECTED_ACTIONS
    }
    for row in records:
        action = str(row["entry_action"])
        samples = np.asarray(row["latency_samples_ns"], dtype=np.int64)
        timing = row["timing_samples_ns"]
        if len(samples) != 5:
            problems["latency_repeat_count"] += 1
        if set(timing) != set(TIMING_KEYS):
            problems["timing_key_set"] += 1
            continue
        if any(len(timing[key]) != len(samples) for key in TIMING_KEYS):
            problems["stage_repeat_count"] += 1
            continue
        totals = np.asarray(timing["total_end_to_end_ns"], dtype=np.int64)
        if not np.array_equal(samples, totals):
            problems["total_sample_mismatch"] += 1
        core = np.zeros(len(samples), dtype=np.int64)
        for key in TIMING_KEYS:
            if key not in ("total_end_to_end_ns", "unattributed_framework_ns"):
                core += np.asarray(timing[key], dtype=np.int64)
        unattributed = np.asarray(timing["unattributed_framework_ns"], dtype=np.int64)
        if not np.array_equal(core + unattributed, samples):
            problems["stage_sum_mismatch"] += 1
        if np.any(unattributed < 0):
            problems["negative_unattributed"] += 1
        if not np.isclose(np.percentile(samples, 50), row["latency_p50_ns"]):
            problems["p50_mismatch"] += 1
        if not np.isclose(np.percentile(samples, 95), row["latency_p95_ns"]):
            problems["p95_mismatch"] += 1
        expected_rate = float(np.mean(samples <= 20_000_000))
        if not np.isclose(expected_rate, row["deadline_success_rate"]):
            problems["deadline_rate_mismatch"] += 1
        expected_label = bool(
            row["verified_success"] and np.percentile(samples, 95) <= 20_000_000
        )
        if expected_label != bool(row["verified_success_before_deadline"]):
            problems["deadline_label_mismatch"] += 1
        action_cv[action].append(float(np.std(samples, ddof=1) / np.mean(samples)))
        action_range_ms[action].append(float(np.ptp(samples) / 1e6))
        action_total_seconds[action] += float(np.sum(samples) / 1e9)
        misses = int(np.count_nonzero(samples > 20_000_000))
        deadline[action]["any_repeat_miss"] += misses > 0
        deadline[action]["all_five_repeat_miss"] += misses == 5
        deadline[action]["straddles_20ms"] += 0 < misses < 5
        deadline[action]["empirical_p95_miss"] += row["latency_p95_ns"] > 20_000_000

    # Reconstruct the Latin-rotation order from the recorded deterministic seed.
    selection_seed = None
    # The seed is supplied by the caller later; leave position balance separate.
    return {
        "problem_counts": dict(sorted(problems.items())),
        "all_timing_checks_pass": not problems,
        "repeat_count": 5,
        "empirical_p95_definition": (
            "NumPy linear percentile over five samples; P95 is 0.8*maximum + "
            "0.2*second-largest, not a well-resolved per-query tail estimate"
        ),
        "per_query_repeat_cv": {
            action: distribution(values) for action, values in action_cv.items()
        },
        "per_query_repeat_range_ms": {
            action: distribution(values) for action, values in action_range_ms.items()
        },
        "total_measured_action_seconds": dict(action_total_seconds),
        "deadline_noise": {
            action: dict(counts) for action, counts in deadline.items()
        },
    }


def order_balance(selection_seed: int, query_count: int) -> dict[str, Any]:
    positions = {action: Counter() for action in COLLECTED_ACTIONS}
    for query_index in range(query_count):
        rng = np.random.default_rng(selection_seed + query_index)
        base = list(COLLECTED_ACTIONS)
        rng.shuffle(base)
        for repeat in range(5):
            offset = repeat % len(base)
            order = base[offset:] + base[:offset]
            for position, action in enumerate(order):
                positions[action][position] += 1
    ideal = query_count * 5 / len(COLLECTED_ACTIONS)
    deviations = [
        abs(count - ideal) / ideal
        for counts in positions.values()
        for count in counts.values()
    ]
    return {
        "counts_by_action_and_position": {
            action: {str(pos): int(counts[pos]) for pos in range(4)}
            for action, counts in positions.items()
        },
        "ideal_count_per_action_position": ideal,
        "maximum_relative_deviation_from_ideal": float(max(deviations)),
        "first_four_repeats_exact_latin_balance_per_query": True,
        "fifth_repeat_reuses_the_base_order": True,
    }


def fixed_easy_audit(
    grouped: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    mismatches: Counter[str] = Counter()
    deadline_mismatches = 0
    p50_delta_ms: list[float] = []
    p95_delta_ms: list[float] = []
    for rows in grouped.values():
        easy = rows["easy"]
        fixed = rows["fixed_robust"]
        for field in SEMANTIC_FIELDS:
            if not values_equal(easy[field], fixed[field]):
                mismatches[field] += 1
        if not easy["fixed_robust_matches_easy"] or not fixed["fixed_robust_matches_easy"]:
            mismatches["fixed_robust_matches_easy_flag"] += 1
        deadline_mismatches += (
            bool(easy["verified_success_before_deadline"])
            != bool(fixed["verified_success_before_deadline"])
        )
        p50_delta_ms.append(
            (float(fixed["latency_p50_ns"]) - float(easy["latency_p50_ns"])) / 1e6
        )
        p95_delta_ms.append(
            (float(fixed["latency_p95_ns"]) - float(easy["latency_p95_ns"])) / 1e6
        )
    return {
        "semantic_mismatch_counts": dict(sorted(mismatches.items())),
        "semantic_equivalence_pass": not mismatches,
        "deadline_label_mismatch_count_due_only_to_timing": int(deadline_mismatches),
        "fixed_minus_easy_p50_ms": distribution(p50_delta_ms),
        "fixed_minus_easy_p95_ms": distribution(p95_delta_ms),
    }


def winner_and_class_audit(
    grouped: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    full_winners: Counter[str] = Counter()
    repeat_winners: Counter[str] = Counter()
    repeat_agreement: list[float] = []
    leave_one_out_agreement: list[float] = []
    supported_four_of_five = 0
    all_repeat_same = 0
    p50_p95_same = 0
    margins_ms: list[float] = []
    margins_over_noise: list[float] = []
    action_success_disagreement = 0
    deadline_disagreement = 0
    fail_all = 0
    semantic_fail_all = 0
    deadline_only_fail_all = 0
    feasible_fail_all = 0
    category: dict[str, dict[str, Any]] = {}
    category_groups: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)

    for rows in grouped.values():
        category_groups[str(rows["easy"]["category"])].append(rows)
        semantic = {bool(rows[action]["verified_success"]) for action in DECISION_ACTIONS}
        deadline = {
            bool(rows[action]["verified_success_before_deadline"])
            for action in DECISION_ACTIONS
        }
        action_success_disagreement += len(semantic) > 1
        deadline_disagreement += len(deadline) > 1
        eligible = [
            action
            for action in DECISION_ACTIONS
            if rows[action]["verified_success_before_deadline"]
        ]
        if not eligible:
            fail_all += 1
            any_semantic = any(rows[action]["verified_success"] for action in DECISION_ACTIONS)
            semantic_fail_all += not any_semantic
            deadline_only_fail_all += any_semantic
            feasible_fail_all += bool(
                rows["easy"]["expected_reachable"]
                and rows["easy"]["continuity_feasible"]
            )
            full_winners["fail_all"] += 1
            continue

        sample_matrix = np.asarray(
            [rows[action]["latency_samples_ns"] for action in eligible],
            dtype=np.float64,
        )
        p95 = np.percentile(sample_matrix, 95, axis=1)
        p50 = np.percentile(sample_matrix, 50, axis=1)
        order = np.argsort(p95)
        full_index = int(order[0])
        full_action = eligible[full_index]
        full_winners[full_action] += 1
        p50_action = eligible[int(np.argmin(p50))]
        p50_p95_same += p50_action == full_action
        if len(eligible) > 1:
            margins_ms.append(float((p95[order[1]] - p95[order[0]]) / 1e6))
        repeat_indices = np.argmin(sample_matrix, axis=0)
        repeat_actions = [eligible[int(index)] for index in repeat_indices]
        repeat_winners.update(repeat_actions)
        agreement = float(np.mean(repeat_indices == full_index))
        repeat_agreement.append(agreement)
        all_repeat_same += len(set(repeat_actions)) == 1
        leave_one_out: list[int] = []
        for repeat in range(sample_matrix.shape[1]):
            reduced = np.delete(sample_matrix, repeat, axis=1)
            leave_one_out.append(int(np.argmin(np.percentile(reduced, 95, axis=1))))
        loo_agreement = float(np.mean(np.asarray(leave_one_out) == full_index))
        leave_one_out_agreement.append(loo_agreement)
        supported_four_of_five += sum(index == full_index for index in leave_one_out) >= 4
        if len(eligible) > 1:
            noise = float(np.median(np.std(sample_matrix, axis=1, ddof=1))) / 1e6
            margins_over_noise.append(margins_ms[-1] / max(noise, 1e-12))

    for name, groups in sorted(category_groups.items()):
        winners: Counter[str] = Counter()
        semantic_fail = 0
        deadline_only = 0
        feasible = 0
        for rows in groups:
            eligible = [
                action
                for action in DECISION_ACTIONS
                if rows[action]["verified_success_before_deadline"]
            ]
            if not eligible:
                winners["fail_all"] += 1
                any_semantic = any(rows[action]["verified_success"] for action in DECISION_ACTIONS)
                semantic_fail += not any_semantic
                deadline_only += any_semantic
            else:
                winner = min(eligible, key=lambda action: rows[action]["latency_p95_ns"])
                winners[winner] += 1
            feasible += bool(
                rows["easy"]["expected_reachable"]
                and rows["easy"]["continuity_feasible"]
            )
        category[name] = {
            "count": len(groups),
            "contract_feasible_count": feasible,
            "semantic_fail_all_count": semantic_fail,
            "deadline_only_fail_all_count": deadline_only,
            "oracle_empirical_p95_winner_counts": dict(winners),
        }

    eligible_count = len(grouped) - fail_all
    trivial_fail_all = sum(
        category.get(name, {}).get("oracle_empirical_p95_winner_counts", {}).get(
            "fail_all", 0
        )
        for name in ("large_step", "unreachable")
    )
    return {
        "query_count": len(grouped),
        "action_semantic_success_disagreement_count": int(action_success_disagreement),
        "action_semantic_success_disagreement_rate": float(
            action_success_disagreement / len(grouped)
        ),
        "action_deadline_label_disagreement_count": int(deadline_disagreement),
        "action_deadline_label_disagreement_rate": float(deadline_disagreement / len(grouped)),
        "fail_all": {
            "count": int(fail_all),
            "rate": float(fail_all / len(grouped)),
            "semantic_solver_fail_all_count": int(semantic_fail_all),
            "deadline_only_fail_all_count": int(deadline_only_fail_all),
            "contract_feasible_fail_all_count": int(feasible_fail_all),
            "large_step_plus_unreachable_fraction": float(
                trivial_fail_all / fail_all if fail_all else 0.0
            ),
        },
        "oracle_empirical_p95_winner_counts": dict(full_winners),
        "repeat_level_winner_counts_among_deadline_eligible_queries": dict(
            repeat_winners
        ),
        "winner_stability": {
            "deadline_eligible_query_count": eligible_count,
            "mean_fraction_of_five_repeat_winners_equal_to_full_p95_winner": float(
                np.mean(repeat_agreement)
            ),
            "all_five_repeat_winners_identical_fraction": float(
                all_repeat_same / eligible_count
            ),
            "mean_leave_one_repeat_out_agreement_with_full_winner": float(
                np.mean(leave_one_out_agreement)
            ),
            "full_winner_supported_by_at_least_four_of_five_leave_one_out_runs_fraction": float(
                supported_four_of_five / eligible_count
            ),
            "p50_and_p95_winner_agreement_fraction": float(
                p50_p95_same / eligible_count
            ),
            "winner_margin_ms": distribution(margins_ms),
            "winner_margin_divided_by_within_query_repeat_noise": distribution(
                margins_over_noise
            ),
        },
        "category_breakdown": category,
    }


def schema_and_uniqueness(
    records: list[dict[str, Any]], robot: str, seed: int
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, Any]]]]:
    schema_mismatch = 0
    metadata_mismatch: Counter[str] = Counter()
    key_counts: Counter[tuple[int, str]] = Counter()
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        schema_mismatch += set(row) != RECORD_KEYS
        if row["robot"] != robot:
            metadata_mismatch["robot"] += 1
        if int(row["training_seed"]) != seed:
            metadata_mismatch["training_seed"] += 1
        if row["source_role"] != "risk_validation_queries":
            metadata_mismatch["source_role"] += 1
        if row["query_sha256"] != row["source_query_sha256"]:
            metadata_mismatch["source_query_sha256"] += 1
        query_index = int(row["query_index"])
        action = str(row["entry_action"])
        key_counts[(query_index, action)] += 1
        grouped[query_index][action] = row
    action_set_mismatches = sum(
        set(rows) != set(COLLECTED_ACTIONS) for rows in grouped.values()
    )
    query_indices = sorted(grouped)
    return (
        {
            "record_count": len(records),
            "query_count": len(grouped),
            "record_schema_mismatch_count": int(schema_mismatch),
            "metadata_mismatch_counts": dict(metadata_mismatch),
            "duplicate_query_action_key_count": int(
                sum(count - 1 for count in key_counts.values() if count > 1)
            ),
            "query_action_set_mismatch_count": int(action_set_mismatches),
            "query_indices_contiguous_from_zero": query_indices == list(range(len(grouped))),
            "all_schema_and_uniqueness_checks_pass": bool(
                not schema_mismatch
                and not metadata_mismatch
                and not action_set_mismatches
                and all(count == 1 for count in key_counts.values())
                and query_indices == list(range(len(grouped)))
            ),
        },
        grouped,
    )


def summary_crosscheck(
    grouped: dict[int, dict[str, dict[str, Any]]], summary: dict[str, Any]
) -> dict[str, Any]:
    mismatches: list[str] = []
    if int(summary["record_count"]) != len(grouped) * len(COLLECTED_ACTIONS):
        mismatches.append("record_count")
    winner = winner_and_class_audit(grouped)
    if summary["oracle_min_p95_action_counts"] != winner["oracle_empirical_p95_winner_counts"]:
        mismatches.append("oracle_min_p95_action_counts")
    if not np.isclose(summary["fail_all_rate"], winner["fail_all"]["rate"]):
        mismatches.append("fail_all_rate")
    if not np.isclose(
        summary["action_success_disagreement_rate"],
        winner["action_semantic_success_disagreement_rate"],
    ):
        mismatches.append("action_success_disagreement_rate")
    for action in COLLECTED_ACTIONS:
        rows = [grouped[index][action] for index in sorted(grouped)]
        expected = {
            "query_count": len(rows),
            "verified_success_rate": np.mean([row["verified_success"] for row in rows]),
            "verified_success_before_deadline_rate": np.mean(
                [row["verified_success_before_deadline"] for row in rows]
            ),
            "mean_function_evaluations": np.mean(
                [row["function_evaluations"] for row in rows]
            ),
            "fallback_rate": np.mean([row["fallback_used"] for row in rows]),
        }
        for key, value in expected.items():
            if not np.isclose(float(summary["actions"][action][key]), float(value)):
                mismatches.append(f"{action}.{key}")
    return {"mismatches": mismatches, "pilot_summary_reproduced": not mismatches}


def environment_and_cost(
    summary: dict[str, Any], selection_seed: int
) -> dict[str, Any]:
    events = summary["quiet_wait_events"]
    query_indices: list[int] = []
    for event in events:
        match = re.search(r"/query(\d+)/", event["context"])
        if match:
            query_indices.append(int(match.group(1)))
    elapsed = float(summary["wall_time_seconds_excluding_warmup_and_writes"])
    waits = float(summary["quiet_wait_seconds"])
    return {
        "contaminated_query_retries": int(summary["contaminated_query_retries"]),
        "quiet_wait_event_count": int(summary["quiet_wait_event_count"]),
        "quiet_wait_seconds": waits,
        "quiet_wait_fraction_of_recorded_wall_time": waits / elapsed,
        "retry_query_indices_from_event_context": query_indices,
        "unique_retry_query_count": len(set(query_indices)),
        "repeatedly_contaminated_query_indices": dict(
            Counter(index for index in query_indices if query_indices.count(index) > 1)
        ),
        "retained_records_have_no_discarded_sample_audit_trail": True,
        "seconds_per_query_four_actions_five_repeats": float(
            summary["seconds_per_query_four_collected_actions"]
        ),
        "wall_time_seconds_excluding_warmup_and_writes": elapsed,
        "action_order_balance": order_balance(selection_seed, int(summary["query_count"])),
    }


def audit_combination(
    workspace: Path,
    output_root: Path,
    run_manifest: dict[str, Any],
    robot: str,
    seed: int,
) -> dict[str, Any]:
    root = output_root / robot / f"seed{seed}"
    records = load_records(root / "counterfactual_records.jsonl.gz")
    summary = strict_json(root / "pilot_summary.json")
    selection = strict_json(root / "selection_manifest.json")
    artifact = strict_json(root / "artifact_manifest.json")
    schema, grouped = schema_and_uniqueness(records, robot, seed)
    hashes = hash_audit(
        workspace, output_root, root, run_manifest, selection, artifact
    )
    source_npz = source_and_npz_audit(
        records, grouped, root / "counterfactual_labels.npz", selection
    )
    timing = timing_audit(records, grouped)
    fixed_easy = fixed_easy_audit(grouped)
    classes = winner_and_class_audit(grouped)
    summary_check = summary_crosscheck(grouped, summary)
    environment = environment_and_cost(summary, int(selection["selection_seed"]))
    integrity_pass = bool(
        schema["all_schema_and_uniqueness_checks_pass"]
        and not hashes["generated_artifact_mismatches"]
        and not hashes["release_artifact_mismatches"]
        and hashes["source_sha256_matches"]
        and hashes["config_sha256_matches_current_file"]
        and hashes["runner_sha256_matches_current_file"]
        and hashes["release_manifest_sha256_matches"]
        and hashes["release_equivalence_sha256_matches"]
        and source_npz["all_raw_npz_source_checks_pass"]
        and timing["all_timing_checks_pass"]
        and fixed_easy["semantic_equivalence_pass"]
        and summary_check["pilot_summary_reproduced"]
    )
    return {
        "integrity_pass": integrity_pass,
        "hash_and_provenance": hashes,
        "schema_and_uniqueness": schema,
        "source_and_npz_consistency": source_npz,
        "fixed_robust_vs_easy": fixed_easy,
        "timing_repeat_quality": timing,
        "classes_and_winner_stability": classes,
        "environment_resampling_and_cost": environment,
        "summary_crosscheck": summary_check,
    }


def scale_projection(results: dict[str, Any]) -> dict[str, Any]:
    rates = {
        robot: results[f"{robot}/seed{SEED}"]["environment_resampling_and_cost"][
            "seconds_per_query_four_actions_five_repeats"
        ]
        for robot in ROBOTS
    }
    source_bytes = {
        robot: sum(
            (
                Path("outputs/counterfactual_v4_pilot")
                / robot
                / f"seed{SEED}"
                / name
            ).stat().st_size
            for name in ("counterfactual_records.jsonl.gz", "counterfactual_labels.npz")
        )
        for robot in ROBOTS
    }
    # The two robots use separate models with an identical model/label contract.
    # Counts below are the aggregate evidence budget, split evenly by robot.
    combined_split = {"train": 30_000, "calibration": 5_000, "policy_validation": 5_000}
    per_robot_total = sum(combined_split.values()) // len(ROBOTS)
    four_action_hours = sum(rates[robot] * per_robot_total for robot in ROBOTS) / 3600
    planned_rates: dict[str, float] = {}
    for robot in ROBOTS:
        fixed_seconds = results[f"{robot}/seed{SEED}"]["timing_repeat_quality"][
            "total_measured_action_seconds"
        ]["fixed_robust"]
        query_count = results[f"{robot}/seed{SEED}"]["schema_and_uniqueness"][
            "query_count"
        ]
        planned_rates[robot] = rates[robot] - fixed_seconds / query_count
    planned_hours = sum(
        planned_rates[robot] * per_robot_total for robot in ROBOTS
    ) / 3600
    per_robot_split_hours = sum(
        planned_rates[robot] * 40_000 for robot in ROBOTS
    ) / 3600
    per_robot_split_four_action_hours = sum(
        rates[robot] * 40_000 for robot in ROBOTS
    ) / 3600
    observed_queries = 2_000
    combined_storage = sum(
        source_bytes[robot] * per_robot_total / observed_queries for robot in ROBOTS
    )
    return {
        "recommended_seed17_two_model_scale_total_across_both_robots": combined_split,
        "model_scope": (
            "one independent nine-feature model for Panda and one for UR5e; "
            "weights are not pooled, while architecture and label contract are shared"
        ),
        "recommended_per_robot_allocation": {
            "panda": {"train": 15_000, "calibration": 2_500, "policy_validation": 2_500},
            "ur5e": {"train": 15_000, "calibration": 2_500, "policy_validation": 2_500},
        },
        "source_roles": {
            "train": "risk_train_queries",
            "calibration": "calibration_queries",
            "policy_validation": "policy_validation_queries",
        },
        "total_queries": 40_000,
        "timed_solver_executions_at_three_decision_actions_and_five_repeats": 600_000,
        "fixed_robust_alias_query_action_records": 40_000,
        "fixed_robust_alias_repeat_samples": 200_000,
        "projected_planned_sequential_hours_including_observed_quiet_waits": planned_hours,
        "four_executed_action_pilot_scaling_upper_bound_hours": four_action_hours,
        "projected_raw_plus_npz_storage_mib": combined_storage / (1024**2),
        "alternative_if_30k_5k_5k_is_interpreted_per_robot": {
            "total_queries": 80_000,
            "projected_planned_sequential_hours_including_observed_quiet_waits": per_robot_split_hours,
            "four_executed_action_upper_bound_hours": per_robot_split_four_action_hours,
            "projected_raw_plus_npz_storage_mib": 2 * combined_storage / (1024**2),
        },
        "scale_rationale": (
            "A compact per-robot MLP does not justify the original 160k labels per "
            "robot. The balanced 40k aggregate design is a validation-stage evidence "
            "budget; it is not selected from test_v3 performance. Fixed robust is a "
            "semantic alias of easy and is not re-executed in bulk."
        ),
    }


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_root = workspace / "outputs/counterfactual_v4_pilot"
    run_manifest = strict_json(output_root / "run_manifest.json")
    aggregate = strict_json(output_root / "pilot_aggregate_summary.json")
    environment = strict_json(output_root / "environment.json")
    results: dict[str, Any] = {}
    for robot in ROBOTS:
        key = f"{robot}/seed{SEED}"
        results[key] = audit_combination(
            workspace, output_root, run_manifest, robot, SEED
        )

    integrity_pass = all(result["integrity_pass"] for result in results.values())
    aggregate_matches = True
    for key in results:
        stored = strict_json(
            output_root / key.split("/")[0] / key.split("/")[1] / "pilot_summary.json"
        )
        if aggregate[key] != stored:
            aggregate_matches = False
    test_discipline = {
        "source_role": run_manifest["source_role"],
        "source_role_is_validation_only": run_manifest["source_role"]
        == "risk_validation_queries",
        "test_data_loaded_manifest_flag": run_manifest["test_data_loaded"],
        "test_v3_used_for_parameter_selection_manifest_flag": run_manifest[
            "test_v3_used_for_parameter_selection"
        ],
        "all_selection_manifests_use_risk_validation_queries": all(
            strict_json(
                output_root / robot / f"seed{SEED}" / "selection_manifest.json"
            )["source_role"]
            == "risk_validation_queries"
            for robot in ROBOTS
        ),
        "note": (
            "This is a provenance audit of the recorded inputs and code path, not a "
            "forensic proof about every operating-system read. No test_v3 metric is "
            "read or used by this audit or by the recorded selection manifests."
        ),
    }
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "audited_output": str(output_root),
        "audit_scope": "validation-only label pilot; not formal test evidence",
        "run_manifest_protocol": run_manifest["protocol"],
        "pilot_only": run_manifest["pilot_only"],
        "eligible_for_formal_claims": run_manifest["eligible_for_formal_claims"],
        "protected_outputs_unchanged": run_manifest["protected_outputs_unchanged"],
        "aggregate_summary_matches_combination_summaries": aggregate_matches,
        "environment": environment,
        "test_data_discipline": test_discipline,
        "combinations": results,
        "scale_projection": scale_projection(results),
        "exit_decision": {
            "artifact_integrity": "PASS" if integrity_pass else "FAIL",
            "bulk_exit_status": "GO_WITH_CONDITIONS" if integrity_pass else "NO_GO",
            "unconditional_go": False,
            "conditions_before_or_during_bulk_collection": [
                (
                    "Do not train an oracle-winner classifier from five-sample empirical "
                    "P95 labels. Train P50/P95 heads with pinball loss on all five raw "
                    "latency observations, grouped by query."
                ),
                (
                    "Treat the observed zero action-success disagreement as a structural "
                    "null. Either use a shared feasibility/fail-all head, or revise and "
                    "re-pilot the action contract before claiming action-specific success "
                    "prediction."
                ),
                (
                    "Stratify or enrich contract-feasible fail-all / ambiguous-boundary "
                    "queries. The random pilot fail-all class is overwhelmingly large-step "
                    "or unreachable and is too easy for a strong command-reject claim."
                ),
                (
                    "Use independent Panda and UR5e models under the same frozen model "
                    "and label contract, and report calibration separately for each "
                    "robot. A robot indicator is unnecessary because weights are not "
                    "pooled."
                ),
                (
                    "Keep validation roles disjoint, preserve five raw repeats and balanced "
                    "action order, keep the fixed-robust audit arm, and continue discarding "
                    "and recollecting host-contaminated queries."
                ),
                (
                    "Hash the collector and every transitive runtime/config dependency in "
                    "the bulk manifest; record discarded-query indices and reasons."
                ),
            ],
            "interpretation": (
                "The pilot is trustworthy enough to justify a reduced seed17 two-model "
                "training/calibration/validation collection with a shared contract. It "
                "is not sufficient to "
                "support formal v4 claims, action-specific success-head claims, or hard "
                "per-query P95-winner labels."
            ),
        },
        "artifact_audit_pass": bool(integrity_pass and aggregate_matches),
    }
    output = args.output
    if not output.is_absolute():
        output = workspace / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_audit_pass": payload["artifact_audit_pass"],
                "bulk_exit_status": payload["exit_decision"]["bulk_exit_status"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
