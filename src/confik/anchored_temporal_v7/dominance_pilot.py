"""One-shot development holdout for Anchored Temporal V7 dominance.

The original V7 ``calibration_no_go`` is immutable and remains authoritative
under its exact-vector protocol.  This runner performs a new, explicitly
amended development protocol: it re-selects R from the already frozen
calibration records under trajectory-set dominance, seals that choice, and
only then opens and evaluates the previously unused policy-validation role.

There is intentionally no smoke mode: even a partial solver run on the sealed
policy-validation role would consume the one-shot holdout.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..experiments.provenance import environment_payload
from ..hierarchical_v5.pilot import (
    _artifact,
    _sha256_file,
    _verified_release_inputs,
    _write_json,
)
from ..hierarchical_v5_lite.pilot import (
    _snapshot_digest,
    _tree_snapshot,
)
from .dominance_selection import (
    R_VALUES,
    NoDominanceEligibleCandidate,
    load_frozen_v7_calibration_records,
    select_dominance_reanchor_interval,
)
from .pilot import (
    FROZEN_H,
    METHODS,
    OCCUPANCY_CODE,
    POLICY_VALIDATION_ROLE,
    TrajectoryView,
    _build_policy_validation_methods,
    _save_benchmark,
    _verify_v6_baseline,
    _view,
    _warm_methods,
    benchmark_policy_validation,
)
from .trajectories import (
    CALIBRATION_ROLE,
    FAMILIES,
    load_trajectory_role,
)


PROTOCOL = "anchored_temporal_v7_dominance_development_holdout_v1"
FROZEN_V7_PROTOCOL = "anchored_temporal_v7_development_pilot_v1"
OUTPUT_NAME = "anchored_temporal_v7_dominance_pilot"
TRAJECTORIES = 40
FRAMES_PER_TRAJECTORY = 150
SELECTION_OBJECTIVES = (
    "maximum_whole_trajectory_completion_count",
    "minimum_total_latency_sum_over_all_fixed_150_frame_trajectories",
    "minimum_frame_level_p99_latency",
    "minimum_learned_seed_invocation_rate",
    "minimum_mean_fev",
    "minimum_local_commitment_horizon",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _forbid_formal_path(value: str | Path, *, name: str) -> None:
    lowered = str(value).casefold()
    if "test_v3" in lowered or "test_v4" in lowered:
        raise ValueError(f"{name} must not reference formal test data: {value}")


def _tree_descriptors(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): _artifact(path, relative_to=root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(_safe_json(row) for row in values)


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create and fsync the immutable selection seal exactly once."""

    encoded = (
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def validate_config(config: Mapping[str, Any], *, workspace: Path) -> None:
    """Fail closed on every preregistered dominance protocol control."""

    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected V7-Dominance protocol")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("V7-Dominance requires Panda and UR5e")
    if dict(config.get("roles", {})) != {
        "calibration": CALIBRATION_ROLE,
        "policy_validation": POLICY_VALIDATION_ROLE,
    }:
        raise ValueError("development roles changed")
    amendment = config.get("protocol_amendment", {})
    if (
        amendment.get("prior_protocol") != FROZEN_V7_PROTOCOL
        or amendment.get("prior_result_required_status") != "calibration_no_go"
        or amendment.get("prior_result_read_only") is not True
        or amendment.get("prior_result_preserved_without_supersession") is not True
        or amendment.get("change_scope") != "calibration_selection_rule_only"
        or amendment.get("change_recorded_before_policy_validation_outcomes_opened")
        is not True
        or amendment.get("development_holdout_only") is not True
        or amendment.get("formal_test") is not False
    ):
        raise ValueError("protocol-amendment declaration changed")
    frozen = config.get("frozen_inputs", {})
    if (
        frozen.get("calibration_source") != "frozen_v7_calibration_records"
        or int(frozen.get("calibration_solver_calls", -1)) != 0
        or int(frozen.get("trajectory_generator_calls", -1)) != 0
        or frozen.get("require_v7_manifest_policy_validation_outcomes_computed")
        is not False
        or int(
            frozen.get("require_v7_manifest_policy_validation_run_count_per_robot", -1)
        )
        != 0
        or frozen.get("verify_manifest_artifact_hashes_before_read") is not True
    ):
        raise ValueError("frozen input contract changed")
    required_hashes = dict(frozen.get("required_sha256", {}))
    expected_hash_names = {
        "run_manifest.json",
        "trajectory_split_manifest.json",
        "panda_calibration_records.npz",
        "ur5e_calibration_records.npz",
        "panda_trajectory_policy_validation.npz",
        "ur5e_trajectory_policy_validation.npz",
    }
    if set(required_hashes) != expected_hash_names or any(
        not isinstance(value, str) or len(value) != 64
        for value in required_hashes.values()
    ):
        raise ValueError("required frozen V7 hash set changed")
    boundary = config.get("data_boundary", {})
    if (
        tuple(boundary.get("allowed_roles", ()))
        != (CALIBRATION_ROLE, POLICY_VALIDATION_ROLE)
        or any(
            boundary.get(key) is not True
            for key in (
                "calibration_records_only_before_selection_seal",
                "policy_validation_queries_open_after_selection_seal_only",
                "policy_validation_outcomes_open_after_selection_seal_only",
                "reuse_existing_fresh_v7_policy_validation_role",
                "formal_test_data_forbidden",
                "formal_test_source_forbidden",
                "reject_test_v3_v4_paths",
            )
        )
        or boundary.get("policy_validation_used_for_selection") is not False
        or int(boundary.get("policy_validation_run_count_per_robot", -1)) != 1
        or boundary.get("split_unit") != "complete_trajectory"
        or int(boundary.get("fixed_trajectories_per_robot", -1)) != TRAJECTORIES
        or int(boundary.get("fixed_frames_per_trajectory", -1))
        != FRAMES_PER_TRAJECTORY
        or boundary.get("regenerate_trajectories") is not False
    ):
        raise ValueError("one-shot development boundary changed")
    if dict(config.get("frozen_temporal_v6_hold_frames", {})) != FROZEN_H:
        raise ValueError("frozen V6 hold frames changed")
    anchor = config.get("anchor_policy", {})
    if (
        tuple(anchor.get("local_commitment_horizon", ())) != R_VALUES
        or anchor.get("runtime_implementation")
        != "unchanged_anchored_temporal_v7"
        or anchor.get("new_networks") is not False
        or anchor.get("new_temporal_parameters") is not False
    ):
        raise ValueError("V7 runtime/R grid contract changed")
    selection = config.get("dominance_selection", {})
    if (
        selection.get("required_role") != CALIBRATION_ROLE
        or selection.get("select_separately_per_robot") is not True
        or selection.get("eligibility")
        != "always_hard_completed_trajectory_uid_set_subset_of_anchor"
        or dict(selection.get("eligibility_requirements", {}))
        != {
            "lost_trajectory_uids_empty": True,
            "gained_trajectory_uids_allowed": True,
            "completion_count_not_below_always_hard": True,
        }
        or tuple(selection.get("objective_order", ())) != SELECTION_OBJECTIVES
        or selection.get("cumulative_latency_definition")
        != "sum_total_latency_ns_per_trajectory_over_exactly_150_frames"
        or selection.get("cumulative_latency_aggregate")
        != "sum_over_all_40_fixed_trajectories"
        or tuple(selection.get("cumulative_latency_report_statistics", ()))
        != ("mean", "median", "p95")
        or selection.get("policy_validation_used_for_selection") is not False
    ):
        raise ValueError("dominance selection contract changed")
    seal = config.get("selection_seal", {})
    if not all(value is True for value in seal.values()) or set(seal) != {
        "write_before_policy_validation_open",
        "immutable",
        "include_selected_r_per_robot",
        "include_selection_rule",
        "include_selection_input_file_sha256",
        "include_selection_output_file_sha256",
        "verify_before_policy_validation_open",
    }:
        raise ValueError("selection-seal contract changed")
    if tuple(config.get("strategies", ())) != METHODS:
        raise ValueError(f"method registry must remain exactly {METHODS}")
    timing = config.get("timing", {})
    if (
        timing.get("clock") != "perf_counter_ns"
        or int(timing.get("warmup_frames", -1)) != 200
        or int(timing.get("trajectory_repeats", -1)) != 1
        or timing.get("method_order") != "seeded_four_by_four_latin_blocks"
        or timing.get("disk_io_inside_timed_interval") is not False
        or timing.get("logging_serialization_inside_timed_interval") is not False
    ):
        raise ValueError("timing contract changed")
    runtime = config.get("runtime", {})
    if set(runtime) != {
        "device",
        "intra_op_threads",
        "inter_op_threads",
        "deterministic_algorithms",
        "progress_every_trajectories",
    }:
        raise ValueError("undeclared runtime control")
    goals = config.get("policy_validation_gate", {})
    if (
        goals.get("always_hard_completed_trajectory_uid_set_subset_of_anchor")
        is not True
        or goals.get("anchored_completion_count_not_below_always_hard") is not True
        or float(goals.get("aggregate_cumulative_latency_ratio_vs_always_hard_max", np.nan))
        != 0.80
        or float(goals.get("learned_seed_invocation_rate_max", np.nan)) != 0.50
        or float(goals.get("p99_ratio_vs_always_hard_max", np.nan)) != 1.05
        or float(goals.get("mean_fev_ratio_vs_always_hard_max", np.nan)) != 1.05
    ):
        raise ValueError("policy-validation gate changed")
    exclusions = config.get("scope_exclusions", {})
    if set(exclusions) != {
        "ood",
        "external_solver",
        "collision",
        "real_robot",
        "new_network",
        "new_temporal_parameter",
        "formal_test",
    } or any(value is not True for value in exclusions.values()):
        raise ValueError("scope exclusions changed")
    for key in (
        "source_config",
        "release_v3_root",
        "release_v4_root",
        "temporal_v6_root",
        "frozen_v7_root",
        "output_root",
    ):
        _forbid_formal_path(config.get(key, ""), name=key)
    expected_old = (workspace / "outputs" / "anchored_temporal_v7_pilot").resolve()
    expected_new = (workspace / "outputs" / OUTPUT_NAME).resolve()
    expected_paths = {
        "source_config": (workspace / "configs" / "paper_v2.yaml").resolve(),
        "release_v3_root": (workspace / "outputs" / "release_v3_locked").resolve(),
        "release_v4_root": (workspace / "outputs" / "release_v4_locked").resolve(),
        "temporal_v6_root": (
            workspace / "outputs" / "temporal_event_v6_pilot"
        ).resolve(),
    }
    for key, expected_path in expected_paths.items():
        if resolve_path(dict(config), str(config[key])) != expected_path:
            raise ValueError(f"{key} must resolve exactly to {expected_path}")
    if resolve_path(dict(config), str(config["frozen_v7_root"])) != expected_old:
        raise ValueError("frozen_v7_root resolves outside the immutable V7 result")
    if resolve_path(dict(config), str(config["output_root"])) != expected_new:
        raise ValueError("output_root changed")


def _verify_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    relative = Path(str(descriptor.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe artifact descriptor path: {relative}")
    expected = (root / relative).resolve()
    if expected != path.resolve() or not expected.is_relative_to(root.resolve()):
        raise RuntimeError(f"artifact descriptor path mismatch: {path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"artifact missing or unsafe: {path}")
    actual = _artifact(path, relative_to=root)
    if actual != dict(descriptor):
        raise RuntimeError(f"artifact identity mismatch: {path}")
    return actual


def verify_frozen_v7(
    *,
    workspace: Path,
    root: Path,
    required_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete old no-go tree without parsing its PV NPZ."""

    expected = (workspace / "outputs" / "anchored_temporal_v7_pilot").resolve()
    if root.resolve() != expected or not root.is_dir() or root.is_symlink():
        raise RuntimeError("frozen V7 root is missing, unsafe, or unexpected")
    manifest_path = root / "run_manifest.json"
    if _sha256_file(manifest_path) != required_sha256["run_manifest.json"]:
        raise RuntimeError("frozen V7 manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != FROZEN_V7_PROTOCOL
        or manifest.get("status") != "calibration_no_go"
        or manifest.get("policy_validation_outcomes_computed") is not False
        or int(manifest.get("policy_validation_run_count_per_robot", -1)) != 0
        or manifest.get("policy_validation_used_for_selection") is not False
        or manifest.get("formal_test_data_opened") is not False
        or manifest.get("formal_test_source_opened") is not False
        or int(manifest.get("test_v3_test_v4_files_opened", -1)) != 0
        or manifest.get("protected_trees_unchanged") is not True
        or dict(manifest.get("selected_reanchor_interval", {}))
        != {"panda": None, "ur5e": 20}
        or list(manifest.get("calibration_no_go", {}).get("robots_without_eligible_r", ()))
        != ["panda"]
    ):
        raise RuntimeError("frozen V7 no-go contract changed")
    artifacts = dict(manifest.get("artifacts", {}))
    if not artifacts:
        raise RuntimeError("frozen V7 manifest has no artifacts")
    for name, descriptor in artifacts.items():
        _verify_descriptor(root / name, descriptor, root=root)
    physical = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if physical != set(artifacts) | {"run_manifest.json"}:
        raise RuntimeError("frozen V7 output tree contains missing or extra files")
    for name, expected_hash in required_sha256.items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"required frozen V7 input changed: {name}")
    sources = dict(manifest.get("implementation_sources", {}))
    if not sources:
        raise RuntimeError("frozen V7 implementation provenance is empty")
    for relative, descriptor in sources.items():
        _verify_descriptor(workspace / relative, descriptor, root=workspace)
    evidence = {
        "root": str(root.relative_to(workspace)),
        "run_manifest": _artifact(manifest_path, relative_to=workspace),
        "protocol": manifest["protocol"],
        "status": manifest["status"],
        "old_exact_vector_no_go_preserved": True,
        "old_policy_validation_outcomes_computed": False,
        "old_policy_validation_run_count_per_robot": 0,
        "output_artifact_count": len(artifacts),
        "all_output_artifacts_verified": True,
        "all_implementation_sources_verified": True,
        "policy_validation_npz_parsed_during_verification": False,
        "policy_validation_file_bytes_hashed_for_identity": True,
    }
    return manifest, evidence


def _trajectory_runs(mask: np.ndarray) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    runs: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def trajectory_metric_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, method in enumerate(METHODS):
        for uid, indices in data.role.groups():
            latency = data.latency_ns[indices, column].astype(np.int64)
            occupancy = data.occupancy_mode[indices, column]
            is_stateful = method in (
                "temporal_event_cghik_v6",
                "anchored_temporal_cghik_v7",
            )
            if is_stateful and not np.all(
                np.isin(occupancy, tuple(OCCUPANCY_CODE.values()))
            ):
                raise RuntimeError("stateful occupancy contains an invalid code")
            robust_runs = (
                _trajectory_runs(occupancy == OCCUPANCY_CODE["robust"])
                if is_stateful
                else []
            )
            family = str(data.role.dataset.category[int(indices[0])])
            periodic = data.anchor_kind[indices, column] == "periodic"
            rows.append(
                {
                    "robot": data.robot,
                    "method": method,
                    "trajectory_uid": uid,
                    "family": family,
                    "frame_count": int(len(indices)),
                    "whole_trajectory_completed": bool(
                        np.all(data.accepted[indices, column])
                    ),
                    "cumulative_latency_ns": int(np.sum(latency, dtype=np.int64)),
                    "cumulative_latency_ms": float(
                        np.sum(latency, dtype=np.int64) / 1e6
                    ),
                    "p50_frame_latency_ms": float(np.quantile(latency, 0.50) / 1e6),
                    "p95_frame_latency_ms": float(np.quantile(latency, 0.95) / 1e6),
                    "p99_frame_latency_ms": float(np.quantile(latency, 0.99) / 1e6),
                    "mean_fev": float(
                        np.mean(data.function_evaluations[indices, column])
                    ),
                    "learned_seed_invocation_count": int(
                        np.sum(data.seed_invoked[indices, column])
                    ),
                    "learned_seed_invocation_rate": float(
                        np.mean(data.seed_invoked[indices, column])
                    ),
                    "anchor_frame_count": (
                        int(np.sum(occupancy == OCCUPANCY_CODE["anchor"]))
                        if is_stateful
                        else None
                    ),
                    "periodic_reanchor_count": (
                        int(np.sum(periodic)) if is_stateful else None
                    ),
                    "local_frame_count": (
                        int(np.sum(occupancy == OCCUPANCY_CODE["local"]))
                        if is_stateful
                        else None
                    ),
                    "robust_frame_count": (
                        int(np.sum(occupancy == OCCUPANCY_CODE["robust"]))
                        if is_stateful
                        else None
                    ),
                    "longest_robust_run": max(robust_runs, default=0)
                    if is_stateful
                    else None,
                }
            )
    return rows


def summarize_holdout(
    data: Any, trajectory_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, method in enumerate(METHODS):
        method_trajectories = [
            row for row in trajectory_rows if row["method"] == method
        ]
        completed = [
            str(row["trajectory_uid"])
            for row in method_trajectories
            if bool(row["whole_trajectory_completed"])
        ]
        cumulative = np.asarray(
            [int(row["cumulative_latency_ns"]) for row in method_trajectories],
            dtype=np.int64,
        )
        latency = data.latency_ns[:, column].astype(np.int64)
        is_stateful = method in (
            "temporal_event_cghik_v6",
            "anchored_temporal_cghik_v7",
        )
        occupancy = data.occupancy_mode[:, column]
        periodic_counts = [
            int(row["periodic_reanchor_count"])
            for row in method_trajectories
            if row["periodic_reanchor_count"] is not None
        ]
        longest = [
            int(row["longest_robust_run"])
            for row in method_trajectories
            if row["longest_robust_run"] is not None
        ]
        if is_stateful and not np.all(
            np.isin(occupancy, tuple(OCCUPANCY_CODE.values()))
        ):
            raise RuntimeError("stateful occupancy contains an invalid code")
        rows.append(
            {
                "robot": data.robot,
                "method": method,
                "trajectory_count": TRAJECTORIES,
                "frames_per_trajectory": FRAMES_PER_TRAJECTORY,
                "completion_trajectory_uids": completed,
                "whole_trajectory_completion_count": len(completed),
                "whole_trajectory_completion_rate": len(completed) / TRAJECTORIES,
                "frame_verified_success": float(np.mean(data.accepted[:, column])),
                "total_latency_over_all_frames_ns": int(
                    np.sum(latency, dtype=np.int64)
                ),
                "total_latency_over_all_frames_seconds": float(
                    np.sum(latency, dtype=np.int64) / 1e9
                ),
                "trajectory_cumulative_latency_mean_ms": float(
                    np.mean(cumulative) / 1e6
                ),
                "trajectory_cumulative_latency_median_ms": float(
                    np.median(cumulative) / 1e6
                ),
                "trajectory_cumulative_latency_p95_ms": float(
                    np.quantile(cumulative, 0.95) / 1e6
                ),
                "p50_frame_latency_ms": float(np.quantile(latency, 0.50) / 1e6),
                "p95_frame_latency_ms": float(np.quantile(latency, 0.95) / 1e6),
                "p99_frame_latency_ms": float(np.quantile(latency, 0.99) / 1e6),
                "mean_fev": float(
                    np.mean(data.function_evaluations[:, column])
                ),
                "learned_seed_invocation_rate": float(
                    np.mean(data.seed_invoked[:, column])
                ),
                "anchor_occupancy": (
                    float(np.mean(occupancy == OCCUPANCY_CODE["anchor"]))
                    if is_stateful
                    else None
                ),
                "local_occupancy": (
                    float(np.mean(occupancy == OCCUPANCY_CODE["local"]))
                    if is_stateful
                    else None
                ),
                "robust_occupancy": (
                    float(np.mean(occupancy == OCCUPANCY_CODE["robust"]))
                    if is_stateful
                    else None
                ),
                "periodic_reanchor_count_total": (
                    int(sum(periodic_counts)) if is_stateful else None
                ),
                "periodic_reanchor_count_per_trajectory_mean": (
                    float(np.mean(periodic_counts)) if is_stateful else None
                ),
                "periodic_reanchor_count_per_trajectory_median": (
                    float(np.median(periodic_counts)) if is_stateful else None
                ),
                "periodic_reanchor_count_per_trajectory_p95": (
                    float(np.quantile(periodic_counts, 0.95)) if is_stateful else None
                ),
                "longest_robust_run": max(longest, default=0)
                if is_stateful
                else None,
            }
        )
    return rows


def completion_identity(
    robot: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    methods = {str(row["method"]): row for row in rows}
    hard_set = set(methods["always_hard"]["completion_trajectory_uids"])
    anchor_set = set(
        methods["anchored_temporal_cghik_v7"]["completion_trajectory_uids"]
    )
    order = list(methods["always_hard"]["completion_trajectory_uids"])
    del order
    role_order = sorted(hard_set | anchor_set)
    return {
        "robot": robot,
        "dominance_definition": "S_hard is a subset of S_anchor",
        "always_hard_completion_trajectory_uids": sorted(hard_set),
        "anchored_completion_trajectory_uids": sorted(anchor_set),
        "lost_trajectory_uids": sorted(hard_set - anchor_set),
        "gained_trajectory_uids": sorted(anchor_set - hard_set),
        "lost_trajectory_count": len(hard_set - anchor_set),
        "gained_trajectory_count": len(anchor_set - hard_set),
        "always_hard_completion_count": len(hard_set),
        "anchored_completion_count": len(anchor_set),
        "always_hard_set_subset_of_anchor": hard_set.issubset(anchor_set),
        "union_uid_count": len(role_order),
    }


def family_rows(
    data: Any, trajectory_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        column = METHODS.index(method)
        for family in FAMILIES:
            selected = [
                row
                for row in trajectory_rows
                if row["method"] == method and row["family"] == family
            ]
            if len(selected) != 10:
                raise RuntimeError("each method/family must contain ten trajectories")
            mask = data.role.dataset.category == family
            cumulative = np.asarray(
                [int(row["cumulative_latency_ns"]) for row in selected],
                dtype=np.int64,
            )
            completed = [
                str(row["trajectory_uid"])
                for row in selected
                if bool(row["whole_trajectory_completed"])
            ]
            occupancy = data.occupancy_mode[mask, column]
            is_stateful = method in (
                "temporal_event_cghik_v6",
                "anchored_temporal_cghik_v7",
            )
            output.append(
                {
                    "robot": data.robot,
                    "method": method,
                    "family": family,
                    "trajectory_count": len(selected),
                    "completion_trajectory_uids": completed,
                    "whole_trajectory_completion_count": len(completed),
                    "whole_trajectory_completion_rate": len(completed) / len(selected),
                    "total_cumulative_latency_ns": int(
                        np.sum(cumulative, dtype=np.int64)
                    ),
                    "total_cumulative_latency_seconds": float(
                        np.sum(cumulative, dtype=np.int64) / 1e9
                    ),
                    "trajectory_cumulative_latency_mean_ms": float(
                        np.mean(cumulative) / 1e6
                    ),
                    "trajectory_cumulative_latency_median_ms": float(
                        np.median(cumulative) / 1e6
                    ),
                    "trajectory_cumulative_latency_p95_ms": float(
                        np.quantile(cumulative, 0.95) / 1e6
                    ),
                    "frame_verified_success": float(
                        np.mean(data.accepted[mask, column])
                    ),
                    "mean_fev": float(
                        np.mean(data.function_evaluations[mask, column])
                    ),
                    "learned_seed_invocation_rate": float(
                        np.mean(data.seed_invoked[mask, column])
                    ),
                    "anchor_occupancy": (
                        float(np.mean(occupancy == OCCUPANCY_CODE["anchor"]))
                        if is_stateful
                        else None
                    ),
                    "local_occupancy": (
                        float(np.mean(occupancy == OCCUPANCY_CODE["local"]))
                        if is_stateful
                        else None
                    ),
                    "robust_occupancy": (
                        float(np.mean(occupancy == OCCUPANCY_CODE["robust"]))
                        if is_stateful
                        else None
                    ),
                }
            )
    return output


def dominance_gate(
    main_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    robots: dict[str, Any] = {}
    for robot in ("panda", "ur5e"):
        methods = {
            str(row["method"]): row
            for row in main_rows
            if row["robot"] == robot
        }
        hard = methods["always_hard"]
        anchor = methods["anchored_temporal_cghik_v7"]
        hard_set = set(hard["completion_trajectory_uids"])
        anchor_set = set(anchor["completion_trajectory_uids"])
        cumulative_ratio = float(anchor["total_latency_over_all_frames_ns"]) / float(
            hard["total_latency_over_all_frames_ns"]
        )
        p99_ratio = float(anchor["p99_frame_latency_ms"]) / float(
            hard["p99_frame_latency_ms"]
        )
        fev_ratio = float(anchor["mean_fev"]) / float(hard["mean_fev"])
        checks = {
            "always_hard_completion_set_subset_of_anchor": hard_set.issubset(
                anchor_set
            ),
            "anchored_completion_count_not_below_always_hard": int(
                anchor["whole_trajectory_completion_count"]
            )
            >= int(hard["whole_trajectory_completion_count"]),
            "aggregate_cumulative_latency_ratio_at_most_0_80": cumulative_ratio
            <= 0.80,
            "learned_seed_invocation_rate_at_most_0_50": float(
                anchor["learned_seed_invocation_rate"]
            )
            <= 0.50,
            "p99_ratio_at_most_1_05": p99_ratio <= 1.05,
            "mean_fev_ratio_at_most_1_05": fev_ratio <= 1.05,
        }
        robots[robot] = {
            "pass": all(checks.values()),
            "checks": checks,
            "lost_trajectory_uids": sorted(hard_set - anchor_set),
            "gained_trajectory_uids": sorted(anchor_set - hard_set),
            "aggregate_cumulative_latency_ratio_vs_always_hard": cumulative_ratio,
            "aggregate_cumulative_latency_reduction_fraction": 1.0
            - cumulative_ratio,
            "p50_ratio_vs_always_hard": float(anchor["p50_frame_latency_ms"])
            / float(hard["p50_frame_latency_ms"]),
            "p95_ratio_vs_always_hard": float(anchor["p95_frame_latency_ms"])
            / float(hard["p95_frame_latency_ms"]),
            "p99_ratio_vs_always_hard": p99_ratio,
            "mean_fev_ratio_vs_always_hard": fev_ratio,
        }
    passed = all(value["pass"] for value in robots.values())
    return {
        "status": "pass" if passed else "fail",
        "all_robots_pass": passed,
        "robots": robots,
        "development_holdout_only": True,
        "formal_test": False,
        "fresh_trajectory_evaluation_authorized": passed,
    }


def _main_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Robot | Method | Completion | Total s | P50 ms | P95 ms | P99 ms | Mean FEV | Seed rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {robot} | {method} | {whole_trajectory_completion_count}/40 | "
            "{total_latency_over_all_frames_seconds:.3f} | "
            "{p50_frame_latency_ms:.4f} | {p95_frame_latency_ms:.4f} | "
            "{p99_frame_latency_ms:.4f} | {mean_fev:.3f} | "
            "{learned_seed_invocation_rate:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _verify_selection_seal(
    staging: Path,
    *,
    expected_sha256: str,
    selected_r: Mapping[str, int],
    workspace: Path,
    frozen_v7_root: Path,
    expected_release_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    path = staging / "selection_seal.json"
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("selection seal bytes changed")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if (
        seal.get("protocol") != PROTOCOL
        or dict(seal.get("selected_reanchor_interval", {})) != dict(selected_r)
        or seal.get("dominance_rule") != "S_hard is a subset of S_anchor"
        or tuple(seal.get("objective_order", ())) != SELECTION_OBJECTIVES
        or seal.get("policy_validation_npz_semantically_opened_before_seal") is not False
        or seal.get("policy_validation_outcomes_computed_before_seal") is not False
        or int(seal.get("selection_solver_call_count", -1)) != 0
        or int(
            seal.get("calibration_outcome_collection_solver_call_count", -1)
        )
        != 0
        or int(seal.get("trajectory_generator_call_count", -1)) != 0
    ):
        raise RuntimeError("selection seal semantic contract changed")
    for descriptor in (
        seal["selection_report"],
        seal["candidate_metrics"],
        seal["protocol_config"],
    ):
        _verify_descriptor(
            staging / str(descriptor["path"]), descriptor, root=staging
        )
    _verify_descriptor(
        workspace / str(seal["old_exact_vector_no_go_manifest"]["path"]),
        seal["old_exact_vector_no_go_manifest"],
        root=workspace,
    )
    _verify_descriptor(
        frozen_v7_root / str(seal["old_trajectory_split_manifest"]["path"]),
        seal["old_trajectory_split_manifest"],
        root=frozen_v7_root,
    )
    for group in ("selection_inputs", "sealed_calibration_roles", "sealed_unopened_policy_validation_roles"):
        for descriptor in seal[group].values():
            _verify_descriptor(
                frozen_v7_root / str(descriptor["path"]),
                descriptor,
                root=frozen_v7_root,
            )
    for descriptor in seal["implementation_sources"].values():
        _verify_descriptor(
            workspace / str(descriptor["path"]), descriptor, root=workspace
        )
    if dict(seal.get("verified_release_inputs", {})) != dict(
        expected_release_inputs
    ):
        raise RuntimeError("selection seal release inputs changed")
    selection = json.loads(
        (staging / str(seal["selection_report"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    candidates = json.loads(
        (staging / str(seal["candidate_metrics"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        selection.get("protocol") != PROTOCOL
        or candidates.get("protocol") != PROTOCOL
        or dict(selection.get("selected_reanchor_interval", {}))
        != dict(selected_r)
        or dict(candidates.get("robots", {})) != dict(selection.get("reports", {}))
    ):
        raise RuntimeError("selection seal reports are not semantically bound")
    for robot, value in selected_r.items():
        selected = selection["reports"][robot].get("selected")
        if not isinstance(selected, dict) or int(selected["reanchor_interval"]) != int(
            value
        ):
            raise RuntimeError("selection seal does not bind the selected candidate")
    return seal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    release_v3_root = resolve_path(config, str(config["release_v3_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    temporal_v6_root = resolve_path(config, str(config["temporal_v6_root"]))
    frozen_v7_root = resolve_path(config, str(config["frozen_v7_root"]))
    output_root = resolve_path(config, str(config["output_root"]))
    for name, value in (
        ("source_config", source_config_path),
        ("release_v3_root", release_v3_root),
        ("release_v4_root", release_v4_root),
        ("temporal_v6_root", temporal_v6_root),
        ("frozen_v7_root", frozen_v7_root),
        ("output_root", output_root),
    ):
        _forbid_formal_path(value, name=name)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"overwrite forbidden: {output_root}")
    stale = sorted(output_root.parent.glob(f".{output_root.name}.incomplete.*"))
    if stale:
        raise FileExistsError(
            "stale staging requires inspection: "
            + ", ".join(str(path) for path in stale)
        )
    lock = output_root.with_name(f".{output_root.name}.lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)

    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    try:
        git_commit_start = _git(workspace, "rev-parse", "HEAD")
        git_status_start = _git(workspace, "status", "--short").splitlines()
        if git_status_start:
            raise RuntimeError("V7-Dominance requires a clean committed worktree")
        started_at = _utc()
        required_sha256 = dict(config["frozen_inputs"]["required_sha256"])
        frozen_manifest, frozen_evidence = verify_frozen_v7(
            workspace=workspace,
            root=frozen_v7_root,
            required_sha256=required_sha256,
        )
        v6_evidence, _ = _verify_v6_baseline(
            workspace=workspace, temporal_v6_root=temporal_v6_root
        )
        release_inputs = {
            robot: _verified_release_inputs(
                workspace=workspace,
                release_v3_root=release_v3_root,
                release_v4_root=release_v4_root,
                robot=robot,
            )
            for robot in ("panda", "ur5e")
        }
        protected_roots = {
            "anchored_temporal_v7_pilot": frozen_v7_root,
            "anchored_temporal_v7_smoke": workspace
            / "outputs"
            / "anchored_temporal_v7_smoke",
            "hierarchical_v5_pilot": workspace / "outputs" / "hierarchical_v5_pilot",
            "hierarchical_v5_lite_pilot": workspace
            / "outputs"
            / "hierarchical_v5_lite_pilot",
            "legacy_temporal_v6_pilot": workspace / "outputs" / "temporal_v6_pilot",
            "temporal_event_v6_pilot": temporal_v6_root,
            "release_v3_locked": release_v3_root,
            "release_v4_locked": release_v4_root,
        }
        protected_before = {
            name: _tree_snapshot(path) for name, path in protected_roots.items()
        }
        protected_digest_before = {
            name: _snapshot_digest(snapshot)
            for name, snapshot in protected_before.items()
        }
        implementation_paths = (
            Path(__file__).resolve(),
            Path(__file__).with_name("dominance_selection.py").resolve(),
            Path(__file__).with_name("pilot.py").resolve(),
            Path(__file__).with_name("runtime.py").resolve(),
            Path(__file__).with_name("state.py").resolve(),
            Path(__file__).with_name("trajectories.py").resolve(),
            Path(config["_config_path"]).resolve(),
            source_config_path.resolve(),
            (workspace / "scripts" / "run_anchored_temporal_v7_dominance.sh").resolve(),
            (workspace / "tests" / "test_anchored_temporal_v7_dominance.py").resolve(),
        )
        implementation_before = {
            str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
            for path in implementation_paths
        }
        staging.mkdir(parents=True, exist_ok=False)

        # Phase A: only the SHA-pinned calibration records are parsed.  The PV
        # files have been byte-hashed for identity, but their NPZ contents have
        # not been opened and no solver outcome has been computed.
        selection_reports: dict[str, Any] = {}
        selected_r: dict[str, int] = {}
        for robot in ("panda", "ur5e"):
            name = f"{robot}_calibration_records.npz"
            records = load_frozen_v7_calibration_records(
                frozen_v7_root / name,
                robot=robot,
                expected_artifact=frozen_manifest["artifacts"][name],
            )
            try:
                value, report = select_dominance_reanchor_interval(records)
            except NoDominanceEligibleCandidate:
                raise RuntimeError(
                    f"{robot}: dominance amendment has no eligible R; PV remains unopened"
                )
            selected_r[robot] = value
            selection_reports[robot] = report

        candidate_payload = {
            "protocol": PROTOCOL,
            "selection_role": CALIBRATION_ROLE,
            "selection_solver_call_count": 0,
            "calibration_outcome_collection_solver_call_count": 0,
            "trajectory_generator_call_count": 0,
            "policy_validation_npz_semantically_opened": False,
            "policy_validation_outcomes_computed": False,
            "robots": selection_reports,
        }
        _write_json(staging / "dominance_candidate_metrics.json", candidate_payload)
        selection_payload = {
            "protocol": PROTOCOL,
            "status": "selected_on_frozen_calibration",
            "protocol_rule_changed_before_policy_validation_outcomes_opened": True,
            "prior_exact_vector_no_go_preserved": True,
            "prior_exact_vector_no_go_superseded": False,
            "development_holdout_only": True,
            "formal_test": False,
            "selected_reanchor_interval": selected_r,
            "frozen_temporal_v6_hold_frames": FROZEN_H,
            "dominance_rule": "S_hard is a subset of S_anchor",
            "objective_order": list(
                config["dominance_selection"]["objective_order"]
            ),
            "selection_role": CALIBRATION_ROLE,
            "policy_validation_used_for_selection": False,
            "policy_validation_npz_semantically_opened": False,
            "policy_validation_outcomes_computed": False,
            "selection_solver_call_count": 0,
            "calibration_outcome_collection_solver_call_count": 0,
            "trajectory_generator_call_count": 0,
            "reports": selection_reports,
        }
        _write_json(staging / "dominance_selection.json", selection_payload)
        effective_config = {
            key: value for key, value in config.items() if key != "_config_path"
        }
        (staging / "anchored_temporal_v7_dominance.yaml").write_text(
            yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
        )
        seal_payload = {
            "protocol": PROTOCOL,
            "sealed_at": _utc(),
            "selected_reanchor_interval": selected_r,
            "frozen_temporal_v6_hold_frames": FROZEN_H,
            "dominance_rule": "S_hard is a subset of S_anchor",
            "objective_order": list(
                config["dominance_selection"]["objective_order"]
            ),
            "old_exact_vector_no_go_preserved": True,
            "old_exact_vector_no_go_manifest": frozen_evidence["run_manifest"],
            "old_trajectory_split_manifest": frozen_manifest["artifacts"][
                "trajectory_split_manifest.json"
            ],
            "selection_solver_call_count": 0,
            "calibration_outcome_collection_solver_call_count": 0,
            "warmup_role": CALIBRATION_ROLE,
            "warmup_frames_per_robot": int(config["timing"]["warmup_frames"]),
            "warmup_method_calls_per_robot": int(
                config["timing"]["warmup_frames"]
            )
            * len(METHODS),
            "warmup_outcomes_retained": False,
            "trajectory_generator_call_count": 0,
            "policy_validation_file_bytes_hashed_for_identity_before_seal": True,
            "policy_validation_npz_semantically_opened_before_seal": False,
            "policy_validation_outcomes_computed_before_seal": False,
            "policy_validation_run_count_per_robot_before_seal": 0,
            "selection_inputs": {
                robot: frozen_manifest["artifacts"][
                    f"{robot}_calibration_records.npz"
                ]
                for robot in ("panda", "ur5e")
            },
            "sealed_calibration_roles": {
                robot: frozen_manifest["artifacts"][
                    f"{robot}_trajectory_calibration.npz"
                ]
                for robot in ("panda", "ur5e")
            },
            "sealed_unopened_policy_validation_roles": {
                robot: frozen_manifest["artifacts"][
                    f"{robot}_trajectory_policy_validation.npz"
                ]
                for robot in ("panda", "ur5e")
            },
            "selection_report": _artifact(
                staging / "dominance_selection.json", relative_to=staging
            ),
            "candidate_metrics": _artifact(
                staging / "dominance_candidate_metrics.json", relative_to=staging
            ),
            "protocol_config": _artifact(
                staging / "anchored_temporal_v7_dominance.yaml",
                relative_to=staging,
            ),
            "implementation_sources": implementation_before,
            "verified_release_inputs": release_inputs,
            "git_commit": git_commit_start,
        }
        _write_exclusive_json(staging / "selection_seal.json", seal_payload)
        seal_sha = _sha256_file(staging / "selection_seal.json")
        _verify_selection_seal(
            staging,
            expected_sha256=seal_sha,
            selected_r=selected_r,
            workspace=workspace,
            frozen_v7_root=frozen_v7_root,
            expected_release_inputs=release_inputs,
        )

        torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
        torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
        torch.use_deterministic_algorithms(
            bool(config["runtime"]["deterministic_algorithms"])
        )
        device = str(config["runtime"]["device"])
        warmup = int(config["timing"]["warmup_frames"])
        progress = int(config["runtime"]["progress_every_trajectories"])
        main_rows: list[dict[str, Any]] = []
        all_trajectory_rows: list[dict[str, Any]] = []
        all_family_rows: list[dict[str, Any]] = []
        identities: dict[str, Any] = {}
        access_ledger: dict[str, Any] = {
            "selection_sealed_at": seal_payload["sealed_at"],
            "selection_seal_sha256": seal_sha,
            "policy_validation_npz_semantically_opened_before_seal": False,
            "policy_validation_outcomes_computed_before_seal": False,
            "selection_solver_call_count": 0,
            "calibration_outcome_collection_solver_call_count": 0,
            "warmup_role": CALIBRATION_ROLE,
            "warmup_frames_per_robot": warmup,
            "warmup_method_calls_per_robot": warmup * len(METHODS),
            "warmup_outcomes_retained": False,
            "trajectory_generator_call_count": 0,
            "robots": {},
        }

        # Phase B: this is the first semantic opening of the frozen PV roles,
        # followed by exactly one retained policy-validation pass per robot.
        for robot in ("panda", "ur5e"):
            _verify_selection_seal(
                staging,
                expected_sha256=seal_sha,
                selected_r=selected_r,
                workspace=workspace,
                frozen_v7_root=frozen_v7_root,
                expected_release_inputs=release_inputs,
            )
            current_release = _verified_release_inputs(
                workspace=workspace,
                release_v3_root=release_v3_root,
                release_v4_root=release_v4_root,
                robot=robot,
            )
            if current_release != release_inputs[robot]:
                raise RuntimeError("frozen release inputs changed before holdout")
            current_v6, _ = _verify_v6_baseline(
                workspace=workspace, temporal_v6_root=temporal_v6_root
            )
            if current_v6 != v6_evidence:
                raise RuntimeError("frozen V6 baseline changed before holdout")
            kinematics = load_robot(source_config, robot)
            calibration_name = f"{robot}_trajectory_calibration.npz"
            calibration_role = load_trajectory_role(
                frozen_v7_root / calibration_name,
                robot=robot,
                expected_role=CALIBRATION_ROLE,
                expected_artifact=frozen_manifest["artifacts"][calibration_name],
                kinematics=kinematics,
            )
            warmup_view: TrajectoryView = _view(calibration_role, smoke=False)
            methods = _build_policy_validation_methods(
                source_config=source_config,
                release_v3_root=release_v3_root,
                release_v4_root=release_v4_root,
                robot=robot,
                kinematics=kinematics,
                device=device,
                hold_frames=FROZEN_H[robot],
                reanchor_interval=selected_r[robot],
            )
            _warm_methods(methods, warmup_view, frames=warmup)

            # Finish all non-PV construction and warmup before irreversibly
            # marking and semantically opening the one-shot holdout.
            policy_name = f"{robot}_trajectory_policy_validation.npz"
            attempt_started_at = _utc()
            _write_exclusive_json(
                staging / f"{robot}_policy_validation_attempt_started.json",
                {
                    "protocol": PROTOCOL,
                    "robot": robot,
                    "attempt_started_at": attempt_started_at,
                    "selection_seal_sha256": seal_sha,
                    "retained_run_ordinal": 1,
                    "policy_validation_npz_semantically_opened_before_this_marker": False,
                    "policy_validation_outcomes_computed_before_this_marker": False,
                },
            )
            policy_opened_at = _utc()
            full_policy_role = load_trajectory_role(
                frozen_v7_root / policy_name,
                robot=robot,
                expected_role=POLICY_VALIDATION_ROLE,
                expected_artifact=frozen_manifest["artifacts"][policy_name],
                kinematics=kinematics,
            )
            if set(full_policy_role.source_query_hash.astype(str).tolist()) & set(
                calibration_role.source_query_hash.astype(str).tolist()
            ) or set(full_policy_role.trajectory_order) & set(
                calibration_role.trajectory_order
            ):
                raise RuntimeError(
                    "sealed calibration and policy-validation roles overlap"
                )
            policy_view: TrajectoryView = _view(full_policy_role, smoke=False)
            benchmark_started_at = _utc()
            result = benchmark_policy_validation(
                policy_view,
                methods=methods,
                warmup_role=warmup_view,
                order_seed=920_000 + (862_711 if robot == "panda" else 862_712),
                warmup_frames=0,
                progress_every=progress,
            )
            benchmark_finished_at = _utc()
            _write_exclusive_json(
                staging / f"{robot}_policy_validation_attempt_completed.json",
                {
                    "protocol": PROTOCOL,
                    "robot": robot,
                    "attempt_started_at": attempt_started_at,
                    "attempt_completed_at": benchmark_finished_at,
                    "selection_seal_sha256": seal_sha,
                    "retained_run_ordinal": 1,
                    "policy_validation_frame_count": policy_view.count,
                    "methods_per_frame": len(METHODS),
                },
            )
            _save_benchmark(
                staging / f"{robot}_policy_validation_records.npz", result
            )
            trajectory_rows = trajectory_metric_rows(result)
            robot_main = summarize_holdout(result, trajectory_rows)
            robot_family = family_rows(result, trajectory_rows)
            main_rows.extend(robot_main)
            all_trajectory_rows.extend(trajectory_rows)
            all_family_rows.extend(robot_family)
            identities[robot] = completion_identity(robot, robot_main)
            access_ledger["robots"][robot] = {
                "policy_validation_role_semantically_opened_at": policy_opened_at,
                "benchmark_started_at": benchmark_started_at,
                "benchmark_finished_at": benchmark_finished_at,
                "policy_validation_retained_run_count": 1,
                "policy_validation_frame_count": policy_view.count,
                "method_calls_per_frame": 1,
                "policy_validation_used_for_retuning": False,
                "method_order_seed": 920_000
                + (862_711 if robot == "panda" else 862_712),
            }
            del methods, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _verify_selection_seal(
            staging,
            expected_sha256=seal_sha,
            selected_r=selected_r,
            workspace=workspace,
            frozen_v7_root=frozen_v7_root,
            expected_release_inputs=release_inputs,
        )
        gate = dominance_gate(main_rows)
        _write_json(staging / "main_table.json", main_rows)
        _write_csv(staging / "main_table.csv", main_rows)
        (staging / "main_table.md").write_text(
            _main_markdown(main_rows), encoding="utf-8"
        )
        _write_json(staging / "trajectory_metrics.json", all_trajectory_rows)
        _write_csv(staging / "trajectory_metrics.csv", all_trajectory_rows)
        _write_json(staging / "family_summary.json", all_family_rows)
        _write_csv(staging / "family_summary.csv", all_family_rows)
        _write_json(staging / "completion_uid_sets.json", identities)
        _write_json(staging / "policy_validation_access_ledger.json", access_ledger)
        _write_json(staging / "pilot_gate.json", gate)
        environment = environment_payload()
        environment.update(
            {
                "development_holdout_only": True,
                "formal_test": False,
                "formal_test_data_opened": False,
                "formal_test_source_opened": False,
                "test_v3_test_v4_files_opened": 0,
                "timing_clock": "perf_counter_ns",
                "calibration_solver_calls": 0,
                "trajectory_generator_calls": 0,
                "policy_validation_run_count_per_robot": 1,
            }
        )
        _write_json(staging / "environment.json", environment)

        protected_after = {
            name: _tree_snapshot(path) for name, path in protected_roots.items()
        }
        protected_digest_after = {
            name: _snapshot_digest(snapshot)
            for name, snapshot in protected_after.items()
        }
        if protected_digest_after != protected_digest_before:
            raise RuntimeError("a protected V5/V6/V7/release tree changed")
        git_commit_end = _git(workspace, "rev-parse", "HEAD")
        git_status_end = _git(workspace, "status", "--short").splitlines()
        implementation_after = {
            str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
            for path in implementation_paths
        }
        if (
            git_commit_end != git_commit_start
            or git_status_end != git_status_start
            or implementation_after != implementation_before
        ):
            raise RuntimeError("git/source state changed during V7-Dominance")
        expected_files = {
            "anchored_temporal_v7_dominance.yaml",
            "dominance_candidate_metrics.json",
            "dominance_selection.json",
            "selection_seal.json",
            "main_table.json",
            "main_table.csv",
            "main_table.md",
            "trajectory_metrics.json",
            "trajectory_metrics.csv",
            "family_summary.json",
            "family_summary.csv",
            "completion_uid_sets.json",
            "policy_validation_access_ledger.json",
            "pilot_gate.json",
            "environment.json",
        }
        for robot in ("panda", "ur5e"):
            expected_files.update(
                {
                    f"{robot}_policy_validation_attempt_started.json",
                    f"{robot}_policy_validation_attempt_completed.json",
                    f"{robot}_policy_validation_records.npz",
                }
            )
        actual_files = {
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise RuntimeError(
                "V7-Dominance output closure changed: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        manifest = {
            "protocol": PROTOCOL,
            "status": "complete_policy_validation_holdout",
            "started_at": started_at,
            "finished_at": _utc(),
            "git_commit_start": git_commit_start,
            "git_commit_end": git_commit_end,
            "git_status_start": git_status_start,
            "git_status_end": git_status_end,
            "implementation_sources": implementation_before,
            "artifacts": _tree_descriptors(staging),
            "protocol_amendment": dict(config["protocol_amendment"]),
            "old_exact_vector_no_go_preserved": True,
            "old_exact_vector_no_go_superseded": False,
            "verified_frozen_v7": frozen_evidence,
            "verified_temporal_v6_baseline": v6_evidence,
            "verified_release_inputs": release_inputs,
            "selected_reanchor_interval": selected_r,
            "frozen_temporal_v6_hold_frames": FROZEN_H,
            "selection_seal_sha256": seal_sha,
            "selection_sealed_before_policy_validation_semantic_open": True,
            "policy_validation_file_bytes_hashed_for_identity_before_seal": True,
            "selection_solver_call_count": 0,
            "calibration_outcome_collection_solver_call_count": 0,
            "warmup_role": CALIBRATION_ROLE,
            "warmup_frames_per_robot": warmup,
            "warmup_method_calls_per_robot": warmup * len(METHODS),
            "warmup_outcomes_retained": False,
            "trajectory_generator_call_count": 0,
            "policy_validation_outcomes_computed_before_seal": False,
            "policy_validation_run_count_per_robot": 1,
            "policy_validation_used_for_selection": False,
            "policy_validation_used_for_retuning": False,
            "policy_validation_trajectory_count_per_robot": TRAJECTORIES,
            "frames_per_trajectory": FRAMES_PER_TRAJECTORY,
            "methods": list(METHODS),
            "formal_test": False,
            "formal_test_data_opened": False,
            "formal_test_source_opened": False,
            "test_v3_test_v4_files_opened": 0,
            "protected_tree_digest_before": protected_digest_before,
            "protected_tree_digest_after": protected_digest_after,
            "protected_trees_unchanged": True,
            "pilot_gate": gate,
        }
        _write_json(staging / "run_manifest.json", manifest)
        os.replace(staging, output_root)
        print(
            f"[anchored-v7-dominance] complete: output={output_root}, "
            f"R={selected_r}, gate={gate['status']}",
            flush=True,
        )
        return manifest
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def main() -> None:
    args = _parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL",
    "completion_identity",
    "dominance_gate",
    "family_rows",
    "run",
    "summarize_holdout",
    "trajectory_metric_rows",
    "validate_config",
]
