#!/usr/bin/env python3
"""Independent, read-only audit of the completed counterfactual-v4 bulk labels.

The implementation deliberately does not import ``bulk_runner``.  It reads no
formal test result: protected test-v3 paths are checked with ``stat`` metadata
only.  The only write is the requested audit JSON below ``docs/audits``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL = "counterfactual_v4_bulk_training_validation_v1"
SCHEMA_VERSION = 3
ROBOTS = ("panda", "ur5e")
SEED = 17
ROLES = ("risk_train_queries", "calibration_queries", "policy_validation_queries")
ROLE_COUNTS = {
    "risk_train_queries": 15_000,
    "calibration_queries": 2_500,
    "policy_validation_queries": 2_500,
}
TRAIN_CATEGORY_COUNTS = {
    "id": 3_500,
    "hard_valid": 2_500,
    "near_singular": 2_500,
    "near_limit": 2_500,
    "workspace_boundary": 2_500,
    "large_step": 750,
    "unreachable": 750,
}
DECISION_ACTIONS = ("easy", "medium", "hard")
ALL_ACTIONS = DECISION_ACTIONS + ("fixed_robust",)
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
SELECTION_KEYS = {
    "source_indices",
    "query_sha256",
    "category",
    "expected_reachable",
    "continuity_feasible",
}
LABEL_KEYS = {
    "feature_names",
    "action_names",
    "decision_action_names",
    "features",
    "query_indices",
    "source_indices",
    "query_sha256",
    "category",
    "expected_reachable",
    "continuity_feasible",
    "verified_success",
    "verified_success_before_deadline",
    "latency_samples_ns",
    "latency_p50_ms",
    "latency_p95_ms",
    "function_evaluations",
    "fallback_used",
    "measurement_is_alias",
    "failure_reason",
    "command_q",
    "max_joint_step_rad",
    "max_joint_velocity_rad_s",
    "max_velocity_limit_utilization",
    "max_joint_acceleration_rad_s2",
    "max_joint_jerk_rad_s3",
    "dynamic_history_available",
}
RECORD_KEYS = {
    "aliased_from_action",
    "category",
    "chunk_name",
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
    "measurement_executed",
    "measurement_mode",
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
CHUNK_MANIFEST_KEYS = {
    "action_summary",
    "artifacts",
    "chunk_payload_sha256",
    "contaminated_attempt_events",
    "contaminated_query_retries",
    "created_utc",
    "decision_action_execution_count",
    "environment_contaminated",
    "fixed_robust_execution_count",
    "fixed_robust_measurement_mode",
    "label_schema_version",
    "post_chunk_busy_processes",
    "protocol",
    "query_count",
    "query_start",
    "query_stop_exclusive",
    "quiet_check_count",
    "quiet_wait_event_count",
    "quiet_wait_events",
    "quiet_wait_seconds",
    "record_count",
    "robot",
    "seconds_per_query",
    "source_role",
    "test_data_loaded",
    "training_seed",
    "wall_time_seconds_excluding_warmup_and_writes",
}
PROTECTED_PATTERNS = (
    "paper_v2_*",
    "paper_v2_aggregate",
    "latency_pilot_v3",
    "release_v3_locked",
    "counterfactual_v4_pilot",
    "counterfactual_v4_smoke*",
    "counterfactual_v4_readiness_smoke*",
)
TEST_METADATA_PATTERNS = ("test_v3_seed*", "test_v3_aggregate")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--bulk-root", type=Path, default=Path("outputs/counterfactual_v4_bulk")
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/audits/counterfactual_v4_bulk/bulk_audit.json"),
    )
    return parser.parse_args()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def strict_line(line: str, path: Path, number: int) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r} in {path}:{number}")

    value = json.loads(line, parse_constant=reject)
    if not isinstance(value, dict):
        raise TypeError(f"record is not an object in {path}:{number}")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            safe(dict(payload)),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def mapping_digest(payload: Mapping[str, Any]) -> str:
    return sha256(canonical_bytes(payload)).hexdigest()


def relative_or_absolute(workspace: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else workspace / candidate


def add_issue(counter: Counter[str], condition: bool, name: str) -> None:
    if not condition:
        counter[name] += 1


def arrays_equal(left: Any, right: Any) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.dtype.kind in "fc" or second.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def query_digest(source: Mapping[str, np.ndarray], index: int, dt: float) -> str:
    digest = sha256()
    for name in ("previous_q", "target_position", "target_rotation"):
        digest.update(
            np.ascontiguousarray(source[name][index], dtype=np.float64).tobytes()
        )
    digest.update(np.asarray([dt], dtype=np.float64).tobytes())
    return digest.hexdigest()


def selection_seed(release_commit: str, robot: str, role: str) -> int:
    material = f"counterfactual_v4_bulk_v1|{release_commit}|{robot}|17|{role}".encode()
    return int.from_bytes(sha256(material).digest()[:8], "big", signed=False)


def replay_selection(
    category: np.ndarray, *, role: str, count: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    if role != "risk_train_queries":
        return np.asarray(
            generator.choice(len(category), size=count, replace=False), dtype=np.int64
        )
    parts: list[np.ndarray] = []
    observed = category.astype(str)
    for name, quota in TRAIN_CATEGORY_COUNTS.items():
        available = np.flatnonzero(observed == name)
        parts.append(generator.choice(available, size=quota, replace=False))
    selected = np.concatenate(parts)
    generator.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def audit_file_manifest(
    workspace: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    mismatches: list[str] = []
    for name, metadata in manifest.items():
        path = relative_or_absolute(workspace, str(name))
        if not path.is_file() or path.is_symlink():
            mismatches.append(f"missing_or_symlink:{name}")
            continue
        if path.stat().st_size != int(metadata["size"]):
            mismatches.append(f"size:{name}")
        if file_sha256(path) != str(metadata["sha256"]):
            mismatches.append(f"sha256:{name}")
    return {"file_count": len(manifest), "mismatches": mismatches, "pass": not mismatches}


def audit_frozen_provenance(
    workspace: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    frozen = manifest["frozen_provenance"]
    add_issue(
        issues,
        mapping_digest(frozen) == manifest["frozen_provenance_sha256"],
        "frozen_provenance_digest",
    )
    config_path = relative_or_absolute(workspace, frozen["config"]["path"])
    source_config = relative_or_absolute(workspace, frozen["source_config"]["path"])
    add_issue(
        issues,
        config_path.is_file() and file_sha256(config_path) == frozen["config"]["sha256"],
        "bulk_config_hash",
    )
    add_issue(
        issues,
        source_config.is_file()
        and file_sha256(source_config) == frozen["source_config"]["sha256"],
        "source_config_hash",
    )
    code = audit_file_manifest(workspace, frozen["code"]["files"])
    add_issue(issues, code["pass"], "code_file_manifest")
    add_issue(
        issues,
        mapping_digest(frozen["code"]["files"]) == frozen["code"]["sha256"],
        "code_manifest_digest",
    )
    current_code = {
        str(path.relative_to(workspace))
        for path in (workspace / "src/confik").rglob("*.py")
        if path.is_file()
    }
    add_issue(
        issues,
        current_code == set(frozen["code"]["files"]),
        "code_file_set",
    )
    release = audit_file_manifest(workspace, frozen["release"]["files"])
    add_issue(issues, release["pass"], "release_file_manifest")
    add_issue(
        issues,
        mapping_digest(frozen["release"]["files"]) == frozen["release"]["sha256"],
        "release_manifest_digest",
    )
    source_dataset_issues: list[str] = []
    for key, metadata in frozen["source_datasets"].items():
        path = relative_or_absolute(workspace, metadata["path"])
        if not path.is_file() or path.is_symlink():
            source_dataset_issues.append(f"missing_or_symlink:{key}")
            continue
        if path.stat().st_size != int(metadata["size"]):
            source_dataset_issues.append(f"size:{key}")
        if file_sha256(path) != metadata["sha256"]:
            source_dataset_issues.append(f"sha256:{key}")
    add_issue(issues, not source_dataset_issues, "source_dataset_files")
    add_issue(
        issues,
        mapping_digest(frozen["source_datasets"])
        == frozen["source_datasets_sha256"],
        "source_datasets_digest",
    )
    alias_results: dict[str, Any] = {}
    for key, evidence in frozen["fixed_robust_alias_evidence"].items():
        result = audit_file_manifest(workspace, evidence["files"])
        add_issue(issues, result["pass"], f"alias_evidence_files:{key}")
        add_issue(
            issues,
            mapping_digest(evidence["files"]) == evidence["files_sha256"],
            f"alias_evidence_digest:{key}",
        )
        add_issue(
            issues,
            all(bool(value) for value in evidence["semantic_checks"].values()),
            f"alias_semantic_checks:{key}",
        )
        alias_results[key] = result
    add_issue(
        issues,
        mapping_digest(frozen["fixed_robust_alias_evidence"])
        == frozen["fixed_robust_alias_evidence_sha256"],
        "all_alias_evidence_digest",
    )
    add_issue(issues, frozen["test_data_loaded"] is False, "test_data_loaded")
    add_issue(issues, tuple(frozen["allowed_roles"]) == ROLES, "allowed_roles")
    return {
        "issue_counts": dict(issues),
        "pass": not issues,
        "code": code,
        "release": release,
        "source_datasets": {
            "file_count": len(frozen["source_datasets"]),
            "mismatches": source_dataset_issues,
            "pass": not source_dataset_issues,
        },
        "alias_evidence": alias_results,
    }


def hashed_snapshot(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in root.glob(pattern) if path.is_dir())
    files: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected directory is a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            stat = path.stat()
            files[str(path.relative_to(root))] = {
                "sha256": file_sha256(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return {
        "directories": [str(path.relative_to(root)) for path in sorted(directories)],
        "file_count": len(files),
        "files": files,
    }


def test_metadata_snapshot(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    """Stat formal-test paths without opening their content."""

    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in root.glob(pattern) if path.is_dir())
    files: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"formal test directory is a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            stat = path.stat(follow_symlinks=False)
            files[str(path.relative_to(root))] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "content_opened": False,
            }
    return {
        "directories": [str(path.relative_to(root)) for path in sorted(directories)],
        "file_count": len(files),
        "files": files,
        "content_opened": False,
    }


def czy_snapshot(workspace: Path) -> dict[str, Any]:
    root = workspace / "czy"
    files: dict[str, Any] = {}
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files[str(path.relative_to(workspace))] = {
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
    return {"files": files, "sha256": mapping_digest(files)}


def audit_protected_snapshot(
    workspace: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = manifest["protected_baseline"]
    outputs = hashed_snapshot(workspace / "outputs", PROTECTED_PATTERNS)
    # This is the only operation on formal test paths and uses stat only.
    tests = test_metadata_snapshot(workspace / "outputs", TEST_METADATA_PATTERNS)
    czy = czy_snapshot(workspace)
    observed = {"outputs": outputs, "test_v3_metadata_only": tests, "czy": czy}
    return {
        "protected_baseline_sha256_matches": mapping_digest(baseline)
        == manifest["protected_baseline_sha256"],
        "current_snapshot_matches_baseline": observed == baseline,
        "test_v3_content_files_opened": 0,
        "test_v3_metadata_content_opened_flags_false": bool(
            tests["content_opened"] is False
            and all(
                entry["content_opened"] is False for entry in tests["files"].values()
            )
        ),
        "protected_output_file_count": outputs["file_count"],
        "test_v3_metadata_file_count": tests["file_count"],
        "czy_file_count": len(czy["files"]),
    }


def expected_chunk_names(count: int) -> list[str]:
    return [
        f"chunk_{start:06d}_{min(start + 250, count) - 1:06d}"
        for start in range(0, count, 250)
    ]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        try:
            return arrays_equal(left, right)
        except (TypeError, ValueError):
            return left == right
    if left is None or right is None:
        return left is right
    return left == right


def audit_chunk(
    *,
    chunk_dir: Path,
    chunk_name: str,
    robot: str,
    role: str,
    start: int,
    stop: int,
    selection: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    dt: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    issues: Counter[str] = Counter()
    manifest_path = chunk_dir / "chunk_manifest.json"
    records_path = chunk_dir / "counterfactual_records.jsonl.gz"
    labels_path = chunk_dir / "counterfactual_labels.npz"
    actual_files = {path.name for path in chunk_dir.iterdir()}
    add_issue(issues, chunk_dir.is_dir() and not chunk_dir.is_symlink(), "chunk_directory")
    add_issue(
        issues,
        actual_files
        == {"chunk_manifest.json", "counterfactual_records.jsonl.gz", "counterfactual_labels.npz"},
        "chunk_file_set",
    )
    add_issue(
        issues,
        all(not path.is_symlink() for path in chunk_dir.iterdir()),
        "chunk_symlink",
    )
    manifest = strict_json(manifest_path)
    add_issue(issues, set(manifest) == CHUNK_MANIFEST_KEYS, "manifest_schema")
    identity = {
        "protocol": PROTOCOL,
        "label_schema_version": SCHEMA_VERSION,
        "robot": robot,
        "training_seed": SEED,
        "source_role": role,
        "query_start": start,
        "query_stop_exclusive": stop,
        "query_count": stop - start,
        "record_count": (stop - start) * 4,
        "decision_action_execution_count": (stop - start) * 3 * 5,
        "fixed_robust_execution_count": 0,
        "fixed_robust_measurement_mode": "semantic_alias_of_easy",
        "environment_contaminated": False,
        "post_chunk_busy_processes": [],
        "test_data_loaded": False,
    }
    for name, expected in identity.items():
        add_issue(issues, manifest.get(name) == expected, f"manifest_identity:{name}")
    add_issue(
        issues,
        set(manifest.get("artifacts", {}))
        == {"counterfactual_records.jsonl.gz", "counterfactual_labels.npz"},
        "artifact_manifest_set",
    )
    for name, path in (
        ("counterfactual_records.jsonl.gz", records_path),
        ("counterfactual_labels.npz", labels_path),
    ):
        metadata = manifest.get("artifacts", {}).get(name, {})
        add_issue(issues, path.is_file() and not path.is_symlink(), f"artifact_exists:{name}")
        if path.is_file():
            add_issue(issues, path.stat().st_size == metadata.get("size"), f"artifact_size:{name}")
            add_issue(issues, file_sha256(path) == metadata.get("sha256"), f"artifact_sha:{name}")
    payload = dict(manifest)
    claimed = payload.pop("chunk_payload_sha256", None)
    add_issue(issues, mapping_digest(payload) == claimed, "chunk_payload_sha256")
    add_issue(
        issues,
        int(manifest["quiet_check_count"])
        == (stop - start) + int(manifest["contaminated_query_retries"]),
        "quiet_check_count",
    )
    events = manifest["contaminated_attempt_events"]
    add_issue(
        issues,
        len(events) == int(manifest["contaminated_query_retries"]),
        "contaminated_event_count",
    )
    for event in events:
        add_issue(issues, event.get("feature_and_rows_discarded") is True, "retry_not_discarded")
        add_issue(issues, bool(event.get("busy_processes")), "retry_without_busy_process")
        add_issue(
            issues,
            start <= int(event.get("query_index", -1)) < stop,
            "retry_query_outside_chunk",
        )
    wait_events = manifest["quiet_wait_events"]
    add_issue(
        issues,
        len(wait_events) == int(manifest["quiet_wait_event_count"]),
        "quiet_wait_event_count",
    )
    add_issue(
        issues,
        np.isclose(
            sum(float(event["wait_seconds"]) for event in wait_events),
            float(manifest["quiet_wait_seconds"]),
            rtol=0.0,
            atol=1e-9,
        ),
        "quiet_wait_seconds",
    )
    add_issue(
        issues,
        np.isclose(
            float(manifest["seconds_per_query"]),
            float(manifest["wall_time_seconds_excluding_warmup_and_writes"]) / (stop - start),
            rtol=0.0,
            atol=1e-12,
        ),
        "seconds_per_query",
    )

    labels = load_npz(labels_path)
    add_issue(issues, set(labels) == LABEL_KEYS, "npz_schema")
    count = stop - start
    nq = 7 if robot == "panda" else 6
    expected_shapes = {
        "features": (count, 9),
        "query_indices": (count,),
        "source_indices": (count,),
        "query_sha256": (count,),
        "category": (count,),
        "expected_reachable": (count,),
        "continuity_feasible": (count,),
        "verified_success": (count, 4),
        "verified_success_before_deadline": (count, 4),
        "latency_samples_ns": (count, 4, 5),
        "latency_p50_ms": (count, 4),
        "latency_p95_ms": (count, 4),
        "function_evaluations": (count, 4),
        "fallback_used": (count, 4),
        "measurement_is_alias": (count, 4),
        "failure_reason": (count, 4),
        "command_q": (count, 4, nq),
        "max_joint_step_rad": (count, 4),
        "max_joint_velocity_rad_s": (count, 4),
        "max_velocity_limit_utilization": (count, 4),
        "max_joint_acceleration_rad_s2": (count, 4),
        "max_joint_jerk_rad_s3": (count, 4),
        "dynamic_history_available": (count, 4),
    }
    for name, shape in expected_shapes.items():
        add_issue(issues, labels.get(name, np.empty(0)).shape == shape, f"npz_shape:{name}")
    add_issue(issues, tuple(labels["feature_names"].astype(str)) == FEATURE_NAMES, "feature_names")
    add_issue(issues, tuple(labels["action_names"].astype(str)) == ALL_ACTIONS, "action_names")
    add_issue(
        issues,
        tuple(labels["decision_action_names"].astype(str)) == DECISION_ACTIONS,
        "decision_action_names",
    )
    expected_query_indices = np.arange(start, stop, dtype=np.int64)
    expected_source_indices = selection["source_indices"][start:stop]
    add_issue(issues, arrays_equal(labels["query_indices"], expected_query_indices), "query_indices")
    add_issue(issues, arrays_equal(labels["source_indices"], expected_source_indices), "source_indices")
    add_issue(issues, np.all(labels["latency_samples_ns"] > 0), "nonpositive_latency")
    expected_alias = np.zeros((count, 4), dtype=bool)
    expected_alias[:, 3] = True
    add_issue(issues, arrays_equal(labels["measurement_is_alias"], expected_alias), "alias_matrix")
    add_issue(issues, np.all(np.isnan(labels["max_joint_acceleration_rad_s2"])), "acceleration_nan")
    add_issue(issues, np.all(np.isnan(labels["max_joint_jerk_rad_s3"])), "jerk_nan")
    add_issue(issues, not np.any(labels["dynamic_history_available"]), "dynamic_history_false")

    records: list[dict[str, Any]] = []
    with gzip.open(records_path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            records.append(strict_line(line, records_path, number))
    add_issue(issues, len(records) == count * 4, "raw_record_count")
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_keys = 0
    for row in records:
        add_issue(issues, set(row) == RECORD_KEYS, "raw_schema")
        query_index = int(row["query_index"])
        action = str(row["entry_action"])
        if action in grouped[query_index]:
            duplicate_keys += 1
        grouped[query_index][action] = row
    add_issue(issues, duplicate_keys == 0, "duplicate_query_action")
    add_issue(issues, set(grouped) == set(range(start, stop)), "raw_query_index_set")
    add_issue(
        issues,
        all(set(rows) == set(ALL_ACTIONS) for rows in grouped.values()),
        "raw_action_set",
    )

    action_summary: dict[str, Any] = {}
    semantic_fail_all = 0
    deadline_fail_all = 0
    deadline_only_fail_all = 0
    contract_feasible_semantic_fail_all = 0
    semantic_disagreement = 0
    deadline_disagreement = 0
    fail_by_category: Counter[str] = Counter()
    deadline_fail_by_category: Counter[str] = Counter()
    action_counts: dict[str, Counter[str]] = {action: Counter() for action in ALL_ACTIONS}
    alias_differences: Counter[str] = Counter()
    alias_exclusions = {
        "entry_action",
        "measurement_mode",
        "measurement_executed",
        "aliased_from_action",
    }
    action_index = {action: index for index, action in enumerate(ALL_ACTIONS)}
    for query_index in range(start, stop):
        local = query_index - start
        rows = grouped[query_index]
        easy = rows["easy"]
        fixed = rows["fixed_robust"]
        for field in RECORD_KEYS - alias_exclusions:
            if not value_equal(easy[field], fixed[field]):
                alias_differences[field] += 1
        add_issue(issues, fixed["measurement_mode"] == "semantic_alias", "fixed_mode")
        add_issue(issues, fixed["measurement_executed"] is False, "fixed_executed")
        add_issue(issues, fixed["aliased_from_action"] == "easy", "fixed_source")
        add_issue(issues, easy["measurement_mode"] == "executed", "easy_mode")
        add_issue(issues, easy["measurement_executed"] is True, "easy_executed")
        add_issue(issues, easy["aliased_from_action"] is None, "easy_alias_source")

        source_index = int(expected_source_indices[local])
        expected_hash = str(selection["query_sha256"][query_index])
        source_hash = query_digest(source, source_index, dt)
        add_issue(issues, source_hash == expected_hash, "source_query_hash")
        add_issue(issues, str(labels["query_sha256"][local]) == expected_hash, "npz_query_hash")
        add_issue(
            issues,
            str(labels["category"][local]) == str(source["category"][source_index]),
            "npz_category_source",
        )
        add_issue(
            issues,
            bool(labels["expected_reachable"][local])
            == bool(source["expected_reachable"][source_index]),
            "npz_expected_source",
        )
        add_issue(
            issues,
            bool(labels["continuity_feasible"][local])
            == bool(source["continuity_feasible"][source_index]),
            "npz_continuity_source",
        )
        reference_features = np.asarray(easy["risk_features"], dtype=np.float64)
        add_issue(issues, reference_features.shape == (9,), "raw_feature_shape")
        add_issue(issues, np.all(np.isfinite(reference_features)), "raw_feature_finite")
        add_issue(issues, arrays_equal(labels["features"][local], reference_features), "raw_npz_features")
        semantic_values = []
        deadline_values = []
        for action in ALL_ACTIONS:
            row = rows[action]
            column = action_index[action]
            add_issue(issues, row["robot"] == robot, "raw_robot")
            add_issue(issues, int(row["training_seed"]) == SEED, "raw_seed")
            add_issue(issues, row["source_role"] == role, "raw_role")
            add_issue(issues, row["chunk_name"] == chunk_name, "raw_chunk")
            add_issue(issues, int(row["source_index"]) == source_index, "raw_source_index")
            add_issue(issues, row["query_sha256"] == expected_hash, "raw_query_hash")
            add_issue(issues, row["source_query_sha256"] == expected_hash, "raw_source_query_hash")
            add_issue(issues, arrays_equal(row["risk_features"], reference_features), "raw_feature_repeat")
            add_issue(
                issues,
                str(row["category"]) == str(source["category"][source_index]),
                "raw_category_source",
            )
            add_issue(
                issues,
                int(row["trajectory_id"]) == int(source["trajectory_id"][source_index]),
                "raw_trajectory_source",
            )
            add_issue(
                issues,
                int(row["time_index"]) == int(source["time_index"][source_index]),
                "raw_time_source",
            )
            samples = np.asarray(row["latency_samples_ns"], dtype=np.int64)
            add_issue(issues, samples.shape == (5,), "raw_latency_shape")
            add_issue(issues, np.all(samples > 0), "raw_latency_positive")
            timing = row["timing_samples_ns"]
            add_issue(issues, set(timing) == set(TIMING_KEYS), "timing_key_set")
            if set(timing) == set(TIMING_KEYS):
                add_issue(
                    issues,
                    all(len(timing[name]) == 5 for name in TIMING_KEYS),
                    "timing_repeat_count",
                )
                total = np.asarray(timing["total_end_to_end_ns"], dtype=np.int64)
                add_issue(issues, arrays_equal(total, samples), "timing_total_samples")
                core = np.zeros(5, dtype=np.int64)
                for name in TIMING_KEYS:
                    if name not in ("total_end_to_end_ns", "unattributed_framework_ns"):
                        core += np.asarray(timing[name], dtype=np.int64)
                unattributed = np.asarray(timing["unattributed_framework_ns"], dtype=np.int64)
                add_issue(issues, arrays_equal(core + unattributed, samples), "timing_stage_sum")
                add_issue(issues, np.all(unattributed >= 0), "negative_unattributed")
            add_issue(
                issues,
                np.isclose(float(row["latency_p50_ns"]), np.percentile(samples, 50)),
                "raw_p50",
            )
            add_issue(
                issues,
                np.isclose(float(row["latency_p95_ns"]), np.percentile(samples, 95)),
                "raw_p95",
            )
            expected_deadline_rate = float(np.mean(samples <= 20_000_000)) if row["verified_success"] else 0.0
            add_issue(
                issues,
                np.isclose(float(row["deadline_success_rate"]), expected_deadline_rate),
                "deadline_success_rate",
            )
            expected_before = bool(
                row["verified_success"] and np.percentile(samples, 95) <= 20_000_000
            )
            add_issue(
                issues,
                bool(row["verified_success_before_deadline"]) == expected_before,
                "deadline_label",
            )
            scalar_pairs = {
                "success": bool(labels["verified_success"][local, column])
                == bool(row["verified_success"]),
                "deadline": bool(labels["verified_success_before_deadline"][local, column])
                == bool(row["verified_success_before_deadline"]),
                "latency_samples": arrays_equal(labels["latency_samples_ns"][local, column], samples),
                "latency_p50": np.isclose(
                    labels["latency_p50_ms"][local, column], float(row["latency_p50_ns"]) / 1e6
                ),
                "latency_p95": np.isclose(
                    labels["latency_p95_ms"][local, column], float(row["latency_p95_ns"]) / 1e6
                ),
                "fev": int(labels["function_evaluations"][local, column])
                == int(row["function_evaluations"]),
                "fallback": bool(labels["fallback_used"][local, column])
                == bool(row["fallback_used"]),
                "failure": str(labels["failure_reason"][local, column])
                == str(row["failure_reason"]),
            }
            for field, passed in scalar_pairs.items():
                add_issue(issues, passed, f"raw_npz:{field}")
            expected_command = (
                np.full(nq, np.nan)
                if row["command_q"] is None
                else np.asarray(row["command_q"], dtype=np.float64)
            )
            add_issue(
                issues,
                arrays_equal(labels["command_q"][local, column], expected_command),
                "raw_npz:command",
            )
            if action in DECISION_ACTIONS:
                add_issue(issues, row["measurement_mode"] == "executed", "decision_mode")
                add_issue(issues, row["measurement_executed"] is True, "decision_executed")
                add_issue(issues, row["aliased_from_action"] is None, "decision_alias_source")
                semantic_values.append(bool(row["verified_success"]))
                deadline_values.append(bool(row["verified_success_before_deadline"]))
            action_counts[action]["query_count"] += 1
            action_counts[action]["verified_success_count"] += bool(row["verified_success"])
            action_counts[action]["verified_success_before_deadline_count"] += bool(
                row["verified_success_before_deadline"]
            )
            action_counts[action]["function_evaluations_sum"] += int(row["function_evaluations"])
            action_counts[action]["fallback_count"] += bool(row["fallback_used"])
            if row["failure_reason"]:
                action_counts[action][f"failure:{row['failure_reason']}"] += 1
        semantic_disagreement += len(set(semantic_values)) > 1
        deadline_disagreement += len(set(deadline_values)) > 1
        semantic_fail = not any(semantic_values)
        deadline_fail = not any(deadline_values)
        category = str(easy["category"])
        semantic_fail_all += semantic_fail
        deadline_fail_all += deadline_fail
        deadline_only_fail_all += deadline_fail and not semantic_fail
        if semantic_fail:
            fail_by_category[category] += 1
        if deadline_fail:
            deadline_fail_by_category[category] += 1
        contract_feasible_semantic_fail_all += bool(
            semantic_fail
            and easy["expected_reachable"]
            and easy["continuity_feasible"]
        )
    for field, count_value in alias_differences.items():
        issues[f"easy_fixed_alias:{field}"] += count_value

    for action in ALL_ACTIONS:
        totals = action_counts[action]
        expected_summary = {
            "query_count": int(totals["query_count"]),
            "verified_success_count": int(totals["verified_success_count"]),
            "verified_success_before_deadline_count": int(
                totals["verified_success_before_deadline_count"]
            ),
            "function_evaluations_sum": int(totals["function_evaluations_sum"]),
            "fallback_count": int(totals["fallback_count"]),
            "failure_reason_counts": {
                key.removeprefix("failure:"): int(value)
                for key, value in totals.items()
                if key.startswith("failure:")
            },
            "measurement_mode": "semantic_alias"
            if action == "fixed_robust"
            else "executed",
        }
        action_summary[action] = expected_summary
        add_issue(
            issues,
            manifest["action_summary"].get(action) == expected_summary,
            f"action_summary:{action}",
        )
    return (
        {
            "chunk": chunk_name,
            "issue_counts": dict(issues),
            "pass": not issues,
            "contaminated_query_retries": int(manifest["contaminated_query_retries"]),
            "quiet_wait_events": int(manifest["quiet_wait_event_count"]),
            "quiet_wait_seconds": float(manifest["quiet_wait_seconds"]),
        },
        {
            "query_count": count,
            "record_count": count * 4,
            "decision_action_execution_count": count * 3 * 5,
            "fixed_alias_query_action_records": count,
            "fixed_alias_repeat_samples": count * 5,
            "wall_time_seconds": float(manifest["wall_time_seconds_excluding_warmup_and_writes"]),
            "contaminated_query_retries": int(manifest["contaminated_query_retries"]),
            "quiet_wait_events": int(manifest["quiet_wait_event_count"]),
            "quiet_wait_seconds": float(manifest["quiet_wait_seconds"]),
            "semantic_fail_all": int(semantic_fail_all),
            "deadline_fail_all": int(deadline_fail_all),
            "deadline_only_fail_all": int(deadline_only_fail_all),
            "contract_feasible_semantic_fail_all": int(
                contract_feasible_semantic_fail_all
            ),
            "semantic_action_disagreement": int(semantic_disagreement),
            "deadline_action_disagreement": int(deadline_disagreement),
            "semantic_fail_all_by_category": dict(fail_by_category),
            "deadline_fail_all_by_category": dict(deadline_fail_by_category),
            "action_summary": action_summary,
        },
    )


def merge_counts(target: Counter[str], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def audit_combination(
    *,
    workspace: Path,
    bulk_root: Path,
    manifest: Mapping[str, Any],
    robot: str,
    role: str,
    pilot_hashes: set[str],
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    key = f"{robot}/seed17/{role}"
    expected = manifest["selections"][key]
    role_root = bulk_root / robot / "seed17" / role
    selection_manifest = strict_json(role_root / "selection_manifest.json")
    selection = load_npz(role_root / "selection.npz")
    source_path = relative_or_absolute(workspace, expected["source_path"])
    source = load_npz(source_path)
    issues: Counter[str] = Counter()
    add_issue(issues, selection_manifest == expected, "selection_manifest_run_manifest")
    add_issue(issues, set(selection) == SELECTION_KEYS, "selection_npz_schema")
    add_issue(issues, file_sha256(role_root / "selection.npz") == expected["selection_artifact_sha256"], "selection_artifact_sha")
    add_issue(issues, file_sha256(source_path) == expected["source_sha256"], "source_sha")
    count = ROLE_COUNTS[role]
    selected = np.asarray(selection["source_indices"], dtype=np.int64)
    hashes = selection["query_sha256"].astype(str)
    add_issue(issues, len(selected) == count, "selection_count")
    add_issue(issues, len(np.unique(selected)) == count, "selection_source_unique")
    add_issue(issues, len(np.unique(hashes)) == count, "selection_hash_unique")
    add_issue(
        issues,
        sha256(np.ascontiguousarray(selected).tobytes()).hexdigest()
        == expected["selection_indices_sha256"],
        "selection_indices_digest",
    )
    add_issue(
        issues,
        sha256(np.ascontiguousarray(hashes.astype("S64")).tobytes()).hexdigest()
        == expected["selection_query_hashes_sha256"],
        "selection_hashes_digest",
    )
    computed_seed = selection_seed(manifest["frozen_provenance"]["release_commit"], robot, role)
    add_issue(issues, int(expected["selection_seed"]) == computed_seed, "selection_seed")
    replayed = replay_selection(
        source["category"], role=role, count=count, seed=computed_seed
    )
    add_issue(issues, arrays_equal(selected, replayed), "selection_exact_replay")
    computed_hashes = np.asarray(
        [query_digest(source, int(index), 0.02) for index in selected], dtype="U64"
    )
    add_issue(issues, arrays_equal(hashes, computed_hashes), "selection_query_hash_recompute")
    for field in ("category", "expected_reachable", "continuity_feasible"):
        add_issue(issues, arrays_equal(selection[field], source[field][selected]), f"selection_source:{field}")
    category_counts = dict(Counter(selection["category"].astype(str).tolist()))
    add_issue(issues, category_counts == expected["selected_category_counts"], "selection_category_manifest")
    if role == "risk_train_queries":
        add_issue(issues, category_counts == TRAIN_CATEGORY_COUNTS, "training_category_quota")
    overlap_pilot = len(set(hashes.tolist()) & pilot_hashes)
    add_issue(issues, overlap_pilot == 0, "pilot_overlap")

    chunks_root = role_root / "chunks"
    expected_names = expected_chunk_names(count)
    actual_committed = sorted(
        path.name for path in chunks_root.iterdir() if path.is_dir() and path.name.startswith("chunk_")
    )
    add_issue(issues, actual_committed == expected_names, "committed_chunk_set")
    incomplete = sorted(
        str(path.relative_to(bulk_root))
        for path in role_root.iterdir()
        if path.is_dir() and path.name.startswith(".chunk_")
    )
    chunk_issues: Counter[str] = Counter()
    failed_chunks: list[str] = []
    totals: Counter[str] = Counter()
    category_semantic: Counter[str] = Counter()
    category_deadline: Counter[str] = Counter()
    action_totals: dict[str, Counter[str]] = {action: Counter() for action in ALL_ACTIONS}
    for chunk_number, chunk_name in enumerate(expected_names):
        start = chunk_number * 250
        stop = min(start + 250, count)
        report, aggregate = audit_chunk(
            chunk_dir=chunks_root / chunk_name,
            chunk_name=chunk_name,
            robot=robot,
            role=role,
            start=start,
            stop=stop,
            selection=selection,
            source=source,
            dt=0.02,
        )
        if not report["pass"]:
            failed_chunks.append(chunk_name)
            merge_counts(chunk_issues, report["issue_counts"])
        for field in (
            "query_count",
            "record_count",
            "decision_action_execution_count",
            "fixed_alias_query_action_records",
            "fixed_alias_repeat_samples",
            "contaminated_query_retries",
            "quiet_wait_events",
            "semantic_fail_all",
            "deadline_fail_all",
            "deadline_only_fail_all",
            "contract_feasible_semantic_fail_all",
            "semantic_action_disagreement",
            "deadline_action_disagreement",
        ):
            totals[field] += int(aggregate[field])
        totals["wall_time_microseconds"] += int(round(aggregate["wall_time_seconds"] * 1e6))
        totals["quiet_wait_microseconds"] += int(round(aggregate["quiet_wait_seconds"] * 1e6))
        merge_counts(category_semantic, aggregate["semantic_fail_all_by_category"])
        merge_counts(category_deadline, aggregate["deadline_fail_all_by_category"])
        for action, values in aggregate["action_summary"].items():
            for field in (
                "query_count",
                "verified_success_count",
                "verified_success_before_deadline_count",
                "function_evaluations_sum",
                "fallback_count",
            ):
                action_totals[action][field] += int(values[field])
    return (
        {
            "pass": not issues and not failed_chunks,
            "issue_counts": dict(issues),
            "failed_chunk_count": len(failed_chunks),
            "failed_chunks": failed_chunks[:20],
            "chunk_issue_counts": dict(chunk_issues),
            "expected_chunk_count": len(expected_names),
            "incomplete_attempt_directories": incomplete,
            "selection": {
                "count": count,
                "category_counts": category_counts,
                "pilot_overlap_count": overlap_pilot,
            },
            "counts": {
                key: int(value) for key, value in totals.items() if not key.endswith("microseconds")
            },
            "wall_time_seconds": totals["wall_time_microseconds"] / 1e6,
            "quiet_wait_seconds": totals["quiet_wait_microseconds"] / 1e6,
            "semantic_fail_all_by_category": dict(category_semantic),
            "deadline_fail_all_by_category": dict(category_deadline),
            "action_totals": {action: dict(values) for action, values in action_totals.items()},
        },
        {
            "totals": totals,
            "action_totals": action_totals,
            "semantic_categories": category_semantic,
            "deadline_categories": category_deadline,
        },
        set(hashes.tolist()),
    )


def audit_bulk_summary(
    bulk_root: Path,
    manifest: Mapping[str, Any],
    combinations: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    summary_path = bulk_root / "bulk_summary.json"
    environment_path = bulk_root / "environment_final.json"
    add_issue(issues, file_sha256(summary_path) == manifest["bulk_summary_sha256"], "bulk_summary_sha")
    add_issue(issues, file_sha256(environment_path) == manifest["environment_final_sha256"], "environment_sha")
    summary = strict_json(summary_path)
    add_issue(issues, summary["protocol"] == PROTOCOL, "summary_protocol")
    add_issue(issues, summary["test_data_loaded"] is False, "summary_test_flag")
    add_issue(issues, set(summary["combinations"]) == set(combinations), "summary_combination_set")
    for key, audited in combinations.items():
        observed = summary["combinations"][key]
        add_issue(issues, int(observed["query_count"]) == audited["selection"]["count"], f"summary_query_count:{key}")
        add_issue(
            issues,
            np.isclose(float(observed["wall_time_seconds_excluding_warmup_and_writes"]), audited["wall_time_seconds"], atol=5e-5),
            f"summary_wall_time:{key}",
        )
        for action in ALL_ACTIONS:
            source = audited["action_totals"][action]
            target = observed["actions"][action]
            for field in (
                "query_count",
                "verified_success_count",
                "verified_success_before_deadline_count",
                "function_evaluations_sum",
                "fallback_count",
            ):
                add_issue(issues, int(target[field]) == int(source[field]), f"summary:{key}:{action}:{field}")
            add_issue(
                issues,
                target["measurement_mode"]
                == ("semantic_alias" if action == "fixed_robust" else "executed"),
                f"summary_mode:{key}:{action}",
            )
    return {"pass": not issues, "issue_counts": dict(issues)}


def audit_manifest_counts(
    bulk_root: Path, manifest: Mapping[str, Any], combinations: Mapping[str, Any]
) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    add_issue(issues, manifest["protocol"] == PROTOCOL, "protocol")
    add_issue(issues, int(manifest["label_schema_version"]) == SCHEMA_VERSION, "schema_version")
    add_issue(issues, manifest["status"] == "complete", "status")
    add_issue(issues, int(manifest["completed_chunk_count"]) == 160, "completed_chunks")
    add_issue(issues, int(manifest["completed_query_count"]) == 40_000, "completed_queries")
    add_issue(issues, int(manifest["total_query_count"]) == 40_000, "total_queries")
    add_issue(issues, int(manifest["environment_contaminated_chunk_count"]) == 0, "contaminated_chunks")
    add_issue(issues, manifest["test_data_loaded"] is False, "test_data_loaded")
    add_issue(issues, manifest["test_v3_used_for_selection"] is False, "test_selection")
    add_issue(issues, manifest["formal_test_eligible"] is False, "formal_test_eligible")
    add_issue(issues, manifest["protected_outputs_unchanged"] is True, "protected_flag")
    add_issue(issues, manifest["selection_roles_disjoint"] is True, "selection_disjoint_flag")
    add_issue(issues, tuple(manifest["decision_actions"]) == DECISION_ACTIONS, "decision_actions")
    add_issue(issues, tuple(manifest["collected_actions"]) == ALL_ACTIONS, "collected_actions")
    fixed = manifest["fixed_robust"]
    add_issue(
        issues,
        fixed == {
            "measurement_mode": "semantic_alias",
            "aliased_from_action": "easy",
            "executed_in_bulk": False,
            "reason": "the validation pilot proved exact command/outcome/FEV semantics",
        },
        "fixed_alias_manifest",
    )
    add_issue(issues, set(manifest["selections"]) == set(combinations), "selection_keys")
    expected_combination_progress = {
        key: {"completed_queries": ROLE_COUNTS[key.split("/")[-1]], "total_queries": ROLE_COUNTS[key.split("/")[-1]]}
        for key in combinations
    }
    add_issue(issues, manifest["combinations"] == expected_combination_progress, "combination_progress")
    failure_path = bulk_root / "failure_events.jsonl"
    failure_lines = 0
    if failure_path.exists():
        with failure_path.open("rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                strict_line(line, failure_path, number)
                failure_lines += 1
    add_issue(issues, int(manifest["failure_event_count"]) == failure_lines, "failure_event_count")
    return {
        "pass": not issues,
        "issue_counts": dict(issues),
        "failure_event_count": failure_lines,
    }


def main() -> None:
    args = arguments()
    workspace = args.workspace.resolve()
    bulk_root = args.bulk_root
    if not bulk_root.is_absolute():
        bulk_root = workspace / bulk_root
    bulk_root = bulk_root.resolve()
    expected_root = (workspace / "outputs/counterfactual_v4_bulk").resolve()
    if bulk_root != expected_root or bulk_root.is_symlink():
        raise ValueError(f"bulk root must be exactly {expected_root}")
    manifest = strict_json(bulk_root / "run_manifest.json")
    if args.require_complete and manifest.get("status") != "complete":
        print(json.dumps({"audit_pass": False, "reason": "bulk_not_complete"}))
        raise SystemExit(2)

    pilot_hashes: dict[str, set[str]] = {}
    for robot in ROBOTS:
        pilot_path = workspace / f"outputs/counterfactual_v4_pilot/{robot}/seed17/counterfactual_labels.npz"
        with np.load(pilot_path, allow_pickle=False) as pilot:
            pilot_hashes[robot] = set(pilot["query_sha256"].astype(str).tolist())

    combinations: dict[str, Any] = {}
    aggregate_context: dict[str, Any] = {}
    role_hashes: dict[str, dict[str, set[str]]] = defaultdict(dict)
    global_totals: Counter[str] = Counter()
    for robot in ROBOTS:
        for role in ROLES:
            key = f"{robot}/seed17/{role}"
            report, context, hashes = audit_combination(
                workspace=workspace,
                bulk_root=bulk_root,
                manifest=manifest,
                robot=robot,
                role=role,
                pilot_hashes=pilot_hashes[robot],
            )
            combinations[key] = report
            aggregate_context[key] = context
            role_hashes[robot][role] = hashes
            for field, value in context["totals"].items():
                global_totals[field] += int(value)

    disjoint: dict[str, Any] = {}
    split_disjoint_pass = True
    for robot in ROBOTS:
        pairs: dict[str, int] = {}
        for left_index, left in enumerate(ROLES):
            for right in ROLES[left_index + 1 :]:
                overlap = len(role_hashes[robot][left] & role_hashes[robot][right])
                pairs[f"{left}__{right}"] = overlap
                split_disjoint_pass &= overlap == 0
        disjoint[robot] = pairs

    expected_global = {
        "query_count": 40_000,
        "record_count": 160_000,
        "decision_action_execution_count": 600_000,
        "fixed_alias_query_action_records": 40_000,
        "fixed_alias_repeat_samples": 200_000,
    }
    count_checks = {
        name: int(global_totals[name]) == expected for name, expected in expected_global.items()
    }
    frozen = audit_frozen_provenance(workspace, manifest)
    protected = audit_protected_snapshot(workspace, manifest)
    summary = audit_bulk_summary(bulk_root, manifest, combinations)
    manifest_audit = audit_manifest_counts(bulk_root, manifest, combinations)

    support: dict[str, Any] = {}
    broad_reject_policy_validation_pass = True
    for robot in ROBOTS:
        roles: dict[str, Any] = {}
        aggregate_feasible_semantic_fail = 0
        for role in ROLES:
            values = combinations[f"{robot}/seed17/{role}"]["counts"]
            count = int(values["contract_feasible_semantic_fail_all"])
            roles[role] = {
                "semantic_fail_all": int(values["semantic_fail_all"]),
                "deadline_fail_all": int(values["deadline_fail_all"]),
                "deadline_only_fail_all": int(values["deadline_only_fail_all"]),
                "contract_feasible_semantic_fail_all": count,
                "meets_frozen_minimum_30": count >= 30,
                "semantic_action_disagreement": int(values["semantic_action_disagreement"]),
                "deadline_action_disagreement": int(values["deadline_action_disagreement"]),
            }
            aggregate_feasible_semantic_fail += count
        broad_reject_policy_validation_pass &= roles["policy_validation_queries"][
            "meets_frozen_minimum_30"
        ]
        support[robot] = {
            "roles": roles,
            "aggregate_contract_feasible_semantic_fail_all": aggregate_feasible_semantic_fail,
            "aggregate_meets_minimum_30": aggregate_feasible_semantic_fail >= 30,
        }

    combination_pass = all(item["pass"] for item in combinations.values())
    structural_pass = bool(
        combination_pass
        and all(count_checks.values())
        and split_disjoint_pass
        and frozen["pass"]
        and protected["protected_baseline_sha256_matches"]
        and protected["current_snapshot_matches_baseline"]
        and protected["test_v3_content_files_opened"] == 0
        and protected["test_v3_metadata_content_opened_flags_false"]
        and summary["pass"]
        and manifest_audit["pass"]
    )
    blockers: list[str] = []
    if not combination_pass:
        blockers.append("one_or_more_selection_or_chunk_audits_failed")
    if not all(count_checks.values()):
        blockers.append("global_count_contract_failed")
    if not split_disjoint_pass:
        blockers.append("development_split_overlap")
    if not frozen["pass"]:
        blockers.append("frozen_provenance_mismatch")
    if not protected["current_snapshot_matches_baseline"]:
        blockers.append("protected_tree_changed")
    if not summary["pass"]:
        blockers.append("bulk_summary_mismatch")
    if not manifest_audit["pass"]:
        blockers.append("run_manifest_mismatch")

    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "audited_bulk_root": str(bulk_root),
        "scope": "training/calibration/policy-validation bulk labels only",
        "test_v3_performance_values_read": False,
        "artifact_audit_pass": structural_pass,
        "blocking_issues": blockers,
        "manifest": manifest_audit,
        "frozen_provenance": frozen,
        "protected_snapshot": protected,
        "global_contract": {
            "expected": expected_global,
            "observed": {name: int(global_totals[name]) for name in expected_global},
            "checks": count_checks,
            "expected_chunk_count": 160,
            "observed_chunk_count": sum(item["expected_chunk_count"] for item in combinations.values()),
        },
        "selection_disjointness": {
            "pairwise_overlap_counts": disjoint,
            "all_zero": split_disjoint_pass,
            "pilot_overlap_all_zero": all(
                item["selection"]["pilot_overlap_count"] == 0
                for item in combinations.values()
            ),
        },
        "combinations": combinations,
        "bulk_summary": summary,
        "fail_all_support": {
            "by_robot_and_role": support,
            "broad_reject_policy_validation_support_gate_pass": broad_reject_policy_validation_pass,
            "gate_interpretation": (
                "The minimum-30 broad-reject support gate is evaluated conservatively "
                "on policy-validation separately for each robot. Training and calibration "
                "support are reported separately and are not pooled to rescue a failed "
                "policy-validation gate."
            ),
        },
        "claim_status": {
            "bulk_labels_structurally_usable": structural_pass,
            "broad_reject_claim_supported_by_count_gate": broad_reject_policy_validation_pass,
            "formal_test_eligible": False,
            "note": (
                "Passing this audit authorizes development training only. It does not "
                "freeze a v4 release or authorize a fresh formal test."
            ),
        },
    }
    destination = args.json
    if not destination.is_absolute():
        destination = workspace / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_audit_pass": structural_pass,
                "blocking_issues": blockers,
                "broad_reject_policy_validation_support_gate_pass": broad_reject_policy_validation_pass,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if structural_pass else 1)


if __name__ == "__main__":
    main()
