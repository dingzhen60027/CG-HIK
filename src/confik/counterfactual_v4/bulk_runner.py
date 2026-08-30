from __future__ import annotations

"""Resumable training/validation-only counterfactual label collection.

This runner deliberately has no test-data code path.  It executes the three
decision entries (easy, medium, and hard), while recording ``fixed_robust`` as
an explicitly marked semantic alias of easy.  The latter equivalence was
established by the completed validation pilot and is re-audited from that
pilot before a bulk run is initialized.

Every chunk is committed by an atomic directory rename and contains its own
hash manifest.  Resume is allowed only when the complete Python source tree,
configuration, frozen datasets, sealed v3 release, and alias-evidence hashes
are unchanged.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime, query_digest, query_from_dataset
from ..latency_pilot_v3.optimized_inference import SeedEngine, cached_risk_features
from ..types import IKQuery
from .collector import (
    ACTIONS,
    COLLECTED_ACTIONS,
    STAGE_TIMING_KEYS,
    _failure_reason,
    _semantic_signature,
    _timed_call,
    validate_source_role,
)
from .runner import (
    FEATURE_NAMES,
    _build_runtimes,
    _busy_unrelated_processes,
    _source_dataset_path,
    _wait_for_quiet_environment,
)


PROTOCOL = "counterfactual_v4_bulk_training_validation_v1"
LABEL_SCHEMA_VERSION = 3
ROLE_ORDER = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)
FIXED_ALIAS_ACTION = "fixed_robust"
FIXED_ALIAS_SOURCE = "easy"
PROTECTED_OUTPUT_PATTERNS = (
    "paper_v2_*",
    "paper_v2_aggregate",
    "latency_pilot_v3",
    "release_v3_locked",
    "counterfactual_v4_pilot",
    "counterfactual_v4_smoke*",
    "counterfactual_v4_readiness_smoke*",
)
TEST_V3_METADATA_ONLY_PATTERNS = ("test_v3_seed*", "test_v3_aggregate")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect resumable development-only v4 counterfactual labels"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional operational pause after N newly committed chunks; labels are unchanged.",
    )
    return parser


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    separators = None if pretty else (",", ":")
    text = json.dumps(
        _safe(payload),
        indent=2 if pretty else None,
        sort_keys=True,
        allow_nan=False,
        separators=separators,
    )
    return (text + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _append_failure(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_bytes(dict(payload), pretty=False))
        handle.flush()
        os.fsync(handle.fileno())


def _digest_mapping(payload: Mapping[str, Any]) -> str:
    return sha256(_json_bytes(dict(payload), pretty=False)).hexdigest()


def _file_manifest(paths: Iterable[Path], *, relative_to: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required regular file is missing: {path}")
        try:
            key = str(path.relative_to(relative_to.resolve()))
        except ValueError:
            key = str(path)
        manifest[key] = {"sha256": _sha256_file(path), "size": path.stat().st_size}
    return manifest


def _code_manifest(workspace: Path) -> dict[str, Any]:
    paths = list((workspace / "src" / "confik").rglob("*.py"))
    if Path(__file__).resolve() not in {path.resolve() for path in paths}:
        raise RuntimeError("bulk runner is absent from the source-code fingerprint")
    files = _file_manifest(paths, relative_to=workspace)
    return {"files": files, "sha256": _digest_mapping(files)}


def _directory_file_manifest(directory: Path, *, relative_to: Path) -> dict[str, Any]:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(directory)
    files = _file_manifest(
        (path for path in directory.rglob("*") if path.is_file()),
        relative_to=relative_to,
    )
    return {"files": files, "sha256": _digest_mapping(files)}


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("protocol_version", -1)) != 4:
        raise ValueError("bulk v4 requires protocol_version: 4")
    robots = tuple(str(item) for item in config.get("robots", ()))
    if robots != ("panda", "ur5e"):
        raise ValueError("the frozen bulk design requires robots: [panda, ur5e]")
    seeds = tuple(int(item) for item in config.get("training_seeds", ()))
    if seeds != (17,):
        raise ValueError("the frozen bulk design requires shared training seed 17")
    counts = config.get("data", {}).get("role_counts", {})
    required = {
        # Per robot. Across Panda and UR5e this yields the frozen aggregate
        # design of 30k / 5k / 5k development queries.
        "risk_train_queries": 15_000,
        "calibration_queries": 2_500,
        "policy_validation_queries": 2_500,
    }
    if {str(key): int(value) for key, value in counts.items()} != required:
        raise ValueError(f"role_counts must equal the frozen design {required}")
    for role in counts:
        validate_source_role(str(role))
        if "test" in str(role).lower():
            raise ValueError("test roles are forbidden")
    if int(config.get("timing", {}).get("repeats", 0)) != 5:
        raise ValueError("the frozen bulk design requires exactly five timing repeats")
    if int(config.get("bulk", {}).get("chunk_size", 0)) != 250:
        raise ValueError("the frozen bulk design requires chunk_size: 250")
    if float(config.get("timing", {}).get("deadline_ms", 0.0)) != 20.0:
        raise ValueError("the frozen command deadline is 20 ms")
    if float(config.get("data", {}).get("dt", 0.0)) <= 0.0:
        raise ValueError("data.dt must be positive")
    if int(config.get("runtime", {}).get("environment_check_every_queries", 0)) != 1:
        raise ValueError(
            "the frozen latency-label protocol requires environment_check_every_queries: 1"
        )


def _assert_output_scope(workspace: Path, output_root: Path) -> None:
    expected_parent = (workspace / "outputs").resolve()
    resolved = output_root.resolve()
    if resolved.parent != expected_parent or resolved.name != "counterfactual_v4_bulk":
        raise ValueError(
            "bulk output must be exactly outputs/counterfactual_v4_bulk; "
            f"got {resolved}"
        )
    if output_root.is_symlink():
        raise RuntimeError("bulk output cannot be a symlink")


def _hashed_directory_snapshot(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in root.glob(pattern) if path.is_dir())
    files: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output cannot be a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = str(path.relative_to(root))
            stat = path.stat()
            files[relative] = {
                "sha256": _sha256_file(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return {
        "directories": [str(path.relative_to(root)) for path in sorted(directories)],
        "file_count": len(files),
        "files": files,
    }


def _metadata_directory_snapshot(root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    """Protect formal test outputs without opening or hashing their contents."""

    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in root.glob(pattern) if path.is_dir())
    files: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected test output cannot be a symlink: {directory}")
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


def _protected_snapshot(workspace: Path) -> dict[str, Any]:
    output_root = workspace / "outputs"
    output_snapshot = _hashed_directory_snapshot(output_root, PROTECTED_OUTPUT_PATTERNS)
    test_v3_metadata = _metadata_directory_snapshot(
        output_root, TEST_V3_METADATA_ONLY_PATTERNS
    )
    czy_root = workspace / "czy"
    czy_files = (
        _file_manifest(
            (path for path in czy_root.rglob("*") if path.is_file()),
            relative_to=workspace,
        )
        if czy_root.is_dir()
        else {}
    )
    return {
        "outputs": output_snapshot,
        "test_v3_metadata_only": test_v3_metadata,
        "czy": {"files": czy_files, "sha256": _digest_mapping(czy_files)},
    }


def _selection_seed(
    release_commit: str, robot: str, training_seed: int, role: str
) -> int:
    material = (
        f"counterfactual_v4_bulk_v1|{release_commit}|{robot}|{training_seed}|{role}"
    ).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big", signed=False)


def _selected_indices(dataset: QueryDataset, *, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count > len(dataset):
        raise ValueError(f"selection count must be in [1, {len(dataset)}], got {count}")
    selected = np.random.default_rng(seed).choice(len(dataset), size=count, replace=False)
    return np.asarray(selected, dtype=np.int64)


def _query_hashes(
    dataset: QueryDataset, selected: np.ndarray, *, dt: float
) -> np.ndarray:
    return np.asarray(
        [query_digest(query_from_dataset(dataset, int(index), dt=dt)) for index in selected],
        dtype="U64",
    )


def _alias_evidence(workspace: Path, *, robot: str, training_seed: int) -> dict[str, Any]:
    root = workspace / "outputs" / "counterfactual_v4_pilot" / robot / f"seed{training_seed}"
    labels_path = root / "counterfactual_labels.npz"
    records_path = root / "counterfactual_records.jsonl.gz"
    summary_path = root / "pilot_summary.json"
    for path in (labels_path, records_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(
                "fixed_robust aliasing requires the completed validation pilot: " + str(path)
            )
    with np.load(labels_path, allow_pickle=False) as data:
        actions = data["action_names"].astype(str).tolist()
        easy = actions.index(FIXED_ALIAS_SOURCE)
        fixed = actions.index(FIXED_ALIAS_ACTION)
        checks = {
            "verified_success": bool(
                np.array_equal(data["verified_success"][:, easy], data["verified_success"][:, fixed])
            ),
            "function_evaluations": bool(
                np.array_equal(
                    data["function_evaluations"][:, easy],
                    data["function_evaluations"][:, fixed],
                )
            ),
            "fallback_used": bool(
                np.array_equal(data["fallback_used"][:, easy], data["fallback_used"][:, fixed])
            ),
            "failure_reason": bool(
                np.array_equal(data["failure_reason"][:, easy], data["failure_reason"][:, fixed])
            ),
            "command_q": bool(
                np.allclose(
                    data["command_q"][:, easy],
                    data["command_q"][:, fixed],
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            ),
        }
        query_count = int(data["verified_success"].shape[0])
    if not all(checks.values()):
        raise RuntimeError(f"validation pilot does not prove easy/fixed semantic alias: {checks}")
    files = _file_manifest((labels_path, records_path, summary_path), relative_to=workspace)
    return {
        "pilot_root": str(root),
        "query_count": query_count,
        "semantic_checks": checks,
        "files": files,
        "files_sha256": _digest_mapping(files),
    }


def _frozen_provenance(
    *,
    workspace: Path,
    config_path: Path,
    config: Mapping[str, Any],
    source_config_path: Path,
    release_root: Path,
    release_commit: str,
) -> dict[str, Any]:
    source_datasets: dict[str, Any] = {}
    alias_evidence: dict[str, Any] = {}
    for robot in config["robots"]:
        for training_seed in config["training_seeds"]:
            for role in ROLE_ORDER:
                path = _source_dataset_path(
                    workspace,
                    robot=str(robot),
                    training_seed=int(training_seed),
                    role=role,
                )
                source_datasets[f"{robot}/seed{int(training_seed)}/{role}"] = {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                }
            alias_evidence[f"{robot}/seed{int(training_seed)}"] = _alias_evidence(
                workspace, robot=str(robot), training_seed=int(training_seed)
            )
    return {
        "protocol": PROTOCOL,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "source_config": {
            "path": str(source_config_path),
            "sha256": _sha256_file(source_config_path),
        },
        "code": _code_manifest(workspace),
        "source_datasets": source_datasets,
        "source_datasets_sha256": _digest_mapping(source_datasets),
        "release": _directory_file_manifest(release_root, relative_to=workspace),
        "release_root": str(release_root),
        "release_commit": release_commit,
        "fixed_robust_alias_evidence": alias_evidence,
        "fixed_robust_alias_evidence_sha256": _digest_mapping(alias_evidence),
        "test_data_loaded": False,
        "allowed_roles": list(ROLE_ORDER),
    }


def _validate_frozen_provenance(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if dict(expected) != dict(actual):
        expected_digest = _digest_mapping(dict(expected))
        actual_digest = _digest_mapping(dict(actual))
        raise RuntimeError(
            "resume provenance mismatch; refusing to mix code/config/source states: "
            f"expected={expected_digest}, actual={actual_digest}"
        )


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _initialize_output(
    *,
    workspace: Path,
    output_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    release_commit: str,
    frozen: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    staging = output_root.with_name(f".{output_root.name}.initializing.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    selections: dict[str, Any] = {}
    role_query_hashes: dict[tuple[str, int], dict[str, set[str]]] = {}
    try:
        dt = float(config["data"]["dt"])
        counts = {str(key): int(value) for key, value in config["data"]["role_counts"].items()}
        for robot in config["robots"]:
            for training_seed in config["training_seeds"]:
                role_sets: dict[str, set[str]] = {}
                for role in ROLE_ORDER:
                    path = _source_dataset_path(
                        workspace,
                        robot=str(robot),
                        training_seed=int(training_seed),
                        role=role,
                    )
                    dataset = QueryDataset.load(path)
                    seed = _selection_seed(release_commit, str(robot), int(training_seed), role)
                    selected = _selected_indices(dataset, count=counts[role], seed=seed)
                    hashes = _query_hashes(dataset, selected, dt=dt)
                    if len(np.unique(hashes)) != len(hashes):
                        raise RuntimeError(f"duplicate selected queries in {robot}/seed{training_seed}/{role}")
                    role_sets[role] = set(hashes.tolist())
                    destination = staging / str(robot) / f"seed{int(training_seed)}" / role
                    (destination / "chunks").mkdir(parents=True, exist_ok=False)
                    selection_path = destination / "selection.npz"
                    _write_npz_atomic(
                        selection_path,
                        source_indices=selected,
                        query_sha256=hashes,
                        category=dataset.category[selected],
                        expected_reachable=dataset.expected_reachable[selected],
                        continuity_feasible=dataset.continuity_feasible[selected],
                    )
                    selection = {
                        "robot": str(robot),
                        "training_seed": int(training_seed),
                        "source_role": role,
                        "source_path": str(path),
                        "source_sha256": _sha256_file(path),
                        "source_query_count": len(dataset),
                        "selected_query_count": len(selected),
                        "selection_seed": seed,
                        "selection_indices_sha256": sha256(
                            np.ascontiguousarray(selected).tobytes()
                        ).hexdigest(),
                        "selection_query_hashes_sha256": sha256(
                            np.ascontiguousarray(hashes.astype("S64")).tobytes()
                        ).hexdigest(),
                        "selection_artifact_sha256": _sha256_file(selection_path),
                        "selected_category_counts": dict(
                            sorted(Counter(dataset.category[selected].astype(str).tolist()).items())
                        ),
                        "test_named_dataset_loaded": False,
                    }
                    _atomic_json(destination / "selection_manifest.json", selection)
                    selections[f"{robot}/seed{int(training_seed)}/{role}"] = selection
                for left_index, left in enumerate(ROLE_ORDER):
                    for right in ROLE_ORDER[left_index + 1 :]:
                        overlap = role_sets[left] & role_sets[right]
                        if overlap:
                            raise RuntimeError(
                                f"development-role query overlap in {robot}/seed{training_seed}: "
                                f"{left} vs {right}, count={len(overlap)}"
                            )
                role_query_hashes[(str(robot), int(training_seed))] = role_sets

        manifest = {
            "protocol": PROTOCOL,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "status": "in_progress",
            "created_utc": _utc(),
            "updated_utc": _utc(),
            "config_path": str(config_path),
            "frozen_provenance": dict(frozen),
            "frozen_provenance_sha256": _digest_mapping(dict(frozen)),
            "protected_baseline": dict(protected),
            "protected_baseline_sha256": _digest_mapping(dict(protected)),
            "selections": selections,
            "selection_roles_disjoint": True,
            "decision_actions": list(ACTIONS),
            "collected_actions": list(COLLECTED_ACTIONS),
            "fixed_robust": {
                "measurement_mode": "semantic_alias",
                "aliased_from_action": FIXED_ALIAS_SOURCE,
                "executed_in_bulk": False,
                "reason": "the validation pilot proved exact command/outcome/FEV semantics",
            },
            "test_data_loaded": False,
            "test_v3_used_for_selection": False,
            "formal_test_eligible": False,
            "completed_chunk_count": 0,
            "completed_query_count": 0,
            "total_query_count": int(sum(item["selected_query_count"] for item in selections.values())),
            "failure_event_count": 0,
        }
        _atomic_json(staging / "run_manifest.json", manifest)
        _atomic_json(
            staging / "environment_initial.json",
            {
                **environment_payload(),
                "captured_utc": _utc(),
                "python_executable": os.sys.executable,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            },
        )
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        _append_failure(
            staging / "failure_events.jsonl",
            {
                "failed_utc": _utc(),
                "phase": "initialization",
                "traceback": traceback.format_exc(),
                "test_data_loaded": False,
            },
        )
        raise


def _load_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"existing bulk output lacks run_manifest.json: {output_root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("existing output belongs to a different protocol")
    return payload


def _selection_for_role(role_root: Path, expected: Mapping[str, Any]) -> np.ndarray:
    selection_path = role_root / "selection.npz"
    if _sha256_file(selection_path) != expected["selection_artifact_sha256"]:
        raise RuntimeError(f"selection artifact changed: {selection_path}")
    with np.load(selection_path, allow_pickle=False) as data:
        selected = np.asarray(data["source_indices"], dtype=np.int64)
        hashes = data["query_sha256"].astype(str)
    if len(selected) != int(expected["selected_query_count"]):
        raise RuntimeError(f"selection count changed: {selection_path}")
    if sha256(np.ascontiguousarray(selected).tobytes()).hexdigest() != expected[
        "selection_indices_sha256"
    ]:
        raise RuntimeError(f"selection indices changed: {selection_path}")
    if sha256(np.ascontiguousarray(hashes.astype("S64")).tobytes()).hexdigest() != expected[
        "selection_query_hashes_sha256"
    ]:
        raise RuntimeError(f"selection query hashes changed: {selection_path}")
    return selected


def _fixed_alias_record(easy_record: Mapping[str, Any]) -> dict[str, Any]:
    if easy_record.get("entry_action") != FIXED_ALIAS_SOURCE:
        raise ValueError("fixed robust can only alias an easy record")
    record = deepcopy(dict(easy_record))
    record.update(
        {
            "entry_action": FIXED_ALIAS_ACTION,
            "measurement_mode": "semantic_alias",
            "measurement_executed": False,
            "aliased_from_action": FIXED_ALIAS_SOURCE,
            "fixed_robust_matches_easy": True,
        }
    )
    return record


def _record_from_outcomes(
    *,
    action: str,
    outcomes: list[Any],
    query: IKQuery,
    query_index: int,
    source_index: int,
    dataset: QueryDataset,
    runtime: ProfiledCascadeRuntime,
    deadline_ms: float,
) -> dict[str, Any]:
    reference = outcomes[-1]
    signature = _semantic_signature(reference)
    if any(_semantic_signature(outcome) != signature for outcome in outcomes):
        raise RuntimeError(f"counterfactual action {action} changed semantics across repeats")
    latencies = np.asarray(
        [outcome.timings_ns["total_end_to_end_ns"] for outcome in outcomes], dtype=np.int64
    )
    accepted = bool(reference.accepted)
    if accepted and reference.q is not None:
        kinematics = runtime.kinematics
        joint_delta = np.abs(kinematics.difference(reference.q, query.previous_q))
        joint_step_max = float(np.max(joint_delta))
        joint_velocity_max = float(np.max(joint_delta / query.dt))
        velocity_utilization_max = float(
            np.max(joint_delta / (kinematics.limits.velocity * query.dt))
        )
    else:
        joint_step_max = None
        joint_velocity_max = None
        velocity_utilization_max = None
    return {
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
        "deadline_success_rate": float(np.mean(latencies <= int(deadline_ms * 1e6)))
        if accepted
        else 0.0,
        "latency_samples_ns": latencies.tolist(),
        "latency_p50_ns": float(np.percentile(latencies, 50)),
        "latency_p95_ns": float(np.percentile(latencies, 95)),
        "function_evaluations": int(reference.function_evaluations),
        "iterations": int(reference.iterations),
        "fallback_used": bool(reference.fallback_used),
        "executed_stages": list(reference.executed_stages),
        "failure_reason": _failure_reason(reference),
        "verification_reasons": list(reference.verification_reasons),
        "command_q": None
        if reference.q is None
        else np.asarray(reference.q, dtype=np.float64).tolist(),
        "max_joint_step_rad": joint_step_max,
        "max_joint_velocity_rad_s": joint_velocity_max,
        "max_velocity_limit_utilization": velocity_utilization_max,
        "max_joint_acceleration_rad_s2": None,
        "max_joint_jerk_rad_s3": None,
        "dynamic_history_available": False,
        "measurement_mode": "executed",
        "measurement_executed": True,
        "aliased_from_action": None,
        "fixed_robust_matches_easy": action == FIXED_ALIAS_SOURCE,
        "timing_samples_ns": {
            key: [int(outcome.timings_ns.get(key, 0)) for outcome in outcomes]
            for key in STAGE_TIMING_KEYS
        },
    }


def _collect_query(
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
    if set(runtimes) != set(ACTIONS):
        raise ValueError(f"bulk runtimes must contain exactly {ACTIONS}")
    if repeats != 5:
        raise ValueError("bulk collection requires exactly five repeats")
    collected: dict[str, list[Any]] = {action: [] for action in ACTIONS}
    rng = np.random.default_rng(order_seed)
    base_order = list(ACTIONS)
    rng.shuffle(base_order)
    for repeat in range(repeats):
        offset = repeat % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        if (query_index // len(base_order)) % 2:
            order = list(reversed(order))
        for action in order:
            collected[action].append(_timed_call(runtimes[action], query))
    rows = [
        _record_from_outcomes(
            action=action,
            outcomes=collected[action],
            query=query,
            query_index=query_index,
            source_index=source_index,
            dataset=dataset,
            runtime=runtimes[action],
            deadline_ms=deadline_ms,
        )
        for action in ACTIONS
    ]
    easy = next(row for row in rows if row["entry_action"] == FIXED_ALIAS_SOURCE)
    rows.append(_fixed_alias_record(easy))
    prepared = seed_engine.prepare(query)
    features = cached_risk_features(query, prepared, reuse_best_pose=True)
    return np.asarray(features, dtype=np.float64).copy(), rows


def _warmup(
    *,
    runtimes: Mapping[str, ProfiledCascadeRuntime],
    dataset: QueryDataset,
    selected: np.ndarray,
    iterations: int,
    dt: float,
) -> None:
    for index in range(iterations):
        query = query_from_dataset(dataset, int(selected[index % len(selected)]), dt=dt)
        actions = list(ACTIONS)
        if index % 2:
            actions.reverse()
        for action in actions:
            runtimes[action].solve(query)


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("xb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0) as compressed:
            for record in records:
                compressed.write(_json_bytes(record, pretty=False))
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _chunk_matrix(
    *,
    path: Path,
    features: list[np.ndarray],
    records: list[dict[str, Any]],
    selected_indices: np.ndarray,
    dataset: QueryDataset,
    query_start: int,
    nq: int,
    repeats: int,
) -> None:
    count = len(selected_indices)
    action_index = {action: index for index, action in enumerate(COLLECTED_ACTIONS)}
    shape = (count, len(COLLECTED_ACTIONS))
    success = np.zeros(shape, dtype=bool)
    deadline = np.zeros(shape, dtype=bool)
    latency_samples = np.zeros((*shape, repeats), dtype=np.int64)
    latency_p50 = np.zeros(shape, dtype=np.float64)
    latency_p95 = np.zeros(shape, dtype=np.float64)
    evaluations = np.zeros(shape, dtype=np.int64)
    fallback = np.zeros(shape, dtype=bool)
    aliases = np.zeros(shape, dtype=bool)
    commands = np.full((*shape, nq), np.nan, dtype=np.float64)
    failure = np.full(shape, "", dtype="U256")
    joint_step = np.full(shape, np.nan, dtype=np.float64)
    joint_velocity = np.full(shape, np.nan, dtype=np.float64)
    velocity_utilization = np.full(shape, np.nan, dtype=np.float64)
    query_hashes = np.full(count, "", dtype="U64")
    for row in records:
        local = int(row["query_index"]) - query_start
        action = action_index[str(row["entry_action"])]
        success[local, action] = bool(row["verified_success"])
        deadline[local, action] = bool(row["verified_success_before_deadline"])
        latency_samples[local, action] = np.asarray(row["latency_samples_ns"], dtype=np.int64)
        latency_p50[local, action] = float(row["latency_p50_ns"]) / 1e6
        latency_p95[local, action] = float(row["latency_p95_ns"]) / 1e6
        evaluations[local, action] = int(row["function_evaluations"])
        fallback[local, action] = bool(row["fallback_used"])
        aliases[local, action] = not bool(row["measurement_executed"])
        failure[local, action] = str(row["failure_reason"])
        query_hashes[local] = str(row["query_sha256"])
        if row["max_joint_step_rad"] is not None:
            joint_step[local, action] = float(row["max_joint_step_rad"])
            joint_velocity[local, action] = float(row["max_joint_velocity_rad_s"])
            velocity_utilization[local, action] = float(
                row["max_velocity_limit_utilization"]
            )
        if row["command_q"] is not None:
            commands[local, action] = np.asarray(row["command_q"], dtype=np.float64)
    _write_npz_atomic(
        path,
        feature_names=np.asarray(FEATURE_NAMES, dtype=np.str_),
        action_names=np.asarray(COLLECTED_ACTIONS, dtype=np.str_),
        decision_action_names=np.asarray(ACTIONS, dtype=np.str_),
        features=np.asarray(features, dtype=np.float64),
        query_indices=np.arange(query_start, query_start + count, dtype=np.int64),
        source_indices=np.asarray(selected_indices, dtype=np.int64),
        query_sha256=query_hashes,
        category=dataset.category[selected_indices],
        expected_reachable=dataset.expected_reachable[selected_indices],
        continuity_feasible=dataset.continuity_feasible[selected_indices],
        verified_success=success,
        verified_success_before_deadline=deadline,
        latency_samples_ns=latency_samples,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        function_evaluations=evaluations,
        fallback_used=fallback,
        measurement_is_alias=aliases,
        failure_reason=failure,
        command_q=commands,
        max_joint_step_rad=joint_step,
        max_joint_velocity_rad_s=joint_velocity,
        max_velocity_limit_utilization=velocity_utilization,
        max_joint_acceleration_rad_s2=np.full(shape, np.nan, dtype=np.float64),
        max_joint_jerk_rad_s3=np.full(shape, np.nan, dtype=np.float64),
        dynamic_history_available=np.zeros(shape, dtype=bool),
    )


def _chunk_name(start: int, stop: int) -> str:
    if start < 0 or stop <= start:
        raise ValueError("invalid half-open chunk interval")
    return f"chunk_{start:06d}_{stop - 1:06d}"


def _validate_chunk_directory(
    chunk_dir: Path,
    *,
    robot: str,
    training_seed: int,
    role: str,
    start: int,
    stop: int,
) -> dict[str, Any]:
    manifest_path = chunk_dir / "chunk_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"committed chunk lacks manifest: {chunk_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "robot": robot,
        "training_seed": training_seed,
        "source_role": role,
        "query_start": start,
        "query_stop_exclusive": stop,
    }
    actual_identity = {key: manifest.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"chunk identity mismatch at {chunk_dir}: {actual_identity} != {expected_identity}"
        )
    for name, metadata in manifest.get("artifacts", {}).items():
        path = chunk_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"chunk artifact missing: {path}")
        if path.stat().st_size != int(metadata["size"]):
            raise RuntimeError(f"chunk artifact size mismatch: {path}")
        if _sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"chunk artifact hash mismatch: {path}")
    payload = dict(manifest)
    claimed = str(payload.pop("chunk_payload_sha256", ""))
    if _digest_mapping(payload) != claimed:
        raise RuntimeError(f"chunk payload hash mismatch: {chunk_dir}")
    return manifest


def _commit_chunk(
    *,
    role_root: Path,
    robot: str,
    training_seed: int,
    role: str,
    dataset: QueryDataset,
    selected: np.ndarray,
    query_start: int,
    query_stop: int,
    runtimes: Mapping[str, ProfiledCascadeRuntime],
    seed_engine: SeedEngine,
    config: Mapping[str, Any],
    selection_seed: int,
) -> dict[str, Any]:
    name = _chunk_name(query_start, query_stop)
    final = role_root / "chunks" / name
    if final.exists():
        return _validate_chunk_directory(
            final,
            robot=robot,
            training_seed=training_seed,
            role=role,
            start=query_start,
            stop=query_stop,
        )
    attempt = role_root / f".{name}.incomplete.{int(time.time_ns())}.{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        repeats = int(config["timing"]["repeats"])
        deadline_ms = float(config["timing"]["deadline_ms"])
        dt = float(config["data"]["dt"])
        features: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        quiet_check_count = 0
        quiet_wait_events: list[dict[str, Any]] = []
        contaminated_attempt_events: list[dict[str, Any]] = []
        contaminated_query_retries = 0
        started = time.perf_counter()
        for query_index in range(query_start, query_stop):
            source_index = int(selected[query_index])
            query = query_from_dataset(dataset, source_index, dt=dt)
            attempt_index = 0
            while True:
                quiet_event = _wait_for_quiet_environment(
                    config,
                    context=(
                        f"{robot}/seed{training_seed}/{role}/{name}/query{query_index}/"
                        f"attempt{attempt_index}/before"
                    ),
                )
                quiet_check_count += 1
                if quiet_event["had_busy_process"]:
                    quiet_wait_events.append(
                        {
                            "query_index": query_index,
                            "attempt_index": attempt_index,
                            **quiet_event,
                        }
                    )
                feature, rows = _collect_query(
                    query=query,
                    query_index=query_index,
                    source_index=source_index,
                    dataset=dataset,
                    runtimes=runtimes,
                    seed_engine=seed_engine,
                    repeats=repeats,
                    deadline_ms=deadline_ms,
                    order_seed=selection_seed + query_index,
                )
                busy_after = _busy_unrelated_processes(
                    cpu_threshold_percent=float(
                        config["runtime"]["max_unrelated_cpu_percent"]
                    )
                )
                if not busy_after:
                    break
                # Neither the feature nor any timing rows from this attempt
                # enter the chunk. Preserve the contamination audit, regain a
                # quiet host, and recollect the complete query/action block.
                contaminated_query_retries += 1
                contaminated_attempt_events.append(
                    {
                        "query_index": query_index,
                        "source_index": source_index,
                        "attempt_index": attempt_index,
                        "detected_utc": _utc(),
                        "busy_processes": busy_after,
                        "feature_and_rows_discarded": True,
                    }
                )
                print(
                    "[counterfactual-v4-bulk] discarding contaminated query attempt "
                    f"{robot}/seed{training_seed}/{role}/{name}/query{query_index}/"
                    f"attempt{attempt_index}: {busy_after}",
                    flush=True,
                )
                attempt_index += 1
            digest = query_digest(query)
            for row in rows:
                row.update(
                    {
                        "robot": robot,
                        "training_seed": training_seed,
                        "source_role": role,
                        "query_sha256": digest,
                        "source_query_sha256": digest,
                        "risk_features": feature.tolist(),
                        "chunk_name": name,
                    }
                )
            features.append(feature)
            records.extend(rows)
        elapsed = time.perf_counter() - started
        # This final guard closes the narrow interval between the last
        # per-query after-check and chunk commit.  A dirty chunk is preserved
        # as an incomplete attempt and can never be atomically committed.
        post_busy = _busy_unrelated_processes(
            cpu_threshold_percent=float(config["runtime"]["max_unrelated_cpu_percent"])
        )
        if post_busy:
            raise RuntimeError(
                "refusing to commit an environment-contaminated latency chunk; "
                f"post_chunk_busy_processes={post_busy}"
            )
        records_path = attempt / "counterfactual_records.jsonl.gz"
        labels_path = attempt / "counterfactual_labels.npz"
        _write_records(records_path, records)
        _chunk_matrix(
            path=labels_path,
            features=features,
            records=records,
            selected_indices=selected[query_start:query_stop],
            dataset=dataset,
            query_start=query_start,
            nq=int(runtimes["easy"].kinematics.nq),
            repeats=repeats,
        )
        action_summary: dict[str, Any] = {}
        for action in COLLECTED_ACTIONS:
            rows = [row for row in records if row["entry_action"] == action]
            action_summary[action] = {
                "query_count": len(rows),
                "verified_success_count": int(sum(bool(row["verified_success"]) for row in rows)),
                "verified_success_before_deadline_count": int(
                    sum(bool(row["verified_success_before_deadline"]) for row in rows)
                ),
                "function_evaluations_sum": int(
                    sum(int(row["function_evaluations"]) for row in rows)
                ),
                "fallback_count": int(sum(bool(row["fallback_used"]) for row in rows)),
                "failure_reason_counts": dict(
                    Counter(str(row["failure_reason"]) for row in rows if row["failure_reason"])
                ),
                "measurement_mode": "semantic_alias"
                if action == FIXED_ALIAS_ACTION
                else "executed",
            }
        artifacts = {
            path.name: {"sha256": _sha256_file(path), "size": path.stat().st_size}
            for path in (records_path, labels_path)
        }
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "created_utc": _utc(),
            "robot": robot,
            "training_seed": training_seed,
            "source_role": role,
            "query_start": query_start,
            "query_stop_exclusive": query_stop,
            "query_count": query_stop - query_start,
            "record_count": len(records),
            "decision_action_execution_count": (query_stop - query_start) * len(ACTIONS) * repeats,
            "fixed_robust_execution_count": 0,
            "fixed_robust_measurement_mode": "semantic_alias_of_easy",
            "wall_time_seconds_excluding_warmup_and_writes": elapsed,
            "seconds_per_query": elapsed / (query_stop - query_start),
            "environment_contaminated": False,
            "post_chunk_busy_processes": [],
            "quiet_check_count": quiet_check_count,
            "quiet_wait_event_count": len(quiet_wait_events),
            "quiet_wait_seconds": float(
                sum(float(event["wait_seconds"]) for event in quiet_wait_events)
            ),
            "quiet_wait_events": quiet_wait_events,
            "contaminated_query_retries": contaminated_query_retries,
            "contaminated_attempt_events": contaminated_attempt_events,
            "action_summary": action_summary,
            "artifacts": artifacts,
            "test_data_loaded": False,
        }
        manifest["chunk_payload_sha256"] = _digest_mapping(manifest)
        _atomic_json(attempt / "chunk_manifest.json", manifest)
        os.replace(attempt, final)
        return _validate_chunk_directory(
            final,
            robot=robot,
            training_seed=training_seed,
            role=role,
            start=query_start,
            stop=query_stop,
        )
    except BaseException:
        _append_failure(
            attempt / "failure_event.jsonl",
            {
                "failed_utc": _utc(),
                "robot": robot,
                "training_seed": training_seed,
                "source_role": role,
                "query_start": query_start,
                "query_stop_exclusive": query_stop,
                "traceback": traceback.format_exc(),
                "partial_attempt_preserved": True,
                "test_data_loaded": False,
            },
        )
        raise


def _chunk_intervals(count: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, count)) for start in range(0, count, chunk_size)]


def _scan_progress(output_root: Path, manifest: Mapping[str, Any], chunk_size: int) -> dict[str, Any]:
    combinations: dict[str, Any] = {}
    completed_chunks = 0
    completed_queries = 0
    contaminated_chunks = 0
    for key, selection in manifest["selections"].items():
        robot, seed_text, role = key.split("/")
        seed = int(seed_text.removeprefix("seed"))
        role_root = output_root / robot / seed_text / role
        count = int(selection["selected_query_count"])
        completed = 0
        for start, stop in _chunk_intervals(count, chunk_size):
            chunk = role_root / "chunks" / _chunk_name(start, stop)
            if not chunk.exists():
                continue
            details = _validate_chunk_directory(
                chunk,
                robot=robot,
                training_seed=seed,
                role=role,
                start=start,
                stop=stop,
            )
            completed += stop - start
            completed_chunks += 1
            completed_queries += stop - start
            contaminated_chunks += int(bool(details["environment_contaminated"]))
        combinations[key] = {"completed_queries": completed, "total_queries": count}
    return {
        "completed_chunk_count": completed_chunks,
        "completed_query_count": completed_queries,
        "total_query_count": int(sum(item["selected_query_count"] for item in manifest["selections"].values())),
        "environment_contaminated_chunk_count": contaminated_chunks,
        "combinations": combinations,
    }


def _refresh_manifest(
    output_root: Path,
    manifest: dict[str, Any],
    *,
    chunk_size: int,
    status: str,
) -> dict[str, Any]:
    progress = _scan_progress(output_root, manifest, chunk_size)
    manifest.update(progress)
    manifest["status"] = status
    manifest["updated_utc"] = _utc()
    _atomic_json(output_root / "run_manifest.json", manifest)
    return manifest


def _aggregate_completed(output_root: Path, manifest: Mapping[str, Any], chunk_size: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"protocol": PROTOCOL, "combinations": {}}
    for key, selection in manifest["selections"].items():
        robot, seed_text, role = key.split("/")
        seed = int(seed_text.removeprefix("seed"))
        role_root = output_root / robot / seed_text / role
        count = int(selection["selected_query_count"])
        action_totals: dict[str, Counter[str]] = {action: Counter() for action in COLLECTED_ACTIONS}
        elapsed = 0.0
        for start, stop in _chunk_intervals(count, chunk_size):
            chunk = role_root / "chunks" / _chunk_name(start, stop)
            details = _validate_chunk_directory(
                chunk,
                robot=robot,
                training_seed=seed,
                role=role,
                start=start,
                stop=stop,
            )
            elapsed += float(details["wall_time_seconds_excluding_warmup_and_writes"])
            for action, values in details["action_summary"].items():
                for field in (
                    "query_count",
                    "verified_success_count",
                    "verified_success_before_deadline_count",
                    "function_evaluations_sum",
                    "fallback_count",
                ):
                    action_totals[action][field] += int(values[field])
        actions: dict[str, Any] = {}
        for action, totals in action_totals.items():
            query_count = totals["query_count"]
            actions[action] = {
                **dict(totals),
                "verified_success_rate": totals["verified_success_count"] / query_count,
                "verified_success_before_deadline_rate": totals[
                    "verified_success_before_deadline_count"
                ]
                / query_count,
                "mean_function_evaluations": totals["function_evaluations_sum"] / query_count,
                "fallback_rate": totals["fallback_count"] / query_count,
                "measurement_mode": "semantic_alias"
                if action == FIXED_ALIAS_ACTION
                else "executed",
            }
        summary["combinations"][key] = {
            "query_count": count,
            "wall_time_seconds_excluding_warmup_and_writes": elapsed,
            "seconds_per_query": elapsed / count,
            "actions": actions,
        }
    summary["test_data_loaded"] = False
    summary["created_utc"] = _utc()
    return summary


def _acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another v4 bulk collector holds {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_utc={_utc()}\n")
    handle.flush()
    return handle


def main() -> None:
    args = _parser().parse_args()
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be positive")
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("bulk v4 config must be a mapping")
    config["_config_path"] = str(config_path)
    _validate_config(config)
    workspace = resolve_path(config, config.get("workspace", ".."))
    source_config_path = resolve_path(config, config["source_config"])
    source_config = load_config(source_config_path)
    release_root = resolve_path(config, config["release_root"])
    output_root = resolve_path(config, config["output_root"])
    _assert_output_scope(workspace, output_root)
    # Keep the handle strongly referenced for the complete process lifetime;
    # closing it would release the advisory lock immediately.
    lock_handle = _acquire_lock(workspace / "outputs" / ".counterfactual_v4_bulk.lock")

    release_manifest_path = release_root / "release_manifest.json"
    release_equivalence_path = release_root / "release_equivalence.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    equivalence = json.loads(release_equivalence_path.read_text(encoding="utf-8"))
    if release_manifest.get("release_status") != "sealed" or not equivalence.get("all_six_pass"):
        raise RuntimeError("bulk v4 labels require the sealed six-pass exact v3 release")
    release_commit = str(release_manifest["git_commit"])
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))

    protected = _protected_snapshot(workspace)
    frozen = _frozen_provenance(
        workspace=workspace,
        config_path=config_path,
        config=config,
        source_config_path=source_config_path,
        release_root=release_root,
        release_commit=release_commit,
    )
    if output_root.exists():
        manifest = _load_manifest(output_root)
        _validate_frozen_provenance(manifest["frozen_provenance"], frozen)
        if manifest["protected_baseline"] != protected:
            raise RuntimeError("a protected v2/v3/czy artifact changed since bulk initialization")
    else:
        manifest = _initialize_output(
            workspace=workspace,
            output_root=output_root,
            config_path=config_path,
            config=config,
            release_commit=release_commit,
            frozen=frozen,
            protected=protected,
        )

    # A completed collection is immutable.  Validate every committed chunk via
    # the progress scan below, then return without regenerating summaries or
    # timestamps.
    already_complete = manifest.get("status") == "complete"

    chunk_size = int(config["bulk"]["chunk_size"])
    if already_complete:
        progress = _scan_progress(output_root, manifest, chunk_size)
        if progress["completed_query_count"] != progress["total_query_count"]:
            raise RuntimeError("complete manifest has missing chunks")
        print(f"[counterfactual-v4-bulk] already complete and verified: {output_root}")
        lock_handle.close()
        return
    manifest = _refresh_manifest(output_root, manifest, chunk_size=chunk_size, status="in_progress")
    newly_completed = 0
    active_context: dict[str, Any] = {}
    try:
        for robot in config["robots"]:
            for training_seed_value in config["training_seeds"]:
                training_seed = int(training_seed_value)
                seed_engine: SeedEngine | None = None
                runtimes: dict[str, ProfiledCascadeRuntime] | None = None
                warmed = False
                kinematics = None
                for role in ROLE_ORDER:
                    key = f"{robot}/seed{training_seed}/{role}"
                    role_root = output_root / str(robot) / f"seed{training_seed}" / role
                    selected = _selection_for_role(role_root, manifest["selections"][key])
                    dataset_path = _source_dataset_path(
                        workspace,
                        robot=str(robot),
                        training_seed=training_seed,
                        role=role,
                    )
                    dataset = QueryDataset.load(dataset_path)
                    if len(selected) > len(dataset):
                        raise RuntimeError(f"selection exceeds dataset length: {key}")
                    selection_seed = int(manifest["selections"][key]["selection_seed"])
                    for start, stop in _chunk_intervals(len(selected), chunk_size):
                        final = role_root / "chunks" / _chunk_name(start, stop)
                        if final.exists():
                            _validate_chunk_directory(
                                final,
                                robot=str(robot),
                                training_seed=training_seed,
                                role=role,
                                start=start,
                                stop=stop,
                            )
                            continue
                        if runtimes is None or seed_engine is None:
                            kinematics = load_robot(source_config, str(robot))
                            built_seed, built_runtimes, _ = _build_runtimes(
                                source_config=source_config,
                                release_root=release_root,
                                robot=str(robot),
                                training_seed=training_seed,
                                kinematics=kinematics,
                                device=str(config["runtime"]["device"]),
                            )
                            seed_engine = built_seed  # type: ignore[assignment]
                            runtimes = {
                                action: built_runtimes[action] for action in ACTIONS
                            }
                        if not warmed:
                            _wait_for_quiet_environment(
                                config,
                                context=f"{robot}/seed{training_seed}/{role}/warmup",
                            )
                            _warmup(
                                runtimes=runtimes,
                                dataset=dataset,
                                selected=selected,
                                iterations=int(config["timing"]["warmup_iterations"]),
                                dt=float(config["data"]["dt"]),
                            )
                            warmed = True
                        _wait_for_quiet_environment(
                            config,
                            context=f"{robot}/seed{training_seed}/{role}/{_chunk_name(start, stop)}",
                        )
                        active_context = {
                            "robot": str(robot),
                            "training_seed": training_seed,
                            "source_role": role,
                            "query_start": start,
                            "query_stop_exclusive": stop,
                        }
                        details = _commit_chunk(
                            role_root=role_root,
                            robot=str(robot),
                            training_seed=training_seed,
                            role=role,
                            dataset=dataset,
                            selected=selected,
                            query_start=start,
                            query_stop=stop,
                            runtimes=runtimes,
                            seed_engine=seed_engine,
                            config=config,
                            selection_seed=selection_seed,
                        )
                        newly_completed += 1
                        manifest = _refresh_manifest(
                            output_root,
                            manifest,
                            chunk_size=chunk_size,
                            status="in_progress",
                        )
                        print(
                            "[counterfactual-v4-bulk] committed "
                            f"{robot}/seed{training_seed}/{role}/{_chunk_name(start, stop)} "
                            f"({manifest['completed_query_count']}/{manifest['total_query_count']} queries, "
                            f"{details['wall_time_seconds_excluding_warmup_and_writes']:.1f}s)",
                            flush=True,
                        )
                        if args.max_chunks is not None and newly_completed >= args.max_chunks:
                            _refresh_manifest(
                                output_root,
                                manifest,
                                chunk_size=chunk_size,
                                status="paused_by_max_chunks",
                            )
                            print(
                                f"[counterfactual-v4-bulk] clean operational pause after {newly_completed} chunks",
                                flush=True,
                            )
                            return

        if _protected_snapshot(workspace) != manifest["protected_baseline"]:
            raise RuntimeError("a protected v2/v3/czy artifact changed during bulk collection")
        summary = _aggregate_completed(output_root, manifest, chunk_size)
        _atomic_json(output_root / "bulk_summary.json", summary)
        _atomic_json(
            output_root / "environment_final.json",
            {
                **environment_payload(),
                "captured_utc": _utc(),
                "python_executable": os.sys.executable,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            },
        )
        manifest = _refresh_manifest(
            output_root, manifest, chunk_size=chunk_size, status="complete"
        )
        manifest["completed_utc"] = _utc()
        manifest["bulk_summary_sha256"] = _sha256_file(output_root / "bulk_summary.json")
        manifest["environment_final_sha256"] = _sha256_file(
            output_root / "environment_final.json"
        )
        manifest["protected_outputs_unchanged"] = True
        _atomic_json(output_root / "run_manifest.json", manifest)
        print(f"[counterfactual-v4-bulk] complete output={output_root}", flush=True)
    except BaseException as exc:
        failure = {
            "failed_utc": _utc(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "active_context": active_context,
            "test_data_loaded": False,
            "resume_supported": True,
        }
        _append_failure(output_root / "failure_events.jsonl", failure)
        manifest["failure_event_count"] = int(manifest.get("failure_event_count", 0)) + 1
        manifest["last_failure"] = failure
        _refresh_manifest(
            output_root, manifest, chunk_size=chunk_size, status="interrupted_resumable"
        )
        raise


if __name__ == "__main__":
    main()
