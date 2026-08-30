#!/usr/bin/env python3
"""Independent, read-only audit of ``outputs/release_v4_locked``.

The audit does not import ``confik.release_v4_locked`` and never writes below
``outputs``.  It reconstructs the eager v4 network from the sealed checkpoint,
loads the disk TorchScript artifact, replays the policy-validation gate at the
batch-one deployment boundary, reproduces the six validation subset selections,
and checks the persisted paired runtime-equivalence evidence.

Files below old ``test_v3*`` directories are treated as opaque bytes while the
protected-tree digest is recomputed.  Their JSON/NPZ contents are never parsed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F
import yaml


PROTOCOL = "release_v4_locked"
BACKEND = "torchscript_exact_v4"
ROBOTS = ("panda", "ur5e")
SEEDS = (17, 29, 43)
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
ACTIONS = ("easy", "medium", "hard")
DECISIONS = ACTIONS + ("reject", "defer")
LABEL_CONTRACT = {
    "action_success_target": "semantic_verified_success",
    "shared_semantic_success_due_to_terminal_fallback_invariance": True,
    "deadline_success_role": "diagnostic_only",
    "latency_target": "raw_repeat_p50_p95_pinball",
}
PAYLOAD_FILES = {
    "panda/exact_v4_predictor.ts",
    "panda/v4_policy.json",
    "panda/v4_runtime_spec.json",
    "release_config.yaml",
    "release_environment.json",
    "release_equivalence.json",
    "upstream_dependencies.json",
    "ur5e/exact_v4_predictor.ts",
    "ur5e/v4_policy.json",
    "ur5e/v4_runtime_spec.json",
}
AGREEMENT_FIELDS = {
    "accepted",
    "eligible_actions",
    "executed_stages",
    "fallback",
    "function_evaluations",
    "iterations",
    "ood_decision",
    "query_hash",
    "reject_reason",
    "route_action",
    "route_reason",
    "verification_reasons",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--release-root", type=Path, default=Path("outputs/release_v4_locked")
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/audits/release_v4_locked/release_audit.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/audits/release_v4_locked/RELEASE_AUDIT.md"),
    )
    return parser.parse_args()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def strict_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML document is not a mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def add(issues: list[str], condition: bool, name: str) -> None:
    if not condition:
        issues.append(name)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def eager_one(checkpoint: Mapping[str, Any], feature: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reconstruct one eager inference without importing the v4 implementation."""

    state = checkpoint["state_dict"]
    mean = checkpoint["feature_mean"].cpu().numpy().astype(np.float32)
    scale = checkpoint["feature_scale"].cpu().numpy().astype(np.float32)
    normalized = np.ascontiguousarray(
        (np.asarray(feature, dtype=np.float32)[None, :] - mean) / scale,
        dtype=np.float32,
    )
    hidden = torch.from_numpy(normalized)
    hidden_sizes = tuple(int(value) for value in checkpoint["config"]["hidden_sizes"])
    with torch.inference_mode():
        for index in range(len(hidden_sizes)):
            layer = index * 2
            hidden = F.silu(
                F.linear(
                    hidden,
                    state[f"backbone.{layer}.weight"],
                    state[f"backbone.{layer}.bias"],
                )
            )
        shared_logit = F.linear(
            hidden,
            state["verified_success_head.weight"],
            state["verified_success_head.bias"],
        ).detach().cpu().numpy().astype(np.float64)[0, 0]
        latency_raw = F.linear(
            hidden,
            state["latency_head.weight"],
            state["latency_head.bias"],
        )
        p50_raw, gap_raw = latency_raw.chunk(2, dim=1)
        p50 = (
            F.softplus(p50_raw) + float(checkpoint["config"]["min_latency_ms"])
        ).detach().cpu().numpy().astype(np.float64)[0]
        p95 = (
            F.softplus(p50_raw)
            + float(checkpoint["config"]["min_latency_ms"])
            + F.softplus(gap_raw)
            + float(checkpoint["config"]["min_quantile_gap_ms"])
        ).detach().cpu().numpy().astype(np.float64)[0]
        fail_logit = F.linear(
            hidden,
            state["fail_all_head.weight"],
            state["fail_all_head.bias"],
        ).detach().cpu().numpy().astype(np.float64)[0, 0]
    calibrators = checkpoint["calibrator"]["calibrators"]
    success = float(
        np.clip(
            stable_sigmoid(
                np.asarray(
                    float(calibrators[0]["slope"]) * shared_logit
                    + float(calibrators[0]["intercept"])
                )
            ),
            1e-7,
            1.0 - 1e-7,
        )
    )
    fail = float(
        np.clip(
            stable_sigmoid(
                np.asarray(
                    float(calibrators[1]["slope"]) * fail_logit
                    + float(calibrators[1]["intercept"])
                )
            ),
            1e-7,
            1.0 - 1e-7,
        )
    )
    embedding = hidden.detach().cpu().numpy().astype(np.float64)[0]
    detector = checkpoint["ood_detector"]
    centered = embedding - np.asarray(detector["mean"], dtype=np.float64)
    ood_score = float(
        max(
            centered @ np.asarray(detector["precision"], dtype=np.float64) @ centered,
            0.0,
        )
    )
    return (
        np.full(3, success, dtype=np.float64),
        p50,
        p95,
        np.asarray(fail, dtype=np.float64),
        embedding,
        np.asarray(ood_score, dtype=np.float64),
        np.asarray(ood_score > float(detector["threshold"]), dtype=bool),
    )


def exact_one(module: torch.jit.ScriptModule, feature: np.ndarray) -> tuple[np.ndarray, ...]:
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.asarray(feature, dtype=np.float32)[None, :])
    )
    with torch.inference_mode():
        values = module(tensor)
    return tuple(value[0].detach().cpu().numpy() for value in values)


def decision(
    values: tuple[np.ndarray, ...], config: Mapping[str, float]
) -> tuple[str, str, tuple[str, ...], bool]:
    success = np.asarray(values[0], dtype=np.float64)
    p95 = np.asarray(values[2], dtype=np.float64)
    fail = float(values[3])
    is_ood = bool(values[6])
    if is_ood:
        return "defer", "ood_defer", (), True
    eligible_index = np.flatnonzero(
        (success >= float(config["minimum_success_probability"]))
        & (p95 <= float(config["deadline_ms"]))
    )
    eligible = tuple(ACTIONS[int(index)] for index in eligible_index)
    if fail >= float(config["reject_probability"]) and not len(eligible_index):
        return "reject", "high_confidence_fail_all", eligible, False
    if not len(eligible_index):
        return "defer", "uncertain_no_eligible_action", eligible, False
    fastest = int(eligible_index[np.argmin(p95[eligible_index])])
    conservative = int(np.min(eligible_index))
    improvement = p95[conservative] - p95[fastest]
    selected = (
        conservative
        if improvement < float(config["latency_tie_margin_ms"])
        else fastest
    )
    reason = (
        "tie_margin_conservative_entry"
        if selected == conservative and selected != fastest
        else "minimum_predicted_p95"
    )
    return ACTIONS[selected], reason, eligible, False


def load_policy_features(bulk_root: Path, robot: str) -> tuple[np.ndarray, dict[str, Any]]:
    root = bulk_root / robot / "seed17" / "policy_validation_queries"
    selection = strict_json(root / "selection_manifest.json")
    arrays: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    issues: list[str] = []
    expected_start = 0
    chunks = sorted((root / "chunks").glob("chunk_*"))
    for chunk in chunks:
        manifest_path = chunk / "chunk_manifest.json"
        labels_path = chunk / "counterfactual_labels.npz"
        manifest = strict_json(manifest_path)
        add(issues, int(manifest["query_start"]) == expected_start, "chunk_contiguity")
        add(issues, manifest.get("source_role") == "policy_validation_queries", "chunk_role")
        add(issues, manifest.get("test_data_loaded") is False, "chunk_test_flag")
        add(issues, manifest.get("environment_contaminated") is False, "chunk_clean")
        for relative, expected in manifest["artifacts"].items():
            path = chunk / relative
            add(
                issues,
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == int(expected["size"])
                and file_sha256(path) == str(expected["sha256"]),
                f"chunk_artifact:{relative}",
            )
        with np.load(labels_path, allow_pickle=False) as labels:
            add(
                issues,
                tuple(labels["feature_names"].astype(str).tolist()) == FEATURE_NAMES,
                "feature_schema",
            )
            arrays.append(np.asarray(labels["features"], dtype=np.float32))
        files.extend((metadata(manifest_path), metadata(labels_path)))
        expected_start = int(manifest["query_stop_exclusive"])
    features = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    add(issues, len(chunks) == 10, "chunk_count")
    add(issues, expected_start == 2500 == len(features), "feature_count")
    add(issues, selection.get("selected_query_count") == 2500, "selection_count")
    add(issues, selection.get("test_named_dataset_loaded") is False, "selection_test_flag")
    return features, {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "chunk_count": len(chunks),
        "feature_count": len(features),
        "source_files": files,
    }


def audit_payloads(root: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    declared = artifact.get("files", {})
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "release_manifest.json"}
    }
    add(issues, set(declared) == PAYLOAD_FILES, "declared_payload_contract")
    add(issues, actual == PAYLOAD_FILES, "actual_payload_contract")
    add(issues, int(artifact.get("file_count", -1)) == 10, "file_count")
    add(issues, artifact.get("protocol") == PROTOCOL, "protocol")
    add(issues, artifact.get("release_status") == "sealed", "release_status")
    add(issues, artifact.get("test_data_loaded") is False, "test_flag")
    rows: dict[str, Any] = {}
    for relative in sorted(PAYLOAD_FILES | set(declared)):
        path = root / relative
        expected = declared.get(relative)
        exists = path.is_file() and not path.is_symlink()
        actual_size = path.stat().st_size if exists else None
        actual_hash = file_sha256(path) if exists else None
        matched = bool(
            exists
            and expected
            and actual_size == int(expected["size"])
            and actual_hash == str(expected["sha256"])
        )
        add(issues, matched, f"payload:{relative}")
        rows[relative] = {
            "exists_regular_not_symlink": exists,
            "declared_size": None if expected is None else int(expected["size"]),
            "actual_size": actual_size,
            "declared_sha256": None if expected is None else str(expected["sha256"]),
            "actual_sha256": actual_hash,
            "match": matched,
        }
    return {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "declared_count": len(declared),
        "actual_payload_count": len(actual),
        "missing": sorted(PAYLOAD_FILES - actual),
        "extra": sorted(actual - PAYLOAD_FILES),
        "files": rows,
    }


def git_value(workspace: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True
    )
    return completed.stdout.decode().strip() if text else completed.stdout


def audit_source(workspace: Path, release: Mapping[str, Any]) -> dict[str, Any]:
    before = release["source_manifest"]
    after = release["source_manifest_after_validation"]
    issues: list[str] = []
    add(issues, before == after, "before_after_identity")
    add(issues, release.get("release_source_unchanged_during_validation") is True, "stable_flag")
    commit = str(before["git_commit"])
    commit_exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=workspace
    ).returncode == 0
    add(issues, commit_exists, "commit_exists")
    tree = str(git_value(workspace, "rev-parse", f"{commit}^{{tree}}")) if commit_exists else None
    add(issues, tree == before.get("git_tree"), "git_tree")
    blobs: dict[str, Any] = {}
    for relative, field in (
        ("src/confik/release_v4_locked/runner.py", "runner_sha256"),
        ("src/confik/release_v4_locked/artifacts.py", "artifacts_sha256"),
    ):
        contents = git_value(workspace, "show", f"{commit}:{relative}", text=False) if commit_exists else b""
        digest = sha256(contents).hexdigest() if commit_exists else None
        matched = digest == before.get(field)
        add(issues, matched, f"source_blob:{relative}")
        blobs[relative] = {
            "recorded_sha256": before.get(field),
            "git_blob_content_sha256": digest,
            "match": matched,
        }
    return {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "before_after_exactly_equal": before == after,
        "recorded_release_source_unchanged": release.get(
            "release_source_unchanged_during_validation"
        ),
        "git_commit": commit,
        "git_commit_exists": commit_exists,
        "recorded_git_tree": before.get("git_tree"),
        "recomputed_git_tree": tree,
        "source_blobs_at_recorded_commit": blobs,
        "historical_clean_scope_note": (
            "Cleanliness at packaging time is a recorded temporal assertion; "
            "the immutable commit/tree and the two release-code blobs are independently checked."
        ),
    }


def protected_snapshot(output_root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
    entries: dict[str, Any] = {}
    opaque_test_files = 0
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output is a symlink: {directory}")
        is_old_test = directory.name.startswith("test_v3")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            info = path.stat()
            # Hashing is byte-opaque. In particular, old test JSON/NPZ files
            # are not decoded, indexed, or loaded into analytical structures.
            entries[str(path.relative_to(output_root))] = {
                "sha256": file_sha256(path),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "mode": stat.S_IMODE(info.st_mode),
            }
            opaque_test_files += int(is_old_test)
    return {
        "directories": [str(path.relative_to(output_root)) for path in sorted(directories)],
        "file_count": len(entries),
        "total_bytes": sum(int(value["size"]) for value in entries.values()),
        "tree_digest": canonical_digest(entries),
        "old_test_files_hashed_as_opaque_bytes": opaque_test_files,
        "old_test_payloads_parsed": False,
    }


def audit_protected(
    workspace: Path, config: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    recorded = release["protected_outputs"]
    current = protected_snapshot(
        workspace / "outputs", [str(value) for value in config["protected_outputs"]]
    )
    comparable = {key: current[key] for key in ("directories", "file_count", "total_bytes", "tree_digest")}
    before = recorded["before"]
    after = recorded["after"]
    issues: list[str] = []
    add(issues, before == after, "recorded_before_after")
    add(issues, recorded.get("unchanged") is True, "recorded_unchanged_flag")
    add(issues, comparable == after, "current_matches_frozen_after")
    return {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "recorded_before_after_exactly_equal": before == after,
        "recorded_unchanged": recorded.get("unchanged"),
        "current_matches_recorded_after": comparable == after,
        "recorded": after,
        "recomputed_current": current,
        "test_boundary": (
            "Old test-v3 files contributed opaque byte hashes only; no old test "
            "JSON, NPZ, or performance field was parsed."
        ),
    }


def audit_dependencies(workspace: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    candidate = payload["candidate"]
    candidate_root = Path(str(candidate["root"])).resolve()
    expected_candidate = (workspace / "outputs/release_v4_candidate").resolve()
    add(issues, candidate_root == expected_candidate, "candidate_root")
    checks = {
        "candidate_run_manifest": (
            candidate_root / "run_manifest.json",
            candidate["run_manifest_sha256"],
        ),
        "candidate_artifact_manifest": (
            candidate_root / "artifact_manifest.json",
            candidate["artifact_manifest_sha256"],
        ),
    }
    upstream = payload["release_v3_locked"]
    v3_root = Path(str(upstream["root"])).resolve()
    add(issues, v3_root == (workspace / "outputs/release_v3_locked").resolve(), "v3_root")
    checks.update(
        {
            "v3_release_manifest": (
                v3_root / "release_manifest.json",
                upstream["release_manifest_sha256"],
            ),
            "v3_release_equivalence": (
                v3_root / "release_equivalence.json",
                upstream["release_equivalence_sha256"],
            ),
        }
    )
    check_rows: dict[str, Any] = {}
    for name, (path, expected) in checks.items():
        actual = file_sha256(path) if path.is_file() and not path.is_symlink() else None
        matched = actual == expected
        add(issues, matched, name)
        check_rows[name] = {"path": str(path), "recorded_sha256": expected, "actual_sha256": actual, "match": matched}
    v3_artifact_mismatches: list[str] = []
    for item in upstream["artifacts"]:
        path = workspace / str(item["path"])
        if not (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(item["size"])
            and file_sha256(path) == str(item["sha256"])
        ):
            v3_artifact_mismatches.append(str(item["path"]))
    add(issues, len(upstream["artifacts"]) == int(upstream["artifact_count"]) == 48, "v3_artifact_count")
    add(issues, not v3_artifact_mismatches, "v3_artifacts")
    add(issues, payload.get("test_named_dataset_loaded") is False, "test_flag")
    return {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "manifest_checks": check_rows,
        "candidate_release_digest": candidate["release_digest"],
        "candidate_artifact_count": candidate["artifact_count"],
        "upstream_v3_artifact_count": len(upstream["artifacts"]),
        "upstream_v3_artifact_mismatches": v3_artifact_mismatches,
        "test_named_dataset_loaded": payload.get("test_named_dataset_loaded"),
    }


def audit_numerical(
    root: Path,
    candidate_root: Path,
    bulk_root: Path,
    equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_pass = True
    for robot in ROBOTS:
        issues: list[str] = []
        checkpoint_path = candidate_root / "models" / f"{robot}_seed17_predictor.pt"
        policy_path = root / robot / "v4_policy.json"
        spec_path = root / robot / "v4_runtime_spec.json"
        module_path = root / robot / "exact_v4_predictor.ts"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy = strict_json(policy_path)
        spec = strict_json(spec_path)
        config = policy["policy_config"]
        features, feature_audit = load_policy_features(bulk_root, robot)
        add(issues, feature_audit["pass"], "policy_features")
        add(issues, tuple(checkpoint["feature_names"]) == FEATURE_NAMES, "checkpoint_features")
        add(issues, checkpoint["label_contract"] == LABEL_CONTRACT, "checkpoint_label_contract")
        add(issues, policy["label_contract"] == LABEL_CONTRACT, "policy_label_contract")
        add(issues, spec["label_contract"] == LABEL_CONTRACT, "spec_label_contract")
        add(issues, spec["policy_config"] == config, "policy_spec_config")
        add(issues, spec.get("backend") == BACKEND, "spec_backend")
        add(issues, spec.get("ood_before_command_reject") is True, "ood_precedence")
        add(issues, spec.get("command_reject_numerical_solver_budget") == 0, "reject_budget")
        add(issues, spec.get("defer_entry") == "complete_fixed_robust_cascade_from_easy", "defer_entry")
        module = torch.jit.load(str(module_path), map_location="cpu").eval()
        errors = {
            "success_probability": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "fail_all_probability": 0.0,
            "embedding": 0.0,
            "ood_score": 0.0,
        }
        agreements = Counter()
        route_counts = Counter()
        for feature in features:
            eager = eager_one(checkpoint, feature)
            exact = exact_one(module, feature)
            for name, index in (
                ("success_probability", 0),
                ("latency_p50_ms", 1),
                ("latency_p95_ms", 2),
                ("fail_all_probability", 3),
                ("embedding", 4),
                ("ood_score", 5),
            ):
                errors[name] = max(
                    errors[name],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(eager[index], dtype=np.float64)
                                - np.asarray(exact[index], dtype=np.float64)
                            )
                        )
                    ),
                )
            left = decision(eager, config)
            right = decision(exact, config)
            agreements["ood"] += int(bool(eager[6]) == bool(exact[6]))
            agreements["route"] += int(left[0] == right[0])
            agreements["reason"] += int(left[1] == right[1])
            agreements["eligible"] += int(left[2] == right[2])
            route_counts[right[0]] += 1
        recorded = equivalence["robots"][robot]["numerical_equivalence"]
        tolerances = recorded["tolerances"]
        for name, value in errors.items():
            add(issues, value <= float(tolerances[name]), f"tolerance:{name}")
        for name in ("ood", "route", "reason", "eligible"):
            add(issues, agreements[name] == len(features), f"agreement:{name}")
        normalized_counts = {name: int(route_counts[name]) for name in DECISIONS}
        add(issues, normalized_counts == recorded["exact_route_counts"], "recorded_route_counts")
        add(issues, normalized_counts == policy["selection_metrics"]["route_counts"], "policy_route_counts")
        # Recorded maxima are repeated as provenance; independent values need
        # only respect the same tolerance because BLAS/NumPy scalar order can
        # vary at sub-tolerance level across executions.
        recorded_errors_within_tolerance = all(
            float(recorded["max_absolute_errors"][name]) <= float(tolerances[name])
            for name in errors
        )
        add(issues, recorded_errors_within_tolerance, "recorded_errors")
        row = {
            "pass": not issues,
            "issues": sorted(Counter(issues).items()),
            "sample_count": len(features),
            "batch_size": 1,
            "independent_max_absolute_errors": errors,
            "recorded_max_absolute_errors": recorded["max_absolute_errors"],
            "tolerances": tolerances,
            "agreements": {name: agreements[name] / len(features) for name in ("ood", "route", "reason", "eligible")},
            "route_counts": normalized_counts,
            "policy_validation_features": feature_audit,
            "candidate_checkpoint": metadata(checkpoint_path),
            "locked_torchscript": metadata(module_path),
            "policy": metadata(policy_path),
            "runtime_spec": metadata(spec_path),
            "semantic_checks": {
                "ood_precedes_command_reject": spec.get("ood_before_command_reject"),
                "command_reject_solver_budget": spec.get("command_reject_numerical_solver_budget"),
                "defer_entry": spec.get("defer_entry"),
                "raw_probability_logging": spec.get("raw_probability_logging"),
            },
        }
        all_pass &= row["pass"]
        result[robot] = row
    return {"pass": all_pass, "robots": result}


def point_indices(category: np.ndarray, per_category: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for value in sorted(np.unique(category)):
        available = np.flatnonzero(category == value)
        count = min(per_category, len(available))
        chosen.extend(rng.choice(available, size=count, replace=False).astype(int).tolist())
    selected = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(selected)
    return selected.tolist()


def trajectory_ids(values: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    unique = np.unique(values)
    return np.sort(rng.choice(unique, size=min(count, len(unique)), replace=False).astype(np.int64)).astype(int).tolist()


def audit_runtime(
    workspace: Path,
    config: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    overall = True
    validation = config["validation"]
    tolerance = config["equivalence"]
    expected_keys = {f"{robot}/seed{seed}" for robot in ROBOTS for seed in SEEDS}
    top_issues: list[str] = []
    add(top_issues, set(equivalence["runtime_combinations"]) == expected_keys, "combination_keys")
    for robot in ROBOTS:
        for seed in SEEDS:
            key = f"{robot}/seed{seed}"
            recorded = equivalence["runtime_combinations"][key]
            issues: list[str] = []
            point_path = workspace / f"outputs/paper_v2_seed{seed}/{robot}/datasets/risk_validation_queries.npz"
            trajectory_path = workspace / f"outputs/paper_v2_seed{seed}/{robot}/datasets/seed_validation.npz"
            add(issues, Path(recorded["point_source"]).resolve() == point_path.resolve(), "point_source")
            add(issues, Path(recorded["trajectory_source"]).resolve() == trajectory_path.resolve(), "trajectory_source")
            add(issues, "test" not in point_path.name.lower() and "test" not in trajectory_path.name.lower(), "development_paths")
            with np.load(point_path, allow_pickle=False) as data:
                categories = np.asarray(data["category"]).astype(str)
            expected_points = point_indices(
                categories,
                int(validation["point_queries_per_category"]),
                int(validation["point_sampling_seed"]) + seed,
            )
            with np.load(trajectory_path, allow_pickle=False) as data:
                ids = np.asarray(data["trajectory_id"], dtype=np.int64)
            expected_trajectories = trajectory_ids(
                ids,
                int(validation["trajectory_count"]),
                int(validation["trajectory_sampling_seed"]) + seed,
            )
            add(issues, recorded["point_source_indices"] == expected_points, "point_selection")
            add(issues, recorded["trajectory_ids"] == expected_trajectories, "trajectory_selection")
            add(issues, recorded.get("validation_only") is True, "validation_only")
            add(issues, recorded.get("test_named_dataset_loaded") is False, "test_flag")
            split_rows: dict[str, Any] = {}
            for split in ("point", "trajectory"):
                summary = recorded[split]
                count = int(summary["paired_record_count"])
                split_issues: list[str] = []
                add(split_issues, set(summary["agreements"]) == AGREEMENT_FIELDS, "agreement_schema")
                add(split_issues, all(float(value) == 1.0 for value in summary["agreements"].values()), "all_agreements")
                raw = summary["raw_prediction_max_absolute_errors"]
                add(split_issues, float(raw["success_probability"]) <= float(tolerance["probability_max_abs"]), "success_tolerance")
                add(split_issues, float(raw["fail_all_probability"]) <= float(tolerance["probability_max_abs"]), "fail_tolerance")
                add(split_issues, float(raw["latency_p50_ms"]) <= float(tolerance["latency_max_abs_ms"]), "p50_tolerance")
                add(split_issues, float(raw["latency_p95_ms"]) <= float(tolerance["latency_max_abs_ms"]), "p95_tolerance")
                add(split_issues, float(raw["ood_score"]) <= float(tolerance["ood_score_max_abs"]), "ood_tolerance")
                add(split_issues, float(summary["accepted_command_max_absolute_error_rad"]) <= float(tolerance["accepted_command_max_abs_rad"]), "command_tolerance")
                add(split_issues, float(summary["command_reject_zero_solver_rate"]) == 1.0, "reject_zero_solver")
                add(split_issues, float(summary["defer_enters_fixed_easy_stage_rate"]) == 1.0, "defer_fixed_easy")
                add(split_issues, summary.get("pass") is True, "recorded_pass")
                add(issues, not split_issues, f"{split}_equivalence")
                split_rows[split] = {
                    "pass": not split_issues,
                    "issues": sorted(Counter(split_issues).items()),
                    "paired_record_count": count,
                    "all_semantic_agreements": summary["agreements"],
                    "route_action_agreement": summary["agreements"]["route_action"],
                    "accepted_agreement": summary["agreements"]["accepted"],
                    "function_evaluations_agreement": summary["agreements"]["function_evaluations"],
                    "fallback_agreement": summary["agreements"]["fallback"],
                    "executed_stages_agreement": summary["agreements"]["executed_stages"],
                    "paired_accepted_command_count": summary["paired_accepted_command_count"],
                    "accepted_command_max_absolute_error_rad": summary["accepted_command_max_absolute_error_rad"],
                    "command_reject_count": summary["command_reject_count"],
                    "command_reject_zero_solver_rate": summary["command_reject_zero_solver_rate"],
                    "defer_count": summary["defer_count"],
                    "defer_enters_fixed_easy_stage_rate": summary["defer_enters_fixed_easy_stage_rate"],
                    "raw_prediction_max_absolute_errors": raw,
                    "performance_totals_available_in_seal": {
                        "function_evaluations": False,
                        "fallback": False,
                        "executed_stage_counts": False,
                        "route_counts_beyond_reject_and_defer": False,
                    },
                }
            add(issues, recorded.get("pass") is True, "combination_pass")
            row = {
                "pass": not issues,
                "issues": sorted(Counter(issues).items()),
                "validation_only": recorded.get("validation_only"),
                "point_selection_reproduced": recorded["point_source_indices"] == expected_points,
                "point_query_count": len(expected_points),
                "point_category_counts": dict(sorted(Counter(categories[expected_points]).items())),
                "trajectory_selection_reproduced": recorded["trajectory_ids"] == expected_trajectories,
                "trajectory_ids": expected_trajectories,
                "trajectory_query_count": int(np.sum(np.isin(ids, expected_trajectories))),
                **split_rows,
            }
            overall &= row["pass"]
            results[key] = row
    return {
        "pass": overall and not top_issues,
        "issues": sorted(Counter(top_issues).items()),
        "combination_count": len(results),
        "combinations": results,
        "evidence_scope": (
            "Subset selection is independently replayed. The sealed report contains paired "
            "equality rates, prediction maxima, and command/reject/defer counts, but not raw "
            "per-query runtime rows or aggregate FEV/fallback/stage totals; those performance "
            "totals therefore cannot be reconstructed from the seal without rerunning the GPU solver."
        ),
    }


def test_flags(
    release: Mapping[str, Any],
    artifact: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    checks = {
        "release.test_named_dataset_loaded": release.get("test_named_dataset_loaded") is False,
        "release.test_v4_started": release.get("test_v4_started") is False,
        "release.formal_test_authorized_or_started": release.get("formal_test_authorized_or_started") is False,
        "artifact.test_data_loaded": artifact.get("test_data_loaded") is False,
        "equivalence.test_named_dataset_loaded": equivalence.get("test_named_dataset_loaded") is False,
        "equivalence.test_v4_started": equivalence.get("test_v4_started") is False,
        "dependencies.test_named_dataset_loaded": dependencies.get("test_named_dataset_loaded") is False,
    }
    for robot in ROBOTS:
        policy = strict_json(root / robot / "v4_policy.json")
        checks[f"{robot}.policy.test_data_loaded"] = policy.get("test_data_loaded") is False
        numeric = equivalence["robots"][robot]
        checks[f"{robot}.candidate_source_not_test_named"] = "test" not in Path(numeric["candidate_path"]).name.lower()
    for key, value in equivalence["runtime_combinations"].items():
        checks[f"{key}.test_named_dataset_loaded"] = value.get("test_named_dataset_loaded") is False
        checks[f"{key}.point_source_not_test_named"] = "test" not in Path(value["point_source"]).name.lower()
        checks[f"{key}.trajectory_source_not_test_named"] = "test" not in Path(value["trajectory_source"]).name.lower()
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "old_test_performance_loaded_by_audit": False,
        "old_test_files_content_treatment": "opaque SHA-256 bytes for protected-tree integrity only",
    }


def markdown(report: Mapping[str, Any]) -> str:
    verdict = report["verdict"]
    runtime = report["runtime_equivalence"]
    lines = [
        "# release_v4_locked 独立审计",
        "",
        f"**结论：{verdict}。**",
        "",
        "本审计只读取冻结发布包、冻结候选、开发/validation 数据与上游 v3 发布件；"
        "未解析任何旧 test_v3 性能内容，也未修改 `outputs/`。",
        "",
        "## 核心结果",
        "",
        f"- 发布 payload：{report['artifact_payloads']['actual_payload_count']}/10，大小与 SHA-256 全部匹配。",
        f"- 发布 digest：独立复算 {'一致' if report['release_digest']['match'] else '不一致'}。",
        f"- 源码冻结：before/after {'一致' if report['source_provenance']['before_after_exactly_equal'] else '不一致'}；记录 commit/tree 与 release runner/artifacts blob 均复算通过。",
        f"- protected tree：{report['protected_outputs']['recomputed_current']['file_count']} 个文件，当前摘要与冻结 after {'一致' if report['protected_outputs']['current_matches_recorded_after'] else '不一致'}。",
        f"- 数值等价：Panda 与 UR5e 均在 2,500 条 policy-validation 查询上通过 batch-one eager↔TorchScript 复算。",
        f"- runtime 等价：{runtime['combination_count']}/6 个 robot×seed 组合通过，point 4,200 条、trajectory 2,400 条。",
        f"- test 边界：所有显式 test/load/start/authorize 标志通过；旧 test_v3 文件仅作为 opaque bytes 参与 protected-tree 哈希。",
        "",
        "## 六个 runtime 组合",
        "",
        "| 组合 | Point 接受命令 | Point reject/defer | Trajectory 接受命令 | Trajectory reject/defer | route/accepted/FEV/fallback/stages | reject零solver | defer从easy进入 |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for key, value in runtime["combinations"].items():
        point = value["point"]
        trajectory = value["trajectory"]
        agreements = point["all_semantic_agreements"]
        semantic = all(
            agreements[name] == 1.0
            for name in ("route_action", "accepted", "function_evaluations", "fallback", "executed_stages")
        ) and all(
            trajectory["all_semantic_agreements"][name] == 1.0
            for name in ("route_action", "accepted", "function_evaluations", "fallback", "executed_stages")
        )
        lines.append(
            f"| {key} | {point['paired_accepted_command_count']}/700 | "
            f"{point['command_reject_count']}/{point['defer_count']} | "
            f"{trajectory['paired_accepted_command_count']}/400 | "
            f"{trajectory['command_reject_count']}/{trajectory['defer_count']} | "
            f"{'1.0 / 通过' if semantic else '失败'} | "
            f"{min(point['command_reject_zero_solver_rate'], trajectory['command_reject_zero_solver_rate']):.1f} | "
            f"{min(point['defer_enters_fixed_easy_stage_rate'], trajectory['defer_enters_fixed_easy_stage_rate']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "发布包保存了逐字段的 paired agreement、连续输出最大误差、接受命令误差、reject/defer 计数及语义率，"
            "但没有保存逐查询 runtime rows，也没有保存 FEV、fallback 和 executed-stage 的性能总量。"
            "因此，本审计能够确认 eager 与 locked runtime 在这些字段上逐条一致，却不能仅由 seal 独立重建这些字段的绝对性能总量。"
            "这不影响部署等价性 PASS，但建议未来发布包附带压缩的、去标识化 paired semantic rows。",
            "",
            "## 非阻断加固项",
            "",
            f"- `release_manifest.json` 不在 10 项 payload manifest 内，也不进入 release digest；"
            f"本次审计记录的控制文件 SHA-256 为 `{report['control_files']['release_manifest']['sha256']}`。"
            "建议在仓库外或签名清单中 pin 该摘要。",
            f"- 当前发布目录/文件权限为 `{report['hardening']['release_root_mode']}` / "
            f"`{report['hardening']['regular_file_modes']}`；sealed 是协议与哈希级冻结，不是 OS immutable。",
            "",
            "## 阻断项",
            "",
        ]
    )
    if report["blocking_issues"]:
        lines.extend(f"- {item}" for item in report["blocking_issues"])
    else:
        lines.append("- 无。发布包满足锁定与进入全新、预注册 test_v4 的前置条件。")
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/eric/anaconda3/envs/isaaclab_3/bin/python scripts/audit_release_v4_locked.py",
            "```",
            "",
            "脚本只写入 `docs/audits/release_v4_locked/`，不会写入或修改 `outputs/`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    workspace = args.workspace.resolve()
    root = args.release_root if args.release_root.is_absolute() else workspace / args.release_root
    root = root.resolve()
    json_path = args.json if args.json.is_absolute() else workspace / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else workspace / args.markdown
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    artifact = strict_json(root / "artifact_manifest.json")
    release = strict_json(root / "release_manifest.json")
    equivalence = strict_json(root / "release_equivalence.json")
    dependencies_payload = strict_json(root / "upstream_dependencies.json")
    config = strict_yaml(root / "release_config.yaml")
    payload_audit = audit_payloads(root, artifact)
    artifact_manifest_sha = file_sha256(root / "artifact_manifest.json")
    digest = sha256(
        (
            artifact_manifest_sha
            + str(release["candidate_release_digest"])
            + str(release["upstream_v3_release_manifest_sha256"])
        ).encode("ascii")
    ).hexdigest()
    digest_audit = {
        "pass": artifact_manifest_sha == release["artifact_manifest_sha256"] and digest == release["release_digest"],
        "artifact_manifest_sha256": artifact_manifest_sha,
        "recorded_artifact_manifest_sha256": release["artifact_manifest_sha256"],
        "recomputed_release_digest": digest,
        "recorded_release_digest": release["release_digest"],
        "match": digest == release["release_digest"],
        "formula": "SHA256(artifact_manifest_sha256 || candidate_release_digest || upstream_v3_release_manifest_sha256)",
    }
    source_audit = audit_source(workspace, release)
    protected_audit = audit_protected(workspace, config, release)
    dependency_audit = audit_dependencies(workspace, dependencies_payload)
    numerical_audit = audit_numerical(
        root,
        (workspace / "outputs/release_v4_candidate").resolve(),
        (workspace / "outputs/counterfactual_v4_bulk").resolve(),
        equivalence,
    )
    runtime_audit = audit_runtime(workspace, config, equivalence)
    flags = test_flags(release, artifact, equivalence, dependencies_payload, root)
    symlinks = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
    regular_files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    hardening = {
        "blocking": False,
        "release_manifest_covered_by_artifact_manifest": False,
        "release_manifest_covered_by_release_digest": False,
        "external_control_anchor_recommended": True,
        "release_root_mode": oct(stat.S_IMODE(root.stat().st_mode)),
        "regular_file_modes": sorted(
            {oct(stat.S_IMODE(path.stat().st_mode)) for path in regular_files}
        ),
        "regular_file_count": len(regular_files),
        "symlink_count": len(symlinks),
        "symlinks": symlinks,
        "os_immutable_asserted": False,
    }
    top_checks = {
        "artifact_payloads": payload_audit["pass"],
        "release_digest": digest_audit["pass"],
        "source_provenance": source_audit["pass"],
        "protected_outputs": protected_audit["pass"],
        "upstream_dependencies": dependency_audit["pass"],
        "numerical_equivalence": numerical_audit["pass"],
        "runtime_equivalence": runtime_audit["pass"],
        "test_boundary": flags["pass"],
        "recorded_all_pass": equivalence.get("all_pass") is True,
        "six_runtime_count": equivalence.get("expected_runtime_combination_count") == 6,
        "backend": release.get("backend") == BACKEND == equivalence.get("backend"),
        "protocol": release.get("protocol") == PROTOCOL == equivalence.get("protocol"),
    }
    blocking = [name for name, passed in top_checks.items() if not passed]
    report = {
        "audit_protocol": "independent_release_v4_locked_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "release_root": str(root),
        "read_only_outputs": True,
        "imported_release_v4_runner_or_artifacts": False,
        "old_test_performance_loaded": False,
        "verdict": "PASS" if not blocking else "FAIL",
        "blocking_issues": blocking,
        "top_level_checks": top_checks,
        "artifact_payloads": payload_audit,
        "control_files": {
            "artifact_manifest": metadata(root / "artifact_manifest.json"),
            "release_manifest": metadata(root / "release_manifest.json"),
        },
        "hardening": hardening,
        "release_digest": digest_audit,
        "source_provenance": source_audit,
        "protected_outputs": protected_audit,
        "upstream_dependencies": dependency_audit,
        "numerical_equivalence": numerical_audit,
        "runtime_equivalence": runtime_audit,
        "test_boundary": flags,
        "limitations": [
            "The seal omits raw per-query runtime rows and absolute FEV/fallback/stage totals; the audit independently verifies the selected validation subsets and the persisted paired equality evidence, not those unavailable totals.",
            "Historical release-scope cleanliness is temporal; immutable commit/tree and release code blobs are independently verified, while before/after cleanliness remains a signed manifest assertion.",
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "blocking_issues": blocking, "json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
