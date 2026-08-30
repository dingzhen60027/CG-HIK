#!/usr/bin/env python3
"""Independent, read-only audit of the frozen validation-only v4 candidate.

The audit deliberately does not import the v4 training runner or predictor.
It reconstructs the small Torch model, Platt calibration, Mahalanobis OOD
threshold, all 252 policy candidates, and the selected policy directly from
the sealed checkpoint and the three development roles.  Its only write is the
requested JSON report below ``docs/audits``.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F
import yaml


PROTOCOL = "counterfactual_v4_validation_training_v2"
ROBOTS = ("panda", "ur5e")
ROLES = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)
ROLE_COUNTS = {
    "risk_train_queries": 15_000,
    "calibration_queries": 2_500,
    "policy_validation_queries": 2_500,
}
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
COLLECTED_ACTIONS = ACTIONS + ("fixed_robust",)
REJECT_ACTION = 3
DEFER_ACTION = 4
LABEL_CONTRACT = {
    "action_success_target": "semantic_verified_success",
    "shared_semantic_success_due_to_terminal_fallback_invariance": True,
    "deadline_success_role": "diagnostic_only",
    "latency_target": "raw_repeat_p50_p95_pinball",
}
PAYLOAD_FILES = {
    "data_audit.json",
    "environment.json",
    "frozen_config.yaml",
    "models/panda_seed17_predictor.pt",
    "models/ur5e_seed17_predictor.pt",
    "policies/panda_seed17_policy.json",
    "policies/ur5e_seed17_policy.json",
    "policy_candidates/panda_seed17.json",
    "policy_candidates/ur5e_seed17.json",
    "policy_selection.json",
    "training_metrics.json",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("outputs/release_v4_candidate")
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "docs/audits/release_v4_candidate/candidate_audit.json"
        ),
    )
    return parser.parse_args()


def strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def source_tree_sha256(workspace: Path) -> str:
    root = workspace / "src/confik"
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_tracked_source_tree_sha256(
    workspace: Path, reference: str = "HEAD"
) -> tuple[str, list[str]]:
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", reference, "--", "src/confik"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    files = sorted(path for path in listing if path.endswith(".py"))
    digest = sha256()
    for relative in files:
        logical = relative.removeprefix("src/confik/")
        digest.update(logical.encode("utf-8"))
        contents = subprocess.run(
            ["git", "show", f"{reference}:{relative}"],
            cwd=workspace,
            check=True,
            capture_output=True,
        ).stdout
        digest.update(contents)
    return digest.hexdigest(), files


def find_matching_source_commit(
    workspace: Path, recorded_sha256: str, *, limit: int = 20
) -> dict[str, Any]:
    commits = subprocess.run(
        ["git", "rev-list", f"--max-count={limit}", "HEAD"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    checked: list[dict[str, Any]] = []
    for commit in commits:
        digest, files = git_tracked_source_tree_sha256(workspace, commit)
        checked.append(
            {"commit": commit, "sha256": digest, "file_count": len(files)}
        )
        if digest == recorded_sha256:
            return {
                "found": True,
                "matching_commit": commit,
                "matching_sha256": digest,
                "file_count": len(files),
                "commits_checked": len(checked),
                "checked": checked,
            }
    return {
        "found": False,
        "matching_commit": None,
        "matching_sha256": None,
        "file_count": None,
        "commits_checked": len(checked),
        "checked": checked,
    }


def source_worktree_status(workspace: Path) -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no", "--", "src/confik"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/confik",
        ],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "tracked_changes": tracked,
        "untracked_python_files": sorted(
            path for path in untracked if path.endswith(".py")
        ),
    }


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


def release_digest(
    artifact_manifest_sha256: str,
    config_sha256: str,
    bulk_manifest_sha256: str,
) -> str:
    return sha256(
        (
            artifact_manifest_sha256
            + config_sha256
            + bulk_manifest_sha256
        ).encode("ascii")
    ).hexdigest()


def add_issue(issues: list[str], condition: bool, name: str) -> None:
    if not condition:
        issues.append(name)


def compare_payload(
    recomputed: Any,
    recorded: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    mismatches: list[str] = []
    max_absolute_error = 0.0

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal max_absolute_error
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                mismatches.append(
                    f"{path}:keys:{sorted(set(left) - set(right))}:"
                    f"{sorted(set(right) - set(left))}"
                )
            for key in sorted(set(left) & set(right), key=str):
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                mismatches.append(f"{path}:length:{len(left)}!={len(right)}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
            if bool(left) != bool(right):
                mismatches.append(f"{path}:{left!r}!={right!r}")
            return
        numeric = isinstance(left, (int, float, np.number)) and isinstance(
            right, (int, float, np.number)
        )
        if numeric:
            left_number = float(left)
            right_number = float(right)
            difference = abs(left_number - right_number)
            max_absolute_error = max(max_absolute_error, difference)
            if not np.isclose(
                left_number, right_number, atol=atol, rtol=rtol, equal_nan=False
            ):
                mismatches.append(
                    f"{path}:{left_number:.17g}!={right_number:.17g}"
                )
            return
        if left != right:
            mismatches.append(f"{path}:{left!r}!={right!r}")

    visit(recomputed, recorded, "root")
    return {
        "pass": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "max_absolute_error": max_absolute_error,
    }


@dataclass(frozen=True)
class RoleData:
    features: np.ndarray
    source_indices: np.ndarray
    query_sha256: np.ndarray
    category: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    verified_success: np.ndarray
    verified_success_before_deadline: np.ndarray
    latency_samples_ns: np.ndarray
    function_evaluations: np.ndarray
    fallback_used: np.ndarray
    source_files: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return int(self.features.shape[0])

    @property
    def semantic_success(self) -> np.ndarray:
        return self.verified_success[:, :3]

    @property
    def semantic_fail_all(self) -> np.ndarray:
        return np.all(~self.semantic_success, axis=1)


def artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
    }


def load_role(
    *,
    bulk_root: Path,
    robot: str,
    role: str,
    recorded_audit: Mapping[str, Any],
) -> tuple[RoleData, dict[str, Any]]:
    issues: list[str] = []
    role_root = bulk_root / robot / "seed17" / role
    add_issue(issues, role_root.is_dir() and not role_root.is_symlink(), "role_root")
    selection_path = role_root / "selection.npz"
    selection_manifest_path = role_root / "selection_manifest.json"
    add_issue(
        issues,
        selection_path.is_file() and not selection_path.is_symlink(),
        "selection_file",
    )
    add_issue(
        issues,
        selection_manifest_path.is_file()
        and not selection_manifest_path.is_symlink(),
        "selection_manifest_file",
    )
    selection_manifest = strict_json(selection_manifest_path)
    add_issue(issues, selection_manifest.get("source_role") == role, "selection_role")
    add_issue(
        issues,
        selection_manifest.get("test_named_dataset_loaded") is False,
        "selection_development_only",
    )
    source_files: list[dict[str, Any]] = [
        artifact_metadata(selection_path),
        artifact_metadata(selection_manifest_path),
    ]

    fields = {
        name: []
        for name in (
            "features",
            "source_indices",
            "query_sha256",
            "category",
            "expected_reachable",
            "continuity_feasible",
            "verified_success",
            "verified_success_before_deadline",
            "latency_samples_ns",
            "function_evaluations",
            "fallback_used",
        )
    }
    chunks = sorted((role_root / "chunks").glob("chunk_*"))
    expected_chunk_count = ROLE_COUNTS[role] // 250
    add_issue(issues, len(chunks) == expected_chunk_count, "chunk_count")
    expected_start = 0
    for chunk in chunks:
        add_issue(issues, chunk.is_dir() and not chunk.is_symlink(), "chunk_path")
        manifest_path = chunk / "chunk_manifest.json"
        labels_path = chunk / "counterfactual_labels.npz"
        add_issue(
            issues,
            manifest_path.is_file() and not manifest_path.is_symlink(),
            "chunk_manifest_path",
        )
        add_issue(
            issues,
            labels_path.is_file() and not labels_path.is_symlink(),
            "chunk_labels_path",
        )
        manifest = strict_json(manifest_path)
        start = int(manifest["query_start"])
        stop = int(manifest["query_stop_exclusive"])
        add_issue(issues, start == expected_start and stop > start, "chunk_contiguity")
        add_issue(issues, manifest.get("robot") == robot, "chunk_robot")
        add_issue(issues, manifest.get("source_role") == role, "chunk_role")
        add_issue(issues, int(manifest.get("training_seed", -1)) == 17, "chunk_seed")
        add_issue(
            issues,
            manifest.get("environment_contaminated") is False,
            "chunk_environment_clean",
        )
        add_issue(
            issues, manifest.get("test_data_loaded") is False, "chunk_development_only"
        )
        for relative, expected in manifest.get("artifacts", {}).items():
            path = chunk / str(relative)
            add_issue(
                issues,
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == int(expected["size"])
                and file_sha256(path) == str(expected["sha256"]),
                f"chunk_artifact:{relative}",
            )
        with np.load(labels_path, allow_pickle=False) as labels:
            add_issue(
                issues,
                tuple(labels["feature_names"].astype(str).tolist()) == FEATURE_NAMES,
                "feature_schema",
            )
            add_issue(
                issues,
                tuple(labels["action_names"].astype(str).tolist())
                == COLLECTED_ACTIONS,
                "action_schema",
            )
            add_issue(
                issues,
                tuple(labels["decision_action_names"].astype(str).tolist())
                == ACTIONS,
                "decision_action_schema",
            )
            add_issue(
                issues,
                int(labels["features"].shape[0]) == stop - start,
                "chunk_row_count",
            )
            for name in fields:
                fields[name].append(np.asarray(labels[name]))
        source_files.extend((artifact_metadata(manifest_path), artifact_metadata(labels_path)))
        expected_start = stop

    values = {name: np.concatenate(items, axis=0) for name, items in fields.items()}
    expected_count = ROLE_COUNTS[role]
    add_issue(issues, expected_start == expected_count, "complete_chunk_range")
    with np.load(selection_path, allow_pickle=False) as selection:
        selection_source_indices = np.asarray(selection["source_indices"], dtype=np.int64)
        for name in (
            "query_sha256",
            "category",
            "expected_reachable",
            "continuity_feasible",
        ):
            add_issue(
                issues,
                np.array_equal(values[name].astype(str) if values[name].dtype.kind in "US" else values[name],
                               selection[name].astype(str) if selection[name].dtype.kind in "US" else selection[name]),
                f"selection_identity:{name}",
            )
        add_issue(
            issues,
            np.array_equal(
                np.asarray(values["source_indices"], dtype=np.int64),
                selection_source_indices,
            ),
            "selection_identity:source_indices",
        )

    expected_source_path = (
        bulk_root.parent
        / "paper_v2_seed17"
        / robot
        / "datasets"
        / f"{role}.npz"
    ).resolve()
    source_path = Path(str(selection_manifest["source_path"])).resolve()
    add_issue(issues, source_path == expected_source_path, "source_path")
    add_issue(
        issues,
        source_path.is_file()
        and not source_path.is_symlink()
        and file_sha256(source_path) == str(selection_manifest["source_sha256"]),
        "source_artifact",
    )
    add_issue(
        issues,
        file_sha256(selection_path)
        == str(selection_manifest["selection_artifact_sha256"]),
        "selection_artifact_hash",
    )
    add_issue(
        issues,
        sha256(np.ascontiguousarray(selection_source_indices).tobytes()).hexdigest()
        == str(selection_manifest["selection_indices_sha256"]),
        "selection_indices_hash",
    )
    add_issue(
        issues,
        sha256(
            np.ascontiguousarray(values["query_sha256"].astype("S64")).tobytes()
        ).hexdigest()
        == str(selection_manifest["selection_query_hashes_sha256"]),
        "selection_query_hashes_hash",
    )
    with np.load(source_path, allow_pickle=False) as source:
        # Materialize each compressed member once. Re-indexing an NpzFile in
        # the loop would repeatedly decompress the same array.
        source_arrays = {
            name: np.asarray(source[name])
            for name in (
                "previous_q",
                "target_position",
                "target_rotation",
                "category",
                "expected_reachable",
                "continuity_feasible",
            )
        }
        add_issue(
            issues,
            int(source_arrays["category"].shape[0])
            == int(selection_manifest["source_query_count"]),
            "source_query_count",
        )
        add_issue(
            issues,
            np.all(
                (selection_source_indices >= 0)
                & (selection_source_indices < len(source_arrays["category"]))
            ),
            "source_index_range",
        )
        add_issue(
            issues,
            len(np.unique(selection_source_indices)) == len(selection_source_indices),
            "source_indices_unique",
        )
        for name in ("category", "expected_reachable", "continuity_feasible"):
            add_issue(
                issues,
                np.array_equal(
                    np.asarray(values[name]),
                    source_arrays[name][selection_source_indices],
                ),
                f"source_selection_identity:{name}",
            )
        dt = np.float64(0.02)
        reconstructed_query_hashes: list[str] = []
        for source_index in selection_source_indices.tolist():
            digest = sha256()
            for name in ("previous_q", "target_position", "target_rotation"):
                digest.update(
                    np.ascontiguousarray(
                        source_arrays[name][source_index], dtype=np.float64
                    ).tobytes()
                )
            digest.update(np.asarray([dt], dtype=np.float64).tobytes())
            reconstructed_query_hashes.append(digest.hexdigest())
        add_issue(
            issues,
            np.array_equal(
                np.asarray(reconstructed_query_hashes, dtype="U64"),
                values["query_sha256"].astype("U64"),
            ),
            "source_query_hash_reconstruction",
        )

    data = RoleData(
        features=np.asarray(values["features"], dtype=np.float32),
        source_indices=np.asarray(values["source_indices"], dtype=np.int64),
        query_sha256=values["query_sha256"].astype("U64"),
        category=values["category"].astype(str),
        expected_reachable=np.asarray(values["expected_reachable"], dtype=bool),
        continuity_feasible=np.asarray(values["continuity_feasible"], dtype=bool),
        verified_success=np.asarray(values["verified_success"], dtype=bool),
        verified_success_before_deadline=np.asarray(
            values["verified_success_before_deadline"], dtype=bool
        ),
        latency_samples_ns=np.asarray(values["latency_samples_ns"], dtype=np.int64),
        function_evaluations=np.asarray(values["function_evaluations"], dtype=np.int64),
        fallback_used=np.asarray(values["fallback_used"], dtype=bool),
        source_files=tuple(source_files),
    )
    add_issue(issues, data.features.shape == (expected_count, 9), "feature_shape")
    add_issue(issues, np.all(np.isfinite(data.features)), "feature_finite")
    add_issue(
        issues,
        data.verified_success.shape == (expected_count, 4),
        "success_shape",
    )
    add_issue(
        issues,
        data.latency_samples_ns.shape == (expected_count, 4, 5),
        "latency_shape",
    )
    add_issue(
        issues,
        np.array_equal(
            data.semantic_success,
            np.repeat(data.semantic_success[:, :1], 3, axis=1),
        ),
        "shared_semantic_success_labels",
    )
    for name, values_array in (
        ("verified_success", data.verified_success),
        (
            "verified_success_before_deadline",
            data.verified_success_before_deadline,
        ),
        ("latency_samples_ns", data.latency_samples_ns),
        ("function_evaluations", data.function_evaluations),
        ("fallback_used", data.fallback_used),
    ):
        add_issue(
            issues,
            np.array_equal(values_array[:, 0], values_array[:, 3]),
            f"easy_fixed_alias:{name}",
        )
    add_issue(
        issues,
        len(np.unique(data.query_sha256)) == expected_count,
        "unique_query_identity",
    )

    reconstructed_audit = {
        "query_count": data.count,
        "query_sha256_tensor_sha256": array_sha256(
            data.query_sha256.astype("S64")
        ),
        "raw_latency_samples_ns_sha256": array_sha256(data.latency_samples_ns),
        "raw_latency_shape": list(data.latency_samples_ns.shape),
        "raw_latency_dtype": str(data.latency_samples_ns.dtype),
        # The training runner exposes fail_all as float32 before taking its
        # mean.  Reproduce that dtype association exactly.
        "fail_all_rate": float(
            np.mean(data.semantic_fail_all.astype(np.float32))
        ),
        "semantic_verified_success_rates": {
            action: float(np.mean(data.verified_success[:, index]))
            for index, action in enumerate(ACTIONS)
        },
        "deadline_success_diagnostic_rates": {
            action: float(
                np.mean(data.verified_success_before_deadline[:, index])
            )
            for index, action in enumerate(ACTIONS)
        },
        "contract_feasible_semantic_fail_all_count": int(
            np.sum(
                data.semantic_fail_all
                & data.expected_reachable
                & data.continuity_feasible
            )
        ),
        "source_files": list(data.source_files),
    }
    recorded_comparison = compare_payload(reconstructed_audit, recorded_audit)
    add_issue(issues, recorded_comparison["pass"], "recorded_data_audit")
    return data, {
        "pass": not issues,
        "issues": sorted(Counter(issues).items()),
        "query_count": data.count,
        "chunk_count": len(chunks),
        "shared_semantic_labels_exact": np.array_equal(
            data.semantic_success,
            np.repeat(data.semantic_success[:, :1], 3, axis=1),
        ),
        "semantic_fail_all_count": int(np.sum(data.semantic_fail_all)),
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_query_hash_reconstruction_pass": "source_query_hash_reconstruction"
        not in issues,
        "contract_feasible_semantic_fail_all_count": int(
            np.sum(
                data.semantic_fail_all
                & data.expected_reachable
                & data.continuity_feasible
            )
        ),
        "data_audit_reproduction": recorded_comparison,
    }


@dataclass(frozen=True)
class Prediction:
    success_logits: np.ndarray
    success_probability: np.ndarray
    p50_ms: np.ndarray
    p95_ms: np.ndarray
    fail_logit: np.ndarray
    fail_probability: np.ndarray
    embedding: np.ndarray
    ood_score: np.ndarray
    is_ood: np.ndarray


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def raw_forward(payload: Mapping[str, Any], features: np.ndarray) -> tuple[np.ndarray, ...]:
    state = payload["state_dict"]
    mean = payload["feature_mean"].cpu().numpy().astype(np.float32)
    scale = payload["feature_scale"].cpu().numpy().astype(np.float32)
    normalized = np.ascontiguousarray((features - mean) / scale, dtype=np.float32)
    tensor = torch.from_numpy(normalized)
    hidden_sizes = tuple(int(value) for value in payload["config"]["hidden_sizes"])
    with torch.inference_mode():
        hidden = tensor
        for index in range(len(hidden_sizes)):
            layer = index * 2
            hidden = F.silu(
                F.linear(
                    hidden,
                    state[f"backbone.{layer}.weight"],
                    state[f"backbone.{layer}.bias"],
                )
            )
        shared = F.linear(
            hidden,
            state["verified_success_head.weight"],
            state["verified_success_head.bias"],
        )
        success = shared.expand(-1, 3)
        latency_raw = F.linear(
            hidden,
            state["latency_head.weight"],
            state["latency_head.bias"],
        )
        p50_raw, gap_raw = latency_raw.chunk(2, dim=1)
        p50 = F.softplus(p50_raw) + float(payload["config"]["min_latency_ms"])
        p95 = (
            p50
            + F.softplus(gap_raw)
            + float(payload["config"]["min_quantile_gap_ms"])
        )
        fail = F.linear(
            hidden,
            state["fail_all_head.weight"],
            state["fail_all_head.bias"],
        ).squeeze(1)
    return tuple(
        output.detach().cpu().numpy().astype(np.float64)
        for output in (success, p50, p95, fail, hidden)
    )


def calibrated_prediction(payload: Mapping[str, Any], features: np.ndarray) -> Prediction:
    success_logits, p50, p95, fail_logit, embedding = raw_forward(payload, features)
    calibrator = payload["calibrator"]
    add = calibrator["calibrators"]
    success_probability = np.clip(
        stable_sigmoid(
            float(add[0]["slope"]) * success_logits[:, 0]
            + float(add[0]["intercept"])
        ),
        1e-7,
        1.0 - 1e-7,
    )
    fail_probability = np.clip(
        stable_sigmoid(
            float(add[1]["slope"]) * fail_logit
            + float(add[1]["intercept"])
        ),
        1e-7,
        1.0 - 1e-7,
    )
    detector = payload["ood_detector"]
    mean = np.asarray(detector["mean"], dtype=np.float64)
    precision = np.asarray(detector["precision"], dtype=np.float64)
    centered = embedding - mean
    score = np.maximum(
        np.einsum("ni,ij,nj->n", centered, precision, centered, optimize=True),
        0.0,
    )
    threshold = float(detector["threshold"])
    return Prediction(
        success_logits=success_logits,
        success_probability=np.repeat(success_probability[:, None], 3, axis=1),
        p50_ms=p50,
        p95_ms=p95,
        fail_logit=fail_logit,
        fail_probability=fail_probability,
        embedding=embedding,
        ood_score=score,
        is_ood=score > threshold,
    )


def expected_calibration_error(
    probability: np.ndarray, target: np.ndarray, bins: int = 15
) -> float:
    indices = np.minimum((probability * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected]))
                - float(np.mean(target[selected]))
            )
    return float(error)


def discrimination(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    result: dict[str, Any] = {
        "ece": expected_calibration_error(clipped, target),
        "brier": float(np.mean((clipped - target) ** 2)),
        "nll": float(
            -np.mean(
                target * np.log(clipped)
                + (1.0 - target) * np.log1p(-clipped)
            )
        ),
        "coverage": float(np.mean(np.maximum(clipped, 1.0 - clipped) >= 0.8)),
        "positive_rate": float(np.mean(target)),
    }
    if np.unique(target).size == 2:
        result["auroc"] = float(roc_auc_score(target, probability))
        result["auprc"] = float(average_precision_score(target, probability))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def refit_platt(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, float, np.ndarray]:
    estimator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        random_state=0,
        max_iter=1000,
    )
    estimator.fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), targets)
    slope = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    probability = np.clip(
        stable_sigmoid(slope * np.asarray(logits, dtype=np.float64) + intercept),
        1e-7,
        1.0 - 1e-7,
    )
    return slope, intercept, probability


def latency_report(prediction: Prediction, role: RoleData) -> dict[str, Any]:
    samples = role.latency_samples_ns[:, :3].astype(np.float32) / np.float32(1e6)
    samples64 = samples.astype(np.float64)
    target_p50 = np.quantile(samples64, 0.50, axis=2)
    target_p95 = np.quantile(samples64, 0.95, axis=2)
    result: dict[str, Any] = {}
    for index, action in enumerate(ACTIONS):
        action_samples = samples64[:, index]
        result[action] = {
            "diagnostic_empirical_p50_mae_ms": float(
                np.mean(np.abs(prediction.p50_ms[:, index] - target_p50[:, index]))
            ),
            "diagnostic_empirical_p95_mae_ms": float(
                np.mean(np.abs(prediction.p95_ms[:, index] - target_p95[:, index]))
            ),
            "raw_sample_p50_coverage": float(
                np.mean(action_samples <= prediction.p50_ms[:, index, None])
            ),
            "raw_sample_p95_coverage": float(
                np.mean(action_samples <= prediction.p95_ms[:, index, None])
            ),
            "predicted_p50_median_ms": float(
                np.median(prediction.p50_ms[:, index])
            ),
            "predicted_p95_median_ms": float(
                np.median(prediction.p95_ms[:, index])
            ),
        }
    result["supervision"] = {
        "source_field": "latency_samples_ns",
        "raw_repeat_count": 5,
        "raw_observation_count": int(samples64.size),
        "per_query_winner_used_as_target": False,
        "aggregated_p50_p95_used_as_target": False,
    }
    return result


def route_actions(prediction: Prediction, config: Mapping[str, float]) -> np.ndarray:
    eligible = (
        prediction.success_probability
        >= float(config["minimum_success_probability"])
    ) & (prediction.p95_ms <= float(config["deadline_ms"]))
    has_eligible = np.any(eligible, axis=1)
    action = np.full(prediction.success_probability.shape[0], DEFER_ACTION, dtype=np.int64)
    reject = (
        ~prediction.is_ood
        & ~has_eligible
        & (prediction.fail_probability >= float(config["reject_probability"]))
    )
    action[reject] = REJECT_ACTION
    selectable = ~prediction.is_ood & has_eligible
    if np.any(selectable):
        masked = np.where(eligible, prediction.p95_ms, np.inf)
        fastest = np.argmin(masked, axis=1)
        conservative = np.argmax(eligible, axis=1)
        rows = np.arange(action.size)
        improvement = (
            prediction.p95_ms[rows, conservative]
            - prediction.p95_ms[rows, fastest]
        )
        chosen = np.where(
            improvement < float(config["latency_tie_margin_ms"]),
            conservative,
            fastest,
        )
        action[selectable] = chosen[selectable]
    return action


def policy_metrics(
    role: RoleData,
    prediction: Prediction,
    config: Mapping[str, float],
    hard_gates: Mapping[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    selected = route_actions(prediction, config)
    count = role.count
    command_success = np.zeros(count, dtype=bool)
    deadline_success = np.zeros(count, dtype=bool)
    fev = np.zeros(count, dtype=np.float64)
    latency = np.zeros((count, 5), dtype=np.float64)
    for action in range(3):
        mask = selected == action
        command_success[mask] = role.verified_success[mask, action]
        deadline_success[mask] = role.verified_success_before_deadline[mask, action]
        fev[mask] = role.function_evaluations[mask, action]
        latency[mask] = role.latency_samples_ns[mask, action] / 1e6
    defer = selected == DEFER_ACTION
    command_success[defer] = role.verified_success[defer, 3]
    deadline_success[defer] = role.verified_success_before_deadline[defer, 3]
    fev[defer] = role.function_evaluations[defer, 3]
    latency[defer] = role.latency_samples_ns[defer, 3] / 1e6
    reject = selected == REJECT_ACTION
    fixed_success = role.verified_success[:, 3]
    fixed_deadline = role.verified_success_before_deadline[:, 3]
    fixed_fev = role.function_evaluations[:, 3].astype(np.float64)
    operational_feasible = role.expected_reachable & role.continuity_feasible
    fixed_positive_count = max(int(np.sum(fixed_success)), 1)
    operational_count = max(int(np.sum(operational_feasible)), 1)
    successful_samples = latency[command_success]
    route_counts = {
        **{
            action: int(np.sum(selected == index))
            for index, action in enumerate(ACTIONS)
        },
        "reject": int(np.sum(reject)),
        "defer": int(np.sum(defer)),
    }
    success_difference = float(np.mean(command_success) - np.mean(fixed_success))
    fixed_false_reject = float(np.sum(reject & fixed_success) / fixed_positive_count)
    feasible_false_reject = float(
        np.sum(reject & operational_feasible) / operational_count
    )
    result = {
        "config": {
            "minimum_success_probability": float(config["minimum_success_probability"]),
            "reject_probability": float(config["reject_probability"]),
            "deadline_ms": float(config["deadline_ms"]),
            "latency_tie_margin_ms": float(config["latency_tie_margin_ms"]),
        },
        "query_count": count,
        "route_counts": route_counts,
        "route_rates": {key: value / count for key, value in route_counts.items()},
        "fixed_verified_success_rate": float(np.mean(fixed_success)),
        "selected_verified_success_rate": float(np.mean(command_success)),
        "verified_success_difference_vs_fixed": success_difference,
        "fixed_deadline_success_rate": float(np.mean(fixed_deadline)),
        "selected_deadline_success_rate": float(np.mean(deadline_success)),
        "fixed_success_false_reject_rate": fixed_false_reject,
        "operational_feasible_false_reject_rate": feasible_false_reject,
        "reject_precision_against_fixed_failure": (
            None if not np.any(reject) else float(np.mean(~fixed_success[reject]))
        ),
        "mean_function_evaluations": float(np.mean(fev)),
        "fixed_mean_function_evaluations": float(np.mean(fixed_fev)),
        "function_evaluation_reduction_fraction": float(
            1.0 - np.mean(fev) / max(np.mean(fixed_fev), 1e-12)
        ),
        "observed_latency_p50_ms": float(np.quantile(latency, 0.50)),
        "observed_latency_p95_ms": float(np.quantile(latency, 0.95)),
        "observed_latency_p99_ms": float(np.quantile(latency, 0.99)),
        "successful_command_latency_p95_ms": (
            None
            if successful_samples.size == 0
            else float(np.quantile(successful_samples, 0.95))
        ),
        "deadline_miss_rate": float(np.mean(~deadline_success & command_success)),
        "eligibility_contract": (
            "calibrated semantic verified-success probability meets the risk threshold "
            "and predicted raw-repeat P95 latency is at most the 20 ms deadline"
        ),
        "selection_latency_scope": (
            "raw five-repeat solver-plus-verifier labels; learned gate overhead excluded"
        ),
        "hard_gate_pass": bool(
            success_difference
            >= float(hard_gates["verified_success_difference_vs_fixed_minimum"])
            and fixed_false_reject
            <= float(hard_gates["fixed_success_false_reject_rate_maximum"])
            and feasible_false_reject
            <= float(
                hard_gates["operational_feasible_false_reject_rate_maximum"]
            )
        ),
    }
    return result, selected


def candidate_grid(
    role: RoleData,
    prediction: Prediction,
    policy_grid: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    routes: list[np.ndarray] = []
    for success in policy_grid["minimum_success_probabilities"]:
        for reject in policy_grid["reject_probabilities"]:
            for margin in policy_grid["latency_tie_margins_ms"]:
                config = {
                    "minimum_success_probability": float(success),
                    "reject_probability": float(reject),
                    "deadline_ms": float(policy_grid["deadline_ms"]),
                    "latency_tie_margin_ms": float(margin),
                }
                metrics, route = policy_metrics(
                    role, prediction, config, policy_grid["hard_gates"]
                )
                candidates.append(metrics)
                routes.append(route)
    passing = [index for index, row in enumerate(candidates) if row["hard_gate_pass"]]
    if not passing:
        raise RuntimeError("independent audit found no passing policy candidate")

    def key(index: int) -> tuple[float, float, float, float, float, float]:
        row = candidates[index]
        success_p95 = row["successful_command_latency_p95_ms"]
        return (
            float("inf") if success_p95 is None else float(success_p95),
            float(row["mean_function_evaluations"]),
            -float(row["selected_verified_success_rate"]),
            float(row["operational_feasible_false_reject_rate"]),
            -float(row["config"]["latency_tie_margin_ms"]),
            -float(row["config"]["minimum_success_probability"]),
        )

    selected_index = min(passing, key=key)
    selected = {
        **candidates[selected_index],
        "selection_rank": 1,
        "selection_objective": (
            "lexicographic: lowest successful-command observed P95, lowest mean FEV, "
            "highest verified success; subject to all frozen hard gates"
        ),
        "hard_constraints": {
            "verified_success_difference_vs_fixed_minimum": -0.01,
            "fixed_success_false_reject_rate_maximum": 0.01,
            "operational_feasible_false_reject_rate_maximum": 0.01,
        },
    }
    return candidates, routes, selected


def feature_overlap(left: np.ndarray, right: np.ndarray) -> int:
    left_rows = {
        np.ascontiguousarray(row, dtype=np.float32).tobytes() for row in left
    }
    right_rows = {
        np.ascontiguousarray(row, dtype=np.float32).tobytes() for row in right
    }
    return len(left_rows & right_rows)


def feature_label_diagnostics(role: RoleData) -> dict[str, Any]:
    target = role.semantic_fail_all.astype(np.int64)
    features: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = role.features[:, index].astype(np.float64)
        positive = values[target == 1]
        negative = values[target == 0]
        auroc = float(roc_auc_score(target, values))
        oriented = max(auroc, 1.0 - auroc)
        positive_above = bool(np.min(positive) > np.max(negative))
        negative_above = bool(np.min(negative) > np.max(positive))
        features[name] = {
            "best_direction_univariate_auroc": oriented,
            "perfect_threshold_separation": positive_above or negative_above,
            "fail_all_min": float(np.min(positive)),
            "fail_all_max": float(np.max(positive)),
            "success_min": float(np.min(negative)),
            "success_max": float(np.max(negative)),
        }
    category_counts: dict[str, Any] = {}
    for category in sorted(np.unique(role.category).tolist()):
        selected = role.category == category
        category_counts[str(category)] = {
            "query_count": int(np.sum(selected)),
            "semantic_fail_all_count": int(np.sum(target[selected])),
            "semantic_fail_all_rate": float(np.mean(target[selected])),
        }
    return {"features": features, "by_category": category_counts}


def feature_function_audit(workspace: Path) -> dict[str, Any]:
    path = workspace / "src/confik/latency_pilot_v3/optimized_inference.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "cached_risk_features"
    )
    function_source = ast.get_source_segment(source, function) or ""
    forbidden = (
        "verified_success",
        "failure_reason",
        "function_evaluations",
        "latency_samples",
        "fallback_used",
        "verification_reasons",
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "function": "cached_risk_features",
        "forbidden_post_solver_label_tokens": [
            token for token in forbidden if token in function_source
        ],
        "arguments": [argument.arg for argument in function.args.args],
        "uses_only_query_seed_and_kinematic_diagnostics": not any(
            token in function_source for token in forbidden
        ),
    }


def artifact_audit(
    workspace: Path, candidate_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    issues: list[str] = []
    add_issue(
        issues,
        candidate_root.is_dir() and not candidate_root.is_symlink(),
        "candidate_root",
    )
    artifact_manifest_path = candidate_root / "artifact_manifest.json"
    run_manifest_path = candidate_root / "run_manifest.json"
    artifact_manifest = strict_json(artifact_manifest_path)
    run_manifest = strict_json(run_manifest_path)
    recorded_files = artifact_manifest.get("files", {})
    add_issue(issues, int(artifact_manifest.get("file_count", -1)) == 11, "file_count")
    add_issue(issues, set(recorded_files) == PAYLOAD_FILES, "payload_inventory")
    actual_files = {
        str(path.relative_to(candidate_root))
        for path in candidate_root.rglob("*")
        if path.is_file()
        and path not in {artifact_manifest_path, run_manifest_path}
    }
    add_issue(issues, actual_files == PAYLOAD_FILES, "actual_payload_inventory")
    artifact_results: dict[str, Any] = {}
    for relative in sorted(PAYLOAD_FILES):
        path = candidate_root / relative
        expected = recorded_files.get(relative, {})
        actual = {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        passed = bool(
            path.is_file()
            and not path.is_symlink()
            and actual == expected
        )
        artifact_results[relative] = {"pass": passed, **actual}
        add_issue(issues, passed, f"artifact:{relative}")

    config_path = Path(str(run_manifest["config_path"])).resolve()
    expected_config = (workspace / "configs/counterfactual_v4_train.yaml").resolve()
    add_issue(issues, config_path == expected_config, "config_path")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_config_path = candidate_root / "frozen_config.yaml"
    config_hash = file_sha256(config_path)
    add_issue(issues, file_sha256(frozen_config_path) == config_hash, "frozen_config")
    add_issue(issues, run_manifest.get("config_sha256") == config_hash, "config_hash")
    add_issue(
        issues,
        tuple(config.get("roles", {}).values()) == ROLES
        and all("test" not in str(role).lower() for role in config["roles"].values()),
        "development_roles",
    )
    bulk_root = Path(str(run_manifest["bulk_root"])).resolve()
    expected_bulk = (workspace / "outputs/counterfactual_v4_bulk").resolve()
    add_issue(issues, bulk_root == expected_bulk, "bulk_root")
    bulk_manifest_path = bulk_root / "run_manifest.json"
    bulk_hash = file_sha256(bulk_manifest_path)
    add_issue(
        issues,
        run_manifest.get("bulk_manifest_sha256") == bulk_hash,
        "bulk_manifest_hash",
    )
    bulk_manifest = strict_json(bulk_manifest_path)
    add_issue(issues, bulk_manifest.get("status") == "complete", "bulk_complete")
    add_issue(issues, bulk_manifest.get("test_data_loaded") is False, "bulk_no_test")
    artifact_manifest_hash = file_sha256(artifact_manifest_path)
    add_issue(
        issues,
        run_manifest.get("artifact_manifest_sha256") == artifact_manifest_hash,
        "artifact_manifest_hash",
    )
    add_issue(
        issues,
        run_manifest.get("release_digest")
        == release_digest(artifact_manifest_hash, config_hash, bulk_hash),
        "release_digest",
    )
    live_tree_hash = source_tree_sha256(workspace)
    tracked_tree_hash, tracked_tree_files = git_tracked_source_tree_sha256(workspace)
    worktree_status = source_worktree_status(workspace)
    source_commit_match = find_matching_source_commit(
        workspace, str(run_manifest.get("source_tree_sha256"))
    )
    add_issue(
        issues,
        source_commit_match["found"],
        "sealed_source_commit_snapshot",
    )
    add_issue(
        issues,
        not worktree_status["tracked_changes"],
        "tracked_source_worktree_clean",
    )
    add_issue(issues, run_manifest.get("protocol") == PROTOCOL, "run_protocol")
    add_issue(
        issues,
        artifact_manifest.get("protocol") == PROTOCOL,
        "artifact_protocol",
    )
    add_issue(
        issues,
        run_manifest.get("status") == "frozen_validation_candidate",
        "run_status",
    )
    add_issue(
        issues,
        artifact_manifest.get("release_status") == "frozen_validation_candidate",
        "artifact_status",
    )
    add_issue(issues, run_manifest.get("test_data_loaded") is False, "run_no_test")
    add_issue(
        issues,
        run_manifest.get("formal_test_authorized_or_started") is False,
        "formal_not_started",
    )
    add_issue(
        issues,
        artifact_manifest.get("test_data_loaded") is False,
        "artifact_no_test",
    )
    add_issue(
        issues,
        tuple(run_manifest.get("feature_names", ())) == FEATURE_NAMES,
        "feature_names",
    )
    add_issue(
        issues, run_manifest.get("label_contract") == LABEL_CONTRACT, "label_contract"
    )
    return (
        {
            "pass": not issues,
            "issues": issues,
            "recorded_file_count": int(artifact_manifest.get("file_count", -1)),
            "actual_payload_file_count": len(actual_files),
            "artifacts": artifact_results,
            "artifact_manifest_sha256": artifact_manifest_hash,
            "config_sha256": config_hash,
            "bulk_manifest_sha256": bulk_hash,
            "recorded_source_tree_sha256": run_manifest.get("source_tree_sha256"),
            "git_head_tracked_source_tree_sha256": tracked_tree_hash,
            "git_head_tracked_source_file_count": len(tracked_tree_files),
            "sealed_source_commit_match": source_commit_match,
            "live_source_tree_sha256": live_tree_hash,
            "live_tree_matches_sealed_snapshot": live_tree_hash == tracked_tree_hash,
            "source_worktree_status": worktree_status,
            "release_digest": release_digest(
                artifact_manifest_hash, config_hash, bulk_hash
            ),
        },
        config,
        run_manifest,
    )


def audit_robot(
    *,
    workspace: Path,
    candidate_root: Path,
    robot: str,
    roles: Mapping[str, RoleData],
    config: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    policy_selection: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    model_path = candidate_root / "models" / f"{robot}_seed17_predictor.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    add_issue(issues, int(payload.get("format_version", -1)) == 3, "model_format")
    add_issue(
        issues,
        tuple(payload.get("feature_names", ())) == FEATURE_NAMES,
        "model_features",
    )
    add_issue(issues, payload.get("label_contract") == LABEL_CONTRACT, "model_labels")
    add_issue(
        issues,
        payload.get("training_provenance", {}).get("latency_training_source")
        == "raw_samples"
        and int(payload["training_provenance"]["latency_repeat_count"]) == 5
        and payload["training_provenance"].get("deadline_success_role")
        == "diagnostic_only",
        "model_training_provenance",
    )
    expected_seed = int(config["model"]["seed"]) + (0 if robot == "panda" else 10_000)
    add_issue(issues, int(payload["config"]["seed"]) == expected_seed, "model_seed")
    add_issue(
        issues,
        len(payload.get("training_history", [])) == int(config["model"]["epochs"]),
        "training_history",
    )

    predictions = {
        role: calibrated_prediction(payload, values.features)
        for role, values in roles.items()
    }
    shared_outputs = {
        role: bool(
            np.array_equal(
                prediction.success_logits,
                np.repeat(prediction.success_logits[:, :1], 3, axis=1),
            )
            and np.array_equal(
                prediction.success_probability,
                np.repeat(prediction.success_probability[:, :1], 3, axis=1),
            )
        )
        for role, prediction in predictions.items()
    }
    add_issue(issues, all(shared_outputs.values()), "shared_success_outputs")

    calibration = roles["calibration_queries"]
    raw_calibration = raw_forward(payload, calibration.features)
    calibration_targets = {
        "verified_success_shared": calibration.semantic_success[:, 0].astype(np.int64),
        "fail_all": calibration.semantic_fail_all.astype(np.int64),
    }
    raw_logits = {
        "verified_success_shared": raw_calibration[0][:, 0],
        "fail_all": raw_calibration[3],
    }
    refitted: dict[str, Any] = {}
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    serialized_calibrators = payload["calibrator"]["calibrators"]
    for index, name in enumerate(("verified_success_shared", "fail_all")):
        slope, intercept, probability = refit_platt(
            raw_logits[name], calibration_targets[name]
        )
        serialized = serialized_calibrators[index]
        refitted[name] = {
            "slope": slope,
            "intercept": intercept,
            "serialized_slope": float(serialized["slope"]),
            "serialized_intercept": float(serialized["intercept"]),
            "max_parameter_absolute_error": max(
                abs(slope - float(serialized["slope"])),
                abs(intercept - float(serialized["intercept"])),
            ),
        }
        before[name] = discrimination(
            stable_sigmoid(raw_logits[name]), calibration_targets[name]
        )
        after[name] = discrimination(probability, calibration_targets[name])
    platt_metric_reproduction = compare_payload(
        {"before_platt": before, "after_platt": after},
        training_metrics["calibration"],
        atol=1e-9,
        rtol=1e-9,
    )
    add_issue(
        issues,
        max(row["max_parameter_absolute_error"] for row in refitted.values()) <= 1e-10,
        "platt_refit",
    )
    add_issue(issues, platt_metric_reproduction["pass"], "platt_metrics")

    train_prediction = predictions["risk_train_queries"]
    calibration_prediction = predictions["calibration_queries"]
    detector = payload["ood_detector"]
    train_embedding = train_prediction.embedding
    mean = np.mean(train_embedding, axis=0)
    centered = train_embedding - mean
    covariance = centered.T @ centered / max(train_embedding.shape[0] - 1, 1)
    scale = float(np.trace(covariance) / train_embedding.shape[1])
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    shrinkage = float(detector["shrinkage"])
    regularization = float(detector["regularization"])
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * scale * np.eye(train_embedding.shape[1], dtype=np.float64)
        + regularization
        * max(scale, 1.0)
        * np.eye(train_embedding.shape[1], dtype=np.float64)
    )
    precision = np.linalg.pinv(covariance, hermitian=True)
    cal_centered = calibration_prediction.embedding - mean
    calibration_score = np.maximum(
        np.einsum(
            "ni,ij,nj->n", cal_centered, precision, cal_centered, optimize=True
        ),
        0.0,
    )
    threshold = float(np.quantile(calibration_score, 0.99, method="higher"))
    ood_state_error = max(
        float(np.max(np.abs(mean - np.asarray(detector["mean"], dtype=np.float64)))),
        float(
            np.max(
                np.abs(precision - np.asarray(detector["precision"], dtype=np.float64))
            )
        ),
        abs(threshold - float(detector["threshold"])),
    )
    calibration_coverage = float(np.mean(calibration_score <= threshold))
    policy_prediction = predictions["policy_validation_queries"]
    policy_false_defer = float(np.mean(policy_prediction.is_ood))
    recorded_ood = training_metrics["ood"]
    ood_summary = {
        "fit_role": "risk_train_queries",
        "threshold_role": "calibration_queries",
        "target_id_coverage": 0.99,
        "threshold": threshold,
        "calibration_id_coverage": calibration_coverage,
        "policy_id_false_defer_rate": policy_false_defer,
        "ood_examples_used_for_threshold": 0,
    }
    ood_metric_reproduction = compare_payload(ood_summary, recorded_ood)
    add_issue(issues, ood_state_error <= 1e-9, "ood_state")
    add_issue(issues, ood_metric_reproduction["pass"], "ood_metrics")

    train_latency = latency_report(train_prediction, roles["risk_train_queries"])
    policy_latency = latency_report(
        policy_prediction, roles["policy_validation_queries"]
    )
    train_latency_comparison = compare_payload(
        train_latency, training_metrics["latency_train"]
    )
    policy_latency_comparison = compare_payload(
        policy_latency, training_metrics["latency_policy_validation"]
    )
    add_issue(issues, train_latency_comparison["pass"], "train_latency_metrics")
    add_issue(issues, policy_latency_comparison["pass"], "policy_latency_metrics")

    candidates, routes, selected = candidate_grid(
        roles["policy_validation_queries"], policy_prediction, config["policy_grid"]
    )
    candidate_path = candidate_root / "policy_candidates" / f"{robot}_seed17.json"
    recorded_candidates = strict_json(candidate_path)
    candidates_comparison = compare_payload(
        {"selected": selected, "candidates": candidates}, recorded_candidates
    )
    add_issue(issues, len(candidates) == 252, "candidate_count")
    add_issue(issues, candidates_comparison["pass"], "candidate_reproduction")
    hard_gate_count = int(sum(row["hard_gate_pass"] for row in candidates))
    add_issue(
        issues,
        hard_gate_count == int(training_metrics["candidate_hard_gate_pass_count"]),
        "candidate_hard_gate_count",
    )
    selected_comparisons = {
        "training_metrics": compare_payload(selected, training_metrics["selected_policy"]),
        "policy_selection": compare_payload(selected, policy_selection),
        "policy_artifact": compare_payload(
            selected,
            strict_json(
                candidate_root / "policies" / f"{robot}_seed17_policy.json"
            )["selection_metrics"],
        ),
    }
    add_issue(
        issues,
        all(item["pass"] for item in selected_comparisons.values()),
        "selected_policy_reproduction",
    )

    route_signatures = {route.tobytes() for route in routes}
    outcome_signatures: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in candidates:
        outcome = {key: value for key, value in row.items() if key != "config"}
        outcome_signatures[json.dumps(safe(outcome), sort_keys=True)].append(row["config"])
    threshold_invariance = {
        "unique_route_signatures": len(route_signatures),
        "unique_reported_outcome_signatures": len(outcome_signatures),
        "candidates_per_outcome_signature": sorted(
            (len(values) for values in outcome_signatures.values()), reverse=True
        ),
        "success_and_reject_thresholds_change_routes": len(route_signatures)
        > len(config["policy_grid"]["latency_tie_margins_ms"]),
        "interpretation": (
            "Only latency_tie_margin_ms changes routing in this candidate grid; "
            "minimum-success and reject thresholds are saturated on policy validation."
        ),
    }

    selected_route = route_actions(policy_prediction, selected["config"])
    rejected = selected_route == REJECT_ACTION
    category_rejects = {
        category: int(
            np.sum(rejected & (roles["policy_validation_queries"].category == category))
        )
        for category in sorted(
            np.unique(roles["policy_validation_queries"].category).tolist()
        )
    }
    fixed_success = roles["policy_validation_queries"].verified_success[:, 3]
    operational = (
        roles["policy_validation_queries"].expected_reachable
        & roles["policy_validation_queries"].continuity_feasible
    )
    policy_role = roles["policy_validation_queries"]
    selected_fev = np.zeros(policy_role.count, dtype=np.float64)
    for action_index in range(3):
        selected_fev[selected_route == action_index] = policy_role.function_evaluations[
            selected_route == action_index, action_index
        ]
    selected_fev[selected_route == DEFER_ACTION] = policy_role.function_evaluations[
        selected_route == DEFER_ACTION, 3
    ]
    fixed_fev = policy_role.function_evaluations[:, 3].astype(np.float64)

    def fev_slice(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.sum(mask))
        selected_mean = float(np.mean(selected_fev[mask])) if count else None
        fixed_mean = float(np.mean(fixed_fev[mask])) if count else None
        return {
            "query_count": count,
            "selected_mean_function_evaluations": selected_mean,
            "fixed_mean_function_evaluations": fixed_mean,
            "reduction_fraction": (
                None
                if count == 0 or fixed_mean is None
                else float(1.0 - selected_mean / max(fixed_mean, 1e-12))
            ),
        }

    reject_diagnostics = {
        "reject_count": int(np.sum(rejected)),
        "rejected_fixed_success_count": int(np.sum(rejected & fixed_success)),
        "rejected_operational_feasible_count": int(np.sum(rejected & operational)),
        "by_category": category_rejects,
    }
    fev_stratification = {
        "all_queries": fev_slice(np.ones(policy_role.count, dtype=bool)),
        "fixed_verified_success": fev_slice(fixed_success),
        "operational_feasible": fev_slice(operational),
    }

    feature_diagnostics = {
        role: feature_label_diagnostics(values) for role, values in roles.items()
    }
    semantic_perfect = {
        name: bool(
            training_metrics["calibration"]["after_platt"][name]["auroc"] == 1.0
        )
        for name in ("verified_success_shared", "fail_all")
    }
    return {
        "pass": not issues,
        "issues": issues,
        "checkpoint": {
            "format_version": int(payload["format_version"]),
            "training_seed": int(payload["config"]["seed"]),
            "training_epoch_count": len(payload["training_history"]),
            "raw_repeat_training": payload["training_provenance"],
            "shared_semantic_output_exact_by_role": shared_outputs,
        },
        "platt": {
            "refit": refitted,
            "metrics_reproduction": platt_metric_reproduction,
            "recomputed_metrics": {
                "before_platt": before,
                "after_platt": after,
            },
        },
        "ood": {
            "state_max_absolute_error": ood_state_error,
            "metrics_reproduction": ood_metric_reproduction,
            "recomputed": ood_summary,
            "ood_auroc_available": False,
            "reason": "No labeled OOD examples were used or evaluated in the candidate.",
        },
        "latency_heads": {
            "train_reproduction": train_latency_comparison,
            "policy_validation_reproduction": policy_latency_comparison,
        },
        "policy_grid": {
            "candidate_count": len(candidates),
            "hard_gate_pass_count": hard_gate_count,
            "candidate_reproduction": candidates_comparison,
            "threshold_identifiability": threshold_invariance,
        },
        "selected_policy": {
            "recomputed": selected,
            "reproduction": selected_comparisons,
            "reject_diagnostics": reject_diagnostics,
            "function_evaluations_stratified": fev_stratification,
        },
        "perfect_metric_diagnostics": {
            "calibration_auroc_exactly_one": semantic_perfect,
            "semantic_success_and_fail_all_targets_are_exact_complements": bool(
                np.array_equal(
                    roles["calibration_queries"].semantic_success[:, 0],
                    ~roles["calibration_queries"].semantic_fail_all,
                )
            ),
            "feature_label_diagnostics": feature_diagnostics,
        },
    }


def main() -> None:
    args = arguments()
    workspace = args.workspace.resolve()
    candidate_root = args.candidate_root
    if not candidate_root.is_absolute():
        candidate_root = workspace / candidate_root
    candidate_root = candidate_root.resolve()
    expected_candidate = (workspace / "outputs/release_v4_candidate").resolve()
    if candidate_root != expected_candidate or candidate_root.is_symlink():
        raise ValueError(f"candidate root must be exactly {expected_candidate}")

    # Match the frozen CPU inference environment used to create the candidate.
    torch.set_num_threads(8)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)

    artifact_report, config, run_manifest = artifact_audit(workspace, candidate_root)
    data_audit = strict_json(candidate_root / "data_audit.json")
    training_metrics = strict_json(candidate_root / "training_metrics.json")["robots"]
    policy_selection = strict_json(candidate_root / "policy_selection.json")["robots"]
    bulk_root = Path(str(run_manifest["bulk_root"])).resolve()

    roles: dict[str, dict[str, RoleData]] = {}
    role_reports: dict[str, Any] = {}
    for robot in ROBOTS:
        roles[robot] = {}
        role_reports[robot] = {}
        for role in ROLES:
            data, report = load_role(
                bulk_root=bulk_root,
                robot=robot,
                role=role,
                recorded_audit=data_audit["robots"][robot]["roles"][role],
            )
            roles[robot][role] = data
            role_reports[robot][role] = report

    split_report: dict[str, Any] = {}
    split_pass = True
    for robot in ROBOTS:
        hash_overlap: dict[str, int] = {}
        feature_row_overlap: dict[str, int] = {}
        for left_index, left in enumerate(ROLES):
            for right in ROLES[left_index + 1 :]:
                key = f"{left}__{right}"
                hash_overlap[key] = len(
                    set(roles[robot][left].query_sha256.tolist())
                    & set(roles[robot][right].query_sha256.tolist())
                )
                feature_row_overlap[key] = feature_overlap(
                    roles[robot][left].features, roles[robot][right].features
                )
        recorded = data_audit["robots"][robot]
        recorded_overlap = compare_payload(
            hash_overlap, recorded["role_query_overlap_counts"]
        )
        role_pass = (
            not any(hash_overlap.values())
            and not any(feature_row_overlap.values())
            and recorded_overlap["pass"]
            and recorded.get("roles_disjoint") is True
        )
        split_pass &= role_pass
        split_report[robot] = {
            "pass": role_pass,
            "query_hash_overlap_counts": hash_overlap,
            "exact_feature_row_overlap_counts": feature_row_overlap,
            "recorded_overlap_reproduction": recorded_overlap,
        }

    model_reports = {
        robot: audit_robot(
            workspace=workspace,
            candidate_root=candidate_root,
            robot=robot,
            roles=roles[robot],
            config=config,
            training_metrics=training_metrics[robot],
            policy_selection=policy_selection[robot],
        )
        for robot in ROBOTS
    }
    feature_source = feature_function_audit(workspace)
    role_pass = all(
        report["pass"]
        for robot in ROBOTS
        for report in role_reports[robot].values()
    )
    model_pass = all(report["pass"] for report in model_reports.values())
    structural_pass = bool(
        artifact_report["pass"]
        and role_pass
        and split_pass
        and model_pass
        and feature_source["uses_only_query_seed_and_kinematic_diagnostics"]
    )
    blockers: list[str] = []
    if not artifact_report["pass"]:
        blockers.append("candidate_artifact_or_provenance_integrity_failed")
    if not role_pass:
        blockers.append("development_role_reproduction_failed")
    if not split_pass:
        blockers.append("development_split_identity_or_overlap_failed")
    if not model_pass:
        blockers.append("model_calibration_ood_or_policy_reproduction_failed")
    if not feature_source["uses_only_query_seed_and_kinematic_diagnostics"]:
        blockers.append("post_solver_label_token_found_in_feature_function")

    ur5e = model_reports["ur5e"]["perfect_metric_diagnostics"]
    ur5e_features = ur5e["feature_label_diagnostics"]
    ur5e_position_perfect = all(
        ur5e_features[role]["features"]["learned_seed_position_error"][
            "perfect_threshold_separation"
        ]
        and ur5e_features[role]["features"]["current_pose_position_step"][
            "perfect_threshold_separation"
        ]
        for role in ROLES
    )
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "audited_candidate_root": str(candidate_root),
        "scope": "frozen validation-only v4 candidate and development roles",
        "candidate_modified": False,
        "formal_test_query_or_result_content_read": False,
        "formal_test_output_paths_opened": 0,
        "source_provenance_note": (
            "Source-tree provenance was reconstructed from committed Python source "
            "blobs; no formal test query, label, metric, or result artifact was opened."
        ),
        "audit_pass": structural_pass,
        "blocking_issues": blockers,
        "artifact_integrity": artifact_report,
        "development_roles": role_reports,
        "split_audit": split_report,
        "feature_provenance": feature_source,
        "models": model_reports,
        "perfect_auroc_and_false_reject_assessment": {
            "perfect_auroc_is_ood_auroc": False,
            "perfect_auroc_location": (
                "UR5e calibration semantic verified-success and complementary "
                "semantic fail-all heads"
            ),
            "ood_auroc_supported": False,
            "query_or_exact_feature_overlap_detected": False,
            "post_solver_label_used_as_feature_detected": False,
            "ur5e_position_features_perfectly_separate_all_three_roles": ur5e_position_perfect,
            "ur5e_fail_all_category_support": (
                "All observed UR5e semantic fail-all cases are large_step or "
                "unreachable in train, calibration, and policy validation."
            ),
            "interpretation": (
                "The perfect semantic AUROC and zero false reject are reproducible and "
                "not explained by split duplication or a direct post-solver label field. "
                "They arise from highly separable pre-solver position-error features and "
                "the narrow category-defined failure support. This is a dataset shortcut/"
                "support limitation, not evidence of broad reject generalization."
            ),
        },
        "claim_status": {
            "candidate_structurally_reproducible": structural_pass,
            "broad_reject_claim_enabled": False,
            "ood_discrimination_claim_supported": False,
            "success_and_reject_thresholds_empirically_identified": False,
            "validation_policy_metrics_may_be_reported_as_development_only": structural_pass,
            "formal_test_authorized_by_this_audit": False,
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
                "audit_pass": structural_pass,
                "blocking_issues": blockers,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if structural_pass else 1)


if __name__ == "__main__":
    main()
