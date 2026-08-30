from __future__ import annotations

"""Validation-only training, calibration, and policy locking for CG-HIK v4.

The runner has a deliberately narrow input contract: it reads only the three
development roles produced by :mod:`counterfactual_v4.bulk_runner`.  It never
discovers a paper/test directory and refuses a bulk artifact whose role or
completion manifest is inconsistent.  Latency supervision is taken from all
five raw ``latency_samples_ns`` observations; the stored per-query P50/P95
fields are used only for diagnostics.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import yaml

from ..config import resolve_path
from ..experiments.provenance import environment_payload, source_tree_hash
from .calibration import binary_calibration_metrics, sigmoid
from .model import (
    CALIBRATION_HEAD_NAMES,
    FEATURE_NAMES,
    LABEL_CONTRACT,
    CounterfactualPrediction,
    CounterfactualTrainingConfig,
    CounterfactualV4Predictor,
)
from .policy import DECISION_ENTRIES, V4PolicyConfig


PROTOCOL = "counterfactual_v4_validation_training_v2"
ROLE_ORDER = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)
COLLECTED_ACTIONS = (*DECISION_ENTRIES, "fixed_robust")
REJECT_ACTION = 3
DEFER_ACTION = 4


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and lock the validation-only v4 routing candidate"
    )
    parser.add_argument("--config", required=True)
    return parser


@dataclass(frozen=True)
class RoleArrays:
    features: np.ndarray
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
    def decision_verified_success(self) -> np.ndarray:
        """Semantic solver-plus-verifier success for the learned action heads."""

        return self.verified_success[:, : len(DECISION_ENTRIES)].astype(
            np.float32
        )

    @property
    def decision_deadline_success(self) -> np.ndarray:
        """Deadline diagnostic; never a model or calibration target."""

        return self.verified_success_before_deadline[
            :, : len(DECISION_ENTRIES)
        ].astype(np.float32)

    @property
    def fail_all(self) -> np.ndarray:
        return np.all(
            ~self.verified_success[:, : len(DECISION_ENTRIES)], axis=1
        ).astype(np.float32)

    @property
    def decision_latency_samples_ms(self) -> np.ndarray:
        # Conversion occurs only after the exact raw-nanosecond tensor has
        # passed schema validation and been included in the provenance hash.
        return (
            self.latency_samples_ns[:, : len(DECISION_ENTRIES)].astype(np.float32)
            / np.float32(1e6)
        )


def _artifact_metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size}


def _release_artifact_metadata(path: Path, release_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(release_root)),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _validate_chunk_manifest(chunk: Path) -> dict[str, Any]:
    manifest_path = chunk / "chunk_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"missing regular chunk manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("environment_contaminated", True)):
        raise RuntimeError(f"contaminated timing chunk is not trainable: {chunk}")
    if bool(manifest.get("test_data_loaded", True)):
        raise RuntimeError(f"test-loaded chunk is forbidden: {chunk}")
    for name, expected in manifest.get("artifacts", {}).items():
        artifact = chunk / str(name)
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"missing chunk artifact: {artifact}")
        if artifact.stat().st_size != int(expected["size"]):
            raise RuntimeError(f"chunk size mismatch: {artifact}")
        if _sha256_file(artifact) != str(expected["sha256"]):
            raise RuntimeError(f"chunk hash mismatch: {artifact}")
    return manifest


def _load_role(bulk_root: Path, robot: str, training_seed: int, role: str) -> RoleArrays:
    if role not in ROLE_ORDER or "test" in role.lower():
        raise ValueError(f"forbidden development role: {role}")
    role_root = bulk_root / robot / f"seed{training_seed}" / role
    selection_path = role_root / "selection.npz"
    selection_manifest_path = role_root / "selection_manifest.json"
    if not selection_path.is_file() or not selection_manifest_path.is_file():
        raise FileNotFoundError(f"bulk selection is incomplete: {role_root}")
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_manifest.get("source_role") != role:
        raise RuntimeError(f"selection role mismatch: {selection_manifest_path}")
    if bool(selection_manifest.get("test_named_dataset_loaded", True)):
        raise RuntimeError(f"test-named source is forbidden: {selection_manifest_path}")

    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "features",
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
    source_files: list[dict[str, Any]] = [
        _artifact_metadata(selection_path),
        _artifact_metadata(selection_manifest_path),
    ]
    chunks = sorted((role_root / "chunks").glob("chunk_*"))
    if not chunks:
        raise RuntimeError(f"role contains no committed chunks: {role_root}")
    expected_start = 0
    feature_names: tuple[str, ...] | None = None
    for chunk in chunks:
        if not chunk.is_dir() or chunk.is_symlink():
            raise RuntimeError(f"invalid chunk path: {chunk}")
        manifest = _validate_chunk_manifest(chunk)
        start = int(manifest["query_start"])
        stop = int(manifest["query_stop_exclusive"])
        if (
            manifest.get("robot") != robot
            or int(manifest.get("training_seed", -1)) != training_seed
            or manifest.get("source_role") != role
            or start != expected_start
            or stop <= start
        ):
            raise RuntimeError(f"non-contiguous or mismatched chunk: {chunk}")
        labels_path = chunk / "counterfactual_labels.npz"
        with np.load(labels_path, allow_pickle=False) as labels:
            actions = tuple(labels["action_names"].astype(str).tolist())
            decisions = tuple(labels["decision_action_names"].astype(str).tolist())
            names = tuple(labels["feature_names"].astype(str).tolist())
            if actions != COLLECTED_ACTIONS or decisions != DECISION_ENTRIES:
                raise RuntimeError(f"action schema mismatch: {labels_path}")
            if names != FEATURE_NAMES:
                raise RuntimeError(
                    f"canonical feature schema mismatch: {labels_path}; "
                    f"expected={FEATURE_NAMES}, got={names}"
                )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise RuntimeError(f"feature schema changed across chunks: {labels_path}")
            count = stop - start
            if labels["features"].shape[0] != count:
                raise RuntimeError(f"chunk row count mismatch: {labels_path}")
            for name in fields:
                fields[name].append(np.asarray(labels[name]))
        source_files.extend(
            _artifact_metadata(path)
            for path in (chunk / "chunk_manifest.json", labels_path)
        )
        expected_start = stop

    arrays = {name: np.concatenate(values, axis=0) for name, values in fields.items()}
    expected_count = int(selection_manifest["selected_query_count"])
    if expected_start != expected_count or arrays["features"].shape[0] != expected_count:
        raise RuntimeError(f"role is not complete: {role_root}")
    with np.load(selection_path, allow_pickle=False) as selection:
        if not np.array_equal(
            arrays["query_sha256"].astype(str), selection["query_sha256"].astype(str)
        ):
            raise RuntimeError(f"chunk queries do not match frozen selection: {role_root}")

    features = np.asarray(arrays["features"], dtype=np.float32)
    success = np.asarray(arrays["verified_success"], dtype=bool)
    deadline = np.asarray(arrays["verified_success_before_deadline"], dtype=bool)
    latency_ns = np.asarray(arrays["latency_samples_ns"])
    evaluations = np.asarray(arrays["function_evaluations"], dtype=np.int64)
    fallback = np.asarray(arrays["fallback_used"], dtype=bool)
    expected_matrix = (expected_count, len(COLLECTED_ACTIONS))
    if features.ndim != 2 or features.shape[1] != 9 or not np.all(np.isfinite(features)):
        raise RuntimeError(f"invalid feature matrix: {role_root}")
    if success.shape != expected_matrix or deadline.shape != expected_matrix:
        raise RuntimeError(f"invalid success label matrix: {role_root}")
    decision_success = success[:, : len(DECISION_ENTRIES)]
    if not np.array_equal(
        decision_success,
        np.repeat(decision_success[:, :1], len(DECISION_ENTRIES), axis=1),
    ):
        raise RuntimeError(
            "semantic verified-success differs across terminal-fallback-invariant "
            f"actions: {role_root}"
        )
    if evaluations.shape != expected_matrix or fallback.shape != expected_matrix:
        raise RuntimeError(f"invalid solver label matrix: {role_root}")
    if latency_ns.shape != (*expected_matrix, 5):
        raise RuntimeError(
            "formal v4 requires raw latency_samples_ns with shape "
            f"({expected_count}, 4, 5), got {latency_ns.shape}"
        )
    if latency_ns.dtype.kind not in "iu" or np.any(latency_ns <= 0):
        raise RuntimeError(f"raw latency samples must be positive integer ns: {role_root}")
    if len(np.unique(arrays["query_sha256"].astype(str))) != expected_count:
        raise RuntimeError(f"duplicate query identities in role: {role_root}")
    # fixed_robust is an audited semantic alias of easy in the bulk artifact.
    for name, values in (
        ("verified_success", success),
        ("verified_success_before_deadline", deadline),
        ("function_evaluations", evaluations),
        ("fallback_used", fallback),
        ("latency_samples_ns", latency_ns),
    ):
        if not np.array_equal(values[:, 0], values[:, 3]):
            raise RuntimeError(f"easy/fixed alias mismatch in {name}: {role_root}")

    return RoleArrays(
        features=features,
        query_sha256=arrays["query_sha256"].astype("U64"),
        category=arrays["category"].astype(str),
        expected_reachable=np.asarray(arrays["expected_reachable"], dtype=bool),
        continuity_feasible=np.asarray(arrays["continuity_feasible"], dtype=bool),
        verified_success=success,
        verified_success_before_deadline=deadline,
        latency_samples_ns=latency_ns.astype(np.int64, copy=False),
        function_evaluations=evaluations,
        fallback_used=fallback,
        source_files=tuple(source_files),
    )


def _binary_discrimination(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    result: dict[str, Any] = binary_calibration_metrics(probability, target)
    if np.unique(target).size == 2:
        result.update(
            {
                "auroc": float(roc_auc_score(target, probability)),
                "auprc": float(average_precision_score(target, probability)),
            }
        )
    else:
        result.update({"auroc": None, "auprc": None})
    result["positive_rate"] = float(np.mean(target))
    return result


def _calibration_report(
    predictor: CounterfactualV4Predictor,
    role: RoleArrays,
    *,
    raw_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = predictor.predict(role.features)
    probabilities = np.column_stack(
        [prediction.verified_success_probability[:, 0], prediction.fail_all_probability]
    )
    targets = np.column_stack([role.decision_verified_success[:, 0], role.fail_all])
    report = {
        name: _binary_discrimination(probabilities[:, index], targets[:, index])
        for index, name in enumerate(CALIBRATION_HEAD_NAMES)
    }
    if raw_logits is not None:
        raw_probability = sigmoid(raw_logits)
        report = {
            "before_platt": {
                name: _binary_discrimination(raw_probability[:, index], targets[:, index])
                for index, name in enumerate(CALIBRATION_HEAD_NAMES)
            },
            "after_platt": report,
        }
    return report


def _latency_report(prediction: CounterfactualPrediction, role: RoleArrays) -> dict[str, Any]:
    samples = role.decision_latency_samples_ms.astype(np.float64)
    target_p50 = np.quantile(samples, 0.50, axis=2)
    target_p95 = np.quantile(samples, 0.95, axis=2)
    result: dict[str, Any] = {}
    for index, action in enumerate(DECISION_ENTRIES):
        p50 = prediction.latency_p50_ms[:, index]
        p95 = prediction.latency_p95_ms[:, index]
        action_samples = samples[:, index]
        result[action] = {
            "diagnostic_empirical_p50_mae_ms": float(np.mean(np.abs(p50 - target_p50[:, index]))),
            "diagnostic_empirical_p95_mae_ms": float(np.mean(np.abs(p95 - target_p95[:, index]))),
            "raw_sample_p50_coverage": float(np.mean(action_samples <= p50[:, None])),
            "raw_sample_p95_coverage": float(np.mean(action_samples <= p95[:, None])),
            "predicted_p50_median_ms": float(np.median(p50)),
            "predicted_p95_median_ms": float(np.median(p95)),
        }
    result["supervision"] = {
        "source_field": "latency_samples_ns",
        "raw_repeat_count": int(samples.shape[2]),
        "raw_observation_count": int(samples.size),
        "per_query_winner_used_as_target": False,
        "aggregated_p50_p95_used_as_target": False,
    }
    return result


def _route_actions(
    prediction: CounterfactualPrediction,
    config: V4PolicyConfig,
) -> np.ndarray:
    count = prediction.verified_success_probability.shape[0]
    if prediction.is_ood is None:
        raise RuntimeError("policy selection requires a calibrated OOD detector")
    is_ood = np.asarray(prediction.is_ood, dtype=bool)
    eligible = (
        (prediction.verified_success_probability >= config.minimum_success_probability)
        & (prediction.latency_p95_ms <= config.deadline_ms)
    )
    has_eligible = np.any(eligible, axis=1)
    action = np.full(count, DEFER_ACTION, dtype=np.int64)
    reject = (
        ~is_ood
        & ~has_eligible
        & (prediction.fail_all_probability >= config.reject_probability)
    )
    action[reject] = REJECT_ACTION
    selectable = ~is_ood & has_eligible
    if np.any(selectable):
        masked = np.where(eligible, prediction.latency_p95_ms, np.inf)
        fastest = np.argmin(masked, axis=1)
        conservative = np.argmax(eligible, axis=1)  # first True index
        rows = np.arange(count)
        improvement = (
            prediction.latency_p95_ms[rows, conservative]
            - prediction.latency_p95_ms[rows, fastest]
        )
        chosen = np.where(
            improvement < config.latency_tie_margin_ms, conservative, fastest
        )
        action[selectable] = chosen[selectable]
    return action


def _policy_metrics(
    role: RoleArrays,
    prediction: CounterfactualPrediction,
    config: V4PolicyConfig,
) -> dict[str, Any]:
    selected = _route_actions(prediction, config)
    count = role.count
    rows = np.arange(count)
    fixed = 3
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
    command_success[defer] = role.verified_success[defer, fixed]
    deadline_success[defer] = role.verified_success_before_deadline[defer, fixed]
    fev[defer] = role.function_evaluations[defer, fixed]
    latency[defer] = role.latency_samples_ns[defer, fixed] / 1e6
    reject = selected == REJECT_ACTION
    fixed_success = role.verified_success[:, fixed]
    fixed_deadline = role.verified_success_before_deadline[:, fixed]
    fixed_fev = role.function_evaluations[:, fixed].astype(np.float64)
    operational_feasible = role.expected_reachable & role.continuity_feasible
    fixed_positive_count = max(int(np.sum(fixed_success)), 1)
    operational_count = max(int(np.sum(operational_feasible)), 1)
    successful_command_samples = latency[command_success]
    route_counts = {
        **{
            DECISION_ENTRIES[index]: int(np.sum(selected == index))
            for index in range(3)
        },
        "reject": int(np.sum(reject)),
        "defer": int(np.sum(defer)),
    }
    return {
        "config": asdict(config),
        "query_count": count,
        "route_counts": route_counts,
        "route_rates": {key: value / count for key, value in route_counts.items()},
        "fixed_verified_success_rate": float(np.mean(fixed_success)),
        "selected_verified_success_rate": float(np.mean(command_success)),
        "verified_success_difference_vs_fixed": float(
            np.mean(command_success) - np.mean(fixed_success)
        ),
        "fixed_deadline_success_rate": float(np.mean(fixed_deadline)),
        "selected_deadline_success_rate": float(np.mean(deadline_success)),
        "fixed_success_false_reject_rate": float(
            np.sum(reject & fixed_success) / fixed_positive_count
        ),
        "operational_feasible_false_reject_rate": float(
            np.sum(reject & operational_feasible) / operational_count
        ),
        "reject_precision_against_fixed_failure": (
            None
            if not np.any(reject)
            else float(np.mean(~fixed_success[reject]))
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
            if successful_command_samples.size == 0
            else float(np.quantile(successful_command_samples, 0.95))
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
            np.mean(command_success) - np.mean(fixed_success) >= -0.01
            and np.sum(reject & fixed_success) / fixed_positive_count <= 0.01
            and np.sum(reject & operational_feasible) / operational_count <= 0.01
        ),
    }


def _select_policy_configuration(
    role: RoleArrays,
    prediction: CounterfactualPrediction,
    *,
    minimum_success_probabilities: Sequence[float],
    reject_probabilities: Sequence[float],
    latency_tie_margins_ms: Sequence[float],
    deadline_ms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for success_threshold in minimum_success_probabilities:
        for reject_threshold in reject_probabilities:
            for tie_margin in latency_tie_margins_ms:
                config = V4PolicyConfig(
                    minimum_success_probability=float(success_threshold),
                    reject_probability=float(reject_threshold),
                    deadline_ms=float(deadline_ms),
                    latency_tie_margin_ms=float(tie_margin),
                )
                candidates.append(_policy_metrics(role, prediction, config))
    passing = [row for row in candidates if row["hard_gate_pass"]]
    if not passing:
        raise RuntimeError(
            "no policy configuration passed fixed-success noninferiority and false-reject gates"
        )

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
        # The policy role is used only once: first minimize observed successful
        # command P95, then FEV, with deterministic conservative tie breakers.
        success_p95 = row["successful_command_latency_p95_ms"]
        return (
            float("inf") if success_p95 is None else float(success_p95),
            float(row["mean_function_evaluations"]),
            -float(row["selected_verified_success_rate"]),
            float(row["operational_feasible_false_reject_rate"]),
            -float(row["config"]["latency_tie_margin_ms"]),
            -float(row["config"]["minimum_success_probability"]),
        )

    selected = min(passing, key=key)
    selected = {
        **selected,
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
    return selected, candidates


def _validate_config(config: Mapping[str, Any], config_path: Path) -> tuple[Path, Path, Path]:
    if int(config.get("protocol_version", -1)) != 4:
        raise ValueError("v4 training requires protocol_version: 4")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("v4 training requires robots: [panda, ur5e]")
    if int(config.get("training_seed", -1)) != 17:
        raise ValueError("v4 training requires the frozen development seed 17")
    roles = tuple(config.get("roles", {}).values())
    if roles != ROLE_ORDER or any("test" in str(role).lower() for role in roles):
        raise ValueError(f"roles must be exactly {ROLE_ORDER}")
    local = dict(config)
    local["_config_path"] = str(config_path)
    workspace = resolve_path(local, config.get("workspace", ".."))
    bulk_root = resolve_path(local, config["bulk_root"])
    output_root = resolve_path(local, config["output_root"])
    if bulk_root.resolve() != (workspace / "outputs" / "counterfactual_v4_bulk").resolve():
        raise ValueError("bulk_root must be exactly outputs/counterfactual_v4_bulk")
    if output_root.resolve() != (workspace / "outputs" / "release_v4_candidate").resolve():
        raise ValueError("output_root must be exactly outputs/release_v4_candidate")
    if bulk_root.is_symlink() or output_root.is_symlink():
        raise RuntimeError("bulk and release roots cannot be symlinks")
    model = config.get("model", {})
    hidden = tuple(int(value) for value in model.get("hidden_sizes", ()))
    if not hidden or len(hidden) > 3 or max(hidden) > 128:
        raise ValueError("v4 requires a compact MLP with at most three <=128-wide layers")
    if str(config.get("calibration", {}).get("method")) != "platt":
        raise ValueError("formal v4 calibration is frozen to Platt scaling")
    coverage = float(config.get("ood", {}).get("target_id_coverage", 0.0))
    if coverage != 0.99:
        raise ValueError("OOD threshold must be the ID calibration 99% coverage quantile")
    hard_gates = config.get("policy_grid", {}).get("hard_gates", {})
    if hard_gates != {
        "verified_success_difference_vs_fixed_minimum": -0.01,
        "fixed_success_false_reject_rate_maximum": 0.01,
        "operational_feasible_false_reject_rate_maximum": 0.01,
    }:
        raise ValueError(
            "policy hard gates must remain frozen at -1 pp success and 1% false reject"
        )
    return workspace, bulk_root, output_root


def _load_bulk_manifest(bulk_root: Path) -> dict[str, Any]:
    path = bulk_root / "run_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("counterfactual bulk collection is not complete")
    if tuple(manifest.get("decision_actions", ())) != DECISION_ENTRIES:
        raise RuntimeError("bulk decision-action schema mismatch")
    if tuple(manifest.get("collected_actions", ())) != COLLECTED_ACTIONS:
        raise RuntimeError("bulk collected-action schema mismatch")
    if not bool(manifest.get("selection_roles_disjoint", False)):
        raise RuntimeError("bulk roles are not declared disjoint")
    if bool(manifest.get("frozen_provenance", {}).get("test_data_loaded", True)):
        raise RuntimeError("bulk provenance indicates forbidden test access")
    return manifest


def _model_config(config: Mapping[str, Any], robot: str) -> CounterfactualTrainingConfig:
    values = dict(config["model"])
    values["hidden_sizes"] = tuple(int(width) for width in values["hidden_sizes"])
    base_seed = int(values.pop("seed"))
    values["seed"] = base_seed + (0 if robot == "panda" else 10_000)
    return CounterfactualTrainingConfig(**values)


def _fit_robot(
    *,
    robot: str,
    roles: Mapping[str, RoleArrays],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    train = roles["risk_train_queries"]
    calibration = roles["calibration_queries"]
    policy_role = roles["policy_validation_queries"]
    predictor = CounterfactualV4Predictor(
        _model_config(config, robot), device=str(config["runtime"]["device"])
    )
    predictor.fit(
        train.features,
        train.decision_verified_success,
        train.decision_latency_samples_ms,
        train.fail_all,
    )
    if not predictor.training_provenance["formal_v4_eligible"]:
        raise RuntimeError("model did not preserve raw-repeat training provenance")
    raw = predictor.predict(calibration.features)
    raw_logits = np.column_stack(
        [raw.verified_success_logits[:, 0], raw.fail_all_logit]
    )
    predictor.calibrate(
        calibration.features,
        calibration.decision_verified_success,
        calibration.fail_all,
        method="platt",
    )
    predictor.fit_ood_detector(
        train.features,
        calibration.features,
        target_id_coverage=float(config["ood"]["target_id_coverage"]),
        shrinkage=float(config["ood"]["shrinkage"]),
    )
    calibration_report = _calibration_report(
        predictor, calibration, raw_logits=raw_logits
    )
    policy_prediction = predictor.predict(policy_role.features)
    selected, candidates = _select_policy_configuration(
        policy_role,
        policy_prediction,
        minimum_success_probabilities=config["policy_grid"][
            "minimum_success_probabilities"
        ],
        reject_probabilities=config["policy_grid"]["reject_probabilities"],
        latency_tie_margins_ms=config["policy_grid"]["latency_tie_margins_ms"],
        deadline_ms=float(config["policy_grid"]["deadline_ms"]),
    )

    model_path = output_dir / "models" / f"{robot}_seed17_predictor.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(model_path)
    restored = CounterfactualV4Predictor.load(model_path, device="cpu")
    expected = predictor.predict(policy_role.features[: min(512, policy_role.count)])
    actual = restored.predict(policy_role.features[: min(512, policy_role.count)])
    equivalence: dict[str, Any] = {}
    for name in (
        "verified_success_logits",
        "verified_success_probability",
        "latency_p50_ms",
        "latency_p95_ms",
        "fail_all_probability",
        "embedding",
        "ood_score",
    ):
        left = np.asarray(getattr(expected, name))
        right = np.asarray(getattr(actual, name))
        equivalence[name] = {
            "max_absolute_error": float(np.max(np.abs(left - right))),
            "exact": bool(np.array_equal(left, right)),
        }
    equivalence["is_ood_exact"] = bool(
        np.array_equal(expected.is_ood, actual.is_ood)
    )
    equivalence["shared_semantic_success_logits_exact"] = bool(
        np.array_equal(
            actual.verified_success_logits,
            np.repeat(actual.verified_success_logits[:, :1], 3, axis=1),
        )
    )
    equivalence["shared_semantic_success_probabilities_exact"] = bool(
        np.array_equal(
            actual.verified_success_probability,
            np.repeat(actual.verified_success_probability[:, :1], 3, axis=1),
        )
    )
    if not all(
        row["exact"] for row in equivalence.values() if isinstance(row, dict)
    ) or not all(
        bool(equivalence[name])
        for name in (
            "is_ood_exact",
            "shared_semantic_success_logits_exact",
            "shared_semantic_success_probabilities_exact",
        )
    ):
        raise RuntimeError(f"saved model round-trip is not exact for {robot}")

    policy_path = output_dir / "policies" / f"{robot}_seed17_policy.json"
    _write_json(
        policy_path,
        {
            "robot": robot,
            "training_seed": 17,
            "policy_config": selected["config"],
            "selection_metrics": selected,
            "selection_role": "policy_validation_queries",
            "label_contract": dict(LABEL_CONTRACT),
            "shared_semantic_success_due_to_terminal_fallback_invariance": True,
            "test_data_loaded": False,
        },
    )
    train_prediction = predictor.predict(train.features)
    metrics = {
        "robot": robot,
        "training_seed": 17,
        "feature_names": list(FEATURE_NAMES),
        "label_contract": dict(LABEL_CONTRACT),
        "shared_semantic_success_due_to_terminal_fallback_invariance": True,
        "split_counts": {role: values.count for role, values in roles.items()},
        "training_provenance": predictor.training_provenance,
        "reject_claim_support": {
            "definition": (
                "semantic fail-all among queries marked expected_reachable and "
                "continuity_feasible by the frozen dataset contract"
            ),
            "risk_train_count": int(
                np.sum(
                    (train.fail_all > 0.5)
                    & train.expected_reachable
                    & train.continuity_feasible
                )
            ),
            "minimum_for_broad_reject_claim": 30,
            "broad_reject_claim_enabled": bool(
                np.sum(
                    (train.fail_all > 0.5)
                    & train.expected_reachable
                    & train.continuity_feasible
                )
                >= 30
            ),
            "insufficient_support_does_not_block_training": True,
        },
        "training_final_epoch": predictor.training_history[-1],
        "calibration": calibration_report,
        "latency_train": _latency_report(train_prediction, train),
        "latency_policy_validation": _latency_report(policy_prediction, policy_role),
        "ood": {
            "fit_role": "risk_train_queries",
            "threshold_role": "calibration_queries",
            "target_id_coverage": float(config["ood"]["target_id_coverage"]),
            "threshold": float(predictor.ood_detector.threshold),
            "calibration_id_coverage": float(
                1.0 - np.mean(predictor.predict(calibration.features).is_ood)
            ),
            "policy_id_false_defer_rate": float(np.mean(policy_prediction.is_ood)),
            "ood_examples_used_for_threshold": 0,
        },
        "selected_policy": selected,
        "candidate_count": len(candidates),
        "candidate_hard_gate_pass_count": int(
            sum(bool(row["hard_gate_pass"]) for row in candidates)
        ),
        "serialization_equivalence": equivalence,
        "model_artifact": _release_artifact_metadata(model_path, output_dir),
        "policy_artifact": _release_artifact_metadata(policy_path, output_dir),
        "test_data_loaded": False,
    }
    candidate_path = output_dir / "policy_candidates" / f"{robot}_seed17.json"
    _write_json(candidate_path, {"selected": selected, "candidates": candidates})
    metrics["candidate_artifact"] = _release_artifact_metadata(
        candidate_path, output_dir
    )
    return metrics


def _candidate_release_digest(
    artifact_manifest_sha256: str,
    config_sha256: str,
    bulk_manifest_sha256: str,
) -> str:
    return sha256(
        (artifact_manifest_sha256 + config_sha256 + bulk_manifest_sha256).encode(
            "ascii"
        )
    ).hexdigest()


def _verify_existing_candidate(
    output_root: Path,
    *,
    config_path: Path,
    bulk_root: Path,
) -> None:
    if output_root.is_symlink():
        raise RuntimeError(f"candidate root cannot be a symlink: {output_root}")
    artifact_manifest = output_root / "artifact_manifest.json"
    run_manifest = output_root / "run_manifest.json"
    if (
        not artifact_manifest.is_file()
        or artifact_manifest.is_symlink()
        or not run_manifest.is_file()
        or run_manifest.is_symlink()
    ):
        raise RuntimeError(f"candidate path exists but is not a sealed candidate: {output_root}")
    artifacts = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    run = json.loads(run_manifest.read_text(encoding="utf-8"))
    if artifacts.get("protocol") != PROTOCOL or artifacts.get("release_status") != (
        "frozen_validation_candidate"
    ):
        raise RuntimeError("candidate artifact manifest has the wrong protocol or status")
    if bool(artifacts.get("test_data_loaded", True)):
        raise RuntimeError("candidate artifact manifest indicates forbidden test access")
    if run.get("protocol") != PROTOCOL or run.get("status") != (
        "frozen_validation_candidate"
    ):
        raise RuntimeError("candidate run manifest has the wrong protocol or status")
    if bool(run.get("test_data_loaded", True)) or bool(
        run.get("formal_test_authorized_or_started", True)
    ):
        raise RuntimeError("candidate run manifest violates the validation-only boundary")
    artifact_manifest_sha256 = _sha256_file(artifact_manifest)
    config_sha256 = _sha256_file(config_path)
    bulk_manifest_sha256 = _sha256_file(bulk_root / "run_manifest.json")
    if run.get("artifact_manifest_sha256") != artifact_manifest_sha256:
        raise RuntimeError("candidate artifact-manifest hash chain is broken")
    if run.get("config_path") != str(config_path) or run.get("config_sha256") != config_sha256:
        raise RuntimeError("candidate was created from a different training config")
    if run.get("bulk_root") != str(bulk_root) or run.get(
        "bulk_manifest_sha256"
    ) != bulk_manifest_sha256:
        raise RuntimeError("candidate was created from a different bulk artifact")
    if run.get("source_tree_sha256") != source_tree_hash():
        raise RuntimeError("candidate source tree differs from the current training code")
    if tuple(run.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("candidate feature schema is not canonical")
    if run.get("label_contract") != LABEL_CONTRACT:
        raise RuntimeError("candidate label contract is not semantic verified-success")
    if not bool(
        run.get("shared_semantic_success_due_to_terminal_fallback_invariance", False)
    ):
        raise RuntimeError("candidate does not enforce shared semantic-success outputs")
    expected_release_digest = _candidate_release_digest(
        artifact_manifest_sha256, config_sha256, bulk_manifest_sha256
    )
    if run.get("release_digest") != expected_release_digest:
        raise RuntimeError("candidate release digest does not match its provenance chain")
    files = artifacts.get("files")
    if not isinstance(files, dict) or int(artifacts.get("file_count", -1)) != len(files):
        raise RuntimeError("candidate artifact file inventory is malformed")
    expected_files: set[str] = set()
    for relative, metadata in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe candidate artifact path: {relative}")
        path = output_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(metadata["size"])
            or _sha256_file(path) != metadata["sha256"]
        ):
            raise RuntimeError(f"sealed candidate artifact changed: {path}")
        expected_files.add(str(relative_path))
    actual_files = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
        and path not in {artifact_manifest, run_manifest}
    }
    if actual_files != expected_files:
        raise RuntimeError(
            "candidate payload inventory changed: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )


def run(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("v4 training config must be a YAML mapping")
    workspace, bulk_root, output_root = _validate_config(config, path)
    if output_root.exists():
        _verify_existing_candidate(
            output_root,
            config_path=path,
            bulk_root=bulk_root,
        )
        print(f"[counterfactual-v4-train] sealed candidate already verified: {output_root}")
        return output_root
    bulk_manifest = _load_bulk_manifest(bulk_root)
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    try:
        torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(config["runtime"]["inter_op_threads"]):
            raise
    if bool(config["runtime"].get("deterministic_algorithms", True)):
        torch.use_deterministic_algorithms(True)

    staging = output_root.with_name(
        f".{output_root.name}.incomplete.{time_ns()}.{os.getpid()}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    try:
        all_roles: dict[str, dict[str, RoleArrays]] = {}
        data_audit: dict[str, Any] = {
            "robots": {},
            "feature_names": list(FEATURE_NAMES),
            "label_contract": dict(LABEL_CONTRACT),
            "shared_semantic_success_due_to_terminal_fallback_invariance": True,
            "test_data_loaded": False,
        }
        for robot in config["robots"]:
            roles = {
                role: _load_role(bulk_root, str(robot), int(config["training_seed"]), role)
                for role in ROLE_ORDER
            }
            all_roles[str(robot)] = roles
            hash_sets = {role: set(values.query_sha256.tolist()) for role, values in roles.items()}
            overlaps = {
                f"{left}__{right}": len(hash_sets[left] & hash_sets[right])
                for index, left in enumerate(ROLE_ORDER)
                for right in ROLE_ORDER[index + 1 :]
            }
            if any(overlaps.values()):
                raise RuntimeError(f"development role overlap for {robot}: {overlaps}")
            data_audit["robots"][str(robot)] = {
                "roles": {
                    role: {
                        "query_count": values.count,
                        "query_sha256_tensor_sha256": _sha256_arrays(values.query_sha256.astype("S64")),
                        "raw_latency_samples_ns_sha256": _sha256_arrays(values.latency_samples_ns),
                        "raw_latency_shape": list(values.latency_samples_ns.shape),
                        "raw_latency_dtype": str(values.latency_samples_ns.dtype),
                        "fail_all_rate": float(np.mean(values.fail_all)),
                        "semantic_verified_success_rates": {
                            action: float(
                                np.mean(values.verified_success[:, action_index])
                            )
                            for action_index, action in enumerate(DECISION_ENTRIES)
                        },
                        "deadline_success_diagnostic_rates": {
                            action: float(
                                np.mean(
                                    values.verified_success_before_deadline[
                                        :, action_index
                                    ]
                                )
                            )
                            for action_index, action in enumerate(DECISION_ENTRIES)
                        },
                        "contract_feasible_semantic_fail_all_count": int(
                            np.sum(
                                (values.fail_all > 0.5)
                                & values.expected_reachable
                                & values.continuity_feasible
                            )
                        ),
                        "source_files": list(values.source_files),
                    }
                    for role, values in roles.items()
                },
                "role_query_overlap_counts": overlaps,
                "roles_disjoint": True,
            }

        robot_metrics = {
            robot: _fit_robot(
                robot=robot,
                roles=all_roles[robot],
                config=config,
                output_dir=staging,
            )
            for robot in config["robots"]
        }
        _write_json(staging / "data_audit.json", data_audit)
        _write_json(staging / "training_metrics.json", {"robots": robot_metrics})
        _write_json(
            staging / "policy_selection.json",
            {
                "robots": {
                    robot: metrics["selected_policy"]
                    for robot, metrics in robot_metrics.items()
                },
                "selection_data_role": "policy_validation_queries",
                "test_data_loaded": False,
            },
        )
        shutil.copyfile(path, staging / "frozen_config.yaml")
        _write_json(staging / "environment.json", environment_payload())
        files = {
            str(item.relative_to(staging)): {
                "sha256": _sha256_file(item),
                "size": item.stat().st_size,
            }
            for item in sorted(staging.rglob("*"))
            if item.is_file()
        }
        artifact_manifest = {
            "protocol": PROTOCOL,
            "release_status": "frozen_validation_candidate",
            "created_utc": _utc(),
            "files": files,
            "file_count": len(files),
            "test_data_loaded": False,
        }
        _write_json(staging / "artifact_manifest.json", artifact_manifest)
        artifact_manifest_sha = _sha256_file(staging / "artifact_manifest.json")
        release_digest = _candidate_release_digest(
            artifact_manifest_sha,
            _sha256_file(path),
            _sha256_file(bulk_root / "run_manifest.json"),
        )
        _write_json(
            staging / "run_manifest.json",
            {
                "protocol": PROTOCOL,
                "status": "frozen_validation_candidate",
                "created_utc": _utc(),
                "release_digest": release_digest,
                "config_path": str(path),
                "config_sha256": _sha256_file(path),
                "source_tree_sha256": source_tree_hash(),
                "bulk_root": str(bulk_root),
                "bulk_manifest_sha256": _sha256_file(bulk_root / "run_manifest.json"),
                "bulk_status": bulk_manifest["status"],
                "artifact_manifest_sha256": artifact_manifest_sha,
                "roles": list(ROLE_ORDER),
                "robots": list(config["robots"]),
                "training_seed": int(config["training_seed"]),
                "feature_names": list(FEATURE_NAMES),
                "label_contract": dict(LABEL_CONTRACT),
                "shared_semantic_success_due_to_terminal_fallback_invariance": True,
                "eligibility_rule": (
                    "semantic verified-success probability threshold AND predicted "
                    "P95 latency <= deadline_ms"
                ),
                "latency_supervision": "raw latency_samples_ns converted once to ms",
                "test_data_loaded": False,
                "formal_test_authorized_or_started": False,
            },
        )
        os.replace(staging, output_root)
        print(f"[counterfactual-v4-train] frozen candidate: {output_root}")
        return output_root
    except BaseException:
        _write_json(
            staging / "failure.json",
            {
                "failed_utc": _utc(),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
                "test_data_loaded": False,
            },
        )
        raise


def time_ns() -> int:
    """Small indirection for deterministic unit tests of staging names."""

    import time

    return time.time_ns()


def main() -> None:
    args = _parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
