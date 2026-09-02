"""Development-only training and paired policy-validation pilot for v5.

The runner has no formal-test data path.  It reuses the frozen selections from
``counterfactual_v4_bulk`` for the three development roles, recomputes only
seed-free geometric features, and benchmarks five methods on the exact same
policy-validation queries.  Every timed command is produced by a numerical
solver and accepted only by the shared deterministic verifier.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..counterfactual_v4.runner import _build_runtimes, _wait_for_quiet_environment
from ..counterfactual_v4.runtime_v4 import wrap_profiled_runtime
from ..data.datasets import QueryDataset
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.benchmark import ProfiledOutcome, query_digest, query_from_dataset
from ..latency_pilot_v3.runner import _solver_components
from ..release_v3_locked.artifacts import load_locked_seed_engine
from ..release_v4_locked.artifacts import (
    FrozenV4Policy,
    TorchScriptV4Inference,
    load_exact_v4_predictor,
    load_policy_config,
)
from ..test_v3_locked.runner import _release_paths
from .features import CHEAP_FEATURE_NAMES, prepare_cheap_features
from .model import (
    FastGatePredictor,
    FastGateTrainingConfig,
    TorchScriptFastGateInference,
    export_exact_torchscript,
    load_exact_torchscript,
    numerical_equivalence,
)
from .policy import (
    FastGatePolicy,
    FastGatePolicyConfig,
    ThresholdSelectionConfig,
    save_policy,
    select_thresholds,
)
from .runtime import (
    AlwaysLocalRuntime,
    HierarchicalOutcome,
    HierarchicalRuntime,
    LocalSolveOutcome,
    wrap_no_easy_profiled_runtime,
)


PROTOCOL = "hierarchical_v5_policy_validation_pilot_v1"
ROLE_ORDER = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)
METHODS = (
    "always_local",
    "fixed_easy_cascade",
    "always_hard",
    "counterfactual_cghik_v4",
    "hierarchical_cghik_v5",
)
ROUTE_STATES = (
    "fast_verified",
    "fast_failed_then_robust",
    "direct_robust",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    key = str(path if relative_to is None else path.relative_to(relative_to))
    return {"path": key, "sha256": _sha256_file(path), "size": path.stat().st_size}


def _verified_release_inputs(
    *,
    workspace: Path,
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
) -> dict[str, Any]:
    """Verify and describe every frozen deployment artifact opened by v5."""

    v3_manifest_path = release_v3_root / "release_manifest.json"
    v3_manifest = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    if (
        v3_manifest.get("release_status") != "sealed"
        or not bool(v3_manifest.get("release_equivalence_all_six_pass", False))
        or bool(v3_manifest.get("test_named_dataset_loaded", True))
    ):
        raise RuntimeError("release_v3_locked is not an eligible sealed release")
    v3_index = {
        (workspace / str(row["path"])).resolve(): row
        for row in v3_manifest.get("artifacts", ())
    }
    v3_paths = _release_paths(release_v3_root, robot, 17)
    v3_used: dict[str, Any] = {}
    for key in (
        "torchscript",
        "normalization",
        "runtime_spec",
        "solver_metadata",
        "seed_bank",
    ):
        path = Path(v3_paths[key]).resolve()
        expected = v3_index.get(path)
        if expected is None:
            raise RuntimeError(f"v3 release manifest omits {path}")
        actual = _artifact(path)
        if actual["sha256"] != expected.get("sha256") or actual["size"] != int(
            expected.get("size", -1)
        ):
            raise RuntimeError(f"sealed v3 artifact changed: {path}")
        v3_used[key] = actual

    v4_manifest_path = release_v4_root / "release_manifest.json"
    v4_artifact_manifest_path = release_v4_root / "artifact_manifest.json"
    v4_manifest = json.loads(v4_manifest_path.read_text(encoding="utf-8"))
    v4_artifact_manifest = json.loads(
        v4_artifact_manifest_path.read_text(encoding="utf-8")
    )
    if (
        v4_manifest.get("release_status") != "sealed"
        or not bool(v4_manifest.get("all_six_validation_runtime_equivalence_pass", False))
        or bool(v4_manifest.get("test_named_dataset_loaded", True))
        or v4_artifact_manifest.get("release_status") != "sealed"
        or bool(v4_artifact_manifest.get("test_data_loaded", True))
    ):
        raise RuntimeError("release_v4_locked is not an eligible sealed release")
    v4_index = dict(v4_artifact_manifest.get("files", {}))
    v4_used: dict[str, Any] = {}
    for key, relative in (
        ("exact_predictor", f"{robot}/exact_v4_predictor.ts"),
        ("policy", f"{robot}/v4_policy.json"),
    ):
        path = (release_v4_root / relative).resolve()
        expected = v4_index.get(relative)
        if expected is None:
            raise RuntimeError(f"v4 artifact manifest omits {relative}")
        actual = _artifact(path)
        if actual["sha256"] != expected.get("sha256") or actual["size"] != int(
            expected.get("size", -1)
        ):
            raise RuntimeError(f"sealed v4 artifact changed: {path}")
        v4_used[key] = actual
    return {
        "release_v3_manifest": _artifact(v3_manifest_path),
        "release_v3_loaded_artifacts": v3_used,
        "release_v4_manifest": _artifact(v4_manifest_path),
        "release_v4_artifact_manifest": _artifact(v4_artifact_manifest_path),
        "release_v4_loaded_artifacts": v4_used,
    }


def _forbid_test_role_or_path(value: str | Path, *, name: str) -> None:
    text = str(value).lower()
    if "test" in text:
        raise ValueError(f"{name} must not contain a test role or path: {value}")


def validate_config(config: Mapping[str, Any], *, workspace: Path | None = None) -> None:
    """Fail closed on method, feature, split, and output-scope drift."""

    if config.get("protocol_version") != "hierarchical_v5_pilot_v1":
        raise ValueError("hierarchical v5 requires protocol_version hierarchical_v5_pilot_v1")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("hierarchical v5 pilot requires Panda and UR5e")
    if int(config.get("training_seed", -1)) != 17:
        raise ValueError("hierarchical v5 pilot requires development seed 17")
    roles = config.get("roles", {})
    actual_roles = (
        str(roles.get("train", "")),
        str(roles.get("calibration", "")),
        str(roles.get("policy_validation", "")),
    )
    if actual_roles != ROLE_ORDER:
        raise ValueError(f"development roles must be exactly {ROLE_ORDER}")
    allowed = tuple(config.get("data_boundary", {}).get("allowed_roles", ()))
    if allowed != ROLE_ORDER or not bool(
        config.get("data_boundary", {}).get("test_data_forbidden", False)
    ):
        raise ValueError("development allowlist or test-data prohibition changed")
    for role in actual_roles:
        _forbid_test_role_or_path(role, name="development role")
    feature_config = config.get("cheap_features", {})
    if int(feature_config.get("dimension", -1)) != len(CHEAP_FEATURE_NAMES):
        raise ValueError("cheap feature dimension changed")
    if tuple(feature_config.get("names", ())) != CHEAP_FEATURE_NAMES:
        raise ValueError("cheap feature schema differs from the implementation")
    if not bool(feature_config.get("learned_seed_features_forbidden", False)) or not bool(
        feature_config.get("learned_seed_ensemble_forbidden", False)
    ):
        raise ValueError("first-level learned-seed prohibition is not explicit")
    if int(config.get("fast_path", {}).get("fast_iterations", -1)) != 1:
        raise ValueError("the pilot fast budget must remain one DLS iteration")
    if tuple(config.get("fast_path", {}).get("slow_path_allowed_entries", ())) != (
        "medium",
        "hard",
    ):
        raise ValueError("v5 slow path must contain only medium and hard entries")
    if tuple(config.get("model", {}).get("hidden_sizes", ())) != (16, 16):
        raise ValueError("the fast gate must remain a 16x16 MLP")
    if tuple(config.get("strategies", ())) != METHODS:
        raise ValueError(f"the pilot strategy registry must be exactly {METHODS}")
    if int(config.get("timing", {}).get("repeats", -1)) != len(METHODS):
        raise ValueError("five methods require exactly five Latin timing repeats")
    if config.get("timing", {}).get("clock") != "perf_counter_ns":
        raise ValueError("pilot timing must use perf_counter_ns")
    if not bool(config.get("data_boundary", {}).get("reject_test_named_paths", False)):
        raise ValueError("test-named paths must be rejected")
    for key in ("bulk_root", "release_v4_root", "output_root"):
        _forbid_test_role_or_path(str(config.get(key, "")), name=key)
    if workspace is not None:
        output = resolve_path(dict(config), str(config["output_root"]))
        expected = (workspace / "outputs" / "hierarchical_v5_pilot").resolve()
        if output != expected:
            raise ValueError(f"output_root must resolve to {expected}")


def local_success_label_from_record(record: Mapping[str, Any]) -> bool:
    """Development audit helper: terminal cascade success is not local success."""

    return bool(
        record.get("entry_action") == "easy"
        and record.get("verified_success", False)
        and tuple(record.get("executed_stages", ())) == ("easy",)
    )


def latin_method_orders(methods: Sequence[str], seed: int) -> tuple[tuple[str, ...], ...]:
    names = tuple(str(item) for item in methods)
    if not names or len(set(names)) != len(names):
        raise ValueError("Latin methods must be non-empty and unique")
    base = list(names)
    np.random.default_rng(seed).shuffle(base)
    return tuple(
        tuple(base[offset:] + base[:offset]) for offset in range(len(base))
    )


@dataclass(frozen=True)
class DevelopmentRole:
    robot: str
    role: str
    dataset: QueryDataset
    source_indices: np.ndarray
    query_sha256: np.ndarray
    category: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    hard_latency_samples_ns: np.ndarray
    hard_verified_success: np.ndarray
    hard_function_evaluations: np.ndarray
    source_manifest: dict[str, Any]

    @property
    def count(self) -> int:
        return int(self.source_indices.size)

    def query(self, index: int, *, dt: float):
        return query_from_dataset(
            self.dataset, int(self.source_indices[index]), dt=dt
        )

    def subset(self, count: int) -> "DevelopmentRole":
        if count <= 0 or count > self.count:
            raise ValueError("invalid development-role subset count")
        selected = slice(0, count)
        return DevelopmentRole(
            robot=self.robot,
            role=self.role,
            dataset=self.dataset,
            source_indices=self.source_indices[selected].copy(),
            query_sha256=self.query_sha256[selected].copy(),
            category=self.category[selected].copy(),
            expected_reachable=self.expected_reachable[selected].copy(),
            continuity_feasible=self.continuity_feasible[selected].copy(),
            hard_latency_samples_ns=self.hard_latency_samples_ns[selected].copy(),
            hard_verified_success=self.hard_verified_success[selected].copy(),
            hard_function_evaluations=self.hard_function_evaluations[selected].copy(),
            source_manifest=dict(self.source_manifest),
        )


@dataclass(frozen=True)
class LocalDevelopmentMeasurements:
    features: np.ndarray
    local_success: np.ndarray
    local_total_samples_ns: np.ndarray
    local_function_evaluations: np.ndarray
    direct_robust_total_samples_ns: np.ndarray
    direct_robust_verified_success: np.ndarray
    direct_robust_function_evaluations: np.ndarray


def _load_development_role(
    *,
    workspace: Path,
    bulk_root: Path,
    robot: str,
    role: str,
    training_seed: int,
    dt: float,
) -> DevelopmentRole:
    if role not in ROLE_ORDER:
        raise ValueError(f"role is outside the v5 development allowlist: {role}")
    _forbid_test_role_or_path(role, name="role")
    role_root = bulk_root / robot / f"seed{training_seed}" / role
    selection_path = role_root / "selection.npz"
    manifest_path = role_root / "selection_manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(role_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_role") != role or bool(
        manifest.get("test_named_dataset_loaded", True)
    ):
        raise RuntimeError("bulk selection violates the development-only contract")
    expected_source = (
        workspace
        / "outputs"
        / f"paper_v2_seed{training_seed}"
        / robot
        / "datasets"
        / f"{role}.npz"
    ).resolve()
    _forbid_test_role_or_path(expected_source, name="development source")
    if Path(str(manifest.get("source_path", ""))).resolve() != expected_source:
        raise RuntimeError("bulk selection points at an unexpected source dataset")
    if _sha256_file(expected_source) != str(manifest.get("source_sha256")):
        raise RuntimeError("development source dataset changed after bulk collection")
    dataset = QueryDataset.load(expected_source)
    with np.load(selection_path, allow_pickle=False) as selection:
        source_indices = np.asarray(selection["source_indices"], dtype=np.int64)
        selection_hashes = selection["query_sha256"].astype(str)
    if len(source_indices) != int(manifest.get("selected_query_count", -1)):
        raise RuntimeError("selection count mismatch")

    fields: dict[str, list[np.ndarray]] = {
        "query_indices": [],
        "source_indices": [],
        "query_sha256": [],
        "category": [],
        "expected_reachable": [],
        "continuity_feasible": [],
        "verified_success": [],
        "latency_samples_ns": [],
        "function_evaluations": [],
    }
    chunks = sorted((role_root / "chunks").glob("chunk_*"))
    if not chunks:
        raise RuntimeError(f"development role has no committed chunks: {role_root}")
    for chunk in chunks:
        chunk_manifest_path = chunk / "chunk_manifest.json"
        labels_path = chunk / "counterfactual_labels.npz"
        chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
        if bool(chunk_manifest.get("environment_contaminated", True)) or bool(
            chunk_manifest.get("test_data_loaded", True)
        ):
            raise RuntimeError(f"unusable development timing chunk: {chunk}")
        artifact = chunk_manifest.get("artifacts", {}).get(labels_path.name, {})
        if _sha256_file(labels_path) != artifact.get("sha256"):
            raise RuntimeError(f"bulk label artifact changed: {labels_path}")
        with np.load(labels_path, allow_pickle=False) as labels:
            actions = tuple(labels["action_names"].astype(str).tolist())
            if actions != ("easy", "medium", "hard", "fixed_robust"):
                raise RuntimeError("bulk action schema changed")
            for name in fields:
                fields[name].append(np.asarray(labels[name]))
    arrays = {name: np.concatenate(values, axis=0) for name, values in fields.items()}
    if not np.array_equal(arrays["query_indices"], np.arange(len(source_indices))):
        raise RuntimeError("bulk chunk query order is not contiguous")
    if not np.array_equal(arrays["source_indices"], source_indices):
        raise RuntimeError("bulk chunks no longer match the frozen selection")
    hashes = arrays["query_sha256"].astype(str)
    if not np.array_equal(hashes, selection_hashes):
        raise RuntimeError("bulk chunk query hashes differ from the selection")
    for local_index, source_index in enumerate(source_indices):
        query = query_from_dataset(dataset, int(source_index), dt=dt)
        if query_digest(query) != hashes[local_index]:
            raise RuntimeError("development source query hash mismatch")
    hard = 2
    source_record = {
        **manifest,
        "selection": _artifact(selection_path),
        "selection_manifest": _artifact(manifest_path),
        "source": _artifact(expected_source),
        "chunk_count": len(chunks),
    }
    return DevelopmentRole(
        robot=robot,
        role=role,
        dataset=dataset,
        source_indices=source_indices,
        query_sha256=hashes,
        category=arrays["category"].astype(str),
        expected_reachable=np.asarray(arrays["expected_reachable"], dtype=bool),
        continuity_feasible=np.asarray(arrays["continuity_feasible"], dtype=bool),
        hard_latency_samples_ns=np.asarray(arrays["latency_samples_ns"][:, hard], dtype=np.int64),
        hard_verified_success=np.asarray(arrays["verified_success"][:, hard], dtype=bool),
        hard_function_evaluations=np.asarray(
            arrays["function_evaluations"][:, hard], dtype=np.int64
        ),
        source_manifest=source_record,
    )


def _collect_local_measurements(
    role: DevelopmentRole,
    *,
    kinematics: object,
    dls: object,
    verifier: object,
    direct_robust_runtime: object | None,
    repeats: int,
    dt: float,
    warmup: int,
    progress_every: int,
) -> LocalDevelopmentMeasurements:
    runtime = AlwaysLocalRuntime(dls, verifier, iterations=1)
    features = np.empty((role.count, len(CHEAP_FEATURE_NAMES)), dtype=np.float32)
    success = np.zeros(role.count, dtype=bool)
    latency = np.zeros((role.count, repeats), dtype=np.int64)
    evaluations = np.zeros(role.count, dtype=np.int64)
    robust_latency = np.zeros(
        (role.count, repeats if direct_robust_runtime is not None else 0),
        dtype=np.int64,
    )
    robust_success = np.zeros(role.count, dtype=bool)
    robust_evaluations = np.zeros(role.count, dtype=np.int64)

    for warmup_index in range(warmup):
        query = role.query(warmup_index % role.count, dt=dt)
        prepare_cheap_features(kinematics, dls, query)
        runtime.solve(query)
        if direct_robust_runtime is not None:
            direct_robust_runtime.solve(query)  # type: ignore[attr-defined]

    for query_index in range(role.count):
        query = role.query(query_index, dt=dt)
        local_signature: tuple[Any, ...] | None = None
        robust_signature: tuple[Any, ...] | None = None
        for repeat in range(repeats):
            if direct_robust_runtime is None:
                order = ("local",)
            elif (query_index + repeat) % 2 == 0:
                order = ("local", "robust")
            else:
                order = ("robust", "local")
            for action in order:
                if action == "robust":
                    started = perf_counter_ns()
                    robust = direct_robust_runtime.solve(query)  # type: ignore[union-attr]
                    robust_latency[query_index, repeat] = perf_counter_ns() - started
                    if any(
                        str(stage).lower().split(":", 1)[0] == "easy"
                        for stage in robust.executed_stages
                    ):
                        raise RuntimeError(
                            "direct robust label path executed forbidden EASY"
                        )
                    signature = _outcome_signature(robust)
                    if robust_signature is None:
                        robust_signature = signature
                        robust_success[query_index] = bool(robust.accepted)
                        robust_evaluations[query_index] = int(
                            robust.function_evaluations
                        )
                    elif signature != robust_signature:
                        raise RuntimeError(
                            "direct robust semantics changed across timing repeats"
                        )
                    continue

                started = perf_counter_ns()
                prepared = prepare_cheap_features(kinematics, dls, query)
                outcome = runtime.solve(query)
                latency[query_index, repeat] = perf_counter_ns() - started
                if repeat == 0:
                    features[query_index] = prepared.features
                elif not np.array_equal(features[query_index], prepared.features):
                    raise RuntimeError("cheap feature values changed across repeats")
                signature = (
                    bool(outcome.accepted),
                    int(outcome.function_evaluations),
                    int(outcome.iterations),
                    tuple(outcome.verification_reasons),
                    None
                    if outcome.q is None
                    else np.asarray(outcome.q, dtype=np.float64).tobytes(),
                )
                if local_signature is None:
                    local_signature = signature
                    success[query_index] = outcome.accepted
                    evaluations[query_index] = outcome.function_evaluations
                elif signature != local_signature:
                    raise RuntimeError(
                        "local solver semantics changed across timing repeats"
                    )
        if progress_every > 0 and (query_index + 1) % progress_every == 0:
            print(
                f"[hierarchical-v5] {role.robot}/{role.role} local labels "
                f"{query_index + 1}/{role.count}",
                flush=True,
            )
    return LocalDevelopmentMeasurements(
        features=np.ascontiguousarray(features),
        local_success=success,
        local_total_samples_ns=latency,
        local_function_evaluations=evaluations,
        direct_robust_total_samples_ns=robust_latency,
        direct_robust_verified_success=robust_success,
        direct_robust_function_evaluations=robust_evaluations,
    )


def _benefit_labels(
    measurements: LocalDevelopmentMeasurements,
    *,
    gate_overhead_ns: float,
) -> np.ndarray:
    local_p95 = np.percentile(measurements.local_total_samples_ns, 95, axis=1)
    if measurements.direct_robust_total_samples_ns.shape[1] == 0:
        raise ValueError("latency-benefit labels require paired direct-robust timings")
    robust_p95 = np.percentile(
        measurements.direct_robust_total_samples_ns, 95, axis=1
    )
    return measurements.local_success & (
        (local_p95 + gate_overhead_ns) < robust_p95
    )


def _model_config(config: Mapping[str, Any], *, smoke: bool) -> FastGateTrainingConfig:
    values = config["model"]
    return FastGateTrainingConfig(
        epochs=3 if smoke else int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        gradient_clip_norm=float(values["gradient_clip_norm"]),
        seed=int(values["seed"]),
    )


def _gate_latency_microbenchmark(
    policy: FastGatePolicy,
    features: np.ndarray,
    *,
    warmup: int,
    samples: int,
) -> dict[str, float]:
    if len(features) == 0:
        raise ValueError("gate microbenchmark requires features")
    for index in range(warmup):
        policy.decide(features[index % len(features)])
    timings = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        started = perf_counter_ns()
        policy.decide(features[index % len(features)])
        timings[index] = perf_counter_ns() - started
    return {
        "samples": float(samples),
        "p50_ns": float(np.percentile(timings, 50)),
        "p95_ns": float(np.percentile(timings, 95)),
        "p99_ns": float(np.percentile(timings, 99)),
        "mean_ns": float(np.mean(timings)),
    }


def _train_robot_gate(
    *,
    robot: str,
    config: Mapping[str, Any],
    train: LocalDevelopmentMeasurements,
    calibration: LocalDevelopmentMeasurements,
    output_dir: Path,
    smoke: bool,
) -> tuple[FastGatePolicy, dict[str, Any]]:
    training_config = _model_config(config, smoke=smoke)
    preliminary_train_benefit = _benefit_labels(train, gate_overhead_ns=0.0)
    preliminary_cal_benefit = _benefit_labels(calibration, gate_overhead_ns=0.0)
    preliminary = FastGatePredictor(training_config, device="cpu")
    preliminary.fit(
        train.features,
        train.local_success,
        preliminary_train_benefit,
        role="risk_train_queries",
        provenance={"robot": robot, "phase": "gate-overhead-estimation"},
    ).calibrate(
        calibration.features,
        calibration.local_success,
        preliminary_cal_benefit,
        role="calibration_queries",
        provenance={"robot": robot, "phase": "gate-overhead-estimation"},
    )
    preliminary_path = output_dir / f"{robot}_preliminary_fast_gate.ts"
    export_exact_torchscript(preliminary, preliminary_path)
    preliminary_backend = TorchScriptFastGateInference(
        load_exact_torchscript(preliminary_path, device="cpu")
    )
    preliminary_policy = FastGatePolicy(
        preliminary_backend,
        FastGatePolicyConfig(
            local_success_threshold=0.5,
            latency_benefit_threshold=0.5,
            minimum_fast_precision=0.0,
            minimum_positive_benefit_rate=0.0,
            calibration_count=len(calibration.features),
        ),
    )
    overhead = _gate_latency_microbenchmark(
        preliminary_policy,
        calibration.features,
        warmup=20 if smoke else 200,
        samples=min(250 if smoke else 2500, max(len(calibration.features), 1)),
    )
    gate_overhead_ns = float(overhead["p95_ns"])
    train_benefit = _benefit_labels(train, gate_overhead_ns=gate_overhead_ns)
    calibration_benefit = _benefit_labels(
        calibration, gate_overhead_ns=gate_overhead_ns
    )

    predictor = FastGatePredictor(training_config, device="cpu")
    predictor.fit(
        train.features,
        train.local_success,
        train_benefit,
        role="risk_train_queries",
        provenance={
            "robot": robot,
            "gate_overhead_ns_for_latency_benefit": gate_overhead_ns,
            "test_data_loaded": False,
        },
    ).calibrate(
        calibration.features,
        calibration.local_success,
        calibration_benefit,
        role="calibration_queries",
        provenance={"robot": robot, "test_data_loaded": False},
    )
    prediction = predictor.predict(calibration.features)
    calibration_config = config["calibration"]
    precision = float(calibration_config["minimum_fast_path_precision"])
    threshold_config = ThresholdSelectionConfig(
        local_success_grid=tuple(
            float(item)
            for item in calibration_config["local_success_probability_grid"]
        ),
        latency_benefit_grid=tuple(
            float(item)
            for item in calibration_config["latency_benefit_probability_grid"]
        ),
        minimum_fast_precision=0.50 if smoke else precision,
        minimum_positive_benefit_rate=0.50 if smoke else precision,
        minimum_fast_count=1 if smoke else 25,
    )
    policy_config, selection = select_thresholds(
        prediction.local_success_probability,
        prediction.latency_benefit_probability,
        calibration.local_success,
        calibration_benefit,
        role="calibration_queries",
        config=threshold_config,
    )

    eager_path = output_dir / f"{robot}_fast_gate_predictor.pt"
    exact_path = output_dir / f"{robot}_exact_fast_gate.ts"
    policy_path = output_dir / f"{robot}_fast_gate_policy.json"
    predictor.save(eager_path, provenance={"selection_role": "calibration_queries"})
    export_meta = export_exact_torchscript(predictor, exact_path)
    exact_module = load_exact_torchscript(exact_path, device="cpu")
    equivalence = numerical_equivalence(
        predictor,
        exact_module,
        calibration.features[: min(1000, len(calibration.features))],
        success_threshold=policy_config.local_success_threshold,
        benefit_threshold=policy_config.latency_benefit_threshold,
        atol=1.0e-9,
    )
    if not equivalence["passed"]:
        raise RuntimeError(f"{robot} fast-gate TorchScript equivalence failed")
    save_policy(
        policy_path,
        policy_config,
        selection,
        provenance={"robot": robot, "test_data_loaded": False},
    )
    backend = TorchScriptFastGateInference(exact_module)
    policy = FastGatePolicy(backend, policy_config)
    final_overhead = _gate_latency_microbenchmark(
        policy,
        calibration.features,
        warmup=20 if smoke else 200,
        samples=min(250 if smoke else 2500, max(len(calibration.features), 1)),
    )
    report = {
        "robot": robot,
        "architecture": [7, 16, 16, 2],
        "training_query_count": len(train.features),
        "calibration_query_count": len(calibration.features),
        "training_local_success_rate": float(np.mean(train.local_success)),
        "calibration_local_success_rate": float(np.mean(calibration.local_success)),
        "training_benefit_rate": float(np.mean(train_benefit)),
        "calibration_benefit_rate": float(np.mean(calibration_benefit)),
        "preliminary_gate_overhead": overhead,
        "final_gate_overhead": final_overhead,
        "threshold_selection": selection,
        "calibration_metrics": predictor.calibration_metrics(
            calibration.features,
            calibration.local_success,
            calibration_benefit,
        ),
        "numerical_equivalence": equivalence,
        "artifacts": {
            "eager": _artifact(eager_path, relative_to=output_dir),
            "torchscript": _artifact(exact_path, relative_to=output_dir),
            "policy": _artifact(policy_path, relative_to=output_dir),
            "preliminary_torchscript": _artifact(
                preliminary_path, relative_to=output_dir
            ),
        },
        "export": export_meta,
        "test_data_loaded": False,
    }
    return policy, report


def _build_no_easy_slow_runtime(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
    kinematics: object,
    device: str,
) -> tuple[object, object, object]:
    release_paths = _release_paths(release_v3_root, robot, 17)
    dls, verifier, fallback, bank, cascade = _solver_components(
        source_config,
        {
            "solver_metadata": release_paths["solver_metadata"],
            "seed_bank": release_paths["seed_bank"],
        },
        kinematics,
    )
    v4_model_path = release_v4_root / robot / "exact_v4_predictor.ts"
    v4_policy_path = release_v4_root / robot / "v4_policy.json"
    v4_config, _ = load_policy_config(v4_policy_path)
    policy = FrozenV4Policy(
        TorchScriptV4Inference(load_exact_v4_predictor(v4_model_path, device="cpu")),
        v4_config,
    )
    runtime = wrap_no_easy_profiled_runtime(
        name="hierarchical_v5_medium_hard_slow",
        policy=policy,
        kinematics=kinematics,
        seed_engine=load_locked_seed_engine(
            kinematics=kinematics,
            torchscript_path=release_paths["torchscript"],
            normalization_path=release_paths["normalization"],
            runtime_spec_path=release_paths["runtime_spec"],
            device=device,
        ),
        dls=dls,
        verifier=verifier,
        seed_bank=bank,
        fallback=fallback,
        cascade_config=cascade,
    )
    return runtime, dls, verifier


def _build_policy_validation_methods(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
    kinematics: object,
    device: str,
    fast_gate: FastGatePolicy,
) -> dict[str, object]:
    _, fixed, release_paths = _build_runtimes(
        source_config=source_config,
        release_root=release_v3_root,
        robot=robot,
        training_seed=17,
        kinematics=kinematics,
        device=device,
    )

    def fresh_components():
        return _solver_components(
            source_config,
            {
                "solver_metadata": release_paths["solver_metadata"],
                "seed_bank": release_paths["seed_bank"],
            },
            kinematics,
        )

    def fresh_seed_engine():
        return load_locked_seed_engine(
            kinematics=kinematics,
            torchscript_path=release_paths["torchscript"],
            normalization_path=release_paths["normalization"],
            runtime_spec_path=release_paths["runtime_spec"],
            device=device,
        )

    v4_model_path = release_v4_root / robot / "exact_v4_predictor.ts"
    v4_policy_path = release_v4_root / robot / "v4_policy.json"
    v4_config, _ = load_policy_config(v4_policy_path)

    current_dls, current_verifier, current_fallback, current_bank, current_cascade = (
        fresh_components()
    )
    current_policy = FrozenV4Policy(
        TorchScriptV4Inference(load_exact_v4_predictor(v4_model_path, device="cpu")),
        v4_config,
    )
    current = wrap_profiled_runtime(
        name="counterfactual_cghik_v4",
        policy=current_policy,  # type: ignore[arg-type]
        kinematics=kinematics,
        seed_engine=fresh_seed_engine(),
        dls=current_dls,
        verifier=current_verifier,
        seed_bank=current_bank,
        fallback=current_fallback,
        cascade_config=current_cascade,
    )

    slow, slow_dls, slow_verifier = _build_no_easy_slow_runtime(
        source_config=source_config,
        release_v3_root=release_v3_root,
        release_v4_root=release_v4_root,
        robot=robot,
        kinematics=kinematics,
        device=device,
    )
    hierarchical = HierarchicalRuntime(
        kinematics=kinematics,
        dls=slow_dls,
        verifier=slow_verifier,
        fast_gate=fast_gate,
        slow_runtime=slow,
        fast_iterations=1,
    )
    local_dls, local_verifier, _, _, _ = fresh_components()
    return {
        "always_local": AlwaysLocalRuntime(local_dls, local_verifier, iterations=1),
        "fixed_easy_cascade": fixed["easy"],
        "always_hard": fixed["hard"],
        "counterfactual_cghik_v4": current,
        "hierarchical_cghik_v5": hierarchical,
    }


def _outcome_signature(outcome: object) -> tuple[Any, ...]:
    q = getattr(outcome, "q", None)
    return (
        bool(getattr(outcome, "accepted")),
        int(getattr(outcome, "function_evaluations")),
        int(getattr(outcome, "iterations")),
        bool(getattr(outcome, "fallback_used", False)),
        tuple(getattr(outcome, "verification_reasons", ())),
        str(getattr(outcome, "route", getattr(outcome, "entry_action", ""))),
        bool(getattr(outcome, "local_attempted", False)),
        bool(getattr(outcome, "local_accepted", False)),
        bool(getattr(outcome, "learned_seed_ensemble_invoked", False)),
        tuple(str(stage) for stage in getattr(outcome, "executed_stages", ())),
        None if q is None else np.asarray(q, dtype=np.float64).tobytes(),
    )


def _instrumentation(outcome: object, method: str) -> dict[str, Any]:
    if isinstance(outcome, HierarchicalOutcome):
        return {
            "route": outcome.route,
            "local_attempted": outcome.local_attempted,
            "local_accepted": outcome.local_accepted,
            "seed_invoked": outcome.learned_seed_ensemble_invoked,
            "gate_local_success_probability": outcome.gate_local_success_probability,
            "gate_latency_benefit_probability": outcome.gate_latency_benefit_probability,
            "executed_stages": list(outcome.executed_stages),
        }
    if isinstance(outcome, LocalSolveOutcome):
        return {
            "route": "always_local_accept" if outcome.accepted else "always_local_failure",
            "local_attempted": True,
            "local_accepted": outcome.accepted,
            "seed_invoked": False,
            "gate_local_success_probability": None,
            "gate_latency_benefit_probability": None,
            "executed_stages": ["local_fast"],
        }
    return {
        "route": str(getattr(outcome, "entry_action", method)),
        "local_attempted": False,
        "local_accepted": False,
        "seed_invoked": True,
        "gate_local_success_probability": None,
        "gate_latency_benefit_probability": None,
        "executed_stages": list(getattr(outcome, "executed_stages", ())),
    }


@dataclass(frozen=True)
class BenchmarkData:
    robot: str
    methods: tuple[str, ...]
    query_sha256: np.ndarray
    category: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    latency_samples_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    route: np.ndarray
    gate_local_success_probability: np.ndarray
    gate_latency_benefit_probability: np.ndarray
    executed_stages: np.ndarray | None = None


def _warmup_methods(
    methods: Mapping[str, object],
    role: DevelopmentRole,
    *,
    iterations: int,
    dt: float,
) -> None:
    for index in range(iterations):
        query = role.query(index % role.count, dt=dt)
        order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
        for method in order:
            methods[method].solve(query)  # type: ignore[attr-defined]


def _benchmark_policy_validation(
    role: DevelopmentRole,
    methods: Mapping[str, object],
    *,
    config: Mapping[str, Any],
    dt: float,
    smoke: bool,
) -> tuple[BenchmarkData, list[dict[str, Any]]]:
    if tuple(methods) != METHODS:
        raise ValueError("runtime registry differs from the five-method contract")
    repeats = int(config["timing"]["repeats"])
    _warmup_methods(
        methods,
        role,
        iterations=5 if smoke else int(config["timing"]["warmup_iterations"]),
        dt=dt,
    )
    count = role.count
    method_count = len(METHODS)
    latency = np.zeros((count, method_count, repeats), dtype=np.int64)
    accepted = np.zeros((count, method_count), dtype=bool)
    evaluations = np.zeros((count, method_count), dtype=np.int64)
    seed_invoked = np.zeros((count, method_count), dtype=bool)
    local_attempted = np.zeros((count, method_count), dtype=bool)
    local_accepted = np.zeros((count, method_count), dtype=bool)
    route = np.full((count, method_count), "", dtype="U64")
    executed_stages = np.full((count, method_count), "", dtype="U256")
    gate_success = np.full((count, method_count), np.nan, dtype=np.float64)
    gate_benefit = np.full((count, method_count), np.nan, dtype=np.float64)
    quiet_events: list[dict[str, Any]] = []
    index_of = {name: index for index, name in enumerate(METHODS)}
    for query_index in range(count):
        if query_index % max(
            int(config["runtime"]["environment_check_every_queries"]), 1
        ) == 0:
            event = _wait_for_quiet_environment(
                dict(config), context=f"hierarchical-v5/{role.robot}/query{query_index}"
            )
            if event["had_busy_process"]:
                quiet_events.append(event)
        query = role.query(query_index, dt=dt)
        signatures: dict[str, tuple[Any, ...]] = {}
        orders = latin_method_orders(METHODS, int(config["model"]["seed"]) + query_index)
        for repeat, order in enumerate(orders):
            for method in order:
                started = perf_counter_ns()
                outcome = methods[method].solve(query)  # type: ignore[attr-defined]
                elapsed = perf_counter_ns() - started
                method_index = index_of[method]
                latency[query_index, method_index, repeat] = elapsed
                signature = _outcome_signature(outcome)
                if method not in signatures:
                    signatures[method] = signature
                    info = _instrumentation(outcome, method)
                    accepted[query_index, method_index] = bool(
                        getattr(outcome, "accepted")
                    )
                    evaluations[query_index, method_index] = int(
                        getattr(outcome, "function_evaluations")
                    )
                    seed_invoked[query_index, method_index] = bool(
                        info["seed_invoked"]
                    )
                    local_attempted[query_index, method_index] = bool(
                        info["local_attempted"]
                    )
                    local_accepted[query_index, method_index] = bool(
                        info["local_accepted"]
                    )
                    route[query_index, method_index] = str(info["route"])
                    executed_stages[query_index, method_index] = "|".join(
                        str(stage) for stage in info["executed_stages"]
                    )
                    if info["gate_local_success_probability"] is not None:
                        gate_success[query_index, method_index] = float(
                            info["gate_local_success_probability"]
                        )
                    if info["gate_latency_benefit_probability"] is not None:
                        gate_benefit[query_index, method_index] = float(
                            info["gate_latency_benefit_probability"]
                        )
                elif signatures[method] != signature:
                    raise RuntimeError(
                        f"{role.robot} {method} changed semantics across timing repeats"
                    )
        if (query_index + 1) % (10 if smoke else 100) == 0:
            print(
                f"[hierarchical-v5] {role.robot} policy pilot "
                f"{query_index + 1}/{count}",
                flush=True,
            )
    return (
        BenchmarkData(
            robot=role.robot,
            methods=METHODS,
            query_sha256=role.query_sha256.copy(),
            category=role.category.copy(),
            expected_reachable=role.expected_reachable.copy(),
            continuity_feasible=role.continuity_feasible.copy(),
            latency_samples_ns=latency,
            accepted=accepted,
            function_evaluations=evaluations,
            seed_invoked=seed_invoked,
            local_attempted=local_attempted,
            local_accepted=local_accepted,
            route=route,
            gate_local_success_probability=gate_success,
            gate_latency_benefit_probability=gate_benefit,
            executed_stages=executed_stages,
        ),
        quiet_events,
    )


def _q(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_benchmark(data: BenchmarkData) -> list[dict[str, Any]]:
    """Compute query-level operational metrics from five raw repeats."""

    count, method_count, repeats = data.latency_samples_ns.shape
    if data.methods != METHODS or method_count != len(METHODS) or repeats != 5:
        raise ValueError("benchmark tensor differs from the five-by-five contract")
    if len(set(data.query_sha256.astype(str))) != count:
        raise ValueError("policy-validation query hashes are not unique")
    feasible = data.expected_reachable & data.continuity_feasible
    if not np.any(feasible):
        raise ValueError("benchmark contains no operational-feasible queries")
    per_query_ms = np.median(data.latency_samples_ns, axis=2) / 1e6
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(data.methods):
        selected_latency = per_query_ms[feasible, method_index]
        attempts = data.local_attempted[feasible, method_index]
        hits = data.local_accepted[feasible, method_index]
        failed_attempts = attempts & ~hits
        recovered = failed_attempts & data.accepted[feasible, method_index]
        row = {
            "robot": data.robot,
            "method": method,
            "feasible_queries": int(np.sum(feasible)),
            "verified_success": float(np.mean(data.accepted[feasible, method_index])),
            "p50_ms": _q(selected_latency, 50),
            "p95_ms": _q(selected_latency, 95),
            "p99_ms": _q(selected_latency, 99),
            "mean_fev": float(
                np.mean(data.function_evaluations[feasible, method_index])
            ),
            "learned_seed_ensemble_invocation_rate": float(
                np.mean(data.seed_invoked[feasible, method_index])
            ),
            "fast_path_attempt_rate": float(np.mean(attempts)),
            "fast_path_hit_rate": float(np.mean(hits)),
            "fast_path_precision": (
                float(np.sum(hits) / np.sum(attempts)) if np.any(attempts) else None
            ),
            "fast_path_failure_recovery_rate": (
                float(np.sum(recovered) / np.sum(failed_attempts))
                if np.any(failed_attempts)
                else None
            ),
            "paired_v5_minus_always_hard_median_ms": None,
            "paired_v5_minus_current_cghik_median_ms": None,
        }
        rows.append(row)
    return rows


def _paired_summary(data: BenchmarkData, comparator: str) -> dict[str, Any]:
    if comparator not in data.methods:
        raise ValueError(f"unknown comparator: {comparator}")
    feasible = data.expected_reachable & data.continuity_feasible
    median_ms = np.median(data.latency_samples_ns, axis=2) / 1e6
    v5 = data.methods.index("hierarchical_cghik_v5")
    other = data.methods.index(comparator)
    delta = median_ms[feasible, v5] - median_ms[feasible, other]
    return {
        "robot": data.robot,
        "comparator": comparator,
        "paired_query_count": int(delta.size),
        "query_hash_digest": sha256(
            np.ascontiguousarray(data.query_sha256[feasible].astype("S64")).tobytes()
        ).hexdigest(),
        "v5_minus_comparator_ms": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "p05": _q(delta, 5),
            "p95": _q(delta, 95),
            "fraction_v5_faster": float(np.mean(delta < 0.0)),
        },
    }


def _family_routes(data: BenchmarkData) -> list[dict[str, Any]]:
    v5 = data.methods.index("hierarchical_cghik_v5")
    result: list[dict[str, Any]] = []
    for family in sorted(set(data.category.astype(str))):
        selected = data.category == family
        attempts = data.local_attempted[selected, v5]
        hits = data.local_accepted[selected, v5]
        states = np.where(
            hits,
            "fast_verified",
            np.where(attempts, "fast_failed_then_robust", "direct_robust"),
        )
        total = int(np.sum(selected))
        for state in ROUTE_STATES:
            count = int(np.sum(states == state))
            result.append(
                {
                    "robot": data.robot,
                    "query_family": family,
                    "route_state": state,
                    "count": count,
                    "rate": count / total,
                    "family_queries": total,
                }
            )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_main_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "robot",
        "method",
        "verified_success",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_fev",
        "learned_seed_ensemble_invocation_rate",
        "fast_path_hit_rate",
        "fast_path_precision",
    )
    lines = [
        "# Hierarchical v5 policy-validation pilot",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append("--" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_fast_usage(rows: Sequence[Mapping[str, Any]], output: Path, dpi: int) -> None:
    v5 = [row for row in rows if row["method"] == "hierarchical_cghik_v5"]
    robots = [str(row["robot"]) for row in v5]
    labels = ("Attempt", "Hit", "Precision")
    values = np.asarray(
        [
            [
                float(row["fast_path_attempt_rate"]),
                float(row["fast_path_hit_rate"]),
                float(row["fast_path_precision"] or 0.0),
            ]
            for row in v5
        ]
    )
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(labels))
    width = 0.34
    for index, robot in enumerate(robots):
        axis.bar(x + (index - 0.5) * width, values[index], width, label=robot)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("Fast-path use and verifier-confirmed success")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_latency(rows: Sequence[Mapping[str, Any]], output: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=False)
    labels = [
        "Local",
        "Fixed easy",
        "Always hard",
        "CG-HIK",
        "Hierarchical",
    ]
    for axis, robot in zip(axes, ("panda", "ur5e"), strict=True):
        robot_rows = [row for row in rows if row["robot"] == robot]
        x = np.arange(len(robot_rows))
        width = 0.24
        for offset, metric, label in (
            (-1, "p50_ms", "P50"),
            (0, "p95_ms", "P95"),
            (1, "p99_ms", "P99"),
        ):
            axis.bar(
                x + offset * width,
                [float(row[metric]) for row in robot_rows],
                width,
                label=label,
            )
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_ylabel("Query-median latency (ms)")
        axis.set_title(robot.upper())
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Five-strategy latency quantiles on policy validation")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_family_routes(
    rows: Sequence[Mapping[str, Any]], output: Path, dpi: int
) -> None:
    colors = {
        "fast_verified": "#2a9d8f",
        "fast_failed_then_robust": "#e9c46a",
        "direct_robust": "#264653",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True)
    for axis, robot in zip(axes, ("panda", "ur5e"), strict=True):
        robot_rows = [row for row in rows if row["robot"] == robot]
        families = sorted(set(str(row["query_family"]) for row in robot_rows))
        y = np.arange(len(families))
        left = np.zeros(len(families), dtype=np.float64)
        for state in ROUTE_STATES:
            values = np.asarray(
                [
                    next(
                        float(row["rate"])
                        for row in robot_rows
                        if row["query_family"] == family
                        and row["route_state"] == state
                    )
                    for family in families
                ]
            )
            axis.barh(y, values, left=left, color=colors[state], label=state)
            left += values
        axis.set_yticks(y, families)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Route fraction")
        axis.set_title(robot.upper())
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Fast/robust routing by development query family")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_benchmark(path: Path, data: BenchmarkData) -> None:
    np.savez_compressed(
        path,
        method_names=np.asarray(data.methods, dtype=np.str_),
        query_sha256=data.query_sha256,
        category=data.category,
        expected_reachable=data.expected_reachable,
        continuity_feasible=data.continuity_feasible,
        latency_samples_ns=data.latency_samples_ns,
        accepted=data.accepted,
        function_evaluations=data.function_evaluations,
        learned_seed_ensemble_invoked=data.seed_invoked,
        local_attempted=data.local_attempted,
        local_accepted=data.local_accepted,
        route=data.route,
        gate_local_success_probability=data.gate_local_success_probability,
        gate_latency_benefit_probability=data.gate_latency_benefit_probability,
        executed_stages=(
            np.full(data.accepted.shape, "", dtype="U1")
            if data.executed_stages is None
            else data.executed_stages
        ),
    )


def _forbidden_easy_stage_count(data: BenchmarkData) -> int:
    if data.executed_stages is None:
        raise ValueError("executed stages are required for the no-EASY audit")
    method_index = data.methods.index("hierarchical_cghik_v5")
    count = 0
    for encoded in data.executed_stages[:, method_index].astype(str):
        for stage in encoded.split("|"):
            if stage and stage.lower().split(":", 1)[0] == "easy":
                count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small development-only plumbing check in a separate output root.",
    )
    return parser


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    bulk_root = resolve_path(config, str(config["bulk_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    release_v3_root = (workspace / "outputs" / "release_v3_locked").resolve()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status_at_start = subprocess.run(
        ["git", "status", "--short"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not smoke and git_status_at_start:
        raise RuntimeError(
            "full hierarchical v5 pilot requires a clean committed worktree: "
            + "; ".join(git_status_at_start)
        )
    output_root = resolve_path(config, str(config["output_root"]))
    if smoke:
        output_root = (workspace / "outputs" / "hierarchical_v5_smoke").resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"v5 output already exists; overwrite is forbidden: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    torch.use_deterministic_algorithms(
        bool(config["runtime"].get("deterministic_algorithms", True))
    )
    dt = float(source_config["data"]["dt"])
    repeats = int(config["timing"]["repeats"])
    all_main_rows: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    training_reports: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    quiet_events: dict[str, Any] = {}
    started_utc = _utc()
    for robot in config["robots"]:
        robot = str(robot)
        kinematics = load_robot(source_config, robot)
        source_records[f"{robot}/sealed_release_inputs"] = _verified_release_inputs(
            workspace=workspace,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
        )
        label_slow, local_dls, local_verifier = _build_no_easy_slow_runtime(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
            kinematics=kinematics,
            device=str(config["runtime"]["device"]),
        )
        roles: dict[str, DevelopmentRole] = {}
        measurements: dict[str, LocalDevelopmentMeasurements] = {}
        for role in ROLE_ORDER:
            loaded = _load_development_role(
                workspace=workspace,
                bulk_root=bulk_root,
                robot=robot,
                role=role,
                training_seed=17,
                dt=dt,
            )
            if smoke:
                smoke_counts = {
                    "risk_train_queries": 128,
                    "calibration_queries": 64,
                    "policy_validation_queries": 32,
                }
                loaded = loaded.subset(min(smoke_counts[role], loaded.count))
            roles[role] = loaded
            source_records[f"{robot}/{role}"] = loaded.source_manifest

        role_sets = {
            role: set(values.query_sha256.astype(str))
            for role, values in roles.items()
        }
        overlaps = {
            f"{left}__{right}": len(role_sets[left] & role_sets[right])
            for left_index, left in enumerate(ROLE_ORDER)
            for right in ROLE_ORDER[left_index + 1 :]
        }
        if any(overlaps.values()):
            raise RuntimeError(
                f"{robot} development roles overlap by query hash: {overlaps}"
            )
        source_records[f"{robot}/role_isolation"] = {
            "query_counts": {role: len(role_sets[role]) for role in ROLE_ORDER},
            "pairwise_overlap_counts": overlaps,
            "all_pairwise_disjoint": True,
        }
        quiet_events[robot] = {"development_measurements": {}}
        for role in ROLE_ORDER:
            loaded = roles[role]
            quiet_event = _wait_for_quiet_environment(
                dict(config), context=f"hierarchical-v5/{robot}/{role}/labels"
            )
            quiet_events[robot]["development_measurements"][role] = quiet_event
            measurements[role] = _collect_local_measurements(
                loaded,
                kinematics=kinematics,
                dls=local_dls,
                verifier=local_verifier,
                direct_robust_runtime=(
                    label_slow
                    if role in {"risk_train_queries", "calibration_queries"}
                    else None
                ),
                repeats=repeats,
                dt=dt,
                warmup=5 if smoke else int(config["timing"]["warmup_iterations"]),
                progress_every=0 if smoke else 1000,
            )
            np.savez_compressed(
                staging / f"{robot}_{role}_seed_free_local_labels.npz",
                feature_names=np.asarray(CHEAP_FEATURE_NAMES, dtype=np.str_),
                query_sha256=loaded.query_sha256,
                category=loaded.category,
                features=measurements[role].features,
                local_verified_success=measurements[role].local_success,
                local_total_samples_ns=measurements[role].local_total_samples_ns,
                local_function_evaluations=measurements[role].local_function_evaluations,
                direct_robust_total_samples_ns=measurements[
                    role
                ].direct_robust_total_samples_ns,
                direct_robust_verified_success=measurements[
                    role
                ].direct_robust_verified_success,
                direct_robust_function_evaluations=measurements[
                    role
                ].direct_robust_function_evaluations,
                bulk_fixed_hard_latency_samples_ns=loaded.hard_latency_samples_ns,
            )

        gate, training_report = _train_robot_gate(
            robot=robot,
            config=config,
            train=measurements["risk_train_queries"],
            calibration=measurements["calibration_queries"],
            output_dir=staging,
            smoke=smoke,
        )
        training_reports[robot] = training_report
        methods = _build_policy_validation_methods(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
            kinematics=kinematics,
            device=str(config["runtime"]["device"]),
            fast_gate=gate,
        )
        benchmark, robot_quiet = _benchmark_policy_validation(
            roles["policy_validation_queries"],
            methods,
            config=config,
            dt=dt,
            smoke=smoke,
        )
        quiet_events[robot]["policy_benchmark"] = robot_quiet
        _save_benchmark(staging / f"{robot}_policy_validation_records.npz", benchmark)
        rows = summarize_benchmark(benchmark)
        all_main_rows.extend(rows)
        family = _family_routes(benchmark)
        all_family_rows.extend(family)
        robot_paired = [
            _paired_summary(benchmark, comparator)
            for comparator in config["reporting"]["paired_comparators"]
        ]
        paired.extend(robot_paired)
        by_method = {row["method"]: row for row in rows}
        v5 = by_method["hierarchical_cghik_v5"]
        v5["paired_v5_minus_always_hard_median_ms"] = next(
            item["v5_minus_comparator_ms"]["median"]
            for item in robot_paired
            if item["comparator"] == "always_hard"
        )
        v5["paired_v5_minus_current_cghik_median_ms"] = next(
            item["v5_minus_comparator_ms"]["median"]
            for item in robot_paired
            if item["comparator"] == "counterfactual_cghik_v4"
        )
        current = by_method["counterfactual_cghik_v4"]
        hard = by_method["always_hard"]
        checks = {
            "verified_success_vs_current": v5["verified_success"]
            >= current["verified_success"],
            "verified_success_vs_always_hard": v5["verified_success"]
            >= hard["verified_success"],
            "p50_improves_current": v5["p50_ms"] < current["p50_ms"],
            "p95_not_worse_current": v5["p95_ms"] <= current["p95_ms"],
            "slow_path_easy_stage_count_zero": _forbidden_easy_stage_count(
                benchmark
            )
            == 0,
        }
        gates[robot] = {
            "checks": checks,
            "all_pass": all(bool(value) for value in checks.values()),
            "selection_frozen_before_policy_validation": True,
            "policy_validation_used_for_retuning": False,
        }

    _write_csv(staging / "main_table.csv", all_main_rows)
    _write_json(staging / "main_table.json", all_main_rows)
    _write_main_markdown(staging / "main_table.md", all_main_rows)
    _write_csv(staging / "query_family_routes.csv", all_family_rows)
    _write_json(staging / "query_family_routes.json", all_family_rows)
    _write_json(staging / "paired_latency_summary.json", paired)
    _write_json(staging / "training_report.json", training_reports)
    pilot_gate = {
        "robots": gates,
        "all_robots_pass": all(bool(row["all_pass"]) for row in gates.values()),
        "development_only": True,
        "fresh_evaluation_started": False,
        "test_data_loaded": False,
    }
    _write_json(staging / "pilot_gate.json", pilot_gate)
    dpi = int(config["reporting"]["png_dpi"])
    _plot_fast_usage(all_main_rows, staging / "fig1_fast_path_use_success", dpi)
    _plot_latency(all_main_rows, staging / "fig2_five_strategy_latency_quantiles", dpi)
    _plot_family_routes(
        all_family_rows, staging / "fig3_query_family_fast_robust_routes", dpi
    )
    _write_json(
        staging / "environment.json",
        {
            **environment_payload(),
            "started_utc": started_utc,
            "completed_utc": _utc(),
            "quiet_events": quiet_events,
        },
    )
    config_copy = staging / "hierarchical_v5_pilot.yaml"
    config_copy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    generated = sorted(path for path in staging.iterdir() if path.is_file())
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("features.py").resolve(),
        Path(__file__).with_name("model.py").resolve(),
        Path(__file__).with_name("policy.py").resolve(),
        Path(__file__).with_name("runtime.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
        Path(config["_config_path"]).resolve(),
        (workspace / "scripts" / "run_hierarchical_v5_pilot.sh").resolve(),
        (workspace / "tests" / "test_hierarchical_v5.py").resolve(),
    ]
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_smoke" if smoke else "complete_policy_validation_pilot",
        "development_roles": list(ROLE_ORDER),
        "robots": list(config["robots"]),
        "training_seed": 17,
        "methods": list(METHODS),
        "feature_names": list(CHEAP_FEATURE_NAMES),
        "source_config": _artifact(source_config_path),
        "pilot_config": _artifact(Path(config["_config_path"])),
        "source_records": source_records,
        "git_commit": git_commit,
        "git_status_at_start": git_status_at_start,
        "implementation_sources": {
            str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
            for path in implementation_paths
        },
        "test_data_loaded": False,
        "formal_test_started": False,
        "policy_validation_used_for_retuning": False,
        "pilot_gate": pilot_gate,
        "artifacts": {
            path.name: _artifact(path, relative_to=staging) for path in generated
        },
    }
    _write_json(staging / "run_manifest.json", manifest)
    os.replace(staging, output_root)
    print(f"[hierarchical-v5] completed: {output_root}", flush=True)
    return pilot_gate


def main() -> None:
    args = _parser().parse_args()
    run(args.config, smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
