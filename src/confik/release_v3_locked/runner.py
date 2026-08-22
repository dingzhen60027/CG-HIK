from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot
from ..data.datasets import QueryDataset, RiskDataset, TransitionDataset
from ..experiments.provenance import environment_payload
from ..experiments.policy_selection import action_predictions
from ..latency_pilot_v3.benchmark import (
    ProfiledCascadeRuntime,
    benchmark_points,
    benchmark_trajectories,
    warmup_runtimes,
)
from ..latency_pilot_v3.optimized_inference import (
    EagerRiskEngine,
    EagerSeedEngine,
    ExactSingleCallSeedEnsemble,
)
from ..latency_pilot_v3.runner import (
    _build_runtimes,
    _load_gate_config,
    _module_batch_predictions,
    _risk_metric_deltas,
    _solver_components,
)
from ..latency_pilot_v3.validation import (
    method_validation_metrics,
    record_equivalence,
    risk_probability_metrics,
    stratified_point_subset,
    trajectory_validation_subset,
)
from ..models.risk import RiskModel
from ..models.seed import TorchSeedEnsemble, encode_seed_inputs
from .artifacts import (
    export_frozen_risk,
    export_normalization,
    load_frozen_risk,
    load_locked_seed_engine,
)


BACKEND = "torchscript_exact"
ROBOTS = ("panda", "ur5e")
TRAINING_SEEDS = (17, 29, 43)
TOP_LEVEL_FILES = (
    "release_manifest.json",
    "release_equivalence.json",
    "release_environment.json",
    "release_config.yaml",
    "release_commands.sh",
    "source_tree_manifest.json",
)
COMBINATION_FILES = (
    "exact_seed_ensemble.ts",
    "hgb_vectorized_parameters.npz",
    "isotonic_calibration_parameters.npz",
    "normalization_parameters.npz",
    "solver_metadata.json",
    "route_thresholds.json",
    "seed_bank.npz",
    "runtime_spec.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal the locked v3 deployment artifacts")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Exercise one combination in a temporary directory; never writes release outputs.",
    )
    return parser


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML document must be a mapping: {path}")
    return payload


def _git(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_tree_manifest(workspace: Path, latency_config: Path) -> dict[str, Any]:
    top_level = Path(_git(workspace, "rev-parse", "--show-toplevel")).resolve()
    if top_level != workspace.resolve():
        raise RuntimeError(f"Git root must be the workspace: {top_level}")
    status = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"release requires a clean Git worktree:\n{status}")
    tracked = [value for value in _git(workspace, "ls-files", "-z").split("\0") if value]
    entries: dict[str, dict[str, Any]] = {}
    for relative in sorted(tracked):
        path = workspace / relative
        if not path.is_file():
            raise RuntimeError(f"tracked path is not a regular file: {relative}")
        metadata = path.stat()
        entries[relative] = {
            "sha256": _sha256_file(path),
            "size": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
    return {
        "captured_utc": _utc(),
        "git_repository_present": True,
        "git_root": str(top_level),
        "git_commit": _git(workspace, "rev-parse", "HEAD"),
        "git_commit_tree": _git(workspace, "rev-parse", "HEAD^{tree}"),
        "git_worktree_clean": True,
        "git_status_porcelain": "",
        "tracked_file_count": len(entries),
        "code_tree_sha256": _json_digest(entries),
        "latency_pilot_config_path": str(latency_config),
        "latency_pilot_config_sha256": _sha256_file(latency_config),
        "entries": entries,
    }


def _protected_snapshot(output_root: Path, patterns: list[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output must not be a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            metadata = path.stat()
            relative = str(path.relative_to(output_root))
            entries[relative] = {
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
            total_bytes += metadata.st_size
    return {
        "directories": [str(path.relative_to(output_root)) for path in sorted(directories)],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_digest": _json_digest(entries),
        "entries": entries,
    }


def _source_paths(workspace: Path, robot: str, seed: int) -> dict[str, Path]:
    root = workspace / "outputs" / f"paper_v2_seed{seed}" / robot
    paths = {
        "seed_model": root / "models" / "seed_ensemble.pt",
        "risk_model": root / "models" / "risk_model.joblib",
        "seed_bank": root / "models" / "seed_bank.npz",
        "solver_metadata": root / "models" / "solver_metadata.json",
        "policy_selection": root / "results" / "policy_selection_v2.json",
        "seed_validation": root / "datasets" / "seed_validation.npz",
        "risk_validation": root / "datasets" / "risk_validation.npz",
        "risk_validation_queries": root / "datasets" / "risk_validation_queries.npz",
        "policy_validation": root / "datasets" / "policy_validation.npz",
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {robot}/seed{seed} {role}: {path}")
        if "test" in path.name.lower():
            raise RuntimeError(f"release equivalence refuses test-named input: {path}")
    return paths


def _artifact_record(
    *,
    path: Path,
    workspace: Path,
    robot: str,
    seed: int,
    role: str,
    source_hashes: dict[str, str],
    export_script_hash: str,
    thread_config: dict[str, int],
) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(
            Path("outputs")
            / "release_v3_locked"
            / robot
            / f"seed{seed}"
            / path.name
        ),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "robot": robot,
        "training_seed": seed,
        "source_checkpoint_sha256": source_hashes["seed_model"],
        "source_risk_model_sha256": source_hashes["risk_model"],
        "source_policy_selection_sha256": source_hashes["policy_selection"],
        "source_solver_metadata_sha256": source_hashes["solver_metadata"],
        "source_seed_bank_sha256": source_hashes["seed_bank"],
        "export_script_sha256": export_script_hash,
        "backend": BACKEND,
        "device": "cuda:0",
        "dtype": "float32" if role == "exact_seed_ensemble" else "float64_or_metadata",
        "thread_configuration": thread_config,
    }


def _metric_deltas(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    keys = (
        "point_feasible_success",
        "point_rejectable_rejection",
        "point_feasible_mean_function_evaluations",
        "point_rejectable_mean_function_evaluations",
        "trajectory_completion",
        "trajectory_mean_function_evaluations",
        "trajectory_command_spike",
    )
    return {key: float(candidate[key]) - float(reference[key]) for key in keys}


def _additional_record_equivalence(
    records: list[dict[str, object]],
    *,
    robot: str,
    reference_backend: str,
    candidate_backend: str,
) -> dict[str, Any]:
    def rows(backend: str) -> dict[tuple[str, str, int], dict[str, object]]:
        return {
            (str(row["method"]), str(row["split"]), int(row["query_index"])): row
            for row in records
            if str(row["robot"]) == robot and str(row["backend"]) == backend
        }

    reference = rows(reference_backend)
    candidate = rows(candidate_backend)
    if set(reference) != set(candidate):
        raise RuntimeError("deployment equivalence record keys differ")
    keys = sorted(reference)
    fallback_matches = 0
    verification_matches = 0
    stage_matches = 0
    risk_probability_max_error = 0.0
    risk_score_max_error = 0.0
    query_hash_matches = 0
    for key in keys:
        left = reference[key]
        right = candidate[key]
        query_hash_matches += int(left["query_sha256"] == right["query_sha256"])
        fallback_matches += int(bool(left["fallback_used"]) == bool(right["fallback_used"]))
        verification_matches += int(
            tuple(left["verification_reasons"]) == tuple(right["verification_reasons"])
        )
        stage_matches += int(tuple(left["executed_stages"]) == tuple(right["executed_stages"]))
        risk_probability_max_error = max(
            risk_probability_max_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(left["risk_probabilities"], dtype=np.float64)
                        - np.asarray(right["risk_probabilities"], dtype=np.float64)
                    )
                )
            ),
        )
        risk_score_max_error = max(
            risk_score_max_error,
            abs(float(left["risk_score"]) - float(right["risk_score"])),
        )
    count = len(keys)
    return {
        "paired_record_count": count,
        "query_hash_agreement": query_hash_matches / count if count else float("nan"),
        "fallback_agreement": fallback_matches / count if count else float("nan"),
        "verification_reason_agreement": verification_matches / count if count else float("nan"),
        "executed_stage_agreement": stage_matches / count if count else float("nan"),
        "runtime_risk_probability_max_abs_error": risk_probability_max_error,
        "runtime_risk_score_max_abs_error": risk_score_max_error,
    }


def _environment(
    workspace: Path,
    latency_environment_path: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    current = environment_payload()
    current.update(
        {
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "environment_variables": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "gpu_index_after_visibility_filter": 0,
            "gpu_uuid": subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().splitlines()[0],
            "nvidia_driver": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().splitlines()[0],
            "git_commit": source_manifest["git_commit"],
            "code_tree_sha256": source_manifest["code_tree_sha256"],
        }
    )
    validation = json.loads(latency_environment_path.read_text(encoding="utf-8"))
    compared = ("torch", "cuda_version", "pinocchio", "scikit_learn", "gpu")
    matches = {key: current.get(key) == validation.get(key) for key in compared}
    current["latency_pilot_environment"] = {
        "path": str(latency_environment_path.relative_to(workspace)),
        "sha256": _sha256_file(latency_environment_path),
        "compared_fields": list(compared),
        "field_matches": matches,
        "all_compared_fields_match": all(matches.values()),
    }
    if not current["latency_pilot_environment"]["all_compared_fields_match"]:
        raise RuntimeError("current environment differs from latency_pilot_v3")
    return current


def _export_combination(
    *,
    workspace: Path,
    destination: Path,
    source_config: dict[str, Any],
    robot: str,
    seed: int,
    config: dict[str, Any],
    smoke: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _source_paths(workspace, robot, seed)
    source_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    destination.mkdir(parents=True, exist_ok=False)
    kinematics = load_robot(source_config, robot)
    gate_config, gate_source = _load_gate_config(paths["policy_selection"])
    ensemble = TorchSeedEnsemble.load(paths["seed_model"], kinematics, device="cuda:0")
    ensemble.members.eval()
    risk_model = RiskModel.load(paths["risk_model"])

    exact_source = ExactSingleCallSeedEnsemble(ensemble).eval()
    example = torch.empty(
        (1, kinematics.nq + 9), dtype=torch.float32, device=ensemble.device
    )
    with torch.inference_mode():
        traced = torch.jit.trace(exact_source, example, strict=True).eval()
    seed_artifact = destination / "exact_seed_ensemble.ts"
    torch.jit.save(traced, str(seed_artifact))
    del traced, exact_source
    torch.cuda.synchronize()

    forest_artifact = destination / "hgb_vectorized_parameters.npz"
    calibration_artifact = destination / "isotonic_calibration_parameters.npz"
    export_frozen_risk(risk_model, forest_artifact, calibration_artifact)
    normalization_artifact = destination / "normalization_parameters.npz"
    runtime_metadata = export_normalization(ensemble, normalization_artifact)

    shutil.copy2(paths["solver_metadata"], destination / "solver_metadata.json")
    shutil.copy2(paths["seed_bank"], destination / "seed_bank.npz")
    route_payload = {
        "backend": BACKEND,
        "selection_split": gate_source["selection_split"],
        "selected_config": asdict(gate_config),
        "source_policy_selection_path": str(paths["policy_selection"]),
        "source_policy_selection_sha256": source_hashes["policy_selection"],
        "consumed_json_keys": gate_source["consumed_json_keys"],
        "test_metrics_consumed": False,
    }
    _write_json(destination / "route_thresholds.json", route_payload)
    solver_keys = ("solver", "verifier", "cascade", "fallback")
    urdf_path = Path(source_config["robots"][robot]["urdf"])
    runtime_spec = {
        **runtime_metadata,
        "schema_version": 1,
        "robot": robot,
        "training_seed": seed,
        "backend": BACKEND,
        "device": "cuda:0",
        "dtype": "float32",
        "torchscript_load_only": True,
        "retrace_allowed": False,
        "source_artifacts": {
            role: {"path": str(path), "sha256": source_hashes[role]}
            for role, path in paths.items()
        },
        "source_urdf": {
            "path": str(urdf_path),
            "sha256": _sha256_file(urdf_path),
        },
        "locked_solver_configuration": {
            key: source_config[key] for key in solver_keys
        },
        "thread_configuration": {
            "intra_op_threads": 8,
            "inter_op_threads": 1,
        },
    }
    _write_json(destination / "runtime_spec.json", runtime_spec)

    # Deployment path begins here: all inference artifacts are loaded from
    # disk.  No trace or vector-parameter extraction occurs after this point.
    disk_seed = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=seed_artifact,
        normalization_path=normalization_artifact,
        runtime_spec_path=destination / "runtime_spec.json",
        device="cuda:0",
    )
    disk_risk = load_frozen_risk(forest_artifact, calibration_artifact)

    seed_validation = TransitionDataset.load(paths["seed_validation"])
    if smoke:
        seed_count = min(100, len(seed_validation))
        seed_slice = slice(0, seed_count)
    else:
        seed_count = len(seed_validation)
        seed_slice = slice(None)
    features = np.ascontiguousarray(
        encode_seed_inputs(
            kinematics,
            seed_validation.previous_q[seed_slice],
            seed_validation.target_position[seed_slice],
            seed_validation.target_rotation[seed_slice],
            use_history=ensemble.config.use_history,
        ).astype(np.float32)
    )
    reference_seed = ensemble.predict_deltas_batch(
        seed_validation.previous_q[seed_slice],
        seed_validation.target_position[seed_slice],
        seed_validation.target_rotation[seed_slice],
        batch_size=256 if smoke else 2048,
    )
    artifact_seed = _module_batch_predictions(
        disk_seed.module,
        features,
        batch_size=256 if smoke else 2048,
    )
    seed_error = float(np.max(np.abs(reference_seed - artifact_seed)))

    risk_splits: dict[str, Any] = {}
    risk_probability_error = 0.0
    risk_score_error = 0.0
    stored_route_agreement = 1.0
    risk_metric_max_delta = 0.0
    for split in ("risk_validation", "policy_validation"):
        risk_dataset = RiskDataset.load(paths[split])
        if smoke:
            selected = slice(0, min(500, len(risk_dataset)))
            split_features = risk_dataset.features[selected]
            split_labels = risk_dataset.labels[selected]
        else:
            split_features = risk_dataset.features
            split_labels = risk_dataset.labels
        reference_probability = risk_model.predict_proba(split_features)
        artifact_probability = disk_risk.predict_proba(split_features)
        probability_error = float(
            np.max(np.abs(reference_probability - artifact_probability))
        )
        score_error = float(
            np.max(
                np.abs(
                    reference_probability[:, 2:].sum(axis=1)
                    - artifact_probability[:, 2:].sum(axis=1)
                )
            )
        )
        reference_actions = action_predictions(reference_probability, gate_config)
        artifact_actions = action_predictions(artifact_probability, gate_config)
        route_agreement = float(np.mean(reference_actions == artifact_actions))
        reference_metrics = risk_probability_metrics(
            reference_probability, split_labels, gate_config
        )
        artifact_metrics = risk_probability_metrics(
            artifact_probability, split_labels, gate_config
        )
        metric_deltas = _risk_metric_deltas(artifact_metrics, reference_metrics)
        risk_probability_error = max(risk_probability_error, probability_error)
        risk_score_error = max(risk_score_error, score_error)
        stored_route_agreement = min(stored_route_agreement, route_agreement)
        risk_metric_max_delta = max(
            risk_metric_max_delta,
            max((abs(float(value)) for value in metric_deltas.values()), default=0.0),
        )
        risk_splits[split] = {
            "sample_count": len(split_features),
            "probability_max_abs_error": probability_error,
            "risk_score_max_abs_error": score_error,
            "routing_action_agreement": route_agreement,
            "metric_deltas": metric_deltas,
        }

    query_dataset = QueryDataset.load(paths["risk_validation_queries"])
    point_per_category = (
        3 if smoke else int(config["validation"]["point_queries_per_category"])
    )
    point_dataset, point_indices = stratified_point_subset(
        query_dataset,
        per_category=point_per_category,
        seed=int(config["validation"]["point_sampling_seed"]),
    )
    trajectory_dataset, trajectory_ids = trajectory_validation_subset(
        seed_validation,
        trajectory_count=1 if smoke else int(config["validation"]["trajectory_count"]),
        seed=int(config["validation"]["trajectory_sampling_seed"]),
    )
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {
            "solver_metadata": destination / "solver_metadata.json",
            "seed_bank": destination / "seed_bank.npz",
        },
        kinematics,
    )
    engines = {
        "eager_reference": (
            EagerSeedEngine(ensemble),
            EagerRiskEngine(risk_model),
            False,
        ),
        "torchscript_exact_disk": (disk_seed, disk_risk, True),
    }
    runtimes = _build_runtimes(
        kinematics=kinematics,
        gate_config=gate_config,
        engines=engines,
        dls=dls,
        verifier=verifier,
        fallback=fallback,
        seed_bank=seed_bank,
        cascade=cascade,
    )
    warmup_runtimes(
        runtimes,
        point_dataset,
        iterations=3 if smoke else int(config["validation"]["warmup_iterations"]),
        dt=float(source_config["data"].get("dt", 0.02)),
    )
    point_records, _ = benchmark_points(
        robot,
        runtimes,
        point_dataset,
        repeats=1,
        dt=float(source_config["data"].get("dt", 0.02)),
        order_seed=int(config["validation"]["method_order_seed"]),
    )
    trajectory_records = benchmark_trajectories(
        robot,
        runtimes,
        trajectory_dataset,
        dt=float(source_config["data"].get("dt", 0.02)),
        order_seed=int(config["validation"]["method_order_seed"]),
    )
    records = point_records + trajectory_records
    standard_records = record_equivalence(
        records,
        robot=robot,
        reference_backend="eager_reference",
        candidate_backend="torchscript_exact_disk",
    )
    additional_records = _additional_record_equivalence(
        records,
        robot=robot,
        reference_backend="eager_reference",
        candidate_backend="torchscript_exact_disk",
    )
    validation_metrics: dict[str, Any] = {}
    metric_deltas: dict[str, Any] = {}
    for method in ("baseline", "proposed"):
        reference_metrics = method_validation_metrics(
            records, robot, "eager_reference", method
        )
        artifact_metrics = method_validation_metrics(
            records, robot, "torchscript_exact_disk", method
        )
        validation_metrics[method] = {
            "reference": reference_metrics,
            "artifact": artifact_metrics,
        }
        metric_deltas[method] = _metric_deltas(artifact_metrics, reference_metrics)

    tolerance = config["equivalence"]
    exact_agreements = (
        standard_records["accepted_agreement"] == 1.0
        and standard_records["function_evaluations_agreement"] == 1.0
        and standard_records["point_function_evaluations_agreement"] == 1.0
        and standard_records["trajectory_function_evaluations_agreement"] == 1.0
        and standard_records["point_route_action_agreement"] == 1.0
        and standard_records["trajectory_route_action_agreement"] == 1.0
        and standard_records["all_route_action_agreement"] == 1.0
        and additional_records["query_hash_agreement"] == 1.0
        and additional_records["fallback_agreement"] == 1.0
        and additional_records["verification_reason_agreement"] == 1.0
        and additional_records["executed_stage_agreement"] == 1.0
    )
    invariant_metric_keys = (
        "point_feasible_success",
        "point_rejectable_rejection",
        "trajectory_completion",
        "trajectory_command_spike",
    )
    metric_gate = all(
        abs(float(metric_deltas[method][key]))
        <= float(tolerance["metric_abs_tolerance"])
        for method in ("baseline", "proposed")
        for key in invariant_metric_keys
    )
    point_fev_gate = all(
        abs(float(metric_deltas[method][key]))
        <= float(tolerance["point_mean_function_evaluations_abs_tolerance"])
        for method in ("baseline", "proposed")
        for key in (
            "point_feasible_mean_function_evaluations",
            "point_rejectable_mean_function_evaluations",
        )
    )
    trajectory_fev_gate = all(
        abs(float(metric_deltas[method]["trajectory_mean_function_evaluations"]))
        <= float(tolerance["trajectory_mean_function_evaluations_abs_tolerance"])
        for method in ("baseline", "proposed")
    )
    passed = bool(
        seed_error <= float(tolerance["seed_max_abs_tolerance"])
        and risk_probability_error
        <= float(tolerance["risk_probability_max_abs_tolerance"])
        and risk_score_error <= float(tolerance["risk_score_max_abs_tolerance"])
        and stored_route_agreement >= float(tolerance["routing_action_agreement_min"])
        and risk_metric_max_delta <= float(tolerance["metric_abs_tolerance"])
        and standard_records["accepted_command_max_abs_error_rad"]
        <= float(tolerance["seed_max_abs_tolerance"])
        and exact_agreements
        and metric_gate
        and point_fev_gate
        and trajectory_fev_gate
    )
    equivalence = {
        "robot": robot,
        "training_seed": seed,
        "backend": BACKEND,
        "deployment_artifacts_loaded_from_disk": True,
        "torchscript_retraced_during_equivalence": False,
        "seed_validation_sample_count": seed_count,
        "seed_output_max_absolute_error": seed_error,
        "risk_probability_max_absolute_error": risk_probability_error,
        "risk_score_max_absolute_error": risk_score_error,
        "stored_feature_routing_action_agreement": stored_route_agreement,
        "risk_metric_max_absolute_delta": risk_metric_max_delta,
        "risk_splits": risk_splits,
        "runtime_records": {**standard_records, **additional_records},
        "validation_metrics": validation_metrics,
        "validation_metric_deltas": metric_deltas,
        "point_source_indices": point_indices,
        "trajectory_ids": trajectory_ids,
        "gates": {
            "seed_output": seed_error <= float(tolerance["seed_max_abs_tolerance"]),
            "risk_probability": risk_probability_error
            <= float(tolerance["risk_probability_max_abs_tolerance"]),
            "risk_score": risk_score_error
            <= float(tolerance["risk_score_max_abs_tolerance"]),
            "stored_routing": stored_route_agreement
            >= float(tolerance["routing_action_agreement_min"]),
            "risk_metrics": risk_metric_max_delta
            <= float(tolerance["metric_abs_tolerance"]),
            "exact_runtime_agreements": exact_agreements,
            "accepted_joint_command": standard_records[
                "accepted_command_max_abs_error_rad"
            ]
            <= float(tolerance["seed_max_abs_tolerance"]),
            "success_rejection_trajectory_metrics": metric_gate,
            "point_mean_fev": point_fev_gate,
            "trajectory_mean_fev": trajectory_fev_gate,
        },
        "pass": passed,
    }

    export_script_hash = _sha256_file(Path(__file__))
    thread_config = {"intra_op_threads": 8, "inter_op_threads": 1}
    artifact_roles = {
        "exact_seed_ensemble.ts": "exact_seed_ensemble",
        "hgb_vectorized_parameters.npz": "frozen_hgb_vector_parameters",
        "isotonic_calibration_parameters.npz": "original_isotonic_parameters",
        "normalization_parameters.npz": "normalization_parameters",
        "solver_metadata.json": "solver_metadata",
        "route_thresholds.json": "route_thresholds",
        "seed_bank.npz": "seed_bank",
        "runtime_spec.json": "runtime_spec",
    }
    artifacts = [
        _artifact_record(
            path=destination / filename,
            workspace=workspace,
            robot=robot,
            seed=seed,
            role=role,
            source_hashes=source_hashes,
            export_script_hash=export_script_hash,
            thread_config=thread_config,
        )
        for filename, role in artifact_roles.items()
    ]
    if set(path.name for path in destination.iterdir()) != set(COMBINATION_FILES):
        raise RuntimeError(f"unexpected artifact set for {robot}/seed{seed}")
    return equivalence, artifacts


def _run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[3]
    config = _read_yaml(config_path)
    if config.get("backend") != BACKEND:
        raise RuntimeError("release backend is not the validation-locked torchscript_exact")
    if tuple(config.get("robots", ())) != ROBOTS:
        raise RuntimeError("release robot list differs from the preregistered pair")
    if tuple(int(value) for value in config.get("training_seeds", ())) != TRAINING_SEEDS:
        raise RuntimeError("release seed list differs from 17/29/43")
    runtime = config["runtime"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(runtime["cuda_visible_devices"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly set to 0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Quadro RTX 8000":
        raise RuntimeError("locked Quadro RTX 8000 is not available as cuda:0")
    torch.set_num_threads(int(runtime["intra_op_threads"]))
    torch.set_num_interop_threads(int(runtime["inter_op_threads"]))
    if torch.get_num_threads() != 8 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("locked torch thread configuration is not active")

    latency_config = (config_path.parent / config["latency_pilot_config"]).resolve()
    source_config_path = (config_path.parent / config["source_config"]).resolve()
    source_config = load_config(source_config_path)
    latency_output = (config_path.parent / config["latency_pilot_output"]).resolve()
    latency_run = json.loads((latency_output / "run_manifest.json").read_text(encoding="utf-8"))
    latency_equivalence = json.loads(
        (latency_output / "numerical_equivalence.json").read_text(encoding="utf-8")
    )
    latency_gate = json.loads(
        (latency_output / "validation_gate_v3.json").read_text(encoding="utf-8")
    )
    latency_config_payload = _read_yaml(latency_config)
    if latency_run["selected_backend"] != BACKEND:
        raise RuntimeError("latency pilot did not lock torchscript_exact")
    if not latency_equivalence["all_pass"] or not latency_gate["all_robots_pass"]:
        raise RuntimeError("latency pilot readiness/equivalence gate was not passed")
    if config["equivalence"] != latency_config_payload["equivalence"]:
        raise RuntimeError("release equivalence tolerances differ from latency_pilot_v3")

    source_manifest = _source_tree_manifest(workspace, latency_config)
    protected_patterns = [str(value) for value in config["protected_outputs"]]
    protected_before = _protected_snapshot(workspace / "outputs", protected_patterns)
    if "latency_pilot_v3" not in protected_before["directories"]:
        raise RuntimeError("latency_pilot_v3 is missing from protected snapshot")

    declared_output = (config_path.parent / config["output_directory"]).resolve()
    expected_output = (workspace / "outputs" / "release_v3_locked").resolve()
    if declared_output != expected_output:
        raise RuntimeError(f"release output must resolve exactly to {expected_output}")
    if declared_output.exists() or declared_output.is_symlink():
        raise RuntimeError("release_v3_locked already exists; overwrite is forbidden")

    combinations = (
        (("panda", 17),)
        if smoke
        else tuple((robot, seed) for robot in ROBOTS for seed in TRAINING_SEEDS)
    )
    if smoke:
        temporary = tempfile.TemporaryDirectory(prefix="confik_release_smoke_")
        staging = Path(temporary.name) / "release_v3_locked"
        staging.mkdir(parents=True)
    else:
        temporary = None
        staging = workspace / "outputs" / (
            f".release_v3_locked.incomplete.{os.getpid()}"
        )
        if staging.exists():
            raise RuntimeError(f"staging path already exists: {staging}")
        staging.mkdir(parents=False)

    equivalence_combinations: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    try:
        for robot, seed in combinations:
            print(f"[release-v3] exporting {robot}/seed{seed}", flush=True)
            result, combination_artifacts = _export_combination(
                workspace=workspace,
                destination=staging / robot / f"seed{seed}",
                source_config=source_config,
                robot=robot,
                seed=seed,
                config=config,
                smoke=smoke,
            )
            equivalence_combinations[f"{robot}/seed{seed}"] = result
            artifacts.extend(combination_artifacts)
            print(
                f"[release-v3] {robot}/seed{seed} equivalence pass={result['pass']}",
                flush=True,
            )
            if not result["pass"]:
                raise RuntimeError(f"packaging/export equivalence failed for {robot}/seed{seed}")
        all_pass = len(equivalence_combinations) == len(combinations) and all(
            value["pass"] for value in equivalence_combinations.values()
        )
        if smoke:
            assert temporary is not None
            temporary.cleanup()
            return {
                "smoke": True,
                "all_pass": all_pass,
                "combinations": equivalence_combinations,
                "formal_release_written": False,
                "test_v3_started": False,
            }
        if not all_pass or len(combinations) != 6:
            raise RuntimeError("all six locked deployment artifacts must pass")

        environment = _environment(
            workspace,
            latency_output / "environment.json",
            source_manifest,
        )
        release_equivalence = {
            "protocol_version": "release_v3_locked",
            "created_utc": _utc(),
            "backend": BACKEND,
            "reference_backend": "eager_original_validation_only",
            "tolerances": config["equivalence"],
            "backend_reselection_performed": False,
            "test_queries_loaded": False,
            "test_v3_started": False,
            "combination_count": len(equivalence_combinations),
            "combinations": equivalence_combinations,
            "all_six_pass": all_pass,
        }
        output_config = dict(config)
        output_config.update(
            {
                "frozen_backend": BACKEND,
                "backend_reselection_allowed": False,
                "source_config_sha256": _sha256_file(source_config_path),
                "latency_pilot_config_sha256": _sha256_file(latency_config),
                "latency_pilot_run_manifest_sha256": _sha256_file(
                    latency_output / "run_manifest.json"
                ),
                "git_commit": source_manifest["git_commit"],
                "code_tree_sha256": source_manifest["code_tree_sha256"],
                "paper_v2_config_snapshot": source_config,
            }
        )
        _write_json(staging / "release_equivalence.json", release_equivalence)
        _write_json(staging / "release_environment.json", environment)
        (staging / "release_config.yaml").write_text(
            yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        commands = """#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=\"$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}\"
PYTHON_BIN=\"${CONFIK_PYTHON:-/home/eric/anaconda3/envs/isaaclab_3/bin/python}\"

# Phase A: idempotence is intentionally forbidden; never overwrite the release.
\"${PYTHON_BIN}\" -m confik.release_v3_locked.runner --config configs/release_v3_locked.yaml

# Phase B is deliberately not invoked here.  It requires a separately frozen
# test_v3_preregistration.json after release_equivalence all_six_pass=true.
"""
        commands_path = staging / "release_commands.sh"
        commands_path.write_text(commands, encoding="utf-8")
        commands_path.chmod(0o755)
        _write_json(staging / "source_tree_manifest.json", source_manifest)

        # Verify every combination artifact after all serialization and before
        # publishing the staging directory.
        for artifact in artifacts:
            artifact_path = workspace / artifact["path"]
            # Manifest paths point at the final directory; translate to staging.
            staged_path = staging / Path(artifact["path"]).relative_to(
                "outputs/release_v3_locked"
            )
            if _sha256_file(staged_path) != artifact["sha256"]:
                raise RuntimeError(f"artifact hash changed after write: {artifact_path}")
        protected_after = _protected_snapshot(workspace / "outputs", protected_patterns)
        protected_unchanged = protected_before["entries"] == protected_after["entries"]
        if not protected_unchanged:
            before_keys = set(protected_before["entries"])
            after_keys = set(protected_after["entries"])
            changed = sorted(
                (before_keys ^ after_keys)
                | {
                    key
                    for key in before_keys & after_keys
                    if protected_before["entries"][key] != protected_after["entries"][key]
                }
            )
            raise RuntimeError(f"protected formal output changed: {changed[:10]}")
        release_manifest = {
            "protocol_version": "release_v3_locked",
            "created_utc": _utc(),
            "release_status": "sealed",
            "backend": BACKEND,
            "backend_reselection_performed": False,
            "robot_seed_combinations": [
                {"robot": robot, "training_seed": seed}
                for robot, seed in combinations
            ],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "all_artifact_hashes_verified_after_write": True,
            "release_equivalence_all_six_pass": True,
            "formal_test_v3_started": False,
            "test_named_dataset_loaded": False,
            "git_commit": source_manifest["git_commit"],
            "git_worktree_clean_before_release": True,
            "code_tree_sha256": source_manifest["code_tree_sha256"],
            "latency_pilot_config_sha256": _sha256_file(latency_config),
            "protected_outputs": {
                "before_tree_digest": protected_before["tree_digest"],
                "after_tree_digest": protected_after["tree_digest"],
                "file_count": protected_before["file_count"],
                "total_bytes": protected_before["total_bytes"],
                "unchanged": True,
            },
            "top_level_files": list(TOP_LEVEL_FILES),
            "thread_configuration": {
                "intra_op_threads": 8,
                "inter_op_threads": 1,
            },
            "device": "cuda:0",
            "dtype": "float32",
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        if set(path.name for path in staging.iterdir() if path.is_file()) != set(
            TOP_LEVEL_FILES
        ):
            raise RuntimeError("release top-level file set is incomplete or unexpected")
        staging.rename(declared_output)
        if _git(workspace, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError("Git worktree became dirty after release")
        print(f"[release-v3] sealed at {declared_output}", flush=True)
        return release_manifest
    except Exception as error:
        if not smoke:
            _write_json(
                staging / "PACKAGING_ERROR.json",
                {
                    "created_utc": _utc(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "backend_change_attempted": False,
                    "test_v3_started": False,
                },
            )
            print(f"[release-v3] packaging stopped; diagnostics retained at {staging}", flush=True)
        raise


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = _run(Path(args.config).resolve(), smoke=args.smoke)
    if args.smoke:
        print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
