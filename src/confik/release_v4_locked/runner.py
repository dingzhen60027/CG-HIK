"""Seal the exact validation-frozen CG-HIK v4 deployment package.

This module never discovers or reads a test-named dataset.  ``--smoke`` uses a
temporary directory and one robot/seed combination; the formal path requires
the release-relevant source/configuration scope to be clean, validates both
frozen input releases, runs all six validation-only runtime equivalence
checks, and atomically renames the staging directory only after every gate
passes.  Unrelated user documents may remain dirty and are reported without
being modified or treated as release code.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot
from ..counterfactual_v4.model import (
    FEATURE_NAMES,
    LABEL_CONTRACT,
    CounterfactualV4Predictor,
)
from ..counterfactual_v4.runtime_v4 import wrap_profiled_runtime
from ..data.datasets import QueryDataset, TransitionDataset
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.benchmark import ProfiledOutcome, query_digest, query_from_dataset
from ..latency_pilot_v3.runner import _solver_components
from ..latency_pilot_v3.validation import (
    stratified_point_subset,
    trajectory_validation_subset,
)
from ..release_v3_locked.artifacts import load_locked_seed_engine
from .artifacts import (
    EagerV4Inference,
    FrozenV4Policy,
    TorchScriptV4Inference,
    V4InferenceOutput,
    _decision_from_output,
    decision_record,
    export_exact_v4_predictor,
    load_exact_v4_predictor,
    load_policy_config,
)


PROTOCOL = "release_v4_locked"
BACKEND = "torchscript_exact_v4"
ROBOTS = ("panda", "ur5e")
UPSTREAM_SEEDS = (17, 29, 43)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal the exact v4 deployment release")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate panda/seed17 in a temporary directory; never write the release",
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


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML document must contain a mapping: {path}")
    return payload


def _git(workspace: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_snapshot(output_root: Path, patterns: Iterable[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
    entries: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output cannot be a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            metadata = path.stat()
            entries[str(path.relative_to(output_root))] = {
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return {
        "directories": [str(path.relative_to(output_root)) for path in sorted(directories)],
        "file_count": len(entries),
        "total_bytes": sum(int(value["size"]) for value in entries.values()),
        "tree_digest": _json_digest(entries),
    }


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError(f"artifact manifest has no file mapping: {root}")
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"frozen artifact is missing or is a symlink: {path}")
        if path.stat().st_size != int(expected["size"]):
            raise RuntimeError(f"frozen artifact size changed: {path}")
        if _sha256_file(path) != str(expected["sha256"]):
            raise RuntimeError(f"frozen artifact hash changed: {path}")


def _verify_candidate(candidate_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidate_root.is_dir() or candidate_root.is_symlink():
        raise FileNotFoundError(candidate_root)
    artifact_path = candidate_root / "artifact_manifest.json"
    run_path = candidate_root / "run_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "frozen_validation_candidate":
        raise RuntimeError("v4 candidate is not validation-frozen")
    if run.get("protocol") != "counterfactual_v4_validation_training_v2":
        raise RuntimeError("v4 candidate protocol differs from the shared-success contract")
    if bool(run.get("test_data_loaded", True)) or bool(
        run.get("formal_test_authorized_or_started", True)
    ):
        raise RuntimeError("v4 candidate provenance indicates forbidden formal-test access")
    if run.get("label_contract") != LABEL_CONTRACT:
        raise RuntimeError("v4 candidate label contract differs from release code")
    if _sha256_file(artifact_path) != run.get("artifact_manifest_sha256"):
        raise RuntimeError("v4 candidate artifact-manifest hash changed")
    _verify_manifest_files(candidate_root, artifact)
    return run, artifact


def _verify_v3_release(workspace: Path, release_root: Path) -> dict[str, Any]:
    manifest_path = release_root / "release_manifest.json"
    equivalence_path = release_root / "release_equivalence.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if (
        manifest.get("release_status") != "sealed"
        or manifest.get("backend") != "torchscript_exact"
        or not bool(manifest.get("release_equivalence_all_six_pass", False))
        or not bool(equivalence.get("all_six_pass", False))
    ):
        raise RuntimeError("upstream release_v3_locked is not eligible")
    if int(manifest.get("artifact_count", -1)) != 48:
        raise RuntimeError("upstream v3 release must contain 48 artifacts")
    for item in manifest["artifacts"]:
        path = workspace / str(item["path"])
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"upstream v3 artifact is missing: {path}")
        if path.stat().st_size != int(item["size"]) or _sha256_file(path) != str(
            item["sha256"]
        ):
            raise RuntimeError(f"upstream v3 artifact changed: {path}")
    return manifest


def _v3_paths(release_root: Path, robot: str, seed: int) -> dict[str, Path]:
    root = release_root / robot / f"seed{seed}"
    result = {
        "torchscript": root / "exact_seed_ensemble.ts",
        "normalization": root / "normalization_parameters.npz",
        "runtime_spec": root / "runtime_spec.json",
        "solver_metadata": root / "solver_metadata.json",
        "seed_bank": root / "seed_bank.npz",
    }
    for path in result.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    return result


def _development_dataset_path(
    workspace: Path, robot: str, seed: int, filename: str
) -> Path:
    if "test" in filename.lower():
        raise RuntimeError(f"test-named dataset is forbidden during release: {filename}")
    path = (
        workspace
        / "outputs"
        / f"paper_v2_seed{seed}"
        / robot
        / "datasets"
        / filename
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _load_policy_validation_features(
    bulk_root: Path, robot: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    role = bulk_root / robot / "seed17" / "policy_validation_queries"
    selection_manifest = json.loads(
        (role / "selection_manifest.json").read_text(encoding="utf-8")
    )
    if bool(selection_manifest.get("test_named_dataset_loaded", True)):
        raise RuntimeError("policy-validation selection indicates test access")
    arrays: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    expected_start = 0
    for chunk in sorted((role / "chunks").glob("chunk_*")):
        manifest_path = chunk / "chunk_manifest.json"
        labels_path = chunk / "counterfactual_labels.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            bool(manifest.get("environment_contaminated", True))
            or bool(manifest.get("test_data_loaded", True))
            or int(manifest.get("query_start", -1)) != expected_start
        ):
            raise RuntimeError(f"invalid policy-validation chunk: {chunk}")
        for name, expected in manifest["artifacts"].items():
            artifact = chunk / name
            if (
                not artifact.is_file()
                or artifact.stat().st_size != int(expected["size"])
                or _sha256_file(artifact) != str(expected["sha256"])
            ):
                raise RuntimeError(f"policy-validation chunk hash mismatch: {artifact}")
        with np.load(labels_path, allow_pickle=False) as data:
            if tuple(data["feature_names"].astype(str).tolist()) != FEATURE_NAMES:
                raise RuntimeError("policy-validation feature schema changed")
            arrays.append(np.asarray(data["features"], dtype=np.float32))
        expected_start = int(manifest["query_stop_exclusive"])
        sources.extend(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in (manifest_path, labels_path)
        )
    features = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    if len(features) != int(selection_manifest["selected_query_count"]):
        raise RuntimeError("policy-validation feature set is incomplete")
    return features, sources


def _module_batch(
    module: torch.jit.ScriptModule, features: np.ndarray, batch_size: int = 2048
) -> tuple[np.ndarray, ...]:
    collected: list[list[np.ndarray]] = [[] for _ in range(7)]
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(features[start : start + batch_size], dtype=np.float32)
            )
            values = module(tensor)
            for index, value in enumerate(values):
                collected[index].append(value.detach().cpu().numpy())
    return tuple(np.concatenate(values, axis=0) for values in collected)


def _output_at(values: tuple[np.ndarray, ...], index: int) -> V4InferenceOutput:
    return V4InferenceOutput(
        values[0][index],
        values[1][index],
        values[2][index],
        float(values[3][index]),
        values[4][index],
        float(values[5][index]),
        bool(values[6][index]),
    )


def _numerical_equivalence(
    *,
    candidate_path: Path,
    module: torch.jit.ScriptModule,
    features: np.ndarray,
    policy_config: Any,
    tolerance: Mapping[str, float],
) -> dict[str, Any]:
    # The deployable contract is batch one.  GEMM kernels can legitimately
    # choose different reduction paths for N=2500 and N=1, so equivalence is
    # measured at the exact batch-one API boundary used by the robot runtime.
    eager_backend = EagerV4Inference(candidate_path)
    exact_backend = TorchScriptV4Inference(module)
    errors = {
        "success_probability": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "fail_all_probability": 0.0,
        "embedding": 0.0,
        "ood_score": 0.0,
    }
    ood_matches = 0
    route_matches = 0
    reason_matches = 0
    eligible_matches = 0
    action_counts: dict[str, int] = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "reject": 0,
        "defer": 0,
    }
    for index in range(len(features)):
        eager_output = eager_backend.infer(features[index])
        exact_output = exact_backend.infer(features[index])
        for name, left_value, right_value in (
            (
                "success_probability",
                eager_output.success_probabilities,
                exact_output.success_probabilities,
            ),
            ("latency_p50_ms", eager_output.latency_p50_ms, exact_output.latency_p50_ms),
            ("latency_p95_ms", eager_output.latency_p95_ms, exact_output.latency_p95_ms),
            (
                "fail_all_probability",
                [eager_output.fail_all_probability],
                [exact_output.fail_all_probability],
            ),
            ("embedding", eager_output.embedding, exact_output.embedding),
            ("ood_score", [eager_output.ood_score], [exact_output.ood_score]),
        ):
            errors[name] = max(
                errors[name],
                float(
                    np.max(
                        np.abs(
                            np.asarray(left_value, dtype=np.float64)
                            - np.asarray(right_value, dtype=np.float64)
                        )
                    )
                ),
            )
        ood_matches += int(eager_output.is_ood == exact_output.is_ood)
        left = _decision_from_output(eager_output, policy_config)
        right = _decision_from_output(exact_output, policy_config)
        route_matches += int(left.action == right.action)
        reason_matches += int(left.reason == right.reason)
        eligible_matches += int(left.eligible_actions == right.eligible_actions)
        action_counts[right.action] += 1
    ood_agreement = ood_matches / len(features)
    thresholds = {
        "success_probability": float(tolerance["probability_max_abs"]),
        "latency_p50_ms": float(tolerance["latency_max_abs_ms"]),
        "latency_p95_ms": float(tolerance["latency_max_abs_ms"]),
        "fail_all_probability": float(tolerance["probability_max_abs"]),
        "embedding": float(tolerance["embedding_max_abs"]),
        "ood_score": float(tolerance["ood_score_max_abs"]),
    }
    passed = (
        all(errors[name] <= threshold for name, threshold in thresholds.items())
        and ood_agreement == 1.0
        and route_matches == len(features)
        and reason_matches == len(features)
        and eligible_matches == len(features)
    )
    return {
        "sample_count": len(features),
        "inference_batch_size": 1,
        "max_absolute_errors": errors,
        "tolerances": thresholds,
        "ood_decision_agreement": ood_agreement,
        "route_action_agreement": route_matches / len(features),
        "route_reason_agreement": reason_matches / len(features),
        "eligible_action_agreement": eligible_matches / len(features),
        "exact_route_counts": action_counts,
        "raw_probabilities_pass_through_calibrated_risk": False,
        "pass": passed,
    }


def _semantic_row(
    outcome: ProfiledOutcome,
    decision: Any,
    query_sha256: str,
) -> dict[str, Any]:
    return {
        "query_sha256": query_sha256,
        "accepted": bool(outcome.accepted),
        "command_q": None if outcome.q is None else np.asarray(outcome.q).copy(),
        "function_evaluations": int(outcome.function_evaluations),
        "iterations": int(outcome.iterations),
        "fallback_used": bool(outcome.fallback_used),
        "verification_reasons": tuple(outcome.verification_reasons),
        "executed_stages": tuple(outcome.executed_stages),
        "reject_reason": outcome.reject_reason,
        "decision": decision,
    }


def _run_runtime(
    runtime: Any,
    dataset: QueryDataset,
    *,
    dt: float,
    closed_loop: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states: dict[int, np.ndarray] = {}
    for index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[index])
        previous = states.get(trajectory_id) if closed_loop else None
        query = query_from_dataset(dataset, index, previous_q=previous, dt=dt)
        outcome = runtime.solve(query)
        decision = runtime.last_decision
        if decision is None:
            raise RuntimeError("v4 runtime did not expose its frozen raw decision")
        if closed_loop:
            states[trajectory_id] = (
                np.asarray(outcome.q, dtype=np.float64).copy()
                if outcome.accepted and outcome.q is not None
                else np.asarray(query.previous_q, dtype=np.float64).copy()
            )
        rows.append(_semantic_row(outcome, decision, query_digest(query)))
    return rows


def _runtime_equivalence(
    reference: list[dict[str, Any]],
    exact: list[dict[str, Any]],
    *,
    tolerance: Mapping[str, float],
) -> dict[str, Any]:
    if len(reference) != len(exact):
        raise RuntimeError("runtime equivalence record counts differ")
    counts = {
        "query_hash": 0,
        "accepted": 0,
        "function_evaluations": 0,
        "iterations": 0,
        "fallback": 0,
        "verification_reasons": 0,
        "executed_stages": 0,
        "reject_reason": 0,
        "route_action": 0,
        "route_reason": 0,
        "eligible_actions": 0,
        "ood_decision": 0,
    }
    raw_errors = {
        "success_probability": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "fail_all_probability": 0.0,
        "ood_score": 0.0,
    }
    command_error = 0.0
    paired_commands = 0
    reject_zero_fev = 0
    reject_count = 0
    defer_enters_easy = 0
    defer_count = 0
    for left, right in zip(reference, exact, strict=True):
        counts["query_hash"] += int(left["query_sha256"] == right["query_sha256"])
        for key in (
            "accepted",
            "function_evaluations",
            "iterations",
            "fallback_used",
            "verification_reasons",
            "executed_stages",
            "reject_reason",
        ):
            target = "fallback" if key == "fallback_used" else key
            counts[target] += int(left[key] == right[key])
        left_decision = left["decision"]
        right_decision = right["decision"]
        counts["route_action"] += int(left_decision.action == right_decision.action)
        counts["route_reason"] += int(left_decision.reason == right_decision.reason)
        counts["eligible_actions"] += int(
            left_decision.eligible_actions == right_decision.eligible_actions
        )
        counts["ood_decision"] += int(left_decision.is_ood == right_decision.is_ood)
        for name, left_value, right_value in (
            (
                "success_probability",
                left_decision.predicted_success,
                right_decision.predicted_success,
            ),
            ("latency_p50_ms", left_decision.predicted_p50_ms, right_decision.predicted_p50_ms),
            ("latency_p95_ms", left_decision.predicted_p95_ms, right_decision.predicted_p95_ms),
            (
                "fail_all_probability",
                [left_decision.fail_all_probability],
                [right_decision.fail_all_probability],
            ),
            ("ood_score", [left_decision.ood_score], [right_decision.ood_score]),
        ):
            raw_errors[name] = max(
                raw_errors[name],
                float(
                    np.max(
                        np.abs(
                            np.asarray(left_value, dtype=np.float64)
                            - np.asarray(right_value, dtype=np.float64)
                        )
                    )
                ),
            )
        if left["command_q"] is not None or right["command_q"] is not None:
            if left["command_q"] is None or right["command_q"] is None:
                command_error = float("inf")
            else:
                paired_commands += 1
                command_error = max(
                    command_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(left["command_q"], dtype=np.float64)
                                - np.asarray(right["command_q"], dtype=np.float64)
                            )
                        )
                    ),
                )
        if right_decision.action == "reject":
            reject_count += 1
            reject_zero_fev += int(
                right["function_evaluations"] == 0 and not right["executed_stages"]
            )
        if right_decision.action == "defer":
            defer_count += 1
            defer_enters_easy += int(
                bool(right["executed_stages"])
                and right["executed_stages"][0] == "easy"
            )
    count = len(reference)
    agreements = {key: value / count if count else float("nan") for key, value in counts.items()}
    exact_semantics = all(value == 1.0 for value in agreements.values())
    numerical_pass = bool(
        raw_errors["success_probability"] <= float(tolerance["probability_max_abs"])
        and raw_errors["fail_all_probability"]
        <= float(tolerance["probability_max_abs"])
        and raw_errors["latency_p50_ms"] <= float(tolerance["latency_max_abs_ms"])
        and raw_errors["latency_p95_ms"] <= float(tolerance["latency_max_abs_ms"])
        and raw_errors["ood_score"] <= float(tolerance["ood_score_max_abs"])
        and command_error <= float(tolerance["accepted_command_max_abs_rad"])
    )
    return {
        "paired_record_count": count,
        "agreements": agreements,
        "raw_prediction_max_absolute_errors": raw_errors,
        "raw_prediction_tolerances": {
            "probability": float(tolerance["probability_max_abs"]),
            "latency_ms": float(tolerance["latency_max_abs_ms"]),
            "ood_score": float(tolerance["ood_score_max_abs"]),
        },
        "accepted_command_max_absolute_error_rad": command_error,
        "accepted_command_tolerance_rad": float(
            tolerance["accepted_command_max_abs_rad"]
        ),
        "paired_accepted_command_count": paired_commands,
        "command_reject_count": reject_count,
        "command_reject_zero_solver_rate": (
            reject_zero_fev / reject_count if reject_count else 1.0
        ),
        "defer_count": defer_count,
        "defer_enters_fixed_easy_stage_rate": (
            defer_enters_easy / defer_count if defer_count else 1.0
        ),
        "pass": bool(
            exact_semantics
            and numerical_pass
            and reject_zero_fev == reject_count
            and defer_enters_easy == defer_count
        ),
    }


def _runtime_validation(
    *,
    workspace: Path,
    source_config: dict[str, Any],
    release_v3_root: Path,
    robot: str,
    seed: int,
    candidate_path: Path,
    exact_module_path: Path,
    policy_path: Path,
    config: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    kinematics = load_robot(source_config, robot)
    paths = _v3_paths(release_v3_root, robot, seed)
    seed_engine = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=paths["torchscript"],
        normalization_path=paths["normalization"],
        runtime_spec_path=paths["runtime_spec"],
        device="cuda:0",
    )
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    policy_config, _ = load_policy_config(policy_path)
    eager_policy = FrozenV4Policy(EagerV4Inference(candidate_path), policy_config)
    exact_policy = FrozenV4Policy(
        TorchScriptV4Inference(load_exact_v4_predictor(exact_module_path)),
        policy_config,
    )
    runtimes = {
        "eager": wrap_profiled_runtime(
            name="v4_eager_validation_reference",
            policy=eager_policy,  # type: ignore[arg-type]
            kinematics=kinematics,
            seed_engine=seed_engine,
            dls=dls,
            verifier=verifier,
            seed_bank=seed_bank,
            fallback=fallback,
            cascade_config=cascade,
        ),
        "exact": wrap_profiled_runtime(
            name="v4_torchscript_exact_disk",
            policy=exact_policy,  # type: ignore[arg-type]
            kinematics=kinematics,
            seed_engine=seed_engine,
            dls=dls,
            verifier=verifier,
            seed_bank=seed_bank,
            fallback=fallback,
            cascade_config=cascade,
        ),
    }

    validation = config["validation"]
    point_source = QueryDataset.load(
        _development_dataset_path(
            workspace, robot, seed, "risk_validation_queries.npz"
        )
    )
    point_count = 2 if smoke else int(validation["point_queries_per_category"])
    points, point_indices = stratified_point_subset(
        point_source,
        per_category=point_count,
        seed=int(validation["point_sampling_seed"]) + seed,
    )
    trajectory_source = TransitionDataset.load(
        _development_dataset_path(workspace, robot, seed, "seed_validation.npz")
    )
    trajectories, trajectory_ids = trajectory_validation_subset(
        trajectory_source,
        trajectory_count=1 if smoke else int(validation["trajectory_count"]),
        seed=int(validation["trajectory_sampling_seed"]) + seed,
    )
    dt = float(source_config["data"].get("dt", 0.02))
    # Warm both backends on the same validation-only query. No timing result is
    # used for release selection.
    if len(points):
        warm = query_from_dataset(points, 0, dt=dt)
        for index in range(2 if smoke else int(validation["warmup_iterations"])):
            order = ("exact", "eager") if index % 2 else ("eager", "exact")
            for name in order:
                runtimes[name].solve(warm)

    point_reference = _run_runtime(runtimes["eager"], points, dt=dt, closed_loop=False)
    point_exact = _run_runtime(runtimes["exact"], points, dt=dt, closed_loop=False)
    trajectory_reference = _run_runtime(
        runtimes["eager"], trajectories, dt=dt, closed_loop=True
    )
    trajectory_exact = _run_runtime(
        runtimes["exact"], trajectories, dt=dt, closed_loop=True
    )
    point_result = _runtime_equivalence(
        point_reference, point_exact, tolerance=config["equivalence"]
    )
    trajectory_result = _runtime_equivalence(
        trajectory_reference, trajectory_exact, tolerance=config["equivalence"]
    )
    return {
        "robot": robot,
        "upstream_seed": seed,
        "validation_only": True,
        "test_named_dataset_loaded": False,
        "point_source": str(
            _development_dataset_path(
                workspace, robot, seed, "risk_validation_queries.npz"
            )
        ),
        "trajectory_source": str(
            _development_dataset_path(workspace, robot, seed, "seed_validation.npz")
        ),
        "point_source_indices": point_indices,
        "trajectory_ids": trajectory_ids,
        "point": point_result,
        "trajectory": trajectory_result,
        "pass": bool(point_result["pass"] and trajectory_result["pass"]),
    }


def _environment(candidate_root: Path, release_v3_root: Path) -> dict[str, Any]:
    payload = environment_payload()
    payload.update(
        {
            "captured_utc": _utc(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "candidate_environment_sha256": _sha256_file(
                candidate_root / "environment.json"
            ),
            "upstream_v3_environment_sha256": _sha256_file(
                release_v3_root / "release_environment.json"
            ),
        }
    )
    return payload


def _validate_environment(
    candidate_root: Path,
    release_v3_root: Path,
    *,
    expected_gpu_name: str,
) -> dict[str, Any]:
    current = environment_payload()
    candidate = json.loads((candidate_root / "environment.json").read_text(encoding="utf-8"))
    upstream = json.loads(
        (release_v3_root / "release_environment.json").read_text(encoding="utf-8")
    )
    fields = ("torch", "cuda_version", "pinocchio", "scikit_learn", "gpu")
    candidate_matches = {
        field: current.get(field) == candidate.get(field) for field in fields
    }
    upstream_matches = {
        field: current.get(field) == upstream.get(field) for field in fields
    }
    current_gpu = str(current.get("gpu"))
    if current_gpu != expected_gpu_name:
        raise RuntimeError(
            f"release_v4 expected GPU {expected_gpu_name!r}, found {current_gpu!r}"
        )
    gpu_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    uuid_match = gpu_uuid == upstream.get("gpu_uuid")
    driver_match = driver == upstream.get("nvidia_driver")
    passed = bool(
        all(candidate_matches.values())
        and all(upstream_matches.values())
        and uuid_match
        and driver_match
    )
    result = {
        "fields": list(fields),
        "candidate_matches": candidate_matches,
        "upstream_v3_matches": upstream_matches,
        "gpu_uuid": gpu_uuid,
        "gpu_uuid_matches_upstream_v3": uuid_match,
        "nvidia_driver": driver,
        "nvidia_driver_matches_upstream_v3": driver_match,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"release_v4 environment differs from frozen validation: {result}")
    return result


def _formal_source_manifest(workspace: Path) -> dict[str, Any]:
    top = Path(_git(workspace, "rev-parse", "--show-toplevel")).resolve()
    if top != workspace.resolve():
        raise RuntimeError(f"Git root differs from workspace: {top}")
    scoped_paths = (
        "src/confik",
        "configs/paper_v2.yaml",
        "configs/counterfactual_v4_train.yaml",
        "configs/release_v4_locked.yaml",
        "scripts/run_release_v4.sh",
        "tests/test_release_v4_locked.py",
    )
    scoped_status = _git(
        workspace,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scoped_paths,
    )
    if scoped_status:
        raise RuntimeError(
            "formal v4 release requires a clean release-source scope:\n"
            f"{scoped_status}"
        )
    full_status = _git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all"
    )
    return {
        "git_commit": _git(workspace, "rev-parse", "HEAD"),
        "git_tree": _git(workspace, "rev-parse", "HEAD^{tree}"),
        "release_source_scope": list(scoped_paths),
        "release_source_scope_clean": True,
        "git_worktree_clean": not bool(full_status),
        "out_of_scope_changes_present": bool(full_status),
        "out_of_scope_change_count": len(full_status.splitlines()),
        "runner_sha256": _sha256_file(Path(__file__)),
        "artifacts_sha256": _sha256_file(Path(__file__).with_name("artifacts.py")),
    }


def _verify_formal_source_stable(
    workspace: Path, initial: Mapping[str, Any]
) -> dict[str, Any]:
    """Recheck the release source immediately before the atomic freeze."""

    current = _formal_source_manifest(workspace)
    stable_fields = (
        "git_commit",
        "git_tree",
        "release_source_scope",
        "release_source_scope_clean",
        "runner_sha256",
        "artifacts_sha256",
    )
    mismatched = {
        field: {"initial": initial.get(field), "current": current.get(field)}
        for field in stable_fields
        if initial.get(field) != current.get(field)
    }
    if mismatched:
        raise RuntimeError(
            "formal v4 release source changed during validation: "
            f"{mismatched}"
        )
    return current


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _read_yaml(path)
    workspace = Path(__file__).resolve().parents[3]
    if config.get("protocol_version") != PROTOCOL:
        raise RuntimeError("release_v4 protocol differs from the frozen contract")
    if tuple(config.get("robots", ())) != ROBOTS or tuple(
        int(value) for value in config.get("upstream_training_seeds", ())
    ) != UPSTREAM_SEEDS:
        raise RuntimeError("release_v4 must validate Panda/UR5e x seeds 17/29/43")
    if config.get("backend") != BACKEND:
        raise RuntimeError("release_v4 backend must be torchscript_exact_v4")
    candidate_root = (path.parent / config["candidate_directory"]).resolve()
    release_v3_root = (path.parent / config["upstream_v3_directory"]).resolve()
    output_root = (path.parent / config["output_directory"]).resolve()
    if output_root != (workspace / "outputs" / "release_v4_locked").resolve():
        raise RuntimeError("formal release path must be outputs/release_v4_locked")

    runtime = config["runtime"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(runtime["cuda_visible_devices"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly locked")
    torch.set_num_threads(int(runtime["intra_op_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["inter_op_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(runtime["inter_op_threads"]):
            raise
    if not torch.cuda.is_available():
        raise RuntimeError("upstream exact seed deployment requires cuda:0")

    candidate_run, candidate_artifacts = _verify_candidate(candidate_root)
    v3_manifest = _verify_v3_release(workspace, release_v3_root)
    environment_equivalence = _validate_environment(
        candidate_root,
        release_v3_root,
        expected_gpu_name=str(runtime["expected_gpu_name"]),
    )
    source_config_path = (path.parent / config["source_config"]).resolve()
    source_config = load_config(source_config_path)
    bulk_root = Path(str(candidate_run["bulk_root"]))
    if bulk_root.resolve() != (workspace / "outputs" / "counterfactual_v4_bulk").resolve():
        raise RuntimeError("candidate bulk provenance points outside the frozen v4 bulk root")

    source_manifest = (
        {
            "smoke": True,
            "formal_git_cleanliness_not_asserted": True,
            "runner_sha256": _sha256_file(Path(__file__)),
            "artifacts_sha256": _sha256_file(Path(__file__).with_name("artifacts.py")),
        }
        if smoke
        else _formal_source_manifest(workspace)
    )
    if not smoke and (output_root.exists() or output_root.is_symlink()):
        raise RuntimeError("release_v4_locked already exists; overwrite is forbidden")

    protected_before = _tree_snapshot(
        workspace / "outputs", [str(value) for value in config["protected_outputs"]]
    )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if smoke:
        temporary = tempfile.TemporaryDirectory(prefix="confik_release_v4_smoke_")
        staging = Path(temporary.name) / "release_v4_locked"
    else:
        staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    combinations = (("panda", 17),) if smoke else tuple(
        (robot, seed) for robot in ROBOTS for seed in UPSTREAM_SEEDS
    )
    robot_payload: dict[str, Any] = {}
    runtime_payload: dict[str, Any] = {}
    try:
        for robot in ("panda",) if smoke else ROBOTS:
            candidate_path = candidate_root / "models" / f"{robot}_seed17_predictor.pt"
            policy_source = candidate_root / "policies" / f"{robot}_seed17_policy.json"
            destination = staging / robot
            destination.mkdir(parents=True)
            module_path = destination / "exact_v4_predictor.ts"
            metadata = export_exact_v4_predictor(candidate_path, module_path)
            policy_config, policy_payload = load_policy_config(policy_source)
            if (
                policy_payload.get("selection_role") != "policy_validation_queries"
                or not bool(
                    policy_payload.get("selection_metrics", {}).get(
                        "hard_gate_pass", False
                    )
                )
            ):
                raise RuntimeError(
                    f"v4 policy is not a passed validation-only selection: {robot}"
                )
            frozen_policy = {
                **policy_payload,
                "backend": BACKEND,
                "source_candidate_path": str(candidate_path),
                "source_candidate_sha256": _sha256_file(candidate_path),
                "source_policy_path": str(policy_source),
                "source_policy_sha256": _sha256_file(policy_source),
                "test_data_loaded": False,
            }
            _write_json(destination / "v4_policy.json", frozen_policy)
            runtime_spec = {
                **metadata,
                "protocol": PROTOCOL,
                "robot": robot,
                "backend": BACKEND,
                "gate_device": "cpu",
                "seed_device": "cuda:0",
                "thread_configuration": {
                    "intra_op_threads": int(runtime["intra_op_threads"]),
                    "inter_op_threads": int(runtime["inter_op_threads"]),
                },
                "policy_config": policy_config.__dict__,
                "ood_before_command_reject": True,
                "defer_entry": "complete_fixed_robust_cascade_from_easy",
                "command_reject_numerical_solver_budget": 0,
                "raw_probability_logging": "V4Decision; never CalibratedRisk",
            }
            _write_json(destination / "v4_runtime_spec.json", runtime_spec)
            features, feature_sources = _load_policy_validation_features(bulk_root, robot)
            module = load_exact_v4_predictor(module_path)
            numeric = _numerical_equivalence(
                candidate_path=candidate_path,
                module=module,
                features=features,
                policy_config=policy_config,
                tolerance=config["equivalence"],
            )
            selected_counts = {
                str(key): int(value)
                for key, value in policy_payload["selection_metrics"][
                    "route_counts"
                ].items()
            }
            numeric["policy_artifact_route_counts"] = selected_counts
            numeric["policy_artifact_route_counts_agreement"] = (
                numeric["exact_route_counts"] == selected_counts
            )
            numeric["pass"] = bool(
                numeric["pass"]
                and numeric["policy_artifact_route_counts_agreement"]
            )
            robot_payload[robot] = {
                "robot": robot,
                "candidate_path": str(candidate_path),
                "candidate_sha256": _sha256_file(candidate_path),
                "policy_validation_sources": feature_sources,
                "numerical_equivalence": numeric,
                "artifacts": {
                    name: {
                        "path": str(destination / name),
                        "sha256": _sha256_file(destination / name),
                        "size": (destination / name).stat().st_size,
                    }
                    for name in (
                        "exact_v4_predictor.ts",
                        "v4_policy.json",
                        "v4_runtime_spec.json",
                    )
                },
            }
            if not numeric["pass"]:
                raise RuntimeError(f"exact v4 numerical/route equivalence failed: {robot}")

        for robot, seed in combinations:
            print(f"[release-v4] validation equivalence {robot}/seed{seed}", flush=True)
            result = _runtime_validation(
                workspace=workspace,
                source_config=source_config,
                release_v3_root=release_v3_root,
                robot=robot,
                seed=seed,
                candidate_path=candidate_root
                / "models"
                / f"{robot}_seed17_predictor.pt",
                exact_module_path=staging / robot / "exact_v4_predictor.ts",
                policy_path=candidate_root
                / "policies"
                / f"{robot}_seed17_policy.json",
                config=config,
                smoke=smoke,
            )
            runtime_payload[f"{robot}/seed{seed}"] = result
            if not result["pass"]:
                raise RuntimeError(f"v4 runtime equivalence failed: {robot}/seed{seed}")

        all_pass = all(
            value["numerical_equivalence"]["pass"] for value in robot_payload.values()
        ) and all(value["pass"] for value in runtime_payload.values())
        equivalence = {
            "protocol": PROTOCOL,
            "created_utc": _utc(),
            "backend": BACKEND,
            "reference": "disk_loaded_release_v4_candidate_eager_validation_only",
            "deployment_artifact_loaded_from_disk": True,
            "torchscript_retraced_during_equivalence": False,
            "test_named_dataset_loaded": False,
            "test_v4_started": False,
            "tolerances": config["equivalence"],
            "environment_equivalence": environment_equivalence,
            "robots": robot_payload,
            "runtime_combinations": runtime_payload,
            "expected_runtime_combination_count": 1 if smoke else 6,
            "all_pass": all_pass,
        }
        _write_json(staging / "release_equivalence.json", equivalence)
        if not all_pass:
            raise RuntimeError("release_v4 equivalence gate failed")
        if smoke:
            assert temporary is not None
            result = {
                "smoke": True,
                "all_pass": True,
                "formal_release_written": False,
                "test_v4_started": False,
                "equivalence": equivalence,
            }
            temporary.cleanup()
            return result

        _write_json(staging / "release_environment.json", _environment(candidate_root, release_v3_root))
        shutil.copyfile(path, staging / "release_config.yaml")
        dependencies = {
            "candidate": {
                "root": str(candidate_root),
                "run_manifest_sha256": _sha256_file(candidate_root / "run_manifest.json"),
                "artifact_manifest_sha256": _sha256_file(
                    candidate_root / "artifact_manifest.json"
                ),
                "release_digest": candidate_run["release_digest"],
                "artifact_count": candidate_artifacts["file_count"],
            },
            "release_v3_locked": {
                "root": str(release_v3_root),
                "release_manifest_sha256": _sha256_file(
                    release_v3_root / "release_manifest.json"
                ),
                "release_equivalence_sha256": _sha256_file(
                    release_v3_root / "release_equivalence.json"
                ),
                "artifact_count": v3_manifest["artifact_count"],
                "artifacts": v3_manifest["artifacts"],
            },
            "test_named_dataset_loaded": False,
        }
        _write_json(staging / "upstream_dependencies.json", dependencies)
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
            "release_status": "sealed",
            "created_utc": _utc(),
            "files": files,
            "file_count": len(files),
            "all_six_validation_runtime_equivalence_pass": True,
            "test_data_loaded": False,
        }
        _write_json(staging / "artifact_manifest.json", artifact_manifest)
        release_digest = sha256(
            (
                _sha256_file(staging / "artifact_manifest.json")
                + str(candidate_run["release_digest"])
                + _sha256_file(release_v3_root / "release_manifest.json")
            ).encode("ascii")
        ).hexdigest()
        source_manifest_after = _verify_formal_source_stable(
            workspace, source_manifest
        )
        protected_after = _tree_snapshot(
            workspace / "outputs", [str(value) for value in config["protected_outputs"]]
        )
        if protected_before != protected_after:
            raise RuntimeError("a protected output changed during release_v4 packaging")
        release_manifest = {
            "protocol": PROTOCOL,
            "release_status": "sealed",
            "created_utc": _utc(),
            "release_digest": release_digest,
            "backend": BACKEND,
            "robots": list(ROBOTS),
            "upstream_training_seeds": list(UPSTREAM_SEEDS),
            "runtime_combination_count": 6,
            "all_six_validation_runtime_equivalence_pass": True,
            "all_artifact_hashes_verified_after_write": True,
            "test_named_dataset_loaded": False,
            "test_v4_started": False,
            "formal_test_authorized_or_started": False,
            "source_manifest": source_manifest,
            "source_manifest_after_validation": source_manifest_after,
            "release_source_unchanged_during_validation": True,
            "artifact_manifest_sha256": _sha256_file(
                staging / "artifact_manifest.json"
            ),
            "candidate_release_digest": candidate_run["release_digest"],
            "upstream_v3_release_manifest_sha256": _sha256_file(
                release_v3_root / "release_manifest.json"
            ),
            "protected_outputs": {
                "before": protected_before,
                "after": protected_after,
                "unchanged": True,
            },
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        for relative, metadata in artifact_manifest["files"].items():
            target = staging / relative
            if _sha256_file(target) != metadata["sha256"]:
                raise RuntimeError(f"release artifact changed after manifest: {target}")
        os.replace(staging, output_root)
        print(f"[release-v4] sealed {output_root}", flush=True)
        return release_manifest
    except BaseException:
        if staging.exists():
            _write_json(
                staging / "failure.json",
                {
                    "failed_utc": _utc(),
                    "formal_release_written": False,
                    "test_v4_started": False,
                    "partial_artifacts_preserved": not smoke,
                },
            )
        raise


def main() -> None:
    args = _parser().parse_args()
    run(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
