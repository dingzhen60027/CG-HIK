from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot
from ..data.datasets import QueryDataset, RiskDataset, TransitionDataset
from ..experiments.metrics import binary_calibration_error
from ..experiments.policy_selection import action_predictions
from ..experiments.provenance import environment_payload, source_tree_hash
from ..models.risk import ConstantRiskProvider, RiskModel
from ..models.seed import TorchSeedEnsemble, encode_seed_inputs
from ..runtime.cascade import (
    ActionGateConfig,
    CalibratedActionGate,
    CascadeConfig,
    CascadedHybridIK,
    EntryAction,
    FixedEntryGate,
)
from ..solvers.dls import AdaptiveDLS, DLSConfig
from ..solvers.fallback import KDTreeSeedBank, TRFConfig, TRFFallbackSolver
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..types import IKQuery, Pose
from .benchmark import (
    ConstantRiskEngine,
    ProfiledCascadeRuntime,
    benchmark_points,
    benchmark_trajectories,
    cached_risk_features,
    compare_backends,
    distribution_summary,
    latency_breakdown_summary,
    paired_latency_summary,
    query_from_dataset,
    warmup_runtimes,
)
from .optimized_inference import (
    EagerRiskEngine,
    EagerSeedEngine,
    ExactSingleCallSeedEnsemble,
    OptimizedSeedEngine,
    VectorizedHGBRiskModel,
    VectorizedSeedMLP,
)
from .validation import (
    method_validation_metrics,
    record_equivalence,
    risk_probability_metrics,
    stratified_point_subset,
    trajectory_validation_subset,
)


OUTPUT_FILENAMES = (
    "latency_breakdown.json",
    "paired_latency_summary.json",
    "numerical_equivalence.json",
    "validation_gate_v3.json",
    "optimization_changes.md",
    "run_manifest.json",
    "environment.json",
)

ALLOWED_DATASETS = {
    "seed_train.npz",
    "seed_validation.npz",
    "risk_train.npz",
    "risk_train_queries.npz",
    "risk_validation.npz",
    "risk_validation_queries.npz",
    "calibration.npz",
    "calibration_queries.npz",
    "policy_validation.npz",
    "policy_validation_queries.npz",
}

ALLOWED_MODELS = {
    "seed_ensemble.pt",
    "risk_model.joblib",
    "seed_bank.npz",
    "solver_metadata.json",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only latency pilot for frozen paper-v2 models"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced in-memory validation check; never writes output files.",
    )
    return parser


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("latency pilot config must be a mapping")
    return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _frozen_snapshot(output_root: Path) -> dict[str, Any]:
    directories = sorted(
        path
        for path in output_root.glob("paper_v2_*")
        if path.is_dir()
    )
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for directory in directories:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            metadata = path.stat()
            relative = str(path.relative_to(output_root.parent))
            entries[relative] = {
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
            total_bytes += metadata.st_size
    return {
        "directories": [str(path.relative_to(output_root.parent)) for path in directories],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_digest": _json_digest(entries),
        "entries": entries,
    }


def _assert_allowed(path: Path, source_root: Path, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{role} escapes the frozen seed-17 source root: {resolved}") from error
    if "test" in resolved.name.lower():
        raise RuntimeError(f"validation-only pilot refuses test-named input: {resolved.name}")
    allowed = ALLOWED_DATASETS | ALLOWED_MODELS | {"policy_selection_v2.json"}
    if resolved.name not in allowed:
        raise RuntimeError(f"input is not on the latency-pilot allowlist: {resolved.name}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _robot_paths(source_root: Path, robot: str) -> dict[str, Path]:
    robot_root = source_root / robot
    raw = {
        "seed_train": robot_root / "datasets" / "seed_train.npz",
        "seed_validation": robot_root / "datasets" / "seed_validation.npz",
        "risk_train": robot_root / "datasets" / "risk_train.npz",
        "risk_train_queries": robot_root / "datasets" / "risk_train_queries.npz",
        "risk_validation": robot_root / "datasets" / "risk_validation.npz",
        "risk_validation_queries": robot_root / "datasets" / "risk_validation_queries.npz",
        "calibration": robot_root / "datasets" / "calibration.npz",
        "calibration_queries": robot_root / "datasets" / "calibration_queries.npz",
        "policy_validation": robot_root / "datasets" / "policy_validation.npz",
        "policy_validation_queries": robot_root / "datasets" / "policy_validation_queries.npz",
        "seed_model": robot_root / "models" / "seed_ensemble.pt",
        "risk_model": robot_root / "models" / "risk_model.joblib",
        "seed_bank": robot_root / "models" / "seed_bank.npz",
        "solver_metadata": robot_root / "models" / "solver_metadata.json",
        "policy_selection": robot_root / "results" / "policy_selection_v2.json",
    }
    return {role: _assert_allowed(path, source_root, role) for role, path in raw.items()}


def _row_bytes(*arrays: np.ndarray) -> set[bytes]:
    matrices = [np.asarray(array).reshape(len(array), -1) for array in arrays]
    numeric = np.ascontiguousarray(np.concatenate(matrices, axis=1), dtype=np.float64)
    width = numeric.dtype.itemsize * numeric.shape[1]
    return set(numeric.view(np.dtype((np.void, width))).reshape(-1).tolist())


def _split_audit(paths: dict[str, Path]) -> dict[str, Any]:
    seed_train = TransitionDataset.load(paths["seed_train"])
    seed_validation = TransitionDataset.load(paths["seed_validation"])
    train_rows = _row_bytes(
        seed_train.previous_q,
        seed_train.target_q,
        seed_train.target_position,
        seed_train.target_rotation,
    )
    validation_rows = _row_bytes(
        seed_validation.previous_q,
        seed_validation.target_q,
        seed_validation.target_position,
        seed_validation.target_rotation,
    )
    query_roles = (
        "risk_train_queries",
        "risk_validation_queries",
        "calibration_queries",
        "policy_validation_queries",
    )
    query_sets: dict[str, set[bytes]] = {}
    query_ids: dict[str, set[int]] = {}
    query_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    for role in query_roles:
        dataset = QueryDataset.load(paths[role])
        rows = _row_bytes(dataset.previous_q, dataset.target_position, dataset.target_rotation)
        query_sets[role] = rows
        query_ids[role] = set(dataset.trajectory_id.astype(int).tolist())
        query_counts[role] = len(dataset)
        duplicate_counts[role] = len(dataset) - len(rows)
    overlaps: dict[str, int] = {}
    id_overlaps: dict[str, int] = {}
    for first_index, first in enumerate(query_roles):
        for second in query_roles[first_index + 1 :]:
            key = f"{first}__{second}"
            overlaps[key] = len(query_sets[first] & query_sets[second])
            id_overlaps[key] = len(query_ids[first] & query_ids[second])
    paired_row_counts = {
        split: {
            "risk_rows": len(RiskDataset.load(paths[split])),
            "query_rows": len(QueryDataset.load(paths[f"{split}_queries"])),
        }
        for split in ("risk_train", "risk_validation", "calibration", "policy_validation")
    }
    paired_row_counts_match = all(
        counts["risk_rows"] == counts["query_rows"] for counts in paired_row_counts.values()
    )
    return {
        "seed_train_rows": len(seed_train),
        "seed_validation_rows": len(seed_validation),
        "seed_train_validation_exact_overlap": len(train_rows & validation_rows),
        "query_counts": query_counts,
        "within_split_exact_duplicate_counts": duplicate_counts,
        "cross_split_exact_query_overlap_counts": overlaps,
        "cross_split_local_trajectory_id_overlap_counts": id_overlaps,
        "risk_query_paired_row_counts": paired_row_counts,
        "risk_query_paired_row_counts_match": paired_row_counts_match,
        "grouping_key": ["robot", "training_seed", "split_name", "trajectory_id"],
        "trajectory_id_scope_warning": "trajectory_id is split-local and must not be joined alone",
        "passed": bool(
            len(train_rows & validation_rows) == 0
            and all(value == 0 for value in duplicate_counts.values())
            and all(value == 0 for value in overlaps.values())
            and paired_row_counts_match
        ),
    }


def _load_gate_config(path: Path) -> tuple[ActionGateConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    learned = payload["learned_gate"]
    if learned.get("selection_split") != "policy_validation":
        raise RuntimeError("frozen action gate was not selected on policy_validation")
    selected = learned["selected_config"]
    exact_keys = {"easy_probability", "hard_probability", "reject_probability"}
    if set(selected) != exact_keys:
        raise RuntimeError("frozen action gate has an unexpected threshold schema")
    config = ActionGateConfig(**{key: float(value) for key, value in selected.items()})
    return config, {
        "selection_split": learned["selection_split"],
        "selected_config": asdict(config),
        "source_file": str(path),
        "consumed_json_keys": [
            "learned_gate.selection_split",
            "learned_gate.selected_config",
        ],
        "test_metrics_consumed": False,
    }


def _solver_components(
    source_config: dict[str, Any],
    paths: dict[str, Path],
    kinematics: object,
) -> tuple[AdaptiveDLS, SolutionVerifier, TRFFallbackSolver, KDTreeSeedBank, CascadeConfig]:
    solver_values = dict(source_config.get("solver", {}))
    metadata = json.loads(paths["solver_metadata"].read_text(encoding="utf-8"))
    solver_values["sigma_threshold"] = metadata["sigma_threshold"]
    dls_allowed = DLSConfig.__dataclass_fields__.keys()
    dls = AdaptiveDLS(
        kinematics,  # type: ignore[arg-type]
        DLSConfig(**{key: value for key, value in solver_values.items() if key in dls_allowed}),
    )
    verifier = SolutionVerifier(
        kinematics,  # type: ignore[arg-type]
        VerifierConfig(**dict(source_config.get("verifier", {}))),
    )
    fallback_values = dict(source_config.get("fallback", {}))
    fallback_allowed = TRFConfig.__dataclass_fields__.keys()
    fallback = TRFFallbackSolver(
        kinematics,  # type: ignore[arg-type]
        TRFConfig(
            **{key: value for key, value in fallback_values.items() if key in fallback_allowed}
        ),
    )
    with np.load(paths["seed_bank"]) as data:
        seed_bank_joints = np.asarray(data["joints"], dtype=np.float64)
    seed_bank = KDTreeSeedBank(kinematics).fit(seed_bank_joints)  # type: ignore[arg-type]
    cascade_values = dict(source_config.get("cascade", {}))
    cascade_allowed = CascadeConfig.__dataclass_fields__.keys()
    cascade = CascadeConfig(
        **{key: value for key, value in cascade_values.items() if key in cascade_allowed}
    )
    return dls, verifier, fallback, seed_bank, cascade


def _thread_probe(
    *,
    python: str,
    source_config_path: Path,
    source_root: Path,
    robots: list[str],
    candidates: list[list[int]],
    warmup: int,
    repeats: int,
    workspace: Path,
) -> tuple[tuple[int, int], dict[str, Any]]:
    reports: dict[str, Any] = {}
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    for candidate in candidates:
        intra, inter = map(int, candidate)
        key = f"intra_{intra}_inter_{inter}"
        reports[key] = {}
        for robot in robots:
            command = [
                python,
                "-m",
                "confik.latency_pilot_v3.thread_probe",
                "--config",
                str(source_config_path),
                "--source-root",
                str(source_root),
                "--robot",
                robot,
                "--intra",
                str(intra),
                "--inter",
                str(inter),
                "--warmup",
                str(warmup),
                "--repeats",
                str(repeats),
            ]
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            reports[key][robot] = json.loads(completed.stdout.strip().splitlines()[-1])
    def score(item: tuple[str, Any]) -> tuple[float, int]:
        key, report = item
        worst = max(
            float(robot_report["backends_ms"]["optimized_pytorch"]["p95"])
            for robot_report in report.values()
        )
        intra = int(key.split("_")[1])
        inter = int(key.split("_")[3])
        return worst, intra * inter
    selected_key, _ = min(reports.items(), key=score)
    parts = selected_key.split("_")
    selected = (int(parts[1]), int(parts[3]))
    return selected, {
        "selection_metric": "minimum worst two-robot P95 for the optimized full ensemble",
        "selected": {"intra_op_threads": selected[0], "inter_op_threads": selected[1]},
        "candidates": reports,
        "single_member_used_for_selection": False,
    }


def _module_batch_predictions(
    module: torch.nn.Module,
    features: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(features[start : start + batch_size])
            ).to(device)
            chunks.append(module(tensor).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def _risk_microbenchmark(
    eager: EagerRiskEngine,
    optimized: VectorizedHGBRiskModel,
    features: np.ndarray,
    *,
    warmup: int = 200,
    repeats: int = 1500,
) -> dict[str, Any]:
    rows = np.asarray(features, dtype=np.float64)
    for index in range(warmup):
        row = rows[index % len(rows)]
        eager.predict(row)
        optimized.predict(row)
    samples = {"eager_sklearn": [], "vectorized_frozen_hgb": []}
    for repeat in range(min(repeats, len(rows))):
        row = rows[repeat]
        order = (
            (("vectorized_frozen_hgb", optimized), ("eager_sklearn", eager))
            if repeat % 2
            else (("eager_sklearn", eager), ("vectorized_frozen_hgb", optimized))
        )
        for name, engine in order:
            started = perf_counter_ns()
            engine.predict(row)
            samples[name].append((perf_counter_ns() - started) / 1e6)
    payload = {name: distribution_summary(values) for name, values in samples.items()}
    payload["p95_savings_ms"] = float(payload["eager_sklearn"]["p95"]) - float(
        payload["vectorized_frozen_hgb"]["p95"]
    )
    return payload


def _feature_cache_microbenchmark(
    engine: OptimizedSeedEngine,
    dataset: QueryDataset,
    *,
    samples: int = 300,
) -> dict[str, Any]:
    uncached: list[float] = []
    cached: list[float] = []
    max_error = 0.0
    count = min(samples, len(dataset))
    for index in range(count):
        query = query_from_dataset(dataset, index)
        prepared = engine.prepare(query)
        order = (False, True) if index % 2 == 0 else (True, False)
        outputs: dict[bool, np.ndarray] = {}
        for reuse in order:
            started = perf_counter_ns()
            outputs[reuse] = cached_risk_features(query, prepared, reuse_best_pose=reuse)
            elapsed = (perf_counter_ns() - started) / 1e6
            (cached if reuse else uncached).append(elapsed)
        max_error = max(max_error, float(np.max(np.abs(outputs[False] - outputs[True]))))
    uncached_summary = distribution_summary(uncached)
    cached_summary = distribution_summary(cached)
    return {
        "uncached_ms": uncached_summary,
        "cached_ms": cached_summary,
        "p95_savings_ms": float(uncached_summary["p95"]) - float(cached_summary["p95"]),
        "risk_feature_max_abs_error": max_error,
    }


def _single_member_diagnostic(
    ensemble: TorchSeedEnsemble,
    features: np.ndarray,
    *,
    warmup: int = 200,
    repeats: int = 1000,
) -> dict[str, Any]:
    member = ensemble.members[0].eval()
    values: list[float] = []
    with torch.inference_mode():
        for index in range(warmup):
            member(torch.from_numpy(features[index % len(features) : index % len(features) + 1]))
        for repeat in range(min(repeats, len(features))):
            tensor = torch.from_numpy(features[repeat : repeat + 1])
            started = perf_counter_ns()
            member(tensor)
            values.append((perf_counter_ns() - started) / 1e6)
    return {
        "eligible_for_backend_selection": False,
        "reason": "diagnostic only; v3 retains all five frozen ensemble members",
        "member_index": 0,
        "forward_ms": distribution_summary(values),
    }


def _build_runtimes(
    *,
    kinematics: object,
    gate_config: ActionGateConfig,
    engines: dict[str, tuple[object, object, bool]],
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    fallback: TRFFallbackSolver,
    seed_bank: KDTreeSeedBank,
    cascade: CascadeConfig,
) -> dict[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]]:
    runtimes: dict[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]] = {}
    for backend, (seed_engine, risk_engine, reuse) in engines.items():
        common = {
            "kinematics": kinematics,
            "seed_engine": seed_engine,
            "dls": dls,
            "verifier": verifier,
            "seed_bank": seed_bank,
            "fallback": fallback,
            "cascade_config": cascade,
            "reuse_candidate_features": reuse,
        }
        baseline = ProfiledCascadeRuntime(
            name="baseline",
            risk_engine=ConstantRiskEngine(),
            gate=FixedEntryGate(EntryAction.EASY),
            **common,  # type: ignore[arg-type]
        )
        proposed = ProfiledCascadeRuntime(
            name="proposed",
            risk_engine=risk_engine,  # type: ignore[arg-type]
            gate=CalibratedActionGate(gate_config),
            **common,  # type: ignore[arg-type]
        )
        runtimes[backend] = (baseline, proposed)
    return runtimes


def _production_wrapper_equivalence(
    *,
    kinematics: object,
    ensemble: TorchSeedEnsemble,
    risk_model: RiskModel,
    gate_config: ActionGateConfig,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    fallback: TRFFallbackSolver,
    seed_bank: KDTreeSeedBank,
    cascade: CascadeConfig,
    eager_runtimes: tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime],
    dataset: QueryDataset,
    dt: float,
) -> dict[str, Any]:
    """Verify the timing shell against the frozen production solve method."""

    production = (
        CascadedHybridIK(
            kinematics,  # type: ignore[arg-type]
            ensemble,
            ConstantRiskProvider(np.array([1.0, 0.0, 0.0, 0.0])),
            dls,
            verifier,
            gate=FixedEntryGate(EntryAction.EASY),
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade,
        ),
        CascadedHybridIK(
            kinematics,  # type: ignore[arg-type]
            ensemble,
            risk_model,
            dls,
            verifier,
            gate=CalibratedActionGate(gate_config),
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade,
        ),
    )
    indices: list[int] = []
    for category in sorted(np.unique(dataset.category)):
        indices.extend(np.flatnonzero(dataset.category == category)[:2].astype(int).tolist())
    accepted_matches = 0
    route_matches = 0
    fev_matches = 0
    fallback_matches = 0
    verification_matches = 0
    command_pairs = 0
    max_command_error = 0.0
    comparisons = 0
    for query_index in indices:
        query = query_from_dataset(dataset, query_index, dt=dt)
        for original_runtime, profiled_runtime in zip(production, eager_runtimes, strict=True):
            original = original_runtime.solve(query)
            profiled = profiled_runtime.solve(query)
            comparisons += 1
            accepted_matches += int(bool(original.accepted) == bool(profiled.accepted))
            route_matches += int(
                str(original.metadata["entry_action"]) == profiled.entry_action
            )
            fev_matches += int(
                sum(trace.function_evaluations for trace in original.traces)
                == profiled.function_evaluations
            )
            fallback_matches += int(bool(original.fallback_used) == bool(profiled.fallback_used))
            original_reasons = (
                tuple(original.verification.reasons)
                if original.verification is not None
                else ()
            )
            verification_matches += int(original_reasons == profiled.verification_reasons)
            if original.q is not None and profiled.q is not None:
                command_pairs += 1
                max_command_error = max(
                    max_command_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(original.q, dtype=np.float64)
                                - np.asarray(profiled.q, dtype=np.float64)
                            )
                        )
                    ),
                )
    payload = {
        "sampling": "first two records from each category in the preregistered point subset",
        "comparison_count": comparisons,
        "accepted_agreement": accepted_matches / comparisons if comparisons else float("nan"),
        "routing_action_agreement": route_matches / comparisons if comparisons else float("nan"),
        "function_evaluations_agreement": fev_matches / comparisons if comparisons else float("nan"),
        "fallback_agreement": fallback_matches / comparisons if comparisons else float("nan"),
        "verification_reasons_agreement": verification_matches / comparisons
        if comparisons
        else float("nan"),
        "accepted_command_pair_count": command_pairs,
        "accepted_command_max_abs_error_rad": max_command_error,
    }
    payload["pass"] = bool(
        comparisons > 0
        and payload["accepted_agreement"] == 1.0
        and payload["routing_action_agreement"] == 1.0
        and payload["function_evaluations_agreement"] == 1.0
        and payload["fallback_agreement"] == 1.0
        and payload["verification_reasons_agreement"] == 1.0
        and max_command_error <= 1e-12
    )
    return payload


def _metric_delta(candidate: dict[str, Any], reference: dict[str, Any], key: str) -> float:
    return float(candidate[key]) - float(reference[key])


def _risk_metric_deltas(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    """Return explicit derived-metric deltas for the frozen risk model."""

    deltas = {
        key: _metric_delta(candidate, reference, key)
        for key in ("fail_auroc", "fail_ece", "multiclass_ece", "argmax_macro_f1")
    }
    candidate_policy = candidate["action_policy"]
    reference_policy = reference["action_policy"]
    deltas.update(
        {
            f"action_policy.{key}": _metric_delta(candidate_policy, reference_policy, key)
            for key in ("reject_recall", "false_reject_rate", "nonreject_macro_f1")
        }
    )
    return deltas


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    safe = _json_safe(payload)
    path.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report_markdown(
    *,
    selected_backend: str | None,
    gate_payload: dict[str, Any],
    paired_payload: dict[str, Any],
    equivalence: dict[str, Any],
    microbenchmarks: dict[str, Any],
    thread_report: dict[str, Any],
) -> str:
    selected_label = selected_backend or "无（等价门失败）"
    lines = [
        "# latency_pilot_v3 实现级优化报告",
        "",
        "## 技术摘要",
        "",
        f"本轮仅使用 training/validation 数据，锁定候选后端为 `{selected_label}`。"
        f"双机器人 validation 总门结果为 `{gate_payload['all_robots_pass']}`；未启动任何 test_v3。",
        "",
        "## 最大固定开销与逐项收益",
        "",
        "所有收益均来自 validation-only 同口径计时；正数表示降低延迟。共享同一调用路径的阶段净差不可直接相加。",
        "",
        "| 修改内容 | 为什么降低延迟 | 改变算法语义 | Validation 收益 | 数值等价性 |",
        "|---|---|---|---|---|",
        "| `model.eval()` + `torch.inference_mode()` | 关闭训练态与 autograd bookkeeping | 否；v2 已使用，v3 显式强制 | 已有保障，新增收益按 0 ms 计 | 全模型输出检查 |",
    ]
    for robot in sorted(microbenchmarks):
        item = microbenchmarks[robot]
        thread_selected = thread_report["selected"]
        selected_thread_key = (
            f"intra_{thread_selected['intra_op_threads']}_"
            f"inter_{thread_selected['inter_op_threads']}"
        )
        # Thread selection is intentionally measured on the full optimized
        # CPU PyTorch ensemble; the exact export backend executes on CUDA.
        deployment_backend = "optimized_pytorch"
        selected_thread_p95 = float(
            thread_report["candidates"][selected_thread_key][robot]["backends_ms"]
            [deployment_backend]["p95"]
        )
        legacy_thread_p95 = float(
            thread_report["candidates"]["intra_1_inter_16"][robot]["backends_ms"]
            [deployment_backend]["p95"]
        )
        lines.extend(
            [
                f"| {robot}: 冻结 HGB 树向量化 + 原 isotonic | 消除 sklearn batch-one 逐树/逐校准器调度 | 否 | P95 节省 {item['risk']['p95_savings_ms']:.3f} ms | risk max error {equivalence['robots'][robot]['risk_probability_max_abs_error']:.3e} |",
                f"| {robot}: 原 CUDA 成员循环改为 CPU 五成员堆叠前向 | 各层用一次 batched matmul，移除成员循环与逐成员同步 | 数学语义否；若 FEV 等价门失败则不锁定 | learned-seed 阶段组合净节省 {item['stage_savings']['vectorized_seed_p95_ms']:.3f} ms | seed max error {equivalence['robots'][robot]['optimized_pytorch']['seed_max_abs_error']:.3e} |",
                f"| {robot}: 连续 float32 + 预分配 CPU 输入 + 缓存归一化 | 避免逐查询 tensor 构造、H2D/D2H 与不必要复制 | 否 | conversion 阶段组合净节省 {item['stage_savings']['conversion_p95_ms']:.3f} ms | 路由/solver 门见下节 |",
                f"| {robot}: 复用候选 FK、裕量与 joint-step | 删除 candidate scoring 与 risk feature 的重复几何计算 | 否 | P95 节省 {item['feature_cache']['p95_savings_ms']:.3f} ms | feature max error {item['feature_cache']['risk_feature_max_abs_error']:.3e} |",
                f"| {robot}: 精确 single-call TorchScript 图 | 在 warmup 前将五成员调用展开为无 Python 成员循环的 CUDA 推理图 | 否 | 相对 eager member-loop 的 seed P95 净节省 {item['stage_savings']['exact_export_seed_p95_ms']:.3f} ms | seed max error {equivalence['robots'][robot]['torchscript_exact']['seed_max_abs_error']:.3e} |",
                f"| {robot}: 日志对象与 JSON 序列化移出 core | 不让记录开销污染 solver E2E | 否 | core 外 logging P95 {item['logging_p95_ms']:.3f} ms | 不影响结果字段 |",
                f"| {robot}: CPU 线程固定为 intra={thread_selected['intra_op_threads']}, inter={thread_selected['inter_op_threads']} | batch-one 小模型避免过度线程调度 | 否 | 相对 v2 的 intra=1/inter=16 seed-forward P95 节省 {legacy_thread_p95 - selected_thread_p95:.3f} ms | single-member 未参与选择 |",
                f"| {robot}: single-member 诊断（不参与选择） | 仅量化移除 ensemble 的理论开销下界 | 是，故禁止选用 | 单成员 forward P95 {item['single_member']['forward_ms']['p95']:.3f} ms | `eligible_for_backend_selection=false` |",
                f"| {robot}: 最大阶段诊断 | 分离真正的固定推理开销与数值求解尾部 | 否 | eager 最大 P95 为 {item['largest_stage_eager']['stage']}={item['largest_stage_eager']['p95_ms']:.3f} ms；锁定后端最大 P95 为 {item['largest_stage_selected']['stage']}={item['largest_stage_selected']['p95_ms']:.3f} ms | 同一批 validation 查询 |",
            ]
        )
    lines.extend(
        [
            "",
            "## 数据与计时边界",
            "",
            "- 主 checkpoint 固定为预注册 seed 17；未从三次正式 test 结果中挑选训练 seed。",
            "- 点查询来自 `risk_validation_queries` 的类别分层子集，风险指标使用完整 `risk_validation`/`policy_validation`，轨迹来自 `seed_validation`。",
            "- 输入 allowlist 在读取前拒绝任何文件名含 `test` 的 dataset；正式输出运行前后执行 SHA-256、size、mtime 和 mode 快照。",
            "- `perf_counter_ns` 记录九个指定阶段；点查询对 baseline/proposed 严格 AB/BA 交错。核心 E2E 在 verification 后停止，随后仅在内存中计时日志对象和 JSON 序列化，所有查询完成后才写盘。",
            "",
            "## 数值等价性与 Validation 门",
            "",
            f"整体数值等价门：`{equivalence['all_pass']}`。整体 validation gate：`{gate_payload['all_robots_pass']}`。",
        ]
    )
    for robot, robot_gate in gate_payload["robots"].items():
        paired = paired_payload["summaries"][f"{robot}/{selected_backend}/point_feasible"] if selected_backend else None
        if paired is not None:
            lines.append(
                f"- {robot}: feasible P95 ratio={paired['p95_ratio_proposed_over_baseline']:.3f}, "
                f"相对 eager proposed P95 降低={paired_payload['eager_comparison'][robot]['p95_reduction']:.1%}, "
                f"success/rejection/risk/trajectory/latency={robot_gate['success_gate_pass']}/"
                f"{robot_gate['rejection_gate_pass']}/{robot_gate['risk_gate_pass']}/"
                f"{robot_gate['trajectory_gate_pass']}/{robot_gate['latency_gate_pass']}。"
            )
    lines.extend(
        [
            "",
            "## 限制与稳健性说明",
            "",
            "- 这是 validation pilot，不是新的独立 test；P99 仍受 pilot 样本量与主机调度噪声影响。",
            "- CPU 向量化 PyTorch 后端保留为性能/数值诊断；若其微小舍入差改变 FEV，则不会被锁定。精确导出后端使用未 freeze 的 traced TorchScript，以保留原 CUDA 成员算术顺序。",
            "- v2 协议中不同训练 seed 的 hard-valid 查询实际会受各 seed 的 sigma threshold 影响；本轮将 seed17 checkpoint 与 seed17 validation 严格配对，不修订冻结 v2。",
            "- 这里只验证精确 URDF 运动学，不扩展为 Isaac Lab 物理、碰撞或硬件实时结论。",
            "",
            "## 推荐下一步",
            "",
            "只有当两个机器人全部门通过且正式输出快照不变时，才具备锁定 v3 实现的条件。即使满足，本轮也不会生成或运行 test_v3；新 test 集必须在下一轮独立版本中创建并一次性冻结。",
            "",
            "## 尚待回答的问题",
            "",
            "- 在不同 CPU 微架构上，所选线程数与 `torch.export` 后端是否仍保持相同排序？",
            "- 新的独立 test_v3 是否能复现 validation 中的 P95 比率与 30% 降幅？",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    v3_config = _read_yaml(config_path)
    workspace = Path(__file__).resolve().parents[3]
    declared_output = Path(
        os.path.abspath(config_path.parent / v3_config["output_directory"])
    )
    expected_output = Path(
        os.path.abspath(workspace / "outputs" / "latency_pilot_v3")
    )
    if declared_output != expected_output:
        raise RuntimeError(f"v3 output must resolve exactly to {expected_output}")
    if declared_output.is_symlink():
        raise RuntimeError("v3 output directory must not be a symbolic link")
    output_dir = declared_output
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError("v3 output path exists but is not a directory")
    for filename in OUTPUT_FILENAMES:
        if (output_dir / filename).is_symlink():
            raise RuntimeError(f"v3 output file must not be a symbolic link: {filename}")
    source_config_path = (config_path.parent / v3_config["source_config"]).resolve()
    source_config = load_config(source_config_path)
    source_root = (workspace / "outputs" / str(v3_config["source_experiment"])).resolve()
    if source_root.name != "paper_v2_seed17" or int(v3_config["training_seed"]) != 17:
        raise RuntimeError("latency pilot is preregistered to the seed-17 validation run")
    robots = [str(robot) for robot in v3_config["robots"]]
    if robots != ["ur5e", "panda"]:
        raise RuntimeError("latency pilot must evaluate both locked robots in order")

    print("[v3] hashing frozen paper_v2 outputs before any benchmark", flush=True)
    frozen_before = _frozen_snapshot(workspace / "outputs")
    if frozen_before["file_count"] == 0:
        raise RuntimeError("no frozen paper_v2 outputs were found")
    paths_by_robot = {robot: _robot_paths(source_root, robot) for robot in robots}
    split_audits: dict[str, Any] = {}
    input_manifest: dict[str, Any] = {}
    for robot in robots:
        print(f"[v3] auditing train/validation boundaries for {robot}", flush=True)
        split_audits[robot] = _split_audit(paths_by_robot[robot])
        input_manifest[robot] = {
            role: {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size}
            for role, path in paths_by_robot[robot].items()
        }
    if not all(audit["passed"] for audit in split_audits.values()):
        raise RuntimeError("training/validation split audit failed")

    validation_cfg = dict(v3_config["validation"])
    thread_cfg = dict(v3_config["thread_probe"])
    if args.smoke:
        validation_cfg.update(
            {
                "point_queries_per_category": 5,
                "trajectory_count": 1,
                "warmup_iterations": 5,
                "point_timing_repeats": 1,
            }
        )
        thread_cfg.update(
            {
                "candidates": [[1, 1], [2, 1]],
                "warmup_iterations": 20,
                "timing_repeats": 80,
            }
        )

    print("[v3] probing CPU intra/inter-op thread counts in isolated subprocesses", flush=True)
    selected_threads, thread_report = _thread_probe(
        python=sys.executable,
        source_config_path=source_config_path,
        source_root=source_root,
        robots=robots,
        candidates=thread_cfg["candidates"],
        warmup=int(thread_cfg["warmup_iterations"]),
        repeats=int(thread_cfg["timing_repeats"]),
        workspace=workspace,
    )
    torch.set_num_threads(selected_threads[0])
    torch.set_num_interop_threads(selected_threads[1])

    all_records: list[dict[str, object]] = []
    equivalence_robots: dict[str, Any] = {}
    validation_metrics: dict[str, Any] = {}
    risk_validation_reports: dict[str, Any] = {}
    microbenchmarks: dict[str, Any] = {}
    selection_inputs: dict[str, Any] = {}
    for robot in robots:
        print(f"[v3] loading frozen seed17 validation artifacts for {robot}", flush=True)
        paths = paths_by_robot[robot]
        kinematics = load_robot(source_config, robot)
        gate_config, gate_source = _load_gate_config(paths["policy_selection"])
        risk_model = RiskModel.load(paths["risk_model"])
        vector_risk = VectorizedHGBRiskModel(risk_model)
        eager_risk = EagerRiskEngine(risk_model)
        eager_ensemble = TorchSeedEnsemble.load(
            paths["seed_model"],
            kinematics,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        cpu_ensemble = TorchSeedEnsemble.load(paths["seed_model"], kinematics, device="cpu")
        cpu_ensemble.members.eval()
        vector_module = VectorizedSeedMLP.from_ensemble(cpu_ensemble, device="cpu").eval()
        exact_graph_source = ExactSingleCallSeedEnsemble(eager_ensemble).eval()
        exact_example = torch.empty(
            (1, kinematics.nq + 9), dtype=torch.float32, device=eager_ensemble.device
        )
        exported_module = torch.jit.trace(
            exact_graph_source, exact_example, strict=True
        ).eval()
        eager_seed = EagerSeedEngine(eager_ensemble)
        optimized_seed = OptimizedSeedEngine(
            cpu_ensemble, vector_module, name="optimized_pytorch", device="cpu"
        )
        exported_seed = OptimizedSeedEngine(
            eager_ensemble,
            exported_module,
            name="torchscript_exact",
            device=eager_ensemble.device,
        )

        seed_validation = TransitionDataset.load(paths["seed_validation"])
        encoded = np.ascontiguousarray(
            encode_seed_inputs(
                kinematics,
                seed_validation.previous_q,
                seed_validation.target_position,
                seed_validation.target_rotation,
                use_history=cpu_ensemble.config.use_history,
            ).astype(np.float32)
        )
        print(f"[v3] checking full seed/risk numerical equivalence for {robot}", flush=True)
        reference_seed = eager_ensemble.predict_deltas_batch(
            seed_validation.previous_q,
            seed_validation.target_position,
            seed_validation.target_rotation,
            batch_size=1,
        )
        optimized_seed_predictions = _module_batch_predictions(
            vector_module, encoded, batch_size=1
        )
        exported_seed_predictions = _module_batch_predictions(
            exported_module, encoded, batch_size=1
        )
        optimized_seed_error = float(np.max(np.abs(reference_seed - optimized_seed_predictions)))
        exported_seed_error = float(np.max(np.abs(reference_seed - exported_seed_predictions)))

        risk_splits: dict[str, Any] = {}
        risk_max_error = 0.0
        risk_score_max_error = 0.0
        route_agreement_min = 1.0
        for split_name in ("risk_validation", "policy_validation"):
            dataset = RiskDataset.load(paths[split_name])
            eager_probabilities = risk_model.predict_proba(dataset.features)
            optimized_probabilities = vector_risk.predict_proba(dataset.features)
            probability_error = float(
                np.max(np.abs(eager_probabilities - optimized_probabilities))
            )
            score_error = float(
                np.max(
                    np.abs(
                        eager_probabilities[:, 2:].sum(axis=1)
                        - optimized_probabilities[:, 2:].sum(axis=1)
                    )
                )
            )
            eager_actions = action_predictions(eager_probabilities, gate_config)
            optimized_actions = action_predictions(optimized_probabilities, gate_config)
            agreement = float(np.mean(eager_actions == optimized_actions))
            risk_max_error = max(risk_max_error, probability_error)
            risk_score_max_error = max(risk_score_max_error, score_error)
            route_agreement_min = min(route_agreement_min, agreement)
            optimized_metrics = risk_probability_metrics(
                optimized_probabilities, dataset.labels, gate_config
            )
            eager_metrics = risk_probability_metrics(
                eager_probabilities, dataset.labels, gate_config
            )
            risk_splits[split_name] = {
                "metrics": optimized_metrics,
                "eager_metrics": eager_metrics,
                "metric_deltas_vs_eager": _risk_metric_deltas(
                    optimized_metrics, eager_metrics
                ),
                "risk_probability_max_abs_error": probability_error,
                "risk_score_max_abs_error": score_error,
                "routing_action_agreement": agreement,
                "routing_action_mismatch_count": int(np.sum(eager_actions != optimized_actions)),
            }
        risk_validation_reports[robot] = risk_splits

        point_source = QueryDataset.load(paths["risk_validation_queries"])
        point_dataset, point_indices = stratified_point_subset(
            point_source,
            per_category=int(validation_cfg["point_queries_per_category"]),
            seed=int(validation_cfg["point_sampling_seed"]),
        )
        trajectory_dataset, trajectory_ids = trajectory_validation_subset(
            seed_validation,
            trajectory_count=int(validation_cfg["trajectory_count"]),
            seed=int(validation_cfg["trajectory_sampling_seed"]),
        )
        dls, verifier, fallback, seed_bank, cascade = _solver_components(
            source_config, paths, kinematics
        )
        engines = {
            "eager_original": (eager_seed, eager_risk, False),
            "optimized_pytorch": (optimized_seed, vector_risk, True),
            "torchscript_exact": (exported_seed, vector_risk, True),
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
        production_wrapper_check = _production_wrapper_equivalence(
            kinematics=kinematics,
            ensemble=eager_ensemble,
            risk_model=risk_model,
            gate_config=gate_config,
            dls=dls,
            verifier=verifier,
            fallback=fallback,
            seed_bank=seed_bank,
            cascade=cascade,
            eager_runtimes=runtimes["eager_original"],
            dataset=point_dataset,
            dt=float(source_config["data"].get("dt", 0.02)),
        )
        print(f"[v3] warmup and paired AB/BA point benchmark for {robot}", flush=True)
        warmup_runtimes(
            runtimes,
            point_dataset,
            iterations=int(validation_cfg["warmup_iterations"]),
            dt=float(source_config["data"].get("dt", 0.02)),
        )
        point_records, order_counts = benchmark_points(
            robot,
            runtimes,
            point_dataset,
            repeats=int(validation_cfg["point_timing_repeats"]),
            dt=float(source_config["data"].get("dt", 0.02)),
            order_seed=int(validation_cfg["method_order_seed"]),
        )
        print(f"[v3] closed-loop seed_validation trajectory benchmark for {robot}", flush=True)
        trajectory_records = benchmark_trajectories(
            robot,
            runtimes,
            trajectory_dataset,
            dt=float(source_config["data"].get("dt", 0.02)),
            order_seed=int(validation_cfg["method_order_seed"]),
        )
        robot_records = point_records + trajectory_records
        all_records.extend(robot_records)

        robot_metrics: dict[str, Any] = {}
        for backend in engines:
            robot_metrics[backend] = {
                method: method_validation_metrics(robot_records, robot, backend, method)
                for method in ("baseline", "proposed")
            }
        validation_metrics[robot] = robot_metrics
        equivalence_robots[robot] = {
            "risk_probability_max_abs_error": risk_max_error,
            "risk_score_max_abs_error": risk_score_max_error,
            "stored_feature_routing_action_agreement": route_agreement_min,
            "production_runtime_wrapper_equivalence": production_wrapper_check,
            "optimized_pytorch": {
                "seed_max_abs_error": optimized_seed_error,
                "records": record_equivalence(
                    robot_records,
                    robot=robot,
                    reference_backend="eager_original",
                    candidate_backend="optimized_pytorch",
                ),
            },
            "torchscript_exact": {
                "seed_max_abs_error": exported_seed_error,
                "records": record_equivalence(
                    robot_records,
                    robot=robot,
                    reference_backend="eager_original",
                    candidate_backend="torchscript_exact",
                ),
            },
        }
        microbenchmarks[robot] = {
            "risk": _risk_microbenchmark(
                eager_risk,
                vector_risk,
                RiskDataset.load(paths["risk_validation"]).features[
                    np.asarray(point_indices, dtype=np.int64)
                ],
                warmup=50 if args.smoke else 200,
                repeats=100 if args.smoke else 1500,
            ),
            "feature_cache": _feature_cache_microbenchmark(
                optimized_seed, point_dataset, samples=20 if args.smoke else 300
            ),
            "single_member": _single_member_diagnostic(
                cpu_ensemble,
                encoded[: max(100, min(2000, len(encoded)))],
                warmup=20 if args.smoke else 200,
                repeats=80 if args.smoke else 1000,
            ),
            "order_counts": order_counts,
        }
        selection_inputs[robot] = {
            "gate": gate_source,
            "point_source_split": "risk_validation_queries",
            "point_selected_source_indices": point_indices,
            "trajectory_source_split": "seed_validation",
                "selected_trajectory_ids": trajectory_ids,
                "test_queries_loaded": False,
                "export_backend": "single-call traced TorchScript CUDA graph",
                "export_package_persisted": False,
                "export_compile_and_load_inside_timed_interval": False,
            }

    breakdown_summary = latency_breakdown_summary(all_records)
    paired_summaries = paired_latency_summary(all_records)
    equivalence_cfg = dict(v3_config["equivalence"])
    if args.smoke:
        # Five rows/category make one evaluation a 0.1 shift in a stratum mean.
        # This affects only the no-output smoke path; the formal pilot retains
        # the preregistered 0.05-evaluation tolerance.
        equivalence_cfg["point_mean_function_evaluations_abs_tolerance"] = 0.2
    for robot in robots:
        for backend in ("optimized_pytorch", "torchscript_exact"):
            record_check = equivalence_robots[robot][backend]["records"]
            reference_metrics = validation_metrics[robot]["eager_original"]["proposed"]
            candidate_metrics = validation_metrics[robot][backend]["proposed"]
            reference_baseline_metrics = validation_metrics[robot]["eager_original"]["baseline"]
            candidate_baseline_metrics = validation_metrics[robot][backend]["baseline"]
            metric_deltas = {
                key: _metric_delta(candidate_metrics, reference_metrics, key)
                for key in (
                    "point_feasible_success",
                    "point_rejectable_rejection",
                    "point_feasible_mean_function_evaluations",
                    "point_rejectable_mean_function_evaluations",
                    "trajectory_completion",
                    "trajectory_mean_function_evaluations",
                    "trajectory_command_spike",
                )
            }
            equivalence_robots[robot][backend]["metric_deltas_vs_eager"] = metric_deltas
            baseline_metric_deltas = {
                key: _metric_delta(candidate_baseline_metrics, reference_baseline_metrics, key)
                for key in (
                    "point_feasible_success",
                    "point_rejectable_rejection",
                    "point_feasible_mean_function_evaluations",
                    "point_rejectable_mean_function_evaluations",
                    "trajectory_completion",
                    "trajectory_mean_function_evaluations",
                    "trajectory_command_spike",
                )
            }
            equivalence_robots[robot][backend]["baseline_metric_deltas_vs_eager"] = baseline_metric_deltas
            risk_metric_deltas = [
                abs(float(delta))
                for split in risk_validation_reports[robot].values()
                for delta in split["metric_deltas_vs_eager"].values()
            ]
            equivalence_robots[robot][backend]["risk_metric_max_abs_delta_vs_eager"] = max(
                risk_metric_deltas, default=float("nan")
            )
            equivalence_robots[robot][backend]["pass"] = bool(
                equivalence_robots[robot][backend]["seed_max_abs_error"]
                <= float(equivalence_cfg["seed_max_abs_tolerance"])
                and equivalence_robots[robot]["risk_probability_max_abs_error"]
                <= float(equivalence_cfg["risk_probability_max_abs_tolerance"])
                and equivalence_robots[robot]["risk_score_max_abs_error"]
                <= float(equivalence_cfg["risk_score_max_abs_tolerance"])
                and equivalence_robots[robot]["stored_feature_routing_action_agreement"]
                >= float(equivalence_cfg["routing_action_agreement_min"])
                and equivalence_robots[robot][backend]["risk_metric_max_abs_delta_vs_eager"]
                <= float(equivalence_cfg["metric_abs_tolerance"])
                and equivalence_robots[robot]["production_runtime_wrapper_equivalence"]["pass"]
                and record_check["point_route_action_agreement"]
                >= float(equivalence_cfg["routing_action_agreement_min"])
                and record_check["trajectory_route_action_agreement"]
                >= float(equivalence_cfg["routing_action_agreement_min"])
                and record_check["all_route_action_agreement"]
                >= float(equivalence_cfg["routing_action_agreement_min"])
                and record_check["accepted_agreement"] == 1.0
                and all(
                    abs(metric_deltas[key]) <= float(equivalence_cfg["metric_abs_tolerance"])
                    and abs(baseline_metric_deltas[key]) <= float(equivalence_cfg["metric_abs_tolerance"])
                    for key in (
                        "point_feasible_success",
                        "point_rejectable_rejection",
                        "trajectory_completion",
                        "trajectory_command_spike",
                    )
                )
                and all(
                    abs(metric_deltas[key])
                    <= float(equivalence_cfg["point_mean_function_evaluations_abs_tolerance"])
                    and abs(baseline_metric_deltas[key])
                    <= float(equivalence_cfg["point_mean_function_evaluations_abs_tolerance"])
                    for key in (
                        "point_feasible_mean_function_evaluations",
                        "point_rejectable_mean_function_evaluations",
                    )
                )
                and abs(metric_deltas["trajectory_mean_function_evaluations"])
                <= float(equivalence_cfg["trajectory_mean_function_evaluations_abs_tolerance"])
                and abs(baseline_metric_deltas["trajectory_mean_function_evaluations"])
                <= float(equivalence_cfg["trajectory_mean_function_evaluations_abs_tolerance"])
            )
    candidate_backends = [
        backend
        for backend in ("optimized_pytorch", "torchscript_exact")
        if all(equivalence_robots[robot][backend]["pass"] for robot in robots)
    ]
    if candidate_backends:
        selected_backend = min(
            candidate_backends,
            key=lambda backend: max(
                float(
                    paired_summaries[f"{robot}/{backend}/point_feasible"]["proposed_ms"]["p95"]
                )
                for robot in robots
            ),
        )
    else:
        selected_backend = None
    numerical_equivalence = {
        "reference_backend": "eager_original",
        "eligible_candidate_backends": candidate_backends,
        "selected_backend": selected_backend,
        "single_member_eligible_for_selection": False,
        "tolerances": equivalence_cfg,
        "robots": equivalence_robots,
        "risk_validation_metrics": risk_validation_reports,
        "solver_and_trajectory_validation_metrics": validation_metrics,
        "all_pass": bool(selected_backend is not None),
    }

    eager_comparison = (
        compare_backends(
            all_records,
            reference_backend="eager_original",
            candidate_backend=selected_backend,
        )
        if selected_backend is not None
        else {}
    )
    paired_payload: dict[str, Any] = {
        "timing_unit": "milliseconds",
        "paired_difference_sign": "proposed_minus_baseline",
        "core_e2e_excludes_logging_serialization": True,
        "summaries": paired_summaries,
        "eager_comparison": eager_comparison,
        "selected_backend": selected_backend,
        "selection_rule": "validation-only eligible backend minimizing worst two-robot feasible proposed P95",
        "single_member_considered": False,
    }

    # Stage-attributed savings for the required change log.
    for robot in robots:
        def stage_p95(backend: str, stage: str) -> float:
            key = f"{robot}/{backend}/proposed/point_feasible/all"
            return float(breakdown_summary[key][stage]["p95"])
        eager_seed_p95 = stage_p95("eager_original", "learned_seed_inference_ms")
        optimized_seed_p95 = stage_p95("optimized_pytorch", "learned_seed_inference_ms")
        export_seed_p95 = stage_p95("torchscript_exact", "learned_seed_inference_ms")
        eager_conversion = stage_p95("eager_original", "numpy_torch_conversion_ms")
        optimized_conversion = stage_p95("optimized_pytorch", "numpy_torch_conversion_ms")
        logging_p95 = stage_p95(selected_backend or "optimized_pytorch", "logging_serialization_ms")
        microbenchmarks[robot]["stage_savings"] = {
            "vectorized_seed_p95_ms": eager_seed_p95 - optimized_seed_p95,
            "conversion_p95_ms": eager_conversion - optimized_conversion,
            "exact_export_seed_p95_ms": eager_seed_p95 - export_seed_p95,
        }
        microbenchmarks[robot]["logging_p95_ms"] = logging_p95
        stage_names = (
            "feature_preparation_ms",
            "numpy_torch_conversion_ms",
            "learned_seed_inference_ms",
            "uncertainty_risk_inference_ms",
            "routing_decision_ms",
            "numerical_solver_ms",
            "verification_ms",
        )
        for label, backend in (
            ("largest_stage_eager", "eager_original"),
            ("largest_stage_selected", selected_backend or "optimized_pytorch"),
        ):
            group = breakdown_summary[f"{robot}/{backend}/proposed/point_feasible/all"]
            largest = max(stage_names, key=lambda stage: float(group[stage]["p95"]))
            microbenchmarks[robot][label] = {
                "stage": largest,
                "p95_ms": float(group[largest]["p95"]),
            }

    targets = dict(v3_config["validation_targets"])
    gate_robots: dict[str, Any] = {}
    for robot in robots:
        if selected_backend is None:
            gate_robots[robot] = {"all_pass": False, "reason": "no numerically equivalent backend"}
            continue
        baseline = validation_metrics[robot][selected_backend]["baseline"]
        proposed = validation_metrics[robot][selected_backend]["proposed"]
        success_gap = proposed["point_feasible_success"] - baseline["point_feasible_success"]
        completion_gap = proposed["trajectory_completion"] - baseline["trajectory_completion"]
        policy_metrics = risk_validation_reports[robot]["policy_validation"]["metrics"]["action_policy"]
        risk_metrics = risk_validation_reports[robot]["risk_validation"]["metrics"]
        latency = paired_summaries[f"{robot}/{selected_backend}/point_feasible"]
        reduction = eager_comparison[robot]["p95_reduction"]
        category_success = {}
        category_rejection = {}
        for category, proposed_category in proposed["by_category"].items():
            baseline_category = baseline["by_category"][category]
            if proposed_category["feasible_count"]:
                category_success[category] = (
                    proposed_category["feasible_success"] - baseline_category["feasible_success"]
                ) >= float(targets["feasible_success_gap_min"])
            if proposed_category["rejectable_count"]:
                category_rejection[category] = proposed_category["rejectable_rejection"] >= float(
                    targets["rejectable_rejection_min"]
                )
        success_gate = bool(
            success_gap >= float(targets["feasible_success_gap_min"])
            and all(category_success.values())
        )
        rejection_gate = bool(
            proposed["point_rejectable_rejection"] >= float(targets["rejectable_rejection_min"])
            and all(category_rejection.values())
        )
        risk_gate = bool(
            risk_metrics["fail_auroc"] >= float(targets["fail_auroc_min"])
            and risk_metrics["fail_ece"] <= float(targets["fail_ece_max"])
            and policy_metrics["false_reject_rate"] <= float(targets["false_reject_rate_max"])
            and policy_metrics["reject_recall"] >= float(targets["reject_recall_min"])
            and policy_metrics["nonreject_macro_f1"] >= float(targets["nonreject_macro_f1_min"])
        )
        trajectory_gate = bool(
            completion_gap >= float(targets["trajectory_completion_gap_min"])
            and proposed["trajectory_command_spike"] <= float(targets["trajectory_command_spike_max"])
        )
        latency_gate = bool(
            latency["p95_ratio_proposed_over_baseline"]
            <= float(targets["feasible_p95_ratio_max"])
            and reduction >= float(targets["feasible_p95_reduction_vs_eager_min"])
        )
        gate_robots[robot] = {
            "success_gate_pass": success_gate,
            "rejection_gate_pass": rejection_gate,
            "risk_gate_pass": risk_gate,
            "trajectory_gate_pass": trajectory_gate,
            "latency_gate_pass": latency_gate,
            "numerical_equivalence_pass": equivalence_robots[robot][selected_backend]["pass"],
            "all_validation_categories_success_pass": category_success,
            "all_validation_categories_rejection_pass": category_rejection,
            "metrics": {
                "feasible_success_gap": success_gap,
                "point_rejectable_rejection": proposed["point_rejectable_rejection"],
                "fail_auroc": risk_metrics["fail_auroc"],
                "fail_ece": risk_metrics["fail_ece"],
                "reject_recall": policy_metrics["reject_recall"],
                "false_reject_rate": policy_metrics["false_reject_rate"],
                "nonreject_macro_f1": policy_metrics["nonreject_macro_f1"],
                "trajectory_completion_gap": completion_gap,
                "trajectory_command_spike": proposed["trajectory_command_spike"],
                "feasible_p95_ratio": latency["p95_ratio_proposed_over_baseline"],
                "feasible_p95_reduction_vs_eager": reduction,
                "function_evaluations": {
                    "baseline_feasible_mean": baseline["point_feasible_mean_function_evaluations"],
                    "proposed_feasible_mean": proposed["point_feasible_mean_function_evaluations"],
                },
            },
        }
        gate_robots[robot]["all_pass"] = bool(
            success_gate
            and rejection_gate
            and risk_gate
            and trajectory_gate
            and latency_gate
            and equivalence_robots[robot][selected_backend]["pass"]
        )
    validation_gate = {
        "protocol": "latency_pilot_v3 validation-only implementation gate",
        "selected_backend": selected_backend,
        "targets": targets,
        "robots": gate_robots,
        "all_robots_pass": bool(
            selected_backend is not None
            and all(bool(gate_robots[robot].get("all_pass", False)) for robot in robots)
        ),
        "formal_test_v3_started": False,
        "thresholds_modified": False,
        "labels_modified": False,
        "claim_gate_modified": False,
    }

    latency_payload = {
        "schema_version": 3,
        "clock": "time.perf_counter_ns",
        "timing_contract": {
            "feature_preparation": "normalization/input fill, candidate scoring, and nine locked risk features",
            "numpy_torch_conversion": "input/output tensor views or necessary device/dtype transfers",
            "learned_seed_inference": "all five frozen SeedMLP members",
            "uncertainty_risk_inference": "ensemble uncertainty plus frozen calibrated risk model",
            "routing_decision": "locked action gate and policy construction",
            "numerical_solver": "DLS/TRF/seed-bank stage time excluding verifier calls",
            "verification": "shared frozen SolutionVerifier calls",
            "logging_serialization": "record construction and in-memory json.dumps after core timer",
            "total_end_to_end": "feature start through final verification/result decision; excludes logging and disk I/O",
        },
        "warmup_iterations_per_backend_pair": int(validation_cfg["warmup_iterations"]),
        "point_timing_repeats": int(validation_cfg["point_timing_repeats"]),
        "disk_writes_during_timed_intervals": False,
        "microbenchmarks": microbenchmarks,
        "summaries": breakdown_summary,
        "records": all_records,
    }

    print("[v3] re-hashing frozen paper_v2 outputs", flush=True)
    frozen_after = _frozen_snapshot(workspace / "outputs")
    changed_paths = sorted(
        key
        for key in set(frozen_before["entries"]) | set(frozen_after["entries"])
        if frozen_before["entries"].get(key) != frozen_after["entries"].get(key)
    )
    if changed_paths:
        raise RuntimeError(f"frozen paper_v2 outputs changed during v3: {changed_paths[:5]}")

    run_manifest = {
        "protocol_version": v3_config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "source_experiment": v3_config["source_experiment"],
        "training_seed": int(v3_config["training_seed"]),
        "selection_scope": "training/validation only",
        "source_config": str(source_config_path),
        "source_config_sha256": _sha256_file(source_config_path),
        "v3_config": str(config_path),
        "v3_config_sha256": _sha256_file(config_path),
        "source_tree_sha256": source_tree_hash(),
        "output_directory": str(output_dir),
        "expected_output_files": list(OUTPUT_FILENAMES),
        "selected_backend": selected_backend,
        "thread_selection": thread_report,
        "split_roles": {
            "seed_train": "frozen model training source; audit only",
            "seed_validation": "seed equivalence and trajectory validation",
            "risk_train": "frozen risk training source; audit only",
            "risk_validation": "risk metrics/equivalence and point pilot",
            "calibration": "frozen calibration source; audit only",
            "policy_validation": "locked route-threshold source and policy metrics",
            "formal_test": "prohibited and not loaded",
        },
        "split_audit": split_audits,
        "selection_inputs": selection_inputs,
        "input_artifacts": input_manifest,
        "formal_output_snapshot": {
            "before_tree_digest": frozen_before["tree_digest"],
            "after_tree_digest": frozen_after["tree_digest"],
            "file_count": frozen_before["file_count"],
            "total_bytes": frozen_before["total_bytes"],
            "changed_paths": changed_paths,
            "unchanged": not changed_paths,
            "compared_fields": ["sha256", "size", "mtime_ns", "mode"],
        },
        "test_named_dataset_loaded": False,
        "formal_test_v3_started": False,
        "single_member_used_for_selection": False,
        "smoke_mode": bool(args.smoke),
    }
    environment = {
        **environment_payload(),
        "captured_after_thread_selection_utc": datetime.now(timezone.utc).isoformat(),
        "torch_intra_op_threads": torch.get_num_threads(),
        "torch_inter_op_threads": torch.get_num_interop_threads(),
        "thread_probe": thread_report,
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "benchmark_compute_scope": "exact URDF kinematics; no Isaac Lab physics",
    }

    if args.smoke:
        print(
            json.dumps(
                {
                    "smoke": True,
                    "selected_backend": selected_backend,
                    "all_pass": validation_gate["all_robots_pass"],
                    "record_count": len(all_records),
                    "formal_outputs_unchanged": not changed_paths,
                    "equivalence": {
                        robot: {
                            backend: equivalence_robots[robot][backend]
                            for backend in ("optimized_pytorch", "torchscript_exact")
                        }
                        for robot in robots
                    },
                },
                indent=2,
            ),
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report = _report_markdown(
        selected_backend=selected_backend,
        gate_payload=validation_gate,
        paired_payload=paired_payload,
        equivalence=numerical_equivalence,
        microbenchmarks=microbenchmarks,
        thread_report=thread_report,
    )
    _write_json(output_dir / "latency_breakdown.json", latency_payload)
    _write_json(output_dir / "paired_latency_summary.json", paired_payload)
    _write_json(output_dir / "numerical_equivalence.json", numerical_equivalence)
    _write_json(output_dir / "validation_gate_v3.json", validation_gate)
    (output_dir / "optimization_changes.md").write_text(report, encoding="utf-8")
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "environment.json", environment)
    frozen_post_write = _frozen_snapshot(workspace / "outputs")
    post_write_changed = sorted(
        key
        for key in set(frozen_before["entries"]) | set(frozen_post_write["entries"])
        if frozen_before["entries"].get(key) != frozen_post_write["entries"].get(key)
    )
    if post_write_changed:
        raise RuntimeError(
            f"frozen paper_v2 outputs changed while writing v3: {post_write_changed[:5]}"
        )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "selected_backend": selected_backend,
                "all_pass": validation_gate["all_robots_pass"],
                "record_count": len(all_records),
                "formal_outputs_unchanged": not changed_paths,
                "formal_test_v3_started": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
