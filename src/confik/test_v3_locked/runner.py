from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot
from ..data.datasets import QueryDataset
from ..experiments.ablations import FeatureMaskRiskProvider, SingleMemberCandidates
from ..experiments.baselines import TRFOnlyMethod, fixed_hybrid
from ..experiments.baselines_v2 import ThresholdGuardConfig, ThresholdGuardRiskProvider
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.optimized_inference import EagerRiskEngine, EagerSeedEngine
from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime
from ..latency_pilot_v3.runner import _build_runtimes, _load_gate_config, _solver_components
from ..models.risk import RiskModel
from ..models.seed import PreviousStateCandidates, TorchSeedEnsemble
from ..release_v3_locked.artifacts import load_frozen_risk, load_locked_seed_engine
from ..runtime.cascade import (
    ActionGateConfig,
    CalibratedActionGate,
    CascadedHybridIK,
)
from ..solvers.dls import AdaptiveDLS, DLSConfig
from ..types import CandidateSet, IKQuery
from .benchmark import (
    benchmark_methods,
    benchmark_profiled_points,
    benchmark_profiled_single,
    benchmark_profiled_trajectories,
    method_summary,
    warmup_methods,
    warmup_profiled,
)
from .data import (
    dataset_schema,
    derive_seed,
    generate_locked_dataset,
    split_audit,
    validate_schema,
)
from .reporting import (
    PRIMARY_BACKEND,
    cluster_intervals,
    latency_report,
    primary_records,
    write_locked_claim_gate,
)


ROBOTS = ("panda", "ur5e")
TRAINING_SEEDS = (17, 29, 43)
METHODS = (
    "dls_previous_1x50",
    "learned_1x25",
    "fixed_robust_cascade",
    "threshold_guard_cascade",
    "trf_previous",
    "proposed_v2",
    "ablation_no_history",
    "ablation_single_member",
    "ablation_no_uncertainty",
    "ablation_uncalibrated",
    "ablation_no_reject",
    "ablation_no_fallback",
    "ablation_fixed_damping",
)
LOCKED_GATE = {
    "feasible_success_gap_min": -0.01,
    "rejectable_rejection_min": 0.95,
    "feasible_fev_reduction_min": 0.10,
    "feasible_p95_ratio_max": 1.25,
    "rejectable_fev_reduction_min": 0.50,
    "rejectable_latency_reduction_min": 0.50,
    "fail_auroc_min": 0.75,
    "fail_ece_max": 0.10,
    "false_reject_rate_max": 0.02,
    "reject_recall_min": 0.95,
    "nonreject_macro_f1_min": 0.30,
    "trajectory_completion_gap_min": -0.10,
    "trajectory_command_spike_max": 0.0,
    "threshold_success_gap_min": -0.01,
    "threshold_fev_reduction_min": 0.05,
    "threshold_p95_ratio_max": 1.25,
}


class LockedCandidates:
    def __init__(self, engine: object):
        self.engine = engine
        self.kinematics = engine.kinematics

    def candidates(self, query: IKQuery) -> CandidateSet:
        return self.engine.prepare(query).candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the one-shot locked test_v3 protocol")
    parser.add_argument("--config", required=True)
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
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _snapshot(output_root: Path, patterns: list[str]) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
    entries: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output cannot be a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = str(path.relative_to(output_root))
            stat = path.stat()
            entries[relative] = {
                "sha256": _sha256_file(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return {
        "directories": [str(path.relative_to(output_root)) for path in sorted(directories)],
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries.values()),
        "tree_digest": _json_digest(entries),
    }


def _release_paths(release_root: Path, robot: str, seed: int) -> dict[str, Path]:
    root = release_root / robot / f"seed{seed}"
    paths = {
        "torchscript": root / "exact_seed_ensemble.ts",
        "forest": root / "hgb_vectorized_parameters.npz",
        "calibration": root / "isotonic_calibration_parameters.npz",
        "normalization": root / "normalization_parameters.npz",
        "solver_metadata": root / "solver_metadata.json",
        "route_thresholds": root / "route_thresholds.json",
        "seed_bank": root / "seed_bank.npz",
        "runtime_spec": root / "runtime_spec.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _source_paths(workspace: Path, robot: str, seed: int) -> dict[str, Path]:
    root = workspace / "outputs" / f"paper_v2_seed{seed}" / robot
    paths = {
        "seed_model": root / "models" / "seed_ensemble.pt",
        "risk_model": root / "models" / "risk_model.joblib",
        "no_history_seed_model": root / "models" / "seed_no_history.pt",
        "uncalibrated_risk_model": root / "models" / "risk_uncalibrated.joblib",
        "no_uncertainty_risk_model": root / "models" / "risk_no_uncertainty.joblib",
        "threshold_guard": root / "results" / "threshold_guard.json",
        "policy_selection": root / "results" / "policy_selection_v2.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _comparison_files(workspace: Path, robot: str) -> list[tuple[str, Path, str]]:
    result: list[tuple[str, Path, str]] = []
    for seed in TRAINING_SEEDS:
        root = workspace / "outputs" / f"paper_v2_seed{seed}" / robot / "datasets"
        for name, kind in (
            ("seed_train", "transition"),
            ("seed_validation", "transition"),
            ("risk_train_queries", "query"),
            ("risk_validation_queries", "query"),
            ("calibration_queries", "query"),
            ("policy_validation_queries", "query"),
            ("risk_test_queries", "query"),
            ("test_queries", "query"),
        ):
            result.append((f"seed{seed}/{name}", root / f"{name}.npz", kind))
    return result


def _current_environment(release_environment: dict[str, Any]) -> dict[str, Any]:
    current = environment_payload()
    current.update(
        {
            "captured_utc": _utc(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "gpu_uuid": subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[0],
            "nvidia_driver": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[0],
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment_variables": {
                key: os.environ.get(key)
                for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        }
    )
    compared = (
        "hostname", "gpu", "gpu_uuid", "nvidia_driver", "torch", "cuda_version",
        "pinocchio", "scikit_learn", "python_executable",
    )
    matches = {key: current.get(key) == release_environment.get(key) for key in compared}
    current["release_environment_comparison"] = {
        "fields": list(compared),
        "matches": matches,
        "all_match": all(matches.values()),
    }
    if not all(matches.values()):
        raise RuntimeError(f"test_v3 environment differs from release: {matches}")
    return current


def _asset_manifest(workspace: Path, release_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for robot in ROBOTS:
        for seed in TRAINING_SEEDS:
            key = f"{robot}/seed{seed}"
            paths = {**_release_paths(release_root, robot, seed), **_source_paths(workspace, robot, seed)}
            result[key] = {
                role: {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size}
                for role, path in sorted(paths.items())
            }
    return result


def _preregistration(
    *,
    workspace: Path,
    config_path: Path,
    config: dict[str, Any],
    release_root: Path,
    release_manifest: dict[str, Any],
    source_assets: dict[str, Any],
) -> dict[str, Any]:
    release_commit = str(release_manifest["git_commit"])
    generation_seeds = {
        robot: {
            role: derive_seed(release_commit, robot, role)
            for role in ("id_transitions", "point_stress", "hard_valid", "trajectories", "method_order", "bootstrap")
        }
        for robot in ROBOTS
    }
    return {
        "protocol": "one-shot formal test_v3",
        "created_utc": _utc(),
        "release_commit": release_commit,
        "release_manifest_path": str(release_root / "release_manifest.json"),
        "release_manifest_sha256": _sha256_file(release_root / "release_manifest.json"),
        "release_equivalence_sha256": _sha256_file(release_root / "release_equivalence.json"),
        "release_backend": PRIMARY_BACKEND,
        "backend_reselection_allowed": False,
        "runner_git_commit": _git(workspace, "rev-parse", "HEAD"),
        "runner_git_tree": _git(workspace, "rev-parse", "HEAD^{tree}"),
        "runner_git_worktree_clean": True,
        "runner_sha256": _sha256_file(Path(__file__)),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "robots": list(ROBOTS),
        "training_seeds": list(TRAINING_SEEDS),
        "training_seed_role": "sensitivity replicate sharing one test set per robot; not an independent query sample",
        "methods": list(METHODS),
        "data": config["data"],
        "generation_seeds": generation_seeds,
        "timing": config["timing"],
        "statistics": config["statistics"],
        "claim_gate": config["claim_gate"],
        "source_assets": source_assets,
        "test_data_rules": {
            "may_inspect_before_formal_run": ["schema", "counts", "categories", "duplicates", "hashes", "exact_overlap"],
            "may_not_inspect_before_formal_run": ["method outputs", "latency", "FEV", "success", "solver difficulty"],
            "test_set_parameter_selection": False,
        },
        "failure_policy": {
            "gate_failure_rerun_allowed": False,
            "outlier_removal_allowed": False,
            "winsorization_allowed": False,
            "technical_retry_requires_identical_release_dataset_environment": True,
            "retry_count": 0,
        },
    }


def _build_comparators(
    *,
    source_config: dict[str, Any],
    robot: str,
    seed: int,
    kinematics: object,
    release_paths: dict[str, Path],
    source_paths: dict[str, Path],
    locked_seed: object,
    locked_risk: object,
    gate_config: ActionGateConfig,
) -> dict[str, object]:
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": release_paths["solver_metadata"], "seed_bank": release_paths["seed_bank"]},
        kinematics,
    )
    candidates = LockedCandidates(locked_seed)
    eager_ensemble = TorchSeedEnsemble.load(source_paths["seed_model"], kinematics, device="cuda:0")
    no_history = TorchSeedEnsemble.load(source_paths["no_history_seed_model"], kinematics, device="cuda:0")
    uncalibrated = RiskModel.load(source_paths["uncalibrated_risk_model"])
    no_uncertainty = RiskModel.load(source_paths["no_uncertainty_risk_model"])

    def cascade_method(
        candidate_provider: object,
        risk_provider: object,
        *,
        gate: object | None = None,
        solver: AdaptiveDLS = dls,
        use_fallback: bool = True,
    ) -> CascadedHybridIK:
        return CascadedHybridIK(
            kinematics,  # type: ignore[arg-type]
            candidate_provider,  # type: ignore[arg-type]
            risk_provider,  # type: ignore[arg-type]
            solver,
            verifier,
            gate=gate or CalibratedActionGate(gate_config),  # type: ignore[arg-type]
            seed_bank=seed_bank if use_fallback else None,
            fallback=fallback if use_fallback else None,
            config=cascade,
        )

    no_reject = asdict(gate_config)
    no_reject["reject_probability"] = 1.1
    fixed_values = asdict(dls.config)
    fixed_values["lambda_min"] = 0.01
    fixed_values["lambda_max"] = 0.01
    fixed_dls = AdaptiveDLS(kinematics, DLSConfig(**fixed_values))  # type: ignore[arg-type]
    return {
        "dls_previous_1x50": fixed_hybrid(
            kinematics,  # type: ignore[arg-type]
            PreviousStateCandidates(),
            dls,
            verifier,
            candidate_count=1,
            iterations=50,
        ),
        "learned_1x25": fixed_hybrid(
            kinematics, candidates, dls, verifier, candidate_count=1, iterations=25  # type: ignore[arg-type]
        ),
        "trf_previous": TRFOnlyMethod(fallback, verifier),
        "ablation_no_history": cascade_method(no_history, locked_risk),
        "ablation_single_member": cascade_method(SingleMemberCandidates(eager_ensemble), locked_risk),
        "ablation_no_uncertainty": cascade_method(
            candidates,
            FeatureMaskRiskProvider(no_uncertainty, (0, 1, 4, 5, 6, 7, 8)),
        ),
        "ablation_uncalibrated": cascade_method(candidates, uncalibrated),
        "ablation_no_reject": cascade_method(
            candidates,
            locked_risk,
            gate=CalibratedActionGate(ActionGateConfig(**no_reject)),
        ),
        "ablation_no_fallback": cascade_method(candidates, locked_risk, use_fallback=False),
        "ablation_fixed_damping": cascade_method(candidates, locked_risk, solver=fixed_dls),
    }


def _write_raw_records(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(_safe(record), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def _run_combination(
    *,
    workspace: Path,
    destination: Path,
    source_config: dict[str, Any],
    config: dict[str, Any],
    release_root: Path,
    robot: str,
    seed: int,
    dataset: QueryDataset,
    dataset_manifest: dict[str, Any],
    environment: dict[str, Any],
    preregistration_sha256: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    paths = _release_paths(release_root, robot, seed)
    sources = _source_paths(workspace, robot, seed)
    kinematics = load_robot(source_config, robot)
    gate_config, gate_source = _load_gate_config(sources["policy_selection"])
    locked_seed = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=paths["torchscript"],
        normalization_path=paths["normalization"],
        runtime_spec_path=paths["runtime_spec"],
        device="cuda:0",
    )
    locked_risk = load_frozen_risk(paths["forest"], paths["calibration"])
    source_ensemble = TorchSeedEnsemble.load(sources["seed_model"], kinematics, device="cuda:0")
    source_risk = RiskModel.load(sources["risk_model"])
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    runtimes = _build_runtimes(
        kinematics=kinematics,
        gate_config=gate_config,
        engines={
            PRIMARY_BACKEND: (locked_seed, locked_risk, True),
            "eager_reference": (EagerSeedEngine(source_ensemble), EagerRiskEngine(source_risk), False),
        },
        dls=dls,
        verifier=verifier,
        fallback=fallback,
        seed_bank=seed_bank,
        cascade=cascade,
    )
    threshold_values = json.loads(sources["threshold_guard"].read_text(encoding="utf-8"))
    threshold_provider = ThresholdGuardRiskProvider(ThresholdGuardConfig(**threshold_values))
    threshold_runtime = ProfiledCascadeRuntime(
        name="threshold_guard_cascade",
        kinematics=kinematics,
        seed_engine=locked_seed,
        risk_engine=threshold_provider,  # type: ignore[arg-type]
        gate=CalibratedActionGate(gate_config),
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
        reuse_candidate_features=True,
    )
    point_mask = ~np.char.startswith(dataset.category.astype(str), "trajectory_")
    trajectory_mask = ~point_mask
    point = QueryDataset(**{name: getattr(dataset, name)[point_mask] for name in dataset.__dataclass_fields__})
    trajectory = QueryDataset(**{name: getattr(dataset, name)[trajectory_mask] for name in dataset.__dataclass_fields__})
    timing = config["timing"]
    dt = float(config["data"]["dt"])
    order_seed = derive_seed(dataset_manifest["release_commit"], robot, "method_order") + seed

    # No disk writes occur between the first warmup call and the completion of
    # all benchmark calls below.
    warmup_profiled(runtimes, point, iterations=int(timing["warmup_iterations"]), dt=dt)
    records = benchmark_profiled_points(
        robot,
        runtimes,
        point,
        repeats=int(timing["primary_point_repeats"]),
        dt=dt,
        order_seed=order_seed,
    )
    records.extend(
        benchmark_profiled_trajectories(
            robot, runtimes, trajectory, dt=dt, order_seed=order_seed
        )
    )
    warmup_methods(
        {"threshold_guard_cascade": threshold_runtime},
        point,
        iterations=int(timing["warmup_iterations"]),
        dt=dt,
    )
    records.extend(
        benchmark_profiled_single(
            robot,
            backend=PRIMARY_BACKEND,
            method="threshold_guard_cascade",
            runtime=threshold_runtime,
            dataset=point,
            split="point",
            repeats=int(timing["primary_point_repeats"]),
            dt=dt,
        )
    )
    records.extend(
        benchmark_profiled_single(
            robot,
            backend=PRIMARY_BACKEND,
            method="threshold_guard_cascade",
            runtime=threshold_runtime,
            dataset=trajectory,
            split="trajectory",
            repeats=1,
            dt=dt,
        )
    )
    comparators = _build_comparators(
        source_config=source_config,
        robot=robot,
        seed=seed,
        kinematics=kinematics,
        release_paths=paths,
        source_paths=sources,
        locked_seed=locked_seed,
        locked_risk=locked_risk,
        gate_config=gate_config,
    )
    warmup_methods(comparators, point, iterations=int(timing["warmup_iterations"]), dt=dt)
    records.extend(
        benchmark_methods(
            robot,
            comparators,
            point,
            split="point",
            point_repeats=int(timing["comparator_point_repeats"]),
            dt=dt,
            order_seed=order_seed + 1_000_000,
        )
    )
    records.extend(
        benchmark_methods(
            robot,
            comparators,
            trajectory,
            split="trajectory",
            point_repeats=1,
            dt=dt,
            order_seed=order_seed + 2_000_000,
        )
    )

    formal = primary_records(records)
    summary = {
        "robot": robot,
        "training_seed": seed,
        "backend": PRIMARY_BACKEND,
        "method_metrics": method_summary(formal),
        "method_count": len(METHODS),
        "query_count_per_method": len(dataset),
    }
    claim = write_locked_claim_gate(formal, {key: float(value) for key, value in config["claim_gate"].items()})
    latency = latency_report(records, tail_thresholds_ms=[float(value) for value in timing["tail_thresholds_ms"]])
    intervals = cluster_intervals(
        formal,
        samples=int(config["statistics"]["bootstrap_samples"]),
        seed=derive_seed(dataset_manifest["release_commit"], robot, "bootstrap") + seed,
    )
    protocol = {
        "protocol": "test_v3_locked",
        "robot": robot,
        "training_seed": seed,
        "preregistration_sha256": preregistration_sha256,
        "dataset_path": dataset_manifest["datasets"][robot]["path"],
        "dataset_sha256": dataset_manifest["datasets"][robot]["sha256"],
        "release_artifacts": {
            role: {"path": str(path), "sha256": _sha256_file(path)} for role, path in paths.items()
        },
        "gate_source": gate_source,
        "timing": timing,
        "test_parameter_selection_performed": False,
        "outliers_removed": False,
        "winsorization_performed": False,
        "raw_record_count": len(records),
    }
    _write_json(destination / "summary_v3.json", summary)
    _write_json(destination / "claim_gate_v3.json", claim)
    _write_json(destination / "latency_breakdown_v3.json", latency)
    _write_json(destination / "cluster_intervals_v3.json", intervals)
    _write_json(destination / "protocol_manifest_v3.json", protocol)
    _write_json(destination / "environment_v3.json", environment)
    _write_raw_records(destination / "query_records_v3.jsonl.gz", records)
    (destination / "run_log.txt").write_text(
        "\n".join(
            [
                f"completed_utc={_utc()}",
                f"robot={robot}",
                f"training_seed={seed}",
                "natural_exit=true",
                "technical_retry_count=0",
                "test_set_retuning=false",
                "outlier_removal=false",
            ]
        ) + "\n",
        encoding="utf-8",
    )


def _aggregate(workspace: Path, aggregate_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for robot in ROBOTS:
        for seed in TRAINING_SEEDS:
            root = workspace / "outputs" / f"test_v3_seed{seed}" / robot
            claim = json.loads((root / "claim_gate_v3.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "summary_v3.json").read_text(encoding="utf-8"))
            latency = json.loads((root / "latency_breakdown_v3.json").read_text(encoding="utf-8"))
            runs.append({"robot": robot, "seed": seed, "root": str(root), "claim": claim, "summary": summary, "latency": latency})
    metrics = (
        "point_feasible_success_gap",
        "point_feasible_evaluation_reduction",
        "point_feasible_p95_latency_reduction",
        "point_rejectable_evaluation_reduction",
        "point_rejectable_p95_latency_reduction",
        "trajectory_completion_gap",
        "threshold_guard_point_evaluation_reduction",
    )
    by_robot: dict[str, Any] = {}
    for robot in ROBOTS:
        selected = [run for run in runs if run["robot"] == robot]
        by_robot[robot] = {
            "run_count": len(selected),
            "all_run_gates_pass": all(run["claim"]["pilot_gate_pass"] for run in selected),
            "metrics": {
                metric: {
                    "values": [run["claim"][metric] for run in selected],
                    "mean": float(np.mean([run["claim"][metric] for run in selected])),
                    "min": float(np.min([run["claim"][metric] for run in selected])),
                    "max": float(np.max([run["claim"][metric] for run in selected])),
                }
                for metric in metrics
            },
        }
    all_gates = all(run["claim"]["pilot_gate_pass"] for run in runs)
    direction = all(
        run["claim"]["point_feasible_success_gap"] >= -0.01
        and run["claim"]["point_feasible_evaluation_reduction"] > 0.0
        and run["claim"]["point_rejectable_evaluation_reduction"] > 0.0
        and run["claim"]["threshold_guard_point_evaluation_reduction"] > 0.0
        for run in runs
    )
    paper_gate = {
        "protocol": "v3 separate-estimand confirmatory aggregation",
        "robots": list(ROBOTS),
        "training_seeds": list(TRAINING_SEEDS),
        "expected_run_count": 6,
        "observed_run_count": len(runs),
        "all_run_gates_pass": all_gates,
        "effect_direction_consistent": direction,
        "paper_gate_pass": len(runs) == 6 and all_gates and direction,
        "by_robot": by_robot,
        "formal_feasible_latency_gate": "proposed feasible P95 / baseline feasible P95 <= 1.25",
        "validation_readiness_1_15_used_as_paper_gate": False,
        "training_seed_statistical_role": "sensitivity replicate, not an independent query sample",
    }
    aggregate_summary = {
        "runs": [
            {
                "robot": run["robot"],
                "training_seed": run["seed"],
                "root": run["root"],
                "claim_gate": run["claim"],
                "method_metrics": run["summary"]["method_metrics"],
                "primary_latency": run["latency"]["paired"].get(f"{PRIMARY_BACKEND}/point_feasible"),
            }
            for run in runs
        ],
        "by_robot": by_robot,
    }
    consistency = {
        "by_robot": by_robot,
        "all_six_natural_exits": True,
        "shared_queries_within_robot": True,
        "training_seeds_treated_as_independent_queries": False,
        "effect_direction_consistent": direction,
    }
    _write_json(aggregate_dir / "paper_gate_v3.json", paper_gate)
    _write_json(aggregate_dir / "aggregate_summary_v3.json", aggregate_summary)
    _write_json(aggregate_dir / "cross_seed_consistency_v3.json", consistency)
    return paper_gate, aggregate_summary, consistency


def _run(config_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("test_v3 config must be a mapping")
    if tuple(config["robots"]) != ROBOTS or tuple(config["training_seeds"]) != TRAINING_SEEDS:
        raise RuntimeError("test_v3 robot/seed design differs from preregistration")
    if tuple(config["methods"]) != METHODS:
        raise RuntimeError("test_v3 method set differs from paper_v2")
    if {key: float(value) for key, value in config["claim_gate"].items()} != LOCKED_GATE:
        raise RuntimeError("test_v3 claim gate differs from frozen paper_v2 definitions")
    if config["backend"] != PRIMARY_BACKEND:
        raise RuntimeError("test_v3 backend must remain torchscript_exact")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["runtime"]["cuda_visible_devices"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly locked to 0")
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    if torch.get_num_threads() != 8 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("locked thread configuration is not active")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Quadro RTX 8000":
        raise RuntimeError("locked Quadro RTX 8000 is unavailable")
    status = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"formal test_v3 requires a clean Git worktree:\n{status}")

    source_config_path = (config_path.parent / config["source_config"]).resolve()
    source_config = load_config(source_config_path)
    if int(config["data"]["hard_screening_easy_iterations"]) != int(source_config["cascade"]["easy_iterations"]):
        raise RuntimeError("hard-valid easy screening budget differs from paper_v2")
    if int(config["data"]["hard_screening_robust_iterations"]) != int(source_config["cascade"]["hard_iterations"]):
        raise RuntimeError("hard-valid robust screening budget differs from paper_v2")
    release_root = (config_path.parent / config["release_directory"]).resolve()
    release_equivalence = json.loads((release_root / "release_equivalence.json").read_text(encoding="utf-8"))
    release_manifest = json.loads((release_root / "release_manifest.json").read_text(encoding="utf-8"))
    if not release_equivalence["all_six_pass"] or release_manifest["backend"] != PRIMARY_BACKEND:
        raise RuntimeError("release_v3_locked is not eligible for formal testing")
    if int(release_manifest["artifact_count"]) != 48:
        raise RuntimeError("release artifact count differs from the sealed six-combination package")
    for artifact in release_manifest["artifacts"]:
        path = workspace / str(artifact["path"])
        if not path.is_file() or _sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"release artifact hash mismatch: {path}")
    release_environment = json.loads((release_root / "release_environment.json").read_text(encoding="utf-8"))
    environment = _current_environment(release_environment)
    output_root = workspace / "outputs"
    aggregate_dir = (config_path.parent / config["aggregate_directory"]).resolve()
    expected_aggregate = output_root / "test_v3_aggregate"
    if aggregate_dir != expected_aggregate:
        raise RuntimeError("aggregate output path differs from the locked location")
    forbidden = [aggregate_dir] + [output_root / f"test_v3_seed{seed}" for seed in TRAINING_SEEDS]
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise RuntimeError("test_v3 output already exists; a second formal run is forbidden")

    protected_before = _snapshot(output_root, [str(value) for value in config["protected_outputs"]])
    source_assets = _asset_manifest(workspace, release_root)
    aggregate_dir.mkdir(parents=False)
    prereg = _preregistration(
        workspace=workspace,
        config_path=config_path,
        config=config,
        release_root=release_root,
        release_manifest=release_manifest,
        source_assets=source_assets,
    )
    prereg_path = aggregate_dir / "test_v3_preregistration.json"
    _write_json(prereg_path, prereg)
    prereg_hash = _sha256_file(prereg_path)
    print(f"[test-v3] preregistration frozen sha256={prereg_hash}", flush=True)

    datasets_dir = aggregate_dir / "datasets"
    datasets_dir.mkdir()
    dataset_payload: dict[str, Any] = {}
    release_commit = str(release_manifest["git_commit"])
    for robot in ROBOTS:
        kinematics = load_robot(source_config, robot)
        screening_seed = int(config["data"]["hard_screening_release_seed"])
        screen_paths = _release_paths(release_root, robot, screening_seed)
        screening_dls, screening_verifier, _, _, _ = _solver_components(
            source_config,
            {"solver_metadata": screen_paths["solver_metadata"], "seed_bank": screen_paths["seed_bank"]},
            kinematics,
        )
        dataset, generation_seeds = generate_locked_dataset(
            kinematics=kinematics,
            dls=screening_dls,
            verifier=screening_verifier,
            release_commit=release_commit,
            robot=robot,
            config=config,
        )
        schema = dataset_schema(dataset)
        validate_schema(schema, config)
        audit = split_audit(dataset, comparison_files=_comparison_files(workspace, robot))
        if not audit["passed"]:
            raise RuntimeError(f"test_v3 split audit failed for {robot}: {audit}")
        path = datasets_dir / f"{robot}_test_v3_queries.npz"
        dataset.save(path)
        dataset_payload[robot] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
            "generation_seeds": generation_seeds,
            "schema": schema,
            "overlap_audit": audit,
        }
    dataset_manifest = {
        "protocol": "test_v3 locked dataset manifest",
        "frozen_utc": _utc(),
        "release_commit": release_commit,
        "preregistration_sha256": prereg_hash,
        "generation_parameters": config["data"],
        "datasets": dataset_payload,
        "method_outputs_inspected_before_freeze": False,
        "solver_performance_inspected_before_freeze": False,
        "all_schema_and_overlap_gates_pass": True,
    }
    dataset_manifest_path = aggregate_dir / "test_v3_dataset_manifest.json"
    _write_json(dataset_manifest_path, dataset_manifest)
    dataset_manifest_hash = _sha256_file(dataset_manifest_path)
    print(f"[test-v3] datasets frozen sha256={dataset_manifest_hash}", flush=True)

    for seed in TRAINING_SEEDS:
        staging = output_root / f".test_v3_seed{seed}.incomplete.{os.getpid()}"
        if staging.exists():
            raise RuntimeError(f"staging path already exists: {staging}")
        staging.mkdir()
        for robot in ROBOTS:
            print(f"[test-v3] running {robot}/seed{seed}", flush=True)
            dataset = QueryDataset.load(Path(dataset_payload[robot]["path"]))
            _run_combination(
                workspace=workspace,
                destination=staging / robot,
                source_config=source_config,
                config=config,
                release_root=release_root,
                robot=robot,
                seed=seed,
                dataset=dataset,
                dataset_manifest=dataset_manifest,
                environment=environment,
                preregistration_sha256=prereg_hash,
            )
            print(f"[test-v3] completed {robot}/seed{seed} (metrics sealed, not aggregated)", flush=True)
        staging.rename(output_root / f"test_v3_seed{seed}")

    # Paper-level conclusions are computed only after all six runs have
    # naturally exited and all three seed directories have been sealed.
    paper_gate, _, _ = _aggregate(workspace, aggregate_dir)
    protected_after = _snapshot(output_root, [str(value) for value in config["protected_outputs"]])
    if protected_before != protected_after:
        raise RuntimeError("a protected v2/validation/release output changed during test_v3")
    output_files = sorted(
        path
        for root in [aggregate_dir] + [output_root / f"test_v3_seed{seed}" for seed in TRAINING_SEEDS]
        for path in root.rglob("*")
        if path.is_file() and path.name != "test_v3_final_manifest.json"
    )
    final_manifest = {
        "protocol": "test_v3 final immutable evidence manifest",
        "completed_utc": _utc(),
        "release_commit": release_commit,
        "runner_git_commit": _git(workspace, "rev-parse", "HEAD"),
        "preregistration_sha256": prereg_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "formal_run_count": 6,
        "all_six_natural_exits": True,
        "technical_retry_count": 0,
        "test_set_retuning_performed": False,
        "threshold_or_gate_changes_after_test": False,
        "outliers_removed": False,
        "winsorization_performed": False,
        "protected_outputs": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": True,
        },
        "files": [
            {
                "path": str(path.relative_to(workspace)),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in output_files
        ],
        "paper_gate_pass": paper_gate["paper_gate_pass"],
    }
    _write_json(aggregate_dir / "test_v3_final_manifest.json", final_manifest)
    print("[test-v3] all six formal runs sealed; aggregate complete", flush=True)


def main() -> None:
    args = _parser().parse_args()
    _run(Path(args.config).resolve())


if __name__ == "__main__":
    main()
