"""One-shot formal runner for the validation-frozen CG-HIK v4 release.

The runner freezes its preregistration before generating fresh queries, keeps
all method calls interleaved and disk-free inside each timing interval, and
never consumes a performance field from an older formal test.  A failed run is
preserved under an ``.incomplete`` path for audit; overwrite and automatic
retry are intentionally unsupported.
"""

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
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot
from ..data.datasets import QueryDataset
from ..experiments.baselines import TRFOnlyMethod, fixed_hybrid
from ..experiments.baselines_v2 import ThresholdGuardConfig, ThresholdGuardRiskProvider
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime
from ..latency_pilot_v3.runner import _build_runtimes, _load_gate_config, _solver_components
from ..models.seed import PreviousStateCandidates
from ..release_v3_locked.artifacts import load_frozen_risk, load_locked_seed_engine
from ..release_v4_locked.artifacts import (
    FrozenV4Policy,
    TorchScriptV4Inference,
    load_exact_v4_predictor,
    load_policy_config,
)
from ..runtime.cascade import CalibratedActionGate
from ..counterfactual_v4.runtime_v4 import wrap_profiled_runtime
from ..types import CandidateSet, IKQuery
from .benchmark import (
    PRIMARY_METHODS,
    SENSITIVITY_METHODS,
    benchmark_role,
    warmup_methods,
)
from .data import (
    TEST_V4_ROLES,
    audit_freshness,
    dataset_contract,
    default_comparison_sources,
    derive_seed,
    generate_locked_datasets,
    validate_dataset_contract,
)
from .reporting import (
    claim_gate_report,
    method_metrics,
    ood_and_abstention_metrics,
    paired_confirmatory_intervals,
)


PROTOCOL = "test_v4_locked"
ROBOTS = ("panda", "ur5e")
PRIMARY_SEED = 17
SENSITIVITY_SEEDS = (29, 43)
ALL_SEEDS = (17, 29, 43)


class LockedCandidates:
    def __init__(self, engine: object):
        self.engine = engine
        self.kinematics = engine.kinematics

    def candidates(self, query: IKQuery) -> CandidateSet:
        return self.engine.prepare(query).candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the one-shot locked test_v4 protocol")
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
    """Write one JSON artifact atomically outside any timing interval."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML document must contain a mapping: {path}")
    return payload


def _git(workspace: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=workspace, check=True, capture_output=True, text=True
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
        raise RuntimeError("release artifact manifest has no file mapping")
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"frozen release artifact is missing: {path}")
        if path.stat().st_size != int(expected["size"]):
            raise RuntimeError(f"frozen release artifact size changed: {path}")
        if _sha256_file(path) != str(expected["sha256"]):
            raise RuntimeError(f"frozen release artifact hash changed: {path}")


def _verify_release(release_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = release_root / "release_manifest.json"
    artifact_path = release_root / "artifact_manifest.json"
    equivalence_path = release_root / "release_equivalence.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != "release_v4_locked"
        or manifest.get("release_status") != "sealed"
        or manifest.get("backend") != "torchscript_exact_v4"
        or not bool(manifest.get("all_six_validation_runtime_equivalence_pass", False))
        or bool(manifest.get("formal_test_authorized_or_started", True))
        or bool(manifest.get("test_v4_started", True))
        or not bool(equivalence.get("all_pass", False))
    ):
        raise RuntimeError("release_v4_locked is not eligible for one-shot formal testing")
    if _sha256_file(artifact_path) != str(manifest["artifact_manifest_sha256"]):
        raise RuntimeError("release_v4 artifact manifest hash differs from its seal")
    _verify_manifest_files(release_root, artifacts)
    return manifest, artifacts


def _validate_environment(release_root: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    release = json.loads(
        (release_root / "release_environment.json").read_text(encoding="utf-8")
    )
    current = environment_payload()
    current.update(
        {
            "captured_utc": _utc(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment_variables": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        }
    )
    fields = (
        "hostname",
        "gpu",
        "torch",
        "cuda_version",
        "pinocchio",
        "scikit_learn",
        "python_executable",
    )
    matches = {field: current.get(field) == release.get(field) for field in fields}
    matches["intra_op_threads"] = torch.get_num_threads() == int(runtime["intra_op_threads"])
    matches["inter_op_threads"] = torch.get_num_interop_threads() == int(runtime["inter_op_threads"])
    current["release_environment_comparison"] = {
        "matches": matches,
        "all_match": all(matches.values()),
    }
    if not all(matches.values()):
        raise RuntimeError(f"formal test environment differs from release: {matches}")
    return current


def _source_scope_status(workspace: Path) -> dict[str, Any]:
    scope = [
        "src/confik",
        "configs/paper_v2.yaml",
        "configs/test_v4_locked.yaml",
        "scripts/run_test_v4_locked.sh",
        "tests/test_test_v4_benchmark.py",
        "tests/test_test_v4_reporting.py",
        "tests/test_test_v4_runner.py",
    ]
    status = _git(
        workspace,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scope,
    )
    payload = {
        "git_commit": _git(workspace, "rev-parse", "HEAD"),
        "git_tree": _git(workspace, "rev-parse", "HEAD^{tree}"),
        "scope": scope,
        "scope_clean": not bool(status),
        "status": status.splitlines(),
    }
    if status:
        raise RuntimeError(f"formal test_v4 source scope is dirty:\n{status}")
    return payload


def _v3_paths(release_v3_root: Path, robot: str, seed: int) -> dict[str, Path]:
    root = release_v3_root / robot / f"seed{seed}"
    names = {
        "torchscript": "exact_seed_ensemble.ts",
        "forest": "hgb_vectorized_parameters.npz",
        "calibration": "isotonic_calibration_parameters.npz",
        "normalization": "normalization_parameters.npz",
        "solver_metadata": "solver_metadata.json",
        "seed_bank": "seed_bank.npz",
        "runtime_spec": "runtime_spec.json",
    }
    result = {key: root / value for key, value in names.items()}
    for path in result.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    return result


def _source_paths(workspace: Path, robot: str, seed: int) -> dict[str, Path]:
    root = workspace / "outputs" / f"paper_v2_seed{seed}" / robot / "results"
    result = {
        "policy_selection": root / "policy_selection_v2.json",
        "threshold_guard": root / "threshold_guard.json",
    }
    for path in result.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    return result


def _release_v3_root(release_v4_root: Path, workspace: Path) -> Path:
    dependencies = json.loads(
        (release_v4_root / "upstream_dependencies.json").read_text(encoding="utf-8")
    )
    root = Path(str(dependencies["release_v3_locked"]["root"])).resolve()
    expected = (workspace / "outputs" / "release_v3_locked").resolve()
    if root != expected:
        raise RuntimeError("frozen v4 dependency points outside release_v3_locked")
    manifest = root / "release_manifest.json"
    if _sha256_file(manifest) != str(
        dependencies["release_v3_locked"]["release_manifest_sha256"]
    ):
        raise RuntimeError("upstream v3 release manifest changed after v4 sealing")
    return root


def _build_methods(
    *,
    workspace: Path,
    source_config: dict[str, Any],
    release_v4_root: Path,
    release_v3_root: Path,
    robot: str,
    seed: int,
) -> dict[str, object]:
    kinematics = load_robot(source_config, robot)
    paths = _v3_paths(release_v3_root, robot, seed)
    sources = _source_paths(workspace, robot, seed)
    gate_config, _ = _load_gate_config(sources["policy_selection"])
    seed_engine = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=paths["torchscript"],
        normalization_path=paths["normalization"],
        runtime_spec_path=paths["runtime_spec"],
        device="cuda:0",
    )
    risk_engine = load_frozen_risk(paths["forest"], paths["calibration"])
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    profiled_pair = _build_runtimes(
        kinematics=kinematics,
        gate_config=gate_config,
        engines={"torchscript_exact": (seed_engine, risk_engine, True)},
        dls=dls,
        verifier=verifier,
        fallback=fallback,
        seed_bank=seed_bank,
        cascade=cascade,
    )["torchscript_exact"]
    fixed_runtime, proposed_v2_runtime = profiled_pair

    policy_config, _ = load_policy_config(release_v4_root / robot / "v4_policy.json")
    v4_policy = FrozenV4Policy(
        TorchScriptV4Inference(
            load_exact_v4_predictor(
                release_v4_root / robot / "exact_v4_predictor.ts", device="cpu"
            )
        ),
        policy_config,
    )
    v4_runtime = wrap_profiled_runtime(
        name="proposed_v4_torchscript_exact",
        policy=v4_policy,  # type: ignore[arg-type]
        kinematics=kinematics,
        seed_engine=seed_engine,
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
    )
    methods: dict[str, object] = {
        "fixed_robust_cascade": fixed_runtime,
        "proposed_v2": proposed_v2_runtime,
        "proposed_v4": v4_runtime,
    }
    if seed != PRIMARY_SEED:
        return methods

    threshold_values = json.loads(sources["threshold_guard"].read_text(encoding="utf-8"))
    threshold_provider = ThresholdGuardRiskProvider(
        ThresholdGuardConfig(**threshold_values)
    )
    threshold_runtime = ProfiledCascadeRuntime(
        name="threshold_guard_cascade",
        kinematics=kinematics,
        seed_engine=seed_engine,
        risk_engine=threshold_provider,  # type: ignore[arg-type]
        gate=CalibratedActionGate(gate_config),
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
        reuse_candidate_features=True,
    )
    candidates = LockedCandidates(seed_engine)
    methods.update(
        {
            "threshold_guard_cascade": threshold_runtime,
            "learned_1x25": fixed_hybrid(
                kinematics,
                candidates,
                dls,
                verifier,
                candidate_count=1,
                iterations=25,
            ),
            "dls_previous_1x50": fixed_hybrid(
                kinematics,
                PreviousStateCandidates(),
                dls,
                verifier,
                candidate_count=1,
                iterations=50,
            ),
            "trf_previous": TRFOnlyMethod(fallback, verifier),
        }
    )
    return {name: methods[name] for name in PRIMARY_METHODS}


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _safe(record),
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def _run_combination(
    *,
    workspace: Path,
    destination: Path,
    source_config: dict[str, Any],
    config: Mapping[str, Any],
    release_v4_root: Path,
    release_v3_root: Path,
    release_digest: str,
    robot: str,
    seed: int,
    datasets: Mapping[str, QueryDataset],
    preregistration_sha256: str,
    dataset_manifest_sha256: str,
    environment: Mapping[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    methods = _build_methods(
        workspace=workspace,
        source_config=source_config,
        release_v4_root=release_v4_root,
        release_v3_root=release_v3_root,
        robot=robot,
        seed=seed,
    )
    expected_methods = PRIMARY_METHODS if seed == PRIMARY_SEED else SENSITIVITY_METHODS
    if tuple(methods) != expected_methods:
        raise RuntimeError(
            f"formal method order differs for seed {seed}: {tuple(methods)}"
        )
    timing = config["timing"]
    dt = float(config["data"]["dt"])
    warmup_methods(
        methods,
        datasets["id_points"],
        iterations=int(timing["warmup_iterations"]),
        dt=dt,
        synchronize_cuda=True,
    )
    repeats = {
        name: (
            int(timing["primary_point_repeats"])
            if name in SENSITIVITY_METHODS
            else int(timing["comparator_point_repeats"])
        )
        for name in methods
    }
    record_dir = destination / "role_records"
    record_dir.mkdir()
    role_files: list[Path] = []
    order_root = derive_seed(release_digest, robot, f"method_order_seed{seed}")
    for role_index, role in enumerate(TEST_V4_ROLES):
        print(f"[test-v4] timing {robot}/seed{seed}/{role}", flush=True)
        # benchmark_role contains no disk operation.  Serialization begins
        # only after every method has completed this explicit role.
        records = benchmark_role(
            robot=robot,
            training_seed=seed,
            role=role,
            methods=methods,
            dataset=datasets[role],
            repeats_by_method=repeats,
            dt=dt,
            order_seed=order_root + role_index,
            synchronize_cuda=True,
        )
        role_path = record_dir / f"{role}.jsonl.gz"
        _write_records(role_path, records)
        role_files.append(role_path)
        del records

    records = _read_records(role_files)
    summary = {
        "protocol": PROTOCOL,
        "robot": robot,
        "training_seed": seed,
        "seed_statistical_role": (
            "primary confirmatory" if seed == PRIMARY_SEED else "sensitivity only"
        ),
        "methods": list(methods),
        "method_metrics": method_metrics(records),
        "record_count": len(records),
        "expected_record_count": sum(len(dataset) for dataset in datasets.values())
        * len(methods),
    }
    _write_json(destination / "summary_v4.json", summary)
    if seed == PRIMARY_SEED:
        bootstrap_seed = derive_seed(release_digest, robot, "bootstrap")
        intervals = paired_confirmatory_intervals(
            records,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=bootstrap_seed,
            gates={key: float(value) for key, value in config["claim_gate"].items()},
        )
        abstention = ood_and_abstention_metrics(records)
        claim = claim_gate_report(
            records,
            gates={key: float(value) for key, value in config["claim_gate"].items()},
            intervals=intervals,
        )
        _write_json(destination / "paired_intervals_v4.json", intervals)
        _write_json(destination / "ood_abstention_v4.json", abstention)
        _write_json(destination / "claim_gate_v4.json", claim)

    protocol = {
        "protocol": PROTOCOL,
        "robot": robot,
        "training_seed": seed,
        "preregistration_sha256": preregistration_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "release_digest": release_digest,
        "timing": config["timing"],
        "raw_point_repeats_retained": True,
        "same_query_interleaving": True,
        "disk_writes_inside_timed_interval": False,
        "explicit_roles": list(TEST_V4_ROLES),
        "hysteresis": config["runtime"]["hysteresis"],
        "old_test_performance_results_read": False,
        "test_parameter_selection_performed": False,
        "technical_retry_count": 0,
        "outliers_removed": False,
        "winsorization_performed": False,
    }
    _write_json(destination / "protocol_manifest_v4.json", protocol)
    _write_json(destination / "environment_v4.json", environment)
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
        )
        + "\n",
        encoding="utf-8",
    )


def _aggregate(
    *, workspace: Path, aggregate_staging: Path
) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for robot in ROBOTS:
        root = workspace / "outputs" / f"test_v4_seed{PRIMARY_SEED}" / robot
        primary[robot] = {
            "claim_gate": json.loads((root / "claim_gate_v4.json").read_text(encoding="utf-8")),
            "summary": json.loads((root / "summary_v4.json").read_text(encoding="utf-8")),
            "ood_abstention": json.loads((root / "ood_abstention_v4.json").read_text(encoding="utf-8")),
        }
        sensitivity[robot] = {}
        for seed in SENSITIVITY_SEEDS:
            sensitivity[robot][f"seed{seed}"] = json.loads(
                (
                    workspace
                    / "outputs"
                    / f"test_v4_seed{seed}"
                    / robot
                    / "summary_v4.json"
                ).read_text(encoding="utf-8")
            )
    paper_gate = {
        "protocol": "test_v4 robot-level confirmatory aggregation",
        "primary_training_seed": PRIMARY_SEED,
        "sensitivity_training_seeds": list(SENSITIVITY_SEEDS),
        "sensitivity_seeds_are_not_independent_query_samples": True,
        "robot_gates": {
            robot: bool(primary[robot]["claim_gate"]["formal_gate_pass"])
            for robot in ROBOTS
        },
        "both_robot_gates_pass": all(
            bool(primary[robot]["claim_gate"]["formal_gate_pass"])
            for robot in ROBOTS
        ),
        "test_set_retuning_performed": False,
    }
    _write_json(aggregate_staging / "paper_gate_v4.json", paper_gate)
    _write_json(
        aggregate_staging / "aggregate_summary_v4.json",
        {"primary": primary, "sensitivity": sensitivity},
    )
    return paper_gate


def _baseline_availability() -> dict[str, Any]:
    return {
        "fixed_robust_cascade": {"available": True},
        "proposed_v2": {"available": True, "status": "frozen exact v3 deployment"},
        "threshold_guard_cascade": {"available": True},
        "learned_1x25": {"available": True},
        "dls_previous_1x50": {"available": True},
        "trf_previous": {
            "available": True,
            "implementation": "SciPy trust-region reflective fallback from previous state",
            "is_trac_ik": False,
        },
        "trac_ik": {
            "available": False,
            "reason": "No pinned, reproducible TRAC-IK binary/library exists in the frozen environment.",
            "substitution_claimed": False,
        },
        "proposed_v4": {"available": True, "status": "frozen exact TorchScript v4"},
    }


def run(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _read_yaml(path)
    workspace = Path(__file__).resolve().parents[3]
    if config.get("protocol_version") != PROTOCOL:
        raise RuntimeError("test_v4 protocol version differs from locked code")
    if tuple(config.get("robots", ())) != ROBOTS:
        raise RuntimeError("formal robot set differs from preregistration")
    if int(config.get("primary_training_seed")) != PRIMARY_SEED or tuple(
        int(value) for value in config.get("sensitivity_training_seeds", ())
    ) != SENSITIVITY_SEEDS:
        raise RuntimeError("formal seed roles differ from preregistration")
    if tuple(config["methods"]["primary"]) != PRIMARY_METHODS:
        raise RuntimeError("primary seven-method set differs from preregistration")
    if tuple(config["methods"]["sensitivity_only"]) != SENSITIVITY_METHODS:
        raise RuntimeError("sensitivity method set differs from preregistration")
    if tuple(config["data"]["roles"]) != TEST_V4_ROLES:
        raise RuntimeError("explicit test_v4 roles differ from preregistration")
    if int(config["statistics"]["bootstrap_samples"]) != 10_000:
        raise RuntimeError("formal bootstrap count must remain 10,000")
    if str(config["statistics"]["multiplicity_correction"]).lower() != "holm":
        raise RuntimeError("formal multiplicity correction must remain Holm")
    if bool(config["runtime"]["hysteresis"]["enabled"]):
        raise RuntimeError("trajectory hysteresis was disabled before fresh testing")

    runtime = config["runtime"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(runtime["cuda_visible_devices"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly locked to 0")
    torch.set_num_threads(int(runtime["intra_op_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["inter_op_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(runtime["inter_op_threads"]):
            raise
    if not torch.cuda.is_available():
        raise RuntimeError("formal seed deployment requires cuda:0")

    source_manifest = _source_scope_status(workspace)
    release_root = (path.parent / str(config["release_directory"])).resolve()
    expected_release = (workspace / "outputs" / "release_v4_locked").resolve()
    if release_root != expected_release:
        raise RuntimeError("formal v4 release path differs from locked location")
    release_manifest, _ = _verify_release(release_root)
    release_digest = str(release_manifest["release_digest"])
    release_v3_root = _release_v3_root(release_root, workspace)
    environment = _validate_environment(release_root, runtime)
    source_config = load_config((path.parent / str(config["source_config"])).resolve())

    output_root = workspace / "outputs"
    aggregate = (path.parent / str(config["aggregate_directory"])).resolve()
    if aggregate != (output_root / "test_v4_aggregate").resolve():
        raise RuntimeError("formal aggregate path differs from locked location")
    destinations = [aggregate] + [output_root / f"test_v4_seed{seed}" for seed in ALL_SEEDS]
    if any(target.exists() or target.is_symlink() for target in destinations):
        raise RuntimeError("test_v4 output already exists; a second formal run is forbidden")
    stale_staging = list(output_root.glob(".test_v4_*.incomplete.*"))
    if stale_staging:
        raise RuntimeError(f"an earlier incomplete formal run requires audit: {stale_staging}")

    protected_before = _tree_snapshot(
        output_root, [str(value) for value in config["protected_outputs"]]
    )
    aggregate_staging = output_root / f".test_v4_aggregate.incomplete.{os.getpid()}"
    aggregate_staging.mkdir(parents=False)
    preregistration = {
        "protocol": PROTOCOL,
        "created_utc": _utc(),
        "release_digest": release_digest,
        "release_manifest_sha256": _sha256_file(release_root / "release_manifest.json"),
        "runner_git_commit": source_manifest["git_commit"],
        "runner_git_tree": source_manifest["git_tree"],
        "runner_sha256": _sha256_file(Path(__file__)),
        "benchmark_sha256": _sha256_file(Path(__file__).with_name("benchmark.py")),
        "reporting_sha256": _sha256_file(Path(__file__).with_name("reporting.py")),
        "config_sha256": _sha256_file(path),
        "source_manifest": source_manifest,
        "robots": list(ROBOTS),
        "primary_training_seed": PRIMARY_SEED,
        "sensitivity_training_seeds": list(SENSITIVITY_SEEDS),
        "methods": config["methods"],
        "data": config["data"],
        "timing": config["timing"],
        "statistics": config["statistics"],
        "claim_gate": config["claim_gate"],
        "baseline_availability": _baseline_availability(),
        "old_test_access_contract": {
            "identity_arrays_only": [
                "previous_q",
                "target_position",
                "target_rotation",
            ],
            "performance_results_allowed": False,
        },
        "failure_policy": {
            "automatic_rerun_allowed": False,
            "post_test_tuning_allowed": False,
            "outlier_removal_allowed": False,
            "winsorization_allowed": False,
        },
    }
    prereg_path = aggregate_staging / "test_v4_preregistration.json"
    _write_json(prereg_path, preregistration)
    prereg_hash = _sha256_file(prereg_path)
    _write_json(aggregate_staging / "baseline_availability.json", _baseline_availability())
    print(f"[test-v4] preregistration frozen sha256={prereg_hash}", flush=True)

    dataset_root = aggregate_staging / "datasets"
    dataset_root.mkdir()
    datasets_by_robot: dict[str, dict[str, QueryDataset]] = {}
    dataset_payload: dict[str, Any] = {}
    for robot in ROBOTS:
        kinematics = load_robot(source_config, robot)
        screening_paths = _v3_paths(release_v3_root, robot, PRIMARY_SEED)
        screening_dls, screening_verifier, _, _, _ = _solver_components(
            source_config,
            {
                "solver_metadata": screening_paths["solver_metadata"],
                "seed_bank": screening_paths["seed_bank"],
            },
            kinematics,
        )
        datasets, seeds = generate_locked_datasets(
            kinematics=kinematics,
            dls=screening_dls,
            verifier=screening_verifier,
            release_v4_digest=release_digest,
            robot=robot,
            config=config,
        )
        contract = dataset_contract(
            datasets, robot=robot, dt=float(config["data"]["dt"])
        )
        validate_dataset_contract(contract, config)
        freshness = audit_freshness(
            datasets,
            robot=robot,
            dt=float(config["data"]["dt"]),
            comparison_sources=default_comparison_sources(workspace, robot),
        )
        if not freshness["passed"]:
            raise RuntimeError(f"fresh test_v4 overlap audit failed for {robot}")
        datasets_by_robot[robot] = datasets
        role_files: dict[str, Any] = {}
        for role in TEST_V4_ROLES:
            destination = dataset_root / f"{robot}_{role}.npz"
            datasets[role].save(destination)
            role_files[role] = {
                "path": str(destination.relative_to(aggregate_staging)),
                "sha256": _sha256_file(destination),
                "size": destination.stat().st_size,
            }
        dataset_payload[robot] = {
            "seeds": seeds,
            "contract": contract,
            "freshness": freshness,
            "roles": role_files,
        }
    dataset_manifest = {
        "protocol": "test_v4 locked fresh dataset manifest",
        "created_utc": _utc(),
        "release_digest": release_digest,
        "preregistration_sha256": prereg_hash,
        "robots": dataset_payload,
        "method_outputs_inspected_before_freeze": False,
        "solver_performance_inspected_before_freeze": False,
        "old_test_performance_results_read": False,
        "all_contract_and_freshness_gates_pass": True,
    }
    dataset_manifest_path = aggregate_staging / "test_v4_dataset_manifest.json"
    _write_json(dataset_manifest_path, dataset_manifest)
    dataset_manifest_hash = _sha256_file(dataset_manifest_path)
    print(f"[test-v4] fresh datasets frozen sha256={dataset_manifest_hash}", flush=True)

    for seed in ALL_SEEDS:
        seed_staging = output_root / f".test_v4_seed{seed}.incomplete.{os.getpid()}"
        seed_staging.mkdir(parents=False)
        for robot in ROBOTS:
            print(f"[test-v4] starting {robot}/seed{seed}", flush=True)
            _run_combination(
                workspace=workspace,
                destination=seed_staging / robot,
                source_config=source_config,
                config=config,
                release_v4_root=release_root,
                release_v3_root=release_v3_root,
                release_digest=release_digest,
                robot=robot,
                seed=seed,
                datasets=datasets_by_robot[robot],
                preregistration_sha256=prereg_hash,
                dataset_manifest_sha256=dataset_manifest_hash,
                environment=environment,
            )
            print(f"[test-v4] sealed metrics {robot}/seed{seed}", flush=True)
        seed_staging.rename(output_root / f"test_v4_seed{seed}")

    paper_gate = _aggregate(workspace=workspace, aggregate_staging=aggregate_staging)
    protected_after = _tree_snapshot(
        output_root, [str(value) for value in config["protected_outputs"]]
    )
    if protected_before != protected_after:
        raise RuntimeError("a protected evidence tree changed during formal test_v4")

    logical_files: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        root = output_root / f"test_v4_seed{seed}"
        for item in sorted(path for path in root.rglob("*") if path.is_file()):
            logical_files.append(
                {
                    "path": str(item.relative_to(workspace)),
                    "sha256": _sha256_file(item),
                    "size": item.stat().st_size,
                }
            )
    for item in sorted(
        path for path in aggregate_staging.rglob("*")
        if path.is_file() and path.name != "test_v4_final_manifest.json"
    ):
        logical_files.append(
            {
                "path": str(Path("outputs/test_v4_aggregate") / item.relative_to(aggregate_staging)),
                "sha256": _sha256_file(item),
                "size": item.stat().st_size,
            }
        )
    final_manifest = {
        "protocol": "test_v4 final immutable evidence manifest",
        "completed_utc": _utc(),
        "release_digest": release_digest,
        "preregistration_sha256": prereg_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "formal_primary_runs": 2,
        "sensitivity_runs": 4,
        "all_six_natural_exits": True,
        "technical_retry_count": 0,
        "test_set_retuning_performed": False,
        "threshold_or_gate_changes_after_test": False,
        "outliers_removed": False,
        "winsorization_performed": False,
        "old_test_performance_results_read": False,
        "protected_outputs": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": True,
        },
        "paper_gate_pass": bool(paper_gate["both_robot_gates_pass"]),
        "files": logical_files,
    }
    _write_json(aggregate_staging / "test_v4_final_manifest.json", final_manifest)
    aggregate_staging.rename(aggregate)
    print("[test-v4] all formal and sensitivity runs sealed", flush=True)
    return final_manifest


def main() -> None:
    arguments = _parser().parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()


__all__ = ["run"]
