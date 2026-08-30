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
import time
import traceback
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
    dataset_query_hashes,
    default_comparison_sources,
    derive_seed,
    generate_locked_datasets,
    validate_dataset_contract,
)
from .evidence_protocol import (
    ExclusiveRunLock,
    assert_evidence_fingerprint,
    evidence_fingerprint,
)
from .host_guard import FormalHostGuard, QuietHostTechnicalInterruption
from .reporting import (
    claim_gate_report,
    joint_holm_confirmatory,
    method_metrics,
    ood_and_abstention_metrics,
    paired_confirmatory_intervals,
)


PROTOCOL = "test_v4_locked"
ROBOTS = ("panda", "ur5e")
PRIMARY_SEED = 17
SENSITIVITY_SEEDS = (29, 43)
ALL_SEEDS = (17, 29, 43)
SOURCE_SCOPE = (
    "src/confik",
    "configs/paper_v2.yaml",
    "configs/test_v4_locked.yaml",
    "scripts/run_test_v4_locked.sh",
    "tests/test_test_v4_benchmark.py",
    "tests/test_test_v4_data.py",
    "tests/test_test_v4_reporting.py",
    "tests/test_test_v4_runner.py",
    "tests/test_test_v4_host_guard.py",
)

_ACTIVE_PHASE: dict[str, Any] = {"phase": "not_started"}
_ACTIVE_HOST_GUARD: FormalHostGuard | None = None
_ACTIVE_RUN_LOCK: ExclusiveRunLock | None = None
_ACTIVE_CONTROL_PLANE_SEAL_SHA256: str | None = None


class LockedCandidates:
    def __init__(self, engine: object):
        self.engine = engine
        self.kinematics = engine.kinematics

    def candidates(self, query: IKQuery) -> CandidateSet:
        return self.engine.prepare(query).candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the one-shot locked test_v4 protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume atomic technical checkpoints under the identical frozen evidence contract",
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
        # ``czy`` is a user-supplied evidence bundle at workspace root rather
        # than under outputs/. Supporting this one explicit directory name
        # keeps it under the same before/after protected-tree contract.
        if pattern == "czy":
            workspace_candidate = output_root.parent / pattern
            if workspace_candidate.is_dir():
                directories.add(workspace_candidate)

    def logical_path(path: Path) -> str:
        try:
            return str(path.relative_to(output_root))
        except ValueError:
            return str(Path("workspace") / path.relative_to(output_root.parent))

    entries: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected output cannot be a symlink: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            metadata = path.stat()
            entries[logical_path(path)] = {
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return {
        "directories": [logical_path(path) for path in sorted(directories)],
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


def _verify_release(
    release_root: Path,
    expected_lock: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    files = artifacts.get("files")
    if (
        not isinstance(files, Mapping)
        or int(artifacts.get("file_count", -1)) != len(files)
    ):
        raise RuntimeError("release_v4 artifact manifest file count is inconsistent")
    _verify_manifest_files(release_root, artifacts)
    if any(path.is_symlink() for path in release_root.rglob("*")):
        raise RuntimeError("release_v4 contains a symlink")
    allowed_unlisted = {"artifact_manifest.json", "release_manifest.json"}
    actual_files = {
        str(path.relative_to(release_root))
        for path in release_root.rglob("*")
        if path.is_file()
    }
    if actual_files != set(str(value) for value in files) | allowed_unlisted:
        raise RuntimeError("release_v4 contains missing or unexpected payload files")

    dependencies = json.loads(
        (release_root / "upstream_dependencies.json").read_text(encoding="utf-8")
    )
    candidate_digest = str(dependencies["candidate"]["release_digest"])
    upstream_v3_hash = str(
        dependencies["release_v3_locked"]["release_manifest_sha256"]
    )
    if candidate_digest != str(manifest.get("candidate_release_digest")):
        raise RuntimeError("release_v4 candidate digest differs from its dependency seal")
    if upstream_v3_hash != str(
        manifest.get("upstream_v3_release_manifest_sha256")
    ):
        raise RuntimeError("release_v4 upstream-v3 digest differs across manifests")
    calculated_digest = sha256(
        (
            _sha256_file(artifact_path)
            + candidate_digest
            + upstream_v3_hash
        ).encode("ascii")
    ).hexdigest()
    if calculated_digest != str(manifest.get("release_digest")):
        raise RuntimeError("release_v4 digest does not match the frozen packaging formula")
    if expected_lock is not None:
        observed_lock = {
            "release_digest": calculated_digest,
            "release_manifest_sha256": _sha256_file(manifest_path),
            "artifact_manifest_sha256": _sha256_file(artifact_path),
            "upstream_v3_release_manifest_sha256": upstream_v3_hash,
        }
        expected = {str(key): str(value) for key, value in expected_lock.items()}
        if observed_lock != expected:
            raise RuntimeError(
                "release_v4 differs from independent test_v4 control-plane anchors"
            )
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
    required_environment = {
        str(key): str(value)
        for key, value in runtime["required_environment_variables"].items()
    }
    matches["required_environment_variables"] = all(
        os.environ.get(key) == value for key, value in required_environment.items()
    )
    current["release_environment_comparison"] = {
        "matches": matches,
        "all_match": all(matches.values()),
    }
    if not all(matches.values()):
        raise RuntimeError(f"formal test environment differs from release: {matches}")
    return current


def _source_scope_status(workspace: Path) -> dict[str, Any]:
    scope = list(SOURCE_SCOPE)
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
    v3_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = v3_manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != int(v3_manifest.get("artifact_count", -1))
        or len({str(item.get("path")) for item in artifacts}) != len(artifacts)
    ):
        raise RuntimeError("upstream v3 release artifact index is malformed")
    for artifact in artifacts:
        target = (workspace / str(artifact["path"])).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise RuntimeError("upstream v3 artifact escapes the sealed release") from error
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(artifact["size"])
            or _sha256_file(target) != str(artifact["sha256"])
        ):
            raise RuntimeError(f"upstream v3 artifact changed after sealing: {target}")
    return root


def _configured_urdf_paths(source_config_path: Path) -> dict[str, Path]:
    source = _read_yaml(source_config_path)
    result: dict[str, Path] = {}
    for robot in ROBOTS:
        configured = Path(str(source["robots"][robot]["urdf"]))
        if not configured.is_absolute():
            configured = source_config_path.parent / configured
        if configured.is_symlink():
            raise RuntimeError(f"configured source URDF cannot be a symlink: {configured}")
        configured = configured.resolve()
        if not configured.is_file():
            raise RuntimeError(f"configured source URDF is missing: {configured}")
        result[robot] = configured
    return result


def _verify_source_urdf_bindings(
    *, source_config_path: Path, release_v3_root: Path
) -> dict[str, Any]:
    """Bind every robot/seed runtime spec to the live source URDF bytes."""

    source_config_raw = _read_yaml(source_config_path)
    configured = _configured_urdf_paths(source_config_path)
    robots: dict[str, Any] = {}
    for robot in ROBOTS:
        path = configured[robot]
        actual = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        seed_records: dict[str, Any] = {}
        first_contract: tuple[str, str] | None = None
        first_joint_names: tuple[str, ...] | None = None
        for seed in ALL_SEEDS:
            runtime_spec_path = (
                release_v3_root / robot / f"seed{seed}" / "runtime_spec.json"
            )
            if not runtime_spec_path.is_file() or runtime_spec_path.is_symlink():
                raise RuntimeError(f"release_v3 runtime spec is unavailable: {runtime_spec_path}")
            runtime_spec = json.loads(runtime_spec_path.read_text(encoding="utf-8"))
            source_urdf = runtime_spec.get("source_urdf")
            runtime_joint_names = tuple(
                str(value) for value in runtime_spec.get("joint_names", ())
            )
            if (
                int(runtime_spec.get("schema_version", -1)) != 1
                or runtime_spec.get("robot") != robot
                or int(runtime_spec.get("training_seed", -1)) != seed
                or not isinstance(source_urdf, Mapping)
                or set(source_urdf) != {"path", "sha256"}
                or not runtime_joint_names
            ):
                raise RuntimeError(
                    f"release_v3 runtime-spec URDF schema differs for {robot}/seed{seed}"
                )
            recorded_source_path = Path(str(source_urdf["path"]))
            if recorded_source_path.is_symlink():
                raise RuntimeError(
                    f"release_v3 source URDF path is a symlink for {robot}/seed{seed}"
                )
            recorded_path = str(recorded_source_path.resolve())
            recorded_hash = str(source_urdf["sha256"])
            contract = (recorded_path, recorded_hash)
            if first_contract is None:
                first_contract = contract
                first_joint_names = runtime_joint_names
            if (
                contract != first_contract
                or runtime_joint_names != first_joint_names
                or recorded_path != actual["path"]
                or recorded_hash != actual["sha256"]
            ):
                raise RuntimeError(
                    f"source URDF differs from release_v3 runtime spec for {robot}/seed{seed}"
                )
            normalization_path = (
                release_v3_root
                / robot
                / f"seed{seed}"
                / "normalization_parameters.npz"
            )
            if not normalization_path.is_file() or normalization_path.is_symlink():
                raise RuntimeError(
                    f"release_v3 normalization artifact is unavailable: {normalization_path}"
                )
            with np.load(normalization_path, allow_pickle=False) as normalization:
                normalization_joint_names = tuple(
                    str(value) for value in normalization["joint_names"].tolist()
                )
            if normalization_joint_names != runtime_joint_names:
                raise RuntimeError(
                    f"runtime and normalization joint names differ for {robot}/seed{seed}"
                )
            seed_records[f"seed{seed}"] = {
                "runtime_spec_path": str(runtime_spec_path),
                "runtime_spec_sha256": _sha256_file(runtime_spec_path),
                "recorded_path": recorded_path,
                "recorded_sha256": recorded_hash,
                # release_v3 schema did not store size; test_v4 freezes the
                # actual size alongside the recorded path/hash without
                # pretending an absent historical field existed.
                "actual_size": actual["size"],
                "joint_names": list(runtime_joint_names),
                "normalization_sha256": _sha256_file(normalization_path),
                "path_sha256_and_actual_size_verified": True,
            }
        robot_config = source_config_raw["robots"][robot]
        robots[robot] = {
            "actual_source_urdf": actual,
            "base_link": str(robot_config["base_link"]),
            "end_link": str(robot_config["end_link"]),
            "joint_names": list(first_joint_names or ()),
            "runtime_specs": seed_records,
        }
    payload: dict[str, Any] = {
        "source_config_path": str(source_config_path),
        "source_config_sha256": _sha256_file(source_config_path),
        "release_v3_runtime_spec_size_field_present": False,
        "size_basis": "actual URDF bytes frozen by test_v4 evidence fingerprint",
        "robots": robots,
    }
    payload["digest"] = _json_digest(payload)
    return payload


def _formal_asset_paths(
    *,
    workspace: Path,
    config_path: Path,
    release_v4_root: Path,
    release_v3_root: Path,
) -> list[Path]:
    assets = [
        config_path,
        workspace / "configs" / "paper_v2.yaml",
        *[path for path in release_v4_root.rglob("*") if path.is_file()],
        *[path for path in release_v3_root.rglob("*") if path.is_file()],
    ]
    assets.extend(
        _configured_urdf_paths(workspace / "configs" / "paper_v2.yaml").values()
    )
    for robot in ROBOTS:
        for source in default_comparison_sources(workspace, robot):
            assets.append(source.path)
            if source.provenance_path is not None:
                assets.append(source.provenance_path)
        for seed in ALL_SEEDS:
            assets.extend(_source_paths(workspace, robot, seed).values())
    return sorted({path.resolve() for path in assets})


def _formal_evidence_fingerprint(
    *,
    workspace: Path,
    config_path: Path,
    release_v4_root: Path,
    release_v3_root: Path,
) -> dict[str, Any]:
    return evidence_fingerprint(
        workspace=workspace,
        source_scope=SOURCE_SCOPE,
        asset_paths=_formal_asset_paths(
            workspace=workspace,
            config_path=config_path,
            release_v4_root=release_v4_root,
            release_v3_root=release_v3_root,
        ),
    )


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


def _subset_dataset(dataset: QueryDataset, indices: np.ndarray) -> QueryDataset:
    selected = np.asarray(indices, dtype=np.int64)
    return QueryDataset(
        **{
            name: np.asarray(getattr(dataset, name))[selected]
            for name in dataset.__dataclass_fields__
        }
    )


def _checkpoint_specs(
    datasets: Mapping[str, QueryDataset], *, point_chunk_size: int
) -> list[tuple[str, str, QueryDataset, np.ndarray]]:
    if point_chunk_size <= 0:
        raise ValueError("point checkpoint chunk size must be positive")
    result: list[tuple[str, str, QueryDataset, np.ndarray]] = []
    for role in TEST_V4_ROLES:
        dataset = datasets[role]
        if not role.endswith("trajectories"):
            for start in range(0, len(dataset), point_chunk_size):
                stop = min(start + point_chunk_size, len(dataset))
                indices = np.arange(start, stop, dtype=np.int64)
                result.append(
                    (
                        role,
                        f"query_{start:06d}_{stop - 1:06d}",
                        _subset_dataset(dataset, indices),
                        indices,
                    )
                )
            continue
        for trajectory_id in dict.fromkeys(int(value) for value in dataset.trajectory_id):
            indices = np.flatnonzero(dataset.trajectory_id == trajectory_id).astype(
                np.int64
            )
            indices = indices[
                np.argsort(dataset.time_index[indices], kind="stable")
            ]
            result.append(
                (
                    role,
                    f"trajectory_{trajectory_id:06d}",
                    _subset_dataset(dataset, indices),
                    indices,
                )
            )
    return result


def _checkpoint_expected(
    *,
    robot: str,
    seed: int,
    role: str,
    key: str,
    dataset: QueryDataset,
    source_indices: np.ndarray,
    methods: Iterable[str],
    dt: float,
    preregistration_sha256: str,
    dataset_manifest_sha256: str,
    evidence_fingerprint_digest: str,
    quiet_host_config_digest: str,
) -> dict[str, Any]:
    method_names = list(methods)
    return {
        "protocol": "test_v4_atomic_measurement_checkpoint",
        "robot": robot,
        "training_seed": int(seed),
        "role": role,
        "checkpoint_key": key,
        "checkpoint_unit": (
            "complete_trajectory_cluster"
            if role.endswith("trajectories")
            else "fixed_point_query_chunk"
        ),
        "source_indices": np.asarray(source_indices, dtype=np.int64).tolist(),
        "source_query_sha256": dataset_query_hashes(dataset, dt=dt).astype(str).tolist(),
        "methods": method_names,
        "expected_query_count": len(dataset),
        "expected_record_count": len(dataset) * len(method_names),
        "preregistration_sha256": preregistration_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evidence_fingerprint_digest": evidence_fingerprint_digest,
        "quiet_host_config_digest": quiet_host_config_digest,
        "resume_contract": "completed checkpoint is hash-validated and never recomputed",
    }


def _validate_checkpoint(final: Path, expected: Mapping[str, Any]) -> Path:
    manifest_path = final / "checkpoint_manifest.json"
    records_path = final / "records.jsonl.gz"
    if not final.is_dir() or final.is_symlink():
        raise RuntimeError(f"checkpoint is not a regular directory: {final}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = dict(manifest)
    artifacts = frozen.pop("artifacts", None)
    completed_utc = frozen.pop("completed_utc", None)
    environment = frozen.pop("quiet_host_evidence", None)
    if frozen != dict(expected) or not completed_utc or not isinstance(environment, Mapping):
        raise RuntimeError(f"checkpoint contract differs from preregistration: {final}")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"records.jsonl.gz"}:
        raise RuntimeError(f"checkpoint artifact schema differs: {final}")
    recorded = artifacts["records.jsonl.gz"]
    if (
        not records_path.is_file()
        or records_path.is_symlink()
        or records_path.stat().st_size != int(recorded["size"])
        or _sha256_file(records_path) != str(recorded["sha256"])
    ):
        raise RuntimeError(f"checkpoint record hash changed: {records_path}")
    if "expected_query_count" in expected:
        required_environment = {
            "background_monitor": True,
            "synchronous_ps_or_nvidia_smi_per_query": False,
            "contamination_decision_source": "external process state only",
            "latency_or_solver_result_used_for_contamination_decision": False,
        }
        if any(environment.get(key) != value for key, value in required_environment.items()):
            raise RuntimeError(f"checkpoint quiet-host evidence differs: {final}")
        expected_queries = int(expected["expected_query_count"])
        if (
            int(environment.get("query_interval_check_count", -1)) < expected_queries
            or int(environment.get("query_intervals_without_new_monitor_sample", -1)) != 0
            or int(environment.get("minimum_monitor_samples_since_query_start", 0)) < 1
            or environment.get("quiet_host_config_digest")
            != expected["quiet_host_config_digest"]
        ):
            raise RuntimeError(f"checkpoint quiet-host interval coverage is incomplete: {final}")
        records = _read_records([records_path])
        if len(records) != int(expected["expected_record_count"]):
            raise RuntimeError(f"checkpoint record count differs: {final}")
        expected_pairs = {
            (int(index), str(method))
            for index in expected["source_indices"]
            for method in expected["methods"]
        }
        observed_pairs = [
            (int(record.get("query_index", -1)), str(record.get("method", "")))
            for record in records
        ]
        if len(set(observed_pairs)) != len(observed_pairs) or set(observed_pairs) != expected_pairs:
            raise RuntimeError(f"checkpoint method-query cross product differs: {final}")
        expected_hashes = dict(
            zip(expected["source_indices"], expected["source_query_sha256"], strict=True)
        )
        if any(
            record.get("robot") != expected["robot"]
            or int(record.get("training_seed", -1)) != int(expected["training_seed"])
            or record.get("role") != expected["role"]
            or record.get("source_query_sha256")
            != expected_hashes[int(record["query_index"])]
            for record in records
        ):
            raise RuntimeError(f"checkpoint record identity binding differs: {final}")
    return records_path


def _write_checkpoint(
    *,
    root: Path,
    key: str,
    records: list[dict[str, Any]],
    expected: Mapping[str, Any],
    quiet_host_evidence: Mapping[str, Any],
) -> Path:
    final = root / key
    if final.exists() or final.is_symlink():
        return _validate_checkpoint(final, expected)
    attempt = root / f".{key}.incomplete.{time.time_ns()}.{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    records_path = attempt / "records.jsonl.gz"
    _write_records(records_path, records)
    manifest = {
        **dict(expected),
        "completed_utc": _utc(),
        "quiet_host_evidence": dict(quiet_host_evidence),
        "artifacts": {
            "records.jsonl.gz": {
                "sha256": _sha256_file(records_path),
                "size": records_path.stat().st_size,
            }
        },
    }
    _write_json(attempt / "checkpoint_manifest.json", manifest)
    attempt.rename(final)
    return _validate_checkpoint(final, expected)


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
    evidence_fingerprint_digest: str,
    host_guard: FormalHostGuard,
    resume: bool,
) -> None:
    expected_methods = PRIMARY_METHODS if seed == PRIMARY_SEED else SENSITIVITY_METHODS
    quiet_host_config_digest = _json_digest(config["runtime"]["quiet_host"])
    if destination.exists() and not resume:
        raise RuntimeError(f"formal combination destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=resume)
    completion_path = destination / "combination_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("preregistration_sha256") != preregistration_sha256
            or completion.get("dataset_manifest_sha256") != dataset_manifest_sha256
            or completion.get("evidence_fingerprint_digest")
            != evidence_fingerprint_digest
            or tuple(completion.get("methods", ())) != expected_methods
            or not bool(completion.get("all_checkpoints_hash_validated", False))
        ):
            raise RuntimeError(f"completed combination contract changed: {destination}")
        for relative, artifact in completion["artifacts"].items():
            target = destination / relative
            if (
                not target.is_file()
                or target.is_symlink()
                or target.stat().st_size != int(artifact["size"])
                or _sha256_file(target) != str(artifact["sha256"])
            ):
                raise RuntimeError(f"completed combination artifact changed: {target}")
        for role, key, subset, source_indices in _checkpoint_specs(
            datasets,
            point_chunk_size=int(config["checkpointing"]["point_query_chunk_size"]),
        ):
            expected = _checkpoint_expected(
                robot=robot,
                seed=seed,
                role=role,
                key=key,
                dataset=subset,
                source_indices=source_indices,
                methods=expected_methods,
                dt=float(config["data"]["dt"]),
                preregistration_sha256=preregistration_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                evidence_fingerprint_digest=evidence_fingerprint_digest,
                quiet_host_config_digest=quiet_host_config_digest,
            )
            _validate_checkpoint(destination / "checkpoints" / role / key, expected)
        return

    timing = config["timing"]
    dt = float(config["data"]["dt"])
    checkpoint_specs = _checkpoint_specs(
        datasets,
        point_chunk_size=int(config["checkpointing"]["point_query_chunk_size"]),
    )
    checkpoint_root = destination / "checkpoints"
    checkpoint_root.mkdir(exist_ok=resume)
    existing_files: dict[tuple[str, str], Path] = {}
    missing: list[tuple[str, str, QueryDataset, np.ndarray]] = []
    for role, key, subset, source_indices in checkpoint_specs:
        expected = _checkpoint_expected(
            robot=robot,
            seed=seed,
            role=role,
            key=key,
            dataset=subset,
            source_indices=source_indices,
            methods=expected_methods,
            dt=dt,
            preregistration_sha256=preregistration_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            evidence_fingerprint_digest=evidence_fingerprint_digest,
            quiet_host_config_digest=quiet_host_config_digest,
        )
        final = checkpoint_root / role / key
        if final.exists():
            existing_files[(role, key)] = _validate_checkpoint(final, expected)
        else:
            missing.append((role, key, subset, source_indices))

    methods: dict[str, object] | None = None
    if missing:
        methods = _build_methods(
            workspace=workspace,
            source_config=source_config,
            release_v4_root=release_v4_root,
            release_v3_root=release_v3_root,
            robot=robot,
            seed=seed,
        )
        if tuple(methods) != expected_methods:
            raise RuntimeError(
                f"formal method order differs for seed {seed}: {tuple(methods)}"
            )
        host_guard.wait_until_quiet(
            context=f"{robot}/seed{seed}/validation_only_warmup/before"
        )
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
        for name in expected_methods
    }
    order_root = derive_seed(release_digest, robot, f"method_order_seed{seed}")
    role_index = {role: index for index, role in enumerate(TEST_V4_ROLES)}
    for role, key, subset, source_indices in missing:
        assert methods is not None
        checkpoint = host_guard.checkpoint()
        print(f"[test-v4] timing {robot}/seed{seed}/{role}/{key}", flush=True)
        records = benchmark_role(
            robot=robot,
            training_seed=seed,
            role=role,
            methods=methods,
            dataset=subset,
            repeats_by_method=repeats,
            dt=dt,
            order_seed=order_root + role_index[role],
            synchronize_cuda=True,
            environment_guard=host_guard,
            source_indices=source_indices,
        )
        expected = _checkpoint_expected(
            robot=robot,
            seed=seed,
            role=role,
            key=key,
            dataset=subset,
            source_indices=source_indices,
            methods=expected_methods,
            dt=dt,
            preregistration_sha256=preregistration_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            evidence_fingerprint_digest=evidence_fingerprint_digest,
            quiet_host_config_digest=quiet_host_config_digest,
        )
        role_root = checkpoint_root / role
        role_root.mkdir(exist_ok=True)
        quiet_host_evidence = host_guard.summary_since(checkpoint)
        quiet_host_evidence["quiet_host_config_digest"] = quiet_host_config_digest
        if (
            not bool(quiet_host_evidence.get("background_monitor", False))
            or bool(
                quiet_host_evidence.get(
                    "synchronous_ps_or_nvidia_smi_per_query", True
                )
            )
            or int(quiet_host_evidence.get("query_interval_check_count", 0))
            < len(subset)
            or int(
                quiet_host_evidence.get(
                    "query_intervals_without_new_monitor_sample", -1
                )
            )
            != 0
            or int(
                quiet_host_evidence.get(
                    "minimum_monitor_samples_since_query_start", 0
                )
            )
            < 1
            or quiet_host_evidence.get("contamination_decision_source")
            != "external process state only"
            or bool(
                quiet_host_evidence.get(
                    "latency_or_solver_result_used_for_contamination_decision", True
                )
            )
        ):
            raise RuntimeError(
                f"quiet-host evidence contract failed for {robot}/seed{seed}/{role}/{key}"
            )
        existing_files[(role, key)] = _write_checkpoint(
            root=role_root,
            key=key,
            records=records,
            expected=expected,
            quiet_host_evidence=quiet_host_evidence,
        )
        del records

    ordered_record_files = [
        existing_files[(role, key)] for role, key, _, _ in checkpoint_specs
    ]
    records = _read_records(ordered_record_files)
    summary = {
        "protocol": PROTOCOL,
        "robot": robot,
        "training_seed": seed,
        "seed_statistical_role": (
            "primary confirmatory" if seed == PRIMARY_SEED else "sensitivity only"
        ),
        "methods": list(expected_methods),
        "method_metrics": method_metrics(records),
        "record_count": len(records),
        "expected_record_count": sum(len(dataset) for dataset in datasets.values())
        * len(expected_methods),
        "atomic_checkpoint_count": len(checkpoint_specs),
        "completed_checkpoint_count": len(ordered_record_files),
        "completed_checkpoints_recomputed_on_resume": False,
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
        "technical_retry_count": int(resume),
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
                f"technical_retry_count={int(resume)}",
                "test_set_retuning=false",
                "outlier_removal=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        str(path.relative_to(destination)): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(item for item in destination.rglob("*") if item.is_file())
        if path.name != "combination_complete.json"
    }
    _write_json(
        completion_path,
        {
            "protocol": "test_v4_combination_complete_but_not_formal_completion_marker",
            "completed_utc": _utc(),
            "robot": robot,
            "training_seed": seed,
            "methods": list(expected_methods),
            "preregistration_sha256": preregistration_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "evidence_fingerprint_digest": evidence_fingerprint_digest,
            "all_checkpoints_hash_validated": True,
            "scientific_gate_failure_permits_rerun": False,
            "formal_completion_marker": (
                "outputs/test_v4_aggregate/test_v4_final_manifest.json"
            ),
            "eligible_without_aggregate_final_manifest": False,
            "artifacts": artifacts,
        },
    )


def _aggregate(
    *,
    workspace: Path,
    aggregate_staging: Path,
    seed_roots: Mapping[int, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for robot in ROBOTS:
        root = seed_roots[PRIMARY_SEED] / robot
        primary[robot] = {
            "claim_gate": json.loads((root / "claim_gate_v4.json").read_text(encoding="utf-8")),
            "summary": json.loads((root / "summary_v4.json").read_text(encoding="utf-8")),
            "ood_abstention": json.loads((root / "ood_abstention_v4.json").read_text(encoding="utf-8")),
        }
        sensitivity[robot] = {}
        for seed in SENSITIVITY_SEEDS:
            sensitivity[robot][f"seed{seed}"] = json.loads(
                (
                    seed_roots[seed]
                    / robot
                    / "summary_v4.json"
                ).read_text(encoding="utf-8")
            )
    joint_holm = joint_holm_confirmatory(
        {
            robot: json.loads(
                (
                    seed_roots[PRIMARY_SEED]
                    / robot
                    / "paired_intervals_v4.json"
                ).read_text(encoding="utf-8")
            )
            for robot in ROBOTS
        },
        alpha=float(config["statistics"]["familywise_alpha"]),
    )
    robot_gates = {
        robot: bool(primary[robot]["claim_gate"]["formal_gate_pass"])
        for robot in ROBOTS
    }
    paper_gate = {
        "protocol": "test_v4 robot-level confirmatory aggregation",
        "primary_training_seed": PRIMARY_SEED,
        "sensitivity_training_seeds": list(SENSITIVITY_SEEDS),
        "sensitivity_seeds_are_not_independent_query_samples": True,
        "robot_gates": robot_gates,
        "robot_gates_are_pre_joint_holm": True,
        "joint_holm_gate_pass": bool(
            joint_holm["all_confirmatory_nulls_rejected"]
        ),
        "joint_holm_is_required_for_formal_gate": True,
        "both_robot_gates_pass": all(robot_gates.values())
        and bool(joint_holm["all_confirmatory_nulls_rejected"]),
        "test_set_retuning_performed": False,
    }
    _write_json(aggregate_staging / "joint_holm_v4.json", joint_holm)
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


def _load_frozen_test_datasets(
    *,
    aggregate_staging: Path,
    dataset_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, QueryDataset]]:
    result: dict[str, dict[str, QueryDataset]] = {}
    for robot in ROBOTS:
        result[robot] = {}
        for role in TEST_V4_ROLES:
            artifact = dataset_manifest["robots"][robot]["roles"][role]
            path = aggregate_staging / str(artifact["path"])
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(artifact["size"])
                or _sha256_file(path) != str(artifact["sha256"])
            ):
                raise RuntimeError(f"frozen resume dataset changed: {path}")
            dataset = QueryDataset.load(path)
            identity = _dataset_query_identity(
                dataset, dt=float(config["data"]["dt"])
            )
            if any(artifact.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f"frozen resume dataset identity changed: {path}")
            result[robot][role] = dataset
        contract = dataset_contract(
            result[robot], robot=robot, dt=float(config["data"]["dt"])
        )
        validate_dataset_contract(contract, config)
        if contract != dataset_manifest["robots"][robot]["contract"]:
            raise RuntimeError(f"resume dataset contract changed for {robot}")
    return result


def _dataset_query_identity(dataset: QueryDataset, *, dt: float) -> dict[str, Any]:
    hashes = dataset_query_hashes(dataset, dt=dt).astype(str).tolist()
    return {
        "query_count": len(hashes),
        "ordered_query_sha256_digest": _json_digest(hashes),
        "query_sha256_set_digest": _json_digest(sorted(hashes)),
    }


def _assert_frozen_control_plane(
    *,
    aggregate_staging: Path,
    preregistration_sha256: str,
    dataset_manifest_sha256: str,
    control_plane_seal_sha256: str,
    datasets_by_robot: Mapping[str, Mapping[str, QueryDataset]],
    config: Mapping[str, Any],
    evidence_fingerprint_digest: str,
) -> None:
    preregistration_path = aggregate_staging / "test_v4_preregistration.json"
    dataset_manifest_path = aggregate_staging / "test_v4_dataset_manifest.json"
    seal_path = aggregate_staging / "test_v4_control_plane_seal.json"
    if _sha256_file(preregistration_path) != preregistration_sha256:
        raise RuntimeError("frozen preregistration changed during formal test")
    if _sha256_file(dataset_manifest_path) != dataset_manifest_sha256:
        raise RuntimeError("frozen dataset manifest changed during formal test")
    if _sha256_file(seal_path) != control_plane_seal_sha256:
        raise RuntimeError("frozen control-plane seal changed during formal test")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal != {
        "protocol": "test_v4_frozen_control_plane_seal",
        "preregistration_sha256": preregistration_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evidence_fingerprint_digest": evidence_fingerprint_digest,
    }:
        raise RuntimeError("frozen control-plane seal contract differs")
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dt = float(config["data"]["dt"])
    for robot in ROBOTS:
        for role in TEST_V4_ROLES:
            artifact = manifest["robots"][robot]["roles"][role]
            dataset_path = aggregate_staging / str(artifact["path"])
            if (
                not dataset_path.is_file()
                or dataset_path.is_symlink()
                or dataset_path.stat().st_size != int(artifact["size"])
                or _sha256_file(dataset_path) != str(artifact["sha256"])
            ):
                raise RuntimeError(f"frozen dataset artifact changed: {dataset_path}")
            disk_identity = _dataset_query_identity(QueryDataset.load(dataset_path), dt=dt)
            memory_identity = _dataset_query_identity(datasets_by_robot[robot][role], dt=dt)
            if any(artifact.get(key) != value for key, value in disk_identity.items()):
                raise RuntimeError(f"frozen dataset query identity changed: {dataset_path}")
            if memory_identity != disk_identity:
                raise RuntimeError(f"in-memory dataset differs from frozen artifact: {dataset_path}")


def _partial_output_manifest(workspace: Path) -> list[dict[str, Any]]:
    output_root = workspace / "outputs"
    roots = [
        output_root / ".test_v4_aggregate.incomplete",
        *[output_root / f".test_v4_seed{seed}.incomplete" for seed in ALL_SEEDS],
        *[output_root / f"test_v4_seed{seed}" for seed in ALL_SEEDS],
    ]
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name.startswith("failure_manifest"):
                continue
            files.append(
                {
                    "path": str(path.relative_to(workspace)),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return files


def _aggregate_checkpoint_host_evidence(
    seed_roots: Mapping[int, Path],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        for manifest_path in sorted(
            seed_roots[seed].glob("*/checkpoints/*/*/checkpoint_manifest.json")
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = manifest.get("quiet_host_evidence")
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    f"checkpoint lacks quiet-host evidence: {manifest_path}"
                )
            evidence.append(dict(payload))
    if not evidence:
        raise RuntimeError("formal completion has no checkpoint quiet-host evidence")
    quiet_config_digests = {
        str(item.get("quiet_host_config_digest", "")) for item in evidence
    }
    if (
        len(quiet_config_digests) != 1
        or "" in quiet_config_digests
        or any(
            not bool(item.get("background_monitor", False))
            or bool(item.get("synchronous_ps_or_nvidia_smi_per_query", True))
            or int(item.get("query_intervals_without_new_monitor_sample", -1)) != 0
            or int(item.get("minimum_monitor_samples_since_query_start", 0)) < 1
            or item.get("contamination_decision_source")
            != "external process state only"
            or bool(
                item.get(
                    "latency_or_solver_result_used_for_contamination_decision", True
                )
            )
            for item in evidence
        )
    ):
        raise RuntimeError("completed checkpoints lack complete quiet-host coverage")
    summed_fields = (
        "external_state_cache_read_count",
        "background_monitor_sample_count",
        "query_interval_check_count",
        "query_intervals_without_new_monitor_sample",
        "quiet_wait_event_count",
        "contaminated_attempt_count",
    )
    return {
        "checkpoint_count": len(evidence),
        "quiet_host_config_digest": next(iter(quiet_config_digests)),
        "all_checkpoints_used_background_monitor": all(
            bool(item.get("background_monitor", False)) for item in evidence
        ),
        "any_checkpoint_used_synchronous_process_probe_per_query": any(
            bool(item.get("synchronous_ps_or_nvidia_smi_per_query", True))
            for item in evidence
        ),
        "all_contamination_decisions_external_state_only": all(
            item.get("contamination_decision_source")
            == "external process state only"
            and not bool(
                item.get(
                    "latency_or_solver_result_used_for_contamination_decision", True
                )
            )
            for item in evidence
        ),
        "totals": {
            field: sum(int(item.get(field, 0)) for item in evidence)
            for field in summed_fields
        },
        "quiet_wait_seconds_total": float(
            sum(float(item.get("quiet_wait_seconds", 0.0)) for item in evidence)
        ),
        "maximum_post_read_cache_age_seconds": max(
            float(item.get("maximum_post_read_cache_age_seconds", 0.0))
            for item in evidence
        ),
        "maximum_observed_monitor_sample_gap_seconds": max(
            float(item.get("maximum_observed_monitor_sample_gap_seconds", 0.0))
            for item in evidence
        ),
        "per_checkpoint_evidence_retained_in_checkpoint_manifests": True,
    }


def _write_failure_manifest(
    *,
    workspace: Path,
    error: BaseException,
    resume_eligible: bool,
    failure_classification: str,
) -> Path | None:
    staging = workspace / "outputs" / ".test_v4_aggregate.incomplete"
    if not staging.is_dir():
        return None
    failure_root = staging / "failure_manifests"
    failure_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "test_v4_technical_or_integrity_failure_manifest",
        "created_utc": _utc(),
        "phase": dict(_ACTIVE_PHASE),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "failure_classification": failure_classification,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "resume_eligible": bool(resume_eligible),
        "resume_requires_explicit_flag": True,
        "scientific_result_failure_is_resume_eligible": False,
        "completed_checkpoint_recomputation_allowed": False,
        "formal_completion_marker_exists": (
            workspace
            / "outputs"
            / "test_v4_aggregate"
            / "test_v4_final_manifest.json"
        ).is_file(),
        "partial_files": _partial_output_manifest(workspace),
        "quiet_host_evidence": (
            _ACTIVE_HOST_GUARD.total_summary()
            if _ACTIVE_HOST_GUARD is not None
            else None
        ),
        "control_plane_seal_sha256": _ACTIVE_CONTROL_PLANE_SEAL_SHA256,
    }
    destination = failure_root / f"failure_{time.time_ns()}_{os.getpid()}.json"
    _write_json(destination, payload)
    _write_json(staging / "latest_failure_manifest.json", payload)
    return destination


def _classify_failure(error: BaseException) -> tuple[bool, str]:
    if isinstance(error, RuntimeError) and str(error).startswith(
        "formal command contract violation"
    ):
        return False, "non_resumable_scientific_contract_failure"
    if isinstance(error, QuietHostTechnicalInterruption):
        return True, "resumable_external_environment_technical_interruption"
    if isinstance(error, KeyboardInterrupt):
        return True, "resumable_operator_or_signal_technical_interruption"
    if isinstance(error, OSError):
        return True, "resumable_io_or_system_technical_interruption"
    return False, "non_resumable_integrity_or_scientific_failure"


def _run_formal(
    config_path: str | Path,
    *,
    resume: bool,
    stale_lock_recovered: bool,
) -> dict[str, Any]:
    global _ACTIVE_HOST_GUARD, _ACTIVE_CONTROL_PLANE_SEAL_SHA256
    workspace = Path(__file__).resolve().parents[3]
    path = Path(config_path).resolve()
    canonical_config = (workspace / "configs" / "test_v4_locked.yaml").resolve()
    if path != canonical_config:
        raise RuntimeError("formal runner requires the canonical locked test_v4 config")
    config = _read_yaml(path)
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
    if float(config["statistics"].get("familywise_alpha", -1.0)) != 0.05:
        raise RuntimeError("formal joint Holm familywise alpha must remain 0.05")
    if str(config["statistics"]["multiplicity_correction"]).lower() != "holm":
        raise RuntimeError("formal multiplicity correction must remain Holm")
    if bool(config["runtime"]["hysteresis"]["enabled"]):
        raise RuntimeError("trajectory hysteresis was disabled before fresh testing")
    quiet_host = config["runtime"].get("quiet_host", {})
    if (
        quiet_host.get("contamination_decision_source")
        != "external_process_state_only"
        or quiet_host.get("point_discard_scope")
        != "complete_same_query_all_methods_all_repeats"
        or quiet_host.get("trajectory_discard_scope")
        != "complete_trajectory_cluster_all_methods"
        or not bool(quiet_host.get("reject_other_gpu_compute_pids", False))
    ):
        raise RuntimeError("formal quiet-host evidence contract differs from preregistration")
    id_points = config["data"]["roles"]["id_points"]
    if int(id_points.get("hard_screening_max_attempts_per_query", -1)) != 120:
        raise RuntimeError("hard-valid screening attempt bound differs from frozen generator")
    checkpointing = config.get("checkpointing", {})
    if (
        int(checkpointing.get("point_query_chunk_size", 0)) != 250
        or checkpointing.get("trajectory_checkpoint_unit")
        != "complete_trajectory_cluster"
        or not bool(checkpointing.get("atomic_directory_rename", False))
        or bool(checkpointing.get("completed_checkpoint_recomputation_allowed", True))
        or checkpointing.get("resume_scope") != "technical_interruption_only"
        or bool(checkpointing.get("scientific_gate_failure_rerun_allowed", True))
    ):
        raise RuntimeError("formal checkpoint/resume contract differs from preregistration")

    runtime = config["runtime"]
    required_environment = {
        str(key): str(value)
        for key, value in runtime.get("required_environment_variables", {}).items()
    }
    expected_required_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }
    if required_environment != expected_required_environment:
        raise RuntimeError("formal BLAS/CUDA environment contract differs from preregistration")
    mismatched_environment = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in required_environment.items()
        if os.environ.get(key) != value
    }
    if mismatched_environment:
        raise RuntimeError(
            f"formal BLAS/CUDA environment is not locked: {mismatched_environment}"
        )
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
    release_manifest, _ = _verify_release(release_root, config["release_lock"])
    release_digest = str(release_manifest["release_digest"])
    release_v3_root = _release_v3_root(release_root, workspace)
    environment = _validate_environment(release_root, runtime)
    source_config_path = (path.parent / str(config["source_config"])).resolve()
    if source_config_path != (workspace / "configs" / "paper_v2.yaml").resolve():
        raise RuntimeError("formal source configuration differs from paper_v2.yaml")
    source_config = load_config(source_config_path)
    frozen_urdf_bindings = _verify_source_urdf_bindings(
        source_config_path=source_config_path,
        release_v3_root=release_v3_root,
    )
    for robot in ROBOTS:
        loaded_joint_names = list(load_robot(source_config, robot).joint_names)
        if loaded_joint_names != frozen_urdf_bindings["robots"][robot]["joint_names"]:
            raise RuntimeError(
                f"loaded kinematic joint names differ from release_v3 for {robot}"
            )
    frozen_fingerprint = _formal_evidence_fingerprint(
        workspace=workspace,
        config_path=path,
        release_v4_root=release_root,
        release_v3_root=release_v3_root,
    )

    output_root = workspace / "outputs"
    aggregate = (path.parent / str(config["aggregate_directory"])).resolve()
    if aggregate != (output_root / "test_v4_aggregate").resolve():
        raise RuntimeError("formal aggregate path differs from locked location")
    aggregate_staging = output_root / ".test_v4_aggregate.incomplete"
    final_seed_roots = {
        seed: output_root / f"test_v4_seed{seed}" for seed in ALL_SEEDS
    }
    incomplete_seed_roots = {
        seed: output_root / f".test_v4_seed{seed}.incomplete" for seed in ALL_SEEDS
    }
    if aggregate.exists() or aggregate.is_symlink():
        raise RuntimeError("test_v4 final manifest already exists; rerun is forbidden")
    legacy_staging = sorted(output_root.glob(".test_v4_*.incomplete.*"))
    if legacy_staging:
        raise RuntimeError(f"legacy incomplete output requires audit: {legacy_staging}")

    if not resume:
        if aggregate_staging.exists() or any(
            path.exists() for path in [*final_seed_roots.values(), *incomplete_seed_roots.values()]
        ):
            raise RuntimeError("test_v4 evidence already exists; use audited --resume or stop")
        protected_before = _tree_snapshot(
            output_root, [str(value) for value in config["protected_outputs"]]
        )
        aggregate_staging.mkdir(parents=False)
        preregistration = {
            "protocol": PROTOCOL,
            "created_utc": _utc(),
            "release_digest": release_digest,
            "release_manifest_sha256": _sha256_file(release_root / "release_manifest.json"),
            "independent_release_lock": config["release_lock"],
            "runner_git_commit": source_manifest["git_commit"],
            "runner_git_tree": source_manifest["git_tree"],
            "runner_sha256": _sha256_file(Path(__file__)),
            "benchmark_sha256": _sha256_file(Path(__file__).with_name("benchmark.py")),
            "reporting_sha256": _sha256_file(Path(__file__).with_name("reporting.py")),
            "config_sha256": _sha256_file(path),
            "source_manifest": source_manifest,
            "evidence_fingerprint": frozen_fingerprint,
            "source_urdf_bindings": frozen_urdf_bindings,
            "protected_outputs_before": protected_before,
            "robots": list(ROBOTS),
            "primary_training_seed": PRIMARY_SEED,
            "sensitivity_training_seeds": list(SENSITIVITY_SEEDS),
            "methods": config["methods"],
            "data": config["data"],
            "timing": config["timing"],
            "checkpointing": config["checkpointing"],
            "statistics": config["statistics"],
            "claim_gate": config["claim_gate"],
            "baseline_availability": _baseline_availability(),
            "formal_completion_marker": (
                "outputs/test_v4_aggregate/test_v4_final_manifest.json"
            ),
            "seed_directories_eligible_without_completion_marker": False,
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
                "technical_resume_requires_explicit_flag": True,
                "completed_checkpoint_recomputation_allowed": False,
                "scientific_gate_failure_rerun_allowed": False,
                "post_test_tuning_allowed": False,
                "outlier_removal_allowed": False,
                "winsorization_allowed": False,
            },
        }
        prereg_path = aggregate_staging / "test_v4_preregistration.json"
        _write_json(prereg_path, preregistration)
        prereg_hash = _sha256_file(prereg_path)
        _write_json(aggregate_staging / "baseline_availability.json", _baseline_availability())
        _write_json(aggregate_staging / "resume_history.json", {"events": []})
        print(f"[test-v4] preregistration frozen sha256={prereg_hash}", flush=True)

        _ACTIVE_PHASE.clear()
        _ACTIVE_PHASE.update({"phase": "fresh_dataset_generation"})
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
                    **_dataset_query_identity(
                        datasets[role], dt=float(config["data"]["dt"])
                    ),
                }
            dataset_payload[robot] = {
                "seeds": seeds,
                "contract": contract,
                "freshness": freshness,
                "source_kinematics": frozen_urdf_bindings["robots"][robot],
                "roles": role_files,
            }
        dataset_manifest = {
            "protocol": "test_v4 locked fresh dataset manifest",
            "created_utc": _utc(),
            "release_digest": release_digest,
            "preregistration_sha256": prereg_hash,
            "release_manifest_sha256": _sha256_file(
                release_root / "release_manifest.json"
            ),
            "artifact_manifest_sha256": _sha256_file(
                release_root / "artifact_manifest.json"
            ),
            "source_config_sha256": _sha256_file(source_config_path),
            "source_kinematics_contract_sha256": frozen_urdf_bindings["digest"],
            "robots": dataset_payload,
            "method_outputs_inspected_before_freeze": False,
            "preregistered_hard_valid_screening_performed": True,
            "hard_valid_screening": {
                "scope": "id_points/hard_valid_only",
                "selector": "previous-state DLS plus deterministic verifier",
                "learned_seed_or_gate_used": False,
                "release_v3_solver_seed": int(
                    id_points["hard_screening_release_seed"]
                ),
                "easy_iterations": int(id_points["hard_screening_easy_iterations"]),
                "robust_iterations": int(
                    id_points["hard_screening_robust_iterations"]
                ),
                "max_attempts_per_query": int(
                    id_points["hard_screening_max_attempts_per_query"]
                ),
                "acceptance_rule": (
                    "known reference verifies; easy fails verification; robust verifies"
                ),
            },
            "old_test_performance_results_read": False,
            "all_contract_and_freshness_gates_pass": True,
        }
        dataset_manifest_path = aggregate_staging / "test_v4_dataset_manifest.json"
        _write_json(dataset_manifest_path, dataset_manifest)
        dataset_manifest_hash = _sha256_file(dataset_manifest_path)
        control_plane_seal_path = (
            aggregate_staging / "test_v4_control_plane_seal.json"
        )
        _write_json(
            control_plane_seal_path,
            {
                "protocol": "test_v4_frozen_control_plane_seal",
                "preregistration_sha256": prereg_hash,
                "dataset_manifest_sha256": dataset_manifest_hash,
                "evidence_fingerprint_digest": frozen_fingerprint["digest"],
            },
        )
        control_plane_seal_hash = _sha256_file(control_plane_seal_path)
        _ACTIVE_CONTROL_PLANE_SEAL_SHA256 = control_plane_seal_hash
        if _ACTIVE_RUN_LOCK is None:
            raise RuntimeError("formal global lock is unavailable for control-plane seal")
        _ACTIVE_RUN_LOCK.bind_control_plane(control_plane_seal_hash)
        print(f"[test-v4] fresh datasets frozen sha256={dataset_manifest_hash}", flush=True)
    else:
        if not aggregate_staging.is_dir():
            raise RuntimeError("technical resume requires the fixed aggregate staging directory")
        latest_failure = aggregate_staging / "latest_failure_manifest.json"
        if latest_failure.is_file():
            failure = json.loads(latest_failure.read_text(encoding="utf-8"))
            if not bool(failure.get("resume_eligible", False)):
                raise RuntimeError("prior failure was not classified as technically resumable")
        elif not stale_lock_recovered:
            raise RuntimeError("resume lacks a technical failure manifest or stale-lock evidence")
        prereg_path = aggregate_staging / "test_v4_preregistration.json"
        preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
        prereg_hash = _sha256_file(prereg_path)
        if (
            preregistration.get("protocol") != PROTOCOL
            or preregistration.get("release_digest") != release_digest
            or preregistration.get("config_sha256") != _sha256_file(path)
        ):
            raise RuntimeError("resume preregistration differs from the current frozen run")
        assert_evidence_fingerprint(
            preregistration["evidence_fingerprint"],
            frozen_fingerprint,
            context="resume_preflight",
        )
        if preregistration.get("source_urdf_bindings") != frozen_urdf_bindings:
            raise RuntimeError("resume source URDF contract differs from preregistration")
        protected_before = preregistration["protected_outputs_before"]
        protected_current = _tree_snapshot(
            output_root, [str(value) for value in config["protected_outputs"]]
        )
        if protected_current != protected_before:
            raise RuntimeError("protected evidence changed before technical resume")
        dataset_manifest_path = aggregate_staging / "test_v4_dataset_manifest.json"
        if not dataset_manifest_path.is_file():
            raise RuntimeError(
                "technical resume has no frozen dataset manifest; strict dataset-hash "
                "resume is impossible and requires manual audit"
            )
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_manifest_hash = _sha256_file(dataset_manifest_path)
        control_plane_seal_path = (
            aggregate_staging / "test_v4_control_plane_seal.json"
        )
        if not control_plane_seal_path.is_file():
            raise RuntimeError("technical resume has no frozen control-plane seal")
        control_plane_seal_hash = _sha256_file(control_plane_seal_path)
        _ACTIVE_CONTROL_PLANE_SEAL_SHA256 = control_plane_seal_hash
        if latest_failure.is_file():
            if failure.get("control_plane_seal_sha256") != control_plane_seal_hash:
                raise RuntimeError("technical failure manifest does not anchor the control plane")
        else:
            recovered = (
                None
                if _ACTIVE_RUN_LOCK is None or _ACTIVE_RUN_LOCK.payload is None
                else _ACTIVE_RUN_LOCK.payload.get("recovered_lock_payload")
            )
            if not isinstance(recovered, Mapping) or recovered.get(
                "control_plane_seal_sha256"
            ) != control_plane_seal_hash:
                raise RuntimeError("stale-lock resume does not anchor the control plane")
        if _ACTIVE_RUN_LOCK is None:
            raise RuntimeError("formal global lock is unavailable for resumed control plane")
        _ACTIVE_RUN_LOCK.bind_control_plane(control_plane_seal_hash)
        if dataset_manifest.get("preregistration_sha256") != prereg_hash:
            raise RuntimeError("resume dataset manifest is not bound to the preregistration")
        if (
            dataset_manifest.get("source_kinematics_contract_sha256")
            != frozen_urdf_bindings["digest"]
        ):
            raise RuntimeError("resume dataset source-kinematics contract changed")
        datasets_by_robot = _load_frozen_test_datasets(
            aggregate_staging=aggregate_staging,
            dataset_manifest=dataset_manifest,
            config=config,
        )
        history_path = aggregate_staging / "resume_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history["events"].append(
            {
                "resumed_utc": _utc(),
                "pid": os.getpid(),
                "stale_lock_recovered": stale_lock_recovered,
                "preregistration_sha256": prereg_hash,
                "dataset_manifest_sha256": dataset_manifest_hash,
                "evidence_fingerprint_digest": frozen_fingerprint["digest"],
                "completed_checkpoints_recomputed": False,
            }
        )
        _write_json(history_path, history)

    _assert_frozen_control_plane(
        aggregate_staging=aggregate_staging,
        preregistration_sha256=prereg_hash,
        dataset_manifest_sha256=dataset_manifest_hash,
        control_plane_seal_sha256=control_plane_seal_hash,
        datasets_by_robot=datasets_by_robot,
        config=config,
        evidence_fingerprint_digest=frozen_fingerprint["digest"],
    )

    _ACTIVE_HOST_GUARD = FormalHostGuard(config["runtime"]["quiet_host"])
    seed_roots: dict[int, Path] = {}
    for seed in ALL_SEEDS:
        final = final_seed_roots[seed]
        incomplete = incomplete_seed_roots[seed]
        if final.exists() and incomplete.exists():
            raise RuntimeError(f"both final-looking and incomplete seed roots exist: seed{seed}")
        if final.exists():
            if not resume:
                raise RuntimeError(f"unexpected existing seed evidence: {final}")
            seed_roots[seed] = final
        else:
            incomplete.mkdir(exist_ok=resume)
            seed_roots[seed] = incomplete

    for seed in ALL_SEEDS:
        for robot in ROBOTS:
            _ACTIVE_PHASE.clear()
            _ACTIVE_PHASE.update(
                {"phase": "formal_combination", "robot": robot, "training_seed": seed}
            )
            current = _formal_evidence_fingerprint(
                workspace=workspace,
                config_path=path,
                release_v4_root=release_root,
                release_v3_root=release_v3_root,
            )
            assert_evidence_fingerprint(
                frozen_fingerprint, current, context=f"before_{robot}_seed{seed}"
            )
            _assert_frozen_control_plane(
                aggregate_staging=aggregate_staging,
                preregistration_sha256=prereg_hash,
                dataset_manifest_sha256=dataset_manifest_hash,
                control_plane_seal_sha256=control_plane_seal_hash,
                datasets_by_robot=datasets_by_robot,
                config=config,
                evidence_fingerprint_digest=frozen_fingerprint["digest"],
            )
            current_urdf_bindings = _verify_source_urdf_bindings(
                source_config_path=source_config_path,
                release_v3_root=release_v3_root,
            )
            if current_urdf_bindings != frozen_urdf_bindings:
                raise RuntimeError(
                    f"source URDF binding changed before {robot}/seed{seed}"
                )
            print(f"[test-v4] starting {robot}/seed{seed}", flush=True)
            _run_combination(
                workspace=workspace,
                destination=seed_roots[seed] / robot,
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
                evidence_fingerprint_digest=frozen_fingerprint["digest"],
                host_guard=_ACTIVE_HOST_GUARD,
                resume=resume,
            )
            current = _formal_evidence_fingerprint(
                workspace=workspace,
                config_path=path,
                release_v4_root=release_root,
                release_v3_root=release_v3_root,
            )
            assert_evidence_fingerprint(
                frozen_fingerprint, current, context=f"after_{robot}_seed{seed}"
            )
            _assert_frozen_control_plane(
                aggregate_staging=aggregate_staging,
                preregistration_sha256=prereg_hash,
                dataset_manifest_sha256=dataset_manifest_hash,
                control_plane_seal_sha256=control_plane_seal_hash,
                datasets_by_robot=datasets_by_robot,
                config=config,
                evidence_fingerprint_digest=frozen_fingerprint["digest"],
            )
            current_urdf_bindings = _verify_source_urdf_bindings(
                source_config_path=source_config_path,
                release_v3_root=release_v3_root,
            )
            if current_urdf_bindings != frozen_urdf_bindings:
                raise RuntimeError(
                    f"source URDF binding changed after {robot}/seed{seed}"
                )
            print(f"[test-v4] sealed checkpoints {robot}/seed{seed}", flush=True)

    _ACTIVE_PHASE.clear()
    _ACTIVE_PHASE.update({"phase": "aggregate_and_final_integrity"})
    paper_gate = _aggregate(
        workspace=workspace,
        aggregate_staging=aggregate_staging,
        seed_roots=seed_roots,
        config=config,
    )
    protected_after = _tree_snapshot(
        output_root, [str(value) for value in config["protected_outputs"]]
    )
    if protected_before != protected_after:
        raise RuntimeError("a protected evidence tree changed during formal test_v4")

    final_fingerprint = _formal_evidence_fingerprint(
        workspace=workspace,
        config_path=path,
        release_v4_root=release_root,
        release_v3_root=release_v3_root,
    )
    assert_evidence_fingerprint(
        frozen_fingerprint, final_fingerprint, context="before_final_seal"
    )
    _assert_frozen_control_plane(
        aggregate_staging=aggregate_staging,
        preregistration_sha256=prereg_hash,
        dataset_manifest_sha256=dataset_manifest_hash,
        control_plane_seal_sha256=control_plane_seal_hash,
        datasets_by_robot=datasets_by_robot,
        config=config,
        evidence_fingerprint_digest=frozen_fingerprint["digest"],
    )
    final_urdf_bindings = _verify_source_urdf_bindings(
        source_config_path=source_config_path,
        release_v3_root=release_v3_root,
    )
    if final_urdf_bindings != frozen_urdf_bindings:
        raise RuntimeError("source URDF binding changed before final seal")
    logical_files: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        root = seed_roots[seed]
        for item in sorted(path for path in root.rglob("*") if path.is_file()):
            logical_files.append(
                {
                    "path": str(
                        Path(f"outputs/test_v4_seed{seed}") / item.relative_to(root)
                    ),
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
    # Stop the background sampler before sealing its final coverage evidence.
    # All timed combinations are complete at this point.
    _ACTIVE_HOST_GUARD.close()
    quiet_host_summary = _ACTIVE_HOST_GUARD.total_summary()
    checkpoint_host_summary = _aggregate_checkpoint_host_evidence(seed_roots)
    final_manifest = {
        "protocol": "test_v4 final immutable evidence manifest",
        "completed_utc": _utc(),
        "release_digest": release_digest,
        "preregistration_sha256": prereg_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "control_plane_seal_sha256": control_plane_seal_hash,
        "formal_primary_runs": 2,
        "sensitivity_runs": 4,
        "all_six_natural_exits": True,
        "technical_retry_count": len(
            json.loads(
                (aggregate_staging / "resume_history.json").read_text(encoding="utf-8")
            )["events"]
        ),
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
        "formal_completion_marker": True,
        "seed_directories_eligible_without_this_manifest": False,
        "evidence_fingerprint_at_start": frozen_fingerprint,
        "evidence_fingerprint_at_end": final_fingerprint,
        "source_urdf_bindings_at_start": frozen_urdf_bindings,
        "source_urdf_bindings_at_end": final_urdf_bindings,
        "quiet_host_evidence": {
            "current_process_monitor": quiet_host_summary,
            "all_completed_checkpoints": checkpoint_host_summary,
        },
        "files": logical_files,
    }
    _write_json(aggregate_staging / "test_v4_final_manifest.json", final_manifest)
    # Seed roots are intentionally promoted only after all six combinations,
    # aggregate metrics, protected-tree checks, and source/asset rechecks pass.
    for seed in ALL_SEEDS:
        source = seed_roots[seed]
        destination = final_seed_roots[seed]
        if source == destination:
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"seed promotion target already exists: {destination}")
        source.rename(destination)
    aggregate_staging.rename(aggregate)
    print("[test-v4] all formal and sensitivity runs sealed", flush=True)
    return final_manifest


def run(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    global _ACTIVE_HOST_GUARD, _ACTIVE_RUN_LOCK, _ACTIVE_CONTROL_PLANE_SEAL_SHA256
    workspace = Path(__file__).resolve().parents[3]
    lock = ExclusiveRunLock(
        workspace / "outputs" / ".test_v4_locked.global.lock",
        resume=resume,
    )
    lock_payload = lock.acquire()
    _ACTIVE_RUN_LOCK = lock
    stale_recovered = bool(lock_payload.get("stale_lock_archive"))
    caught: BaseException | None = None
    try:
        return _run_formal(
            config_path,
            resume=resume,
            stale_lock_recovered=stale_recovered,
        )
    except BaseException as error:
        caught = error
        resume_eligible, failure_classification = _classify_failure(error)
        frozen_dataset_manifest = (
            workspace
            / "outputs"
            / ".test_v4_aggregate.incomplete"
            / "test_v4_dataset_manifest.json"
        )
        if resume_eligible and not frozen_dataset_manifest.is_file():
            resume_eligible = False
            failure_classification = (
                "non_resumable_pre_dataset_interruption_requires_manual_audit"
            )
        _write_failure_manifest(
            workspace=workspace,
            error=error,
            resume_eligible=resume_eligible,
            failure_classification=failure_classification,
        )
        raise
    finally:
        try:
            if _ACTIVE_HOST_GUARD is not None:
                _ACTIVE_HOST_GUARD.close()
        except BaseException:
            if caught is None:
                raise
        finally:
            _ACTIVE_HOST_GUARD = None
            _ACTIVE_RUN_LOCK = None
            _ACTIVE_CONTROL_PLANE_SEAL_SHA256 = None
            lock.release()


def main() -> None:
    arguments = _parser().parse_args()
    run(arguments.config, resume=arguments.resume)


if __name__ == "__main__":
    main()


__all__ = ["run"]
