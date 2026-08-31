"""One-shot, aggregation-only repair for the sealed test_v4 measurements.

This module deliberately has no execution path into the benchmark runner,
robot models, learned models, or numerical solvers. It validates the already
sealed six combination trees, recomputes only the joint aggregation, and
writes a new evidence namespace without changing the failed formal run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import yaml

from .reporting import joint_holm_confirmatory


PROTOCOL = "test_v4_aggregate_repair_v1"
ROBOTS = ("panda", "ur5e")
SEEDS = (17, 29, 43)
PRIMARY_SEED = 17
SENSITIVITY_SEEDS = (29, 43)
REPAIR_SOURCE_SCOPE = (
    "src/confik/test_v4_locked/aggregate_repair.py",
    "src/confik/test_v4_locked/reporting.py",
    "configs/test_v4_aggregate_repair_v1.yaml",
    "scripts/run_test_v4_aggregate_repair_v1.sh",
    "tests/test_test_v4_aggregate_repair.py",
    "tests/test_test_v4_reporting.py",
)
ALLOWED_ORIGINAL_SOURCE_CHANGE = "src/confik/test_v4_locked/reporting.py"
EXPECTED_REPAIR_DIFF = {
    "M": {ALLOWED_ORIGINAL_SOURCE_CHANGE},
    "A": {
        "src/confik/test_v4_locked/aggregate_repair.py",
        "configs/test_v4_aggregate_repair_v1.yaml",
        "scripts/run_test_v4_aggregate_repair_v1.sh",
        "tests/test_test_v4_aggregate_repair.py",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while sealing aggregation repair evidence")
        view = view[written:]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML document must contain a mapping: {path}")
    return payload


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"repair output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _resolve(config_path: Path, value: object) -> Path:
    return (config_path.parent / str(value)).resolve()


def _regular_file(path: Path, *, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{context} is missing or is a symlink: {path}")


def _tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"repair input root is missing or is a symlink: {root}")
    files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"repair input tree contains a symlink: {item}")
        if item.is_file():
            files.append(
                {
                    "path": str(item.relative_to(root)),
                    "sha256": _sha256_file(item),
                    "size": item.stat().st_size,
                }
            )
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
        "tree_digest": _json_digest(files),
    }


def _protected_tree_snapshot(
    output_root: Path, patterns: Iterable[str]
) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
        if pattern == "czy":
            candidate = output_root.parent / pattern
            if candidate.is_dir():
                directories.add(candidate)

    def logical(path: Path) -> str:
        try:
            return str(path.relative_to(output_root))
        except ValueError:
            return str(Path("workspace") / path.relative_to(output_root.parent))

    entries: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected input cannot be a symlink: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"protected tree contains a symlink: {path}")
            if not path.is_file():
                continue
            metadata = path.stat()
            entries[logical(path)] = {
                "sha256": _sha256_file(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return {
        "directories": [logical(path) for path in sorted(directories)],
        "file_count": len(entries),
        "total_bytes": sum(int(value["size"]) for value in entries.values()),
        "tree_digest": _json_digest(entries),
    }


class _ExclusiveRepairLock:
    def __init__(self, path: Path):
        self.path = path
        self.inode: int | None = None

    def acquire(self) -> None:
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise RuntimeError(f"aggregation repair lock already exists: {self.path}") from error
        payload = {
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "created_utc": _utc(),
            "automatic_resume_allowed": False,
        }
        _write_all(
            descriptor,
            (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
        self.inode = os.fstat(descriptor).st_ino
        os.close(descriptor)

    def release(self) -> None:
        if self.inode is None:
            return
        if self.path.stat().st_ino != self.inode:
            raise RuntimeError("aggregation repair lock inode changed")
        self.path.unlink()
        self.inode = None


def _repair_source_manifest(workspace: Path) -> dict[str, Any]:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *REPAIR_SOURCE_SCOPE,
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"aggregation repair source scope is dirty:\n{status}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files: dict[str, Any] = {}
    for relative in REPAIR_SOURCE_SCOPE:
        path = workspace / relative
        _regular_file(path, context="repair source")
        files[relative] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    payload = {
        "git_commit": commit,
        "git_tree": tree,
        "scope_clean": True,
        "files": files,
    }
    payload["digest"] = _json_digest(payload)
    return payload


def _repair_git_diff(workspace: Path, original_commit: str) -> dict[str, set[str]]:
    lines = subprocess.run(
        ["git", "diff", "--name-status", original_commit, "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    result: dict[str, set[str]] = {}
    for line in lines:
        status, name = line.split("\t", maxsplit=1)
        result.setdefault(status, set()).add(name)
    return result


def _validate_original_evidence_files(
    *,
    workspace: Path,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    fingerprint = preregistration.get("evidence_fingerprint")
    if not isinstance(fingerprint, Mapping) or not isinstance(
        fingerprint.get("files"), Mapping
    ):
        raise RuntimeError("original preregistration lacks an evidence fingerprint")
    changed: dict[str, Any] = {}
    for name, expected in fingerprint["files"].items():
        path = Path(str(name))
        if not path.is_absolute():
            path = workspace / path
        _regular_file(path, context="original formal evidence input")
        observed = {"sha256": _sha256_file(path), "size": path.stat().st_size}
        if observed == expected:
            continue
        if str(name) != ALLOWED_ORIGINAL_SOURCE_CHANGE:
            raise RuntimeError(f"original formal evidence input changed: {name}")
        changed[str(name)] = {"original": expected, "repair": observed}
    if set(changed) != {ALLOWED_ORIGINAL_SOURCE_CHANGE}:
        raise RuntimeError("repair must change exactly the reporting implementation")
    original_commit = str(preregistration.get("runner_git_commit", ""))
    if not original_commit:
        raise RuntimeError("original preregistration has no runner git commit")
    git_diff = _repair_git_diff(workspace, original_commit)
    if git_diff != EXPECTED_REPAIR_DIFF:
        raise RuntimeError(
            "git changes since the original run exceed the aggregation repair scope: "
            f"{git_diff}"
        )
    return {
        "original_fingerprint_digest": fingerprint.get("digest"),
        "unchanged_files_verified": len(fingerprint["files"]) - len(changed),
        "allowed_changed_files": changed,
        "git_diff_from_original_formal_commit": {
            key: sorted(value) for key, value in git_diff.items()
        },
        "new_files_are_confined_to_repair_source_scope": True,
    }


def _expected_anchor(snapshot: Mapping[str, Any], anchor: Mapping[str, Any]) -> None:
    if (
        int(snapshot["file_count"]) != int(anchor["file_count"])
        or str(snapshot["tree_digest"]) != str(anchor["tree_digest"])
    ):
        raise RuntimeError(
            "sealed repair input tree changed: "
            f"root={snapshot['root']}, expected={anchor}, "
            f"observed_file_count={snapshot['file_count']}, "
            f"observed_digest={snapshot['tree_digest']}"
        )


def _validate_completion(
    *,
    root: Path,
    robot: str,
    seed: int,
    expected_sha256: str,
    expected_checkpoint_count: int,
) -> dict[str, Any]:
    completion_path = root / robot / "combination_complete.json"
    _regular_file(completion_path, context="combination completion manifest")
    if _sha256_file(completion_path) != expected_sha256:
        raise RuntimeError(f"combination completion hash changed: seed{seed}/{robot}")
    completion = _read_json(completion_path)
    if (
        completion.get("protocol")
        != "test_v4_combination_complete_but_not_formal_completion_marker"
        or completion.get("robot") != robot
        or int(completion.get("training_seed", -1)) != seed
        or not bool(completion.get("all_checkpoints_hash_validated", False))
        or bool(completion.get("eligible_without_aggregate_final_manifest", True))
    ):
        raise RuntimeError(f"combination completion contract changed: seed{seed}/{robot}")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"combination artifact manifest is missing: seed{seed}/{robot}")
    combination_root = root / robot
    actual_files = {
        str(path.relative_to(combination_root))
        for path in combination_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != set(artifacts) | {"combination_complete.json"}:
        raise RuntimeError(f"combination file set changed: seed{seed}/{robot}")
    for relative, expected in artifacts.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("combination artifact path escapes its sealed root")
        path = combination_root / relative_path
        _regular_file(path, context="combination artifact")
        if (
            path.stat().st_size != int(expected["size"])
            or _sha256_file(path) != str(expected["sha256"])
        ):
            raise RuntimeError(f"combination artifact changed: {path}")
    checkpoint_manifests = sorted(
        combination_root.glob("checkpoints/*/*/checkpoint_manifest.json")
    )
    if len(checkpoint_manifests) != expected_checkpoint_count:
        raise RuntimeError(f"checkpoint count changed: seed{seed}/{robot}")
    for path in checkpoint_manifests:
        checkpoint = _read_json(path)
        quiet = checkpoint.get("quiet_host_evidence")
        if (
            not isinstance(quiet, Mapping)
            or int(quiet.get("query_intervals_without_new_monitor_sample", -1)) != 0
            or int(quiet.get("minimum_monitor_samples_since_query_start", 0)) < 1
            or not bool(quiet.get("background_monitor", False))
            or bool(quiet.get("synchronous_ps_or_nvidia_smi_per_query", True))
        ):
            raise RuntimeError(f"checkpoint quiet-host coverage changed: {path}")
    return {
        "robot": robot,
        "training_seed": seed,
        "completion_manifest_sha256": expected_sha256,
        "artifact_count": len(artifacts),
        "checkpoint_count": len(checkpoint_manifests),
        "all_artifact_hashes_verified": True,
        "all_checkpoint_quiet_host_contracts_verified": True,
    }


def _aggregate_only(
    *, seed_roots: Mapping[int, Path], alpha: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    for robot in ROBOTS:
        root = seed_roots[PRIMARY_SEED] / robot
        primary[robot] = {
            "claim_gate": _read_json(root / "claim_gate_v4.json"),
            "summary": _read_json(root / "summary_v4.json"),
            "ood_abstention": _read_json(root / "ood_abstention_v4.json"),
        }
        intervals[robot] = _read_json(root / "paired_intervals_v4.json")
        sensitivity[robot] = {
            f"seed{seed}": _read_json(
                seed_roots[seed] / robot / "summary_v4.json"
            )
            for seed in SENSITIVITY_SEEDS
        }
    joint_holm = joint_holm_confirmatory(intervals, alpha=alpha)
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
    summary = {"primary": primary, "sensitivity": sensitivity}
    return joint_holm, paper_gate, summary


def _validate_configuration(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise RuntimeError("aggregation repair protocol differs from the frozen repair")
    contract = config.get("execution_contract", {})
    required = {
        "aggregation_only": True,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "original_outputs_mutation_allowed": False,
        "automatic_resume_allowed": False,
    }
    if any(contract.get(key) != value for key, value in required.items()):
        raise RuntimeError("aggregation repair execution contract changed")
    if float(config["statistics"]["familywise_alpha"]) != 0.05:
        raise RuntimeError("aggregation repair Holm alpha must remain 0.05")
    if (
        config["statistics"].get("multiplicity_correction") != "Holm"
        or int(config["statistics"].get("hypothesis_count", -1)) != 8
    ):
        raise RuntimeError("aggregation repair Holm family must remain eight hypotheses")


def _execute_repair(
    config_path: Path,
    *,
    workspace: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    workspace = workspace.resolve()
    config = _read_yaml(config_path)
    _validate_configuration(config)
    output_root = workspace / "outputs"
    final = _resolve(config_path, config["output"]["final_directory"])
    staging = _resolve(config_path, config["output"]["staging_directory"])
    lock_path = _resolve(config_path, config["output"]["lock_path"])
    if (
        final.parent != output_root
        or staging.parent != output_root
        or lock_path.parent != output_root
    ):
        raise RuntimeError("aggregation repair outputs must be direct children of outputs/")

    source_manifest = _repair_source_manifest(workspace)
    formal_config = _resolve(config_path, config["inputs"]["formal_config_path"])
    _regular_file(formal_config, context="frozen formal config")
    if _sha256_file(formal_config) != config["inputs"]["formal_config_sha256"]:
        raise RuntimeError("frozen formal configuration changed")
    formal_config_payload = _read_yaml(formal_config)

    aggregate_root = _resolve(
        config_path, config["inputs"]["aggregate_failure"]["path"]
    )
    seed_roots = {
        int(seed): _resolve(config_path, payload["path"])
        for seed, payload in config["inputs"]["seed_roots"].items()
    }
    if set(seed_roots) != set(SEEDS):
        raise RuntimeError("repair requires exactly the three sealed training seeds")
    input_roots = {"aggregate_failure": aggregate_root, **{
        f"seed{seed}": seed_roots[seed] for seed in SEEDS
    }}
    before = {name: _tree_snapshot(path) for name, path in input_roots.items()}
    _expected_anchor(before["aggregate_failure"], config["inputs"]["aggregate_failure"])
    for seed in SEEDS:
        _expected_anchor(before[f"seed{seed}"], config["inputs"]["seed_roots"][seed])

    control = config["inputs"]["control_plane"]
    control_files = {
        "preregistration": aggregate_root / "test_v4_preregistration.json",
        "dataset_manifest": aggregate_root / "test_v4_dataset_manifest.json",
        "control_plane_seal": aggregate_root / "test_v4_control_plane_seal.json",
        "latest_failure_manifest": aggregate_root / "latest_failure_manifest.json",
    }
    for name, path in control_files.items():
        _regular_file(path, context=f"original {name}")
        if _sha256_file(path) != str(control[f"{name}_sha256"]):
            raise RuntimeError(f"original {name} hash changed")
    preregistration = _read_json(control_files["preregistration"])
    failure = _read_json(control_files["latest_failure_manifest"])
    expected_failure = config["inputs"]["expected_failure"]
    if (
        failure.get("failure_classification")
        != expected_failure["failure_classification"]
        or bool(failure.get("resume_eligible", True))
        or failure.get("exception_type") != expected_failure["exception_type"]
        or failure.get("exception_message") != expected_failure["exception_message"]
        or failure.get("phase") != expected_failure["phase"]
    ):
        raise RuntimeError("original non-resumable aggregate failure contract changed")
    if preregistration.get("release_digest") != control["release_digest"]:
        raise RuntimeError("original preregistration release digest changed")
    if (
        preregistration.get("evidence_fingerprint", {}).get("digest")
        != control["original_evidence_fingerprint_digest"]
    ):
        raise RuntimeError("original evidence fingerprint anchor changed")
    original_source_audit = _validate_original_evidence_files(
        workspace=workspace,
        preregistration=preregistration,
    )
    protected_before = _protected_tree_snapshot(
        output_root, [str(value) for value in formal_config_payload["protected_outputs"]]
    )
    if protected_before != preregistration["protected_outputs_before"]:
        raise RuntimeError("protected evidence changed before aggregation repair")

    combination_validations: list[dict[str, Any]] = []
    for seed in SEEDS:
        anchor = config["inputs"]["seed_roots"][seed]
        for robot in ROBOTS:
            combination_validations.append(
                _validate_completion(
                    root=seed_roots[seed],
                    robot=robot,
                    seed=seed,
                    expected_sha256=str(anchor["combination_complete_sha256"][robot]),
                    expected_checkpoint_count=int(
                        config["inputs"]["expected_checkpoint_count_per_combination"]
                    ),
                )
            )

    repair_config_sha256 = _sha256_file(config_path)

    def revalidate_immutable_inputs() -> dict[str, Any]:
        current_trees = {
            name: _tree_snapshot(path) for name, path in input_roots.items()
        }
        if current_trees != before:
            raise RuntimeError("sealed formal input trees changed during aggregation repair")
        current_protected = _protected_tree_snapshot(
            output_root,
            [str(value) for value in formal_config_payload["protected_outputs"]],
        )
        if current_protected != protected_before:
            raise RuntimeError("protected evidence changed during aggregation repair")
        if _sha256_file(config_path) != repair_config_sha256:
            raise RuntimeError("aggregation repair config changed during execution")
        if _sha256_file(formal_config) != config["inputs"]["formal_config_sha256"]:
            raise RuntimeError("formal test config changed during aggregation repair")
        for name, path in control_files.items():
            if _sha256_file(path) != str(control[f"{name}_sha256"]):
                raise RuntimeError(f"original {name} changed during aggregation repair")
        current_source = _repair_source_manifest(workspace)
        if current_source != source_manifest:
            raise RuntimeError("aggregation repair source changed during execution")
        current_original_source = _validate_original_evidence_files(
            workspace=workspace,
            preregistration=preregistration,
        )
        if current_original_source != original_source_audit:
            raise RuntimeError("original source/asset audit changed during aggregation repair")
        payload = {
            "input_trees": current_trees,
            "protected_tree": current_protected,
            "repair_source_digest": current_source["digest"],
            "original_source_audit": current_original_source,
            "repair_config_sha256": repair_config_sha256,
            "formal_config_sha256": config["inputs"]["formal_config_sha256"],
            "control_plane_hashes": {
                name: str(control[f"{name}_sha256"])
                for name in control_files
            },
        }
        payload["digest"] = _json_digest(payload)
        return payload

    lock = _ExclusiveRepairLock(lock_path)
    lock.acquire()
    staging_created = False
    try:
        if final.exists() or final.is_symlink():
            raise RuntimeError("completed aggregation repair already exists; rerun is forbidden")
        if staging.exists() or staging.is_symlink():
            raise RuntimeError("prior aggregation repair staging exists; overwrite is forbidden")
        staging.mkdir(parents=False)
        staging_created = True
        repair_preregistration = {
            "protocol": PROTOCOL,
            "created_utc": _utc(),
            "repair_reason": (
                "JSON object key order was incorrectly treated as confirmatory "
                "metric membership after sort_keys serialization"
            ),
            "statistical_semantics_changed": False,
            "original_failure_classification_changed": False,
            "execution_contract": config["execution_contract"],
            "statistics": config["statistics"],
            "repair_config_sha256": repair_config_sha256,
            "repair_source_manifest": source_manifest,
            "original_source_audit": original_source_audit,
            "input_tree_digests": {
                name: {
                    "file_count": snapshot["file_count"],
                    "total_bytes": snapshot["total_bytes"],
                    "tree_digest": snapshot["tree_digest"],
                }
                for name, snapshot in before.items()
            },
            "control_plane": control,
            "expected_failure": expected_failure,
            "combination_validations": combination_validations,
            "old_test_performance_used_for_selection": False,
            "threshold_or_gate_changes": False,
        }
        preregistration_path = staging / "aggregation_repair_preregistration.json"
        _write_json_new(preregistration_path, repair_preregistration)
        input_manifest = {
            "protocol": f"{PROTOCOL}_input_manifest",
            "created_utc": _utc(),
            "repair_preregistration_sha256": _sha256_file(preregistration_path),
            "roots": before,
            "combined_tree_digest": _json_digest(before),
            "all_six_combinations_hash_validated": True,
            "query_records_read_for_aggregation": False,
            "query_record_files_hash_validated_only": True,
        }
        input_manifest_path = staging / "aggregation_repair_input_manifest.json"
        _write_json_new(input_manifest_path, input_manifest)

        joint_holm, paper_gate, summary = _aggregate_only(
            seed_roots=seed_roots,
            alpha=float(config["statistics"]["familywise_alpha"]),
        )
        report_payloads = {
            "joint_holm_v4.json": joint_holm,
            "paper_gate_v4.json": paper_gate,
            "aggregate_summary_v4.json": summary,
        }
        report_hashes: dict[str, Any] = {}
        for name, payload in report_payloads.items():
            path = staging / name
            _write_json_new(path, payload)
            report_hashes[name] = {
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }

        final_input_recheck = revalidate_immutable_inputs()
        after = final_input_recheck["input_trees"]
        protected_after = final_input_recheck["protected_tree"]
        integrity = {
            "protocol": f"{PROTOCOL}_integrity",
            "created_utc": _utc(),
            "input_trees_before": before,
            "input_trees_after": after,
            "input_trees_unchanged": True,
            "protected_tree_before": protected_before,
            "protected_tree_after": protected_after,
            "protected_tree_unchanged": True,
            "all_six_combinations_hash_validated": True,
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
            "original_failure_evidence_preserved": True,
            "original_failure_classification_changed": False,
            "final_input_recheck_digest": final_input_recheck["digest"],
        }
        integrity_path = staging / "aggregation_repair_integrity.json"
        _write_json_new(integrity_path, integrity)
        chain = {
            "aggregation_repair_preregistration.json": _sha256_file(
                preregistration_path
            ),
            "aggregation_repair_input_manifest.json": _sha256_file(
                input_manifest_path
            ),
            **{name: value["sha256"] for name, value in report_hashes.items()},
            "aggregation_repair_integrity.json": _sha256_file(integrity_path),
        }
        final_manifest: dict[str, Any] = {
            "protocol": f"{PROTOCOL}_final_manifest",
            "completed_utc": _utc(),
            "aggregation_only_repair": True,
            "authoritative_output_namespace": str(final.relative_to(workspace)),
            "original_formal_runner_natural_exit": False,
            "six_combination_natural_exits": True,
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
            "original_failure_evidence_preserved": True,
            "original_failure_classification_changed": False,
            "threshold_or_statistical_semantics_changed": False,
            "paper_gate_pass": bool(paper_gate["both_robot_gates_pass"]),
            "hash_chain": chain,
            "hash_chain_digest": _json_digest(chain),
            "automatic_rerun_allowed": False,
            "original_incomplete_paths_promoted_or_renamed": False,
            "final_input_recheck_digest": final_input_recheck["digest"],
            "second_pre_promotion_recheck_required": True,
        }
        final_manifest["manifest_payload_digest"] = _json_digest(final_manifest)
        final_manifest_path = staging / "test_v4_repair_final_manifest.json"
        _write_json_new(final_manifest_path, final_manifest)
        final_manifest_sha256 = _sha256_file(final_manifest_path)
        pre_promotion_recheck = revalidate_immutable_inputs()
        if pre_promotion_recheck != final_input_recheck:
            raise RuntimeError("immutable inputs changed immediately before repair promotion")
        staging.rename(final)
        staging_created = False
        return {
            **final_manifest,
            "final_manifest_sha256": final_manifest_sha256,
        }
    except BaseException as error:
        if staging_created:
            failure_path = staging / "aggregation_repair_failure_manifest.json"
            if not failure_path.exists():
                _write_json_new(
                    failure_path,
                    {
                        "protocol": f"{PROTOCOL}_failure",
                        "created_utc": _utc(),
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                        "automatic_resume_allowed": False,
                        "original_outputs_modified": False,
                    },
                )
        raise
    finally:
        lock.release()


def run(config_path: str | Path) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[3]
    canonical = (workspace / "configs" / "test_v4_aggregate_repair_v1.yaml").resolve()
    path = Path(config_path).resolve()
    if path != canonical:
        raise RuntimeError("aggregation repair requires its canonical frozen config")
    config = _read_yaml(path)
    expected_paths = {
        config["inputs"]["aggregate_failure"]["path"]: (
            workspace / "outputs" / ".test_v4_aggregate.incomplete"
        ),
        config["inputs"]["seed_roots"][17]["path"]: (
            workspace / "outputs" / ".test_v4_seed17.incomplete"
        ),
        config["inputs"]["seed_roots"][29]["path"]: (
            workspace / "outputs" / ".test_v4_seed29.incomplete"
        ),
        config["inputs"]["seed_roots"][43]["path"]: (
            workspace / "outputs" / ".test_v4_seed43.incomplete"
        ),
        config["output"]["staging_directory"]: (
            workspace / "outputs" / ".test_v4_aggregate_repair_v1.incomplete"
        ),
        config["output"]["final_directory"]: (
            workspace / "outputs" / "test_v4_aggregate_repair_v1"
        ),
        config["output"]["lock_path"]: (
            workspace / "outputs" / ".test_v4_aggregate_repair_v1.lock"
        ),
    }
    if any(
        _resolve(path, value) != expected.resolve()
        for value, expected in expected_paths.items()
    ):
        raise RuntimeError("aggregation repair input/output paths differ from the frozen repair")
    return _execute_repair(path, workspace=workspace)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    result = run(_parser().parse_args().config)
    print(
        json.dumps(
            {
                "completed": True,
                "final_manifest_sha256": result["final_manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["run"]
