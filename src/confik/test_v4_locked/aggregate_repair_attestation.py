"""Retrospective, read-only attestation for the sealed test_v4 repair v1.

The attestation does not repair, rerun, or replace the v1 aggregation.  It
discloses the split Git metadata/work-tree invocation, proves that the source
scope recorded by v1 is byte-identical to a permanent commit, independently
recomputes the aggregation with the Python standard library, and seals the
result in a new atomic namespace.  The v1 tree and all formal input trees are
strictly read-only inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import time
import traceback
from typing import Any, Iterable, Mapping


PROTOCOL = "test_v4_aggregate_repair_v1_attestation_v1"
ROBOTS = ("panda", "ur5e")
SEEDS = (17, 29, 43)
PRIMARY_SEED = 17
SENSITIVITY_SEEDS = (29, 43)
CONFIRMATORY_METRICS = (
    "feasible_success_gap",
    "feasible_p95_latency_ratio",
    "feasible_p99_latency_ratio",
    "trajectory_completion_gap",
)
ATTESTATION_SOURCE_SCOPE = (
    "src/confik/test_v4_locked/aggregate_repair_attestation.py",
    "configs/test_v4_aggregate_repair_v1_attestation_v1.json",
    "scripts/run_test_v4_aggregate_repair_v1_attestation_v1.sh",
    "tests/test_test_v4_aggregate_repair_attestation.py",
)
EXPECTED_V1_SOURCE_SCOPE = (
    "configs/test_v4_aggregate_repair_v1.yaml",
    "scripts/run_test_v4_aggregate_repair_v1.sh",
    "src/confik/test_v4_locked/aggregate_repair.py",
    "src/confik/test_v4_locked/reporting.py",
    "tests/test_test_v4_aggregate_repair.py",
    "tests/test_test_v4_reporting.py",
)


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


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path, context="JSON input")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return payload


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while sealing attestation evidence")
        view = view[written:]


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"attestation output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, *, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{context} is missing, non-regular, or a symlink: {path}")


def _resolve(config_path: Path, value: object) -> Path:
    return (config_path.parent / str(value)).resolve()


def _safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe evidence-relative path: {value}")
    return path


def _tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evidence root is missing or a symlink: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
        "tree_digest": _json_digest(files),
    }


def _tree_anchor(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": int(snapshot["file_count"]),
        "total_bytes": int(snapshot["total_bytes"]),
        "tree_digest": str(snapshot["tree_digest"]),
    }


def _assert_tree_anchor(
    snapshot: Mapping[str, Any], anchor: Mapping[str, Any], *, context: str
) -> None:
    expected = {
        "file_count": int(anchor["file_count"]),
        "total_bytes": int(anchor["total_bytes"]),
        "tree_digest": str(anchor["tree_digest"]),
    }
    if _tree_anchor(snapshot) != expected:
        raise RuntimeError(
            f"{context} tree differs from its frozen anchor: "
            f"expected={expected}, observed={_tree_anchor(snapshot)}"
        )


def _protected_tree_snapshot(
    workspace: Path, output_root: Path, patterns: Iterable[str]
) -> dict[str, Any]:
    directories: set[Path] = set()
    for pattern in patterns:
        directories.update(path for path in output_root.glob(pattern) if path.is_dir())
        if pattern == "czy":
            candidate = workspace / "czy"
            if candidate.is_dir():
                directories.add(candidate)

    def logical(path: Path) -> str:
        try:
            return str(path.relative_to(output_root))
        except ValueError:
            return str(Path("workspace") / path.relative_to(workspace))

    entries: dict[str, Any] = {}
    for directory in sorted(directories):
        if directory.is_symlink():
            raise RuntimeError(f"protected evidence root is a symlink: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"protected evidence contains a symlink: {path}")
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


def _git(
    workspace: Path, *arguments: str, binary: bool = False
) -> str | bytes:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def _attestation_source_manifest(workspace: Path) -> dict[str, Any]:
    if os.environ.get("GIT_DIR") or os.environ.get("GIT_WORK_TREE"):
        raise RuntimeError("attestation execution forbids GIT_DIR/GIT_WORK_TREE overrides")
    top = Path(str(_git(workspace, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != workspace.resolve():
        raise RuntimeError("attestation source and Git work tree are not co-located")
    git_dir = Path(str(_git(workspace, "rev-parse", "--absolute-git-dir")).strip())
    expected_git_dir = (workspace / ".git").resolve()
    if git_dir.resolve() != expected_git_dir:
        raise RuntimeError("attestation requires the main repository Git directory")
    status = str(
        _git(
            workspace,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *ATTESTATION_SOURCE_SCOPE,
        )
    ).strip()
    if status:
        raise RuntimeError(f"attestation source scope is dirty:\n{status}")
    files: dict[str, Any] = {}
    for relative in ATTESTATION_SOURCE_SCOPE:
        path = workspace / relative
        _regular_file(path, context="attestation source")
        files[relative] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    payload: dict[str, Any] = {
        "git_commit": str(_git(workspace, "rev-parse", "HEAD")).strip(),
        "git_tree": str(_git(workspace, "rev-parse", "HEAD^{tree}")).strip(),
        "git_top_level": str(top),
        "git_dir": str(git_dir.resolve()),
        "scope_clean": True,
        "global_cleanliness_asserted": False,
        "files": files,
    }
    payload["digest"] = _json_digest(payload)
    return payload


class _ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.inode: int | None = None

    def acquire(self) -> None:
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise RuntimeError(f"attestation lock already exists: {self.path}") from error
        try:
            _write_all(
                descriptor,
                (
                    json.dumps(
                        {
                            "protocol": PROTOCOL,
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "created_utc": _utc(),
                            "automatic_resume_allowed": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            os.fsync(descriptor)
            self.inode = os.fstat(descriptor).st_ino
        finally:
            os.close(descriptor)

    def release(self) -> None:
        if self.inode is None:
            return
        metadata = self.path.stat()
        if metadata.st_ino != self.inode:
            raise RuntimeError("attestation lock inode changed")
        self.path.unlink()
        self.inode = None


def _descriptor(path: Path) -> dict[str, Any]:
    _regular_file(path, context="evidence artifact")
    return {"sha256": _sha256_file(path), "size": path.stat().st_size}


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Standard-library Holm step-down, matching the frozen stable ordering."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted[name] = running
    return adjusted


def _independent_recomputation(
    *, v1_root: Path, seed_roots: Mapping[int, Path], alpha: float
) -> dict[str, Any]:
    intervals: dict[str, dict[str, Any]] = {}
    unadjusted: dict[str, float] = {}
    for robot in ROBOTS:
        payload = _read_json(
            seed_roots[PRIMARY_SEED] / robot / "paired_intervals_v4.json"
        )
        intervals[robot] = payload
        family = payload.get("inference_family")
        metrics = payload.get("metrics")
        if (
            not isinstance(family, Mapping)
            or tuple(family.get("members", ())) != CONFIRMATORY_METRICS
            or not isinstance(metrics, Mapping)
            or len(metrics) != len(CONFIRMATORY_METRICS)
            or set(metrics) != set(CONFIRMATORY_METRICS)
        ):
            raise RuntimeError(f"{robot} confirmatory family is not the frozen family")
        for metric in CONFIRMATORY_METRICS:
            metric_payload = metrics[metric]
            if not isinstance(metric_payload, Mapping):
                raise RuntimeError(f"invalid stored metric payload: {robot}/{metric}")
            value = float(metric_payload["one_sided_unadjusted_p"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"invalid stored p-value: {robot}/{metric}")
            unadjusted[f"{robot}/{metric}"] = value
    adjusted = _holm(unadjusted)
    hypotheses = {
        name: {
            "robot": name.split("/", 1)[0],
            "metric": name.split("/", 1)[1],
            "one_sided_unadjusted_p": unadjusted[name],
            "holm_adjusted_p": adjusted[name],
            "reject_margin_null": adjusted[name] <= alpha,
        }
        for name in unadjusted
    }
    recomputed_joint = {
        "method": "Holm",
        "scope": "Panda and UR5e x four prespecified confirmatory claims",
        "alpha": alpha,
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "all_confirmatory_nulls_rejected": all(
            bool(payload["reject_margin_null"]) for payload in hypotheses.values()
        ),
        "operational_finite_test_gates_included": False,
    }

    primary: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for robot in ROBOTS:
        root = seed_roots[PRIMARY_SEED] / robot
        primary[robot] = {
            "claim_gate": _read_json(root / "claim_gate_v4.json"),
            "summary": _read_json(root / "summary_v4.json"),
            "ood_abstention": _read_json(root / "ood_abstention_v4.json"),
        }
        sensitivity[robot] = {
            f"seed{seed}": _read_json(seed_roots[seed] / robot / "summary_v4.json")
            for seed in SENSITIVITY_SEEDS
        }
    robot_gates = {
        robot: bool(primary[robot]["claim_gate"]["formal_gate_pass"])
        for robot in ROBOTS
    }
    recomputed_paper_gate = {
        "protocol": "test_v4 robot-level confirmatory aggregation",
        "primary_training_seed": PRIMARY_SEED,
        "sensitivity_training_seeds": list(SENSITIVITY_SEEDS),
        "sensitivity_seeds_are_not_independent_query_samples": True,
        "robot_gates": robot_gates,
        "robot_gates_are_pre_joint_holm": True,
        "joint_holm_gate_pass": bool(
            recomputed_joint["all_confirmatory_nulls_rejected"]
        ),
        "joint_holm_is_required_for_formal_gate": True,
        "both_robot_gates_pass": all(robot_gates.values())
        and bool(recomputed_joint["all_confirmatory_nulls_rejected"]),
        "test_set_retuning_performed": False,
    }
    recomputed_summary = {"primary": primary, "sensitivity": sensitivity}
    stored = {
        "joint_holm_v4.json": _read_json(v1_root / "joint_holm_v4.json"),
        "paper_gate_v4.json": _read_json(v1_root / "paper_gate_v4.json"),
        "aggregate_summary_v4.json": _read_json(
            v1_root / "aggregate_summary_v4.json"
        ),
    }
    recomputed = {
        "joint_holm_v4.json": recomputed_joint,
        "paper_gate_v4.json": recomputed_paper_gate,
        "aggregate_summary_v4.json": recomputed_summary,
    }
    matches = {name: recomputed[name] == stored[name] for name in recomputed}
    if not all(matches.values()):
        raise RuntimeError(f"independent aggregation differs from v1: {matches}")
    return {
        "protocol": f"{PROTOCOL}_independent_recomputation",
        "implementation": "Python standard library only; no confik reporting import",
        "bootstrap_resamples_executed": 0,
        "query_records_parsed_or_used_for_recomputation": False,
        "query_record_files_hash_verified_only": True,
        "stored_pvalues_reused": True,
        "familywise_alpha": alpha,
        "confirmatory_members": list(CONFIRMATORY_METRICS),
        "semantic_matches": matches,
        "all_semantic_matches": True,
        "stored_artifacts": {
            name: _descriptor(v1_root / name) for name in stored
        },
        "recomputed_payload_digests": {
            name: _json_digest(payload) for name, payload in recomputed.items()
        },
        "observed_results_not_used_as_acceptance_criteria": {
            "robot_gates": robot_gates,
            "joint_holm_gate_pass": bool(
                recomputed_joint["all_confirmatory_nulls_rejected"]
            ),
            "paper_gate_pass": bool(recomputed_paper_gate["both_robot_gates_pass"]),
        },
        "acceptance_rule": (
            "semantic equality to the deterministic aggregation of sealed inputs; "
            "no required direction or pass/fail value for any scientific gate"
        ),
        "outcome_direction_hardcoded": False,
    }


def _parse_bundle_header(path: Path) -> dict[str, Any]:
    references: dict[str, str] = {}
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        signature = handle.readline().decode("ascii").rstrip("\n")
        if signature not in {"# v2 git bundle", "# v3 git bundle"}:
            raise RuntimeError("unrecognized Git bundle signature")
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("truncated Git bundle header")
            if line == b"\n":
                break
            decoded = line.decode("utf-8").rstrip("\n")
            if decoded.startswith("-"):
                prerequisites.append(decoded)
            elif decoded.startswith("@"):  # v3 capability line
                continue
            else:
                object_id, reference = decoded.split(" ", 1)
                references[reference] = object_id
    return {
        "signature": signature,
        "references": references,
        "prerequisites": prerequisites,
        "self_contained": not prerequisites,
    }


def _source_commit_verification(
    *,
    workspace: Path,
    v1_preregistration: Mapping[str, Any],
    source_config: Mapping[str, Any],
    bundle_path: Path,
) -> dict[str, Any]:
    if Path(str(source_config["main_repository"])).resolve() != workspace.resolve():
        raise RuntimeError("permanent repair source repository path changed")
    commit = str(source_config["commit"])
    parent = str(source_config["parent"])
    tree = str(source_config["tree"])
    permanent_ref = str(source_config["permanent_ref"])
    if str(_git(workspace, "rev-parse", permanent_ref)).strip() != commit:
        raise RuntimeError("permanent repair source ref moved")
    if str(_git(workspace, "rev-parse", f"{commit}^")).strip() != parent:
        raise RuntimeError("repair source parent differs from frozen lineage")
    if str(_git(workspace, "rev-parse", f"{commit}^{{tree}}")).strip() != tree:
        raise RuntimeError("repair source tree differs from frozen lineage")
    source_manifest = v1_preregistration.get("repair_source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise RuntimeError("v1 lacks a repair source manifest")
    if (
        source_manifest.get("git_commit") != commit
        or source_manifest.get("git_tree") != tree
        or set(source_manifest.get("files", {})) != set(EXPECTED_V1_SOURCE_SCOPE)
        or not bool(source_manifest.get("scope_clean", False))
    ):
        raise RuntimeError("v1 repair source scope differs from the attested lineage")
    files: dict[str, Any] = {}
    for relative in EXPECTED_V1_SOURCE_SCOPE:
        expected = source_manifest["files"][relative]
        blob = _git(workspace, "show", f"{commit}:{relative}", binary=True)
        if not isinstance(blob, bytes):
            raise AssertionError("binary Git read returned text")
        commit_descriptor = {"sha256": sha256(blob).hexdigest(), "size": len(blob)}
        if commit_descriptor != expected:
            raise RuntimeError(f"v1 source descriptor differs from commit: {relative}")
        physical = _descriptor(workspace / relative)
        if physical != expected:
            raise RuntimeError(f"main work-tree source differs from v1: {relative}")
        files[relative] = {
            "v1_manifest": expected,
            "permanent_commit": commit_descriptor,
            "main_work_tree": physical,
            "all_byte_identical": True,
        }
    source_payload = dict(source_manifest)
    source_digest = source_payload.pop("digest", None)
    if source_digest != _json_digest(source_payload):
        raise RuntimeError("v1 repair source manifest digest is invalid")

    if bundle_path.exists() or bundle_path.is_symlink():
        raise RuntimeError("source bundle output already exists")
    _git(workspace, "bundle", "create", str(bundle_path), permanent_ref)
    _regular_file(bundle_path, context="self-contained source bundle")
    with bundle_path.open("rb") as handle:
        os.fsync(handle.fileno())
    verification = str(_git(workspace, "bundle", "verify", str(bundle_path))).strip()
    header = _parse_bundle_header(bundle_path)
    if not bool(header["self_contained"]):
        raise RuntimeError("repair source bundle has external prerequisites")
    if header["references"].get(permanent_ref) != commit:
        raise RuntimeError("repair source bundle does not pin the permanent ref")
    listed = str(_git(workspace, "bundle", "list-heads", str(bundle_path))).splitlines()
    if f"{commit} {permanent_ref}" not in listed:
        raise RuntimeError("repair source bundle head differs from the permanent ref")
    return {
        "protocol": f"{PROTOCOL}_source_commit_verification",
        "main_repository": str(workspace),
        "permanent_ref": permanent_ref,
        "commit": commit,
        "parent": parent,
        "tree": tree,
        "commit_was_recorded_in_v1_before_this_attestation": True,
        "permanent_ref_was_imported_retrospectively": True,
        "v1_scope_clean_interpretation": "only the six recorded repair source files",
        "global_worktree_cleanliness_asserted_for_v1": False,
        "source_files": files,
        "all_source_files_byte_identical": True,
        "bundle": {
            **_descriptor(bundle_path),
            "path": bundle_path.name,
            "header": header,
            "git_bundle_verify_succeeded": True,
            "git_bundle_verify_output": verification,
        },
    }


def _v1_integrity_reaudit(
    *,
    v1_root: Path,
    original_snapshots: Mapping[str, Mapping[str, Any]],
    protected_current: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    v1_snapshot = _tree_snapshot(v1_root)
    _assert_tree_anchor(v1_snapshot, config["v1"]["tree"], context="repair v1")
    final_path = v1_root / "test_v4_repair_final_manifest.json"
    if _sha256_file(final_path) != str(config["v1"]["final_manifest_sha256"]):
        raise RuntimeError("v1 final manifest hash differs from the frozen anchor")
    final_manifest = _read_json(final_path)
    if final_manifest.get("protocol") != "test_v4_aggregate_repair_v1_final_manifest":
        raise RuntimeError("v1 final manifest protocol changed")
    chain = final_manifest.get("hash_chain")
    if not isinstance(chain, Mapping):
        raise RuntimeError("v1 final manifest lacks a hash chain")
    actual_names = {
        path.name for path in v1_root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_names != set(chain) | {final_path.name}:
        raise RuntimeError("v1 artifact set differs from its final hash chain")
    for name, expected_hash in chain.items():
        path = v1_root / _safe_relative(name)
        if _sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"v1 hash-chain mismatch: {name}")
    if final_manifest.get("hash_chain_digest") != _json_digest(chain):
        raise RuntimeError("v1 hash-chain digest is invalid")
    manifest_payload = dict(final_manifest)
    manifest_digest = manifest_payload.pop("manifest_payload_digest", None)
    if manifest_digest != _json_digest(manifest_payload):
        raise RuntimeError("v1 final manifest payload digest is invalid")

    preregistration_path = v1_root / "aggregation_repair_preregistration.json"
    preregistration = _read_json(preregistration_path)
    input_manifest = _read_json(v1_root / "aggregation_repair_input_manifest.json")
    integrity = _read_json(v1_root / "aggregation_repair_integrity.json")
    if input_manifest.get("repair_preregistration_sha256") != _sha256_file(
        preregistration_path
    ):
        raise RuntimeError("v1 input manifest does not bind its preregistration")
    if input_manifest.get("combined_tree_digest") != _json_digest(
        input_manifest.get("roots")
    ):
        raise RuntimeError("v1 combined input-tree digest is invalid")
    if input_manifest.get("roots") != original_snapshots:
        raise RuntimeError("current formal input trees differ from the v1 input manifest")
    if (
        integrity.get("input_trees_before") != original_snapshots
        or integrity.get("input_trees_after") != original_snapshots
        or not bool(integrity.get("input_trees_unchanged", False))
    ):
        raise RuntimeError("v1 integrity does not preserve identical input trees")
    if (
        integrity.get("protected_tree_before") != protected_current
        or integrity.get("protected_tree_after") != protected_current
        or not bool(integrity.get("protected_tree_unchanged", False))
    ):
        raise RuntimeError("current protected evidence differs from v1")
    zero_fields = (
        "query_rerun_count",
        "solver_invocation_count",
        "model_inference_count",
    )
    if any(int(integrity.get(name, -1)) != 0 for name in zero_fields):
        raise RuntimeError("v1 scientific activity ledger is nonzero")
    if (
        not bool(integrity.get("original_failure_evidence_preserved", False))
        or bool(integrity.get("original_failure_classification_changed", True))
        or bool(final_manifest.get("threshold_or_statistical_semantics_changed", True))
    ):
        raise RuntimeError("v1 preservation/semantic contract changed")
    return (
        {
            "protocol": f"{PROTOCOL}_v1_integrity_reaudit",
            "v1_tree": _tree_anchor(v1_snapshot),
            "v1_final_manifest": _descriptor(final_path),
            "v1_input_manifest": _descriptor(
                v1_root / "aggregation_repair_input_manifest.json"
            ),
            "v1_integrity_artifact": _descriptor(
                v1_root / "aggregation_repair_integrity.json"
            ),
            "v1_hash_chain_valid": True,
            "v1_manifest_payload_digest_valid": True,
            "formal_input_trees": {
                name: _tree_anchor(snapshot)
                for name, snapshot in original_snapshots.items()
            },
            "formal_input_trees_match_v1_before_and_after": True,
            "protected_tree": protected_current,
            "protected_tree_matches_v1_before_and_after": True,
            "query_generation_count": 0,
            "query_rerun_count": 0,
            "query_generation_or_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
            "bootstrap_resamples_executed_by_attestation": 0,
            "original_failure_evidence_preserved": True,
            "original_failure_classification_changed": False,
            "threshold_or_statistical_semantics_changed": False,
            "v1_was_modified_by_attestation": False,
        },
        preregistration,
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise RuntimeError("attestation protocol differs from the frozen protocol")
    contract = config.get("execution_contract")
    expected = {
        "retrospective_attestation": True,
        "modify_v1_allowed": False,
        "modify_original_outputs_allowed": False,
        "query_generation_count": 0,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "bootstrap_resample_count": 0,
        "scientific_gate_direction_is_acceptance_criterion": False,
        "automatic_resume_allowed": False,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("attestation execution contract changed")
    statistics = config.get("statistics")
    if (
        not isinstance(statistics, Mapping)
        or float(statistics.get("familywise_alpha", -1)) != 0.05
        or tuple(statistics.get("confirmatory_metrics", ())) != CONFIRMATORY_METRICS
        or statistics.get("multiplicity_correction") != "Holm"
        or int(statistics.get("hypothesis_count", -1)) != 8
    ):
        raise RuntimeError("attestation statistics contract changed")
    disclosure = config.get("execution_disclosure")
    if (
        not isinstance(disclosure, Mapping)
        or disclosure.get("attestation_timing") != "retrospective"
        or disclosure.get("invocation_claim_source")
        != "operator_and_tool_invocation_record"
        or bool(disclosure.get("invocation_was_independently_traced", True))
        or not bool(disclosure.get("shadow_git_metadata_used", False))
        or not bool(disclosure.get("scope_clean_only", False))
        or bool(disclosure.get("global_worktree_clean_claim", True))
        or disclosure.get("permanent_ref_import_timing")
        != "retrospective_after_v1_execution"
    ):
        raise RuntimeError("attestation disclosure contract changed")


def _execution_provenance_payload(
    config: Mapping[str, Any], attestation_source: Mapping[str, Any]
) -> dict[str, Any]:
    disclosure = dict(config["execution_disclosure"])
    return {
        "protocol": f"{PROTOCOL}_execution_provenance",
        "attestation_timing": "retrospective",
        "original_execution_invocation_source": disclosure[
            "invocation_claim_source"
        ],
        "original_invocation_independently_traced": False,
        "declared_original_invocation": disclosure["declared_original_invocation"],
        "shadow_git_metadata_used": True,
        "git_work_tree_was_main_workspace": True,
        "scope_clean_interpretation": (
            "the six paths in v1 repair_source_manifest matched the shadow "
            "repository index and permanent commit"
        ),
        "global_worktree_cleanliness_asserted": False,
        "global_worktree_cleanliness_required_for_this_attestation": False,
        "source_commit_was_recorded_in_v1_manifest": True,
        "permanent_ref_imported_retrospectively": True,
        "source_commit_content_created_by_attestation": False,
        "verifiable_consequences_not_invocation_claim_alone": [
            "v1 recorded the permanent source commit and tree",
            "all recorded source descriptors match that commit byte-for-byte",
            "all recorded source descriptors match the main work tree byte-for-byte",
            "the exact repair lineage is self-contained in the sealed Git bundle",
        ],
        "attestation_generator_source": attestation_source,
        "scientific_outcome_direction_used_for_acceptance": False,
    }


def _execute(config_path: Path, *, workspace: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    workspace = workspace.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    output_root = workspace / "outputs"
    v1_root = _resolve(config_path, config["inputs"]["v1_root"])
    original_roots = {
        name: _resolve(config_path, payload["path"])
        for name, payload in config["inputs"]["formal_roots"].items()
    }
    final = _resolve(config_path, config["output"]["final_directory"])
    staging = _resolve(config_path, config["output"]["staging_directory"])
    lock_path = _resolve(config_path, config["output"]["lock_path"])
    expected_final = (
        output_root / "test_v4_aggregate_repair_v1_attestation_v1"
    ).resolve()
    expected_staging = (
        output_root / ".test_v4_aggregate_repair_v1_attestation_v1.incomplete"
    ).resolve()
    expected_lock = (
        output_root / ".test_v4_aggregate_repair_v1_attestation_v1.lock"
    ).resolve()
    if (final, staging, lock_path) != (
        expected_final,
        expected_staging,
        expected_lock,
    ):
        raise RuntimeError("attestation output paths differ from the frozen namespace")
    if v1_root != (output_root / "test_v4_aggregate_repair_v1").resolve():
        raise RuntimeError("attestation v1 input path differs from the frozen namespace")
    if _resolve(config_path, config["v1"]["root"]) != v1_root:
        raise RuntimeError("attestation has inconsistent v1 root declarations")
    expected_root_paths = {
        "aggregate_failure": output_root / ".test_v4_aggregate.incomplete",
        "seed17": output_root / ".test_v4_seed17.incomplete",
        "seed29": output_root / ".test_v4_seed29.incomplete",
        "seed43": output_root / ".test_v4_seed43.incomplete",
    }
    if original_roots != {
        key: value.resolve() for key, value in expected_root_paths.items()
    }:
        raise RuntimeError("attestation formal input paths differ from the frozen roots")

    source_manifest = _attestation_source_manifest(workspace)
    config_sha256 = _sha256_file(config_path)
    lock = _ExclusiveLock(lock_path)
    lock.acquire()
    staging_created = False
    try:
        if final.exists() or final.is_symlink():
            raise RuntimeError("completed attestation already exists; rerun is forbidden")
        if staging.exists() or staging.is_symlink():
            raise RuntimeError("prior attestation staging exists; overwrite is forbidden")
        staging.mkdir(parents=False)
        staging_created = True

        v1_before = _tree_snapshot(v1_root)
        _assert_tree_anchor(v1_before, config["v1"]["tree"], context="repair v1")
        formal_before = {
            name: _tree_snapshot(path) for name, path in original_roots.items()
        }
        for name, snapshot in formal_before.items():
            _assert_tree_anchor(
                snapshot,
                config["inputs"]["formal_roots"][name],
                context=name,
            )
        protected_before = _protected_tree_snapshot(
            workspace,
            output_root,
            [str(value) for value in config["inputs"]["protected_patterns"]],
        )
        expected_protected = config["inputs"]["protected_tree"]
        if protected_before != expected_protected:
            raise RuntimeError("protected evidence differs from the v1 sealed snapshot")

        v1_reaudit, v1_preregistration = _v1_integrity_reaudit(
            v1_root=v1_root,
            original_snapshots=formal_before,
            protected_current=protected_before,
            config=config,
        )
        bundle_path = staging / "repair_v1_source.bundle"
        source_verification = _source_commit_verification(
            workspace=workspace,
            v1_preregistration=v1_preregistration,
            source_config=config["source_commit"],
            bundle_path=bundle_path,
        )
        execution_provenance = _execution_provenance_payload(
            config, source_manifest
        )
        seed_roots = {
            seed: original_roots[f"seed{seed}"] for seed in SEEDS
        }
        independent = _independent_recomputation(
            v1_root=v1_root,
            seed_roots=seed_roots,
            alpha=float(config["statistics"]["familywise_alpha"]),
        )

        outputs = {
            "execution_provenance_attestation.json": execution_provenance,
            "source_commit_verification.json": source_verification,
            "v1_integrity_reaudit.json": v1_reaudit,
            "independent_recomputation.json": independent,
        }
        for name, payload in outputs.items():
            _write_json_new(staging / name, payload)

        def revalidate() -> dict[str, Any]:
            current_v1 = _tree_snapshot(v1_root)
            current_formal = {
                name: _tree_snapshot(path) for name, path in original_roots.items()
            }
            current_protected = _protected_tree_snapshot(
                workspace,
                output_root,
                [str(value) for value in config["inputs"]["protected_patterns"]],
            )
            current_source = _attestation_source_manifest(workspace)
            if current_v1 != v1_before:
                raise RuntimeError("repair v1 changed during attestation")
            if current_formal != formal_before:
                raise RuntimeError("formal input trees changed during attestation")
            if current_protected != protected_before:
                raise RuntimeError("protected evidence changed during attestation")
            if current_source != source_manifest:
                raise RuntimeError("attestation source changed during execution")
            if _sha256_file(config_path) != config_sha256:
                raise RuntimeError("attestation config changed during execution")
            if (
                str(
                    _git(
                        workspace,
                        "rev-parse",
                        str(config["source_commit"]["permanent_ref"]),
                    )
                ).strip()
                != str(config["source_commit"]["commit"])
            ):
                raise RuntimeError("permanent repair source ref moved during attestation")
            v1_source_manifest = v1_preregistration["repair_source_manifest"]
            for relative in EXPECTED_V1_SOURCE_SCOPE:
                expected = v1_source_manifest["files"][relative]
                if _descriptor(workspace / relative) != expected:
                    raise RuntimeError(
                        f"main work-tree v1 source changed during attestation: {relative}"
                    )
                blob = _git(
                    workspace,
                    "show",
                    f"{config['source_commit']['commit']}:{relative}",
                    binary=True,
                )
                if not isinstance(blob, bytes) or {
                    "sha256": sha256(blob).hexdigest(),
                    "size": len(blob),
                } != expected:
                    raise RuntimeError(
                        f"permanent v1 source changed during attestation: {relative}"
                    )
            payload = {
                "v1_tree": _tree_anchor(current_v1),
                "formal_trees": {
                    name: _tree_anchor(snapshot)
                    for name, snapshot in current_formal.items()
                },
                "protected_tree": current_protected,
                "attestation_source_digest": current_source["digest"],
                "config_sha256": config_sha256,
                "permanent_source_commit": config["source_commit"]["commit"],
            }
            payload["digest"] = _json_digest(payload)
            return payload

        first_final_recheck = revalidate()
        chain = {
            name: _sha256_file(staging / name)
            for name in (
                "execution_provenance_attestation.json",
                "source_commit_verification.json",
                "v1_integrity_reaudit.json",
                "independent_recomputation.json",
                "repair_v1_source.bundle",
            )
        }
        final_manifest: dict[str, Any] = {
            "protocol": f"{PROTOCOL}_final_manifest",
            "completed_utc": _utc(),
            "retrospective_attestation": True,
            "attestation_is_part_of_v1_tree": False,
            "v1_tree_modified": False,
            "original_outputs_modified": False,
            "scope_clean_not_global_clean": True,
            "shadow_git_invocation_disclosed": True,
            "source_commit_permanently_recoverable_from_bundle": True,
            "query_generation_count": 0,
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
            "bootstrap_resample_count": 0,
            "scientific_gate_direction_used_for_acceptance": False,
            "independent_recomputation_semantically_identical": True,
            "composite_v1_plus_attestation_integrity_pass": True,
            "hash_chain": chain,
            "hash_chain_digest": _json_digest(chain),
            "first_final_recheck_digest": first_final_recheck["digest"],
            "second_pre_promotion_recheck_required": True,
            "automatic_rerun_allowed": False,
        }
        final_manifest["manifest_payload_digest"] = _json_digest(final_manifest)
        final_manifest_path = staging / "attestation_final_manifest.json"
        _write_json_new(final_manifest_path, final_manifest)
        final_manifest_sha256 = _sha256_file(final_manifest_path)
        second_final_recheck = revalidate()
        if second_final_recheck != first_final_recheck:
            raise RuntimeError("attested inputs changed immediately before promotion")
        for name, expected_hash in chain.items():
            if _sha256_file(staging / name) != expected_hash:
                raise RuntimeError(
                    f"attestation artifact changed after hash-chain creation: {name}"
                )
        if _sha256_file(final_manifest_path) != final_manifest_sha256:
            raise RuntimeError("attestation final manifest changed before promotion")
        expected_files = set(chain) | {final_manifest_path.name}
        staging_entries = list(staging.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in staging_entries):
            raise RuntimeError(
                "attestation staging contains a symlink or non-regular entry"
            )
        observed_files = {path.name for path in staging_entries}
        if observed_files != expected_files:
            raise RuntimeError("attestation staging file set is not the frozen set")
        _fsync_directory(staging)
        staging.rename(final)
        staging_created = False
        _fsync_directory(output_root)
        return {**final_manifest, "final_manifest_sha256": final_manifest_sha256}
    except BaseException as error:
        if staging_created:
            failure_path = staging / "attestation_failure_manifest.json"
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
                        "repair_v1_modification_performed_by_attestation": False,
                        "original_outputs_modification_performed_by_attestation": False,
                    },
                )
        raise
    finally:
        lock.release()


def run(config_path: str | Path) -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[3]
    canonical = (
        workspace
        / "configs"
        / "test_v4_aggregate_repair_v1_attestation_v1.json"
    ).resolve()
    path = Path(config_path).resolve()
    if path != canonical:
        raise RuntimeError("attestation requires its canonical frozen config")
    return _execute(path, workspace=workspace)


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
