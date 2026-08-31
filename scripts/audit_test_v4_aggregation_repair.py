#!/usr/bin/env python3
"""Independent, read-only audit for the locked test_v4 aggregation repair.

This program intentionally uses only the Python standard library.  It never
imports ``confik``, NumPy, Torch, SciPy, Pinocchio, or a repair implementation.
It does not create a report file: the only report is emitted to stdout.

The production contract was captured after the six formal combinations had
completed and before any aggregation repair was attempted.  A successful
post-repair audit proves, independently of the repair implementation, that:

* all six pinned combination seals and every artifact named by them remain
  byte-identical;
* exactly 744 checkpoint manifests contain exactly 650,000 complete
  method-query records;
* the original failed-aggregation history remains byte-identical;
* the stored (not re-bootstrapped) eight p-values yield the reported joint
  Holm result regardless of JSON mapping key order;
* independently recomputed robot/Holm/paper decisions equal the repair output
  without treating any particular pass/fail direction as an audit criterion;
* the repair declares zero query generation, solver calls, model inference,
  checkpoint rewriting, and bootstrap resampling; and
* the seven-file authoritative repair namespace is byte-identical to its
  frozen hash chain and leaves all four original incomplete input trees in
  place.

The actual v1 schema under ``outputs/test_v4_aggregate_repair_v1`` is audited:
preregistration, input manifest, integrity record, aggregate summary, joint
Holm, paper gate, and final hash-chain manifest.  These manifests are evidence,
not the sole basis of the audit: raw checkpoint cross-products, Holm, gates,
and all file hashes are independently recomputed here.  A separate execution
attestation is audited by a later layer because the v1 files did not themselves
record the shadow Git environment used for execution.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FORBIDDEN_RUNTIME_MODULES = {
    "confik",
    "numpy",
    "pinocchio",
    "scipy",
    "sklearn",
    "torch",
}

CONFIRMATORY_METRICS = (
    "feasible_success_gap",
    "feasible_p95_latency_ratio",
    "feasible_p99_latency_ratio",
    "trajectory_completion_gap",
)

PRIMARY_METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "threshold_guard_cascade",
    "learned_1x25",
    "dls_previous_1x50",
    "trf_previous",
    "proposed_v4",
)

SENSITIVITY_METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "proposed_v4",
)


class AuditError(RuntimeError):
    """A blocking discrepancy in the formal evidence."""


@dataclass(frozen=True)
class AuditExpectations:
    """Pinned facts known before an aggregation-only repair."""

    robots: tuple[str, ...]
    seeds: tuple[int, ...]
    primary_seed: int
    roles: tuple[str, ...]
    role_query_counts: Mapping[str, int]
    checkpoint_count_per_combination: int
    total_checkpoint_count: int
    total_record_count: int
    combination_record_counts: Mapping[str, int]
    completion_sha256: Mapping[str, str]
    preregistration_sha256: str
    dataset_manifest_sha256: str
    control_plane_seal_sha256: str
    evidence_fingerprint_digest: str
    protected_tree_digest: str
    runner_git_commit: str
    failed_tree_digest: str
    failed_tree_file_count: int
    failure_manifest_count: int
    resume_event_count: int
    robot_gate_expectation: Mapping[str, bool]
    paper_gate_expectation: bool
    repair_namespace: str
    repair_output_sha256: Mapping[str, str]
    repair_git_commit: str
    repair_git_tree: str
    repair_git_parent: str
    repair_git_ref: str
    attestation_namespace: str
    attestation_final_manifest_sha256: str

    def methods_for_seed(self, seed: int) -> tuple[str, ...]:
        return PRIMARY_METHODS if seed == self.primary_seed else SENSITIVITY_METHODS


PRODUCTION = AuditExpectations(
    robots=("panda", "ur5e"),
    seeds=(17, 29, 43),
    primary_seed=17,
    roles=("id_points", "id_trajectories", "ood_points", "ood_trajectories"),
    role_query_counts={
        "id_points": 12_000,
        "id_trajectories": 6_000,
        "ood_points": 4_000,
        "ood_trajectories": 3_000,
    },
    checkpoint_count_per_combination=124,
    total_checkpoint_count=744,
    total_record_count=650_000,
    combination_record_counts={
        "panda/seed17": 175_000,
        "ur5e/seed17": 175_000,
        "panda/seed29": 75_000,
        "ur5e/seed29": 75_000,
        "panda/seed43": 75_000,
        "ur5e/seed43": 75_000,
    },
    completion_sha256={
        "panda/seed17": "6657fce520ad49285bfe113892e78345abf684248ef313216ee76771bb3534be",
        "ur5e/seed17": "88480f1e41bed6fec071ad04bae7dfdb19f3d16c39827bbc8d2a490cd768fba8",
        "panda/seed29": "fdffb3def50c29be3c8b04dc146704ab523846ea6c319928c5e745c98c90fcca",
        "ur5e/seed29": "d4f254efe473166886df86028fcbfb8840886289d17ff0815474dfd4c3392ef0",
        "panda/seed43": "e1a1a7339ee73cc62bd76d8481c25083c97e19d69e3b8200640367b293ec5860",
        "ur5e/seed43": "cc7d3cb8d5aff55beda0f148c9c39a782b77c90edd10f4f0d14489f998beec72",
    },
    preregistration_sha256="7808206ac9b76684f724523123bc461b7eedca4f4b759462eb142100385ad56d",
    dataset_manifest_sha256="77fa670e5de1bf40301f0bcd0292ebcc7860f412ac8bd946a485dc782458d51e",
    control_plane_seal_sha256="e97d13c190842451a20184a5279f3b0fceaadf1c501bcb2d5930817bd4d2b26a",
    evidence_fingerprint_digest="e0dda714ff8d6cb28fcaf8c8c9bf1ed2db0fffa66b476b0d1785f3a30cbabc38",
    protected_tree_digest="e859207753672aeace4e5115cd0ca65c6e07ab7ab00056538508a7b3e81367aa",
    runner_git_commit="e22fe9116ec40a15dc116dadc6d763e149fac72b",
    failed_tree_digest="bd294001b46cd6ebf96f55163030d329d4df582c2fdbf45818465f052aeb6bcd",
    failed_tree_file_count=44,
    failure_manifest_count=42,
    resume_event_count=41,
    robot_gate_expectation={"panda": False, "ur5e": True},
    paper_gate_expectation=False,
    repair_namespace="outputs/test_v4_aggregate_repair_v1",
    repair_output_sha256={
        "aggregate_summary_v4.json": "58b13579893c7ac2020a77357c954c521a6043c8b7864eac8ed0198ccec11851",
        "aggregation_repair_input_manifest.json": "8f6179a3ec294b3cb8f53b96f364e04032ad20ed96e3dcc9efcea991539e8fa9",
        "aggregation_repair_integrity.json": "cb91d8b324393e5626c54089002c852d2e8e7ac6f27ea0bc0393d667f256a997",
        "aggregation_repair_preregistration.json": "2a57ac5d5a115ea49107d60e5fadfcddd3708a7286758ac6c177b5a90b997b0f",
        "joint_holm_v4.json": "a6dc73452d57ca35b7bc7a2f83d4197efa32657dd8446dbd44c2e3a78955eef0",
        "paper_gate_v4.json": "4f9d307d499076daa5242859c3bffdd08805b78e7ab27aa08bef013b90e6a6ab",
        "test_v4_repair_final_manifest.json": "4f3b5024bdb6aa4ec283be4b4a0a8d3438e7a3ba03467154567a418817ba7ccc",
    },
    repair_git_commit="63e2ed6cbd14bbce0db869a247a9fb84e1f6911f",
    repair_git_tree="254024def41e34f9ac0d83aacd7feca4947c8a82",
    repair_git_parent="e22fe9116ec40a15dc116dadc6d763e149fac72b",
    repair_git_ref="refs/heads/codex/v4-aggregation-repair-exec",
    attestation_namespace="outputs/test_v4_aggregate_repair_v1_attestation_v1",
    # Filled with the independently observed final-manifest hash immediately
    # after the one-shot attestation namespace is atomically created.
    attestation_final_manifest_sha256="c1dddbdc2b2d1b89b9a735fe00bcb4804d6cd28bcc3c7925b36705389062034f",
)


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    _fail(path.is_file() and not path.is_symlink(), f"missing/non-regular JSON: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON {path}: {error}") from error


def file_descriptor(path: Path) -> dict[str, Any]:
    _fail(path.is_file() and not path.is_symlink(), f"missing/non-regular file: {path}")
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def _git(
    workspace: Path, arguments: Sequence[str], *, binary: bool = False
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            stderr = error.stderr
            detail = (
                stderr.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes)
                else str(stderr)
            )
        raise AuditError(
            f"git provenance check failed: {' '.join(arguments)}: {detail}"
        ) from error
    return result.stdout


def git_file_descriptor(workspace: Path, commit: str, relative: str) -> dict[str, Any]:
    payload = _git(workspace, ["show", f"{commit}:{relative}"], binary=True)
    assert isinstance(payload, bytes)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def snapshot(paths: Iterable[Path], root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): file_descriptor(path) for path in sorted(paths)
    }


def _assert_descriptor(path: Path, descriptor: Mapping[str, Any], context: str) -> None:
    _fail(set(descriptor) == {"sha256", "size"}, f"bad artifact descriptor: {context}")
    actual = file_descriptor(path)
    _fail(actual == dict(descriptor), f"artifact hash/size changed: {context}")


def _combination_key(robot: str, seed: int) -> str:
    return f"{robot}/seed{seed}"


def resolve_combination_roots(
    workspace: Path, expectations: AuditExpectations
) -> dict[tuple[str, int], Path]:
    """Resolve promoted or still-incomplete roots without accepting ambiguity."""

    resolved: dict[tuple[str, int], Path] = {}
    for seed in expectations.seeds:
        for robot in expectations.robots:
            final = workspace / "outputs" / f"test_v4_seed{seed}" / robot
            incomplete = (
                workspace / "outputs" / f".test_v4_seed{seed}.incomplete" / robot
            )
            candidates = [path for path in (final, incomplete) if path.is_dir()]
            _fail(candidates, f"missing combination root: {robot}/seed{seed}")
            if len(candidates) == 2:
                # A copy-based repair is not automatically trusted.  Both copies
                # must carry the independently pinned completion marker.
                expected = expectations.completion_sha256[_combination_key(robot, seed)]
                for path in candidates:
                    marker = path / "combination_complete.json"
                    _fail(
                        marker.is_file() and sha256_file(marker) == expected,
                        f"ambiguous combination copies differ: {robot}/seed{seed}",
                    )
                resolved[(robot, seed)] = final
            else:
                resolved[(robot, seed)] = candidates[0]
    return resolved


def audit_control_plane(
    workspace: Path,
    failed_aggregate: Path,
    expectations: AuditExpectations,
    *,
    verify_fingerprint_files: bool = True,
) -> dict[str, Any]:
    prereg_path = failed_aggregate / "test_v4_preregistration.json"
    dataset_path = failed_aggregate / "test_v4_dataset_manifest.json"
    seal_path = failed_aggregate / "test_v4_control_plane_seal.json"
    _fail(
        sha256_file(prereg_path) == expectations.preregistration_sha256,
        "preregistration changed",
    )
    _fail(
        sha256_file(dataset_path) == expectations.dataset_manifest_sha256,
        "dataset manifest changed",
    )
    _fail(
        sha256_file(seal_path) == expectations.control_plane_seal_sha256,
        "control-plane seal changed",
    )

    prereg = load_json(prereg_path)
    dataset = load_json(dataset_path)
    seal = load_json(seal_path)
    _fail(prereg.get("protocol") == "test_v4_locked", "wrong preregistration protocol")
    _fail(
        prereg.get("runner_git_commit") == expectations.runner_git_commit,
        "measurement commit changed",
    )
    _fail(
        prereg.get("evidence_fingerprint", {}).get("digest")
        == expectations.evidence_fingerprint_digest,
        "evidence fingerprint anchor changed",
    )
    _fail(
        prereg.get("protected_outputs_before", {}).get("tree_digest")
        == expectations.protected_tree_digest,
        "protected evidence anchor changed",
    )
    _fail(
        seal
        == {
            "protocol": "test_v4_frozen_control_plane_seal",
            "preregistration_sha256": expectations.preregistration_sha256,
            "dataset_manifest_sha256": expectations.dataset_manifest_sha256,
            "evidence_fingerprint_digest": expectations.evidence_fingerprint_digest,
        },
        "control-plane seal payload changed",
    )
    _fail(
        dataset.get("preregistration_sha256") == expectations.preregistration_sha256,
        "dataset manifest lost preregistration binding",
    )

    dataset_artifacts = 0
    for robot in expectations.robots:
        roles = dataset.get("robots", {}).get(robot, {}).get("roles", {})
        _fail(set(roles) == set(expectations.roles), f"dataset roles changed: {robot}")
        for role in expectations.roles:
            descriptor = roles[role]
            path = failed_aggregate / str(descriptor["path"])
            _assert_descriptor(
                path,
                {"sha256": descriptor["sha256"], "size": descriptor["size"]},
                f"dataset/{robot}/{role}",
            )
            _fail(
                int(descriptor["query_count"]) == expectations.role_query_counts[role],
                f"dataset query count changed: {robot}/{role}",
            )
            dataset_artifacts += 1

    fingerprint_files = prereg.get("evidence_fingerprint", {}).get("files", {})
    if verify_fingerprint_files:
        for raw_path, descriptor in fingerprint_files.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = workspace / path
            if path.is_file() and file_descriptor(path) == descriptor:
                continue
            # reporting.py is the sole preregistered, aggregation-only source
            # change.  Its original formal bytes remain content-addressed by
            # the e22 measurement commit even though the physical worktree now
            # contains the repaired implementation.
            _fail(
                raw_path == "src/confik/test_v4_locked/reporting.py",
                f"frozen source/asset changed outside repair scope: {raw_path}",
            )
            _fail(
                git_file_descriptor(workspace, expectations.runner_git_commit, raw_path)
                == descriptor,
                "original reporting.py is not recoverable from measurement commit",
            )

    return {
        "preregistration_sha256": expectations.preregistration_sha256,
        "dataset_manifest_sha256": expectations.dataset_manifest_sha256,
        "control_plane_seal_sha256": expectations.control_plane_seal_sha256,
        "dataset_artifact_count": dataset_artifacts,
        "fingerprint_file_count": len(fingerprint_files),
        "fingerprint_files_rehashed": bool(verify_fingerprint_files),
    }


def audit_failed_tree(
    failed_aggregate: Path, expectations: AuditExpectations
) -> dict[str, Any]:
    manifests = sorted((failed_aggregate / "failure_manifests").glob("*.json"))
    paths = manifests + [
        failed_aggregate / "latest_failure_manifest.json",
        failed_aggregate / "resume_history.json",
    ]
    _fail(
        len(manifests) == expectations.failure_manifest_count,
        "failure-manifest count changed",
    )
    tree = snapshot(paths, failed_aggregate)
    _fail(
        len(tree) == expectations.failed_tree_file_count,
        "failed-tree file count changed",
    )
    digest = json_digest(tree)
    _fail(digest == expectations.failed_tree_digest, "original failed tree changed")

    classifications: Counter[str] = Counter()
    resumable: Counter[bool] = Counter()
    for path in manifests:
        payload = load_json(path)
        classifications[str(payload.get("failure_classification"))] += 1
        resumable[bool(payload.get("resume_eligible", False))] += 1
    _fail(
        classifications
        == {
            "resumable_external_environment_technical_interruption": 41,
            "non_resumable_integrity_or_scientific_failure": 1,
        },
        "failure classifications changed",
    )
    _fail(resumable == {True: 41, False: 1}, "failure eligibility changed")
    latest = load_json(failed_aggregate / "latest_failure_manifest.json")
    _fail(
        latest.get("exception_message") == "panda confirmatory metrics changed",
        "wrong terminal failure",
    )
    _fail(latest.get("resume_eligible") is False, "terminal failure became resumable")
    history = load_json(failed_aggregate / "resume_history.json")
    _fail(
        len(history.get("events", ())) == expectations.resume_event_count,
        "resume history changed",
    )
    return {
        "tree_digest": digest,
        "tree_file_count": len(tree),
        "failure_manifest_count": len(manifests),
        "resume_event_count": len(history["events"]),
        "terminal_failure": latest["exception_message"],
    }


def _validate_checkpoint_records(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    robot: str,
    seed: int,
    methods: Sequence[str],
) -> tuple[int, set[tuple[str, int, str]], set[tuple[str, int]]]:
    checkpoint_dir = manifest_path.parent
    record_path = checkpoint_dir / "records.jsonl.gz"
    artifacts = manifest.get("artifacts", {})
    _fail(
        set(artifacts) == {"records.jsonl.gz"},
        f"checkpoint artifacts changed: {manifest_path}",
    )
    _assert_descriptor(record_path, artifacts["records.jsonl.gz"], str(record_path))

    role = str(manifest["role"])
    indices = [int(value) for value in manifest.get("source_indices", ())]
    hashes = [str(value) for value in manifest.get("source_query_sha256", ())]
    expected_queries = int(manifest["expected_query_count"])
    expected_records = int(manifest["expected_record_count"])
    _fail(
        len(indices) == expected_queries,
        f"checkpoint query count mismatch: {manifest_path}",
    )
    _fail(
        len(hashes) == expected_queries,
        f"checkpoint query hash count mismatch: {manifest_path}",
    )
    _fail(len(set(indices)) == len(indices), f"duplicate source index: {manifest_path}")
    _fail(len(set(hashes)) == len(hashes), f"duplicate source hash: {manifest_path}")
    _fail(
        all(
            len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in hashes
        ),
        f"invalid source hash: {manifest_path}",
    )
    _fail(
        expected_records == expected_queries * len(methods),
        f"bad checkpoint cross-product size: {manifest_path}",
    )

    source_hash_by_index = dict(zip(indices, hashes, strict=True))
    observed: set[tuple[int, str]] = set()
    record_count = 0
    try:
        with gzip.open(record_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line, parse_constant=_reject_json_constant)
                except (ValueError, json.JSONDecodeError) as error:
                    raise AuditError(
                        f"bad record JSON {record_path}:{line_number}"
                    ) from error
                index = int(row.get("query_index"))
                method = str(row.get("method"))
                pair = (index, method)
                _fail(
                    pair not in observed,
                    f"duplicate method-query pair: {record_path}:{pair}",
                )
                observed.add(pair)
                _fail(
                    index in source_hash_by_index,
                    f"unexpected query index: {record_path}:{index}",
                )
                _fail(method in methods, f"unexpected method: {record_path}:{method}")
                _fail(row.get("robot") == robot, f"record robot changed: {record_path}")
                _fail(
                    int(row.get("training_seed")) == seed,
                    f"record seed changed: {record_path}",
                )
                _fail(row.get("role") == role, f"record role changed: {record_path}")
                _fail(
                    row.get("source_query_sha256") == source_hash_by_index[index],
                    f"record source identity changed: {record_path}:{index}",
                )
                record_count += 1
    except (OSError, EOFError) as error:
        raise AuditError(
            f"unreadable gzip checkpoint {record_path}: {error}"
        ) from error

    expected_pairs = {(index, method) for index in indices for method in methods}
    _fail(
        observed == expected_pairs,
        f"incomplete checkpoint cross-product: {record_path}",
    )
    _fail(
        record_count == expected_records,
        f"checkpoint record count changed: {record_path}",
    )
    identities = {(role, index, source_hash_by_index[index]) for index in indices}
    role_indices = {(role, index) for index in indices}
    return record_count, identities, role_indices


def audit_combination(
    root: Path,
    *,
    robot: str,
    seed: int,
    expectations: AuditExpectations,
) -> dict[str, Any]:
    key = _combination_key(robot, seed)
    marker_path = root / "combination_complete.json"
    _fail(
        sha256_file(marker_path) == expectations.completion_sha256[key],
        f"combination completion marker changed: {key}",
    )
    completion = load_json(marker_path)
    methods = expectations.methods_for_seed(seed)
    _fail(completion.get("robot") == robot, f"completion robot changed: {key}")
    _fail(
        int(completion.get("training_seed")) == seed, f"completion seed changed: {key}"
    )
    _fail(
        tuple(completion.get("methods", ())) == methods,
        f"completion methods changed: {key}",
    )
    _fail(
        completion.get("all_checkpoints_hash_validated") is True,
        f"unsealed checkpoints: {key}",
    )
    _fail(
        completion.get("preregistration_sha256") == expectations.preregistration_sha256
        and completion.get("dataset_manifest_sha256")
        == expectations.dataset_manifest_sha256
        and completion.get("evidence_fingerprint_digest")
        == expectations.evidence_fingerprint_digest,
        f"completion anchors changed: {key}",
    )

    artifacts = completion.get("artifacts", {})
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "combination_complete.json"
    }
    _fail(set(artifacts) == actual_files, f"completion artifact set changed: {key}")
    for relative, descriptor in artifacts.items():
        _assert_descriptor(root / relative, descriptor, f"{key}/{relative}")

    manifests = sorted(root.glob("checkpoints/*/*/checkpoint_manifest.json"))
    _fail(
        len(manifests) == expectations.checkpoint_count_per_combination,
        f"checkpoint count changed: {key}",
    )
    total_records = 0
    identities: set[tuple[str, int, str]] = set()
    role_indices: set[tuple[str, int]] = set()
    role_counts: Counter[str] = Counter()
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        _fail(
            manifest.get("protocol") == "test_v4_atomic_measurement_checkpoint",
            f"wrong checkpoint protocol: {manifest_path}",
        )
        _fail(
            manifest.get("robot") == robot, f"checkpoint robot changed: {manifest_path}"
        )
        _fail(
            int(manifest.get("training_seed")) == seed,
            f"checkpoint seed changed: {manifest_path}",
        )
        _fail(
            tuple(manifest.get("methods", ())) == methods,
            f"checkpoint methods changed: {manifest_path}",
        )
        _fail(
            manifest.get("preregistration_sha256")
            == expectations.preregistration_sha256
            and manifest.get("dataset_manifest_sha256")
            == expectations.dataset_manifest_sha256
            and manifest.get("evidence_fingerprint_digest")
            == expectations.evidence_fingerprint_digest,
            f"checkpoint anchors changed: {manifest_path}",
        )
        count, checkpoint_identities, checkpoint_indices = _validate_checkpoint_records(
            manifest_path,
            manifest,
            robot=robot,
            seed=seed,
            methods=methods,
        )
        _fail(
            not (identities & checkpoint_identities),
            f"query identity repeated across checkpoints: {manifest_path}",
        )
        _fail(
            not (role_indices & checkpoint_indices),
            f"query index repeated across checkpoints: {manifest_path}",
        )
        identities.update(checkpoint_identities)
        role_indices.update(checkpoint_indices)
        role_counts[str(manifest["role"])] += int(manifest["expected_query_count"])
        total_records += count

    _fail(
        dict(role_counts) == dict(expectations.role_query_counts),
        f"role query counts changed: {key}",
    )
    _fail(
        total_records == expectations.combination_record_counts[key],
        f"combination record count changed: {key}",
    )
    return {
        "root": str(root),
        "completion_sha256": expectations.completion_sha256[key],
        "checkpoint_count": len(manifests),
        "query_count": len(identities),
        "record_count": total_records,
        "methods": list(methods),
    }


def audit_all_combinations(
    workspace: Path, expectations: AuditExpectations
) -> tuple[dict[str, Any], dict[tuple[str, int], Path]]:
    roots = resolve_combination_roots(workspace, expectations)
    combinations: dict[str, Any] = {}
    checkpoints = 0
    records = 0
    for seed in expectations.seeds:
        for robot in expectations.robots:
            key = _combination_key(robot, seed)
            payload = audit_combination(
                roots[(robot, seed)],
                robot=robot,
                seed=seed,
                expectations=expectations,
            )
            combinations[key] = payload
            checkpoints += int(payload["checkpoint_count"])
            records += int(payload["record_count"])
    _fail(
        checkpoints == expectations.total_checkpoint_count,
        "global checkpoint count changed",
    )
    _fail(records == expectations.total_record_count, "global record count changed")
    return {
        "combination_count": len(combinations),
        "checkpoint_count": checkpoints,
        "record_count": records,
        "combinations": combinations,
    }, roots


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Standard step-down Holm adjustment, independent of JSON key order."""

    ordered = sorted(pvalues.items(), key=lambda item: (float(item[1]), item[0]))
    result: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        _fail(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0,
            f"invalid p-value: {name}",
        )
        running = max(running, min(1.0, float(value) * (count - rank)))
        result[name] = running
    return result


def recompute_joint_holm(
    primary_roots: Mapping[str, Path], *, alpha: float = 0.05
) -> dict[str, Any]:
    pvalues: dict[str, float] = {}
    for robot in ("panda", "ur5e"):
        intervals = load_json(primary_roots[robot] / "paired_intervals_v4.json")
        family = intervals.get("inference_family", {})
        _fail(
            tuple(family.get("members", ())) == CONFIRMATORY_METRICS,
            f"inference family changed: {robot}",
        )
        metrics = intervals.get("metrics", {})
        # Mapping order is intentionally ignored; this is the bug under repair.
        _fail(
            set(metrics) == set(CONFIRMATORY_METRICS),
            f"metric membership changed: {robot}",
        )
        for metric in CONFIRMATORY_METRICS:
            pvalues[f"{robot}/{metric}"] = float(
                metrics[metric]["one_sided_unadjusted_p"]
            )
    adjusted = holm_adjust(pvalues)
    hypotheses = {
        name: {
            "robot": name.split("/", 1)[0],
            "metric": name.split("/", 1)[1],
            "one_sided_unadjusted_p": pvalues[name],
            "holm_adjusted_p": adjusted[name],
            "reject_margin_null": adjusted[name] <= alpha,
        }
        for name in pvalues
    }
    return {
        "method": "Holm",
        "scope": "Panda and UR5e x four prespecified confirmatory claims",
        "alpha": alpha,
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "all_confirmatory_nulls_rejected": all(
            item["reject_margin_null"] for item in hypotheses.values()
        ),
        "operational_finite_test_gates_included": False,
    }


def _float_equal(left: Any, right: Any, *, atol: float = 1e-15) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def audit_recomputed_statistics_and_gates(
    repair_aggregate: Path,
    roots: Mapping[tuple[str, int], Path],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    primary_roots = {
        robot: roots[(robot, expectations.primary_seed)]
        for robot in expectations.robots
    }
    expected_holm = recompute_joint_holm(primary_roots)
    reported_holm = load_json(repair_aggregate / "joint_holm_v4.json")
    _fail(
        set(reported_holm.get("hypotheses", {})) == set(expected_holm["hypotheses"]),
        "reported Holm family changed",
    )
    for name, expected in expected_holm["hypotheses"].items():
        actual = reported_holm["hypotheses"][name]
        _fail(
            actual.get("robot") == expected["robot"]
            and actual.get("metric") == expected["metric"],
            f"Holm identity changed: {name}",
        )
        _fail(
            _float_equal(
                actual.get("one_sided_unadjusted_p"), expected["one_sided_unadjusted_p"]
            ),
            f"Holm raw p changed: {name}",
        )
        _fail(
            _float_equal(actual.get("holm_adjusted_p"), expected["holm_adjusted_p"]),
            f"Holm adjusted p changed: {name}",
        )
        _fail(
            actual.get("reject_margin_null") is expected["reject_margin_null"],
            f"Holm decision changed: {name}",
        )
    for field in (
        "method",
        "scope",
        "hypothesis_count",
        "all_confirmatory_nulls_rejected",
        "operational_finite_test_gates_included",
    ):
        _fail(
            reported_holm.get(field) == expected_holm[field],
            f"Holm summary changed: {field}",
        )
    _fail(
        _float_equal(reported_holm.get("alpha"), expected_holm["alpha"]),
        "Holm alpha changed",
    )

    robot_gates: dict[str, bool] = {}
    failed_checks: dict[str, list[str]] = {}
    for robot in expectations.robots:
        gate = load_json(primary_roots[robot] / "claim_gate_v4.json")
        value = bool(gate.get("formal_gate_pass"))
        robot_gates[robot] = value
        failed_checks[robot] = sorted(
            name for name, passed in gate.get("checks", {}).items() if not bool(passed)
        )

    paper = load_json(repair_aggregate / "paper_gate_v4.json")
    expected_paper = all(robot_gates.values()) and bool(
        expected_holm["all_confirmatory_nulls_rejected"]
    )
    _fail(paper.get("robot_gates") == robot_gates, "reported robot gates changed")
    _fail(
        paper.get("joint_holm_gate_pass")
        is bool(expected_holm["all_confirmatory_nulls_rejected"]),
        "reported joint Holm gate changed",
    )
    _fail(
        paper.get("both_robot_gates_pass") is expected_paper,
        "reported paper gate changed",
    )
    _fail(
        paper.get("test_set_retuning_performed") is False,
        "repair reports test retuning",
    )

    aggregate_summary = load_json(repair_aggregate / "aggregate_summary_v4.json")
    for robot in expectations.robots:
        expected_primary = {
            "claim_gate": load_json(primary_roots[robot] / "claim_gate_v4.json"),
            "summary": load_json(primary_roots[robot] / "summary_v4.json"),
            "ood_abstention": load_json(
                primary_roots[robot] / "ood_abstention_v4.json"
            ),
        }
        _fail(
            aggregate_summary.get("primary", {}).get(robot) == expected_primary,
            f"primary aggregation is not a pure copy: {robot}",
        )
        for seed in expectations.seeds:
            if seed == expectations.primary_seed:
                continue
            expected_summary = load_json(roots[(robot, seed)] / "summary_v4.json")
            _fail(
                aggregate_summary.get("sensitivity", {})
                .get(robot, {})
                .get(f"seed{seed}")
                == expected_summary,
                f"sensitivity aggregation is not a pure copy: {robot}/seed{seed}",
            )
    return {
        "stored_unadjusted_pvalue_count": len(expected_holm["hypotheses"]),
        "all_joint_holm_claims_pass": expected_holm["all_confirmatory_nulls_rejected"],
        "robot_gates": robot_gates,
        "failed_checks": failed_checks,
        "paper_gate_pass": expected_paper,
    }


def _audit_input_tree(
    payload: Mapping[str, Any], *, expected_root: Path, label: str
) -> dict[str, Any]:
    root = Path(str(payload.get("root", "")))
    _fail(
        root.resolve() == expected_root.resolve(), f"repair input root changed: {label}"
    )
    _fail(
        root.is_dir() and not root.is_symlink(), f"repair input root missing: {label}"
    )
    items = payload.get("files", ())
    declared: dict[str, dict[str, Any]] = {}
    for item in items:
        relative = str(item.get("path"))
        _fail(
            relative not in declared, f"duplicate repair input path: {label}/{relative}"
        )
        declared[relative] = {
            "sha256": item.get("sha256"),
            "size": item.get("size"),
        }
    actual = {
        str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()
    }
    _fail(set(declared) == set(actual), f"repair input artifact set changed: {label}")
    for relative, path in actual.items():
        _assert_descriptor(path, declared[relative], f"repair-input/{label}/{relative}")
    _fail(
        int(payload.get("file_count", -1)) == len(declared),
        f"repair input file count changed: {label}",
    )
    _fail(
        int(payload.get("total_bytes", -1))
        == sum(int(item["size"]) for item in declared.values()),
        f"repair input byte count changed: {label}",
    )
    _fail(
        json_digest(items) == payload.get("tree_digest"),
        f"repair input tree digest changed: {label}",
    )
    return {
        "root": str(root),
        "file_count": len(declared),
        "total_bytes": int(payload["total_bytes"]),
        "tree_digest": payload["tree_digest"],
    }


def _audit_repair_source_lineage(
    workspace: Path,
    source: Mapping[str, Any],
    expectations: AuditExpectations,
    *,
    verify_git_lineage: bool,
) -> dict[str, Any]:
    _fail(
        source.get("git_commit") == expectations.repair_git_commit,
        "repair commit changed",
    )
    _fail(
        source.get("git_tree") == expectations.repair_git_tree,
        "repair git tree changed",
    )
    _fail(source.get("scope_clean") is True, "repair source scope was not sealed clean")
    without_digest = {key: value for key, value in source.items() if key != "digest"}
    _fail(
        json_digest(without_digest) == source.get("digest"),
        "repair source manifest digest changed",
    )
    files = source.get("files", {})
    expected_files = {
        "configs/test_v4_aggregate_repair_v1.yaml",
        "scripts/run_test_v4_aggregate_repair_v1.sh",
        "src/confik/test_v4_locked/aggregate_repair.py",
        "src/confik/test_v4_locked/reporting.py",
        "tests/test_test_v4_aggregate_repair.py",
        "tests/test_test_v4_reporting.py",
    }
    _fail(set(files) == expected_files, "repair source file scope changed")
    for relative, descriptor in files.items():
        _assert_descriptor(
            workspace / relative, descriptor, f"repair-source/{relative}"
        )
    if verify_git_lineage:
        resolved = str(
            _git(workspace, ["rev-parse", expectations.repair_git_ref])
        ).strip()
        tree = str(
            _git(workspace, ["rev-parse", f"{expectations.repair_git_commit}^{{tree}}"])
        ).strip()
        parent = str(
            _git(workspace, ["rev-parse", f"{expectations.repair_git_commit}^"])
        ).strip()
        _fail(resolved == expectations.repair_git_commit, "permanent repair ref moved")
        _fail(tree == expectations.repair_git_tree, "permanent repair tree changed")
        _fail(parent == expectations.repair_git_parent, "repair commit parent changed")
        for relative, descriptor in files.items():
            _fail(
                git_file_descriptor(workspace, expectations.repair_git_commit, relative)
                == descriptor,
                f"permanent repair commit bytes differ: {relative}",
            )
    return {
        "git_commit": expectations.repair_git_commit,
        "git_tree": expectations.repair_git_tree,
        "git_parent": expectations.repair_git_parent,
        "permanent_ref": expectations.repair_git_ref,
        "source_file_count": len(files),
        "disk_bytes_match_manifest": True,
        "permanent_git_lineage_verified": bool(verify_git_lineage),
    }


def audit_repair_manifest(
    workspace: Path,
    repair_aggregate: Path,
    failed_aggregate: Path,
    expectations: AuditExpectations,
    *,
    verify_git_lineage: bool = True,
) -> dict[str, Any]:
    prereg_path = repair_aggregate / "aggregation_repair_preregistration.json"
    input_path = repair_aggregate / "aggregation_repair_input_manifest.json"
    integrity_path = repair_aggregate / "aggregation_repair_integrity.json"
    prereg = load_json(prereg_path)
    input_manifest = load_json(input_path)
    integrity = load_json(integrity_path)
    for evidence_path in (prereg_path, input_path, integrity_path):
        _fail(
            sha256_file(evidence_path)
            == expectations.repair_output_sha256[evidence_path.name],
            f"sealed repair evidence bytes changed: {evidence_path.name}",
        )
    _fail(
        prereg.get("protocol") == "test_v4_aggregate_repair_v1",
        "wrong repair preregistration protocol",
    )
    contract = prereg.get("execution_contract", {})
    _fail(contract.get("aggregation_only") is True, "repair scope expanded")
    _fail(
        contract.get("automatic_resume_allowed") is False
        and contract.get("original_outputs_mutation_allowed") is False,
        "repair execution contract permits mutation/resume",
    )
    _fail(
        int(contract.get("query_rerun_count", -1)) == 0
        and int(contract.get("solver_invocation_count", -1)) == 0
        and int(contract.get("model_inference_count", -1)) == 0,
        "repair preregistration reports scientific execution",
    )
    _fail(
        prereg.get("threshold_or_gate_changes") is False
        and prereg.get("statistical_semantics_changed") is False
        and prereg.get("old_test_performance_used_for_selection") is False
        and prereg.get("original_failure_classification_changed") is False,
        "repair preregistration changes frozen semantics/evidence",
    )
    control = prereg.get("control_plane", {})
    _fail(
        control.get("preregistration_sha256") == expectations.preregistration_sha256
        and control.get("dataset_manifest_sha256")
        == expectations.dataset_manifest_sha256
        and control.get("control_plane_seal_sha256")
        == expectations.control_plane_seal_sha256
        and control.get("original_evidence_fingerprint_digest")
        == expectations.evidence_fingerprint_digest,
        "repair control-plane anchors changed",
    )
    expected_failure = prereg.get("expected_failure", {})
    _fail(
        expected_failure.get("exception_message")
        == "panda confirmatory metrics changed"
        and expected_failure.get("failure_classification")
        == "non_resumable_integrity_or_scientific_failure"
        and expected_failure.get("phase") == {"phase": "aggregate_and_final_integrity"},
        "repair target failure changed",
    )
    source_lineage = _audit_repair_source_lineage(
        workspace,
        prereg.get("repair_source_manifest", {}),
        expectations,
        verify_git_lineage=verify_git_lineage,
    )

    _fail(
        input_manifest.get("protocol") == "test_v4_aggregate_repair_v1_input_manifest",
        "wrong repair input-manifest protocol",
    )
    _fail(
        input_manifest.get("repair_preregistration_sha256") == sha256_file(prereg_path),
        "repair input manifest lost preregistration binding",
    )
    _fail(
        input_manifest.get("all_six_combinations_hash_validated") is True
        and input_manifest.get("query_record_files_hash_validated_only") is True
        and input_manifest.get("query_records_read_for_aggregation") is False,
        "repair input access exceeded aggregation contract",
    )
    root_payloads = input_manifest.get("roots", {})
    expected_roots = {
        "aggregate_failure": failed_aggregate,
        **{
            f"seed{seed}": workspace / "outputs" / f".test_v4_seed{seed}.incomplete"
            for seed in expectations.seeds
        },
    }
    _fail(set(root_payloads) == set(expected_roots), "repair input root set changed")
    input_trees = {
        label: _audit_input_tree(
            root_payloads[label], expected_root=expected_roots[label], label=label
        )
        for label in expected_roots
    }
    _fail(
        json_digest(root_payloads) == input_manifest.get("combined_tree_digest"),
        "combined repair input digest changed",
    )
    _fail(
        prereg.get("input_tree_digests")
        == {
            label: {
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "tree_digest": payload["tree_digest"],
            }
            for label, payload in input_trees.items()
        },
        "repair preregistration input digests changed",
    )

    validations = prereg.get("combination_validations", ())
    validation_map = {
        _combination_key(str(item["robot"]), int(item["training_seed"])): item
        for item in validations
    }
    _fail(
        set(validation_map) == set(expectations.completion_sha256),
        "repair combination validation set changed",
    )
    for key, expected_hash in expectations.completion_sha256.items():
        item = validation_map[key]
        _fail(
            item.get("completion_manifest_sha256") == expected_hash,
            f"repair completion hash changed: {key}",
        )
        _fail(
            int(item.get("checkpoint_count", -1))
            == expectations.checkpoint_count_per_combination,
            f"repair checkpoint count changed: {key}",
        )
        _fail(
            item.get("all_artifact_hashes_verified") is True
            and item.get("all_checkpoint_quiet_host_contracts_verified") is True,
            f"repair combination was not fully verified: {key}",
        )

    _fail(
        integrity.get("protocol") == "test_v4_aggregate_repair_v1_integrity",
        "wrong repair integrity protocol",
    )
    _fail(
        integrity.get("input_trees_before") == root_payloads
        and integrity.get("input_trees_after") == root_payloads
        and integrity.get("input_trees_unchanged") is True,
        "repair input trees changed during aggregation",
    )
    _fail(
        integrity.get("protected_tree_before") == integrity.get("protected_tree_after")
        and integrity.get("protected_tree_unchanged") is True
        and integrity.get("protected_tree_before", {}).get("tree_digest")
        == expectations.protected_tree_digest,
        "protected evidence changed during repair",
    )
    _fail(
        integrity.get("original_failure_evidence_preserved") is True
        and integrity.get("original_failure_classification_changed") is False,
        "original failure evidence/classification changed",
    )
    _fail(
        int(integrity.get("query_rerun_count", -1)) == 0
        and int(integrity.get("solver_invocation_count", -1)) == 0
        and int(integrity.get("model_inference_count", -1)) == 0,
        "repair integrity reports scientific execution",
    )
    _fail(
        integrity.get("all_six_combinations_hash_validated") is True,
        "repair integrity did not validate all combinations",
    )
    _fail(
        integrity.get("final_input_recheck_digest") is not None,
        "repair final input recheck is absent",
    )
    return {
        "preregistration_sha256": sha256_file(prereg_path),
        "input_manifest_sha256": sha256_file(input_path),
        "integrity_sha256": sha256_file(integrity_path),
        "input_trees": input_trees,
        "all_input_trees_unchanged": True,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "source_lineage": source_lineage,
        "final_input_recheck_digest": integrity["final_input_recheck_digest"],
    }


def audit_final_seal(
    workspace: Path,
    repair_aggregate: Path,
    expectations: AuditExpectations,
    *,
    expected_paper_gate: bool,
    expected_final_input_recheck_digest: str,
) -> dict[str, Any]:
    final_path = repair_aggregate / "test_v4_repair_final_manifest.json"
    final = load_json(final_path)
    _fail(
        final.get("protocol") == "test_v4_aggregate_repair_v1_final_manifest",
        "wrong final protocol",
    )
    _fail(
        final.get("authoritative_output_namespace") == expectations.repair_namespace,
        "repair namespace changed",
    )
    _fail(
        final.get("aggregation_only_repair") is True
        and final.get("six_combination_natural_exits") is True,
        "final manifest expanded beyond sealed aggregation",
    )
    _fail(
        final.get("original_formal_runner_natural_exit") is False
        and final.get("original_incomplete_paths_promoted_or_renamed") is False,
        "final manifest rewrites original formal-run status/paths",
    )
    _fail(
        final.get("original_failure_evidence_preserved") is True
        and final.get("original_failure_classification_changed") is False,
        "final manifest changes original failure evidence",
    )
    _fail(
        final.get("threshold_or_statistical_semantics_changed") is False,
        "final manifest changes frozen semantics",
    )
    _fail(
        int(final.get("query_rerun_count", -1)) == 0
        and int(final.get("solver_invocation_count", -1)) == 0
        and int(final.get("model_inference_count", -1)) == 0,
        "final manifest reports scientific execution",
    )
    _fail(
        final.get("paper_gate_pass") is expected_paper_gate,
        "final paper gate differs from independent recomputation",
    )
    _fail(
        final.get("final_input_recheck_digest") == expected_final_input_recheck_digest,
        "final input recheck lost integrity binding",
    )

    expected_files = set(expectations.repair_output_sha256)
    actual_files = {path.name for path in repair_aggregate.iterdir() if path.is_file()}
    _fail(
        actual_files == expected_files,
        "repair namespace does not contain exactly the seven sealed files",
    )
    for name, expected_hash in expectations.repair_output_sha256.items():
        path = repair_aggregate / name
        _fail(
            path.is_file() and not path.is_symlink(),
            f"sealed repair artifact missing/non-regular: {name}",
        )
        _fail(
            sha256_file(path) == expected_hash,
            f"sealed repair artifact bytes changed: {name}",
        )

    expected_chain = {
        name: digest
        for name, digest in expectations.repair_output_sha256.items()
        if name != final_path.name
    }
    _fail(final.get("hash_chain") == expected_chain, "repair final hash chain changed")
    _fail(
        final.get("hash_chain_digest") == json_digest(expected_chain),
        "repair hash-chain digest changed",
    )
    payload = {
        key: value for key, value in final.items() if key != "manifest_payload_digest"
    }
    _fail(
        final.get("manifest_payload_digest") == json_digest(payload),
        "repair final-manifest payload digest changed",
    )
    return {
        "final_manifest_sha256": sha256_file(final_path),
        "sealed_artifact_count": len(actual_files),
        "artifact_set_exact": True,
        "all_hashes_valid": True,
        "paper_gate_pass": final["paper_gate_pass"],
        "authoritative_output_namespace": final["authoritative_output_namespace"],
    }


def _descriptor_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        set(left) == {"sha256", "size"}
        and set(right) == {"sha256", "size"}
        and str(left["sha256"]) == str(right["sha256"])
        and int(left["size"]) == int(right["size"])
    )


def _audit_execution_provenance_attestation(
    workspace: Path,
    attestation_root: Path,
    expectations: AuditExpectations,
) -> dict[str, Any]:
    path = attestation_root / "execution_provenance_attestation.json"
    payload = load_json(path)
    _fail(
        str(payload.get("protocol", "")).startswith("test_v4_aggregate_repair_v1"),
        "wrong execution-attestation protocol",
    )
    _fail(
        payload.get("attestation_timing") == "retrospective",
        "attestation timing disclosure changed",
    )
    _fail(
        payload.get("original_execution_invocation_source")
        == "operator_and_tool_invocation_record"
        and payload.get("original_invocation_independently_traced") is False,
        "attestation overstates independent tracing",
    )
    invocation = payload.get("declared_original_invocation", {})
    _fail(
        invocation.get("git_dir") == "/tmp/confik-v4-repair-lineage.ZJRjoy/.git"
        and Path(str(invocation.get("git_work_tree", ""))).resolve()
        == workspace.resolve()
        and invocation.get("source_commit") == expectations.repair_git_commit,
        "shadow Git execution disclosure changed",
    )
    _fail(
        isinstance(invocation.get("launcher"), str)
        and bool(invocation.get("launcher"))
        and isinstance(invocation.get("statement"), str)
        and bool(invocation.get("statement")),
        "original invocation declaration is incomplete",
    )
    _fail(
        payload.get("shadow_git_metadata_used") is True
        and payload.get("git_work_tree_was_main_workspace") is True
        and payload.get("global_worktree_cleanliness_asserted") is False
        and payload.get("global_worktree_cleanliness_required_for_this_attestation")
        is False,
        "shadow/global-cleanliness limitation was not disclosed",
    )
    _fail(
        isinstance(payload.get("scope_clean_interpretation"), str)
        and bool(payload.get("scope_clean_interpretation")),
        "scope-clean interpretation is absent",
    )
    consequences = payload.get("verifiable_consequences_not_invocation_claim_alone")
    _fail(
        isinstance(consequences, list)
        and len(consequences) >= 3
        and all(isinstance(item, str) and item for item in consequences),
        "attestation lacks independently verifiable consequences",
    )
    _fail(
        payload.get("scientific_outcome_direction_used_for_acceptance") is False,
        "attestation used observed scientific direction for acceptance",
    )

    source = payload.get("attestation_generator_source", {})
    _fail(source.get("scope_clean") is True, "attestation generator scope is not clean")
    _fail(
        source.get("global_cleanliness_asserted") is False,
        "attestation generator overstates global cleanliness",
    )
    _fail(
        Path(str(source.get("git_top_level", ""))).resolve() == workspace.resolve(),
        "attestation generator top-level differs",
    )
    source_without_digest = {
        key: value for key, value in source.items() if key != "digest"
    }
    _fail(
        json_digest(source_without_digest) == source.get("digest"),
        "attestation source digest changed",
    )
    commit = str(source.get("git_commit", ""))
    tree = str(source.get("git_tree", ""))
    _fail(
        str(_git(workspace, ["rev-parse", f"{commit}^{{tree}}"])).strip() == tree,
        "attestation generator Git tree changed",
    )
    files = source.get("files", {})
    _fail(
        isinstance(files, dict) and files, "attestation generator source files absent"
    )
    for relative, descriptor in files.items():
        _assert_descriptor(
            workspace / relative, descriptor, f"attestation-source/{relative}"
        )
        _fail(
            git_file_descriptor(workspace, commit, relative) == descriptor,
            f"attestation generator commit bytes differ: {relative}",
        )
    return {
        "protocol": payload.get("protocol"),
        "retrospective": True,
        "independently_traced": False,
        "shadow_git_disclosed": True,
        "global_cleanliness_not_asserted": True,
        "generator_commit": commit,
        "generator_tree": tree,
        "generator_source_file_count": len(files),
        "sha256": sha256_file(path),
    }


def _audit_source_commit_verification(
    workspace: Path,
    attestation_root: Path,
    v1_preregistration: Mapping[str, Any],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    path = attestation_root / "source_commit_verification.json"
    payload = load_json(path)
    _fail(
        str(payload.get("protocol", "")).startswith("test_v4_aggregate_repair_v1"),
        "wrong source-verification protocol",
    )
    repository = payload.get("main_repository")
    repository_path = (
        repository.get("path") if isinstance(repository, Mapping) else repository
    )
    _fail(
        Path(str(repository_path)).resolve() == workspace.resolve(),
        "attested main repository differs",
    )
    _fail(
        payload.get("permanent_ref") == expectations.repair_git_ref,
        "attested permanent ref changed",
    )
    _fail(
        payload.get("commit") == expectations.repair_git_commit,
        "attested repair commit changed",
    )
    _fail(
        payload.get("parent") == expectations.repair_git_parent,
        "attested repair parent changed",
    )
    _fail(
        payload.get("tree") == expectations.repair_git_tree,
        "attested repair tree changed",
    )
    _fail(
        payload.get("global_worktree_cleanliness_asserted_for_v1") is False
        and isinstance(payload.get("v1_scope_clean_interpretation"), str),
        "source verification overstates v1 global cleanliness",
    )

    v1_files = v1_preregistration.get("repair_source_manifest", {}).get("files", {})
    source_files = payload.get("source_files", {})
    _fail(set(source_files) == set(v1_files), "attested repair source file set changed")
    for relative, item in source_files.items():
        _fail(
            item.get("all_byte_identical") is True,
            f"repair source mismatch reported: {relative}",
        )
        descriptors = (
            item.get("v1_manifest", {}),
            item.get("permanent_commit", {}),
            item.get("main_work_tree", {}),
        )
        _fail(
            all(
                _descriptor_equal(descriptor, v1_files[relative])
                for descriptor in descriptors
            ),
            f"repair source descriptor mismatch: {relative}",
        )
        _assert_descriptor(
            workspace / relative, v1_files[relative], f"attested-source/{relative}"
        )
        _fail(
            git_file_descriptor(workspace, expectations.repair_git_commit, relative)
            == v1_files[relative],
            f"permanent repair source bytes changed: {relative}",
        )
    _fail(
        payload.get("all_source_files_byte_identical") is True,
        "attested source equivalence failed",
    )

    bundle = payload.get("bundle", {})
    _fail(bundle.get("path") == "repair_v1_source.bundle", "bundle path changed")
    bundle_path = attestation_root / "repair_v1_source.bundle"
    _assert_descriptor(
        bundle_path,
        {"sha256": bundle.get("sha256"), "size": bundle.get("size")},
        "repair-source-bundle",
    )
    header = bundle.get("header", {})
    _fail(
        header.get("self_contained") is True and header.get("prerequisites") == [],
        "repair bundle is not self-contained",
    )
    _fail(
        isinstance(header.get("signature"), str) and bool(header.get("signature")),
        "repair bundle signature header absent",
    )
    references = header.get("references", {})
    _fail(
        references.get(expectations.repair_git_ref) == expectations.repair_git_commit,
        "repair bundle permanent ref changed",
    )
    _fail(
        bundle.get("git_bundle_verify_succeeded") is True,
        "recorded git bundle verification failed",
    )
    _git(workspace, ["bundle", "verify", str(bundle_path)])
    heads = str(_git(workspace, ["bundle", "list-heads", str(bundle_path)]))
    _fail(
        f"{expectations.repair_git_commit} {expectations.repair_git_ref}"
        in heads.splitlines(),
        "repair bundle does not contain permanent ref",
    )
    return {
        "protocol": payload.get("protocol"),
        "permanent_ref": expectations.repair_git_ref,
        "commit": expectations.repair_git_commit,
        "tree": expectations.repair_git_tree,
        "source_file_count": len(source_files),
        "all_source_files_byte_identical": True,
        "bundle_sha256": bundle["sha256"],
        "bundle_self_contained": True,
        "git_bundle_independently_verified": True,
        "sha256": sha256_file(path),
    }


def _audit_v1_integrity_reattestation(
    repair_aggregate: Path,
    attestation_root: Path,
    repair_evidence: Mapping[str, Any],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    path = attestation_root / "v1_integrity_reaudit.json"
    payload = load_json(path)
    _fail(
        str(payload.get("protocol", "")).startswith("test_v4_aggregate_repair_v1"),
        "wrong v1-integrity-reaudit protocol",
    )
    v1_files = [
        {
            "path": item.name,
            "sha256": sha256_file(item),
            "size": item.stat().st_size,
        }
        for item in sorted(repair_aggregate.iterdir())
        if item.is_file()
    ]
    v1_tree = payload.get("v1_tree", {})
    _fail(
        int(v1_tree.get("file_count", -1)) == len(v1_files),
        "attested v1 file count changed",
    )
    _fail(
        int(v1_tree.get("total_bytes", -1))
        == sum(int(item["size"]) for item in v1_files),
        "attested v1 byte count changed",
    )
    _fail(
        v1_tree.get("tree_digest") == json_digest(v1_files),
        "attested v1 tree digest changed",
    )
    final_path = repair_aggregate / "test_v4_repair_final_manifest.json"
    _fail(
        _descriptor_equal(
            payload.get("v1_final_manifest", {}), file_descriptor(final_path)
        ),
        "attested v1 final manifest changed",
    )
    _fail(
        _descriptor_equal(
            payload.get("v1_input_manifest", {}),
            file_descriptor(
                repair_aggregate / "aggregation_repair_input_manifest.json"
            ),
        ),
        "attested v1 input manifest changed",
    )
    _fail(
        _descriptor_equal(
            payload.get("v1_integrity_artifact", {}),
            file_descriptor(repair_aggregate / "aggregation_repair_integrity.json"),
        ),
        "attested v1 integrity artifact changed",
    )
    expected_formal_trees = {
        label: {
            "file_count": tree["file_count"],
            "total_bytes": tree["total_bytes"],
            "tree_digest": tree["tree_digest"],
        }
        for label, tree in repair_evidence.get("input_trees", {}).items()
    }
    _fail(
        payload.get("formal_input_trees") == expected_formal_trees,
        "attested formal input-tree digests changed",
    )
    v1_integrity = load_json(repair_aggregate / "aggregation_repair_integrity.json")
    _fail(
        payload.get("protected_tree") == v1_integrity.get("protected_tree_after")
        and payload.get("protected_tree", {}).get("tree_digest")
        == expectations.protected_tree_digest,
        "attested protected-tree digest changed",
    )
    for field in (
        "v1_hash_chain_valid",
        "v1_manifest_payload_digest_valid",
        "formal_input_trees_match_v1_before_and_after",
        "protected_tree_matches_v1_before_and_after",
        "original_failure_evidence_preserved",
    ):
        _fail(payload.get(field) is True, f"v1 integrity attestation failed: {field}")
    _fail(
        payload.get("original_failure_classification_changed") is False
        and payload.get("threshold_or_statistical_semantics_changed") is False
        and payload.get("v1_was_modified_by_attestation") is False,
        "attestation changed v1 evidence/semantics",
    )
    zero_fields = (
        "query_generation_count",
        "query_rerun_count",
        "query_generation_or_rerun_count",
        "solver_invocation_count",
        "model_inference_count",
        "bootstrap_resamples_executed_by_attestation",
    )
    _fail(
        all(int(payload.get(field, -1)) == 0 for field in zero_fields),
        "attestation performed forbidden scientific work",
    )
    _fail(
        repair_evidence.get("all_input_trees_unchanged") is True,
        "independent v1 input-tree audit did not pass",
    )
    return {
        "protocol": payload.get("protocol"),
        "v1_tree_digest": v1_tree["tree_digest"],
        "v1_file_count": len(v1_files),
        "v1_final_manifest_sha256": expectations.repair_output_sha256[
            "test_v4_repair_final_manifest.json"
        ],
        "zero_scientific_activity": True,
        "v1_unmodified": True,
        "sha256": sha256_file(path),
    }


def _audit_attestation_independent_recomputation(
    repair_aggregate: Path,
    attestation_root: Path,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    path = attestation_root / "independent_recomputation.json"
    payload = load_json(path)
    _fail(
        str(payload.get("protocol", "")).startswith("test_v4_aggregate_repair_v1"),
        "wrong attestation-recomputation protocol",
    )
    _fail(
        payload.get("implementation") == "python_standard_library_only",
        "attestation recomputation implementation changed",
    )
    _fail(
        int(payload.get("bootstrap_resamples_executed", -1)) == 0
        and payload.get("query_records_read") is False
        and payload.get("stored_pvalues_reused") is True,
        "attestation recomputation exceeded stored-aggregate scope",
    )
    _fail(
        _float_equal(payload.get("familywise_alpha"), 0.05),
        "attestation Holm alpha changed",
    )
    _fail(
        tuple(payload.get("confirmatory_members", ())) == CONFIRMATORY_METRICS,
        "attestation confirmatory family changed",
    )
    artifact_names = (
        "aggregate_summary_v4.json",
        "joint_holm_v4.json",
        "paper_gate_v4.json",
    )
    semantic_matches = payload.get("semantic_matches", {})
    _fail(
        set(semantic_matches) == set(artifact_names) and all(semantic_matches.values()),
        "attestation semantic recomputation mismatch",
    )
    _fail(
        payload.get("all_semantic_matches") is True,
        "attestation semantic equivalence failed",
    )
    stored = payload.get("stored_artifacts", {})
    recomputed_digests = payload.get("recomputed_payload_digests", {})
    _fail(
        set(stored) == set(artifact_names)
        and set(recomputed_digests) == set(artifact_names),
        "attestation recomputation artifact set changed",
    )
    for name in artifact_names:
        artifact_path = repair_aggregate / name
        _assert_descriptor(artifact_path, stored[name], f"attestation-stored/{name}")
        _fail(
            recomputed_digests[name] == json_digest(load_json(artifact_path)),
            f"attestation semantic payload digest changed: {name}",
        )
    observed = payload.get("observed_results_not_used_as_acceptance_criteria", {})
    _fail(
        observed.get("robot_gates") == statistics.get("robot_gates"),
        "attestation observed robot gates changed",
    )
    _fail(
        observed.get("joint_holm_gate_pass")
        is bool(statistics.get("all_joint_holm_claims_pass")),
        "attestation observed Holm result changed",
    )
    _fail(
        observed.get("paper_gate_pass") is bool(statistics.get("paper_gate_pass")),
        "attestation observed paper result changed",
    )
    _fail(
        payload.get("outcome_direction_hardcoded") is False,
        "attestation hardcoded scientific outcome direction",
    )
    _fail(
        isinstance(payload.get("acceptance_rule"), str)
        and payload.get("acceptance_rule"),
        "attestation acceptance rule absent",
    )
    return {
        "protocol": payload.get("protocol"),
        "implementation": payload["implementation"],
        "stored_pvalue_recomputation_only": True,
        "semantic_matches": dict(semantic_matches),
        "observed_results": observed,
        "outcome_direction_hardcoded": False,
        "sha256": sha256_file(path),
    }


def audit_external_execution_attestation(
    workspace: Path,
    repair_aggregate: Path,
    attestation_root: Path,
    expectations: AuditExpectations,
    *,
    repair_evidence: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    _fail(
        attestation_root.resolve()
        == (workspace / expectations.attestation_namespace).resolve(),
        "attestation namespace changed",
    )
    _fail(
        attestation_root.is_dir() and not attestation_root.is_symlink(),
        "attestation namespace missing/non-regular",
    )
    v1_preregistration = load_json(
        repair_aggregate / "aggregation_repair_preregistration.json"
    )
    execution = _audit_execution_provenance_attestation(
        workspace, attestation_root, expectations
    )
    source = _audit_source_commit_verification(
        workspace, attestation_root, v1_preregistration, expectations
    )
    integrity = _audit_v1_integrity_reattestation(
        repair_aggregate,
        attestation_root,
        repair_evidence,
        expectations,
    )
    recomputation = _audit_attestation_independent_recomputation(
        repair_aggregate, attestation_root, statistics
    )

    final_path = attestation_root / "attestation_final_manifest.json"
    final = load_json(final_path)
    _fail(
        str(final.get("protocol", "")).startswith("test_v4_aggregate_repair_v1"),
        "wrong attestation-final protocol",
    )
    expected_names = {
        "execution_provenance_attestation.json",
        "source_commit_verification.json",
        "v1_integrity_reaudit.json",
        "independent_recomputation.json",
        "repair_v1_source.bundle",
        "attestation_final_manifest.json",
    }
    actual_names = {path.name for path in attestation_root.iterdir() if path.is_file()}
    _fail(actual_names == expected_names, "attestation namespace file set changed")
    expected_chain = {
        name: sha256_file(attestation_root / name)
        for name in sorted(expected_names - {final_path.name})
    }
    _fail(final.get("hash_chain") == expected_chain, "attestation hash chain changed")
    _fail(
        final.get("hash_chain_digest") == json_digest(expected_chain),
        "attestation hash-chain digest changed",
    )
    final_without_digest = {
        key: value for key, value in final.items() if key != "manifest_payload_digest"
    }
    _fail(
        final.get("manifest_payload_digest") == json_digest(final_without_digest),
        "attestation final payload digest changed",
    )
    _fail(
        isinstance(final.get("first_final_recheck_digest"), str)
        and bool(final.get("first_final_recheck_digest")),
        "attestation first final recheck digest absent",
    )
    if expectations.attestation_final_manifest_sha256:
        _fail(
            sha256_file(final_path) == expectations.attestation_final_manifest_sha256,
            "attestation final manifest differs from independent post-run pin",
        )
    else:
        raise AuditError(
            "attestation final manifest has not yet been independently pinned in the auditor"
        )
    required_true = (
        "retrospective_attestation",
        "scope_clean_not_global_clean",
        "shadow_git_invocation_disclosed",
        "source_commit_permanently_recoverable_from_bundle",
        "independent_recomputation_semantically_identical",
        "composite_v1_plus_attestation_integrity_pass",
        "second_pre_promotion_recheck_required",
    )
    _fail(
        all(final.get(field) is True for field in required_true),
        "attestation final integrity flag failed",
    )
    _fail(
        final.get("attestation_is_part_of_v1_tree") is False
        and final.get("v1_tree_modified") is False
        and final.get("original_outputs_modified") is False
        and final.get("scientific_gate_direction_used_for_acceptance") is False
        and final.get("automatic_rerun_allowed") is False,
        "attestation final manifest changes/overstates v1",
    )
    zero_fields = (
        "query_generation_count",
        "query_rerun_count",
        "solver_invocation_count",
        "model_inference_count",
        "bootstrap_resample_count",
    )
    _fail(
        all(int(final.get(field, -1)) == 0 for field in zero_fields),
        "attestation final manifest reports scientific execution",
    )
    staging = (
        workspace / "outputs/.test_v4_aggregate_repair_v1_attestation_v1.incomplete"
    )
    _fail(
        not staging.exists() and not staging.is_symlink(),
        "attestation staging remains after promotion",
    )
    return {
        "execution_provenance": execution,
        "source_commit_verification": source,
        "v1_integrity_reaudit": integrity,
        "independent_recomputation": recomputation,
        "final_manifest_sha256": sha256_file(final_path),
        "sealed_file_count": len(actual_names),
        "hash_chain_valid": True,
        "manifest_payload_digest_valid": True,
        "retrospective_limitations_disclosed": True,
        "verdict": "PASS",
    }


def audit_script_has_no_scientific_imports(script_path: Path) -> None:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    _fail(
        not (imported & FORBIDDEN_RUNTIME_MODULES), "auditor imports scientific runtime"
    )


def run_audit(
    workspace: Path,
    *,
    failed_aggregate: Path,
    repair_aggregate: Path,
    attestation_root: Path,
    expectations: AuditExpectations = PRODUCTION,
    verify_fingerprint_files: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    failed_aggregate = failed_aggregate.resolve()
    repair_aggregate = repair_aggregate.resolve()
    attestation_root = attestation_root.resolve()
    audit_script_has_no_scientific_imports(Path(__file__).resolve())
    report: dict[str, Any] = {
        "protocol": "independent_test_v4_aggregation_repair_audit_v1",
        "read_only": True,
        "scientific_runtime_imported_by_auditor": False,
    }
    report["control_plane"] = audit_control_plane(
        workspace,
        failed_aggregate,
        expectations,
        verify_fingerprint_files=verify_fingerprint_files,
    )
    report["original_failed_tree"] = audit_failed_tree(failed_aggregate, expectations)
    combinations, roots = audit_all_combinations(workspace, expectations)
    report["formal_measurements"] = combinations
    report["repair_evidence"] = audit_repair_manifest(
        workspace, repair_aggregate, failed_aggregate, expectations
    )
    report["statistics_and_gates"] = audit_recomputed_statistics_and_gates(
        repair_aggregate, roots, expectations
    )
    report["final_seal"] = audit_final_seal(
        workspace,
        repair_aggregate,
        expectations,
        expected_paper_gate=bool(report["statistics_and_gates"]["paper_gate_pass"]),
        expected_final_input_recheck_digest=str(
            report["repair_evidence"]["final_input_recheck_digest"]
        ),
    )
    report["external_execution_attestation"] = audit_external_execution_attestation(
        workspace,
        repair_aggregate,
        attestation_root,
        expectations,
        repair_evidence=report["repair_evidence"],
        statistics=report["statistics_and_gates"],
    )
    report["verdict"] = "PASS"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--failed-aggregate",
        type=Path,
        default=Path("outputs/.test_v4_aggregate.incomplete"),
        help="original failed aggregation tree; it must remain byte-identical",
    )
    parser.add_argument(
        "--repair-aggregate",
        type=Path,
        default=Path("outputs/test_v4_aggregate_repair_v1"),
        help="sealed, authoritative aggregation-only repair namespace",
    )
    parser.add_argument(
        "--attestation-root",
        type=Path,
        default=Path("outputs/test_v4_aggregate_repair_v1_attestation_v1"),
        help="independent, retrospective execution-attestation namespace",
    )
    parser.add_argument(
        "--skip-fingerprint-file-rehash",
        action="store_true",
        help="diagnostic-only shortcut; never use for the final independent audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace.resolve()
    failed = arguments.failed_aggregate
    repair = arguments.repair_aggregate
    attestation = arguments.attestation_root
    if not failed.is_absolute():
        failed = workspace / failed
    if not repair.is_absolute():
        repair = workspace / repair
    if not attestation.is_absolute():
        attestation = workspace / attestation
    try:
        report = run_audit(
            workspace,
            failed_aggregate=failed,
            repair_aggregate=repair,
            attestation_root=attestation,
            verify_fingerprint_files=not arguments.skip_fingerprint_file_rehash,
        )
    except AuditError as error:
        print(
            json.dumps(
                {
                    "protocol": "independent_test_v4_aggregation_repair_audit_v1",
                    "read_only": True,
                    "verdict": "FAIL",
                    "blocking_error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
