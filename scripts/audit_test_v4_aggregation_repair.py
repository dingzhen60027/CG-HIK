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
* Panda remains failed, UR5e remains passed, and the paper gate remains failed;
* the repair declares zero query generation, solver calls, model inference,
  checkpoint rewriting, and bootstrap resampling; and
* the final manifest hashes the promoted output tree and records an atomic,
  aggregation-only promotion.

The repair-side manifest schema expected by this auditor is deliberately
small.  ``outputs/test_v4_aggregate/aggregation_repair_manifest.json`` must
contain the fields checked by :func:`audit_repair_manifest`.  The manifest is
evidence, not the sole basis of the audit: raw checkpoint cross-products,
Holm, gates, and all file hashes are independently recomputed here.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
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
            _assert_descriptor(path, descriptor, f"frozen-source/{raw_path}")

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
        _fail(
            value is expectations.robot_gate_expectation[robot],
            f"robot gate changed: {robot}",
        )
        failed_checks[robot] = sorted(
            name for name, passed in gate.get("checks", {}).items() if not bool(passed)
        )
    _fail(
        failed_checks["panda"] == ["ood_feasible_false_reject_improvement"],
        "Panda failure reason changed",
    )
    _fail(not failed_checks["ur5e"], "UR5e acquired a failed gate")

    paper = load_json(repair_aggregate / "paper_gate_v4.json")
    expected_paper = all(robot_gates.values()) and bool(
        expected_holm["all_confirmatory_nulls_rejected"]
    )
    _fail(
        expected_paper is expectations.paper_gate_expectation,
        "pinned paper result assumption changed",
    )
    _fail(paper.get("robot_gates") == robot_gates, "reported robot gates changed")
    _fail(paper.get("joint_holm_gate_pass") is True, "joint Holm gate did not pass")
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


def audit_repair_tool_source(path: Path, expected_sha256: str) -> dict[str, Any]:
    _fail(path.is_file() and not path.is_symlink(), f"repair tool missing: {path}")
    _fail(sha256_file(path) == expected_sha256, "repair tool source hash changed")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AuditError(f"repair tool is invalid Python: {error}") from error
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                calls.add(target.id)
            elif isinstance(target, ast.Attribute):
                calls.add(target.attr)
    forbidden_imports = imports & FORBIDDEN_RUNTIME_MODULES
    _fail(
        not forbidden_imports,
        f"repair tool imports scientific runtime: {sorted(forbidden_imports)}",
    )
    forbidden_calls = calls & {
        "_build_methods",
        "_load_frozen_test_datasets",
        "_run_combination",
        "generate_locked_datasets",
        "load_model",
        "run_query",
        "solve",
    }
    _fail(
        not forbidden_calls,
        f"repair tool contains forbidden scientific calls: {sorted(forbidden_calls)}",
    )
    _fail(
        "rename" in calls or "replace" in calls,
        "repair tool has no atomic promotion call",
    )
    return {
        "sha256": expected_sha256,
        "imports": sorted(imports),
        "forbidden_scientific_imports": [],
        "forbidden_scientific_calls": [],
        "atomic_promotion_call_present": True,
    }


def audit_repair_manifest(
    workspace: Path,
    repair_aggregate: Path,
    failed_aggregate: Path,
    expectations: AuditExpectations,
) -> dict[str, Any]:
    manifest_path = repair_aggregate / "aggregation_repair_manifest.json"
    manifest = load_json(manifest_path)
    _fail(
        manifest.get("protocol") == "test_v4_aggregation_only_repair_v1",
        "wrong repair protocol",
    )
    _fail(manifest.get("status") == "completed", "repair is not complete")
    _fail(manifest.get("repair_scope") == "aggregation_only", "repair scope expanded")
    activity = manifest.get("scientific_activity", {})
    required_zero = (
        "query_generation_calls",
        "solver_calls",
        "model_inference_calls",
        "checkpoint_record_writes",
        "bootstrap_resamples",
        "threshold_changes",
        "gate_definition_changes",
    )
    _fail(set(activity) >= set(required_zero), "repair activity ledger is incomplete")
    _fail(
        all(activity[name] == 0 for name in required_zero),
        "repair performed forbidden scientific activity",
    )

    inputs = manifest.get("input_evidence", {})
    _fail(
        inputs.get("preregistration_sha256") == expectations.preregistration_sha256,
        "repair input preregistration differs",
    )
    _fail(
        inputs.get("dataset_manifest_sha256") == expectations.dataset_manifest_sha256,
        "repair input dataset differs",
    )
    _fail(
        inputs.get("control_plane_seal_sha256")
        == expectations.control_plane_seal_sha256,
        "repair input control seal differs",
    )
    _fail(
        inputs.get("evidence_fingerprint_digest")
        == expectations.evidence_fingerprint_digest,
        "repair input source fingerprint differs",
    )
    _fail(
        inputs.get("failed_tree_digest") == expectations.failed_tree_digest,
        "repair did not bind original failure tree",
    )
    _fail(
        inputs.get("combination_completion_sha256")
        == dict(expectations.completion_sha256),
        "repair did not bind six completion markers",
    )
    _fail(
        int(inputs.get("checkpoint_count", -1)) == expectations.total_checkpoint_count,
        "repair checkpoint count differs",
    )
    _fail(
        int(inputs.get("record_count", -1)) == expectations.total_record_count,
        "repair record count differs",
    )

    bug = manifest.get("bug_classification", {})
    _fail(
        bug.get("class") == "json_mapping_key_order_only",
        "repair bug classification changed",
    )
    _fail(
        bug.get("stored_metric_values_changed") is False,
        "repair reports changed stored metrics",
    )
    _fail(
        bug.get("stored_unadjusted_pvalues_reused") is True,
        "repair did not reuse stored p-values",
    )

    promotion = manifest.get("atomic_promotion", {})
    _fail(
        promotion.get("atomic_directory_rename") is True,
        "repair was not atomically promoted",
    )
    _fail(
        promotion.get("same_filesystem") is True,
        "repair promotion filesystem is unproven",
    )
    staging_relative = promotion.get("staging_path")
    final_relative = promotion.get("final_path")
    _fail(
        isinstance(staging_relative, str) and isinstance(final_relative, str),
        "repair promotion paths missing",
    )
    staging = workspace / staging_relative
    final = workspace / final_relative
    _fail(final.resolve() == repair_aggregate.resolve(), "repair final path differs")
    _fail(
        staging.resolve() != failed_aggregate.resolve(),
        "repair reused/destructively promoted failed tree",
    )
    _fail(
        not staging.exists() and not staging.is_symlink(),
        "repair staging remains after promotion",
    )
    _fail(
        repair_aggregate.is_dir() and not repair_aggregate.is_symlink(),
        "final aggregate is not a real directory",
    )

    tool = manifest.get("repair_tool", {})
    tool_path = workspace / str(tool.get("path", ""))
    source_audit = audit_repair_tool_source(tool_path, str(tool.get("sha256", "")))
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "zero_activity_fields": list(required_zero),
        "atomic_staging_absent": True,
        "failed_tree_separate": True,
        "repair_tool": source_audit,
    }


def audit_final_seal(
    workspace: Path,
    repair_aggregate: Path,
    roots: Mapping[tuple[str, int], Path],
    expectations: AuditExpectations,
) -> dict[str, Any]:
    final_path = repair_aggregate / "test_v4_final_manifest.json"
    final = load_json(final_path)
    _fail(
        final.get("protocol") == "test_v4 final immutable evidence manifest",
        "wrong final protocol",
    )
    _fail(
        final.get("formal_completion_marker") is True, "formal completion marker absent"
    )
    _fail(final.get("all_six_natural_exits") is True, "six natural exits not retained")
    _fail(
        final.get("paper_gate_pass") is expectations.paper_gate_expectation,
        "final paper gate changed",
    )
    _fail(
        final.get("test_set_retuning_performed") is False,
        "final manifest reports retuning",
    )
    _fail(
        final.get("threshold_or_gate_changes_after_test") is False,
        "final manifest reports gate changes",
    )
    _fail(
        final.get("outliers_removed") is False
        and final.get("winsorization_performed") is False,
        "final manifest reports data editing",
    )
    _fail(
        final.get("preregistration_sha256") == expectations.preregistration_sha256,
        "final preregistration anchor changed",
    )
    _fail(
        final.get("dataset_manifest_sha256") == expectations.dataset_manifest_sha256,
        "final dataset anchor changed",
    )
    _fail(
        final.get("control_plane_seal_sha256")
        == expectations.control_plane_seal_sha256,
        "final control seal changed",
    )
    protected = final.get("protected_outputs", {})
    _fail(
        protected.get("unchanged") is True
        and protected.get("before") == protected.get("after"),
        "protected evidence changed",
    )
    _fail(
        protected.get("before", {}).get("tree_digest")
        == expectations.protected_tree_digest,
        "protected evidence digest changed",
    )

    declared_list = final.get("files", ())
    declared: dict[str, dict[str, Any]] = {}
    for item in declared_list:
        path = str(item.get("path"))
        _fail(path not in declared, f"duplicate final-manifest path: {path}")
        declared[path] = {"sha256": item.get("sha256"), "size": item.get("size")}

    actual_paths: dict[str, Path] = {}
    for seed in expectations.seeds:
        seed_root = workspace / "outputs" / f"test_v4_seed{seed}"
        _fail(
            seed_root.is_dir() and not seed_root.is_symlink(),
            f"seed root not promoted: seed{seed}",
        )
        for path in seed_root.rglob("*"):
            if path.is_file():
                actual_paths[str(path.relative_to(workspace))] = path
    for path in repair_aggregate.rglob("*"):
        if path.is_file() and path != final_path:
            actual_paths[str(path.relative_to(workspace))] = path
    _fail(
        set(declared) == set(actual_paths),
        "final manifest artifact set differs from disk",
    )
    for relative, path in actual_paths.items():
        _assert_descriptor(path, declared[relative], f"final-seal/{relative}")

    # The promoted combination markers must still be the pinned originals.
    for (robot, seed), root in roots.items():
        final_root = workspace / "outputs" / f"test_v4_seed{seed}" / robot
        _fail(
            final_root.is_dir(),
            f"combination not present in final root: {robot}/seed{seed}",
        )
        _fail(
            sha256_file(final_root / "combination_complete.json")
            == expectations.completion_sha256[_combination_key(robot, seed)],
            f"promoted completion marker changed: {robot}/seed{seed}",
        )
    return {
        "final_manifest_sha256": sha256_file(final_path),
        "sealed_artifact_count": len(declared),
        "artifact_set_exact": True,
        "all_hashes_valid": True,
        "paper_gate_pass": final["paper_gate_pass"],
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
    expectations: AuditExpectations = PRODUCTION,
    verify_fingerprint_files: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    failed_aggregate = failed_aggregate.resolve()
    repair_aggregate = repair_aggregate.resolve()
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
    report["repair_manifest"] = audit_repair_manifest(
        workspace, repair_aggregate, failed_aggregate, expectations
    )
    report["statistics_and_gates"] = audit_recomputed_statistics_and_gates(
        repair_aggregate, roots, expectations
    )
    report["final_seal"] = audit_final_seal(
        workspace, repair_aggregate, roots, expectations
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
        default=Path("outputs/test_v4_aggregate"),
        help="atomically promoted aggregation-only repair",
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
    if not failed.is_absolute():
        failed = workspace / failed
    if not repair.is_absolute():
        repair = workspace / repair
    try:
        report = run_audit(
            workspace,
            failed_aggregate=failed,
            repair_aggregate=repair,
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
