from __future__ import annotations

import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_test_v4_aggregation_repair as audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _descriptor(path: Path) -> dict[str, object]:
    return {"sha256": audit.sha256_file(path), "size": path.stat().st_size}


def _write_records(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _interval(metric_order: tuple[str, ...], p99: float = 0.0001) -> dict[str, object]:
    pvalues = {
        "feasible_success_gap": 0.0001,
        "feasible_p95_latency_ratio": 0.0001,
        "feasible_p99_latency_ratio": p99,
        "trajectory_completion_gap": 0.0001,
    }
    return {
        "inference_family": {
            "members": list(audit.CONFIRMATORY_METRICS),
            "member_count_per_robot": 4,
            "per_robot_holm_applied": False,
            "joint_robot_holm_required": True,
        },
        "metrics": {
            name: {"one_sided_unadjusted_p": pvalues[name]} for name in metric_order
        },
    }


def test_joint_holm_ignores_serialized_mapping_key_order(tmp_path: Path) -> None:
    panda = tmp_path / "panda"
    ur5e = tmp_path / "ur5e"
    _write_json(
        panda / "paired_intervals_v4.json",
        _interval(tuple(reversed(audit.CONFIRMATORY_METRICS))),
    )
    _write_json(
        ur5e / "paired_intervals_v4.json",
        _interval(
            (
                "feasible_p95_latency_ratio",
                "feasible_p99_latency_ratio",
                "feasible_success_gap",
                "trajectory_completion_gap",
            ),
            p99=0.0015,
        ),
    )

    result = audit.recompute_joint_holm({"panda": panda, "ur5e": ur5e})

    assert result["hypothesis_count"] == 8
    assert result["all_confirmatory_nulls_rejected"] is True
    assert set(result["hypotheses"]) == {
        f"{robot}/{metric}"
        for robot in ("panda", "ur5e")
        for metric in audit.CONFIRMATORY_METRICS
    }


def test_statistics_audit_preserves_panda_false_ur5e_true_paper_false(
    tmp_path: Path,
) -> None:
    roots: dict[tuple[str, int], Path] = {}
    for seed in audit.PRODUCTION.seeds:
        for robot in audit.PRODUCTION.robots:
            root = tmp_path / f"seed{seed}" / robot
            roots[(robot, seed)] = root
            summary = {"robot": robot, "seed": seed, "sealed": True}
            _write_json(root / "summary_v4.json", summary)
            if seed == audit.PRODUCTION.primary_seed:
                _write_json(
                    root / "paired_intervals_v4.json",
                    _interval(
                        tuple(reversed(audit.CONFIRMATORY_METRICS)),
                        p99=0.0015 if robot == "ur5e" else 0.0001,
                    ),
                )
                checks = {
                    "ood_feasible_false_reject_improvement": robot == "ur5e",
                    "all_other_frozen_checks": True,
                }
                _write_json(
                    root / "claim_gate_v4.json",
                    {"formal_gate_pass": robot == "ur5e", "checks": checks},
                )
                _write_json(
                    root / "ood_abstention_v4.json",
                    {"robot": robot, "sealed": True},
                )

    aggregate = tmp_path / "aggregate"
    primary_roots = {
        robot: roots[(robot, audit.PRODUCTION.primary_seed)]
        for robot in audit.PRODUCTION.robots
    }
    joint = audit.recompute_joint_holm(primary_roots)
    _write_json(aggregate / "joint_holm_v4.json", joint)
    _write_json(
        aggregate / "paper_gate_v4.json",
        {
            "robot_gates": {"panda": False, "ur5e": True},
            "joint_holm_gate_pass": True,
            "both_robot_gates_pass": False,
            "test_set_retuning_performed": False,
        },
    )
    aggregate_summary = {"primary": {}, "sensitivity": {}}
    for robot in audit.PRODUCTION.robots:
        root = primary_roots[robot]
        aggregate_summary["primary"][robot] = {
            "claim_gate": audit.load_json(root / "claim_gate_v4.json"),
            "summary": audit.load_json(root / "summary_v4.json"),
            "ood_abstention": audit.load_json(root / "ood_abstention_v4.json"),
        }
        aggregate_summary["sensitivity"][robot] = {
            f"seed{seed}": audit.load_json(roots[(robot, seed)] / "summary_v4.json")
            for seed in audit.PRODUCTION.seeds
            if seed != audit.PRODUCTION.primary_seed
        }
    _write_json(aggregate / "aggregate_summary_v4.json", aggregate_summary)

    result = audit.audit_recomputed_statistics_and_gates(
        aggregate, roots, audit.PRODUCTION
    )

    assert result["all_joint_holm_claims_pass"] is True
    assert result["robot_gates"] == {"panda": False, "ur5e": True}
    assert result["paper_gate_pass"] is False

    paper = audit.load_json(aggregate / "paper_gate_v4.json")
    paper["both_robot_gates_pass"] = True
    _write_json(aggregate / "paper_gate_v4.json", paper)
    with pytest.raises(audit.AuditError, match="reported paper gate changed"):
        audit.audit_recomputed_statistics_and_gates(aggregate, roots, audit.PRODUCTION)


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object], tuple[str, ...]]:
    methods = ("fixed_robust_cascade", "proposed_v4")
    checkpoint = tmp_path / "checkpoints" / "id_points" / "query_000000_000001"
    hashes = ("a" * 64, "b" * 64)
    rows = [
        {
            "query_index": index,
            "method": method,
            "robot": "panda",
            "training_seed": 17,
            "role": "id_points",
            "source_query_sha256": hashes[index],
        }
        for index in range(2)
        for method in methods
    ]
    record_path = checkpoint / "records.jsonl.gz"
    _write_records(record_path, rows)
    manifest: dict[str, object] = {
        "role": "id_points",
        "source_indices": [0, 1],
        "source_query_sha256": list(hashes),
        "expected_query_count": 2,
        "expected_record_count": 4,
        "artifacts": {"records.jsonl.gz": _descriptor(record_path)},
    }
    manifest_path = checkpoint / "checkpoint_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest, methods


def test_checkpoint_audit_requires_exact_method_query_cross_product(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, methods = _checkpoint(tmp_path)

    count, identities, indices = audit._validate_checkpoint_records(
        manifest_path,
        manifest,
        robot="panda",
        seed=17,
        methods=methods,
    )

    assert count == 4
    assert len(identities) == 2
    assert indices == {("id_points", 0), ("id_points", 1)}


def test_checkpoint_audit_rejects_missing_pair_even_with_valid_gzip(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, methods = _checkpoint(tmp_path)
    record_path = manifest_path.parent / "records.jsonl.gz"
    rows = []
    with gzip.open(record_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    _write_records(record_path, rows[:-1])
    manifest["artifacts"] = {"records.jsonl.gz": _descriptor(record_path)}

    with pytest.raises(
        audit.AuditError, match="incomplete checkpoint cross-product|record count"
    ):
        audit._validate_checkpoint_records(
            manifest_path,
            manifest,
            robot="panda",
            seed=17,
            methods=methods,
        )


def test_combination_audit_recomputes_sealed_checkpoint_and_record_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "panda"
    checkpoint = root / "checkpoints" / "id_points" / "query_000000_000001"
    methods = audit.PRIMARY_METHODS
    hashes = ("a" * 64, "b" * 64)
    rows = [
        {
            "query_index": index,
            "method": method,
            "robot": "panda",
            "training_seed": 17,
            "role": "id_points",
            "source_query_sha256": hashes[index],
        }
        for index in range(2)
        for method in methods
    ]
    records = checkpoint / "records.jsonl.gz"
    _write_records(records, rows)
    manifest = {
        "protocol": "test_v4_atomic_measurement_checkpoint",
        "robot": "panda",
        "training_seed": 17,
        "role": "id_points",
        "methods": list(methods),
        "source_indices": [0, 1],
        "source_query_sha256": list(hashes),
        "expected_query_count": 2,
        "expected_record_count": 2 * len(methods),
        "preregistration_sha256": audit.PRODUCTION.preregistration_sha256,
        "dataset_manifest_sha256": audit.PRODUCTION.dataset_manifest_sha256,
        "evidence_fingerprint_digest": audit.PRODUCTION.evidence_fingerprint_digest,
        "artifacts": {"records.jsonl.gz": _descriptor(records)},
    }
    _write_json(checkpoint / "checkpoint_manifest.json", manifest)
    artifacts = {
        str(path.relative_to(root)): _descriptor(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    completion = {
        "robot": "panda",
        "training_seed": 17,
        "methods": list(methods),
        "all_checkpoints_hash_validated": True,
        "preregistration_sha256": audit.PRODUCTION.preregistration_sha256,
        "dataset_manifest_sha256": audit.PRODUCTION.dataset_manifest_sha256,
        "evidence_fingerprint_digest": audit.PRODUCTION.evidence_fingerprint_digest,
        "artifacts": artifacts,
    }
    _write_json(root / "combination_complete.json", completion)
    expectations = replace(
        audit.PRODUCTION,
        roles=("id_points",),
        role_query_counts={"id_points": 2},
        checkpoint_count_per_combination=1,
        combination_record_counts={"panda/seed17": 2 * len(methods)},
        completion_sha256={
            "panda/seed17": audit.sha256_file(root / "combination_complete.json")
        },
    )

    result = audit.audit_combination(
        root, robot="panda", seed=17, expectations=expectations
    )

    assert result["checkpoint_count"] == 1
    assert result["query_count"] == 2
    assert result["record_count"] == 14


def _failed_tree(root: Path) -> audit.AuditExpectations:
    failures = root / "failure_manifests"
    failures.mkdir(parents=True)
    last: dict[str, object] | None = None
    for index in range(42):
        resumable = index < 41
        payload: dict[str, object] = {
            "failure_classification": (
                "resumable_external_environment_technical_interruption"
                if resumable
                else "non_resumable_integrity_or_scientific_failure"
            ),
            "resume_eligible": resumable,
            "exception_message": (
                "host interruption"
                if resumable
                else "panda confirmatory metrics changed"
            ),
        }
        _write_json(failures / f"failure_{index:02d}.json", payload)
        last = payload
    assert last is not None
    _write_json(root / "latest_failure_manifest.json", last)
    _write_json(
        root / "resume_history.json", {"events": [{"index": i} for i in range(41)]}
    )
    paths = sorted(failures.glob("*.json")) + [
        root / "latest_failure_manifest.json",
        root / "resume_history.json",
    ]
    digest = audit.json_digest(audit.snapshot(paths, root))
    return replace(audit.PRODUCTION, failed_tree_digest=digest)


def test_failed_tree_is_pinned_and_tampering_is_detected(tmp_path: Path) -> None:
    expectations = _failed_tree(tmp_path)
    result = audit.audit_failed_tree(tmp_path, expectations)
    assert result["failure_manifest_count"] == 42
    assert result["resume_event_count"] == 41

    with (tmp_path / "resume_history.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(audit.AuditError, match="original failed tree changed"):
        audit.audit_failed_tree(tmp_path, expectations)


def _repair_tool(path: Path, *, forbidden: bool = False) -> str:
    source = (
        "import torch\nfrom pathlib import Path\nPath('a').rename('b')\n"
        if forbidden
        else "from pathlib import Path\ndef promote(a: Path, b: Path):\n    a.rename(b)\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return audit.sha256_file(path)


def _repair_manifest(
    workspace: Path,
    repair: Path,
    failed: Path,
    expectations: audit.AuditExpectations,
) -> dict[str, object]:
    tool_path = workspace / "scripts" / "repair_fixture.py"
    tool_hash = _repair_tool(tool_path)
    return {
        "protocol": "test_v4_aggregation_only_repair_v1",
        "status": "completed",
        "repair_scope": "aggregation_only",
        "scientific_activity": {
            "query_generation_calls": 0,
            "solver_calls": 0,
            "model_inference_calls": 0,
            "checkpoint_record_writes": 0,
            "bootstrap_resamples": 0,
            "threshold_changes": 0,
            "gate_definition_changes": 0,
        },
        "input_evidence": {
            "preregistration_sha256": expectations.preregistration_sha256,
            "dataset_manifest_sha256": expectations.dataset_manifest_sha256,
            "control_plane_seal_sha256": expectations.control_plane_seal_sha256,
            "evidence_fingerprint_digest": expectations.evidence_fingerprint_digest,
            "failed_tree_digest": expectations.failed_tree_digest,
            "combination_completion_sha256": dict(expectations.completion_sha256),
            "checkpoint_count": expectations.total_checkpoint_count,
            "record_count": expectations.total_record_count,
        },
        "bug_classification": {
            "class": "json_mapping_key_order_only",
            "stored_metric_values_changed": False,
            "stored_unadjusted_pvalues_reused": True,
        },
        "atomic_promotion": {
            "atomic_directory_rename": True,
            "same_filesystem": True,
            "staging_path": "outputs/.test_v4_aggregate.repair.incomplete",
            "final_path": str(repair.relative_to(workspace)),
        },
        "repair_tool": {
            "path": str(tool_path.relative_to(workspace)),
            "sha256": tool_hash,
        },
    }


def test_repair_manifest_requires_zero_scientific_activity_and_atomic_promotion(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "outputs" / ".test_v4_aggregate.incomplete"
    repair = tmp_path / "outputs" / "test_v4_aggregate"
    failed.mkdir(parents=True)
    repair.mkdir(parents=True)
    manifest = _repair_manifest(tmp_path, repair, failed, audit.PRODUCTION)
    _write_json(repair / "aggregation_repair_manifest.json", manifest)

    result = audit.audit_repair_manifest(tmp_path, repair, failed, audit.PRODUCTION)
    assert result["atomic_staging_absent"] is True
    assert result["failed_tree_separate"] is True

    manifest["scientific_activity"]["solver_calls"] = 1  # type: ignore[index]
    _write_json(repair / "aggregation_repair_manifest.json", manifest)
    with pytest.raises(audit.AuditError, match="forbidden scientific activity"):
        audit.audit_repair_manifest(tmp_path, repair, failed, audit.PRODUCTION)


def test_repair_tool_static_audit_rejects_scientific_runtime_import(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "repair.py"
    digest = _repair_tool(tool, forbidden=True)
    with pytest.raises(audit.AuditError, match="imports scientific runtime"):
        audit.audit_repair_tool_source(tool, digest)


def _sealed_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[tuple[str, int], Path], audit.AuditExpectations]:
    expectations = replace(
        audit.PRODUCTION,
        robots=("panda",),
        seeds=(17,),
        completion_sha256={"panda/seed17": "pending"},
        paper_gate_expectation=False,
    )
    root = tmp_path / "outputs" / "test_v4_seed17" / "panda"
    _write_json(root / "combination_complete.json", {"fixture": True})
    marker_hash = audit.sha256_file(root / "combination_complete.json")
    expectations = replace(
        expectations, completion_sha256={"panda/seed17": marker_hash}
    )
    aggregate = tmp_path / "outputs" / "test_v4_aggregate"
    _write_json(aggregate / "aggregation_repair_manifest.json", {"fixture": True})
    files: list[dict[str, object]] = []
    for path in sorted(
        list((tmp_path / "outputs" / "test_v4_seed17").rglob("*"))
        + list(aggregate.rglob("*"))
    ):
        if path.is_file() and path.name != "test_v4_final_manifest.json":
            descriptor = _descriptor(path)
            files.append(
                {
                    "path": str(path.relative_to(tmp_path)),
                    "sha256": descriptor["sha256"],
                    "size": descriptor["size"],
                }
            )
    final = {
        "protocol": "test_v4 final immutable evidence manifest",
        "formal_completion_marker": True,
        "all_six_natural_exits": True,
        "paper_gate_pass": False,
        "test_set_retuning_performed": False,
        "threshold_or_gate_changes_after_test": False,
        "outliers_removed": False,
        "winsorization_performed": False,
        "preregistration_sha256": expectations.preregistration_sha256,
        "dataset_manifest_sha256": expectations.dataset_manifest_sha256,
        "control_plane_seal_sha256": expectations.control_plane_seal_sha256,
        "protected_outputs": {
            "unchanged": True,
            "before": {"tree_digest": expectations.protected_tree_digest},
            "after": {"tree_digest": expectations.protected_tree_digest},
        },
        "files": files,
    }
    _write_json(aggregate / "test_v4_final_manifest.json", final)
    return aggregate, {("panda", 17): root}, expectations


def test_final_seal_requires_exact_artifact_set_and_hashes(tmp_path: Path) -> None:
    aggregate, roots, expectations = _sealed_fixture(tmp_path)
    result = audit.audit_final_seal(tmp_path, aggregate, roots, expectations)
    assert result["artifact_set_exact"] is True
    assert result["paper_gate_pass"] is False

    with (aggregate / "aggregation_repair_manifest.json").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(" ")
    with pytest.raises(audit.AuditError, match="artifact hash/size changed"):
        audit.audit_final_seal(tmp_path, aggregate, roots, expectations)


def test_auditor_source_has_no_scientific_runtime_imports() -> None:
    audit.audit_script_has_no_scientific_imports(Path(audit.__file__).resolve())
