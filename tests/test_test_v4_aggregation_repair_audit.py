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


def _tree_payload(root: Path) -> dict[str, object]:
    files = [
        {"path": str(path.relative_to(root)), **_descriptor(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "root": str(root.resolve()),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "tree_digest": audit.json_digest(files),
    }


def _repair_evidence_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, audit.AuditExpectations]:
    failed = tmp_path / "outputs" / ".test_v4_aggregate.incomplete"
    seed_root = tmp_path / "outputs" / ".test_v4_seed17.incomplete"
    repair = tmp_path / "outputs" / "test_v4_aggregate_repair_v1"
    _write_json(failed / "failure.json", {"sealed": True})
    _write_json(seed_root / "panda" / "combination_complete.json", {"sealed": True})
    repair.mkdir(parents=True)

    source_files = (
        "configs/test_v4_aggregate_repair_v1.yaml",
        "scripts/run_test_v4_aggregate_repair_v1.sh",
        "src/confik/test_v4_locked/aggregate_repair.py",
        "src/confik/test_v4_locked/reporting.py",
        "tests/test_test_v4_aggregate_repair.py",
        "tests/test_test_v4_reporting.py",
    )
    for relative in source_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    source: dict[str, object] = {
        "files": {
            relative: _descriptor(tmp_path / relative) for relative in source_files
        },
        "git_commit": "fixture-commit",
        "git_tree": "fixture-tree",
        "scope_clean": True,
    }
    source["digest"] = audit.json_digest(source)
    roots = {
        "aggregate_failure": _tree_payload(failed),
        "seed17": _tree_payload(seed_root),
    }
    completion_hash = audit.sha256_file(
        seed_root / "panda" / "combination_complete.json"
    )
    prereg = {
        "protocol": "test_v4_aggregate_repair_v1",
        "execution_contract": {
            "aggregation_only": True,
            "automatic_resume_allowed": False,
            "original_outputs_mutation_allowed": False,
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
        },
        "threshold_or_gate_changes": False,
        "statistical_semantics_changed": False,
        "old_test_performance_used_for_selection": False,
        "original_failure_classification_changed": False,
        "control_plane": {
            "preregistration_sha256": audit.PRODUCTION.preregistration_sha256,
            "dataset_manifest_sha256": audit.PRODUCTION.dataset_manifest_sha256,
            "control_plane_seal_sha256": audit.PRODUCTION.control_plane_seal_sha256,
            "original_evidence_fingerprint_digest": audit.PRODUCTION.evidence_fingerprint_digest,
        },
        "expected_failure": {
            "exception_message": "panda confirmatory metrics changed",
            "failure_classification": "non_resumable_integrity_or_scientific_failure",
            "phase": {"phase": "aggregate_and_final_integrity"},
        },
        "repair_source_manifest": source,
        "input_tree_digests": {
            label: {
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "tree_digest": payload["tree_digest"],
            }
            for label, payload in roots.items()
        },
        "combination_validations": [
            {
                "robot": "panda",
                "training_seed": 17,
                "completion_manifest_sha256": completion_hash,
                "checkpoint_count": 1,
                "all_artifact_hashes_verified": True,
                "all_checkpoint_quiet_host_contracts_verified": True,
            }
        ],
    }
    prereg_path = repair / "aggregation_repair_preregistration.json"
    _write_json(prereg_path, prereg)
    input_manifest = {
        "protocol": "test_v4_aggregate_repair_v1_input_manifest",
        "repair_preregistration_sha256": audit.sha256_file(prereg_path),
        "all_six_combinations_hash_validated": True,
        "query_record_files_hash_validated_only": True,
        "query_records_read_for_aggregation": False,
        "roots": roots,
        "combined_tree_digest": audit.json_digest(roots),
    }
    input_path = repair / "aggregation_repair_input_manifest.json"
    _write_json(input_path, input_manifest)
    integrity = {
        "protocol": "test_v4_aggregate_repair_v1_integrity",
        "input_trees_before": roots,
        "input_trees_after": roots,
        "input_trees_unchanged": True,
        "protected_tree_before": {
            "tree_digest": audit.PRODUCTION.protected_tree_digest
        },
        "protected_tree_after": {"tree_digest": audit.PRODUCTION.protected_tree_digest},
        "protected_tree_unchanged": True,
        "original_failure_evidence_preserved": True,
        "original_failure_classification_changed": False,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "all_six_combinations_hash_validated": True,
        "final_input_recheck_digest": "fixture-recheck",
    }
    integrity_path = repair / "aggregation_repair_integrity.json"
    _write_json(integrity_path, integrity)
    expectations = replace(
        audit.PRODUCTION,
        robots=("panda",),
        seeds=(17,),
        completion_sha256={"panda/seed17": completion_hash},
        checkpoint_count_per_combination=1,
        repair_git_commit="fixture-commit",
        repair_git_tree="fixture-tree",
        repair_output_sha256={
            prereg_path.name: audit.sha256_file(prereg_path),
            input_path.name: audit.sha256_file(input_path),
            integrity_path.name: audit.sha256_file(integrity_path),
        },
    )
    return repair, failed, expectations


def test_repair_evidence_requires_zero_scientific_activity_and_unchanged_inputs(
    tmp_path: Path,
) -> None:
    repair, failed, expectations = _repair_evidence_fixture(tmp_path)
    result = audit.audit_repair_manifest(
        tmp_path,
        repair,
        failed,
        expectations,
        verify_git_lineage=False,
    )
    assert result["all_input_trees_unchanged"] is True
    assert result["solver_invocation_count"] == 0

    integrity_path = repair / "aggregation_repair_integrity.json"
    integrity = audit.load_json(integrity_path)
    integrity["solver_invocation_count"] = 1
    _write_json(integrity_path, integrity)
    expectations = replace(
        expectations,
        repair_output_sha256={
            **expectations.repair_output_sha256,
            integrity_path.name: audit.sha256_file(integrity_path),
        },
    )
    with pytest.raises(audit.AuditError, match="scientific execution"):
        audit.audit_repair_manifest(
            tmp_path,
            repair,
            failed,
            expectations,
            verify_git_lineage=False,
        )


def test_repair_evidence_detects_input_tree_tampering(tmp_path: Path) -> None:
    repair, failed, expectations = _repair_evidence_fixture(tmp_path)
    with (failed / "failure.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(audit.AuditError, match="artifact hash/size changed"):
        audit.audit_repair_manifest(
            tmp_path,
            repair,
            failed,
            expectations,
            verify_git_lineage=False,
        )


def _sealed_fixture(
    tmp_path: Path, *, paper_gate: bool
) -> tuple[Path, audit.AuditExpectations]:
    aggregate = tmp_path / "outputs" / "test_v4_aggregate_repair_v1"
    names = (
        "aggregate_summary_v4.json",
        "aggregation_repair_input_manifest.json",
        "aggregation_repair_integrity.json",
        "aggregation_repair_preregistration.json",
        "joint_holm_v4.json",
        "paper_gate_v4.json",
    )
    for name in names:
        _write_json(aggregate / name, {"fixture": name})
    chain = {name: audit.sha256_file(aggregate / name) for name in names}
    final = {
        "protocol": "test_v4_aggregate_repair_v1_final_manifest",
        "authoritative_output_namespace": "outputs/test_v4_aggregate_repair_v1",
        "aggregation_only_repair": True,
        "six_combination_natural_exits": True,
        "original_formal_runner_natural_exit": False,
        "original_incomplete_paths_promoted_or_renamed": False,
        "original_failure_evidence_preserved": True,
        "original_failure_classification_changed": False,
        "threshold_or_statistical_semantics_changed": False,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "paper_gate_pass": paper_gate,
        "final_input_recheck_digest": "fixture-recheck",
        "hash_chain": chain,
        "hash_chain_digest": audit.json_digest(chain),
    }
    final["manifest_payload_digest"] = audit.json_digest(final)
    final_path = aggregate / "test_v4_repair_final_manifest.json"
    _write_json(final_path, final)
    expectations = replace(
        audit.PRODUCTION,
        repair_output_sha256={
            **chain,
            final_path.name: audit.sha256_file(final_path),
        },
    )
    return aggregate, expectations


def test_final_seal_requires_exact_seven_file_hash_chain_without_result_direction(
    tmp_path: Path,
) -> None:
    aggregate, expectations = _sealed_fixture(tmp_path, paper_gate=True)
    result = audit.audit_final_seal(
        tmp_path,
        aggregate,
        expectations,
        expected_paper_gate=True,
        expected_final_input_recheck_digest="fixture-recheck",
    )
    assert result["artifact_set_exact"] is True
    assert result["paper_gate_pass"] is True

    with (aggregate / "aggregate_summary_v4.json").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(" ")
    with pytest.raises(audit.AuditError, match="sealed repair artifact bytes changed"):
        audit.audit_final_seal(
            tmp_path,
            aggregate,
            expectations,
            expected_paper_gate=True,
            expected_final_input_recheck_digest="fixture-recheck",
        )


def _attestation_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    audit.AuditExpectations,
    dict[str, object],
    dict[str, object],
]:
    repair, expectations = _sealed_fixture(tmp_path, paper_gate=True)
    source_relatives = (
        "configs/test_v4_aggregate_repair_v1.yaml",
        "scripts/run_test_v4_aggregate_repair_v1.sh",
        "src/confik/test_v4_locked/aggregate_repair.py",
        "src/confik/test_v4_locked/reporting.py",
        "tests/test_test_v4_aggregate_repair.py",
        "tests/test_test_v4_reporting.py",
    )
    for relative in source_relatives:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    source_files = {
        relative: _descriptor(tmp_path / relative) for relative in source_relatives
    }
    source_manifest: dict[str, object] = {
        "files": source_files,
        "git_commit": "repair-commit",
        "git_tree": "repair-tree",
        "scope_clean": True,
    }
    source_manifest["digest"] = audit.json_digest(source_manifest)
    _write_json(
        repair / "aggregation_repair_preregistration.json",
        {"repair_source_manifest": source_manifest},
    )
    protected_tree = {
        "directories": ["fixture"],
        "file_count": 1,
        "total_bytes": 1,
        "tree_digest": audit.PRODUCTION.protected_tree_digest,
    }
    _write_json(
        repair / "aggregation_repair_integrity.json",
        {"protected_tree_after": protected_tree},
    )
    _write_json(
        repair / "aggregate_summary_v4.json", {"primary": {}, "sensitivity": {}}
    )
    _write_json(
        repair / "joint_holm_v4.json", {"all_confirmatory_nulls_rejected": True}
    )
    _write_json(
        repair / "paper_gate_v4.json",
        {"robot_gates": {"panda": True, "ur5e": True}, "both_robot_gates_pass": True},
    )
    repair_hashes = {
        path.name: audit.sha256_file(path)
        for path in repair.iterdir()
        if path.is_file()
    }

    attestation = tmp_path / "outputs" / "test_v4_aggregate_repair_v1_attestation_v1"
    generator_relative = "scripts/attestation_generator_fixture.py"
    generator_path = tmp_path / generator_relative
    generator_path.write_text("# generator fixture\n", encoding="utf-8")
    generator_source: dict[str, object] = {
        "git_commit": "generator-commit",
        "git_tree": "generator-tree",
        "git_top_level": str(tmp_path),
        "git_dir": str(tmp_path / ".git"),
        "scope_clean": True,
        "global_cleanliness_asserted": False,
        "files": {generator_relative: _descriptor(generator_path)},
    }
    generator_source["digest"] = audit.json_digest(generator_source)
    execution = {
        "protocol": "test_v4_aggregate_repair_v1_execution_attestation",
        "attestation_timing": "retrospective",
        "original_execution_invocation_source": "operator_and_tool_invocation_record",
        "original_invocation_independently_traced": False,
        "declared_original_invocation": {
            "launcher": "./scripts/run_test_v4_aggregate_repair_v1.sh",
            "git_dir": "/tmp/confik-v4-repair-lineage.ZJRjoy/.git",
            "git_work_tree": str(tmp_path),
            "source_commit": "repair-commit",
            "statement": "Retrospective declaration, not an independent trace.",
        },
        "shadow_git_metadata_used": True,
        "git_work_tree_was_main_workspace": True,
        "scope_clean_interpretation": "Scoped repair lineage only, not global clean.",
        "global_worktree_cleanliness_asserted": False,
        "global_worktree_cleanliness_required_for_this_attestation": False,
        "verifiable_consequences_not_invocation_claim_alone": [
            "source bytes",
            "input hashes",
            "output hashes",
        ],
        "attestation_generator_source": generator_source,
        "scientific_outcome_direction_used_for_acceptance": False,
    }
    _write_json(attestation / "execution_provenance_attestation.json", execution)

    bundle_path = attestation / "repair_v1_source.bundle"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(b"fixture bundle")
    source_verification = {
        "protocol": "test_v4_aggregate_repair_v1_source_verification",
        "main_repository": str(tmp_path),
        "permanent_ref": "refs/heads/fixture-repair",
        "commit": "repair-commit",
        "parent": "repair-parent",
        "tree": "repair-tree",
        "v1_scope_clean_interpretation": "Scoped repair lineage only.",
        "global_worktree_cleanliness_asserted_for_v1": False,
        "source_files": {
            relative: {
                "v1_manifest": descriptor,
                "permanent_commit": descriptor,
                "main_work_tree": descriptor,
                "all_byte_identical": True,
            }
            for relative, descriptor in source_files.items()
        },
        "all_source_files_byte_identical": True,
        "bundle": {
            **_descriptor(bundle_path),
            "path": "repair_v1_source.bundle",
            "header": {
                "signature": "# v2 git bundle",
                "references": {"refs/heads/fixture-repair": "repair-commit"},
                "prerequisites": [],
                "self_contained": True,
            },
            "git_bundle_verify_succeeded": True,
            "git_bundle_verify_output": "fixture verified",
        },
    }
    _write_json(attestation / "source_commit_verification.json", source_verification)

    repair_evidence: dict[str, object] = {
        "all_input_trees_unchanged": True,
        "input_trees": {
            "aggregate_failure": {
                "file_count": 1,
                "total_bytes": 2,
                "tree_digest": "input-tree-a",
            },
            "seed17": {
                "file_count": 1,
                "total_bytes": 3,
                "tree_digest": "input-tree-b",
            },
        },
    }
    v1_files = [
        {
            "path": path.name,
            "sha256": audit.sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(repair.iterdir())
        if path.is_file()
    ]
    integrity = {
        "protocol": "test_v4_aggregate_repair_v1_integrity_reattestation",
        "v1_tree": {
            "file_count": len(v1_files),
            "total_bytes": sum(int(item["size"]) for item in v1_files),
            "tree_digest": audit.json_digest(v1_files),
        },
        "v1_final_manifest": _descriptor(repair / "test_v4_repair_final_manifest.json"),
        "v1_input_manifest": _descriptor(
            repair / "aggregation_repair_input_manifest.json"
        ),
        "v1_integrity_artifact": _descriptor(
            repair / "aggregation_repair_integrity.json"
        ),
        "formal_input_trees": repair_evidence["input_trees"],
        "protected_tree": protected_tree,
        "v1_hash_chain_valid": True,
        "v1_manifest_payload_digest_valid": True,
        "formal_input_trees_match_v1_before_and_after": True,
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
    }
    _write_json(attestation / "v1_integrity_reaudit.json", integrity)

    statistics: dict[str, object] = {
        "robot_gates": {"panda": True, "ur5e": True},
        "all_joint_holm_claims_pass": True,
        "paper_gate_pass": True,
    }
    observed_results = {
        "robot_gates": statistics["robot_gates"],
        "joint_holm_gate_pass": statistics["all_joint_holm_claims_pass"],
        "paper_gate_pass": statistics["paper_gate_pass"],
    }
    semantic_names = (
        "aggregate_summary_v4.json",
        "joint_holm_v4.json",
        "paper_gate_v4.json",
    )
    recomputation = {
        "protocol": "test_v4_aggregate_repair_v1_independent_recomputation",
        "implementation": "Python standard library only; no confik reporting import",
        "bootstrap_resamples_executed": 0,
        "query_records_parsed_or_used_for_recomputation": False,
        "query_record_files_hash_verified_only": True,
        "stored_pvalues_reused": True,
        "familywise_alpha": 0.05,
        "confirmatory_members": list(audit.CONFIRMATORY_METRICS),
        "semantic_matches": {name: True for name in semantic_names},
        "all_semantic_matches": True,
        "stored_artifacts": {
            name: _descriptor(repair / name) for name in semantic_names
        },
        "recomputed_payload_digests": {
            name: audit.json_digest(audit.load_json(repair / name))
            for name in semantic_names
        },
        "observed_results_not_used_as_acceptance_criteria": observed_results,
        "acceptance_rule": "Semantic equality independent of result direction.",
        "outcome_direction_hardcoded": False,
    }
    _write_json(attestation / "independent_recomputation.json", recomputation)

    chain_names = (
        "execution_provenance_attestation.json",
        "source_commit_verification.json",
        "v1_integrity_reaudit.json",
        "independent_recomputation.json",
        "repair_v1_source.bundle",
    )
    chain = {name: audit.sha256_file(attestation / name) for name in chain_names}
    final = {
        "protocol": "test_v4_aggregate_repair_v1_attestation_final",
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
        "hash_chain_digest": audit.json_digest(chain),
        "first_final_recheck_digest": "fixture-recheck",
        "second_pre_promotion_recheck_required": True,
        "automatic_rerun_allowed": False,
    }
    final["manifest_payload_digest"] = audit.json_digest(final)
    final_path = attestation / "attestation_final_manifest.json"
    _write_json(final_path, final)
    expectations = replace(
        expectations,
        repair_output_sha256=repair_hashes,
        repair_git_commit="repair-commit",
        repair_git_tree="repair-tree",
        repair_git_parent="repair-parent",
        repair_git_ref="refs/heads/fixture-repair",
        attestation_namespace="outputs/test_v4_aggregate_repair_v1_attestation_v1",
        attestation_final_manifest_sha256=audit.sha256_file(final_path),
    )
    return attestation, repair, expectations, repair_evidence, statistics


def test_external_attestation_discloses_shadow_git_and_seals_six_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attestation, repair, expectations, repair_evidence, statistics = (
        _attestation_fixture(tmp_path)
    )

    def fake_git(
        workspace: Path, arguments: tuple[str, ...] | list[str], *, binary: bool = False
    ) -> str | bytes:
        del workspace, binary
        if arguments[:2] == ["rev-parse", "generator-commit^{tree}"]:
            return "generator-tree\n"
        if arguments[:2] == ["bundle", "list-heads"]:
            return "repair-commit refs/heads/fixture-repair\n"
        if arguments[:2] == ["bundle", "verify"]:
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(audit, "_git", fake_git)
    monkeypatch.setattr(
        audit,
        "git_file_descriptor",
        lambda workspace, commit, relative: _descriptor(workspace / relative),
    )

    result = audit.audit_external_execution_attestation(
        tmp_path,
        repair,
        attestation,
        expectations,
        repair_evidence=repair_evidence,
        statistics=statistics,
    )

    assert result["verdict"] == "PASS"
    assert result["sealed_file_count"] == 6
    assert result["retrospective_limitations_disclosed"] is True


def test_external_attestation_rejects_global_cleanliness_overclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attestation, _, expectations, _, _ = _attestation_fixture(tmp_path)
    execution_path = attestation / "execution_provenance_attestation.json"
    execution = audit.load_json(execution_path)
    execution["global_worktree_cleanliness_asserted"] = True
    _write_json(execution_path, execution)
    monkeypatch.setattr(
        audit,
        "_git",
        lambda workspace, arguments, binary=False: "generator-tree\n",
    )
    monkeypatch.setattr(
        audit,
        "git_file_descriptor",
        lambda workspace, commit, relative: _descriptor(workspace / relative),
    )
    with pytest.raises(audit.AuditError, match="shadow/global-cleanliness"):
        audit._audit_execution_provenance_attestation(
            tmp_path, attestation, expectations
        )


def test_auditor_source_has_no_scientific_runtime_imports() -> None:
    audit.audit_script_has_no_scientific_imports(Path(audit.__file__).resolve())
