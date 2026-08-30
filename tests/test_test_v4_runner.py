from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml

from confik.test_v4_locked.data import default_comparison_sources
from confik.test_v4_locked.benchmark import PRIMARY_METHODS, SENSITIVITY_METHODS
from confik.test_v4_locked.runner import (
    _aggregate,
    _baseline_availability,
    _classify_failure,
    _formal_asset_paths,
    _safe,
    _tree_snapshot,
    _verify_release,
    _verify_source_urdf_bindings,
    _validate_checkpoint,
    _write_checkpoint,
    _write_json,
)
from confik.test_v4_locked.reporting import CONFIRMATORY_INFERENCE_METRICS
from confik.test_v4_locked.evidence_protocol import (
    ExclusiveRunLock,
    assert_evidence_fingerprint,
)
from confik.test_v4_locked.host_guard import QuietHostTechnicalInterruption


def test_formal_config_locks_seven_primary_and_three_sensitivity_methods() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "test_v4_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert tuple(config["methods"]["primary"]) == PRIMARY_METHODS
    assert tuple(config["methods"]["sensitivity_only"]) == SENSITIVITY_METHODS
    assert config["timing"]["clock"] == "perf_counter_ns"
    assert config["timing"]["interleaved_same_query_methods"]
    assert not config["timing"]["disk_writes_inside_timed_interval"]
    assert config["statistics"]["bootstrap_samples"] == 10_000
    assert config["statistics"]["multiplicity_correction"] == "holm"
    assert config["statistics"]["familywise_alpha"] == 0.05
    quiet = config["runtime"]["quiet_host"]
    assert config["runtime"]["required_environment_variables"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }
    assert quiet["monitor_sample_interval_seconds"] == 0.25
    assert quiet["allowed_persistent_gpu_compute_process_names"] == [
        "/usr/libexec/gnome-remote-desktop-daemon"
    ]
    assert "counterfactual_v4_smoke*" in config["protected_outputs"]
    assert "counterfactual_v4_readiness_smoke*" in config["protected_outputs"]
    assert "czy" in config["protected_outputs"]
    assert len(config["release_lock"]["release_digest"]) == 64
    assert config["claim_gate"]["reject_support_count_min"] == 30
    assert config["claim_gate"]["defer_support_count_min"] == 10
    assert (
        config["claim_gate"][
            "ood_feasible_false_reject_improvement_vs_v3_min"
        ]
        > 0.0
    )


def test_baseline_manifest_does_not_mislabel_trf_as_trac_ik() -> None:
    availability = _baseline_availability()
    assert availability["trf_previous"]["available"]
    assert not availability["trf_previous"]["is_trac_ik"]
    assert not availability["trac_ik"]["available"]
    assert not availability["trac_ik"]["substitution_claimed"]


def test_json_writer_is_atomic_and_converts_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    _write_json(destination, {"finite": 1.0, "nan": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "nan": None,
    }
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert _safe(float("inf")) is None


def test_release_verifier_rejects_a_release_that_authorized_test(tmp_path: Path) -> None:
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps({"files": {}}), encoding="utf-8"
    )
    import hashlib

    digest = hashlib.sha256((tmp_path / "artifact_manifest.json").read_bytes()).hexdigest()
    (tmp_path / "release_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "release_v4_locked",
                "release_status": "sealed",
                "backend": "torchscript_exact_v4",
                "all_six_validation_runtime_equivalence_pass": True,
                "formal_test_authorized_or_started": True,
                "test_v4_started": False,
                "artifact_manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "release_equivalence.json").write_text(
        json.dumps({"all_pass": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not eligible"):
        _verify_release(tmp_path)


def test_release_verifier_recomputes_frozen_digest_and_control_anchor(
    tmp_path: Path,
) -> None:
    equivalence = tmp_path / "release_equivalence.json"
    dependencies = tmp_path / "upstream_dependencies.json"
    equivalence.write_text(json.dumps({"all_pass": True}), encoding="utf-8")
    candidate_digest = "a" * 64
    v3_hash = "b" * 64
    dependencies.write_text(
        json.dumps(
            {
                "candidate": {"release_digest": candidate_digest},
                "release_v3_locked": {"release_manifest_sha256": v3_hash},
            }
        ),
        encoding="utf-8",
    )
    files = {
        path.name: {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in (equivalence, dependencies)
    }
    artifact_path = tmp_path / "artifact_manifest.json"
    artifact_path.write_text(
        json.dumps({"files": files, "file_count": len(files)}), encoding="utf-8"
    )
    artifact_hash = sha256(artifact_path.read_bytes()).hexdigest()
    release_digest = sha256(
        (artifact_hash + candidate_digest + v3_hash).encode("ascii")
    ).hexdigest()
    release_path = tmp_path / "release_manifest.json"
    release_payload = {
        "protocol": "release_v4_locked",
        "release_status": "sealed",
        "backend": "torchscript_exact_v4",
        "all_six_validation_runtime_equivalence_pass": True,
        "formal_test_authorized_or_started": False,
        "test_v4_started": False,
        "artifact_manifest_sha256": artifact_hash,
        "candidate_release_digest": candidate_digest,
        "upstream_v3_release_manifest_sha256": v3_hash,
        "release_digest": release_digest,
    }
    release_path.write_text(json.dumps(release_payload), encoding="utf-8")
    lock = {
        "release_digest": release_digest,
        "release_manifest_sha256": sha256(release_path.read_bytes()).hexdigest(),
        "artifact_manifest_sha256": artifact_hash,
        "upstream_v3_release_manifest_sha256": v3_hash,
    }
    _verify_release(tmp_path, lock)
    release_payload["release_digest"] = "0" * 64
    release_path.write_text(json.dumps(release_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="packaging formula"):
        _verify_release(tmp_path)


def test_o_excl_global_lock_and_stale_resume_recovery(tmp_path: Path) -> None:
    path = tmp_path / ".formal.lock"
    first = ExclusiveRunLock(path, resume=False)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already exists"):
            ExclusiveRunLock(path, resume=False).acquire()
    finally:
        first.release()
    path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    resumed = ExclusiveRunLock(path, resume=True)
    payload = resumed.acquire()
    try:
        assert payload["stale_lock_archive"]
        assert Path(payload["stale_lock_archive"]).is_file()
        resumed.bind_control_plane("a" * 64)
        assert json.loads(path.read_text(encoding="utf-8"))[
            "control_plane_seal_sha256"
        ] == "a" * 64
    finally:
        resumed.release()


def test_checkpoint_is_atomic_hash_validated_and_not_recomputed(tmp_path: Path) -> None:
    expected = {"protocol": "unit", "checkpoint_key": "chunk0"}
    records = [{"query": 0, "method": "fixed"}]
    path = _write_checkpoint(
        root=tmp_path,
        key="chunk0",
        records=records,
        expected=expected,
        quiet_host_evidence={"background_monitor": True},
    )
    assert path == _validate_checkpoint(tmp_path / "chunk0", expected)
    original_hash = sha256(path.read_bytes()).hexdigest()
    second = _write_checkpoint(
        root=tmp_path,
        key="chunk0",
        records=[{"query": 999}],
        expected=expected,
        quiet_host_evidence={"background_monitor": True},
    )
    assert second == path
    assert sha256(path.read_bytes()).hexdigest() == original_hash
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checkpoint record hash changed"):
        _validate_checkpoint(tmp_path / "chunk0", expected)


def test_formal_checkpoint_requires_new_monitor_sample_for_every_query(
    tmp_path: Path,
) -> None:
    expected = {
        "protocol": "test_v4_atomic_measurement_checkpoint",
        "robot": "panda",
        "training_seed": 17,
        "role": "id_points",
        "checkpoint_key": "query_000000_000000",
        "checkpoint_unit": "fixed_point_query_chunk",
        "source_indices": [0],
        "source_query_sha256": ["q0"],
        "methods": ["fixed"],
        "expected_query_count": 1,
        "expected_record_count": 1,
        "preregistration_sha256": "p",
        "dataset_manifest_sha256": "d",
        "evidence_fingerprint_digest": "e",
        "quiet_host_config_digest": "h",
        "resume_contract": "completed checkpoint is hash-validated and never recomputed",
    }
    record = {
        "robot": "panda",
        "training_seed": 17,
        "role": "id_points",
        "query_index": 0,
        "method": "fixed",
        "source_query_sha256": "q0",
    }
    environment = {
        "background_monitor": True,
        "synchronous_ps_or_nvidia_smi_per_query": False,
        "contamination_decision_source": "external process state only",
        "latency_or_solver_result_used_for_contamination_decision": False,
        "query_interval_check_count": 1,
        "query_intervals_without_new_monitor_sample": 0,
        "minimum_monitor_samples_since_query_start": 1,
        "quiet_host_config_digest": "h",
    }
    _write_checkpoint(
        root=tmp_path,
        key="query_000000_000000",
        records=[record],
        expected=expected,
        quiet_host_evidence=environment,
    )
    manifest_path = tmp_path / "query_000000_000000" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quiet_host_evidence"]["query_intervals_without_new_monitor_sample"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="interval coverage"):
        _validate_checkpoint(tmp_path / "query_000000_000000", expected)


def test_source_urdf_binding_checks_all_robot_seed_runtime_specs(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    release_root = tmp_path / "release_v3_locked"
    config_root.mkdir()
    urdfs: dict[str, Path] = {}
    for robot in ("panda", "ur5e"):
        urdf = tmp_path / f"{robot}.urdf"
        urdf.write_text(f"<robot name='{robot}'/>", encoding="utf-8")
        urdfs[robot] = urdf
        for seed in (17, 29, 43):
            root = release_root / robot / f"seed{seed}"
            root.mkdir(parents=True)
            (root / "runtime_spec.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "robot": robot,
                        "training_seed": seed,
                        "joint_names": [f"{robot}_joint"],
                        "source_urdf": {
                            "path": str(urdf),
                            "sha256": sha256(urdf.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                root / "normalization_parameters.npz",
                joint_names=np.asarray([f"{robot}_joint"]),
            )
    source_config = config_root / "paper_v2.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "robots": {
                    robot: {
                        "urdf": str(path),
                        "base_link": f"{robot}_base",
                        "end_link": f"{robot}_tool",
                    }
                    for robot, path in urdfs.items()
                }
            }
        ),
        encoding="utf-8",
    )
    evidence = _verify_source_urdf_bindings(
        source_config_path=source_config,
        release_v3_root=release_root,
    )
    assert set(evidence["robots"]) == {"panda", "ur5e"}
    assert evidence["robots"]["panda"]["actual_source_urdf"]["size"] > 0
    assert len(evidence["robots"]["ur5e"]["runtime_specs"]) == 3


def test_protected_tree_supports_workspace_czy_and_smoke_patterns(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    smoke = outputs / "counterfactual_v4_readiness_smoke_r2"
    smoke.mkdir()
    (smoke / "evidence.json").write_text("{}", encoding="utf-8")
    czy = tmp_path / "czy"
    czy.mkdir()
    (czy / "results.csv").write_text("x\n1\n", encoding="utf-8")
    snapshot = _tree_snapshot(outputs, ["counterfactual_v4_readiness_smoke*", "czy"])
    assert "counterfactual_v4_readiness_smoke_r2" in snapshot["directories"]
    assert "workspace/czy" in snapshot["directories"]
    assert snapshot["file_count"] == 2


def test_formal_assets_cover_every_freshness_registry_source() -> None:
    workspace = Path(__file__).resolve().parents[1]
    assets = set(
        _formal_asset_paths(
            workspace=workspace,
            config_path=workspace / "configs" / "test_v4_locked.yaml",
            release_v4_root=workspace / "outputs" / "release_v4_locked",
            release_v3_root=workspace / "outputs" / "release_v3_locked",
        )
    )
    for robot in ("panda", "ur5e"):
        for source in default_comparison_sources(workspace, robot):
            assert source.path.resolve() in assets
            if source.provenance_path is not None:
                assert source.provenance_path.resolve() in assets
    assert (
        workspace / "czy" / "closed_loop_v3_raw_frame_records.csv"
    ).resolve() in assets
    assert (
        workspace
        / "outputs"
        / ".counterfactual_v4_readiness_smoke.incomplete.1313949"
        / "panda"
        / "seed17"
        / "counterfactual_records.jsonl.gz"
    ).resolve() in assets


def test_failure_classification_never_resumes_command_contract_violation() -> None:
    resumable, classification = _classify_failure(
        RuntimeError("formal command contract violation for panda/proposed_v4")
    )
    assert not resumable
    assert classification == "non_resumable_scientific_contract_failure"
    resumable, classification = _classify_failure(
        QuietHostTechnicalInterruption("foreign GPU compute")
    )
    assert resumable
    assert classification.startswith("resumable_external_environment")


def test_joint_holm_result_is_a_required_aggregate_claim_gate(tmp_path: Path) -> None:
    seed_roots = {seed: tmp_path / f"seed{seed}" for seed in (17, 29, 43)}
    for seed, root in seed_roots.items():
        for robot in ("panda", "ur5e"):
            robot_root = root / robot
            robot_root.mkdir(parents=True)
            (robot_root / "summary_v4.json").write_text("{}", encoding="utf-8")
            if seed == 17:
                (robot_root / "claim_gate_v4.json").write_text(
                    json.dumps({"formal_gate_pass": True}), encoding="utf-8"
                )
                (robot_root / "ood_abstention_v4.json").write_text(
                    "{}", encoding="utf-8"
                )
                pvalue = 0.9 if robot == "ur5e" else 0.001
                intervals = {
                    "inference_family": {
                        "members": list(CONFIRMATORY_INFERENCE_METRICS)
                    },
                    "metrics": {
                        name: {"one_sided_unadjusted_p": pvalue}
                        for name in CONFIRMATORY_INFERENCE_METRICS
                    },
                }
                (robot_root / "paired_intervals_v4.json").write_text(
                    json.dumps(intervals), encoding="utf-8"
                )
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    gate = _aggregate(
        workspace=tmp_path,
        aggregate_staging=aggregate,
        seed_roots=seed_roots,
        config={"statistics": {"familywise_alpha": 0.05}},
    )
    assert gate["robot_gates"] == {"panda": True, "ur5e": True}
    assert not gate["joint_holm_gate_pass"]
    assert not gate["both_robot_gates_pass"]
    assert (aggregate / "joint_holm_v4.json").is_file()


def test_evidence_fingerprint_comparison_rejects_any_asset_change() -> None:
    expected = {"digest": "a", "files": {"asset": {"sha256": "1"}}}
    changed = {"digest": "b", "files": {"asset": {"sha256": "2"}}}
    with pytest.raises(RuntimeError, match=r"changed_files=\['asset'\]"):
        assert_evidence_fingerprint(expected, changed, context="unit")
