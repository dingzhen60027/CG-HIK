from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

import confik.test_v4_locked.aggregate_repair as repair
from confik.test_v4_locked.reporting import CONFIRMATORY_INFERENCE_METRICS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _intervals() -> dict[str, object]:
    return {
        "inference_family": {
            "members": list(CONFIRMATORY_INFERENCE_METRICS),
        },
        # Deliberately use serialization order rather than preregistered order.
        "metrics": {
            name: {"one_sided_unadjusted_p": 0.001}
            for name in sorted(CONFIRMATORY_INFERENCE_METRICS)
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    workspace = tmp_path
    outputs = workspace / "outputs"
    configs = workspace / "configs"
    outputs.mkdir()
    configs.mkdir()

    reporting = workspace / "src/confik/test_v4_locked/reporting.py"
    reporting.parent.mkdir(parents=True)
    reporting.write_text("new", encoding="utf-8")
    old_reporting = {"sha256": sha256(b"old").hexdigest(), "size": 3}

    formal_config = configs / "test_v4_locked.yaml"
    formal_config.write_text(yaml.safe_dump({"protected_outputs": []}), encoding="utf-8")

    aggregate = outputs / ".test_v4_aggregate.incomplete"
    aggregate.mkdir()
    protected = repair._protected_tree_snapshot(outputs, [])
    preregistration = {
        "release_digest": "release",
        "runner_git_commit": "original",
        "protected_outputs_before": protected,
        "evidence_fingerprint": {
            "digest": "original-fingerprint",
            "files": {
                "src/confik/test_v4_locked/reporting.py": old_reporting,
            },
        },
    }
    _write_json(aggregate / "test_v4_preregistration.json", preregistration)
    _write_json(aggregate / "test_v4_dataset_manifest.json", {"frozen": True})
    _write_json(aggregate / "test_v4_control_plane_seal.json", {"frozen": True})
    failure = {
        "failure_classification": "non_resumable_integrity_or_scientific_failure",
        "resume_eligible": False,
        "exception_type": "RuntimeError",
        "exception_message": "panda confirmatory metrics changed",
        "phase": {"phase": "aggregate_and_final_integrity"},
    }
    _write_json(aggregate / "latest_failure_manifest.json", failure)
    _write_json(aggregate / "failure_manifests/failure.json", failure)

    seed_roots: dict[int, Path] = {}
    completion_hashes: dict[int, dict[str, str]] = {}
    for seed in repair.SEEDS:
        seed_root = outputs / f".test_v4_seed{seed}.incomplete"
        seed_roots[seed] = seed_root
        completion_hashes[seed] = {}
        for robot in repair.ROBOTS:
            root = seed_root / robot
            root.mkdir(parents=True)
            summary = {"record_count": 175_000 if seed == 17 else 75_000}
            _write_json(root / "summary_v4.json", summary)
            if seed == repair.PRIMARY_SEED:
                _write_json(root / "claim_gate_v4.json", {"formal_gate_pass": True})
                _write_json(root / "ood_abstention_v4.json", {"available": True})
                _write_json(root / "paired_intervals_v4.json", _intervals())
            artifacts = {
                str(path.relative_to(root)): {
                    "sha256": _sha(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            completion = {
                "protocol": (
                    "test_v4_combination_complete_but_not_formal_completion_marker"
                ),
                "robot": robot,
                "training_seed": seed,
                "all_checkpoints_hash_validated": True,
                "eligible_without_aggregate_final_manifest": False,
                "artifacts": artifacts,
            }
            completion_path = root / "combination_complete.json"
            _write_json(completion_path, completion)
            completion_hashes[seed][robot] = _sha(completion_path)

    aggregate_snapshot = repair._tree_snapshot(aggregate)
    seed_snapshots = {
        seed: repair._tree_snapshot(seed_roots[seed]) for seed in repair.SEEDS
    }
    control_paths = {
        "preregistration": aggregate / "test_v4_preregistration.json",
        "dataset_manifest": aggregate / "test_v4_dataset_manifest.json",
        "control_plane_seal": aggregate / "test_v4_control_plane_seal.json",
        "latest_failure_manifest": aggregate / "latest_failure_manifest.json",
    }
    config = {
        "protocol_version": repair.PROTOCOL,
        "inputs": {
            "formal_config_path": "test_v4_locked.yaml",
            "formal_config_sha256": _sha(formal_config),
            "aggregate_failure": {
                "path": "../outputs/.test_v4_aggregate.incomplete",
                "file_count": aggregate_snapshot["file_count"],
                "tree_digest": aggregate_snapshot["tree_digest"],
            },
            "seed_roots": {
                seed: {
                    "path": f"../outputs/.test_v4_seed{seed}.incomplete",
                    "file_count": seed_snapshots[seed]["file_count"],
                    "tree_digest": seed_snapshots[seed]["tree_digest"],
                    "combination_complete_sha256": completion_hashes[seed],
                }
                for seed in repair.SEEDS
            },
            "expected_checkpoint_count_per_combination": 0,
            "control_plane": {
                **{
                    f"{name}_sha256": _sha(path)
                    for name, path in control_paths.items()
                },
                "release_digest": "release",
                "original_evidence_fingerprint_digest": "original-fingerprint",
            },
            "expected_failure": failure,
        },
        "statistics": {
            "familywise_alpha": 0.05,
            "multiplicity_correction": "Holm",
            "hypothesis_count": 8,
        },
        "execution_contract": {
            "aggregation_only": True,
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
            "original_outputs_mutation_allowed": False,
            "automatic_resume_allowed": False,
        },
        "output": {
            "staging_directory": "../outputs/.test_v4_aggregate_repair_v1.incomplete",
            "final_directory": "../outputs/test_v4_aggregate_repair_v1",
            "lock_path": "../outputs/.test_v4_aggregate_repair_v1.lock",
        },
    }
    config_path = configs / "test_v4_aggregate_repair_v1.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, {
        "aggregate": aggregate,
        **{f"seed{seed}": path for seed, path in seed_roots.items()},
    }


def _source_override(*args: object, **kwargs: object) -> dict[str, object]:
    return {
        "git_commit": "unit",
        "git_tree": "unit",
        "scope_clean": True,
        "files": {},
        "digest": "unit",
    }


def _patch_source_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repair, "_repair_source_manifest", _source_override)
    monkeypatch.setattr(
        repair,
        "_repair_git_diff",
        lambda workspace, original_commit: {
            key: set(value) for key, value in repair.EXPECTED_REPAIR_DIFF.items()
        },
    )


def test_repair_is_structurally_forbidden_from_importing_execution_modules() -> None:
    source = Path(repair.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = ("benchmark", "runner", "models", "solver", "torch")
    assert not any(any(token in name for token in forbidden) for name in imported)


def test_joint_holm_membership_survives_sorted_json_object_round_trip() -> None:
    def interval(reverse: bool) -> dict[str, object]:
        pairs = list(
            zip(
                CONFIRMATORY_INFERENCE_METRICS,
                (0.001, 0.002, 0.003, 0.004),
                strict=True,
            )
        )
        if reverse:
            pairs.reverse()
        return {
            "inference_family": {"members": list(CONFIRMATORY_INFERENCE_METRICS)},
            "metrics": {
                name: {"one_sided_unadjusted_p": value} for name, value in pairs
            },
        }

    canonical = {"panda": interval(False), "ur5e": interval(False)}
    sorted_round_trip = json.loads(json.dumps(canonical, sort_keys=True))
    reversed_mapping = {"panda": interval(True), "ur5e": interval(True)}
    expected = repair.joint_holm_confirmatory(canonical, alpha=0.05)
    assert repair.joint_holm_confirmatory(sorted_round_trip, alpha=0.05) == expected
    assert repair.joint_holm_confirmatory(reversed_mapping, alpha=0.05) == expected


def test_joint_holm_still_rejects_missing_or_extra_metric_members() -> None:
    metrics = {
        name: {"one_sided_unadjusted_p": 0.001}
        for name in CONFIRMATORY_INFERENCE_METRICS
    }
    interval = {
        "inference_family": {"members": list(CONFIRMATORY_INFERENCE_METRICS)},
        "metrics": metrics,
    }
    missing = json.loads(json.dumps(interval))
    missing["metrics"].pop(CONFIRMATORY_INFERENCE_METRICS[0])
    with pytest.raises(RuntimeError, match="confirmatory metrics changed"):
        repair.joint_holm_confirmatory({"panda": missing, "ur5e": interval})
    extra = json.loads(json.dumps(interval))
    extra["metrics"]["unregistered_metric"] = {
        "one_sided_unadjusted_p": 0.001
    }
    with pytest.raises(RuntimeError, match="confirmatory metrics changed"):
        repair.joint_holm_confirmatory({"panda": interval, "ur5e": extra})


def test_aggregation_only_repair_preserves_every_original_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from confik.models.risk import RiskModel
    from confik.models.seed import TorchSeedEnsemble
    from confik.solvers.dls import AdaptiveDLS
    from confik.solvers.fallback import TRFFallbackSolver
    from confik.test_v4_locked import benchmark

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("aggregation repair entered a forbidden execution path")

    monkeypatch.setattr(benchmark, "_solve_once", forbidden)
    monkeypatch.setattr(benchmark, "benchmark_role", forbidden)
    monkeypatch.setattr(benchmark, "warmup_methods", forbidden)
    monkeypatch.setattr(AdaptiveDLS, "solve", forbidden)
    monkeypatch.setattr(TRFFallbackSolver, "solve", forbidden)
    monkeypatch.setattr(TorchSeedEnsemble, "candidates", forbidden)
    monkeypatch.setattr(RiskModel, "predict", forbidden)
    _patch_source_audit(monkeypatch)
    config_path, roots = _fixture(tmp_path)
    before = {name: repair._tree_snapshot(path) for name, path in roots.items()}
    result = repair._execute_repair(
        config_path,
        workspace=tmp_path,
    )
    after = {name: repair._tree_snapshot(path) for name, path in roots.items()}
    assert before == after
    assert result["aggregation_only_repair"]
    assert result["query_rerun_count"] == 0
    assert result["solver_invocation_count"] == 0
    assert result["model_inference_count"] == 0
    final = tmp_path / "outputs/test_v4_aggregate_repair_v1"
    assert final.is_dir()
    assert not (tmp_path / "outputs/.test_v4_aggregate_repair_v1.incomplete").exists()
    manifest = json.loads(
        (final / "test_v4_repair_final_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["hash_chain"].items():
        assert _sha(final / name) == expected
    assert manifest["original_formal_runner_natural_exit"] is False
    assert manifest["original_failure_evidence_preserved"] is True

    with pytest.raises(RuntimeError, match="rerun is forbidden"):
        repair._execute_repair(
            config_path,
            workspace=tmp_path,
        )


def test_repair_rejects_any_change_to_a_sealed_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source_audit(monkeypatch)
    config_path, roots = _fixture(tmp_path)
    summary = roots["seed17"] / "panda/summary_v4.json"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed repair input tree changed"):
        repair._execute_repair(
            config_path,
            workspace=tmp_path,
        )
    assert not (tmp_path / "outputs/test_v4_aggregate_repair_v1").exists()
    assert not (tmp_path / "outputs/.test_v4_aggregate_repair_v1.incomplete").exists()


def test_repair_lock_is_independent_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source_audit(monkeypatch)
    config_path, _ = _fixture(tmp_path)
    lock = tmp_path / "outputs/.test_v4_aggregate_repair_v1.lock"
    lock.write_text("held", encoding="utf-8")
    with pytest.raises(RuntimeError, match="repair lock already exists"):
        repair._execute_repair(
            config_path,
            workspace=tmp_path,
        )


def test_repair_rechecks_inputs_after_final_manifest_before_atomic_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source_audit(monkeypatch)
    config_path, roots = _fixture(tmp_path)
    original_writer = repair._write_json_new

    def mutate_after_final_manifest(path: Path, payload: object) -> None:
        original_writer(path, payload)  # type: ignore[arg-type]
        if path.name == "test_v4_repair_final_manifest.json":
            source = roots["seed17"] / "panda/summary_v4.json"
            source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")

    monkeypatch.setattr(repair, "_write_json_new", mutate_after_final_manifest)
    with pytest.raises(RuntimeError, match="input trees changed"):
        repair._execute_repair(config_path, workspace=tmp_path)
    assert not (tmp_path / "outputs/test_v4_aggregate_repair_v1").exists()
    staging = tmp_path / "outputs/.test_v4_aggregate_repair_v1.incomplete"
    assert staging.is_dir()
    failure = json.loads(
        (staging / "aggregation_repair_failure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["automatic_resume_allowed"] is False
