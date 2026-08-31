from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

import confik.test_v4_locked.aggregate_repair_attestation as attestation


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(workspace: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=workspace, text=True, stderr=subprocess.STDOUT
    ).strip()


def _commit(workspace: Path, message: str) -> str:
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", message)
    return _git(workspace, "rev-parse", "HEAD")


def _descriptor_from_commit(workspace: Path, commit: str, relative: str) -> dict[str, object]:
    payload = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=workspace
    )
    return {"sha256": sha256(payload).hexdigest(), "size": len(payload)}


def _intervals(offset: float = 0.0) -> dict[str, object]:
    return {
        "inference_family": {"members": list(attestation.CONFIRMATORY_METRICS)},
        "metrics": {
            name: {"one_sided_unadjusted_p": 0.001 * (index + 1) + offset}
            for index, name in enumerate(attestation.CONFIRMATORY_METRICS)
        },
    }


def _stored_aggregation(seed_roots: dict[int, Path]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    unadjusted: dict[str, float] = {}
    primary: dict[str, object] = {}
    sensitivity: dict[str, object] = {}
    for robot in attestation.ROBOTS:
        intervals = json.loads(
            (seed_roots[17] / robot / "paired_intervals_v4.json").read_text()
        )
        for metric in attestation.CONFIRMATORY_METRICS:
            unadjusted[f"{robot}/{metric}"] = float(
                intervals["metrics"][metric]["one_sided_unadjusted_p"]
            )
        root = seed_roots[17] / robot
        primary[robot] = {
            "claim_gate": json.loads((root / "claim_gate_v4.json").read_text()),
            "summary": json.loads((root / "summary_v4.json").read_text()),
            "ood_abstention": json.loads(
                (root / "ood_abstention_v4.json").read_text()
            ),
        }
        sensitivity[robot] = {
            f"seed{seed}": json.loads(
                (seed_roots[seed] / robot / "summary_v4.json").read_text()
            )
            for seed in attestation.SENSITIVITY_SEEDS
        }
    adjusted = attestation._holm(unadjusted)
    hypotheses = {
        name: {
            "robot": name.split("/", 1)[0],
            "metric": name.split("/", 1)[1],
            "one_sided_unadjusted_p": unadjusted[name],
            "holm_adjusted_p": adjusted[name],
            "reject_margin_null": adjusted[name] <= 0.05,
        }
        for name in unadjusted
    }
    joint = {
        "method": "Holm",
        "scope": "Panda and UR5e x four prespecified confirmatory claims",
        "alpha": 0.05,
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "all_confirmatory_nulls_rejected": all(
            bool(value["reject_margin_null"]) for value in hypotheses.values()
        ),
        "operational_finite_test_gates_included": False,
    }
    robot_gates = {
        robot: bool(primary[robot]["claim_gate"]["formal_gate_pass"])
        for robot in attestation.ROBOTS
    }
    paper = {
        "protocol": "test_v4 robot-level confirmatory aggregation",
        "primary_training_seed": 17,
        "sensitivity_training_seeds": [29, 43],
        "sensitivity_seeds_are_not_independent_query_samples": True,
        "robot_gates": robot_gates,
        "robot_gates_are_pre_joint_holm": True,
        "joint_holm_gate_pass": bool(joint["all_confirmatory_nulls_rejected"]),
        "joint_holm_is_required_for_formal_gate": True,
        "both_robot_gates_pass": all(robot_gates.values())
        and bool(joint["all_confirmatory_nulls_rejected"]),
        "test_set_retuning_performed": False,
    }
    return joint, paper, {"primary": primary, "sensitivity": sensitivity}


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], str]:
    workspace = tmp_path
    (workspace / "outputs").mkdir()
    (workspace / "configs").mkdir()
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "attestation@example.invalid")
    _git(workspace, "config", "user.name", "Attestation Test")

    reporting = workspace / "src/confik/test_v4_locked/reporting.py"
    reporting.parent.mkdir(parents=True)
    reporting.write_text("old reporting\n", encoding="utf-8")
    test_reporting = workspace / "tests/test_test_v4_reporting.py"
    test_reporting.parent.mkdir(parents=True)
    test_reporting.write_text("unchanged reporting test\n", encoding="utf-8")
    base = _commit(workspace, "formal base")

    reporting.write_text("repaired reporting\n", encoding="utf-8")
    for relative in (
        "src/confik/test_v4_locked/aggregate_repair.py",
        "configs/test_v4_aggregate_repair_v1.yaml",
        "scripts/run_test_v4_aggregate_repair_v1.sh",
        "tests/test_test_v4_aggregate_repair.py",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"repair file {relative}\n", encoding="utf-8")
    repair_commit = _commit(workspace, "repair source")
    permanent_ref = "refs/heads/test-repair-source"
    _git(workspace, "branch", "test-repair-source", repair_commit)

    aggregate = workspace / "outputs/.test_v4_aggregate.incomplete"
    aggregate.mkdir()
    _write_json(
        aggregate / "latest_failure_manifest.json",
        {
            "failure_classification": "non_resumable_integrity_or_scientific_failure",
            "resume_eligible": False,
        },
    )
    seed_roots: dict[int, Path] = {}
    for seed in attestation.SEEDS:
        root = workspace / f"outputs/.test_v4_seed{seed}.incomplete"
        seed_roots[seed] = root
        for robot in attestation.ROBOTS:
            robot_root = root / robot
            robot_root.mkdir(parents=True)
            _write_json(
                robot_root / "summary_v4.json",
                {"robot": robot, "seed": seed, "value": seed},
            )
            if seed == 17:
                _write_json(
                    robot_root / "claim_gate_v4.json",
                    {"formal_gate_pass": robot == "ur5e"},
                )
                _write_json(
                    robot_root / "ood_abstention_v4.json", {"robot": robot}
                )
                _write_json(
                    robot_root / "paired_intervals_v4.json",
                    _intervals(0.0 if robot == "panda" else 0.0001),
                )

    formal_roots = {
        "aggregate_failure": aggregate,
        **{f"seed{seed}": root for seed, root in seed_roots.items()},
    }
    original_snapshots = {
        name: attestation._tree_snapshot(path) for name, path in formal_roots.items()
    }
    protected = attestation._protected_tree_snapshot(
        workspace, workspace / "outputs", []
    )
    source_files = {
        relative: _descriptor_from_commit(workspace, repair_commit, relative)
        for relative in attestation.EXPECTED_V1_SOURCE_SCOPE
    }
    source_manifest: dict[str, object] = {
        "git_commit": repair_commit,
        "git_tree": _git(workspace, "rev-parse", f"{repair_commit}^{{tree}}"),
        "scope_clean": True,
        "files": source_files,
    }
    source_manifest["digest"] = attestation._json_digest(source_manifest)

    v1 = workspace / "outputs/test_v4_aggregate_repair_v1"
    v1.mkdir()
    prereg = {
        "protocol": "test_v4_aggregate_repair_v1",
        "repair_source_manifest": source_manifest,
        "execution_contract": {
            "query_rerun_count": 0,
            "solver_invocation_count": 0,
            "model_inference_count": 0,
        },
    }
    prereg_path = v1 / "aggregation_repair_preregistration.json"
    _write_json(prereg_path, prereg)
    input_manifest = {
        "protocol": "test_v4_aggregate_repair_v1_input_manifest",
        "repair_preregistration_sha256": attestation._sha256_file(prereg_path),
        "roots": original_snapshots,
        "combined_tree_digest": attestation._json_digest(original_snapshots),
    }
    _write_json(v1 / "aggregation_repair_input_manifest.json", input_manifest)
    integrity = {
        "protocol": "test_v4_aggregate_repair_v1_integrity",
        "input_trees_before": original_snapshots,
        "input_trees_after": original_snapshots,
        "input_trees_unchanged": True,
        "protected_tree_before": protected,
        "protected_tree_after": protected,
        "protected_tree_unchanged": True,
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "original_failure_evidence_preserved": True,
        "original_failure_classification_changed": False,
    }
    _write_json(v1 / "aggregation_repair_integrity.json", integrity)
    joint, paper, summary = _stored_aggregation(seed_roots)
    _write_json(v1 / "joint_holm_v4.json", joint)
    _write_json(v1 / "paper_gate_v4.json", paper)
    _write_json(v1 / "aggregate_summary_v4.json", summary)
    chain = {
        path.name: attestation._sha256_file(path)
        for path in sorted(v1.iterdir())
        if path.is_file()
    }
    final: dict[str, object] = {
        "protocol": "test_v4_aggregate_repair_v1_final_manifest",
        "hash_chain": chain,
        "hash_chain_digest": attestation._json_digest(chain),
        "query_rerun_count": 0,
        "solver_invocation_count": 0,
        "model_inference_count": 0,
        "original_failure_evidence_preserved": True,
        "original_failure_classification_changed": False,
        "threshold_or_statistical_semantics_changed": False,
    }
    final["manifest_payload_digest"] = attestation._json_digest(final)
    final_path = v1 / "test_v4_repair_final_manifest.json"
    _write_json(final_path, final)
    v1_snapshot = attestation._tree_snapshot(v1)

    config = {
        "protocol_version": attestation.PROTOCOL,
        "execution_contract": {
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
        },
        "execution_disclosure": {
            "attestation_timing": "retrospective",
            "invocation_claim_source": "operator_and_tool_invocation_record",
            "invocation_was_independently_traced": False,
            "shadow_git_metadata_used": True,
            "scope_clean_only": True,
            "global_worktree_clean_claim": False,
            "permanent_ref_import_timing": "retrospective_after_v1_execution",
            "declared_original_invocation": {
                "git_dir": "/tmp/shadow/.git",
                "git_work_tree": str(workspace),
            },
        },
        "statistics": {
            "familywise_alpha": 0.05,
            "multiplicity_correction": "Holm",
            "hypothesis_count": 8,
            "confirmatory_metrics": list(attestation.CONFIRMATORY_METRICS),
        },
        "source_commit": {
            "main_repository": str(workspace),
            "permanent_ref": permanent_ref,
            "commit": repair_commit,
            "parent": base,
            "tree": _git(workspace, "rev-parse", f"{repair_commit}^{{tree}}"),
        },
        "v1": {
            "root": "../outputs/test_v4_aggregate_repair_v1",
            "tree": attestation._tree_anchor(v1_snapshot),
            "final_manifest_sha256": attestation._sha256_file(final_path),
        },
        "inputs": {
            "v1_root": "../outputs/test_v4_aggregate_repair_v1",
            "formal_roots": {
                name: {
                    "path": f"../outputs/{path.name}",
                    **attestation._tree_anchor(original_snapshots[name]),
                }
                for name, path in formal_roots.items()
            },
            "protected_patterns": [],
            "protected_tree": protected,
        },
        "output": {
            "staging_directory": "../outputs/.test_v4_aggregate_repair_v1_attestation_v1.incomplete",
            "final_directory": "../outputs/test_v4_aggregate_repair_v1_attestation_v1",
            "lock_path": "../outputs/.test_v4_aggregate_repair_v1_attestation_v1.lock",
        },
    }
    config_path = workspace / "configs/test_v4_aggregate_repair_v1_attestation_v1.json"
    _write_json(config_path, config)
    for relative in (
        "src/confik/test_v4_locked/aggregate_repair_attestation.py",
        "scripts/run_test_v4_aggregate_repair_v1_attestation_v1.sh",
        "tests/test_test_v4_aggregate_repair_attestation.py",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"attestation source {relative}\n", encoding="utf-8")
    _commit(workspace, "attestation source")
    return config_path, {"v1": v1, **formal_roots}, repair_commit


def test_attestation_module_has_no_scientific_runtime_imports() -> None:
    tree = ast.parse(Path(attestation.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"numpy", "scipy", "sklearn", "torch", "pinocchio"}
    assert not any(name.split(".", 1)[0] in forbidden for name in imported)
    assert not any("reporting" in name or "benchmark" in name for name in imported)


def test_direct_isolated_script_process_loads_no_scientific_runtime(
    tmp_path: Path,
) -> None:
    script = Path(attestation.__file__).resolve()
    malicious = tmp_path / "confik/__init__.py"
    malicious.parent.mkdir(parents=True)
    malicious.write_text("raise RuntimeError('cwd package imported')\n", encoding="utf-8")
    probe = """
import runpy
import sys
script_path = sys.argv[1]
sys.argv = [script_path, '--help']
try:
    runpy.run_path(script_path, run_name='__main__')
except SystemExit as error:
    if error.code != 0:
        raise
forbidden = ('confik', 'numpy', 'scipy', 'sklearn', 'torch', 'pinocchio')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
if loaded:
    raise RuntimeError(f'forbidden scientific runtime loaded: {loaded}')
print('STANDARD_LIBRARY_ONLY')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "STANDARD_LIBRARY_ONLY" in result.stdout
    launcher = (
        script.parents[3]
        / "scripts/run_test_v4_aggregate_repair_v1_attestation_v1.sh"
    ).read_text(encoding="utf-8")
    assert " -m " not in launcher
    assert "PYTHONPATH" not in launcher
    assert '"${PYTHON_BIN}" -I' in launcher


def test_standard_library_holm_is_stable_and_direction_agnostic() -> None:
    first = {"b": 0.01, "a": 0.01, "c": 0.03}
    assert attestation._holm(first) == {"b": 0.03, "a": 0.03, "c": 0.03}
    assert attestation._holm({"x": 0.9, "y": 0.8}) == {"y": 1.0, "x": 1.0}


def test_attestation_is_atomic_read_only_and_self_contained(tmp_path: Path) -> None:
    config_path, roots, repair_commit = _fixture(tmp_path)
    before = {name: attestation._tree_snapshot(path) for name, path in roots.items()}
    result = attestation._execute(config_path, workspace=tmp_path)
    after = {name: attestation._tree_snapshot(path) for name, path in roots.items()}
    assert before == after
    assert result["composite_v1_plus_attestation_integrity_pass"] is True
    assert result["query_generation_count"] == 0
    assert result["solver_invocation_count"] == 0
    assert result["model_inference_count"] == 0
    assert result["bootstrap_resample_count"] == 0

    final = tmp_path / "outputs/test_v4_aggregate_repair_v1_attestation_v1"
    assert final.is_dir()
    assert not (
        tmp_path / "outputs/.test_v4_aggregate_repair_v1_attestation_v1.incomplete"
    ).exists()
    manifest = json.loads(
        (final / "attestation_final_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["hash_chain"].items():
        assert attestation._sha256_file(final / name) == expected
    recomputation = json.loads(
        (final / "independent_recomputation.json").read_text(encoding="utf-8")
    )
    assert recomputation["all_semantic_matches"] is True
    assert recomputation["outcome_direction_hardcoded"] is False
    assert recomputation["query_records_parsed_or_used_for_recomputation"] is False
    assert recomputation["query_record_files_hash_verified_only"] is True
    assert recomputation["observed_results_not_used_as_acceptance_criteria"][
        "paper_gate_pass"
    ] is False
    source = json.loads(
        (final / "source_commit_verification.json").read_text(encoding="utf-8")
    )
    assert source["commit"] == repair_commit
    assert source["bundle"]["header"]["self_contained"] is True
    clone = tmp_path / "bundle-clone"
    subprocess.run(
        ["git", "clone", str(final / "repair_v1_source.bundle"), str(clone)],
        check=True,
        capture_output=True,
    )
    assert (
        _git(clone, "rev-parse", "refs/remotes/origin/test-repair-source")
        == repair_commit
    )
    with pytest.raises(RuntimeError, match="rerun is forbidden"):
        attestation._execute(config_path, workspace=tmp_path)


def test_attestation_rejects_v1_tampering(tmp_path: Path) -> None:
    config_path, roots, _ = _fixture(tmp_path)
    target = roots["v1"] / "joint_holm_v4.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="repair v1 tree differs"):
        attestation._execute(config_path, workspace=tmp_path)
    assert not (
        tmp_path / "outputs/test_v4_aggregate_repair_v1_attestation_v1"
    ).exists()


def test_attestation_rejects_shadow_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, _ = _fixture(tmp_path)
    monkeypatch.setenv("GIT_DIR", "/tmp/shadow/.git")
    with pytest.raises(RuntimeError, match="forbids GIT_DIR"):
        attestation._execute(config_path, workspace=tmp_path)


def test_attestation_lock_is_exclusive(tmp_path: Path) -> None:
    config_path, _, _ = _fixture(tmp_path)
    lock = tmp_path / "outputs/.test_v4_aggregate_repair_v1_attestation_v1.lock"
    lock.write_text("held", encoding="utf-8")
    with pytest.raises(RuntimeError, match="attestation lock already exists"):
        attestation._execute(config_path, workspace=tmp_path)


def test_attestation_rechecks_v1_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, roots, _ = _fixture(tmp_path)
    writer = attestation._write_json_new

    def mutate_after_final(path: Path, payload: object) -> None:
        writer(path, payload)  # type: ignore[arg-type]
        if path.name == "attestation_final_manifest.json":
            target = roots["v1"] / "joint_holm_v4.json"
            target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")

    monkeypatch.setattr(attestation, "_write_json_new", mutate_after_final)
    with pytest.raises(RuntimeError, match="repair v1 changed"):
        attestation._execute(config_path, workspace=tmp_path)
    assert not (
        tmp_path / "outputs/test_v4_aggregate_repair_v1_attestation_v1"
    ).exists()
    failure = tmp_path / (
        "outputs/.test_v4_aggregate_repair_v1_attestation_v1.incomplete/"
        "attestation_failure_manifest.json"
    )
    assert failure.is_file()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["automatic_resume_allowed"] is False
    assert payload["repair_v1_modification_performed_by_attestation"] is False


def test_production_attestation_has_no_incomplete_state() -> None:
    workspace = Path(attestation.__file__).resolve().parents[3]
    for relative in (
        "outputs/.test_v4_aggregate_repair_v1_attestation_v1.incomplete",
        "outputs/.test_v4_aggregate_repair_v1_attestation_v1.lock",
    ):
        assert not (workspace / relative).exists()
    final = workspace / "outputs/test_v4_aggregate_repair_v1_attestation_v1"
    if final.exists():
        manifest_path = final / "attestation_final_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["protocol"] == f"{attestation.PROTOCOL}_final_manifest"
        assert manifest["composite_v1_plus_attestation_integrity_pass"] is True
        assert {
            path.name for path in final.iterdir() if path.is_file()
        } == set(manifest["hash_chain"]) | {manifest_path.name}
        for name, expected in manifest["hash_chain"].items():
            assert attestation._sha256_file(final / name) == expected
