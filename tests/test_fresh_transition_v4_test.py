from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from confik.config import load_config, load_robot, resolve_path
from confik.data.datasets import QueryDataset
from confik.fresh_transition_v4_test.benchmark import (
    METHODS,
    STAGE_NAMES,
    BenchmarkData,
    FrozenMethod,
    benchmark_trajectories,
    latin_method_orders,
)
from confik.fresh_transition_v4_test.data import (
    DT,
    EXPECTED_FRAMES,
    EXPECTED_TRAJECTORIES,
    FAMILIES,
    FROZEN_POOL_SEEDS,
    PROTOCOL,
    FreshTransitionSpec,
    audit_freshness,
    build_identity_manifest,
    build_prior_identity_registries,
    generate_fresh_transition_dataset,
    load_fresh_dataset,
    load_identity_manifest,
    save_fresh_dataset,
    save_identity_manifest,
)
from confik.fresh_transition_v4_test.reporting import (
    completion_identity,
    family_rows,
    final_gate,
    main_rows,
    trajectory_rows,
    validate_benchmark,
)
from confik.fresh_transition_v4_test.runner import (
    _artifact,
    _parser,
    _verify_sealed_stage_inputs,
    identity_anchors,
    validate_config,
)
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.types import VerificationResult


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "fresh_transition_v4_test.yaml"
SOURCE_CONFIG_PATH = WORKSPACE / "configs" / "paper_v2.yaml"

PROTECTED_TREE_DIGESTS = {
    "outputs/anchored_temporal_v7_dominance_pilot": (
        "b55c47b7f313125c93a285bf9e083a340e5c56f5c0e942fa75f2b9b6f0ba777b"
    ),
    "outputs/temporal_event_v6_pilot": (
        "04605f614cdc6a06680e69927a06efa3903452ff2579fb76685d52a54690d2dc"
    ),
    "outputs/anchored_temporal_v7_pilot": (
        "2ceba3ce4d0f02baa833d98dc422a59f4d42eb094bd3bb6670571f050c718ecc"
    ),
    "outputs/release_v4_locked": (
        "df8b97af4335d019d14c6414564a0eb9cf8770106192ea4e16019a200c6284bd"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink()
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(_sha256_file(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


@pytest.fixture(scope="session")
def panda_fresh() -> tuple[Any, dict[str, Any], Any]:
    source = load_config(SOURCE_CONFIG_PATH)
    model = load_robot(source, "panda")
    urdf = resolve_path(source, str(source["robots"]["panda"]["urdf"]))
    spec = FreshTransitionSpec.frozen(
        "panda", kinematics_identity=_sha256_file(urdf)
    )
    fresh, generation = generate_fresh_transition_dataset(model, spec)
    return fresh, generation, model


@pytest.fixture(scope="session")
def panda_isolation(
    panda_fresh: tuple[Any, dict[str, Any], Any],
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    fresh, _, _ = panda_fresh
    registries, ledger = build_prior_identity_registries(
        WORKSPACE,
        robot="panda",
        dt=DT,
        kinematics_identity=fresh.kinematics_identity,
    )
    audit = audit_freshness(fresh, registries, access_ledger=ledger)
    return registries, ledger, audit


def test_real_yaml_is_fully_frozen_and_rejects_scientific_drift() -> None:
    config = load_config(CONFIG_PATH)
    workspace = resolve_path(config, str(config["workspace"]))
    assert workspace == WORKSPACE
    validate_config(config, workspace=workspace)

    for path, value in (
        (("methods",), [*METHODS, "temporal_event_cghik_v6"]),
        (("fresh_data", "pool_seed", "panda"), 17),
        (("timing", "trajectory_repeats"), 2),
        (("final_gate", "p50_is_gate"), True),
    ):
        changed = deepcopy(config)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError):
            validate_config(changed, workspace=workspace)


def test_exact_three_methods_and_no_retired_runtime_imports() -> None:
    assert METHODS == (
        "fixed_robust_cascade",
        "always_hard",
        "counterfactual_cghik_v4",
    )
    forbidden = {
        "confik.hierarchical_v5",
        "confik.hierarchical_v5_lite",
        "confik.temporal_v6",
        "confik.anchored_temporal_v7",
    }
    for filename in ("benchmark.py", "reporting.py", "runner.py"):
        path = WORKSPACE / "src" / "confik" / "fresh_transition_v4_test" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.lstrip(".")
                imported.add(module)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imported
            for prefix in forbidden
        )
    runner_source = (
        WORKSPACE / "src" / "confik" / "fresh_transition_v4_test" / "runner.py"
    ).read_text(encoding="utf-8")
    assert "benchmark_trajectories(" in runner_source
    assert "v5_v6_v7_runtime_invocation_count\": 0" in runner_source


def test_frozen_fresh_panda_data_matches_committed_identity_anchors(
    panda_fresh: tuple[Any, dict[str, Any], Any],
) -> None:
    fresh, generation, _ = panda_fresh
    config = load_config(CONFIG_PATH)

    assert FROZEN_POOL_SEEDS == {"panda": 864901, "ur5e": 864902}
    assert fresh.pool_seed == 864901
    assert fresh.count == EXPECTED_FRAMES == 80 * 150
    assert len(fresh.trajectory_order) == EXPECTED_TRAJECTORIES == 80
    assert len(set(fresh.trajectory_order)) == 80
    assert len(set(fresh.formal_query_sha256.tolist())) == EXPECTED_FRAMES
    assert len(set(fresh.source_query_hash.tolist())) == EXPECTED_FRAMES
    assert Counter(fresh.dataset.category.tolist()) == Counter(
        {family: 20 * 150 for family in FAMILIES}
    )
    assert all(len(rows) == 150 for _, rows in fresh.groups())
    assert identity_anchors(fresh) == config["fresh_data"]["frozen_identity"]["panda"]
    assert generation["solver_calls"] == 0
    assert generation["verifier_calls"] == 0
    assert generation["learned_model_calls"] == 0
    assert generation["prior_performance_fields_read"] == 0


def test_all_six_development_and_five_formal_identity_registries_are_isolated(
    panda_isolation: tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]],
) -> None:
    registries, ledger, audit = panda_isolation
    classes = Counter(registry.source_class for registry in registries)

    assert len(registries) == 11
    assert classes == Counter({"development": 6, "formal_evaluation": 5})
    assert ledger["source_class_counts"] == {
        "development": 6,
        "formal_evaluation": 5,
    }
    assert ledger["performance_files_opened"] == 0
    assert ledger["performance_arrays_read"] == 0
    assert ledger["formal_result_roots_opened"] == []
    allowed_formal = {
        "previous_q",
        "target_position",
        "target_rotation",
        "category",
        "trajectory_id",
        "time_index",
    }
    for registry in registries:
        assert registry.performance_arrays_read is False
        if registry.source_class == "formal_evaluation":
            assert set(registry.arrays_read) <= allowed_formal
    assert audit["status"] == "pass"
    assert audit["all_isolation_checks_pass"] is True
    assert audit["prior_overlap_counts"] == {
        "formal_query_hash": 0,
        "runtime_query_hash": 0,
        "trajectory_uid": 0,
        "trajectory_id": 0,
        "seed": 0,
    }
    assert audit["formal_performance_consumed"] is False


def test_identity_dataset_and_self_hashed_manifest_roundtrip(
    tmp_path: Path,
    panda_fresh: tuple[Any, dict[str, Any], Any],
    panda_isolation: tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]],
) -> None:
    fresh, _, _ = panda_fresh
    _, _, isolation = panda_isolation
    dataset_path = tmp_path / "panda_fresh.npz"
    dataset_artifact = save_fresh_dataset(dataset_path, fresh)
    manifest = build_identity_manifest(
        fresh, isolation, dataset_artifact=dataset_artifact
    )
    manifest_path = tmp_path / "panda_identity.json"
    manifest_artifact = save_identity_manifest(manifest_path, manifest)

    loaded_manifest = load_identity_manifest(
        manifest_path,
        expected_artifact=manifest_artifact,
        expected_robot="panda",
    )
    loaded = load_fresh_dataset(
        dataset_path,
        robot="panda",
        expected_artifact=dataset_artifact,
        identity_manifest=loaded_manifest,
    )
    assert loaded_manifest == manifest
    assert loaded.trajectory_order == fresh.trajectory_order
    assert np.array_equal(loaded.formal_query_sha256, fresh.formal_query_sha256)

    tampered = deepcopy(manifest)
    tampered["pool_seed"] += 1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(
        __import__("json").dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="self-hash"):
        load_identity_manifest(tampered_path, expected_robot="panda")


class _ToyVerifier:
    def __init__(self, *, reject_check: int | None = None) -> None:
        self.config = ("same-deterministic-contract",)
        self.reject_check = reject_check
        self.calls = 0

    def check(self, q: np.ndarray, query: Any) -> VerificationResult:
        accepted = self.calls != self.reject_check
        self.calls += 1
        return VerificationResult(
            accepted=accepted,
            position_error=0.0 if accepted else 1.0,
            orientation_error=0.0,
            joint_limit_ok=True,
            velocity_ok=accepted,
            finite_ok=bool(np.all(np.isfinite(q))),
            reasons=() if accepted else ("toy_reject",),
        )


class _ToyRuntime:
    def __init__(self, name: str, increment: float) -> None:
        self.name = name
        self.increment = increment
        self.previous_seen: list[np.ndarray] = []
        self.calls = 0

    def solve(self, query: Any) -> ProfiledOutcome:
        self.calls += 1
        self.previous_seen.append(query.previous_q.copy())
        command = query.previous_q + self.increment
        return ProfiledOutcome(
            q=command,
            accepted=True,
            entry_action="easy" if self.name == METHODS[0] else "hard",
            executed_stages=("toy_solver", "deterministic_verifier"),
            risk_probabilities=np.asarray([0.0, 0.0, 1.0, 0.0]),
            risk_score=1.0,
            function_evaluations=2,
            iterations=1,
            fallback_used=False,
            verification_reasons=(),
            reject_reason="",
            candidate_count=1,
            timings_ns={},
        )


def _toy_role() -> SimpleNamespace:
    count = 6
    uid = np.repeat(np.asarray(["a" * 64, "b" * 64]), 3)
    initial = np.repeat(np.asarray([[0.0, 0.0], [10.0, 10.0]]), 3, axis=0)
    dataset = QueryDataset(
        previous_q=initial,
        target_position=np.zeros((count, 3), dtype=np.float64),
        target_rotation=np.repeat(np.eye(3)[None, :, :], count, axis=0),
        reference_q=initial.copy(),
        category=np.repeat(np.asarray([FAMILIES[0], FAMILIES[1]]), 3),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.repeat(np.asarray([1, 2]), 3),
        time_index=np.tile(np.arange(3), 2),
    )
    return SimpleNamespace(
        robot="panda",
        dt=DT,
        dataset=dataset,
        trajectory_uid=uid,
        trajectory_order=("a" * 64, "b" * 64),
        source_query_hash=np.asarray([f"{index + 1:064x}" for index in range(count)]),
    )


def _toy_methods() -> dict[str, FrozenMethod]:
    result: dict[str, FrozenMethod] = {}
    for column, name in enumerate(METHODS):
        runtime = _ToyRuntime(name, float(column + 1))
        verifier = _ToyVerifier(reject_check=1 if name == METHODS[2] else None)
        result[name] = FrozenMethod(
            name=name,
            runtime=runtime,
            verifier=verifier,  # type: ignore[arg-type]
            kinematics=object(),
            dls=object(),
            seed_engine=object(),
            seed_bank=object(),
            fallback=object(),
        )
    return result


def test_toy_benchmark_calls_every_method_once_latin_closes_stages_and_branches() -> None:
    role = _toy_role()
    methods = _toy_methods()
    data = benchmark_trajectories(
        role,
        methods,
        order_seed=940901,
        progress_every=0,
        synchronize_cuda=False,
    )

    assert all(methods[name].runtime.calls == 6 for name in METHODS)
    assert np.all(data.latency_ns > 0)
    assert np.array_equal(data.stage_latency_ns.sum(axis=2), data.latency_ns)
    assert all(
        np.array_equal(np.sort(row), np.arange(3))
        for row in data.method_order_position
    )
    for position in range(3):
        assert int(np.sum(data.method_order_position == position)) == 6
    assert len(latin_method_orders(METHODS, 940901)) == 3

    fixed_seen = methods[METHODS[0]].runtime.previous_seen
    hard_seen = methods[METHODS[1]].runtime.previous_seen
    v4_seen = methods[METHODS[2]].runtime.previous_seen
    assert [row[0] for row in fixed_seen[:3]] == [0.0, 1.0, 2.0]
    assert [row[0] for row in hard_seen[:3]] == [0.0, 2.0, 4.0]
    # The independent verifier rejects V4's second command, so it cannot
    # advance the next-frame closed-loop state despite runtime acceptance.
    assert [row[0] for row in v4_seen[:3]] == [0.0, 3.0, 3.0]
    v4_column = METHODS.index("counterfactual_cghik_v4")
    assert data.accepted[1, v4_column]
    assert data.verifier_checked[1, v4_column]
    assert not data.verifier_accepted[1, v4_column]
    assert data.accepted_contract_violation[1, v4_column]
    assert all(methods[name].verifier.calls == 6 for name in METHODS)


def _synthetic_raw(robot: str = "panda") -> BenchmarkData:
    frames = EXPECTED_FRAMES
    methods = len(METHODS)
    trajectories = EXPECTED_TRAJECTORIES
    order = np.asarray([f"{10_000 + index:064x}" for index in range(trajectories)])
    uid = np.repeat(order, 150)
    category = np.repeat(
        np.asarray([family for family in FAMILIES for _ in range(20)]), 150
    )
    accepted = np.ones((frames, methods), dtype=bool)
    accepted[-1, METHODS.index("always_hard")] = False
    checked = accepted.copy()
    verifier_accepted = accepted.copy()
    latency = np.tile(np.asarray([1500, 1000, 800], dtype=np.int64), (frames, 1))
    stages = np.zeros((frames, methods, len(STAGE_NAMES)), dtype=np.int64)
    stages[:, :, -1] = latency
    command = np.zeros((frames, methods, 2), dtype=np.float64)
    command[~accepted] = np.nan
    fev = np.tile(np.asarray([30, 20, 16], dtype=np.int64), (frames, 1))
    positions = np.tile(np.arange(methods, dtype=np.int8), (frames, 1))
    return BenchmarkData(
        robot=robot,
        method_names=METHODS,
        stage_names=STAGE_NAMES,
        trajectory_order=order,
        source_query_hash=np.asarray([f"{index + 1:064x}" for index in range(frames)]),
        trajectory_uid=uid,
        category=category,
        time_index=np.tile(np.arange(150), trajectories),
        expected_reachable=np.ones(frames, dtype=bool),
        continuity_feasible=np.ones(frames, dtype=bool),
        latency_ns=latency,
        stage_latency_ns=stages,
        accepted=accepted,
        accepted_contract_violation=np.zeros((frames, methods), dtype=bool),
        verifier_checked=checked,
        verifier_accepted=verifier_accepted,
        verifier_position_error=np.where(accepted, 0.0, np.nan),
        verifier_orientation_error=np.where(accepted, 0.0, np.nan),
        verifier_joint_limit_ok=accepted.copy(),
        verifier_velocity_ok=accepted.copy(),
        verifier_finite_ok=accepted.copy(),
        verifier_reasons=np.full((frames, methods), "", dtype="U8"),
        function_evaluations=fev,
        iterations=np.ones((frames, methods), dtype=np.int64),
        fallback_used=np.zeros((frames, methods), dtype=bool),
        learned_seed_invoked=np.ones((frames, methods), dtype=bool),
        candidate_count=np.ones((frames, methods), dtype=np.int64),
        entry_action=np.full((frames, methods), "hard", dtype="U8"),
        executed_stages=np.full((frames, methods), "hard", dtype="U8"),
        reject_reason=np.full((frames, methods), "", dtype="U8"),
        risk_score=np.ones((frames, methods), dtype=np.float64),
        risk_probabilities=np.tile(
            np.asarray([0.0, 0.0, 1.0, 0.0]), (frames, methods, 1)
        ),
        v4_decision_reason=np.full((frames, methods), "", dtype="U8"),
        v4_eligible_actions=np.full((frames, methods), "", dtype="U8"),
        v4_predicted_success=np.full((frames, methods, 3), np.nan),
        v4_predicted_p50_ms=np.full((frames, methods, 3), np.nan),
        v4_predicted_p95_ms=np.full((frames, methods, 3), np.nan),
        v4_fail_all_probability=np.full((frames, methods), np.nan),
        v4_ood_score=np.full((frames, methods), np.nan),
        v4_is_ood=np.zeros((frames, methods), dtype=bool),
        command_q=command,
        executed_query_hash=np.asarray(
            [[f"{index * methods + column + 1:064x}" for column in range(methods)]
             for index in range(frames)]
        ),
        method_order_position=positions,
    )


def test_reporting_recomputes_main_family_trajectory_completion_and_gate() -> None:
    data = _synthetic_raw()
    validate_benchmark(data)
    main = main_rows(data)
    family = family_rows(data)
    trajectory = trajectory_rows(data)
    completion = completion_identity(data)

    assert len(main) == 3
    assert len(family) == 3 * 4
    assert len(trajectory) == 3 * 80
    hard = next(row for row in main if row["method"] == "always_hard")
    v4 = next(row for row in main if row["method"] == "counterfactual_cghik_v4")
    assert hard["whole_trajectory_completion_count"] == 79
    assert v4["whole_trajectory_completion_count"] == 80
    assert v4["total_cumulative_latency_ns"] == EXPECTED_FRAMES * 800
    assert v4["trajectory_cumulative_latency_mean_ms"] == pytest.approx(0.12)
    assert v4["frame_p50_latency_ms"] == pytest.approx(0.0008)
    assert v4["frame_p95_latency_ms"] == pytest.approx(0.0008)
    assert v4["frame_p99_latency_ms"] == pytest.approx(0.0008)
    assert v4["mean_fev"] == pytest.approx(16.0)
    assert completion["v4_lost_vs_always_hard_trajectory_uids"] == []
    assert completion["v4_gained_vs_always_hard_trajectory_uids"] == [
        str(data.trajectory_order[-1])
    ]
    assert all(row["trajectory_count"] == 20 for row in family)

    both = main + [dict(row, robot="ur5e") for row in main]
    gate = final_gate(both)
    assert gate["status"] == "pass"
    assert gate["all_robots_pass"] is True
    assert gate["robots"]["panda"]["ratios_vs_always_hard"][
        "aggregate_cumulative_latency"
    ] == pytest.approx(0.8)
    assert gate["robots"]["panda"]["ratios_vs_always_hard"]["mean_fev"] == pytest.approx(0.8)


def test_reporting_never_counts_a_runtime_accept_rejected_by_the_verifier() -> None:
    data = _synthetic_raw()
    verifier_accepted = data.verifier_accepted.copy()
    violation = data.accepted_contract_violation.copy()
    verifier_accepted[0, 0] = False
    violation[0, 0] = True
    audited = replace(
        data,
        verifier_accepted=verifier_accepted,
        accepted_contract_violation=violation,
    )

    main = main_rows(audited)
    fixed = next(row for row in main if row["method"] == METHODS[0])
    assert fixed["frame_verified_success_count"] == EXPECTED_FRAMES - 1
    assert fixed["whole_trajectory_completion_count"] == 79
    assert fixed["accepted_contract_violation_count"] == 1


def test_any_method_contract_violation_invalidates_the_final_evaluation() -> None:
    panda = main_rows(_synthetic_raw())
    rows = panda + [dict(row, robot="ur5e") for row in panda]
    comparator = next(
        row
        for row in rows
        if row["robot"] == "panda" and row["method"] == "always_hard"
    )
    comparator["accepted_contract_violation_count"] = 1

    gate = final_gate(rows)

    assert gate["status"] == "fail"
    assert gate["all_robots_pass"] is False


@pytest.mark.parametrize(
    ("field", "bad_value", "failed_check"),
    (
        (
            "whole_trajectory_completion_count",
            78,
            "v4_completion_count_not_below_always_hard",
        ),
        (
            "total_cumulative_latency_ns",
            EXPECTED_FRAMES * 851,
            "aggregate_cumulative_latency_ratio_at_most_0_85",
        ),
        ("mean_fev", 17.01, "mean_fev_ratio_at_most_0_85"),
        ("frame_p95_latency_ms", 0.00101, "p95_latency_ratio_at_most_1_0"),
        ("frame_p99_latency_ms", 0.001051, "p99_latency_ratio_at_most_1_05"),
        (
            "accepted_contract_violation_count",
            1,
            "accepted_contract_violation_count_zero",
        ),
    ),
)
def test_each_final_gate_fails_independently(
    field: str, bad_value: Any, failed_check: str
) -> None:
    panda = main_rows(_synthetic_raw())
    rows = panda + [dict(row, robot="ur5e") for row in panda]
    target = next(
        row
        for row in rows
        if row["robot"] == "panda" and row["method"] == "counterfactual_cghik_v4"
    )
    target[field] = bad_value
    gate = final_gate(rows)
    assert gate["status"] == "fail"
    assert gate["robots"]["panda"]["checks"][failed_check] is False
    assert gate["robots"]["ur5e"]["pass"] is True


def test_p50_is_report_only_not_a_gate() -> None:
    panda = main_rows(_synthetic_raw())
    rows = panda + [dict(row, robot="ur5e") for row in panda]
    for row in rows:
        if row["method"] == "counterfactual_cghik_v4":
            row["frame_p50_latency_ms"] = 999.0
    gate = final_gate(rows)
    assert gate["status"] == "pass"
    assert all(value["p50_is_report_only"] for value in gate["robots"].values())


def test_cli_has_no_smoke_resume_or_rerun_surface() -> None:
    parser = _parser()
    args = parser.parse_args(["--config", str(CONFIG_PATH)])
    assert vars(args) == {"config": str(CONFIG_PATH)}
    for forbidden in ("--smoke", "--resume", "--rerun"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", str(CONFIG_PATH), forbidden])
    script = (WORKSPACE / "scripts" / "run_fresh_transition_v4_test.sh").read_text(
        encoding="utf-8"
    )
    assert "if [[ $# -ne 0 ]]" in script
    assert "no smoke, resume, or rerun mode" in script


@pytest.mark.parametrize(
    "tampered_relative",
    (
        "panda_fresh.npz",
        "ur5e_identity.json",
        "freshness_audit.json",
        "effective_config.yaml",
    ),
)
def test_preregistration_seal_rejects_any_bound_input_tamper(
    tmp_path: Path, tampered_relative: str
) -> None:
    relative_files = (
        "panda_fresh.npz",
        "ur5e_fresh.npz",
        "panda_identity.json",
        "ur5e_identity.json",
        "freshness_audit.json",
        "effective_config.yaml",
    )
    for index, relative in enumerate(relative_files):
        (tmp_path / relative).write_bytes(f"sealed-{index}".encode("ascii"))
    descriptor = {
        relative: _artifact(tmp_path / relative, relative_to=tmp_path)
        for relative in relative_files
    }
    seal = {
        "fresh_dataset_artifacts": {
            "panda": descriptor["panda_fresh.npz"],
            "ur5e": descriptor["ur5e_fresh.npz"],
        },
        "identity_manifest_artifacts": {
            "panda": descriptor["panda_identity.json"],
            "ur5e": descriptor["ur5e_identity.json"],
        },
        "freshness_audit": descriptor["freshness_audit.json"],
        "protocol_config": descriptor["effective_config.yaml"],
    }
    _verify_sealed_stage_inputs(tmp_path, seal)

    with (tmp_path / tampered_relative).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="changed after preregistration"):
        _verify_sealed_stage_inputs(tmp_path, seal)


def test_preexisting_protected_trees_match_preimplementation_digests() -> None:
    assert {
        relative: _tree_digest(WORKSPACE / relative)
        for relative in PROTECTED_TREE_DIGESTS
    } == PROTECTED_TREE_DIGESTS
