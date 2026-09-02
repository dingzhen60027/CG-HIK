from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from confik.config import load_config
from confik.hierarchical_v5.pilot import latin_method_orders
from confik.hierarchical_v5_lite.features import (
    lite_feature_dim,
    lite_feature_names,
    prepare_lite_features,
)
from confik.hierarchical_v5_lite.model import (
    LiteGatePredictor,
    LiteGateTrainingConfig,
    TorchScriptLiteGateInference,
    export_exact_torchscript,
    load_exact_torchscript,
    numerical_equivalence,
)
from confik.hierarchical_v5_lite.pilot import (
    METHODS,
    STAGE_NAMES,
    BenchmarkData,
    _pilot_gate,
    _stage_timings,
    family_routes,
    latency_breakdown,
    paired_summary,
    summarize,
    validate_config,
)
from confik.hierarchical_v5_lite.policy import (
    LiteGateDecision,
    ThresholdSelectionConfig,
    select_thresholds,
)
from confik.hierarchical_v5_lite.runtime import HierarchicalLiteRuntime
from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.solvers.verifier import SolutionVerifier
from confik.types import IKQuery, SolveTrace


ROOT = Path(__file__).resolve().parents[1]
TOY_URDF = Path(__file__).parent / "assets" / "toy_arm.urdf"


def _toy() -> URDFKinematics:
    return URDFKinematics.from_file(TOY_URDF)


def test_lite_feature_path_uses_one_fk_and_no_jacobian_or_linear_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _toy()
    q = np.zeros(model.nq, dtype=np.float64)
    target = model.forward(q)
    query = IKQuery(target, q, dt=0.02)
    calls = 0
    original_forward = model.forward

    def counted_forward(values: np.ndarray):
        nonlocal calls
        calls += 1
        return original_forward(values)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("expensive operation reached V5-Lite feature path")

    monkeypatch.setattr(model, "forward", counted_forward)
    monkeypatch.setattr(model, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)

    prepared = prepare_lite_features(model, query)

    assert calls == 1
    assert prepared.features.shape == (model.nq + 9,)
    assert prepared.features.dtype == np.float32
    assert prepared.features.flags.c_contiguous
    assert np.all(np.isfinite(prepared.features))
    assert len(prepared.feature_names) == lite_feature_dim(model.nq)
    assert prepared.feature_names == lite_feature_names(model)


@pytest.mark.parametrize("input_dim", (15, 16))
def test_single_head_exact_torchscript_is_numerically_and_route_equivalent(
    tmp_path: Path, input_dim: int
) -> None:
    rng = np.random.default_rng(71 + input_dim)
    features = rng.normal(size=(96, input_dim)).astype(np.float32)
    labels = (np.arange(96) % 2).astype(np.float32)
    predictor = LiteGatePredictor(
        input_dim,
        LiteGateTrainingConfig(epochs=2, batch_size=24, seed=19),
        device="cpu",
    )
    predictor.fit(features, labels, role="risk_train_queries")
    predictor.calibrate(features, labels, role="calibration_queries")
    artifact = tmp_path / f"gate_{input_dim}.ts"
    export_exact_torchscript(predictor, artifact)
    module = load_exact_torchscript(artifact, device="cpu")
    result = numerical_equivalence(
        predictor, module, features[:32], threshold=0.7, atol=1.0e-10
    )
    adapter = TorchScriptLiteGateInference(module, input_dim)

    assert result["passed"]
    assert result["route_match_rate"] == 1.0
    assert 0.0 <= adapter.predict_one(features[0]).local_success_probability <= 1.0
    with pytest.raises(ValueError, match="forbids"):
        predictor.fit(features, labels, role="policy_validation_queries")
    with pytest.raises(ValueError, match="forbids"):
        predictor.calibrate(features, labels, role="test_v4_queries")


def test_cost_sensitive_threshold_uses_calibration_p95_then_p50() -> None:
    probability = np.asarray([0.9, 0.8, 0.2, 0.1])
    local_success = np.asarray([True, True, False, False])
    hard_success = np.ones(4, dtype=bool)
    primary = np.ones(4, dtype=bool)
    forced_fast = np.asarray([[3] * 5, [3] * 5, [13] * 5, [13] * 5])
    forced_robust = np.full((4, 5), 12, dtype=np.int64)
    forced_fast_seed = ~local_success
    forced_robust_seed = np.ones(4, dtype=bool)

    policy, report = select_thresholds(
        probability,
        forced_fast,
        forced_robust,
        hard_success,
        hard_success,
        local_success,
        forced_fast_seed,
        forced_robust_seed,
        primary,
        role="calibration_queries",
        config=ThresholdSelectionConfig(threshold_grid=(0.0, 0.5, 1.0)),
    )

    assert policy.local_success_threshold == 0.5
    assert report["selected"]["success_vector_equal_to_always_hard"]
    assert report["objective_order"][:3] == [
        "success_vector_equal_to_always_hard",
        "minimum_primary_query_median_p95_ns",
        "minimum_primary_query_median_p50_ns",
    ]
    with pytest.raises(ValueError, match="forbids"):
        select_thresholds(
            probability,
            forced_fast,
            forced_robust,
            hard_success,
            hard_success,
            local_success,
            forced_fast_seed,
            forced_robust_seed,
            primary,
            role="policy_validation_queries",
            config=ThresholdSelectionConfig(threshold_grid=(0.5, 1.0)),
        )


def test_cost_sensitive_threshold_requires_per_query_success_equality() -> None:
    probability = np.asarray([0.95, 0.10])
    local_success = np.asarray([True, False])
    hard_success = np.asarray([False, True])
    costs = np.ones((2, 5), dtype=np.int64)

    policy, report = select_thresholds(
        probability,
        costs,
        np.full((2, 5), 10, dtype=np.int64),
        local_success,
        hard_success,
        local_success,
        np.asarray([False, True]),
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
        role="calibration_queries",
        config=ThresholdSelectionConfig(threshold_grid=(0.5, 1.0)),
    )

    assert policy.local_success_threshold == 1.0
    rejected = next(
        row for row in report["candidates"]
        if row["local_success_threshold"] == 0.5
    )
    assert not rejected["eligible"]
    assert rejected["success_mismatch_count"] == 1


def test_cost_sensitive_threshold_breaks_cost_ties_by_seed_calls_then_threshold() -> None:
    probability = np.asarray([0.9, 0.8])
    success = np.ones(2, dtype=bool)
    costs = np.full((2, 5), 5, dtype=np.int64)

    policy, _ = select_thresholds(
        probability,
        costs,
        costs,
        success,
        success,
        success,
        np.zeros(2, dtype=bool),
        np.ones(2, dtype=bool),
        success,
        role="calibration_queries",
        config=ThresholdSelectionConfig(threshold_grid=(0.0, 0.5, 0.7, 1.0)),
    )

    # 0.0, 0.5 and 0.7 have identical P95/P50 and zero seed calls.  The
    # frozen conservative final tie-break must choose the highest threshold.
    assert policy.local_success_threshold == 0.7


def test_threshold_grid_requires_explicit_never_fast_sentinel() -> None:
    with pytest.raises(ValueError, match="never-FAST"):
        ThresholdSelectionConfig(threshold_grid=(0.0, 0.5, 0.99))


class _FixedTraceDLS:
    def __init__(self, q_result: np.ndarray, fev: int):
        self.q_result = np.asarray(q_result, dtype=np.float64)
        self.fev = int(fev)

    def solve(
        self,
        target,
        seed: np.ndarray,
        max_iterations: int,
        *,
        seed_source: str = "unknown",
    ) -> SolveTrace:
        del target, seed, max_iterations
        return SolveTrace(
            q=self.q_result.copy(),
            converged=True,
            iterations=1,
            position_error=0.0,
            orientation_error=0.0,
            seed_source=seed_source,
            reason="synthetic",
            function_evaluations=self.fev,
        )


class _StaticLiteGate:
    def __init__(self, choose_fast: bool):
        self.choose_fast = bool(choose_fast)

    def decide(self, features: np.ndarray) -> LiteGateDecision:
        return LiteGateDecision(
            action="fast" if self.choose_fast else "robust",
            choose_fast=self.choose_fast,
            reason="synthetic",
            local_success_probability=0.99 if self.choose_fast else 0.01,
            local_success_threshold=0.5,
        )


class _FixedHardRuntime:
    def __init__(self, q: np.ndarray, *, fev: int = 11, stages=("hard",)):
        self.q = np.asarray(q, dtype=np.float64)
        self.fev = int(fev)
        self.stages = tuple(stages)
        self.calls = 0

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        del query
        self.calls += 1
        return ProfiledOutcome(
            q=self.q.copy(),
            accepted=True,
            entry_action="hard",
            executed_stages=self.stages,
            risk_probabilities=np.asarray([0.0, 0.0, 1.0, 0.0]),
            risk_score=1.0,
            function_evaluations=self.fev,
            iterations=4,
            fallback_used=False,
            verification_reasons=(),
            reject_reason="",
            candidate_count=2,
            timings_ns={"total_end_to_end_ns": 1},
        )


def test_lite_runtime_verifier_controls_fast_return_and_hard_recovery() -> None:
    model = _toy()
    previous = np.zeros(model.nq, dtype=np.float64)
    target_q = np.asarray([0.02, -0.01], dtype=np.float64)
    query = IKQuery(model.forward(target_q), previous, dt=0.02)
    verifier = SolutionVerifier(model)

    accepting_hard = _FixedHardRuntime(target_q, fev=11)
    failing_local = HierarchicalLiteRuntime(
        kinematics=model,
        dls=_FixedTraceDLS(previous, fev=3),  # type: ignore[arg-type]
        verifier=verifier,
        fast_gate=_StaticLiteGate(True),
        always_hard_runtime=accepting_hard,
    )
    recovered = failing_local.solve(query)

    assert recovered.accepted
    assert recovered.route == "fast_fail_hard_recovery"
    assert recovered.local_attempted and not recovered.local_accepted
    assert recovered.learned_seed_ensemble_invoked
    assert recovered.function_evaluations == 14
    assert recovered.executed_stages == ("local_fast", "hard")
    assert accepting_hard.calls == 1

    unused_hard = _FixedHardRuntime(target_q)
    accepting_local = HierarchicalLiteRuntime(
        kinematics=model,
        dls=_FixedTraceDLS(target_q, fev=3),  # type: ignore[arg-type]
        verifier=verifier,
        fast_gate=_StaticLiteGate(True),
        always_hard_runtime=unused_hard,
    ).solve(query)
    assert accepting_local.accepted and accepting_local.route == "fast_accept"
    assert not accepting_local.learned_seed_ensemble_invoked
    assert unused_hard.calls == 0


def test_lite_runtime_direct_route_is_fixed_hard_and_rejects_stage_drift() -> None:
    model = _toy()
    previous = np.zeros(model.nq, dtype=np.float64)
    target_q = np.asarray([0.02, -0.01], dtype=np.float64)
    query = IKQuery(model.forward(target_q), previous, dt=0.02)
    verifier = SolutionVerifier(model)
    hard = _FixedHardRuntime(target_q)
    outcome = HierarchicalLiteRuntime(
        kinematics=model,
        dls=_FixedTraceDLS(previous, fev=3),  # type: ignore[arg-type]
        verifier=verifier,
        fast_gate=_StaticLiteGate(False),
        always_hard_runtime=hard,
    ).solve(query)
    assert outcome.accepted and outcome.route == "hard_direct_accept"
    assert not outcome.local_attempted and outcome.function_evaluations == 11

    drifted = HierarchicalLiteRuntime(
        kinematics=model,
        dls=_FixedTraceDLS(previous, fev=3),  # type: ignore[arg-type]
        verifier=verifier,
        fast_gate=_StaticLiteGate(False),
        always_hard_runtime=_FixedHardRuntime(target_q, stages=("medium", "hard")),
    )
    with pytest.raises(RuntimeError, match="non-HARD"):
        drifted.solve(query)


class _TimingOutcome:
    def __init__(self, timings_ns: dict[str, int]):
        self.timings_ns = timings_ns


def test_stage_breakdown_maps_every_top_level_stage_and_closes_outer_total() -> None:
    outer = 100
    v5 = _TimingOutcome(
        {
            "cheap_feature_ns": 20,
            "gate_ns": 5,
            "local_solver_ns": 15,
            "local_verifier_ns": 5,
            "slow_ns": 40,
        }
    )
    lite = _TimingOutcome(
        {
            "feature_extraction_ns": 10,
            "gate_ns": 5,
            "local_path_ns": 20,
            "robust_path_ns": 50,
        }
    )

    assert np.array_equal(
        _stage_timings(v5, "hierarchical_cghik_v5", outer),
        np.asarray([20, 5, 20, 40, 15]),
    )
    assert np.array_equal(
        _stage_timings(lite, "hierarchical_cghik_v5_lite", outer),
        np.asarray([10, 5, 20, 50, 15]),
    )
    assert np.array_equal(
        _stage_timings(_TimingOutcome({}), "always_local", outer),
        np.asarray([0, 0, 100, 0, 0]),
    )
    assert np.array_equal(
        _stage_timings(_TimingOutcome({}), "always_hard", outer),
        np.asarray([0, 0, 0, 100, 0]),
    )


def _synthetic_benchmark() -> BenchmarkData:
    count, method_count, repeats = 4, len(METHODS), 5
    latency = np.zeros((count, method_count, repeats), dtype=np.int64)
    for method_index in range(method_count):
        latency[:, method_index, :] = (method_index + 1) * 1_000_000
    stages = np.zeros((count, method_count, repeats, len(STAGE_NAMES)), dtype=np.int64)
    stages[:, :, :, -1] = latency
    accepted = np.ones((count, method_count), dtype=bool)
    attempted = np.zeros_like(accepted)
    local_ok = np.zeros_like(accepted)
    lite = METHODS.index("hierarchical_cghik_v5_lite")
    attempted[:2, lite] = True
    local_ok[0, lite] = True
    route = np.full((count, method_count), "hard_direct_accept", dtype="U64")
    return BenchmarkData(
        robot="panda",
        query_sha256=np.asarray([f"{index:064x}" for index in range(count)]),
        category=np.asarray(["id", "id", "large_step", "unreachable"]),
        expected_reachable=np.asarray([True, True, True, False]),
        continuity_feasible=np.asarray([True, True, True, False]),
        latency_samples_ns=latency,
        stage_latency_samples_ns=stages,
        accepted=accepted,
        function_evaluations=np.ones_like(accepted, dtype=np.int64),
        seed_invoked=np.ones_like(accepted),
        local_attempted=attempted,
        local_accepted=local_ok,
        route=route,
        gate_probability=np.full((count, method_count), np.nan),
        executed_stages=np.full((count, method_count), "hard", dtype="U32"),
        command_q=np.zeros((count, method_count, 7), dtype=np.float64),
    )


def test_five_strategy_statistics_are_paired_and_family_counts_conserve() -> None:
    data = _synthetic_benchmark()
    rows = summarize(data)
    paired = paired_summary(data, "always_hard")
    families = family_routes(data)
    breakdown = latency_breakdown(data)

    assert [row["method"] for row in rows] == list(METHODS)
    assert paired["paired_query_count"] == 3
    assert paired["lite_minus_comparator_ms"]["median"] == 3.0
    for family in set(data.category.astype(str)):
        selected = [row for row in families if row["query_family"] == family]
        assert sum(row["count"] for row in selected) == selected[0]["family_queries"]
    assert len(breakdown) == len(METHODS) * len(STAGE_NAMES) * 2


def test_policy_validation_gate_requires_per_query_success_equality() -> None:
    data = _synthetic_benchmark()
    hard = METHODS.index("always_hard")
    lite = METHODS.index("hierarchical_cghik_v5_lite")
    data.accepted[:3, hard] = np.asarray([True, False, True])
    data.accepted[:3, lite] = np.asarray([False, True, True])
    rows = summarize(data)
    assert next(row for row in rows if row["method"] == "always_hard")[
        "verified_success"
    ] == next(
        row for row in rows if row["method"] == "hierarchical_cghik_v5_lite"
    )["verified_success"]

    gate = _pilot_gate(
        data,
        rows,
        {
            "p95_ratio_vs_always_hard_max": 10.0,
            "p50_ratio_vs_counterfactual_cghik_v4_max_exclusive": 10.0,
            "learned_seed_ensemble_invocation_rate_max_exclusive": 2.0,
        },
    )

    assert gate["success_mismatch_count_vs_always_hard"] == 2
    assert not gate["checks"]["per_query_verified_success_equal_always_hard"]
    assert not gate["all_pass"]


def test_five_by_five_latin_orders_place_every_method_once_per_position() -> None:
    orders = latin_method_orders(METHODS, 1715)
    assert len(orders) == len(METHODS) == 5
    assert all(set(order) == set(METHODS) for order in orders)
    for position in range(5):
        assert {order[position] for order in orders} == set(METHODS)


def test_config_freezes_roles_methods_goals_and_rejects_test_paths() -> None:
    config = load_config(ROOT / "configs" / "hierarchical_v5_lite_pilot.yaml")
    validate_config(config, workspace=ROOT)
    assert tuple(config["strategies"]) == METHODS
    assert config["pilot_goals"][
        "learned_seed_ensemble_invocation_rate_max_exclusive"
    ] < 1.0
    assert config["pilot_goals"][
        "per_query_verified_success_equal_always_hard"
    ] is True

    changed = deepcopy(config)
    changed["roles"]["policy_validation"] = "test_v4_queries"
    with pytest.raises(ValueError, match="development roles"):
        validate_config(changed, workspace=ROOT)

    changed = deepcopy(config)
    changed["output_root"] = "../outputs/test_v4"
    with pytest.raises(ValueError, match="formal-test"):
        validate_config(changed, workspace=ROOT)

    invalid_calibration_changes = (
        ("include_always_robust_sentinel", False),
        ("threshold_selection_role", "policy_validation_queries"),
        ("retune_after_exact_runtime_check", True),
    )
    for key, value in invalid_calibration_changes:
        changed = deepcopy(config)
        changed["calibration"][key] = value
        with pytest.raises(ValueError, match="calibration selection contract"):
            validate_config(changed, workspace=ROOT)

    changed = deepcopy(config)
    changed["calibration"]["threshold_probability_grid"] = [0.0, 0.5, 0.99]
    with pytest.raises(ValueError, match="calibration selection contract"):
        validate_config(changed, workspace=ROOT)
