from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from confik.config import load_config
from confik.hierarchical_v5.features import (
    CHEAP_FEATURE_DIM,
    CHEAP_FEATURE_NAMES,
    prepare_cheap_features,
)
from confik.hierarchical_v5.model import (
    CALIBRATION_ROLE,
    TRAIN_ROLE,
    FastGateOutput,
    FastGatePredictor,
    FastGateTrainingConfig,
    export_exact_torchscript,
    load_exact_torchscript,
    numerical_equivalence,
)
from confik.hierarchical_v5.pilot import (
    METHODS,
    BenchmarkData,
    LocalDevelopmentMeasurements,
    _benefit_labels,
    _forbidden_easy_stage_count,
    _verified_release_inputs,
    latin_method_orders,
    local_success_label_from_record,
    summarize_benchmark,
    validate_config,
)
from confik.hierarchical_v5.policy import FastGatePolicy, FastGatePolicyConfig
from confik.hierarchical_v5.runtime import (
    FastGateDecision as RuntimeFastGateDecision,
    HierarchicalRuntime,
)
from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.solvers.dls import AdaptiveDLS, DLSConfig
from confik.solvers.verifier import SolutionVerifier
from confik.types import IKQuery, SolveTrace


ROOT = Path(__file__).resolve().parents[1]
TOY_URDF = Path(__file__).parent / "assets" / "toy_arm.urdf"


def _toy_model() -> URDFKinematics:
    return URDFKinematics.from_file(TOY_URDF)


def test_cheap_feature_contract_is_exactly_seven_dimensional_and_seed_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Poison the legacy learned-seed encoder.  First-level feature extraction
    # must remain usable without invoking it.
    import confik.models.seed as seed_module

    def forbidden_seed_encoder(*_args, **_kwargs):
        raise AssertionError("learned seed encoding reached the cheap feature path")

    monkeypatch.setattr(seed_module, "encode_seed_inputs", forbidden_seed_encoder)
    model = _toy_model()
    dls = AdaptiveDLS(model)
    q = np.zeros(model.nq, dtype=np.float64)
    query = IKQuery(model.forward(q), q, dt=0.02)

    prepared = prepare_cheap_features(model, dls, query)

    assert CHEAP_FEATURE_DIM == len(CHEAP_FEATURE_NAMES) == 7
    assert prepared.features.shape == (7,)
    assert prepared.features.dtype == np.float32
    assert prepared.features.flags.c_contiguous
    assert np.all(np.isfinite(prepared.features))
    assert CHEAP_FEATURE_NAMES == (
        "target_position_step",
        "target_orientation_step",
        "previous_joint_limit_margin_min",
        "previous_jacobian_sigma_min",
        "previous_jacobian_condition_number",
        "one_step_dls_max_joint_update",
        "estimated_velocity_limit_utilization_max",
    )


def test_local_success_label_requires_unescalated_verified_easy_stage() -> None:
    assert local_success_label_from_record(
        {
            "entry_action": "easy",
            "verified_success": True,
            "executed_stages": ["easy"],
        }
    )
    assert not local_success_label_from_record(
        {
            "entry_action": "easy",
            "verified_success": True,
            "executed_stages": ["easy", "medium", "hard"],
        }
    )
    assert not local_success_label_from_record(
        {
            "entry_action": "easy",
            "verified_success": False,
            "executed_stages": ["easy"],
        }
    )
    assert not local_success_label_from_record(
        {
            "entry_action": "hard",
            "verified_success": True,
            "executed_stages": ["hard"],
        }
    )


def test_latency_benefit_uses_paired_direct_robust_timings_and_requires_them() -> None:
    measurements = LocalDevelopmentMeasurements(
        features=np.zeros((2, CHEAP_FEATURE_DIM), dtype=np.float32),
        local_success=np.asarray([True, True]),
        local_total_samples_ns=np.asarray(
            [[100, 100, 100, 100, 100], [100, 100, 100, 100, 100]],
            dtype=np.int64,
        ),
        local_function_evaluations=np.ones(2, dtype=np.int64),
        direct_robust_total_samples_ns=np.asarray(
            [[250, 250, 250, 250, 250], [50, 50, 50, 50, 50]],
            dtype=np.int64,
        ),
        direct_robust_verified_success=np.ones(2, dtype=bool),
        direct_robust_function_evaluations=np.ones(2, dtype=np.int64),
    )

    # The two rows differ only in their paired direct-robust timings.  A
    # positive label is therefore possible only if that measured field is the
    # comparator used by the implementation.
    assert np.array_equal(
        _benefit_labels(measurements, gate_overhead_ns=10.0),
        np.asarray([True, False]),
    )

    missing_direct_timings = LocalDevelopmentMeasurements(
        features=measurements.features,
        local_success=measurements.local_success,
        local_total_samples_ns=measurements.local_total_samples_ns,
        local_function_evaluations=measurements.local_function_evaluations,
        direct_robust_total_samples_ns=np.empty((2, 0), dtype=np.int64),
        direct_robust_verified_success=measurements.direct_robust_verified_success,
        direct_robust_function_evaluations=(
            measurements.direct_robust_function_evaluations
        ),
    )
    with pytest.raises(ValueError, match="paired direct-robust timings"):
        _benefit_labels(missing_direct_timings, gate_overhead_ns=10.0)


def test_fast_gate_torchscript_is_numerically_and_route_equivalent(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(96, CHEAP_FEATURE_DIM)).astype(np.float32)
    local = (np.arange(len(features)) % 2).astype(np.float32)
    benefit = ((np.arange(len(features)) // 2) % 2).astype(np.float32)
    predictor = FastGatePredictor(
        FastGateTrainingConfig(epochs=2, batch_size=24, seed=41), device="cpu"
    )
    predictor.fit(features, local, benefit, role=TRAIN_ROLE)
    predictor.calibrate(features, local, benefit, role=CALIBRATION_ROLE)

    artifact = tmp_path / "fast_gate.ts"
    export_exact_torchscript(predictor, artifact)
    module = load_exact_torchscript(artifact, device="cpu")
    result = numerical_equivalence(
        predictor,
        module,
        features[:24],
        success_threshold=0.75,
        benefit_threshold=0.75,
        atol=1.0e-10,
    )

    assert result["passed"]
    assert result["route_match_rate"] == 1.0
    assert result["max_absolute_probability_error"] <= 1.0e-10


class _StaticFastBackend:
    def __init__(self, success: float, benefit: float):
        self.output = FastGateOutput(0.0, 0.0, success, benefit)

    def predict_one(self, features: np.ndarray) -> FastGateOutput:
        assert features.shape == (CHEAP_FEATURE_DIM,)
        return self.output


@pytest.mark.parametrize(
    ("success", "benefit", "expected"),
    (
        (0.90, 0.80, "fast"),
        (0.79, 0.80, "robust"),
        (0.90, 0.69, "robust"),
        (0.79, 0.69, "robust"),
    ),
)
def test_fast_policy_requires_both_calibrated_thresholds(
    success: float, benefit: float, expected: str
) -> None:
    policy = FastGatePolicy(
        _StaticFastBackend(success, benefit),
        FastGatePolicyConfig(
            local_success_threshold=0.80,
            latency_benefit_threshold=0.70,
        ),
    )
    decision = policy.decide(np.zeros(CHEAP_FEATURE_DIM, dtype=np.float32))
    assert decision.action == expected
    assert decision.choose_fast is (expected == "fast")


class _FixedTraceDLS:
    """DLS-compatible test double; feature extraction still uses real geometry."""

    def __init__(self, model: URDFKinematics, q_result: np.ndarray, fev: int):
        self.config = DLSConfig()
        self._damping = AdaptiveDLS(model, self.config)
        self.q_result = np.asarray(q_result, dtype=np.float64)
        self.fev = int(fev)

    def damping(self, sigma_min: float) -> float:
        return self._damping.damping(sigma_min)

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


class _AlwaysFastGate:
    def decide(self, features: np.ndarray) -> RuntimeFastGateDecision:
        assert features.shape == (CHEAP_FEATURE_DIM,)
        return RuntimeFastGateDecision(True, 0.99, 0.99, "synthetic_fast")


class _SlowRuntime:
    def __init__(
        self,
        q: np.ndarray,
        *,
        accepted: bool = True,
        stages: tuple[str, ...] = ("medium", "hard"),
        fev: int = 11,
    ):
        self.q = np.asarray(q, dtype=np.float64)
        self.accepted = accepted
        self.stages = stages
        self.fev = fev
        self.calls = 0

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        del query
        self.calls += 1
        return ProfiledOutcome(
            q=self.q.copy() if self.accepted else None,
            accepted=self.accepted,
            entry_action="medium",
            executed_stages=self.stages,
            risk_probabilities=np.asarray([0.0, 1.0, 0.0, 0.0]),
            risk_score=0.0,
            function_evaluations=self.fev,
            iterations=4,
            fallback_used=False,
            verification_reasons=(),
            reject_reason="" if self.accepted else "synthetic_failure",
            candidate_count=1,
            timings_ns={"total_end_to_end_ns": 1},
        )


def test_fast_verifier_failure_recovers_without_easy_and_accumulates_fev() -> None:
    model = _toy_model()
    q_previous = np.zeros(model.nq, dtype=np.float64)
    q_target = np.asarray([0.02, -0.01], dtype=np.float64)
    query = IKQuery(model.forward(q_target), q_previous, dt=0.02)
    # The synthetic local solver claims convergence but returns q_previous;
    # the shared verifier must reject it on pose error.
    dls = _FixedTraceDLS(model, q_previous, fev=3)
    slow = _SlowRuntime(q_target, fev=11)
    runtime = HierarchicalRuntime(
        kinematics=model,
        dls=dls,  # type: ignore[arg-type]
        verifier=SolutionVerifier(model),
        fast_gate=_AlwaysFastGate(),
        slow_runtime=slow,
        fast_iterations=1,
    )

    outcome = runtime.solve(query)

    assert outcome.accepted
    assert outcome.route == "fast_fail_robust_recovery"
    assert outcome.local_attempted and not outcome.local_accepted
    assert outcome.recovered_after_fast_failure
    assert outcome.learned_seed_ensemble_invoked
    assert slow.calls == 1
    assert outcome.function_evaluations == 3 + 11
    assert outcome.executed_stages == ("local_fast", "medium", "hard")
    assert "easy" not in outcome.executed_stages
    assert outcome.local_verification_reasons

    forbidden = HierarchicalRuntime(
        kinematics=model,
        dls=dls,  # type: ignore[arg-type]
        verifier=SolutionVerifier(model),
        fast_gate=_AlwaysFastGate(),
        slow_runtime=_SlowRuntime(q_target, stages=("easy", "hard")),
        fast_iterations=1,
    )
    with pytest.raises(RuntimeError, match="forbidden EASY"):
        forbidden.solve(query)


def test_fast_verified_return_never_invokes_the_learned_slow_path() -> None:
    model = _toy_model()
    q_previous = np.zeros(model.nq, dtype=np.float64)
    q_target = np.asarray([0.02, -0.01], dtype=np.float64)
    query = IKQuery(model.forward(q_target), q_previous, dt=0.02)
    slow = _SlowRuntime(q_target)
    runtime = HierarchicalRuntime(
        kinematics=model,
        dls=_FixedTraceDLS(model, q_target, fev=3),  # type: ignore[arg-type]
        verifier=SolutionVerifier(model),
        fast_gate=_AlwaysFastGate(),
        slow_runtime=slow,
        fast_iterations=1,
    )

    outcome = runtime.solve(query)

    assert outcome.accepted and outcome.fast_path_hit
    assert outcome.route == "fast_accept"
    assert not outcome.learned_seed_ensemble_invoked
    assert slow.calls == 0
    assert outcome.candidate_count == 0
    assert outcome.executed_stages == ("local_fast",)


def test_five_strategy_latin_rotation_is_complete_and_deterministic() -> None:
    assert METHODS == (
        "always_local",
        "fixed_easy_cascade",
        "always_hard",
        "counterfactual_cghik_v4",
        "hierarchical_cghik_v5",
    )
    orders = latin_method_orders(METHODS, seed=7319)
    assert orders == latin_method_orders(METHODS, seed=7319)
    assert len(orders) == len(METHODS) == 5
    assert all(set(order) == set(METHODS) for order in orders)
    for position in range(len(METHODS)):
        assert {order[position] for order in orders} == set(METHODS)


def _synthetic_benchmark() -> BenchmarkData:
    count = 4
    method_count = len(METHODS)
    repeats = 5
    latency = np.empty((count, method_count, repeats), dtype=np.int64)
    for method_index in range(method_count):
        latency[:, method_index, :] = (method_index + 1) * 1_000_000
    accepted = np.ones((count, method_count), dtype=bool)
    fev = np.ones((count, method_count), dtype=np.int64)
    v5 = METHODS.index("hierarchical_cghik_v5")
    fev[:, v5] = np.asarray([3, 14, 3, 11])
    seed_invoked = np.ones((count, method_count), dtype=bool)
    seed_invoked[:, METHODS.index("always_local")] = False
    seed_invoked[:, v5] = np.asarray([False, True, False, True])
    attempted = np.zeros((count, method_count), dtype=bool)
    local_accepted = np.zeros((count, method_count), dtype=bool)
    attempted[:, METHODS.index("always_local")] = True
    attempted[:, v5] = np.asarray([True, True, True, False])
    local_accepted[:, v5] = np.asarray([True, False, True, False])
    route = np.full((count, method_count), "baseline", dtype="U64")
    route[:, v5] = np.asarray(
        [
            "fast_accept",
            "fast_fail_robust_recovery",
            "fast_accept",
            "robust_direct_accept",
        ]
    )
    return BenchmarkData(
        robot="toy",
        methods=METHODS,
        query_sha256=np.asarray([f"{index:064x}" for index in range(count)]),
        category=np.asarray(["id", "id", "near_limit", "hard_valid"]),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        latency_samples_ns=latency,
        accepted=accepted,
        function_evaluations=fev,
        seed_invoked=seed_invoked,
        local_attempted=attempted,
        local_accepted=local_accepted,
        route=route,
        gate_local_success_probability=np.full((count, method_count), np.nan),
        gate_latency_benefit_probability=np.full((count, method_count), np.nan),
    )


def test_benchmark_summary_uses_query_medians_and_exact_fast_path_denominators() -> None:
    data = _synthetic_benchmark()
    rows = summarize_benchmark(data)
    row = next(item for item in rows if item["method"] == "hierarchical_cghik_v5")

    assert row["feasible_queries"] == 4
    assert row["verified_success"] == 1.0
    assert row["p50_ms"] == row["p95_ms"] == row["p99_ms"] == 5.0
    assert row["mean_fev"] == pytest.approx(7.75)
    assert row["learned_seed_ensemble_invocation_rate"] == 0.5
    assert row["fast_path_attempt_rate"] == 0.75
    assert row["fast_path_hit_rate"] == 0.5
    assert row["fast_path_precision"] == pytest.approx(2.0 / 3.0)
    assert row["fast_path_failure_recovery_rate"] == 1.0

    duplicate = deepcopy(data)
    object.__setattr__(
        duplicate,
        "query_sha256",
        np.asarray(["0" * 64] * len(data.query_sha256)),
    )
    with pytest.raises(ValueError, match="not unique"):
        summarize_benchmark(duplicate)


def test_no_easy_audit_reads_executed_stage_trace() -> None:
    data = deepcopy(_synthetic_benchmark())
    executed_stages = np.full(data.accepted.shape, "hard", dtype="U64")
    v5 = METHODS.index("hierarchical_cghik_v5")
    executed_stages[:, v5] = np.asarray(
        [
            "local_fast|medium|hard",
            "medium|easy:recovery|hard",
            "EASY|hard",
            "medium|easy_recovery|hard",
        ]
    )
    object.__setattr__(data, "executed_stages", executed_stages)

    # Route names deliberately contain no EASY token.  The two violations can
    # only be found by parsing the recorded stage trace (case-insensitively and
    # before any stage metadata suffix).
    assert not any("easy" in value.lower() for value in data.route[:, v5])
    assert _forbidden_easy_stage_count(data) == 2


def test_panda_seed17_sealed_v3_v4_loaded_artifacts_match_their_hashes() -> None:
    release_v3_root = ROOT / "outputs" / "release_v3_locked"
    release_v4_root = ROOT / "outputs" / "release_v4_locked"
    verified = _verified_release_inputs(
        workspace=ROOT,
        release_v3_root=release_v3_root,
        release_v4_root=release_v4_root,
        robot="panda",
    )

    assert set(verified["release_v3_loaded_artifacts"]) == {
        "torchscript",
        "normalization",
        "runtime_spec",
        "solver_metadata",
        "seed_bank",
    }
    assert set(verified["release_v4_loaded_artifacts"]) == {
        "exact_predictor",
        "policy",
    }
    loaded = (
        *verified["release_v3_loaded_artifacts"].values(),
        *verified["release_v4_loaded_artifacts"].values(),
    )
    for descriptor in loaded:
        path = Path(descriptor["path"])
        assert path.is_file()
        assert "panda" in path.parts
        assert descriptor["size"] == path.stat().st_size
        assert descriptor["sha256"] == sha256(path.read_bytes()).hexdigest()

    assert "seed17" in Path(
        verified["release_v3_loaded_artifacts"]["torchscript"]["path"]
    ).parts
    assert Path(
        verified["release_v4_loaded_artifacts"]["exact_predictor"]["path"]
    ).name == "exact_v4_predictor.ts"


def test_config_is_development_only_and_rejects_test_named_roles_and_paths() -> None:
    config = load_config(ROOT / "configs" / "hierarchical_v5_pilot.yaml")
    validate_config(config, workspace=ROOT)
    assert config["timing"]["warmup_iterations"] == 200
    assert config["timing"]["repeats"] == 5
    assert config["fast_path"]["fast_iterations"] == 1
    assert config["model"]["hidden_sizes"] == [16, 16]

    bad_role = deepcopy(config)
    bad_role["roles"]["policy_validation"] = "test_v4_queries"
    with pytest.raises(ValueError):
        validate_config(bad_role, workspace=ROOT)

    bad_path = deepcopy(config)
    bad_path["bulk_root"] = "../outputs/test_v4_seed17"
    with pytest.raises(ValueError, match="must not contain a test"):
        validate_config(bad_path, workspace=ROOT)
