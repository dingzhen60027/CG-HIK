from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from confik.hierarchical_v5.runtime import LocalSolveOutcome
from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.temporal_v6.policy import TemporalPolicyConfig, TemporalPolicyController
from confik.temporal_v6.pilot import (
    CalibrationData,
    FAMILIES,
    TrajectoryRole,
    _expected_grid,
    generate_development_roles,
    select_calibration_policy,
    validate_config,
)
from confik.temporal_v6.runtime import BoundTemporalStream, TemporalCGHIKRuntime
from confik.temporal_v6.state import TemporalMode, TemporalState
from confik.config import load_config, resolve_path
from confik.data.datasets import QueryDataset
from confik.types import IKQuery, SolveTrace


ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def _hard_outcome(query: IKQuery, *, accepted: bool = True, stage: str = "hard") -> ProfiledOutcome:
    return ProfiledOutcome(
        q=query.previous_q.copy() if accepted else None,
        accepted=accepted,
        entry_action="hard",
        executed_stages=(stage,),
        risk_probabilities=np.array([0.0, 0.0, 1.0, 0.0]),
        risk_score=1.0,
        function_evaluations=7,
        iterations=5,
        fallback_used=False,
        verification_reasons=(),
        reject_reason="" if accepted else "failed",
        candidate_count=5,
        timings_ns={"total_end_to_end_ns": 10},
    )


class HardStub:
    def __init__(
        self,
        model: URDFKinematics,
        verifier: SolutionVerifier,
        accepted: list[bool] | None = None,
        log: list[str] | None = None,
    ):
        self.kinematics = model
        self._timed_verifier = SimpleNamespace(verifier=verifier)
        self.accepted = list(accepted or [True])
        self.calls = 0
        self.log = log

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        if self.log is not None:
            self.log.append("hard")
        value = self.accepted[min(self.calls, len(self.accepted) - 1)]
        self.calls += 1
        return _hard_outcome(query, accepted=value)


class PredictorStub:
    def __init__(self, probability: float = 0.99, log: list[str] | None = None):
        self.probability = probability
        self.calls = 0
        self.log = log

    def predict_one(self, features: np.ndarray):
        assert features.dtype == np.float32
        if self.log is not None:
            self.log.append("predictor")
        self.calls += 1
        return {"local_success_probability": self.probability}


def _query(model: URDFKinematics) -> IKQuery:
    q = np.zeros(model.nq, dtype=np.float64)
    return IKQuery(model.forward(q), q)


def test_temporal_state_is_immutable_and_rejects_invalid_modes() -> None:
    state = TemporalState()
    with pytest.raises(Exception):
        state.frames_seen = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        TemporalState(mode=TemporalMode.ROBUST, frames_seen=1, robust_age=0)
    with pytest.raises(ValueError):
        TemporalState(mode=TemporalMode.LOCAL, frames_seen=1, robust_age=1)


def test_schedule_holds_then_checks_every_m_and_reenters_next_frame() -> None:
    policy = TemporalPolicyController(TemporalPolicyConfig(3, 2, 2, 0.9))
    state = policy.initial_state()
    initial = policy.plan(state)
    state = policy.transition(
        state,
        initial,
        local_accepted=None,
        hard_accepted=True,
        probe_executed=False,
        local_success_probability=None,
    ).state_after
    assert state.mode is TemporalMode.LOCAL

    local = policy.plan(state)
    transition = policy.transition(
        state,
        local,
        local_accepted=False,
        hard_accepted=True,
        probe_executed=False,
        local_success_probability=None,
    )
    state = transition.state_after
    assert transition.switch_kind == "local_to_robust"
    assert state.robust_age == 1

    observed = []
    for expected_r in (2, 3, 4, 5):
        plan = policy.plan(state)
        observed.append((plan.robust_frame_number, plan.probe_after_hard))
        probability = 0.95 if plan.probe_after_hard else None
        transition = policy.transition(
            state,
            plan,
            local_accepted=None,
            hard_accepted=True,
            probe_executed=plan.probe_after_hard,
            local_success_probability=probability,
        )
        state = transition.state_after
    assert observed == [(2, False), (3, True), (4, False), (5, True)]
    assert state.mode is TemporalMode.LOCAL
    assert transition.switch_kind == "robust_to_local"
    assert transition.recovery_delay_frames == 5


def test_noncheck_frames_preserve_streak_and_failed_hard_clears_it() -> None:
    policy = TemporalPolicyController(TemporalPolicyConfig(2, 2, 3, 0.8))
    state = TemporalState(
        mode=TemporalMode.ROBUST,
        frames_seen=3,
        robust_age=1,
        reentry_high_streak=0,
    )
    plan = policy.plan(state)
    assert plan.probe_after_hard
    state = policy.transition(
        state,
        plan,
        local_accepted=None,
        hard_accepted=True,
        probe_executed=True,
        local_success_probability=0.9,
    ).state_after
    assert state.reentry_high_streak == 1
    plan = policy.plan(state)
    assert not plan.probe_after_hard
    state = policy.transition(
        state,
        plan,
        local_accepted=None,
        hard_accepted=True,
        probe_executed=False,
        local_success_probability=None,
    ).state_after
    assert state.reentry_high_streak == 1
    plan = policy.plan(state)
    assert plan.probe_after_hard
    state = policy.transition(
        state,
        plan,
        local_accepted=None,
        hard_accepted=False,
        probe_executed=False,
        local_success_probability=None,
    ).state_after
    assert state.reentry_high_streak == 0


def test_unscheduled_robust_hard_failure_also_clears_streak() -> None:
    policy = TemporalPolicyController(TemporalPolicyConfig(2, 2, 3, 0.8))
    state = TemporalState(
        mode=TemporalMode.ROBUST,
        frames_seen=4,
        robust_age=2,
        reentry_high_streak=1,
    )
    plan = policy.plan(state)
    assert not plan.probe_after_hard
    state = policy.transition(
        state,
        plan,
        local_accepted=None,
        hard_accepted=False,
        probe_executed=False,
        local_success_probability=None,
    ).state_after
    assert state.reentry_high_streak == 0


def test_runtime_bootstraps_hard_then_local_without_per_frame_gate() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, [True])
    predictor = PredictorStub()
    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=predictor,
        always_hard_runtime=hard,
        policy_config=TemporalPolicyConfig(5, 3, 2, 0.9),
    )
    query = _query(model)
    first = runtime.step(query, runtime.initial_state())
    assert first.route == "bootstrap_hard_accept"
    assert first.learned_seed_ensemble_invoked
    assert predictor.calls == 0
    second = runtime.step(query, first.state_after)
    assert second.route == "local_accept"
    assert not second.learned_seed_ensemble_invoked
    assert predictor.calls == 0
    assert first.state_before.mode is TemporalMode.INIT
    assert first.state_after.mode is TemporalMode.LOCAL


def test_robust_probe_runs_after_hard_and_only_schedules_next_frame_local() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    order: list[str] = []
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, [False, True], order)
    predictor = PredictorStub(0.99, order)
    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=predictor,
        always_hard_runtime=hard,
        policy_config=TemporalPolicyConfig(2, 1, 1, 0.9),
    )
    query = _query(model)
    first = runtime.step(query, runtime.initial_state())
    assert first.state_after.mode is TemporalMode.ROBUST
    second = runtime.step(query, first.state_after)
    assert order == ["hard", "hard", "predictor"]
    assert second.occupancy_mode == "robust"
    assert second.state_after.mode is TemporalMode.LOCAL
    assert second.reentry_probe_scheduled and second.reentry_probe_executed
    assert second.route == "robust_hard_accept"


def test_failed_scheduled_hard_skips_predictor_and_resets_streak() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, [False])
    predictor = PredictorStub()
    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=predictor,
        always_hard_runtime=hard,
        policy_config=TemporalPolicyConfig(2, 1, 2, 0.8),
    )
    query = _query(model)
    state = TemporalState(
        mode=TemporalMode.ROBUST,
        frames_seen=4,
        robust_age=1,
        reentry_high_streak=1,
    )
    result = runtime.step(query, state)
    assert result.reentry_probe_scheduled
    assert not result.reentry_probe_executed
    assert predictor.calls == 0
    assert result.state_after.reentry_high_streak == 0


def test_local_verifier_failure_invokes_hard_and_accumulates_fev() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, [True])
    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=PredictorStub(),
        always_hard_runtime=hard,
        policy_config=TemporalPolicyConfig(5, 1, 1, 0.8),
    )
    rejected_trace = SolveTrace(
        q=np.zeros(model.nq),
        converged=True,
        iterations=1,
        position_error=0.0,
        orientation_error=0.0,
        function_evaluations=3,
    )
    runtime.local_runtime.solve = lambda query: LocalSolveOutcome(  # type: ignore[method-assign]
        q=None,
        accepted=False,
        trace=rejected_trace,
        verification=None,
        verification_reasons=("velocity_limit",),
        function_evaluations=3,
        iterations=1,
        timings_ns={"local_solver_ns": 2, "local_verifier_ns": 1, "total_ns": 3},
    )
    state = TemporalState(mode=TemporalMode.LOCAL, frames_seen=1)
    result = runtime.step(_query(model), state)
    assert result.route == "local_fail_hard_recovery"
    assert result.function_evaluations == 10
    assert result.executed_stages == ("local", "hard")
    assert result.state_after.mode is TemporalMode.ROBUST


def test_bound_stream_reset_snapshot_and_restore_prevent_trajectory_leakage() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    core = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=PredictorStub(),
        always_hard_runtime=HardStub(model, verifier, [True]),
        policy_config=TemporalPolicyConfig(5, 1, 1, 0.8),
    )
    stream = BoundTemporalStream(core)
    first = stream.solve(_query(model))
    checkpoint = stream.snapshot()
    assert checkpoint.mode is TemporalMode.LOCAL
    stream.solve(_query(model))
    stream.restore(checkpoint)
    assert stream.snapshot() == checkpoint
    stream.reset()
    assert stream.snapshot() == TemporalState()
    assert first.state_before == TemporalState()


def test_fixed_hard_contract_rejects_nonhard_stage() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")

    verifier = SolutionVerifier(model)

    class BadHard(HardStub):
        def solve(self, query: IKQuery) -> ProfiledOutcome:
            return _hard_outcome(query, stage="medium")

    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        predictor=PredictorStub(),
        always_hard_runtime=BadHard(model, verifier),
        policy_config=TemporalPolicyConfig(5, 1, 1, 0.8),
    )
    with pytest.raises(RuntimeError, match="non-HARD"):
        runtime.step(_query(model), TemporalState())


def test_pilot_config_freezes_grid_roles_methods_and_goals() -> None:
    path = Path(__file__).parents[1] / "configs" / "temporal_v6_pilot.yaml"
    config = load_config(path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    assert len(_expected_grid(config)) == 144

    changed = json.loads(json.dumps(config))
    changed["roles"]["policy_validation"] = "formal_test"
    with pytest.raises(ValueError):
        validate_config(changed, workspace=workspace)

    changed = json.loads(json.dumps(config))
    changed["pilot_goals"]["learned_seed_invocation_rate_max"] = 0.8
    with pytest.raises(ValueError, match="pilot goals"):
        validate_config(changed, workspace=workspace)

    changed = json.loads(json.dumps(config))
    changed["temporal_grid"]["hold_frames"] = [6, 11, 21, 31]
    with pytest.raises(ValueError, match="grid axis"):
        validate_config(changed, workspace=workspace)


def test_generated_development_split_is_whole_trajectory_and_family_stratified() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    calibration, policy, manifest = generate_development_roles(
        model,
        robot="toy",
        paths_per_family_pool=2,
        paths_per_family_per_role=1,
        steps=12,
        pool_seed=7021,
        split_seed=7022,
        dt=0.02,
    )
    assert calibration.count == policy.count == 4 * 12
    assert len(calibration.trajectory_order) == len(policy.trajectory_order) == 4
    assert not (set(calibration.trajectory_order) & set(policy.trajectory_order))
    assert manifest["trajectory_uid_overlap_count"] == 0
    for role in (calibration, policy):
        for family in FAMILIES:
            rows = role.dataset.category == family
            assert int(np.sum(rows)) == 12
            assert len(np.unique(role.trajectory_uid[rows])) == 1
        for _, rows in role.groups():
            assert np.array_equal(role.dataset.time_index[rows], np.arange(12))


def _selection_role(frame_count: int = 20) -> TrajectoryRole:
    previous = np.zeros((frame_count, 1), dtype=np.float64)
    dataset = QueryDataset(
        previous_q=previous,
        target_position=np.zeros((frame_count, 3), dtype=np.float64),
        target_rotation=np.repeat(np.eye(3)[None, :, :], frame_count, axis=0),
        reference_q=previous.copy(),
        category=np.asarray(["trajectory_smooth"] * frame_count),
        expected_reachable=np.ones(frame_count, dtype=bool),
        continuity_feasible=np.ones(frame_count, dtype=bool),
        trajectory_id=np.repeat(np.arange(2), frame_count // 2),
        time_index=np.tile(np.arange(frame_count // 2), 2),
    )
    uids = np.asarray(
        ["a"] * (frame_count // 2) + ["b"] * (frame_count // 2), dtype="U64"
    )
    return TrajectoryRole("toy", "trajectory_calibration", dataset, uids, ("a", "b"))


def test_calibration_selection_is_completion_then_p95_seed_and_p50() -> None:
    role = _selection_role()
    configs = (
        TemporalPolicyConfig(5, 1, 1, 0.8),
        TemporalPolicyConfig(10, 1, 1, 0.9),
        TemporalPolicyConfig(20, 3, 2, 0.95),
        TemporalPolicyConfig(30, 5, 3, 0.99),
    )
    n = role.count
    accepted = np.ones((4, n), dtype=bool)
    # Candidate 0 fails one complete trajectory and is ineligible despite speed.
    accepted[0, 0] = False
    latency = np.asarray(
        [
            [1] * n,
            [2] * 18 + [100, 100],
            [3] * 18 + [100, 100],
            [1] * 18 + [100, 100],
        ],
        dtype=np.int64,
    )
    seed = np.asarray(
        [[False] * n, [True] * n, [False] * n, [False] * n], dtype=bool
    )
    zeros_b = np.zeros((4, n), dtype=bool)
    zeros_i = np.zeros((4, n), dtype=np.int32)
    states = np.zeros((4, n), dtype=np.int8)
    strings = np.full((4, n), "", dtype="U40")
    data = CalibrationData(
        configs=configs,
        latency_ns=latency,
        accepted=accepted,
        function_evaluations=zeros_i,
        seed_invoked=seed,
        gate_invoked=zeros_b,
        local_attempted=zeros_b,
        local_accepted=zeros_b,
        hard_accepted=accepted.copy(),
        command_q=np.zeros((4, n, 1), dtype=np.float64),
        candidate_order_index=np.zeros((4, n), dtype=np.uint16),
        state_before=states,
        state_after=states,
        robust_frame_number=np.zeros((4, n), dtype=np.int16),
        probe_scheduled=zeros_b,
        reentry_probability=np.full((4, n), np.nan),
        streak_before=np.zeros((4, n), dtype=np.int16),
        streak_after=np.zeros((4, n), dtype=np.int16),
        mode_switched=zeros_b,
        switch_kind=np.full((4, n), "", dtype="U24"),
        route=strings,
        trajectory_attempt_index=np.zeros(n, dtype=np.int16),
        hard_reference_accepted=np.ones(n, dtype=bool),
        hard_reference_completion=np.ones(2, dtype=bool),
    )
    selected_index, selected, report = select_calibration_policy(data, role)
    assert selected_index == 3
    assert selected == configs[3]
    assert report["eligible_candidate_count"] == 3
    assert report["selected"]["p95_ns"] == report["candidates"][2]["p95_ns"]
    assert report["selected"]["learned_seed_ensemble_invocation_rate"] == 0.0
    assert report["selected"]["p50_ns"] < report["candidates"][2]["p50_ns"]


def test_calibration_selection_uses_conservative_tie_break_and_fails_closed() -> None:
    role = _selection_role()
    configs = (
        TemporalPolicyConfig(5, 1, 1, 0.8),
        TemporalPolicyConfig(30, 5, 3, 0.99),
    )
    n = role.count
    accepted = np.ones((2, n), dtype=bool)
    zeros_b = np.zeros((2, n), dtype=bool)
    zeros_i = np.zeros((2, n), dtype=np.int32)
    states = np.zeros((2, n), dtype=np.int8)
    common = dict(
        configs=configs,
        latency_ns=np.full((2, n), 10, dtype=np.int64),
        accepted=accepted,
        function_evaluations=zeros_i,
        seed_invoked=zeros_b,
        gate_invoked=zeros_b,
        local_attempted=zeros_b,
        local_accepted=zeros_b,
        hard_accepted=accepted.copy(),
        command_q=np.zeros((2, n, 1), dtype=np.float64),
        candidate_order_index=np.zeros((2, n), dtype=np.uint16),
        state_before=states,
        state_after=states,
        robust_frame_number=np.zeros((2, n), dtype=np.int16),
        probe_scheduled=zeros_b,
        reentry_probability=np.full((2, n), np.nan),
        streak_before=np.zeros((2, n), dtype=np.int16),
        streak_after=np.zeros((2, n), dtype=np.int16),
        mode_switched=zeros_b,
        switch_kind=np.full((2, n), "", dtype="U24"),
        route=np.full((2, n), "", dtype="U40"),
        trajectory_attempt_index=np.zeros(n, dtype=np.int16),
        hard_reference_accepted=np.ones(n, dtype=bool),
        hard_reference_completion=np.ones(2, dtype=bool),
    )
    data = CalibrationData(**common)
    index, selected, _ = select_calibration_policy(data, role)
    assert index == 1
    assert selected == configs[1]

    failed = dict(common)
    failed["accepted"] = np.zeros((2, n), dtype=bool)
    with pytest.raises(RuntimeError, match="policy-validation remains unopened"):
        select_calibration_policy(CalibrationData(**failed), role)
