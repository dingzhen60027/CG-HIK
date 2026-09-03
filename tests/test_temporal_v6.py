from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from confik.config import load_config, resolve_path
from confik.data.datasets import QueryDataset
from confik.hierarchical_v5.runtime import LocalSolveOutcome
from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.temporal_v6.pilot import (
    CALIBRATION_ARMS,
    H_VALUES,
    METHODS,
    CalibrationData,
    TrajectoryView,
    benchmark_policy_validation,
    pilot_gate,
    select_hold_frames,
    validate_config,
)
from confik.temporal_v6.runtime import BoundTemporalStream, TemporalCGHIKRuntime
from confik.temporal_v6.state import (
    TemporalMode,
    TemporalPolicyConfig,
    TemporalPolicyController,
    TemporalState,
)
from confik.temporal_v6.trajectories import (
    FAMILIES,
    FreshTrajectorySpec,
    load_trajectory_role,
)
from confik.types import IKQuery, SolveTrace, VerificationResult


WORKSPACE = Path(__file__).resolve().parents[1]
ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def _query(model: URDFKinematics) -> IKQuery:
    q = np.zeros(model.nq, dtype=np.float64)
    return IKQuery(model.forward(q), q)


def _hard_outcome(query: IKQuery, accepted: bool = True) -> ProfiledOutcome:
    return ProfiledOutcome(
        q=query.previous_q.copy() if accepted else None,
        accepted=accepted,
        entry_action="hard",
        executed_stages=("hard",),
        risk_probabilities=np.asarray([0.0, 0.0, 1.0, 0.0]),
        risk_score=1.0,
        function_evaluations=7,
        iterations=5,
        fallback_used=False,
        verification_reasons=() if accepted else ("failed",),
        reject_reason="" if accepted else "failed",
        candidate_count=5,
        timings_ns={"total_end_to_end_ns": 1},
    )


class HardStub:
    def __init__(
        self,
        model: URDFKinematics,
        verifier: SolutionVerifier,
        accepted: list[bool] | None = None,
    ) -> None:
        self.kinematics = model
        self._timed_verifier = SimpleNamespace(verifier=verifier)
        self.accepted = list(accepted or [True])
        self.calls = 0

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        accepted = self.accepted[min(self.calls, len(self.accepted) - 1)]
        self.calls += 1
        return _hard_outcome(query, accepted)


class LocalStub:
    def __init__(self, accepted: list[bool]) -> None:
        self.accepted = accepted
        self.calls = 0

    def solve(self, query: IKQuery) -> LocalSolveOutcome:
        accepted = self.accepted[min(self.calls, len(self.accepted) - 1)]
        self.calls += 1
        q = query.previous_q.copy() if accepted else None
        trace = SolveTrace(q, accepted, 1, 0.0, 0.0, function_evaluations=2)
        verification = (
            VerificationResult(True, 0.0, 0.0, True, True, True) if accepted else None
        )
        return LocalSolveOutcome(
            q=q,
            accepted=accepted,
            trace=trace,
            verification=verification,
            verification_reasons=(),
            function_evaluations=2,
            iterations=1,
            timings_ns={"local_solver_ns": 1, "local_verifier_ns": 1, "total_ns": 2},
        )


def _runtime(
    *, hold: int, hard_accept: list[bool] | None = None, local_accept: list[bool]
) -> tuple[TemporalCGHIKRuntime, HardStub]:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, hard_accept)
    runtime = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=hard,
        policy_config=TemporalPolicyConfig(hold),
    )
    runtime.local_runtime = LocalStub(local_accept)  # type: ignore[assignment]
    return runtime, hard


def _small_view(robot: str = "panda") -> TrajectoryView:
    count, nq = 4, 2
    dataset = QueryDataset(
        previous_q=np.zeros((count, nq)),
        target_position=np.zeros((count, 3)),
        target_rotation=np.repeat(np.eye(3)[None, :, :], count, axis=0),
        reference_q=np.zeros((count, nq)),
        category=np.asarray(["a", "a", "b", "b"]),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.asarray([0, 0, 1, 1]),
        time_index=np.asarray([0, 1, 0, 1]),
    )
    hashes = np.asarray([f"{index:064x}" for index in range(count)], dtype="U64")
    uids = np.asarray(["1" * 64, "1" * 64, "2" * 64, "2" * 64], dtype="U64")
    return TrajectoryView(
        robot,
        "trajectory_calibration",
        dataset,
        hashes,
        uids,
        ("1" * 64, "2" * 64),
        0.02,
    )


def test_policy_has_only_h_and_state_is_explicit_and_immutable() -> None:
    assert [field.name for field in fields(TemporalPolicyConfig)] == ["hold_frames"]
    assert H_VALUES == (2, 5, 10, 20, 30)
    state = TemporalState()
    with pytest.raises(Exception):
        state.frames_seen = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        TemporalState(mode=TemporalMode.ROBUST, frames_seen=1)


def test_exact_h_hard_calls_then_next_frame_local_probe() -> None:
    controller = TemporalPolicyController(TemporalPolicyConfig(2))
    state = controller.initial_state()
    state = controller.transition(
        state, controller.plan(state), local_accepted=None, hard_accepted=True
    ).state_after
    state = controller.transition(
        state, controller.plan(state), local_accepted=False, hard_accepted=True
    ).state_after
    assert state.mode is TemporalMode.ROBUST
    assert state.hard_calls_since_local_attempt == 1
    plan = controller.plan(state)
    assert plan.action == "robust_hard"
    state = controller.transition(
        state, plan, local_accepted=None, hard_accepted=True
    ).state_after
    assert state.hard_calls_since_local_attempt == 2
    plan = controller.plan(state)
    assert plan.action == "robust_local_probe"
    transition = controller.transition(
        state, plan, local_accepted=True, hard_accepted=None
    )
    assert transition.state_after.mode is TemporalMode.LOCAL
    assert transition.recovery_delay_frames == 2


def test_runtime_local_failure_recovers_same_frame_and_probe_success_skips_hard() -> None:
    runtime, hard = _runtime(
        hold=2, hard_accept=[True, True, True], local_accept=[False, True]
    )
    query = _query(runtime.kinematics)  # type: ignore[arg-type]
    first = runtime.step(query, runtime.initial_state())
    second = runtime.step(query, first.state_after)
    assert second.route == "local_fail_hard_recovery"
    assert second.same_frame_hard_recovery
    assert second.function_evaluations == 9
    third = runtime.step(query, second.state_after)
    fourth = runtime.step(query, third.state_after)
    assert fourth.route == "robust_probe_local_accept"
    assert fourth.state_after.mode is TemporalMode.LOCAL
    assert hard.calls == 3
    assert not fourth.learned_seed_ensemble_invoked


def test_failed_probe_invokes_hard_and_restarts_hold_counter() -> None:
    runtime, _ = _runtime(
        hold=2, hard_accept=[False, True, True, True], local_accept=[False]
    )
    query = _query(runtime.kinematics)  # type: ignore[arg-type]
    state = runtime.step(query, runtime.initial_state()).state_after
    state = runtime.step(query, state).state_after
    outcome = runtime.step(query, state)
    assert outcome.route == "robust_probe_local_fail_hard_recovery"
    assert outcome.same_frame_hard_recovery
    assert outcome.state_after.hard_calls_since_local_attempt == 1


def test_bound_stream_snapshot_restore() -> None:
    runtime, _ = _runtime(hold=2, local_accept=[True])
    stream = BoundTemporalStream(runtime)
    initial = stream.snapshot()
    stream.solve(_query(runtime.kinematics))  # type: ignore[arg-type]
    assert stream.snapshot() != initial
    stream.restore(initial)
    assert stream.snapshot() == initial


def test_config_is_exact_and_rejects_extra_policy_parameters_and_formal_paths() -> None:
    path = WORKSPACE / "configs" / "temporal_v6_pilot.yaml"
    config = load_config(path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    changed = dict(config)
    changed["temporal_policy"] = {
        "hold_frames": [2, 5, 10, 20, 30],
        "threshold": 0.9,
    }
    with pytest.raises(ValueError):
        validate_config(changed, workspace=workspace)
    changed = dict(config)
    changed["release_v4_root"] = "../outputs/test_v4_aggregate"
    with pytest.raises(ValueError):
        validate_config(changed, workspace=workspace)
    assert tuple(config["strategies"]) == METHODS
    assert tuple(config["trajectory_data"]["families"]) == FAMILIES


def test_fresh_spec_is_frozen_and_test_named_role_path_fails_before_open() -> None:
    identity = "a" * 64
    panda = FreshTrajectorySpec.frozen("panda", kinematics_identity=identity)
    assert (panda.pool_seed, panda.split_seed, panda.steps) == (861601, 861611, 150)
    with pytest.raises(ValueError):
        FreshTrajectorySpec("panda", 1, 861611, identity)
    with pytest.raises(ValueError):
        load_trajectory_role(
            WORKSPACE / "outputs" / "test_v4" / "queries.npz",
            robot="panda",
            expected_role="trajectory_policy_validation",
        )


def _calibration_data() -> tuple[CalibrationData, TrajectoryView]:
    role = _small_view()
    arms, count, nq = (
        len(CALIBRATION_ARMS),
        role.count,
        role.dataset.previous_q.shape[1],
    )
    accepted = np.ones((arms, count), dtype=bool)
    accepted[0] = [True, True, True, False]
    accepted[1] = [True, False, True, True]
    for arm in range(2, arms):
        accepted[arm] = accepted[0]
    latency = np.full((arms, count), 5_000_000, dtype=np.int64)
    latency[1] = 1_000_000
    return (
        CalibrationData(
            CALIBRATION_ARMS,
            np.asarray((-1,) + H_VALUES, dtype=np.int16),
            latency,
            accepted,
            np.ones((arms, count), dtype=np.int64),
            np.ones((arms, count), dtype=bool),
            np.zeros((arms, count), dtype=bool),
            np.zeros((arms, count), dtype=bool),
            np.ones((arms, count), dtype=bool),
            accepted.copy(),
            np.full((arms, count), -1, dtype=np.int8),
            np.full((arms, count), -1, dtype=np.int8),
            np.full((arms, count), -1, dtype=np.int16),
            np.full((arms, count), -1, dtype=np.int16),
            np.zeros((arms, count), dtype=bool),
            np.zeros((arms, count), dtype=bool),
            np.full((arms, count), "hard", dtype="U64"),
            np.full((arms, count, nq), np.nan),
            np.full((arms, count), "a" * 64, dtype="U64"),
            np.zeros((arms, count), dtype=np.int8),
        ),
        role,
    )


def test_selection_uses_scalar_completion_then_p95_seed_p50() -> None:
    data, role = _calibration_data()
    selected, report = select_hold_frames(data, role)
    assert selected == 2
    row = report["selected"]
    assert row["eligible"] is True
    assert row["completion_vector_hamming_count"] == 2
    assert len(row["gained_trajectory_uids"]) == 1
    assert len(row["lost_trajectory_uids"]) == 1


def test_policy_validation_collector_closes_stages_and_balances_latin_order() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    temporal = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier),
        policy_config=TemporalPolicyConfig(2),
    )
    methods = {
        "always_hard": HardStub(model, verifier),
        "fixed_easy_cascade": HardStub(model, verifier),
        "counterfactual_cghik_v4": HardStub(model, verifier),
        "temporal_event_cghik": temporal,
    }
    q = np.zeros((4, model.nq))
    pose = model.forward(q[0])
    dataset = QueryDataset(
        previous_q=q,
        target_position=np.repeat(pose.position[None, :], 4, axis=0),
        target_rotation=np.repeat(pose.rotation[None, :, :], 4, axis=0),
        reference_q=q.copy(),
        category=np.asarray(["a", "a", "b", "b"]),
        expected_reachable=np.ones(4, dtype=bool),
        continuity_feasible=np.ones(4, dtype=bool),
        trajectory_id=np.asarray([0, 0, 1, 1]),
        time_index=np.asarray([0, 1, 0, 1]),
    )
    role = TrajectoryView(
        "panda",
        "trajectory_policy_validation",
        dataset,
        np.asarray([f"{index + 10:064x}" for index in range(4)], dtype="U64"),
        np.asarray(["3" * 64, "3" * 64, "4" * 64, "4" * 64], dtype="U64"),
        ("3" * 64, "4" * 64),
        0.02,
    )
    result = benchmark_policy_validation(
        role,
        methods=methods,
        warmup_role=role,
        order_seed=11,
        warmup_frames=0,
        progress_every=0,
    )
    assert result.latency_ns.shape == (4, 4)
    assert np.array_equal(result.stage_latency_ns.sum(axis=2), result.latency_ns)
    assert np.all(np.sort(result.method_order_position, axis=1) == np.arange(4))
    assert np.all(result.accepted)


def test_development_gate_requires_all_four_frozen_targets() -> None:
    rows = []
    for robot in ("panda", "ur5e"):
        rows.extend(
            [
                {
                    "robot": robot,
                    "method": "always_hard",
                    "whole_trajectory_completion_count": 40,
                    "p50_latency_ms": 2.0,
                    "p95_latency_ms": 3.0,
                    "learned_seed_invocation_rate": 1.0,
                },
                {
                    "robot": robot,
                    "method": "temporal_event_cghik",
                    "whole_trajectory_completion_count": 40,
                    "p50_latency_ms": 1.0,
                    "p95_latency_ms": 3.0,
                    "learned_seed_invocation_rate": 0.6,
                },
            ]
        )
    assert pilot_gate(rows)["all_robots_pass"] is True
    rows[-1]["p95_latency_ms"] = 3.001
    assert pilot_gate(rows)["all_robots_pass"] is False
