from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from confik.anchored_temporal_v7.pilot import (
    CALIBRATION_ARMS,
    FROZEN_H,
    METHODS,
    R_VALUES,
    NoEligibleReanchorInterval,
    TrajectoryView,
    benchmark_policy_validation,
    pilot_gate,
    select_reanchor_interval,
    summarize_benchmark,
    temporal_run_lengths,
    validate_config,
)
from confik.anchored_temporal_v7.runtime import (
    AnchoredTemporalCGHIKRuntime,
    BoundAnchoredTemporalStream,
)
from confik.anchored_temporal_v7.state import (
    AnchoredTemporalMode,
    AnchoredTemporalPolicyConfig,
    AnchoredTemporalPolicyController,
    AnchoredTemporalState,
)
from confik.anchored_temporal_v7.trajectories import (
    CALIBRATION_ROLE,
    FAMILIES,
    FreshTrajectorySpec,
    POLICY_VALIDATION_ROLE,
    load_trajectory_role,
)
from confik.config import load_config, resolve_path
from confik.data.datasets import QueryDataset
from confik.hierarchical_v5.runtime import LocalSolveOutcome
from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.temporal_v6.runtime import TemporalCGHIKRuntime
from confik.temporal_v6.state import TemporalPolicyConfig
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
        self.accepted = list(accepted)
        self.calls = 0

    def solve(self, query: IKQuery) -> LocalSolveOutcome:
        accepted = self.accepted[min(self.calls, len(self.accepted) - 1)]
        self.calls += 1
        q = query.previous_q.copy() if accepted else None
        trace = SolveTrace(q, accepted, 1, 0.0, 0.0, function_evaluations=2)
        verification = (
            VerificationResult(True, 0.0, 0.0, True, True, True)
            if accepted
            else None
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
    *,
    reanchor: int = 2,
    hold: int = 2,
    hard_accept: list[bool] | None = None,
    local_accept: list[bool] | None = None,
) -> tuple[AnchoredTemporalCGHIKRuntime, HardStub]:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    hard = HardStub(model, verifier, hard_accept)
    runtime = AnchoredTemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=hard,
        policy_config=AnchoredTemporalPolicyConfig(reanchor, hold),
    )
    runtime.local_runtime = LocalStub(local_accept or [True])  # type: ignore[assignment]
    return runtime, hard


def _small_view(robot: str = "panda", *, role: str = CALIBRATION_ROLE) -> TrajectoryView:
    count, nq = 8, 2
    trajectory_id = np.repeat(np.arange(2), 4)
    time_index = np.tile(np.arange(4), 2)
    dataset = QueryDataset(
        previous_q=np.zeros((count, nq)),
        target_position=np.zeros((count, 3)),
        target_rotation=np.repeat(np.eye(3)[None, :, :], count, axis=0),
        reference_q=np.zeros((count, nq)),
        category=np.asarray([FAMILIES[0]] * 4 + [FAMILIES[1]] * 4),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=trajectory_id,
        time_index=time_index,
    )
    hashes = np.asarray([f"{index + 10:064x}" for index in range(count)], dtype="U64")
    uids = np.asarray(["1" * 64] * 4 + ["2" * 64] * 4, dtype="U64")
    return TrajectoryView(
        robot=robot,
        role=role,
        dataset=dataset,
        source_query_hash=hashes,
        trajectory_uid=uids,
        trajectory_order=("1" * 64, "2" * 64),
        phase=np.asarray(["pre"] * count, dtype="U32"),
        phase_index=np.zeros(count, dtype=np.int64),
        transition_boundary=np.asarray([False, False, True, False] * 2),
        dt=0.02,
    )


def test_policy_has_only_r_and_h_and_state_is_explicit_and_immutable() -> None:
    assert [field.name for field in fields(AnchoredTemporalPolicyConfig)] == [
        "reanchor_interval",
        "hold_frames",
    ]
    assert R_VALUES == (20, 25, 30, 40, 50)
    assert FROZEN_H == {"panda": 5, "ur5e": 10}
    state = AnchoredTemporalState()
    with pytest.raises(Exception):
        state.frames_seen = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        AnchoredTemporalState(mode=AnchoredTemporalMode.ROBUST, frames_seen=1)


def test_exact_r_local_commands_then_next_frame_anchor() -> None:
    controller = AnchoredTemporalPolicyController(
        AnchoredTemporalPolicyConfig(reanchor_interval=20, hold_frames=5)
    )
    state = controller.initial_state()
    anchor_frames: list[int] = []
    for frame in range(150):
        plan = controller.plan(state)
        if plan.anchor_scheduled:
            anchor_frames.append(frame)
            transition = controller.transition(
                state, plan, local_accepted=None, hard_accepted=True
            )
        else:
            transition = controller.transition(
                state, plan, local_accepted=True, hard_accepted=None
            )
        state = transition.state_after
    assert anchor_frames == [0, 21, 42, 63, 84, 105, 126, 147]


def test_periodic_anchor_failure_enters_robust_and_counts_as_first_hard() -> None:
    controller = AnchoredTemporalPolicyController(
        AnchoredTemporalPolicyConfig(reanchor_interval=2, hold_frames=2)
    )
    state = controller.initial_state()
    state = controller.transition(
        state, controller.plan(state), local_accepted=None, hard_accepted=True
    ).state_after
    for _ in range(2):
        state = controller.transition(
            state, controller.plan(state), local_accepted=True, hard_accepted=None
        ).state_after
    plan = controller.plan(state)
    assert plan.action == "periodic_anchor_hard"
    transition = controller.transition(
        state, plan, local_accepted=None, hard_accepted=False
    )
    assert transition.state_after.mode is AnchoredTemporalMode.ROBUST
    assert transition.state_after.hard_calls_since_local_attempt == 1
    assert transition.switch_kind == "anchor_to_robust"


def test_local_failure_recovers_same_frame_and_probe_success_skips_hard() -> None:
    runtime, hard = _runtime(
        reanchor=20,
        hold=2,
        hard_accept=[True, True, True],
        local_accept=[False, True],
    )
    query = _query(runtime.kinematics)  # type: ignore[arg-type]
    first = runtime.step(query, runtime.initial_state())
    second = runtime.step(query, first.state_after)
    assert second.route == "local_fail_hard_recovery"
    assert second.same_frame_hard_recovery_attempted
    assert second.same_frame_hard_recovered
    assert second.function_evaluations == 9
    third = runtime.step(query, second.state_after)
    fourth = runtime.step(query, third.state_after)
    assert fourth.route == "robust_probe_local_accept"
    assert fourth.state_after.mode is AnchoredTemporalMode.LOCAL
    assert fourth.state_after.local_streak == 1
    assert not fourth.learned_seed_ensemble_invoked
    assert hard.calls == 3


def test_runtime_fails_closed_when_hard_does_not_share_verifier() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    other_verifier = SolutionVerifier(model)
    with pytest.raises(ValueError, match="share the deterministic verifier"):
        AnchoredTemporalCGHIKRuntime(
            kinematics=model,
            dls=AdaptiveDLS(model),
            verifier=verifier,
            always_hard_runtime=HardStub(model, other_verifier),
            policy_config=AnchoredTemporalPolicyConfig(20, 5),
        )


def test_bound_stream_can_restore_a_trajectory_boundary_snapshot() -> None:
    runtime, _ = _runtime()
    stream = BoundAnchoredTemporalStream(runtime)
    initial = stream.snapshot()
    stream.solve(_query(runtime.kinematics))  # type: ignore[arg-type]
    assert stream.snapshot() != initial
    stream.restore(initial)
    assert stream.snapshot() == initial


def test_config_is_exact_and_rejects_extra_policy_or_formal_paths() -> None:
    path = WORKSPACE / "configs" / "anchored_temporal_v7_pilot.yaml"
    config = load_config(path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    changed = dict(config)
    changed["anchor_policy"] = {
        "local_commitment_horizon": list(R_VALUES),
        "learned_gate_threshold": 0.9,
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
    ur5e = FreshTrajectorySpec.frozen("ur5e", kinematics_identity=identity)
    assert (panda.pool_seed, panda.split_seed, panda.steps) == (862701, 862711, 150)
    assert (ur5e.pool_seed, ur5e.split_seed, ur5e.steps) == (862702, 862712, 150)
    with pytest.raises(ValueError):
        FreshTrajectorySpec("panda", 1, 862711, identity)
    with pytest.raises(ValueError):
        load_trajectory_role(
            WORKSPACE / "outputs" / "test_v4" / "queries.npz",
            robot="panda",
            expected_role=POLICY_VALIDATION_ROLE,
        )


def _selection_data(*, exact_candidate: bool = True) -> SimpleNamespace:
    role = _small_view()
    arms = len(CALIBRATION_ARMS)
    accepted = np.ones((arms, role.count), dtype=bool)
    # HARD completes trajectory 0 and fails trajectory 1.
    accepted[0, 4] = False
    # R=20 has the same scalar completion count but swaps trajectory identity.
    accepted[1, :4] = False
    accepted[1, 4:] = True
    if exact_candidate:
        # R=25 is the first truly eligible candidate.
        accepted[2] = accepted[0]
    else:
        for arm in range(2, arms):
            accepted[arm] = accepted[1]
    latency = np.full((arms, role.count), 4_000_000, dtype=np.int64)
    latency[1] = 100_000  # Ineligible swap must never win.
    latency[2] = 2_000_000
    for arm in range(3, arms):
        if exact_candidate:
            accepted[arm] = accepted[0]
        latency[arm] = 3_000_000 + arm
    return SimpleNamespace(
        accepted=accepted,
        latency_ns=latency,
        seed_invoked=np.ones_like(accepted),
        function_evaluations=np.ones_like(latency),
        reanchor_interval=np.asarray((-1,) + R_VALUES, dtype=np.int16),
    )


def test_selection_requires_exact_completion_identity_not_equal_count() -> None:
    role = _small_view()
    selected, report = select_reanchor_interval(_selection_data(), role)  # type: ignore[arg-type]
    assert selected == 25
    swapped = report["candidate_metrics"][0]
    assert swapped["whole_trajectory_completion_count"] == 1
    assert swapped["completion_vector_hamming_count"] == 2
    assert swapped["eligible"] is False
    assert swapped["gained_trajectory_uids"] == ["2" * 64]
    assert swapped["lost_trajectory_uids"] == ["1" * 64]


def test_selection_stops_when_no_r_preserves_hard_completion_vector() -> None:
    with pytest.raises(NoEligibleReanchorInterval) as error:
        select_reanchor_interval(_selection_data(exact_candidate=False), _small_view())  # type: ignore[arg-type]
    assert error.value.report["eligible_candidate_count"] == 0
    assert error.value.report["selected"] is None


def test_policy_validation_collector_closes_stages_and_resets_state() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    v6 = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier),
        policy_config=TemporalPolicyConfig(2),
    )
    v7 = AnchoredTemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier),
        policy_config=AnchoredTemporalPolicyConfig(20, 2),
    )
    v7.local_runtime = LocalStub([True])  # type: ignore[assignment]
    methods = {
        "always_hard": HardStub(model, verifier),
        "counterfactual_cghik_v4": HardStub(model, verifier),
        "temporal_event_cghik_v6": v6,
        "anchored_temporal_cghik_v7": v7,
    }
    role = _small_view(role=POLICY_VALIDATION_ROLE)
    result = benchmark_policy_validation(
        role,
        methods=methods,
        warmup_role=role,
        order_seed=11,
        warmup_frames=0,
        progress_every=0,
    )
    assert result.latency_ns.shape == (8, 4)
    assert np.array_equal(result.stage_latency_ns.sum(axis=2), result.latency_ns)
    assert np.all(np.sort(result.method_order_position, axis=1) == np.arange(4))
    assert np.all(result.accepted)
    v7_column = METHODS.index("anchored_temporal_cghik_v7")
    assert np.array_equal(
        np.flatnonzero(result.anchor_attempted[:, v7_column]), np.asarray([0, 4])
    )
    run_report = temporal_run_lengths(result)
    assert run_report["robust_required"]["run_count"] == 0
    assert run_report["all_hard_invocation"]["run_count"] == 2


def test_collector_counts_same_frame_recovery_only_when_hard_accepts() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    v6 = TemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier, [True, False, True]),
        policy_config=TemporalPolicyConfig(2),
    )
    v6.local_runtime = LocalStub([False])  # type: ignore[assignment]
    v7 = AnchoredTemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier),
        policy_config=AnchoredTemporalPolicyConfig(20, 2),
    )
    v7.local_runtime = LocalStub([True])  # type: ignore[assignment]
    methods = {
        "always_hard": HardStub(model, verifier),
        "counterfactual_cghik_v4": HardStub(model, verifier),
        "temporal_event_cghik_v6": v6,
        "anchored_temporal_cghik_v7": v7,
    }
    result = benchmark_policy_validation(
        _small_view(role=POLICY_VALIDATION_ROLE),
        methods=methods,
        warmup_role=_small_view(role=CALIBRATION_ROLE),
        order_seed=23,
        warmup_frames=0,
        progress_every=0,
    )
    column = METHODS.index("temporal_event_cghik_v6")
    assert result.same_frame_hard_recovery_attempted[1, column]
    assert not result.hard_accepted[1, column]
    assert not result.same_frame_hard_recovery[1, column]


def test_development_gate_requires_exact_vector_and_all_resource_targets() -> None:
    rows: list[dict[str, object]] = []
    for robot in ("panda", "ur5e"):
        rows.extend(
            [
                {
                    "robot": robot,
                    "method": "always_hard",
                    "whole_trajectory_completion_vector": [True, False],
                    "p50_latency_ms": 2.0,
                    "p95_latency_ms": 3.0,
                    "p99_latency_ms": 5.0,
                    "mean_fev": 8.0,
                    "learned_seed_invocation_rate": 1.0,
                },
                {
                    "robot": robot,
                    "method": "anchored_temporal_cghik_v7",
                    "whole_trajectory_completion_vector": [True, False],
                    "p50_latency_ms": 1.0,
                    "p95_latency_ms": 3.0,
                    "p99_latency_ms": 5.0,
                    "mean_fev": 8.0,
                    "learned_seed_invocation_rate": 0.15,
                },
            ]
        )
    assert pilot_gate(rows)["all_robots_pass"] is True
    rows[-1]["whole_trajectory_completion_vector"] = [False, True]
    assert pilot_gate(rows)["all_robots_pass"] is False
    rows[-1]["whole_trajectory_completion_vector"] = [True, False]
    rows[-1]["p99_latency_ms"] = 5.001
    assert pilot_gate(rows)["all_robots_pass"] is False


def test_summary_partitions_anchor_local_robust_occupancy() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    verifier = SolutionVerifier(model)
    anchored = AnchoredTemporalCGHIKRuntime(
        kinematics=model,
        dls=AdaptiveDLS(model),
        verifier=verifier,
        always_hard_runtime=HardStub(model, verifier),
        policy_config=AnchoredTemporalPolicyConfig(2, 2),
    )
    anchored.local_runtime = LocalStub([True])  # type: ignore[assignment]
    methods = {
        "always_hard": HardStub(model, verifier),
        "counterfactual_cghik_v4": HardStub(model, verifier),
        "temporal_event_cghik_v6": TemporalCGHIKRuntime(
            kinematics=model,
            dls=AdaptiveDLS(model),
            verifier=verifier,
            always_hard_runtime=HardStub(model, verifier),
            policy_config=TemporalPolicyConfig(2),
        ),
        "anchored_temporal_cghik_v7": anchored,
    }
    result = benchmark_policy_validation(
        _small_view(role=POLICY_VALIDATION_ROLE),
        methods=methods,
        warmup_role=_small_view(role=CALIBRATION_ROLE),
        order_seed=19,
        warmup_frames=0,
        progress_every=0,
    )
    row = next(
        item
        for item in summarize_benchmark(result)
        if item["method"] == "anchored_temporal_cghik_v7"
    )
    assert row["anchor_occupancy"] + row["local_occupancy"] + row["robust_occupancy"] == pytest.approx(1.0)
    assert row["anchor_occupancy"] > 0.0


def test_runtime_contains_no_predictor_or_per_frame_gate() -> None:
    source = (WORKSPACE / "src" / "confik" / "anchored_temporal_v7" / "runtime.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert "predictor" not in source
    assert "torchscript" not in source
    assert "prepare_lite_features" not in source
