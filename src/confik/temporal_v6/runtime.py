"""Verifier-backed runtime for Temporal/Event-Triggered CG-HIK."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

import numpy as np

from ..hierarchical_v5.runtime import AlwaysLocalRuntime, LocalSolveOutcome
from ..hierarchical_v5_lite.features import prepare_lite_features
from ..latency_pilot_v3.benchmark import ProfiledOutcome
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier
from ..types import FloatArray, IKQuery
from .policy import TemporalPolicyConfig, TemporalPolicyController
from .state import TemporalState


TEMPORAL_TIMING_KEYS = (
    "state_policy_ns",
    "local_path_ns",
    "hard_path_ns",
    "event_feature_ns",
    "event_gate_ns",
    "total_ns",
)


class LocalSuccessBackend(Protocol):
    def predict_one(self, features: np.ndarray) -> object: ...


class FixedHardRuntime(Protocol):
    def solve(self, query: IKQuery) -> ProfiledOutcome: ...


@dataclass
class TemporalOutcome:
    q: FloatArray | None
    accepted: bool
    route: str
    state_before: TemporalState
    state_after: TemporalState
    occupancy_mode: str
    mode_switched: bool
    switch_kind: str | None
    local_attempted: bool
    local_accepted: bool
    hard_attempted: bool
    hard_accepted: bool
    learned_seed_ensemble_invoked: bool
    reentry_probe_scheduled: bool
    reentry_probe_executed: bool
    reentry_probability: float | None
    robust_frame_number: int
    robust_to_local_recovery_delay_frames: int | None
    executed_stages: tuple[str, ...]
    function_evaluations: int
    iterations: int
    fallback_used: bool
    candidate_count: int
    verification_reasons: tuple[str, ...]
    local_verification_reasons: tuple[str, ...]
    hard_verification_reasons: tuple[str, ...]
    reject_reason: str
    timings_ns: dict[str, int]

    @property
    def local_to_robust_transition(self) -> bool:
        return self.switch_kind == "local_to_robust"

    @property
    def robust_to_local_transition(self) -> bool:
        return self.switch_kind == "robust_to_local"


def _assert_fixed_hard(outcome: ProfiledOutcome) -> None:
    if str(outcome.entry_action).lower() != "hard":
        raise RuntimeError(f"Temporal V6 requires fixed HARD, got {outcome.entry_action!r}")
    stages = tuple(str(stage).lower().split(":", 1)[0] for stage in outcome.executed_stages)
    if not stages or any(stage != "hard" for stage in stages):
        raise RuntimeError(f"Temporal V6 HARD path executed non-HARD stages: {outcome.executed_stages}")


def _probability_from_prediction(value: object) -> float:
    probability = getattr(value, "local_success_probability", None)
    if probability is None and isinstance(value, dict):
        probability = value.get("local_success_probability", value.get("probability"))
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]) or not 0.0 <= values[0] <= 1.0:
        raise ValueError("V5-Lite predictor must return one finite probability in [0, 1]")
    return float(values[0])


class TemporalCGHIKRuntime:
    """Pure explicit-state online stepper.

    ``step`` never mutates the supplied :class:`TemporalState`.  Benchmark
    code can therefore reset or rollback a complete trajectory safely.
    """

    def __init__(
        self,
        *,
        kinematics: object,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        predictor: LocalSuccessBackend,
        always_hard_runtime: FixedHardRuntime,
        policy_config: TemporalPolicyConfig,
        local_iterations: int = 1,
        name: str = "temporal_cghik_v6",
    ) -> None:
        if local_iterations != 1:
            raise ValueError("Temporal V6 LOCAL budget is frozen at one DLS iteration")
        hard_kinematics = getattr(always_hard_runtime, "kinematics", None)
        if hard_kinematics is None:
            raise TypeError("fixed HARD must expose its kinematics for contract checks")
        if hard_kinematics is not kinematics:
            raise ValueError("LOCAL and HARD must share one kinematics instance")
        timed_verifier = getattr(always_hard_runtime, "_timed_verifier", None)
        hard_verifier = getattr(timed_verifier, "verifier", None)
        if hard_verifier is None:
            raise TypeError("fixed HARD must expose its deterministic verifier")
        if hard_verifier is not verifier:
            raise ValueError("LOCAL and HARD must share the deterministic verifier")
        self.name = str(name)
        self.kinematics = kinematics
        self.predictor = predictor
        self.always_hard_runtime = always_hard_runtime
        self.local_runtime = AlwaysLocalRuntime(
            dls, verifier, iterations=1, name=f"{self.name}:local"
        )
        self.controller = TemporalPolicyController(policy_config)

    @property
    def policy_config(self) -> TemporalPolicyConfig:
        return self.controller.config

    def initial_state(self) -> TemporalState:
        return self.controller.initial_state()

    def _hard(self, query: IKQuery) -> ProfiledOutcome:
        outcome = self.always_hard_runtime.solve(query)
        _assert_fixed_hard(outcome)
        return outcome

    def step(self, query: IKQuery, state: TemporalState) -> TemporalOutcome:
        total_started = perf_counter_ns()
        timings = {key: 0 for key in TEMPORAL_TIMING_KEYS}
        started = perf_counter_ns()
        plan = self.controller.plan(state)
        timings["state_policy_ns"] += perf_counter_ns() - started
        local: LocalSolveOutcome | None = None
        hard: ProfiledOutcome | None = None
        probability: float | None = None
        probe_executed = False

        if plan.action == "initial_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = "bootstrap_hard_accept" if hard.accepted else "bootstrap_hard_failure"
        elif plan.action == "local":
            started = perf_counter_ns()
            local = self.local_runtime.solve(query)
            timings["local_path_ns"] = perf_counter_ns() - started
            if not local.accepted:
                started = perf_counter_ns()
                hard = self._hard(query)
                timings["hard_path_ns"] = perf_counter_ns() - started
            if local.accepted:
                route = "local_accept"
            elif hard is not None and hard.accepted:
                route = "local_fail_hard_recovery"
            else:
                route = "local_fail_hard_failure"
        else:
            # HARD must finish before any event probe is evaluated.
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            if plan.probe_after_hard and hard.accepted:
                started = perf_counter_ns()
                prepared = prepare_lite_features(self.kinematics, query)  # type: ignore[arg-type]
                timings["event_feature_ns"] = perf_counter_ns() - started
                started = perf_counter_ns()
                probability = _probability_from_prediction(
                    self.predictor.predict_one(prepared.features)
                )
                timings["event_gate_ns"] = perf_counter_ns() - started
                probe_executed = True
            route = "robust_hard_accept" if hard.accepted else "robust_hard_failure"

        started = perf_counter_ns()
        transition = self.controller.transition(
            state,
            plan,
            local_accepted=None if local is None else bool(local.accepted),
            hard_accepted=None if hard is None else bool(hard.accepted),
            probe_executed=probe_executed,
            local_success_probability=probability,
        )
        timings["state_policy_ns"] += perf_counter_ns() - started

        if local is not None and local.accepted:
            q = None if local.q is None else np.asarray(local.q, dtype=np.float64).copy()
            accepted = True
            verification_reasons = tuple(local.verification_reasons)
            reject_reason = ""
            fallback_used = False
            candidate_count = 0
        else:
            if hard is None:  # pragma: no cover - invariant guard
                raise RuntimeError("non-local outcome completed without HARD")
            q = None if hard.q is None else np.asarray(hard.q, dtype=np.float64).copy()
            accepted = bool(hard.accepted)
            verification_reasons = tuple(hard.verification_reasons)
            reject_reason = str(hard.reject_reason)
            fallback_used = bool(hard.fallback_used)
            candidate_count = int(hard.candidate_count)

        local_fev = 0 if local is None else int(local.function_evaluations)
        hard_fev = 0 if hard is None else int(hard.function_evaluations)
        local_iterations = 0 if local is None else int(local.iterations)
        hard_iterations = 0 if hard is None else int(hard.iterations)
        stages = (() if local is None else ("local",)) + (
            () if hard is None else tuple(str(stage) for stage in hard.executed_stages)
        )
        local_reasons = () if local is None else tuple(local.verification_reasons)
        hard_reasons = () if hard is None else tuple(hard.verification_reasons)

        outcome = TemporalOutcome(
            q=q,
            accepted=accepted,
            route=route,
            state_before=state,
            state_after=transition.state_after,
            occupancy_mode=plan.occupancy_mode.value,
            mode_switched=bool(transition.switched),
            switch_kind=transition.switch_kind,
            local_attempted=local is not None,
            local_accepted=bool(local is not None and local.accepted),
            hard_attempted=hard is not None,
            hard_accepted=bool(hard is not None and hard.accepted),
            learned_seed_ensemble_invoked=hard is not None,
            reentry_probe_scheduled=bool(plan.probe_after_hard),
            reentry_probe_executed=probe_executed,
            reentry_probability=probability,
            robust_frame_number=int(plan.robust_frame_number),
            robust_to_local_recovery_delay_frames=transition.recovery_delay_frames,
            executed_stages=stages,
            function_evaluations=local_fev + hard_fev,
            iterations=local_iterations + hard_iterations,
            fallback_used=fallback_used,
            candidate_count=candidate_count,
            verification_reasons=verification_reasons,
            local_verification_reasons=local_reasons,
            hard_verification_reasons=hard_reasons,
            reject_reason=reject_reason,
            timings_ns=timings,
        )
        # Stop after the API result has been materialized.  The shared timing
        # mapping is intentionally filled last, matching the profiled cascade
        # call-boundary convention.
        timings["total_ns"] = perf_counter_ns() - total_started
        timings["unattributed_ns"] = max(
            timings["total_ns"]
            - sum(timings[key] for key in TEMPORAL_TIMING_KEYS if key != "total_ns"),
            0,
        )
        return outcome


class BoundTemporalStream:
    """Compatibility adapter with explicit reset/snapshot/restore controls."""

    def __init__(self, runtime: TemporalCGHIKRuntime):
        self.runtime = runtime
        self.state = runtime.initial_state()

    @property
    def name(self) -> str:
        return self.runtime.name

    def reset(self) -> None:
        self.state = self.runtime.initial_state()

    def snapshot(self) -> TemporalState:
        return self.state

    def restore(self, state: TemporalState) -> None:
        self.state = state

    def solve(self, query: IKQuery) -> TemporalOutcome:
        outcome = self.runtime.step(query, self.state)
        self.state = outcome.state_after
        return outcome


__all__ = [
    "BoundTemporalStream",
    "TEMPORAL_TIMING_KEYS",
    "TemporalCGHIKRuntime",
    "TemporalOutcome",
]
