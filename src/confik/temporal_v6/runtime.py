"""Verifier-backed runtime for the predictor-free periodic-H protocol."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

import numpy as np

from ..hierarchical_v5.runtime import AlwaysLocalRuntime, LocalSolveOutcome
from ..latency_pilot_v3.benchmark import ProfiledOutcome
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier
from ..types import FloatArray, IKQuery
from .state import (
    TemporalPolicyConfig,
    TemporalPolicyController,
    TemporalState,
)


TEMPORAL_TIMING_KEYS = (
    "state_policy_ns",
    "local_path_ns",
    "hard_path_ns",
    "unattributed_ns",
    "total_ns",
)


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
    same_frame_hard_recovery: bool
    learned_seed_ensemble_invoked: bool
    local_probe_scheduled: bool
    local_probe_executed: bool
    hard_calls_since_local_attempt_before: int
    hard_calls_since_local_attempt_after: int
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
    stages = tuple(
        str(stage).lower().split(":", 1)[0] for stage in outcome.executed_stages
    )
    if not stages or any(stage != "hard" for stage in stages):
        raise RuntimeError(
            "Temporal V6 HARD path executed non-HARD stages: "
            f"{outcome.executed_stages}"
        )
    if outcome.accepted and outcome.q is None:
        raise RuntimeError("fixed HARD reported acceptance without a command")


class TemporalCGHIKRuntime:
    """Pure explicit-state online stepper with a periodic LOCAL probe.

    The first frame always runs fixed HARD.  LOCAL failures invoke fixed HARD
    on the same unchanged query.  In ROBUST mode, exactly ``hold_frames`` HARD
    calls must have completed since the latest LOCAL attempt before the next
    frame probes one-step LOCAL first.  No learned predictor or confidence
    threshold participates in scheduling.

    ``step`` never mutates the supplied :class:`TemporalState`, so callers can
    reset or roll back an entire trajectory without hidden scheduler state.
    """

    def __init__(
        self,
        *,
        kinematics: object,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        always_hard_runtime: FixedHardRuntime,
        policy_config: TemporalPolicyConfig,
        local_iterations: int = 1,
        name: str = "temporal_event_cghik",
    ) -> None:
        if local_iterations != 1:
            raise ValueError("Temporal V6 LOCAL budget is frozen at one DLS iteration")
        if getattr(dls, "kinematics", None) is not kinematics:
            raise ValueError("LOCAL DLS and runtime must share one kinematics instance")
        if getattr(verifier, "kinematics", None) is not kinematics:
            raise ValueError("LOCAL verifier and runtime must share one kinematics instance")
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
        self.always_hard_runtime = always_hard_runtime
        self.local_runtime = AlwaysLocalRuntime(
            dls,
            verifier,
            iterations=1,
            name=f"{self.name}:local",
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

    @staticmethod
    def _accepted_local_q(local: LocalSolveOutcome) -> FloatArray:
        if local.q is None:
            raise RuntimeError("LOCAL reported acceptance without a command")
        return np.asarray(local.q, dtype=np.float64).copy()

    @staticmethod
    def _accepted_hard_q(hard: ProfiledOutcome) -> FloatArray:
        if hard.q is None:  # guarded in _assert_fixed_hard; kept fail-closed here
            raise RuntimeError("fixed HARD reported acceptance without a command")
        return np.asarray(hard.q, dtype=np.float64).copy()

    def step(self, query: IKQuery, state: TemporalState) -> TemporalOutcome:
        total_started = perf_counter_ns()
        timings = {key: 0 for key in TEMPORAL_TIMING_KEYS}

        started = perf_counter_ns()
        plan = self.controller.plan(state)
        timings["state_policy_ns"] += perf_counter_ns() - started

        local: LocalSolveOutcome | None = None
        hard: ProfiledOutcome | None = None

        if plan.action == "initial_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = (
                "bootstrap_hard_accept"
                if hard.accepted
                else "bootstrap_hard_failure"
            )
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
        elif plan.action == "robust_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = "robust_hard_accept" if hard.accepted else "robust_hard_failure"
        elif plan.action == "robust_local_probe":
            # The scheduled probe is a real verifier-governed LOCAL solve.  HARD
            # is skipped on success and invoked on the same query on failure.
            started = perf_counter_ns()
            local = self.local_runtime.solve(query)
            timings["local_path_ns"] = perf_counter_ns() - started
            if not local.accepted:
                started = perf_counter_ns()
                hard = self._hard(query)
                timings["hard_path_ns"] = perf_counter_ns() - started
            if local.accepted:
                route = "robust_probe_local_accept"
            elif hard is not None and hard.accepted:
                route = "robust_probe_local_fail_hard_recovery"
            else:
                route = "robust_probe_local_fail_hard_failure"
        else:  # pragma: no cover - Literal/controller guard
            raise RuntimeError(f"unsupported temporal action: {plan.action!r}")

        started = perf_counter_ns()
        transition = self.controller.transition(
            state,
            plan,
            local_accepted=None if local is None else bool(local.accepted),
            hard_accepted=None if hard is None else bool(hard.accepted),
        )
        timings["state_policy_ns"] += perf_counter_ns() - started

        if local is not None and local.accepted:
            q: FloatArray | None = self._accepted_local_q(local)
            accepted = True
            verification_reasons = tuple(local.verification_reasons)
            reject_reason = ""
            fallback_used = False
            candidate_count = 0
        else:
            if hard is None:  # pragma: no cover - transition invariant guard
                raise RuntimeError("rejected LOCAL completed without same-frame HARD")
            accepted = bool(hard.accepted)
            q = self._accepted_hard_q(hard) if accepted else None
            verification_reasons = tuple(hard.verification_reasons)
            reject_reason = str(hard.reject_reason)
            fallback_used = bool(hard.fallback_used)
            candidate_count = int(hard.candidate_count)

        local_fev = 0 if local is None else int(local.function_evaluations)
        hard_fev = 0 if hard is None else int(hard.function_evaluations)
        local_iterations = 0 if local is None else int(local.iterations)
        hard_iterations = 0 if hard is None else int(hard.iterations)
        executed_stages = (() if local is None else ("local",)) + (
            () if hard is None else tuple(str(stage) for stage in hard.executed_stages)
        )
        local_reasons = () if local is None else tuple(local.verification_reasons)
        hard_reasons = () if hard is None else tuple(hard.verification_reasons)
        probe_scheduled = bool(plan.local_probe_scheduled)
        probe_executed = bool(
            plan.action == "robust_local_probe" and local is not None
        )

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
            same_frame_hard_recovery=bool(
                local is not None and not local.accepted and hard is not None
            ),
            learned_seed_ensemble_invoked=hard is not None,
            local_probe_scheduled=probe_scheduled,
            local_probe_executed=probe_executed,
            hard_calls_since_local_attempt_before=int(
                state.hard_calls_since_local_attempt
            ),
            hard_calls_since_local_attempt_after=int(
                transition.state_after.hard_calls_since_local_attempt
            ),
            robust_to_local_recovery_delay_frames=transition.recovery_delay_frames,
            executed_stages=executed_stages,
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

        # Materialize the public result before stopping the outer API timer.
        timings["total_ns"] = perf_counter_ns() - total_started
        attributed = (
            timings["state_policy_ns"]
            + timings["local_path_ns"]
            + timings["hard_path_ns"]
        )
        if attributed > timings["total_ns"]:  # pragma: no cover - clock invariant
            raise RuntimeError("temporal stage timings exceed total runtime")
        timings["unattributed_ns"] = timings["total_ns"] - attributed
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
        if not isinstance(state, TemporalState):
            raise TypeError("restore requires a TemporalState snapshot")
        self.state = state

    def solve(self, query: IKQuery) -> TemporalOutcome:
        outcome = self.runtime.step(query, self.state)
        self.state = outcome.state_after
        return outcome


__all__ = [
    "BoundTemporalStream",
    "FixedHardRuntime",
    "TEMPORAL_TIMING_KEYS",
    "TemporalCGHIKRuntime",
    "TemporalOutcome",
]
