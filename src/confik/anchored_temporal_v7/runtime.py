"""Verifier-backed runtime for Branch-Anchored Temporal CG-HIK."""

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
    AnchorKind,
    AnchoredTemporalPolicyConfig,
    AnchoredTemporalPolicyController,
    AnchoredTemporalState,
)


ANCHORED_TEMPORAL_TIMING_KEYS = (
    "state_policy_ns",
    "local_path_ns",
    "hard_path_ns",
    "unattributed_ns",
    "total_ns",
)


class FixedHardRuntime(Protocol):
    """Minimal contract for the frozen fixed-HARD cascade."""

    def solve(self, query: IKQuery) -> ProfiledOutcome: ...


@dataclass
class AnchoredTemporalOutcome:
    q: FloatArray | None
    accepted: bool
    route: str
    state_before: AnchoredTemporalState
    state_after: AnchoredTemporalState
    occupancy_mode: str
    mode_switched: bool
    switch_kind: str | None
    local_attempted: bool
    local_accepted: bool
    hard_attempted: bool
    hard_accepted: bool
    same_frame_hard_recovery_attempted: bool
    same_frame_hard_recovered: bool
    learned_seed_ensemble_invoked: bool
    anchor_scheduled: bool
    anchor_attempted: bool
    anchor_accepted: bool
    anchor_kind: AnchorKind | None
    local_probe_scheduled: bool
    local_probe_executed: bool
    local_streak_before: int
    local_streak_after: int
    hard_calls_since_local_attempt_before: int
    hard_calls_since_local_attempt_after: int
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
    def same_frame_hard_recovery(self) -> bool:
        """Whether rejected LOCAL was recovered by accepted same-frame HARD."""

        return self.same_frame_hard_recovered

    @property
    def is_anchor_frame(self) -> bool:
        return self.anchor_attempted

    @property
    def local_to_robust_transition(self) -> bool:
        return self.switch_kind == "local_to_robust"

    @property
    def anchor_to_robust_transition(self) -> bool:
        return self.switch_kind == "anchor_to_robust"

    @property
    def robust_to_local_transition(self) -> bool:
        return self.switch_kind == "robust_to_local"


def _assert_fixed_hard(outcome: ProfiledOutcome) -> None:
    """Fail closed if a supposedly fixed-HARD runtime changes semantics."""

    if str(outcome.entry_action).lower() != "hard":
        raise RuntimeError(
            "Anchored Temporal V7 requires fixed HARD, got "
            f"{outcome.entry_action!r}"
        )
    stages = tuple(
        str(stage).lower().split(":", 1)[0]
        for stage in outcome.executed_stages
    )
    if not stages or any(stage != "hard" for stage in stages):
        raise RuntimeError(
            "Anchored Temporal V7 HARD path executed non-HARD stages: "
            f"{outcome.executed_stages}"
        )
    if outcome.accepted and outcome.q is None:
        raise RuntimeError("fixed HARD reported acceptance without a command")


class AnchoredTemporalCGHIKRuntime:
    """Explicit-state implementation of the frozen Branch-Anchored protocol.

    There is no learned gate or local preview on anchor and non-probe ROBUST
    frames.  Every LOCAL candidate and every HARD result remains subject to the
    same deterministic verifier object.  ``step`` does not mutate its supplied
    state, so one runtime can serve independently reset trajectory streams.
    """

    def __init__(
        self,
        *,
        kinematics: object,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        always_hard_runtime: FixedHardRuntime,
        policy_config: AnchoredTemporalPolicyConfig,
        local_iterations: int = 1,
        name: str = "anchored_temporal_cghik_v7",
    ) -> None:
        if local_iterations != 1:
            raise ValueError(
                "Anchored Temporal V7 LOCAL budget is frozen at one DLS iteration"
            )
        if getattr(dls, "kinematics", None) is not kinematics:
            raise ValueError("LOCAL DLS and runtime must share one kinematics instance")
        if getattr(verifier, "kinematics", None) is not kinematics:
            raise ValueError(
                "LOCAL verifier and runtime must share one kinematics instance"
            )
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
        self.controller = AnchoredTemporalPolicyController(policy_config)

    @property
    def policy_config(self) -> AnchoredTemporalPolicyConfig:
        return self.controller.config

    def initial_state(self) -> AnchoredTemporalState:
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
        if hard.q is None:
            raise RuntimeError("fixed HARD reported acceptance without a command")
        return np.asarray(hard.q, dtype=np.float64).copy()

    def step(
        self,
        query: IKQuery,
        state: AnchoredTemporalState,
    ) -> AnchoredTemporalOutcome:
        total_started = perf_counter_ns()
        timings = {key: 0 for key in ANCHORED_TEMPORAL_TIMING_KEYS}

        started = perf_counter_ns()
        plan = self.controller.plan(state)
        timings["state_policy_ns"] += perf_counter_ns() - started

        local: LocalSolveOutcome | None = None
        hard: ProfiledOutcome | None = None

        if plan.action == "initial_anchor_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = (
                "initial_anchor_hard_accept"
                if hard.accepted
                else "initial_anchor_hard_failure"
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
        elif plan.action == "periodic_anchor_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = (
                "periodic_anchor_hard_accept"
                if hard.accepted
                else "periodic_anchor_hard_failure"
            )
        elif plan.action == "robust_hard":
            started = perf_counter_ns()
            hard = self._hard(query)
            timings["hard_path_ns"] = perf_counter_ns() - started
            route = "robust_hard_accept" if hard.accepted else "robust_hard_failure"
        elif plan.action == "robust_local_probe":
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
            raise RuntimeError(f"unsupported anchored action: {plan.action!r}")

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
        rejected_local = bool(local is not None and not local.accepted)
        same_frame_attempted = bool(rejected_local and hard is not None)
        same_frame_recovered = bool(same_frame_attempted and hard is not None and hard.accepted)
        anchor_attempted = bool(plan.anchor_scheduled and hard is not None)

        outcome = AnchoredTemporalOutcome(
            q=q,
            accepted=accepted,
            route=route,
            state_before=state,
            state_after=transition.state_after,
            occupancy_mode=plan.occupancy_mode,
            mode_switched=bool(transition.switched),
            switch_kind=transition.switch_kind,
            local_attempted=local is not None,
            local_accepted=bool(local is not None and local.accepted),
            hard_attempted=hard is not None,
            hard_accepted=bool(hard is not None and hard.accepted),
            same_frame_hard_recovery_attempted=same_frame_attempted,
            same_frame_hard_recovered=same_frame_recovered,
            learned_seed_ensemble_invoked=hard is not None,
            anchor_scheduled=bool(plan.anchor_scheduled),
            anchor_attempted=anchor_attempted,
            anchor_accepted=bool(anchor_attempted and hard is not None and hard.accepted),
            anchor_kind=plan.anchor_kind,
            local_probe_scheduled=bool(plan.local_probe_scheduled),
            local_probe_executed=bool(
                plan.action == "robust_local_probe" and local is not None
            ),
            local_streak_before=int(state.local_streak),
            local_streak_after=int(transition.state_after.local_streak),
            hard_calls_since_local_attempt_before=int(
                state.hard_calls_since_local_attempt
            ),
            hard_calls_since_local_attempt_after=int(
                transition.state_after.hard_calls_since_local_attempt
            ),
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

        timings["total_ns"] = perf_counter_ns() - total_started
        attributed = (
            timings["state_policy_ns"]
            + timings["local_path_ns"]
            + timings["hard_path_ns"]
        )
        if attributed > timings["total_ns"]:  # pragma: no cover
            raise RuntimeError("anchored temporal stage timings exceed total runtime")
        timings["unattributed_ns"] = timings["total_ns"] - attributed
        return outcome


class BoundAnchoredTemporalStream:
    """Stateful adapter with explicit reset/snapshot/restore controls."""

    def __init__(self, runtime: AnchoredTemporalCGHIKRuntime):
        if not isinstance(runtime, AnchoredTemporalCGHIKRuntime):
            raise TypeError("runtime must be an AnchoredTemporalCGHIKRuntime")
        self.runtime = runtime
        self.state = runtime.initial_state()

    @property
    def name(self) -> str:
        return self.runtime.name

    def reset(self) -> None:
        self.state = self.runtime.initial_state()

    def snapshot(self) -> AnchoredTemporalState:
        return self.state

    def restore(self, state: AnchoredTemporalState) -> None:
        if not isinstance(state, AnchoredTemporalState):
            raise TypeError("restore requires an AnchoredTemporalState snapshot")
        self.state = state

    def solve(self, query: IKQuery) -> AnchoredTemporalOutcome:
        outcome = self.runtime.step(query, self.state)
        self.state = outcome.state_after
        return outcome


__all__ = [
    "ANCHORED_TEMPORAL_TIMING_KEYS",
    "AnchoredTemporalCGHIKRuntime",
    "AnchoredTemporalOutcome",
    "BoundAnchoredTemporalStream",
    "FixedHardRuntime",
]
