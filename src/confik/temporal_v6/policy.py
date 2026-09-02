"""Pure state-transition policy for Temporal/Event-Triggered CG-HIK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .state import TemporalMode, TemporalState


FrameAction = Literal["initial_hard", "local", "robust_hard"]


@dataclass(frozen=True)
class TemporalPolicyConfig:
    hold_frames: int
    probe_interval: int
    consecutive_successes: int
    reentry_threshold: float

    def __post_init__(self) -> None:
        if self.hold_frames <= 0 or self.probe_interval <= 0:
            raise ValueError("hold_frames and probe_interval must be positive")
        if self.consecutive_successes <= 0:
            raise ValueError("consecutive_successes must be positive")
        if not np.isfinite(self.reentry_threshold) or not 0.0 <= self.reentry_threshold <= 1.0:
            raise ValueError("reentry_threshold must be finite and lie in [0, 1]")


@dataclass(frozen=True)
class TemporalFramePlan:
    frame_index: int
    action: FrameAction
    occupancy_mode: TemporalMode
    probe_after_hard: bool
    robust_frame_number: int


@dataclass(frozen=True)
class TemporalTransition:
    state_before: TemporalState
    state_after: TemporalState
    switched: bool
    switch_kind: str | None
    recovery_delay_frames: int | None = None


class TemporalPolicyController:
    """Stateless controller: ``plan`` and ``transition`` never mutate input."""

    def __init__(self, config: TemporalPolicyConfig) -> None:
        self.config = config

    @staticmethod
    def initial_state() -> TemporalState:
        return TemporalState()

    def plan(self, state: TemporalState) -> TemporalFramePlan:
        if state.mode is TemporalMode.INIT:
            return TemporalFramePlan(0, "initial_hard", TemporalMode.ROBUST, False, 1)
        if state.mode is TemporalMode.LOCAL:
            return TemporalFramePlan(
                state.frames_seen, "local", TemporalMode.LOCAL, False, 0
            )
        robust_frame_number = state.robust_age + 1
        due = bool(
            robust_frame_number >= self.config.hold_frames
            and (robust_frame_number - self.config.hold_frames)
            % self.config.probe_interval
            == 0
        )
        return TemporalFramePlan(
            state.frames_seen,
            "robust_hard",
            TemporalMode.ROBUST,
            due,
            robust_frame_number,
        )

    def transition(
        self,
        state: TemporalState,
        plan: TemporalFramePlan,
        *,
        local_accepted: bool | None,
        hard_accepted: bool | None,
        probe_executed: bool,
        local_success_probability: float | None,
    ) -> TemporalTransition:
        if plan.frame_index != state.frames_seen:
            raise ValueError("frame plan does not belong to the supplied state")
        if plan.action == "initial_hard":
            if state.mode is not TemporalMode.INIT or hard_accepted is None:
                raise ValueError("INIT requires one HARD outcome")
            if local_accepted is not None or probe_executed or local_success_probability is not None:
                raise ValueError("INIT must not run LOCAL or a predictor probe")
            after = TemporalState(
                mode=TemporalMode.LOCAL if hard_accepted else TemporalMode.ROBUST,
                frames_seen=1,
                robust_age=0 if hard_accepted else 1,
            )
            return TemporalTransition(state, after, False, None)

        if plan.action == "local":
            if state.mode is not TemporalMode.LOCAL or local_accepted is None:
                raise ValueError("LOCAL plan requires one LOCAL outcome")
            if probe_executed or local_success_probability is not None:
                raise ValueError("LOCAL must not run a predictor probe")
            if local_accepted:
                if hard_accepted is not None:
                    raise ValueError("accepted LOCAL must not also execute HARD")
                after = TemporalState(
                    mode=TemporalMode.LOCAL,
                    frames_seen=state.frames_seen + 1,
                    mode_switch_count=state.mode_switch_count,
                    last_local_failure_frame=state.last_local_failure_frame,
                )
                return TemporalTransition(state, after, False, None)
            if hard_accepted is None:
                raise ValueError("failed LOCAL must execute HARD in the same frame")
            after = TemporalState(
                mode=TemporalMode.ROBUST,
                frames_seen=state.frames_seen + 1,
                robust_age=1,
                mode_switch_count=state.mode_switch_count + 1,
                last_local_failure_frame=state.frames_seen,
            )
            return TemporalTransition(state, after, True, "local_to_robust")

        if state.mode is not TemporalMode.ROBUST or hard_accepted is None:
            raise ValueError("ROBUST plan requires one HARD outcome")
        if probe_executed and not plan.probe_after_hard:
            raise ValueError("unscheduled predictor probe")
        if plan.probe_after_hard and hard_accepted:
            if not probe_executed or local_success_probability is None:
                raise ValueError("accepted scheduled HARD must run one probe")
        elif probe_executed or local_success_probability is not None:
            raise ValueError("failed or unscheduled HARD must not run a probe")

        streak = state.reentry_high_streak
        # Any failed ROBUST command breaks the evidence chain, even on a
        # non-check frame.  Only verified HARD frames may preserve or extend a
        # streak of scheduled high-confidence events.
        if not hard_accepted:
            streak = 0
        elif plan.probe_after_hard:
            if float(local_success_probability) >= self.config.reentry_threshold:
                streak += 1
            else:
                streak = 0
        switch = streak >= self.config.consecutive_successes
        if switch:
            delay = None
            if state.last_local_failure_frame is not None:
                delay = state.frames_seen + 1 - state.last_local_failure_frame
            after = TemporalState(
                mode=TemporalMode.LOCAL,
                frames_seen=state.frames_seen + 1,
                mode_switch_count=state.mode_switch_count + 1,
                last_local_failure_frame=state.last_local_failure_frame,
            )
            return TemporalTransition(state, after, True, "robust_to_local", delay)
        after = TemporalState(
            mode=TemporalMode.ROBUST,
            frames_seen=state.frames_seen + 1,
            robust_age=plan.robust_frame_number,
            reentry_high_streak=streak,
            mode_switch_count=state.mode_switch_count,
            last_local_failure_frame=state.last_local_failure_frame,
        )
        return TemporalTransition(state, after, False, None)


__all__ = [
    "TemporalFramePlan",
    "TemporalPolicyConfig",
    "TemporalPolicyController",
    "TemporalTransition",
]
