"""Immutable state and pure transitions for periodic-H Temporal CG-HIK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Literal


class TemporalMode(str, Enum):
    """Persistent execution mode; ``INIT`` exists only before the first frame."""

    INIT = "init"
    LOCAL = "local"
    ROBUST = "robust"


FrameAction = Literal[
    "initial_hard",
    "local",
    "robust_hard",
    "robust_local_probe",
]


@dataclass(frozen=True)
class TemporalPolicyConfig:
    """The sole temporal parameter.

    ``hold_frames`` is the number of executed HARD calls required after the
    most recent LOCAL attempt before the *next* frame may probe LOCAL again.
    A same-frame HARD recovery after a failed LOCAL attempt is the first of
    those calls.
    """

    hold_frames: int

    def __post_init__(self) -> None:
        if isinstance(self.hold_frames, bool) or not isinstance(
            self.hold_frames, Integral
        ):
            raise ValueError("hold_frames must be a positive integer")
        if self.hold_frames <= 0:
            raise ValueError("hold_frames must be a positive integer")
        object.__setattr__(self, "hold_frames", int(self.hold_frames))


@dataclass(frozen=True)
class TemporalState:
    """Explicit per-trajectory state passed between closed-loop frames.

    ``hard_calls_since_local_attempt`` counts calls, not accepted commands.
    This keeps the pure-H schedule independent of solver success and makes the
    off-by-one convention directly auditable.
    """

    mode: TemporalMode = TemporalMode.INIT
    frames_seen: int = 0
    hard_calls_since_local_attempt: int = 0
    mode_switch_count: int = 0
    last_local_failure_frame: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TemporalMode):
            raise TypeError("mode must be a TemporalMode")
        for name, value in (
            ("frames_seen", self.frames_seen),
            ("hard_calls_since_local_attempt", self.hard_calls_since_local_attempt),
            ("mode_switch_count", self.mode_switch_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if self.last_local_failure_frame is not None:
            value = self.last_local_failure_frame
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("last_local_failure_frame must be an integer or None")
            object.__setattr__(self, "last_local_failure_frame", int(value))
        if self.mode is TemporalMode.INIT:
            if (
                self.frames_seen != 0
                or self.hard_calls_since_local_attempt != 0
                or self.mode_switch_count != 0
                or self.last_local_failure_frame is not None
            ):
                raise ValueError("INIT state must be the zero state")
        elif self.frames_seen < 1:
            raise ValueError("non-INIT state must follow at least one frame")
        elif self.mode is TemporalMode.LOCAL:
            if self.hard_calls_since_local_attempt != 0:
                raise ValueError("LOCAL state must not retain a ROBUST HARD count")
        elif self.hard_calls_since_local_attempt < 1:
            raise ValueError("ROBUST state requires at least one executed HARD call")
        if self.last_local_failure_frame is not None and not (
            0 <= self.last_local_failure_frame < self.frames_seen
        ):
            raise ValueError("last_local_failure_frame must precede the next frame")

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "frames_seen": int(self.frames_seen),
            "hard_calls_since_local_attempt": int(
                self.hard_calls_since_local_attempt
            ),
            "mode_switch_count": int(self.mode_switch_count),
            "last_local_failure_frame": self.last_local_failure_frame,
        }


@dataclass(frozen=True)
class TemporalFramePlan:
    frame_index: int
    action: FrameAction
    occupancy_mode: TemporalMode
    local_probe_scheduled: bool
    hard_calls_since_local_attempt: int


@dataclass(frozen=True)
class TemporalTransition:
    state_before: TemporalState
    state_after: TemporalState
    switched: bool
    switch_kind: str | None
    recovery_delay_frames: int | None = None


class TemporalPolicyController:
    """Stateless pure-H planner and transition validator."""

    def __init__(self, config: TemporalPolicyConfig) -> None:
        if not isinstance(config, TemporalPolicyConfig):
            raise TypeError("config must be a TemporalPolicyConfig")
        self.config = config

    @staticmethod
    def initial_state() -> TemporalState:
        return TemporalState()

    def plan(self, state: TemporalState) -> TemporalFramePlan:
        if state.mode is TemporalMode.INIT:
            return TemporalFramePlan(
                0,
                "initial_hard",
                TemporalMode.ROBUST,
                False,
                0,
            )
        if state.mode is TemporalMode.LOCAL:
            return TemporalFramePlan(
                state.frames_seen,
                "local",
                TemporalMode.LOCAL,
                False,
                0,
            )
        due = state.hard_calls_since_local_attempt >= self.config.hold_frames
        return TemporalFramePlan(
            state.frames_seen,
            "robust_local_probe" if due else "robust_hard",
            TemporalMode.ROBUST,
            due,
            state.hard_calls_since_local_attempt,
        )

    def transition(
        self,
        state: TemporalState,
        plan: TemporalFramePlan,
        *,
        local_accepted: bool | None,
        hard_accepted: bool | None,
    ) -> TemporalTransition:
        if plan != self.plan(state):
            raise ValueError("frame plan does not belong to the supplied state/config")

        if plan.action == "initial_hard":
            if local_accepted is not None or hard_accepted is None:
                raise ValueError("INIT must execute exactly one HARD call")
            after = TemporalState(
                mode=TemporalMode.LOCAL if hard_accepted else TemporalMode.ROBUST,
                frames_seen=1,
                hard_calls_since_local_attempt=0 if hard_accepted else 1,
            )
            return TemporalTransition(state, after, False, None)

        if plan.action == "local":
            if local_accepted is None:
                raise ValueError("LOCAL mode must execute one LOCAL call")
            if local_accepted:
                if hard_accepted is not None:
                    raise ValueError("accepted LOCAL must not execute HARD")
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
                hard_calls_since_local_attempt=1,
                mode_switch_count=state.mode_switch_count + 1,
                last_local_failure_frame=state.frames_seen,
            )
            return TemporalTransition(state, after, True, "local_to_robust")

        if plan.action == "robust_hard":
            if local_accepted is not None or hard_accepted is None:
                raise ValueError("non-probe ROBUST frame must execute exactly one HARD call")
            after = TemporalState(
                mode=TemporalMode.ROBUST,
                frames_seen=state.frames_seen + 1,
                hard_calls_since_local_attempt=(
                    state.hard_calls_since_local_attempt + 1
                ),
                mode_switch_count=state.mode_switch_count,
                last_local_failure_frame=state.last_local_failure_frame,
            )
            return TemporalTransition(state, after, False, None)

        if plan.action != "robust_local_probe":  # pragma: no cover - type guard
            raise ValueError(f"unsupported temporal action: {plan.action!r}")
        if local_accepted is None:
            raise ValueError("scheduled ROBUST probe must execute LOCAL first")
        if local_accepted:
            if hard_accepted is not None:
                raise ValueError("accepted ROBUST LOCAL probe must not execute HARD")
            delay = None
            if state.last_local_failure_frame is not None:
                # Frame-index distance from the failed LOCAL attempt to the
                # successful probe.  With a hold of H HARD calls this is H,
                # not the inclusive count H + 1.
                delay = state.frames_seen - state.last_local_failure_frame
            after = TemporalState(
                mode=TemporalMode.LOCAL,
                frames_seen=state.frames_seen + 1,
                mode_switch_count=state.mode_switch_count + 1,
                last_local_failure_frame=state.last_local_failure_frame,
            )
            return TemporalTransition(
                state,
                after,
                True,
                "robust_to_local",
                delay,
            )
        if hard_accepted is None:
            raise ValueError(
                "failed ROBUST LOCAL probe must execute HARD in the same frame"
            )
        after = TemporalState(
            mode=TemporalMode.ROBUST,
            frames_seen=state.frames_seen + 1,
            hard_calls_since_local_attempt=1,
            mode_switch_count=state.mode_switch_count,
            last_local_failure_frame=state.frames_seen,
        )
        return TemporalTransition(state, after, False, None)


__all__ = [
    "FrameAction",
    "TemporalFramePlan",
    "TemporalMode",
    "TemporalPolicyConfig",
    "TemporalPolicyController",
    "TemporalState",
    "TemporalTransition",
]
