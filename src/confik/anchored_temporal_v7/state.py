"""Immutable state and pure transitions for Branch-Anchored Temporal CG-HIK.

``ANCHOR`` is deliberately a one-frame execution event, not a persistent
state.  The persistent controller state therefore remains ``INIT``, ``LOCAL``
or ``ROBUST`` while frame occupancy can additionally be ``anchor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Literal


class AnchoredTemporalMode(str, Enum):
    """Persistent execution mode; ``INIT`` exists only before frame zero."""

    INIT = "init"
    LOCAL = "local"
    ROBUST = "robust"


AnchoredFrameAction = Literal[
    "initial_anchor_hard",
    "local",
    "periodic_anchor_hard",
    "robust_hard",
    "robust_local_probe",
]
AnchoredOccupancy = Literal["anchor", "local", "robust"]
AnchorKind = Literal["initial", "periodic"]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


@dataclass(frozen=True)
class AnchoredTemporalPolicyConfig:
    """The periodic-anchor parameter ``R`` and frozen V6 robust hold ``H``.

    ``reanchor_interval`` counts verifier-accepted LOCAL commands.  Once the
    state contains exactly that many commands, the *next* frame is a HARD-only
    anchor.  ``hold_frames`` retains the V6 convention: the same-frame HARD
    after a rejected LOCAL command is call one of the ``H`` calls required
    before the next frame may probe LOCAL.
    """

    reanchor_interval: int
    hold_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reanchor_interval",
            _positive_integer(self.reanchor_interval, name="reanchor_interval"),
        )
        object.__setattr__(
            self,
            "hold_frames",
            _positive_integer(self.hold_frames, name="hold_frames"),
        )

    @property
    def R(self) -> int:
        return self.reanchor_interval

    @property
    def H(self) -> int:
        return self.hold_frames


@dataclass(frozen=True)
class AnchoredTemporalState:
    """Explicit per-trajectory state supplied at every closed-loop frame."""

    mode: AnchoredTemporalMode = AnchoredTemporalMode.INIT
    frames_seen: int = 0
    local_streak: int = 0
    hard_calls_since_local_attempt: int = 0
    mode_switch_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AnchoredTemporalMode):
            raise TypeError("mode must be an AnchoredTemporalMode")
        for name in (
            "frames_seen",
            "local_streak",
            "hard_calls_since_local_attempt",
            "mode_switch_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name=name),
            )

        if self.mode is AnchoredTemporalMode.INIT:
            if any(
                (
                    self.frames_seen,
                    self.local_streak,
                    self.hard_calls_since_local_attempt,
                    self.mode_switch_count,
                )
            ):
                raise ValueError("INIT state must be the zero state")
            return

        if self.frames_seen < 1:
            raise ValueError("non-INIT state must follow at least one frame")
        if self.mode_switch_count > self.frames_seen:
            raise ValueError("mode_switch_count cannot exceed frames_seen")
        if self.mode is AnchoredTemporalMode.LOCAL:
            if self.hard_calls_since_local_attempt != 0:
                raise ValueError("LOCAL state must not retain a ROBUST HARD count")
            if self.local_streak > self.frames_seen:
                raise ValueError("local_streak cannot exceed frames_seen")
            return

        if self.local_streak != 0:
            raise ValueError("ROBUST state must have a zero LOCAL streak")
        if self.hard_calls_since_local_attempt < 1:
            raise ValueError("ROBUST state requires at least one executed HARD call")
        if self.hard_calls_since_local_attempt > self.frames_seen:
            raise ValueError("ROBUST HARD count cannot exceed frames_seen")

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "frames_seen": int(self.frames_seen),
            "local_streak": int(self.local_streak),
            "hard_calls_since_local_attempt": int(
                self.hard_calls_since_local_attempt
            ),
            "mode_switch_count": int(self.mode_switch_count),
        }


@dataclass(frozen=True)
class AnchoredTemporalFramePlan:
    frame_index: int
    action: AnchoredFrameAction
    occupancy_mode: AnchoredOccupancy
    anchor_kind: AnchorKind | None
    local_probe_scheduled: bool
    local_streak: int
    hard_calls_since_local_attempt: int

    @property
    def anchor_scheduled(self) -> bool:
        return self.anchor_kind is not None


@dataclass(frozen=True)
class AnchoredTemporalTransition:
    state_before: AnchoredTemporalState
    state_after: AnchoredTemporalState
    switched: bool
    switch_kind: str | None


class AnchoredTemporalPolicyController:
    """Stateless planner and validator for the frozen R/H protocol."""

    def __init__(self, config: AnchoredTemporalPolicyConfig) -> None:
        if not isinstance(config, AnchoredTemporalPolicyConfig):
            raise TypeError("config must be an AnchoredTemporalPolicyConfig")
        self.config = config

    @staticmethod
    def initial_state() -> AnchoredTemporalState:
        return AnchoredTemporalState()

    def plan(self, state: AnchoredTemporalState) -> AnchoredTemporalFramePlan:
        if not isinstance(state, AnchoredTemporalState):
            raise TypeError("state must be an AnchoredTemporalState")
        if state.mode is AnchoredTemporalMode.INIT:
            return AnchoredTemporalFramePlan(
                frame_index=0,
                action="initial_anchor_hard",
                occupancy_mode="anchor",
                anchor_kind="initial",
                local_probe_scheduled=False,
                local_streak=0,
                hard_calls_since_local_attempt=0,
            )
        if state.mode is AnchoredTemporalMode.LOCAL:
            anchor_due = state.local_streak >= self.config.reanchor_interval
            return AnchoredTemporalFramePlan(
                frame_index=state.frames_seen,
                action="periodic_anchor_hard" if anchor_due else "local",
                occupancy_mode="anchor" if anchor_due else "local",
                anchor_kind="periodic" if anchor_due else None,
                local_probe_scheduled=False,
                local_streak=state.local_streak,
                hard_calls_since_local_attempt=0,
            )

        probe_due = (
            state.hard_calls_since_local_attempt >= self.config.hold_frames
        )
        return AnchoredTemporalFramePlan(
            frame_index=state.frames_seen,
            action="robust_local_probe" if probe_due else "robust_hard",
            occupancy_mode="robust",
            anchor_kind=None,
            local_probe_scheduled=probe_due,
            local_streak=0,
            hard_calls_since_local_attempt=state.hard_calls_since_local_attempt,
        )

    def transition(
        self,
        state: AnchoredTemporalState,
        plan: AnchoredTemporalFramePlan,
        *,
        local_accepted: bool | None,
        hard_accepted: bool | None,
    ) -> AnchoredTemporalTransition:
        if plan != self.plan(state):
            raise ValueError("frame plan does not belong to the supplied state/config")

        if plan.action == "initial_anchor_hard":
            if local_accepted is not None or hard_accepted is None:
                raise ValueError("initial anchor must execute exactly one HARD call")
            after = AnchoredTemporalState(
                mode=(
                    AnchoredTemporalMode.LOCAL
                    if hard_accepted
                    else AnchoredTemporalMode.ROBUST
                ),
                frames_seen=1,
                hard_calls_since_local_attempt=0 if hard_accepted else 1,
            )
            return AnchoredTemporalTransition(state, after, False, None)

        if plan.action == "local":
            if local_accepted is None:
                raise ValueError("LOCAL frame must execute one LOCAL call")
            if local_accepted:
                if hard_accepted is not None:
                    raise ValueError("accepted LOCAL must not execute HARD")
                after = AnchoredTemporalState(
                    mode=AnchoredTemporalMode.LOCAL,
                    frames_seen=state.frames_seen + 1,
                    local_streak=state.local_streak + 1,
                    mode_switch_count=state.mode_switch_count,
                )
                return AnchoredTemporalTransition(state, after, False, None)
            if hard_accepted is None:
                raise ValueError("failed LOCAL must execute HARD in the same frame")
            after = AnchoredTemporalState(
                mode=AnchoredTemporalMode.ROBUST,
                frames_seen=state.frames_seen + 1,
                hard_calls_since_local_attempt=1,
                mode_switch_count=state.mode_switch_count + 1,
            )
            return AnchoredTemporalTransition(
                state,
                after,
                True,
                "local_to_robust",
            )

        if plan.action == "periodic_anchor_hard":
            if local_accepted is not None or hard_accepted is None:
                raise ValueError("periodic anchor must execute exactly one HARD call")
            if hard_accepted:
                after = AnchoredTemporalState(
                    mode=AnchoredTemporalMode.LOCAL,
                    frames_seen=state.frames_seen + 1,
                    local_streak=0,
                    mode_switch_count=state.mode_switch_count,
                )
                return AnchoredTemporalTransition(state, after, False, None)
            after = AnchoredTemporalState(
                mode=AnchoredTemporalMode.ROBUST,
                frames_seen=state.frames_seen + 1,
                hard_calls_since_local_attempt=1,
                mode_switch_count=state.mode_switch_count + 1,
            )
            return AnchoredTemporalTransition(
                state,
                after,
                True,
                "anchor_to_robust",
            )

        if plan.action == "robust_hard":
            if local_accepted is not None or hard_accepted is None:
                raise ValueError("non-probe ROBUST frame must execute exactly one HARD call")
            after = AnchoredTemporalState(
                mode=AnchoredTemporalMode.ROBUST,
                frames_seen=state.frames_seen + 1,
                hard_calls_since_local_attempt=(
                    state.hard_calls_since_local_attempt + 1
                ),
                mode_switch_count=state.mode_switch_count,
            )
            return AnchoredTemporalTransition(state, after, False, None)

        if plan.action != "robust_local_probe":  # pragma: no cover
            raise ValueError(f"unsupported anchored action: {plan.action!r}")
        if local_accepted is None:
            raise ValueError("scheduled ROBUST probe must execute LOCAL first")
        if local_accepted:
            if hard_accepted is not None:
                raise ValueError("accepted ROBUST LOCAL probe must not execute HARD")
            after = AnchoredTemporalState(
                mode=AnchoredTemporalMode.LOCAL,
                frames_seen=state.frames_seen + 1,
                # A successful probe returns a LOCAL command and is therefore
                # the first commitment in the new bounded LOCAL streak.
                local_streak=1,
                mode_switch_count=state.mode_switch_count + 1,
            )
            return AnchoredTemporalTransition(
                state,
                after,
                True,
                "robust_to_local",
            )
        if hard_accepted is None:
            raise ValueError(
                "failed ROBUST LOCAL probe must execute HARD in the same frame"
            )
        after = AnchoredTemporalState(
            mode=AnchoredTemporalMode.ROBUST,
            frames_seen=state.frames_seen + 1,
            hard_calls_since_local_attempt=1,
            mode_switch_count=state.mode_switch_count,
        )
        return AnchoredTemporalTransition(state, after, False, None)


__all__ = [
    "AnchorKind",
    "AnchoredFrameAction",
    "AnchoredOccupancy",
    "AnchoredTemporalFramePlan",
    "AnchoredTemporalMode",
    "AnchoredTemporalPolicyConfig",
    "AnchoredTemporalPolicyController",
    "AnchoredTemporalState",
    "AnchoredTemporalTransition",
]
