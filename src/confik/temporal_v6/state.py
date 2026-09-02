"""Immutable per-trajectory state for Temporal/Event-Triggered CG-HIK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TemporalMode(str, Enum):
    INIT = "init"
    LOCAL = "local"
    ROBUST = "robust"


@dataclass(frozen=True)
class TemporalState:
    """State passed explicitly between frames of one closed-loop trajectory."""

    mode: TemporalMode = TemporalMode.INIT
    frames_seen: int = 0
    robust_age: int = 0
    reentry_high_streak: int = 0
    mode_switch_count: int = 0
    last_local_failure_frame: int | None = None

    def __post_init__(self) -> None:
        if self.frames_seen < 0 or self.robust_age < 0 or self.reentry_high_streak < 0:
            raise ValueError("temporal state counters must be nonnegative")
        if self.mode_switch_count < 0:
            raise ValueError("mode_switch_count must be nonnegative")
        if self.mode is TemporalMode.INIT:
            if self.frames_seen or self.robust_age or self.reentry_high_streak:
                raise ValueError("INIT state must have zero counters")
        elif self.mode is TemporalMode.LOCAL:
            if self.robust_age or self.reentry_high_streak:
                raise ValueError("LOCAL state must have zero ROBUST counters")
        elif self.robust_age < 1:
            raise ValueError("ROBUST state must have one-based robust_age")
        if self.last_local_failure_frame is not None and not (
            0 <= self.last_local_failure_frame < self.frames_seen
        ):
            raise ValueError("last_local_failure_frame must precede the next frame")

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "frames_seen": int(self.frames_seen),
            "robust_age": int(self.robust_age),
            "reentry_high_streak": int(self.reentry_high_streak),
            "mode_switch_count": int(self.mode_switch_count),
            "last_local_failure_frame": self.last_local_failure_frame,
        }


__all__ = ["TemporalMode", "TemporalState"]
