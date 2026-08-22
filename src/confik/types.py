from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Pose:
    position: FloatArray
    rotation: FloatArray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)

    @property
    def matrix(self) -> FloatArray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform


@dataclass(frozen=True)
class RobotLimits:
    lower: FloatArray
    upper: FloatArray
    velocity: FloatArray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if not (lower.shape == upper.shape == velocity.shape):
            raise ValueError("lower, upper, and velocity limits must have equal shapes")
        if np.any(lower >= upper):
            raise ValueError("every lower joint limit must be below its upper limit")
        if np.any(velocity <= 0):
            raise ValueError("joint velocity limits must be positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class IKQuery:
    target: Pose
    previous_q: FloatArray
    dt: float = 0.02

    def __post_init__(self) -> None:
        previous_q = np.asarray(self.previous_q, dtype=np.float64)
        if previous_q.ndim != 1:
            raise ValueError("previous_q must be a one-dimensional array")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        object.__setattr__(self, "previous_q", previous_q)


@dataclass
class CandidateSet:
    joints: FloatArray
    scores: FloatArray
    uncertainty_mean: float
    uncertainty_max: float
    source: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.joints.ndim != 2:
            raise ValueError("candidate joints must have shape (candidates, joints)")
        if self.scores.shape != (self.joints.shape[0],):
            raise ValueError("one score is required per candidate")
        if not self.source:
            self.source = ["learned"] * self.joints.shape[0]


@dataclass(frozen=True)
class CalibratedRisk:
    probabilities: FloatArray
    labels: tuple[str, ...] = ("easy", "medium", "hard", "fail")

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.shape != (len(self.labels),):
            raise ValueError("risk probability count must match labels")
        if np.any(probabilities < 0):
            raise ValueError("risk probabilities cannot be negative")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("risk probabilities must have positive mass")
        object.__setattr__(self, "probabilities", probabilities / total)

    def probability(self, label: str) -> float:
        return float(self.probabilities[self.labels.index(label)])

    @property
    def score(self) -> float:
        return self.probability("hard") + self.probability("fail")


class RiskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    REJECT = "reject"


@dataclass(frozen=True)
class SolverPolicy:
    level: RiskLevel
    learned_candidates: int
    dls_iterations_per_candidate: int
    include_previous: bool = False
    use_fallback: bool = False


@dataclass
class SolveTrace:
    q: FloatArray | None
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float
    seed_source: str = "unknown"
    reason: str = ""
    function_evaluations: int = 0
    damping_history: list[float] = field(default_factory=list)
    residual_history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    position_error: float
    orientation_error: float
    joint_limit_ok: bool
    velocity_ok: bool
    finite_ok: bool
    reasons: tuple[str, ...] = ()


@dataclass
class IKResult:
    q: FloatArray | None
    accepted: bool
    risk: CalibratedRisk
    policy: SolverPolicy
    verification: VerificationResult | None
    traces: list[SolveTrace]
    fallback_used: bool
    reject_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
