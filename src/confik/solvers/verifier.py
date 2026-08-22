from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import pose_distance
from ..kinematics.base import KinematicsModel
from ..types import FloatArray, IKQuery, VerificationResult


@dataclass(frozen=True)
class VerifierConfig:
    position_tolerance: float = 1e-3
    orientation_tolerance: float = np.deg2rad(0.5)
    joint_limit_tolerance: float = 1e-9
    velocity_tolerance: float = 1e-4
    enforce_velocity: bool = True


class SolutionVerifier:
    def __init__(self, kinematics: KinematicsModel, config: VerifierConfig | None = None):
        self.kinematics = kinematics
        self.config = config or VerifierConfig()

    def check(self, q: FloatArray, query: IKQuery) -> VerificationResult:
        q_array = np.asarray(q, dtype=np.float64)
        finite_ok = q_array.shape == (self.kinematics.nq,) and bool(np.all(np.isfinite(q_array)))
        if not finite_ok:
            return VerificationResult(
                accepted=False,
                position_error=float("inf"),
                orientation_error=float("inf"),
                joint_limit_ok=False,
                velocity_ok=False,
                finite_ok=False,
                reasons=("non_finite_or_wrong_shape",),
            )

        config = self.config
        limits = self.kinematics.limits
        joint_limit_ok = bool(
            np.all(q_array >= limits.lower - config.joint_limit_tolerance)
            and np.all(q_array <= limits.upper + config.joint_limit_tolerance)
        )
        delta = np.abs(self.kinematics.difference(q_array, query.previous_q))
        allowed = limits.velocity * query.dt + config.velocity_tolerance
        velocity_ok = bool(np.all(delta <= allowed)) if config.enforce_velocity else True
        actual = self.kinematics.forward(q_array)
        position_error, orientation_error = pose_distance(query.target, actual)
        reasons: list[str] = []
        if position_error > config.position_tolerance:
            reasons.append("position_tolerance")
        if orientation_error > config.orientation_tolerance:
            reasons.append("orientation_tolerance")
        if not joint_limit_ok:
            reasons.append("joint_limit")
        if not velocity_ok:
            reasons.append("velocity_limit")
        accepted = not reasons
        return VerificationResult(
            accepted=accepted,
            position_error=position_error,
            orientation_error=orientation_error,
            joint_limit_ok=joint_limit_ok,
            velocity_ok=velocity_ok,
            finite_ok=finite_ok,
            reasons=tuple(reasons),
        )
