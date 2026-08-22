from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import FloatArray, Pose, RobotLimits
from .base import KinematicsModel


class PinocchioKinematics(KinematicsModel):
    """Optional Pinocchio backend for fixed-base serial manipulators."""

    def __init__(self, urdf_path: str | Path, end_frame: str):
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - depends on optional dependency
            raise ImportError("PinocchioKinematics requires the optional 'pin' package") from error
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if self.model.nq != self.model.nv:
            raise ValueError("only fixed-base scalar-joint models are supported")
        self.frame_id = self.model.getFrameId(end_frame)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"frame {end_frame!r} was not found")
        self.name = self.model.name
        self.joint_names = tuple(self.model.names[1:])
        lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        invalid = ~np.isfinite(lower + upper) | ((upper - lower) > 100.0)
        lower[invalid], upper[invalid] = -np.pi, np.pi
        velocity = np.asarray(self.model.velocityLimit, dtype=np.float64).copy()
        velocity[~np.isfinite(velocity) | (velocity <= 0)] = 1.0
        self.limits = RobotLimits(lower, upper, velocity)

    def forward(self, q: FloatArray) -> Pose:
        q_array = np.asarray(q, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, q_array)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        return Pose(np.asarray(placement.translation).copy(), np.asarray(placement.rotation).copy())

    def jacobian(self, q: FloatArray) -> FloatArray:
        q_array = np.asarray(q, dtype=np.float64)
        jacobian = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q_array,
            self.frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(jacobian, dtype=np.float64)

