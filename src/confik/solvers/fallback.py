from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..geometry import pose_distance, pose_error, rotation_6d
from ..kinematics.base import KinematicsModel
from ..types import FloatArray, Pose, SolveTrace


class KDTreeSeedBank:
    def __init__(self, kinematics: KinematicsModel, orientation_scale: float = 0.25):
        self.kinematics = kinematics
        self.orientation_scale = orientation_scale
        self._tree: cKDTree | None = None
        self._joints: FloatArray | None = None

    def _pose_vector(self, pose: Pose) -> FloatArray:
        return np.concatenate([pose.position, self.orientation_scale * rotation_6d(pose.rotation)])

    def fit(self, joints: FloatArray) -> "KDTreeSeedBank":
        joint_array = np.asarray(joints, dtype=np.float64)
        if joint_array.ndim != 2 or joint_array.shape[1] != self.kinematics.nq:
            raise ValueError("seed bank joints have an invalid shape")
        poses = np.stack([self._pose_vector(self.kinematics.forward(q)) for q in joint_array])
        self._tree = cKDTree(poses)
        self._joints = joint_array.copy()
        return self

    @property
    def fitted(self) -> bool:
        return self._tree is not None and self._joints is not None

    def query(self, target: Pose, previous_q: FloatArray, k: int = 3, workspace_candidates: int = 32) -> FloatArray:
        if not self.fitted:
            raise RuntimeError("KDTreeSeedBank must be fitted before query")
        assert self._tree is not None and self._joints is not None
        count = min(max(k, workspace_candidates), len(self._joints))
        _, indices = self._tree.query(self._pose_vector(target), k=count)
        candidate_indices = np.atleast_1d(indices)
        candidates = self._joints[candidate_indices]
        normalized_previous = self.kinematics.normalize(previous_q)
        distances = np.linalg.norm(
            np.stack([self.kinematics.normalize(q) for q in candidates]) - normalized_previous,
            axis=1,
        )
        return candidates[np.argsort(distances)[:k]].copy()


@dataclass(frozen=True)
class TRFConfig:
    position_tolerance: float = 1e-3
    orientation_tolerance: float = np.deg2rad(0.5)
    orientation_weight: float = 0.35
    max_function_evaluations: int = 100


class TRFFallbackSolver:
    def __init__(self, kinematics: KinematicsModel, config: TRFConfig | None = None):
        self.kinematics = kinematics
        self.config = config or TRFConfig()

    def solve(self, target: Pose, seed: FloatArray, *, seed_source: str = "kdtree") -> SolveTrace:
        config = self.config

        def residual(q: FloatArray) -> FloatArray:
            error = pose_error(target, self.kinematics.forward(q))
            error[3:] *= config.orientation_weight
            return error

        result = least_squares(
            residual,
            self.kinematics.clip(seed),
            bounds=(self.kinematics.limits.lower, self.kinematics.limits.upper),
            method="trf",
            max_nfev=config.max_function_evaluations,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        position_error, orientation_error = pose_distance(target, self.kinematics.forward(result.x))
        converged = bool(
            position_error <= config.position_tolerance
            and orientation_error <= config.orientation_tolerance
        )
        return SolveTrace(
            q=np.asarray(result.x, dtype=np.float64),
            converged=converged,
            iterations=int(result.nfev),
            position_error=position_error,
            orientation_error=orientation_error,
            seed_source=seed_source,
            reason="converged" if converged else f"trf_failed:{result.status}",
            function_evaluations=int(result.nfev),
            residual_history=[float(np.linalg.norm(result.fun))],
        )

