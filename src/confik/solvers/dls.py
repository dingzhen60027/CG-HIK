from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import pose_distance, pose_error
from ..kinematics.base import KinematicsModel
from ..types import FloatArray, Pose, SolveTrace


@dataclass(frozen=True)
class DLSConfig:
    position_tolerance: float = 1e-3
    orientation_tolerance: float = np.deg2rad(0.5)
    lambda_min: float = 1e-4
    lambda_max: float = 0.1
    sigma_threshold: float = 0.05
    max_joint_step: float = 0.1
    orientation_weight: float = 0.35
    minimum_improvement: float = 1e-10
    stagnation_patience: int = 8


class AdaptiveDLS:
    def __init__(self, kinematics: KinematicsModel, config: DLSConfig | None = None):
        self.kinematics = kinematics
        self.config = config or DLSConfig()

    def damping(self, sigma_min: float) -> float:
        config = self.config
        ratio = min(max(sigma_min / max(config.sigma_threshold, 1e-12), 0.0), 1.0)
        return config.lambda_min + (config.lambda_max - config.lambda_min) * (1.0 - ratio) ** 2

    def solve(
        self,
        target: Pose,
        seed: FloatArray,
        max_iterations: int,
        *,
        seed_source: str = "unknown",
    ) -> SolveTrace:
        q = self.kinematics.clip(np.asarray(seed, dtype=np.float64))
        if q.shape != (self.kinematics.nq,):
            raise ValueError(f"seed must have shape ({self.kinematics.nq},)")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        damping_history: list[float] = []
        residual_history: list[float] = []
        function_evaluations = 0
        stagnation = 0
        weights = np.diag([1.0, 1.0, 1.0] + [self.config.orientation_weight] * 3)

        for iteration in range(max_iterations + 1):
            current = self.kinematics.forward(q)
            function_evaluations += 1
            position_error, orientation_error = pose_distance(target, current)
            objective = position_error + self.config.orientation_weight * orientation_error
            residual_history.append(objective)
            if (
                position_error <= self.config.position_tolerance
                and orientation_error <= self.config.orientation_tolerance
            ):
                return SolveTrace(
                    q=q.copy(),
                    converged=True,
                    iterations=iteration,
                    position_error=position_error,
                    orientation_error=orientation_error,
                    seed_source=seed_source,
                    reason="converged",
                    function_evaluations=function_evaluations,
                    damping_history=damping_history,
                    residual_history=residual_history,
                )
            if iteration == max_iterations:
                break

            jacobian = self.kinematics.jacobian(q)
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
            damping = self.damping(sigma_min)
            damping_history.append(damping)
            weighted_jacobian = weights @ jacobian
            weighted_error = weights @ pose_error(target, current)
            system = weighted_jacobian @ weighted_jacobian.T + damping**2 * np.eye(6)
            try:
                step = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
            except np.linalg.LinAlgError:
                step = weighted_jacobian.T @ np.linalg.lstsq(system, weighted_error, rcond=None)[0]
            step = np.clip(step, -self.config.max_joint_step, self.config.max_joint_step)

            accepted_step = False
            best_q = q
            best_objective = objective
            for scale in (1.0, 0.5, 0.25, 0.1):
                trial_q = self.kinematics.clip(q + scale * step)
                trial_pose = self.kinematics.forward(trial_q)
                function_evaluations += 1
                trial_position, trial_orientation = pose_distance(target, trial_pose)
                trial_objective = trial_position + self.config.orientation_weight * trial_orientation
                if trial_objective < best_objective:
                    best_q = trial_q
                    best_objective = trial_objective
                    accepted_step = True
                    break
            improvement = objective - best_objective
            q = best_q
            if not accepted_step or improvement <= self.config.minimum_improvement:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= self.config.stagnation_patience:
                current = self.kinematics.forward(q)
                function_evaluations += 1
                position_error, orientation_error = pose_distance(target, current)
                return SolveTrace(
                    q=q.copy(),
                    converged=False,
                    iterations=iteration + 1,
                    position_error=position_error,
                    orientation_error=orientation_error,
                    seed_source=seed_source,
                    reason="stagnation",
                    function_evaluations=function_evaluations,
                    damping_history=damping_history,
                    residual_history=residual_history,
                )

        final_pose = self.kinematics.forward(q)
        function_evaluations += 1
        position_error, orientation_error = pose_distance(target, final_pose)
        return SolveTrace(
            q=q.copy(),
            converged=False,
            iterations=max_iterations,
            position_error=position_error,
            orientation_error=orientation_error,
            seed_source=seed_source,
            reason="iteration_budget_exhausted",
            function_evaluations=function_evaluations,
            damping_history=damping_history,
            residual_history=residual_history,
        )

