"""Seed-free geometric features for the hierarchical v5 fast gate.

The first-level gate must run before the learned seed ensemble.  Consequently,
this module depends only on the query, the robot kinematics, and the frozen DLS
configuration.  In particular, it must never import or call a learned candidate
provider.

The seven-feature schema is deliberately small and fixed.  ``prepare_cheap_features``
also returns the intermediate kinematic context so callers can audit or reuse
the first-order calculation without repeating feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping

import numpy as np

from ..geometry import pose_distance, pose_error
from ..kinematics.base import KinematicsModel
from ..solvers.dls import AdaptiveDLS
from ..types import FloatArray, IKQuery, Pose


CHEAP_FEATURE_NAMES = (
    "target_position_step",
    "target_orientation_step",
    "previous_joint_limit_margin_min",
    "previous_jacobian_sigma_min",
    "previous_jacobian_condition_number",
    "one_step_dls_max_joint_update",
    "estimated_velocity_limit_utilization_max",
)
CHEAP_FEATURE_DIM = len(CHEAP_FEATURE_NAMES)

# Exact singular configurations have infinite mathematical condition number,
# which is not a valid neural-network input.  Saturation preserves their status
# as maximally ill-conditioned while keeping the fixed feature vector finite.
CONDITION_NUMBER_CAP = 1.0e12
_SINGULAR_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CheapFeatureContext:
    """Reusable intermediates from one seed-free feature evaluation."""

    previous_q_clipped: FloatArray
    current_pose: Pose
    jacobian: FloatArray
    singular_values: FloatArray
    sigma_min: float
    condition_number: float
    damping: float
    dls_step_clipped: FloatArray
    estimated_q: FloatArray
    estimated_joint_update: FloatArray


@dataclass(frozen=True)
class PreparedCheapFeatures:
    """Contiguous model input, reusable context, and diagnostic timings."""

    features: np.ndarray
    context: CheapFeatureContext
    timings_ns: Mapping[str, int]

    def __post_init__(self) -> None:
        values = np.asarray(self.features)
        if values.shape != (CHEAP_FEATURE_DIM,):
            raise ValueError(
                f"cheap features must have shape ({CHEAP_FEATURE_DIM},), got {values.shape}"
            )
        if values.dtype != np.float32 or not values.flags.c_contiguous:
            raise ValueError("cheap features must be contiguous float32")
        if not np.all(np.isfinite(values)):
            raise ValueError("cheap features must be finite")


def _condition_number(singular_values: np.ndarray) -> tuple[float, float]:
    if singular_values.size == 0:
        return 0.0, CONDITION_NUMBER_CAP
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    if sigma_min <= _SINGULAR_EPSILON:
        return sigma_min, CONDITION_NUMBER_CAP
    return sigma_min, min(sigma_max / sigma_min, CONDITION_NUMBER_CAP)


def prepare_cheap_features(
    kinematics: KinematicsModel,
    dls: AdaptiveDLS,
    query: IKQuery,
) -> PreparedCheapFeatures:
    """Compute the fixed seven-dimensional, learned-seed-free gate input.

    The first-order estimate mirrors :meth:`AdaptiveDLS.solve` exactly through
    its current-state calculation: the same clipped seed, orientation weight,
    adaptive damping, linear system, least-squares fallback, component-wise DLS
    step clipping, and robot joint-limit clipping are used.  No line search or
    learned seed inference is performed.
    """

    total_started = perf_counter_ns()
    stage_timings: dict[str, int] = {}

    started = perf_counter_ns()
    previous_q = kinematics.clip(np.asarray(query.previous_q, dtype=np.float64))
    if previous_q.shape != (kinematics.nq,):
        raise ValueError(f"previous_q must have shape ({kinematics.nq},)")
    current_pose = kinematics.forward(previous_q)
    position_step, orientation_step = pose_distance(query.target, current_pose)
    margin = float(np.min(kinematics.joint_margin(previous_q)))
    stage_timings["current_geometry_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    jacobian = np.asarray(kinematics.jacobian(previous_q), dtype=np.float64)
    if jacobian.shape != (6, kinematics.nq):
        raise ValueError(
            f"jacobian must have shape (6, {kinematics.nq}), got {jacobian.shape}"
        )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    sigma_min, condition_number = _condition_number(singular_values)
    stage_timings["jacobian_svd_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    config = dls.config
    damping = float(dls.damping(sigma_min))
    weights = np.diag(
        [1.0, 1.0, 1.0] + [float(config.orientation_weight)] * 3
    )
    weighted_jacobian = weights @ jacobian
    weighted_error = weights @ pose_error(query.target, current_pose)
    system = (
        weighted_jacobian @ weighted_jacobian.T
        + damping**2 * np.eye(6, dtype=np.float64)
    )
    try:
        step = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
    except np.linalg.LinAlgError:
        step = weighted_jacobian.T @ np.linalg.lstsq(
            system, weighted_error, rcond=None
        )[0]
    step = np.clip(step, -config.max_joint_step, config.max_joint_step)
    estimated_q = kinematics.clip(previous_q + step)
    estimated_update = np.asarray(
        kinematics.difference(estimated_q, previous_q), dtype=np.float64
    )
    max_update = float(np.max(np.abs(estimated_update)))
    velocity_denominator = np.asarray(
        kinematics.limits.velocity * query.dt, dtype=np.float64
    )
    velocity_utilization = float(
        np.max(np.abs(estimated_update) / np.maximum(velocity_denominator, 1.0e-12))
    )
    stage_timings["first_order_dls_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    features = np.ascontiguousarray(
        np.asarray(
            [
                position_step,
                orientation_step,
                margin,
                sigma_min,
                condition_number,
                max_update,
                velocity_utilization,
            ],
            dtype=np.float32,
        )
    )
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("seed-free feature extraction produced a non-finite value")
    stage_timings["feature_pack_ns"] = perf_counter_ns() - started

    context = CheapFeatureContext(
        previous_q_clipped=previous_q.copy(),
        current_pose=current_pose,
        jacobian=jacobian.copy(),
        singular_values=np.asarray(singular_values, dtype=np.float64).copy(),
        sigma_min=sigma_min,
        condition_number=condition_number,
        damping=damping,
        dls_step_clipped=np.asarray(step, dtype=np.float64).copy(),
        estimated_q=np.asarray(estimated_q, dtype=np.float64).copy(),
        estimated_joint_update=estimated_update.copy(),
    )
    stage_timings["total_ns"] = perf_counter_ns() - total_started
    return PreparedCheapFeatures(features, context, stage_timings)


__all__ = [
    "CHEAP_FEATURE_DIM",
    "CHEAP_FEATURE_NAMES",
    "CONDITION_NUMBER_CAP",
    "CheapFeatureContext",
    "PreparedCheapFeatures",
    "prepare_cheap_features",
]
