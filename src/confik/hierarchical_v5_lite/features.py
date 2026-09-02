"""Truly lightweight, seed-free state features for Hierarchical V5-Lite.

The first-level policy is intentionally limited to one forward-kinematics
evaluation at the previous command and inexpensive vector operations.  Its
signature therefore contains neither a numerical solver nor a learned seed
provider, making the compute boundary explicit in the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping, Sequence

import numpy as np

from ..geometry import pose_error
from ..kinematics.base import KinematicsModel
from ..types import FloatArray, IKQuery, Pose


LITE_SCALAR_FEATURE_NAMES = (
    "pose_error_x",
    "pose_error_y",
    "pose_error_z",
    "pose_error_rx",
    "pose_error_ry",
    "pose_error_rz",
    "position_error_norm",
    "orientation_error_norm",
    "minimum_normalized_joint_margin",
)


def lite_feature_dim(nq: int) -> int:
    """Return the per-robot input dimension, ``nq + 9``."""

    count = int(nq)
    if count <= 0:
        raise ValueError("nq must be positive")
    return count + len(LITE_SCALAR_FEATURE_NAMES)


def lite_feature_names(
    kinematics_or_joint_names: KinematicsModel | Sequence[str],
) -> tuple[str, ...]:
    """Return the ordered, robot-specific feature schema."""

    if isinstance(kinematics_or_joint_names, KinematicsModel):
        joint_names = tuple(kinematics_or_joint_names.joint_names)
    elif hasattr(kinematics_or_joint_names, "joint_names"):
        joint_names = tuple(
            str(name) for name in kinematics_or_joint_names.joint_names  # type: ignore[union-attr]
        )
    else:
        joint_names = tuple(str(name) for name in kinematics_or_joint_names)
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint names must be non-empty and unique")
    normalized_names = tuple(
        f"previous_q_normalized::{joint_name}" for joint_name in joint_names
    )
    return normalized_names + LITE_SCALAR_FEATURE_NAMES


@dataclass(frozen=True)
class LiteFeatureContext:
    """Inspectable intermediates produced by the single-FK feature pass."""

    previous_q_clipped: FloatArray
    normalized_previous_q: FloatArray
    current_pose: Pose
    pose_error_6d: FloatArray
    position_error_norm: float
    orientation_error_norm: float
    minimum_normalized_joint_margin: float


@dataclass(frozen=True)
class PreparedLiteFeatures:
    """Contiguous deployment input plus provenance-friendly diagnostics."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    context: LiteFeatureContext
    timings_ns: Mapping[str, int]

    def __post_init__(self) -> None:
        values = np.asarray(self.features)
        if values.shape != (len(self.feature_names),):
            raise ValueError(
                "lite features must have one value per feature name; "
                f"got {values.shape} and {len(self.feature_names)} names"
            )
        if values.dtype != np.float32 or not values.flags.c_contiguous:
            raise ValueError("lite features must be contiguous float32")
        if not np.all(np.isfinite(values)):
            raise ValueError("lite features must be finite")
        if len(self.feature_names) != lite_feature_dim(
            len(self.context.normalized_previous_q)
        ):
            raise ValueError("lite feature schema is inconsistent with robot nq")


def prepare_lite_features(
    kinematics: KinematicsModel,
    query: IKQuery,
) -> PreparedLiteFeatures:
    """Build the V5-Lite input using exactly one previous-state FK call.

    Feature order is ``normalized q``, the six-dimensional target pose error,
    translation and rotation error norms, and the minimum joint-limit margin
    normalized by each joint's complete range.  The final margin lies in
    ``[0, 0.5]`` after clipping, matching :meth:`KinematicsModel.joint_margin`.
    """

    total_started = perf_counter_ns()
    timings: dict[str, int] = {}

    started = perf_counter_ns()
    previous_q = kinematics.clip(np.asarray(query.previous_q, dtype=np.float64))
    if previous_q.shape != (kinematics.nq,):
        raise ValueError(f"previous_q must have shape ({kinematics.nq},)")
    normalized_q = np.asarray(kinematics.normalize(previous_q), dtype=np.float64)
    if normalized_q.shape != (kinematics.nq,):
        raise ValueError("normalized previous_q has the wrong shape")
    timings["input_preparation_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    current_pose = kinematics.forward(previous_q)
    timings["previous_fk_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    error_6d = np.asarray(pose_error(query.target, current_pose), dtype=np.float64)
    if error_6d.shape != (6,):
        raise ValueError("pose error must have shape (6,)")
    position_norm = float(np.linalg.norm(error_6d[:3]))
    orientation_norm = float(np.linalg.norm(error_6d[3:]))
    margins = np.asarray(kinematics.joint_margin(previous_q), dtype=np.float64)
    if margins.shape != (kinematics.nq,):
        raise ValueError("joint margin has the wrong shape")
    minimum_margin = float(np.min(margins))
    timings["vector_features_ns"] = perf_counter_ns() - started

    started = perf_counter_ns()
    features = np.ascontiguousarray(
        np.concatenate(
            [
                normalized_q,
                error_6d,
                np.asarray(
                    [position_norm, orientation_norm, minimum_margin],
                    dtype=np.float64,
                ),
            ]
        ),
        dtype=np.float32,
    )
    names = lite_feature_names(kinematics)
    if features.shape != (lite_feature_dim(kinematics.nq),):
        raise RuntimeError("internal lite feature dimension mismatch")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("lite feature extraction produced a non-finite value")
    timings["feature_pack_ns"] = perf_counter_ns() - started

    context = LiteFeatureContext(
        previous_q_clipped=np.asarray(previous_q, dtype=np.float64).copy(),
        normalized_previous_q=normalized_q.copy(),
        current_pose=current_pose,
        pose_error_6d=error_6d.copy(),
        position_error_norm=position_norm,
        orientation_error_norm=orientation_norm,
        minimum_normalized_joint_margin=minimum_margin,
    )
    timings["total_ns"] = perf_counter_ns() - total_started
    return PreparedLiteFeatures(features, names, context, timings)


__all__ = [
    "LITE_SCALAR_FEATURE_NAMES",
    "LiteFeatureContext",
    "PreparedLiteFeatures",
    "lite_feature_dim",
    "lite_feature_names",
    "prepare_lite_features",
]
