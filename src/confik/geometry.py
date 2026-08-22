from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import FloatArray, Pose


def skew(vector: ArrayLike) -> FloatArray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rpy_matrix(rpy: ArrayLike) -> FloatArray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: ArrayLike, angle: float) -> FloatArray:
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_array)
    if norm <= 1e-15:
        return np.eye(3, dtype=np.float64)
    unit = axis_array / norm
    cross = skew(unit)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rotation_log(rotation: ArrayLike) -> FloatArray:
    r = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return 0.5 * np.array(
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            dtype=np.float64,
        )
    if np.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(r) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != pivot:
                    axis[index] = (r[index, pivot] + r[pivot, index]) / (4.0 * axis[pivot])
            axis /= np.linalg.norm(axis)
        return angle * axis
    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    )


def pose_error(target: Pose, current: Pose) -> FloatArray:
    translation = target.position - current.position
    orientation = rotation_log(target.rotation @ current.rotation.T)
    return np.concatenate([translation, orientation])


def pose_distance(target: Pose, current: Pose) -> tuple[float, float]:
    error = pose_error(target, current)
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def rotation_6d(rotation: ArrayLike) -> FloatArray:
    r = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([r[:, 0], r[:, 1]])


def wrap_to_pi(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def transform(rotation: ArrayLike | None = None, translation: ArrayLike | None = None) -> FloatArray:
    matrix = np.eye(4, dtype=np.float64)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix

