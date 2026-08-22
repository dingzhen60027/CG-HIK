from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from ..types import FloatArray, Pose, RobotLimits


class KinematicsModel(ABC):
    name: str
    joint_names: tuple[str, ...]
    limits: RobotLimits

    @property
    def nq(self) -> int:
        return len(self.joint_names)

    @abstractmethod
    def forward(self, q: FloatArray) -> Pose:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, q: FloatArray) -> FloatArray:
        """Return a 6 x nq geometric Jacobian in the world frame."""
        raise NotImplementedError

    def clip(self, q: FloatArray, margin: float = 0.0) -> FloatArray:
        span = self.limits.upper - self.limits.lower
        return np.clip(q, self.limits.lower + margin * span, self.limits.upper - margin * span)

    def normalize(self, q: FloatArray) -> FloatArray:
        center = (self.limits.lower + self.limits.upper) / 2.0
        half_span = (self.limits.upper - self.limits.lower) / 2.0
        return (np.asarray(q, dtype=np.float64) - center) / half_span

    def denormalize(self, normalized_q: FloatArray) -> FloatArray:
        center = (self.limits.lower + self.limits.upper) / 2.0
        half_span = (self.limits.upper - self.limits.lower) / 2.0
        return center + np.asarray(normalized_q, dtype=np.float64) * half_span

    def joint_margin(self, q: FloatArray) -> FloatArray:
        q_array = np.asarray(q, dtype=np.float64)
        span = self.limits.upper - self.limits.lower
        return np.minimum(q_array - self.limits.lower, self.limits.upper - q_array) / span

    def difference(self, q: FloatArray, reference: FloatArray) -> FloatArray:
        """Configuration difference; subclasses may wrap genuinely continuous joints."""
        return np.asarray(q, dtype=np.float64) - np.asarray(reference, dtype=np.float64)

    def min_singular_value(self, q: FloatArray) -> float:
        singular_values = np.linalg.svd(self.jacobian(q), compute_uv=False)
        return float(singular_values[-1]) if singular_values.size else 0.0

    def random_configuration(self, rng: Generator, margin: float = 0.1) -> FloatArray:
        span = self.limits.upper - self.limits.lower
        low = self.limits.lower + margin * span
        high = self.limits.upper - margin * span
        return rng.uniform(low, high)
