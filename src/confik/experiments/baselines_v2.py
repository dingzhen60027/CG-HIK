from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.datasets import RiskDataset
from ..types import CalibratedRisk, FloatArray


@dataclass(frozen=True)
class ThresholdGuardConfig:
    position_step_threshold: float
    orientation_step_threshold: float
    quantile: float


class ThresholdGuardRiskProvider:
    """Interpretable reject guard fitted only on non-reject training actions.

    It deliberately predicts only easy or reject. When it predicts easy, the
    same fixed cascade escalates normally, so it is a strong baseline for the
    claim that a four-action learned gate adds value beyond a distance guard.
    """

    def __init__(self, config: ThresholdGuardConfig):
        self.config = config

    @classmethod
    def fit(cls, dataset: RiskDataset, *, quantile: float = 0.995) -> "ThresholdGuardRiskProvider":
        if not 0.5 < quantile <= 1.0:
            raise ValueError("threshold guard quantile must lie in (0.5, 1.0]")
        non_reject = dataset.labels != 3
        if not np.any(non_reject):
            raise ValueError("threshold guard requires non-reject training actions")
        return cls(
            ThresholdGuardConfig(
                position_step_threshold=float(np.quantile(dataset.features[non_reject, 7], quantile)),
                orientation_step_threshold=float(np.quantile(dataset.features[non_reject, 8], quantile)),
                quantile=quantile,
            )
        )

    def predict(self, features: FloatArray) -> CalibratedRisk:
        row = np.asarray(features, dtype=np.float64)
        reject = bool(
            row[7] > self.config.position_step_threshold
            or row[8] > self.config.orientation_step_threshold
        )
        return CalibratedRisk(
            np.array([0.0, 0.0, 0.0, 1.0]) if reject else np.array([1.0, 0.0, 0.0, 0.0])
        )

    def predict_actions(self, features: FloatArray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 9:
            raise ValueError("threshold guard features must have shape (samples, >=9)")
        rejected = (
            (rows[:, 7] > self.config.position_step_threshold)
            | (rows[:, 8] > self.config.orientation_step_threshold)
        )
        return np.where(rejected, 3, 0).astype(np.int64)
