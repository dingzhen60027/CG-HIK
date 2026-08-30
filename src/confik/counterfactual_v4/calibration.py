"""Validation-only probability calibration and calibration metrics for v4.

The classes in this module operate on logits rather than already clipped
probabilities.  They intentionally expose ``fit`` separately from model
training so callers can enforce the train/calibration/test split boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["platt", "temperature"]
_EPS = 1e-7


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite array")
    return array


def _targets(values: np.ndarray) -> np.ndarray:
    target = _as_vector(values, name="targets")
    if np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("binary targets must lie in [0, 1]")
    return target


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid returning float64 probabilities."""

    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Equal-width binary expected calibration error."""

    probability = _as_vector(probabilities, name="probabilities")
    target = _targets(targets)
    if probability.shape != target.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    # ``right=True`` keeps p=1 in the final bin and p=0 in the first.
    indices = np.minimum((probability * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if not np.any(selected):
            continue
        error += float(np.mean(selected)) * abs(
            float(np.mean(probability[selected])) - float(np.mean(target[selected]))
        )
    return float(error)


def binary_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
    confidence_threshold: float = 0.8,
) -> dict[str, float]:
    """Return ECE, Brier, NLL, and high-confidence decision coverage.

    ``coverage`` is the fraction of samples for which either class has at
    least ``confidence_threshold`` probability.  It measures how often a
    calibrated head is confident enough to support a selective decision; it
    is not a success or accuracy metric.
    """

    probability = _as_vector(probabilities, name="probabilities")
    target = _targets(targets)
    if probability.shape != target.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0.5, 1]")
    clipped = np.clip(probability, _EPS, 1.0 - _EPS)
    confident = np.maximum(clipped, 1.0 - clipped) >= confidence_threshold
    return {
        "ece": expected_calibration_error(clipped, target, bins=bins),
        "brier": float(np.mean((clipped - target) ** 2)),
        "nll": float(
            -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
        ),
        "coverage": float(np.mean(confident)),
    }


@dataclass
class PlattCalibrator:
    """One-dimensional logistic (Platt) scaling."""

    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "PlattCalibrator":
        score = _as_vector(logits, name="logits")
        target = _targets(targets)
        if score.shape != target.shape:
            raise ValueError("logits and targets must have the same shape")
        hard_target = np.rint(target).astype(np.int64)
        if not np.allclose(target, hard_target):
            raise ValueError("Platt scaling requires binary 0/1 targets")
        if np.unique(hard_target).size == 1:
            # A smoothed empirical prior is defined even on a one-class
            # calibration split and avoids an opaque sklearn failure.
            prior = (float(np.sum(hard_target)) + 0.5) / (len(hard_target) + 1.0)
            self.slope = 0.0
            self.intercept = log(prior / (1.0 - prior))
        else:
            estimator = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                random_state=0,
                max_iter=1000,
            )
            estimator.fit(score[:, None], hard_target)
            self.slope = float(estimator.coef_[0, 0])
            self.intercept = float(estimator.intercept_[0])
        self.fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator has not been fitted")
        return np.clip(
            sigmoid(self.slope * np.asarray(logits, dtype=np.float64) + self.intercept),
            _EPS,
            1.0 - _EPS,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "kind": "platt",
            "slope": self.slope,
            "intercept": self.intercept,
            "fitted": self.fitted,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PlattCalibrator":
        if state.get("kind") != "platt":
            raise ValueError("state is not a Platt calibrator")
        return cls(
            slope=float(state["slope"]),
            intercept=float(state["intercept"]),
            fitted=bool(state["fitted"]),
        )


@dataclass
class TemperatureCalibrator:
    """Positive scalar temperature scaling for one binary logit head."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "TemperatureCalibrator":
        score = _as_vector(logits, name="logits")
        target = _targets(targets)
        if score.shape != target.shape:
            raise ValueError("logits and targets must have the same shape")

        def objective(log_temperature: float) -> float:
            probability = np.clip(sigmoid(score / exp(log_temperature)), _EPS, 1.0 - _EPS)
            return float(
                -np.mean(
                    target * np.log(probability)
                    + (1.0 - target) * np.log1p(-probability)
                )
            )

        optimum = minimize_scalar(
            objective,
            bounds=(log(0.05), log(20.0)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not optimum.success or not np.isfinite(optimum.x):
            raise RuntimeError(f"temperature optimization failed: {optimum.message}")
        self.temperature = float(exp(float(optimum.x)))
        self.fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator has not been fitted")
        return np.clip(
            sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature),
            _EPS,
            1.0 - _EPS,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "kind": "temperature",
            "temperature": self.temperature,
            "fitted": self.fitted,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TemperatureCalibrator":
        if state.get("kind") != "temperature":
            raise ValueError("state is not a temperature calibrator")
        return cls(
            temperature=float(state["temperature"]),
            fitted=bool(state["fitted"]),
        )


class MultiOutputCalibrator:
    """Independent validation-only calibrators for named binary heads."""

    def __init__(self, method: CalibrationMethod, head_names: Sequence[str]):
        if method not in {"platt", "temperature"}:
            raise ValueError(f"unsupported calibration method: {method}")
        names = tuple(str(name) for name in head_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("head_names must be non-empty and unique")
        self.method = method
        self.head_names = names
        calibrator_type = PlattCalibrator if method == "platt" else TemperatureCalibrator
        self.calibrators = [calibrator_type() for _ in names]
        self.fitted = False

    @staticmethod
    def _matrix(values: np.ndarray, width: int, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1 and width == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != width or array.shape[0] == 0:
            raise ValueError(f"{name} must have shape (N, {width})")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "MultiOutputCalibrator":
        score = self._matrix(logits, len(self.head_names), name="logits")
        target = self._matrix(targets, len(self.head_names), name="targets")
        if score.shape != target.shape:
            raise ValueError("logits and targets must have the same shape")
        for index, calibrator in enumerate(self.calibrators):
            calibrator.fit(score[:, index], target[:, index])
        self.fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator has not been fitted")
        score = self._matrix(logits, len(self.head_names), name="logits")
        return np.column_stack(
            [
                calibrator.predict_proba(score[:, index])
                for index, calibrator in enumerate(self.calibrators)
            ]
        )

    def metrics(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        *,
        bins: int = 15,
        confidence_threshold: float = 0.8,
    ) -> dict[str, dict[str, float]]:
        target = self._matrix(targets, len(self.head_names), name="targets")
        probability = self.predict_proba(logits)
        return {
            name: binary_calibration_metrics(
                probability[:, index],
                target[:, index],
                bins=bins,
                confidence_threshold=confidence_threshold,
            )
            for index, name in enumerate(self.head_names)
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "head_names": list(self.head_names),
            "fitted": self.fitted,
            "calibrators": [calibrator.to_state() for calibrator in self.calibrators],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MultiOutputCalibrator":
        instance = cls(str(state["method"]), tuple(state["head_names"]))  # type: ignore[arg-type]
        restored = []
        for calibrator_state in state["calibrators"]:
            kind = calibrator_state.get("kind")
            if kind == "platt":
                restored.append(PlattCalibrator.from_state(calibrator_state))
            elif kind == "temperature":
                restored.append(TemperatureCalibrator.from_state(calibrator_state))
            else:
                raise ValueError(f"unknown calibrator kind: {kind}")
        if len(restored) != len(instance.head_names):
            raise ValueError("calibrator count does not match head_names")
        instance.calibrators = restored
        instance.fitted = bool(state["fitted"])
        return instance
