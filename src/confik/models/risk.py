from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..types import CalibratedRisk, FloatArray

RISK_LABELS = ("easy", "medium", "hard", "fail")


class RiskModel:
    def __init__(self, kind: Literal["mlp", "gradient_boosting"] = "gradient_boosting", seed: int = 17):
        self.kind = kind
        self.seed = seed
        if kind == "mlp":
            self.estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 64),
                            activation="relu",
                            max_iter=300,
                            early_stopping=True,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        elif kind == "gradient_boosting":
            self.estimator = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1e-3,
                random_state=seed,
            )
        else:
            raise ValueError(f"unknown risk model kind: {kind}")
        self.calibrators: list[IsotonicRegression] | None = None
        self.class_indices: FloatArray | None = None

    def fit(self, features: FloatArray, labels: FloatArray) -> "RiskModel":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        self.estimator.fit(x, y)
        classes = np.asarray(self.estimator.classes_, dtype=np.int64)
        if not set(classes).issubset({0, 1, 2, 3}):
            raise ValueError("risk labels must use class indices 0..3")
        self.class_indices = classes
        return self

    def _raw_full_probabilities(self, features: FloatArray) -> FloatArray:
        raw = np.asarray(self.estimator.predict_proba(features), dtype=np.float64)
        full = np.zeros((raw.shape[0], 4), dtype=np.float64)
        assert self.class_indices is not None
        for raw_index, class_index in enumerate(self.class_indices.astype(int)):
            full[:, class_index] = raw[:, raw_index]
        return full

    def calibrate(self, features: FloatArray, labels: FloatArray) -> "RiskModel":
        y = np.asarray(labels, dtype=np.int64)
        raw = self._raw_full_probabilities(np.asarray(features, dtype=np.float64))
        self.calibrators = []
        for class_index in range(4):
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw[:, class_index], (y == class_index).astype(np.float64))
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        raw = self._raw_full_probabilities(x)
        if self.calibrators is None:
            return raw
        calibrated = np.column_stack(
            [calibrator.predict(raw[:, index]) for index, calibrator in enumerate(self.calibrators)]
        )
        totals = calibrated.sum(axis=1, keepdims=True)
        zero = totals[:, 0] <= 1e-12
        calibrated[zero] = raw[zero]
        totals = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.maximum(totals, 1e-12)

    def predict(self, features: FloatArray) -> CalibratedRisk:
        return CalibratedRisk(self.predict_proba(features)[0])

    def validation_nll(self, features: FloatArray, labels: FloatArray) -> float:
        return float(log_loss(labels, self.predict_proba(features), labels=[0, 1, 2, 3]))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RiskModel":
        model = joblib.load(path)
        if not isinstance(model, RiskModel):
            raise TypeError("artifact is not a RiskModel")
        return model


class ConstantRiskProvider:
    def __init__(self, probabilities: FloatArray | None = None):
        self.risk = CalibratedRisk(
            np.asarray(probabilities if probabilities is not None else [0.9, 0.07, 0.02, 0.01])
        )

    def predict(self, features: FloatArray) -> CalibratedRisk:
        del features
        return self.risk


def select_risk_model(
    train_features: FloatArray,
    train_labels: FloatArray,
    validation_features: FloatArray,
    validation_labels: FloatArray,
    *,
    seed: int = 17,
) -> tuple[RiskModel, dict[str, float]]:
    candidates = [RiskModel("mlp", seed), RiskModel("gradient_boosting", seed)]
    scores: dict[str, float] = {}
    for model in candidates:
        model.fit(train_features, train_labels)
        scores[model.kind] = model.validation_nll(validation_features, validation_labels)
    best = min(candidates, key=lambda model: scores[model.kind])
    return best, scores

