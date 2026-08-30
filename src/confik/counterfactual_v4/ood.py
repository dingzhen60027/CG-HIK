"""Embedding-space Mahalanobis OOD detector for v4 defer routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _embedding_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def ood_detection_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Return OOD AUROC/AUPRC and optional threshold operating points."""

    known = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    shifted = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if known.size == 0 or shifted.size == 0:
        raise ValueError("both ID and OOD score arrays must be non-empty")
    if not np.all(np.isfinite(known)) or not np.all(np.isfinite(shifted)):
        raise ValueError("OOD scores must be finite")
    labels = np.concatenate(
        [np.zeros(known.size, dtype=np.int64), np.ones(shifted.size, dtype=np.int64)]
    )
    scores = np.concatenate([known, shifted])
    result = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    if threshold is not None:
        result.update(
            {
                "id_coverage": float(np.mean(known <= threshold)),
                "id_false_defer_rate": float(np.mean(known > threshold)),
                "ood_recall": float(np.mean(shifted > threshold)),
            }
        )
    return result


@dataclass
class EmbeddingMahalanobisOOD:
    """Shrinkage Mahalanobis detector with a validation-calibrated threshold.

    ``fit`` estimates embedding location and covariance from training data.
    ``calibrate_threshold`` must then use an ID validation/pilot split.  This
    separation prevents a test set from silently influencing the defer rule.
    """

    shrinkage: float = 0.05
    regularization: float = 1e-6
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None
    threshold: float | None = None
    target_id_coverage: float | None = None

    def fit(self, train_embeddings: np.ndarray) -> "EmbeddingMahalanobisOOD":
        embedding = _embedding_matrix(train_embeddings, name="train_embeddings")
        if embedding.shape[0] < 2:
            raise ValueError("at least two training embeddings are required")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        self.mean = np.mean(embedding, axis=0)
        centered = embedding - self.mean
        covariance = centered.T @ centered / max(embedding.shape[0] - 1, 1)
        scale = float(np.trace(covariance) / embedding.shape[1])
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * scale * np.eye(embedding.shape[1], dtype=np.float64)
            + self.regularization * max(scale, 1.0) * np.eye(
                embedding.shape[1], dtype=np.float64
            )
        )
        self.precision = np.linalg.pinv(covariance, hermitian=True)
        self.threshold = None
        self.target_id_coverage = None
        return self

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.precision is not None

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OOD detector has not been fitted")
        embedding = _embedding_matrix(embeddings, name="embeddings")
        assert self.mean is not None and self.precision is not None
        if embedding.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"embedding width {embedding.shape[1]} != fitted width {self.mean.shape[0]}"
            )
        centered = embedding - self.mean
        squared_distance = np.einsum(
            "ni,ij,nj->n", centered, self.precision, centered, optimize=True
        )
        return np.maximum(squared_distance, 0.0)

    def calibrate_threshold(
        self,
        id_validation_embeddings: np.ndarray,
        *,
        target_id_coverage: float = 0.95,
    ) -> float:
        if not 0.0 < target_id_coverage < 1.0:
            raise ValueError("target_id_coverage must lie strictly between 0 and 1")
        scores = self.score_samples(id_validation_embeddings)
        self.threshold = float(
            np.quantile(scores, target_id_coverage, method="higher")
        )
        self.target_id_coverage = float(target_id_coverage)
        return self.threshold

    def predict_ood(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("OOD threshold has not been calibrated on validation data")
        return self.score_samples(embeddings) > self.threshold

    def id_coverage(self, id_embeddings: np.ndarray) -> float:
        if self.threshold is None:
            raise RuntimeError("OOD threshold has not been calibrated on validation data")
        return float(np.mean(self.score_samples(id_embeddings) <= self.threshold))

    def to_state(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted OOD detector")
        assert self.mean is not None and self.precision is not None
        return {
            "shrinkage": self.shrinkage,
            "regularization": self.regularization,
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "threshold": self.threshold,
            "target_id_coverage": self.target_id_coverage,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "EmbeddingMahalanobisOOD":
        detector = cls(
            shrinkage=float(state["shrinkage"]),
            regularization=float(state["regularization"]),
        )
        detector.mean = np.asarray(state["mean"], dtype=np.float64)
        detector.precision = np.asarray(state["precision"], dtype=np.float64)
        if detector.precision.shape != (detector.mean.size, detector.mean.size):
            raise ValueError("serialized OOD precision has an invalid shape")
        threshold = state.get("threshold")
        detector.threshold = None if threshold is None else float(threshold)
        coverage = state.get("target_id_coverage")
        detector.target_id_coverage = None if coverage is None else float(coverage)
        return detector

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_state(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingMahalanobisOOD":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise TypeError("OOD artifact must contain a JSON object")
        return cls.from_state(state)
