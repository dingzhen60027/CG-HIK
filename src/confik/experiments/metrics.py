from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from ..data.datasets import RiskDataset
from ..models.risk import RiskModel
from ..types import FloatArray


def expected_calibration_error(
    probabilities: FloatArray,
    labels: FloatArray,
    bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            accuracy = np.mean(predictions[mask] == labels[mask])
            mean_confidence = np.mean(confidence[mask])
            error += np.mean(mask) * abs(float(accuracy - mean_confidence))
    return float(error)


def binary_calibration_error(probabilities: FloatArray, targets: FloatArray, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities > lower) & (probabilities <= upper)
        if np.any(mask):
            error += np.mean(mask) * abs(float(np.mean(targets[mask]) - np.mean(probabilities[mask])))
    return float(error)


def risk_metrics(model: RiskModel, dataset: RiskDataset) -> dict[str, object]:
    probabilities = model.predict_proba(dataset.features)
    predictions = np.argmax(probabilities, axis=1)
    fail_targets = (dataset.labels == 3).astype(int)
    p_fail = probabilities[:, 3]
    metrics = {
        "macro_f1": float(f1_score(dataset.labels, predictions, average="macro", zero_division=0)),
        "classwise_f1": f1_score(
            dataset.labels, predictions, average=None, labels=[0, 1, 2, 3], zero_division=0
        ).tolist(),
        "balanced_accuracy": float(balanced_accuracy_score(dataset.labels, predictions)),
        "confusion_matrix": confusion_matrix(dataset.labels, predictions, labels=[0, 1, 2, 3]).tolist(),
        "nll": float(log_loss(dataset.labels, probabilities, labels=[0, 1, 2, 3])),
        "ece": expected_calibration_error(probabilities, dataset.labels),
        "fail_brier": float(brier_score_loss(fail_targets, p_fail)),
        "fail_ece": binary_calibration_error(p_fail, fail_targets),
        "action_accuracy": float(np.mean(predictions == dataset.labels)),
    }
    non_reject = dataset.labels != 3
    severity = probabilities[:, 1] + 2.0 * probabilities[:, 2]
    metrics["action_effort_spearman"] = float(
        spearmanr(severity[non_reject], dataset.iterations[non_reject]).statistic
    ) if np.sum(non_reject) >= 3 else float("nan")
    if np.unique(fail_targets).size == 2:
        metrics["fail_auroc"] = float(roc_auc_score(fail_targets, p_fail))
        metrics["fail_auprc"] = float(average_precision_score(fail_targets, p_fail))
    else:
        metrics["fail_auroc"] = float("nan")
        metrics["fail_auprc"] = float("nan")
    return metrics
