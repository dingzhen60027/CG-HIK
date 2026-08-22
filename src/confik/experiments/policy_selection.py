from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import f1_score

from ..data.datasets import RiskDataset
from ..models.risk import RiskModel
from ..runtime.cascade import ActionGateConfig, CalibratedActionGate
from ..types import CalibratedRisk
from .baselines_v2 import ThresholdGuardRiskProvider


def action_predictions(probabilities: np.ndarray, config: ActionGateConfig) -> np.ndarray:
    gate = CalibratedActionGate(config)
    return np.asarray(
        [int(gate.choose(CalibratedRisk(row))) for row in np.asarray(probabilities)],
        dtype=np.int64,
    )


def action_policy_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predicted.shape:
        raise ValueError("labels and action predictions must have equal shapes")
    reject = truth == 3
    non_reject = ~reject
    false_reject_rate = float(np.mean(predicted[non_reject] == 3)) if np.any(non_reject) else float("nan")
    reject_recall = float(np.mean(predicted[reject] == 3)) if np.any(reject) else float("nan")
    return {
        "sample_count": int(len(truth)),
        "action_accuracy": float(np.mean(predicted == truth)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "nonreject_action_accuracy": float(np.mean(predicted[non_reject] == truth[non_reject]))
        if np.any(non_reject)
        else float("nan"),
        "nonreject_macro_f1": float(
            f1_score(truth[non_reject], predicted[non_reject], labels=[0, 1, 2], average="macro", zero_division=0)
        ) if np.any(non_reject) else float("nan"),
        "false_reject_rate": false_reject_rate,
        "reject_recall": reject_recall,
        "entry_action_counts": np.bincount(predicted, minlength=4).tolist(),
    }


def _feasible(metrics: dict[str, Any], max_false_reject_rate: float, min_reject_recall: float) -> bool:
    return bool(
        metrics["false_reject_rate"] <= max_false_reject_rate
        and metrics["reject_recall"] >= min_reject_recall
    )


def tune_action_gate(
    model: RiskModel,
    dataset: RiskDataset,
    *,
    easy_grid: Iterable[float],
    hard_grid: Iterable[float],
    reject_grid: Iterable[float],
    max_false_reject_rate: float,
    min_reject_recall: float,
) -> tuple[ActionGateConfig, dict[str, Any]]:
    """Select probability thresholds on a policy-validation split only.

    The constrained objective first protects rejection behavior, then maximizes
    non-reject routing macro-F1. Test labels are never passed to this function.
    """
    probabilities = model.predict_proba(dataset.features)
    candidates: list[tuple[ActionGateConfig, dict[str, Any]]] = []
    for easy, hard, reject in product(easy_grid, hard_grid, reject_grid):
        config = ActionGateConfig(float(easy), float(hard), float(reject))
        metrics = action_policy_metrics(dataset.labels, action_predictions(probabilities, config))
        candidates.append((config, metrics))
    if not candidates:
        raise ValueError("action-gate search grids cannot be empty")
    constrained = [
        item for item in candidates
        if _feasible(item[1], max_false_reject_rate, min_reject_recall)
    ]
    pool = constrained or candidates
    selected, selected_metrics = max(
        pool,
        key=lambda item: (
            bool(_feasible(item[1], max_false_reject_rate, min_reject_recall)),
            item[1]["nonreject_macro_f1"],
            item[1]["reject_recall"],
            -item[1]["false_reject_rate"],
            item[1]["action_accuracy"],
        ),
    )
    return selected, {
        "selection_split": "policy_validation",
        "candidate_count": len(candidates),
        "constraint_feasible_candidates": len(constrained),
        "constraints_satisfied": bool(constrained),
        "constraints": {
            "max_false_reject_rate": max_false_reject_rate,
            "min_reject_recall": min_reject_recall,
        },
        "selected_config": asdict(selected),
        "validation_metrics": selected_metrics,
    }


def tune_threshold_guard(
    train: RiskDataset,
    policy_validation: RiskDataset,
    *,
    quantiles: Iterable[float],
    max_false_reject_rate: float,
    min_reject_recall: float,
) -> tuple[ThresholdGuardRiskProvider, dict[str, Any]]:
    """Fit and select the strongest distance guard under the same reject constraints."""
    candidates: list[tuple[ThresholdGuardRiskProvider, dict[str, Any]]] = []
    for quantile in quantiles:
        provider = ThresholdGuardRiskProvider.fit(train, quantile=float(quantile))
        metrics = action_policy_metrics(
            policy_validation.labels,
            provider.predict_actions(policy_validation.features),
        )
        candidates.append((provider, metrics))
    if not candidates:
        raise ValueError("threshold-guard quantile grid cannot be empty")
    constrained = [
        item for item in candidates
        if _feasible(item[1], max_false_reject_rate, min_reject_recall)
    ]
    pool = constrained or candidates
    # The smallest admissible quantile is the most aggressive/strongest compute baseline.
    selected, selected_metrics = min(
        pool,
        key=lambda item: (
            not _feasible(item[1], max_false_reject_rate, min_reject_recall),
            -item[1]["reject_recall"],
            item[0].config.quantile,
            item[1]["false_reject_rate"],
        ),
    )
    return selected, {
        "selection_split": "policy_validation",
        "candidate_count": len(candidates),
        "constraint_feasible_candidates": len(constrained),
        "constraints_satisfied": bool(constrained),
        "constraints": {
            "max_false_reject_rate": max_false_reject_rate,
            "min_reject_recall": min_reject_recall,
        },
        "selected_config": asdict(selected.config),
        "validation_metrics": selected_metrics,
        "candidate_table": [
            {"config": asdict(provider.config), "metrics": metrics}
            for provider, metrics in candidates
        ],
    }
