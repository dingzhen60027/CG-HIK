from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


DECISION_ENTRIES = ("easy", "medium", "hard")
FINAL_ACTIONS = DECISION_ENTRIES + ("reject", "defer")


@dataclass(frozen=True)
class V4Prediction:
    success_probabilities: np.ndarray
    latency_p50_ms: np.ndarray
    latency_p95_ms: np.ndarray
    fail_all_probability: float
    embedding: np.ndarray

    def __post_init__(self) -> None:
        success = np.asarray(self.success_probabilities, dtype=np.float64)
        p50 = np.asarray(self.latency_p50_ms, dtype=np.float64)
        p95 = np.asarray(self.latency_p95_ms, dtype=np.float64)
        embedding = np.asarray(self.embedding, dtype=np.float64)
        if success.shape != (3,) or p50.shape != (3,) or p95.shape != (3,):
            raise ValueError("v4 action predictions must have exactly three entries")
        if not np.all(np.isfinite(np.concatenate([success, p50, p95, embedding]))):
            raise ValueError("v4 prediction contains a non-finite value")
        if np.any((success < 0.0) | (success > 1.0)):
            raise ValueError("success probabilities must lie in [0, 1]")
        if not 0.0 <= float(self.fail_all_probability) <= 1.0:
            raise ValueError("fail-all probability must lie in [0, 1]")
        if np.any(p50 < 0.0) or np.any(p95 < p50):
            raise ValueError("latency quantiles must be nonnegative and non-crossing")
        object.__setattr__(self, "success_probabilities", success)
        object.__setattr__(self, "latency_p50_ms", p50)
        object.__setattr__(self, "latency_p95_ms", p95)
        object.__setattr__(self, "embedding", embedding)


class Predictor(Protocol):
    def predict(self, features: np.ndarray) -> V4Prediction: ...


class OODScorer(Protocol):
    threshold: float

    def score(self, embedding: np.ndarray) -> float: ...


@dataclass(frozen=True)
class V4PolicyConfig:
    minimum_success_probability: float = 0.95
    reject_probability: float = 0.95
    deadline_ms: float = 20.0
    latency_tie_margin_ms: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_success_probability < 1.0:
            raise ValueError("minimum_success_probability must lie in (0, 1)")
        if not 0.0 < self.reject_probability < 1.0:
            raise ValueError("reject_probability must lie in (0, 1)")
        if self.deadline_ms <= 0.0 or self.latency_tie_margin_ms < 0.0:
            raise ValueError("deadline must be positive and tie margin nonnegative")


@dataclass(frozen=True)
class V4Decision:
    action: str
    reason: str
    ood_score: float
    is_ood: bool
    eligible_actions: tuple[str, ...]
    predicted_success: tuple[float, ...]
    predicted_p50_ms: tuple[float, ...]
    predicted_p95_ms: tuple[float, ...]
    fail_all_probability: float


class TwoSidedAbstentionPolicy:
    """Latency-aware entry selection with distinct reject and defer semantics."""

    def __init__(
        self,
        predictor: Predictor,
        ood_scorer: OODScorer,
        config: V4PolicyConfig,
    ):
        self.predictor = predictor
        self.ood_scorer = ood_scorer
        self.config = config

    def decide(self, features: np.ndarray) -> V4Decision:
        prediction = self.predictor.predict(np.asarray(features, dtype=np.float64))
        score = float(self.ood_scorer.score(prediction.embedding))
        is_ood = score > float(self.ood_scorer.threshold)
        base = {
            "ood_score": score,
            "is_ood": is_ood,
            "predicted_success": tuple(float(x) for x in prediction.success_probabilities),
            "predicted_p50_ms": tuple(float(x) for x in prediction.latency_p50_ms),
            "predicted_p95_ms": tuple(float(x) for x in prediction.latency_p95_ms),
            "fail_all_probability": float(prediction.fail_all_probability),
        }
        if is_ood:
            return V4Decision(
                action="defer",
                reason="ood_defer",
                eligible_actions=(),
                **base,
            )

        eligible_index = np.flatnonzero(
            (prediction.success_probabilities >= self.config.minimum_success_probability)
            & (prediction.latency_p95_ms <= self.config.deadline_ms)
        )
        eligible = tuple(DECISION_ENTRIES[int(index)] for index in eligible_index)

        # Reject only when the two heads agree that no portfolio action is
        # dependable.  Contradictory high success and high fail-all estimates
        # are treated as uncertainty and therefore defer.
        if (
            prediction.fail_all_probability >= self.config.reject_probability
            and len(eligible_index) == 0
        ):
            return V4Decision(
                action="reject",
                reason="high_confidence_fail_all",
                eligible_actions=eligible,
                **base,
            )
        if len(eligible_index) == 0:
            return V4Decision(
                action="defer",
                reason="uncertain_no_eligible_action",
                eligible_actions=eligible,
                **base,
            )

        fastest = int(
            eligible_index[np.argmin(prediction.latency_p95_ms[eligible_index])]
        )
        conservative = int(np.min(eligible_index))
        improvement = (
            prediction.latency_p95_ms[conservative]
            - prediction.latency_p95_ms[fastest]
        )
        selected = (
            conservative
            if improvement < self.config.latency_tie_margin_ms
            else fastest
        )
        return V4Decision(
            action=DECISION_ENTRIES[selected],
            reason=(
                "tie_margin_conservative_entry"
                if selected == conservative and selected != fastest
                else "minimum_predicted_p95"
            ),
            eligible_actions=eligible,
            **base,
        )


@dataclass
class TemporalDecisionState:
    previous_action: str | None = None
    previous_predicted_p95_ms: float | None = None


def apply_latency_hysteresis(
    decision: V4Decision,
    state: TemporalDecisionState,
    *,
    switch_margin_ms: float,
) -> V4Decision:
    if switch_margin_ms < 0.0:
        raise ValueError("switch_margin_ms must be nonnegative")
    if decision.action in {"reject", "defer"}:
        state.previous_action = decision.action
        state.previous_predicted_p95_ms = None
        return decision
    current_index = DECISION_ENTRIES.index(decision.action)
    current_cost = decision.predicted_p95_ms[current_index]
    previous = state.previous_action
    if previous in DECISION_ENTRIES and previous in decision.eligible_actions:
        previous_index = DECISION_ENTRIES.index(previous)
        previous_cost = decision.predicted_p95_ms[previous_index]
        if previous_cost - current_cost < switch_margin_ms:
            held = V4Decision(
                action=previous,
                reason="latency_hysteresis_hold",
                ood_score=decision.ood_score,
                is_ood=decision.is_ood,
                eligible_actions=decision.eligible_actions,
                predicted_success=decision.predicted_success,
                predicted_p50_ms=decision.predicted_p50_ms,
                predicted_p95_ms=decision.predicted_p95_ms,
                fail_all_probability=decision.fail_all_probability,
            )
            state.previous_action = previous
            state.previous_predicted_p95_ms = previous_cost
            return held
    state.previous_action = decision.action
    state.previous_predicted_p95_ms = current_cost
    return decision
