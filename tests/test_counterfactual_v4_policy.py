import numpy as np

from confik.counterfactual_v4.policy import (
    TemporalDecisionState,
    TwoSidedAbstentionPolicy,
    V4PolicyConfig,
    V4Prediction,
    apply_latency_hysteresis,
)


class Predictor:
    def __init__(self, prediction: V4Prediction):
        self.prediction = prediction

    def predict(self, features: np.ndarray) -> V4Prediction:
        del features
        return self.prediction


class OOD:
    def __init__(self, score: float, threshold: float = 5.0):
        self.value = score
        self.threshold = threshold

    def score(self, embedding: np.ndarray) -> float:
        del embedding
        return self.value


def prediction(
    success=(0.99, 0.99, 0.99),
    p95=(3.0, 2.0, 1.0),
    fail=0.01,
) -> V4Prediction:
    return V4Prediction(
        np.asarray(success),
        np.asarray(p95) * 0.8,
        np.asarray(p95),
        fail,
        np.zeros(4),
    )


def policy(pred: V4Prediction, ood_score: float = 0.0) -> TwoSidedAbstentionPolicy:
    return TwoSidedAbstentionPolicy(
        Predictor(pred),
        OOD(ood_score),
        V4PolicyConfig(latency_tie_margin_ms=0.15),
    )


def test_ood_always_defers_even_when_fail_all_is_high() -> None:
    decision = policy(prediction(success=(0.1, 0.1, 0.1), fail=0.999), 10.0).decide(
        np.zeros(9)
    )
    assert decision.action == "defer"
    assert decision.reason == "ood_defer"


def test_high_confidence_id_fail_all_rejects_without_solver_action() -> None:
    decision = policy(prediction(success=(0.1, 0.2, 0.3), fail=0.99)).decide(
        np.zeros(9)
    )
    assert decision.action == "reject"


def test_policy_selects_fastest_eligible_action_and_uses_tie_margin() -> None:
    selected = policy(prediction(p95=(3.0, 2.0, 1.0))).decide(np.zeros(9))
    assert selected.action == "hard"
    tied = policy(prediction(p95=(1.00, 0.94, 0.90))).decide(np.zeros(9))
    assert tied.action == "easy"
    assert tied.reason == "tie_margin_conservative_entry"


def test_no_eligible_action_defers_when_fail_all_is_not_confident() -> None:
    decision = policy(prediction(success=(0.5, 0.6, 0.7), fail=0.5)).decide(
        np.zeros(9)
    )
    assert decision.action == "defer"


def test_hysteresis_holds_previous_route_for_small_gain() -> None:
    state = TemporalDecisionState(previous_action="medium")
    decision = policy(prediction(p95=(3.0, 2.0, 1.95))).decide(np.zeros(9))
    held = apply_latency_hysteresis(decision, state, switch_margin_ms=0.1)
    assert held.action == "medium"
    assert held.reason == "latency_hysteresis_hold"
