import numpy as np

from confik.counterfactual_v4.policy import (
    TwoSidedAbstentionPolicy,
    V4PolicyConfig,
    V4Prediction,
)
from confik.counterfactual_v4.runtime_v4 import PolicyEntryGate, PolicyRiskEngine
from confik.runtime.cascade import EntryAction


class Predictor:
    def __init__(self, prediction: V4Prediction):
        self.value = prediction

    def predict(self, features: np.ndarray) -> V4Prediction:
        del features
        return self.value


class OOD:
    def __init__(self, score: float):
        self.value = score
        self.threshold = 5.0

    def score(self, embedding: np.ndarray) -> float:
        del embedding
        return self.value


def engine(*, score: float, fail: float, success: tuple[float, float, float]):
    prediction = V4Prediction(
        np.asarray(success),
        np.asarray([1.0, 1.5, 2.0]),
        np.asarray([1.2, 1.7, 2.2]),
        fail,
        np.zeros(2),
    )
    policy = TwoSidedAbstentionPolicy(
        Predictor(prediction), OOD(score), V4PolicyConfig()
    )
    return PolicyRiskEngine(policy)


def test_defer_maps_to_full_fixed_cascade_entry() -> None:
    risk_engine = engine(score=10.0, fail=0.99, success=(0.1, 0.1, 0.1))
    risk = risk_engine.predict(np.zeros(9))
    assert risk_engine.last_decision is not None
    assert risk_engine.last_decision.action == "defer"
    assert PolicyEntryGate(risk_engine).choose(risk) == EntryAction.EASY


def test_command_reject_maps_to_zero_solver_entry() -> None:
    risk_engine = engine(score=0.0, fail=0.99, success=(0.1, 0.1, 0.1))
    risk = risk_engine.predict(np.zeros(9))
    assert risk_engine.last_decision is not None
    assert risk_engine.last_decision.action == "reject"
    assert PolicyEntryGate(risk_engine).choose(risk) == EntryAction.REJECT
