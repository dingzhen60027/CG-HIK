from __future__ import annotations

import numpy as np

from ..models.seed import TorchSeedEnsemble
from ..types import CalibratedRisk, CandidateSet, FloatArray, IKQuery


class SingleMemberCandidates:
    def __init__(self, ensemble: TorchSeedEnsemble, member_index: int = 0):
        self.ensemble = ensemble
        self.member_index = member_index

    def candidates(self, query: IKQuery) -> CandidateSet:
        deltas = self.ensemble.predict_deltas(query)
        delta = deltas[self.member_index]
        q = self.ensemble.kinematics.clip(query.previous_q + delta)
        return CandidateSet(
            joints=q[None, :],
            scores=np.array([self.ensemble._score(query, q)]),
            uncertainty_mean=0.0,
            uncertainty_max=0.0,
            source=[f"single_member:{self.member_index}"],
        )


class FeatureMaskRiskProvider:
    def __init__(self, model: object, keep_indices: tuple[int, ...]):
        self.model = model
        self.keep_indices = keep_indices

    def predict(self, features: FloatArray) -> CalibratedRisk:
        masked = np.asarray(features, dtype=np.float64)[list(self.keep_indices)]
        return self.model.predict(masked)  # type: ignore[no-any-return,attr-defined]

