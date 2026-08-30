from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime, ProfiledOutcome
from ..runtime.cascade import EntryAction
from ..types import CalibratedRisk, IKQuery
from .policy import TwoSidedAbstentionPolicy, V4Decision


class PolicyRiskEngine:
    """Adapter that evaluates the v4 predictor inside the timed risk stage."""

    name = "v4_multihead_two_sided_abstention"

    def __init__(self, policy: TwoSidedAbstentionPolicy):
        self.policy = policy
        self.last_decision: V4Decision | None = None

    def predict(self, features: np.ndarray) -> CalibratedRisk:
        decision = self.policy.decide(np.asarray(features, dtype=np.float64))
        self.last_decision = decision
        # ProfiledCascadeRuntime retains the legacy four-mass logging field.
        # These masses are diagnostics only; PolicyEntryGate consumes the
        # non-normalized v4 decision stored above.
        masses = np.asarray(
            [*decision.predicted_success, decision.fail_all_probability],
            dtype=np.float64,
        )
        return CalibratedRisk(np.maximum(masses, 1e-12))


class PolicyEntryGate:
    def __init__(self, engine: PolicyRiskEngine):
        self.engine = engine

    def choose(self, risk: CalibratedRisk) -> EntryAction:
        del risk
        decision = self.engine.last_decision
        if decision is None:
            raise RuntimeError("v4 gate was called before v4 risk inference")
        return {
            "easy": EntryAction.EASY,
            "medium": EntryAction.MEDIUM,
            "hard": EntryAction.HARD,
            "reject": EntryAction.REJECT,
            # Model deferral is deliberately the complete fixed robust cascade.
            "defer": EntryAction.EASY,
        }[decision.action]


class V4ProfiledRuntime:
    """Profiled runtime preserving the v3 solver and verifier acceptance path."""

    def __init__(self, base: ProfiledCascadeRuntime, engine: PolicyRiskEngine):
        self.base = base
        self.engine = engine
        self.kinematics = base.kinematics
        self.last_decision: V4Decision | None = None

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        outcome = self.base.solve(query)
        decision = self.engine.last_decision
        if decision is None:
            raise RuntimeError("v4 runtime completed without a routing decision")
        self.last_decision = decision
        reject_reason = outcome.reject_reason
        if decision.action == "reject":
            reject_reason = "command_reject_high_confidence_fail_all"
        elif decision.action == "defer" and not outcome.accepted:
            reject_reason = "deferred_fixed_robust_failed"
        return replace(
            outcome,
            entry_action=decision.action,
            reject_reason=reject_reason,
        )


def wrap_profiled_runtime(
    *,
    name: str,
    policy: TwoSidedAbstentionPolicy,
    kinematics: object,
    seed_engine: object,
    dls: object,
    verifier: object,
    seed_bank: object,
    fallback: object,
    cascade_config: object,
) -> V4ProfiledRuntime:
    engine = PolicyRiskEngine(policy)
    gate = PolicyEntryGate(engine)
    base = ProfiledCascadeRuntime(
        name=name,
        kinematics=kinematics,
        seed_engine=seed_engine,  # type: ignore[arg-type]
        risk_engine=engine,
        gate=gate,
        dls=dls,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        seed_bank=seed_bank,  # type: ignore[arg-type]
        fallback=fallback,  # type: ignore[arg-type]
        cascade_config=cascade_config,
        reuse_candidate_features=True,
    )
    return V4ProfiledRuntime(base, engine)
