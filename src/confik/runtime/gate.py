from __future__ import annotations

from dataclasses import dataclass

from ..types import CalibratedRisk, RiskLevel, SolverPolicy


@dataclass(frozen=True)
class GateConfig:
    easy_probability: float = 0.80
    easy_fail_ceiling: float = 0.05
    medium_fail_ceiling: float = 0.25
    easy_iterations: int = 8
    medium_candidates: int = 3
    medium_iterations: int = 15
    hard_candidates: int = 5
    hard_iterations: int = 25


class ConfidenceGate:
    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()

    def make_policy(self, risk: CalibratedRisk) -> SolverPolicy:
        config = self.config
        p_easy = risk.probability("easy")
        p_fail = risk.probability("fail")
        if p_easy >= config.easy_probability and p_fail < config.easy_fail_ceiling:
            return SolverPolicy(RiskLevel.EASY, 1, config.easy_iterations)
        if p_fail < config.medium_fail_ceiling:
            return SolverPolicy(
                RiskLevel.MEDIUM,
                config.medium_candidates,
                config.medium_iterations,
            )
        return SolverPolicy(
            RiskLevel.HARD,
            config.hard_candidates,
            config.hard_iterations,
            include_previous=True,
            use_fallback=True,
        )

