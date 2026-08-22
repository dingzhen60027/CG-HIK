from __future__ import annotations

from hashlib import blake2b

import numpy as np

from ..kinematics.base import KinematicsModel
from ..models.risk import ConstantRiskProvider
from ..runtime.hybrid import HybridIK
from ..solvers.dls import AdaptiveDLS
from ..solvers.fallback import KDTreeSeedBank, TRFFallbackSolver
from ..solvers.verifier import SolutionVerifier
from ..types import (
    CalibratedRisk,
    CandidateSet,
    IKQuery,
    IKResult,
    RiskLevel,
    SolverPolicy,
)


class FixedPolicyGate:
    def __init__(self, policy: SolverPolicy):
        self.policy = policy

    def make_policy(self, risk: CalibratedRisk) -> SolverPolicy:
        del risk
        return self.policy


class RandomCandidates:
    def __init__(self, kinematics: KinematicsModel, count: int = 5, seed: int = 17):
        self.kinematics = kinematics
        self.count = count
        self.seed = seed

    def candidates(self, query: IKQuery) -> CandidateSet:
        digest = blake2b(
            np.concatenate([query.previous_q, query.target.position]).tobytes(),
            digest_size=8,
            person=str(self.seed).encode()[:16],
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        joints = np.stack([self.kinematics.random_configuration(rng, margin=0.0) for _ in range(self.count)])
        return CandidateSet(
            joints=joints,
            scores=np.arange(self.count, dtype=np.float64),
            uncertainty_mean=0.0,
            uncertainty_max=0.0,
            source=[f"random:{index}" for index in range(self.count)],
        )


class KDTreeCandidates:
    def __init__(self, bank: KDTreeSeedBank, count: int = 3):
        self.bank = bank
        self.count = count

    def candidates(self, query: IKQuery) -> CandidateSet:
        joints = self.bank.query(query.target, query.previous_q, k=self.count)
        return CandidateSet(
            joints=joints,
            scores=np.arange(len(joints), dtype=np.float64),
            uncertainty_mean=0.0,
            uncertainty_max=0.0,
            source=[f"kdtree:{index}" for index in range(len(joints))],
        )


class TRFOnlyMethod:
    def __init__(self, solver: TRFFallbackSolver, verifier: SolutionVerifier):
        self.solver = solver
        self.verifier = verifier
        self._risk = CalibratedRisk(np.array([0.0, 0.0, 0.0, 1.0]))
        self._policy = SolverPolicy(RiskLevel.HARD, 1, 0, use_fallback=True)

    def solve(self, query: IKQuery) -> IKResult:
        trace = self.solver.solve(query.target, query.previous_q, seed_source="previous")
        verification = self.verifier.check(trace.q, query) if trace.q is not None else None
        accepted = bool(trace.converged and verification is not None and verification.accepted)
        return IKResult(
            q=trace.q if accepted else None,
            accepted=accepted,
            risk=self._risk,
            policy=self._policy,
            verification=verification,
            traces=[trace],
            fallback_used=True,
            reject_reason="" if accepted else trace.reason,
            metadata={
                "total_iterations": trace.iterations,
                "total_function_evaluations": trace.function_evaluations,
            },
        )


def fixed_hybrid(
    kinematics: KinematicsModel,
    candidate_provider: object,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    *,
    candidate_count: int,
    iterations: int,
    include_previous: bool = False,
) -> HybridIK:
    policy = SolverPolicy(
        RiskLevel.MEDIUM,
        candidate_count,
        iterations,
        include_previous=include_previous,
        use_fallback=False,
    )
    return HybridIK(
        kinematics,
        candidate_provider,  # type: ignore[arg-type]
        ConstantRiskProvider(np.array([0.0, 1.0, 0.0, 0.0])),
        dls,
        verifier,
        gate=FixedPolicyGate(policy),  # type: ignore[arg-type]
    )

