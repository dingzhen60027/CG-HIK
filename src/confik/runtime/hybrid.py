from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from ..geometry import pose_distance
from ..kinematics.base import KinematicsModel
from ..solvers.dls import AdaptiveDLS
from ..solvers.fallback import KDTreeSeedBank, TRFFallbackSolver
from ..solvers.verifier import SolutionVerifier
from ..types import (
    CalibratedRisk,
    CandidateSet,
    FloatArray,
    IKQuery,
    IKResult,
    SolveTrace,
)
from .gate import ConfidenceGate


class CandidateProvider(Protocol):
    def candidates(self, query: IKQuery) -> CandidateSet: ...


class RiskProvider(Protocol):
    def predict(self, features: FloatArray) -> CalibratedRisk: ...


def risk_features(
    kinematics: KinematicsModel,
    query: IKQuery,
    candidates: CandidateSet,
) -> FloatArray:
    best = candidates.joints[0]
    seed_pose = kinematics.forward(best)
    position_error, orientation_error = pose_distance(query.target, seed_pose)
    current_pose = kinematics.forward(query.previous_q)
    step_position, step_orientation = pose_distance(query.target, current_pose)
    margin = float(np.min(kinematics.joint_margin(best)))
    joint_step = float(np.linalg.norm(kinematics.difference(best, query.previous_q)))
    return np.array(
        [
            position_error,
            orientation_error,
            candidates.uncertainty_mean,
            candidates.uncertainty_max,
            kinematics.min_singular_value(best),
            margin,
            joint_step,
            step_position,
            step_orientation,
        ],
        dtype=np.float64,
    )


class HybridIK:
    def __init__(
        self,
        kinematics: KinematicsModel,
        candidate_provider: CandidateProvider,
        risk_provider: RiskProvider,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        gate: ConfidenceGate | None = None,
        seed_bank: KDTreeSeedBank | None = None,
        fallback: TRFFallbackSolver | None = None,
        fallback_seed_count: int = 3,
    ):
        self.kinematics = kinematics
        self.candidate_provider = candidate_provider
        self.risk_provider = risk_provider
        self.dls = dls
        self.verifier = verifier
        self.gate = gate or ConfidenceGate()
        self.seed_bank = seed_bank
        self.fallback = fallback
        self.fallback_seed_count = fallback_seed_count

    def solve(self, query: IKQuery) -> IKResult:
        started = perf_counter()
        if query.previous_q.shape != (self.kinematics.nq,):
            raise ValueError("query joint dimension does not match the robot")
        candidates = self.candidate_provider.candidates(query)
        if candidates.joints.shape[0] == 0:
            raise ValueError("candidate provider returned no seeds")
        features = risk_features(self.kinematics, query, candidates)
        risk = self.risk_provider.predict(features)
        policy = self.gate.make_policy(risk)
        traces: list[SolveTrace] = []
        attempted: list[tuple[FloatArray, str]] = []
        for q, source in zip(
            candidates.joints[: policy.learned_candidates],
            candidates.source[: policy.learned_candidates],
            strict=False,
        ):
            attempted.append((q, source))
        if policy.include_previous:
            attempted.append((query.previous_q, "previous"))

        for seed, source in attempted:
            trace = self.dls.solve(
                query.target,
                seed,
                policy.dls_iterations_per_candidate,
                seed_source=source,
            )
            traces.append(trace)
            if trace.q is not None:
                verification = self.verifier.check(trace.q, query)
                if trace.converged and verification.accepted:
                    return IKResult(
                        q=trace.q,
                        accepted=True,
                        risk=risk,
                        policy=policy,
                        verification=verification,
                        traces=traces,
                        fallback_used=False,
                        metadata=self._metadata(started, candidates, features, traces),
                    )

        fallback_used = False
        if policy.use_fallback and self.seed_bank is not None and self.fallback is not None:
            fallback_used = True
            fallback_seeds = self.seed_bank.query(
                query.target,
                query.previous_q,
                k=self.fallback_seed_count,
            )
            for index, seed in enumerate(fallback_seeds):
                trace = self.fallback.solve(query.target, seed, seed_source=f"kdtree:{index}")
                traces.append(trace)
                if trace.q is not None:
                    verification = self.verifier.check(trace.q, query)
                    if trace.converged and verification.accepted:
                        return IKResult(
                            q=trace.q,
                            accepted=True,
                            risk=risk,
                            policy=policy,
                            verification=verification,
                            traces=traces,
                            fallback_used=True,
                            metadata=self._metadata(started, candidates, features, traces),
                        )

        return IKResult(
            q=None,
            accepted=False,
            risk=risk,
            policy=policy,
            verification=None,
            traces=traces,
            fallback_used=fallback_used,
            reject_reason=self._reject_reason(traces),
            metadata=self._metadata(started, candidates, features, traces),
        )

    @staticmethod
    def _reject_reason(traces: list[SolveTrace]) -> str:
        if not traces:
            return "no_solver_attempt"
        reasons = [trace.reason for trace in traces]
        return "all_attempts_failed:" + ",".join(reasons)

    @staticmethod
    def _metadata(
        started: float,
        candidates: CandidateSet,
        features: FloatArray,
        traces: list[SolveTrace],
    ) -> dict[str, object]:
        return {
            "elapsed_seconds": perf_counter() - started,
            "candidate_count": int(candidates.joints.shape[0]),
            "risk_features": features.tolist(),
            "total_iterations": int(sum(trace.iterations for trace in traces)),
            "total_function_evaluations": int(sum(trace.function_evaluations for trace in traces)),
        }
