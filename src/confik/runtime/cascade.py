from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter
from typing import Protocol

import numpy as np

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
    RiskLevel,
    SolveTrace,
    SolverPolicy,
)
from .hybrid import CandidateProvider, RiskProvider, risk_features


class EntryAction(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    REJECT = 3

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel(self.name.lower())


class EntryGate(Protocol):
    def choose(self, risk: CalibratedRisk) -> EntryAction: ...


@dataclass(frozen=True)
class ActionGateConfig:
    easy_probability: float = 0.70
    hard_probability: float = 0.45
    reject_probability: float = 0.85


class CalibratedActionGate:
    """Choose a cascade entry action from locked calibrated probabilities."""

    def __init__(self, config: ActionGateConfig | None = None):
        self.config = config or ActionGateConfig()

    def choose(self, risk: CalibratedRisk) -> EntryAction:
        config = self.config
        p_easy = risk.probability("easy")
        p_hard = risk.probability("hard")
        p_reject = risk.probability("fail")
        if p_reject >= config.reject_probability:
            return EntryAction.REJECT
        if p_easy >= config.easy_probability:
            return EntryAction.EASY
        if p_hard >= config.hard_probability or int(np.argmax(risk.probabilities[:3])) == 2:
            return EntryAction.HARD
        return EntryAction.MEDIUM


class FixedEntryGate:
    def __init__(self, action: EntryAction = EntryAction.EASY):
        self.action = action

    def choose(self, risk: CalibratedRisk) -> EntryAction:
        del risk
        return self.action


@dataclass(frozen=True)
class CascadeConfig:
    easy_iterations: int = 1
    medium_iterations: int = 8
    hard_iterations: int = 25
    hard_learned_candidates: int = 1
    fallback_seed_count: int = 2
    escalate_on_failure: bool = True


@dataclass
class StageOutcome:
    accepted: bool
    q: FloatArray | None
    verification: object | None
    traces: list[SolveTrace]
    fallback_used: bool


class CascadedHybridIK:
    """Verified heterogeneous cascade with a learned entry action.

    EASY uses the previous state and a tiny DLS budget. MEDIUM uses the best
    history-conditioned learned seed. HARD uses longer DLS attempts from
    learned and previous seeds followed by bounded TRF from heterogeneous
    seeds. A failed non-reject entry escalates to later stages, preserving the
    robust-cascade success envelope except for explicit confidence rejection.
    """

    def __init__(
        self,
        kinematics: KinematicsModel,
        candidate_provider: CandidateProvider,
        risk_provider: RiskProvider,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        *,
        gate: EntryGate,
        seed_bank: KDTreeSeedBank | None = None,
        fallback: TRFFallbackSolver | None = None,
        config: CascadeConfig | None = None,
    ):
        self.kinematics = kinematics
        self.candidate_provider = candidate_provider
        self.risk_provider = risk_provider
        self.dls = dls
        self.verifier = verifier
        self.gate = gate
        self.seed_bank = seed_bank
        self.fallback = fallback
        self.config = config or CascadeConfig()

    def _verified_dls(
        self,
        query: IKQuery,
        seed: FloatArray,
        iterations: int,
        source: str,
    ) -> StageOutcome:
        trace = self.dls.solve(query.target, seed, iterations, seed_source=source)
        verification = self.verifier.check(trace.q, query) if trace.q is not None else None
        accepted = bool(trace.converged and verification is not None and verification.accepted)
        return StageOutcome(
            accepted=accepted,
            q=trace.q if accepted else None,
            verification=verification,
            traces=[trace],
            fallback_used=False,
        )

    def run_stage(
        self,
        query: IKQuery,
        candidates: CandidateSet,
        action: EntryAction,
    ) -> StageOutcome:
        config = self.config
        if action == EntryAction.EASY:
            return self._verified_dls(
                query,
                query.previous_q,
                config.easy_iterations,
                "easy:previous",
            )
        if action == EntryAction.MEDIUM:
            return self._verified_dls(
                query,
                candidates.joints[0],
                config.medium_iterations,
                "medium:learned",
            )
        if action == EntryAction.REJECT:
            return StageOutcome(False, None, None, [], False)

        traces: list[SolveTrace] = []
        learned_count = min(config.hard_learned_candidates, len(candidates.joints))
        dls_seeds: list[tuple[FloatArray, str]] = [
            (candidates.joints[index], f"hard:learned:{index}") for index in range(learned_count)
        ]
        dls_seeds.append((query.previous_q, "hard:previous"))
        for seed, source in dls_seeds:
            outcome = self._verified_dls(query, seed, config.hard_iterations, source)
            traces.extend(outcome.traces)
            if outcome.accepted:
                outcome.traces = traces
                return outcome

        fallback_used = False
        if self.fallback is not None:
            fallback_used = True
            fallback_seeds: list[tuple[FloatArray, str]] = [
                (query.previous_q, "trf:previous"),
                (candidates.joints[0], "trf:learned"),
            ]
            if self.seed_bank is not None and config.fallback_seed_count > 0:
                retrieved = self.seed_bank.query(
                    query.target,
                    query.previous_q,
                    k=config.fallback_seed_count,
                )
                fallback_seeds.extend(
                    (seed, f"trf:kdtree:{index}") for index, seed in enumerate(retrieved)
                )
            unique: list[tuple[FloatArray, str]] = []
            for seed, source in fallback_seeds:
                if all(
                    np.max(np.abs(self.kinematics.difference(seed, kept))) >= 1e-6
                    for kept, _ in unique
                ):
                    unique.append((seed, source))
            for seed, source in unique:
                trace = self.fallback.solve(query.target, seed, seed_source=source)
                traces.append(trace)
                verification = self.verifier.check(trace.q, query) if trace.q is not None else None
                if trace.converged and verification is not None and verification.accepted:
                    return StageOutcome(True, trace.q, verification, traces, True)
        return StageOutcome(False, None, None, traces, fallback_used)

    def oracle_action(self, query: IKQuery, candidates: CandidateSet | None = None) -> tuple[EntryAction, list[StageOutcome]]:
        candidates = candidates or self.candidate_provider.candidates(query)
        outcomes: list[StageOutcome] = []
        for action in (EntryAction.EASY, EntryAction.MEDIUM, EntryAction.HARD):
            outcome = self.run_stage(query, candidates, action)
            outcomes.append(outcome)
            if outcome.accepted:
                return action, outcomes
        return EntryAction.REJECT, outcomes

    def solve(self, query: IKQuery) -> IKResult:
        started = perf_counter()
        candidates = self.candidate_provider.candidates(query)
        features = risk_features(self.kinematics, query, candidates)
        risk = self.risk_provider.predict(features)
        entry = self.gate.choose(risk)
        policy = SolverPolicy(
            entry.risk_level,
            learned_candidates=(0 if entry in {EntryAction.EASY, EntryAction.REJECT} else 1),
            dls_iterations_per_candidate={
                EntryAction.EASY: self.config.easy_iterations,
                EntryAction.MEDIUM: self.config.medium_iterations,
                EntryAction.HARD: self.config.hard_iterations,
                EntryAction.REJECT: 0,
            }[entry],
            include_previous=entry in {EntryAction.EASY, EntryAction.HARD},
            use_fallback=entry == EntryAction.HARD,
        )
        traces: list[SolveTrace] = []
        executed: list[str] = []
        fallback_used = False
        if entry == EntryAction.REJECT:
            return IKResult(
                q=None,
                accepted=False,
                risk=risk,
                policy=policy,
                verification=None,
                traces=[],
                fallback_used=False,
                reject_reason="confidence_reject",
                metadata=self._metadata(started, features, candidates, entry, executed, traces),
            )

        last_verification = None
        stages = list(range(int(entry), int(EntryAction.HARD) + 1))
        if not self.config.escalate_on_failure:
            stages = stages[:1]
        for stage_index in stages:
            action = EntryAction(stage_index)
            executed.append(action.name.lower())
            outcome = self.run_stage(query, candidates, action)
            traces.extend(outcome.traces)
            fallback_used = fallback_used or outcome.fallback_used
            last_verification = outcome.verification
            if outcome.accepted:
                return IKResult(
                    q=outcome.q,
                    accepted=True,
                    risk=risk,
                    policy=policy,
                    verification=outcome.verification,  # type: ignore[arg-type]
                    traces=traces,
                    fallback_used=fallback_used,
                    metadata=self._metadata(started, features, candidates, entry, executed, traces),
                )
        return IKResult(
            q=None,
            accepted=False,
            risk=risk,
            policy=policy,
            verification=last_verification,  # type: ignore[arg-type]
            traces=traces,
            fallback_used=fallback_used,
            reject_reason="all_cascade_stages_failed",
            metadata=self._metadata(started, features, candidates, entry, executed, traces),
        )

    @staticmethod
    def _metadata(
        started: float,
        features: FloatArray,
        candidates: CandidateSet,
        entry: EntryAction,
        executed: list[str],
        traces: list[SolveTrace],
    ) -> dict[str, object]:
        return {
            "elapsed_seconds": perf_counter() - started,
            "entry_action": entry.name.lower(),
            "executed_stages": list(executed),
            "candidate_count": len(candidates.joints),
            "risk_features": features.tolist(),
            "total_iterations": int(sum(trace.iterations for trace in traces)),
            "total_function_evaluations": int(sum(trace.function_evaluations for trace in traces)),
        }

