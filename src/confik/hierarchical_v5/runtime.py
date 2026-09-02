"""Verifier-governed hierarchical runtime for online inverse kinematics.

The hierarchy has exactly two compute levels:

1. seed-free geometric features and a cheap fast gate;
2. either a small-budget previous-state DLS attempt or the existing learned
   CG-HIK robust path.

The fast gate never emits a command.  A local command is returned only after
the shared :class:`~confik.solvers.verifier.SolutionVerifier` accepts it.  If
the local attempt fails, the complete failure cost is retained and the query
enters the learned robust path.  That slow path is constrained to start at
MEDIUM or HARD; it can never repeat EASY.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Mapping, Protocol

import numpy as np

from ..counterfactual_v4.runtime_v4 import (
    PolicyRiskEngine,
    V4ProfiledRuntime,
)
from ..latency_pilot_v3.benchmark import ProfiledCascadeRuntime, ProfiledOutcome
from ..runtime.cascade import EntryAction
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier
from ..types import CalibratedRisk, FloatArray, IKQuery, SolveTrace, VerificationResult
from .features import CHEAP_FEATURE_DIM, PreparedCheapFeatures, prepare_cheap_features


REQUIRED_HIERARCHICAL_TIMING_KEYS = (
    "cheap_feature_ns",
    "gate_ns",
    "local_solver_ns",
    "local_verifier_ns",
    "slow_ns",
    "total_ns",
)


@dataclass(frozen=True)
class FastGateDecision:
    """Normalized binary decision returned by a first-level gate."""

    take_fast_path: bool
    local_success_probability: float | None = None
    latency_benefit_probability: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("local_success_probability", "latency_benefit_probability"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")


class FastGate(Protocol):
    """Minimal policy contract used by :class:`HierarchicalRuntime`."""

    def decide(self, features: np.ndarray) -> FastGateDecision: ...


class SlowRuntime(Protocol):
    """Verifier-backed, learned-seed slow runtime contract."""

    def solve(self, query: IKQuery) -> ProfiledOutcome: ...


@dataclass(frozen=True)
class LocalSolveOutcome:
    """Result of exactly one bounded previous-state DLS stage."""

    q: FloatArray | None
    accepted: bool
    trace: SolveTrace
    verification: VerificationResult | None
    verification_reasons: tuple[str, ...]
    function_evaluations: int
    iterations: int
    timings_ns: Mapping[str, int]


@dataclass
class HierarchicalOutcome:
    """Common instrumented result emitted by the hierarchical method."""

    q: FloatArray | None
    accepted: bool
    route: str
    local_attempted: bool
    local_accepted: bool
    learned_seed_ensemble_invoked: bool
    gate_local_success_probability: float | None
    gate_latency_benefit_probability: float | None
    gate_reason: str
    cheap_features: np.ndarray
    slow_entry_action: str | None
    executed_stages: tuple[str, ...]
    function_evaluations: int
    iterations: int
    fallback_used: bool
    candidate_count: int
    verification_reasons: tuple[str, ...]
    local_verification_reasons: tuple[str, ...]
    reject_reason: str
    timings_ns: dict[str, int]

    @property
    def fast_path_hit(self) -> bool:
        return self.local_attempted and self.local_accepted

    @property
    def recovered_after_fast_failure(self) -> bool:
        return self.local_attempted and not self.local_accepted and self.accepted


def _empty_top_level_timings() -> dict[str, int]:
    return {key: 0 for key in REQUIRED_HIERARCHICAL_TIMING_KEYS}


def _coerce_probability(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError("fast-gate probability must be scalar")
    return float(array[0])


def _coerce_gate_decision(value: Any) -> FastGateDecision:
    """Accept the native decision plus simple mapping/bool test doubles."""

    if isinstance(value, FastGateDecision):
        return value
    if isinstance(value, (bool, np.bool_)):
        return FastGateDecision(bool(value), reason="boolean_gate")
    if isinstance(value, Mapping):
        action = str(value.get("action", "")).lower()
        take = value.get(
            "take_fast_path",
            value.get(
                "use_fast_path",
                value.get(
                    "choose_fast", action in {"fast", "local", "local_fast"}
                ),
            ),
        )
        return FastGateDecision(
            bool(take),
            _coerce_probability(
                value.get("local_success_probability", value.get("success_probability"))
            ),
            _coerce_probability(
                value.get("latency_benefit_probability", value.get("benefit_probability"))
            ),
            str(value.get("reason", action or "mapping_gate")),
        )

    action = str(getattr(value, "action", "")).lower()
    take = getattr(value, "take_fast_path", None)
    if take is None:
        take = getattr(value, "use_fast_path", None)
    if take is None:
        take = getattr(value, "choose_fast", None)
    if take is None and action:
        take = action in {"fast", "local", "local_fast"}
    if take is None:
        raise TypeError(
            "fast gate must return FastGateDecision, bool, a mapping, or an object "
            "with take_fast_path/use_fast_path/action"
        )
    return FastGateDecision(
        bool(take),
        _coerce_probability(
            getattr(
                value,
                "local_success_probability",
                getattr(value, "success_probability", None),
            )
        ),
        _coerce_probability(
            getattr(
                value,
                "latency_benefit_probability",
                getattr(value, "benefit_probability", None),
            )
        ),
        str(getattr(value, "reason", action or "object_gate")),
    )


def _invoke_fast_gate(gate: object, features: np.ndarray) -> FastGateDecision:
    if hasattr(gate, "decide"):
        raw = gate.decide(features)  # type: ignore[attr-defined]
    elif hasattr(gate, "predict"):
        # ``predict`` is supported for small deployment adapters, while policy
        # implementations should prefer the explicit ``decide`` contract.
        raw = gate.predict(features)  # type: ignore[attr-defined]
    else:
        raise TypeError("fast gate must implement decide(features) or predict(features)")
    return _coerce_gate_decision(raw)


class StandaloneLocalRuntime:
    """Always-local baseline using the same DLS instance and verifier as v5."""

    def __init__(
        self,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        *,
        iterations: int = 1,
        name: str = "always_local",
    ):
        if iterations <= 0:
            raise ValueError("local DLS iteration budget must be positive")
        self.dls = dls
        self.verifier = verifier
        self.iterations = int(iterations)
        self.name = name

    def solve(self, query: IKQuery) -> LocalSolveOutcome:
        solver_started = perf_counter_ns()
        trace = self.dls.solve(
            query.target,
            query.previous_q,
            self.iterations,
            seed_source="hierarchical_v5:previous",
        )
        solver_ns = perf_counter_ns() - solver_started

        verification: VerificationResult | None = None
        verifier_ns = 0
        if trace.q is not None:
            verifier_started = perf_counter_ns()
            verification = self.verifier.check(trace.q, query)
            verifier_ns = perf_counter_ns() - verifier_started
        # Solver convergence can prevent acceptance, but it can never grant it:
        # every returned command still requires the deterministic verifier.
        accepted = bool(
            trace.converged
            and verification is not None
            and verification.accepted
        )
        reasons = tuple(verification.reasons) if verification is not None else ()
        return LocalSolveOutcome(
            q=(
                np.asarray(trace.q, dtype=np.float64).copy()
                if accepted and trace.q is not None
                else None
            ),
            accepted=accepted,
            trace=trace,
            verification=verification,
            verification_reasons=reasons,
            function_evaluations=int(trace.function_evaluations),
            iterations=int(trace.iterations),
            timings_ns={
                "local_solver_ns": int(solver_ns),
                "local_verifier_ns": int(verifier_ns),
                "total_ns": int(solver_ns + verifier_ns),
            },
        )


# More concise public alias used by pilot code.
AlwaysLocalRuntime = StandaloneLocalRuntime
LocalPathRuntime = StandaloneLocalRuntime


class NoEasyPolicyEntryGate:
    """Map frozen v4 policy decisions onto the MEDIUM/HARD-only slow path.

    EASY, MEDIUM, and DEFER all enter MEDIUM.  A failed MEDIUM stage escalates
    through the unchanged cascade to HARD, whose stage still contains learned
    and previous-state DLS attempts followed by the frozen fallback.  REJECT
    remains a zero-numerical-solver command rejection, although learned seed and
    policy inference have necessarily already run on this second level.
    """

    def __init__(self, engine: PolicyRiskEngine):
        self.engine = engine
        self.last_policy_action: str | None = None
        self.last_mapped_entry: EntryAction | None = None

    def choose(self, risk: CalibratedRisk) -> EntryAction:
        del risk
        decision = self.engine.last_decision
        if decision is None:
            raise RuntimeError("v5 slow gate was called before v4 policy inference")
        action = str(decision.action)
        try:
            mapped = {
                "easy": EntryAction.MEDIUM,
                "medium": EntryAction.MEDIUM,
                "defer": EntryAction.MEDIUM,
                "hard": EntryAction.HARD,
                "reject": EntryAction.REJECT,
            }[action]
        except KeyError as error:
            raise RuntimeError(f"unsupported v4 slow-path action: {action!r}") from error
        self.last_policy_action = action
        self.last_mapped_entry = mapped
        return mapped


def wrap_no_easy_profiled_runtime(
    *,
    name: str,
    policy: object,
    kinematics: object,
    seed_engine: object,
    dls: object,
    verifier: object,
    seed_bank: object,
    fallback: object,
    cascade_config: object,
) -> V4ProfiledRuntime:
    """Build the learned second level while making EASY structurally unreachable."""

    engine = PolicyRiskEngine(policy)  # type: ignore[arg-type]
    gate = NoEasyPolicyEntryGate(engine)
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
    runtime = V4ProfiledRuntime(base, engine)
    # These public diagnostics let the pilot attest that it used the intended
    # v5 slow-path mapping without reaching into the cascade internals.
    runtime.no_easy_enforced = True  # type: ignore[attr-defined]
    runtime.no_easy_gate = gate  # type: ignore[attr-defined]
    return runtime


def _assert_no_easy_slow_outcome(outcome: ProfiledOutcome) -> None:
    offending = [
        stage
        for stage in outcome.executed_stages
        if str(stage).lower().split(":", 1)[0] == "easy"
    ]
    if offending:
        raise RuntimeError(
            "hierarchical v5 slow path executed forbidden EASY stage(s): "
            + ", ".join(str(stage) for stage in offending)
        )


class HierarchicalRuntime:
    """Seed-free fast gate with verified local return and robust recovery."""

    def __init__(
        self,
        *,
        kinematics: object,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        fast_gate: object,
        slow_runtime: SlowRuntime,
        fast_iterations: int = 1,
        name: str = "hierarchical_cg_hik_v5",
    ):
        if fast_iterations <= 0:
            raise ValueError("fast_iterations must be positive")
        self.name = name
        self.kinematics = kinematics
        self.dls = dls
        self.verifier = verifier
        self.fast_gate = fast_gate
        self.slow_runtime = slow_runtime
        self.fast_iterations = int(fast_iterations)
        self.local_runtime = StandaloneLocalRuntime(
            dls,
            verifier,
            iterations=fast_iterations,
            name=f"{name}:local",
        )

    def solve(self, query: IKQuery) -> HierarchicalOutcome:
        total_started = perf_counter_ns()
        timings = _empty_top_level_timings()

        feature_started = perf_counter_ns()
        prepared: PreparedCheapFeatures = prepare_cheap_features(
            self.kinematics,  # type: ignore[arg-type]
            self.dls,
            query,
        )
        timings["cheap_feature_ns"] = perf_counter_ns() - feature_started
        for key, value in prepared.timings_ns.items():
            timings[f"cheap_detail_{key}"] = int(value)

        gate_started = perf_counter_ns()
        decision = _invoke_fast_gate(self.fast_gate, prepared.features)
        timings["gate_ns"] = perf_counter_ns() - gate_started

        local: LocalSolveOutcome | None = None
        slow: ProfiledOutcome | None = None
        if decision.take_fast_path:
            local = self.local_runtime.solve(query)
            timings["local_solver_ns"] = int(local.timings_ns["local_solver_ns"])
            timings["local_verifier_ns"] = int(local.timings_ns["local_verifier_ns"])

        if local is None or not local.accepted:
            slow_started = perf_counter_ns()
            slow = self.slow_runtime.solve(query)
            timings["slow_ns"] = perf_counter_ns() - slow_started
            _assert_no_easy_slow_outcome(slow)
            for key, value in slow.timings_ns.items():
                timings[f"slow_detail_{key}"] = int(value)

        if local is not None and local.accepted:
            route = "fast_accept"
            q = local.q
            accepted = True
            slow_entry = None
            executed = ("local_fast",)
            fallback_used = False
            candidate_count = 0
            verification_reasons = local.verification_reasons
            reject_reason = ""
        else:
            if slow is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("non-accepted local route completed without slow recovery")
            q = None if slow.q is None else np.asarray(slow.q, dtype=np.float64).copy()
            accepted = bool(slow.accepted)
            slow_entry = str(slow.entry_action)
            executed = (
                (("local_fast",) if local is not None else ())
                + tuple(str(stage) for stage in slow.executed_stages)
            )
            fallback_used = bool(slow.fallback_used)
            candidate_count = int(slow.candidate_count)
            verification_reasons = tuple(slow.verification_reasons)
            reject_reason = str(slow.reject_reason)
            if local is not None:
                route = (
                    "fast_fail_robust_recovery"
                    if accepted
                    else "fast_fail_robust_failure"
                )
            elif slow_entry == "reject":
                route = "robust_direct_reject"
            else:
                route = "robust_direct_accept" if accepted else "robust_direct_failure"

        local_fev = 0 if local is None else int(local.function_evaluations)
        slow_fev = 0 if slow is None else int(slow.function_evaluations)
        local_iterations = 0 if local is None else int(local.iterations)
        slow_iterations = 0 if slow is None else int(slow.iterations)
        local_reasons = () if local is None else tuple(local.verification_reasons)

        outcome = HierarchicalOutcome(
            q=q,
            accepted=accepted,
            route=route,
            local_attempted=local is not None,
            local_accepted=bool(local is not None and local.accepted),
            learned_seed_ensemble_invoked=slow is not None,
            gate_local_success_probability=decision.local_success_probability,
            gate_latency_benefit_probability=decision.latency_benefit_probability,
            gate_reason=decision.reason,
            cheap_features=prepared.features.copy(),
            slow_entry_action=slow_entry,
            executed_stages=executed,
            function_evaluations=local_fev + slow_fev,
            iterations=local_iterations + slow_iterations,
            fallback_used=fallback_used,
            candidate_count=candidate_count,
            verification_reasons=verification_reasons,
            local_verification_reasons=local_reasons,
            reject_reason=reject_reason,
            timings_ns=timings,
        )
        timings["total_ns"] = perf_counter_ns() - total_started
        timings["unattributed_ns"] = max(
            timings["total_ns"]
            - sum(
                timings[key]
                for key in REQUIRED_HIERARCHICAL_TIMING_KEYS
                if key != "total_ns"
            ),
            0,
        )
        return outcome


__all__ = [
    "AlwaysLocalRuntime",
    "FastGate",
    "FastGateDecision",
    "HierarchicalOutcome",
    "HierarchicalRuntime",
    "LocalSolveOutcome",
    "LocalPathRuntime",
    "NoEasyPolicyEntryGate",
    "REQUIRED_HIERARCHICAL_TIMING_KEYS",
    "StandaloneLocalRuntime",
    "wrap_no_easy_profiled_runtime",
]
