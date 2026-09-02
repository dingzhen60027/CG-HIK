"""Verifier-governed runtime for selective activation of learned IK priors."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Mapping, Protocol

import numpy as np

from ..hierarchical_v5.runtime import AlwaysLocalRuntime, LocalSolveOutcome
from ..latency_pilot_v3.benchmark import ProfiledOutcome
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier
from ..types import FloatArray, IKQuery
from .features import PreparedLiteFeatures, prepare_lite_features


REQUIRED_LITE_TIMING_KEYS = (
    "feature_extraction_ns",
    "gate_ns",
    "local_path_ns",
    "robust_path_ns",
    "total_ns",
)


@dataclass(frozen=True)
class LiteFastGateDecision:
    """Normalized one-head decision consumed by the V5-Lite runtime."""

    take_fast_path: bool
    local_success_probability: float | None = None
    threshold: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("local_success_probability", "threshold"):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")


class LiteFastGate(Protocol):
    def decide(self, features: np.ndarray) -> LiteFastGateDecision: ...


class AlwaysHardRuntime(Protocol):
    """Contract implemented by the frozen fixed-HARD profiled cascade."""

    def solve(self, query: IKQuery) -> ProfiledOutcome: ...


@dataclass
class HierarchicalLiteOutcome:
    q: FloatArray | None
    accepted: bool
    route: str
    local_attempted: bool
    local_accepted: bool
    learned_seed_ensemble_invoked: bool
    gate_local_success_probability: float | None
    gate_threshold: float | None
    gate_reason: str
    lite_features: np.ndarray
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


def _coerce_optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError("gate probability or threshold must be scalar")
    return float(array[0])


def _coerce_gate_decision(value: Any) -> LiteFastGateDecision:
    if isinstance(value, LiteFastGateDecision):
        return value
    if isinstance(value, (bool, np.bool_)):
        return LiteFastGateDecision(bool(value), reason="boolean_gate")
    if isinstance(value, Mapping):
        action = str(value.get("action", "")).lower()
        take = value.get(
            "take_fast_path",
            value.get(
                "use_fast_path",
                value.get("choose_fast", action in {"fast", "local", "local_fast"}),
            ),
        )
        return LiteFastGateDecision(
            bool(take),
            _coerce_optional_probability(
                value.get(
                    "local_success_probability",
                    value.get("success_probability", value.get("probability")),
                )
            ),
            _coerce_optional_probability(
                value.get("threshold", value.get("local_success_threshold"))
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
            "lite gate must return LiteFastGateDecision, bool, a mapping, or "
            "an object with take_fast_path/use_fast_path/choose_fast/action"
        )
    return LiteFastGateDecision(
        bool(take),
        _coerce_optional_probability(
            getattr(
                value,
                "local_success_probability",
                getattr(value, "success_probability", getattr(value, "probability", None)),
            )
        ),
        _coerce_optional_probability(
            getattr(value, "threshold", getattr(value, "local_success_threshold", None))
        ),
        str(getattr(value, "reason", action or "object_gate")),
    )


def _invoke_gate(gate: object, features: np.ndarray) -> LiteFastGateDecision:
    if hasattr(gate, "decide"):
        raw = gate.decide(features)  # type: ignore[attr-defined]
    elif hasattr(gate, "predict_one"):
        raw = gate.predict_one(features)  # type: ignore[attr-defined]
    elif hasattr(gate, "predict"):
        raw = gate.predict(features)  # type: ignore[attr-defined]
    else:
        raise TypeError(
            "lite gate must implement decide(features), predict_one(features), "
            "or predict(features)"
        )
    return _coerce_gate_decision(raw)


def _assert_fixed_hard_outcome(outcome: ProfiledOutcome) -> None:
    if str(outcome.entry_action).lower() != "hard":
        raise RuntimeError(
            "V5-Lite robust path must use the fixed always-HARD entry; received "
            f"{outcome.entry_action!r}"
        )
    stages = tuple(str(stage).lower().split(":", 1)[0] for stage in outcome.executed_stages)
    if not stages or any(stage != "hard" for stage in stages):
        raise RuntimeError(
            "V5-Lite robust path executed a non-HARD cascade stage: "
            + ", ".join(str(stage) for stage in outcome.executed_stages)
        )


def _assert_shared_runtime_contract(
    *,
    kinematics: object,
    verifier: SolutionVerifier,
    always_hard_runtime: object,
) -> None:
    slow_kinematics = getattr(always_hard_runtime, "kinematics", None)
    if slow_kinematics is not None and slow_kinematics is not kinematics:
        raise ValueError("local and always-HARD paths must share one kinematics instance")
    timed_verifier = getattr(always_hard_runtime, "_timed_verifier", None)
    slow_verifier = getattr(timed_verifier, "verifier", None)
    if slow_verifier is not None and slow_verifier is not verifier:
        raise ValueError(
            "local and always-HARD paths must share the same deterministic verifier"
        )


class HierarchicalLiteRuntime:
    """One-head lightweight gate followed by local DLS or fixed always-HARD.

    The gate never accepts a command.  A FAST result is returned only after
    the shared deterministic verifier accepts the one-step local solve.  Every
    non-FAST query and every failed FAST attempt calls the complete learned
    fixed-HARD path.
    """

    def __init__(
        self,
        *,
        kinematics: object,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        fast_gate: object,
        always_hard_runtime: AlwaysHardRuntime,
        fast_iterations: int = 1,
        name: str = "hierarchical_cghik_v5_lite",
    ) -> None:
        if fast_iterations <= 0:
            raise ValueError("fast_iterations must be positive")
        _assert_shared_runtime_contract(
            kinematics=kinematics,
            verifier=verifier,
            always_hard_runtime=always_hard_runtime,
        )
        self.name = str(name)
        self.kinematics = kinematics
        self.dls = dls
        self.verifier = verifier
        self.fast_gate = fast_gate
        self.always_hard_runtime = always_hard_runtime
        self.fast_iterations = int(fast_iterations)
        self.local_runtime = AlwaysLocalRuntime(
            dls,
            verifier,
            iterations=self.fast_iterations,
            name=f"{self.name}:local",
        )

    def solve(self, query: IKQuery) -> HierarchicalLiteOutcome:
        total_started = perf_counter_ns()
        timings = {key: 0 for key in REQUIRED_LITE_TIMING_KEYS}

        started = perf_counter_ns()
        prepared: PreparedLiteFeatures = prepare_lite_features(
            self.kinematics,  # type: ignore[arg-type]
            query,
        )
        timings["feature_extraction_ns"] = perf_counter_ns() - started
        for key, value in prepared.timings_ns.items():
            timings[f"feature_detail_{key}"] = int(value)

        started = perf_counter_ns()
        decision = _invoke_gate(self.fast_gate, prepared.features)
        timings["gate_ns"] = perf_counter_ns() - started

        local: LocalSolveOutcome | None = None
        hard: ProfiledOutcome | None = None
        if decision.take_fast_path:
            started = perf_counter_ns()
            local = self.local_runtime.solve(query)
            timings["local_path_ns"] = perf_counter_ns() - started
            timings["local_solver_ns"] = int(local.timings_ns["local_solver_ns"])
            timings["local_verifier_ns"] = int(local.timings_ns["local_verifier_ns"])

        if local is None or not local.accepted:
            started = perf_counter_ns()
            hard = self.always_hard_runtime.solve(query)
            timings["robust_path_ns"] = perf_counter_ns() - started
            _assert_fixed_hard_outcome(hard)
            for key, value in hard.timings_ns.items():
                timings[f"robust_detail_{key}"] = int(value)

        if local is not None and local.accepted:
            q = local.q
            accepted = True
            route = "fast_accept"
            executed_stages = ("local_fast",)
            fallback_used = False
            candidate_count = 0
            verification_reasons = tuple(local.verification_reasons)
            reject_reason = ""
        else:
            if hard is None:  # pragma: no cover - invariant guard
                raise RuntimeError("non-accepted FAST route completed without HARD recovery")
            q = None if hard.q is None else np.asarray(hard.q, dtype=np.float64).copy()
            accepted = bool(hard.accepted)
            executed_stages = (
                (("local_fast",) if local is not None else ())
                + tuple(str(stage) for stage in hard.executed_stages)
            )
            fallback_used = bool(hard.fallback_used)
            candidate_count = int(hard.candidate_count)
            verification_reasons = tuple(hard.verification_reasons)
            reject_reason = str(hard.reject_reason)
            if local is not None:
                route = (
                    "fast_fail_hard_recovery"
                    if accepted
                    else "fast_fail_hard_failure"
                )
            else:
                route = "hard_direct_accept" if accepted else "hard_direct_failure"

        local_fev = 0 if local is None else int(local.function_evaluations)
        hard_fev = 0 if hard is None else int(hard.function_evaluations)
        local_iterations = 0 if local is None else int(local.iterations)
        hard_iterations = 0 if hard is None else int(hard.iterations)
        local_reasons = () if local is None else tuple(local.verification_reasons)

        outcome = HierarchicalLiteOutcome(
            q=q,
            accepted=accepted,
            route=route,
            local_attempted=local is not None,
            local_accepted=bool(local is not None and local.accepted),
            learned_seed_ensemble_invoked=hard is not None,
            gate_local_success_probability=decision.local_success_probability,
            gate_threshold=decision.threshold,
            gate_reason=decision.reason,
            lite_features=prepared.features.copy(),
            executed_stages=executed_stages,
            function_evaluations=local_fev + hard_fev,
            iterations=local_iterations + hard_iterations,
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
                for key in REQUIRED_LITE_TIMING_KEYS
                if key != "total_ns"
            ),
            0,
        )
        return outcome


__all__ = [
    "AlwaysHardRuntime",
    "HierarchicalLiteOutcome",
    "HierarchicalLiteRuntime",
    "LiteFastGate",
    "LiteFastGateDecision",
    "REQUIRED_LITE_TIMING_KEYS",
]
