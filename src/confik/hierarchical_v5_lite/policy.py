"""Calibration-only, cost-sensitive threshold policy for V5-Lite.

The single learned score never accepts a command.  It chooses whether to try
one verified local DLS step before the frozen always-hard path.  Thresholds are
selected from measured five-repeat component costs on the calibration role;
test and policy-validation roles are rejected at the API boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .model import CALIBRATION_ROLE, LiteGateOutput


FAST = "fast"
ROBUST = "robust"
FINAL_ACTIONS = (FAST, ROBUST)
TIMING_REPEATS = 5


class LiteGateBackend(Protocol):
    def predict_one(self, features: np.ndarray) -> LiteGateOutput: ...


def _require_calibration_role(role: str, *, operation: str) -> str:
    normalized = str(role).strip()
    lowered = normalized.lower()
    if "test" in lowered or "policy_validation" in lowered:
        raise ValueError(f"{operation} forbids test and policy-validation roles")
    if normalized != CALIBRATION_ROLE:
        raise ValueError(
            f"{operation} requires role {CALIBRATION_ROLE!r}; got {normalized!r}"
        )
    return normalized


def _safe_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    state = {} if values is None else dict(values)
    try:
        return json.loads(json.dumps(state, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("policy provenance must be JSON serializable") from error


def _probabilities(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64).reshape(-1)
    if probability.size == 0 or not np.all(np.isfinite(probability)):
        raise ValueError("local_success_probability must be a non-empty finite vector")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("local_success_probability must lie in [0, 1]")
    return probability


def _boolean_vector(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return numeric.astype(bool)


def _latency_matrix(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    samples = np.asarray(values)
    if samples.shape != (count, TIMING_REPEATS):
        raise ValueError(
            f"{name} must have shape ({count}, {TIMING_REPEATS}); "
            "V5-Lite selection requires actual five-repeat measurements"
        )
    numeric = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0):
        raise ValueError(f"{name} must contain finite nonnegative nanoseconds")
    return numeric


def _threshold_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(sorted(set(float(value) for value in values)))
    if not grid or any(not np.isfinite(value) or not 0 <= value <= 1 for value in grid):
        raise ValueError("threshold_grid must contain finite values in [0, 1]")
    return grid


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


@dataclass(frozen=True)
class LiteGatePolicyConfig:
    local_success_threshold: float
    calibration_role: str = CALIBRATION_ROLE
    calibration_count: int = 0
    timing_repeats: int = TIMING_REPEATS
    success_contract: str = "per_query_equal_to_always_hard"

    def __post_init__(self) -> None:
        if not 0 <= float(self.local_success_threshold) <= 1:
            raise ValueError("local_success_threshold must lie in [0, 1]")
        _require_calibration_role(
            self.calibration_role, operation="LiteGatePolicyConfig"
        )
        if self.calibration_count <= 0:
            raise ValueError("calibration_count must be positive")
        if self.timing_repeats != TIMING_REPEATS:
            raise ValueError("V5-Lite policy requires exactly five timing repeats")
        if self.success_contract != "per_query_equal_to_always_hard":
            raise ValueError("V5-Lite success contract cannot be relaxed")


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    """Pre-registered candidate grid and immutable lexicographic objective."""

    threshold_grid: tuple[float, ...] = (
        0.0,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        0.995,
        1.0,
    )
    timing_repeats: int = TIMING_REPEATS
    primary_aggregation: str = "per_query_median_then_cross_query_percentile"

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold_grid", _threshold_grid(self.threshold_grid))
        if 1.0 not in self.threshold_grid:
            raise ValueError("threshold grid must include the never-FAST 1.0 sentinel")
        if self.timing_repeats != TIMING_REPEATS:
            raise ValueError("threshold selection requires exactly five repeats")
        if self.primary_aggregation != (
            "per_query_median_then_cross_query_percentile"
        ):
            raise ValueError("V5-Lite latency aggregation cannot be changed")


@dataclass(frozen=True)
class LiteGateDecision:
    action: str
    choose_fast: bool
    reason: str
    local_success_probability: float
    local_success_threshold: float

    def __post_init__(self) -> None:
        if self.action not in FINAL_ACTIONS:
            raise ValueError(f"unsupported V5-Lite action: {self.action}")
        if self.choose_fast != (self.action == FAST):
            raise ValueError("choose_fast must agree with action")
        if not 0 <= self.local_success_probability <= 1:
            raise ValueError("local_success_probability must lie in [0, 1]")


def select_thresholds(
    local_success_probability: np.ndarray,
    forced_fast_total_samples_ns: np.ndarray,
    forced_robust_total_samples_ns: np.ndarray,
    forced_fast_verified_success: np.ndarray,
    forced_robust_verified_success: np.ndarray,
    forced_fast_local_verified_success: np.ndarray,
    forced_fast_seed_invoked: np.ndarray,
    forced_robust_seed_invoked: np.ndarray,
    primary_mask: np.ndarray,
    *,
    role: str,
    config: ThresholdSelectionConfig | None = None,
) -> tuple[LiteGatePolicyConfig, dict[str, Any]]:
    """Select one threshold using measured calibration counterfactual costs.

    The two latency inputs are aligned ``(query, repeat)`` matrices measured at
    the complete :class:`HierarchicalLiteRuntime` API boundary.  The forced
    FAST arm therefore includes feature extraction, the real TorchScript gate,
    local solve and verifier, result construction, and (when local verification
    fails) the actual fixed-HARD fallback.  The forced ROBUST arm includes the
    same front end followed directly by fixed HARD.  Candidate policies only
    select one of these two exact measured arms for each calibration query.

    Each candidate must first reproduce the forced-ROBUST (and therefore
    always-HARD) verified-success vector exactly.  Eligible candidates are then
    ordered by P95, P50, learned-seed invocation rate, and finally the higher
    threshold.  Latency quantiles are computed after taking the median of the
    five repeats for each query; repeat/query flattening is forbidden.
    """

    selected_role = _require_calibration_role(
        role, operation="select_thresholds"
    )
    selection = config or ThresholdSelectionConfig()
    probability = _probabilities(local_success_probability)
    count = len(probability)
    fast_total = _latency_matrix(
        forced_fast_total_samples_ns,
        count,
        name="forced_fast_total_samples_ns",
    )
    robust_total = _latency_matrix(
        forced_robust_total_samples_ns,
        count,
        name="forced_robust_total_samples_ns",
    )
    fast_success = _boolean_vector(
        forced_fast_verified_success,
        count,
        name="forced_fast_verified_success",
    )
    robust_success = _boolean_vector(
        forced_robust_verified_success,
        count,
        name="forced_robust_verified_success",
    )
    local_success = _boolean_vector(
        forced_fast_local_verified_success,
        count,
        name="forced_fast_local_verified_success",
    )
    fast_seed = _boolean_vector(
        forced_fast_seed_invoked,
        count,
        name="forced_fast_seed_invoked",
    )
    robust_seed = _boolean_vector(
        forced_robust_seed_invoked,
        count,
        name="forced_robust_seed_invoked",
    )
    primary = _boolean_vector(primary_mask, count, name="primary_mask")
    if not np.any(primary):
        raise ValueError("primary_mask must select at least one calibration query")

    robust_query_latency = np.median(robust_total, axis=1)
    fast_query_latency = np.median(fast_total, axis=1)

    candidates: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for threshold in selection.threshold_grid:
        # The frozen endpoint 1.0 is an explicit never-FAST sentinel, not a
        # numeric threshold that a saturated calibrated sigmoid may cross.
        take_fast = (threshold < 1.0) & (probability >= threshold)
        strategy_success = np.where(
            take_fast,
            fast_success,
            robust_success,
        )
        # Selection is fail-closed: aggregate equality is insufficient because
        # a success could otherwise move from one query to another.
        mismatch = strategy_success != robust_success
        success_equal = bool(not np.any(mismatch))
        strategy_latency = np.where(
            take_fast[:, None], fast_total, robust_total
        )
        query_latency = np.median(strategy_latency, axis=1)
        seed_invoked = np.where(take_fast, fast_seed, robust_seed)
        fast_attempt_count = int(np.sum(take_fast))
        fast_hit_count = int(np.sum(take_fast & local_success))
        primary_latency = query_latency[primary]
        row: dict[str, Any] = {
            "local_success_threshold": float(threshold),
            "eligible": success_equal,
            "success_vector_equal_to_always_hard": success_equal,
            "success_mismatch_count": int(np.sum(mismatch)),
            "verified_success_count": int(np.sum(strategy_success)),
            "always_hard_verified_success_count": int(np.sum(robust_success)),
            "primary_verified_success": float(np.mean(strategy_success[primary])),
            "primary_always_hard_verified_success": float(
                np.mean(robust_success[primary])
            ),
            "p50_ns": _percentile(primary_latency, 50),
            "p95_ns": _percentile(primary_latency, 95),
            "p99_ns": _percentile(primary_latency, 99),
            "primary_learned_seed_invocation_rate": float(
                np.mean(seed_invoked[primary])
            ),
            "primary_fast_attempt_rate": float(np.mean(take_fast[primary])),
            "primary_fast_hit_rate": float(
                np.mean((take_fast & local_success)[primary])
            ),
            "fast_attempt_count": fast_attempt_count,
            "fast_hit_count": fast_hit_count,
            "fast_path_precision": (
                float(fast_hit_count / fast_attempt_count)
                if fast_attempt_count
                else None
            ),
        }
        candidates.append(row)
        if success_equal:
            eligible_rows.append(row)

    if not eligible_rows:
        raise RuntimeError(
            "no calibration threshold reproduces always-hard success per query"
        )
    selected = min(
        eligible_rows,
        key=lambda row: (
            row["p95_ns"],
            row["p50_ns"],
            row["primary_learned_seed_invocation_rate"],
            -row["local_success_threshold"],
        ),
    )
    policy = LiteGatePolicyConfig(
        local_success_threshold=float(selected["local_success_threshold"]),
        calibration_role=selected_role,
        calibration_count=count,
    )
    report = {
        "selection_role": selected_role,
        "selection_data_usage": "calibration_counterfactual_cost_selection_only",
        "query_count": count,
        "primary_query_count": int(np.sum(primary)),
        "timing_repeats": TIMING_REPEATS,
        "timing_units": "ns",
        "exact_arm_cost_contract": {
            "forced_robust": (
                "complete runtime: feature + gate + fixed always_hard + "
                "result construction"
            ),
            "forced_fast": (
                "complete runtime: feature + gate + one_step_local + verifier + "
                "conditional fixed always_hard + result construction"
            ),
        },
        "success_constraint": "per-query verified-success equality to always-hard",
        "objective_order": [
            "success_vector_equal_to_always_hard",
            "minimum_primary_query_median_p95_ns",
            "minimum_primary_query_median_p50_ns",
            "minimum_primary_learned_seed_invocation_rate",
            "higher_threshold",
        ],
        "aggregation": selection.primary_aggregation,
        "selected": dict(selected),
        "eligible_candidate_count": len(eligible_rows),
        "candidate_count": len(candidates),
        "threshold_grid": list(selection.threshold_grid),
        "candidates": candidates,
        "lite_direct_robust_reference": {
            "primary_verified_success": float(np.mean(robust_success[primary])),
            "p50_ns": _percentile(robust_query_latency[primary], 50),
            "p95_ns": _percentile(robust_query_latency[primary], 95),
            "p99_ns": _percentile(robust_query_latency[primary], 99),
        },
        "lite_force_fast_reference": {
            "primary_verified_success": float(np.mean(fast_success[primary])),
            "p50_ns": _percentile(fast_query_latency[primary], 50),
            "p95_ns": _percentile(fast_query_latency[primary], 95),
            "p99_ns": _percentile(fast_query_latency[primary], 99),
        },
        "test_data_loaded": False,
        "policy_validation_used_for_selection": False,
    }
    return policy, report


class LiteFastGatePolicy:
    """Apply the frozen calibrated single threshold and emit FAST or ROBUST."""

    def __init__(self, backend: LiteGateBackend, config: LiteGatePolicyConfig):
        self.backend = backend
        self.config = config

    def decide_output(self, output: LiteGateOutput) -> LiteGateDecision:
        probability = float(output.local_success_probability)
        threshold = float(self.config.local_success_threshold)
        choose_fast = threshold < 1.0 and probability >= threshold
        return LiteGateDecision(
            action=FAST if choose_fast else ROBUST,
            choose_fast=bool(choose_fast),
            reason=(
                "calibrated_local_success_at_or_above_threshold"
                if choose_fast
                else "calibrated_local_success_below_threshold"
            ),
            local_success_probability=probability,
            local_success_threshold=threshold,
        )

    def predict(self, features: np.ndarray) -> LiteGateDecision:
        return self.decide_output(self.backend.predict_one(features))

    def decide(self, features: np.ndarray) -> LiteGateDecision:
        return self.predict(features)


def save_policy(
    path: str | Path,
    config: LiteGatePolicyConfig,
    selection_report: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    report = _safe_mapping(selection_report)
    if report.get("selection_role") != CALIBRATION_ROLE:
        raise ValueError("policy selection report is not calibration-only")
    if bool(report.get("policy_validation_used_for_selection", True)):
        raise ValueError("policy-validation data cannot select a V5-Lite threshold")
    if bool(report.get("test_data_loaded", True)):
        raise ValueError("formal test data cannot select a V5-Lite threshold")
    payload = {
        "format_version": 1,
        "policy": asdict(config),
        "selection_report": report,
        "provenance": _safe_mapping(provenance),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_policy(
    path: str | Path,
) -> tuple[LiteGatePolicyConfig, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported V5-Lite policy artifact")
    config = LiteGatePolicyConfig(**dict(payload["policy"]))
    report = dict(payload["selection_report"])
    if report.get("selection_role") != CALIBRATION_ROLE:
        raise ValueError("loaded policy was not selected on calibration data")
    if bool(report.get("policy_validation_used_for_selection", True)) or bool(
        report.get("test_data_loaded", True)
    ):
        raise ValueError("loaded policy provenance violates development boundaries")
    return config, payload


# Explicit and compatibility names used by orchestration/tests.
FastGatePolicyConfig = LiteGatePolicyConfig
FastGateDecision = LiteGateDecision
FastGatePolicy = LiteFastGatePolicy
select_cost_sensitive_threshold = select_thresholds
select_cost_sensitive_thresholds = select_thresholds


__all__ = [
    "CALIBRATION_ROLE",
    "FAST",
    "FINAL_ACTIONS",
    "FastGateDecision",
    "FastGatePolicy",
    "FastGatePolicyConfig",
    "LiteFastGatePolicy",
    "LiteGateBackend",
    "LiteGateDecision",
    "LiteGatePolicyConfig",
    "ROBUST",
    "TIMING_REPEATS",
    "ThresholdSelectionConfig",
    "load_policy",
    "save_policy",
    "select_cost_sensitive_threshold",
    "select_cost_sensitive_thresholds",
    "select_thresholds",
]
