"""Frozen two-threshold policy for the hierarchical v5 fast gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .model import CALIBRATION_ROLE, FastGateOutput


FAST = "fast"
ROBUST = "robust"
FINAL_ACTIONS = (FAST, ROBUST)


class FastGateBackend(Protocol):
    def predict_one(self, features: np.ndarray) -> FastGateOutput: ...


def _require_calibration_role(role: str) -> str:
    normalized = str(role).strip()
    if "test" in normalized.lower():
        raise ValueError("fast-gate threshold selection forbids every test role")
    if normalized != CALIBRATION_ROLE:
        raise ValueError(
            "fast-gate threshold selection requires the calibration role; "
            f"received {normalized!r}"
        )
    return normalized


def _probabilities(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _binary(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({count},)")
    rounded = np.rint(array)
    if not np.allclose(array, rounded) or np.any((rounded < 0.0) | (rounded > 1.0)):
        raise ValueError(f"{name} must contain binary 0/1 labels")
    return rounded.astype(bool)


def _threshold_grid(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    grid = tuple(sorted(set(float(value) for value in values)))
    if not grid or any(not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return grid


def _safe_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    state = {} if values is None else dict(values)
    try:
        return json.loads(json.dumps(state, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("policy provenance must be JSON serializable") from error


@dataclass(frozen=True)
class FastGatePolicyConfig:
    """Frozen thresholds; both calibrated probabilities must pass."""

    local_success_threshold: float
    latency_benefit_threshold: float
    calibration_role: str = CALIBRATION_ROLE
    minimum_fast_precision: float = 0.99
    minimum_positive_benefit_rate: float = 0.95
    calibration_count: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("local_success_threshold", self.local_success_threshold),
            ("latency_benefit_threshold", self.latency_benefit_threshold),
            ("minimum_fast_precision", self.minimum_fast_precision),
            ("minimum_positive_benefit_rate", self.minimum_positive_benefit_rate),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        _require_calibration_role(self.calibration_role)
        if self.calibration_count < 0:
            raise ValueError("calibration_count must be nonnegative")


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    """Calibration-only grid and selective-precision requirements."""

    local_success_grid: tuple[float, ...] = (
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
    )
    latency_benefit_grid: tuple[float, ...] = (
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
    )
    minimum_fast_precision: float = 0.99
    minimum_positive_benefit_rate: float = 0.95
    minimum_fast_count: int = 25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_success_grid",
            _threshold_grid(self.local_success_grid, name="local_success_grid"),
        )
        object.__setattr__(
            self,
            "latency_benefit_grid",
            _threshold_grid(self.latency_benefit_grid, name="latency_benefit_grid"),
        )
        if not 0.0 <= self.minimum_fast_precision <= 1.0:
            raise ValueError("minimum_fast_precision must lie in [0, 1]")
        if not 0.0 <= self.minimum_positive_benefit_rate <= 1.0:
            raise ValueError("minimum_positive_benefit_rate must lie in [0, 1]")
        if self.minimum_fast_count <= 0:
            raise ValueError("minimum_fast_count must be positive")


@dataclass(frozen=True)
class FastGateDecision:
    action: str
    choose_fast: bool
    reason: str
    local_success_probability: float
    latency_benefit_probability: float
    local_success_threshold: float
    latency_benefit_threshold: float

    def __post_init__(self) -> None:
        if self.action not in FINAL_ACTIONS:
            raise ValueError(f"unsupported fast-gate action: {self.action}")
        if self.choose_fast != (self.action == FAST):
            raise ValueError("choose_fast must agree with action")


def select_thresholds(
    local_success_probability: np.ndarray,
    latency_benefit_probability: np.ndarray,
    local_verified_success: np.ndarray,
    fast_latency_benefit: np.ndarray,
    *,
    role: str,
    config: ThresholdSelectionConfig | None = None,
) -> tuple[FastGatePolicyConfig, dict[str, Any]]:
    """Select thresholds on calibration only, maximizing safe FAST coverage.

    A candidate is eligible only when FAST-routed queries satisfy both the
    local verified-success precision floor and the positive-latency-benefit
    floor.  Policy-validation labels are therefore never inputs to selection.
    """

    selected_role = _require_calibration_role(role)
    selection = config or ThresholdSelectionConfig()
    local_probability = _probabilities(
        local_success_probability, name="local_success_probability"
    )
    benefit_probability = _probabilities(
        latency_benefit_probability, name="latency_benefit_probability"
    )
    count = len(local_probability)
    if benefit_probability.shape != (count,):
        raise ValueError("the two probability vectors must have the same shape")
    local_target = _binary(
        local_verified_success, count, name="local_verified_success"
    )
    benefit_target = _binary(
        fast_latency_benefit, count, name="fast_latency_benefit"
    )

    candidates: list[dict[str, Any]] = []
    for local_threshold in selection.local_success_grid:
        for benefit_threshold in selection.latency_benefit_grid:
            fast = (local_probability >= local_threshold) & (
                benefit_probability >= benefit_threshold
            )
            fast_count = int(np.sum(fast))
            if fast_count == 0:
                local_precision = 0.0
                benefit_rate = 0.0
                joint_precision = 0.0
            else:
                local_precision = float(np.mean(local_target[fast]))
                benefit_rate = float(np.mean(benefit_target[fast]))
                joint_precision = float(
                    np.mean(local_target[fast] & benefit_target[fast])
                )
            eligible = bool(
                fast_count >= selection.minimum_fast_count
                and local_precision >= selection.minimum_fast_precision
                and benefit_rate >= selection.minimum_positive_benefit_rate
            )
            candidates.append(
                {
                    "local_success_threshold": local_threshold,
                    "latency_benefit_threshold": benefit_threshold,
                    "fast_count": fast_count,
                    "fast_rate": fast_count / count,
                    "fast_precision": local_precision,
                    "positive_benefit_rate": benefit_rate,
                    "joint_positive_rate": joint_precision,
                    "eligible": eligible,
                }
            )

    eligible_candidates = [row for row in candidates if row["eligible"]]
    if not eligible_candidates:
        raise RuntimeError(
            "no calibration-grid threshold pair meets the frozen FAST precision, "
            "positive-benefit, and minimum-count constraints"
        )
    # Primary objective: route the largest safe fraction to FAST.  Ties prefer
    # better joint precision, then stricter thresholds for deterministic
    # conservative behavior.
    selected = max(
        eligible_candidates,
        key=lambda row: (
            row["fast_count"],
            row["joint_positive_rate"],
            row["fast_precision"],
            row["positive_benefit_rate"],
            row["local_success_threshold"],
            row["latency_benefit_threshold"],
        ),
    )
    policy = FastGatePolicyConfig(
        local_success_threshold=float(selected["local_success_threshold"]),
        latency_benefit_threshold=float(selected["latency_benefit_threshold"]),
        calibration_role=selected_role,
        minimum_fast_precision=selection.minimum_fast_precision,
        minimum_positive_benefit_rate=selection.minimum_positive_benefit_rate,
        calibration_count=count,
    )
    report = {
        "selection_role": selected_role,
        "selection_data_usage": "threshold_selection_only",
        "query_count": count,
        "objective": "maximum FAST coverage subject to two precision constraints",
        "selected": dict(selected),
        "constraints": {
            "minimum_fast_precision": selection.minimum_fast_precision,
            "minimum_positive_benefit_rate": selection.minimum_positive_benefit_rate,
            "minimum_fast_count": selection.minimum_fast_count,
        },
        "grid": {
            "local_success": list(selection.local_success_grid),
            "latency_benefit": list(selection.latency_benefit_grid),
        },
        "eligible_candidate_count": len(eligible_candidates),
        "candidate_count": len(candidates),
    }
    return policy, report


class HierarchicalFastGatePolicy:
    """Run a calibrated batch-one backend and emit FAST or ROBUST."""

    def __init__(self, backend: FastGateBackend, config: FastGatePolicyConfig):
        self.backend = backend
        self.config = config

    def decide_output(self, output: FastGateOutput) -> FastGateDecision:
        local_pass = (
            output.local_success_probability
            >= self.config.local_success_threshold
        )
        benefit_pass = (
            output.latency_benefit_probability
            >= self.config.latency_benefit_threshold
        )
        choose_fast = bool(local_pass and benefit_pass)
        if choose_fast:
            reason = "both_calibrated_fast_gates_pass"
        elif not local_pass and not benefit_pass:
            reason = "local_success_and_latency_benefit_below_threshold"
        elif not local_pass:
            reason = "local_success_below_threshold"
        else:
            reason = "latency_benefit_below_threshold"
        return FastGateDecision(
            action=FAST if choose_fast else ROBUST,
            choose_fast=choose_fast,
            reason=reason,
            local_success_probability=float(output.local_success_probability),
            latency_benefit_probability=float(
                output.latency_benefit_probability
            ),
            local_success_threshold=self.config.local_success_threshold,
            latency_benefit_threshold=self.config.latency_benefit_threshold,
        )

    def predict(self, features: np.ndarray) -> FastGateDecision:
        """Runtime-compatible prediction API with probabilities and route."""

        return self.decide_output(self.backend.predict_one(features))

    def decide(self, features: np.ndarray) -> FastGateDecision:
        return self.predict(features)


def save_policy(
    path: str | Path,
    config: FastGatePolicyConfig,
    selection_report: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    report = _safe_mapping(selection_report)
    if report.get("selection_role") != CALIBRATION_ROLE:
        raise ValueError("policy selection report is not calibration-only")
    payload = {
        "format_version": 1,
        "policy": asdict(config),
        "selection_report": report,
        "provenance": _safe_mapping(provenance),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_policy(
    path: str | Path,
) -> tuple[FastGatePolicyConfig, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported hierarchical v5 policy artifact")
    config = FastGatePolicyConfig(**dict(payload["policy"]))
    report = dict(payload["selection_report"])
    if report.get("selection_role") != CALIBRATION_ROLE:
        raise ValueError("loaded policy was not selected on calibration data")
    return config, payload


# Short name used by the runtime and tests.
FastGatePolicy = HierarchicalFastGatePolicy

