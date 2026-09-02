"""Single-head deployment model for the development-only V5-Lite fast gate.

The model has one deliberately narrow responsibility: estimate the probability
that one previous-state DLS iteration will pass the frozen deterministic
verifier.  Model weights and feature-normalization statistics are fitted only
on ``risk_train_queries``.  A one-dimensional Platt scaler is fitted only on
``calibration_queries``.  Threshold selection is intentionally left to
``policy.py`` so latency costs can never leak into model fitting.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..counterfactual_v4.calibration import (
    PlattCalibrator,
    binary_calibration_metrics,
    sigmoid,
)


TRAIN_ROLE = "risk_train_queries"
CALIBRATION_ROLE = "calibration_queries"
HEAD_NAME = "local_verified_success"
NON_JOINT_FEATURE_NAMES = (
    "pose_error_x",
    "pose_error_y",
    "pose_error_z",
    "pose_error_rx",
    "pose_error_ry",
    "pose_error_rz",
    "position_error_norm",
    "orientation_error_norm",
    "previous_joint_limit_margin_min",
)
LABEL_CONTRACT = {
    HEAD_NAME: (
        "one previous-state DLS iteration passes the frozen deterministic "
        "verifier without requiring the always-hard fallback"
    ),
    "command_acceptance": "deterministic_verifier_only",
    "training_role": TRAIN_ROLE,
    "calibration_role": CALIBRATION_ROLE,
    "formal_test_data_forbidden": True,
}


def feature_names_for_input_dim(input_dim: int) -> tuple[str, ...]:
    """Return the fixed V5-Lite schema for a robot-specific input width."""

    width = int(input_dim)
    nq = width - len(NON_JOINT_FEATURE_NAMES)
    if nq <= 0:
        raise ValueError("input_dim must equal nq + 9 for a positive nq")
    return tuple(f"normalized_previous_joint_{index}" for index in range(nq)) + (
        NON_JOINT_FEATURE_NAMES
    )


def _require_role(role: str, expected: str, *, operation: str) -> str:
    normalized = str(role).strip()
    lowered = normalized.lower()
    if "test" in lowered or "policy_validation" in lowered:
        raise ValueError(f"{operation} forbids test and policy-validation roles")
    if normalized != expected:
        raise ValueError(f"{operation} requires role {expected!r}; got {normalized!r}")
    return normalized


def _safe_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    state = {} if values is None else dict(values)
    try:
        return json.loads(json.dumps(state, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must be JSON serializable") from error


def _feature_matrix(features: np.ndarray, input_dim: int) -> tuple[np.ndarray, bool]:
    rows = np.asarray(features, dtype=np.float32)
    single = rows.ndim == 1
    if single:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != input_dim:
        raise ValueError(f"features must have shape (N, {input_dim})")
    if not np.all(np.isfinite(rows)):
        raise ValueError("features must be finite")
    return np.ascontiguousarray(rows), single


def _binary_targets(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.shape != (count,) or not np.all(np.isfinite(target)):
        raise ValueError(f"{name} must be a finite vector with shape ({count},)")
    rounded = np.rint(target)
    if not np.allclose(target, rounded) or np.any((rounded < 0) | (rounded > 1)):
        raise ValueError(f"{name} must contain binary 0/1 labels")
    return rounded.astype(np.float32)


@dataclass(frozen=True)
class LiteGateTrainingConfig:
    """Training parameters; the two 16-wide hidden layers are immutable."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    seed: int = 1705

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


class LiteGateMLP(nn.Module):
    """Exactly ``input_dim -> 16 -> 16 -> 1`` with SiLU activations."""

    def __init__(self, input_dim: int):
        super().__init__()
        feature_names_for_input_dim(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(int(input_dim), 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


@dataclass(frozen=True)
class LiteGatePrediction:
    local_success_logit: np.ndarray
    local_success_probability: np.ndarray


@dataclass(frozen=True)
class LiteGateOutput:
    local_success_logit: float
    local_success_probability: float


class LiteGatePredictor:
    """Train-only fitting, calibration-only Platt scaling, and inference."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        config: LiteGateTrainingConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        default_names = feature_names_for_input_dim(self.input_dim)
        names = default_names if feature_names is None else tuple(map(str, feature_names))
        if len(names) != self.input_dim or len(set(names)) != len(names):
            raise ValueError("feature_names must be unique and match input_dim")
        self.feature_names = names
        self.config = config or LiteGateTrainingConfig()
        self.device = torch.device(device)
        self.model = self._new_model().to(self.device)
        self.feature_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.feature_scale = np.ones(self.input_dim, dtype=np.float32)
        self.calibrator: PlattCalibrator | None = None
        self.fitted = False
        self.fit_count = 0
        self.calibration_count = 0
        self.training_history: list[dict[str, float]] = []
        self.user_provenance: dict[str, Any] = {}

    def _new_model(self) -> LiteGateMLP:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            return LiteGateMLP(self.input_dim)

    def _normalized_tensor(self, features: np.ndarray) -> Tensor:
        rows, _ = _feature_matrix(features, self.input_dim)
        normalized = np.ascontiguousarray(
            (rows - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )
        return torch.from_numpy(normalized).to(self.device)

    def fit(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        *,
        role: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LiteGatePredictor":
        _require_role(role, TRAIN_ROLE, operation="LiteGatePredictor.fit")
        rows, _ = _feature_matrix(features, self.input_dim)
        target = _binary_targets(
            local_verified_success,
            len(rows),
            name=HEAD_NAME,
        )

        # These are the only normalization statistics stored in the model.
        self.feature_mean = np.mean(rows, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(rows, axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
        normalized = np.ascontiguousarray(
            (rows - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )

        self.model = self._new_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        feature_tensor = torch.from_numpy(normalized)
        target_tensor = torch.from_numpy(target[:, None])
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.training_history = []
        self.model.train()
        for epoch in range(self.config.epochs):
            order = torch.randperm(len(rows), generator=generator)
            total_loss = 0.0
            batches = 0
            for start in range(0, len(rows), self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                inputs = feature_tensor[index].to(self.device)
                expected = target_tensor[index].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits, expected)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            self.training_history.append(
                {"epoch": float(epoch + 1), "local_success_bce": total_loss / batches}
            )

        self.model.eval()
        self.fitted = True
        self.fit_count = len(rows)
        self.calibrator = None
        self.calibration_count = 0
        self.user_provenance = _safe_mapping(provenance)
        return self

    def _raw_logits(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LiteGatePredictor has not been fitted")
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(self._normalized_tensor(features))
        return logits.detach().cpu().numpy().astype(np.float64)[:, 0]

    def calibrate(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        *,
        role: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LiteGatePredictor":
        _require_role(role, CALIBRATION_ROLE, operation="LiteGatePredictor.calibrate")
        rows, _ = _feature_matrix(features, self.input_dim)
        target = _binary_targets(
            local_verified_success,
            len(rows),
            name=HEAD_NAME,
        )
        self.calibrator = PlattCalibrator().fit(self._raw_logits(rows), target)
        self.calibration_count = len(rows)
        calibration_provenance = _safe_mapping(provenance)
        if calibration_provenance:
            self.user_provenance["calibration"] = calibration_provenance
        return self

    def predict(self, features: np.ndarray) -> LiteGatePrediction:
        rows, _ = _feature_matrix(features, self.input_dim)
        logits = self._raw_logits(rows)
        probability = (
            sigmoid(logits)
            if self.calibrator is None
            else self.calibrator.predict_proba(logits)
        )
        return LiteGatePrediction(logits.copy(), probability.copy())

    def predict_one(self, features: np.ndarray) -> LiteGateOutput:
        rows, _ = _feature_matrix(features, self.input_dim)
        if len(rows) != 1:
            raise ValueError("predict_one requires exactly one feature row")
        prediction = self.predict(rows)
        return LiteGateOutput(
            local_success_logit=float(prediction.local_success_logit[0]),
            local_success_probability=float(prediction.local_success_probability[0]),
        )

    def calibration_metrics(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        *,
        bins: int = 15,
    ) -> dict[str, float]:
        rows, _ = _feature_matrix(features, self.input_dim)
        target = _binary_targets(local_verified_success, len(rows), name=HEAD_NAME)
        return binary_calibration_metrics(
            self.predict(rows).local_success_probability,
            target,
            bins=bins,
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "input_dim": self.input_dim,
            "feature_names": list(self.feature_names),
            "head_name": HEAD_NAME,
            "architecture": [self.input_dim, 16, 16, 1],
            "fit_role": TRAIN_ROLE,
            "fit_count": self.fit_count,
            "calibration_role": (
                CALIBRATION_ROLE if self.calibrator is not None else None
            ),
            "calibration_count": self.calibration_count,
            "label_contract": dict(LABEL_CONTRACT),
            "user": _safe_mapping(self.user_provenance),
        }

    def save(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted LiteGatePredictor")
        if provenance is not None:
            self.user_provenance["artifact"] = _safe_mapping(provenance)
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "input_dim": self.input_dim,
            "feature_names": list(self.feature_names),
            "head_name": HEAD_NAME,
            "label_contract": dict(LABEL_CONTRACT),
            "config": asdict(self.config),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "feature_mean": torch.from_numpy(self.feature_mean.copy()),
            "feature_scale": torch.from_numpy(self.feature_scale.copy()),
            "fitted": self.fitted,
            "training_history": self.training_history,
            "calibrator": (
                None if self.calibrator is None else self.calibrator.to_state()
            ),
            "provenance": self.provenance,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "LiteGatePredictor":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported V5-Lite gate artifact")
        if payload.get("head_name") != HEAD_NAME:
            raise ValueError("V5-Lite output schema mismatch")
        if payload.get("label_contract") != LABEL_CONTRACT:
            raise ValueError("V5-Lite label contract mismatch")
        instance = cls(
            int(payload["input_dim"]),
            LiteGateTrainingConfig(**dict(payload["config"])),
            device=device,
            feature_names=tuple(payload["feature_names"]),
        )
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device).eval()
        instance.feature_mean = payload["feature_mean"].cpu().numpy().astype(np.float32)
        instance.feature_scale = payload["feature_scale"].cpu().numpy().astype(np.float32)
        instance.fitted = bool(payload["fitted"])
        instance.training_history = [dict(row) for row in payload["training_history"]]
        state = dict(payload["provenance"])
        if state.get("fit_role") != TRAIN_ROLE:
            raise ValueError("artifact was not fitted on the training role")
        if state.get("calibration_role") not in {None, CALIBRATION_ROLE}:
            raise ValueError("artifact was calibrated on an invalid role")
        if tuple(state.get("architecture", ())) != (instance.input_dim, 16, 16, 1):
            raise ValueError("V5-Lite architecture provenance mismatch")
        instance.fit_count = int(state["fit_count"])
        instance.calibration_count = int(state["calibration_count"])
        instance.user_provenance = _safe_mapping(state.get("user", {}))
        if payload["calibrator"] is not None:
            instance.calibrator = PlattCalibrator.from_state(payload["calibrator"])
            if not instance.calibrator.fitted:
                raise ValueError("stored Platt calibrator is not fitted")
        return instance


class ExactLiteGateModule(nn.Module):
    """TorchScript graph containing normalization, MLP, and Platt scaling."""

    def __init__(self, predictor: LiteGatePredictor):
        super().__init__()
        if not predictor.fitted or predictor.calibrator is None:
            raise RuntimeError("exact export requires fitted and calibrated predictor")
        self.model = deepcopy(predictor.model).cpu().eval()
        self.register_buffer("feature_mean", torch.from_numpy(predictor.feature_mean.copy()))
        self.register_buffer("feature_scale", torch.from_numpy(predictor.feature_scale.copy()))
        self.register_buffer(
            "platt_slope",
            torch.tensor(float(predictor.calibrator.slope), dtype=torch.float64),
        )
        self.register_buffer(
            "platt_intercept",
            torch.tensor(float(predictor.calibrator.intercept), dtype=torch.float64),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (features - self.feature_mean) / self.feature_scale
        logits = self.model(normalized).to(torch.float64)
        probability = torch.sigmoid(
            logits * self.platt_slope + self.platt_intercept
        ).clamp(1.0e-7, 1.0 - 1.0e-7)
        return logits, probability


def export_exact_torchscript(
    predictor: LiteGatePredictor,
    path: str | Path,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with torch.inference_mode():
        scripted = torch.jit.script(ExactLiteGateModule(predictor).eval())
        scripted = torch.jit.freeze(scripted.eval())
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(destination))
    return {
        **predictor.provenance,
        "torchscript_load_only": True,
        "input_dtype": "float32",
        "input_shape": [1, predictor.input_dim],
        "output_dtype": "float64",
    }


def load_exact_torchscript(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.jit.ScriptModule:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    return torch.jit.load(str(source), map_location=torch.device(device)).eval()


class TorchScriptLiteGateInference:
    """Allocation-light batch-one adapter for the online runtime."""

    def __init__(self, module: torch.jit.ScriptModule, input_dim: int):
        self.module = module.eval()
        self.input_dim = int(input_dim)
        feature_names_for_input_dim(self.input_dim)
        self._input = np.empty((1, self.input_dim), dtype=np.float32)

    def infer(self, features: np.ndarray) -> LiteGateOutput:
        row = np.asarray(features, dtype=np.float32)
        if row.shape != (self.input_dim,) or not np.all(np.isfinite(row)):
            raise ValueError(
                f"features must be finite with shape ({self.input_dim},)"
            )
        np.copyto(self._input[0], row)
        with torch.inference_mode():
            logits, probability = self.module(torch.from_numpy(self._input))
        return LiteGateOutput(
            local_success_logit=float(logits[0, 0].item()),
            local_success_probability=float(probability[0, 0].item()),
        )

    def predict_one(self, features: np.ndarray) -> LiteGateOutput:
        return self.infer(features)


def numerical_equivalence(
    predictor: LiteGatePredictor,
    module: torch.jit.ScriptModule,
    features: np.ndarray,
    *,
    threshold: float | None = None,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    rows, _ = _feature_matrix(features, predictor.input_dim)
    exact = TorchScriptLiteGateInference(module, predictor.input_dim)
    max_logit_error = 0.0
    max_probability_error = 0.0
    route_matches = 0
    for row in rows:
        eager = predictor.predict_one(row)
        deployed = exact.infer(row)
        max_logit_error = max(
            max_logit_error,
            abs(eager.local_success_logit - deployed.local_success_logit),
        )
        max_probability_error = max(
            max_probability_error,
            abs(eager.local_success_probability - deployed.local_success_probability),
        )
        if threshold is None:
            route_matches += 1
        else:
            route_matches += int(
                (eager.local_success_probability >= threshold)
                == (deployed.local_success_probability >= threshold)
            )
    passed = bool(
        max_logit_error <= atol
        and max_probability_error <= atol
        and route_matches == len(rows)
    )
    return {
        "query_count": len(rows),
        "max_absolute_logit_error": max_logit_error,
        "max_absolute_probability_error": max_probability_error,
        "route_match_count": route_matches,
        "route_match_rate": route_matches / len(rows),
        "absolute_tolerance": float(atol),
        "passed": passed,
    }


# Concise compatibility aliases for orchestration code.
FastGateTrainingConfig = LiteGateTrainingConfig
FastGatePredictor = LiteGatePredictor
FastGateOutput = LiteGateOutput
TorchScriptFastGateInference = TorchScriptLiteGateInference


__all__ = [
    "CALIBRATION_ROLE",
    "FastGateOutput",
    "FastGatePredictor",
    "FastGateTrainingConfig",
    "HEAD_NAME",
    "LABEL_CONTRACT",
    "LiteGateMLP",
    "LiteGateOutput",
    "LiteGatePrediction",
    "LiteGatePredictor",
    "LiteGateTrainingConfig",
    "NON_JOINT_FEATURE_NAMES",
    "TRAIN_ROLE",
    "TorchScriptFastGateInference",
    "TorchScriptLiteGateInference",
    "export_exact_torchscript",
    "feature_names_for_input_dim",
    "load_exact_torchscript",
    "numerical_equivalence",
]
