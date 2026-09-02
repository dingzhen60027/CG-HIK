"""Compact, seed-free fast-gate model for hierarchical CG-HIK v5.

The module deliberately knows only two development roles: model weights are
fit on ``risk_train_queries`` and Platt scalers are fit on
``calibration_queries``.  It never discovers datasets or paths by itself.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..counterfactual_v4.calibration import (
    MultiOutputCalibrator,
    binary_calibration_metrics,
    sigmoid,
)


FEATURE_DIM = 7
FEATURE_NAMES = (
    "target_position_step",
    "target_orientation_step",
    "previous_joint_limit_margin_min",
    "previous_jacobian_sigma_min",
    "previous_jacobian_condition_number",
    "one_step_dls_max_joint_update",
    "estimated_velocity_limit_utilization_max",
)
HEAD_NAMES = ("local_verified_success", "fast_latency_benefit")
TRAIN_ROLE = "risk_train_queries"
CALIBRATION_ROLE = "calibration_queries"
LABEL_CONTRACT = {
    "local_verified_success": (
        "previous-state local DLS completes within the frozen fast budget and "
        "passes the deterministic verifier without escalation"
    ),
    "fast_latency_benefit": (
        "measured local-fast-path total latency is lower than measured direct "
        "robust-path total latency for the same development query"
    ),
    "command_acceptance": "deterministic_verifier_only",
    "learned_seed_features_forbidden": True,
}


def _require_role(role: str, expected: str, *, operation: str) -> str:
    normalized = str(role).strip()
    if "test" in normalized.lower():
        raise ValueError(f"{operation} forbids every test role")
    if normalized != expected:
        raise ValueError(
            f"{operation} requires role {expected!r}; received {normalized!r}"
        )
    return normalized


def _feature_matrix(features: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(features, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (N, {FEATURE_DIM})")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return np.ascontiguousarray(array), single


def _binary_targets(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.shape != (count,) or not np.all(np.isfinite(target)):
        raise ValueError(f"{name} must be a finite vector with shape ({count},)")
    rounded = np.rint(target)
    if not np.allclose(target, rounded) or np.any((rounded < 0.0) | (rounded > 1.0)):
        raise ValueError(f"{name} must contain binary 0/1 labels")
    return rounded.astype(np.float32)


def _safe_provenance(values: Mapping[str, Any] | None) -> dict[str, Any]:
    state = {} if values is None else dict(values)
    try:
        # Round-tripping removes non-standard mapping/list implementations and
        # guarantees that the state remains inspectable outside PyTorch.
        return json.loads(json.dumps(state, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must be JSON serializable") from error


@dataclass(frozen=True)
class FastGateTrainingConfig:
    """Training hyperparameters; the 7-16-16-2 architecture is not tunable."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    local_success_loss_weight: float = 1.0
    latency_benefit_loss_weight: float = 1.0
    gradient_clip_norm: float = 5.0
    seed: int = 17

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.local_success_loss_weight <= 0.0:
            raise ValueError("local_success_loss_weight must be positive")
        if self.latency_benefit_loss_weight <= 0.0:
            raise ValueError("latency_benefit_loss_weight must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


class FastGateMLP(nn.Module):
    """Exactly two 16-wide hidden layers and two independent binary logits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIM, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, len(HEAD_NAMES)),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


@dataclass(frozen=True)
class FastGatePrediction:
    local_success_logit: np.ndarray
    latency_benefit_logit: np.ndarray
    local_success_probability: np.ndarray
    latency_benefit_probability: np.ndarray


@dataclass(frozen=True)
class FastGateOutput:
    local_success_logit: float
    latency_benefit_logit: float
    local_success_probability: float
    latency_benefit_probability: float


class FastGatePredictor:
    """Fit, calibrate, serialize, and run the hierarchical fast gate."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: FastGateTrainingConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or FastGateTrainingConfig()
        self.device = torch.device(device)
        self.model = self._new_model().to(self.device)
        self.feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feature_scale = np.ones(FEATURE_DIM, dtype=np.float32)
        self.calibrator: MultiOutputCalibrator | None = None
        self.fitted = False
        self.training_history: list[dict[str, float]] = []
        self.fit_count = 0
        self.calibration_count = 0
        self.user_provenance: dict[str, Any] = {}

    def _new_model(self) -> FastGateMLP:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            return FastGateMLP()

    def _normalized_tensor(self, features: np.ndarray) -> Tensor:
        rows, _ = _feature_matrix(features)
        normalized = np.ascontiguousarray(
            (rows - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )
        return torch.from_numpy(normalized).to(self.device)

    def fit(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        fast_latency_benefit: np.ndarray,
        *,
        role: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "FastGatePredictor":
        """Fit model weights using the training role and no other role."""

        _require_role(role, TRAIN_ROLE, operation="FastGatePredictor.fit")
        rows, _ = _feature_matrix(features)
        count = len(rows)
        local = _binary_targets(
            local_verified_success, count, name="local_verified_success"
        )
        benefit = _binary_targets(
            fast_latency_benefit, count, name="fast_latency_benefit"
        )
        targets = np.column_stack([local, benefit]).astype(np.float32)

        self.feature_mean = np.mean(rows, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(rows, axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
        normalized = np.ascontiguousarray(
            (rows - self.feature_mean) / self.feature_scale, dtype=np.float32
        )

        self.model = self._new_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        feature_tensor = torch.from_numpy(normalized)
        target_tensor = torch.from_numpy(targets)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.training_history = []
        self.model.train()
        for epoch in range(self.config.epochs):
            order = torch.randperm(count, generator=generator)
            totals = np.zeros(3, dtype=np.float64)
            batches = 0
            for start in range(0, count, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                inputs = feature_tensor[index].to(self.device)
                target = target_tensor[index].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                local_loss = F.binary_cross_entropy_with_logits(
                    logits[:, 0], target[:, 0]
                )
                benefit_loss = F.binary_cross_entropy_with_logits(
                    logits[:, 1], target[:, 1]
                )
                loss = (
                    self.config.local_success_loss_weight * local_loss
                    + self.config.latency_benefit_loss_weight * benefit_loss
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                totals += np.asarray(
                    [
                        float(loss.detach().cpu()),
                        float(local_loss.detach().cpu()),
                        float(benefit_loss.detach().cpu()),
                    ]
                )
                batches += 1
            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(totals[0] / batches),
                    "local_success_bce": float(totals[1] / batches),
                    "latency_benefit_bce": float(totals[2] / batches),
                }
            )

        self.model.eval()
        self.fitted = True
        self.fit_count = count
        self.calibration_count = 0
        self.calibrator = None
        self.user_provenance = _safe_provenance(provenance)
        return self

    def _raw_logits(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fast-gate predictor has not been fitted")
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(self._normalized_tensor(features))
        return logits.detach().cpu().numpy().astype(np.float64)

    def calibrate(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        fast_latency_benefit: np.ndarray,
        *,
        role: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "FastGatePredictor":
        """Fit independent Platt scalers using calibration data only."""

        _require_role(role, CALIBRATION_ROLE, operation="FastGatePredictor.calibrate")
        rows, _ = _feature_matrix(features)
        local = _binary_targets(
            local_verified_success, len(rows), name="local_verified_success"
        )
        benefit = _binary_targets(
            fast_latency_benefit, len(rows), name="fast_latency_benefit"
        )
        targets = np.column_stack([local, benefit])
        self.calibrator = MultiOutputCalibrator("platt", HEAD_NAMES).fit(
            self._raw_logits(rows), targets
        )
        self.calibration_count = len(rows)
        calibration_provenance = _safe_provenance(provenance)
        if calibration_provenance:
            self.user_provenance["calibration"] = calibration_provenance
        return self

    def predict(self, features: np.ndarray) -> FastGatePrediction:
        rows, _ = _feature_matrix(features)
        logits = self._raw_logits(rows)
        probabilities = (
            sigmoid(logits)
            if self.calibrator is None
            else self.calibrator.predict_proba(logits)
        )
        return FastGatePrediction(
            local_success_logit=logits[:, 0].copy(),
            latency_benefit_logit=logits[:, 1].copy(),
            local_success_probability=probabilities[:, 0].copy(),
            latency_benefit_probability=probabilities[:, 1].copy(),
        )

    def predict_one(self, features: np.ndarray) -> FastGateOutput:
        rows, _ = _feature_matrix(features)
        if len(rows) != 1:
            raise ValueError("predict_one requires exactly one feature row")
        prediction = self.predict(rows)
        return FastGateOutput(
            local_success_logit=float(prediction.local_success_logit[0]),
            latency_benefit_logit=float(prediction.latency_benefit_logit[0]),
            local_success_probability=float(
                prediction.local_success_probability[0]
            ),
            latency_benefit_probability=float(
                prediction.latency_benefit_probability[0]
            ),
        )

    def calibration_metrics(
        self,
        features: np.ndarray,
        local_verified_success: np.ndarray,
        fast_latency_benefit: np.ndarray,
        *,
        bins: int = 15,
    ) -> dict[str, dict[str, float]]:
        rows, _ = _feature_matrix(features)
        targets = np.column_stack(
            [
                _binary_targets(
                    local_verified_success, len(rows), name="local_verified_success"
                ),
                _binary_targets(
                    fast_latency_benefit, len(rows), name="fast_latency_benefit"
                ),
            ]
        )
        prediction = self.predict(rows)
        probabilities = np.column_stack(
            [
                prediction.local_success_probability,
                prediction.latency_benefit_probability,
            ]
        )
        return {
            name: binary_calibration_metrics(
                probabilities[:, index], targets[:, index], bins=bins
            )
            for index, name in enumerate(HEAD_NAMES)
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "head_names": list(HEAD_NAMES),
            "architecture": [FEATURE_DIM, 16, 16, len(HEAD_NAMES)],
            "fit_role": TRAIN_ROLE,
            "fit_count": self.fit_count,
            "calibration_role": CALIBRATION_ROLE if self.calibrator is not None else None,
            "calibration_count": self.calibration_count,
            "label_contract": dict(LABEL_CONTRACT),
            "user": _safe_provenance(self.user_provenance),
        }

    def save(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted fast-gate predictor")
        if provenance is not None:
            self.user_provenance["artifact"] = _safe_provenance(provenance)
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "head_names": list(HEAD_NAMES),
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
            "calibrator": None
            if self.calibrator is None
            else self.calibrator.to_state(),
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
    ) -> "FastGatePredictor":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported hierarchical v5 fast-gate artifact")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("fast-gate feature schema mismatch")
        if tuple(payload.get("head_names", ())) != HEAD_NAMES:
            raise ValueError("fast-gate output schema mismatch")
        if payload.get("label_contract") != LABEL_CONTRACT:
            raise ValueError("fast-gate label contract mismatch")
        instance = cls(FastGateTrainingConfig(**dict(payload["config"])), device=device)
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device).eval()
        instance.feature_mean = (
            payload["feature_mean"].cpu().numpy().astype(np.float32)
        )
        instance.feature_scale = (
            payload["feature_scale"].cpu().numpy().astype(np.float32)
        )
        instance.fitted = bool(payload["fitted"])
        instance.training_history = [dict(row) for row in payload["training_history"]]
        state = dict(payload["provenance"])
        if state.get("fit_role") != TRAIN_ROLE:
            raise ValueError("artifact was not fitted on the required training role")
        calibration_role = state.get("calibration_role")
        if calibration_role not in {None, CALIBRATION_ROLE}:
            raise ValueError("artifact was calibrated on an invalid role")
        if tuple(state.get("architecture", ())) != (FEATURE_DIM, 16, 16, len(HEAD_NAMES)):
            raise ValueError("fast-gate architecture provenance mismatch")
        instance.fit_count = int(state["fit_count"])
        instance.calibration_count = int(state["calibration_count"])
        instance.user_provenance = _safe_provenance(state.get("user", {}))
        if payload["calibrator"] is not None:
            calibrator = MultiOutputCalibrator.from_state(payload["calibrator"])
            if calibrator.method != "platt" or calibrator.head_names != HEAD_NAMES:
                raise ValueError("fast-gate requires two-head Platt calibration")
            instance.calibrator = calibrator
        return instance


class ExactFastGateModule(nn.Module):
    """Deployment graph containing normalization, MLP, and Platt scaling."""

    def __init__(self, predictor: FastGatePredictor):
        super().__init__()
        if not predictor.fitted or predictor.calibrator is None:
            raise RuntimeError("exact export requires a fitted and calibrated predictor")
        states = predictor.calibrator.to_state()["calibrators"]
        if any(state.get("kind") != "platt" for state in states):
            raise ValueError("exact export supports Platt calibration only")
        self.model = deepcopy(predictor.model).cpu().eval()
        self.register_buffer(
            "feature_mean", torch.from_numpy(predictor.feature_mean.copy())
        )
        self.register_buffer(
            "feature_scale", torch.from_numpy(predictor.feature_scale.copy())
        )
        self.register_buffer(
            "platt_slope",
            torch.tensor([float(state["slope"]) for state in states], dtype=torch.float64),
        )
        self.register_buffer(
            "platt_intercept",
            torch.tensor(
                [float(state["intercept"]) for state in states], dtype=torch.float64
            ),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (features - self.feature_mean) / self.feature_scale
        logits_f32 = self.model(normalized)
        logits = logits_f32.to(torch.float64)
        probability = torch.sigmoid(
            logits * self.platt_slope + self.platt_intercept
        ).clamp(1.0e-7, 1.0 - 1.0e-7)
        return logits, probability


def export_exact_torchscript(
    predictor: FastGatePredictor,
    path: str | Path,
) -> dict[str, Any]:
    """Export a load-only exact batch-one-compatible TorchScript module."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    module = ExactFastGateModule(predictor).eval()
    with torch.inference_mode():
        scripted = torch.jit.script(module)
        scripted = torch.jit.freeze(scripted.eval())
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(destination))
    return {
        **predictor.provenance,
        "torchscript_load_only": True,
        "input_dtype": "float32",
        "input_shape": [1, FEATURE_DIM],
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


class TorchScriptFastGateInference:
    """Allocation-light batch-one adapter used by the online runtime."""

    def __init__(self, module: torch.jit.ScriptModule):
        self.module = module.eval()
        self._input = np.empty((1, FEATURE_DIM), dtype=np.float32)

    def infer(self, features: np.ndarray) -> FastGateOutput:
        row = np.asarray(features, dtype=np.float32)
        if row.shape != (FEATURE_DIM,) or not np.all(np.isfinite(row)):
            raise ValueError(f"features must be finite with shape ({FEATURE_DIM},)")
        np.copyto(self._input[0], row)
        with torch.inference_mode():
            logits, probability = self.module(torch.from_numpy(self._input))
        return FastGateOutput(
            local_success_logit=float(logits[0, 0].item()),
            latency_benefit_logit=float(logits[0, 1].item()),
            local_success_probability=float(probability[0, 0].item()),
            latency_benefit_probability=float(probability[0, 1].item()),
        )

    def predict_one(self, features: np.ndarray) -> FastGateOutput:
        return self.infer(features)


def numerical_equivalence(
    predictor: FastGatePredictor,
    module: torch.jit.ScriptModule,
    features: np.ndarray,
    *,
    success_threshold: float | None = None,
    benefit_threshold: float | None = None,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    """Compare eager and exported inference at the batch-one API boundary."""

    rows, _ = _feature_matrix(features)
    exact = TorchScriptFastGateInference(module)
    max_logit_error = 0.0
    max_probability_error = 0.0
    route_matches = 0
    for row in rows:
        eager = predictor.predict_one(row)
        deployed = exact.infer(row)
        max_logit_error = max(
            max_logit_error,
            abs(eager.local_success_logit - deployed.local_success_logit),
            abs(eager.latency_benefit_logit - deployed.latency_benefit_logit),
        )
        max_probability_error = max(
            max_probability_error,
            abs(
                eager.local_success_probability
                - deployed.local_success_probability
            ),
            abs(
                eager.latency_benefit_probability
                - deployed.latency_benefit_probability
            ),
        )
        if success_threshold is None or benefit_threshold is None:
            route_matches += 1
        else:
            eager_fast = (
                eager.local_success_probability >= success_threshold
                and eager.latency_benefit_probability >= benefit_threshold
            )
            deployed_fast = (
                deployed.local_success_probability >= success_threshold
                and deployed.latency_benefit_probability >= benefit_threshold
            )
            route_matches += int(eager_fast == deployed_fast)
    return {
        "query_count": len(rows),
        "max_absolute_logit_error": max_logit_error,
        "max_absolute_probability_error": max_probability_error,
        "route_match_count": route_matches,
        "route_match_rate": route_matches / len(rows),
        "absolute_tolerance": float(atol),
        "passed": bool(
            max_logit_error <= atol
            and max_probability_error <= atol
            and route_matches == len(rows)
        ),
    }


# Explicit long name used by orchestration code; the shorter alias is handy in tests.
HierarchicalFastGatePredictor = FastGatePredictor
