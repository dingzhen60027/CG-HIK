"""Disk-only exact inference artifacts for the frozen CG-HIK v4 gate.

The deployment graph contains every learned numerical transformation: the
float32 feature normalization and compact shared-semantic-success MLP, the
float64 Platt mappings, and the float64 Mahalanobis OOD calculation.  The
policy itself remains a small, auditable Python adapter whose thresholds are
loaded from the frozen validation-selected JSON artifact.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..counterfactual_v4.model import (
    ACTION_NAMES,
    CALIBRATION_HEAD_NAMES,
    FEATURE_DIM,
    FEATURE_NAMES,
    LABEL_CONTRACT,
    CounterfactualV4Predictor,
)
from ..counterfactual_v4.policy import DECISION_ENTRIES, V4Decision, V4PolicyConfig


ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class V4InferenceOutput:
    success_probabilities: np.ndarray
    latency_p50_ms: np.ndarray
    latency_p95_ms: np.ndarray
    fail_all_probability: float
    embedding: np.ndarray
    ood_score: float
    is_ood: bool

    def __post_init__(self) -> None:
        success = np.asarray(self.success_probabilities, dtype=np.float64)
        p50 = np.asarray(self.latency_p50_ms, dtype=np.float64)
        p95 = np.asarray(self.latency_p95_ms, dtype=np.float64)
        embedding = np.asarray(self.embedding, dtype=np.float64)
        if success.shape != (3,) or p50.shape != (3,) or p95.shape != (3,):
            raise ValueError("v4 inference output must contain exactly three actions")
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError("v4 embedding must be a non-empty vector")
        numeric = np.concatenate(
            [success, p50, p95, embedding, [self.fail_all_probability, self.ood_score]]
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("v4 inference output contains a non-finite value")
        if np.any((success < 0.0) | (success > 1.0)):
            raise ValueError("v4 success probabilities must lie in [0, 1]")
        if not 0.0 <= float(self.fail_all_probability) <= 1.0:
            raise ValueError("v4 fail-all probability must lie in [0, 1]")
        if np.any(p50 < 0.0) or np.any(p95 < p50):
            raise ValueError("v4 latency quantiles are invalid")
        object.__setattr__(self, "success_probabilities", success)
        object.__setattr__(self, "latency_p50_ms", p50)
        object.__setattr__(self, "latency_p95_ms", p95)
        object.__setattr__(self, "embedding", embedding)


class V4InferenceBackend(Protocol):
    def infer(self, features: np.ndarray) -> V4InferenceOutput: ...


class ExactV4ForwardModule(nn.Module):
    """Single-call, batch-capable numerical deployment graph.

    Float32 operations exactly mirror the trained PyTorch MLP.  Conversion to
    float64 occurs only after the MLP, matching the eager predictor's NumPy
    conversion before Platt calibration and Mahalanobis scoring.
    """

    def __init__(self, predictor: CounterfactualV4Predictor):
        super().__init__()
        if not predictor.fitted:
            raise ValueError("cannot export an unfitted v4 predictor")
        if predictor.calibrator is None or not predictor.calibrator.fitted:
            raise ValueError("v4 predictor lacks frozen probability calibration")
        if predictor.ood_detector is None:
            raise ValueError("v4 predictor lacks a frozen OOD detector")
        if (
            predictor.ood_detector.mean is None
            or predictor.ood_detector.precision is None
            or predictor.ood_detector.threshold is None
        ):
            raise ValueError("v4 OOD detector is not fully frozen")
        if tuple(predictor.calibrator.head_names) != CALIBRATION_HEAD_NAMES:
            raise ValueError("v4 calibration head schema differs from the frozen contract")
        if predictor.calibrator.method != "platt" or any(
            item.__class__.__name__ != "PlattCalibrator"
            or not bool(getattr(item, "fitted", False))
            for item in predictor.calibrator.calibrators
        ):
            raise ValueError("v4 exact deployment requires fitted Platt calibrators")
        if predictor.ood_detector.target_id_coverage != 0.99:
            raise ValueError("v4 OOD threshold must remain the ID calibration 99% quantile")
        provenance = predictor.training_provenance
        if (
            provenance.get("action_success_target") != "semantic_verified_success"
            or not bool(
                provenance.get(
                    "shared_semantic_success_due_to_terminal_fallback_invariance", False
                )
            )
            or not bool(provenance.get("formal_v4_eligible", False))
            or provenance.get("latency_training_source") != "raw_samples"
        ):
            raise ValueError("v4 predictor does not enforce shared semantic success")

        # Copy the trained modules so exporting cannot mutate the in-memory
        # candidate used as the eager equivalence reference.
        model = deepcopy(predictor.model).cpu().eval()
        self.backbone = model.backbone
        self.verified_success_head = model.verified_success_head
        self.latency_head = model.latency_head
        self.fail_all_head = model.fail_all_head
        self.action_count = 3
        self.min_latency_ms = float(model.min_latency_ms)
        self.min_quantile_gap_ms = float(model.min_quantile_gap_ms)

        self.register_buffer(
            "feature_mean", torch.as_tensor(predictor.feature_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(predictor.feature_scale, dtype=torch.float32)
        )
        calibrators = predictor.calibrator.calibrators
        self.register_buffer(
            "platt_slope",
            torch.tensor([float(item.slope) for item in calibrators], dtype=torch.float64),
        )
        self.register_buffer(
            "platt_intercept",
            torch.tensor(
                [float(item.intercept) for item in calibrators], dtype=torch.float64
            ),
        )
        self.register_buffer(
            "ood_mean",
            torch.as_tensor(predictor.ood_detector.mean, dtype=torch.float64),
        )
        self.register_buffer(
            "ood_precision",
            torch.as_tensor(predictor.ood_detector.precision, dtype=torch.float64),
        )
        self.register_buffer(
            "ood_threshold",
            torch.tensor(float(predictor.ood_detector.threshold), dtype=torch.float64),
        )

    def forward(
        self, raw_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        normalized = (raw_features - self.feature_mean) / self.feature_scale
        embedding_f32 = self.backbone(normalized)
        shared_success_logit = self.verified_success_head(embedding_f32)
        latency_raw = self.latency_head(embedding_f32)
        p50_raw, gap_raw = latency_raw.chunk(2, dim=1)
        p50_f32 = F.softplus(p50_raw) + self.min_latency_ms
        p95_f32 = p50_f32 + F.softplus(gap_raw) + self.min_quantile_gap_ms
        fail_all_logit = self.fail_all_head(embedding_f32).squeeze(1)

        shared_probability = torch.sigmoid(
            shared_success_logit.to(torch.float64) * self.platt_slope[0]
            + self.platt_intercept[0]
        ).clamp(1.0e-7, 1.0 - 1.0e-7)
        success_probability = shared_probability.expand(-1, self.action_count)
        fail_all_probability = torch.sigmoid(
            fail_all_logit.to(torch.float64) * self.platt_slope[1]
            + self.platt_intercept[1]
        ).clamp(1.0e-7, 1.0 - 1.0e-7)

        embedding = embedding_f32.to(torch.float64)
        centered = embedding - self.ood_mean
        ood_score = torch.sum(
            torch.matmul(centered, self.ood_precision) * centered, dim=1
        ).clamp_min(0.0)
        is_ood = ood_score > self.ood_threshold
        return (
            success_probability,
            p50_f32.to(torch.float64),
            p95_f32.to(torch.float64),
            fail_all_probability,
            embedding,
            ood_score,
            is_ood,
        )


def _candidate_metadata(predictor: CounterfactualV4Predictor) -> dict[str, Any]:
    assert predictor.calibrator is not None
    assert predictor.ood_detector is not None
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "input": {
            "feature_names": list(FEATURE_NAMES),
            "shape": ["batch", FEATURE_DIM],
            "dtype": "float32",
            "contiguous": True,
        },
        "output": {
            "action_names": list(ACTION_NAMES),
            "layout": [
                "calibrated_semantic_success_probability",
                "latency_p50_ms",
                "latency_p95_ms",
                "calibrated_fail_all_probability",
                "embedding",
                "mahalanobis_ood_score",
                "is_ood",
            ],
            "calibration_dtype": "float64",
            "ood_dtype": "float64",
        },
        "label_contract": dict(LABEL_CONTRACT),
        "training_provenance": predictor.training_provenance,
        "model_config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in predictor.config.__dict__.items()
        },
        "calibration": predictor.calibrator.to_state(),
        "ood": predictor.ood_detector.to_state(),
        "torchscript_load_only": True,
        "retrace_allowed": False,
    }


def export_exact_v4_predictor(
    candidate_path: str | Path,
    torchscript_path: str | Path,
) -> dict[str, Any]:
    """Load a frozen candidate from disk and persist its exact deployment graph."""

    candidate = Path(candidate_path)
    destination = Path(torchscript_path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(candidate)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    predictor = CounterfactualV4Predictor.load(candidate, device="cpu")
    predictor.model.eval()
    module = ExactV4ForwardModule(predictor).eval()
    with torch.inference_mode():
        scripted = torch.jit.script(module)
        scripted = torch.jit.freeze(scripted.eval())
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(destination))
    return _candidate_metadata(predictor)


def load_exact_v4_predictor(
    torchscript_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.jit.ScriptModule:
    path = Path(torchscript_path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return torch.jit.load(str(path), map_location=torch.device(device)).eval()


class EagerV4Inference:
    """Disk-loaded candidate used only as the validation equivalence reference."""

    def __init__(self, candidate_path: str | Path):
        self.predictor = CounterfactualV4Predictor.load(candidate_path, device="cpu")
        if self.predictor.ood_detector is None or self.predictor.ood_detector.threshold is None:
            raise ValueError("candidate OOD detector is not frozen")

    def infer(self, features: np.ndarray) -> V4InferenceOutput:
        prediction = self.predictor.predict(np.asarray(features, dtype=np.float32))
        if prediction.ood_score is None or prediction.is_ood is None:
            raise RuntimeError("eager candidate returned no OOD decision")
        return V4InferenceOutput(
            prediction.verified_success_probability[0],
            prediction.latency_p50_ms[0],
            prediction.latency_p95_ms[0],
            float(prediction.fail_all_probability[0]),
            prediction.embedding[0],
            float(prediction.ood_score[0]),
            bool(prediction.is_ood[0]),
        )


class TorchScriptV4Inference:
    """Batch-one adapter around the persisted exact TorchScript graph."""

    def __init__(self, module: torch.jit.ScriptModule):
        self.module = module.eval()
        self._input = np.empty((1, FEATURE_DIM), dtype=np.float32)

    def infer(self, features: np.ndarray) -> V4InferenceOutput:
        row = np.asarray(features, dtype=np.float32)
        if row.shape != (FEATURE_DIM,) or not np.all(np.isfinite(row)):
            raise ValueError(f"features must have shape ({FEATURE_DIM},) and be finite")
        np.copyto(self._input[0], row)
        tensor = torch.from_numpy(self._input)
        with torch.inference_mode():
            values = self.module(tensor)
        success, p50, p95, fail, embedding, ood_score, is_ood = values
        return V4InferenceOutput(
            success[0].detach().cpu().numpy(),
            p50[0].detach().cpu().numpy(),
            p95[0].detach().cpu().numpy(),
            float(fail[0].item()),
            embedding[0].detach().cpu().numpy(),
            float(ood_score[0].item()),
            bool(is_ood[0].item()),
        )


def _decision_from_output(
    output: V4InferenceOutput,
    config: V4PolicyConfig,
) -> V4Decision:
    base = {
        "ood_score": float(output.ood_score),
        "is_ood": bool(output.is_ood),
        "predicted_success": tuple(float(value) for value in output.success_probabilities),
        "predicted_p50_ms": tuple(float(value) for value in output.latency_p50_ms),
        "predicted_p95_ms": tuple(float(value) for value in output.latency_p95_ms),
        "fail_all_probability": float(output.fail_all_probability),
    }
    if output.is_ood:
        return V4Decision(
            action="defer", reason="ood_defer", eligible_actions=(), **base
        )
    eligible_index = np.flatnonzero(
        (output.success_probabilities >= config.minimum_success_probability)
        & (output.latency_p95_ms <= config.deadline_ms)
    )
    eligible = tuple(DECISION_ENTRIES[int(index)] for index in eligible_index)
    if output.fail_all_probability >= config.reject_probability and not len(
        eligible_index
    ):
        return V4Decision(
            action="reject",
            reason="high_confidence_fail_all",
            eligible_actions=eligible,
            **base,
        )
    if not len(eligible_index):
        return V4Decision(
            action="defer",
            reason="uncertain_no_eligible_action",
            eligible_actions=eligible,
            **base,
        )
    fastest = int(eligible_index[np.argmin(output.latency_p95_ms[eligible_index])])
    conservative = int(np.min(eligible_index))
    improvement = output.latency_p95_ms[conservative] - output.latency_p95_ms[fastest]
    selected = conservative if improvement < config.latency_tie_margin_ms else fastest
    return V4Decision(
        action=DECISION_ENTRIES[selected],
        reason=(
            "tie_margin_conservative_entry"
            if selected == conservative and selected != fastest
            else "minimum_predicted_p95"
        ),
        eligible_actions=eligible,
        **base,
    )


class FrozenV4Policy:
    """Frozen policy adapter preserving raw, independently calibrated outputs."""

    def __init__(self, backend: V4InferenceBackend, config: V4PolicyConfig):
        self.backend = backend
        self.config = config
        self.last_output: V4InferenceOutput | None = None
        self.last_decision: V4Decision | None = None

    def decide(self, features: np.ndarray) -> V4Decision:
        output = self.backend.infer(np.asarray(features, dtype=np.float64))
        decision = _decision_from_output(output, self.config)
        self.last_output = output
        self.last_decision = decision
        return decision


def load_policy_config(path: str | Path) -> tuple[V4PolicyConfig, dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("v4 policy artifact must contain a JSON mapping")
    if bool(payload.get("test_data_loaded", True)):
        raise RuntimeError("v4 policy artifact indicates forbidden test access")
    if payload.get("label_contract") != LABEL_CONTRACT:
        raise RuntimeError("v4 policy label contract differs from the candidate model")
    return V4PolicyConfig(**dict(payload["policy_config"])), dict(payload)


def decision_record(decision: V4Decision) -> dict[str, Any]:
    """Return formal raw prediction fields without CalibratedRisk normalization."""

    return {
        "decision_action": decision.action,
        "decision_reason": decision.reason,
        "eligible_actions": list(decision.eligible_actions),
        "predicted_success": list(decision.predicted_success),
        "predicted_p50_ms": list(decision.predicted_p50_ms),
        "predicted_p95_ms": list(decision.predicted_p95_ms),
        "fail_all_probability": decision.fail_all_probability,
        "ood_score": decision.ood_score,
        "is_ood": decision.is_ood,
    }
