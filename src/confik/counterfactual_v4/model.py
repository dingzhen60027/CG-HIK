"""Compact multi-head counterfactual latency predictor for CG-HIK v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .calibration import (
    MultiOutputCalibrator,
    binary_calibration_metrics,
    sigmoid,
)
from .ood import EmbeddingMahalanobisOOD


FEATURE_DIM = 9
ACTION_NAMES = ("easy", "medium", "hard")
CALIBRATION_HEAD_NAMES = (
    "deadline_success_easy",
    "deadline_success_medium",
    "deadline_success_hard",
    "fail_all",
)


@dataclass(frozen=True)
class CounterfactualTrainingConfig:
    hidden_sizes: tuple[int, ...] = (32, 32)
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    success_loss_weight: float = 1.0
    latency_loss_weight: float = 1.0
    fail_all_loss_weight: float = 1.0
    gradient_clip_norm: float = 5.0
    min_latency_ms: float = 1e-4
    min_quantile_gap_ms: float = 1e-5
    seed: int = 17

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.min_latency_ms <= 0.0 or self.min_quantile_gap_ms <= 0.0:
            raise ValueError("latency lower bounds must be positive")


class CounterfactualMultiHeadMLP(nn.Module):
    """Shared compact embedding with success, latency, and fail-all heads."""

    def __init__(self, config: CounterfactualTrainingConfig):
        super().__init__()
        layers: list[nn.Module] = []
        previous = FEATURE_DIM
        for width in config.hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.deadline_success_head = nn.Linear(previous, len(ACTION_NAMES))
        self.latency_head = nn.Linear(previous, 2 * len(ACTION_NAMES))
        self.fail_all_head = nn.Linear(previous, 1)
        self.min_latency_ms = float(config.min_latency_ms)
        self.min_quantile_gap_ms = float(config.min_quantile_gap_ms)

    @property
    def embedding_dim(self) -> int:
        return self.fail_all_head.in_features

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] != FEATURE_DIM:
            raise ValueError(f"inputs must have shape (N, {FEATURE_DIM})")
        embedding = self.backbone(inputs)
        success_logits = self.deadline_success_head(embedding)
        latency_raw = self.latency_head(embedding)
        p50_raw, gap_raw = latency_raw.chunk(2, dim=1)
        latency_p50 = F.softplus(p50_raw) + self.min_latency_ms
        latency_p95 = (
            latency_p50 + F.softplus(gap_raw) + self.min_quantile_gap_ms
        )
        fail_all_logit = self.fail_all_head(embedding).squeeze(1)
        return success_logits, latency_p50, latency_p95, fail_all_logit, embedding


def pinball_loss(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    """Standard quantile regression (pinball) loss."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


@dataclass(frozen=True)
class CounterfactualPrediction:
    deadline_success_logits: np.ndarray
    deadline_success_probability: np.ndarray
    latency_p50_ms: np.ndarray
    latency_p95_ms: np.ndarray
    fail_all_logit: np.ndarray
    fail_all_probability: np.ndarray
    embedding: np.ndarray
    ood_score: np.ndarray | None
    is_ood: np.ndarray | None


def _feature_matrix(features: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(features, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM or array.shape[0] == 0:
        raise ValueError(f"features must have shape (N, {FEATURE_DIM})")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return np.ascontiguousarray(array), single


def _target_matrix(values: np.ndarray, count: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (count, len(ACTION_NAMES)):
        raise ValueError(f"{name} must have shape ({count}, {len(ACTION_NAMES)})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _latency_sample_tensor(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    expected_prefix = (count, len(ACTION_NAMES))
    if array.ndim != 3 or array.shape[:2] != expected_prefix:
        raise ValueError(
            "latency_samples_ms must have shape "
            f"({count}, {len(ACTION_NAMES)}, R)"
        )
    if array.shape[2] < 4:
        raise ValueError("latency_samples_ms requires at least four repeats per action")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("latency_samples_ms must be finite and strictly positive")
    return np.ascontiguousarray(array)


class CounterfactualV4Predictor:
    """Training, calibration, OOD fitting, inference, and safe serialization.

    Model fitting uses training labels only.  ``calibrate`` and
    ``fit_ood_detector`` are explicit calls intended for validation/pilot
    arrays.  No method discovers or reads a dataset path on its own.
    """

    FORMAT_VERSION = 2

    def __init__(
        self,
        config: CounterfactualTrainingConfig | None = None,
        *,
        device: str = "cpu",
    ):
        self.config = config or CounterfactualTrainingConfig()
        self.device = torch.device(device)
        self.model = self._new_model().to(self.device)
        self.feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feature_scale = np.ones(FEATURE_DIM, dtype=np.float32)
        self.fitted = False
        self.calibrator: MultiOutputCalibrator | None = None
        self.ood_detector: EmbeddingMahalanobisOOD | None = None
        self.training_history: list[dict[str, float]] = []
        self.latency_training_source: str | None = None
        self.latency_repeat_count: int | None = None

    def _new_model(self) -> CounterfactualMultiHeadMLP:
        # Isolate initialization from the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            return CounterfactualMultiHeadMLP(self.config)

    def _normalized_tensor(self, features: np.ndarray) -> Tensor:
        array, _ = _feature_matrix(features)
        normalized = (array - self.feature_mean) / self.feature_scale
        return torch.from_numpy(np.ascontiguousarray(normalized)).to(self.device)

    def fit(
        self,
        features: np.ndarray,
        deadline_success: np.ndarray,
        latency_samples_ms: np.ndarray,
        fail_all: np.ndarray | None = None,
    ) -> "CounterfactualV4Predictor":
        """Fit the formal v4 model from every raw timing repeat.

        ``latency_samples_ms`` must contain at least four interleaved timing
        repeats for every query/action.  The P50 and P95 heads are optimized
        against all ``N * 3 * R`` observations; per-query empirical winners
        or pre-aggregated percentiles never enter this formal path.
        """

        inputs, _ = _feature_matrix(features)
        count = inputs.shape[0]
        success = _target_matrix(deadline_success, count, name="deadline_success")
        latency_samples = _latency_sample_tensor(latency_samples_ms, count)
        if np.any((success < 0.0) | (success > 1.0)):
            raise ValueError("deadline_success targets must lie in [0, 1]")
        fail_target = self._fail_all_target(success, fail_all)
        return self._fit_validated(
            inputs=inputs,
            success=success,
            fail_target=fail_target,
            latency_samples=latency_samples,
            aggregated_p50=None,
            aggregated_p95=None,
            latency_training_source="raw_samples",
        )

    def fit_aggregated_for_testing(
        self,
        features: np.ndarray,
        deadline_success: np.ndarray,
        latency_p50_ms: np.ndarray,
        latency_p95_ms: np.ndarray,
        fail_all: np.ndarray | None = None,
    ) -> "CounterfactualV4Predictor":
        """Compatibility path for unit tests, never for a formal v4 artifact.

        The explicit method name and serialized ``aggregated_test_only``
        marker prevent a model trained on unstable aggregated labels from
        being mistaken for the formal raw-repeat model.
        """

        inputs, _ = _feature_matrix(features)
        count = inputs.shape[0]
        success = _target_matrix(deadline_success, count, name="deadline_success")
        p50 = _target_matrix(latency_p50_ms, count, name="latency_p50_ms")
        p95 = _target_matrix(latency_p95_ms, count, name="latency_p95_ms")
        if np.any((success < 0.0) | (success > 1.0)):
            raise ValueError("deadline_success targets must lie in [0, 1]")
        if np.any(p50 <= 0.0) or np.any(p95 < p50):
            raise ValueError("latency targets must be positive with P95 >= P50")
        fail_target = self._fail_all_target(success, fail_all)
        return self._fit_validated(
            inputs=inputs,
            success=success,
            fail_target=fail_target,
            latency_samples=None,
            aggregated_p50=p50,
            aggregated_p95=p95,
            latency_training_source="aggregated_test_only",
        )

    @staticmethod
    def _fail_all_target(
        success: np.ndarray,
        fail_all: np.ndarray | None,
    ) -> np.ndarray:
        count = success.shape[0]
        if fail_all is None:
            fail_target = np.all(success < 0.5, axis=1).astype(np.float32)
        else:
            fail_target = np.asarray(fail_all, dtype=np.float32).reshape(-1)
            if fail_target.shape != (count,) or not np.all(np.isfinite(fail_target)):
                raise ValueError(f"fail_all must have shape ({count},) and be finite")
            if np.any((fail_target < 0.0) | (fail_target > 1.0)):
                raise ValueError("fail_all targets must lie in [0, 1]")
        return fail_target

    def _fit_validated(
        self,
        *,
        inputs: np.ndarray,
        success: np.ndarray,
        fail_target: np.ndarray,
        latency_samples: np.ndarray | None,
        aggregated_p50: np.ndarray | None,
        aggregated_p95: np.ndarray | None,
        latency_training_source: str,
    ) -> "CounterfactualV4Predictor":
        count = inputs.shape[0]
        raw_latency = latency_samples is not None
        if raw_latency:
            if aggregated_p50 is not None or aggregated_p95 is not None:
                raise ValueError("raw and aggregated latency targets are mutually exclusive")
            assert latency_samples is not None
        elif aggregated_p50 is None or aggregated_p95 is None:
            raise ValueError("aggregated compatibility training requires P50 and P95")

        self.feature_mean = np.mean(inputs, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(inputs, axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        normalized = np.ascontiguousarray(
            (inputs - self.feature_mean) / self.feature_scale, dtype=np.float32
        )

        self.model = self._new_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if raw_latency:
            tensors = (
                torch.from_numpy(normalized),
                torch.from_numpy(success),
                torch.from_numpy(latency_samples),
                torch.from_numpy(fail_target),
            )
        else:
            assert aggregated_p50 is not None and aggregated_p95 is not None
            tensors = (
                torch.from_numpy(normalized),
                torch.from_numpy(success),
                torch.from_numpy(aggregated_p50),
                torch.from_numpy(aggregated_p95),
                torch.from_numpy(fail_target),
            )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.training_history = []
        self.model.train()
        for epoch in range(self.config.epochs):
            permutation = torch.randperm(count, generator=generator)
            totals = np.zeros(4, dtype=np.float64)
            batches = 0
            for start in range(0, count, self.config.batch_size):
                index = permutation[start : start + self.config.batch_size]
                batch = [tensor[index].to(self.device) for tensor in tensors]
                optimizer.zero_grad(set_to_none=True)
                success_logits, predicted_p50, predicted_p95, fail_logit, _ = self.model(
                    batch[0]
                )
                success_loss = F.binary_cross_entropy_with_logits(
                    success_logits, batch[1]
                )
                if raw_latency:
                    # Broadcast each predicted query/action quantile across
                    # every timing repeat.  No empirical per-query P95 winner
                    # is constructed or used as a supervised target.
                    latency_loss = pinball_loss(
                        predicted_p50.unsqueeze(-1), batch[2], 0.50
                    ) + pinball_loss(predicted_p95.unsqueeze(-1), batch[2], 0.95)
                    fail_batch = batch[3]
                else:
                    latency_loss = pinball_loss(
                        predicted_p50, batch[2], 0.50
                    ) + pinball_loss(predicted_p95, batch[3], 0.95)
                    fail_batch = batch[4]
                fail_loss = F.binary_cross_entropy_with_logits(fail_logit, fail_batch)
                loss = (
                    self.config.success_loss_weight * success_loss
                    + self.config.latency_loss_weight * latency_loss
                    + self.config.fail_all_loss_weight * fail_loss
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                totals += np.asarray(
                    [
                        float(loss.detach().cpu()),
                        float(success_loss.detach().cpu()),
                        float(latency_loss.detach().cpu()),
                        float(fail_loss.detach().cpu()),
                    ]
                )
                batches += 1
            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(totals[0] / batches),
                    "success_bce": float(totals[1] / batches),
                    "latency_pinball": float(totals[2] / batches),
                    "fail_all_bce": float(totals[3] / batches),
                }
            )
        self.model.eval()
        self.fitted = True
        self.latency_training_source = latency_training_source
        self.latency_repeat_count = (
            int(latency_samples.shape[2]) if latency_samples is not None else None
        )
        # These artifacts depend on the previous embedding/logits and cannot
        # survive a refit.
        self.calibrator = None
        self.ood_detector = None
        return self

    @property
    def training_provenance(self) -> dict[str, Any]:
        """Auditable declaration of the latency supervision used by ``fit``."""

        return {
            "latency_training_source": self.latency_training_source,
            "latency_repeat_count": self.latency_repeat_count,
            "raw_sample_pinball": self.latency_training_source == "raw_samples",
            "formal_v4_eligible": (
                self.latency_training_source == "raw_samples"
                and self.latency_repeat_count is not None
                and self.latency_repeat_count >= 4
            ),
        }

    def _raw_numpy(self, features: np.ndarray) -> tuple[np.ndarray, ...]:
        if not self.fitted:
            raise RuntimeError("predictor has not been fitted")
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(self._normalized_tensor(features))
        return tuple(output.detach().cpu().numpy().astype(np.float64) for output in outputs)

    def embeddings(self, features: np.ndarray) -> np.ndarray:
        return self._raw_numpy(features)[4]

    def calibrate(
        self,
        validation_features: np.ndarray,
        validation_deadline_success: np.ndarray,
        validation_fail_all: np.ndarray,
        *,
        method: str = "platt",
    ) -> "CounterfactualV4Predictor":
        inputs, _ = _feature_matrix(validation_features)
        success = _target_matrix(
            validation_deadline_success,
            inputs.shape[0],
            name="validation_deadline_success",
        )
        fail = np.asarray(validation_fail_all, dtype=np.float64).reshape(-1)
        if fail.shape != (inputs.shape[0],):
            raise ValueError(f"validation_fail_all must have shape ({inputs.shape[0]},)")
        success_logits, _, _, fail_logit, _ = self._raw_numpy(inputs)
        logits = np.column_stack([success_logits, fail_logit])
        targets = np.column_stack([success, fail])
        self.calibrator = MultiOutputCalibrator(
            method, CALIBRATION_HEAD_NAMES  # type: ignore[arg-type]
        ).fit(logits, targets)
        return self

    def fit_ood_detector(
        self,
        train_features: np.ndarray,
        id_validation_features: np.ndarray,
        *,
        target_id_coverage: float = 0.95,
        shrinkage: float = 0.05,
    ) -> "CounterfactualV4Predictor":
        detector = EmbeddingMahalanobisOOD(shrinkage=shrinkage)
        detector.fit(self.embeddings(train_features))
        detector.calibrate_threshold(
            self.embeddings(id_validation_features),
            target_id_coverage=target_id_coverage,
        )
        self.ood_detector = detector
        return self

    def predict(self, features: np.ndarray) -> CounterfactualPrediction:
        inputs, _ = _feature_matrix(features)
        success_logits, p50, p95, fail_logit, embedding = self._raw_numpy(inputs)
        combined_logits = np.column_stack([success_logits, fail_logit])
        if self.calibrator is None:
            combined_probability = sigmoid(combined_logits)
        else:
            combined_probability = self.calibrator.predict_proba(combined_logits)
        if self.ood_detector is None:
            ood_score = None
            is_ood = None
        else:
            ood_score = self.ood_detector.score_samples(embedding)
            is_ood = (
                None
                if self.ood_detector.threshold is None
                else self.ood_detector.predict_ood(embedding)
            )
        return CounterfactualPrediction(
            deadline_success_logits=success_logits,
            deadline_success_probability=combined_probability[:, : len(ACTION_NAMES)],
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            fail_all_logit=fail_logit,
            fail_all_probability=combined_probability[:, -1],
            embedding=embedding,
            ood_score=ood_score,
            is_ood=is_ood,
        )

    def calibration_metrics(
        self,
        features: np.ndarray,
        deadline_success: np.ndarray,
        fail_all: np.ndarray,
        *,
        bins: int = 15,
        confidence_threshold: float = 0.8,
    ) -> dict[str, dict[str, float]]:
        inputs, _ = _feature_matrix(features)
        success = _target_matrix(deadline_success, inputs.shape[0], name="deadline_success")
        fail = np.asarray(fail_all, dtype=np.float64).reshape(-1)
        if fail.shape != (inputs.shape[0],):
            raise ValueError(f"fail_all must have shape ({inputs.shape[0]},)")
        prediction = self.predict(inputs)
        probabilities = np.column_stack(
            [prediction.deadline_success_probability, prediction.fail_all_probability]
        )
        targets = np.column_stack([success, fail])
        return {
            name: binary_calibration_metrics(
                probabilities[:, index],
                targets[:, index],
                bins=bins,
                confidence_threshold=confidence_threshold,
            )
            for index, name in enumerate(CALIBRATION_HEAD_NAMES)
        }

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted predictor")
        payload: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "config": asdict(self.config),
            "state_dict": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "feature_mean": torch.from_numpy(self.feature_mean.copy()),
            "feature_scale": torch.from_numpy(self.feature_scale.copy()),
            "fitted": self.fitted,
            "training_history": self.training_history,
            "training_provenance": self.training_provenance,
            "calibrator": None if self.calibrator is None else self.calibrator.to_state(),
            "ood_detector": (
                None if self.ood_detector is None else self.ood_detector.to_state()
            ),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "CounterfactualV4Predictor":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported counterfactual v4 model artifact")
        config_state = dict(payload["config"])
        config_state["hidden_sizes"] = tuple(config_state["hidden_sizes"])
        instance = cls(CounterfactualTrainingConfig(**config_state), device=device)
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device).eval()
        instance.feature_mean = payload["feature_mean"].cpu().numpy().astype(np.float32)
        instance.feature_scale = payload["feature_scale"].cpu().numpy().astype(np.float32)
        instance.fitted = bool(payload["fitted"])
        instance.training_history = [dict(row) for row in payload["training_history"]]
        provenance = dict(payload["training_provenance"])
        instance.latency_training_source = str(provenance["latency_training_source"])
        repeat_count = provenance["latency_repeat_count"]
        instance.latency_repeat_count = None if repeat_count is None else int(repeat_count)
        if bool(provenance["raw_sample_pinball"]) != (
            instance.latency_training_source == "raw_samples"
        ):
            raise ValueError("inconsistent latency training provenance")
        if instance.latency_training_source == "raw_samples" and (
            instance.latency_repeat_count is None or instance.latency_repeat_count < 4
        ):
            raise ValueError("formal raw-sample artifact has an invalid repeat count")
        if payload["calibrator"] is not None:
            instance.calibrator = MultiOutputCalibrator.from_state(payload["calibrator"])
        if payload["ood_detector"] is not None:
            instance.ood_detector = EmbeddingMahalanobisOOD.from_state(
                payload["ood_detector"]
            )
        return instance


# Short alias used by experiment code and papers.
CounterfactualPredictor = CounterfactualV4Predictor
