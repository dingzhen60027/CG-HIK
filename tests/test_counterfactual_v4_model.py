from __future__ import annotations

import numpy as np
import pytest
import torch

from confik.counterfactual_v4.calibration import (
    MultiOutputCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    binary_calibration_metrics,
)
from confik.counterfactual_v4.model import (
    ACTION_NAMES,
    FEATURE_DIM,
    CounterfactualMultiHeadMLP,
    CounterfactualTrainingConfig,
    CounterfactualV4Predictor,
    pinball_loss,
)
from confik.counterfactual_v4.ood import (
    EmbeddingMahalanobisOOD,
    ood_detection_metrics,
)


def _synthetic(count: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, FEATURE_DIM)).astype(np.float32)
    success_score = np.column_stack(
        [
            1.8 - 0.7 * features[:, 0],
            1.2 - 0.5 * features[:, 1],
            0.8 - 0.4 * features[:, 2],
        ]
    )
    success = (success_score + rng.normal(scale=0.4, size=success_score.shape) > 0).astype(
        np.float32
    )
    fail_all = np.all(success == 0.0, axis=1).astype(np.float32)
    base = np.column_stack(
        [
            0.8 + 0.10 * np.abs(features[:, 3]),
            1.5 + 0.15 * np.abs(features[:, 4]),
            2.4 + 0.20 * np.abs(features[:, 5]),
        ]
    ).astype(np.float32)
    repeat_noise = rng.lognormal(
        mean=-2.0,
        sigma=0.35,
        size=(count, len(ACTION_NAMES), 7),
    ).astype(np.float32)
    latency_samples = base[:, :, None] + repeat_noise
    return features, success, latency_samples.astype(np.float32), fail_all


def _config() -> CounterfactualTrainingConfig:
    return CounterfactualTrainingConfig(
        hidden_sizes=(16, 12),
        epochs=18,
        batch_size=32,
        learning_rate=3e-3,
        seed=23,
    )


def test_architecture_shapes_positive_ordered_latency_and_pinball() -> None:
    model = CounterfactualMultiHeadMLP(_config())
    outputs = model(torch.zeros(7, FEATURE_DIM))
    success, p50, p95, fail_all, embedding = outputs
    assert success.shape == (7, len(ACTION_NAMES))
    assert p50.shape == p95.shape == (7, len(ACTION_NAMES))
    assert fail_all.shape == (7,)
    assert embedding.shape == (7, 12)
    assert torch.all(p50 > 0.0)
    assert torch.all(p95 > p50)
    assert pinball_loss(torch.tensor([1.0]), torch.tensor([2.0]), 0.5).item() == 0.5
    with pytest.raises(ValueError):
        model(torch.zeros(2, FEATURE_DIM - 1))


def test_training_is_deterministic_and_round_trip_is_exact(tmp_path) -> None:
    train = _synthetic(96, 1)
    validation = _synthetic(64, 2)
    first = CounterfactualV4Predictor(_config()).fit(*train)
    second = CounterfactualV4Predictor(_config()).fit(*train)
    first_raw = first.predict(validation[0])
    second_raw = second.predict(validation[0])
    np.testing.assert_array_equal(
        first_raw.deadline_success_logits, second_raw.deadline_success_logits
    )
    np.testing.assert_array_equal(first_raw.latency_p50_ms, second_raw.latency_p50_ms)
    assert np.all(first_raw.latency_p50_ms > 0.0)
    assert np.all(first_raw.latency_p95_ms > first_raw.latency_p50_ms)
    assert first.training_provenance == {
        "latency_training_source": "raw_samples",
        "latency_repeat_count": 7,
        "raw_sample_pinball": True,
        "formal_v4_eligible": True,
    }

    first.calibrate(validation[0], validation[1], validation[3], method="platt")
    first.fit_ood_detector(
        train[0], validation[0], target_id_coverage=0.90, shrinkage=0.1
    )
    artifact = tmp_path / "counterfactual_v4.pt"
    first.save(artifact)
    restored = CounterfactualV4Predictor.load(artifact)
    assert restored.training_provenance == first.training_provenance
    expected = first.predict(validation[0])
    actual = restored.predict(validation[0])
    for name in (
        "deadline_success_probability",
        "latency_p50_ms",
        "latency_p95_ms",
        "fail_all_probability",
        "embedding",
        "ood_score",
        "is_ood",
    ):
        np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))

    report = restored.calibration_metrics(validation[0], validation[1], validation[3])
    assert set(report) == {
        "deadline_success_easy",
        "deadline_success_medium",
        "deadline_success_hard",
        "fail_all",
    }
    assert all(set(row) == {"ece", "brier", "nll", "coverage"} for row in report.values())


def test_raw_latency_shape_is_mandatory_and_aggregated_path_is_test_only() -> None:
    features, success, samples, fail_all = _synthetic(24, 11)
    predictor = CounterfactualV4Predictor(
        CounterfactualTrainingConfig(
            hidden_sizes=(8,), epochs=2, batch_size=12, seed=5
        )
    )
    with pytest.raises(ValueError, match="shape"):
        predictor.fit(features, success, samples[:, :, 0], fail_all)
    with pytest.raises(ValueError, match="at least four"):
        predictor.fit(features, success, samples[:, :, :3], fail_all)

    p50 = np.quantile(samples, 0.50, axis=2)
    p95 = np.quantile(samples, 0.95, axis=2)
    predictor.fit_aggregated_for_testing(features, success, p50, p95, fail_all)
    assert predictor.training_provenance == {
        "latency_training_source": "aggregated_test_only",
        "latency_repeat_count": None,
        "raw_sample_pinball": False,
        "formal_v4_eligible": False,
    }


def test_platt_temperature_and_multihead_calibration_metrics() -> None:
    logits = np.linspace(-3.0, 3.0, 80)
    targets = (logits + np.sin(logits) > 0.0).astype(np.float64)
    for calibrator in (PlattCalibrator(), TemperatureCalibrator()):
        probabilities = calibrator.fit(logits, targets).predict_proba(logits)
        assert np.all((probabilities > 0.0) & (probabilities < 1.0))
        metrics = binary_calibration_metrics(
            probabilities, targets, bins=8, confidence_threshold=0.75
        )
        assert set(metrics) == {"ece", "brier", "nll", "coverage"}
        assert all(np.isfinite(value) for value in metrics.values())

    matrix_logits = np.column_stack([logits, -logits])
    matrix_targets = np.column_stack([targets, 1.0 - targets])
    multi = MultiOutputCalibrator("temperature", ("a", "b")).fit(
        matrix_logits, matrix_targets
    )
    assert multi.predict_proba(matrix_logits).shape == (80, 2)
    restored = MultiOutputCalibrator.from_state(multi.to_state())
    np.testing.assert_array_equal(
        multi.predict_proba(matrix_logits), restored.predict_proba(matrix_logits)
    )


def test_mahalanobis_ood_validation_threshold_metrics_and_save(tmp_path) -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(300, 8))
    validation = rng.normal(size=(120, 8))
    shifted = rng.normal(loc=5.0, size=(80, 8))
    detector = EmbeddingMahalanobisOOD(shrinkage=0.1).fit(train)
    threshold = detector.calibrate_threshold(validation, target_id_coverage=0.95)
    assert detector.id_coverage(validation) >= 0.95
    assert np.mean(detector.predict_ood(shifted)) > 0.95
    metrics = ood_detection_metrics(
        detector.score_samples(validation),
        detector.score_samples(shifted),
        threshold=threshold,
    )
    assert metrics["auroc"] > 0.99
    assert metrics["auprc"] > 0.99
    assert metrics["id_coverage"] >= 0.95

    artifact = tmp_path / "ood.json"
    detector.save(artifact)
    restored = EmbeddingMahalanobisOOD.load(artifact)
    np.testing.assert_array_equal(
        detector.score_samples(shifted), restored.score_samples(shifted)
    )
