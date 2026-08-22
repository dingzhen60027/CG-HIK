import numpy as np

from confik.models.risk import RiskModel, select_risk_model


def _risk_data(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 9))
    score = features[:, 0] + 0.7 * features[:, 1] - 0.4 * features[:, 4]
    labels = np.digitize(score, [-0.7, 0.4, 1.2])
    return features, labels


def test_risk_selection_calibration_and_persistence(tmp_path) -> None:
    train_x, train_y = _risk_data(1, 500)
    validation_x, validation_y = _risk_data(2, 200)
    calibration_x, calibration_y = _risk_data(3, 200)
    model, scores = select_risk_model(train_x, train_y, validation_x, validation_y)
    assert set(scores) == {"mlp", "gradient_boosting"}
    model.calibrate(calibration_x, calibration_y)
    probabilities = model.predict_proba(validation_x[:5])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    artifact = tmp_path / "risk.joblib"
    model.save(artifact)
    loaded = RiskModel.load(artifact)
    np.testing.assert_allclose(loaded.predict_proba(validation_x[:5]), probabilities)

