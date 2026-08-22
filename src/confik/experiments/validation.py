from __future__ import annotations

import numpy as np

from ..data.datasets import TransitionDataset
from ..geometry import pose_distance
from ..models.seed import TorchSeedEnsemble


def seed_validation_metrics(
    ensemble: TorchSeedEnsemble,
    dataset: TransitionDataset,
    *,
    pose_samples: int = 2000,
) -> dict[str, object]:
    predictions = ensemble.predict_deltas_batch(
        dataset.previous_q,
        dataset.target_position,
        dataset.target_rotation,
    )
    target_delta = np.stack(
        [
            ensemble.kinematics.difference(target, previous)
            for target, previous in zip(dataset.target_q, dataset.previous_q, strict=True)
        ]
    )
    error = predictions - target_delta[:, None, :]
    member_l2 = np.linalg.norm(error, axis=2)
    mean_prediction = np.mean(predictions, axis=1)
    mean_q = ensemble.kinematics.clip(dataset.previous_q + mean_prediction)

    pairwise_max: list[np.ndarray] = []
    for first in range(predictions.shape[1]):
        for second in range(first + 1, predictions.shape[1]):
            pairwise_max.append(np.max(np.abs(predictions[:, first] - predictions[:, second]), axis=1))
    diversity = np.max(np.stack(pairwise_max), axis=0) if pairwise_max else np.zeros(len(dataset))

    indices = np.linspace(0, len(dataset) - 1, min(pose_samples, len(dataset)), dtype=int)
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for index in indices:
        position, orientation = pose_distance(
            ensemble.kinematics.forward(dataset.target_q[index]),
            ensemble.kinematics.forward(mean_q[index]),
        )
        position_errors.append(position)
        orientation_errors.append(orientation)

    return {
        "sample_count": len(dataset),
        "joint_l2_error_rad": {
            "mean": float(np.mean(np.mean(member_l2, axis=1))),
            "p50": float(np.percentile(np.mean(member_l2, axis=1), 50)),
            "p95": float(np.percentile(np.mean(member_l2, axis=1), 95)),
        },
        "ensemble_diversity_max_joint_rad": {
            "mean": float(np.mean(diversity)),
            "p50": float(np.percentile(diversity, 50)),
            "p95": float(np.percentile(diversity, 95)),
            "max": float(np.max(diversity)),
            "fraction_above_0_05": float(np.mean(diversity >= 0.05)),
        },
        "mean_seed_pose_error": {
            "position_mm_mean": 1000.0 * float(np.mean(position_errors)),
            "position_mm_p95": 1000.0 * float(np.percentile(position_errors, 95)),
            "orientation_deg_mean": float(np.rad2deg(np.mean(orientation_errors))),
            "orientation_deg_p95": float(np.rad2deg(np.percentile(orientation_errors, 95))),
            "pose_sample_count": len(indices),
        },
    }

