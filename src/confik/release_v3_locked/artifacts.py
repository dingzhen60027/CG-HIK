from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..latency_pilot_v3.optimized_inference import (
    OptimizedSeedEngine,
    VectorizedHGBRiskModel,
)
from ..models.risk import RiskModel
from ..models.seed import SeedTrainingConfig, TorchSeedEnsemble


FOREST_SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1
NORMALIZATION_SCHEMA_VERSION = 1


def _locked_array(values: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.flags.writeable = False
    return result


def export_frozen_risk(
    model: RiskModel,
    forest_path: str | Path,
    calibration_path: str | Path,
) -> VectorizedHGBRiskModel:
    """Persist the exact arrays consumed by the locked NumPy HGB evaluator."""

    frozen = VectorizedHGBRiskModel(model)
    np.savez(
        forest_path,
        schema_version=np.asarray(FOREST_SCHEMA_VERSION, dtype=np.int64),
        n_classes=np.asarray(frozen.n_classes, dtype=np.int64),
        trees_per_iteration=np.asarray(frozen.trees_per_iteration, dtype=np.int64),
        iterations=np.asarray(frozen.iterations, dtype=np.int64),
        tree_count=np.asarray(frozen.tree_count, dtype=np.int64),
        max_depth=np.asarray(frozen.max_depth, dtype=np.int64),
        feature_idx=frozen.feature_idx,
        threshold=frozen.threshold,
        missing_left=frozen.missing_left,
        left=frozen.left,
        right=frozen.right,
        is_leaf=frozen.is_leaf,
        value=frozen.value,
        baseline=frozen.baseline,
    )
    calibration: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CALIBRATION_SCHEMA_VERSION, dtype=np.int64),
        "n_classes": np.asarray(frozen.n_classes, dtype=np.int64),
    }
    for class_index, (x_values, y_values) in enumerate(
        zip(frozen.calibration_x, frozen.calibration_y, strict=True)
    ):
        calibration[f"x_{class_index}"] = x_values
        calibration[f"y_{class_index}"] = y_values
    np.savez(calibration_path, **calibration)
    return frozen


def load_frozen_risk(
    forest_path: str | Path,
    calibration_path: str | Path,
) -> VectorizedHGBRiskModel:
    """Load frozen risk parameters without importing the sklearn checkpoint."""

    frozen = VectorizedHGBRiskModel.__new__(VectorizedHGBRiskModel)
    with np.load(forest_path, allow_pickle=False) as data:
        if int(data["schema_version"]) != FOREST_SCHEMA_VERSION:
            raise RuntimeError("unsupported frozen HGB schema")
        frozen.n_classes = int(data["n_classes"])
        frozen.trees_per_iteration = int(data["trees_per_iteration"])
        frozen.iterations = int(data["iterations"])
        frozen.tree_count = int(data["tree_count"])
        frozen.max_depth = int(data["max_depth"])
        frozen.feature_idx = _locked_array(data["feature_idx"], np.int64)
        frozen.threshold = _locked_array(data["threshold"], np.float64)
        frozen.missing_left = _locked_array(data["missing_left"], bool)
        frozen.left = _locked_array(data["left"], np.int64)
        frozen.right = _locked_array(data["right"], np.int64)
        frozen.is_leaf = _locked_array(data["is_leaf"], bool)
        frozen.value = _locked_array(data["value"], np.float64)
        frozen.baseline = _locked_array(data["baseline"], np.float64)
    with np.load(calibration_path, allow_pickle=False) as data:
        if int(data["schema_version"]) != CALIBRATION_SCHEMA_VERSION:
            raise RuntimeError("unsupported isotonic calibration schema")
        if int(data["n_classes"]) != frozen.n_classes:
            raise RuntimeError("forest/calibration class count mismatch")
        frozen.calibration_x = [
            _locked_array(data[f"x_{index}"], np.float64)
            for index in range(frozen.n_classes)
        ]
        frozen.calibration_y = [
            _locked_array(data[f"y_{index}"], np.float64)
            for index in range(frozen.n_classes)
        ]
    if frozen.n_classes != 4 or frozen.trees_per_iteration != 4:
        raise RuntimeError("locked release requires the four-action HGB model")
    if frozen.tree_count != frozen.iterations * frozen.trees_per_iteration:
        raise RuntimeError("frozen HGB tree dimensions are inconsistent")
    frozen._tree_ids = np.arange(frozen.tree_count, dtype=np.int64)
    frozen._single_indices = np.zeros(frozen.tree_count, dtype=np.int64)
    return frozen


def export_normalization(
    ensemble: TorchSeedEnsemble,
    path: str | Path,
) -> dict[str, Any]:
    lower = np.asarray(ensemble.kinematics.limits.lower, dtype=np.float64)
    upper = np.asarray(ensemble.kinematics.limits.upper, dtype=np.float64)
    center = (lower + upper) / 2.0
    half_span = (upper - lower) / 2.0
    np.savez(
        path,
        schema_version=np.asarray(NORMALIZATION_SCHEMA_VERSION, dtype=np.int64),
        lower=lower,
        upper=upper,
        center=center,
        half_span=half_span,
        joint_names=np.asarray(ensemble.kinematics.joint_names, dtype=np.str_),
        use_history=np.asarray(ensemble.config.use_history, dtype=bool),
        input_dtype=np.asarray("float32", dtype=np.str_),
        input_size=np.asarray(ensemble.kinematics.nq + 9, dtype=np.int64),
    )
    return {
        "seed_training_config": asdict(ensemble.config),
        "joint_names": list(ensemble.kinematics.joint_names),
        "input_size": ensemble.kinematics.nq + 9,
        "input_dtype": "float32",
        "output_layout": "batch,members,joints",
    }


def load_locked_seed_engine(
    *,
    kinematics: object,
    torchscript_path: str | Path,
    normalization_path: str | Path,
    runtime_spec_path: str | Path,
    device: str | torch.device,
) -> OptimizedSeedEngine:
    """Load the persisted TorchScript graph and normalization from disk.

    No trace, script, freeze, compilation, or source-checkpoint loading occurs
    on this deployment path.
    """

    spec = json.loads(Path(runtime_spec_path).read_text(encoding="utf-8"))
    config_payload = dict(spec["seed_training_config"])
    config_payload["hidden_sizes"] = tuple(config_payload["hidden_sizes"])
    seed_config = SeedTrainingConfig(**config_payload)
    with np.load(normalization_path, allow_pickle=False) as data:
        if int(data["schema_version"]) != NORMALIZATION_SCHEMA_VERSION:
            raise RuntimeError("unsupported normalization schema")
        joint_names = tuple(str(value) for value in data["joint_names"].tolist())
        if joint_names != tuple(kinematics.joint_names):
            raise RuntimeError("normalization joint names do not match robot")
        lower = np.asarray(data["lower"], dtype=np.float64)
        upper = np.asarray(data["upper"], dtype=np.float64)
        if not np.array_equal(lower, np.asarray(kinematics.limits.lower, dtype=np.float64)):
            raise RuntimeError("normalization lower limits do not match robot")
        if not np.array_equal(upper, np.asarray(kinematics.limits.upper, dtype=np.float64)):
            raise RuntimeError("normalization upper limits do not match robot")
        center = np.asarray(data["center"], dtype=np.float64).copy()
        half_span = np.asarray(data["half_span"], dtype=np.float64).copy()
        use_history = bool(data["use_history"])
        input_dtype = str(data["input_dtype"])
        input_size = int(data["input_size"])
    if use_history != seed_config.use_history or input_dtype != "float32":
        raise RuntimeError("normalization metadata differs from locked seed configuration")
    if input_size != int(kinematics.nq) + 9:
        raise RuntimeError("normalization input size does not match robot")
    target_device = torch.device(device)
    module = torch.jit.load(str(torchscript_path), map_location=target_device).eval()
    descriptor = SimpleNamespace(kinematics=kinematics, config=seed_config)
    engine = OptimizedSeedEngine(
        descriptor,  # type: ignore[arg-type]
        module,
        name="torchscript_exact",
        device=target_device,
    )
    engine._center = center
    engine._half_span = half_span
    return engine

