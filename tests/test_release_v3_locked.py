from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from confik.config import load_config, load_robot
from confik.data.datasets import RiskDataset, TransitionDataset
from confik.latency_pilot_v3.optimized_inference import ExactSingleCallSeedEnsemble
from confik.models.risk import RiskModel
from confik.models.seed import TorchSeedEnsemble, encode_seed_inputs
from confik.release_v3_locked.artifacts import (
    export_frozen_risk,
    export_normalization,
    load_frozen_risk,
    load_locked_seed_engine,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def test_release_config_keeps_locked_backend_and_tolerances() -> None:
    release = yaml.safe_load((WORKSPACE / "configs/release_v3_locked.yaml").read_text())
    pilot = yaml.safe_load((WORKSPACE / "configs/latency_pilot_v3.yaml").read_text())
    assert release["backend"] == "torchscript_exact"
    assert release["robots"] == ["panda", "ur5e"]
    assert release["training_seeds"] == [17, 29, 43]
    assert release["runtime"]["intra_op_threads"] == 8
    assert release["runtime"]["inter_op_threads"] == 1
    assert release["equivalence"] == pilot["equivalence"]


def test_frozen_hgb_and_isotonic_round_trip(tmp_path: Path) -> None:
    root = WORKSPACE / "outputs/paper_v2_seed17/panda"
    if not root.exists():
        pytest.skip("paper_v2 seed17 artifacts are not available")
    model = RiskModel.load(root / "models/risk_model.joblib")
    forest = tmp_path / "forest.npz"
    calibration = tmp_path / "calibration.npz"
    export_frozen_risk(model, forest, calibration)
    loaded = load_frozen_risk(forest, calibration)
    features = RiskDataset.load(root / "datasets/risk_validation.npz").features[:500]
    reference = model.predict_proba(features)
    candidate = loaded.predict_proba(features)
    np.testing.assert_array_equal(candidate, reference)


def test_disk_loaded_torchscript_seed_round_trip(tmp_path: Path) -> None:
    root = WORKSPACE / "outputs/paper_v2_seed17/ur5e"
    if not root.exists():
        pytest.skip("paper_v2 seed17 artifacts are not available")
    config = load_config(WORKSPACE / "configs/paper_v2.yaml")
    kinematics = load_robot(config, "ur5e")
    ensemble = TorchSeedEnsemble.load(
        root / "models/seed_ensemble.pt", kinematics, device="cpu"
    )
    source = ExactSingleCallSeedEnsemble(ensemble).eval()
    example = torch.empty((1, kinematics.nq + 9), dtype=torch.float32)
    traced = torch.jit.trace(source, example, strict=True).eval()
    torchscript = tmp_path / "seed.ts"
    torch.jit.save(traced, str(torchscript))
    normalization = tmp_path / "normalization.npz"
    runtime = export_normalization(ensemble, normalization)
    runtime.update({"robot": "ur5e", "training_seed": 17})
    runtime_path = tmp_path / "runtime_spec.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    loaded = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=torchscript,
        normalization_path=normalization,
        runtime_spec_path=runtime_path,
        device="cpu",
    )
    validation = TransitionDataset.load(root / "datasets/seed_validation.npz")
    features = np.ascontiguousarray(
        encode_seed_inputs(
            kinematics,
            validation.previous_q[:64],
            validation.target_position[:64],
            validation.target_rotation[:64],
            use_history=ensemble.config.use_history,
        ).astype(np.float32)
    )
    with torch.inference_mode():
        reference = source(torch.from_numpy(features)).numpy()
        candidate = loaded.module(torch.from_numpy(features)).numpy()
    assert np.array_equal(reference, candidate)
