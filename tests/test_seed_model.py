from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from confik.data.generate import generate_smooth_transitions
from confik.kinematics.urdf import URDFKinematics
from confik.models.seed import SeedTrainingConfig, TorchSeedEnsemble
from confik.types import IKQuery

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_seed_ensemble_trains_predicts_and_round_trips(tmp_path) -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    dataset = generate_smooth_transitions(model, trajectories=3, steps_per_trajectory=5, seed=4)
    config = SeedTrainingConfig(
        members=2,
        hidden_sizes=(16,),
        epochs=1,
        batch_size=8,
        fk_position_weight=0.1,
        fk_orientation_weight=0.1,
    )
    ensemble = TorchSeedEnsemble(model, config, device="cpu").fit(dataset)
    query = IKQuery(model.forward(dataset.target_q[0]), dataset.previous_q[0])
    candidates = ensemble.candidates(query)
    assert candidates.joints.shape[1] == model.nq
    assert candidates.joints.shape[0] >= 1
    artifact = tmp_path / "seed.pt"
    ensemble.save(artifact)
    loaded = TorchSeedEnsemble.load(artifact, model, device="cpu")
    np.testing.assert_allclose(loaded.predict_deltas(query), ensemble.predict_deltas(query))
