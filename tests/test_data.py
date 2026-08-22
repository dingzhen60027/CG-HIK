from pathlib import Path

import numpy as np

from confik.data.generate import generate_smooth_transitions
from confik.kinematics.urdf import URDFKinematics

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_transition_generation_is_deterministic_and_velocity_limited(tmp_path) -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    first = generate_smooth_transitions(model, trajectories=3, steps_per_trajectory=5, seed=9)
    second = generate_smooth_transitions(model, trajectories=3, steps_per_trajectory=5, seed=9)
    assert len(first) == 15
    np.testing.assert_allclose(first.target_q, second.target_q)
    assert np.all(np.abs(first.target_q - first.previous_q) <= model.limits.velocity * 0.02 + 1e-12)
    output = tmp_path / "transitions.npz"
    first.save(output)
    loaded = type(first).load(output)
    np.testing.assert_allclose(loaded.target_position, first.target_position)

