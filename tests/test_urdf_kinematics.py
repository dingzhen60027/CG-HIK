from pathlib import Path

import numpy as np
import pytest

from confik.geometry import rotation_log
from confik.kinematics.urdf import URDFKinematics

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_toy_forward_kinematics() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    assert model.nq == 2
    np.testing.assert_allclose(model.forward(np.zeros(2)).position, [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        model.forward(np.array([np.pi / 2, 0.0])).position,
        [0.0, 2.0, 0.0],
        atol=1e-9,
    )


def test_geometric_jacobian_matches_finite_difference() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    q = np.array([0.4, -0.7])
    jacobian = model.jacobian(q)
    epsilon = 1e-7
    base = model.forward(q)
    numerical = np.zeros_like(jacobian)
    for index in range(model.nq):
        shifted = q.copy()
        shifted[index] += epsilon
        pose = model.forward(shifted)
        numerical[:3, index] = (pose.position - base.position) / epsilon
        numerical[3:, index] = rotation_log(pose.rotation @ base.rotation.T) / epsilon
    np.testing.assert_allclose(jacobian, numerical, atol=2e-6)


def test_bounded_revolute_difference_is_not_wrapped() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    difference = model.difference(np.array([3.1, 0.0]), np.array([-3.1, 0.0]))
    assert difference[0] == pytest.approx(6.2)
