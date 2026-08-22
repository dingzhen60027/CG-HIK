from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from confik.kinematics.pinocchio_adapter import PinocchioKinematics
from confik.kinematics.urdf import URDFKinematics

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_numpy_urdf_backend_matches_pinocchio() -> None:
    numpy_model = URDFKinematics.from_file(ASSET, end_link="tool")
    pin_model = PinocchioKinematics(ASSET, "tool")
    q = np.array([0.4, -0.7])
    numpy_pose = numpy_model.forward(q)
    pin_pose = pin_model.forward(q)
    np.testing.assert_allclose(numpy_pose.position, pin_pose.position, atol=1e-10)
    np.testing.assert_allclose(numpy_pose.rotation, pin_pose.rotation, atol=1e-10)
    np.testing.assert_allclose(numpy_model.jacobian(q), pin_model.jacobian(q), atol=1e-10)
