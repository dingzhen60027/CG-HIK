import numpy as np

from confik.geometry import axis_angle_matrix, rotation_log, rpy_matrix, wrap_to_pi


def test_rotation_log_round_trip() -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.7
    recovered = rotation_log(axis_angle_matrix(axis, angle))
    np.testing.assert_allclose(recovered, axis * angle, atol=1e-9)


def test_rpy_identity_and_angle_wrapping() -> None:
    np.testing.assert_allclose(rpy_matrix([0, 0, 0]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(wrap_to_pi([0, 3 * np.pi, -3 * np.pi]), [0, -np.pi, -np.pi])

