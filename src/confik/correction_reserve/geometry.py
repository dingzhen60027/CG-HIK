"""SO(3)-consistent local target correction, in public world-frame axes."""
from dataclasses import dataclass
import numpy as np

from ..geometry import pose_error, rotation_log, axis_angle_matrix
from ..types import Pose
from ..continuation_mechanism.nonlinear_reference_math import so3_right_jacobian_inverse


def predict_target(current, last):
    if last is None:
        return Pose(current.position.copy(), current.rotation.copy())
    rotation_increment = rotation_log(last.rotation.T @ current.rotation)
    rotation = current.rotation @ axis_angle_matrix(rotation_increment, np.linalg.norm(rotation_increment))
    return Pose(2*current.position-last.position, rotation)


def task_scale(verifier):
    return np.array([verifier.config.position_tolerance]*3 +
                    [verifier.config.orientation_tolerance]*3)


def residual_linearization(kin, target, q, scale):
    residual = pose_error(target, kin.forward(q))
    geometric = kin.jacobian(q)
    derivative = np.vstack((-geometric[:3],
        -so3_right_jacobian_inverse(residual[3:]) @ geometric[3:]))
    return residual/scale, derivative/scale[:, None], geometric


@dataclass
class CorrectionMap:
    matrix: object
    singular_values: np.ndarray
    normalized_joint_map: object
    rank: int
    compensation_error: float

    @property
    def full_rank(self):
        return self.rank == 6 and self.matrix is not None


def correction_map(kin, target, z, scale, step, rank_rtol=1e-10):
    """A B = C, not A B = I when a nonzero orientation residual is present.

    Target perturbation: p'=p+D_p dy_p, R'=Exp(D_R dy_R) R.
    e_R=Log(R_target R_actual^T). Its target derivative is J_l^-1(e_R);
    its q derivative is -J_r^-1(e_R) J_world_angular.
    S=diag(step); B=S (A S)^right_inverse C minimizes scaled joint correction.
    A failed rank test returns no positive correction guarantee; no SVD clipping.
    """
    e, derivative, _ = residual_linearization(kin, target, z, scale)
    A = -derivative
    C = np.eye(6)
    C[3:, 3:] = so3_right_jacobian_inverse(-e[3:]*scale[3:])
    G = A*step
    U, singular, Vt = np.linalg.svd(G, full_matrices=False)
    threshold = max(1e-12, rank_rtol*singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    if rank < 6:
        return CorrectionMap(None, singular, None, rank, float("inf"))
    normalized = (Vt.T / singular) @ U.T @ C
    B = step[:, None]*normalized
    error = float(np.linalg.norm(A @ B-C, ord=np.inf))
    if error > 1e-7:
        return CorrectionMap(None, singular, None, rank, error)
    return CorrectionMap(B, singular, normalized, rank, error)


def reserve_value(kin, target, q, z, scale, step, rank_rtol=1e-10):
    mapping = correction_map(kin, target, z, scale, step, rank_rtol)
    slack = np.minimum.reduce([step-np.abs(z-q), z-kin.limits.lower, kin.limits.upper-z])
    if not mapping.full_rank:
        return 0., mapping, slack
    row_norm = np.linalg.norm(mapping.matrix, axis=1)
    quotients = np.divide(np.maximum(0., slack), row_norm,
                          out=np.full(len(step), np.inf), where=row_norm>1e-15)
    return float(max(0., np.min(quotients))), mapping, slack


def perturb_target(target, direction, amplitude, scale):
    delta = np.asarray(direction)*float(amplitude)*scale
    return Pose(target.position+delta[:3],
                axis_angle_matrix(delta[3:], np.linalg.norm(delta[3:])) @ target.rotation)
