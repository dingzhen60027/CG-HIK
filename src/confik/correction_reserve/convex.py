"""Small native Clarabel SOCPs; no CVXPY model generation or nonlinear multistart."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import sparse
import clarabel

from .geometry import residual_linearization, correction_map


@dataclass(frozen=True)
class OptimizerConfig:
    outer_iterations: int = 3
    trust_step: float = 0.5
    pose_interior: float = 1e-5
    native_max_iterations: int = 40
    native_time_limit_ms: float = 2.0
    total_soft_limit_ms: float = 18.0
    reserve_tie_relative: float = 0.001
    reserve_tie_absolute: float = 1e-8
    current_residual_weight: float = 1.0
    rank_rtol: float = 1e-10
    sigma_difference_step: float = 1e-4


class ConeProgram:
    def __init__(self, n):
        self.n = n
        self.rows, self.rhs, self.cones = [], [], []

    def linear(self, A, b):
        A = np.atleast_2d(A)
        b = np.atleast_1d(b)
        self.rows.append(A); self.rhs.append(b)
        self.cones.append(clarabel.NonnegativeConeT(len(b)))

    def norm(self, F, offset, radius):
        A = np.zeros((len(offset)+1, self.n))
        A[1:] = -F
        self.rows.append(A); self.rhs.append(np.r_[radius, offset])
        self.cones.append(clarabel.SecondOrderConeT(len(offset)+1))

    def run(self, P, c, cfg, remaining_seconds):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = cfg.native_max_iterations
        settings.time_limit = max(1e-6, min(cfg.native_time_limit_ms/1000, remaining_seconds))
        settings.max_threads = 1
        settings.tol_feas = 1e-9
        settings.tol_gap_abs = 1e-8
        settings.tol_gap_rel = 1e-8
        solver = clarabel.DefaultSolver(
            sparse.triu(sparse.csc_matrix(P)).tocsc(), np.asarray(c),
            sparse.csc_matrix(np.vstack(self.rows)), np.concatenate(self.rhs),
            self.cones, settings)
        result = solver.solve()
        x = np.asarray(result.x)
        return x, dict(status=str(result.status), iterations=int(result.iterations),
                       native_seconds=float(result.solve_time),
                       primal_residual=float(result.r_prim), dual_residual=float(result.r_dual))


def sigma_value(kin, target, q, scale, step):
    J = kin.jacobian(q)/scale[:, None]*step
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def subproblem(kin, verifier, target, predicted, previous, q0, z0, lower, upper,
               mode, cfg, deadline):
    single = mode == "single"
    n = kin.nq
    size = n+1 if single else 2*n+1
    scale = np.array([verifier.config.position_tolerance]*3 +
                     [verifier.config.orientation_tolerance]*3)
    step = kin.limits.velocity*0.02+verifier.config.velocity_tolerance
    # All callers use the frozen 20 ms interval. Runtime checks this explicitly.
    u0, v0 = (q0-previous)/step, (z0-previous)/step
    program = ConeProgram(size)
    I = np.eye(size)
    ulo, uhi = (lower-previous)/step, (upper-previous)/step
    for i in range(n):
        program.linear(I[i], min(uhi[i], u0[i]+cfg.trust_step))
        program.linear(-I[i], -max(ulo[i], u0[i]-cfg.trust_step))
    e, J, _ = residual_linearization(kin, target, q0, scale)
    F = np.zeros((6, size)); F[:, :n] = J*step
    off = e-F[:, :n] @ u0
    for s in [slice(0, 3), slice(3, 6)]:
        program.norm(F[s], off[s], 1-cfg.pose_interior)
    motion = np.zeros((n if single else 2*n, size))
    motion[:n, :n] = np.eye(n)
    if not single:
        for i in range(n):
            lo = max((kin.limits.lower[i]-previous[i])/step[i], v0[i]-cfg.trust_step)
            hi = min((kin.limits.upper[i]-previous[i])/step[i], v0[i]+cfg.trust_step)
            program.linear(I[n+i], hi); program.linear(-I[n+i], -lo)
            program.linear(I[n+i]-I[i], 1); program.linear(I[i]-I[n+i], 1)
        ep, Jp, _ = residual_linearization(kin, predicted, z0, scale)
        Fp = np.zeros((6, size)); Fp[:, n:2*n] = Jp*step
        offp = ep-Fp[:, n:2*n] @ v0
        for s in [slice(0, 3), slice(3, 6)]:
            program.norm(Fp[s], offp[s], 1-cfg.pose_interior)
        motion[n:, :n] = -np.eye(n); motion[n:, n:2*n] = np.eye(n)
    program.linear(-I[-1], 0)
    map_info = None
    if mode in ("reserve", "single"):
        node = q0 if single else z0
        aim = target if single else predicted
        mapping = correction_map(kin, aim, node, scale, step, cfg.rank_rtol)
        map_info = dict(rank=mapping.rank,
                        compensation_error=mapping.compensation_error if np.isfinite(mapping.compensation_error) else None)
        if not mapping.full_rank:
            return None, [dict(status="rank_deficient_no_reserve", **map_info)]
        beta = np.linalg.norm(mapping.normalized_joint_map, axis=1)
        for i in range(n):
            at = i if single else n+i
            move = I[at] if single else I[at]-I[i]
            reserve = beta[i]*I[-1]
            program.linear(move+reserve, 1); program.linear(-move+reserve, 1)
            program.linear(-I[at]+reserve, (previous[i]-kin.limits.lower[i])/step[i])
            program.linear(I[at]+reserve, (kin.limits.upper[i]-previous[i])/step[i])
    elif mode == "sigma":
        sigma0 = sigma_value(kin, predicted, z0, scale, step)
        grad = np.zeros(n)
        for i in range(n):
            delta = np.zeros(n); delta[i] = cfg.sigma_difference_step*step[i]
            grad[i] = (sigma_value(kin,predicted,z0+delta,scale,step)-
                       sigma_value(kin,predicted,z0-delta,scale,step))/(2*cfg.sigma_difference_step)
        row = I[-1].copy(); row[n:2*n] -= grad
        program.linear(row, sigma0-grad @ v0)
    elif mode == "predictive":
        program.linear(I[-1], 0)
    else:
        raise ValueError(mode)
    # Same secondary objective for every mode; gamma has no invented weighted sum.
    P = 2*(motion.T @ motion+cfg.current_residual_weight*F.T @ F)
    c = 2*cfg.current_residual_weight*F.T @ off
    stages = []
    if perf_counter() >= deadline:
        return None, [dict(status="outer_soft_limit")]
    if mode != "predictive":
        objective = np.zeros(size); objective[-1] = -1
        x, status = program.run(np.zeros((size,size)), objective, cfg, deadline-perf_counter())
        stages.append(dict(stage="primary", **status))
        if status["status"] not in ("Solved", "AlmostSolved") or not np.isfinite(x).all():
            return None, stages
        gamma = max(0., float(x[-1]))
        program.linear(-I[-1], -(gamma-max(cfg.reserve_tie_absolute,cfg.reserve_tie_relative*gamma)))
        if perf_counter() >= deadline:
            return x, stages
        first = x
    else:
        first = None
    x, status = program.run(P, c, cfg, deadline-perf_counter())
    stages.append(dict(stage="secondary", **status))
    if status["status"] not in ("Solved", "AlmostSolved") or not np.isfinite(x).all():
        return first, stages
    return x, stages
