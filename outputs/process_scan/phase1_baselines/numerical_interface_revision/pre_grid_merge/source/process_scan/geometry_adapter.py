"""Geometric adapter to the frozen local GN mathematics, with NO time argument.

This is not an online-IK call with a fictitious dt. Bounds and step scales have
units of radians, and neither a velocity limit nor an online IKQuery is created.
The pose/finite/joint checks use the same public geometry and URDF limits as the
online verifier; physical rate validation belongs to the timed path.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import sys
import numpy as np

# Preserve the previously measured NumPy path; do not upgrade Numba/coverage.
try:
    import numba  # noqa: F401
except (ImportError, AttributeError):
    sys.modules['numba'] = None

from ..bounded_gn import box_qp
from ..task_balance_gn import CompletionSettings, completion_dual_direction, task_value
from ..correction_reserve.geometry import residual_linearization
from ..correction_reserve.native_geometry import NativeGeometry
from ..geometry import pose_distance
from ..kinematics.urdf import URDFKinematics


@dataclass(frozen=True)
class GeometrySettings:
    max_iterations: int = 30
    max_backtracking: int = 8
    damping: float = .01
    kappa: float = 1.
    budget_s: float = 1.


class GeometryAdapter:
    def __init__(self, urdf, base_link, end_link, settings=None):
        self.public = URDFKinematics.from_file(urdf, base_link=base_link, end_link=end_link)
        self.native = NativeGeometry(self.public, urdf)
        self.settings = settings or GeometrySettings()
        self.local = CompletionSettings(forcing=.25)

    def verify(self, q, target_pose, physical_bounds, continuity_box, pose_tolerances):
        q = np.asarray(q, float)
        finite = q.shape == (self.public.nq,) and np.isfinite(q).all()
        if not finite:
            return dict(accepted=False, reason='non_finite_or_wrong_shape', position_error=None, orientation_error=None)
        lo, hi = np.maximum(physical_bounds[0], continuity_box[0]), np.minimum(physical_bounds[1], continuity_box[1])
        ep, er = pose_distance(target_pose, self.public.forward(q))
        reasons = []
        if ep > pose_tolerances[0]: reasons.append('position_tolerance')
        if er > pose_tolerances[1]: reasons.append('orientation_tolerance')
        if np.any(q < lo-1e-12) or np.any(q > hi+1e-12): reasons.append('geometric_joint_box')
        return dict(accepted=not reasons, reason='+'.join(reasons) or 'accepted',
                    position_error=float(ep), orientation_error=float(er))

    def solve(self, target_pose, seed_q, physical_bounds, continuity_box, step_scale,
              pose_tolerances, method='tb'):
        if method not in ('tb', 'gn'): raise ValueError(method)
        start = perf_counter_ns(); cfg = self.settings
        lower, upper = (np.asarray(x, float) for x in physical_bounds)
        lo = np.maximum(lower, continuity_box[0]); hi = np.minimum(upper, continuity_box[1])
        S = np.broadcast_to(np.asarray(step_scale, float), lower.shape)
        if np.any(lo > hi) or np.any(S <= 0) or not np.isfinite([lo, hi, S]).all():
            raise ValueError('Invalid untimed geometric box/scale')
        q = np.clip(np.asarray(seed_q, float), lo, hi)
        scale = np.repeat(pose_tolerances, 3)
        span = upper-lower; w = S/span; lam = cfg.damping
        deadline = start + int(cfg.budget_s*1e9)
        trace = []; calls = 0
        def evaluate(x):
            nonlocal calls
            calls += 1
            e, J, _ = residual_linearization(self.native, target_pose, x, scale)
            return e, J
        def verified(x):
            return self.verify(x, target_pose, physical_bounds, continuity_box, pose_tolerances)['accepted']
        objective = task_value if method == 'tb' else lambda e: .5*float(e@e)
        bestq = q.copy(); best = np.inf; status = 'iteration_limit'
        for iteration in range(cfg.max_iterations):
            if perf_counter_ns() >= deadline: status = 'geometric_budget'; break
            e, J = evaluate(q); f = objective(e)
            if f < best: best, bestq = f, q.copy()
            if verified(q): bestq=q.copy(); status='accepted'; break
            G = J*S; posture = (q-.5*(lower+upper))/span
            H = G.T@G + lam*np.eye(len(q)) + cfg.kappa*np.diag(w*w)
            g = G.T@e + cfg.kappa*w*posture
            dl, du = np.maximum((lo-q)/S, -1.), np.minimum((hi-q)/S, 1.)
            d, nit = box_qp(H, g, dl, du)
            if verified(np.clip(q+S*d, lo, hi)):
                bestq=np.clip(q+S*d, lo, hi); status='first_qp_accepted'; break
            info = dict(reason='original_gn', dual_updates=1)
            if method == 'tb':
                d, info = completion_dual_direction(e,G,dl,du,lam,cfg.kappa,posture,w,
                    box_qp,(H,g,d,nit),lambda v: verified(np.clip(q+S*v,lo,hi)),self.local,deadline)
            trace.append(dict(iteration=iteration, objective=f, **info))
            if d is None or not np.isfinite(d).all(): status='no_direction'; break
            adopted=False
            for k in range(cfg.max_backtracking):
                trial=np.clip(q+2.**(-k)*S*d,lo,hi); en,_=evaluate(trial)
                if verified(trial) or objective(en) < f-1e-12:
                    q=trial; adopted=True; lam=max(cfg.damping/100, lam*.5); break
            if not adopted:
                lam*=10
                if lam > 1e8: status='no_descent'; break
        en,_=evaluate(q)
        if objective(en) < best or verified(q): bestq=q.copy()
        verdict=self.verify(bestq,target_pose,physical_bounds,continuity_box,pose_tolerances)
        return dict(q=bestq, **verdict, method=method, status=status, trace=trace,
                    residual_evaluations=calls, elapsed_ns=perf_counter_ns()-start,
                    physical_velocity_checked=False, nearest_projection_claim=False)
