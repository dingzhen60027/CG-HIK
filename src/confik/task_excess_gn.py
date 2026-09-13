"""Current-frame task-set distance, using the frozen NumPy box-QP.

Projection derivatives, generalized GN, and L-BFGS-B are established tools.
This module implements the supplied development hypothesis, not a fallback
policy or a claim of global convergence. Only the original verifier accepts.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from scipy.optimize import minimize
from .bounded_gn import box_qp
from .correction_reserve.native_geometry import NativeGeometry
from .correction_reserve.geometry import residual_linearization, task_scale
from .types import IKQuery, Pose


@dataclass(frozen=True)
class Settings:
    damping: float = .01
    maximum_iterations: int = 30
    backtracking_steps: int = 8
    normalized_margin: float = 1e-6
    deadline_ms: float = 20.


def projected_task_model(e, radius=1., isotropic=False):
    """Squared distance, gradient and residual-space Hessian away from boundary.

    At a block boundary select the inside (zero) generalized derivative.
    The isotropic option is a diagnostic only: it omits radial curvature,
    without changing the true objective or gradient.
    """
    e = np.asarray(e, dtype=float)
    if e.shape != (6,) or not np.isfinite(e).all():
        raise ValueError('Expected finite six-dimensional normalized residual')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid radius')
    v = np.zeros(6)
    W = np.zeros((6, 6))
    for start in (0, 3):
        b = e[start:start+3]
        r = np.linalg.norm(b)
        if r > radius:
            u = b/r
            alpha = 1-radius/r
            v[start:start+3] = (r-radius)*u
            W[start:start+3, start:start+3] = alpha*np.eye(3)
            if not isotropic:
                W[start:start+3, start:start+3] += (radius/r)*np.outer(u, u)
    return .5*float(v@v), v, W


def dynamic_interval(previous, dt, limits, velocity_tolerance):
    """One interval for the entire frame, never recentered on an iterate."""
    S = limits.velocity*dt+velocity_tolerance
    lo = np.maximum(limits.lower+1e-12, previous-S*(1-1e-12))
    hi = np.minimum(limits.upper-1e-12, previous+S*(1-1e-12))
    if not np.isfinite([*lo, *hi, *S]).all() or dt <= 0 or np.any(lo > hi):
        raise ValueError('Invalid or empty current-frame interval')
    return S, lo, hi


class _Stop(Exception):
    pass


class TaskExcessGN:
    """A single bounded iterator or standalone same-merit standard reference.

    ``isotropic`` and ``point`` are restricted to same-input diagnostics.
    No witness, failed endpoint, future target or reference path is an input.
    The 20 ms check is soft; returned late calls remain in outer timing.
    """
    def __init__(self, kin, verifier, urdf, mode='task', settings=None, trace=False):
        if mode not in ('task', 'lbfgsb', 'isotropic', 'point'):
            raise ValueError(mode)
        self.kin, self.verifier = kin, verifier
        self.native = NativeGeometry(kin, urdf)
        self.scale = task_scale(verifier)
        self.mode, self.settings, self.trace = mode, settings or Settings(), trace
        self.method = {'task':'task_excess_gn', 'lbfgsb':'task_excess_lbfgsb',
                       'isotropic':'task_excess_isotropic', 'point':'point_loop_control'}[mode]

    def reset(self, q):
        pass

    def close(self):
        pass

    def solve(self, position, rotation, previous, dt=.02):
        start = perf_counter_ns()
        cfg = self.settings
        query = IKQuery(Pose(np.asarray(position, float), np.asarray(rotation, float)),
                        np.asarray(previous, float), dt)
        p = query.previous_q
        S, lo, hi = dynamic_interval(p, dt, self.kin.limits,
                                     self.verifier.config.velocity_tolerance)
        q = np.clip(p, lo, hi)
        radius = 1-cfg.normalized_margin
        evaluations = verifications = iterations = updates = backtracks = 0
        status = 'iteration_limit'
        traces = []
        best_q, best_f = q.copy(), np.inf
        native_status = None

        def expired():
            return perf_counter_ns()-start >= cfg.deadline_ms*1e6

        def evaluate(x):
            nonlocal evaluations
            evaluations += 1
            e, J, _ = residual_linearization(self.native, query.target, x, self.scale)
            return e, J

        def verify(x):
            nonlocal verifications
            verifications += 1
            return self.verifier.check(x, query)

        def model(e):
            if self.mode == 'point':
                return .5*float(e@e), e, np.eye(6)
            return projected_task_model(e, radius, self.mode == 'isotropic')

        def remember(x, f):
            nonlocal best_q, best_f
            if f < best_f:
                best_q, best_f = x.copy(), float(f)

        if self.mode == 'lbfgsb':
            # Normalized joint variables give the same task/joint scaling as GN.
            # Task verification, not f convergence, is the acceptance condition.
            # No relative-f early stop: Phi is squared distance and is tiny near
            # the public boundary. gtol is a normal numerical gradient test;
            # neither it nor an optimizer success code can accept a command.
            def fun(y):
                nonlocal q, status
                if expired():
                    status = 'deadline'
                    raise _Stop
                x = np.clip(p+S*y, lo, hi)
                e, J = evaluate(x)
                f, v, _ = model(e)
                remember(x, f)
                verdict = verify(x)
                if self.trace:
                    traces.append(dict(evaluation=evaluations, q=x.tolist(), objective=f,
                                       e=e.tolist(), accepted=bool(verdict.accepted)))
                if verdict.accepted:
                    q, status = x.copy(), 'task_admissible'
                    raise _Stop
                return f, (J*S).T@v

            try:
                result = minimize(fun, (q-p)/S, method='L-BFGS-B', jac=True,
                    bounds=list(zip((lo-p)/S, (hi-p)/S)),
                    options=dict(maxiter=cfg.maximum_iterations, maxfun=240,
                                 maxls=20, gtol=1e-8, ftol=0.))
                iterations = int(result.nit)
                native_status = dict(code=int(result.status), success=bool(result.success),
                                     message=str(result.message))
                status = 'optimizer_stop'
            except _Stop:
                pass
            if status != 'task_admissible':
                q = best_q.copy()
        else:
            lam = cfg.damping
            for it in range(cfg.maximum_iterations):
                if expired():
                    status = 'deadline'
                    break
                iterations += 1
                e, J = evaluate(q)
                f, v, W = model(e)
                remember(q, f)
                verdict = verify(q)
                row = None
                if self.trace:
                    row = dict(iterate=it, q=q.tolist(), e=e.tolist(), objective=f,
                               accepted=bool(verdict.accepted), damping=lam)
                    traces.append(row)
                if verdict.accepted:
                    status = 'task_admissible'
                    break
                G = J*S
                H, g = G.T@W@G+lam*np.eye(len(q)), G.T@v
                dl, du = np.maximum((lo-q)/S, -1.), np.minimum((hi-q)/S, 1.)
                d, count = box_qp(H, g, dl, du)
                updates += int(count)
                gd, dHd = float(g@d), float(d@H@d)
                violation = float(max(0., np.max(dl-d), np.max(d-du)))
                if row is not None:
                    row.update(H=H.tolist(), g=g.tolist(), step_lower=dl.tolist(),
                               step_upper=du.tolist(), direction=d.tolist(), gd=gd,
                               dHd=dHd, descent_inequality_residual=gd+dHd,
                               bound_violation=violation, box_qp_updates=int(count))
                if not np.isfinite(d).all() or violation > 1e-10:
                    status = 'invalid_qp_direction'
                    break
                if np.linalg.norm(d) < 1e-12:
                    status = 'stationary'
                    break
                adopted = False
                for k in range(cfg.backtracking_steps):
                    if expired():
                        status = 'deadline'
                        break
                    backtracks += 1
                    alpha = 2.**(-k)
                    candidate = np.clip(q+alpha*S*d, lo, hi)
                    en, _ = evaluate(candidate)
                    fn = model(en)[0]
                    candidate_verdict = verify(candidate)
                    # A legal command can terminate, otherwise true nonlinear
                    # Armijo descent is required. Never trust a QP return code.
                    if candidate_verdict.accepted or (gd < 0 and fn <= f+1e-4*alpha*gd):
                        q = candidate
                        remember(q, fn)
                        lam = max(cfg.damping/100., lam*.5)
                        adopted = True
                        if row is not None:
                            row.update(alpha=alpha, new_objective=fn,
                                       new_q=q.tolist(), new_e=en.tolist())
                        if candidate_verdict.accepted:
                            status = 'task_admissible'
                        break
                if status in ('task_admissible', 'deadline'):
                    break
                if not adopted:
                    lam *= 10.
                    if lam > 1e8:
                        status = 'no_descent'
                        break
            if status != 'task_admissible':
                q = best_q.copy()

        # One independent final check in all cases, including timeouts.
        verdict = verify(q)
        if verdict.accepted:
            status = 'task_admissible'
        norms = np.array([verdict.position_error/self.scale[0],
                          verdict.orientation_error/self.scale[3]])
        phi = .5*float(np.maximum(norms-radius, 0.)@np.maximum(norms-radius, 0.))
        out = dict(method=self.method, q=q.tolist(), accepted=bool(verdict.accepted),
            internal_status=status, internal_ok=status == 'task_admissible',
            native_status=native_status, iterations=iterations, evaluations=evaluations,
            verification_calls=verifications, box_qp_updates=updates,
            backtracking_evaluations=backtracks, task_excess_objective=phi,
            internal_radius=radius, finite=bool(verdict.finite_ok),
            joint_limit_ok=bool(verdict.joint_limit_ok), velocity_ok=bool(verdict.velocity_ok),
            position_error=float(verdict.position_error), orientation_error=float(verdict.orientation_error),
            verification_reasons=list(verdict.reasons),
            failure_kind='accepted' if verdict.accepted else '+'.join(verdict.reasons),
            velocity_utilization=float(np.max(np.abs(self.kin.difference(q,p))/S)))
        if self.trace:
            out.update(trace=traces, frame_lower=lo.tolist(), frame_upper=hi.tolist(),
                       step_scale=S.tolist(), diagnostic_trace=True)
        out['total_latency_ns'] = perf_counter_ns()-start
        out['accepted_within_20ms'] = bool(verdict.accepted and out['total_latency_ns'] <= 20_000_000)
        return out
