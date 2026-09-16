"""Nearest-method controls. Frozen Task Balance GN is never patched globally.

The range-loss control ports a published loss, not the full RangedIK system.
All configurations are single-start, current-input-only numerical solvers.
"""
import types
from time import perf_counter_ns
import numpy as np
from scipy.optimize import minimize
from .task_balance_gn import CompletedTaskBalanceGN, solve_completion, task_value
from .correction_reserve.geometry import residual_linearization, task_scale
from .correction_reserve.native_geometry import NativeGeometry
from .types import IKQuery, Pose


class FixedLocalBudget(CompletedTaskBalanceGN):
    """One/two weighted QPs PER local model, not one/two IK iterations."""
    def __init__(self, kin, verifier, urdf, count):
        if count not in (1, 2):
            raise ValueError('fixed comparison only supports 1 or 2 QPs')
        super().__init__(kin, verifier, urdf, 1., .25)
        self.count = count
        self.method = f'fixed_qp{count}'
        core = types.FunctionType(solve_completion.__code__,
            dict(solve_completion.__globals__, completion_dual_direction=self.direction),
            solve_completion.__name__, solve_completion.__defaults__)
        adapter = CompletedTaskBalanceGN.solve
        bound = types.FunctionType(adapter.__code__, dict(adapter.__globals__, solve_completion=core),
                                   adapter.__name__, adapter.__defaults__)
        self.solve = types.MethodType(bound, self)

    def direction(self, e, G, lo, hi, lam, kappa, posture, w, box_qp,
                  first, acceptable, settings, deadline, trace=False):
        theta = .5
        pzero = task_value(e) + .5*kappa*float(posture@posture)
        best = None; upper = np.inf; besttheta = theta
        active = 0; calls = 0; reason = 'fixed_budget'
        H, g, d, nit = first
        for k in range(self.count):
            # The already computed first QP is retained even at the deadline.
            if k:
                if perf_counter_ns() >= deadline:
                    reason = 'deadline'; break
                H = theta*C0+(1-theta)*C1+commonH
                g = theta*c0+(1-theta)*c1+commong
                d, nit = box_qp(H, g, lo, hi)
            calls += 1; active += int(nit)
            if not np.isfinite(d).all() or np.max(np.maximum(lo-d, d-hi)) > 1e-10:
                reason = 'invalid_qp'; break
            if k and acceptable(d):
                return d, dict(reason='task', theta=theta, dual_updates=calls,
                    active_updates=active, upper=None, lower=None, gap=None,
                    pzero=pzero, model_fraction=None, trace=[])
            rp = e[:3]+G[:3]@d; rr = e[3:]+G[3:]@d
            fp = float(rp@rp); fr = float(rr@rr)
            reg = .5*lam*float(d@d)+.5*kappa*float(np.sum((posture+w*d)**2))
            value = max(fp, fr)+reg
            if value < upper:
                upper = value; best = d.copy(); besttheta = theta
            if k+1 == self.count or fp == fr:
                break
            # Same safeguarded dual-Newton/bracket choice as frozen core.
            # No lower-bound, forcing-gap, or certificate work is performed.
            C0 = 2*G[:3].T@G[:3]; C1 = 2*G[3:].T@G[3:]
            c0 = 2*G[:3].T@e[:3]; c1 = 2*G[3:].T@e[3:]
            commonH = lam*np.eye(G.shape[1])+kappa*np.diag(w*w)
            commong = kappa*w*posture
            left, right = (.5, 1.) if fp > fr else (0., .5)
            free = np.flatnonzero((d > lo+1e-10) & (d < hi-1e-10))
            z = (C0-C1)@d+c0-c1; curvature = 0.
            if len(free):
                curvature = -float(z[free]@np.linalg.solve(H[np.ix_(free, free)], z[free]))
            proposed = theta-(fp-fr)/curvature if curvature < -1e-18 else .5*(left+right)
            proposed = float(np.clip(proposed, 0, 1))
            if not left < proposed < right:
                if not ((proposed == 0 and left == 0) or (proposed == 1 and right == 1)):
                    proposed = .5*(left+right)
            if abs(proposed-theta) < 1e-14:
                proposed = .5*(left+right)
            theta = proposed
        return best, dict(reason=reason, theta=besttheta, dual_updates=calls,
            active_updates=active, upper=None if best is None else upper,
            lower=None, gap=None, pzero=pzero, model_fraction=None, trace=[])


def swamp_groove(r):
    """Official ranged-ik Swamp Groove at g=0,l=-1,u=1,c=2.

    Source parameters f1=1,f2=.01,f3=100,p=20 are unchanged. Applying
    this scalar formula to a block norm is an explicitly matched-set port.
    Returns value and derivative wrt the nonnegative norm r.
    """
    r = float(r)
    a = (-np.log(.05))*r**20 if r < 10 else np.inf
    expa = np.exp(-a)
    expg = np.exp(-r*r/8)
    value = -expg+.01*r*r-100*np.expm1(-a)
    barrier_grad = 0. if r == 0 or not np.isfinite(a) else 2000*a*expa/r
    return float(value), float(r*expg/4+.02*r+barrier_grad)


def range_objective(e, J):
    f = 0.; grad = np.zeros(J.shape[1])
    for block in (slice(0, 3), slice(3, 6)):
        r = float(np.linalg.norm(e[block])); value, derivative = swamp_groove(r)
        f += value
        if r:
            grad += derivative/r*(e[block]@J[block])
    return f, grad


class _TaskAccepted(Exception): pass
class _Deadline(Exception): pass


class CurrentOptimizer:
    """Direct constrained SLSQP / bounded range-loss L-BFGS-B reference."""
    def __init__(self, kin, verifier, urdf, method):
        if method not in ('direct_sqp', 'range_loss_matched'):
            raise ValueError(method)
        self.kin, self.verifier, self.method = kin, verifier, method
        self.native = NativeGeometry(kin, urdf); self.scale = task_scale(verifier)
        self.settings = dict(maxiter=100, deadline_ms=20., single_start='actual previous_q',
            optimizer='SLSQP' if method == 'direct_sqp' else 'L-BFGS-B',
            ftol=1e-8 if method == 'direct_sqp' else 1e-12,
            gtol=None if method == 'direct_sqp' else 1e-8, maxls=20)

    def reset(self, q): pass
    def close(self): pass

    def solve(self, position, rotation, previous, dt=.02):
        start = perf_counter_ns(); deadline = start+20_000_000
        query = IKQuery(Pose(np.asarray(position, float), np.asarray(rotation, float)),
                        np.asarray(previous, float), dt)
        p = query.previous_q; S = self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        lo = np.maximum(self.kin.limits.lower+1e-12, p-S*(1-1e-12))
        hi = np.minimum(self.kin.limits.upper-1e-12, p+S*(1-1e-12))
        yl, yh = (lo-p)/S, (hi-p)/S
        bestq = np.clip(p, lo, hi); best = np.inf
        cy = ce = cJ = None; calls = checks = iterations = 0
        status = 'not_started'; native_code = None; native_success = False

        def model(y):
            nonlocal cy, ce, cJ, bestq, best, calls, checks
            if perf_counter_ns() >= deadline:
                raise _Deadline()
            if cy is None or not np.array_equal(cy, y):
                q = p+S*y
                e, J, _ = residual_linearization(self.native, query.target, q, self.scale)
                calls += 1; cy = y.copy(); ce = e; cJ = J*S
                value = task_value(e)
                if value < best:
                    best = value; bestq = q.copy()
                if value <= 1.:
                    checks += 1
                    if self.verifier.check(q, query).accepted:
                        bestq = q.copy()
                        raise _TaskAccepted()
            return ce, cJ

        def objective(y):
            e, J = model(y)
            return (.5*float(y@y), y.copy()) if self.method == 'direct_sqp' else range_objective(e, J)

        def constraints(y):
            e, J = model(y)
            return np.array([1-e[:3]@e[:3], 1-e[3:]@e[3:]])

        def derivatives(y):
            e, J = model(y)
            return -2*np.vstack([e[:3]@J[:3], e[3:]@J[3:]])

        def callback(y):
            nonlocal iterations
            iterations += 1; model(y)

        try:
            kwargs = dict(fun=objective, x0=np.clip(np.zeros(len(p)), yl, yh), jac=True,
                          bounds=list(zip(yl, yh)), callback=callback)
            if self.method == 'direct_sqp':
                result = minimize(**kwargs, method='SLSQP',
                    constraints={'type':'ineq','fun':constraints,'jac':derivatives},
                    options={'maxiter':100,'ftol':1e-8})
            else:
                result = minimize(**kwargs, method='L-BFGS-B',
                    options={'maxiter':100,'ftol':1e-12,'gtol':1e-8,'maxls':20})
            native_code = int(result.status); native_success = bool(result.success)
            status = 'optimizer:'+str(result.message)
        except _TaskAccepted:
            status = 'task_accepted_early'
        except _Deadline:
            status = 'deadline'
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            status = 'numeric_error:'+str(exc)
        verdict = self.verifier.check(bestq, query); checks += 1
        row = dict(method=self.method,q=bestq.tolist(),accepted=bool(verdict.accepted),
            internal_status=status,internal_ok=native_success,native_return_code=native_code,
            finite=bool(verdict.finite_ok),joint_limit_ok=bool(verdict.joint_limit_ok),
            velocity_ok=bool(verdict.velocity_ok),position_error=float(verdict.position_error),
            orientation_error=float(verdict.orientation_error),verification_reasons=list(verdict.reasons),
            failure_kind='accepted' if verdict.accepted else '+'.join(verdict.reasons),
            velocity_utilization=float(np.max(abs(self.kin.difference(bestq,p))/S)),
            evaluations=calls,iterations=iterations,verification_calls=checks,
            dual_updates=0,box_qp_updates=0,backtracking_evaluations=None,subproblems=[])
        row['total_latency_ns'] = perf_counter_ns()-start
        row['accepted_within_20ms'] = bool(verdict.accepted and row['total_latency_ns'] <= 20_000_000)
        return row
