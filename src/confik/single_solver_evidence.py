"""Isolated local-QP controls; the supplied BoundedGN sources stay unchanged.

Only the function's private global dictionary differs. There is no global
monkey-patch, new IK iteration, recovery solver, or change to the merit test.
"""
from time import perf_counter_ns
from types import FunctionType, MethodType
import numpy as np
from scipy import sparse
import osqp


def bind_qp(engine, callback):
    original = type(engine).solve
    namespace = dict(original.__globals__)
    namespace['box_qp'] = callback
    private = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    engine.solve = MethodType(private, engine)


def clipped_qp(H, g, lo, hi):
    return np.clip(np.linalg.solve(H, -g), lo, hi), 1


def quality(H, g, lo, hi, x):
    """Independent projected KKT check, in original and problem-scaled units."""
    grad = H @ x + g
    scale = max(1., np.linalg.norm(H, np.inf)*np.linalg.norm(x, np.inf)
                + np.linalg.norm(g, np.inf))
    return dict(box_violation=float(max(0., np.max(lo-x), np.max(x-hi))),
        projected_kkt=float(np.max(np.abs(x-np.clip(x-grad, lo, hi)))),
        normalized_kkt=float(np.max(np.abs(x-np.clip(x-grad/scale, lo, hi)))),
        gradient_scale=float(scale), objective=float(.5*x@H@x+g@x),
        active_bounds=int(np.sum((np.abs(x-lo)<1e-8)|(np.abs(x-hi)<1e-8))))


class CachedOSQP:
    """Same SPD box QP, full upper CSC pattern, numeric updates only.

    Raw numerical violations are logged. An inward projection of at most 1e-8
    in dimensionless step coordinates removes OSQP roundoff at an active bound;
    larger violations/nonfinite results produce a zero update, not another solve.
    This does not alter the task verifier. Native status is not an optimality proof.
    """
    def __init__(self, n, eps_abs=1e-8):
        start = perf_counter_ns()
        self.n = n
        self.cols = np.repeat(np.arange(n), np.arange(1, n+1))
        self.rows = np.concatenate([np.arange(j+1) for j in range(n)])
        indptr = np.r_[0, np.cumsum(np.arange(1, n+1))]
        P = sparse.csc_matrix((np.eye(n)[self.rows,self.cols], self.rows, indptr), shape=(n,n))
        self.settings = dict(eps_abs=eps_abs, eps_rel=0., polishing=True,
            max_iter=20000, warm_starting=True, adaptive_rho=True,
            adaptive_rho_interval=25, verbose=False)
        self.solver = osqp.OSQP()
        self.solver.setup(P=P, q=np.zeros(n), A=sparse.eye(n,format='csc'),
                          l=-np.ones(n), u=np.ones(n), **self.settings)
        self.initialization_ns = perf_counter_ns()-start
        self.last = {}

    def reset(self, q=None):
        self.solver.warm_start(x=np.zeros(self.n), y=np.zeros(self.n))

    def __call__(self, H, g, lo, hi):
        start = perf_counter_ns()
        values = np.ascontiguousarray(H[self.rows,self.cols])
        linear = np.ascontiguousarray(g)
        lower, upper = np.ascontiguousarray(lo), np.ascontiguousarray(hi)
        converted = perf_counter_ns()
        self.solver.update(Px=values, q=linear, l=lower, u=upper)
        updated = perf_counter_ns()
        result = self.solver.solve(raise_error=False)
        solved = perf_counter_ns()
        raw = result.x.copy()
        finite = bool(np.isfinite(raw).all())
        violation = float(max(0.,np.max(lo-raw),np.max(raw-hi))) if finite else float('inf')
        usable = finite and violation<=1e-8
        x = np.clip(raw,lo,hi) if usable else np.clip(np.zeros(self.n),lo,hi)
        self.last = dict(conversion_ns=converted-start, update_ns=updated-converted,
            solve_ns=solved-updated, status=result.info.status,
            status_val=int(result.info.status_val), native_iterations=int(result.info.iter),
            raw_x=raw, raw_box_violation=violation, usable=usable,
            projection_delta=float(np.max(np.abs(x-raw))) if finite else None)
        self.last['check_ns'] = perf_counter_ns()-solved
        self.last['callback_ns'] = perf_counter_ns()-start
        return x, result.info.iter


def outer_counts(row):
    """Recover task evaluations/backtracking from unchanged core counters."""
    it = row['iterations']; status = row['internal_status']
    evaluations = it-(status=='deadline')
    calls = it-(status in ('deadline','task_candidate','nonfinite_model'))
    trial_evaluations = row['evaluations']-evaluations-1  # final FK evaluation
    attempted = calls-(status=='stationary')
    return dict(qp_calls=int(calls), task_evaluation_iterations=int(evaluations),
        line_search_evaluations=int(trial_evaluations),
        extra_backtracking_evaluations=int(trial_evaluations-attempted))
