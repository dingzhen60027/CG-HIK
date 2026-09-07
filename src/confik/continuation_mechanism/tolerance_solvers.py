"""Existing TRAC-IK/DLS with explicit stopping criteria and public acceptance.

No look-ahead, candidate search, post-solve refinement, or solver modification.
Both DLS variants use the same query-bounded clip adapter; all other DLS
parameters and update/line-search code are the unchanged AdaptiveDLS.
"""
from dataclasses import replace
import ctypes as ct
from time import perf_counter_ns

import numpy as np

from ..revision_compute_allocation.trac_ik import TracIK
from ..solvers.dls import AdaptiveDLS, DLSConfig
from ..types import IKQuery, Pose
from .observation import representable_interior

STRICT_EPS = 1e-5
ITERATIONS = 25
METHODS = ('trac_strict_5ms','trac_task_5ms','trac_strict_20ms','trac_task_20ms',
           'dls_strict','dls_task')


def cartesian_box(config):
    # diffRelative is target-frame translation and rotation vector, not RPY.
    # Native success zeroes components <= bounds, then checks eps. With every
    # bound > eps the union is exactly the component box, NOT bounds + eps.
    box = np.nextafter(np.asarray([config.position_tolerance]*3+
                                  [config.orientation_tolerance]*3)/np.sqrt(3.), 0.)
    assert np.all(box > STRICT_EPS)
    return np.ascontiguousarray(box)


class QueryBoundedKinematics:
    def __init__(self, kin):
        self.kin = kin
        self.lower, self.upper = kin.limits.lower.copy(), kin.limits.upper.copy()

    def __getattr__(self, name):
        return getattr(self.kin, name)

    def clip(self, q, margin=0.):
        return np.clip(self.kin.clip(q, margin), self.lower, self.upper)


def feedback(previous, observation):
    return np.asarray(observation['q']).copy() if observation['accepted'] else np.asarray(previous).copy()


class ToleranceSolver:
    def __init__(self, method, kin, verifier, config, library=None, urdf=None):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.kin, self.verifier = method, kin, verifier
        self.trac = None
        if method.startswith('trac'):
            self.budget_ms = 20 if '20ms' in method else 5
            self.trac = TracIK(library, urdf, kin, verifier, self.budget_ms, STRICT_EPS)
            ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            self.trac.lib.tolerance_trac_solve.argtypes = [ct.c_void_p,ct.c_int]+[ptr]*7
            self.trac.lib.tolerance_relative_error.argtypes = [ptr]*5+[ct.c_double,ptr]
            self.bounds = cartesian_box(verifier.config) if '_task_' in method else np.zeros(6)
        else:
            self.budget_ms = None
            self.bounded = QueryBoundedKinematics(kin)
            cfg = DLSConfig(**config['solver'])
            if method == 'dls_strict':
                cfg = replace(cfg, position_tolerance=STRICT_EPS, orientation_tolerance=STRICT_EPS)
            else:
                cfg = replace(cfg, position_tolerance=verifier.config.position_tolerance,
                              orientation_tolerance=verifier.config.orientation_tolerance)
            self.dls = AdaptiveDLS(self.bounded, cfg)

    def solve(self, target_position, target_rotation, previous_q, dt=.02):
        # Timing begins BEFORE required input conversion. Offline metrics below
        # the stop time do not participate in solving, selection or acceptance.
        start = perf_counter_ns()
        query = IKQuery(Pose(target_position, target_rotation), np.asarray(previous_q, dtype=float), dt)
        lower, upper = representable_interior(self.kin, query, self.verifier)
        seed = np.ascontiguousarray(np.clip(query.previous_q, lower, upper))
        conversion_ns = perf_counter_ns()-start
        solving_start = perf_counter_ns()
        if self.trac:
            raw = np.full(self.kin.nq, np.nan)
            rc = int(self.trac.lib.tolerance_trac_solve(self.trac.handle,self.kin.nq,seed,
                np.ascontiguousarray(query.target.position),np.ascontiguousarray(query.target.rotation),
                lower,upper,self.bounds,raw))
            if rc == -999:
                raise RuntimeError(self.trac.lib.revision_trac_error().decode())
            internal_ok = rc >= 0
            status = 'native_success' if internal_ok else 'native_no_solution'
            iterations = fev = None
        else:
            self.bounded.lower, self.bounded.upper = lower, upper
            result = self.dls.solve(query.target,seed,ITERATIONS,seed_source='actual_previous_q')
            raw = result.q
            internal_ok, status = bool(result.converged), result.reason
            rc, iterations, fev = None, int(result.iterations), int(result.function_evaluations)
        solve_ns = perf_counter_ns()-solving_start
        check_start = perf_counter_ns()
        check = self.verifier.check(raw, query)
        verification_ns = perf_counter_ns()-check_start
        elapsed = perf_counter_ns()-start
        # Shared terminal semantics: a returned finite configuration is accepted
        # iff the public verifier passes, even if internal strict precision was
        # not reached. Native success is a separate endpoint, not validity proof.
        accepted = bool(check.accepted)
        out = dict(method=self.method, internal_ok=internal_ok, internal_status=status,
            native_return_code=rc, iterations=iterations, solver_function_evaluations=fev,
            finite=bool(check.finite_ok), q=raw.tolist() if check.finite_ok else None,
            accepted=accepted, verification_reasons=list(check.reasons),
            position_error=float(check.position_error) if check.finite_ok else None,
            orientation_error=float(check.orientation_error) if check.finite_ok else None,
            total_latency_ns=elapsed, conversion_ns=conversion_ns, solve_ns=solve_ns,
            verification_ns=verification_ns, budget_ms=self.budget_ms,
            returned_within_5ms=elapsed<=5_000_000,returned_within_20ms=elapsed<=20_000_000,
            lower=lower.tolist(), upper=upper.tolist(), seed_q=seed.tolist())
        if check.finite_ok:
            delta = np.abs(self.kin.difference(raw, query.previous_q))
            allowed = self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
            out.update(velocity_utilization=float(np.max(delta/allowed)),
                nominal_velocity_utilization=float(np.max(delta/(self.kin.limits.velocity*dt))),
                velocity_utilization_per_joint=(delta/allowed).tolist(),
                joint_limit_margin=float(np.min(np.minimum(raw-self.kin.limits.lower,self.kin.limits.upper-raw))))
        else:
            out.update(velocity_utilization=None, nominal_velocity_utilization=None,
                       velocity_utilization_per_joint=None, joint_limit_margin=None)
        labels = []
        if not internal_ok: labels.append('internal_nonconvergence')
        labels += list(check.reasons)
        out['failure_kind'] = 'accepted' if accepted else '+'.join(labels)
        return out

    def close(self):
        if self.trac:
            self.trac.close()
