"""Retain numerical candidates even when the public verifier rejects them."""
from __future__ import annotations

from time import perf_counter_ns
import numpy as np
from scipy.optimize import least_squares

from ..geometry import pose_error
from ..revision_compute_allocation.trac_ik import TracIK, allowed_interval


def metrics(kin, verifier, query, q):
    if q is None:
        return dict(candidate_available=False, finite=None, verifier_accepted=False,
                    verification_reasons=['candidate_not_retained'], position_error=None,
                    orientation_error=None, joint_limit_margin=None, max_velocity_utilization=None,
                    velocity_margin_rad=None, contract_velocity_utilization=None, q=None)
    q = np.asarray(q, dtype=float)
    check = verifier.check(q, query)
    out = dict(candidate_available=True, finite=check.finite_ok,
               verifier_accepted=check.accepted, verification_reasons=list(check.reasons),
               q=q.tolist() if check.finite_ok else None,
               position_error=None, orientation_error=None, joint_limit_margin=None,
               max_velocity_utilization=None, velocity_margin_rad=None,
               contract_velocity_utilization=None)
    if check.finite_ok:
        step = np.abs(kin.difference(q, query.previous_q))
        allowed = kin.limits.velocity * query.dt + verifier.config.velocity_tolerance
        out.update(position_error=float(check.position_error), orientation_error=float(check.orientation_error),
                   joint_limit_margin=float(np.min(np.minimum(q-kin.limits.lower, kin.limits.upper-q))),
                   max_velocity_utilization=float(np.max(step/(kin.limits.velocity*query.dt))),
                   contract_velocity_utilization=float(np.max(step/allowed)),
                   velocity_margin_rad=float(np.min(allowed-step)),
                   velocity_excess_rad=float(np.max(step-allowed)),
                   velocity_utilization_per_joint=(step/allowed).tolist(),
                   sigma_min=float(np.linalg.svd(kin.jacobian(q), compute_uv=False)[-1]))
    return out


def failure_kind(accepted, solver_ok, observation):
    if accepted:
        return 'accepted'
    if not solver_ok:
        return 'solver_failure'
    reasons = observation['verification_reasons']
    if not observation['finite']:
        return 'non_finite'
    kinds = []
    if 'position_tolerance' in reasons or 'orientation_tolerance' in reasons:
        kinds.append('pose_error')
    if 'joint_limit' in reasons:
        kinds.append('joint_limit')
    if 'velocity_limit' in reasons:
        kinds.append('velocity_violation')
    return '+'.join(kinds) or 'solver_failure'


def representable_interior(kin, query, verifier):
    """One-ULP interior bounds, with the unchanged subtraction-based verifier.

    Diagnostic implementation control only: no tolerance is relaxed and no
    mathematical feasibility guarantee is implied by successful rounding.
    """
    lower, upper = allowed_interval(kin, query, verifier)
    lower = np.nextafter(lower, upper)
    upper = np.nextafter(upper, lower)
    allowed = kin.limits.velocity*query.dt + verifier.config.velocity_tolerance
    for _ in range(8):
        bad_lower = np.abs(kin.difference(lower, query.previous_q)) > allowed
        bad_upper = np.abs(kin.difference(upper, query.previous_q)) > allowed
        if not (bad_lower.any() or bad_upper.any()):
            return np.ascontiguousarray(lower), np.ascontiguousarray(upper)
        lower[bad_lower] = np.nextafter(lower[bad_lower], upper[bad_lower])
        upper[bad_upper] = np.nextafter(upper[bad_upper], lower[bad_upper])
    raise AssertionError('unable to construct floating-point-safe interval')


class DiagnosticTracIK(TracIK):
    """The same official library/ABI, now retaining its raw returned joints."""
    def observe(self, query, *, seed=None, interior=False):
        start = perf_counter_ns()
        lower, upper = (representable_interior if interior else allowed_interval)(
            self.kinematics, query, self.verifier)
        seed = query.previous_q if seed is None else np.asarray(seed, dtype=float)
        seed = np.ascontiguousarray(np.clip(seed, lower, upper))
        raw = np.full(self.kinematics.nq, np.nan)
        rc = self.lib.revision_trac_solve(self.handle, self.kinematics.nq, seed,
             np.ascontiguousarray(query.target.position), np.ascontiguousarray(query.target.rotation),
             lower, upper, raw)
        if rc == -999:
            raise RuntimeError(self.lib.revision_trac_error().decode())
        obs = metrics(self.kinematics, self.verifier, query, raw)
        accepted = bool(rc >= 0 and obs['verifier_accepted'])
        obs.update(solver_return_code=int(rc), solver_ok=bool(rc >= 0), accepted=accepted,
                   failure_kind=failure_kind(accepted, rc >= 0, obs),
                   latency_ns=perf_counter_ns()-start, budget_ms=self.budget_ms,
                   seed_q=seed.tolist(), lower=lower.tolist(), upper=upper.tolist(),
                   interior_bounds=interior,
                   bound_excess_rad=float(max(np.max(lower-raw), np.max(raw-upper))) if obs['finite'] else None)
        return obs


def refine_candidate(kin, verifier, query, candidate):
    """Bounded, tight-residual offline diagnostic, not a deployed solver change."""
    lower, upper = representable_interior(kin, query, verifier)
    x0 = np.clip(np.asarray(candidate), lower, upper)
    result = least_squares(lambda q: pose_error(query.target, kin.forward(q)), x0,
                           bounds=(lower, upper), ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=200, method='trf')
    obs = metrics(kin, verifier, query, result.x)
    obs.update(solver_status=int(result.status), solver_nfev=int(result.nfev),
               strict_residual_met=bool(obs['position_error'] <= 1e-7 and obs['orientation_error'] <= 1e-7))
    return obs


def observe_internal(method, query):
    """Observe unchanged run_stage calls on an isolated runtime instance."""
    base = method.runtime.base
    cascade = base._cascade
    original = cascade.run_stage
    traces = []

    def observed_stage(*args, **kwargs):
        result = original(*args, **kwargs)
        for t in result.traces:
            row = metrics(method.kinematics, method.verifier, query, t.q)
            row.update(solver_ok=bool(t.converged), solver_status=t.reason, seed_source=t.seed_source,
                       iterations=t.iterations, accepted=bool(t.converged and row['verifier_accepted']))
            row['failure_kind'] = failure_kind(row['accepted'], t.converged, row)
            traces.append(row)
        return result

    cascade.run_stage = observed_stage
    try:
        result = method.solve(query)
    finally:
        cascade.run_stage = original
    return result, traces
