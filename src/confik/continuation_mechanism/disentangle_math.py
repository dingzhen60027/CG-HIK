"""Fixed, non-learning interventions; no modification of a frozen runtime."""
from time import perf_counter_ns

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize

from ..geometry import pose_error
from .observation import metrics, representable_interior

MATCH_POSITION = 1e-8
MATCH_ORIENTATION = 1e-8
MIN_CONFIGURATION_DISTANCE = 1e-5
PERTURBATIONS = (0.25, -0.25, 0.5, -0.5, 0.1, -0.1)


def scales(kin, verifier, dt):
    task = np.array([verifier.config.position_tolerance]*3 +
                    [verifier.config.orientation_tolerance]*3)
    step = kin.limits.velocity*dt + verifier.config.velocity_tolerance
    return task, step


def residual_objective(kin, target, q):
    e = pose_error(target, kin.forward(q))
    return float(e @ e)


def uniform_refine(kin, verifier, query, original):
    """Identical objective, bounds and budget to the old refinement.

    Count every residual callback (including finite differences), separately
    from scipy.nfev. The unavailable native TRAC-IK FEV is never called zero.
    """
    start = perf_counter_ns()
    original = np.asarray(original)
    before = verifier.check(original, query)
    if not before.accepted:
        raise ValueError('refinement requires an already legal candidate')
    objective_before = before.position_error**2 + before.orientation_error**2
    lower, upper = representable_interior(kin, query, verifier)
    calls = 0

    def residual(q):
        nonlocal calls
        calls += 1
        return pose_error(query.target, kin.forward(q))

    result = least_squares(residual, np.clip(original, lower, upper),
                           bounds=(lower, upper), method='trf', max_nfev=200,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    checked = verifier.check(result.x, query)
    objective_after = checked.position_error**2 + checked.orientation_error**2
    improved = bool(checked.accepted and objective_after < objective_before)
    selected = result.x.copy() if improved else original.copy()
    assert verifier.check(selected, query).accepted
    elapsed = perf_counter_ns()-start
    return dict(q=selected.tolist(), refined_q=result.x.tolist(), improved=improved,
                residual_calls=calls, scipy_nfev=int(result.nfev), solver_status=int(result.status),
                objective_before=objective_before, objective_after=objective_after,
                refined_verifier_reasons=list(checked.reasons), refinement_latency_ns=elapsed,
                strict_residual_met=bool(checked.position_error<=1e-7 and checked.orientation_error<=1e-7),
                verifier_calls=3)


def candidate_features(kin, verifier, query, q):
    q = np.asarray(q); task, step = scales(kin, verifier, query.dt)
    e = pose_error(query.target, kin.forward(q))
    j = kin.jacobian(q)/task[:, None]*step[None, :]
    singular = np.linalg.svd(j, compute_uv=False)
    delta = kin.difference(q, query.previous_q)
    return dict(**metrics(kin, verifier, query, q),
                position_residual_vector=e[:3].tolist(), orientation_residual_vector=e[3:].tolist(),
                normalized_residual_vector=(e/task).tolist(),
                normalized_residual_norm=float(np.linalg.norm(e/task)),
                normalized_position_norm=float(np.linalg.norm(e[:3]/task[:3])),
                normalized_orientation_norm=float(np.linalg.norm(e[3:]/task[3:])),
                joint_step=delta.tolist(), normalized_joint_step=(delta/step).tolist(),
                joint_lower_margin=(q-kin.limits.lower).tolist(), joint_upper_margin=(kin.limits.upper-q).tolist(),
                distance_to_previous=float(np.linalg.norm(delta)),
                scaled_singular_values=singular.tolist(), scaled_sigma_min=float(singular[-1]),
                scaled_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None)


def matched_projection(kin, verifier, query, center, amplitude):
    """Fix the strongest null coordinate, project the other six nonlinearly.

    Target is the CENTER'S achieved full pose, never the commanded target or
    its scalar error norm. The current common previous state supplies bounds.
    """
    start = perf_counter_ns(); center = np.asarray(center)
    task, step = scales(kin, verifier, query.dt)
    target = kin.forward(center)
    matrix = kin.jacobian(center)/task[:, None]*step[None, :]
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    direction = vh[-1].copy()
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot]<0: direction *= -1
    direction /= np.max(np.abs(direction))
    lower, upper = representable_interior(kin, query, verifier)
    seed = np.clip(center + amplitude*step*direction, lower, upper)
    free = np.arange(kin.nq) != pivot
    calls = 0

    def expand(x):
        q = seed.copy(); q[free] = x
        return q

    def residual(x):
        nonlocal calls
        calls += 1
        return pose_error(target, kin.forward(expand(x)))/task

    fit = least_squares(residual, seed[free], bounds=(lower[free], upper[free]),
                        method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    q = expand(fit.x); error = pose_error(target, kin.forward(q))
    match_p = float(np.linalg.norm(error[:3])); match_r = float(np.linalg.norm(error[3:]))
    distance = float(np.max(np.abs(q-center)))
    check = verifier.check(q, query)
    usable = bool(check.accepted and match_p<=MATCH_POSITION and match_r<=MATCH_ORIENTATION
                  and distance>=MIN_CONFIGURATION_DISTANCE)
    return dict(q=q.tolist(), center_q=center.tolist(), amplitude=amplitude, pivot_joint=pivot,
                center_actual_position=target.position.tolist(), center_actual_rotation=target.rotation.tolist(),
                position_match_vector=error[:3].tolist(), orientation_match_vector=error[3:].tolist(),
                position_match_error=match_p, orientation_match_error=match_r,
                max_configuration_difference=distance, usable=usable,
                current_verifier_accepted=bool(check.accepted), verifier_reasons=list(check.reasons),
                solver_status=int(fit.status), scipy_nfev=int(fit.nfev), residual_calls=calls,
                latency_ns=perf_counter_ns()-start)


def linear_minimum_step(matrix, error, lower, upper):
    """Minimize infinity-norm joint-speed utilization with two L2 pose balls.

    All seven redundant coordinates are variables. Bounds are joint ranges
    in per-joint velocity units, not a selected pseudoinverse solution.
    Numerical failure yields UNKNOWN, never mathematical infeasibility.
    """
    n = matrix.shape[1]
    initial = lsq_linear(matrix, error, bounds=(lower, upper), tol=1e-12, max_iter=200)
    z = initial.x
    x0 = np.r_[z, np.max(np.abs(z)) + 1e-6]

    def con(x):
        e = error-matrix@x[:n]
        return np.r_[1-e[:3]@e[:3], 1-e[3:]@e[3:], x[n]-x[:n], x[n]+x[:n]]

    def jac(x):
        e = error-matrix@x[:n]
        out = np.zeros((2+2*n,n+1))
        out[0,:n] = 2*e[:3]@matrix[:3]; out[1,:n] = 2*e[3:]@matrix[3:]
        out[2:2+n,:n] = -np.eye(n); out[2+n:,:n] = np.eye(n)
        out[2:,-1] = 1
        return out

    gradient = np.r_[np.zeros(n), 1.]
    result = minimize(lambda x: x[-1], x0, jac=lambda x: gradient, method='SLSQP',
        bounds=[*zip(lower,upper), (0,None)], constraints={'type':'ineq','fun':con,'jac':jac},
        options={'maxiter':300,'ftol':1e-10})
    feasible = bool(np.min(con(result.x))>=-1e-7 and np.all(result.x[:n]>=lower-1e-9)
                    and np.all(result.x[:n]<=upper+1e-9))
    return dict(z=result.x[:n].tolist(), demand=float(np.max(np.abs(result.x[:n]))) if feasible else None,
                optimum_reported=bool(result.success and feasible), linear_feasible=feasible,
                solver_status=int(result.status), solver_message=str(result.message),
                function_evaluations=int(result.nfev), iterations=int(result.nit),
                constraint_min=float(np.min(con(result.x))))


def next_target_demand(kin, verifier, next_query):
    start=perf_counter_ns(); q=np.asarray(next_query.previous_q)
    task,step=scales(kin,verifier,next_query.dt)
    e=pose_error(next_query.target,kin.forward(q))/task
    matrix=kin.jacobian(q)/task[:,None]*step[None,:]
    # Physical joint range is retained even if the minimum demand exceeds one
    # nominal frame. Only the unchanged nonlinear verifier decides acceptance.
    lower=(kin.limits.lower-q)/step; upper=(kin.limits.upper-q)/step
    result=linear_minimum_step(matrix,e,lower,upper)
    next_q=q+step*np.asarray(result['z'])
    linear_error=e-matrix@np.asarray(result['z'])
    nonlinear=metrics(kin,verifier,next_query,next_q)
    result.update(next_q=next_q.tolist(),next_target_position=next_query.target.position.tolist(),
        next_target_rotation=next_query.target.rotation.tolist(),
        linear_position_norm=float(np.linalg.norm(linear_error[:3])),
        linear_orientation_norm=float(np.linalg.norm(linear_error[3:])),
        nonlinear_verification=nonlinear,latency_ns=perf_counter_ns()-start,
        interpretation='offline local-linear demand with full redundant coordinates; not an infeasibility certificate')
    return result
