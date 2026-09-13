"""Task-set squared-distance generalized GN: an unvalidated research prototype.

Only current-state variables are optimized. Uses the supplied reference box-QP;
there is no fallback, prediction, scenario expansion or learned gate. The two
three-dimensional normalized pose blocks are projected onto norm balls.
The projection and generalized GN are established mathematical ingredients;
this source does not claim novelty, global convergence or real-time performance.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from bounded_gn_reference import box_qp


def projected_task_model(e: np.ndarray, radius: float = 1.0):
    """Return .5 dist(e,C)^2, its residual-space gradient, and generalized metric.

    C is the product of two 3-D balls. At their boundary the inside derivative
    (zero) is selected. For each active block, the radial eigenvalue is one and
    the two tangential eigenvalues are 1-radius/||e_block||.
    """
    e = np.asarray(e, dtype=float)
    if e.shape != (6,) or not np.isfinite(e).all():
        raise ValueError('Expected a finite, six-dimensional normalized residual')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive')
    v = np.zeros(6)
    W = np.zeros((6, 6))
    for start in (0, 3):
        b = e[start:start + 3]
        norm = np.linalg.norm(b)
        if norm > radius:
            u = b / norm
            v[start:start + 3] = (norm - radius) * u
            W[start:start + 3, start:start + 3] = (
                (1.0 - radius / norm) * np.eye(3)
                + (radius / norm) * np.outer(u, u))
    return 0.5 * float(v @ v), v, W


@dataclass(frozen=True)
class PrototypeSettings:
    damping: float = 0.01
    maximum_iterations: int = 30
    backtracking_steps: int = 8
    # A fixed *stricter* internal numerical margin, not a relaxed verifier.
    normalized_margin: float = 1e-6


def solve_task_excess(previous: np.ndarray, dt: float, lower: np.ndarray,
                      upper: np.ndarray, velocity: np.ndarray,
                      evaluate: Callable, verification: Callable,
                      settings: PrototypeSettings | None = None,
                      velocity_tolerance: float = 1e-4,
                      initial: np.ndarray | None = None,
                      point_loss_control: bool = False) -> dict:
    """Diagnostic numerical solve; no hardware deadline claim is made.

    evaluate(q) -> normalized residual (6,), residual Jacobian (6,n).
    verification(q) -> bool under the ORIGINAL current-query contract.
    initial is a diagnostic-only iterate. It never changes the true previous
    state used to define the available current-frame joint interval.
    point_loss_control changes only the residual-space model/merit to .5||e||².
    """
    cfg = settings or PrototypeSettings()
    p, low, high, vel = (np.asarray(x, dtype=float) for x in
                         (previous, lower, upper, velocity))
    if p.ndim != 1 or not all(x.shape == p.shape for x in (low, high, vel)):
        raise ValueError('Inconsistent joint dimensions')
    if not all(np.isfinite(x).all() for x in (p, low, high, vel)) or dt <= 0:
        raise ValueError('Nonfinite inputs or invalid dt')
    if np.any(low >= high) or np.any(vel <= 0):
        raise ValueError('Invalid robot limits')
    S = vel * dt + velocity_tolerance
    lo = np.maximum(low + 1e-12, p - S * (1.0 - 1e-12))
    hi = np.minimum(high - 1e-12, p + S * (1.0 - 1e-12))
    if np.any(lo > hi):
        raise ValueError('Empty admissible current-frame interval')
    q = np.clip(p if initial is None else np.asarray(initial, dtype=float), lo, hi)
    lam = cfg.damping
    trace = []
    qp_updates = 0
    status = 'iteration_limit'
    radius = 1.0 - cfg.normalized_margin

    def residual_model(e):
        if point_loss_control:
            return 0.5 * float(e @ e), e, np.eye(6)
        return projected_task_model(e, radius)

    for it in range(cfg.maximum_iterations):
        e, J = evaluate(q)
        e, J = np.asarray(e, float), np.asarray(J, float)
        if e.shape != (6,) or J.shape != (6, len(q)):
            raise ValueError('Invalid residual or Jacobian shape')
        f, v, W = residual_model(e)
        legal = bool(verification(q))
        trace.append(dict(iterate=it, q=q.tolist(), objective=f,
                          position_normalized=float(np.linalg.norm(e[:3])),
                          orientation_normalized=float(np.linalg.norm(e[3:])),
                          accepted=legal))
        if legal:
            status = 'task_admissible'
            break
        G = J * S
        H, g = G.T @ W @ G + lam * np.eye(len(q)), G.T @ v
        d, updates = box_qp(H, g, np.maximum((lo-q)/S, -1.),
                            np.minimum((hi-q)/S, 1.))
        qp_updates += int(updates)
        gd = float(g @ d)
        if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-12:
            status = 'stationary'
            break
        adopted = False
        for k in range(cfg.backtracking_steps):
            alpha = 2.0 ** (-k)
            candidate = np.clip(q + alpha * S * d, lo, hi)
            en, _ = evaluate(candidate)
            fn = residual_model(en)[0]
            # A standard Armijo check of the SAME nonlinear task merit.
            if verification(candidate) or (gd < 0 and fn <= f + 1e-4*alpha*gd):
                q = candidate
                lam = max(cfg.damping/100., lam*0.5)
                adopted = True
                break
        if not adopted:
            lam *= 10.
            if lam > 1e8:
                status = 'no_descent'
                break
    verdict = bool(verification(q))
    if verdict:
        status = 'task_admissible'
    e, _ = evaluate(q)
    return dict(q=q.tolist(), accepted=verdict, status=status,
                iterations=len(trace), box_qp_updates=qp_updates,
                position_normalized=float(np.linalg.norm(e[:3])),
                orientation_normalized=float(np.linalg.norm(e[3:])),
                trace=trace, point_loss_control=point_loss_control,
                internal_radius=radius, timed=False)
