"""Exact, one-task-dimensional checks of CR-IK's fixed-correction reserve.

This is a mathematical unit test, NOT a Panda/UR5e robot benchmark.
All quantities are dimensionless. The cases illustrate that a valid sufficient
radius for a specified exact-residual-cancelling map need not rank candidates by
the size of their task-admissible correction set.

Run: python check_reserve_proxy.py
Dependencies: Python 3.10+, NumPy, SciPy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def evaluate(j: list[float], half_widths: list[float], residual: float,
             tolerance: float = 1.0) -> dict:
    a = np.asarray(j, dtype=float)
    h = np.asarray(half_widths, dtype=float)
    if a.ndim != 1 or a.size != h.size or not np.all(h > 0):
        raise ValueError('J and positive half-widths must be matching vectors')
    if np.dot(a, a) == 0 or abs(residual) > tolerance:
        raise ValueError('Nonzero Jacobian and an admissible nominal residual required')
    # Fixed Euclidean right inverse; nominal scaling S=I.
    b = a / np.dot(a, a)
    positive_rows = np.abs(b) > 1e-15
    proxy = float(np.min(h[positive_rows] / np.abs(b[positive_rows])))
    # Residual after target shift w and correction u is residual + w - J u.
    # There is an acceptable correction iff w belongs to the following interval.
    reach_half_width = float(np.abs(a) @ h)
    lo = -reach_half_width - tolerance - residual
    hi = reach_half_width + tolerance - residual
    radius = float(max(0., min(-lo, hi)))
    # Independently find both extreme feasible target shifts with an LP.
    # Decision [u_1,...,u_n,w], with |residual+w-Ju| <= tolerance.
    A = np.vstack([np.r_[-a, 1.], np.r_[a, -1.]])
    rhs = np.array([tolerance-residual, tolerance+residual])
    bounds = [(-float(x), float(x)) for x in h] + [(None,None)]
    c = np.r_[np.zeros(a.size), 1.]
    left = linprog(c, A_ub=A, b_ub=rhs, bounds=bounds, method='highs')
    right = linprog(-c, A_ub=A, b_ub=rhs, bounds=bounds, method='highs')
    if not left.success or not right.success:
        raise RuntimeError('LP verification failed')
    assert np.isclose(left.x[-1], lo, atol=1e-10)
    assert np.isclose(right.x[-1], hi, atol=1e-10)
    assert proxy <= radius + 1e-10  # the old bound is valid, just insufficient for ranking
    return dict(jacobian=a.tolist(), correction_half_widths=h.tolist(),
                nominal_residual=residual, task_tolerance=tolerance,
                fixed_right_inverse=b.tolist(), fixed_policy_radius=proxy,
                admissible_shift_interval=[lo,hi], exact_task_radius=radius,
                left_endpoint_witness=left.x.tolist(),
                right_endpoint_witness=right.x.tolist())


def main() -> None:
    results = {
      'scope': 'Exact linear one-dimensional mathematical counterexamples; not robot performance',
      'residual_allowance': {
        'A': evaluate([1.0], [0.6], 0.9),
        'B': evaluate([1.0], [0.4], 0.0),
      },
      'redundancy_allocation': {
        'A': evaluate([1.0,1.0], [0.1,1.0], 0.0),
        'B': evaluate([1.0,1.0], [0.4,0.4], 0.0),
      },
    }
    a,b = results['residual_allowance'].values()
    assert a['fixed_policy_radius'] > b['fixed_policy_radius']
    assert a['exact_task_radius'] < b['exact_task_radius']
    a,b = results['redundancy_allocation'].values()
    assert a['fixed_policy_radius'] < b['fixed_policy_radius']
    assert a['exact_task_radius'] > b['exact_task_radius']
    results['checks_passed'] = True
    out = Path(__file__).with_name('reserve_proxy_counterexamples.json')
    out.write_text(json.dumps(results,indent=2), encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    main()
