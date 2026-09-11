"""Linear mathematical tests for task-admissible recourse, NOT robot benchmarks.

Run: python test_linear_recourse.py --output linear_recourse_results.json
Dependencies: numpy, scipy. Each check uses one LP containing all recourse nodes.
No fixed inverse or nested per-scenario optimization is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def task_shortfall(j, half_widths, residual, tolerance, perturbations):
    """min d >= 0 s.t. |e+w_j-J u_j| <= tolerance*(1+d), |u_ji| <= h_i."""
    J = np.asarray(j, dtype=float)
    h = np.asarray(half_widths, dtype=float)
    w = np.asarray(perturbations, dtype=float)
    if (J.ndim != 1 or h.shape != J.shape or J.size == 0 or
        w.ndim != 1 or w.size == 0 or np.any(h < 0) or tolerance <= 0 or
        not np.all(np.isfinite(np.r_[J, h, w, residual, tolerance]))):
        raise ValueError('Expected finite scalar-task data, matching nonnegative bounds, and positive tolerance.')
    n, m = J.size, w.size
    A, b = [], []
    for k, wk in enumerate(w):
        row = np.zeros(m*n+1)
        row[k*n:(k+1)*n] = -J
        row[-1] = -tolerance
        A.append(row); b.append(tolerance-residual-wk)
        row = np.zeros(m*n+1)
        row[k*n:(k+1)*n] = J
        row[-1] = -tolerance
        A.append(row); b.append(tolerance+residual+wk)
    c = np.zeros(m*n+1); c[-1] = 1.
    bounds = [(-float(hi), float(hi)) for _ in range(m) for hi in h]+[(0., None)]
    result = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(f'LP failed: {result.message}')
    u = result.x[:-1].reshape(m,n); d = float(result.x[-1])
    errors = residual + w - u@J
    if np.any(np.abs(errors) > tolerance*(1+d)+1e-8):
        raise AssertionError('Returned recourse does not satisfy the modeled pose bound.')
    if np.any(np.abs(u) > h+1e-8):
        raise AssertionError('Returned recourse exceeds its joint bounds.')
    return dict(shortfall=d, corrections=u.tolist(), residuals=errors.tolist(),
                perturbations=w.tolist())


def shared_current_test(shared=True):
    """Toy joint problem: current q in [-.5,.5], next |z-q|<=.5;
    targets +/-1.2; next tolerance .5. One shared current has optimal d=.4.
    Allowing separate current decisions is anticipatory and spuriously gives d=0.
    """
    targets = [-1.2, 1.2]
    nq = 1 if shared else 2
    size = nq+2+1; A=[]; b=[]
    for k,t in enumerate(targets):
        qi = 0 if shared else k; zi=nq+k
        row=np.zeros(size); row[zi]=1; row[qi]=-1
        A.append(row); b.append(.5)
        A.append(-row); b.append(.5)
        row=np.zeros(size); row[zi]=1; row[-1]=-.5
        A.append(row); b.append(.5+t)
        row=np.zeros(size); row[zi]=-1; row[-1]=-.5
        A.append(row); b.append(.5-t)
    c=np.zeros(size); c[-1]=1
    r=linprog(c, A_ub=np.array(A),b_ub=np.array(b),
              bounds=[(-.5,.5)]*nq+[(None,None)]*2+[(0,None)],method='highs')
    if not r.success: raise RuntimeError(r.message)
    return dict(shared_current=shared, shortfall=float(r.x[-1]),
                current=r.x[:nq].tolist(), next=r.x[nq:-1].tolist())


def run():
    # Prior counterexample 1: old fixed inverse favors A; actual task recourse favors B.
    a=task_shortfall([1.],[.6],.9,1.,[-1.,0.,1.])
    b=task_shortfall([1.],[.4],0.,1.,[-1.,0.,1.])
    assert np.isclose(a['shortfall'],.3,atol=1e-9)
    assert np.isclose(b['shortfall'],0.,atol=1e-9)
    # Prior counterexample 2: unrestricted redundant correction reverses the fixed-inverse ranking.
    c=task_shortfall([1.,1.],[.1,1.],0.,1.,[-2.,0.,2.])
    d=task_shortfall([1.,1.],[.4,.4],0.,1.,[-2.,0.,2.])
    assert np.isclose(c['shortfall'],0.,atol=1e-9)
    assert np.isclose(d['shortfall'],.2,atol=1e-9)
    # Rank deficiency is not automatically task failure when deviation is tolerated.
    rank0=task_shortfall([0.,0.],[.1,.1],0.,1.,[-.5,0.,.5])
    assert np.isclose(rank0['shortfall'],0.,atol=1e-9)
    shared, cheating=shared_current_test(True),shared_current_test(False)
    assert np.isclose(shared['shortfall'],.4,atol=1e-9)
    assert np.isclose(cheating['shortfall'],0.,atol=1e-9)
    # Convex interpolation check in a COMMON affine model only.
    # Every convex combination of feasible endpoint corrections remains feasible.
    u=np.array(c['corrections']); errors=[]
    for alpha in np.linspace(0.,1.,21):
        wk=alpha*(-2.)+(1-alpha)*2.
        uk=alpha*u[0]+(1-alpha)*u[2]
        ek=wk-np.sum(uk); errors.append(float(ek))
        assert abs(ek)<=1.+1e-8
        assert np.all(np.abs(uk)<=np.array([.1,1.])+1e-8)
    return dict(scope='Linear scalar-task mathematical tests only; not Panda/UR5e performance or a novelty claim.',
        tolerance_example=dict(A=a,B=b),redundancy_example=dict(A=c,B=d),
        rank_deficient_tolerance_example=rank0,
        nonanticipativity=dict(correct=shared,incorrect_separate_current=cheating),
        affine_convex_interpolation=dict(checked_points=21,max_abs_error=max(map(abs,errors))),
        checks_passed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('linear_recourse_results.json'))
    args=parser.parse_args(); data=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2))
