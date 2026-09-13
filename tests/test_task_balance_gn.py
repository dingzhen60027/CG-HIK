import importlib.util
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('old_excess',ROOT/'scripts/run_task_excess_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from confik.task_balance_gn import minimax_step,worst_squared,TaskBalanceGN
from confik.task_balance_reference import EpigraphReference,objective
from confik.bounded_gn import box_qp
from confik.correction_reserve.study import context


def test_worst_threshold_and_scalar():
    rng=np.random.default_rng(17)
    for _ in range(100):
        e=rng.normal(size=6)
        assert (worst_squared(e)<=1)==(np.linalg.norm(e[:3])<=1 and np.linalg.norm(e[3:])<=1)
    e=np.array([0.,0,0,1.4,0,0]);G=np.array([1.,0,0,-.5,0,0])[:,None]
    d,s=minimax_step(e,G,np.array([0.]),np.array([2.]),damping=1e-9)
    assert abs(d[0]-14/15)<1e-7 and s['gap']<1e-7


@pytest.mark.parametrize('n',[1,6,7])
@pytest.mark.parametrize('kappa',[0.,1.])
def test_symmetric_step_and_epigraph(n,kappa):
    rng=np.random.default_rng(100+n);ref=EpigraphReference(n)
    for k in range(12):
        e=rng.normal(size=6);G=rng.normal(size=(6,n));lo=-rng.uniform(.1,1,n);hi=rng.uniform(.1,1,n)
        c=rng.uniform(-.5,.5,n);w=rng.uniform(.01,.1,n);lam=.01
        H=G.T@G+lam*np.eye(n)+kappa*np.diag(w*w);g=G.T@e+kappa*w*c
        original,_=box_qp(H,g,lo,hi)
        d,info=minimax_step(e,G,lo,hi,posture=c,posture_scale=w,posture_weight=kappa,acceptable=lambda d:True)
        np.testing.assert_allclose(d,original,rtol=1e-12,atol=1e-12)
        assert info['theta']==.5 and info['task_feasible_early'] and not info['converged'] and info['gap'] is None
        d,info=minimax_step(e,G,lo,hi,posture=c,posture_scale=w,posture_weight=kappa,max_dual_updates=32)
        d2,s=ref.solve(e,G,lo,hi,posture=c,posture_scale=w,posture_weight=kappa)
        assert abs(info['primal_upper']-s['objective'])/(1+abs(s['objective']))<2e-6
        assert info['dual_lower']<=s['objective']+1e-7*(1+s['objective'])
        assert s['raw_box_violation']<1e-7
        assert k==0 or s['updated']


def test_deadline_does_not_invent_gap_or_command():
    d,s=minimax_step(np.ones(6),np.ones((6,7)),-np.ones(7),np.ones(7),deadline_ns=0)
    assert d is None and s['deadline'] and s['gap'] is None and not s['converged']


def test_dual_derivative_and_fixed_face_curvature():
    rng=np.random.default_rng(912);G=rng.normal(size=(6,3));e=rng.normal(size=6)
    P0=2*G[:3].T@G[:3];P1=2*G[3:].T@G[3:]
    g0=2*G[:3].T@e[:3];g1=2*G[3:].T@e[3:];theta=.43;h=1e-5
    def dual(t):
        H=t*P0+(1-t)*P1+.1*np.eye(3);g=t*g0+(1-t)*g1
        d=np.linalg.solve(H,-g);r=e+G@d
        return t*(r[:3]@r[:3])+(1-t)*(r[3:]@r[3:])+.05*(d@d),d,H
    v,d,H=dual(theta);r=e+G@d;z=(P0-P1)@d+g0-g1
    np.testing.assert_allclose((dual(theta+h)[0]-dual(theta-h)[0])/(2*h),r[:3]@r[:3]-r[3:]@r[3:],rtol=1e-7)
    np.testing.assert_allclose((dual(theta+h)[0]-2*v+dual(theta-h)[0])/h**2,-z@np.linalg.solve(H,z),rtol=2e-5)


def test_inexact_cap_is_not_optimality():
    e=np.array([0.,0,0,1.4,0,0]);G=np.array([1.,0,0,-.5,0,0])[:,None]
    d,s=minimax_step(e,G,np.array([0.]),np.array([2.]),max_dual_updates=1)
    assert s['status']=='inexact' and not s['converged'] and s['gap']>0 and not s['task_feasible_early']


@pytest.mark.parametrize('robot',['panda','ur5e'])
@pytest.mark.parametrize('kappa',[0.,1.])
def test_online_stationary_is_unchanged(robot,kappa):
    cfg=yaml.safe_load((ROOT/'configs/task_excess_development.yaml').read_text())
    _,kin,v,urdf=context(robot,cfg);solver=TaskBalanceGN(kin,v,urdf,kappa)
    q=.5*(kin.limits.lower+kin.limits.upper);t=kin.forward(q)
    r=solver.solve(t.position,t.rotation,q,.02)
    assert r['accepted'] and r['dual_updates']==0 and r['box_qp_updates']==0
    np.testing.assert_array_equal(q,r['q'])
