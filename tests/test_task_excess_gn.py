"""Projection identities, real derivatives, bounded descent and input semantics."""
import importlib.util
from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('old_single', ROOT/'scripts/run_single_solver_development.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.configure_backend()
from confik.task_excess_gn import projected_task_model, TaskExcessGN, dynamic_interval
from confik.bounded_gn import box_qp
from confik.correction_reserve.study import context
from confik.correction_reserve.geometry import residual_linearization
from confik.types import Pose
import yaml


def test_zero_set_and_inactive_block():
    rng=np.random.default_rng(44)
    for _ in range(300):
        e=rng.normal(size=6)*rng.uniform(.1,2)
        f,v,W=projected_task_model(e)
        assert (f==0)==(np.linalg.norm(e[:3])<=1 and np.linalg.norm(e[3:])<=1)
        for s in (0,3):
            if np.linalg.norm(e[s:s+3])<=1:
                assert not np.any(v[s:s+3]) and not np.any(W[s:s+3])
        assert np.linalg.eigvalsh(W).min()>=-1e-14
    e=np.array([1.,0,0,0,0,1.])
    assert not np.any(projected_task_model(e)[2])
    assert projected_task_model(e,1-1e-6)[0]>0


def test_gradient_metric_and_isotropic():
    e=np.array([1.2,.5,-.3,.05,.03,.04]);h=1e-6
    f,v,W=projected_task_model(e)
    eye=np.eye(6)
    grad=np.array([(projected_task_model(e+h*d)[0]-projected_task_model(e-h*d)[0])/(2*h) for d in eye])
    Hess=np.column_stack([(projected_task_model(e+h*d)[1]-projected_task_model(e-h*d)[1])/(2*h) for d in eye])
    np.testing.assert_allclose(v,grad,atol=1e-9)
    np.testing.assert_allclose(W,Hess,atol=1e-9)
    fi,vi,Wi=projected_task_model(e,isotropic=True)
    assert fi==f
    np.testing.assert_array_equal(v,vi)
    assert np.linalg.norm(W-Wi)>.5


def test_bounded_descent_and_positive_damped_metric():
    rng=np.random.default_rng(61)
    for n in (6,7):
        for _ in range(100):
            G=rng.normal(size=(6,n));e=rng.normal(size=6)*2
            _,v,W=projected_task_model(e)
            H=G.T@W@G+.01*np.eye(n);g=G.T@v
            lo=-rng.uniform(0,1,n);hi=rng.uniform(0,1,n)
            d,_=box_qp(H,g,lo,hi)
            assert np.linalg.eigvalsh(H).min()>0
            assert np.max(lo-d)<1e-9 and np.max(d-hi)<1e-9
            assert g@d+d@H@d<=1e-8*(1+abs(g@d))


def test_scalar_counterexample():
    x=.56
    e=np.array([x,0,0,1.4-.5*x,0,0])
    J=np.array([1.,0,0,-.5,0,0])[:,None]
    assert abs(float((J.T@e)[0]))<1e-14
    _,v,W=projected_task_model(e)
    assert float((J.T@v)[0])<0
    assert projected_task_model(np.array([.8,0,0,1.,0,0]))[0]==0


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_original_backend_derivative_and_dynamic_interval(robot):
    cfg=yaml.safe_load((ROOT/'configs/task_excess_development.yaml').read_text())
    _,kin,verifier,urdf=context(robot,cfg)
    solver=TaskExcessGN(kin,verifier,urdf)
    p=(kin.limits.lower+kin.limits.upper)/2
    S,lo,hi=dynamic_interval(p,.02,kin.limits,verifier.config.velocity_tolerance)
    target=kin.forward(p+.2*S)
    e,J,_=residual_linearization(solver.native,target,p,solver.scale)
    h=1e-7
    fd=np.column_stack([(residual_linearization(solver.native,target,p+h*d,solver.scale)[0]-
                        residual_linearization(solver.native,target,p-h*d,solver.scale)[0])/(2*h) for d in np.eye(kin.nq)])
    np.testing.assert_allclose(J,fd,atol=1e-5,rtol=1e-6)
    assert np.all(lo>=p-S) and np.all(hi<=p+S)
    stationary=kin.forward(p)
    out=solver.solve(stationary.position,stationary.rotation,p,.02)
    assert out['accepted'] and out['box_qp_updates']==0
    np.testing.assert_array_equal(out['q'],p)


def test_selected_input_recovers_without_other_seed():
    cfg=yaml.safe_load((ROOT/'configs/task_excess_development.yaml').read_text())
    data=json.loads((ROOT/'outputs/single_solver_evidence/task_excess_development/task_package/task_excess_gn/selected_input_results.json').read_text())['input']
    _,kin,v,urdf=context('panda',cfg)
    solver=TaskExcessGN(kin,v,urdf)
    p=np.array(data['previous_q'])
    stationary=kin.forward((kin.limits.lower+kin.limits.upper)/2)
    solver.solve(stationary.position,stationary.rotation,(kin.limits.lower+kin.limits.upper)/2,.02)
    result=solver.solve(data['target_position'],data['target_rotation'],p,data['dt'])
    assert result['accepted']
    assert result['velocity_ok'] and result['joint_limit_ok']
    assert result['position_error']<=v.config.position_tolerance
    assert result['orientation_error']<=v.config.orientation_tolerance
