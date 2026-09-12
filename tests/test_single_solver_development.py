"""Integration checks, not additional trajectory experiments or parameter scans."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('single_study',ROOT/'scripts/run_single_solver_development.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
study.configure_backend()
from confik.bounded_gn import BoundedGN,Settings,box_qp
from confik.correction_reserve.geometry import residual_linearization,task_scale
import yaml


def test_package_sources_unchanged():
    study.check_supplied_sources()


def test_box_qp_kkt():
    rng=np.random.default_rng(2026091211)
    for n in (6,7):
        for _ in range(10):
            A=rng.normal(size=(10,n));H=A.T@A+.1*np.eye(n);g=5*rng.normal(size=n)
            lo=-rng.uniform(.01,1,n);hi=rng.uniform(.01,1,n)
            x,it=box_qp(H,g,lo,hi)
            assert np.all(x>=lo) and np.all(x<=hi) and it<=50
            assert np.max(np.abs(x-np.clip(x-(H@x+g),lo,hi)))<1e-7


@pytest.mark.parametrize('robot',['panda','ur5e'])
@pytest.mark.parametrize('method',['single_gn_k0','single_gn_k1'])
def test_adapter_contract_and_scaled_derivative(robot,method):
    cfg=yaml.safe_load((ROOT/'configs/single_solver_development.yaml').read_text())
    solver,kin,v=study.factory(method,robot,cfg)
    q=(kin.limits.lower+kin.limits.upper)/2
    pose=kin.forward(q);result=solver.solve(pose.position,pose.rotation,q,.02)
    assert result['accepted'] and result['total_latency_ns']>0
    np.testing.assert_allclose(result['q'],q,atol=0,rtol=0)
    target=kin.forward(q+.1*(kin.limits.velocity*.02+v.config.velocity_tolerance))
    _,A,_=residual_linearization(solver.native,target,q,task_scale(v))
    h=1e-6
    J=np.column_stack([(residual_linearization(solver.native,target,q+np.eye(kin.nq)[j]*h,task_scale(v))[0]-
                       residual_linearization(solver.native,target,q-np.eye(kin.nq)[j]*h,task_scale(v))[0])/(2*h) for j in range(kin.nq)])
    np.testing.assert_allclose(A,J,atol=2e-6,rtol=2e-6)
    result=solver.solve(target.position,target.rotation,q,.02)
    from confik.types import IKQuery
    verdict=v.check(np.array(result['q']),IKQuery(target,q,.02))
    assert verdict.accepted==result['accepted']
    assert result['iterations']<=30
    assert not any(k in result for k in ('trac_result','trf_result','nominal_next_verified'))
    solver.close()


def test_verifier_has_final_acceptance():
    engine=BoundedGN([-1.]*6,[1.]*6,[1.]*6,settings=Settings())
    out=engine.solve(np.zeros(6),.02,lambda q:(np.zeros(6),np.eye(6)),lambda q:SimpleNamespace(accepted=False))
    assert out['internal_status']=='task_candidate' and not out['accepted']
