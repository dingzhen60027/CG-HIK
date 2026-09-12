import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import pytest

from confik.current_frame_trf import CurrentFrameTRF, TracThenCurrentTRF
from confik.correction_reserve.study import context
from confik.correction_reserve.geometry import residual_linearization, task_scale, perturb_target


CFG={'source_config':'configs/paper_v2.yaml'}


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_scaled_analytic_jacobian(robot):
    _,kin,v,urdf=context(robot,CFG);solver=CurrentFrameTRF(kin,v,urdf)
    q=(kin.limits.lower+kin.limits.upper)/2
    scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
    target=perturb_target(kin.forward(q),np.array([.2,-.4,.3,.5,.1,-.2]),1,scale)
    e,J,_=residual_linearization(solver.geometry,target,q,scale)
    h=1e-5
    fd=np.column_stack([(residual_linearization(solver.geometry,target,q+np.eye(kin.nq)[i]*step*h,scale)[0]-
                         residual_linearization(solver.geometry,target,q-np.eye(kin.nq)[i]*step*h,scale)[0])/(2*h) for i in range(kin.nq)])
    np.testing.assert_allclose(J*step,fd,rtol=1e-6,atol=2e-6)


@pytest.mark.parametrize('index',[1,2,3,4])
def test_supplied_recoverable_inputs(index):
    case=json.loads(Path('outputs/current_frame_recovery/task_package/recorded_inputs.json').read_text())[index]
    _,kin,v,urdf=context('panda',CFG);solver=CurrentFrameTRF(kin,v,urdf)
    result=solver.solve(case['p'],case['R'],case['previous'])
    assert result['accepted']
    assert result['native_nfev']<=50 and result['counts']['residual_evaluations']<=50
    assert result['velocity_utilization']<=1 and result['max_joint_step_rad']>0
    assert result['accounting_remainder_ns']>=0
    assert result['iteration_trace'][-1]['accepted']
    assert result['internal_status']=='task_admissible' and result['native_return_code']==-2


def test_previous_command_and_expired_period():
    _,kin,v,urdf=context('ur5e',CFG);s=CurrentFrameTRF(kin,v,urdf)
    q=(kin.limits.lower+kin.limits.upper)/2;p=kin.forward(q)
    r=s.solve(p.position,p.rotation,q)
    assert r['accepted'] and not r['optimizer_called']
    np.testing.assert_array_equal(r['q'],q)
    r=s.solve(p.position+np.array([.1,0,0]),p.rotation,q,deadline_ns=perf_counter_ns()-1)
    assert not r['optimizer_called'] and r['internal_status']=='deadline_before_solve'


def test_composition_one_recovery_same_input_and_absolute_deadline():
    calls=[]
    class Trac:
        def solve(self,p,R,q,dt):return dict(accepted=False,q=q,method='trac_task_5ms')
    class Recovery:
        def solve(self,p,R,q,dt,*,deadline_ns):
            calls.append((p,R,q,dt,deadline_ns));return dict(accepted=True,q=q)
    solver=TracThenCurrentTRF(Trac(),Recovery());q=np.zeros(7);p=np.zeros(3);R=np.eye(3)
    before=perf_counter_ns();r=solver.solve(p,R,q)
    assert len(calls)==1 and calls[0][2] is q and calls[0][0] is p
    assert before<calls[0][4]<=perf_counter_ns()+20_000_000
    assert r['accepted'] and r['recovery_called']
    solver.trac=SimpleNamespace(solve=lambda *args:dict(accepted=True,q=q))
    r=solver.solve(p,R,q);assert r['accepted'] and not r['recovery_called'] and len(calls)==1


def test_no_recovery_after_expired_trac(monkeypatch):
    import confik.current_frame_trf as module
    clock=iter([0,21_000_000,21_000_000,22_000_000])
    monkeypatch.setattr(module,'perf_counter_ns',lambda:next(clock))
    trac=SimpleNamespace(solve=lambda *args:dict(accepted=False,q=None))
    recovery=SimpleNamespace(solve=lambda *a,**kw:pytest.fail('must not call TRF after deadline'))
    r=TracThenCurrentTRF(trac,recovery).solve(np.zeros(3),np.eye(3),np.zeros(6))
    assert not r['recovery_called'] and not r['accepted'] and r['remaining_before_trf_ns']==0
