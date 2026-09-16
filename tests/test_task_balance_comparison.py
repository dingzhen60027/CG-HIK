import importlib.util
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison_entry',ROOT/'scripts/run_task_balance_comparison.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from confik.task_balance_comparison import FixedLocalBudget, CurrentOptimizer, range_objective, swamp_groove
from confik.task_balance_gn import solve_completion,completion_dual_direction,CompletedTaskBalanceGN,CompletionSettings,box_qp
from confik.correction_reserve.geometry import residual_linearization,task_scale
from confik.correction_reserve.native_geometry import NativeGeometry

@pytest.mark.parametrize('radius',[0.,.1,.7,.99,1.,1.05,2.,20.])
def test_ported_range_derivative(radius):
    h=1e-6
    if radius:
        numerical=(swamp_groove(radius+h)[0]-swamp_groove(radius-h)[0])/(2*h)
        assert swamp_groove(radius)[1]==pytest.approx(numerical,rel=1e-6,abs=1e-6)
    else: assert swamp_groove(radius)==(-1.,0.)

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_native_analytic_block_derivatives(robot):
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context(robot,cfg)
    native=NativeGeometry(kin,urdf);q=np.array(cfg['application']['initial_q'][robot])
    target=kin.forward(q+.005);scale=task_scale(v)
    e,J,_=residual_linearization(native,target,q,scale);h=1e-7
    for j in range(kin.nq):
        d=np.eye(kin.nq)[j]*h
        ep=residual_linearization(native,target,q+d,scale)[0]
        em=residual_linearization(native,target,q-d,scale)[0]
        np.testing.assert_allclose((ep-em)/(2*h),J[:,j],rtol=1e-5,atol=2e-5)
        analytic=-2*np.array([e[:3]@J[:3,j],e[3:]@J[3:,j]])
        finite=(np.array([1-ep[:3]@ep[:3],1-ep[3:]@ep[3:]])-np.array([1-em[:3]@em[:3],1-em[3:]@em[3:]]))/(2*h)
        np.testing.assert_allclose(analytic,finite,rtol=1e-5,atol=1e-4)

@pytest.mark.parametrize('count',[1,2])
def test_fixed_budget_same_outer_no_global_patch(count):
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context('panda',cfg)
    s=FixedLocalBudget(kin,v,urdf,count)
    assert s.solve.__func__.__code__ is CompletedTaskBalanceGN.solve.__code__
    assert s.solve.__func__.__globals__['solve_completion'].__code__ is solve_completion.__code__
    assert solve_completion.__globals__['completion_dual_direction'] is completion_dual_direction
    rng=np.random.default_rng(91);e=rng.normal(size=6)*4;G=rng.normal(size=(6,7))
    lo=-np.ones(7);hi=np.ones(7);w=np.full(7,.01);c=np.zeros(7)
    H=G.T@G+.01*np.eye(7)+np.diag(w*w);g=G.T@e;d,nit=box_qp(H,g,lo,hi);calls=[]
    def tracked(*a):calls.append(1);return box_qp(*a)
    x,info=s.direction(e,G,lo,hi,.01,1.,c,w,tracked,(H,g,d,nit),lambda d:False,CompletionSettings(),perf_counter_ns()+10**10)
    assert len(calls)==count-1 and info['dual_updates']==count
    assert info['gap'] is None and info['lower'] is None
    assert np.all(x>=lo) and np.all(x<=hi)

@pytest.mark.parametrize('method',['direct_sqp','range_loss_matched','fixed_qp1','fixed_qp2'])
@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_first_admissible_current_input_and_final_authority(method,robot):
    cfg=yaml.safe_load(entry.CONFIG.read_text());s,kin,v,_=entry.factory(method,robot,cfg)
    q=np.array(cfg['application']['initial_q'][robot]);pose=kin.forward(q)
    result=s.solve(pose.position,pose.rotation,q,.02)
    assert result['accepted'];np.testing.assert_array_equal(q,result['q'])
    if method in ('direct_sqp','range_loss_matched'):
        assert result['internal_status']=='task_accepted_early' and not result['internal_ok']
    pose=kin.forward(q+.005)
    result=s.solve(pose.position,pose.rotation,q,.02)
    assert v.check(np.array(result['q']),entry.old.query_of(dict(previous_q=q,target_position=pose.position,target_rotation=pose.rotation,dt=.02))).accepted==result['accepted']
