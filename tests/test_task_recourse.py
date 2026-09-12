"""The actual joint SOCP, nonlinear geometry, and causal command semantics."""
import numpy as np
import pytest
import copy
import json
from pathlib import Path

from confik.task_recourse import JointSOCP, RecourseConfig, scenario_nodes, common_model, TaskRecourseIK
from confik.correction_reserve.geometry import task_scale, perturb_target, residual_linearization
from confik.correction_reserve.study import context
from confik.geometry import pose_error
from confik.types import Pose


CFG={'source_config':'configs/paper_v2.yaml'}
MATH=RecourseConfig(native_time_limit_ms=1000,native_max_iterations=100,pose_interior=0,rate_interior=0)


def scalar_case(J,h,e,w,tolerance=1,shared=False):
    J=np.asarray(J,float);h=np.asarray(h,float);w=np.asarray(w,float)
    n,m=len(J),len(w);solver=JointSOCP(n,m,config=MATH)
    F=np.zeros((6,n));F[0]=-J/tolerance
    off=np.zeros(6);off[0]=e/tolerance
    nodes=np.zeros((m,6));nodes[:,0]=w/tolerance
    x,status=solver.solve(np.zeros(n),np.zeros(n),np.zeros((m,n)),
        np.full(n,-.5) if shared else np.zeros(n),np.full(n,.5) if shared else np.zeros(n),
        -h,h,np.zeros(6),np.zeros((6,n)),off,F,np.eye(6),nodes,
        trust=100,rate_limit=.5 if shared else 100,current_radius=1)
    z=x[n:-1].reshape(m,n)
    actual=max(0,float(np.max(np.abs(e+w-z@J)/tolerance)-1))
    assert status['status'] in ('Solved','AlmostSolved')
    assert np.max(np.abs(z)-h)<1e-7
    assert actual<=x[-1]+1e-6
    return dict(tau=float(x[-1]),actual_tau=actual,q=x[:n].tolist(),z=z.tolist(),status=status),solver


def mathematical_regressions():
    cases={}
    for name,J,h,e,w,expected in [
        ('tolerance_A',[1],[.6],.9,[-1,0,1],.3),
        ('tolerance_B',[1],[.4],0,[-1,0,1],0),
        ('allocation_A',[1,1],[.1,1],0,[-2,0,2],0),
        ('allocation_B',[1,1],[.4,.4],0,[-2,0,2],.2),
        ('rank_zero',[0,0],[.1,.1],0,[-.5,0,.5],0)]:
        result,_=scalar_case(J,h,e,w);cases[name]=result
        assert abs(result['actual_tau']-expected)<1e-5
    shared,_=scalar_case([1],[10],0,[-1.2,1.2],tolerance=.5,shared=True)
    assert abs(shared['actual_tau']-.4)<1e-6
    cases['shared_current']=shared
    # Incorrect independent current choices are known feasible, but never used
    # by JointSOCP: q=(-.5,.5), z=(-1,1) => normalized error=.4 <=1.
    assert np.all(np.abs(np.array([-1,1])-np.array([-.5,.5]))<=.5)
    cases['incorrect_independent_current']={'q':[-.5,.5],'z':[-1,1],'actual_tau':0.}
    zs=np.array(cases['allocation_A']['z'])
    errors=[]
    for a in np.linspace(0,1,21):
        z=a*zs[0]+(1-a)*zs[-1];w=a*(-2)+(1-a)*2
        assert np.all(np.abs(z)<=np.array([.1,1])+1e-7)
        errors.append(abs(w-z.sum()));assert errors[-1]<=1+1e-6
    cases['common_affine_interpolation']={'points':21,'max_error':max(errors)}
    return cases


def test_mathematics():mathematical_regressions()


def test_nodes():
    w=scenario_nodes();assert w.shape==(13,6)
    assert np.all(np.sum(np.abs(w[1:]),axis=1)==1)
    assert np.array_equal(w[0],np.zeros(6))


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_common_world_derivatives(robot):
    _,kin,v,_=context(robot,CFG)
    q=(kin.limits.lower+kin.limits.upper)/2
    scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
    target=perturb_target(kin.forward(q),np.array([.5,-.2,.3,.4,-.6,.2]),1,scale)
    e,F,C,B=common_model(kin,target,q,scale,step)
    h=1e-5
    for k in range(6):
        w=np.eye(6)[k]
        plus=pose_error(perturb_target(target,w,h,scale),kin.forward(q))/scale
        minus=pose_error(perturb_target(target,w,-h,scale),kin.forward(q))/scale
        np.testing.assert_allclose((plus-minus)/(2*h),C[:,k],atol=2e-6)
    for k in range(kin.nq):
        dq=np.eye(kin.nq)[k]*step*h
        plus=pose_error(target,kin.forward(q+dq))/scale
        minus=pose_error(target,kin.forward(q-dq))/scale
        np.testing.assert_allclose((plus-minus)/(2*h),F[:,k],atol=2e-6)


def test_fixed_rank_zero_uses_least_squares():
    solver=JointSOCP(2,3,fixed=True,config=MATH)
    nodes=np.zeros((3,6));nodes[:,0]=[-.5,0,.5]
    x,status=solver.solve(np.zeros(2),np.zeros(2),np.zeros((3,2)),
        np.zeros(2),np.zeros(2),np.full(2,-.1),np.full(2,.1),
        np.zeros(6),np.zeros((6,2)),np.zeros(6),np.zeros((6,2)),np.eye(6),nodes,B=np.zeros((2,6)),trust=100)
    assert status['status']=='Solved'
    assert x[-1]<1e-3


def test_sparse_update_keeps_structure():
    result,solver=scalar_case([1],[.6],.9,[-1,0,1])
    indices=solver.A.indices.copy();indptr=solver.A.indptr.copy()
    F=np.zeros((6,1));F[0,0]=-1
    w=np.zeros((3,6));w[:,0]=[-1,0,1]
    x,status=solver.solve(np.zeros(1),np.zeros(1),np.zeros((3,1)),
        np.zeros(1),np.zeros(1),np.array([-.4]),np.array([.4]),np.zeros(6),np.zeros((6,1)),
        np.zeros(6),F,np.eye(6),w,trust=100,rate_limit=100)
    assert solver.setup_count==1 and solver.update_count==1
    assert np.array_equal(indices,solver.A.indices) and np.array_equal(indptr,solver.A.indptr)
    assert x[-1]<1e-3


def test_zero_objective_returns_exact_backup():
    source,kin,v,urdf=context('panda',CFG)
    solver=TaskRecourseIK(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf,mode='nominal')
    q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q)
    try:
        r=solver.solve(pose.position,pose.rotation,q)
        assert r['accepted'] and not r['optimization_called']
        assert np.array_equal(r['q'],r['backup']['q'])
        assert r['all_nodes_verified'] and r['tau_actual']==0
    finally:solver.close()


@pytest.mark.parametrize('robot,site',[('panda','trajectory_00'),('ur5e','trajectory_00'),('ur5e','trajectory_22')])
def test_corrected_initialization_and_exact_anchor(robot,site):
    data=json.loads(Path(f'outputs/task_recourse/mechanism_{robot}/{site}_inputs.json').read_text())
    frozen=next(x['result']['backup'] for x in data['methods'] if x['method']=='tar_free')
    source,kin,v,urdf=context(robot,CFG)
    solver=TaskRecourseIK(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf)
    solver.backup.close()
    class Replay:
        calls=0
        def solve(self,p,R,q,dt):
            self.calls+=1
            np.testing.assert_array_equal(q,data['previous_q'])
            return copy.deepcopy(frozen)
        def close(self):pass
    replay=Replay();solver.backup=replay
    solver.last_target=Pose(np.array(data['last_position']),np.array(data['last_rotation']))
    try:
        r=solver.solve(data['target_position'],data['target_rotation'],np.array(data['previous_q']))
        assert replay.calls==1 and r['zero_cost_verified']
        assert r['objective_actual']==0 and r['tau_actual']==0
        np.testing.assert_array_equal(r['q'],frozen['q'])
        assert not np.all(np.array(r['planned_z'])==np.array(frozen['q']))
        assert all(r['planned_node_verified'])
        assert r['total_latency_ns']>=sum(r['phase_times_ns'].values())
        assert len(r['native_status'])<=4
    finally:solver.close()


def test_zero_requires_original_checks_not_small_scalar():
    # A zero-cost witness requires all true task constraints, not a tiny solver
    # epigraph. Deliberately invalid rate fails even for exact FK at its target.
    from confik.types import IKQuery
    source,kin,v,urdf=context('panda',CFG)
    solver=TaskRecourseIK(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf)
    q=(kin.limits.lower+kin.limits.upper)/2
    step=kin.limits.velocity*.02+v.config.velocity_tolerance
    zs=np.tile(q,(13,1));zs[0,0]+=1.1*step[0]
    try:
        info=solver.inspect(q,zs,IKQuery(kin.forward(q),q,.02),[kin.forward(z) for z in zs],q,step)
        assert info['tau']<1e-10 and not info['geometric']
    finally:solver.close()
