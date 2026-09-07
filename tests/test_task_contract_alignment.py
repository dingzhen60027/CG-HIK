from dataclasses import asdict
from pathlib import Path
import ctypes as ct
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from confik.config import load_config,load_robot,resolve_path
from confik.task_contract_alignment.contract import METHODS,verifier_for,native_mapping,task_interval
from confik.task_contract_alignment.outcomes import ContractSolver,taxonomy
from confik.continuation_mechanism.tolerance_solvers import ToleranceSolver,feedback
from confik.types import Pose,IKQuery


def setup(robot='panda',scale=1.):
    c=load_config('configs/paper_v2.yaml');k=load_robot(c,robot)
    return c,k,verifier_for(k,c,scale)


@pytest.mark.parametrize('robot',['panda','ur5e'])
@pytest.mark.parametrize('scale',[.5,1.,2.])
def test_mapping_conservative_and_scale_changes_only_pose(robot,scale):
    c,k,v=setup(robot,scale);original=verifier_for(k,c)
    for m in METHODS[:6]:
        mapping=native_mapping(m,v);b=np.array(mapping['effective_component_limits'])
        assert np.linalg.norm(b[:3])<=v.config.position_tolerance
        assert np.linalg.norm(b[3:])<=v.config.orientation_tolerance
        assert mapping['frame']=='target frame'
    assert v.config.velocity_tolerance==original.config.velocity_tolerance
    assert v.config.joint_limit_tolerance==original.config.joint_limit_tolerance


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_dls_observation_does_not_change_original_solver(robot):
    c,k,v=setup(robot);q=(k.limits.lower+k.limits.upper)/2;p=k.forward(q+.003)
    for m in ['dls_strict','dls_task']:
        a=ContractSolver(m,k,v,c);b=ToleranceSolver(m,k,v,c)
        result=a.solve(p.position,p.rotation,q,trace=True);old=b.solve(p.position,p.rotation,q)
        assert np.array_equal(result['q'],old['q'])
        assert result['iterations']==old['iterations']
        assert result['solver_function_evaluations']==old['solver_function_evaluations']
        trace=result['dls_trace']
        assert len(trace['evaluations'])==result['solver_function_evaluations']
        assert trace['evaluations'][0]['is_iterate']
        assert result['total_latency_ns']==sum(result[k] for k in ['conversion_ns','bounds_setup_ns','solve_ns','verification_ns','accounting_remainder_ns'])


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_native_smoke_kinematics_mapping_and_exact_targets(robot):
    lib='tmp/task_contract_build/libcontract_trac.so'
    assert Path(lib).exists(),'native adapter must be built before protocol smoke'
    c,k,v=setup(robot);rng=np.random.default_rng(971008800)
    for m in METHODS[:6]:
        s=ContractSolver(m,k,v,c,lib,resolve_path(c,c['robots'][robot]['urdf']))
        try:
            for _ in range(5):
                q=k.random_configuration(rng,.2);p=k.forward(q)
                np_,nr=s.native.forward(q)
                assert np.max(np.abs(np_-p.position))<1e-12
                assert np.max(np.abs(nr-p.rotation))<1e-12
                result=s.solve(p.position,p.rotation,q)
                assert result['accepted'] and result['internal_ok']
                assert result['accounting_remainder_ns']>=0
        finally:s.close()


def test_real_native_rotation_vector_not_rpy():
    c,k,v=setup();lib='tmp/task_contract_build/libcontract_trac.so'
    s=ContractSolver('trac_task_5ms',k,v,c,lib,resolve_path(c,c['robots']['panda']['urdf']))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    s.native.lib.tolerance_relative_error.argtypes=[ptr]*5+[ct.c_double,ptr]
    rng=np.random.default_rng(971008801)
    try:
        for _ in range(50):
            p=rng.normal(size=3);r=Rotation.random(random_state=rng).as_matrix()
            e=rng.uniform(-.99,.99,6)*s.native.bounds
            actual_p=np.ascontiguousarray(p+r@e[:3]);actual_r=np.ascontiguousarray(r@Rotation.from_rotvec(e[3:]).as_matrix())
            out=np.empty(6)
            rc=s.native.lib.tolerance_relative_error(p,np.ascontiguousarray(r),actual_p,actual_r,s.native.bounds,1e-5,out)
            assert rc==1 and np.max(np.abs(out-e))<1e-12
    finally:s.close()


def test_three_success_semantics_and_feedback():
    assert len({taxonomy(i,a) for i in [False,True] for a in [False,True]})==4
    q=np.arange(7.)
    assert np.array_equal(feedback(q,dict(q=q+1,accepted=False)),q)
    # Internal status has no veto over an admissible returned command.
    assert np.array_equal(feedback(q,dict(q=q+.01,accepted=True,internal_ok=False)),q+.01)


def test_existing_joint_tolerance_is_not_silently_changed():
    c,k,v=setup();assert v.config.joint_limit_tolerance==1e-9
    q=(k.limits.lower+k.limits.upper)/2;p=k.forward(q)
    query=IKQuery(p,q,.02);lo,hi=task_interval(k,query,v)
    allowed=k.limits.velocity*.02+v.config.velocity_tolerance
    assert np.all(np.abs(lo-q)<=allowed) and np.all(np.abs(hi-q)<=allowed)
    assert np.all(lo>=k.limits.lower) and np.all(hi<=k.limits.upper)


def test_repetition_is_not_an_independent_unit_and_no_false_deadline_success():
    from confik.task_contract_alignment.aggregate import unit_table,summarize
    rows=[dict(uid='a',repeat=i,family='local',accepted=True,internal_ok=i==0,
        total_latency_ns=21_000_000,returned_within_20ms=False,returned_within_5ms=False,
        position_error=0.,orientation_error=0.,verification_reasons=[]) for i in range(3)]
    assert len(unit_table(rows))==1
    summary=summarize(rows)
    assert summary['units']==1 and summary['calls']==3
    assert summary['accepted_within_20ms']==0 and summary['verified_success']==1
    assert summary['internal_failure__task_accept']==2
