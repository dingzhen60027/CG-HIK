"""Bounded elastic demand, unchanged task acceptance, and matched control tests."""
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pytest
from confik.config import load_config,load_robot,resolve_path
from confik.types import IKQuery
from confik.solvers.verifier import SolutionVerifier,VerifierConfig
from confik.correction_reserve.elastic_runtime import ElasticInterventionIK,objective,improves
from confik.correction_reserve.elastic_convex import ElasticCone
from confik.correction_reserve.convex import OptimizerConfig
from confik.continuation_mechanism.observation import representable_interior


@pytest.fixture(params=['panda','ur5e'])
def case(request):
    source=load_config('configs/paper_v2.yaml');kin=load_robot(source,request.param)
    v=SolutionVerifier(kin,VerifierConfig(**source['verifier']))
    q=kin.random_configuration(np.random.default_rng(301),margin=.2)
    return source,kin,v,q,resolve_path(source,source['robots'][request.param]['urdf'])


def make(case,mu=1.,demand=.1):
    source,kin,v,_,urdf=case
    return ElasticInterventionIK(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf,
        demand_parameters={'minimum':demand,'initial':demand},mu=mu)


def test_actual_partial_objective_is_useful_without_full_demand():
    q0=np.zeros(6);q=np.full(6,.01);step=np.ones(6)
    original,xi,_=objective(q0,q0,q0,q0,step,1.,.1,1.)
    changed,xi2,_=objective(q,q,q0,q0,step,1.,.4,1.)
    assert 0<xi2<xi and improves(changed,original)
    zero0=objective(q0,q0,q0,q0,step,1.,.1,0.)[0]
    zero1=objective(q,q,q0,q0,step,1.,.4,0.)[0]
    assert not improves(zero1,zero0)
    assert not improves(original-1e-12,original)


def test_same_variable_and_sparse_structure_for_all_mu():
    cones=[ElasticCone(7,OptimizerConfig(),mu) for mu in (0.,.25,1.,4.)]
    for cone in cones:
        assert cone.size==15 and cone.P.nnz==15
        np.testing.assert_array_equal(cone.sparse.indptr,cones[0].sparse.indptr)
        np.testing.assert_array_equal(cone.sparse.indices,cones[0].sparse.indices)
        assert cone.solver is None


def test_sufficient_pair_is_exact_backup_without_cone(case):
    _,kin,v,q,_=case;s=make(case,demand=1e-6)
    try:
        target=kin.forward(q)
        for _ in range(2):
            r=s.solve(target.position,target.rotation,q)
            assert r['direct_return'] and r['demand_met'] and not r['optimization_called']
            assert np.array_equal(r['q'],r['backup_q']) and s.cone is None
            assert r['actual_objective']==0 and r['xi_actual']==0
            assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
    finally:s.close()


def test_rank_failure_forces_full_shortfall_but_keeps_geometry(case,monkeypatch):
    import confik.correction_reserve.elastic_convex as ec
    _,kin,v,q,_=case;s=make(case)
    try:
        pose=kin.forward(q);step=kin.limits.velocity*.02+v.config.velocity_tolerance
        lo,hi=representable_interior(kin,IKQuery(pose,q,.02),v)
        monkeypatch.setattr(ec,'correction_map',lambda *a,**kw:SimpleNamespace(full_rank=False,rank=5))
        cone=ElasticCone(kin.nq,s.config,1.)
        x,status=cone.solve(s.counted,pose,pose,q,q,q,q,q,lo,hi,s.scale,step,.1,perf_counter()+.1)
        assert x is not None and abs(x[-1]-1)<1e-6 and not status['reserve_map_available']
        assert status['solver_called'] and cone.creations==1
        x,status=cone.solve(s.counted,pose,pose,q,q,q,q,q,lo,hi,s.scale,step,.1,perf_counter()+.1)
        assert x is not None and status['data_updated'] and cone.updates==1
        assert v.check(q+step*x[:kin.nq],IKQuery(pose,q,.02)).accepted
    finally:s.close()


def test_recovery_without_legal_backup_never_relaxes_task_contract(case):
    _,kin,v,q,_=case;s=make(case,demand=100.)
    original=s.backup.solve
    def fail(*args,**kwargs):
        result=original(*args,**kwargs)
        result.update(accepted=False,finite=False,q=None,internal_ok=False,verification_reasons=['injected_missing'])
        return result
    s.backup.solve=fail
    try:
        target=kin.forward(q);r=s.solve(target.position,target.rotation,q)
        assert r['recovery_mode'] and r['backup_failed_recovery'] and r['accepted']
        assert r['decision']=='geometric_recovery' and not r['demand_met'] and r['conic_calls']<=2
        assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
    finally:s.close()


@pytest.mark.parametrize('mu',[0.,.25,1.,4.])
def test_online_actual_cost_feedback_and_nonlinear_checks(case,mu):
    _,kin,v,q,_=case;s=make(case,mu=mu,demand=2.)
    origin=q.copy()
    try:
        for frame in range(6):
            target=kin.forward(origin+.001*frame)
            r=s.solve(target.position,target.rotation,q)
            assert r['conic_calls']<=2 and r['demand']==2.
            phases=['conversion_ns','backup_ns','demand_estimation_ns','initial_pair_ns','cone_construction_ns','cone_solve_ns','nonlinear_validation_ns']
            assert all(r[p]>=0 for p in phases) and sum(r[p] for p in phases)<=r['total_latency_ns']
            if r['optimized_pair_selected'] and r['initial_pair_legal']:
                assert improves(r['actual_objective'],r['initial_objective'])
                assert any(t['objective_improved'] for t in r['nonlinear_trials'])
            if mu==0 and r['initial_pair_legal']:assert r['command_unchanged']
            if r['xi_actual'] is not None:assert 0<=r['xi_actual']<=r['demand']
            if r['accepted']:
                assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
                q=np.array(r['q'])
    finally:s.close()
