"""Causal demand, exact backup preservation and nonlinear acceptance tests."""
from time import perf_counter
import numpy as np
import pytest

from confik.config import load_config, load_robot, resolve_path
from confik.types import Pose, IKQuery
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.correction_reserve.minimal_demand import PredictionDemand, empirical_quantile
from confik.correction_reserve.minimal_runtime import MinimalInterventionIK


def test_empirical_order_statistic_and_no_demand_clipping():
    assert empirical_quantile(range(1,21),.95)==19
    d=PredictionDemand(np.ones(6),.1,.2)
    for i in range(25):
        _,r=d.observe(Pose([1000.*i*i,0,0],np.eye(3)))
    assert r['demand']>1000 and r['demand_history_count']==20


def test_startup_fixed_and_prefix_causality():
    a=PredictionDemand(np.ones(6),.1,.2)
    b=PredictionDemand(np.ones(6),.1,.2,fixed=True)
    target=Pose(np.zeros(3),np.eye(3))
    for i in range(21):
        pa,ra=a.observe(target);pb,rb=b.observe(target)
        np.testing.assert_array_equal(pa.matrix,pb.matrix)
        assert ra['observed_eta']==rb['observed_eta']
        assert rb['demand']==.2
        assert ra['demand']==(.2 if i<20 else .1)
    a.reset();assert a.forecast is None and not a.errors
    with pytest.raises(ValueError):PredictionDemand(np.ones(6),0,0,window=10)


@pytest.fixture(params=['panda','ur5e'])
def case(request):
    source=load_config('configs/paper_v2.yaml');kin=load_robot(source,request.param)
    verifier=SolutionVerifier(kin,VerifierConfig(**source['verifier']))
    q=kin.random_configuration(np.random.default_rng(301),margin=.2)
    urdf=resolve_path(source,source['robots'][request.param]['urdf'])
    return source,kin,verifier,q,urdf


def make(case,mode='minimal',demand=1e-6):
    source,kin,v,_,urdf=case
    return MinimalInterventionIK(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf,
        demand_parameters={'minimum':demand,'initial':demand},mode=mode)


def test_good_pair_shortcut_never_builds_cone_and_preserves_exact_backup(case):
    _,kin,v,q,_=case;s=make(case)
    try:
        target=kin.forward(q)
        for _ in range(2):
            r=s.solve(target.position,target.rotation,q)
            assert r['accepted'] and r['demand_met'] and r['direct_return']
            assert np.array_equal(r['q'],r['backup_q']) and r['intervention_normalized_l2']==0
            assert not r['optimization_called'] and s.cone is None
            assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
            q=np.array(r['q'])
        assert r['warm_start_used']
    finally:s.close()


def test_no_shortcut_solves_but_preserves_zero_objective_and_updates_cache(case):
    _,kin,_,q,_=case;s=make(case,'no_shortcut')
    try:
        target=kin.forward(q)
        for _ in range(2):
            r=s.solve(target.position,target.rotation,q)
            assert r['accepted'] and r['demand_met'] and not r['direct_return']
            assert r['optimization_called'] and r['conic_calls']<=2
            assert np.array_equal(r['q'],r['backup_q'])
        assert s.cone.creations==1 and s.cone.updates>=1
    finally:s.close()


def test_unsatisfied_large_demand_returns_backup_without_claim(case):
    _,kin,v,q,_=case;s=make(case,demand=1e9)
    try:
        target=kin.forward(q);r=s.solve(target.position,target.rotation,q)
        assert r['accepted'] and not r['demand_met'] and r['demand']==1e9
        assert np.array_equal(r['q'],r['backup_q']) and r['conic_calls']<=2
        assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
    finally:s.close()


def test_failed_backup_uses_verified_predictive_recovery(case):
    _,kin,v,q,_=case;s=make(case,demand=1e9)
    original=s.backup.solve
    def failed(*args,**kwargs):
        result=original(*args,**kwargs)
        result.update(accepted=False,finite=False,q=None,internal_ok=False,
                      verification_reasons=['injected_missing_command'])
        return result
    s.backup.solve=failed
    try:
        target=kin.forward(q);r=s.solve(target.position,target.rotation,q)
        assert r['recovery_mode'] and r['accepted'] and not r['demand_met']
        assert r['conic_calls']<=2
        assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
    finally:s.close()


def test_returned_demand_is_rechecked_nonlinearly_and_timing_includes_work(case):
    _,kin,v,q,_=case;s=make(case,demand=.1)
    try:
        for i in range(4):
            target=kin.forward(q+.001)
            r=s.solve(target.position,target.rotation,q)
            assert r['conic_calls']<=2
            phases=['conversion_ns','backup_ns','demand_estimation_ns','initial_pair_ns',
                    'cone_construction_ns','cone_solve_ns','nonlinear_validation_ns']
            assert all(r[k]>=0 for k in phases)
            assert sum(r[k] for k in phases)<=r['total_latency_ns']
            if r['accepted']:
                assert v.check(np.array(r['q']),IKQuery(target,q,.02)).accepted
            if r['demand_met']:
                step=kin.limits.velocity*.02+v.config.velocity_tolerance
                pred=Pose(r['predicted_position'],r['predicted_rotation'])
                check=s.pair_check(np.array(r['q']),np.array(r['nominal_next_q']),IKQuery(target,q,.02),pred,step,r['demand'])
                assert check['legal'] and check['demand_met']
            if r['accepted']:q=np.array(r['q'])
    finally:s.close()
