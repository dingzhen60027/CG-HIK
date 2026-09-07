import numpy as np
from confik.config import load_config, load_robot
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.types import IKQuery
from confik.continuation_mechanism.disentangle_math import (
    linear_minimum_step, uniform_refine, matched_projection, MATCH_POSITION, MATCH_ORIENTATION)
from confik.continuation_mechanism.observation import refine_candidate


def setup():
    cfg=load_config('configs/paper_v2.yaml'); kin=load_robot(cfg,'panda')
    v=SolutionVerifier(kin,VerifierConfig(**cfg['verifier']))
    q=(kin.limits.lower+kin.limits.upper)/2
    return kin,v,q


def test_demand_uses_redundancy_and_pose_tolerance():
    a=np.zeros((6,7)); a[0,0]=1; a[0,6]=1
    r=linear_minimum_step(a,np.array([3.,0,0,0,0,0]),-np.ones(7)*10,np.ones(7)*10)
    assert r['optimum_reported'] and abs(r['demand']-1)<1e-5
    r=linear_minimum_step(a,np.array([3.,0,0,0,0,0]),-np.ones(7)*10,np.array([.2,10,10,10,10,10,10]))
    assert r['optimum_reported'] and abs(r['demand']-1.8)<1e-5


def test_demand_allows_zero_step_within_pose_contract():
    a=np.zeros((6,7));a[:6,:6]=np.eye(6)
    r=linear_minimum_step(a,np.array([.5,0,0,0,0,.5]),-np.ones(7),np.ones(7))
    assert r['optimum_reported'] and r['demand']<1e-6


def test_uniform_refinement_keeps_same_objective_budget_and_fallback():
    kin,v,q=setup(); target=q.copy();target[0]+=.001
    query=IKQuery(kin.forward(target),q,.02)
    old=refine_candidate(kin,v,query,q); new=uniform_refine(kin,v,query,q)
    assert np.allclose(old['q'],new['refined_q'],atol=1e-12,rtol=0)
    assert v.check(np.asarray(new['q']),query).accepted
    assert new['residual_calls']>=new['scipy_nfev'] and new['scipy_nfev']<=200
    exact=IKQuery(kin.forward(q),q,.02)
    result=uniform_refine(kin,v,exact,q)
    assert not result['improved'] and np.array_equal(result['q'],q)


def test_projection_matches_actual_pose_and_common_step():
    kin,v,q=setup(); query=IKQuery(kin.forward(q),q,.02)
    projected=matched_projection(kin,v,query,q,.25)
    assert projected['usable']
    assert projected['position_match_error']<=MATCH_POSITION
    assert projected['orientation_match_error']<=MATCH_ORIENTATION
    assert v.check(np.asarray(projected['q']),query).accepted
    assert projected['max_configuration_difference']>=1e-5
