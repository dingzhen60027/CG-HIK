import importlib.util
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('set_boundary_entry',ROOT/'scripts/run_task_set_boundary.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from confik.task_set_controls import weighted_cost,weighted_hessian,weighted_gradient,FixedWeightGN,ClarabelBalance
from confik.bounded_gn_adapter import SingleBoundedGN
from confik.task_balance_gn import CompletedTaskBalanceGN,solve_completion,box_qp,CompletionSettings,completion_dual_direction
from confik.task_balance_reference import objective

@pytest.mark.parametrize('theta',[.2,.35,.5,.65,.8])
def test_weighted_coefficients_and_derivative(theta):
    rng=np.random.default_rng(901);e=rng.normal(size=6);G=rng.normal(size=(6,7));d=rng.normal(size=7)
    h=1e-5
    derivative=(weighted_cost(e+h*G@d,theta)-weighted_cost(e-h*G@d,theta))/(2*h)
    assert derivative==pytest.approx(weighted_gradient(G,e,theta)@d,rel=1e-9,abs=1e-9)
    curvature=(weighted_gradient(G,e+h*G@d,theta)-weighted_gradient(G,e-h*G@d,theta))/(2*h)
    np.testing.assert_allclose(curvature,weighted_hessian(G,theta)@d,rtol=1e-8,atol=1e-8)

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_half_weight_same_frozen_gn_and_no_global_change(robot):
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context(robot,cfg)
    original=SingleBoundedGN(kin,v,urdf);weighted=FixedWeightGN(kin,v,urdf,.5)
    rng=np.random.default_rng(511)
    for _ in range(4):
        p=kin.random_configuration(rng,.1);q=kin.clip(p+.5*kin.limits.velocity*.02*rng.uniform(-1,1,kin.nq),margin=0.)
        target=kin.forward(q);a=original.solve(target.position,target.rotation,p);b=weighted.solve(target.position,target.rotation,p)
        np.testing.assert_array_equal(a['q'],b['q'])
        for key in ('accepted','internal_status','iterations','evaluations','box_qp_updates'):assert a[key]==b[key]
    assert original.engine.solve.__func__ is entry.environment.factory.__globals__.get('unused',original.engine.__class__.solve)

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_conic_shares_exact_outer_and_first_half_opportunity(robot):
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context(robot,cfg)
    conic=ClarabelBalance(kin,v,urdf);normal=CompletedTaskBalanceGN(kin,v,urdf)
    assert conic.solve.__func__.__code__ is CompletedTaskBalanceGN.solve.__code__
    k=conic.solve.__func__.__globals__['solve_completion'];assert k.__code__ is solve_completion.__code__
    assert solve_completion.__globals__['completion_dual_direction'] is completion_dual_direction
    def forbidden(*args,**kwargs):raise AssertionError('Unnecessary conic solve')
    conic.epigraph.solve=forbidden
    p=(kin.limits.lower+kin.limits.upper)/2;target=kin.forward(p+.05*kin.limits.velocity*.02)
    a=normal.solve(target.position,target.rotation,p);b=conic.solve(target.position,target.rotation,p)
    assert a['accepted'] and b['accepted'];np.testing.assert_array_equal(a['q'],b['q'])

def test_same_conic_local_objective_matches_scalar_tight():
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context('panda',cfg);conic=ClarabelBalance(kin,v,urdf)
    rng=np.random.default_rng(916)
    for _ in range(4):
        e=rng.normal(size=6)*3;G=rng.normal(size=(6,7));lo=-np.ones(7);hi=np.ones(7);c=rng.normal(size=7)*.1;w=np.full(7,.01)
        H=G.T@G+.01*np.eye(7)+np.diag(w*w);g=G.T@e+w*c;d,nit=box_qp(H,g,lo,hi);first=(H,g,d,nit)
        args=(e,G,lo,hi,.01,1.,c,w,box_qp,first,lambda _:False,CompletionSettings(forcing=None),perf_counter_ns()+10**12)
        dc,ic=conic.direction(*args);ds,is_=completion_dual_direction(*args)
        assert ic['bounds_valid'] and ic['conic']['status']=='Solved'
        assert objective(dc,e,G,.01,c,w,1.) <= objective(ds,e,G,.01,c,w,1.)+1e-6

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_geometry_grid_witnesses_and_shared_directions(robot):
    cfg=yaml.safe_load(entry.CONFIG.read_text());cfg['development']['anchors_per_family']=1
    records=entry.generate_robot(cfg,robot,'development');assert len(records)==27
    _,kin,v,_=entry.context(robot,cfg)
    for r in records:assert v.check(np.array(r['q_witness']),entry.query_of(r)).accepted
    for anchor in {r['anchor_uid'] for r in records}:
        rows=[r for r in records if r['anchor_uid']==anchor]
        assert len(rows)==9
        for r in rows:
            np.testing.assert_array_equal(r['previous_q'],rows[0]['previous_q'])
            np.testing.assert_array_equal(r['direction_p'],rows[0]['direction_p'])
            np.testing.assert_array_equal(r['direction_R'],rows[0]['direction_R'])

def test_current_contract_not_weighted_stop():
    from types import SimpleNamespace
    cfg=yaml.safe_load(entry.CONFIG.read_text());_,kin,v,urdf=entry.context('panda',cfg)
    # A .2-position weighting cannot declare an over-tolerance position admissible.
    s=FixedWeightGN(kin,v,urdf,.2);p=(kin.limits.lower+kin.limits.upper)/2
    r=s.engine.solve(p,.02,lambda q:(np.array([1.1,0,0,0,0,0]),np.zeros((6,kin.nq))),lambda q:SimpleNamespace(accepted=False))
    assert not r['accepted'] and r['internal_status']!='task_candidate'
