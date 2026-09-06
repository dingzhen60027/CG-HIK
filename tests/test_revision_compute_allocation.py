from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from confik.config import load_config,load_robot
from confik.release_v4_locked.artifacts import FrozenV4Policy,V4InferenceOutput
from confik.counterfactual_v4.policy import V4PolicyConfig
from confik.revision_compute_allocation.policies import ControlledPolicy
from confik.revision_compute_allocation.data import points,trajectories
from confik.revision_compute_allocation.trac_ik import allowed_interval
from confik.revision_compute_allocation.common import json_write,paired_interval
from confik.revision_compute_allocation.decomposition import trajectory_decomposition
from confik.latency_pilot_v3.benchmark import query_from_dataset
from confik.solvers.verifier import SolutionVerifier,VerifierConfig
from confik.types import IKQuery,Pose

ROOT=Path(__file__).resolve().parents[1]


def policy(variant,*,ood=False,success=.99,fail=.01,p50=(.9,.4,.5),p95=(3.,2.,1.)):
    output=V4InferenceOutput(np.full(3,success),np.array(p50),np.array(p95),fail,np.ones(4),float(ood),ood)
    backend=SimpleNamespace(infer=lambda _:output)
    return ControlledPolicy(FrozenV4Policy(backend,V4PolicyConfig(latency_tie_margin_ms=.15)),variant,(.01,.05))


@pytest.mark.parametrize('variant',['geometry_threshold','reject_only_hard','routing_only','p50_selection'])
def test_defer_unchanged(variant):
    assert policy(variant,ood=True).decide(np.zeros(9)).action=='defer'
    assert policy(variant,success=.2).decide(np.zeros(9)).action=='defer'


@pytest.mark.parametrize('variant',['geometry_threshold','reject_only_hard','p50_selection'])
def test_reject_unchanged(variant):
    assert policy(variant,success=.1,fail=.99).decide(np.zeros(9)).action=='reject'


def test_no_reject_runs_hard():
    assert policy('routing_only',success=.1,fail=.99).decide(np.zeros(9)).action=='hard'


def test_p50_changes_ranking_but_not_p95_eligibility():
    d=policy('p50_selection').decide(np.zeros(9))
    assert d.action=='medium' and d.eligible_actions==('easy','medium','hard')
    p=policy('p50_selection',p95=(3.,21.,1.))
    d=p.decide(np.zeros(9));assert d.action=='hard' and 'medium' not in d.eligible_actions


def test_p50_tie_preserves_earliest_eligible():
    assert policy('p50_selection',p50=(.51,.5,.6)).decide(np.zeros(9)).action=='easy'


@pytest.mark.parametrize('variant',['geometry_threshold','reject_only_hard'])
def test_no_unnecessary_entry_sort(variant,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('unused entry ranking')
    monkeypatch.setattr(np,'argmin',forbidden)
    assert policy(variant).decide(np.zeros(9)).action in ['easy','hard']


def test_p50_has_only_one_ranking(monkeypatch):
    original=np.argmin;calls=[]
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(np,'argmin',counted)
    assert policy('p50_selection').decide(np.zeros(9)).action=='medium'
    assert len(calls)==1


def test_geometry_both_thresholds_and_reject_only_hard():
    p=policy('geometry_threshold');f=np.zeros(9)
    assert p.decide(f).action=='easy'
    f[8]=.051;assert p.decide(f).action=='hard'
    assert policy('reject_only_hard').decide(np.zeros(9)).action=='hard'


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_new_data_witnesses_and_reference_contract(robot):
    cfg=load_config(ROOT/'configs/revision_compute_allocation.yaml');cfg['_source']=load_config(ROOT/'configs/paper_v2.yaml')
    kin=load_robot(cfg['_source'],robot);verifier=SolutionVerifier(kin,VerifierConfig(**cfg['_source']['verifier']))
    p,ids,oracle=points(kin,cfg,robot,smoke=True)
    assert len(p)==20 and len(oracle)==8
    assert len(set(r['query_hash'] for r in ids))==20
    for i in range(len(p)):
        if p.continuity_feasible[i]:assert verifier.check(p.reference_q[i],query_from_dataset(p,i)).accepted
        else:assert np.isnan(p.reference_q[i]).all() and np.linalg.norm(p.target_position[i])>ids[i]['position_reach_upper_bound_m']
    t,tids=trajectories(kin,cfg,robot,smoke=True)
    assert len(tids)==4 and len(t)==32
    for i in range(len(t)):
        assert verifier.check(t.reference_q[i],query_from_dataset(t,i)).accepted


def test_continuous_interval_crosses_representation_boundary():
    kin=SimpleNamespace(nq=2,continuous_mask=np.array([True,False]),limits=SimpleNamespace(lower=np.array([-np.pi,-2.]),upper=np.array([np.pi,2.]),velocity=np.ones(2)))
    q=IKQuery(Pose(np.zeros(3),np.eye(3)),np.array([np.pi-.001,1.999]),dt=.02)
    v=SimpleNamespace(config=SimpleNamespace(velocity_tolerance=.0001))
    low,high=allowed_interval(kin,q,v)
    assert high[0]>np.pi and high[1]==2.


def test_exclusive_output(tmp_path):
    json_write(tmp_path/'result.json',{'a':1})
    with pytest.raises(FileExistsError):json_write(tmp_path/'result.json',{'a':2})


def test_paired_units():
    center,low,high=paired_interval([2,4,8],[1,2,4],['a','b','b'],repeats=100)
    assert (center,low,high)==(2.,2.,2.)


def test_decomposition_adds_all_four_groups():
    accepted=np.array([[1,1],[0,1],[1,0],[0,0]],bool)
    data=dict(method_names=np.array(['always_hard','counterfactual_cghik_v4']),trajectory_uid=np.array(['a','b','c','d']),
              category=np.array(['x']*4),time_index=np.zeros(4,int),accepted=accepted,latency_ns=np.array([[10,11],[20,10],[30,15],[40,5]]),
              function_evaluations=np.ones((4,2),int),entry_action=np.full((4,2),'hard'),fallback_used=np.zeros((4,2),bool),reject_reason=np.full((4,2),'failed'))
    aggregate,individual=trajectory_decomposition(data,'toy')
    assert [r['trajectory_count'] for r in aggregate]==[1,1,1,1]
    assert sum(r['hard_total_latency_ns'] for r in aggregate)==100
    assert sum(r['cghik_total_latency_ns'] for r in aggregate)==41
