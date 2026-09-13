import importlib.util
from pathlib import Path
from dataclasses import replace
from time import perf_counter_ns
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completion_entry_test',ROOT/'scripts/run_task_balance_completion.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from confik.task_balance_gn import solve_completion,CompletionSettings,relative_stop,box_lower_bound,box_qp

@pytest.mark.parametrize('forcing',[None,.25])
def test_external_rejection_has_final_authority(forcing):
    r=solve_completion(np.zeros(1),.02,np.array([-2.]),np.array([2.]),np.array([1.]),
        lambda q:(np.zeros(6),np.zeros((6,1))),lambda q:False,box_qp,CompletionSettings(forcing=forcing))
    assert not r['accepted'] and r['verification_calls']==1 and r['dual_updates']==0

@pytest.mark.parametrize('forcing',[None,.25])
def test_expired_absolute_clock_and_fixed_box(forcing):
    seen=[];previous=np.array([.99]);low=np.array([-1.]);high=np.array([1.]);velocity=np.ones(1)
    def evaluate(q):seen.append(q.copy());return np.array([2.,0,0,2.,0,0]),np.ones((6,1))
    r=solve_completion(previous,.02,low,high,velocity,evaluate,lambda q:False,box_qp,
        CompletionSettings(forcing=forcing),start_ns=perf_counter_ns()-30_000_000)
    assert r['status']=='deadline' and r['dual_updates']==0 and not r['accepted_within_20ms']
    for q in seen:assert np.all(q>=r['frame_lower']) and np.all(q<=r['frame_upper'])
    assert r['frame_upper'][0]<1 and r['frame_lower'][0]>previous[0]-.0201

@pytest.mark.parametrize('kappa',[0.,1.])
def test_first_qp_identical_to_gn(kappa):
    rng=np.random.default_rng(11);G=rng.normal(size=(6,7));e=rng.normal(size=6)*2
    lo=-np.ones(7);hi=np.ones(7);c=rng.normal(size=7);w=rng.uniform(.01,.1,7)
    a=dict(e=e,G=G,lo=lo,hi=hi,damping=.01,posture=c,posture_scale=w,posture_weight=kappa)
    expected,_=box_qp(G.T@G+.01*np.eye(7)+kappa*np.diag(w*w),G.T@e+kappa*w*c,lo,hi)
    for f in (None,.25):
        actual,info=entry.cached_step(a,f,lambda d:True)
        np.testing.assert_array_equal(actual,expected);assert info['reason']=='task_first' and info['gap'] is None

def test_invalid_bounds_never_claim_progress():
    assert relative_stop(10,2,1,.25)
    for p,u,l in [(1,2,1),(10,1,2),(10,np.inf,2),(np.nan,2,1),(10,2,np.nan)]:
        assert not relative_stop(p,u,l,.25)

def test_box_minorant_without_normal_cone_guess():
    H=np.diag([.2,1.]);g=np.array([2.,-1.]);lo=np.array([-.3,-.2]);hi=np.array([.1,.4])
    optimum=np.clip(-g/np.diag(H),lo,hi);d=lo+1e-9
    val=lambda x:.5*x@H@x+g@x
    bound=box_lower_bound(val(d),H@d+g,.2,d,lo,hi)
    assert bound<=val(optimum)+1e-14

def test_cached_modes_share_core_and_only_forcing():
    assert replace(CompletionSettings(forcing=None),forcing=.25)==CompletionSettings(forcing=.25)
