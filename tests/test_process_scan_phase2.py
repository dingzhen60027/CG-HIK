import inspect
import numpy as np
from confik.process_scan.phase2_graph import (diverse_indices,displacement_capacity,
    time_lower_bound,velocity_envelope,transition_controls,shortest_path,load_library)
from confik.process_scan.phase2_optimize import optimize_transition_time
from confik.process_scan.planning import spline_basis,joint_time_calibrated
from confik.process_scan.task import ScanTask


def test_acceleration_bound_has_known_rest_to_rest_solutions():
    args=(np.array([-10.]),np.array([10.]),np.array([2.]),np.array([1.]))
    for delta,expected in ((1.,2.),(9.,6.5)):
        r=time_lower_bound(np.array([0.]),np.array([delta]),*args,first=True,last=True)
        np.testing.assert_allclose(r['value_s'],expected,atol=1e-10)
        assert r['value_s']<=expected


def test_free_boundary_envelope_does_not_impose_false_stops():
    r=time_lower_bound(np.array([0.]),np.array([1.]),np.array([-10.]),np.array([10.]),np.array([2.]),np.array([1.]))
    np.testing.assert_allclose(r['value_s'],.5,atol=1e-10)
    assert displacement_capacity(1,0,0,2,1)==.25
    v=velocity_envelope(np.array([.01]),np.array([0.]),np.array([10.]),np.array([2.]),np.array([1.]))
    np.testing.assert_allclose(v,np.sqrt(.02))


def test_candidate_cap_and_determinism():
    q=np.random.default_rng(17).normal(size=(40,7))
    a=diverse_indices(q,13,np.ones(7));b=diverse_indices(q,13,np.ones(7))
    assert a==b and len(a)==12 and a[0]==13


def test_local_support_does_not_change_scan_interior():
    for direction in ('u','v'):
        t=ScanTask('plane',0,direction,'panda',.003,.001)
        b=spline_basis(t,96);a,regions=transition_controls(t,b)
        assert 0<len(a)<94
        s=np.linspace(0,1,3001);outside=np.array([not any(lo<=v<=hi for lo,hi in regions) for v in s])
        assert np.max(abs(b(s[outside])[:,a]))==0
        assert 0 not in a and 95 not in a


def test_graph_global_closure_and_shared_pruning():
    lib=dict(endpoints=[[0],[0,1],[0]],grid=np.array([0.,.5,1.]),
             layers=[np.array([[0.]]),np.array([[1.],[2.]]),np.array([[3.]])])
    def e(k,a,b,d,t):return dict(layer=k,source=a,target=b,retained=True,distance_cost=d,
                                time_bound=dict(value_s=t),witness_indices=[a,b])
    graph=dict(edges=[[e(0,0,0,1,2),e(0,0,1,2,1)],[e(1,0,0,1,2),e(1,1,0,2,1)]])
    d=shortest_path(lib,graph,'distance');t=shortest_path(lib,graph,'time')
    np.testing.assert_array_equal(d['q'].ravel(),[0,1,3]);np.testing.assert_array_equal(t['q'].ravel(),[0,2,3])


def test_no_baseline_paths_are_candidate_inputs():
    src=inspect.getsource(load_library)
    assert 'candidate_graph.json.gz' in src
    for bad in ('shared_b1','planned_path','runs/'):
        assert bad not in src


def test_shared_optimizer_keeps_validated_equations_and_settings():
    old=inspect.getsource(joint_time_calibrated);new=inspect.getsource(optimize_transition_time)
    for line in old.splitlines():
        if any(k in line for k in ('g.append(','objective=','duration=','qscale=','ipopt.','hess=','H=ca.hessian','timed=retime')) and 'all_coefficients_variable=' not in line:
            assert line in new
    assert 'fixed=np.setdiff1d' in new and 'frozen_time=' in new


def test_fixed_coefficients_cannot_bypass_joint_bounds():
    from types import SimpleNamespace
    import pytest
    adapter=SimpleNamespace(public=SimpleNamespace(limits=SimpleNamespace(lower=np.array([0.]),upper=np.array([1.]))))
    with pytest.raises(ValueError,match='SAME full-variable'):
        optimize_transition_time(None,adapter,None,np.array([[.2],[-.01],[.8]]),None,None,active_controls=[2])
