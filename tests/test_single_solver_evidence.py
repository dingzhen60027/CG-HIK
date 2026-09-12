import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('old_entry',ROOT/'scripts/run_single_solver_development.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert old.configure_backend()['engine']=='numpy_supplied_fallback'
from confik.bounded_gn import BoundedGN,box_qp
from confik.single_solver_evidence import bind_qp,clipped_qp,CachedOSQP,quality,outer_counts


def test_same_qp_and_cached_pattern():
    H=np.array([[4.,2.],[2.,3.]]);g=np.array([-6.,-1.])
    lo=np.array([-.2,-.4]);hi=np.array([.3,.4])
    a,_=box_qp(H,g,lo,hi);s=CachedOSQP(2)
    for matrix in (H,np.diag([4.,3.]),H):
        s.reset();x,_=s(matrix,g,lo,hi);ref,_=box_qp(matrix,g,lo,hi)
        np.testing.assert_allclose(x,ref,atol=1e-7)
        assert quality(matrix,g,lo,hi,x)['normalized_kkt']<1e-8
    assert len(s.solver._derivative_cache['P'].data)==3
    np.testing.assert_array_equal(clipped_qp(H,g,lo,hi)[0],np.clip(np.linalg.solve(H,-g),lo,hi))
    assert quality(H,g,lo,hi,clipped_qp(H,g,lo,hi)[0])['objective']>quality(H,g,lo,hi,a)['objective']


def test_private_callback_and_default_path():
    a=BoundedGN([-2.,-2.],[2.,2.],[1.,1.]);b=BoundedGN([-2.,-2.],[2.,2.],[1.,1.])
    original=BoundedGN.solve.__globals__['box_qp'];captured=[]
    def callback(H,g,lo,hi):
        captured.append((H.copy(),g.copy(),lo.copy(),hi.copy()))
        return original(H,g,lo,hi)
    bind_qp(b,callback)
    def evaluate(q):
        return np.r_[100*(q-.01),np.zeros(4)],np.vstack([100*np.eye(2),np.zeros((4,2))])
    verify=lambda q:SimpleNamespace(accepted=True)
    aa=a.solve(np.zeros(2),.02,evaluate,verify);bb=b.solve(np.zeros(2),.02,evaluate,verify)
    assert captured and BoundedGN.solve.__globals__['box_qp'] is original
    np.testing.assert_array_equal(aa['q'],bb['q'])
    assert aa['evaluations']==bb['evaluations']
    assert outer_counts(aa)['qp_calls']==len(captured)
    bind_qp(b,clipped_qp);b.solve(np.zeros(2),.02,evaluate,verify)
    np.testing.assert_array_equal(aa['q'],a.solve(np.zeros(2),.02,evaluate,verify)['q'])


def test_osqp_reset_and_bound_roundoff():
    solver=CachedOSQP(3);H=np.eye(3);g=np.array([100.,-100.,2.])
    lo=np.array([0.,-.1,-.2]);hi=np.array([.1,.1,.2])
    x,_=solver(H,g,lo,hi);assert np.all(x>=lo) and np.all(x<=hi)
    solver.reset();y,_=solver(H,g,lo,hi)
    np.testing.assert_allclose(x,y,atol=1e-9)
