"""Adapter/diagnostic checks; not IK performance evidence."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence_entry',ROOT/'scripts/run_single_solver_evidence.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from confik.constrained_qp_reference import QPOases,activity,active_set_trace,qualified
from confik.bounded_gn import box_qp
from confik.single_solver_evidence import quality
LIB=ROOT/'tmp/single_solver_qpoases/build/libbox_reference.so'


def test_matrix_shift_reused_input_memory():
    s=QPOases(2,LIB);H=np.eye(2);g=np.array([-4.,1.]);lo=np.array([-1.,-1.]);hi=-lo
    try:
        for scale in [1.,10.,.5,3.,20.,1.]:
            H[:]=[[2*scale,.2*scale],[.2*scale,scale]]
            x,it=s(H,g,lo,hi);ref,_=box_qp(H,g,lo,hi)
            assert s.last['status']==0
            assert np.allclose(x,ref,atol=1e-8,rtol=0)
            assert qualified(quality(H,g,lo,hi,x))
        s.reset();x,_=s(H,g,lo,hi);assert s.last['initial_solve']
    finally:s.close()


def test_activity_requires_multiplier():
    H=np.eye(2);lo=-np.ones(2);hi=np.ones(2)
    weak=activity(H,np.array([-1.,0.]),lo,hi,np.array([1.,0.]))
    assert weak['active_count']==0 and weak['weak_contacts']==1
    strong=activity(H,np.array([-2.,0.]),lo,hi,np.array([1.,0.]))
    assert strong['state']==[1,0] and strong['qualified']


def test_trace_is_original_function():
    H=np.array([[3.,2.],[2.,3.]]);g=np.array([-6.,-1.]);lo=-np.ones(2);hi=np.ones(2)
    x,it,events=active_set_trace(box_qp,H,g,lo,hi)
    y,j=box_qp(H,g,lo,hi)
    assert np.array_equal(x,y) and it==j and events
    assert any(e['after']!=0 for e in events)


def test_official_on_changing_real_bank():
    for robot in ['panda','ur5e']:
        with np.load(ROOT/f'outputs/single_solver_evidence/qp_benchmark/{robot}_qp_matrices.npz') as b:
            arrays={k:b[k] for k in b.files}
        s=QPOases(arrays['g'].shape[1],LIB)
        try:
            for i in range(len(arrays['g'])):
                args=[arrays[k][i] for k in ['H','g','lo','hi']]
                x,_=s(*args)
                assert s.last['status']==0
                assert qualified(quality(*args,x)),(robot,i)
        finally:s.close()
