"""Offline local-QP references only; never installed into the main IK method."""
import ctypes as ct
import sys
from time import perf_counter_ns
import numpy as np
from .single_solver_evidence import quality


def qualified(v):
    return (v['box_violation'] <= 1e-8 and v['projected_kkt'] <= 1e-5
            and v['normalized_kkt'] <= 1e-8)


def activity(H,g,lo,hi,x):
    """Strong KKT-active constraints, weak contacts separate; 0-based joints."""
    v=quality(H,g,lo,hi,x); grad=H@x+g
    lower=(np.abs(x-lo)<=1e-8); upper=(np.abs(x-hi)<=1e-8)
    # Sign of the multiplier plus non-negligible absolute gradient, not proximity alone.
    state=np.zeros(len(x),int)
    state[lower & (grad>1e-9)]=-1
    state[upper & (grad< -1e-9)]=1
    return dict(quality=v,qualified=qualified(v),state=state.tolist(),
                active_count=int(np.count_nonzero(state)),
                weak_contacts=int(np.sum((lower|upper)&(state==0))),gradient=grad.tolist())


def active_set_trace(fun,H,g,lo,hi):
    """Trace the actual unchanged Python function OUTSIDE IK and timing runs."""
    events=[]; last=None
    def trace(frame,event,arg):
        nonlocal last
        if frame.f_code is not fun.__code__:return None
        if event=='line' and 'state' in frame.f_locals:
            now=frame.f_locals['state'].copy()
            if last is not None and np.any(now!=last):
                for i in np.where(now!=last)[0]:
                    events.append(dict(iteration=frame.f_locals.get('it',-1),joint=int(i),
                                       before=int(last[i]),after=int(now[i])))
            last=now
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace); x,it=fun(H,g,lo,hi)
    finally:sys.settrace(previous)
    return x,it,events


class QPOases:
    """Cached C++ SQProblem with changing H, double-buffered shallow matrix data."""
    def __init__(self,n,library):
        start=perf_counter_ns(); self.n=n
        self.lib=ct.CDLL(str(library)); L=self.lib
        ptr=ct.POINTER(ct.c_double)
        L.qpr_create.argtypes=[ct.c_int]; L.qpr_create.restype=ct.c_void_p
        L.qpr_destroy.argtypes=[ct.c_void_p];L.qpr_reset.argtypes=[ct.c_void_p]
        L.qpr_solve.argtypes=[ct.c_void_p]+[ptr]*5+[ct.POINTER(ct.c_int),ct.POINTER(ct.c_longlong)]
        L.qpr_solve.restype=ct.c_int
        self.p=L.qpr_create(n);self.ptr=ptr
        self.x=np.empty(n);self.info=np.empty(4,np.int32);self.times=np.empty(3,np.int64)
        self.xptr=self.x.ctypes.data_as(ptr)
        self.iptr=self.info.ctypes.data_as(ct.POINTER(ct.c_int))
        self.tptr=self.times.ctypes.data_as(ct.POINTER(ct.c_longlong))
        self.initialization_ns=perf_counter_ns()-start
        self.last={}
    def reset(self):self.lib.qpr_reset(self.p)
    def __call__(self,H,g,lo,hi):
        start=perf_counter_ns()
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in (H,g,lo,hi)]
        pointers=[a.ctypes.data_as(self.ptr) for a in arrays]
        converted=perf_counter_ns()
        self.lib.qpr_solve(self.p,*pointers,self.xptr,self.iptr,self.tptr)
        solved=perf_counter_ns();x=self.x.copy()
        self.last=dict(conversion_ns=converted-start,bridge_ns=solved-converted,
            matrix_copy_ns=int(self.times[0]),native_solve_ns=int(self.times[1]),
            native_result_ns=int(self.times[2]),status=int(self.info[0]),
            primal_status=int(self.info[2]),initial_solve=bool(self.info[3]),
            iterations=int(self.info[1]),callback_ns=perf_counter_ns()-start)
        return x,int(self.info[1])
    def close(self):
        if self.p:self.lib.qpr_destroy(self.p);self.p=None
