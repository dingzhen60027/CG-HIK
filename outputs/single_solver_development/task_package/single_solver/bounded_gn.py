"""A single bound-constrained Gauss--Newton IK iteration, no fallback solver.

The method is an implemented numerical prototype, NOT a claimed new optimization
principle. Active-set box QPs, damping and posture regularization are established
ideas. The recorded development results refer to the accompanying JIT Panda
implementation; this portable adapter must be timed in its actual environment.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fun): return fun
        return wrap

@njit(cache=True)
def box_qp(H, g, lo, hi):
    """Primal active-set minimizer for .5*x.T H x + g.T x, lo<=x<=hi.

    H must be positive definite. Boundary variables are released by their KKT
    signs. Returns the feasible iterate and the number of active-set updates.
    """
    n=len(g)
    x=np.minimum(np.maximum(np.zeros(n),lo),hi)
    state=np.zeros(n,np.int64)
    for k in range(n):
        if abs(x[k]-lo[k])<1e-14: state[k]=-1
        if abs(x[k]-hi[k])<1e-14: state[k]=1
    for it in range(50):
        ids=np.where(state==0)[0]
        if len(ids)>0:
            grad=H@x+g
            Hf=np.empty((len(ids),len(ids))); gf=np.empty(len(ids))
            for i in range(len(ids)):
                gf[i]=grad[ids[i]]
                for j in range(len(ids)): Hf[i,j]=H[ids[i],ids[j]]
            d=np.linalg.solve(Hf,-gf)
            alpha=1.;hit=-1;side=0
            for j in range(len(ids)):
                i=ids[j]
                if d[j]>1e-14 and (hi[i]-x[i])/d[j]<alpha:
                    alpha=(hi[i]-x[i])/d[j];hit=i;side=1
                if d[j]<-1e-14 and (lo[i]-x[i])/d[j]<alpha:
                    alpha=(lo[i]-x[i])/d[j];hit=i;side=-1
            x[ids]+=max(0.,alpha)*d
            x=np.minimum(np.maximum(x,lo),hi)
            if hit>=0:
                state[hit]=side;continue
        grad=H@x+g;worst=1e-9;release=-1
        for i in range(n):
            violation=state[i]*grad[i]
            if state[i]!=0 and violation>worst:
                worst=violation;release=i
        if release<0: return x,it+1
        state[release]=0
    return x,50

@dataclass(frozen=True)
class Settings:
    damping: float=0.01
    posture_weight: float=1.0
    task_stop: float=1.0
    maximum_iterations: int=30
    deadline_ms: float=20.0

class BoundedGN:
    """Model-independent current-frame solver.

    evaluate(q) returns normalized [position; SO(3)] residual and its Jacobian.
    verification(q) must implement the unchanged, full external task contract.
    No future/reference state is accepted by this API.
    """
    def __init__(self, lower, upper, velocity, velocity_tolerance=1e-4, settings=None):
        self.lower=np.asarray(lower,dtype=float)
        self.upper=np.asarray(upper,dtype=float)
        self.velocity=np.asarray(velocity,dtype=float)
        self.velocity_tolerance=float(velocity_tolerance)
        self.settings=settings or Settings()
        if not (self.lower.shape==self.upper.shape==self.velocity.shape):
            raise ValueError('Inconsistent joint dimensions')
        if not np.all(np.isfinite([self.lower,self.upper,self.velocity])):
            raise ValueError('This prototype requires finite joint bounds')
        if np.any(self.upper<=self.lower) or np.any(self.velocity<=0):
            raise ValueError('Invalid limits')

    def solve(self, previous, dt, evaluate: Callable, verification: Callable):
        s=self.settings;started=perf_counter_ns()
        previous=np.asarray(previous,dtype=float)
        if previous.shape!=self.lower.shape or not np.isfinite(previous).all() or dt<=0:
            raise ValueError('Invalid query')
        step=self.velocity*dt+self.velocity_tolerance
        # Tiny inward margins are numerical construction, not verifier changes.
        lo=np.maximum(self.lower+1e-12,previous-step*(1-1e-12))
        hi=np.minimum(self.upper-1e-12,previous+step*(1-1e-12))
        if np.any(lo>hi): raise ValueError('Empty current step interval')
        q=np.clip(previous,lo,hi);lam=s.damping
        bestq=q.copy();bestcost=np.inf;evals=qpit=0;status='iteration_limit'
        mid=.5*(self.lower+self.upper);span=self.upper-self.lower;w=step/span
        for iteration in range(s.maximum_iterations):
            if perf_counter_ns()-started>s.deadline_ms*1e6:
                status='deadline';break
            e,A=evaluate(q);evals+=1
            e=np.asarray(e,float);A=np.asarray(A,float)
            if not np.isfinite(e).all() or not np.isfinite(A).all():
                status='nonfinite_model';break
            cost=.5*float(e@e)
            if cost<bestcost: bestcost=cost;bestq=q.copy()
            if np.linalg.norm(e[:3])<=s.task_stop and np.linalg.norm(e[3:])<=s.task_stop:
                bestq=q.copy();status='task_candidate';break
            G=A*step
            H=G.T@G+lam*np.eye(len(q))+s.posture_weight*np.diag(w*w)
            g=G.T@e+s.posture_weight*w*((q-mid)/span)
            d,ni=box_qp(H,g,np.maximum((lo-q)/step,-1.),np.minimum((hi-q)/step,1.))
            qpit+=int(ni)
            if np.linalg.norm(d)<1e-12:
                status='stationary';break
            accepted_update=False
            for k in range(8):
                candidate=q+(2.**(-k))*step*d
                er,_=evaluate(candidate);evals+=1
                if .5*float(er@er)<cost-1e-12:
                    q=candidate;lam=max(s.damping/100,lam*.5)
                    accepted_update=True;break
            if not accepted_update:
                lam*=10
                if lam>1e8:
                    status='no_descent';break
        # Include the final updated iterate if the last iteration improved it.
        # The shipped measured JIT prototype stops up to one update earlier at
        # its cap. Both are preserved; no claim of bitwise cap-path equivalence.
        eq,_=evaluate(q);evals+=1
        if .5*float(eq@eq)<bestcost:bestq=q.copy()
        verdict=verification(bestq)
        elapsed=perf_counter_ns()-started
        return dict(q=bestq,accepted=bool(verdict.accepted),verification=verdict,
                    internal_status=status,iterations=iteration+1,evaluations=evals,
                    box_qp_updates=qpit,total_latency_ns=elapsed,
                    accepted_within_20ms=bool(verdict.accepted and elapsed<=20_000_000))
