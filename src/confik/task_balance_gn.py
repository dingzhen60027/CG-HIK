"""Two-block task minimax GN: the supplied scalar-dual numerical mode.

Convex duality and the box-QP are established components. No other IK solver,
future target, narrower public tolerance or adaptive policy is used here.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from .bounded_gn import box_qp
from .correction_reserve.native_geometry import NativeGeometry
from .correction_reserve.geometry import residual_linearization,task_scale
from .types import IKQuery,Pose


def worst_squared(e):
    return max(float(e[:3]@e[:3]),float(e[3:]@e[3:]))


def box_normal_residual(d,grad,lo,hi):
    r=grad.copy()
    r[d==lo]=np.minimum(r[d==lo],0.)
    r[d==hi]=np.maximum(r[d==hi],0.)
    r[lo==hi]=0.
    return r


def minimax_step(e,G,lo,hi,damping=.01,theta0=.5,max_dual_updates=16,
                 gap_atol=1e-9,gap_rtol=1e-7,posture=None,posture_scale=None,
                 posture_weight=0.,acceptable=None,deadline_ns=None):
    """One convex local problem, solved through its scalar dual.

    Lower bounds include an inner-stationarity correction using H >= damping I.
    These are floating-point diagnostics, not interval-arithmetic certificates.
    An early task-accepted step has no invented gap or optimality claim.
    """
    if damping<=0 or not np.isfinite(damping):raise ValueError('damping>0 required')
    e=np.asarray(e,float);G=np.asarray(G,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if e.shape!=(6,) or G.ndim!=2 or G.shape[0]!=6 or lo.shape!=(G.shape[1],) or hi.shape!=lo.shape:
        raise ValueError('Inconsistent dimensions')
    if not all(np.isfinite(v).all() for v in (e,G,lo,hi)) or np.any(lo>hi):raise ValueError('Invalid data or box')
    if max_dual_updates<1 or posture_weight<0:raise ValueError('Invalid cap or posture weight')
    n=G.shape[1];eye=np.eye(n)
    P0=2*G[:3].T@G[:3];P1=2*G[3:].T@G[3:]
    c0=2*G[:3].T@e[:3];c1=2*G[3:].T@e[3:]
    posture=np.zeros(n) if posture is None else np.asarray(posture,float)
    posture_scale=np.zeros(n) if posture_scale is None else np.asarray(posture_scale,float)
    common_H=damping*eye+posture_weight*np.diag(posture_scale**2)
    common_g=posture_weight*posture_scale*posture
    C=P0-P1;c=c0-c1
    theta=float(np.clip(theta0,0,1));left=0.;right=1.
    upper=np.inf;lower=-np.inf;best=None;besttheta=theta;updates=0;history=[];timed_out=False
    for k in range(max_dual_updates):
        if deadline_ns is not None and perf_counter_ns()>=deadline_ns:
            timed_out=True;break
        H=theta*P0+(1-theta)*P1+common_H;g=theta*c0+(1-theta)*c1+common_g
        d,nit=box_qp(H,g,lo,hi);updates+=int(nit)
        if not np.isfinite(d).all() or max(np.max(lo-d),np.max(d-hi))>1e-10:
            return best,dict(theta=besttheta,dual_updates=k+1,box_qp_updates=updates,
                primal_upper=None if best is None else upper,dual_lower=None,gap=None,
                converged=False,task_feasible_early=False,deadline=False,status='invalid_qp_direction',trace=history)
        # Deadline is absolute and shared with the adapter and outer loop.
        if acceptable is not None and (deadline_ns is None or perf_counter_ns()<deadline_ns) and acceptable(d):
            history.append(dict(theta=theta,task_feasible_early=True,primal=None,dual_lower=None,gap=None))
            return d,dict(theta=theta,dual_updates=k+1,box_qp_updates=updates,primal_upper=None,
                dual_lower=None,gap=None,converged=False,task_feasible_early=True,deadline=False,
                status='task_feasible_early',trace=history)
        r0=e[:3]+G[:3]@d;r1=e[3:]+G[3:]@d;f0=float(r0@r0);f1=float(r1@r1)
        reg=.5*damping*float(d@d)+.5*posture_weight*float(np.sum((posture+posture_scale*d)**2))
        primal=max(f0,f1)+reg
        normal=box_normal_residual(d,H@d+g,lo,hi)
        dual=theta*f0+(1-theta)*f1+reg-float(normal@normal)/(2*damping)
        if primal<upper:upper=primal;best=d.copy();besttheta=theta
        lower=max(lower,dual);gap=max(0.,upper-lower);diff=f0-f1
        history.append(dict(theta=theta,primal=primal,dual_lower=dual,task_squared=[f0,f1],
            inner_residual=float(np.linalg.norm(normal)),gap=gap,raw_gap=upper-lower))
        if gap<=gap_atol+gap_rtol*max(1.,abs(upper),abs(lower)):break
        if diff>0:left=theta
        elif diff<0:right=theta
        else:break
        free=np.where((d>lo+1e-10)&(d<hi-1e-10))[0];z=C@d+c;curvature=0.
        if len(free):
            zf=z[free];curvature=-float(zf@np.linalg.solve(H[np.ix_(free,free)],zf))
        proposed=theta-diff/curvature if curvature< -1e-18 else .5*(left+right)
        proposed=float(np.clip(proposed,0,1))
        if not(left<proposed<right):
            if proposed==0 and left==0 and theta!=0:pass
            elif proposed==1 and right==1 and theta!=1:pass
            else:proposed=.5*(left+right)
        if abs(proposed-theta)<1e-14:proposed=.5*(left+right)
        theta=proposed
    if best is None:
        return None,dict(theta=theta,dual_updates=len(history),box_qp_updates=updates,
            primal_upper=None,dual_lower=None,gap=None,converged=False,task_feasible_early=False,
            deadline=timed_out,status='deadline' if timed_out else 'no_direction',trace=history)
    converged=upper-lower<=gap_atol+gap_rtol*max(1.,abs(upper),abs(lower))
    return best,dict(theta=besttheta,dual_updates=len(history),box_qp_updates=updates,
        primal_upper=upper,dual_lower=lower,gap=max(0.,upper-lower),raw_gap=upper-lower,
        converged=converged,task_feasible_early=False,deadline=timed_out,
        status='deadline' if timed_out else 'gap_converged' if converged else 'inexact',trace=history)


@dataclass(frozen=True)
class Settings:
    damping:float=.01
    posture_weight:float=1.
    max_iterations:int=30
    backtracking_steps:int=8
    max_dual_updates:int=16
    deadline_ms:float=20.


class TaskBalanceGN:
    def __init__(self,kin,verifier,urdf,posture_weight=1.,trace=False):
        self.kin,self.verifier=kin,verifier
        self.native=NativeGeometry(kin,urdf);self.scale=task_scale(verifier)
        self.settings=Settings(posture_weight=posture_weight);self.trace=trace
        self.method='task_balance_k'+str(int(posture_weight))

    def reset(self,q):pass
    def close(self):pass

    def solve(self,position,rotation,previous,dt=.02):
        start=perf_counter_ns();cfg=self.settings;deadline=start+int(cfg.deadline_ms*1e6)
        query=IKQuery(Pose(np.asarray(position,float),np.asarray(rotation,float)),np.asarray(previous,float),dt)
        p=query.previous_q;lower=self.kin.limits.lower;upper=self.kin.limits.upper
        S=self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        lo=np.maximum(lower+1e-12,p-S*(1-1e-12));hi=np.minimum(upper-1e-12,p+S*(1-1e-12))
        if dt<=0 or not np.isfinite([*lo,*hi]).all() or np.any(lo>hi):raise ValueError('Invalid interval')
        q=np.clip(p,lo,hi);lam=cfg.damping;bestq=q.copy();best=np.inf
        evals=checks=updates=duals=iterations=backtracks=0
        early=False;early_first=False;extended=False;status='iteration_limit';trace=[];infos=[]
        def expired():return perf_counter_ns()>=deadline
        def evaluate(x):
            nonlocal evals
            evals+=1;e,J,_=residual_linearization(self.native,query.target,x,self.scale);return e,J
        def verify(x):
            nonlocal checks
            checks+=1;return self.verifier.check(x,query)
        for it in range(cfg.max_iterations):
            if expired():status='deadline';break
            iterations+=1;e,J=evaluate(q);f=worst_squared(e)
            if f<best:best=f;bestq=q.copy()
            legal=bool(verify(q).accepted)
            row=None
            if self.trace:
                row=dict(iteration=it,q=q.tolist(),rho=float(np.sqrt(f)),e=e.tolist(),accepted=legal)
                trace.append(row)
            if legal:bestq=q.copy();status='initial_task_admissible' if it==0 else 'task_admissible';break
            G=J*S;posture=(q-.5*(lower+upper))/(upper-lower)
            def acceptable(d):
                if expired():return False
                trial=np.clip(q+S*d,lo,hi);ee,_=evaluate(trial)
                return bool(np.linalg.norm(ee[:3])<=1 and np.linalg.norm(ee[3:])<=1 and verify(trial).accepted)
            dl=np.maximum((lo-q)/S,-1.);du=np.minimum((hi-q)/S,1.)
            d,info=minimax_step(e,G,dl,du,damping=lam,theta0=.5,max_dual_updates=cfg.max_dual_updates,
                posture=posture,posture_scale=S/(upper-lower),posture_weight=cfg.posture_weight,
                acceptable=acceptable,deadline_ns=deadline)
            updates+=info['box_qp_updates'];duals+=info['dual_updates'];extended|=info['dual_updates']>1
            # Numeric trace is saved (including early-stop labels), not fabricated
            # zero optimality gaps. Full matrices are diagnostic-only.
            infos.append(info)
            if row is not None:row.update(subproblem=info,G=G.tolist(),step_lower=dl.tolist(),step_upper=du.tolist(),
                damping=lam,posture=posture.tolist(),posture_scale=(S/(upper-lower)).tolist(),direction=None if d is None else d.tolist())
            if info['task_feasible_early']:
                q=np.clip(q+S*d,lo,hi);bestq=q.copy();early=True
                early_first=it==0 and info['dual_updates']==1
                status='task_feasible_early';break
            if info['deadline'] or expired():status='deadline';break
            if d is None or not np.isfinite(d).all():status='no_valid_direction';break
            if np.linalg.norm(d)<1e-12:status='stationary';break
            predicted=f+.5*cfg.posture_weight*float(posture@posture)-info['primal_upper']
            adopted=False
            for k in range(cfg.backtracking_steps):
                if expired():status='deadline';break
                backtracks+=1;a=2.**(-k);trial=np.clip(q+a*S*d,lo,hi)
                en,_=evaluate(trial);fn=worst_squared(en)
                if verify(trial).accepted or (predicted>0 and fn<=f-1e-4*a*predicted):
                    q=trial;lam=max(cfg.damping/100,lam*.5);adopted=True
                    if row is not None:row.update(alpha=a,new_rho=float(np.sqrt(fn)),new_q=q.tolist())
                    break
            if status=='deadline':break
            if not adopted:
                lam*=10
                if lam>1e8:status='no_descent';break
        # Keep the best actual nonlinear iterate, as in the supplied prototype.
        e,_=evaluate(q)
        if worst_squared(e)<best:bestq=q.copy()
        verdict=verify(bestq);er,_=evaluate(bestq)
        if verdict.accepted and status not in ('task_feasible_early','initial_task_admissible'):status='task_admissible'
        result=dict(method=self.method,q=bestq.tolist(),accepted=bool(verdict.accepted),
            internal_status=status,internal_ok=bool(verdict.accepted),rho=float(np.sqrt(worst_squared(er))),
            position_normalized=float(np.linalg.norm(er[:3])),orientation_normalized=float(np.linalg.norm(er[3:])),
            finite=bool(verdict.finite_ok),joint_limit_ok=bool(verdict.joint_limit_ok),velocity_ok=bool(verdict.velocity_ok),
            position_error=verdict.position_error,orientation_error=verdict.orientation_error,
            verification_reasons=list(verdict.reasons),failure_kind='accepted' if verdict.accepted else '+'.join(verdict.reasons),
            velocity_utilization=float(np.max(abs(self.kin.difference(bestq,p))/S)),
            iterations=iterations,evaluations=evals,verification_calls=checks,box_qp_updates=updates,dual_updates=duals,
            backtracking_evaluations=backtracks,task_feasible_early=early,theta_half_first_step_return=early_first,
            continued_dual=extended,continued_dual_accepted=extended and bool(verdict.accepted),
            subproblems=infos)
        if self.trace:result.update(trace=trace,frame_lower=lo.tolist(),frame_upper=hi.tolist(),step_scale=S.tolist())
        result['total_latency_ns']=perf_counter_ns()-start
        result['accepted_within_20ms']=bool(verdict.accepted and result['total_latency_ns']<=20_000_000)
        return result
