"""Current-frame, task-normalized two-block minimax GN.

Candidate contribution: relative model-decrease stopping in the scalar dual.
Cache reuse/one external final verifier are engineering, not new mathematics.
All bounds, losses and posture regularization retain the preceding experiment.
A local certificate is not nonlinear feasibility or a hard deadline guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Settings:
    damping: float = .01
    posture_weight: float = 1.
    max_iterations: int = 30
    max_dual_updates: int = 16
    max_backtracking: int = 8
    # None: original tight subproblem criterion; .25: >=80% model reduction.
    forcing: float | None = .25
    gap_atol: float = 1e-9
    gap_rtol: float = 1e-7
    armijo: float = 1e-4
    deadline_ms: float = 20.


def task_value(e: np.ndarray) -> float:
    return max(float(e[:3] @ e[:3]), float(e[3:] @ e[3:]))


def box_lower_bound(value, grad, strong_convexity, d, lo, hi):
    """Valid real-arithmetic lower bound on a strongly convex weighted QP.

    Minimize its separable quadratic lower model over the ORIGINAL box.
    Does not infer exact normal-cone membership from nearly-active coordinates.
    Floating-point implementation is a diagnostic, not interval arithmetic.
    """
    delta = np.clip(-grad / strong_convexity, lo-d, hi-d)
    return float(value + grad @ delta + .5 * strong_convexity * (delta @ delta))


def relative_stop(pzero, upper, lower, forcing, roundoff=0.):
    decrease = pzero-upper
    raw_gap = upper-lower
    # Reject inconsistent numerical bounds rather than claiming a certificate.
    valid = np.isfinite([pzero,upper,lower]).all() and raw_gap >= -roundoff
    return bool(valid and decrease>0 and max(0.,raw_gap)<=forcing*decrease)


def dual_direction(e,G,lo,hi,lam,kappa,posture,w,box_qp,first,
                   acceptable,settings,deadline,trace=False):
    """Same convex problem; exact/inexact modes differ ONLY in stopping rule."""
    n=G.shape[1]
    C0=2.*(G[:3].T@G[:3]); C1=2.*(G[3:].T@G[3:])
    c0=2.*(G[:3].T@e[:3]); c1=2.*(G[3:].T@e[3:])
    commonH=lam*np.eye(n)+kappa*np.diag(w*w); commong=kappa*w*posture
    zH=C0-C1; zg=c0-c1
    theta=.5; left=0.; right=1.
    pzero=task_value(e)+.5*kappa*float(posture@posture)
    upper=np.inf; lower=-np.inf; best=None; besttheta=.5
    qps=0; active_updates=0; history=[]; mode='cap'; gap=None
    for k in range(settings.max_dual_updates):
        if perf_counter_ns()>=deadline:mode='deadline';break
        if k==0:
            H,g,d,nit=first
        else:
            H=theta*C0+(1.-theta)*C1+commonH
            g=theta*c0+(1.-theta)*c1+commong
            d,nit=box_qp(H,g,lo,hi)
        qps+=1; active_updates+=int(nit)
        d=np.asarray(d,float)
        if not np.isfinite(d).all() or np.max(np.maximum(lo-d,d-hi))>1e-10:
            mode='invalid_qp';break
        if k>0 and acceptable(d):
            return d, dict(reason='task',theta=theta,dual_updates=qps,
                active_updates=active_updates,upper=None,lower=None,gap=None,
                pzero=pzero,model_fraction=None,trace=history)
        r0=e[:3]+G[:3]@d; r1=e[3:]+G[3:]@d
        f0=float(r0@r0);f1=float(r1@r1)
        reg=.5*lam*float(d@d)+.5*kappa*float(np.sum((posture+w*d)**2))
        primal=max(f0,f1)+reg
        h=theta*f0+(1.-theta)*f1+reg
        lb=box_lower_bound(h,H@d+g,lam,d,lo,hi)
        if primal<upper:upper=primal;best=d.copy();besttheta=theta
        lower=max(lower,lb);rawgap=upper-lower
        rounding=64*np.finfo(float).eps*max(1.,abs(upper),abs(lower),abs(pzero))
        valid=rawgap>=-rounding
        gap=max(0.,rawgap)
        reduction=pzero-upper
        fraction=(reduction/(pzero-lower)) if (valid and reduction>0 and pzero>lower) else None
        if trace:
            history.append(dict(theta=theta,upper=upper,lower=lower,raw_gap=rawgap,
                decrease=reduction,fraction_lower=fraction,fp=f0,fr=f1,
                projected_kkt=float(np.linalg.norm(d-np.clip(d-(H@d+g),lo,hi),np.inf))))
        if (settings.forcing is not None and
            relative_stop(pzero,upper,lower,settings.forcing,rounding)):
            mode='relative_progress';break
        if valid and gap<=settings.gap_atol+settings.gap_rtol*max(1.,abs(upper),abs(lower)):
            mode='tight_gap';break
        diff=f0-f1
        if diff>0:left=theta
        elif diff<0:right=theta
        else:mode='zero_dual_derivative';break
        free=np.where((d>lo+1e-10)&(d<hi-1e-10))[0]
        z=zH@d+zg; curvature=0.
        if len(free):
            curvature=-float(z[free]@np.linalg.solve(H[np.ix_(free,free)],z[free]))
        proposed=theta-diff/curvature if curvature < -1e-18 else .5*(left+right)
        proposed=float(np.clip(proposed,0.,1.))
        if not left<proposed<right:
            if not ((proposed==0 and left==0 and theta!=0) or
                    (proposed==1 and right==1 and theta!=1)):
                proposed=.5*(left+right)
        if abs(proposed-theta)<1e-14:proposed=.5*(left+right)
        theta=proposed
    reduction=pzero-upper if best is not None else None
    fraction=reduction/(pzero-lower) if (best is not None and reduction>0 and pzero>lower) else None
    return best,dict(reason=mode,theta=besttheta,dual_updates=qps,
        active_updates=active_updates,upper=None if best is None else upper,
        lower=None if not np.isfinite(lower) else lower,gap=gap,pzero=pzero,
        model_fraction=fraction,trace=history)


def solve(previous,dt,lower,upper,velocity,evaluate:Callable,verify:Callable,
          box_qp:Callable,settings:Settings|None=None,velocity_tolerance=1e-4,trace=False):
    """Model-independent core. verify is called on the FINAL selected command.

    Local norm tests can stop internal iterations but cannot grant acceptance.
    Rejected final commands remain rejected; no second IK or fallback is run.
    """
    cfg=settings or Settings();start=perf_counter_ns()
    if cfg.forcing is not None and not (0.<cfg.forcing<1.):
        raise ValueError('forcing must be None or in (0,1)')
    if cfg.damping<=0 or cfg.posture_weight<0 or min(cfg.max_iterations,cfg.max_dual_updates,cfg.max_backtracking)<1:
        raise ValueError('invalid settings')
    p=np.asarray(previous,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float);velocity=np.asarray(velocity,float)
    if p.ndim!=1 or not (p.shape==lower.shape==upper.shape==velocity.shape):raise ValueError('shape mismatch')
    if not np.isfinite(dt) or dt<=0 or not np.isfinite([p,lower,upper,velocity]).all():raise ValueError('invalid inputs')
    if np.any(upper<=lower) or np.any(velocity<=0):raise ValueError('invalid joint bounds/velocities')
    n=len(p);S=velocity*dt+velocity_tolerance
    lo=np.maximum(lower+1e-12,p-S*(1-1e-12));hi=np.minimum(upper-1e-12,p+S*(1-1e-12))
    if np.any(lo>hi):raise ValueError('empty interval')
    q=np.clip(p,lo,hi);mid=.5*(lower+upper);span=upper-lower;w=S/span
    deadline=start+int(cfg.deadline_ms*1e6);lam=cfg.damping
    evals=0;dualupdates=0;boxupdates=0;backtracks=0;iterations=0
    final_checks=0;first_accept=False;reason_counts={};infos=[]
    bestq=q.copy();best=np.inf;status='iteration_limit'
    cached_q=None;cached_e=None;cached_J=None
    def model(x):
        nonlocal evals,cached_q,cached_e,cached_J
        if cached_q is None or not np.array_equal(x,cached_q):
            e,J=evaluate(x);evals+=1
            e=np.asarray(e,float);J=np.asarray(J,float)
            if e.shape!=(6,) or J.shape!=(6,n) or not np.isfinite(e).all() or not np.isfinite(J).all():
                raise ValueError('invalid residual/Jacobian')
            cached_q=x.copy();cached_e=e;cached_J=J
        return cached_e,cached_J
    try:
        e,J=model(q)
        for it in range(cfg.max_iterations):
            iterations+=1
            if perf_counter_ns()>=deadline:status='deadline';break
            f=task_value(e)
            if f<best:best=f;bestq=q.copy()
            if f<=1.:bestq=q.copy();status='native_task_candidate';break
            G=J*S;posture=(q-mid)/span
            # Same algebraic first step as frozen GN. Extra block products are lazy.
            H=G.T@G+lam*np.eye(n)+cfg.posture_weight*np.diag(w*w)
            g=G.T@e+cfg.posture_weight*w*posture
            dl=np.maximum((lo-q)/S,-1.);du=np.minimum((hi-q)/S,1.)
            d,nit=box_qp(H,g,dl,du)
            if not np.isfinite(d).all() or np.max(np.maximum(dl-d,d-du))>1e-10:
                status='invalid_qp';break
            trial=np.clip(q+S*d,lo,hi);en,Jn=model(trial)
            # Keep evaluated trials for reuse when the chosen dual step is revisited.
            seen={d.tobytes():(trial.copy(),en.copy(),Jn.copy())}
            if task_value(en)<=1.:
                q=trial;e=en;J=Jn;bestq=q.copy();best=task_value(e)
                dualupdates+=1;boxupdates+=int(nit);first_accept=(it==0)
                status='native_task_candidate';reason_counts['task_first']=reason_counts.get('task_first',0)+1
                break
            def acceptable(candidate_d):
                if perf_counter_ns()>=deadline:return False
                key=candidate_d.tobytes()
                if key not in seen:
                    qq=np.clip(q+S*candidate_d,lo,hi);ee,JJ=model(qq)
                    seen[key]=(qq.copy(),ee.copy(),JJ.copy())
                return task_value(seen[key][1])<=1.
            selected,info=dual_direction(e,G,dl,du,lam,cfg.posture_weight,posture,w,
                box_qp,(H,g,np.asarray(d),nit),acceptable,cfg,deadline,trace)
            dualupdates+=info['dual_updates'];boxupdates+=info['active_updates']
            reason_counts[info['reason']]=reason_counts.get(info['reason'],0)+1
            if trace:infos.append(info)
            if info['reason']=='task':
                q,e,J=seen[selected.tobytes()];bestq=q.copy();best=task_value(e);status='native_task_candidate';break
            if info['reason']=='deadline' or perf_counter_ns()>=deadline:status='deadline';break
            if selected is None:status='no_direction';break
            if np.linalg.norm(selected)<1e-12:status='stationary';break
            predicted=info['pzero']-info['upper'];adopted=False
            for bt in range(cfg.max_backtracking):
                if perf_counter_ns()>=deadline:status='deadline';break
                backtracks+=1;alpha=2.**(-bt);key=(alpha*selected).tobytes()
                if key in seen:trial,en,Jn=seen[key]
                else:trial=np.clip(q+alpha*S*selected,lo,hi);en,Jn=model(trial)
                fn=task_value(en)
                if fn<=1. or (predicted>0 and fn<=f-cfg.armijo*alpha*predicted):
                    q=trial.copy();e=en.copy();J=Jn.copy();lam=max(cfg.damping/100,lam*.5)
                    adopted=True
                    if fn<best:best=fn;bestq=q.copy()
                    if fn<=1.:status='native_task_candidate'
                    break
            if status in ('native_task_candidate','deadline'):break
            if not adopted:
                lam*=10.
                if lam>1e8:status='no_descent';break
        # Only this external checker has final authority; preserve late acceptance.
        t=perf_counter_ns();verdict=verify(bestq);final_checks+=1;verify_ns=perf_counter_ns()-t
        accepted=bool(verdict.accepted) if hasattr(verdict,'accepted') else bool(verdict)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        status='numeric_error:'+str(exc);t=perf_counter_ns();verdict=verify(bestq);final_checks+=1;verify_ns=perf_counter_ns()-t
        accepted=bool(verdict.accepted) if hasattr(verdict,'accepted') else bool(verdict)
    elapsed=perf_counter_ns()-start
    return dict(q=bestq,accepted=accepted,status=status,iterations=iterations,
        evaluations=evals,dual_updates=dualupdates,box_qp_updates=boxupdates,
        verification_calls=final_checks,verification_ns=verify_ns,
        backtracking_evaluations=backtracks,reason_counts=reason_counts,
        first_half_accepted=first_accept,elapsed_ns=elapsed,
        accepted_within_20ms=bool(accepted and elapsed<=20_000_000),
        trace=infos)
