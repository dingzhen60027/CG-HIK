"""Research prototype: two-block minimax GN on a fixed current-frame box.

Not a new claim about minimax optimization. A scalar dual selects the weights of
TWO task residual blocks; they are not hand-tuned priorities. Only the current
joint configuration is optimized. No solver fallback or future information.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from bounded_gn_reference import box_qp


def worst_squared(e):
    return max(float(e[:3] @ e[:3]), float(e[3:] @ e[3:]))


def box_normal_residual(d, grad, lo, hi):
    # Exact boundary membership for a valid normal-cone subgradient.
    r = grad.copy()
    r[d == lo] = np.minimum(r[d == lo], 0.)
    r[d == hi] = np.maximum(r[d == hi], 0.)
    r[lo == hi] = 0.
    return r


def minimax_step(e, G, lo, hi, damping=.01, theta0=.5,
                 max_dual_updates=16, gap_atol=1e-9, gap_rtol=1e-7,
                 posture=None, posture_scale=None, posture_weight=0., acceptable=None):
    """Solve min_d max_b ||e_b+G_b d||² + damping/2||d||², lo<=d<=hi.

    Standard convex duality reduces its two epigraph multipliers to theta in
    [0,1]. Every primal trial is feasible. Bounds include a conservative
    strong-convexity correction for numerical inner-QP stationarity error.
    A finite cap returns an inexact, explicitly labelled direction, not a
    false claim of exact subproblem solution.
    """
    if damping <= 0 or not np.isfinite(damping): raise ValueError('damping>0 required')
    e=np.asarray(e,float); G=np.asarray(G,float)
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if e.shape!=(6,) or G.ndim!=2 or G.shape[0]!=6 or lo.shape!=(G.shape[1],) or hi.shape!=lo.shape:
        raise ValueError('Inconsistent residual, Jacobian or box shapes')
    if not all(np.isfinite(v).all() for v in (e,G,lo,hi)) or np.any(lo>hi):
        raise ValueError('Nonfinite data or empty box')
    if max_dual_updates<1 or posture_weight<0:
        raise ValueError('Invalid iteration cap or posture weight')
    n=G.shape[1]; eye=np.eye(n)
    P0=2*G[:3].T@G[:3]; P1=2*G[3:].T@G[3:]
    c0=2*G[:3].T@e[:3]; c1=2*G[3:].T@e[3:]
    posture=np.zeros(n) if posture is None else np.asarray(posture,float)
    posture_scale=np.zeros(n) if posture_scale is None else np.asarray(posture_scale,float)
    common_H=damping*eye+posture_weight*np.diag(posture_scale**2)
    common_g=posture_weight*posture_scale*posture
    C=P0-P1; c=c0-c1
    theta=float(np.clip(theta0,0.,1.)); left=0.; right=1.
    upper=np.inf; lower=-np.inf; best=None; qp_updates=0; history=[]
    for k in range(max_dual_updates):
        H=theta*P0+(1.-theta)*P1+common_H
        g=theta*c0+(1.-theta)*c1+common_g
        d, nit=box_qp(H,g,lo,hi); qp_updates+=int(nit)
        if acceptable is not None and acceptable(d):
            return d, dict(theta=theta,dual_updates=k+1,box_qp_updates=qp_updates,
                primal_upper=None,dual_lower=None,gap=None,converged=False,
                task_feasible_early=True,trace=history)
        r0=e[:3]+G[:3]@d; r1=e[3:]+G[3:]@d
        f0=float(r0@r0); f1=float(r1@r1)
        regularizer=.5*damping*float(d@d)+.5*posture_weight*float(np.sum((posture+posture_scale*d)**2))
        primal=max(f0,f1)+regularizer
        normres=box_normal_residual(d,H@d+g,lo,hi)
        dual=theta*f0+(1.-theta)*f1+regularizer-float(normres@normres)/(2*damping)
        if primal < upper:
            upper=primal; best=d.copy(); besttheta=theta
        lower=max(lower,dual); gap=max(0.,upper-lower)
        diff=f0-f1
        history.append(dict(theta=theta,primal=primal,dual_lower=dual,
                            task_squared=[f0,f1],inner_residual=float(np.linalg.norm(normres)),
                            gap=gap))
        if gap <= gap_atol+gap_rtol*max(1.,abs(upper),abs(lower)):break
        if diff>0: left=theta
        elif diff<0:right=theta
        else:break
        # Dual Newton curvature with the CURRENT locally constant active face.
        free=np.where((d>lo+1e-10)&(d<hi-1e-10))[0]
        z=C@d+c
        curvature=0.
        if len(free):
            zf=z[free]; curvature=-float(zf@np.linalg.solve(H[np.ix_(free,free)],zf))
        proposed=theta-diff/curvature if curvature < -1e-18 else .5*(left+right)
        proposed=float(np.clip(proposed,0.,1.))
        if not (left < proposed < right):
            # Endpoints are allowed once, so single-block optima are recognized.
            if proposed==0. and left==0. and theta!=0.: pass
            elif proposed==1. and right==1. and theta!=1.:pass
            else:proposed=.5*(left+right)
        if abs(proposed-theta)<1e-14:proposed=.5*(left+right)
        theta=proposed
    return best, dict(theta=besttheta,dual_updates=len(history),box_qp_updates=qp_updates,
        primal_upper=upper,dual_lower=lower,gap=max(0.,upper-lower),
        converged=upper-lower <= gap_atol+gap_rtol*max(1.,abs(upper),abs(lower)),task_feasible_early=False,trace=history)


@dataclass(frozen=True)
class Settings:
    damping: float=.01
    posture_weight: float=1.
    early_feasible: bool=True
    max_iterations: int=30
    backtracking_steps: int=8
    max_dual_updates: int=16
    # Optional diagnostic allowance; caller must measure total adapter time.
    deadline_ms: float=20.


def solve_balanced(previous, dt, lower, upper, velocity, evaluate, verification,
                   settings=None, velocity_tolerance=1e-4, keep_trace=False):
    cfg=settings or Settings(); start=perf_counter_ns()
    if min(cfg.max_iterations,cfg.backtracking_steps,cfg.max_dual_updates)<1 or cfg.posture_weight<0:
        raise ValueError('Invalid numerical settings')
    previous=np.asarray(previous,float); n=len(previous)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);velocity=np.asarray(velocity,float)
    if not (lower.shape==upper.shape==velocity.shape==previous.shape):raise ValueError('shape mismatch')
    if not np.isfinite(dt) or dt<=0 or not all(np.isfinite(v).all() for v in (previous,lower,upper,velocity)):
        raise ValueError('invalid input')
    if np.any(velocity<=0) or np.any(upper<=lower):raise ValueError('Invalid joint limits')
    S=velocity*dt+velocity_tolerance
    lo=np.maximum(lower+1e-12,previous-S*(1-1e-12))
    hi=np.minimum(upper-1e-12,previous+S*(1-1e-12))
    if np.any(lo>hi):raise ValueError('empty interval')
    q=np.clip(previous,lo,hi);lam=cfg.damping;theta=.5
    trace=[]; evaluations=0;updates=0;duals=0;status='iteration_limit';gaps=[]
    bestq=q.copy();best=np.inf
    for it in range(cfg.max_iterations):
        if (perf_counter_ns()-start)/1e6 > cfg.deadline_ms:status='deadline';break
        e,J=evaluate(q); evaluations+=1;f=worst_squared(e)
        if f<best:best=f;bestq=q.copy()
        legal=bool(verification(q))
        if keep_trace:trace.append(dict(iteration=it,q=q.tolist(),rho=float(np.sqrt(f)),
                             norms=[float(np.linalg.norm(e[:3])),float(np.linalg.norm(e[3:]))],accepted=legal))
        if legal:bestq=q.copy();status='task_admissible';break
        G=J*S
        posture=(q-.5*(lower+upper))/(upper-lower)
        def acceptable(d):
            nonlocal evaluations
            trial=np.clip(q+S*d,lo,hi)
            ee,_=evaluate(trial);evaluations+=1
            return bool(np.linalg.norm(ee[:3])<=1 and np.linalg.norm(ee[3:])<=1 and verification(trial))
        d,info=minimax_step(e,G,np.maximum((lo-q)/S,-1.),np.minimum((hi-q)/S,1.),
                      damping=lam,theta0=.5,max_dual_updates=cfg.max_dual_updates,
                      posture=posture,posture_scale=S/(upper-lower),posture_weight=cfg.posture_weight,
                      acceptable=acceptable if cfg.early_feasible else None)
        theta=info['theta'];updates+=info['box_qp_updates'];duals+=info['dual_updates'];gaps.append(info['gap'])
        if keep_trace:trace[-1]['subproblem']=info
        if info['task_feasible_early']:
            q=np.clip(q+S*d,lo,hi);bestq=q.copy();status='task_admissible';break
        predicted_decrease=f+.5*cfg.posture_weight*float(posture@posture)-info['primal_upper']
        if not np.isfinite(d).all():status='nonfinite_direction';break
        if np.linalg.norm(d)<1e-12:status='stationary';break
        adopted=False
        for k in range(cfg.backtracking_steps):
            a=2.**(-k); trial=np.clip(q+a*S*d,lo,hi)
            en,_=evaluate(trial);evaluations+=1;fn=worst_squared(en)
            # Do not bisect to the first feasible point: accept the full proposal
            # whenever possible. All acceptance is still original contract.
            if verification(trial) or (predicted_decrease>0 and fn<=f-1e-4*a*predicted_decrease):
                q=trial;lam=max(cfg.damping/100,lam*.5);adopted=True;break
        if not adopted:
            lam*=10
            if lam>1e8:status='no_descent';break
    e,_=evaluate(q);evaluations+=1
    if worst_squared(e)<best:bestq=q.copy()
    legal=bool(verification(bestq))
    if legal:status='task_admissible'
    er,_=evaluate(bestq);evaluations+=1
    return dict(q=bestq,accepted=legal,status=status,iterations=it+1,
        evaluations=evaluations,dual_updates=duals,box_qp_updates=updates,
        rho=float(np.sqrt(worst_squared(er))),
        position_normalized=float(np.linalg.norm(er[:3])),orientation_normalized=float(np.linalg.norm(er[3:])),
        subproblem_gaps=gaps,elapsed_ns=perf_counter_ns()-start,trace=trace)
