"""Research prototype: two-block minimax GN on a fixed current-frame box.

Not a new claim about minimax optimization. A scalar dual selects the weights of
TWO task residual blocks; they are not hand-tuned priorities. Only the current
joint configuration is optimized. No solver fallback or future information.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from bounded_gn_reference import box_qp


def half_worst_squared(e):
    return .5 * max(float(e[:3] @ e[:3]), float(e[3:] @ e[3:]))


def box_normal_residual(d, grad, lo, hi):
    # Exact boundary membership for a valid normal-cone subgradient.
    r = grad.copy()
    r[d == lo] = np.minimum(r[d == lo], 0.)
    r[d == hi] = np.maximum(r[d == hi], 0.)
    r[lo == hi] = 0.
    return r


def minimax_step(e, G, lo, hi, damping=.01, theta0=.5,
                 max_dual_updates=16, gap_atol=1e-9, gap_rtol=1e-7):
    """Solve min_d max_b .5||e_b+G_b d||² + damping/2||d||², lo<=d<=hi.

    Standard convex duality reduces its two epigraph multipliers to theta in
    [0,1]. Every primal trial is feasible. Bounds include a conservative
    strong-convexity correction for numerical inner-QP stationarity error.
    A finite cap returns an inexact, explicitly labelled direction, not a
    false claim of exact subproblem solution.
    """
    if damping <= 0 or not np.isfinite(damping): raise ValueError('damping>0 required')
    n=G.shape[1]; eye=np.eye(n)
    P0=G[:3].T@G[:3]; P1=G[3:].T@G[3:]
    c0=G[:3].T@e[:3]; c1=G[3:].T@e[3:]
    C=P0-P1; c=c0-c1
    theta=float(np.clip(theta0,0.,1.)); left=0.; right=1.
    upper=np.inf; lower=-np.inf; best=None; qp_updates=0; history=[]
    for k in range(max_dual_updates):
        H=theta*P0+(1.-theta)*P1+damping*eye
        g=theta*c0+(1.-theta)*c1
        d, nit=box_qp(H,g,lo,hi); qp_updates+=int(nit)
        r0=e[:3]+G[:3]@d; r1=e[3:]+G[3:]@d
        f0=.5*float(r0@r0); f1=.5*float(r1@r1)
        regularizer=.5*damping*float(d@d)
        primal=max(f0,f1)+regularizer
        normres=box_normal_residual(d,H@d+g,lo,hi)
        dual=theta*f0+(1.-theta)*f1+regularizer-float(normres@normres)/(2*damping)
        if primal < upper:
            upper=primal; best=d.copy(); besttheta=theta
        lower=max(lower,dual); gap=max(0.,upper-lower)
        diff=f0-f1
        history.append(dict(theta=theta,primal=primal,dual_lower=dual,
                            task_half_squared=[f0,f1],inner_residual=float(np.linalg.norm(normres)),
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
        converged=upper-lower <= gap_atol+gap_rtol*max(1.,abs(upper),abs(lower)),trace=history)


@dataclass(frozen=True)
class Settings:
    damping: float=.01
    max_iterations: int=30
    backtracking_steps: int=8
    max_dual_updates: int=16
    # Optional diagnostic allowance; caller must measure total adapter time.
    deadline_ms: float=20.


def solve_balanced(previous, dt, lower, upper, velocity, evaluate, verification,
                   settings=None, velocity_tolerance=1e-4, keep_trace=False):
    cfg=settings or Settings(); start=perf_counter_ns()
    previous=np.asarray(previous,float); n=len(previous)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);velocity=np.asarray(velocity,float)
    if not (lower.shape==upper.shape==velocity.shape==previous.shape):raise ValueError('shape mismatch')
    if not np.isfinite(dt) or dt<=0 or not all(np.isfinite(v).all() for v in (previous,lower,upper,velocity)):
        raise ValueError('invalid input')
    S=velocity*dt+velocity_tolerance
    lo=np.maximum(lower+1e-12,previous-S*(1-1e-12))
    hi=np.minimum(upper-1e-12,previous+S*(1-1e-12))
    if np.any(lo>hi):raise ValueError('empty interval')
    q=np.clip(previous,lo,hi);lam=cfg.damping;theta=.5
    trace=[]; evaluations=0;updates=0;duals=0;status='iteration_limit';gaps=[]
    bestq=q.copy();best=np.inf
    for it in range(cfg.max_iterations):
        if (perf_counter_ns()-start)/1e6 > cfg.deadline_ms:status='deadline';break
        e,J=evaluate(q); evaluations+=1;f=half_worst_squared(e)
        if f<best:best=f;bestq=q.copy()
        legal=bool(verification(q))
        if keep_trace:trace.append(dict(iteration=it,q=q.tolist(),rho=float(np.sqrt(2*f)),
                             norms=[float(np.linalg.norm(e[:3])),float(np.linalg.norm(e[3:]))],accepted=legal))
        if legal:bestq=q.copy();status='task_admissible';break
        G=J*S
        d,info=minimax_step(e,G,np.maximum((lo-q)/S,-1.),np.minimum((hi-q)/S,1.),
                      damping=lam,theta0=theta,max_dual_updates=cfg.max_dual_updates)
        theta=info['theta'];updates+=info['box_qp_updates'];duals+=info['dual_updates'];gaps.append(info['gap'])
        if keep_trace:trace[-1]['subproblem']=info
        predicted_decrease=f-info['primal_upper']
        if not np.isfinite(d).all():status='nonfinite_direction';break
        if np.linalg.norm(d)<1e-12:status='stationary';break
        adopted=False
        for k in range(cfg.backtracking_steps):
            a=2.**(-k); trial=np.clip(q+a*S*d,lo,hi)
            en,_=evaluate(trial);evaluations+=1;fn=half_worst_squared(en)
            # Do not bisect to the first feasible point: accept the full proposal
            # whenever possible. All acceptance is still original contract.
            if verification(trial) or (predicted_decrease>0 and fn<=f-1e-4*a*predicted_decrease):
                q=trial;lam=max(cfg.damping/100,lam*.5);adopted=True;break
        if not adopted:
            lam*=10
            if lam>1e8:status='no_descent';break
    e,_=evaluate(q);evaluations+=1
    if half_worst_squared(e)<best:bestq=q.copy()
    legal=bool(verification(bestq))
    if legal:status='task_admissible'
    er,_=evaluate(bestq);evaluations+=1
    return dict(q=bestq,accepted=legal,status=status,iterations=it+1,
        evaluations=evaluations,dual_updates=duals,box_qp_updates=updates,
        rho=float(np.sqrt(2*half_worst_squared(er))),
        position_normalized=float(np.linalg.norm(er[:3])),orientation_normalized=float(np.linalg.norm(er[3:])),
        subproblem_gaps=gaps,elapsed_ns=perf_counter_ns()-start,trace=trace)
