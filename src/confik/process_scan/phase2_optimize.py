"""Shared Phase 2 path/time kernel; derived verbatim from validated Phase 1.5.

Only added operation: fix inactive coefficient/time increments at their initial
values. No change to objective, constraints, derivatives, or acceptance.
The historical implementation and outputs remain untouched.
"""
from time import perf_counter
import numpy as np
import casadi as ca
from scipy.sparse import csc_matrix
from .models import symbolic_fk
from .planning import numerical_grid,process_constraints,ray_center_function,dense_validate,retime


def optimize_transition_time(task,adapter,basis,initial,timing,collision,budget=30.,
                          speed_reserve=.90,grid_n=201,derivative_check=False,active_controls=None,active_intervals=None):
    """Phase 1.5 validated NLP, with optional exact fixing of inactive variables.

    Eliminate u exactly via x[k+1]-x[k]=2 ds u[k], and fixed endpoint x=0.
    With no mask, equations and solver settings equal joint_time_calibrated.
    Both Phase 2 all-variable and local runs use this one implementation.
    Final dense checks and common TOPPRA remain mandatory.
    """
    start=perf_counter();nctrl,n=initial.shape
    grid=numerical_grid(np.linspace(0,1,grid_n),task.knots,basis.t);m=len(grid);ds=np.diff(grid)
    Y=ca.MX.sym('coefficient_increment',nctrl,n);X=ca.MX.sym('scaled_squared_speed',m-2)
    qscale=.15;xscale=1/timing['duration']**2
    C=ca.DM(initial)+qscale*Y;x=xscale*ca.vertcat(0,X,0);u=(x[1:]-x[:-1])/(2*ca.DM(ds))
    B=ca.DM(csc_matrix(basis(grid)));D1=ca.DM(csc_matrix(basis(grid,nu=1)));D2=ca.DM(csc_matrix(basis(grid,nu=2)))
    Q=B@C;Qs=D1@C;Qss=D2@C;fk=symbolic_fk(adapter.public)
    c,normal,b,_,tags=task.references(grid);lim=adapter.public.limits;g=[];lb=[];ub=[];collision_indices=[]
    for k in range(m):
        g.append(process_constraints(Q[k,:].T,(c[k],normal[k],b[k]),fk))
        lb.extend([-1,-np.inf,np.cos(np.deg2rad(10)),0]);ub.extend([1,1,np.inf,np.inf])
        collision_indices.append(len(lb));g.append(collision(Q[k,:].T)/.005);lb.append(1.);ub.append(np.inf)
    for k in range(m-1):
        for i in (k,k+1):
            g.append(Qs[i,:].T**2*x[i]/ca.DM((.5*lim.velocity)**2))
            lb.extend(np.full(n,-np.inf));ub.extend(np.ones(n))
            g.append((Qss[i,:].T*x[i]+Qs[i,:].T*u[k])/2)
            lb.extend(np.full(n,-1.));ub.extend(np.ones(n))
    ray=ray_center_function(task,adapter)
    for k in range(m):
        if tags[k]:
            J=ray(Q[k,:].T)[1];g.append(ca.sumsqr(J@Qs[k,:].T)*x[k]/.198**2)
            lb.append(-np.inf);ub.append(1.)
    duration=ca.sum1(2*ca.DM(ds)/(ca.sqrt(x[:-1])+ca.sqrt(x[1:])))
    objective=duration/timing['duration'];v=ca.vertcat(ca.vec(Y),X)
    smooth_g=ca.vertcat(*[term for index,term in enumerate(g) if not(index<2*m and index%2==1)])
    g=ca.vertcat(*g)
    v0=np.r_[np.zeros(nctrl*n),np.interp(grid,timing['s'],timing['x'])[1:-1]/xscale]
    vlo=((np.tile(lim.lower,(nctrl,1))-initial)/qscale);vhi=((np.tile(lim.upper,(nctrl,1))-initial)/qscale)
    vlo[[0,-1],:]=0.;vhi[[0,-1],:]=0.
    if active_controls is not None:
        fixed=np.setdiff1d(np.arange(nctrl),np.asarray(active_controls,int))
        vlo[fixed,:]=0.;vhi[fixed,:]=0.
    vlo=np.r_[vlo.ravel(order='F'),np.full(m-2,1e-10)];vhi=np.r_[vhi.ravel(order='F'),np.full(m-2,np.inf)]
    if active_intervals is not None:
        time_active=np.array([any(a<=s<=b for a,b in active_intervals) for s in grid[1:-1]])
        frozen_time=nctrl*n+np.flatnonzero(~time_active)
        vlo[frozen_time]=v0[frozen_time];vhi[frozen_time]=v0[frozen_time]
    evaluate=ca.Function('joint_evaluate',[v],[objective,g]);gradient=ca.Function('joint_derivative',[v],[ca.gradient(objective,v),ca.jacobian(g,v)])
    lb=np.asarray(lb);ub=np.asarray(ub)
    def unpack(values):
        coef=initial+qscale*values[:nctrl*n].reshape((nctrl,n),order='F')
        xx=xscale*np.r_[0,values[nctrl*n:],0];uu=np.diff(xx)/(2*ds)
        tt=np.r_[0,np.cumsum(2*ds/(np.sqrt(xx[:-1])+np.sqrt(xx[1:])))]
        return coef,dict(s=grid,x=xx,u=uu,t=tt,duration=float(tt[-1]))
    initial_check=dense_validate(task,adapter,basis,initial,collision.numeric)
    if not initial_check['feasible']:raise ValueError('B2 requires a densely verified common initial path')
    history=[dict(available_s=perf_counter()-start,objective=timing['duration'],coeff=initial.copy(),timing=timing,source='verified_initial_path')]
    checks=[]
    class Recorder(ca.Callback):
        def __init__(self):ca.Callback.__init__(self);self.construct('joint_incumbent',{})
        def get_n_in(self):return ca.nlpsol_n_out()
        def get_n_out(self):return 1
        def get_name_in(self,i):return ca.nlpsol_out(i)
        def get_name_out(self,i):return 'ret'
        def get_sparsity_in(self,i):
            name=ca.nlpsol_out(i)
            return ca.Sparsity.scalar() if name=='f' else ca.Sparsity.dense(v.numel(),1) if name in ('x','lam_x') else ca.Sparsity.dense(g.numel(),1) if name in ('g','lam_g') else ca.Sparsity.dense(0,1)
        def get_sparsity_out(self,i):return ca.Sparsity.scalar()
        def inspect(self,values,native_objective):
            if perf_counter()-start>budget:return
            ff,gg=evaluate(values);gg=np.asarray(gg).ravel()
            violation=max(0.,float(np.max(lb-gg)),float(np.max(gg-ub)))
            rec=dict(available_s=perf_counter()-start,native_objective=float(native_objective),
                     recomputed_objective=float(ff),sample_violation=violation)
            checks.append(rec)
            if violation>1e-6:return
            coef,proposed=unpack(values)
            if np.max(abs(coef-initial))<1e-9:return
            valid=dense_validate(task,adapter,basis,coef,collision.numeric)
            rec['dense_feasible']=valid['feasible']
            if not valid['feasible']:return
            timed=retime(task,adapter,basis,coef,speed_reserve=speed_reserve)
            rec['retimed_duration']=timed.get('duration');rec['validated_at_s']=perf_counter()-start
            if timed['feasible'] and timed['duration']<min(h['objective'] for h in history)-1e-8 and rec['validated_at_s']<=budget:
                timed['joint_nlp_proposed']=proposed
                history.append(dict(available_s=rec['validated_at_s'],objective=timed['duration'],coeff=coef.copy(),timing=timed,source='nonlinear_validated_iterate'))
        def eval(self,args):
            vv={ca.nlpsol_out(i):a for i,a in enumerate(args)}
            self.inspect(np.array(vv['x']).ravel(),float(vv['f']))
            return [int(perf_counter()-start>=budget)]
    recorder=Recorder();audit=None
    if derivative_check:
        rng=np.random.default_rng(1703);direction=rng.normal(size=len(v0));direction[vlo==vhi]=0.;direction/=np.linalg.norm(direction)
        eps=1e-6;fp,gp=evaluate(v0+eps*direction);fm,gm=evaluate(v0-eps*direction);df,dg=gradient(v0)
        analytic_g=np.array(dg@direction).ravel();fdg=np.array((gp-gm)/(2*eps)).ravel()
        audit=dict(objective_directional_error=abs(float(df.T@direction)-float((fp-fm)/(2*eps))),
                   constraints_directional_max_error=float(np.max(abs(analytic_g-fdg))),
                   constraints_scaled_relative_error=float(np.max(abs(analytic_g-fdg)/(1+abs(fdg)))),epsilon=eps)
    # Exact objective/process/rate curvature, locally affine nearest-distance
    # constraint. Coal exposes only first derivatives; do not label its omitted
    # second derivative as exact. All constraint values/Jacobians remain exact.
    lam=ca.MX.sym('lam_g',g.numel());sigma=ca.MX.sym('lam_f')
    smooth_indices=np.setdiff1d(np.arange(g.numel()),collision_indices)
    H=ca.hessian(sigma*objective+ca.dot(lam[smooth_indices],smooth_g),v)[0]
    hess=ca.Function('joint_hessian',[v,ca.MX.sym('p',0),sigma,lam],[ca.triu(H)])
    options={'ipopt.print_level':0,'print_time':False,'ipopt.max_iter':400,
        'ipopt.max_wall_time':float(budget),'ipopt.tol':1e-6,'ipopt.constr_viol_tol':1e-7,
        'hess_lag':hess,'ipopt.mu_strategy':'adaptive',
        'ipopt.bound_relax_factor':0.,'iteration_callback':recorder,'iteration_callback_step':5}
    solver=ca.nlpsol('joint_full','ipopt',dict(x=v,f=objective,g=g),options);built=perf_counter()
    answer=solver(x0=v0,lbx=vlo,ubx=vhi,lbg=lb,ubg=ub)
    values=np.array(answer['x']).ravel();recorder.inspect(values,float(answer['f']))
    ff,gg=evaluate(values);gg=np.asarray(gg).ravel()
    checkpoints={}
    for deadline in (1.,5.,10.,30.):
        eligible=[h for h in history if h['available_s']<=deadline]
        checkpoints[str(deadline)]=min(eligible,key=lambda h:h['objective']) if eligible else None
    return dict(status=solver.stats()['return_status'],iterations=solver.stats().get('iter_count'),
        elapsed=perf_counter()-start,build_time=built-start,solver_time=perf_counter()-built,
        incumbents=history,checkpoints=checkpoints,checks=checks,derivative_check=audit,
        last_iterate_coeff=unpack(values)[0],sample_constraint_violation=max(0.,float(np.max(lb-gg)),float(np.max(gg-ub))),
        objective=float(ff)*timing['duration'],numeric_grid=grid,all_coefficients_variable=active_controls is None,
        active_controls=None if active_controls is None else np.asarray(active_controls),
        active_intervals=active_intervals,
        timing_u_eliminated_exactly=True,scales=dict(q_rad=qscale,x_per_s2=xscale))

