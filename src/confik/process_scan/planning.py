"""Common constrained cubic geometry, TOPP-RA, and dense joint/time baseline.

No proposed sparse update method is present. B2 is a task-matched full NLP,
not a reproduction of the cited bilevel optimizer.
"""
from time import perf_counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix
import casadi as ca
import toppra as ta
from toppra.constraint import LinearConstraint, JointVelocityConstraint, JointAccelerationConstraint
from .models import symbolic_fk


def numerical_grid(*pieces):
    """Merge representations of the same geometric point, not close path nodes.

    For v rasters, accumulated CAD lengths and linspace can encode 0.25 with
    a 2.8e-17 difference. Such zero-length cells corrupt dx/(2 ds). A 1e-12
    progress tolerance is many orders below all physical sampling intervals.
    """
    values=np.unique(np.concatenate([np.asarray(v).ravel() for v in pieces]))
    keep=np.r_[True,np.diff(values)>1e-12];result=values[keep]
    result[0]=0.;result[-1]=1.
    return result


def spline_basis(task,nctrl=64):
    # C2 simple internal knots, concentrating the same fixed capacity in turns.
    points=list(task.knots[1:-1])
    for i,segment in enumerate(task.segments):
        if not segment[5]:
            for fraction in np.linspace(0,1,5 if nctrl==64 else 7)[1:-1]:
                points.append(task.knots[i]+fraction*(task.knots[i+1]-task.knots[i]))
    candidates=np.linspace(0,1,8*nctrl+1)[1:-1]
    while len(points)<nctrl-4:
        available=np.array([0.,*sorted(points),1.])
        distances=np.min(abs(candidates[:,None]-available[None,:]),axis=1)
        points.append(float(candidates[np.argmax(distances)]))
    if len(points)>nctrl-4:raise ValueError('Insufficient common spline dimension')
    knots=np.r_[np.zeros(4),sorted(points),np.ones(4)]
    return BSpline(knots,np.eye(nctrl),3)


def initialize(task,adapter,start_q,end_q,method,nodes=401):
    t=perf_counter();grid=np.unique(np.r_[np.linspace(0,1,nodes),task.knots]);poses=task.poses(grid)
    limits=adapter.public.limits;q=start_q.copy();path=[];rows=[]
    tolerance=(.00005,np.deg2rad(.05)) if method=='B0' else (.001,np.deg2rad(.5))
    for s,target in zip(grid,poses):
        result=adapter.solve(target,q,(limits.lower,limits.upper),(q-.35,q+.35),.15,tolerance,
                             method='tb' if method=='B1' else 'gn')
        path.append(result['q']);rows.append(dict(s=float(s),**result))
        if not result['accepted']:
            return dict(feasible=False,q=np.array(path),s=grid[:len(path)],records=rows,time=perf_counter()-t)
        q=result['q']
    path=np.array(path);path[0]=start_q;path[-1]=end_q
    return dict(feasible=True,q=path,s=grid,records=rows,time=perf_counter()-t)


class SplinePath(ta.interpolator.AbstractGeometricPath):
    def __init__(self,basis,coeff):self.basis=basis;self.coeff=np.asarray(coeff)
    @property
    def dof(self):return self.coeff.shape[1]
    @property
    def path_interval(self):return np.array([0.,1.])
    def __call__(self,s,order=0):return self.basis(s,nu=order)@self.coeff


class SensorConstraint(LinearConstraint):
    def __init__(self,task,adapter):
        super().__init__();self.task=task;self.center=ray_center_function(task,adapter)
    def compute_constraint_params(self,path,gridpoints):
        g=np.asarray(gridpoints)
        speed=np.array([np.linalg.norm(np.array(self.center(q)[1])@dq) for q,dq in zip(path(g),path(g,1))])
        tags=self.task.references(g)[4]
        bound=np.where(tags,(.198/np.maximum(speed,1e-12))**2,1e8)
        return None,None,None,None,None,None,np.column_stack([np.zeros(len(g)),bound])


def ray_center_function(task,adapter):
    q=ca.SX.sym('ray_q',adapter.public.nq);fk=symbolic_fk(adapter.public);p,R=fk(q)
    pl=ca.DM(task.world_R.T)@(p-ca.DM(task.center));a=ca.DM(task.world_R.T)@R[:,2]
    if task.family=='plane':lam=-pl[2]/a[2]
    else:
        if task.family=='cylinder':
            A=a[0]**2+a[2]**2;B=2*(pl[0]*a[0]+(pl[2]+.35)*a[2]);C=pl[0]**2+(pl[2]+.35)**2-.35**2
        else:
            A=-a[0]**2+a[1]**2;B=a[2]-2*pl[0]*a[0]+2*pl[1]*a[1];C=pl[2]-pl[0]**2+pl[1]**2
        lam=-2*C/(B-ca.sqrt(ca.fmax(B*B-4*A*C,1e-16)))
    center=p+lam*R[:,2]
    return ca.Function('raycenter',[q],[center,ca.jacobian(center,q)])


def retime(task,adapter,basis,coeff,grid_n=801):
    t=perf_counter();path=SplinePath(basis,coeff);grid=numerical_grid(np.linspace(0,1,grid_n),task.knots,basis.t)
    constraints=[JointVelocityConstraint(.5*adapter.public.limits.velocity),
                 JointAccelerationConstraint(np.full(adapter.public.nq,2.),discretization_scheme=ta.constraint.DiscretizationType.Interpolation),
                 SensorConstraint(task,adapter)]
    algo=ta.algorithm.TOPPRA(constraints,path,gridpoints=grid,solver_wrapper='seidel')
    sdd,sd,_=algo.compute_parameterization(0.,0.)
    if sd is None or not np.isfinite(sd).all():return dict(feasible=False,time=perf_counter()-t)
    dt=2*np.diff(grid)/(sd[:-1]+sd[1:]);ts=np.r_[0,np.cumsum(dt)]
    result=dict(feasible=True,time=0.,s=grid,x=sd*sd,u=sdd,t=ts,duration=float(ts[-1]),raw_toppra_duration=float(ts[-1]))
    dense=timed_validate(adapter,basis,coeff,result)
    # Common validation-based time dilation, not a new path optimizer. Preserve
    # raw TOPP-RA time and discretization error; never call the dilated result
    # a proven continuous-time optimum. A common 2% path-speed reserve was
    # fixed using the reference calibration, including every 1 ms physics step.
    dilation=max(1.,dense['velocity_ratio_max'],np.sqrt(dense['acceleration_ratio_max']))/.98
    result.update(x=result['x']/dilation**2,u=result['u']/dilation**2,
                  t=result['t']*dilation,duration=result['duration']*dilation,
                  validation_dilation=dilation,pre_dilation_validation=dense)
    result['validation']=timed_validate(adapter,basis,coeff,result)
    result['feasible']=result['validation']['feasible'];result['time']=perf_counter()-t
    return result


def process_constraints(q,reference,fk):
    c,n,b=reference;p,R=fk(q);a=R[:,2];delta=ca.DM(c)-p
    distance=ca.dot(delta,a);perp=delta-distance*a
    projected=R[:,0]-ca.DM(n)*ca.dot(n,R[:,0])
    # Squared sign-invariant line-angle condition; no artificial wrist flip.
    return ca.vertcat((distance-.15)/.005,ca.sumsqr(perp)/1e-6,
        -ca.dot(a,n),ca.dot(projected,b)**2-ca.cos(np.deg2rad(5))**2*ca.sumsqr(projected))


def optimize(task,adapter,basis,initial,mode='smooth',timing=None,budget=30.,collision=None):
    """All spline coefficients participate. Ipopt's native wall limit is recorded.

    Returned iterates still require independent dense validation. Checkpoint
    incumbents are supplied by callbacks only after full nonlinear validation.
    """
    start=perf_counter();nctrl,n=initial.shape;C=ca.MX.sym('C',nctrl,n)
    grid=numerical_grid(np.linspace(0,1,801),task.knots,basis.t)
    # This is the native sparsity of an ordinary cubic B-spline, NOT a proposed
    # block-selection algorithm: every control coefficient remains a variable.
    B=ca.DM(csc_matrix(basis(grid)));D1=ca.DM(csc_matrix(basis(grid,nu=1)));D2=ca.DM(csc_matrix(basis(grid,nu=2)))
    # Validate the available input BEFORE symbolic graph construction. A late
    # final iterate must never be substituted for an earlier checkpoint.
    initial_incumbent=None
    initial_valid=dense_validate(task,adapter,basis,initial,collision.numeric if collision is not None else None)
    if initial_valid['feasible']:
        seedtime=timing if timing is not None else retime(task,adapter,basis,initial)
        if seedtime['feasible']:
            span=adapter.public.limits.upper-adapter.public.limits.lower
            d1=basis(grid,nu=1)@initial;d2=basis(grid,nu=2)@initial
            initial_objective=seedtime['duration'] if mode=='joint_time' else float(np.mean(np.sum((d1/span)**2+(d2/span)**2,axis=1))/1e4)
            initial_incumbent=dict(available_s=perf_counter()-start,objective=initial_objective,
                                  coeff=initial.copy(),timing=seedtime,source='verified_initial_path')
    Q=B@C;Qs=D1@C;Qss=D2@C;fk=symbolic_fk(adapter.public);c,normal,b,_,tags=task.references(grid)
    g=[];lb=[];ub=[];limits=adapter.public.limits
    for k in range(len(grid)):
        g.append(process_constraints(Q[k,:].T,(c[k],normal[k],b[k]),fk))
        # Do not impose redundant bounds norm²>=0 and cos<=1. Their zero
        # gradients at exact nominal poses degrade interior-point restoration.
        lb.extend([-1,-np.inf,np.cos(np.deg2rad(10)),0]);ub.extend([1,1,np.inf,np.inf])
        if collision is not None:
            g.append(collision(Q[k,:].T)/.005);lb.append(1.);ub.append(np.inf)
    g.extend([C[0,:].T,C[-1,:].T]);lb.extend(initial[0]);lb.extend(initial[-1]);ub.extend(initial[0]);ub.extend(initial[-1])
    span=ca.DM(limits.upper-limits.lower).T
    objective=(ca.sumsqr(Qs/ca.repmat(span,len(grid),1))+ca.sumsqr(Qss/ca.repmat(span,len(grid),1)))/len(grid)/1e4
    v=ca.vec(C);x0=initial.ravel(order='F');vlo=np.tile(limits.lower,(nctrl,1)).ravel(order='F');vhi=np.tile(limits.upper,(nctrl,1)).ravel(order='F')
    if mode=='joint_time':
        # Same x=sdot^2, u=sddot and segment time as the fixed-path reference.
        x=ca.MX.sym('x',len(grid));u=ca.MX.sym('u',len(grid)-1);ds=np.diff(grid)
        v=ca.vertcat(v,x,u);old_x=np.interp(grid,timing['s'],timing['x'])
        x0=np.r_[x0,old_x,np.diff(old_x)/(2*ds)];vlo=np.r_[vlo,np.zeros(len(grid)),np.full(len(grid)-1,-np.inf)];vhi=np.r_[vhi,np.full(len(grid),np.inf),np.full(len(grid)-1,np.inf)]
        vlo[nctrl*n]=vhi[nctrl*n]=0.;vlo[nctrl*n+len(grid)-1]=vhi[nctrl*n+len(grid)-1]=0.
        for k in range(len(grid)-1):
            g.append(x[k+1]-x[k]-2*ds[k]*u[k]);lb.append(0);ub.append(0)
            for index in (k,k+1):
                g.append(Qs[index,:].T**2*x[index]);lb.extend(np.zeros(n));ub.extend((.5*limits.velocity)**2)
                g.append(Qss[index,:].T*x[index]+Qs[index,:].T*u[k]);lb.extend(np.full(n,-2.));ub.extend(np.full(n,2.))
        raycenter=ray_center_function(task,adapter)
        for k in range(len(grid)):
            if tags[k]:
                _,Jc=raycenter(Q[k,:].T)
                g.append(ca.sumsqr(Jc@Qs[k,:].T)*x[k]);lb.append(0);ub.append(.198**2)
        objective=ca.sum1(2*ca.DM(ds)/(ca.sqrt(x[:-1]+1e-16)+ca.sqrt(x[1:]+1e-16)))
    if mode not in ('smooth','joint_time'):raise ValueError(mode)
    numeric_collision=collision.numeric if collision is not None else None
    recorder=FeasibleRecorder(task,adapter,basis,initial,mode,grid,nctrl,n,int(v.numel()),len(lb),
                              numeric_collision,start,initial_incumbent)
    opts={'ipopt.print_level':0,'print_time':False,'ipopt.max_iter':200,
          'ipopt.max_wall_time':float(budget),'ipopt.tol':1e-6,'ipopt.constr_viol_tol':1e-7,
          'ipopt.hessian_approximation':'limited-memory',
          'iteration_callback':recorder,'iteration_callback_step':5}
    solver=ca.nlpsol('geometry','ipopt',dict(x=v,f=objective,g=ca.vertcat(*g)),opts)
    built=perf_counter();answer=solver(x0=x0,lbx=vlo,ubx=vhi,lbg=lb,ubg=ub)
    values=np.array(answer['x']).ravel();result=values[:nctrl*n].reshape((nctrl,n),order='F')
    recorder.inspect(values,float(answer['f']))
    violations=max(float(np.max(np.asarray(lb)-np.array(answer['g']).ravel())),float(np.max(np.array(answer['g']).ravel()-np.asarray(ub))),0.)
    selected=recorder.select(10.)
    return dict(coeff=None if selected is None else selected['coeff'],last_iterate_coeff=result,
                incumbent_feasible=selected is not None,incumbents=recorder.history,
                checkpoints={str(t):recorder.select(t) for t in (1.,5.,10.,30.)},
                elapsed=perf_counter()-start,build_time=built-start,
                solver_time=perf_counter()-built,status=solver.stats()['return_status'],
                objective=float(answer['f']),sample_constraint_violation=violations,
                solver_feasible=violations<=1e-5,iterations=solver.stats().get('iter_count'),
                timing_x=values[nctrl*n:nctrl*n+len(grid)] if mode=='joint_time' else None,
                timing_s=grid if mode=='joint_time' else None)


class FeasibleRecorder(ca.Callback):
    """Numerical incumbent tracking; never backdates late validation."""
    def __init__(self,task,adapter,basis,initial,mode,grid,nctrl,n,nvars,ncons,collision,start,initial_incumbent):
        ca.Callback.__init__(self)
        self.task,self.adapter,self.basis=task,adapter,basis
        self.mode,self.grid,self.nctrl,self.n=mode,grid,nctrl,n
        self.nvars,self.ncons=nvars,ncons;self.collision=collision;self.start=start;self.history=[]
        self.construct('incumbent',{})
        if initial_incumbent is not None:self.history.append(initial_incumbent)
    def get_n_in(self):return ca.nlpsol_n_out()
    def get_n_out(self):return 1
    def get_name_in(self,i):return ca.nlpsol_out(i)
    def get_name_out(self,i):return 'ret'
    def get_sparsity_in(self,i):
        name=ca.nlpsol_out(i)
        return ca.Sparsity.scalar() if name=='f' else ca.Sparsity.dense(self.nvars,1) if name in ('x','lam_x') else ca.Sparsity.dense(self.ncons,1) if name in ('g','lam_g') else ca.Sparsity.dense(0,1)
    def get_sparsity_out(self,i):return ca.Sparsity.scalar()
    def eval(self,args):
        values={ca.nlpsol_out(i):v for i,v in enumerate(args)}
        self.inspect(np.array(values['x']).ravel(),float(values['f']))
        return [int(perf_counter()-self.start>=30.)]
    def inspect(self,values,objective):
        if self.history and objective>=min(r['objective'] for r in self.history):return
        if perf_counter()-self.start>30:return
        coeff=values[:self.nctrl*self.n].reshape((self.nctrl,self.n),order='F')
        valid=dense_validate(self.task,self.adapter,self.basis,coeff,self.collision)
        if not valid['feasible']:return
        if self.mode=='smooth':timing=retime(self.task,self.adapter,self.basis,coeff)
        else:
            x=values[self.nctrl*self.n:self.nctrl*self.n+len(self.grid)];ds=np.diff(self.grid)
            if np.any(x<0):return
            dt=2*ds/(np.sqrt(x[:-1])+np.sqrt(x[1:]));u=np.diff(x)/(2*ds)
            # Full common retiming after a geometric update, required by the
            # protocol. Retain the joint NLP's time variables as diagnostics.
            proposed=dict(s=self.grid,x=x,u=u,t=np.r_[0,np.cumsum(dt)],duration=float(sum(dt)))
            timing=retime(self.task,self.adapter,self.basis,coeff)
            timing['joint_nlp_proposed']=proposed;objective=timing.get('duration',np.inf)
            if self.history and objective>=min(r['objective'] for r in self.history):return
        elapsed=perf_counter()-self.start
        if timing['feasible'] and elapsed<=30:
            self.history.append(dict(available_s=elapsed,objective=objective,coeff=coeff.copy(),timing=timing,source='nonlinear_validated_iterate'))
    def select(self,seconds):
        eligible=[r for r in self.history if r['available_s']<=seconds]
        return min(eligible,key=lambda r:r['objective']) if eligible else None


def timed_validate(adapter,basis,coeff,timing):
    grid=np.linspace(0,1,6401);x=np.interp(grid,timing['s'],timing['x'])
    index=np.minimum(np.searchsorted(timing['s'],grid,side='right')-1,len(timing['u'])-1)
    u=timing['u'][np.maximum(0,index)];qs=basis(grid,nu=1)@coeff;qss=basis(grid,nu=2)@coeff
    vmax=float(np.max(abs(qs)*np.sqrt(np.maximum(x,0))[:,None]/(.5*adapter.public.limits.velocity)))
    amax=float(np.max(abs(qss*x[:,None]+qs*u[:,None]))/2)
    return dict(feasible=bool(vmax<=1.001 and amax<=1.001),velocity_ratio_max=vmax,acceleration_ratio_max=amax,
                numerical_discretization_relative_tolerance=.001)


def dense_validate(task,adapter,basis,coeff,collision=None,count=3201):
    grid=np.unique(np.r_[np.linspace(0,1,count),task.knots]);q=basis(grid)@coeff
    records=[];minclear=np.inf;valid=True
    for s,qq in zip(grid,q):
        pose=adapter.native.forward(qq);v=task.process_values(pose.position,pose.rotation,s)
        legal=task.process_legal(pose.position,pose.rotation,s)
        if collision is not None:
            clearance=float(collision(qq));minclear=min(minclear,clearance);legal&=clearance>=.005
        legal&=bool(np.all(qq>=adapter.public.limits.lower)&np.all(qq<=adapter.public.limits.upper))
        records.append(v);valid&=legal
    return dict(feasible=bool(valid),s=grid,q=q,process=records,
                min_clearance=None if not np.isfinite(minclear) else minclear)
