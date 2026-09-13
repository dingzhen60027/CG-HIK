"""Same local minimax epigraph, for fixed-subproblem comparison only."""
from time import perf_counter_ns
import numpy as np
from scipy import sparse
import clarabel
from .task_balance_gn import box_normal_residual


def objective(d,e,G,damping,posture,posture_scale,posture_weight):
    r=e+G@d
    return max(float(r[:3]@r[:3]),float(r[3:]@r[3:]))+.5*damping*float(d@d)+.5*posture_weight*float(np.sum((posture+posture_scale*d)**2))


def quality(d,e,G,lo,hi,damping,posture,posture_scale,posture_weight,theta):
    r=e+G@d;f0=float(r[:3]@r[:3]);f1=float(r[3:]@r[3:])
    reg=.5*damping*float(d@d)+.5*posture_weight*float(np.sum((posture+posture_scale*d)**2))
    H=2*theta*G[:3].T@G[:3]+2*(1-theta)*G[3:].T@G[3:]+damping*np.eye(len(d))+posture_weight*np.diag(posture_scale**2)
    g=2*theta*G[:3].T@e[:3]+2*(1-theta)*G[3:].T@e[3:]+posture_weight*posture_scale*posture
    nr=box_normal_residual(d,H@d+g,lo,hi)
    grad=H@d+g
    # A second numerical lower bound that also works for a point just inside an
    # active face: minimize the lambda-strong-convex affine/quadratic minorant
    # over the original box. No snapping to an unearned normal cone is needed.
    delta=np.clip(-grad/damping,lo-d,hi-d)
    box_lower=theta*f0+(1-theta)*f1+reg+float(grad@delta)+.5*damping*float(delta@delta)
    return dict(objective=max(f0,f1)+reg,feasible_box_violation=float(max(0.,np.max(lo-d),np.max(d-hi))),
        normal_residual=float(np.linalg.norm(nr)),dual_lower=theta*f0+(1-theta)*f1+reg-float(nr@nr)/(2*damping),
        box_minorant_lower=box_lower,
        task_squared=[f0,f1])


class EpigraphReference:
    """Cached sparse structure; includes conversion/update/solve/quality time."""
    def __init__(self,n):
        start=perf_counter_ns();self.n=n;self.solver=None
        rows=2*n+10;cols=n+1
        self.A=np.zeros((rows,cols));self.b=np.zeros(rows);self.c=np.zeros(cols)
        self.A[:n,:n]=np.eye(n);self.A[n:2*n,:n]=-np.eye(n)
        # Retain the pattern even when a Jacobian entry becomes exactly zero.
        pattern=np.zeros_like(self.A,dtype=bool);pattern[:2*n,:n]=self.A[:2*n,:n]!=0
        for b in range(2):
            s=2*n+5*b;pattern[s,-1]=True;pattern[s+4,-1]=True;pattern[s+1:s+4,:n]=True
        self.As=sparse.csc_matrix(pattern.astype(float));self.rr=self.As.indices.copy()
        self.cc=np.repeat(np.arange(cols),np.diff(self.As.indptr))
        self.P=sparse.csc_matrix((np.ones(n), (np.arange(n),np.arange(n))),shape=(cols,cols))
        self.settings=clarabel.DefaultSettings();self.settings.verbose=False;self.settings.max_threads=1
        self.settings.max_iter=100;self.settings.tol_feas=1e-9
        self.settings.tol_gap_abs=self.settings.tol_gap_rel=1e-9
        self.settings.presolve_enable=False;self.settings.chordal_decomposition_enable=False
        self.settings.input_sparse_dropzeros=False
        self.cones=[clarabel.NonnegativeConeT(2*n),clarabel.SecondOrderConeT(5),clarabel.SecondOrderConeT(5)]
        self.initialization_ns=perf_counter_ns()-start

    def solve(self,e,G,lo,hi,damping=.01,posture=None,posture_scale=None,posture_weight=0.):
        start=perf_counter_ns();n=self.n
        e=np.asarray(e,float);G=np.asarray(G,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
        posture=np.zeros(n) if posture is None else np.asarray(posture,float)
        w=np.zeros(n) if posture_scale is None else np.asarray(posture_scale,float)
        self.P.data[:]=damping+posture_weight*w*w;self.c[:n]=posture_weight*w*posture;self.c[-1]=1.
        self.b[:n]=hi;self.b[n:2*n]=-lo
        for block in range(2):
            s=2*n+5*block;self.A[s,-1]=-1;self.b[s]=1
            self.A[s+1:s+4,:n]=-2*G[3*block:3*block+3];self.b[s+1:s+4]=2*e[3*block:3*block+3]
            self.A[s+4,-1]=-1;self.b[s+4]=-1
        self.As.data[:]=self.A[self.rr,self.cc];updated=self.solver is not None
        if updated:
            assert self.solver.is_data_update_allowed()
            self.solver.update(P=self.P.data,q=self.c,A=self.As.data,b=self.b)
        else:self.solver=clarabel.DefaultSolver(self.P,self.c,self.As,self.b,self.cones,self.settings)
        setup=perf_counter_ns()-start;t=perf_counter_ns();s=self.solver.solve();solve=perf_counter_ns()-t
        x=np.array(s.x);raw=x[:n];d=np.clip(raw,lo,hi)
        # SOC epigraph multipliers: dL/dt=1-sum(z0+zlast)=0.
        native_dual=np.array(s.z)
        weights=np.array([native_dual[2*n+5*b]+native_dual[2*n+5*b+4] for b in range(2)])
        theta=float(np.clip(weights[0]/weights.sum(),0,1)) if weights.sum()>0 else .5
        q=quality(d,e,G,lo,hi,damping,posture,w,posture_weight,theta)
        q.update(raw_box_violation=float(max(0.,np.max(lo-raw),np.max(raw-hi))),
            roundoff_projection=float(np.max(abs(raw-d))),raw_epigraph_violation=max(0.,max(q['task_squared'])-x[-1]),
            theta=theta,native_theta_weights=weights.tolist(),iterations=int(s.iterations),status=str(s.status),
            native_primal_objective=float(s.obj_val)+.5*posture_weight*float(posture@posture),
            native_dual_objective=float(s.obj_val_dual)+.5*posture_weight*float(posture@posture),
            native_primal_residual=float(s.r_prim),native_dual_residual=float(s.r_dual),
            setup_update_ns=setup,solve_ns=solve,native_seconds=float(s.solve_time),updated=updated)
        q['total_ns']=perf_counter_ns()-start
        return d,q
