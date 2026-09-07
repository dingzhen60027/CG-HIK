"""Fixed-initial-state nonlinear horizon reference, not an online policy.

Joint variables are scaled positions relative to the immutable current q.
Position and SO(3) errors are in the same world frame as the geometric Jacobian.
The numerical result supplies an upper bound from a found path, never a proof
that no path exists or that the reported rho is globally optimal.
"""
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.optimize import Bounds, minimize

from ..geometry import pose_error
from ..solvers.dls import AdaptiveDLS, DLSConfig
from ..solvers.verifier import SolutionVerifier
from ..types import IKQuery


@dataclass(frozen=True)
class SearchBudget:
    max_iterations: int = 100
    max_seconds: float = 15.
    ftol: float = 1e-10
    # Fixed conservative numerical interior, NOT a relaxed public tolerance.
    # The original verifier and public epsilon are used for every stored path.
    normalized_squared_interior: float = 1e-8
    dls_iterations: int = 25


class CountingKinematics:
    def __init__(self, kin):
        self.kin = kin
        self.counts = dict(fk_calls=0, geometric_jacobian_calls=0)

    def __getattr__(self, name):
        return getattr(self.kin,name)

    def forward(self,q):
        self.counts['fk_calls'] += 1
        return self.kin.forward(q)

    def jacobian(self,q):
        self.counts['geometric_jacobian_calls'] += 1
        return self.kin.jacobian(q)


def so3_right_jacobian_inverse(e):
    """d Log(R Exp(delta)) / d delta in world-frame rotation-vector axes."""
    theta2 = float(e@e)
    x,y,z = e
    cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    if theta2 < 1e-8:
        coefficient = 1/12 + theta2/720 + theta2*theta2/30240
    else:
        theta = np.sqrt(theta2)
        coefficient = (1-.5*theta/np.tan(.5*theta))/theta2
    return np.eye(3)+.5*cross+coefficient*(cross@cross)


def pose_residual_jacobian(kin,target,q):
    e = pose_error(target,kin.forward(q))
    j = kin.jacobian(q)
    return e,np.vstack([-j[:3],-so3_right_jacobian_inverse(e[3:])@j[3:]])


class HorizonProblem:
    def __init__(self,kin,verifier,q0,targets,dt=.02,interior=1e-8):
        self.kin, self.verifier = kin, verifier
        self.q0 = np.asarray(q0,dtype=float).copy()
        self.q0.setflags(write=False)
        self.targets, self.dt = tuple(targets), dt
        self.L, self.n = len(targets),kin.nq
        self.step = kin.limits.velocity*dt+verifier.config.velocity_tolerance
        self.task = np.array([verifier.config.position_tolerance]*3+
                             [verifier.config.orientation_tolerance]*3)
        self.interior = interior
        self.counts = dict(objective_calls=0, objective_jacobian_calls=0,
            pose_constraint_calls=0, pose_constraint_jacobian_calls=0,
            step_constraint_calls=0, step_constraint_jacobian_calls=0)
        self._key = None
        self._errors = None
        self._derivatives = None
        self.best_pose_path = None
        self.best_verified_path = None
        # z_k=(q_k-q0)/step. q0 is a constant, never an optimization variable.
        self.A = np.zeros((2*self.L*self.n,self.L*self.n+1))
        for k in range(self.L):
            for j in range(self.n):
                row = k*self.n+j
                self.A[row,row] = -1
                self.A[row+self.L*self.n,row] = 1
                if k:
                    self.A[row,row-self.n] = 1
                    self.A[row+self.L*self.n,row-self.n] = -1
        self.A[:,-1] = 1

    def unpack(self,x):
        return self.q0+self.step*np.asarray(x[:-1]).reshape(self.L,self.n)

    def pack(self,path):
        z = (np.asarray(path)-self.q0)/self.step
        dz = np.diff(np.vstack([np.zeros(self.n),z]),axis=0)
        return np.r_[z.ravel(),max(0.,float(np.max(np.abs(dz))))+1e-8]

    def bounds(self):
        lo = np.tile((self.kin.limits.lower-self.q0)/self.step,self.L)
        hi = np.tile((self.kin.limits.upper-self.q0)/self.step,self.L)
        return Bounds(np.r_[lo,0.],np.r_[hi,np.inf])

    def objective(self,x):
        self.counts['objective_calls'] += 1
        return float(x[-1])

    def objective_jacobian(self,x):
        self.counts['objective_jacobian_calls'] += 1
        out = np.zeros(len(x));out[-1] = 1
        return out

    def _evaluate(self,x,need_jacobian=False):
        z = np.asarray(x[:-1])
        if self._key is None or not np.array_equal(self._key,z):
            self._key = z.copy();self._derivatives = None
            path = self.unpack(x)
            self._errors = np.array([pose_error(t,self.kin.forward(q))/self.task
                                      for t,q in zip(self.targets,path)])
            # Preserve all actually evaluated pose/joint-feasible paths, not
            # only the optimizer's last iterate. Recompute achieved rho from q.
            if np.all(np.linalg.norm(self._errors[:,:3],axis=1)<=1) and np.all(np.linalg.norm(self._errors[:,3:],axis=1)<=1):
                if np.all(path>=self.kin.limits.lower) and np.all(path<=self.kin.limits.upper):
                    rho = float(np.max(np.abs(np.diff(np.vstack([self.q0,path]),axis=0))/self.step))
                    if self.best_pose_path is None or rho < self.best_pose_path[0]:
                        self.best_pose_path = (rho,path.copy(),float(x[-1]))
                    if rho<=1 and (self.best_verified_path is None or rho<self.best_verified_path[0]):
                        previous = self.q0
                        passed = True
                        for q,target in zip(path,self.targets):
                            if not self.verifier.check(q,IKQuery(target,previous,self.dt)).accepted:
                                passed = False;break
                            previous = q
                        if passed:
                            self.best_verified_path = (rho,path.copy(),float(x[-1]))
        if need_jacobian and self._derivatives is None:
            path = self.unpack(x)
            matrices = []
            for target,q,e in zip(self.targets,path,self._errors):
                j = self.kin.jacobian(q)
                raw_e = e*self.task
                de = np.vstack([-j[:3],-so3_right_jacobian_inverse(raw_e[3:])@j[3:]])
                matrices.append(de/self.task[:,None]*self.step[None,:])
            self._derivatives = np.asarray(matrices)

    def pose_constraints(self,x):
        self.counts['pose_constraint_calls'] += 1
        self._evaluate(x)
        e = self._errors
        return np.stack([1-self.interior-np.sum(e[:,:3]**2,axis=1),
                         1-self.interior-np.sum(e[:,3:]**2,axis=1)],axis=1).ravel()

    def pose_constraint_jacobian(self,x):
        self.counts['pose_constraint_jacobian_calls'] += 1
        self._evaluate(x,True)
        out = np.zeros((2*self.L,self.L*self.n+1))
        for k,(e,j) in enumerate(zip(self._errors,self._derivatives)):
            out[2*k,k*self.n:(k+1)*self.n] = -2*e[:3]@j[:3]
            out[2*k+1,k*self.n:(k+1)*self.n] = -2*e[3:]@j[3:]
        return out

    def step_constraints(self,x):
        self.counts['step_constraint_calls'] += 1
        return self.A@x

    def step_constraint_jacobian(self,x):
        self.counts['step_constraint_jacobian_calls'] += 1
        return self.A


def validate_path(kin,verifier,q0,targets,path,dt=.02,reported_rho=None):
    """Always advance through the proposed path for constraints, not held state.

    A path is a witness only if EVERY proposed transition passes. Failures are
    never repaired by replacing a node with some other candidate's state.
    """
    previous = np.asarray(q0)
    step = kin.limits.velocity*dt+verifier.config.velocity_tolerance
    rows = []
    for k,(q,target) in enumerate(zip(np.asarray(path),targets),1):
        query = IKQuery(target,previous,dt)
        check = verifier.check(q,query)
        e = pose_error(target,kin.forward(q))
        normalized_step = np.abs(q-previous)/step
        rows.append(dict(offset=k,previous_q=previous.tolist(),q=q.tolist(),
            target_position=target.position.tolist(),target_rotation=target.rotation.tolist(),dt=dt,
            position_residual_vector=e[:3].tolist(),orientation_residual_vector=e[3:].tolist(),
            position_error=float(check.position_error),orientation_error=float(check.orientation_error),
            position_constraint_excess=float(check.position_error-verifier.config.position_tolerance),
            orientation_constraint_excess=float(check.orientation_error-verifier.config.orientation_tolerance),
            joint_lower_margin=(q-kin.limits.lower).tolist(),joint_upper_margin=(kin.limits.upper-q).tolist(),
            normalized_step=normalized_step.tolist(),accepted=bool(check.accepted),reasons=list(check.reasons)))
        previous = q
    actual_rho = max(float(max(r['normalized_step'])) for r in rows)
    pose_ok = all(r['position_constraint_excess']<=0 and r['orientation_constraint_excess']<=0 for r in rows)
    range_ok = all(min(r['joint_lower_margin'])>=0 and min(r['joint_upper_margin'])>=0 for r in rows)
    verified = all(r['accepted'] for r in rows)
    return dict(q0=np.asarray(q0).tolist(),path=np.asarray(path).tolist(),actual_rho=actual_rho,
        optimizer_reported_rho=reported_rho,
        step_constraint_excess_at_reported_rho=None if reported_rho is None else actual_rho-reported_rho,
        pose_and_range_feasible=pose_ok and range_ok,
        found_legal_continuation=bool(actual_rho<=1 and verified and range_ok),
        all_frames_verified=verified,frames=rows,
        max_position_constraint_excess=max(r['position_constraint_excess'] for r in rows),
        max_orientation_constraint_excess=max(r['orientation_constraint_excess'] for r in rows),
        minimum_joint_margin=min(min(r['joint_lower_margin']+r['joint_upper_margin']) for r in rows))


def make_initialization(kind,kin,q0,targets,prefix,config,budget):
    if kind == 'hold':
        return np.tile(q0,(len(targets),1))
    if kind == 'historical_prefix':
        kept = [np.asarray(q) for q in prefix[:len(targets)]]
        last = kept[-1] if kept else np.asarray(q0)
        return np.asarray(kept+[last.copy() for _ in range(len(targets)-len(kept))])
    if kind == 'dls_prediction':
        dls = AdaptiveDLS(kin,DLSConfig(**config['solver']))
        path = [];q = np.asarray(q0).copy()
        for target in targets:
            result = dls.solve(target,q,budget.dls_iterations,seed_source='fixed_horizon_initialization')
            q = result.q.copy();path.append(q)
        return np.asarray(path)
    raise ValueError(kind)


class TimeBudgetExceeded(Exception):
    pass


def search_one(kin,verifier_config,q0,targets,prefix,kind,config,budget=SearchBudget()):
    total_start = perf_counter_ns()
    counted = CountingKinematics(kin)
    verifier = SolutionVerifier(counted,verifier_config)
    start = perf_counter_ns()
    initial = make_initialization(kind,counted,q0,targets,prefix,config,budget)
    initialization_ns = perf_counter_ns()-start
    initial_counts = dict(counted.counts)
    problem = HorizonProblem(counted,verifier,q0,targets,interior=budget.normalized_squared_interior)
    x0 = problem.pack(initial)
    problem.pose_constraints(x0)
    optimization_start = perf_counter_ns()
    last_x = x0.copy();iterations = 0

    def callback(x):
        nonlocal last_x,iterations
        last_x = x.copy();iterations += 1
        if (perf_counter_ns()-optimization_start)/1e9 > budget.max_seconds:
            raise TimeBudgetExceeded()

    try:
        result = minimize(problem.objective,x0,jac=problem.objective_jacobian,method='SLSQP',
            bounds=problem.bounds(),constraints=[
                dict(type='ineq',fun=problem.pose_constraints,jac=problem.pose_constraint_jacobian),
                dict(type='ineq',fun=problem.step_constraints,jac=problem.step_constraint_jacobian)],
            callback=callback,options=dict(maxiter=budget.max_iterations,ftol=budget.ftol))
        last_x = result.x.copy()
        status = dict(success=bool(result.success),code=int(result.status),message=str(result.message),
            iterations=int(result.nit),scipy_nfev=int(result.nfev),scipy_njev=int(result.njev))
    except TimeBudgetExceeded:
        status = dict(success=False,code='time_budget',message='fixed wall limit checked at major-iteration callback',
                      iterations=iterations,scipy_nfev=None,scipy_njev=None)
    problem.pose_constraints(last_x)
    optimization_ns = perf_counter_ns()-optimization_start
    before_validation = dict(counted.counts)
    # Prefer the lowest actual rho among evaluated pose/joint-feasible paths.
    # If no such path was found, retain last iterate diagnostics and mark unknown.
    best = problem.best_pose_path
    chosen = best[1] if best else problem.unpack(last_x)
    declared = best[2] if best else float(last_x[-1])
    start = perf_counter_ns()
    validation = validate_path(counted,verifier,q0,targets,chosen,reported_rho=declared)
    last_validation = validate_path(counted,verifier,q0,targets,problem.unpack(last_x),reported_rho=float(last_x[-1]))
    validation_ns = perf_counter_ns()-start
    total_ns = perf_counter_ns()-total_start
    return dict(initialization=kind,horizon=len(targets),optimizer=status,
        best_found_rho=float(best[0]) if best else None,
        initial_path=initial.tolist(),initial_rho=float(x0[-1]),
        validation=validation,last_iterate_validation=last_validation,
        found_legal_continuation=validation['found_legal_continuation'],
        outcome='found_verified' if validation['found_legal_continuation'] else 'not_found_within_fixed_search',
        timing_ns=dict(initialization=initialization_ns,optimization=optimization_ns,
                       validation=validation_ns,total=total_ns),
        counts=dict(initialization=initial_counts,
            optimization={k:before_validation[k]-initial_counts[k] for k in before_validation},
            validation={k:counted.counts[k]-before_validation[k] for k in before_validation},
            total=dict(counted.counts),callbacks=problem.counts),
        fixed_q0_unchanged=bool(np.array_equal(problem.q0,q0)),
        interpretation='rho is achieved by a numerically found path, not a globally optimal lower bound; not_found is not mathematical infeasibility')
