"""Shared-command, task-admissible recourse. No inherited CR-IK policy.

One current command, parallel recourse configurations, and one future pose
shortfall. The three modes differ only in the scenario set / affine recourse
restriction. Only the current independently verified command is executable.
"""
from dataclasses import dataclass
from time import perf_counter_ns

import clarabel
import numpy as np
from scipy import sparse

from .types import Pose, IKQuery
from .geometry import pose_error
from .correction_reserve.geometry import predict_target, perturb_target, task_scale, residual_linearization
from .correction_reserve.native_geometry import NativeGeometry
from .continuation_mechanism.nonlinear_reference_math import so3_right_jacobian_inverse
from .continuation_mechanism.observation import representable_interior
from .task_contract_alignment.outcomes import ContractSolver


@dataclass(frozen=True)
class RecourseConfig:
    outer_iterations: int = 2
    trust_step: float = .5
    native_max_iterations: int = 40
    native_time_limit_ms: float = 2.
    total_soft_limit_ms: float = 18.
    pose_interior: float = 1e-5
    rate_interior: float = 1e-10
    objective_atol: float = 1e-10
    objective_rtol: float = 1e-8
    rank_rtol: float = 1e-10
    anchor_first: bool = True
    fixed_updates: int = 2


def scenario_nodes(nominal=False):
    return np.zeros((1, 6)) if nominal else np.vstack((np.zeros(6), np.array(
        [sign*np.eye(6)[i] for i in range(6) for sign in (1., -1.)])))


def common_model(kin, predicted, z, scale, step, rank_rtol=1e-10):
    e, derivative, _ = residual_linearization(kin, predicted, z, scale)
    F = derivative*step
    C = np.eye(6)
    C[3:, 3:] = so3_right_jacobian_inverse(-e[3:]*scale[3:])
    # Least squares even at rank zero. No full-rank gate in either program.
    B = -np.linalg.pinv(F, rcond=rank_rtol) @ C
    return e, F, C, B


class JointSOCP:
    """Fixed sparse structure with Clarabel data updates, also used by toy tests.

    Coordinates are u=(q-previous)/S, v_j=(z_j-previous)/S, tau.
    There is exactly one u block. No regularizer is added to the v blocks.
    Both residual blocks have three components; scalar tests use zero padding.
    """
    def __init__(self, n, m, fixed=False, config=None):
        self.n, self.m, self.fixed = n, m, fixed
        self.cfg = config or RecourseConfig()
        self.size = n*(1+m)+1
        size = self.size
        rr, cc, segments = [], [], {}

        def block(name, rows, cols):
            r, c = np.broadcast_arrays(np.asarray(rows), np.asarray(cols))
            segments[name] = (r.ravel(), c.ravel(), r.shape)
            rr.extend(r.ravel()); cc.extend(c.ravel())

        offset = 0
        self.eq_rows = n*(m-1) if fixed else 0
        if fixed:
            for j in range(1, m):
                row = offset+np.arange(n)
                block(f'eq{j}', row, n*(1+j)+np.arange(n))
                block(f'eq0_{j}', row, n+np.arange(n))
                offset += n
        self.bounds_start = offset
        # Two bounds per coordinate; a future bound intersects the trust box.
        count = n*(m+1)
        ids = np.arange(count)
        block('upper', offset+ids, ids); offset += count
        block('lower', offset+ids, ids); offset += count
        self.rate_start = offset
        js = np.arange(m*n)
        block('rate_v', offset+js, n+js)
        block('rate_u', offset+js, js % n); offset += m*n
        block('rate_minus_v', offset+js, n+js)
        block('rate_minus_u', offset+js, js % n); offset += m*n
        block('nonnegative_tau', [offset], [size-1]); offset += 1
        linear_end = offset
        self.soc_rows = []
        for j in range(m+1):
            pair = []
            for part in range(2):
                pair.append(offset)
                cols = np.arange(n)+n*j
                block(f'F{j}_{part}', (offset+1+np.arange(3))[:, None], cols[None, :])
                if j:
                    block(f'tau{j}_{part}', [offset], [size-1])
                offset += 4
            self.soc_rows.append(pair)
        self.A = sparse.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(offset, size)).tocsc()
        if self.A.nnz != len(rr):
            raise AssertionError('overlapping sparse slots')
        lookup = {(int(self.A.indices[k]), c): k for c in range(size)
                  for k in range(self.A.indptr[c], self.A.indptr[c+1])}
        self.slots = {name: (np.array([lookup[(int(r), int(c))] for r, c in zip(rs, cs)]), shape)
                      for name, (rs, cs, shape) in segments.items()}
        self.A.data.fill(0)
        for name, val in [('upper',1),('lower',-1),('rate_v',1),('rate_u',-1),
                          ('rate_minus_v',-1),('rate_minus_u',1),('nonnegative_tau',-1)]:
            self.put(name, val)
        for j in range(1,m):
            if fixed:
                self.put(f'eq{j}',1);self.put(f'eq0_{j}',-1)
        for j in range(1,m+1):
            for part in range(2):self.put(f'tau{j}_{part}',-1)
        self.b = np.zeros(offset)
        self.c = np.zeros(size)
        diag = np.r_[np.ones(n),np.zeros(n*m),1.]
        self.P = sparse.diags(diag,format='csc')
        self.cones = ([clarabel.ZeroConeT(self.eq_rows)] if fixed and self.eq_rows else [])
        self.cones += [clarabel.NonnegativeConeT(linear_end-self.eq_rows)]
        self.cones += [clarabel.SecondOrderConeT(4) for _ in range(2*(m+1))]
        self.solver = None
        self.setup_count = self.update_count = 0

    def put(self, name, value):
        idx, shape = self.slots[name]
        self.A.data[idx] = np.broadcast_to(value, shape).ravel()

    def solve(self, anchor, u0, vs, lower, upper, global_lower, global_upper,
              current_e, current_F, future_e, future_F, C, nodes, B=None,
              trust=None, time_limit=None, rate_limit=None, current_radius=None):
        begin = perf_counter_ns()
        cfg, n, m = self.cfg, self.n, self.m
        trust = cfg.trust_step if trust is None else trust
        rate_limit = 1-cfg.rate_interior if rate_limit is None else rate_limit
        current_radius = 1-cfg.pose_interior if current_radius is None else current_radius
        lo = np.r_[np.maximum(lower,u0-trust), np.maximum(global_lower,vs-trust).ravel()]
        hi = np.r_[np.minimum(upper,u0+trust), np.minimum(global_upper,vs+trust).ravel()]
        count = len(lo); at = self.bounds_start
        self.b[at:at+count] = hi
        self.b[at+count:at+2*count] = -lo
        self.b[self.rate_start:self.rate_start+2*m*n] = rate_limit
        self.c[:n] = -anchor
        if self.fixed:
            self.b[:self.eq_rows] = (nodes[1:] @ B.T).ravel()
        cur = current_e-current_F@u0
        # ONE expansion point / Jacobian / target derivative for all scenarios.
        fut = future_e-future_F@vs[0]
        for j, rows in enumerate(self.soc_rows):
            F = current_F if j == 0 else future_F
            off = cur if j == 0 else fut+C@nodes[j-1]
            for part,row in enumerate(rows):
                self.put(f'F{j}_{part}', -F[part*3:part*3+3])
                self.b[row] = current_radius if j == 0 else 1.
                self.b[row+1:row+4] = off[part*3:part*3+3]
        settings = clarabel.DefaultSettings()
        settings.verbose = False;settings.max_threads = 1
        settings.max_iter = cfg.native_max_iterations
        settings.time_limit = cfg.native_time_limit_ms/1000 if time_limit is None else time_limit
        settings.tol_feas = 1e-9;settings.tol_gap_abs = 1e-8;settings.tol_gap_rel = 1e-8
        # Updates require no presolve/chordal structural modification.
        settings.presolve_enable = False
        settings.chordal_decomposition_enable = False
        if self.solver is None:
            self.solver = clarabel.DefaultSolver(self.P,self.c,self.A,self.b,self.cones,settings)
            self.setup_count += 1
        else:
            self.solver.update(A=self.A,b=self.b,q=self.c,settings=settings)
            self.update_count += 1
        build_ns = perf_counter_ns()-begin
        solve_start = perf_counter_ns(); result = self.solver.solve()
        return np.asarray(result.x), dict(status=str(result.status),iterations=int(result.iterations),
            native_seconds=float(result.solve_time),primal_residual=float(result.r_prim),
            dual_residual=float(result.r_dual),build_ns=build_ns,
            solve_ns=perf_counter_ns()-solve_start,affine_tau=float(result.x[-1]),
            variables=self.size,sparse_nonzeros=self.A.nnz,updates=self.update_count)


class TaskRecourseIK:
    def __init__(self, kin, verifier, source, library, urdf, mode='free', config=None):
        if mode not in ('free','fixed','nominal'):raise ValueError(mode)
        self.kin,self.verifier,self.mode=kin,verifier,mode
        self.cfg=RecourseConfig(**(config or {}))
        if self.cfg.outer_iterations != 2:raise ValueError('joint update budget remains two')
        if not 1 <= self.cfg.fixed_updates <= 6:raise ValueError('bounded fixed-current updates: one to six')
        self.geometry=NativeGeometry(kin,urdf)
        self.backup=ContractSolver('trac_task_5ms',kin,verifier,source,library,urdf)
        self.nodes=scenario_nodes(mode=='nominal')
        self.scale=task_scale(verifier)
        self.program=JointSOCP(kin.nq,len(self.nodes),mode=='fixed',self.cfg)
        self.last_target=None

    def reset(self, q):
        self.last_target=None

    def close(self):self.backup.close()

    def inspect(self, q, zs, query, targets, anchor, step):
        """True nonlinear merit. Pose slack never relaxes joint/rate constraints."""
        current=self.verifier.check(q,query)
        finite=bool(np.isfinite(zs).all())
        bounds=finite and bool(np.all(zs>=self.kin.limits.lower) and np.all(zs<=self.kin.limits.upper))
        rate=finite and bool(np.all(np.abs(zs-q)<=step))
        residual=np.array([pose_error(t,self.geometry.forward(z))/self.scale
                           for t,z in zip(targets,zs)]) if finite else np.full((len(zs),6),np.inf)
        errors=np.stack((np.linalg.norm(residual[:,:3],axis=1),np.linalg.norm(residual[:,3:],axis=1)),axis=1)
        tau=float(max(0.,np.max(errors)-1))
        objective=float(.5*np.sum(((q-anchor)/step)**2)+.5*tau*tau)
        return dict(current=current,geometric=bool(current.accepted and bounds and rate),
                    future_bounds_ok=bounds,future_rate_ok=rate,tau=tau,objective=objective,
                    residuals=residual,normalized_errors=errors)

    def initialize_future(self, predicted, anchor, dt, step):
        """Two common geometric evaluations, no per-node IK and no witness input.

        The existing scaled nominal predictor supplies the expansion point.
        A common least-squares map supplies node INITIAL guesses only; the joint
        program keeps all z blocks free, including at deficient rank.
        """
        lo,hi=representable_interior(self.kin,IKQuery(predicted,anchor,dt),self.verifier)
        e,J,_=residual_linearization(self.geometry,predicted,anchor,self.scale)
        nominal=np.clip(anchor+np.linalg.lstsq(J*step,-e,rcond=1e-12)[0]*step,lo,hi)
        e,F,C,_=common_model(self.geometry,predicted,nominal,self.scale,step,self.cfg.rank_rtol)
        delta=np.linalg.lstsq(F,-(e+self.nodes@C.T).T,rcond=1e-12)[0].T
        return np.clip(nominal+delta*step,lo,hi)

    def solve(self, position, rotation, previous, dt=.02):
        start=perf_counter_ns()
        if dt != .02:raise ValueError('the frozen online contract uses dt=.02')
        query=IKQuery(Pose(np.asarray(position,float),np.asarray(rotation,float)),np.asarray(previous,float),dt)
        step=self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        predicted=predict_target(query.target,self.last_target)
        self.last_target=query.target
        targets=[perturb_target(predicted,w,1.,self.scale) for w in self.nodes]
        lower,upper=representable_interior(self.kin,query,self.verifier)
        phases=dict(input_prediction_ns=perf_counter_ns()-start)
        t=perf_counter_ns();backup=self.backup.solve(position,rotation,previous,dt)
        phases['backup_ns']=perf_counter_ns()-t
        anchor=np.asarray(backup['q'],float) if backup['accepted'] else query.previous_q.copy()
        q=anchor.copy();zs=np.tile(anchor,(len(self.nodes),1));B=None
        corrected=bool(self.mode=='free' and self.cfg.anchor_first)
        t=perf_counter_ns()
        if corrected:
            zs=self.initialize_future(predicted,anchor,dt,step)
        elif self.mode=='fixed':
            _,_,_,B=common_model(self.geometry,predicted,anchor,self.scale,step,self.cfg.rank_rtol)
            zs=anchor+self.nodes@B.T*step
        phases['initialization_ns']=perf_counter_ns()-t
        t=perf_counter_ns()
        initial=self.inspect(q,zs,query,targets,anchor,step)
        if corrected and backup['accepted']:
            # Keep the known feasible hold plan as an incumbent, not as every
            # node's default initial guess. Initial guesses never justify losing
            # a lower true objective already available at this exact anchor.
            held=np.tile(anchor,(len(self.nodes),1))
            held_info=self.inspect(anchor,held,query,targets,anchor,step)
            if held_info['geometric'] and (not initial['geometric'] or held_info['objective']<initial['objective']):
                zs,initial=held,held_info
        def verified_zero(q,zs,info):
            if not (info['geometric'] and np.array_equal(q,anchor) and info['tau']<=self.cfg.objective_atol):return False
            # The scalar tolerance only avoids unnecessary checks: it does NOT
            # establish zero. All original verifier checks and exact bounds/rates
            # are required. Canonical zero is justified by those actual checks.
            legal=all(self.verifier.check(z,IKQuery(goal,anchor,dt)).accepted for z,goal in zip(zs,targets))
            if legal:info['tau']=0.;info['objective']=0.
            return legal
        zero=verified_zero(q,zs,initial) if corrected else bool(initial['geometric'] and initial['objective']==0.)
        best=(q.copy(),zs.copy(),initial,B) if initial['geometric'] else None
        phases['initial_check_ns']=perf_counter_ns()-t
        records=[];trials=[];phases['linearization_ns']=0;phases['cone_build_ns']=0
        phases['cone_solve_ns']=0;phases['nonlinear_check_ns']=0
        stage_compute=dict(fixed_current_ns=0,joint_update_ns=0)
        zero_stage='initialization' if zero else None
        stages=([('fixed',self.cfg.fixed_updates)] if corrected and backup['accepted'] else [])+[('joint',self.cfg.outer_iterations)]
        for stage,limit in stages:
            for update in range(0 if zero else limit):
                if perf_counter_ns()-start>=self.cfg.total_soft_limit_ms*1e6:break
                outer=len(records);t=perf_counter_ns()
                u=(q-query.previous_q)/step;vs=(zs-query.previous_q)/step
                ec,Fc,_=residual_linearization(self.geometry,query.target,q,self.scale);Fc=Fc*step
                ef,Ff,C,B=common_model(self.geometry,predicted,zs[0],self.scale,step,self.cfg.rank_rtol)
                lin_ns=perf_counter_ns()-t;phases['linearization_ns']+=lin_ns
                ua=(anchor-query.previous_q)/step
                fixed=stage=='fixed'
                # Exactly the same sparse program; only q's lower/upper bounds
                # are clamped. A verified anchor must not be excluded by the
                # optional native pose-interior margin used in the released step.
                x,status=self.program.solve(ua,u,vs,
                    ua if fixed else (lower-query.previous_q)/step,
                    ua if fixed else (upper-query.previous_q)/step,
                    (self.kin.limits.lower-query.previous_q)/step,(self.kin.limits.upper-query.previous_q)/step,
                    ec,Fc,ef,Ff,C,self.nodes,B,current_radius=1. if fixed else None,
                    time_limit=max(1e-6,min(self.cfg.native_time_limit_ms/1000,
                        self.cfg.total_soft_limit_ms/1000-(perf_counter_ns()-start)/1e9)))
                records.append(dict(outer=outer,phase=stage,phase_update=update,**status))
                phases['cone_build_ns']+=status['build_ns'];phases['cone_solve_ns']+=status['solve_ns']
                stage_compute['fixed_current_ns' if fixed else 'joint_update_ns']+=lin_ns+status['build_ns']+status['solve_ns']
                if not np.isfinite(x).all():continue
                candidate_q=anchor.copy() if fixed else query.previous_q+step*x[:self.kin.nq]
                candidate_z=query.previous_q+step*x[self.kin.nq:-1].reshape(len(self.nodes),self.kin.nq)
                t=perf_counter_ns();adopted=False
                for alpha in (1.,.5,.25):
                    qt=anchor.copy() if fixed else q+alpha*(candidate_q-q)
                    zt=zs+alpha*(candidate_z-zs)
                    if self.mode=='fixed':zt=zt[0]+self.nodes@B.T*step
                    excess=max(float(np.max(lower-qt)),float(np.max(qt-upper)),0.)
                    if not fixed and excess<=1e-8:qt=np.clip(qt,lower,upper)
                    future_roundoff=0.
                    if corrected:
                        zl,zu=representable_interior(self.kin,IKQuery(predicted,qt,dt),self.verifier)
                        future_roundoff=max(float(np.max(zl-zt)),float(np.max(zt-zu)),0.)
                        # Same reconstruction-only threshold already used for q.
                        # No native/true constraint is relaxed, and substantial
                        # violations are not repaired. Recompute FK after clipping.
                        if future_roundoff<=1e-8:zt=np.clip(zt,zl,zu)
                    info=self.inspect(qt,zt,query,targets,anchor,step)
                    found_zero=verified_zero(qt,zt,info) if corrected else False
                    threshold=0 if best is None else self.cfg.objective_atol+self.cfg.objective_rtol*abs(best[2]['objective'])
                    improve=info['geometric'] and (best is None or info['objective']<best[2]['objective']-threshold or
                        (found_zero and best[2]['objective']>0))
                    trials.append(dict(outer=outer,phase=stage,alpha=alpha,q=qt.copy(),z=zt.copy(),
                        tau_actual=info['tau'],objective_actual=info['objective'],geometric=info['geometric'],
                        current_accepted=info['current'].accepted,current_reasons=list(info['current'].reasons),
                        future_bounds_ok=info['future_bounds_ok'],future_rate_ok=info['future_rate_ok'],
                        adopted=bool(improve),zero_verified=found_zero,fixed_B_normalized=B.copy() if self.mode=='fixed' else None))
                    if corrected:trials[-1]['future_roundoff_reconstruction_rad']=future_roundoff
                    if improve:
                        q,zs=qt,zt;best=(qt.copy(),zt.copy(),info,B.copy());adopted=True
                        if found_zero:zero=True;zero_stage=f'{stage}_{update+1}'
                        break
                phases['nonlinear_check_ns']+=perf_counter_ns()-t
                if zero or not adopted:break
        t=perf_counter_ns()
        if best is not None:
            q,zs,info,best_B=best
            check=self.verifier.check(q,query)
        else:
            q=anchor if backup['accepted'] else np.full(self.kin.nq,np.nan)
            check=self.verifier.check(q,query);info=None;best_B=None
        node_checks=[]
        if info is not None:
            node_checks=[self.verifier.check(z,IKQuery(target,q,dt)) for z,target in zip(zs,targets)]
        phases['final_verification_ns']=perf_counter_ns()-t
        changed=bool(check.accepted and backup['accepted'] and not np.array_equal(q,anchor))
        out=dict(method='tar_'+self.mode,q=q.tolist() if check.finite_ok else None,
            accepted=bool(check.accepted),finite=bool(check.finite_ok),
            joint_limit_ok=bool(check.joint_limit_ok),velocity_ok=bool(check.velocity_ok),
            position_error=float(check.position_error) if check.finite_ok else None,
            orientation_error=float(check.orientation_error) if check.finite_ok else None,
            verification_reasons=list(check.reasons),failure_kind='accepted' if check.accepted else '+'.join(check.reasons),
            internal_status=records[-1]['status'] if records else ('zero_objective' if zero else 'no_program'),
            internal_ok=bool(records and records[-1]['status'] in ('Solved','AlmostSolved')),
            optimization_called=bool(records),optimization_calls=len(records),
            decision='zero_objective_unchanged' if zero else ('optimized' if changed else
                ('geometric_recovery' if check.accepted and not backup['accepted'] else 'backup' if check.accepted else 'failure')),
            backup_used=bool(check.accepted and backup['accepted'] and not changed),backup=backup,
            initial_geometric=initial['geometric'],initial_tau=initial['tau'],initial_objective=initial['objective'],
            objective_actual=info['objective'] if info else None,tau_actual=info['tau'] if info else None,
            planned_z=zs.copy() if info else None,planned_normalized_errors=info['normalized_errors'] if info else None,
            planned_residuals=info['residuals'] if info else None,
            planned_node_verified=[bool(c.accepted) for c in node_checks],
            all_nodes_verified=bool(node_checks and all(c.accepted for c in node_checks)),
            fixed_B_normalized=best_B if self.mode=='fixed' else None,
            predicted_position=predicted.position,predicted_rotation=predicted.rotation,
            nominal_state_is_executable=False,command_changed=changed,
            intervention=float(np.linalg.norm((q-anchor)/step)) if check.finite_ok else None,
            max_difference_from_backup=float(np.max(np.abs(q-anchor))) if check.finite_ok else None,
            velocity_utilization=float(np.max(np.abs(q-query.previous_q)/step)) if check.finite_ok else None,
            native_status=records,nonlinear_trials=trials,phase_times_ns=phases)
        out.update(numerical_completion=corrected,zero_cost_verified=bool(corrected and zero and check.accepted and
            np.array_equal(q,anchor) and node_checks and all(c.accepted for c in node_checks)),zero_found_stage=zero_stage,
            numerical_phase_times_ns=dict(initialization_ns=phases['initialization_ns'],**stage_compute,
                fk_check_ns=phases['initial_check_ns']+phases['nonlinear_check_ns']+phases['final_verification_ns']))
        # Include conversion, full numerical work, acceptance, and record assembly;
        # disk serialization / experiment-only future diagnostics are excluded.
        elapsed=perf_counter_ns()-start
        out.update(total_latency_ns=elapsed,accounting_remainder_ns=elapsed-sum(phases.values()),
                   accepted_within_20ms=bool(check.accepted and elapsed<=20_000_000))
        return out
