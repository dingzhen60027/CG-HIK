"""Demand-satisfying, minimum command intervention; frozen CR-IK is untouched."""
from time import perf_counter, perf_counter_ns
import numpy as np

from ..types import Pose, IKQuery
from ..continuation_mechanism.observation import representable_interior
from .runtime import CorrectionReserveIK
from .geometry import residual_linearization, reserve_value
from .convex import subproblem
from .minimal_demand import PredictionDemand
from .minimal_convex import MinimalCone


MODES = {'minimal': 'cr_ik_minimal', 'fixed': 'cr_ik_minimal_fixed',
         'no_shortcut': 'cr_ik_minimal_no_shortcut'}


class MinimalInterventionIK(CorrectionReserveIK):
    def __init__(self, kin, verifier, source, library, urdf, *, demand_parameters,
                 mode='minimal', config=None):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(kin,verifier,source,library,urdf,mode='predictive',config=config)
        self.variant = mode
        self.demand_model = PredictionDemand(self.scale,demand_parameters['minimum'],
            demand_parameters['initial'],fixed=mode=='fixed',window=20)
        self.last_nominal = None
        self.cone = None  # no solver/program is created on a shortcut-only stream

    def reset(self, q=None):
        super().reset(q)
        self.demand_model.reset()
        self.last_nominal = None

    def pair_check(self, q, z, query, predicted, step, demand, current_known=False):
        current = None if current_known else self.verifier.check(q,query)
        current_ok = current_known or current.accepted
        if not current_ok:
            return dict(legal=False,demand_met=False,gamma=None,rank=None,current_check=current)
        next_check = self.verifier.check(z,IKQuery(predicted,q,query.dt))
        if not next_check.accepted:
            return dict(legal=False,demand_met=False,gamma=None,rank=None,current_check=current,
                        next_reasons=list(next_check.reasons))
        gamma,mapping,slack = reserve_value(self.counted,predicted,q,z,self.scale,step,self.config.rank_rtol)
        needed = demand*np.linalg.norm(mapping.matrix,axis=1) if mapping.full_rank else None
        met = bool(mapping.full_rank and np.all(needed <= slack))
        return dict(legal=True,demand_met=met,gamma=gamma,rank=mapping.rank,
            compensation_error=mapping.compensation_error,slack=slack,
            required_joint_correction=needed,current_check=current)

    def solve(self, position, rotation, previous, dt=.02):
        start = perf_counter_ns()
        if dt != .02:
            raise ValueError('This mode retains dt=0.02 s')
        self.counted.counts = dict(forward=0,jacobian=0)
        target = Pose(position,rotation); previous = np.asarray(previous,float)
        query = IKQuery(target,previous,dt)
        step = self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        conversion_ns = perf_counter_ns()-start
        phase = perf_counter_ns()
        backup = self.backup.solve(position,rotation,previous,dt)
        backup_ns = perf_counter_ns()-phase
        phase = perf_counter_ns()
        predicted, demand_record = self.demand_model.observe(target)
        self.last_target = target
        demand = demand_record['demand']
        demand_ns = perf_counter_ns()-phase
        phase = perf_counter_ns()
        lower,upper = representable_interior(self.kin,query,self.verifier)
        qb = np.asarray(backup['q'],float).copy() if backup['accepted'] else previous.copy()
        zl,zu = representable_interior(self.kin,IKQuery(predicted,qb,dt),self.verifier)
        warm_used = self.last_nominal is not None
        warm = np.clip(self.last_nominal,zl,zu) if warm_used else qb.copy()
        e,J,_ = residual_linearization(self.counted,predicted,warm,self.scale)
        delta = np.linalg.lstsq(J*step,-e,rcond=1e-12)[0]*step
        initial_z = np.clip(warm+delta,zl,zu)
        initial = self.pair_check(qb,initial_z,query,predicted,step,demand,current_known=backup['accepted'])
        initial_pair_ns = perf_counter_ns()-phase
        direct = bool(backup['accepted'] and initial['demand_met'] and self.variant!='no_shortcut')
        initial_good = bool(backup['accepted'] and initial['demand_met'])
        best = dict(q=qb.copy(),z=initial_z.copy(),check=initial,cost=0.,iteration=-1) if initial_good else None
        q0,z0 = qb.copy(),initial_z.copy()
        stages=[];construction_ns=solve_ns=validation_ns=0;updates=0;recovery=not backup['accepted']
        deadline=start/1e9+self.config.total_soft_limit_ms/1000
        if not direct:
            for iteration in range(2):  # one solve, at most one extra linearization
                if perf_counter() >= deadline:
                    stages.append(dict(status='outer_soft_limit',solver_called=False));break
                if recovery:
                    # The frozen same-core predictive feasibility path. It never
                    # changes the actual feedback until the original verifier accepts.
                    before = perf_counter_ns()
                    x,details = subproblem(self.counted,self.verifier,target,predicted,previous,
                        q0,z0,lower,upper,'predictive',self.config,deadline)
                    duration = perf_counter_ns()-before
                    native_duration = int(sum(s.get('native_seconds',0) for s in details)*1e9)
                    build_duration = max(0,duration-native_duration)
                    construction_ns += build_duration; solve_ns += duration-build_duration
                    stages.extend(dict(update=iteration,recovery=True,solver_called='stage' in d,**d) for d in details)
                else:
                    before = perf_counter_ns()
                    if self.cone is None:
                        self.cone = MinimalCone(self.kin.nq,self.config)
                    lazy_ns = perf_counter_ns()-before
                    x,detail = self.cone.solve(self.counted,target,predicted,previous,q0,z0,qb,initial_z,
                        lower,upper,self.scale,step,demand,deadline)
                    detail['construction_ns'] += lazy_ns
                    construction_ns += detail['construction_ns']; solve_ns += detail['solve_ns']
                    stages.append(dict(update=iteration,recovery=False,**detail))
                if x is None:
                    break
                q1 = previous+step*x[:self.kin.nq]
                z1 = previous+step*x[self.kin.nq:2*self.kin.nq]
                updates += 1
                phase = perf_counter_ns()
                found=False
                for alpha in (1.,.5,.25):
                    qt,zt = q0+alpha*(q1-q0),z0+alpha*(z1-z0)
                    check = self.pair_check(qt,zt,query,predicted,step,demand)
                    if check['legal'] and (check['demand_met'] or recovery):
                        cost = float(np.sum(((qt-qb)/step)**2)+np.sum(((zt-initial_z)/step)**2))
                        if best is None or cost < best['cost']:
                            best = dict(q=qt.copy(),z=zt.copy(),check=check,cost=cost,iteration=iteration)
                        found=True;break
                    if perf_counter() >= deadline:
                        break
                validation_ns += perf_counter_ns()-phase
                q0,z0 = q1,z1
                if found:
                    break
        # Outside recovery, only a truly demand-satisfying pair may replace backup.
        selected = best is not None and best['iteration'] >= 0
        raw = best['q'] if selected else (np.asarray(backup['q'],float) if backup['q'] is not None
                                        else np.full(self.kin.nq,np.nan))
        selected_pair = best if best is not None else (
            dict(q=qb,z=initial_z,check=initial,cost=0.,iteration=-1) if backup['accepted'] and initial['legal'] else None)
        phase = perf_counter_ns()
        final = self.verifier.check(raw,query) if selected else None
        # An unchanged backup already passed this same verifier during the timed
        # backup call. Do not perform redundant FK merely to relabel it.
        validation_ns += perf_counter_ns()-phase
        accepted = bool(final.accepted) if final is not None else bool(backup['accepted'])
        finite = bool(final.finite_ok) if final is not None else bool(backup['finite'])
        met = bool(accepted and selected_pair is not None and selected_pair['check']['demand_met'])
        self.last_nominal = selected_pair['z'].copy() if accepted and selected_pair is not None else None
        equal_backup = bool(backup['accepted'] and np.array_equal(raw,np.asarray(backup['q'])))
        intervention = float(np.linalg.norm((raw-np.asarray(backup['q']))/step)) if accepted and backup['accepted'] else None
        elapsed = perf_counter_ns()-start
        out = dict(backup)
        if final is not None:
            out.update(q=raw.tolist() if final.finite_ok else None,finite=bool(final.finite_ok),
                accepted=bool(final.accepted),joint_limit_ok=bool(final.joint_limit_ok),velocity_ok=bool(final.velocity_ok),
                position_error=float(final.position_error) if final.finite_ok else None,
                orientation_error=float(final.orientation_error) if final.finite_ok else None,
                verification_reasons=list(final.reasons))
        out.update(method=MODES[self.variant],internal_ok=bool(selected or backup['internal_ok']),
            internal_status='verified_recovery_pair' if selected and recovery else
                'verified_minimal_pair' if selected else 'verified_backup_no_intervention' if direct else 'trac_backup',
            total_latency_ns=elapsed,conversion_ns=conversion_ns,backup_ns=backup_ns,
            demand_estimation_ns=demand_ns,initial_pair_ns=initial_pair_ns,
            cone_construction_ns=construction_ns,cone_solve_ns=solve_ns,nonlinear_validation_ns=validation_ns,
            optimization_ns=construction_ns+solve_ns+validation_ns,
            backup_verification_ns=backup['verification_ns'],
            timing_remainder_ns=elapsed-conversion_ns-backup_ns-demand_ns-initial_pair_ns-construction_ns-solve_ns-validation_ns,
            backup_used=bool(not selected or equal_backup),backup_accepted=bool(backup['accepted']),backup_q=backup['q'],
            backup_total_latency_ns=backup['total_latency_ns'],backup_internal_ok=backup['internal_ok'],
            returned_within_5ms=elapsed<=5_000_000,returned_within_20ms=elapsed<=20_000_000,
            accepted_within_20ms=bool(accepted and elapsed<=20_000_000),
            predicted_position=predicted.position.tolist(),predicted_rotation=predicted.rotation.tolist(),
            nominal_next_q=selected_pair['z'].tolist() if selected_pair is not None else None,
            nominal_next_verified=bool(accepted and selected_pair is not None),
            predicted_gamma=selected_pair['check']['gamma'] if selected_pair is not None else None,
            demand_met=met,initial_pair_legal=bool(initial['legal']),initial_demand_met=initial_good,
            initial_pair_gamma=initial['gamma'],direct_return=direct,
            command_unchanged=equal_backup,intervention_normalized_l2=intervention,
            max_difference_from_backup=float(np.max(np.abs(raw-np.asarray(backup['q'])))) if finite and backup['q'] is not None else None,
            optimization_called=any(s.get('solver_called',False) for s in stages),
            conic_calls=sum(bool(s.get('solver_called',False)) for s in stages),
            linearization_updates=updates,native_status=stages,recovery_mode=recovery,warm_start_used=warm_used,
            initial_nominal_q=initial_z.tolist(),correction_rank=selected_pair['check']['rank'] if selected_pair is not None else None,
            public_python_kinematics_calls=dict(self.counted.counts),**demand_record)
        # Native subcall measurements are retained under explicit backup names;
        # do not present their old accounting as the new method's total timing.
        for key in ('bounds_setup_ns','solve_ns','verification_ns','accounting_remainder_ns'):
            if key in out:
                out['backup_'+key]=out.pop(key)
        out['taxonomy']=('internal_success' if out['internal_ok'] else 'internal_failure')+(
            '__task_accept' if accepted else '__task_reject')
        out['failure_kind']='accepted' if accepted else '+'.join(out['verification_reasons'])
        out['velocity_utilization']=float(np.max(np.abs(self.kin.difference(raw,previous))/step)) if finite else None
        return out
