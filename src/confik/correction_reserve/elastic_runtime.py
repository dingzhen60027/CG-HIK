"""Elastic correction demand with nonlinear, cost-based command acceptance.

Only the added correction reserve can have a shortfall. Current and nominal
next commands always use the original independent task verifier.
"""
from time import perf_counter, perf_counter_ns
import numpy as np
from ..types import Pose, IKQuery
from ..continuation_mechanism.observation import representable_interior
from .minimal_runtime import MinimalInterventionIK
from .geometry import residual_linearization
from .convex import subproblem
from .elastic_convex import ElasticCone

MU_METHODS={0.:'cr_ik_elastic_mu0',.25:'cr_ik_elastic_mu025',1.:'cr_ik_elastic_mu1',4.:'cr_ik_elastic_mu4'}
OBJECTIVE_ATOL=1e-10
OBJECTIVE_RTOL=1e-9


def objective(q,z,backup_q,initial_z,step,demand,gamma,mu):
    xi=max(0.,float(demand)-float(gamma))
    movement=.5*float(np.sum(((q-backup_q)/step)**2)+np.sum(((z-initial_z)/step)**2))
    return movement+.5*mu*(xi/demand)**2,xi,movement


def improves(candidate,reference):
    return bool(candidate < reference-(OBJECTIVE_ATOL+OBJECTIVE_RTOL*abs(reference)))


class ElasticInterventionIK(MinimalInterventionIK):
    def __init__(self,*args,mu=1.,**kwargs):
        if mu not in MU_METHODS:raise ValueError('mu must be 0, 0.25, 1 or 4')
        super().__init__(*args,mode='minimal',**kwargs)
        self.mu=float(mu)
        if min(self.demand_model.minimum,self.demand_model.initial)<=0:raise ValueError('r must be positive')

    def evaluate(self,q,z,query,predicted,step,demand,qb,zi,current_known=False):
        check=self.pair_check(q,z,query,predicted,step,demand,current_known)
        if check['legal']:
            value,xi,movement=objective(q,z,qb,zi,step,demand,check['gamma'],self.mu)
            check.update(objective=value,xi_actual=xi,movement_cost=movement,
                reserve_map_available=bool(check['rank']==6 and check['compensation_error']<=1e-7))
        else:check.update(objective=None,xi_actual=None,movement_cost=None,reserve_map_available=False)
        return check

    def solve(self,position,rotation,previous,dt=.02):
        start=perf_counter_ns()
        if dt!=.02:raise ValueError('frozen dt=0.02 only')
        self.counted.counts=dict(forward=0,jacobian=0)
        target=Pose(position,rotation);previous=np.asarray(previous,float)
        query=IKQuery(target,previous,dt);step=self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        conversion_ns=perf_counter_ns()-start;phase=perf_counter_ns()
        backup=self.backup.solve(position,rotation,previous,dt);backup_ns=perf_counter_ns()-phase
        phase=perf_counter_ns();predicted,demand_record=self.demand_model.observe(target)
        self.last_target=target;demand=demand_record['demand'];demand_ns=perf_counter_ns()-phase
        phase=perf_counter_ns();lower,upper=representable_interior(self.kin,query,self.verifier)
        qb=np.asarray(backup['q'],float).copy() if backup['accepted'] else previous.copy()
        zl,zu=representable_interior(self.kin,IKQuery(predicted,qb,dt),self.verifier)
        warm_used=self.last_nominal is not None
        warm=np.clip(self.last_nominal,zl,zu) if warm_used else qb.copy()
        e,J,_=residual_linearization(self.counted,predicted,warm,self.scale)
        delta=np.linalg.lstsq(J*step,-e,rcond=1e-12)[0]*step;zi=np.clip(warm+delta,zl,zu)
        initial=self.evaluate(qb,zi,query,predicted,step,demand,qb,zi,current_known=backup['accepted'])
        initial_ns=perf_counter_ns()-phase
        initial_legal=bool(backup['accepted'] and initial['legal'])
        direct=bool(initial_legal and initial['demand_met'])
        best=dict(q=qb.copy(),z=zi.copy(),check=initial,iteration=-1) if initial_legal else None
        recovery=not initial_legal
        q0,z0=qb.copy(),zi.copy();stages=[];trials=[];build_ns=solve_ns=validation_ns=updates=0
        deadline=start/1e9+self.config.total_soft_limit_ms/1000
        if not direct:
            for iteration in range(2):
                if perf_counter()>=deadline:
                    stages.append(dict(status='outer_soft_limit',solver_called=False));break
                if recovery:
                    # The shared, frozen ordinary predictive feasibility fallback;
                    # all mu values use it when the initial pair is unavailable.
                    phase=perf_counter_ns()
                    x,details=subproblem(self.counted,self.verifier,target,predicted,previous,
                        q0,z0,lower,upper,'predictive',self.config,deadline)
                    duration=perf_counter_ns()-phase
                    native=int(sum(d.get('native_seconds',0) for d in details)*1e9)
                    build=max(0,duration-native);build_ns+=build;solve_ns+=duration-build
                    stages.extend(dict(update=iteration,recovery=True,solver_called='stage' in d,**d) for d in details)
                else:
                    phase=perf_counter_ns()
                    if self.cone is None:self.cone=ElasticCone(self.kin.nq,self.config,self.mu)
                    lazy=perf_counter_ns()-phase
                    x,detail=self.cone.solve(self.counted,target,predicted,previous,q0,z0,qb,zi,
                        lower,upper,self.scale,step,demand,deadline)
                    detail['construction_ns']+=lazy;build_ns+=detail['construction_ns'];solve_ns+=detail['solve_ns']
                    stages.append(dict(update=iteration,recovery=False,**detail))
                if x is None:break
                q1=previous+step*x[:self.kin.nq];z1=previous+step*x[self.kin.nq:2*self.kin.nq]
                updates+=1;phase=perf_counter_ns();found=False
                for alpha in (1.,.5,.25):
                    qt,zt=q0+alpha*(q1-q0),z0+alpha*(z1-z0)
                    check=self.evaluate(qt,zt,query,predicted,step,demand,qb,zi)
                    better=bool(check['legal'] and (best is None or improves(check['objective'],best['check']['objective'])))
                    current=check.get('current_check')
                    trials.append(dict(update=iteration,alpha=alpha,q=qt.tolist(),z=zt.tolist(),
                        legal=bool(check['legal']),demand_met=bool(check['demand_met']),
                        gamma_actual=check['gamma'],xi_actual=check['xi_actual'],actual_objective=check['objective'],
                        rank=check['rank'],reserve_map_available=check['reserve_map_available'],
                        current_reasons=list(current.reasons) if current is not None else [],
                        next_reasons=check.get('next_reasons',[]),objective_improved=better,
                        compared_objective=best['check']['objective'] if best is not None else None,
                        hard_demand_would_reject=bool(check['legal'] and not check['demand_met'])))
                    if better:
                        best=dict(q=qt.copy(),z=zt.copy(),check=check,iteration=iteration);found=True;break
                    if perf_counter()>=deadline:break
                validation_ns+=perf_counter_ns()-phase;q0,z0=q1,z1
                if found:break
        selected=best is not None and best['iteration']>=0
        raw=best['q'] if selected else (np.asarray(backup['q'],float) if backup['q'] is not None else np.full(self.kin.nq,np.nan))
        phase=perf_counter_ns();final=self.verifier.check(raw,query) if selected else None
        validation_ns+=perf_counter_ns()-phase
        accepted=bool(final.accepted) if final is not None else bool(backup['accepted'])
        finite=bool(final.finite_ok) if final is not None else bool(backup['finite'])
        selected_check=best['check'] if best is not None else None
        met=bool(accepted and selected_check is not None and selected_check['demand_met'])
        self.last_nominal=best['z'].copy() if accepted and best is not None else None
        unchanged=bool(backup['accepted'] and np.array_equal(raw,backup['q']))
        partial=bool(selected and initial_legal and not met and selected_check['xi_actual']<initial['xi_actual'])
        if selected and recovery:decision='geometric_recovery'
        elif direct or (selected and met):decision='full_demand_satisfied'
        elif partial:decision='partial_effective_correction'
        elif selected:decision='objective_improvement'
        else:decision='backup_without_improvement' if backup['accepted'] else 'no_legal_command'
        if selected or direct:fallback_reason=None
        elif any(t['legal'] for t in trials):fallback_reason='no_actual_objective_improvement'
        elif trials:fallback_reason='nonlinear_pair_rejected'
        elif any(d.get('solver_called',False) for d in stages):fallback_reason='convex_solve_unsuccessful'
        else:fallback_reason='outer_soft_limit'
        intervention=float(np.linalg.norm((raw-np.asarray(backup['q']))/step)) if accepted and backup['accepted'] else None
        elapsed=perf_counter_ns()-start
        out=dict(backup)
        if final is not None:
            out.update(q=raw.tolist() if final.finite_ok else None,finite=bool(final.finite_ok),accepted=bool(final.accepted),
                joint_limit_ok=bool(final.joint_limit_ok),velocity_ok=bool(final.velocity_ok),
                position_error=float(final.position_error) if final.finite_ok else None,
                orientation_error=float(final.orientation_error) if final.finite_ok else None,verification_reasons=list(final.reasons))
        out.update(method=MU_METHODS[self.mu],mu=self.mu,internal_ok=bool(selected or backup['internal_ok']),
            internal_status=decision,decision=decision,fallback_reason=fallback_reason,
            total_latency_ns=elapsed,conversion_ns=conversion_ns,backup_ns=backup_ns,
            demand_estimation_ns=demand_ns,initial_pair_ns=initial_ns,cone_construction_ns=build_ns,
            cone_solve_ns=solve_ns,nonlinear_validation_ns=validation_ns,optimization_ns=build_ns+solve_ns+validation_ns,
            timing_remainder_ns=elapsed-conversion_ns-backup_ns-demand_ns-initial_ns-build_ns-solve_ns-validation_ns,
            backup_used=bool(not selected or unchanged),backup_accepted=bool(backup['accepted']),backup_q=backup['q'],
            backup_total_latency_ns=backup['total_latency_ns'],backup_internal_ok=backup['internal_ok'],
            returned_within_5ms=elapsed<=5_000_000,returned_within_20ms=elapsed<=20_000_000,
            accepted_within_20ms=bool(accepted and elapsed<=20_000_000),
            predicted_position=predicted.position.tolist(),predicted_rotation=predicted.rotation.tolist(),
            nominal_next_q=best['z'].tolist() if best is not None else None,nominal_next_verified=bool(accepted and best is not None),
            predicted_gamma=selected_check['gamma'] if selected_check is not None else None,
            xi_actual=selected_check['xi_actual'] if selected_check is not None else None,
            actual_objective=selected_check['objective'] if selected_check is not None else None,
            initial_objective=initial['objective'],initial_xi_actual=initial['xi_actual'],
            actual_objective_improvement=initial['objective']-selected_check['objective'] if initial_legal and best is not None else None,
            actual_shortfall_reduction=initial['xi_actual']-selected_check['xi_actual'] if initial_legal and best is not None else None,
            objective_comparison_atol=OBJECTIVE_ATOL,objective_comparison_rtol=OBJECTIVE_RTOL,
            demand_met=met,partial_correction_adopted=partial,optimized_pair_selected=selected,
            initial_pair_legal=bool(initial['legal']),initial_demand_met=bool(initial['demand_met']),
            initial_pair_gamma=initial['gamma'],initial_pair_rank=initial['rank'],direct_return=direct,
            command_unchanged=unchanged,effective_command_adjustment=bool(accepted and backup['accepted'] and not unchanged),
            intervention_normalized_l2=intervention,
            max_difference_from_backup=float(np.max(np.abs(raw-np.asarray(backup['q'])))) if finite and backup['q'] is not None else None,
            optimization_called=any(d.get('solver_called',False) for d in stages),conic_calls=sum(bool(d.get('solver_called',False)) for d in stages),
            linearization_updates=updates,native_status=stages,nonlinear_trials=trials,recovery_mode=recovery,
            backup_failed_recovery=bool(not backup['accepted']),warm_start_used=warm_used,initial_nominal_q=zi.tolist(),
            correction_rank=selected_check['rank'] if selected_check is not None else None,
            reserve_map_available=selected_check['reserve_map_available'] if selected_check is not None else False,
            public_python_kinematics_calls=dict(self.counted.counts),**demand_record)
        for key in ('bounds_setup_ns','solve_ns','verification_ns','accounting_remainder_ns'):
            if key in out:out['backup_'+key]=out.pop(key)
        out['taxonomy']=('internal_success' if out['internal_ok'] else 'internal_failure')+('__task_accept' if accepted else '__task_reject')
        out['failure_kind']='accepted' if accepted else '+'.join(out['verification_reasons'])
        out['velocity_utilization']=float(np.max(np.abs(self.kin.difference(raw,previous))/step)) if finite else None
        return out
