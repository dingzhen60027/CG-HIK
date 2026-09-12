"""Standard bounded SciPy TRF with a current-command task contract.

No predictive policy, scenario program, candidate pool, or learned component.
The optional composition makes one TRAC call and at most one TRF call, sharing
one absolute 20 ms soft deadline. Callback checks cannot preempt native work.
"""
from time import perf_counter_ns

import numpy as np
from scipy.optimize import least_squares

from .types import IKQuery, Pose
from .correction_reserve.native_geometry import NativeGeometry
from .correction_reserve.geometry import residual_linearization, task_scale
from .continuation_mechanism.observation import representable_interior


SETTINGS=dict(max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10,
              method='trf',x_scale=1.,tr_solver='exact',loss='linear')


class CurrentFrameTRF:
    method='current_trf'

    def __init__(self,kin,verifier,urdf):
        self.kin,self.verifier=kin,verifier
        self.geometry=NativeGeometry(kin,urdf)
        self.scale=task_scale(verifier)

    def reset(self,q):pass
    def close(self):pass

    def solve(self,position,rotation,previous,dt=.02,*,deadline_ns=None):
        start=perf_counter_ns()
        deadline=start+20_000_000 if deadline_ns is None else int(deadline_ns)
        query=IKQuery(Pose(np.asarray(position,float),np.asarray(rotation,float)),np.asarray(previous,float),float(dt))
        step=self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        phase={'input_ns':perf_counter_ns()-start}
        t=perf_counter_ns();lower,upper=representable_interior(self.kin,query,self.verifier)
        lo=(lower-query.previous_q)/step;hi=(upper-query.previous_q)/step
        x0=np.clip(np.zeros(self.kin.nq),lo,hi)
        phase['bounds_setup_ns']=perf_counter_ns()-t
        counts=dict(residual_evaluations=0,analytic_jacobian_evaluations=0,jacobian_requests=0,verifier_calls=0,iterations=0)
        trace=[];cache_x=None;cache_e=cache_J=None;roundoff=0.
        def decode(x):
            nonlocal roundoff
            raw=query.previous_q+step*x
            q=np.clip(raw,lower,upper)
            error=float(np.max(np.abs(q-raw)));roundoff=max(roundoff,error)
            if error>1e-12:raise ValueError('non-roundoff bound violation from native TRF')
            return q
        def evaluate(x):
            nonlocal cache_x,cache_e,cache_J
            if cache_x is None or not np.array_equal(cache_x,x):
                cache_e,J,_=residual_linearization(self.geometry,query.target,decode(x),self.scale)
                cache_J=J*step;cache_x=np.array(x,copy=True)
                counts['residual_evaluations']+=1;counts['analytic_jacobian_evaluations']+=1
            return cache_e
        def jac(x):
            counts['jacobian_requests']+=1;evaluate(x);return cache_J
        def check(q):
            counts['verifier_calls']+=1
            return self.verifier.check(q,query)
        t=perf_counter_ns();initial=check(query.previous_q)
        phase['initial_verification_ns']=perf_counter_ns()-t
        status='initial_command_admissible' if initial.accepted else 'not_started'
        native=None;stop_reason=None;callback_verify_ns=0
        q=query.previous_q.copy() if initial.accepted else decode(x0)
        def callback(intermediate_result):
            nonlocal stop_reason,callback_verify_ns
            counts['iterations']+=1
            at=perf_counter_ns();current=decode(intermediate_result.x);verdict=check(current)
            callback_verify_ns+=perf_counter_ns()-at
            trace.append(dict(iteration=counts['iterations'],nfev=int(intermediate_result.nfev),
                q=current.copy(),position_error=verdict.position_error,orientation_error=verdict.orientation_error,
                accepted=verdict.accepted,elapsed_ns=perf_counter_ns()-start))
            if verdict.accepted:
                stop_reason='task_admissible';raise StopIteration
            if perf_counter_ns()>=deadline:
                stop_reason='deadline';raise StopIteration
        t=perf_counter_ns()
        if not initial.accepted:
            if perf_counter_ns()>=deadline:status='deadline_before_solve'
            else:
                native=least_squares(evaluate,x0,jac=jac,bounds=(lo,hi),callback=callback,**SETTINGS)
                q=decode(native.x)
                status=stop_reason or 'native_termination'
        phase['trf_including_callback_ns']=perf_counter_ns()-t
        t=perf_counter_ns();final=check(q)
        phase['final_verification_ns']=perf_counter_ns()-t
        out=dict(method=self.method,q=q.tolist(),accepted=bool(final.accepted),finite=bool(final.finite_ok),
            joint_limit_ok=bool(final.joint_limit_ok),velocity_ok=bool(final.velocity_ok),
            position_error=float(final.position_error),orientation_error=float(final.orientation_error),
            verification_reasons=list(final.reasons),failure_kind='accepted' if final.accepted else '+'.join(final.reasons),
            internal_status=status,internal_ok=bool(native is not None and native.success),
            native_return_code=int(native.status) if native is not None else None,
            native_message=str(native.message) if native is not None else None,
            native_nfev=int(native.nfev) if native is not None else 0,
            native_njev=int(native.njev) if native is not None else 0,
            lower=lower.tolist(),upper=upper.tolist(),step_scale=step.tolist(),task_scale=self.scale.tolist(),
            seed_q=decode(x0).tolist(),initial_previous_admissible=bool(initial.accepted),
            velocity_utilization=float(np.max(np.abs(q-query.previous_q)/step)),
            max_joint_step_rad=float(np.max(np.abs(q-query.previous_q))),
            optimizer_called=native is not None,counts=counts,iteration_trace=trace,
            callback_verification_ns=callback_verify_ns,callback_verification_is_nested_in_trf_time=True,
            roundoff_reconstruction_rad=roundoff,phase_times_ns=phase,
            available_budget_at_entry_ns=max(0,deadline-start),deadline_reason=stop_reason=='deadline' or status=='deadline_before_solve')
        elapsed=perf_counter_ns()-start
        out.update(total_latency_ns=elapsed,accounting_remainder_ns=elapsed-sum(phase.values()),
            accepted_within_20ms=bool(final.accepted and elapsed<=20_000_000),
            absolute_deadline_exceeded=perf_counter_ns()>deadline)
        return out


class TracThenCurrentTRF:
    method='trac_then_current_trf'

    def __init__(self,trac,trf):self.trac,self.trf=trac,trf
    def reset(self,q):self.trf.reset(q)
    def close(self):self.trac.close();self.trf.close()

    def solve(self,position,rotation,previous,dt=.02):
        start=perf_counter_ns();deadline=start+20_000_000
        trac=self.trac.solve(position,rotation,previous,dt)
        trac_outer=perf_counter_ns()-start
        remaining=deadline-perf_counter_ns();recovery=None;recovery_outer=0
        if not trac['accepted'] and remaining>0:
            t=perf_counter_ns()
            recovery=self.trf.solve(position,rotation,previous,dt,deadline_ns=deadline)
            recovery_outer=perf_counter_ns()-t
        selected=trac if recovery is None else recovery
        result=dict(selected)
        result.update(method=self.method,trac_result=trac,trf_result=recovery,
            recovery_called=recovery is not None,recovery_accepted=bool(recovery and recovery['accepted']),
            remaining_before_trf_ns=max(0,remaining),trac_outer_ns=trac_outer,trf_outer_ns=recovery_outer,
            decision='trac_accepted' if trac['accepted'] else 'single_trf' if recovery is not None else 'deadline_no_recovery')
        elapsed=perf_counter_ns()-start
        result.update(total_latency_ns=elapsed,accepted_within_20ms=bool(result['accepted'] and elapsed<=20_000_000),
            composition_remainder_ns=elapsed-trac_outer-recovery_outer)
        return result
