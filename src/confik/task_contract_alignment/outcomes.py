"""Three independent endpoints; a numerical return code never grants acceptance."""
from time import perf_counter_ns
import numpy as np
from ..types import Pose, IKQuery
from .contract import task_interval
from .trac_adapter import NativeTrac
from .dls_adapter import NativeDLS


def taxonomy(internal_ok, accepted):
    return ('internal_success' if internal_ok else 'internal_failure')+'__'+('task_accept' if accepted else 'task_reject')


class ContractSolver:
    def __init__(self, method, kin, verifier, source, library=None, urdf=None):
        self.method,self.kin,self.verifier=method,kin,verifier
        self.native=NativeTrac(method,library,urdf,kin,verifier) if method.startswith('trac') else NativeDLS(method,kin,verifier,source)

    def solve(self, position, rotation, previous, dt=.02, trace=False):
        start=perf_counter_ns()
        query=IKQuery(Pose(np.ascontiguousarray(position,dtype=float),np.ascontiguousarray(rotation,dtype=float)),
                      np.ascontiguousarray(previous,dtype=float),float(dt))
        conversion=perf_counter_ns()-start
        setup_start=perf_counter_ns()
        lower,upper=task_interval(self.kin,query,self.verifier)
        seed=np.ascontiguousarray(np.clip(query.previous_q,lower,upper))
        setup=perf_counter_ns()-setup_start
        solve_start=perf_counter_ns()
        if self.method.startswith('trac'):
            raw,result=self.native.run(query,seed,lower,upper)
        else:
            raw,result=self.native.run(query,seed,lower,upper,trace=trace)
        solve_outer=perf_counter_ns()-solve_start
        verify_start=perf_counter_ns()
        check=self.verifier.check(raw,query)
        verification=perf_counter_ns()-verify_start
        elapsed=perf_counter_ns()-start
        native_conv=result.get('native_conversion_ns',0); native_setup=result.get('native_setup_ns',0)
        measured_solve=result.get('native_solve_ns',solve_outer)
        result.update(method=self.method,q=raw.tolist() if check.finite_ok else None,
            finite=bool(check.finite_ok),accepted=bool(check.accepted),
            joint_limit_ok=bool(check.joint_limit_ok),velocity_ok=bool(check.velocity_ok),
            verification_reasons=list(check.reasons),
            position_error=float(check.position_error) if check.finite_ok else None,
            orientation_error=float(check.orientation_error) if check.finite_ok else None,
            total_latency_ns=elapsed,conversion_ns=conversion+native_conv,
            bounds_setup_ns=setup+native_setup,solve_ns=measured_solve,verification_ns=verification,
            accounting_remainder_ns=elapsed-conversion-native_conv-setup-native_setup-measured_solve-verification,
            returned_within_5ms=elapsed<=5_000_000,returned_within_20ms=elapsed<=20_000_000,
            accepted_within_20ms=bool(check.accepted and elapsed<=20_000_000),
            lower=lower.tolist(),upper=upper.tolist(),seed_q=seed.tolist(),
            budget_ms=(20 if '20ms' in self.method else 5) if self.method.startswith('trac') else None,
            taxonomy=taxonomy(result['internal_ok'],check.accepted),trace_instrumented=trace)
        result['failure_kind']='accepted' if check.accepted else '+'.join(
            ([] if result['internal_ok'] else ['internal_nonconvergence'])+list(check.reasons))
        result['velocity_utilization']=float(np.max(np.abs(self.kin.difference(raw,query.previous_q))/
            (self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance))) if check.finite_ok else None
        if trace and not self.method.startswith('trac'):
            result['dls_trace']=self.native.trace(query,result)
        return result

    def close(self):
        self.native.close()
