"""Observe the original AdaptiveDLS without changing its operations or stops.

The kinematics wrapper captures actual FK evaluations, not re-executed solves.
Only point runs enable tracing; BOTH DLS settings pay the same instrumentation.
Trace postprocessing/extra verifier replay is outside reported solve latency.
"""
from dataclasses import replace
import inspect
import sys
import numpy as np
from ..solvers.dls import AdaptiveDLS, DLSConfig
from ..continuation_mechanism.tolerance_solvers import QueryBoundedKinematics
from ..geometry import pose_error
from .contract import STRICT


class ObservedKinematics(QueryBoundedKinematics):
    def __init__(self, kin):
        super().__init__(kin)
        lines, start = inspect.getsourcelines(AdaptiveDLS.solve)
        self.iterate_line = start + next(i for i,s in enumerate(lines) if 'current = self.kinematics.forward(q)' in s)
        self.events = []
        self.record = False

    def forward(self, q):
        pose = self.kin.forward(q)
        if self.record:
            caller = sys._getframe(1)
            self.events.append((q.copy(), pose, int(caller.f_locals.get('iteration', -1)),
                caller.f_lineno == self.iterate_line, 'trial_q' in caller.f_locals and caller.f_lineno != self.iterate_line))
        return pose


class NativeDLS:
    def __init__(self, method, kin, verifier, source):
        self.kin = ObservedKinematics(kin)
        self.verifier = verifier
        cfg = DLSConfig(**source['solver'])
        cfg = replace(cfg, position_tolerance=STRICT if method=='dls_strict' else verifier.config.position_tolerance,
                      orientation_tolerance=STRICT if method=='dls_strict' else verifier.config.orientation_tolerance)
        self.dls = AdaptiveDLS(self.kin, cfg)

    def run(self, query, seed, lower, upper, trace=False):
        self.kin.lower,self.kin.upper = lower,upper
        self.kin.events=[]; self.kin.record=trace
        result=self.dls.solve(query.target,seed,25,seed_source='actual_previous_q')
        self.kin.record=False
        return result.q, dict(internal_ok=bool(result.converged),internal_status=result.reason,
            native_return_code=None,iterations=int(result.iterations),
            solver_function_evaluations=int(result.function_evaluations))

    def trace(self, query, result):
        rows=[]
        for i,(q,pose,iteration,is_iterate,_) in enumerate(self.kin.events):
            check=self.verifier.check(q,query)
            rows.append(dict(fev_index=i+1,iteration=iteration,is_iterate=is_iterate,q=q.tolist(),
                residual_vector=pose_error(query.target,pose).tolist(),position_error=float(check.position_error),
                orientation_error=float(check.orientation_error),task_admissible=bool(check.accepted)))
        first=next((r for r in rows if r['is_iterate'] and r['task_admissible']),None)
        first_eval=next((r['fev_index'] for r in rows if r['task_admissible']),None)
        return dict(evaluations=rows,first_admissible_evaluation=first_eval,
            first_admissible_iteration=first['iteration'] if first else None,
            stopping_iteration=result['iterations'],
            excess_iterations_after_task_admissibility=result['iterations']-first['iteration'] if first else None,
            excess_fk_residual_evaluations=result['solver_function_evaluations']-first['fev_index'] if first else None,
            extra_time_ns=None,
            extra_time_note='not separately estimated: logging perturbs phase timing; exact observed iteration/evaluation counts only')

    def close(self):
        pass
