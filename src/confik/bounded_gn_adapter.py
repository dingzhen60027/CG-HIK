"""Copy alongside confik/current_frame_trf.py; no TRAC instance is created.

Requires bounded_gn.py in the same confik package. Use the existing repository
runner/factory to instantiate this class; no new experiment framework required.
"""
from time import perf_counter_ns
import numpy as np
from .bounded_gn import BoundedGN,Settings
from .types import Pose,IKQuery
from .correction_reserve.native_geometry import NativeGeometry
from .correction_reserve.geometry import residual_linearization,task_scale

class SingleBoundedGN:
    method='single_bounded_gn'
    def __init__(self,kin,verifier,urdf,posture_weight=1.):
        self.kin,self.verifier=kin,verifier
        self.native=NativeGeometry(kin,urdf)
        self.scale=task_scale(verifier)
        self.engine=BoundedGN(kin.limits.lower,kin.limits.upper,kin.limits.velocity,
            verifier.config.velocity_tolerance,Settings(posture_weight=posture_weight))
    def reset(self,q):pass
    def close(self):pass
    def solve(self,position,rotation,previous,dt=.02):
        start=perf_counter_ns()
        query=IKQuery(Pose(np.asarray(position,float),np.asarray(rotation,float)),np.asarray(previous,float),dt)
        def evaluate(q):
            e,A,_=residual_linearization(self.native,query.target,q,self.scale)
            return e,A
        raw=self.engine.solve(query.previous_q,dt,evaluate,lambda q:self.verifier.check(q,query))
        verdict=raw.pop('verification');q=raw['q'];out=dict(raw)
        out.update(method=self.method,q=q.tolist(),finite=bool(verdict.finite_ok),
            joint_limit_ok=bool(verdict.joint_limit_ok),velocity_ok=bool(verdict.velocity_ok),
            position_error=float(verdict.position_error),orientation_error=float(verdict.orientation_error),
            verification_reasons=list(verdict.reasons),failure_kind='accepted' if verdict.accepted else '+'.join(verdict.reasons),
            internal_ok=raw['internal_status']=='task_candidate',
            velocity_utilization=float(np.max(np.abs(self.kin.difference(q,query.previous_q))/(self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance))))
        out['total_latency_ns']=perf_counter_ns()-start
        out['accepted_within_20ms']=bool(verdict.accepted and out['total_latency_ns']<=20_000_000)
        return out
