"""Official TRAC-IK Speed adapter with the unchanged public verifier."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from ..latency_pilot_v3.benchmark import ProfiledOutcome


def allowed_interval(kin, query, verifier):
    step = kin.limits.velocity * query.dt + verifier.config.velocity_tolerance
    lower = np.maximum(kin.limits.lower, query.previous_q - step)
    upper = np.minimum(kin.limits.upper, query.previous_q + step)
    continuous = getattr(kin, 'continuous_mask', np.zeros(kin.nq, bool))
    # Search an unwrapped local interval for continuous variables. Normalize the
    # result into the public representation before its common verifier check.
    lower[continuous] = query.previous_q[continuous] - step[continuous]
    upper[continuous] = query.previous_q[continuous] + step[continuous]
    return np.ascontiguousarray(lower), np.ascontiguousarray(upper)


class TracIK:
    def __init__(self, library, urdf, kin, verifier, budget_ms, epsilon=1e-5):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.kinematics, self.verifier, self.budget_ms = kin, verifier, budget_ms
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.revision_trac_create.argtypes = [ct.c_char_p]*3 + [ptr,ptr,ct.c_int,ct.c_double,ct.c_double]
        self.lib.revision_trac_create.restype = ct.c_void_p
        self.lib.revision_trac_names.argtypes = [ct.c_void_p]
        self.lib.revision_trac_names.restype = ct.c_char_p
        self.lib.revision_trac_error.restype = ct.c_char_p
        self.lib.revision_trac_solve.argtypes = [ct.c_void_p,ct.c_int]+[ptr]*6
        self.lib.revision_trac_fk.argtypes = [ct.c_void_p,ct.c_int,ptr,ptr]
        self.lib.revision_trac_destroy.argtypes = [ct.c_void_p]
        self.handle = self.lib.revision_trac_create(str(urdf).encode(),kin.base_link.encode(),kin.end_link.encode(),kin.limits.lower,kin.limits.upper,kin.nq,budget_ms/1000,epsilon)
        if not self.handle:
            raise RuntimeError(self.lib.revision_trac_error().decode())
        self.joint_names = tuple(self.lib.revision_trac_names(self.handle).decode().split('|'))
        if self.joint_names != kin.joint_names:
            raise RuntimeError('TRAC-IK joint ordering differs from public model')

    def forward(self, q):
        out = np.empty(12)
        rc = self.lib.revision_trac_fk(self.handle,self.kinematics.nq,np.ascontiguousarray(q),out)
        if rc < 0: raise RuntimeError('KDL FK failed')
        return out[:3], out[3:].reshape(3,3)

    def solve(self, query):
        start = perf_counter_ns()
        lower, upper = allowed_interval(self.kinematics, query, self.verifier)
        q = np.empty(self.kinematics.nq)
        rc = self.lib.revision_trac_solve(self.handle,self.kinematics.nq,np.ascontiguousarray(query.previous_q),
                np.ascontiguousarray(query.target.position),np.ascontiguousarray(query.target.rotation),lower,upper,q)
        if rc == -999:
            raise RuntimeError(self.lib.revision_trac_error().decode())
        continuous = getattr(self.kinematics,'continuous_mask',np.zeros(len(q),bool))
        q[continuous] = (q[continuous] + np.pi) % (2*np.pi) - np.pi
        check = self.verifier.check(q, query) if rc >= 0 else None
        accepted = bool(check and check.accepted)
        out = ProfiledOutcome(q.copy() if accepted else None,accepted,'trac_ik_speed',('trac_ik',),np.zeros(4),0.,
                              0,0,False,tuple(check.reasons) if check else (),
                              '' if accepted else 'trac_ik_no_solution' if rc < 0 else 'public_verifier_rejected',0,{})
        out.timings_ns['total_end_to_end_ns'] = perf_counter_ns()-start
        self.last_solver_return_code = rc
        return out

    def close(self):
        if self.handle:
            self.lib.revision_trac_destroy(self.handle)
            self.handle = None
