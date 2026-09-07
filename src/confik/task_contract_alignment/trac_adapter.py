"""Official Speed solver; tolerances/budgets are protocol constants."""
import ctypes as ct
import numpy as np
from ..revision_compute_allocation.trac_ik import TracIK
from .contract import native_mapping, STRICT


class NativeTrac(TracIK):
    def __init__(self, method, library, urdf, kin, verifier):
        super().__init__(library, urdf, kin, verifier, 20 if '20ms' in method else 5, STRICT)
        self.mapping = native_mapping(method, verifier)
        self.bounds = np.ascontiguousarray(self.mapping['bounds'])
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        iptr = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.lib.contract_trac_solve.argtypes = [ct.c_void_p, ct.c_int]+[ptr]*7+[iptr]

    def run(self, query, seed, lower, upper):
        raw = np.full(self.kinematics.nq, np.nan)
        times = np.zeros(3, dtype=np.int64)
        rc = int(self.lib.contract_trac_solve(self.handle,self.kinematics.nq,seed,
            query.target.position,query.target.rotation,lower,upper,self.bounds,raw,times))
        if rc == -999:
            raise RuntimeError(self.lib.revision_trac_error().decode())
        return raw, dict(internal_ok=rc>=0, internal_status='native_success' if rc>=0 else 'native_no_solution',
            native_return_code=rc, iterations=None, solver_function_evaluations=None,
            native_conversion_ns=int(times[0]), native_setup_ns=int(times[1]),
            native_solve_ns=int(times[2]), native_bounds=self.bounds.tolist())
