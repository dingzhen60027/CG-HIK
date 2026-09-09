"""One fixed-shape, data-updatable minimum-intervention SOCP, no gamma variable."""
from time import perf_counter, perf_counter_ns
import clarabel
import numpy as np
from scipy import sparse

from .geometry import residual_linearization, correction_map


class MinimalCone:
    """Lazily create Clarabel only when required; keep dimensions/CSC pattern.

    A has explicitly stored entries for its fixed block pattern, including zeros.
    Presolve and chordal decomposition are disabled as required by the upstream
    data-update API. No unsupported primal/dual warm-start claim is made.
    """
    def __init__(self, n, cfg):
        self.n, self.cfg, self.solver = n, cfg, None
        self.size, self.linear_rows = 2*n, 8*n
        self.rows = self.linear_rows + 16
        self.A = np.zeros((self.rows, self.size))
        self.b = np.zeros(self.rows)
        self.c = np.zeros(self.size)
        self.P = sparse.eye(self.size, format='csc') * 2
        eye = np.eye(n)
        for block, columns, signs in ((0, slice(0,n),1), (1,slice(0,n),-1),
            (2,slice(n,2*n),1),(3,slice(n,2*n),-1)):
            self.A[block*n:(block+1)*n,columns] = signs*eye
        self.A[4*n:5*n,:n] = -eye; self.A[4*n:5*n,n:] = eye
        self.A[5*n:6*n,:n] = eye; self.A[5*n:6*n,n:] = -eye
        self.A[6*n:7*n,n:] = eye; self.A[7*n:8*n,n:] = -eye
        pattern = self.A != 0
        for k in range(4):
            row = self.linear_rows + 4*k
            columns = slice(0,n) if k < 2 else slice(n,2*n)
            pattern[row+1:row+4,columns] = True
        self.sparse = sparse.csc_matrix(pattern.astype(float))
        self.row_ids = self.sparse.indices.copy()
        self.col_ids = np.repeat(np.arange(self.size), np.diff(self.sparse.indptr))
        self.cones = [clarabel.NonnegativeConeT(self.linear_rows)] + [clarabel.SecondOrderConeT(4) for _ in range(4)]
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = False
        self.settings.max_iter = cfg.native_max_iterations
        self.settings.max_threads = 1
        self.settings.presolve_enable = False
        self.settings.chordal_decomposition_enable = False
        self.settings.input_sparse_dropzeros = False
        self.settings.tol_feas = 1e-9
        self.settings.tol_gap_abs = self.settings.tol_gap_rel = 1e-8
        self.creations = self.updates = 0

    def solve(self, kin, target, predicted, previous, q0, z0, backup_q, initial_z,
              lower, upper, scale, step, demand, deadline):
        start = perf_counter_ns()
        n, cfg = self.n, self.cfg
        u0, v0 = (q0-previous)/step, (z0-previous)/step
        mapping = correction_map(kin,predicted,z0,scale,step,cfg.rank_rtol)
        if not mapping.full_rank:
            return None, dict(status='rank_deficient_no_six_dimensional_reserve',
                rank=mapping.rank, construction_ns=perf_counter_ns()-start,
                solve_ns=0, solver_called=False, data_updated=False)
        required = demand * np.linalg.norm(mapping.normalized_joint_map,axis=1)
        # Nonnegative RHS is not assumed; impossible demand remains impossible.
        self.b[:n] = np.minimum((upper-previous)/step,u0+cfg.trust_step)
        self.b[n:2*n] = -np.maximum((lower-previous)/step,u0-cfg.trust_step)
        zlower, zupper = np.nextafter(kin.limits.lower,np.inf), np.nextafter(kin.limits.upper,-np.inf)
        self.b[2*n:3*n] = np.minimum((zupper-previous)/step,v0+cfg.trust_step)
        self.b[3*n:4*n] = -np.maximum((zlower-previous)/step,v0-cfg.trust_step)
        self.b[4*n:6*n] = np.tile(1-required,2)
        self.b[6*n:7*n] = (zupper-previous)/step-required
        self.b[7*n:8*n] = (previous-zlower)/step-required
        for pose_index,(aim,node,center) in enumerate(((target,q0,u0),(predicted,z0,v0))):
            e,J,_ = residual_linearization(kin,aim,node,scale)
            F = J*step; off = e-F@center
            for part in range(2):
                row = self.linear_rows+4*(2*pose_index+part)
                columns = slice(pose_index*n,(pose_index+1)*n)
                self.A[row+1:row+4,columns] = -F[3*part:3*part+3]
                self.b[row] = 1-cfg.pose_interior
                self.b[row+1:row+4] = off[3*part:3*part+3]
        self.c[:n] = -2*(backup_q-previous)/step
        self.c[n:] = -2*(initial_z-previous)/step  # lambda=1, no residual or gamma cost
        self.sparse.data[:] = self.A[self.row_ids,self.col_ids]
        remaining = deadline-perf_counter()
        if remaining <= 0:
            return None, dict(status='outer_soft_limit', construction_ns=perf_counter_ns()-start,
                              solve_ns=0, solver_called=False, data_updated=False)
        self.settings.time_limit = max(1e-6,min(cfg.native_time_limit_ms/1000,remaining))
        updated = self.solver is not None
        if updated:
            if not self.solver.is_data_update_allowed():
                raise RuntimeError('frozen Clarabel structure unexpectedly disallows data update')
            self.solver.update(q=self.c,b=self.b,A=self.sparse.data,settings=self.settings)
            self.updates += 1
        else:
            self.solver = clarabel.DefaultSolver(self.P,self.c,self.sparse,self.b,self.cones,self.settings)
            self.creations += 1
        construction = perf_counter_ns()-start
        native_start = perf_counter_ns()
        result = self.solver.solve()
        elapsed = perf_counter_ns()-native_start
        x = np.asarray(result.x,float)
        status = dict(status=str(result.status), iterations=int(result.iterations),
            primal_residual=float(result.r_prim),dual_residual=float(result.r_dual),
            native_seconds=float(result.solve_time),construction_ns=construction,solve_ns=elapsed,
            solver_called=True,data_updated=updated,rank=mapping.rank,required_step_fraction=required.tolist())
        if str(result.status) not in ('Solved','AlmostSolved') or not np.isfinite(x).all():
            return None,status
        return x,status
