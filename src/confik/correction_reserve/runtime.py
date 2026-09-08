"""Two-step command optimization, with an independently verified TRAC backup."""
from dataclasses import asdict
from time import perf_counter, perf_counter_ns
import numpy as np

from ..types import IKQuery, Pose
from ..task_contract_alignment.outcomes import ContractSolver
from ..continuation_mechanism.observation import representable_interior
from .geometry import predict_target, task_scale, residual_linearization, reserve_value
from .convex import OptimizerConfig, subproblem, sigma_value
from .native_geometry import NativeGeometry


class CountedKinematics:
    def __init__(self, kin):
        self.kin = kin
        self.counts = dict(forward=0, jacobian=0)

    def __getattr__(self, key):
        return getattr(self.kin, key)

    def forward(self, q):
        self.counts["forward"] += 1
        return self.kin.forward(q)

    def jacobian(self, q):
        self.counts["jacobian"] += 1
        return self.kin.jacobian(q)


class CorrectionReserveIK:
    def __init__(self, kin, verifier, source, library, urdf, mode="reserve", config=None):
        self.kin, self.verifier = kin, verifier
        self.native_geometry = NativeGeometry(kin,urdf)
        self.counted = CountedKinematics(self.native_geometry)
        self.backup = ContractSolver("trac_task_5ms", kin, verifier, source, library, urdf)
        self.mode = mode
        self.config = config if isinstance(config, OptimizerConfig) else OptimizerConfig(**(config or {}))
        self.last_target = None
        self.scale = task_scale(verifier)

    def reset(self, q=None):
        self.last_target = None

    def close(self):
        self.backup.close()

    def secondary_cost(self, q, z, previous, target, step):
        from ..geometry import pose_error
        e = pose_error(target,self.counted.forward(q))/self.scale
        value = float(np.sum(((q-previous)/step)**2)+
                      self.config.current_residual_weight*np.sum(e**2))
        if self.mode != "single":
            value += float(np.sum(((z-q)/step)**2))
        return value

    def solve(self, position, rotation, previous, dt=.02, last_target=None):
        start = perf_counter_ns()
        if dt != .02:
            raise ValueError("CR-IK first implementation is fixed to the public dt=0.02 s")
        self.counted.counts = dict(forward=0, jacobian=0)
        target = Pose(position,rotation)
        previous = np.asarray(previous,dtype=float)
        query = IKQuery(target,previous,dt)
        last = self.last_target if last_target is None else last_target
        conversion_ns = perf_counter_ns()-start
        backup_start = perf_counter_ns()
        backup = self.backup.solve(position,rotation,previous,dt)
        backup_ns = perf_counter_ns()-backup_start
        prediction_start = perf_counter_ns()
        predicted = predict_target(target,last)
        # History tracks observed targets even after a rejected command.
        self.last_target = target
        lower, upper = representable_interior(self.kin,query,self.verifier)
        step = self.kin.limits.velocity*dt+self.verifier.config.velocity_tolerance
        q0 = np.asarray(backup["q"]).copy() if backup["accepted"] else previous.copy()
        z0 = q0.copy()
        if self.mode != "single":
            e, J, _ = residual_linearization(self.counted,predicted,q0,self.scale)
            dz = np.linalg.lstsq(J*step,-e,rcond=1e-12)[0]*step
            zl,zu = representable_interior(self.kin,IKQuery(predicted,q0,dt),self.verifier)
            z0 = np.clip(q0+dz,zl,zu)
        prediction_ns = perf_counter_ns()-prediction_start
        optimize_start = perf_counter_ns()
        deadline = start/1e9+self.config.total_soft_limit_ms/1000
        best = None
        baseline_gamma = None
        native_stages = []
        completed_updates = 0

        def consider(q,z,iteration):
            nonlocal best, baseline_gamma
            current_check = self.verifier.check(q,query)
            next_check = current_check if self.mode=="single" else self.verifier.check(z,IKQuery(predicted,q,dt))
            if not current_check.accepted or not next_check.accepted:
                return False
            reserve_q = previous if self.mode=="single" else q
            reserve_z = q if self.mode=="single" else z
            reserve_target = target if self.mode=="single" else predicted
            gamma, mapping, slack = reserve_value(self.counted,reserve_target,reserve_q,reserve_z,
                                                  self.scale,step,self.config.rank_rtol)
            if iteration == -1:
                baseline_gamma = gamma
            cost = self.secondary_cost(q,z,previous,target,step)
            score = -cost if self.mode=="predictive" else (
                sigma_value(self.counted,predicted,z,self.scale,step) if self.mode=="sigma" else gamma)
            if self.mode in ("reserve","single") and not mapping.full_rank:
                return False
            key = (score,-cost)
            if best is None or key > best["key"]:
                best = dict(q=q.copy(),z=z.copy(),gamma=gamma,score=score,cost=cost,key=key,
                            rank=mapping.rank,compensation_error=mapping.compensation_error,
                            sigma_min=float(mapping.singular_values[-1]),slack=slack.copy(),iteration=iteration)
            return True

        consider(q0,z0,-1)
        for iteration in range(self.config.outer_iterations):
            if perf_counter()>=deadline:
                native_stages.append(dict(status="outer_soft_limit"));break
            x, stages = subproblem(self.counted,self.verifier,target,predicted,previous,
                q0,z0,lower,upper,self.mode,self.config,deadline)
            native_stages.extend(dict(update=iteration,**s) for s in stages)
            if x is None:
                break
            q1 = previous+step*x[:self.kin.nq]
            z1 = q1.copy() if self.mode=="single" else previous+step*x[self.kin.nq:2*self.kin.nq]
            # Trial iterates may fail nonlinear pose checks, but are never accepted
            # without the original verifier. Re-linearization does not execute them.
            for alpha in (1.,.5,.25):
                qtrial = q0+alpha*(q1-q0)
                ztrial = z0+alpha*(z1-z0)
                if consider(qtrial,ztrial,iteration):
                    break  # Ordinary first-feasible backtracking, not a candidate pool.
                if perf_counter()>=deadline:
                    break
            q0,z0 = q1,z1
            completed_updates += 1
        optimize_ns = perf_counter_ns()-optimize_start
        selected = best is not None and best["iteration"]>=0
        raw = best["q"] if selected else (
            np.asarray(backup["q"]) if backup["q"] is not None else np.full(self.kin.nq,np.nan))
        verification_start = perf_counter_ns()
        check = self.verifier.check(raw,query)
        verification_ns = perf_counter_ns()-verification_start
        elapsed = perf_counter_ns()-start
        # Everything required for prediction, optimization, ranking and verification
        # above is in elapsed. Only offline record packaging below is excluded.
        out = dict(method={"reserve":"cr_ik","predictive":"two_step_predictive",
                           "single":"single_step_reserve","sigma":"two_step_sigma"}[self.mode],
            q=raw.tolist() if check.finite_ok else None,
            accepted=bool(check.accepted),finite=bool(check.finite_ok),
            internal_ok=bool(selected or backup["internal_ok"]),
            internal_status="verified_optimized_pair" if selected else "trac_backup",
            native_status=native_stages,native_return_code=backup.get("native_return_code"),
            verification_reasons=list(check.reasons),
            position_error=float(check.position_error) if check.finite_ok else None,
            orientation_error=float(check.orientation_error) if check.finite_ok else None,
            joint_limit_ok=bool(check.joint_limit_ok),velocity_ok=bool(check.velocity_ok),
            total_latency_ns=elapsed,conversion_ns=conversion_ns,backup_ns=backup_ns,
            prediction_ns=prediction_ns,optimization_ns=optimize_ns,verification_ns=verification_ns,
            returned_within_5ms=elapsed<=5_000_000,returned_within_20ms=elapsed<=20_000_000,
            accepted_within_20ms=bool(check.accepted and elapsed<=20_000_000),
            backup_used=not selected,backup_accepted=backup["accepted"],backup_q=backup["q"],
            backup_internal_ok=backup["internal_ok"],backup_total_latency_ns=backup["total_latency_ns"],
            predicted_position=predicted.position.tolist(),predicted_rotation=predicted.rotation.tolist(),
            nominal_next_q=best["z"].tolist() if selected else None,
            predicted_gamma=best["gamma"] if selected else None,initial_pair_gamma=baseline_gamma,
            actual_selected_score=best["score"] if selected else None,
            correction_rank=best["rank"] if selected else None,
            compensation_error=best["compensation_error"] if selected else None,
            nominal_next_verified=bool(selected),linearization_updates=completed_updates,
            public_python_kinematics_calls=dict(self.counted.counts),
            counts_scope="optimization API calls to cached Pinocchio geometry; excludes hidden native TRAC evaluations and unchanged URDF final-verifier FK",
            optimization_geometry="Pinocchio analytic FK/Jacobian, same URDF and checked against public backend",
            geometry_agreement_error=self.native_geometry.agreement_error,
            failure_kind="accepted" if check.accepted else "+".join(check.reasons),
            max_difference_from_backup=float(np.max(np.abs(raw-np.asarray(backup["q"])))) if check.finite_ok and backup["q"] is not None else None)
        out["velocity_utilization"]=float(np.max(np.abs(self.kin.difference(raw,previous))/step)) if check.finite_ok else None
        return out
