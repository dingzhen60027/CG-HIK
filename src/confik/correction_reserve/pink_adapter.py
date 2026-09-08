"""B2: one official Pink differential-IK QP, with the original task verifier.

This is an adapter, not a reimplementation of Pink.  A FrameTask and weak
previous-state PostureTask are assembled by ``pink.build_ik``; OSQP solves the
one resulting displacement QP.  No future target, candidate pool, nonlinear
iteration, or fallback solver is used.  Its soft SE(3) task objective is not the
public separate position/orientation norm-ball contract.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from time import perf_counter_ns
import warnings

import numpy as np
from scipy.sparse import csc_matrix

from ..continuation_mechanism.observation import representable_interior
from ..types import IKQuery, Pose


PINK_SETTINGS = {
    "qp_solver": "osqp",
    "frame_gain": 1.0,
    "frame_lm_damping": 0.0,
    "posture_cost": 1e-3,
    "posture_gain": 1.0,
    "tikhonov_damping": 1e-12,
    "configuration_limit_gain": 1.0,
    "qp_eps_abs": 1e-9,
    "qp_eps_rel": 0.0,
    "qp_max_iter": 2000,
    "qp_time_limit_s": 0.018,
    "qp_polishing": True,
    "qp_adaptive_rho_interval": 25,
    "bound_roundoff_clip_max_rad": 1e-8,
    "differential_steps_per_query": 1,
}


class PinkAdapter:
    """Single-frame Pink baseline for the fixed scalar-joint Panda/UR5e URDFs.

    ``source`` is retained only to identify the source contract configuration;
    it does not silently override these predeclared baseline settings.  Every
    call starts from the supplied actual previous accepted configuration.
    ``last_target`` is accepted for the common runner API and is never read.
    """

    method = "pink_qp"

    def __init__(self, kin, verifier, source, urdf):
        import pinocchio as pin
        import pink
        import qpsolvers
        from pink.limits import ConfigurationLimit, VelocityLimit
        from pink.tasks import FrameTask, PostureTask

        self.kin, self.verifier = kin, verifier
        self.pin, self.pink, self.qpsolvers = pin, pink, qpsolvers
        self.urdf = str(Path(urdf).resolve())
        self.source = source.get("_config_path") if isinstance(source, dict) else str(source)
        self.model = pin.buildModelFromUrdf(self.urdf)
        if "osqp" not in qpsolvers.available_solvers:
            raise ImportError("the predeclared Pink baseline requires installed OSQP")
        if self.model.nq != kin.nq or self.model.nv != kin.nq:
            raise ValueError("Pink baseline requires the same fixed-base scalar-joint chain")
        if tuple(self.model.names[1:]) != tuple(kin.joint_names):
            raise ValueError("Pinocchio and public URDF joint ordering do not agree")
        if any(j.nq != 1 or j.nv != 1 for j in list(self.model.joints)[1:]):
            raise ValueError("non-scalar/continuous joint representations are not supported here")
        for value, expected in (
            (self.model.lowerPositionLimit, kin.limits.lower),
            (self.model.upperPositionLimit, kin.limits.upper),
            (self.model.velocityLimit, kin.limits.velocity),
        ):
            if not np.allclose(value, expected, rtol=0.0, atol=1e-12):
                raise ValueError("Pink and public URDF limits do not agree")
        for frame in (kin.base_link, kin.end_link):
            if self.model.getFrameId(frame) >= self.model.nframes:
                raise ValueError(f"missing Pinocchio frame: {frame}")
        self.data = self.model.createData()
        self.settings = dict(PINK_SETTINGS)
        self.frame_task = FrameTask(
            kin.end_link,
            position_cost=1.0 / verifier.config.position_tolerance,
            orientation_cost=1.0 / verifier.config.orientation_tolerance,
            gain=self.settings["frame_gain"],
            lm_damping=self.settings["frame_lm_damping"],
        )
        self.posture_task = PostureTask(
            cost=self.settings["posture_cost"], gain=self.settings["posture_gain"]
        )
        self.tasks = [self.frame_task, self.posture_task]
        self.limits = [
            ConfigurationLimit(self.model, config_limit_gain=1.0),
            VelocityLimit(self.model),
        ]
        self.model_agreement = self._validate_model()
        self.versions = {name: version(name) for name in ("pin-pink", "pin", "qpsolvers", "osqp")}
        self.previous_accepted_q = None

    def _configuration(self, q):
        return self.pink.Configuration(self.model, self.data, q, copy_data=False)

    def _validate_model(self):
        """Construction-time FK/Jacobian checks; not experimental solver calls."""
        center = (self.kin.limits.lower + self.kin.limits.upper) / 2.0
        span = self.kin.limits.upper - self.kin.limits.lower
        direction = np.where(np.arange(self.kin.nq) % 2, -1.0, 1.0)
        largest_p = largest_r = largest_j = 0.0
        base = None
        for fraction in (0.0, 0.07, -0.07, 0.19, -0.19):
            q = center + fraction * span * direction
            cfg = self._configuration(q)
            current_base = cfg.get_transform_frame_to_world(self.kin.base_link)
            if base is None:
                base = current_base.copy()
            elif not np.allclose(base.homogeneous, current_base.homogeneous, rtol=0.0, atol=1e-12):
                raise ValueError("the public base frame must be fixed")
            fk = current_base.actInv(cfg.get_transform_frame_to_world(self.kin.end_link))
            public = self.kin.forward(q)
            largest_p = max(largest_p, float(np.max(np.abs(fk.translation - public.position))))
            largest_r = max(largest_r, float(np.max(np.abs(fk.rotation - public.rotation))))
            jac = np.asarray(self.pin.getFrameJacobian(
                self.model, cfg.data, self.model.getFrameId(self.kin.end_link),
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )).copy()
            jac[:3] = current_base.rotation.T @ jac[:3]
            jac[3:] = current_base.rotation.T @ jac[3:]
            largest_j = max(largest_j, float(np.max(np.abs(jac - self.kin.jacobian(q)))))
        if max(largest_p, largest_r, largest_j) > 1e-11:
            raise ValueError("Pink Pinocchio geometry differs from the public kinematics")
        self.base_to_world = base
        return {
            "configurations": 5,
            "joint_order_equal": True,
            "fk_max_position_component_error_m": largest_p,
            "fk_max_rotation_component_error": largest_r,
            "jacobian_max_component_error": largest_j,
            "public_backend": type(self.kin).__name__,
            "solver_backend": "Pinocchio",
        }

    def metadata(self):
        return {
            "method": self.method,
            "versions": dict(self.versions),
            "settings": dict(self.settings),
            "urdf": self.urdf,
            "joint_order": list(self.kin.joint_names),
            "base_frame": self.kin.base_link,
            "end_frame": self.kin.end_link,
            "source_configuration": self.source,
            "model_agreement": dict(self.model_agreement),
            "task_error": "Pinocchio SE(3) body log; linear followed by angular components",
            "task_jacobian": "official Pink FrameTask Jlog6 derivative",
            "pose_cost": [1.0 / self.verifier.config.position_tolerance] * 3
                + [1.0 / self.verifier.config.orientation_tolerance] * 3,
            "posture_target": "actual previous accepted q, updated each query",
            "uses_future_targets": False,
            "public_verifier_modified": False,
        }

    def reset(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.kin.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("invalid initial q")
        self.previous_accepted_q = q.copy()

    def solve(self, position, rotation, previous, dt=0.02, last_target=None):
        start = perf_counter_ns()
        query = IKQuery(
            Pose(np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)),
            np.asarray(previous, dtype=float), float(dt),
        )
        if query.previous_q.shape != (self.kin.nq,) or not np.all(np.isfinite(query.previous_q)):
            raise ValueError("previous must be a finite scalar-joint configuration")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        conversion_ns = perf_counter_ns() - start
        setup_start = perf_counter_ns()
        lower, upper = representable_interior(self.kin, query, self.verifier)
        self.model.lowerPositionLimit = lower
        self.model.upperPositionLimit = upper
        # The original public allowance is velocity*dt + epsilon_v.  The
        # ConfigurationLimit additionally carries its one-ULP interior form.
        self.model.velocityLimit = (
            self.kin.limits.velocity + self.verifier.config.velocity_tolerance / dt
        )
        self.frame_task.set_target(self.base_to_world * self.pin.SE3(
            query.target.rotation, query.target.position
        ))
        self.posture_task.set_target(query.previous_q)
        bounds_setup_ns = perf_counter_ns() - setup_start
        build_start = perf_counter_ns()
        cfg = self._configuration(query.previous_q)
        problem = self.pink.build_ik(
            cfg, self.tasks, dt, damping=self.settings["tikhonov_damping"], limits=self.limits,
        )
        # OSQP requires CSC.  Conversion is measured, not deferred to logging.
        problem.P = csc_matrix(problem.P)
        problem.G = csc_matrix(problem.G)
        qp_build_ns = perf_counter_ns() - build_start
        solve_start = perf_counter_ns()
        raw_q = np.full(self.kin.nq, np.nan)
        q = raw_q.copy()
        internal_ok, native_status, native_code = False, "not_run", None
        iterations = None
        primal_residual = dual_residual = objective = None
        raw_bound_excess = clipping = None
        native_times = {}
        caught_warnings = []
        native_output = StringIO()
        with warnings.catch_warnings(record=True) as warning_records, redirect_stdout(native_output):
            warnings.simplefilter("always")
            try:
                solution = self.qpsolvers.solve_problem(
                    problem, solver="osqp", verbose=False,
                    eps_abs=self.settings["qp_eps_abs"], eps_rel=self.settings["qp_eps_rel"],
                    max_iter=self.settings["qp_max_iter"],
                    time_limit=self.settings["qp_time_limit_s"],
                    polishing=self.settings["qp_polishing"],
                    adaptive_rho_interval=self.settings["qp_adaptive_rho_interval"],
                    raise_error=False,
                )
                internal_ok = bool(solution.found)
                info = solution.extras.get("info")
                if info is not None:
                    native_status = str(info.status)
                    native_code = int(info.status_val)
                    iterations = int(info.iter)
                    primal_residual = float(info.prim_res)
                    dual_residual = float(info.dual_res)
                    objective = float(info.obj_val)
                    native_times = {
                        f"native_{label}_ns": int(round(getattr(info, field) * 1e9))
                        if getattr(info, field, None) is not None else None
                        for label, field in (("setup", "setup_time"), ("solve", "solve_time"),
                                             ("polish", "polish_time"), ("run", "run_time"))
                    }
                if solution.x is not None and np.all(np.isfinite(solution.x)):
                    raw_q = np.asarray(self.pin.integrate(self.model, query.previous_q, solution.x)).copy()
                    q = raw_q.copy()
                    raw_bound_excess = float(max(0.0, np.max(lower - q), np.max(q - upper)))
                    # Only correct tiny QP/float reconstruction overshoot, not
                    # a substantive infeasible QP result.  This is not a pose
                    # refinement; the unmodified raw q remains in the record.
                    if raw_bound_excess <= self.settings["bound_roundoff_clip_max_rad"]:
                        q = np.clip(q, lower, upper)
                    clipping = float(np.max(np.abs(q - raw_q)))
            except (ValueError, RuntimeError) as error:
                native_status = f"exception:{type(error).__name__}:{error}"
            caught_warnings = [str(w.message) for w in warning_records]
        solve_ns = perf_counter_ns() - solve_start
        verify_start = perf_counter_ns()
        check = self.verifier.check(q, query)
        raw_check = self.verifier.check(raw_q, query) if clipping is not None and clipping > 0 else check
        verification_ns = perf_counter_ns() - verify_start
        total_latency_ns = perf_counter_ns() - start
        finite = bool(check.finite_ok)
        phases = dict(conversion_ns=conversion_ns, bounds_setup_ns=bounds_setup_ns,
                      qp_build_ns=qp_build_ns, solve_ns=solve_ns, verification_ns=verification_ns)
        if check.accepted:
            self.previous_accepted_q = q.copy()
        allowed = self.kin.limits.velocity * dt + self.verifier.config.velocity_tolerance
        step = np.abs(self.kin.difference(q, query.previous_q)) if finite else None
        return {
            "method": self.method,
            "q": q.tolist() if finite else None,
            "raw_q": raw_q.tolist() if np.all(np.isfinite(raw_q)) else None,
            "accepted": bool(check.accepted),
            "internal_ok": internal_ok,
            "internal_status": native_status,
            "native_status": native_status,
            "native_return_code": native_code,
            "finite": finite,
            "joint_limit_ok": bool(check.joint_limit_ok),
            "velocity_ok": bool(check.velocity_ok),
            "position_error": float(check.position_error) if finite else None,
            "orientation_error": float(check.orientation_error) if finite else None,
            "verification_reasons": list(check.reasons),
            "failure_kind": "accepted" if check.accepted else "+".join(
                ([] if internal_ok else ["internal_nonconvergence"]) + list(check.reasons)
            ),
            "velocity_utilization": float(np.max(step / allowed)) if finite else None,
            "velocity_utilization_per_joint": (step / allowed).tolist() if finite else None,
            "joint_limit_margin": float(np.min(np.minimum(q - self.kin.limits.lower,
                self.kin.limits.upper - q))) if finite else None,
            "total_latency_ns": total_latency_ns,
            **phases,
            "phase_times_ns": phases,
            "accounting_remainder_ns": total_latency_ns - sum(phases.values()),
            "returned_within_5ms": total_latency_ns <= 5_000_000,
            "returned_within_20ms": total_latency_ns <= 20_000_000,
            "accepted_within_20ms": bool(check.accepted and total_latency_ns <= 20_000_000),
            "budget_ms": self.settings["qp_time_limit_s"] * 1000.0,
            "qp_time_limit_s": self.settings["qp_time_limit_s"],
            **native_times,
            "iterations": iterations,
            "qp_primal_residual": primal_residual if primal_residual is not None and np.isfinite(primal_residual) else None,
            "qp_dual_residual": dual_residual if dual_residual is not None and np.isfinite(dual_residual) else None,
            "qp_objective": objective if objective is not None and np.isfinite(objective) else None,
            "raw_bound_excess_rad": raw_bound_excess,
            "bound_roundoff_clip_rad": clipping,
            "bound_roundoff_clipped": bool(clipping is not None and clipping > 0),
            "raw_verifier_accepted": bool(raw_check.accepted),
            "raw_verification_reasons": list(raw_check.reasons),
            "raw_position_error": float(raw_check.position_error) if raw_check.finite_ok else None,
            "raw_orientation_error": float(raw_check.orientation_error) if raw_check.finite_ok else None,
            "native_warnings": caught_warnings,
            "native_stdout": native_output.getvalue(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "seed_q": query.previous_q.tolist(),
            "backup_used": False,
            "pinocchio_joint_jacobian_and_fk_calls": 1,
            "pink_frame_error_calls": 1,
            "pink_frame_error_jacobian_calls": 1,
            "public_verifier_calls": 1 + int(clipping is not None and clipping > 0),
            "solver_function_evaluations": None,
        }

    def close(self):
        """No retained native solver session: every query owns its one QP."""
        return None
