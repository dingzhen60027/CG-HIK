"""Bounded, accepted-history-aware adapter for the official RangedIK core.

The optional positive-range build differs from upstream at exactly two cutoff
conditions, enabling its *existing* ranged loss for positive sub-0.01 bounds.
Neither its soft pose objectives nor PANOC's native status prove admissibility.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.spatial.transform import Rotation

from ..continuation_mechanism.observation import representable_interior
from ..types import IKQuery, Pose

UPSTREAM_SHA = "1c48d2ae408b4e024ee037641aac1e728267984e"
UPSTREAM_URL = "https://github.com/uwgraphics/relaxed_ik_core"
RANGE_MODES = ("upstream", "positive_range")
NATIVE_STATUS = {
    0: "PANOC_converged", 1: "PANOC_max_iterations",
    2: "PANOC_time_limit", -1: "PANOC_error", -999: "bridge_error",
}


class _Stats(ct.Structure):
    _fields_ = [
        ("code", ct.c_int), ("iterations", ct.c_uint64),
        ("objective_calls", ct.c_uint64), ("gradient_calls", ct.c_uint64),
        ("setup_ns", ct.c_uint64), ("solve_ns", ct.c_uint64),
        ("cost", ct.c_double), ("fpr_norm", ct.c_double),
    ]


def range_mapping(verifier, mode="positive_range"):
    """Inscribed target-frame component ranges, not hard native constraints."""
    if mode not in RANGE_MODES:
        raise ValueError(f"range_mode must be one of {RANGE_MODES}")
    cfg = verifier.config
    b = np.r_[np.full(3, np.nextafter(cfg.position_tolerance / np.sqrt(3.0), 0.0)),
              np.full(3, np.nextafter(cfg.orientation_tolerance / np.sqrt(3.0), 0.0))]
    return {
        "bounds": b.tolist(), "frame": "target frame",
        "position": "R_target.T @ (p_actual - p_target)",
        "orientation": "Log(R_target.T @ R_actual), principal scaled-axis vector",
        "orientation_term": "absolute value of each scaled-axis component",
        "units": ["m"] * 3 + ["rad"] * 3,
        "formula": "nextafter(public_norm_tolerance / sqrt(3), 0)",
        "conservativeness": "the component box is inside the public norm ball, but native ranges are soft objectives",
        "native_range_mode": mode,
        "range_activation_cutoff": 1e-2 if mode == "upstream" else 0.0,
        "ranged_loss_active": (b > (1e-2 if mode == "upstream" else 0.0)).tolist(),
    }


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


class RangedAdapter:
    """One native RangedIK solve, then the unchanged independent task verifier.

    `source` is the existing robot configuration (retained for API symmetry), not
    a future trajectory. A build is never triggered by construction or solve.
    Use `native/ranged_build.py` explicitly. Default mode is transparently labeled
    positive-range adaptation. Pass `range_mode='upstream'` for original cutoffs.
    """

    def __init__(self, kin, verifier, source, urdf, robot=None, *,
                 range_mode="positive_range", library=None, time_limit_ms=18.0,
                 collision_objective=False):
        self.kin = kin
        self.kinematics = kin
        self.verifier = verifier
        self.source = source
        self.robot = robot or kin.name
        self.urdf = Path(urdf).resolve()
        self.range_mode = range_mode
        if not isinstance(collision_objective, bool):
            raise ValueError("collision_objective must be explicitly boolean")
        self.collision_objective = collision_objective
        self.mapping = range_mapping(verifier, range_mode)
        self.native_bounds = np.ascontiguousarray(self.mapping["bounds"], dtype=np.float64)
        self.time_limit_ms = time_limit_ms
        if time_limit_ms is not None and (not np.isfinite(time_limit_ms) or time_limit_ms <= 0):
            raise ValueError("native time limit must be positive or None")
        self.max_duration_ns = 0 if time_limit_ms is None else int(time_limit_ms * 1_000_000)
        self.root = Path(__file__).resolve().parents[3]
        manifest_path = self.root / "tmp/crik_dependencies/ranged_adapter_build/manifest.json"
        self.build_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if library is None:
            if not self.build_manifest:
                raise FileNotFoundError("RangedIK bridge is not built; run src/confik/correction_reserve/native/ranged_build.py")
            library = self.root / self.build_manifest["variants"][range_mode]["library"]
        self.library_path = Path(library).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        self.library_sha256 = hashlib.sha256(self.library_path.read_bytes()).hexdigest()
        if self.build_manifest is not None:
            variant = self.build_manifest["variants"][range_mode]
            if self.build_manifest["upstream_sha"] != UPSTREAM_SHA:
                raise ValueError("RangedIK source pin mismatch")
            if self.library_sha256 != variant["library_sha256"]:
                raise ValueError("RangedIK library does not match selected range mode manifest")
        if np.any(getattr(kin, "continuous_mask", np.zeros(kin.nq, dtype=bool))):
            raise ValueError("this adapter only covers the bounded revolute Panda/UR5e study chains")
        for joint in getattr(kin, "chain", ()):
            if joint.active and not any(np.array_equal(joint.axis, s * e)
                    for e in np.eye(3) for s in (-1, 1)):
                raise ValueError("official RangedIK parser requires coordinate-aligned URDF joint axes")
        self.lib = ct.CDLL(str(self.library_path))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.crik_ranged_new.argtypes = [ct.c_char_p] * 3 + [ct.c_size_t, ptr, ct.c_int]
        self.lib.crik_ranged_new.restype = ct.c_void_p
        self.lib.crik_ranged_free.argtypes = [ct.c_void_p]
        self.lib.crik_ranged_reset.argtypes = [ct.c_void_p, ct.c_size_t, ptr]
        self.lib.crik_ranged_error.restype = ct.c_char_p
        self.lib.crik_ranged_fk.argtypes = [ct.c_void_p, ct.c_size_t, ptr, ptr]
        self.lib.crik_ranged_components.argtypes = [ct.c_void_p, ct.c_size_t] + [ptr] * 5
        self.lib.crik_ranged_weights.argtypes = [ct.c_void_p, ct.c_size_t, ptr]
        self.lib.crik_ranged_solve.argtypes = [ct.c_void_p, ct.c_size_t] + [ptr] * 6 + [ct.c_uint64, ptr, ct.POINTER(_Stats)]
        initial = np.ascontiguousarray((kin.limits.lower + kin.limits.upper) / 2)
        self.handle = self.lib.crik_ranged_new(self.urdf.read_bytes(), kin.base_link.encode(),
            kin.end_link.encode(), kin.nq, initial, int(collision_objective))
        if not self.handle:
            raise RuntimeError(self.lib.crik_ranged_error().decode())
        self._history = None
        try:
            # Initialization-only model/order checks; no future targets or q_ref.
            self.fk_checks = []
            span = kin.limits.upper - kin.limits.lower
            for fraction in (0.0, 0.03125, -0.0625):
                test_q = np.ascontiguousarray(initial + fraction * span)
                native_pose = self.native_forward(test_q)
                reference_pose = kin.forward(test_q)
                pe = float(np.linalg.norm(native_pose.position - reference_pose.position))
                re = float(np.linalg.norm(Rotation.from_matrix(reference_pose.rotation.T @ native_pose.rotation).as_rotvec()))
                if pe > 1e-9 or re > 1e-9:
                    raise ValueError(f"official/common FK or joint ordering mismatch: {pe} m, {re} rad")
                self.fk_checks.append({"fraction": fraction, "position_error": pe, "orientation_error": re})
            capacity = 6 + kin.nq + 4 + (kin.nq - 1) * (kin.nq - 2) // 2
            weights = np.zeros(capacity, dtype=np.float64)
            count = self.lib.crik_ranged_weights(self.handle, capacity, weights)
            if count < 0:
                raise RuntimeError(self.lib.crik_ranged_error().decode())
            self.active_objective_weights = weights[:count].tolist()
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self):
        return {
            "method": "RangedIK (positive-range adapter)" if self.range_mode == "positive_range" else "RangedIK (official cutoff)",
            "upstream_url": UPSTREAM_URL, "upstream_branch": "ranged-ik", "upstream_sha": UPSTREAM_SHA,
            "license": "MIT", "range_mode": self.range_mode, "mapping": self.mapping,
            "library_sha256": self.library_sha256,
            "native_max_iterations": 100, "native_panoc_tolerance": 0.0005,
            "native_max_duration_ms": self.time_limit_ms,
            "native_cache": {"tolerance": 1e-14, "lbfgs_memory": 10},
            "native_weights": {"position_each": 50.0, "rotation_each": 10.0,
                "joint_limit_each": 0.1, "velocity": 0.7, "acceleration": 0.5,
                "jerk": 0.3, "manipulability": 1.0,
                "native_self_collision_each": 0.01 if self.collision_objective else None},
            "collision_objective": self.collision_objective,
            "active_objective_weights": self.active_objective_weights,
            "active_objective_count": len(self.active_objective_weights),
            "dynamic_bounds": "unchanged static URDF penalty; hard PANOC Rectangle is narrowed to representable single-frame interval",
            "history": "actual accepted or held command each frame, never the rejected raw candidate",
            "native_status": "PANOC convergence only; independent task acceptance may differ",
            "joint_names": list(self.kin.joint_names), "base_link": self.kin.base_link,
            "end_link": self.kin.end_link, "urdf_sha256": hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
            "public_kinematics_backend": f"{type(self.kin).__module__}.{type(self.kin).__name__}",
            "native_kinematics_backend": "official relaxed_ik_core::spacetime::Robot / Arm (nalgebra, URDF via k)",
            "fk_checks": self.fk_checks,
            "collision_scope": ("official self-collision tail explicitly enabled as a development configuration"
                if self.collision_objective else "out-of-contract official self-collision tail removed; all other objectives/weights unchanged"),
            "timing_scope": "input conversion through actual accepted/held history update; offline metric/list packaging excluded and reported separately",
        }

    def native_forward(self, q):
        out = np.zeros(7, dtype=np.float64)
        code = self.lib.crik_ranged_fk(self.handle, self.kin.nq,
            np.ascontiguousarray(q, dtype=np.float64), out)
        if code:
            raise RuntimeError(self.lib.crik_ranged_error().decode())
        return Pose(out[:3], Rotation.from_quat(out[3:]).as_matrix())

    def native_component_losses(self, q, position, rotation):
        """Non-solving implementation probe for the six native Cartesian terms."""
        out = np.zeros(6, dtype=np.float64)
        code = self.lib.crik_ranged_components(self.handle, self.kin.nq,
            np.ascontiguousarray(q, dtype=np.float64),
            np.ascontiguousarray(position, dtype=np.float64),
            np.ascontiguousarray(Rotation.from_matrix(rotation).as_quat()), self.native_bounds, out)
        if code:
            raise RuntimeError(self.lib.crik_ranged_error().decode())
        return out.tolist()

    def reset(self, q):
        q = np.ascontiguousarray(q, dtype=np.float64)
        if q.shape != (self.kin.nq,) or not np.isfinite(q).all():
            raise ValueError("reset requires one finite actual joint configuration")
        if np.any(q < self.kin.limits.lower) or np.any(q > self.kin.limits.upper):
            raise ValueError("reset is outside the original URDF bounds")
        code = self.lib.crik_ranged_reset(self.handle, self.kin.nq, q)
        if code:
            raise RuntimeError(self.lib.crik_ranged_error().decode())
        self._history = np.repeat(q[None, :], 4, axis=0)

    def solve(self, position, rotation, previous, dt=0.02, last_target=None):
        start = perf_counter_ns()
        previous = np.ascontiguousarray(previous, dtype=np.float64)
        position = np.ascontiguousarray(position, dtype=np.float64)
        rotation = np.ascontiguousarray(rotation, dtype=np.float64)
        if previous.shape != (self.kin.nq,) or not np.isfinite(previous).all():
            raise ValueError("previous must be the finite actual accepted configuration")
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("target must be a finite SE(3) pose")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0) or np.linalg.det(rotation) <= 0:
            raise ValueError("target rotation is not an SO(3) matrix")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        quaternion = np.ascontiguousarray(Rotation.from_matrix(rotation).as_quat())
        query = IKQuery(Pose(position, rotation), previous, float(dt))
        conversion_end = perf_counter_ns()
        resynced = self._history is None or not np.array_equal(previous, self._history[0])
        if resynced:
            self.reset(previous)
        history = np.ascontiguousarray(self._history)
        lower, upper = representable_interior(self.kin, query, self.verifier)
        raw = np.full(self.kin.nq, np.nan, dtype=np.float64)
        stats = _Stats()
        setup_end = perf_counter_ns()
        code = int(self.lib.crik_ranged_solve(self.handle, self.kin.nq, position, quaternion,
            self.native_bounds, history, lower, upper, self.max_duration_ns, raw, ct.byref(stats)))
        native_end = perf_counter_ns()
        check = self.verifier.check(raw, query)
        verification_end = perf_counter_ns()
        accepted = bool(check.accepted)
        commanded = raw if accepted else previous
        self._history = np.vstack((commanded.copy(), history[:3]))
        history_end = perf_counter_ns()
        total_latency_ns = history_end - start
        # These are offline diagnostic metrics and JSON-compatible packaging.
        # They are not required to produce, verify, or commit the joint command.
        allowed = self.kin.limits.velocity * dt + self.verifier.config.velocity_tolerance
        velocity = float(np.max(np.abs(self.kin.difference(raw, previous)) / allowed)) if check.finite_ok else None
        result = {
            "method": "ranged_ik_positive_range" if self.range_mode == "positive_range" else "ranged_ik_upstream",
            "q": raw.tolist() if check.finite_ok else None,
            "raw_returned_q": raw.tolist() if check.finite_ok else None,
            "submitted_q": raw.tolist() if accepted else None,
            "held_previous_on_failure": not accepted,
            "accepted": accepted, "internal_ok": code == 0,
            "native_status": NATIVE_STATUS.get(code, f"unknown_{code}"),
            "native_return_code": code, "native_iterations": int(stats.iterations),
            "native_objective_calls": int(stats.objective_calls),
            "native_gradient_calls": int(stats.gradient_calls),
            "solver_function_evaluations": None,
            "native_cost": _finite_or_none(stats.cost), "native_fixed_point_residual": _finite_or_none(stats.fpr_norm),
            "finite": bool(check.finite_ok), "position_error": _finite_or_none(check.position_error),
            "orientation_error": _finite_or_none(check.orientation_error),
            "joint_limit_ok": bool(check.joint_limit_ok), "velocity_ok": bool(check.velocity_ok),
            "verification_reasons": list(check.reasons), "velocity_utilization": velocity,
            "previous_q": previous.tolist(), "history_q": history.tolist(),
            "history_resynchronized": resynced, "native_lower": lower.tolist(),
            "native_upper": upper.tolist(), "native_bounds": self.native_bounds.tolist(),
            "range_mode": self.range_mode, "native_time_limit_ms": self.time_limit_ms,
            "collision_objective": self.collision_objective,
            "conversion_latency_ns": conversion_end - start,
            "bounds_setup_latency_ns": setup_end - conversion_end + int(stats.setup_ns),
            "solve_latency_ns": int(stats.solve_ns),
            "native_call_latency_ns": native_end - setup_end,
            "verification_latency_ns": verification_end - native_end,
            "history_update_latency_ns": history_end - verification_end,
            "total_latency_ns": total_latency_ns,
            "accepted_within_20ms": accepted and total_latency_ns <= 20_000_000,
        }
        if code < 0:
            result["native_error"] = self.lib.crik_ranged_error().decode()
        result["offline_packaging_latency_ns"] = perf_counter_ns() - history_end
        return result

    def close(self):
        if getattr(self, "handle", None):
            self.lib.crik_ranged_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
