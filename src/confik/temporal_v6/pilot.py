"""Development-only trajectory calibration and policy-validation for Temporal V6.

The pilot creates a new, outcome-blind trajectory pool, splits it only at the
complete-trajectory level, selects one temporal policy on calibration, seals
that policy, and opens policy-validation exactly once.  There is intentionally
no formal-test loader in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..counterfactual_v4.runner import _wait_for_quiet_environment
from ..data.datasets import QueryDataset
from ..data.generate_v2 import generate_reference_trajectory_tests
from ..experiments.provenance import environment_payload
from ..hierarchical_v5.pilot import (
    _artifact,
    _sha256_file,
    _verified_release_inputs,
    _write_json,
    latin_method_orders,
)
from ..hierarchical_v5_lite.features import lite_feature_dim, prepare_lite_features
from ..hierarchical_v5_lite.model import (
    TorchScriptLiteGateInference,
    load_exact_torchscript,
)
from ..hierarchical_v5_lite.pilot import (
    _build_current_v4,
    _fresh_fixed_hard,
    _fresh_shared_fixed_hard,
    _snapshot_digest,
    _tree_snapshot,
)
from ..hierarchical_v5_lite.policy import LiteFastGatePolicy, load_policy
from ..hierarchical_v5_lite.runtime import HierarchicalLiteRuntime
from ..latency_pilot_v3.benchmark import ProfiledOutcome, query_digest
from ..types import IKQuery, Pose
from .policy import TemporalPolicyConfig
from .runtime import TemporalCGHIKRuntime, TemporalOutcome
from .state import TemporalMode, TemporalState


PROTOCOL = "temporal_v6_development_pilot_v1"
ROLES = ("trajectory_calibration", "trajectory_policy_validation")
FAMILIES = (
    "trajectory_smooth",
    "trajectory_orientation",
    "trajectory_singular",
    "trajectory_limit",
)
METHODS = (
    "always_hard",
    "counterfactual_cghik_v4",
    "hierarchical_cghik_v5_lite",
    "temporal_cghik_v6",
)
STAGES = ("temporal", "feature", "gate", "local", "robust", "unattributed")
PARAMETER_NAMES = (
    "hold_frames",
    "probe_interval",
    "consecutive_successes",
    "reentry_threshold",
)
MODE_CODE = {
    TemporalMode.INIT: 0,
    TemporalMode.LOCAL: 1,
    TemporalMode.ROBUST: 2,
}


class DevelopmentTrajectoryGuard:
    """Lightweight, honestly scoped trajectory-start environment check.

    The development pilot uses randomized/Latin interleaving for latency
    balance.  It checks for a currently busy host before each complete path,
    but does not claim the continuous background coverage required by a
    confirmatory formal test.
    """

    max_contaminated_attempts = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.events: list[dict[str, Any]] = []

    def wait_until_quiet(self, *, context: str) -> dict[str, Any]:
        event = _wait_for_quiet_environment(self.config, context=context)
        event = {**event, "monitor_sample_index": 0}
        self.events.append(event)
        return event

    @staticmethod
    def observe(*, context: str, since_sample_index: int) -> dict[str, Any]:
        del since_sample_index
        return {
            "context": context,
            "busy": False,
            "monitoring_scope": "trajectory_start_preflight_only",
            "busy_sample_since_query_start": None,
        }

    @staticmethod
    def record_contamination(**_: Any) -> None:  # pragma: no cover - no post monitor
        raise RuntimeError("preflight-only guard cannot report post-hoc contamination")

    def total_summary(self) -> dict[str, Any]:
        return {
            "monitoring_scope": "trajectory_start_preflight_only",
            "background_monitor": False,
            "trajectory_start_check_count": len(self.events),
            "wait_event_count": int(
                sum(bool(event.get("had_busy_process")) for event in self.events)
            ),
            "wait_seconds": float(
                sum(float(event.get("wait_seconds", 0.0)) for event in self.events)
            ),
            "events": self.events,
        }

    @staticmethod
    def close() -> None:
        return None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _forbid_formal_test(value: str | Path, *, name: str) -> None:
    if "test" in str(value).lower():
        raise ValueError(f"{name} must not name formal-test data: {value}")


def _expected_grid(config: Mapping[str, Any]) -> tuple[TemporalPolicyConfig, ...]:
    grid = config["temporal_grid"]
    exact_axes = {
        "hold_frames": (5, 10, 20, 30),
        "probe_interval": (1, 3, 5),
        "consecutive_successes": (1, 2, 3),
        "reentry_threshold": (0.80, 0.90, 0.95, 0.99),
    }
    for key, expected in exact_axes.items():
        observed = tuple(grid.get(key, ()))
        if observed != expected:
            raise ValueError(f"temporal grid axis {key} changed: {observed!r}")
    if int(grid.get("expected_candidate_count", -1)) != 144:
        raise ValueError("expected_candidate_count must remain 144")
    result = tuple(
        TemporalPolicyConfig(int(h), int(m), int(k), float(tau))
        for h, m, k, tau in product(
            grid["hold_frames"],
            grid["probe_interval"],
            grid["consecutive_successes"],
            grid["reentry_threshold"],
        )
    )
    if len(result) != int(grid["expected_candidate_count"]):
        raise ValueError("temporal grid does not contain the preregistered 144 candidates")
    if len(set(result)) != len(result):
        raise ValueError("temporal grid contains duplicate candidates")
    return result


def validate_config(config: Mapping[str, Any], *, workspace: Path) -> None:
    """Fail closed on data roles, method registry, grid, and selection order."""

    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected Temporal V6 protocol")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("Temporal V6 requires Panda and UR5e")
    if int(config.get("training_seed", -1)) != 17:
        raise ValueError("the frozen deployment seed must remain 17")
    roles = config.get("roles", {})
    if (roles.get("calibration"), roles.get("policy_validation")) != ROLES:
        raise ValueError(f"trajectory roles must be exactly {ROLES}")
    boundary = config.get("data_boundary", {})
    if tuple(boundary.get("allowed_roles", ())) != ROLES:
        raise ValueError("development role allowlist changed")
    required_boundary = (
        "formal_test_data_forbidden",
        "reject_test_named_paths",
        "split_before_outcome_collection",
        "compute_policy_validation_outcomes_only_after_calibration_seal",
    )
    if not all(boundary.get(key) is True for key in required_boundary):
        raise ValueError("development data boundary is incomplete")
    data = config.get("trajectory_data", {})
    if data.get("generator") != "reference_trajectory_tests":
        raise ValueError("trajectory generator changed")
    if tuple(data.get("families", ())) != FAMILIES:
        raise ValueError("the four trajectory families changed")
    if (
        int(data.get("paths_per_family_pool", -1)) != 20
        or int(data.get("paths_per_family_per_role", -1)) != 10
        or int(data.get("steps_per_trajectory", -1)) != 150
        or float(data.get("dt", np.nan)) != 0.02
        or data.get("split_unit") != "complete_trajectory"
        or data.get("preserve_time_order") is not True
    ):
        raise ValueError("40+40 by 150 trajectory contract changed")
    _expected_grid(config)
    selection = config.get("calibration_selection", {})
    if (
        selection.get("required_role") != "trajectory_calibration"
        or selection.get("eligibility")
        != "trajectory_completion_vector_exactly_equal_always_hard"
        or tuple(selection.get("objective_order", ()))
        != (
            "minimum_p95_end_to_end_latency",
            "minimum_learned_seed_ensemble_invocation_rate",
            "minimum_p50_end_to_end_latency",
        )
        or tuple(selection.get("conservative_exact_tie_break", ()))
        != (
            "higher_reentry_threshold",
            "higher_consecutive_successes",
            "higher_hold_frames",
            "higher_probe_interval",
        )
        or selection.get("policy_validation_used_for_selection") is not False
        or selection.get("no_eligible_candidate_action")
        != "stop_before_policy_validation"
    ):
        raise ValueError("calibration selection contract changed")
    if tuple(config.get("strategies", ())) != METHODS:
        raise ValueError(f"method registry must be exactly {METHODS}")
    timing = config.get("timing", {})
    if (
        timing.get("clock") != "perf_counter_ns"
        or int(timing.get("trajectory_repeats", -1)) != 1
        or timing.get("method_order") != "seeded_four_by_four_latin_blocks"
        or tuple(timing.get("stage_names", ())) != STAGES
        or timing.get("disk_io_inside_timed_interval") is not False
        or timing.get("logging_serialization_inside_timed_interval") is not False
    ):
        raise ValueError("stateful timing contract changed")
    if config.get("runtime", {}).get("trajectory_environment_check") != "preflight_only":
        raise ValueError("trajectory environment check contract changed")
    goals = config.get("pilot_goals", {})
    if (
        goals.get("trajectory_completion_vector_equal_always_hard") is not True
        or float(goals.get("p95_ratio_vs_always_hard_max", np.nan)) != 1.0
        or float(
            goals.get("p50_ratio_vs_counterfactual_cghik_v4_max_exclusive", np.nan)
        )
        != 1.0
        or float(goals.get("learned_seed_invocation_rate_max", np.nan)) != 0.70
    ):
        raise ValueError("pilot goals changed")
    for key in (
        "source_config",
        "release_v3_root",
        "release_v4_root",
        "frozen_v5_lite_root",
        "output_root",
    ):
        _forbid_formal_test(config.get(key, ""), name=key)
    expected_output = (workspace / "outputs" / "temporal_v6_pilot").resolve()
    if resolve_path(dict(config), str(config["output_root"])) != expected_output:
        raise ValueError(f"output_root must resolve to {expected_output}")


def _subset(dataset: QueryDataset, indices: Sequence[int]) -> QueryDataset:
    selected = np.asarray(indices, dtype=np.int64)
    return QueryDataset(
        previous_q=dataset.previous_q[selected],
        target_position=dataset.target_position[selected],
        target_rotation=dataset.target_rotation[selected],
        reference_q=dataset.reference_q[selected],
        category=dataset.category[selected],
        expected_reachable=dataset.expected_reachable[selected],
        continuity_feasible=dataset.continuity_feasible[selected],
        trajectory_id=dataset.trajectory_id[selected],
        time_index=dataset.time_index[selected],
    )


def _trajectory_uid(robot: str, dataset: QueryDataset, indices: np.ndarray) -> str:
    ordered = indices[np.argsort(dataset.time_index[indices], kind="stable")]
    digest = sha256()
    digest.update(str(robot).encode("utf-8"))
    digest.update(str(dataset.category[ordered[0]]).encode("utf-8"))
    for values in (
        dataset.previous_q[ordered],
        dataset.target_position[ordered],
        dataset.target_rotation[ordered],
        dataset.reference_q[ordered],
        dataset.time_index[ordered],
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrajectoryRole:
    robot: str
    role: str
    dataset: QueryDataset
    trajectory_uid: np.ndarray
    trajectory_order: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.dataset)

    def groups(self) -> list[tuple[str, np.ndarray]]:
        return [
            (uid, np.flatnonzero(self.trajectory_uid == uid).astype(np.int64))
            for uid in self.trajectory_order
        ]

    def query(self, index: int, *, previous_q: np.ndarray | None = None, dt: float) -> IKQuery:
        return IKQuery(
            Pose(
                self.dataset.target_position[index],
                self.dataset.target_rotation[index],
            ),
            self.dataset.previous_q[index] if previous_q is None else previous_q,
            dt=dt,
        )


def _role_from_ids(
    robot: str,
    role: str,
    pool: QueryDataset,
    selected_ids: Sequence[int],
    uid_by_id: Mapping[int, str],
) -> TrajectoryRole:
    indices: list[int] = []
    order: list[str] = []
    for trajectory_id in selected_ids:
        rows = np.flatnonzero(pool.trajectory_id == int(trajectory_id)).astype(np.int64)
        rows = rows[np.argsort(pool.time_index[rows], kind="stable")]
        indices.extend(rows.tolist())
        order.append(uid_by_id[int(trajectory_id)])
    dataset = _subset(pool, indices)
    uid = np.asarray(
        [uid_by_id[int(value)] for value in dataset.trajectory_id], dtype="U64"
    )
    return TrajectoryRole(robot, role, dataset, uid, tuple(order))


def generate_development_roles(
    kinematics: object,
    *,
    robot: str,
    paths_per_family_pool: int,
    paths_per_family_per_role: int,
    steps: int,
    pool_seed: int,
    split_seed: int,
    dt: float,
) -> tuple[TrajectoryRole, TrajectoryRole, dict[str, Any]]:
    """Generate one outcome-blind pool and split each family by trajectory."""

    pool = generate_reference_trajectory_tests(
        kinematics,  # type: ignore[arg-type]
        paths_per_type=paths_per_family_pool,
        steps=steps,
        seed=pool_seed,
        dt=dt,
    )
    uid_by_id: dict[int, str] = {}
    family_by_id: dict[int, str] = {}
    split_ids: dict[str, list[int]] = {role: [] for role in ROLES}
    family_assignment: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family_index, family in enumerate(FAMILIES):
        family_rows = np.flatnonzero(pool.category == family)
        ids = np.unique(pool.trajectory_id[family_rows]).astype(np.int64)
        if len(ids) != paths_per_family_pool:
            raise RuntimeError(f"{robot}/{family} pool has {len(ids)} trajectories")
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, family_index]))
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        calibration_ids = shuffled[:paths_per_family_per_role]
        policy_ids = shuffled[paths_per_family_per_role : 2 * paths_per_family_per_role]
        if len(policy_ids) != paths_per_family_per_role:
            raise RuntimeError("trajectory pool is too small for a complete 10+10 split")
        family_assignment[family] = {}
        for role, values in zip(ROLES, (calibration_ids, policy_ids), strict=True):
            split_ids[role].extend(int(value) for value in values)
            family_assignment[family][role] = []
            for value in values:
                rows = np.flatnonzero(pool.trajectory_id == int(value)).astype(np.int64)
                if len(rows) != steps or not np.array_equal(
                    np.sort(pool.time_index[rows]), np.arange(steps)
                ):
                    raise RuntimeError("trajectory is incomplete or time indices are invalid")
                uid = uid_by_id.setdefault(
                    int(value), _trajectory_uid(robot, pool, rows)
                )
                family_by_id[int(value)] = family
                family_assignment[family][role].append(
                    {"source_trajectory_id": int(value), "trajectory_uid": uid}
                )
    # Interleave families in the execution order while preserving every path.
    for role_index, role in enumerate(ROLES):
        per_family = {
            family: [
                int(row["source_trajectory_id"])
                for row in family_assignment[family][role]
            ]
            for family in FAMILIES
        }
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, 90 + role_index]))
        for values in per_family.values():
            rng.shuffle(values)
        ordered: list[int] = []
        for rank in range(paths_per_family_per_role):
            family_order = list(FAMILIES)
            rng.shuffle(family_order)
            ordered.extend(per_family[family][rank] for family in family_order)
        split_ids[role] = ordered
    calibration = _role_from_ids(
        robot, ROLES[0], pool, split_ids[ROLES[0]], uid_by_id
    )
    policy = _role_from_ids(robot, ROLES[1], pool, split_ids[ROLES[1]], uid_by_id)
    overlap = set(calibration.trajectory_order) & set(policy.trajectory_order)
    if overlap:
        raise RuntimeError(f"complete-trajectory split overlap: {sorted(overlap)}")
    manifest = {
        "robot": robot,
        "generator": "generate_reference_trajectory_tests",
        "pool_seed": int(pool_seed),
        "split_seed": int(split_seed),
        "steps_per_trajectory": int(steps),
        "paths_per_family_pool": int(paths_per_family_pool),
        "paths_per_family_per_role": int(paths_per_family_per_role),
        "pool_trajectory_count": int(len(np.unique(pool.trajectory_id))),
        "pool_frame_count": int(len(pool)),
        "roles": {
            role.role: _role_audit(role) for role in (calibration, policy)
        },
        "family_assignment": family_assignment,
        "trajectory_uid_overlap_count": 0,
        "policy_validation_outcomes_computed_during_split": False,
    }
    return calibration, policy, manifest


def _role_audit(role: TrajectoryRole) -> dict[str, Any]:
    by_family: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        rows = role.dataset.category == family
        by_family[family] = {
            "trajectory_count": int(len(np.unique(role.trajectory_uid[rows]))),
            "frame_count": int(np.sum(rows)),
        }
    ordered = list(role.trajectory_order)
    return {
        "trajectory_count": len(ordered),
        "unique_trajectory_count": len(set(ordered)),
        "frame_count": int(role.count),
        "ordered_trajectory_digest": sha256(
            json.dumps(ordered, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "trajectory_set_digest": sha256(
            "\n".join(sorted(set(ordered))).encode("utf-8")
        ).hexdigest(),
        "by_family": by_family,
    }


def _save_role(path: Path, role: TrajectoryRole) -> None:
    np.savez_compressed(
        path,
        previous_q=role.dataset.previous_q,
        target_position=role.dataset.target_position,
        target_rotation=role.dataset.target_rotation,
        reference_q=role.dataset.reference_q,
        category=role.dataset.category,
        expected_reachable=role.dataset.expected_reachable,
        continuity_feasible=role.dataset.continuity_feasible,
        trajectory_id=role.dataset.trajectory_id,
        time_index=role.dataset.time_index,
        trajectory_uid=role.trajectory_uid,
        trajectory_order=np.asarray(role.trajectory_order, dtype="U64"),
        role=np.asarray([role.role], dtype="U32"),
    )


def _load_role(path: Path, *, robot: str, expected_role: str) -> TrajectoryRole:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"sealed trajectory role is unavailable: {path}")
    with np.load(path, allow_pickle=False) as payload:
        stored_role = str(np.asarray(payload["role"]).reshape(-1)[0])
        if stored_role != expected_role:
            raise RuntimeError(
                f"trajectory role changed: expected={expected_role}, got={stored_role}"
            )
        dataset = QueryDataset(
            previous_q=payload["previous_q"].copy(),
            target_position=payload["target_position"].copy(),
            target_rotation=payload["target_rotation"].copy(),
            reference_q=payload["reference_q"].copy(),
            category=payload["category"].copy(),
            expected_reachable=payload["expected_reachable"].copy(),
            continuity_feasible=payload["continuity_feasible"].copy(),
            trajectory_id=payload["trajectory_id"].copy(),
            time_index=payload["time_index"].copy(),
        )
        uid = payload["trajectory_uid"].astype("U64", copy=True)
        order = tuple(payload["trajectory_order"].astype(str).tolist())
    loaded = TrajectoryRole(robot, expected_role, dataset, uid, order)
    if loaded.count == 0 or len(order) != len(set(order)):
        raise RuntimeError("sealed trajectory role is empty or contains duplicate paths")
    return loaded


def _verify_frozen_lite(root: Path, robot: str) -> tuple[Path, Path]:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_policy_validation_pilot"
        or bool(manifest.get("test_data_loaded", True))
        or bool(manifest.get("formal_test_started", True))
        or bool(manifest.get("policy_validation_used_for_retuning", True))
    ):
        raise RuntimeError("frozen V5-Lite output is not an eligible development artifact")
    model = root / f"{robot}_exact_lite_gate.ts"
    policy = root / f"{robot}_lite_gate_policy.json"
    artifacts = manifest.get("artifacts", {})
    for path in (model, policy):
        descriptor = artifacts.get(path.name)
        if not isinstance(descriptor, Mapping):
            raise RuntimeError(f"frozen V5-Lite manifest omits {path.name}")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(descriptor.get("size", -1))
            or _sha256_file(path) != str(descriptor.get("sha256", ""))
        ):
            raise RuntimeError(f"frozen V5-Lite artifact changed: {path}")
    workspace = root.resolve().parents[1]
    sources = manifest.get("implementation_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise RuntimeError("frozen V5-Lite manifest omits implementation sources")
    for relative, descriptor in sources.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("unsafe V5-Lite implementation path")
        source = workspace / relative_path
        if not isinstance(descriptor, Mapping) or (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != int(descriptor.get("size", -1))
            or _sha256_file(source) != str(descriptor.get("sha256", ""))
        ):
            raise RuntimeError(f"frozen V5-Lite source changed: {source}")
    return model, policy


def _descriptor_matches(root: Path, descriptor: Mapping[str, Any]) -> bool:
    relative = Path(str(descriptor.get("path", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        return False
    path = root / relative
    return bool(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(descriptor.get("size", -1))
        and _sha256_file(path) == str(descriptor.get("sha256", ""))
    )


def _sealed_lite_inputs(root: Path, robot: str, *, workspace: Path) -> dict[str, Any]:
    model, policy = _verify_frozen_lite(root, robot)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest": _artifact(manifest_path, relative_to=workspace),
        "model": _artifact(model, relative_to=workspace),
        "policy": _artifact(policy, relative_to=workspace),
        "implementation_sources": dict(manifest["implementation_sources"]),
    }


def _verify_calibration_seal(
    seal_path: Path,
    *,
    expected_sha256: str,
    staging: Path,
    workspace: Path,
) -> dict[str, Any]:
    if _sha256_file(seal_path) != expected_sha256:
        raise RuntimeError("calibration seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name in ("split_manifest", "calibration_selection", "candidate_metrics"):
        if not _descriptor_matches(staging, seal["sealed_reports"][name]):
            raise RuntimeError(f"sealed report changed: {name}")
    selection = json.loads(
        (staging / seal["sealed_reports"]["calibration_selection"]["path"])
        .read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (staging / seal["sealed_reports"]["candidate_metrics"]["path"])
        .read_text(encoding="utf-8")
    )
    for robot, descriptors in seal["sealed_development_artifacts"].items():
        for name in (
            "trajectory_calibration",
            "trajectory_policy_validation",
            "calibration_records",
        ):
            if not _descriptor_matches(staging, descriptors[name]):
                raise RuntimeError(f"sealed {robot}/{name} artifact changed")
        lite = descriptors["frozen_v5_lite"]
        for name in ("manifest", "model", "policy"):
            if not _descriptor_matches(workspace, lite[name]):
                raise RuntimeError(f"sealed {robot}/V5-Lite {name} changed")
        for relative, descriptor in lite["implementation_sources"].items():
            if str(descriptor.get("path", "")) != str(relative):
                raise RuntimeError(f"sealed V5-Lite source key/path differs: {relative}")
            if not _descriptor_matches(workspace, descriptor):
                raise RuntimeError(f"sealed V5-Lite source changed: {relative}")
        selected = dict(seal["selected_policy"][robot])
        if selected != dict(selection["selected"][robot]):
            raise RuntimeError(f"sealed {robot} policy differs from selection report")
        report_selected = metrics[robot]["selected"]
        keys = ("candidate_index",) + PARAMETER_NAMES
        for key in keys:
            if selected[key] != report_selected[key]:
                raise RuntimeError(
                    f"sealed {robot} policy differs from candidate metrics: {key}"
                )
        calibration_path = staging / descriptors["calibration_records"]["path"]
        with np.load(calibration_path, allow_pickle=False) as raw:
            names = tuple(raw["parameter_names"].astype(str).tolist())
            parameters = raw["candidate_parameters"]
        if names != PARAMETER_NAMES or parameters.shape != (144, 4):
            raise RuntimeError(f"sealed {robot} calibration parameter schema changed")
        index = int(selected["candidate_index"])
        expected = np.asarray([selected[name] for name in PARAMETER_NAMES], dtype=np.float64)
        if index < 0 or index >= 144 or not np.array_equal(parameters[index], expected):
            raise RuntimeError(f"sealed {robot} policy differs from raw candidate row")
    return seal


def _lite_backend(model_path: Path, nq: int) -> TorchScriptLiteGateInference:
    return TorchScriptLiteGateInference(
        load_exact_torchscript(model_path, device="cpu"),
        lite_feature_dim(int(nq)),
    )


def _build_lite_runtime(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    frozen_lite_root: Path,
    robot: str,
    kinematics: object,
    device: str,
) -> HierarchicalLiteRuntime:
    model_path, policy_path = _verify_frozen_lite(frozen_lite_root, robot)
    policy_config, _ = load_policy(policy_path)
    hard, dls, verifier = _fresh_shared_fixed_hard(
        source_config=source_config,
        release_root=release_v3_root,
        robot=robot,
        kinematics=kinematics,
        device=device,
    )
    return HierarchicalLiteRuntime(
        kinematics=kinematics,
        dls=dls,
        verifier=verifier,
        fast_gate=LiteFastGatePolicy(
            _lite_backend(model_path, int(kinematics.nq)), policy_config  # type: ignore[attr-defined]
        ),
        always_hard_runtime=hard,
        fast_iterations=1,
        name="hierarchical_cghik_v5_lite",
    )


def _build_temporal_runtime(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    frozen_lite_root: Path,
    robot: str,
    kinematics: object,
    device: str,
    policy_config: TemporalPolicyConfig,
    shared_components: tuple[object, object, object] | None = None,
    predictor: object | None = None,
) -> TemporalCGHIKRuntime:
    model_path, _ = _verify_frozen_lite(frozen_lite_root, robot)
    hard, dls, verifier = (
        _fresh_shared_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        if shared_components is None
        else shared_components
    )
    return TemporalCGHIKRuntime(
        kinematics=kinematics,
        dls=dls,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        predictor=(
            _lite_backend(model_path, int(kinematics.nq))  # type: ignore[attr-defined]
            if predictor is None
            else predictor
        ),
        always_hard_runtime=hard,  # type: ignore[arg-type]
        policy_config=policy_config,
        local_iterations=1,
    )


def _completion_vector(accepted: np.ndarray, role: TrajectoryRole) -> np.ndarray:
    values = np.asarray(accepted, dtype=bool)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != role.count:
        raise ValueError("accepted array does not match trajectory role")
    return np.asarray(
        [
            [bool(np.all(values[candidate, indices])) for _, indices in role.groups()]
            for candidate in range(values.shape[0])
        ],
        dtype=bool,
    )


def _state_code(state: TemporalState) -> int:
    return MODE_CODE[state.mode]


@dataclass(frozen=True)
class CalibrationData:
    configs: tuple[TemporalPolicyConfig, ...]
    latency_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    gate_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    hard_accepted: np.ndarray
    command_q: np.ndarray
    candidate_order_index: np.ndarray
    state_before: np.ndarray
    state_after: np.ndarray
    robust_frame_number: np.ndarray
    probe_scheduled: np.ndarray
    reentry_probability: np.ndarray
    streak_before: np.ndarray
    streak_after: np.ndarray
    mode_switched: np.ndarray
    switch_kind: np.ndarray
    route: np.ndarray
    trajectory_attempt_index: np.ndarray
    hard_reference_accepted: np.ndarray
    hard_reference_completion: np.ndarray


def _run_hard_reference(
    role: TrajectoryRole,
    runtime: object,
    *,
    dt: float,
) -> np.ndarray:
    accepted = np.zeros(role.count, dtype=bool)
    for _, indices in role.groups():
        previous = role.dataset.previous_q[int(indices[0])].copy()
        for index in indices:
            query = role.query(int(index), previous_q=previous, dt=dt)
            outcome = runtime.solve(query)  # type: ignore[attr-defined]
            accepted[index] = bool(outcome.accepted)
            if outcome.accepted and outcome.q is not None:
                previous = np.asarray(outcome.q, dtype=np.float64).copy()
    return accepted


def collect_calibration_grid(
    role: TrajectoryRole,
    *,
    configs: tuple[TemporalPolicyConfig, ...],
    runtimes: Sequence[TemporalCGHIKRuntime],
    hard_reference_runtime: object,
    dt: float,
    order_seed: int,
    progress_every: int,
    environment_guard: DevelopmentTrajectoryGuard | None = None,
) -> CalibrationData:
    """Run every grid member in closed loop on calibration only.

    Candidate order is randomized independently at every frame.  Each
    candidate owns its explicit temporal state and closed-loop command state;
    the numerical components are shared only because calls are sequential and
    those components are stateless with respect to trajectory identity.
    """

    candidate_count = len(configs)
    if len(runtimes) != candidate_count:
        raise ValueError("one temporal runtime is required per grid candidate")
    count = role.count
    nq = role.dataset.previous_q.shape[1]
    latency = np.zeros((candidate_count, count), dtype=np.int64)
    accepted = np.zeros((candidate_count, count), dtype=bool)
    fev = np.zeros((candidate_count, count), dtype=np.int32)
    seed = np.zeros((candidate_count, count), dtype=bool)
    gate = np.zeros((candidate_count, count), dtype=bool)
    local_attempted = np.zeros((candidate_count, count), dtype=bool)
    local_accepted = np.zeros((candidate_count, count), dtype=bool)
    hard_accepted = np.zeros((candidate_count, count), dtype=bool)
    command_q = np.full((candidate_count, count, nq), np.nan, dtype=np.float64)
    candidate_order = np.zeros((candidate_count, count), dtype=np.uint16)
    state_before = np.full((candidate_count, count), -1, dtype=np.int8)
    state_after = np.full_like(state_before, -1)
    robust_number = np.zeros((candidate_count, count), dtype=np.int16)
    probe_scheduled = np.zeros((candidate_count, count), dtype=bool)
    probability = np.full((candidate_count, count), np.nan, dtype=np.float64)
    streak_before = np.zeros((candidate_count, count), dtype=np.int16)
    streak_after = np.zeros((candidate_count, count), dtype=np.int16)
    switched = np.zeros((candidate_count, count), dtype=bool)
    switch_kind = np.full((candidate_count, count), "", dtype="U24")
    route = np.full((candidate_count, count), "", dtype="U40")
    attempt_index = np.zeros(count, dtype=np.int16)
    hard_reference = _run_hard_reference(role, hard_reference_runtime, dt=dt)

    groups = role.groups()
    for trajectory_number, (uid, indices) in enumerate(groups):
        attempt = 0
        while True:
            token = (
                None
                if environment_guard is None
                else environment_guard.wait_until_quiet(
                    context=(
                        f"temporal-v6/{role.robot}/calibration/{uid}/"
                        f"attempt{attempt}/before"
                    )
                )
            )
            latency[:, indices] = 0
            accepted[:, indices] = False
            fev[:, indices] = 0
            seed[:, indices] = False
            gate[:, indices] = False
            local_attempted[:, indices] = False
            local_accepted[:, indices] = False
            hard_accepted[:, indices] = False
            command_q[:, indices, :] = np.nan
            candidate_order[:, indices] = 0
            state_before[:, indices] = -1
            state_after[:, indices] = -1
            robust_number[:, indices] = 0
            probe_scheduled[:, indices] = False
            probability[:, indices] = np.nan
            streak_before[:, indices] = 0
            streak_after[:, indices] = 0
            switched[:, indices] = False
            switch_kind[:, indices] = ""
            route[:, indices] = ""
            states = [runtime.initial_state() for runtime in runtimes]
            previous = np.repeat(
                role.dataset.previous_q[int(indices[0])][None, :],
                candidate_count,
                axis=0,
            )
            for index in indices:
                rng = np.random.default_rng(
                    np.random.SeedSequence([order_seed, int(index)])
                )
                order = np.arange(candidate_count, dtype=np.int64)
                rng.shuffle(order)
                for position, candidate in enumerate(order):
                    candidate = int(candidate)
                    candidate_order[candidate, index] = position
                    query = role.query(
                        int(index), previous_q=previous[candidate], dt=dt
                    )
                    started = perf_counter_ns()
                    outcome = runtimes[candidate].step(query, states[candidate])
                    latency[candidate, index] = perf_counter_ns() - started
                    states[candidate] = outcome.state_after
                    accepted[candidate, index] = bool(outcome.accepted)
                    fev[candidate, index] = int(outcome.function_evaluations)
                    seed[candidate, index] = bool(
                        outcome.learned_seed_ensemble_invoked
                    )
                    gate[candidate, index] = bool(outcome.reentry_probe_executed)
                    local_attempted[candidate, index] = bool(outcome.local_attempted)
                    local_accepted[candidate, index] = bool(outcome.local_accepted)
                    hard_accepted[candidate, index] = bool(outcome.hard_accepted)
                    state_before[candidate, index] = _state_code(outcome.state_before)
                    state_after[candidate, index] = _state_code(outcome.state_after)
                    robust_number[candidate, index] = int(
                        outcome.robust_frame_number
                    )
                    probe_scheduled[candidate, index] = bool(
                        outcome.reentry_probe_scheduled
                    )
                    if outcome.reentry_probability is not None:
                        probability[candidate, index] = float(
                            outcome.reentry_probability
                        )
                    streak_before[candidate, index] = int(
                        outcome.state_before.reentry_high_streak
                    )
                    streak_after[candidate, index] = int(
                        outcome.state_after.reentry_high_streak
                    )
                    switched[candidate, index] = bool(outcome.mode_switched)
                    switch_kind[candidate, index] = str(outcome.switch_kind or "")
                    route[candidate, index] = str(outcome.route)
                    if outcome.accepted and outcome.q is not None:
                        command = np.asarray(outcome.q, dtype=np.float64)
                        command_q[candidate, index] = command
                        previous[candidate] = command
            observation = (
                None
                if environment_guard is None
                else environment_guard.observe(
                    context=(
                        f"temporal-v6/{role.robot}/calibration/{uid}/"
                        f"attempt{attempt}/after"
                    ),
                    since_sample_index=int(token["monitor_sample_index"]),
                )
            )
            if observation is None or not bool(observation["busy"]):
                attempt_index[indices] = attempt
                break
            environment_guard.record_contamination(
                context=f"temporal-v6/{role.robot}/calibration/{uid}",
                observation=observation,
                attempt_index=attempt,
                discarded_scope="complete_trajectory_all_144_candidates",
            )
            print(
                f"[temporal-v6] discarded contaminated calibration trajectory "
                f"{role.robot}/{uid} attempt={attempt}: "
                f"{observation.get('busy_sample_since_query_start') or observation}",
                flush=True,
            )
            attempt += 1
            if attempt >= environment_guard.max_contaminated_attempts:
                raise RuntimeError(
                    f"calibration trajectory remained contaminated: {role.robot}/{uid}; "
                    f"last_observation={observation}"
                )
        if progress_every and (trajectory_number + 1) % progress_every == 0:
            print(
                f"[temporal-v6] {role.robot} calibration "
                f"{trajectory_number + 1}/{len(groups)} trajectories",
                flush=True,
            )
    return CalibrationData(
        configs=configs,
        latency_ns=latency,
        accepted=accepted,
        function_evaluations=fev,
        seed_invoked=seed,
        gate_invoked=gate,
        local_attempted=local_attempted,
        local_accepted=local_accepted,
        hard_accepted=hard_accepted,
        command_q=command_q,
        candidate_order_index=candidate_order,
        state_before=state_before,
        state_after=state_after,
        robust_frame_number=robust_number,
        probe_scheduled=probe_scheduled,
        reentry_probability=probability,
        streak_before=streak_before,
        streak_after=streak_after,
        mode_switched=switched,
        switch_kind=switch_kind,
        route=route,
        trajectory_attempt_index=attempt_index,
        hard_reference_accepted=hard_reference,
        hard_reference_completion=_completion_vector(hard_reference, role)[0],
    )


def select_calibration_policy(
    data: CalibrationData,
    role: TrajectoryRole,
) -> tuple[int, TemporalPolicyConfig, dict[str, Any]]:
    """Apply the exact preregistered lexicographic selection order."""

    completion = _completion_vector(data.accepted, role)
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(data.configs):
        completion_equal = bool(
            np.array_equal(completion[index], data.hard_reference_completion)
        )
        rows.append(
            {
                "candidate_index": index,
                **asdict(config),
                "eligible": completion_equal,
                "trajectory_completion_vector_equal_always_hard": completion_equal,
                "trajectory_completion_rate": float(np.mean(completion[index])),
                "always_hard_trajectory_completion_rate": float(
                    np.mean(data.hard_reference_completion)
                ),
                "p95_ns": float(np.percentile(data.latency_ns[index], 95)),
                "p50_ns": float(np.percentile(data.latency_ns[index], 50)),
                "learned_seed_ensemble_invocation_rate": float(
                    np.mean(data.seed_invoked[index])
                ),
                "event_gate_invocation_rate": float(np.mean(data.gate_invoked[index])),
                "frame_verified_success": float(np.mean(data.accepted[index])),
                "mean_function_evaluations": float(
                    np.mean(data.function_evaluations[index])
                ),
                "mode_switch_rate": float(np.mean(data.mode_switched[index])),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no calibration candidate preserves the complete always-hard "
            "trajectory-completion vector; policy-validation remains unopened"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["p95_ns"],
            row["learned_seed_ensemble_invocation_rate"],
            row["p50_ns"],
            -row["reentry_threshold"],
            -row["consecutive_successes"],
            -row["hold_frames"],
            -row["probe_interval"],
        ),
    )
    selected_index = int(selected["candidate_index"])
    report = {
        "selection_role": role.role,
        "policy_validation_used_for_selection": False,
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "eligibility": "trajectory completion vector exactly equal to always-hard",
        "objective_order": [
            "minimum P95 end-to-end latency",
            "minimum learned-seed invocation rate",
            "minimum P50 end-to-end latency",
        ],
        "conservative_exact_tie_break": [
            "higher reentry threshold",
            "higher consecutive successes",
            "higher hold frames",
            "higher probe interval",
        ],
        "selected": dict(selected),
        "candidates": rows,
    }
    return selected_index, data.configs[selected_index], report


def _save_calibration(path: Path, data: CalibrationData, role: TrajectoryRole) -> None:
    parameters = np.asarray(
        [
            [
                cfg.hold_frames,
                cfg.probe_interval,
                cfg.consecutive_successes,
                cfg.reentry_threshold,
            ]
            for cfg in data.configs
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="U32"),
        candidate_parameters=parameters,
        trajectory_uid=role.trajectory_uid,
        category=role.dataset.category,
        time_index=role.dataset.time_index,
        latency_ns=data.latency_ns,
        accepted=data.accepted,
        function_evaluations=data.function_evaluations,
        learned_seed_ensemble_invoked=data.seed_invoked,
        event_gate_invoked=data.gate_invoked,
        local_attempted=data.local_attempted,
        local_accepted=data.local_accepted,
        hard_accepted=data.hard_accepted,
        command_q=data.command_q,
        candidate_order_index=data.candidate_order_index,
        state_before=data.state_before,
        state_after=data.state_after,
        robust_frame_number=data.robust_frame_number,
        probe_scheduled=data.probe_scheduled,
        reentry_probability=data.reentry_probability,
        reentry_high_streak_before=data.streak_before,
        reentry_high_streak_after=data.streak_after,
        mode_switched=data.mode_switched,
        switch_kind=data.switch_kind,
        route=data.route,
        trajectory_attempt_index=data.trajectory_attempt_index,
        always_hard_accepted=data.hard_reference_accepted,
        always_hard_trajectory_completion=data.hard_reference_completion,
    )


@dataclass(frozen=True)
class BenchmarkData:
    robot: str
    trajectory_uid: np.ndarray
    category: np.ndarray
    time_index: np.ndarray
    source_frame_sha256: np.ndarray
    executed_query_sha256: np.ndarray
    method_order_index: np.ndarray
    latency_ns: np.ndarray
    stage_latency_ns: np.ndarray
    accepted: np.ndarray
    command_q: np.ndarray
    function_evaluations: np.ndarray
    iterations: np.ndarray
    fallback_used: np.ndarray
    candidate_count: np.ndarray
    verification_reasons: np.ndarray
    reject_reason: np.ndarray
    learned_seed_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    route: np.ndarray
    executed_stages: np.ndarray
    temporal_state_before: np.ndarray
    temporal_state_after: np.ndarray
    temporal_robust_frame_number: np.ndarray
    temporal_probe_scheduled: np.ndarray
    temporal_probe_executed: np.ndarray
    temporal_reentry_probability: np.ndarray
    temporal_streak_before: np.ndarray
    temporal_streak_after: np.ndarray
    temporal_switch_kind: np.ndarray
    temporal_recovery_delay_frames: np.ndarray
    trajectory_attempt_index: np.ndarray


def _build_policy_validation_methods(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    release_v4_root: Path,
    frozen_lite_root: Path,
    robot: str,
    kinematics: object,
    device: str,
    temporal_config: TemporalPolicyConfig,
) -> dict[str, object]:
    methods: dict[str, object] = {
        "always_hard": _fresh_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        ),
        "counterfactual_cghik_v4": _build_current_v4(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        ),
        "hierarchical_cghik_v5_lite": _build_lite_runtime(
            source_config=source_config,
            release_v3_root=release_v3_root,
            frozen_lite_root=frozen_lite_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        ),
        "temporal_cghik_v6": _build_temporal_runtime(
            source_config=source_config,
            release_v3_root=release_v3_root,
            frozen_lite_root=frozen_lite_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
            policy_config=temporal_config,
        ),
    }
    if tuple(methods) != METHODS:
        raise RuntimeError("Temporal V6 four-method registry drifted")
    return methods


def _stage_vector(method: str, outcome: object, outer_ns: int) -> np.ndarray:
    values = np.zeros(len(STAGES), dtype=np.int64)
    if method in {"always_hard", "counterfactual_cghik_v4"}:
        values[STAGES.index("robust")] = int(outer_ns)
        return values
    timings = dict(getattr(outcome, "timings_ns", {}))
    if method == "hierarchical_cghik_v5_lite":
        values[STAGES.index("feature")] = int(timings.get("feature_extraction_ns", 0))
        values[STAGES.index("gate")] = int(timings.get("gate_ns", 0))
        values[STAGES.index("local")] = int(timings.get("local_path_ns", 0))
        values[STAGES.index("robust")] = int(timings.get("robust_path_ns", 0))
    elif method == "temporal_cghik_v6":
        values[STAGES.index("temporal")] = int(timings.get("state_policy_ns", 0))
        values[STAGES.index("feature")] = int(timings.get("event_feature_ns", 0))
        values[STAGES.index("gate")] = int(timings.get("event_gate_ns", 0))
        values[STAGES.index("local")] = int(timings.get("local_path_ns", 0))
        values[STAGES.index("robust")] = int(timings.get("hard_path_ns", 0))
    else:  # pragma: no cover - registry guard
        raise KeyError(method)
    attributed = int(np.sum(values))
    if attributed > int(outer_ns):
        raise RuntimeError(
            f"stage attribution exceeds outer latency for {method}: "
            f"{attributed}>{outer_ns}"
        )
    values[STAGES.index("unattributed")] = int(outer_ns) - attributed
    return values


def _warm_methods(
    methods: Mapping[str, object],
    role: TrajectoryRole,
    *,
    frames: int,
    dt: float,
) -> None:
    if frames <= 0:
        return
    temporal = methods["temporal_cghik_v6"]
    temporal_state = temporal.initial_state()  # type: ignore[attr-defined]
    previous = {
        method: role.dataset.previous_q[0].copy() for method in METHODS
    }
    prior_uid = str(role.trajectory_uid[0])
    for warm_index in range(frames):
        index = warm_index % role.count
        uid = str(role.trajectory_uid[index])
        if uid != prior_uid or int(role.dataset.time_index[index]) == 0:
            previous = {
                method: role.dataset.previous_q[index].copy() for method in METHODS
            }
            temporal_state = temporal.initial_state()  # type: ignore[attr-defined]
            prior_uid = uid
        for method, runtime in methods.items():
            query = role.query(index, previous_q=previous[method], dt=dt)
            if method == "temporal_cghik_v6":
                outcome = runtime.step(query, temporal_state)  # type: ignore[attr-defined]
                temporal_state = outcome.state_after
            else:
                outcome = runtime.solve(query)  # type: ignore[attr-defined]
            if outcome.accepted and outcome.q is not None:
                previous[method] = np.asarray(outcome.q, dtype=np.float64).copy()


def benchmark_policy_validation(
    role: TrajectoryRole,
    *,
    methods: Mapping[str, object],
    dt: float,
    order_seed: int,
    progress_every: int,
    environment_guard: DevelopmentTrajectoryGuard | None = None,
) -> BenchmarkData:
    """Run each method exactly once per closed-loop frame."""

    count = role.count
    method_count = len(METHODS)
    nq = role.dataset.previous_q.shape[1]
    executed_hash = np.full((count, method_count), "", dtype="U64")
    order_index = np.zeros((count, method_count), dtype=np.int8)
    latency = np.zeros((count, method_count), dtype=np.int64)
    stages = np.zeros((count, method_count, len(STAGES)), dtype=np.int64)
    accepted = np.zeros((count, method_count), dtype=bool)
    command = np.full((count, method_count, nq), np.nan, dtype=np.float64)
    fev = np.zeros((count, method_count), dtype=np.int32)
    iterations = np.zeros((count, method_count), dtype=np.int32)
    fallback = np.zeros((count, method_count), dtype=bool)
    candidate_count = np.zeros((count, method_count), dtype=np.int16)
    verification_reasons = np.full((count, method_count), "", dtype="U160")
    reject_reason = np.full((count, method_count), "", dtype="U80")
    seed = np.zeros((count, method_count), dtype=bool)
    local_attempted = np.zeros((count, method_count), dtype=bool)
    local_accepted = np.zeros((count, method_count), dtype=bool)
    route = np.full((count, method_count), "", dtype="U48")
    executed_stages = np.full((count, method_count), "", dtype="U96")
    state_before = np.full(count, -1, dtype=np.int8)
    state_after = np.full(count, -1, dtype=np.int8)
    robust_number = np.zeros(count, dtype=np.int16)
    probe_scheduled = np.zeros(count, dtype=bool)
    probe_executed = np.zeros(count, dtype=bool)
    reentry_probability = np.full(count, np.nan, dtype=np.float64)
    streak_before = np.zeros(count, dtype=np.int16)
    streak_after = np.zeros(count, dtype=np.int16)
    switch_kind = np.full(count, "", dtype="U24")
    recovery_delay = np.full(count, -1, dtype=np.int16)
    attempt_index = np.zeros(count, dtype=np.int16)
    source_hash = np.asarray(
        [query_digest(role.query(i, dt=dt)) for i in range(count)], dtype="U64"
    )

    for trajectory_number, (uid, indices) in enumerate(role.groups()):
        attempt = 0
        while True:
            token = (
                None
                if environment_guard is None
                else environment_guard.wait_until_quiet(
                    context=(
                        f"temporal-v6/{role.robot}/policy-validation/{uid}/"
                        f"attempt{attempt}/before"
                    )
                )
            )
            executed_hash[indices, :] = ""
            order_index[indices, :] = 0
            latency[indices, :] = 0
            stages[indices, :, :] = 0
            accepted[indices, :] = False
            command[indices, :, :] = np.nan
            fev[indices, :] = 0
            iterations[indices, :] = 0
            fallback[indices, :] = False
            candidate_count[indices, :] = 0
            verification_reasons[indices, :] = ""
            reject_reason[indices, :] = ""
            seed[indices, :] = False
            local_attempted[indices, :] = False
            local_accepted[indices, :] = False
            route[indices, :] = ""
            executed_stages[indices, :] = ""
            state_before[indices] = -1
            state_after[indices] = -1
            robust_number[indices] = 0
            probe_scheduled[indices] = False
            probe_executed[indices] = False
            reentry_probability[indices] = np.nan
            streak_before[indices] = 0
            streak_after[indices] = 0
            switch_kind[indices] = ""
            recovery_delay[indices] = -1
            previous = {
                method: role.dataset.previous_q[int(indices[0])].copy()
                for method in METHODS
            }
            temporal_runtime = methods["temporal_cghik_v6"]
            temporal_state = temporal_runtime.initial_state()  # type: ignore[attr-defined]
            orders = latin_method_orders(METHODS, order_seed + trajectory_number)
            for local_frame, index_value in enumerate(indices):
                index = int(index_value)
                order = orders[local_frame % len(orders)]
                for position, method in enumerate(order):
                    column = METHODS.index(method)
                    order_index[index, column] = position
                    query = role.query(index, previous_q=previous[method], dt=dt)
                    executed_hash[index, column] = query_digest(query)
                    started = perf_counter_ns()
                    if method == "temporal_cghik_v6":
                        outcome = methods[method].step(query, temporal_state)  # type: ignore[attr-defined]
                    else:
                        outcome = methods[method].solve(query)  # type: ignore[attr-defined]
                    outer_ns = perf_counter_ns() - started
                    if method == "temporal_cghik_v6":
                        temporal_state = outcome.state_after
                    latency[index, column] = int(outer_ns)
                    stages[index, column] = _stage_vector(method, outcome, outer_ns)
                    accepted[index, column] = bool(outcome.accepted)
                    fev[index, column] = int(outcome.function_evaluations)
                    iterations[index, column] = int(outcome.iterations)
                    fallback[index, column] = bool(outcome.fallback_used)
                    candidate_count[index, column] = int(outcome.candidate_count)
                    verification_reasons[index, column] = "|".join(
                        str(value) for value in outcome.verification_reasons
                    )
                    reject_reason[index, column] = str(outcome.reject_reason)
                    route[index, column] = str(
                        getattr(outcome, "route", getattr(outcome, "entry_action", ""))
                    )
                    executed_stages[index, column] = "|".join(
                        str(value)
                        for value in getattr(outcome, "executed_stages", ())
                    )
                    if method in {"always_hard", "counterfactual_cghik_v4"}:
                        seed[index, column] = True
                    else:
                        seed[index, column] = bool(
                            outcome.learned_seed_ensemble_invoked
                        )
                        local_attempted[index, column] = bool(outcome.local_attempted)
                        local_accepted[index, column] = bool(outcome.local_accepted)
                    if outcome.accepted and outcome.q is not None:
                        q = np.asarray(outcome.q, dtype=np.float64).copy()
                        command[index, column] = q
                        previous[method] = q
                    if method == "temporal_cghik_v6":
                        state_before[index] = _state_code(outcome.state_before)
                        state_after[index] = _state_code(outcome.state_after)
                        robust_number[index] = int(outcome.robust_frame_number)
                        probe_scheduled[index] = bool(
                            outcome.reentry_probe_scheduled
                        )
                        probe_executed[index] = bool(outcome.reentry_probe_executed)
                        if outcome.reentry_probability is not None:
                            reentry_probability[index] = float(
                                outcome.reentry_probability
                            )
                        streak_before[index] = int(
                            outcome.state_before.reentry_high_streak
                        )
                        streak_after[index] = int(
                            outcome.state_after.reentry_high_streak
                        )
                        switch_kind[index] = str(outcome.switch_kind or "")
                        if outcome.robust_to_local_recovery_delay_frames is not None:
                            recovery_delay[index] = int(
                                outcome.robust_to_local_recovery_delay_frames
                            )
            observation = (
                None
                if environment_guard is None
                else environment_guard.observe(
                    context=(
                        f"temporal-v6/{role.robot}/policy-validation/{uid}/"
                        f"attempt{attempt}/after"
                    ),
                    since_sample_index=int(token["monitor_sample_index"]),
                )
            )
            if observation is None or not bool(observation["busy"]):
                attempt_index[indices] = attempt
                break
            environment_guard.record_contamination(
                context=f"temporal-v6/{role.robot}/policy-validation/{uid}",
                observation=observation,
                attempt_index=attempt,
                discarded_scope="complete_trajectory_all_four_methods",
            )
            print(
                f"[temporal-v6] discarded contaminated policy-validation trajectory "
                f"{role.robot}/{uid} attempt={attempt}: "
                f"{observation.get('busy_sample_since_query_start') or observation}",
                flush=True,
            )
            attempt += 1
            if attempt >= environment_guard.max_contaminated_attempts:
                raise RuntimeError(
                    f"policy-validation trajectory remained contaminated: "
                    f"{role.robot}/{uid}; last_observation={observation}"
                )
        if progress_every and (trajectory_number + 1) % progress_every == 0:
            print(
                f"[temporal-v6] {role.robot} policy-validation "
                f"{trajectory_number + 1}/{len(role.trajectory_order)} trajectories",
                flush=True,
            )
    for row in order_index:
        if not np.array_equal(np.sort(row), np.arange(method_count, dtype=np.int8)):
            raise RuntimeError("method order is not a four-method permutation")
    # A 150-frame trajectory contains 37 full 4x4 Latin blocks plus two
    # positions.  Each complete trajectory must therefore be balanced to
    # within one occurrence per position; the exact aggregate counts are
    # preserved in the manifest rather than incorrectly forced to equality.
    for _, indices in role.groups():
        for column in range(method_count):
            counts = np.bincount(
                order_index[indices, column], minlength=method_count
            )
            if int(np.max(counts) - np.min(counts)) > 1:
                raise RuntimeError("four-method Latin order is not path-balanced")
    if not np.array_equal(np.sum(stages, axis=2), latency):
        raise RuntimeError("policy-validation stage latency does not close to total")
    return BenchmarkData(
        role.robot,
        role.trajectory_uid.copy(),
        role.dataset.category.copy(),
        role.dataset.time_index.copy(),
        source_hash,
        executed_hash,
        order_index,
        latency,
        stages,
        accepted,
        command,
        fev,
        iterations,
        fallback,
        candidate_count,
        verification_reasons,
        reject_reason,
        seed,
        local_attempted,
        local_accepted,
        route,
        executed_stages,
        state_before,
        state_after,
        robust_number,
        probe_scheduled,
        probe_executed,
        reentry_probability,
        streak_before,
        streak_after,
        switch_kind,
        recovery_delay,
        attempt_index,
    )


def _save_benchmark(path: Path, data: BenchmarkData) -> None:
    np.savez_compressed(
        path,
        method_names=np.asarray(METHODS, dtype="U40"),
        stage_names=np.asarray(STAGES, dtype="U20"),
        trajectory_uid=data.trajectory_uid,
        category=data.category,
        time_index=data.time_index,
        source_frame_sha256=data.source_frame_sha256,
        executed_query_sha256=data.executed_query_sha256,
        method_order_index=data.method_order_index,
        latency_ns=data.latency_ns,
        stage_latency_ns=data.stage_latency_ns,
        accepted=data.accepted,
        command_q=data.command_q,
        function_evaluations=data.function_evaluations,
        iterations=data.iterations,
        fallback_used=data.fallback_used,
        candidate_count=data.candidate_count,
        verification_reasons=data.verification_reasons,
        reject_reason=data.reject_reason,
        learned_seed_ensemble_invoked=data.learned_seed_invoked,
        local_attempted=data.local_attempted,
        local_accepted=data.local_accepted,
        route=data.route,
        executed_stages=data.executed_stages,
        temporal_state_before=data.temporal_state_before,
        temporal_state_after=data.temporal_state_after,
        temporal_robust_frame_number=data.temporal_robust_frame_number,
        temporal_probe_scheduled=data.temporal_probe_scheduled,
        temporal_probe_executed=data.temporal_probe_executed,
        temporal_reentry_probability=data.temporal_reentry_probability,
        temporal_streak_before=data.temporal_streak_before,
        temporal_streak_after=data.temporal_streak_after,
        temporal_switch_kind=data.temporal_switch_kind,
        temporal_recovery_delay_frames=data.temporal_recovery_delay_frames,
        trajectory_attempt_index=data.trajectory_attempt_index,
    )


def _completion_by_method(data: BenchmarkData) -> np.ndarray:
    values = np.zeros((len(METHODS), len(set(data.trajectory_uid.astype(str)))), dtype=bool)
    order = list(dict.fromkeys(data.trajectory_uid.astype(str).tolist()))
    for method in range(len(METHODS)):
        values[method] = [
            bool(np.all(data.accepted[data.trajectory_uid == uid, method]))
            for uid in order
        ]
    return values


def summarize_benchmark(data: BenchmarkData) -> list[dict[str, Any]]:
    completion = _completion_by_method(data)
    rows: list[dict[str, Any]] = []
    temporal_column = METHODS.index("temporal_cghik_v6")
    for column, method in enumerate(METHODS):
        latency_ms = data.latency_ns[:, column].astype(np.float64) / 1e6
        row: dict[str, Any] = {
            "robot": data.robot,
            "method": method,
            "trajectory_count": int(completion.shape[1]),
            "trajectory_completion": float(np.mean(completion[column])),
            "frame_verified_success": float(np.mean(data.accepted[:, column])),
            "p50_ms": float(np.percentile(latency_ms, 50)),
            "p95_ms": float(np.percentile(latency_ms, 95)),
            "p99_ms": float(np.percentile(latency_ms, 99)),
            "mean_fev": float(np.mean(data.function_evaluations[:, column])),
            "learned_seed_invocation_rate": float(
                np.mean(data.learned_seed_invoked[:, column])
            ),
            "local_occupancy": None,
            "robust_occupancy": None,
            "mode_switches_per_trajectory": None,
            "local_to_robust_transition_count": None,
            "local_to_robust_transition_latency_p50_ms": None,
            "local_to_robust_transition_latency_p95_ms": None,
            "robust_to_local_transition_count": None,
            "robust_to_local_recovery_delay_mean_frames": None,
            "robust_to_local_recovery_delay_p95_frames": None,
        }
        if column == temporal_column:
            local = data.temporal_state_before == MODE_CODE[TemporalMode.LOCAL]
            robust = ~local
            switches = data.temporal_switch_kind != ""
            per_trajectory_switch = [
                int(np.sum(switches[data.trajectory_uid == uid]))
                for uid in dict.fromkeys(data.trajectory_uid.astype(str).tolist())
            ]
            l2r = data.temporal_switch_kind == "local_to_robust"
            r2l = data.temporal_switch_kind == "robust_to_local"
            l2r_latency = latency_ms[l2r]
            delays = data.temporal_recovery_delay_frames[r2l]
            delays = delays[delays >= 0]
            row.update(
                {
                    "local_occupancy": float(np.mean(local)),
                    "robust_occupancy": float(np.mean(robust)),
                    "mode_switches_per_trajectory": float(
                        np.mean(per_trajectory_switch)
                    ),
                    "local_to_robust_transition_count": int(np.sum(l2r)),
                    "local_to_robust_transition_latency_p50_ms": (
                        float(np.percentile(l2r_latency, 50))
                        if l2r_latency.size
                        else None
                    ),
                    "local_to_robust_transition_latency_p95_ms": (
                        float(np.percentile(l2r_latency, 95))
                        if l2r_latency.size
                        else None
                    ),
                    "robust_to_local_transition_count": int(np.sum(r2l)),
                    "robust_to_local_recovery_delay_mean_frames": (
                        float(np.mean(delays)) if delays.size else None
                    ),
                    "robust_to_local_recovery_delay_p95_frames": (
                        float(np.percentile(delays, 95)) if delays.size else None
                    ),
                }
            )
        rows.append(row)
    return rows


def family_mode_distribution(data: BenchmarkData) -> list[dict[str, Any]]:
    column = METHODS.index("temporal_cghik_v6")
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        selected = data.category == family
        total = int(np.sum(selected))
        local = data.temporal_state_before[selected] == MODE_CODE[TemporalMode.LOCAL]
        completion_values = []
        for uid in dict.fromkeys(data.trajectory_uid[selected].astype(str).tolist()):
            frames = data.trajectory_uid == uid
            completion_values.append(bool(np.all(data.accepted[frames, column])))
        rows.append(
            {
                "robot": data.robot,
                "trajectory_family": family,
                "frame_count": total,
                "trajectory_count": len(completion_values),
                "local_occupancy": float(np.mean(local)),
                "robust_occupancy": float(1.0 - np.mean(local)),
                "learned_seed_invocation_rate": float(
                    np.mean(data.learned_seed_invoked[selected, column])
                ),
                "frame_verified_success": float(
                    np.mean(data.accepted[selected, column])
                ),
                "trajectory_completion": float(np.mean(completion_values)),
                "mode_switch_rate": float(
                    np.mean(data.temporal_switch_kind[selected] != "")
                ),
            }
        )
    return rows


def paired_latency(data: BenchmarkData) -> list[dict[str, Any]]:
    temporal = METHODS.index("temporal_cghik_v6")
    rows: list[dict[str, Any]] = []
    key_digest = sha256(
        np.ascontiguousarray(
            np.asarray(
                [
                    f"{uid}:{time_index}"
                    for uid, time_index in zip(
                        data.trajectory_uid.astype(str), data.time_index, strict=True
                    )
                ],
                dtype="S96",
            )
        ).tobytes()
    ).hexdigest()
    for comparator in METHODS[:-1]:
        column = METHODS.index(comparator)
        delta = (
            data.latency_ns[:, temporal].astype(np.float64)
            - data.latency_ns[:, column].astype(np.float64)
        ) / 1e6
        rows.append(
            {
                "robot": data.robot,
                "comparison": f"temporal_cghik_v6_minus_{comparator}",
                "paired_frame_count": int(delta.size),
                "paired_key_digest": key_digest,
                "mean_difference_ms": float(np.mean(delta)),
                "median_difference_ms": float(np.median(delta)),
                "p05_difference_ms": float(np.percentile(delta, 5)),
                "p95_difference_ms": float(np.percentile(delta, 95)),
                "fraction_temporal_faster": float(np.mean(delta < 0.0)),
            }
        )
    return rows


def stage_latency_summary(data: BenchmarkData) -> list[dict[str, Any]]:
    """Summarize unconditional and invoked-only stage costs from raw frames."""

    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        for stage_index, stage in enumerate(STAGES):
            values = data.stage_latency_ns[:, method_index, stage_index].astype(
                np.float64
            ) / 1e6
            invoked = values[values > 0.0]
            rows.append(
                {
                    "robot": data.robot,
                    "method": method,
                    "stage": stage,
                    "frame_count": int(values.size),
                    "invoked_frame_count": int(invoked.size),
                    "invocation_rate": float(np.mean(values > 0.0)),
                    "unconditional_mean_ms": float(np.mean(values)),
                    "unconditional_p50_ms": float(np.percentile(values, 50)),
                    "unconditional_p95_ms": float(np.percentile(values, 95)),
                    "unconditional_p99_ms": float(np.percentile(values, 99)),
                    "invoked_mean_ms": float(np.mean(invoked)) if invoked.size else None,
                    "invoked_p50_ms": (
                        float(np.percentile(invoked, 50)) if invoked.size else None
                    ),
                    "invoked_p95_ms": (
                        float(np.percentile(invoked, 95)) if invoked.size else None
                    ),
                    "invoked_p99_ms": (
                        float(np.percentile(invoked, 99)) if invoked.size else None
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _main_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Temporal V6 development-only trajectory pilot",
        "",
        "| Robot | Method | Completion | Frame success | P50 ms | P95 ms | P99 ms | Mean FEV | Seed rate | LOCAL occupancy | Switches/traj |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        local = "--" if row["local_occupancy"] is None else f"{float(row['local_occupancy']):.4f}"
        switches = "--" if row["mode_switches_per_trajectory"] is None else f"{float(row['mode_switches_per_trajectory']):.3f}"
        lines.append(
            f"| {row['robot']} | {row['method']} | {float(row['trajectory_completion']):.4f} | "
            f"{float(row['frame_verified_success']):.4f} | {float(row['p50_ms']):.4f} | "
            f"{float(row['p95_ms']):.4f} | {float(row['p99_ms']):.4f} | "
            f"{float(row['mean_fev']):.4f} | {float(row['learned_seed_invocation_rate']):.4f} | "
            f"{local} | {switches} |"
        )
    return "\n".join(lines) + "\n"


def _plot_results(
    main_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    benchmarks: Mapping[str, BenchmarkData],
    output: Path,
    *,
    dpi: int,
) -> dict[str, Any]:
    labels = ["Hard", "V4", "V5-Lite", "Temporal V6"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, robot in zip(axes, ("panda", "ur5e"), strict=True):
        rows = [row for row in main_rows if row["robot"] == robot]
        x = np.arange(len(METHODS))
        for offset, metric in enumerate(("p50_ms", "p95_ms", "p99_ms")):
            ax.bar(
                x + (offset - 1) * 0.24,
                [float(row[metric]) for row in rows],
                0.24,
                label=metric.upper(),
            )
        ax.set_xticks(x, labels, rotation=18)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(robot.upper())
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "fig1_four_method_latency.png", dpi=dpi)
    fig.savefig(output / "fig1_four_method_latency.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, robot in zip(axes, ("panda", "ur5e"), strict=True):
        rows = [row for row in family_rows if row["robot"] == robot]
        y = np.arange(len(FAMILIES))
        local = np.asarray([float(row["local_occupancy"]) for row in rows])
        robust = 1.0 - local
        ax.barh(y, local, color="#2a9d8f", label="LOCAL")
        ax.barh(y, robust, left=local, color="#264653", label="ROBUST")
        ax.set_yticks(y, [family.replace("trajectory_", "") for family in FAMILIES])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Frame occupancy")
        ax.set_title(robot.upper())
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "fig2_family_mode_distribution.png", dpi=dpi)
    fig.savefig(output / "fig2_family_mode_distribution.pdf")
    plt.close(fig)

    representative: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=False)
    temporal_column = METHODS.index("temporal_cghik_v6")
    for ax, robot in zip(axes, ("panda", "ur5e"), strict=True):
        data = benchmarks[robot]
        candidates = sorted(
            set(data.trajectory_uid[data.category == "trajectory_singular"].astype(str))
        )
        uid = candidates[0]
        selected = data.trajectory_uid == uid
        time_index = data.time_index[selected]
        latency_ms = data.latency_ns[selected, temporal_column] / 1e6
        fev = data.function_evaluations[selected, temporal_column]
        local = data.temporal_state_before[selected] == MODE_CODE[TemporalMode.LOCAL]
        for index in range(len(time_index)):
            ax.axvspan(
                time_index[index] - 0.5,
                time_index[index] + 0.5,
                color="#d8f3dc" if local[index] else "#e9ecef",
                alpha=0.55,
                linewidth=0,
            )
        latency_line = ax.plot(time_index, latency_ms, color="#1d3557", lw=1.3, label="Latency (ms)")[0]
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{robot.upper()} — first prespecified low-manipulability trajectory")
        twin = ax.twinx()
        fev_line = twin.plot(time_index, fev, color="#e76f51", lw=1.0, alpha=0.85, label="FEV")[0]
        twin.set_ylabel("FEV")
        ax.legend([latency_line, fev_line], ["Latency", "FEV"], frameon=False, loc="upper right")
        ax.set_xlabel("Frame")
        representative[robot] = {
            "trajectory_uid": uid,
            "family": "trajectory_singular",
            "selection_rule": "lexicographically first trajectory UID",
        }
    fig.tight_layout()
    fig.savefig(output / "fig3_representative_temporal_trace.png", dpi=dpi)
    fig.savefig(output / "fig3_representative_temporal_trace.pdf")
    plt.close(fig)
    return representative


def _pilot_gate(rows: Sequence[Mapping[str, Any]], data: Mapping[str, BenchmarkData]) -> dict[str, Any]:
    by_key = {(str(row["robot"]), str(row["method"])): row for row in rows}
    robots: dict[str, Any] = {}
    for robot in ("panda", "ur5e"):
        hard = by_key[(robot, "always_hard")]
        v4 = by_key[(robot, "counterfactual_cghik_v4")]
        v6 = by_key[(robot, "temporal_cghik_v6")]
        completion = _completion_by_method(data[robot])
        exact = bool(
            np.array_equal(
                completion[METHODS.index("temporal_cghik_v6")],
                completion[METHODS.index("always_hard")],
            )
        )
        checks = {
            "trajectory_completion_vector_equal_always_hard": exact,
            "p95_not_above_always_hard": float(v6["p95_ms"]) <= float(hard["p95_ms"]),
            "p50_below_counterfactual_cghik_v4": float(v6["p50_ms"]) < float(v4["p50_ms"]),
            "learned_seed_invocation_rate_at_most_0_70": float(
                v6["learned_seed_invocation_rate"]
            )
            <= 0.70,
        }
        robots[robot] = {
            "all_pass": bool(all(checks.values())),
            "checks": checks,
            "observed": {
                "completion_v6": v6["trajectory_completion"],
                "completion_always_hard": hard["trajectory_completion"],
                "p95_ratio_vs_always_hard": float(v6["p95_ms"]) / float(hard["p95_ms"]),
                "p50_ratio_vs_counterfactual_cghik_v4": float(v6["p50_ms"]) / float(v4["p50_ms"]),
                "learned_seed_invocation_rate": v6["learned_seed_invocation_rate"],
            },
        }
    return {
        "status": "pass" if all(value["all_pass"] for value in robots.values()) else "fail",
        "all_robots_pass": bool(all(value["all_pass"] for value in robots.values())),
        "fresh_or_formal_evaluation_authorized": False,
        "policy_validation_used_for_retuning": False,
        "robots": robots,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1+1 trajectory-per-family development smoke.",
    )
    return parser


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    if smoke:
        config = json.loads(json.dumps(config))
        config["trajectory_data"]["paths_per_family_pool"] = 2
        config["trajectory_data"]["paths_per_family_per_role"] = 1
        config["trajectory_data"]["steps_per_trajectory"] = 24
        config["timing"]["warmup_frames"] = 4
        config["runtime"]["progress_every_trajectories"] = 0
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    release_v3_root = resolve_path(config, str(config["release_v3_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    frozen_lite_root = resolve_path(config, str(config["frozen_v5_lite_root"]))
    output_root = resolve_path(config, str(config["output_root"]))
    if smoke:
        output_root = (workspace / "outputs" / "temporal_v6_smoke").resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"overwrite forbidden: {output_root}")
    stale = sorted(output_root.parent.glob(f".{output_root.name}.incomplete.*"))
    if stale:
        raise FileExistsError(
            "stale Temporal V6 staging requires inspection: "
            + ", ".join(str(path) for path in stale)
        )
    git_status_start = subprocess.run(
        ["git", "status", "--short"], cwd=workspace, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    if not smoke and git_status_start:
        raise RuntimeError("full Temporal V6 pilot requires a clean committed worktree")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    protected_roots = {
        "hierarchical_v5_pilot": workspace / "outputs" / "hierarchical_v5_pilot",
        "hierarchical_v5_lite_pilot": frozen_lite_root,
        "release_v3_locked": release_v3_root,
        "release_v4_locked": release_v4_root,
    }
    protected_before = {
        name: _tree_snapshot(path) for name, path in protected_roots.items()
    }
    protected_before_digest = {
        name: _snapshot_digest(snapshot)
        for name, snapshot in protected_before.items()
    }
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("state.py").resolve(),
        Path(__file__).with_name("policy.py").resolve(),
        Path(__file__).with_name("runtime.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
        Path(config["_config_path"]).resolve(),
        source_config_path.resolve(),
        (workspace / "scripts" / "run_temporal_v6_pilot.sh").resolve(),
        (workspace / "tests" / "test_temporal_v6.py").resolve(),
    )
    implementation_sources = {
        str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
        for path in implementation_paths
    }
    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    staging.mkdir(parents=True)
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
    started = _utc()
    dt = float(config["trajectory_data"]["dt"])
    device = str(config["runtime"]["device"])
    grid = _expected_grid(config)
    contexts: dict[str, dict[str, Any]] = {}
    split_manifest: dict[str, Any] = {}
    calibration_reports: dict[str, Any] = {}
    release_inputs: dict[str, Any] = {}
    frozen_lite_inputs: dict[str, Any] = {}
    quiet_events: dict[str, Any] = {}
    host_guard = DevelopmentTrajectoryGuard(config)
    quiet_events["preflight"] = host_guard.wait_until_quiet(
        context="temporal-v6/preflight"
    )

    # Phase A: generate/split outcome-blind pools and use calibration only.
    for robot in config["robots"]:
        robot = str(robot)
        kinematics = load_robot(source_config, robot)
        calibration_role, policy_role, split = generate_development_roles(
            kinematics,
            robot=robot,
            paths_per_family_pool=int(
                config["trajectory_data"]["paths_per_family_pool"]
            ),
            paths_per_family_per_role=int(
                config["trajectory_data"]["paths_per_family_per_role"]
            ),
            steps=int(config["trajectory_data"]["steps_per_trajectory"]),
            pool_seed=int(config["trajectory_data"]["pool_seed"][robot]),
            split_seed=int(config["trajectory_data"]["split_seed"][robot]),
            dt=dt,
        )
        _save_role(staging / f"{robot}_trajectory_calibration.npz", calibration_role)
        _save_role(
            staging / f"{robot}_trajectory_policy_validation.npz", policy_role
        )
        split_manifest[robot] = split
        release_inputs[robot] = _verified_release_inputs(
            workspace=workspace,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
        )
        model_path, _ = _verify_frozen_lite(frozen_lite_root, robot)
        frozen_lite_inputs[robot] = _sealed_lite_inputs(
            frozen_lite_root, robot, workspace=workspace
        )
        shared = _fresh_shared_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        predictor = _lite_backend(model_path, int(kinematics.nq))
        runtimes = [
            _build_temporal_runtime(
                source_config=source_config,
                release_v3_root=release_v3_root,
                frozen_lite_root=frozen_lite_root,
                robot=robot,
                kinematics=kinematics,
                device=device,
                policy_config=policy,
                shared_components=shared,
                predictor=predictor,
            )
            for policy in grid
        ]
        # Warm the shared HARD implementation and the exact predictor without
        # carrying any temporal state into calibration.
        warm_query = calibration_role.query(0, dt=dt)
        for _ in range(int(config["timing"]["warmup_frames"])):
            shared[0].solve(warm_query)  # type: ignore[attr-defined]
        predictor.predict_one(prepare_lite_features(kinematics, warm_query).features)
        hard_reference = _fresh_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        calibration_data = collect_calibration_grid(
            calibration_role,
            configs=grid,
            runtimes=runtimes,
            hard_reference_runtime=hard_reference,
            dt=dt,
            order_seed=881_000 + int(config["trajectory_data"]["split_seed"][robot]),
            progress_every=int(config["runtime"]["progress_every_trajectories"]),
            environment_guard=host_guard,
        )
        selected_index, selected_policy, report = select_calibration_policy(
            calibration_data, calibration_role
        )
        calibration_path = staging / f"{robot}_calibration_records.npz"
        _save_calibration(calibration_path, calibration_data, calibration_role)
        calibration_reports[robot] = report
        contexts[robot] = {
            "kinematics": kinematics,
            "calibration_role": calibration_role,
            "selected_index": selected_index,
            "selected_policy": selected_policy,
        }
        print(
            f"[temporal-v6] {robot} selected {asdict(selected_policy)}",
            flush=True,
        )

    _write_json(staging / "trajectory_split_manifest.json", split_manifest)
    _write_json(staging / "calibration_candidate_metrics.json", calibration_reports)
    selected_payload = {
        robot: {
            "candidate_index": int(contexts[robot]["selected_index"]),
            **asdict(contexts[robot]["selected_policy"]),
        }
        for robot in ("panda", "ur5e")
    }
    _write_json(
        staging / "calibration_selection.json",
        {
            "selection_role": "trajectory_calibration",
            "policy_validation_outcomes_computed": False,
            "policy_validation_used_for_selection": False,
            "selected": selected_payload,
        },
    )
    calibration_seal_payload = {
        "status": "sealed_before_policy_validation",
        "sealed_at": _utc(),
        "protocol": PROTOCOL,
        "git_commit": git_commit,
        "implementation_sources": implementation_sources,
        "selected_policy": selected_payload,
        "release_inputs": release_inputs,
        "sealed_reports": {
            "split_manifest": _artifact(
                staging / "trajectory_split_manifest.json", relative_to=staging
            ),
            "calibration_selection": _artifact(
                staging / "calibration_selection.json", relative_to=staging
            ),
            "candidate_metrics": _artifact(
                staging / "calibration_candidate_metrics.json", relative_to=staging
            ),
        },
        "sealed_development_artifacts": {
            robot: {
                "trajectory_calibration": _artifact(
                    staging / f"{robot}_trajectory_calibration.npz",
                    relative_to=staging,
                ),
                "trajectory_policy_validation": _artifact(
                    staging / f"{robot}_trajectory_policy_validation.npz",
                    relative_to=staging,
                ),
                "calibration_records": _artifact(
                    staging / f"{robot}_calibration_records.npz",
                    relative_to=staging,
                ),
                "frozen_v5_lite": frozen_lite_inputs[robot],
            }
            for robot in ("panda", "ur5e")
        },
        "formal_test_data_loaded": False,
    }
    _write_json(staging / "calibration_seal.json", calibration_seal_payload)
    seal_sha256 = _sha256_file(staging / "calibration_seal.json")

    # Phase B: the frozen policy is now used exactly once on policy-validation.
    benchmarks: dict[str, BenchmarkData] = {}
    for robot in ("panda", "ur5e"):
        sealed = _verify_calibration_seal(
            staging / "calibration_seal.json",
            expected_sha256=seal_sha256,
            staging=staging,
            workspace=workspace,
        )
        context = contexts[robot]
        kinematics = context["kinematics"]
        policy_role = _load_role(
            staging / f"{robot}_trajectory_policy_validation.npz",
            robot=robot,
            expected_role="trajectory_policy_validation",
        )
        selected = sealed["selected_policy"][robot]
        selected_policy = TemporalPolicyConfig(
            hold_frames=int(selected["hold_frames"]),
            probe_interval=int(selected["probe_interval"]),
            consecutive_successes=int(selected["consecutive_successes"]),
            reentry_threshold=float(selected["reentry_threshold"]),
        )
        methods = _build_policy_validation_methods(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            frozen_lite_root=frozen_lite_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
            temporal_config=selected_policy,
        )
        _warm_methods(
            methods,
            context["calibration_role"],
            frames=int(config["timing"]["warmup_frames"]),
            dt=dt,
        )
        benchmark = benchmark_policy_validation(
            policy_role,
            methods=methods,
            dt=dt,
            order_seed=886_000 + int(config["trajectory_data"]["split_seed"][robot]),
            progress_every=int(config["runtime"]["progress_every_trajectories"]),
            environment_guard=host_guard,
        )
        benchmarks[robot] = benchmark
        _save_benchmark(staging / f"{robot}_policy_validation_records.npz", benchmark)

    _verify_calibration_seal(
        staging / "calibration_seal.json",
        expected_sha256=seal_sha256,
        staging=staging,
        workspace=workspace,
    )
    quiet_events["trajectory_start_preflight"] = host_guard.total_summary()
    host_guard.close()

    main_rows = [
        row for robot in ("panda", "ur5e") for row in summarize_benchmark(benchmarks[robot])
    ]
    family_rows = [
        row for robot in ("panda", "ur5e") for row in family_mode_distribution(benchmarks[robot])
    ]
    paired_rows = [
        row for robot in ("panda", "ur5e") for row in paired_latency(benchmarks[robot])
    ]
    stage_rows = [
        row
        for robot in ("panda", "ur5e")
        for row in stage_latency_summary(benchmarks[robot])
    ]
    _write_json(staging / "main_table.json", main_rows)
    _write_csv(staging / "main_table.csv", main_rows)
    (staging / "main_table.md").write_text(_main_markdown(main_rows), encoding="utf-8")
    _write_json(staging / "query_family_mode_distribution.json", family_rows)
    _write_csv(staging / "query_family_mode_distribution.csv", family_rows)
    _write_json(staging / "paired_latency_summary.json", paired_rows)
    _write_json(staging / "latency_breakdown.json", stage_rows)
    _write_csv(staging / "latency_breakdown.csv", stage_rows)
    gate = _pilot_gate(main_rows, benchmarks)
    _write_json(staging / "pilot_gate.json", gate)
    representative = _plot_results(
        main_rows,
        family_rows,
        benchmarks,
        staging,
        dpi=int(config["reporting"]["png_dpi"]),
    )
    _write_json(staging / "representative_trajectory.json", representative)
    _write_json(
        staging / "environment.json",
        {
            **environment_payload(),
            "quiet_events": quiet_events,
            "test_data_loaded": False,
            "formal_test_started": False,
        },
    )
    effective_config = {
        key: value for key, value in config.items() if not str(key).startswith("_")
    }
    (staging / "temporal_v6_pilot.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
    )

    protected_after = {
        name: _tree_snapshot(path) for name, path in protected_roots.items()
    }
    protected_after_digest = {
        name: _snapshot_digest(snapshot)
        for name, snapshot in protected_after.items()
    }
    if protected_before != protected_after:
        raise RuntimeError("an existing V5 output tree changed during Temporal V6")
    git_status_end = subprocess.run(
        ["git", "status", "--short"], cwd=workspace, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    if git_status_end != git_status_start:
        raise RuntimeError("source worktree changed during Temporal V6")
    git_commit_end = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if git_commit_end != git_commit:
        raise RuntimeError("git HEAD changed during Temporal V6")
    for relative, descriptor in implementation_sources.items():
        path = workspace / relative
        if _artifact(path, relative_to=workspace) != descriptor:
            raise RuntimeError(f"implementation source changed during run: {relative}")

    artifacts = {
        path.name: _artifact(path, relative_to=staging)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "status": "complete_smoke" if smoke else "complete_policy_validation_pilot",
        "protocol": PROTOCOL,
        "started_at": started,
        "completed_at": _utc(),
        "git_commit": git_commit,
        "git_commit_end": git_commit_end,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "source_state_unchanged_during_run": True,
        "implementation_sources": implementation_sources,
        "release_inputs": release_inputs,
        "protocol_config": _artifact(
            Path(config["_config_path"]), relative_to=workspace
        ),
        "source_config": _artifact(source_config_path, relative_to=workspace),
        "effective_config": _artifact(
            staging / "temporal_v6_pilot.yaml", relative_to=staging
        ),
        "smoke_overrides": (
            {
                "paths_per_family_pool": 2,
                "paths_per_family_per_role": 1,
                "steps_per_trajectory": 24,
                "warmup_frames": 4,
                "progress_every_trajectories": 0,
            }
            if smoke
            else {}
        ),
        "frozen_v5_lite_root": str(frozen_lite_root),
        "protected_output_trees": {
            name: {
                "root": str(protected_roots[name]),
                "before_digest": protected_before_digest[name],
                "after_digest": protected_after_digest[name],
                "unchanged": True,
            }
            for name in protected_roots
        },
        "trajectory_roles": split_manifest,
        "calibration_seal_sha256": seal_sha256,
        "selected_policy": selected_payload,
        "policy_frozen_before_policy_validation": True,
        "policy_validation_used_for_selection": False,
        "policy_validation_used_for_retuning": False,
        "policy_validation_retained_passes_per_robot": {
            "panda": 1,
            "ur5e": 1,
        },
        "trajectory_environment_evidence": quiet_events,
        "method_order_position_counts": {
            robot: {
                method: np.bincount(
                    benchmarks[robot].method_order_index[:, column],
                    minlength=len(METHODS),
                ).astype(int).tolist()
                for column, method in enumerate(METHODS)
            }
            for robot in ("panda", "ur5e")
        },
        "test_data_loaded": False,
        "formal_test_started": False,
        "fresh_evaluation_started": False,
        "pilot_gate": gate,
        "artifacts": artifacts,
    }
    _write_json(staging / "run_manifest.json", manifest)
    os.replace(staging, output_root)
    print(
        f"[temporal-v6] complete: output={output_root} gate={gate['status']}",
        flush=True,
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    run(args.config, smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
