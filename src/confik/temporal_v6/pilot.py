"""Development-only pilot for pure-H Temporal Event-Triggered CG-HIK.

The module has no formal-test loader.  It generates a fresh trajectory pool,
selects the sole parameter H on complete calibration trajectories, seals that
selection, and evaluates policy validation exactly once.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..counterfactual_v4.runner import _build_runtimes
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
from ..hierarchical_v5_lite.pilot import (
    _build_current_v4,
    _fresh_fixed_hard,
    _fresh_shared_fixed_hard,
    _snapshot_digest,
    _tree_snapshot,
)
from ..latency_pilot_v3.benchmark import query_digest
from ..types import IKQuery, Pose
from .runtime import TemporalCGHIKRuntime, TemporalOutcome
from .state import TemporalMode, TemporalPolicyConfig, TemporalState
from .trajectories import (
    AllowedSeedRegistry,
    AllowedSourceHashRegistry,
    FAMILIES,
    PROTOCOL as TRAJECTORY_PROTOCOL,
    FreshTrajectoryRole,
    FreshTrajectorySpec,
    generate_fresh_development_roles,
    load_trajectory_role,
    save_split_audit_manifest,
    save_trajectory_role,
    runtime_query_hashes,
    source_query_hashes,
)


PROTOCOL = "temporal_event_v6_development_pilot_v1"
H_VALUES = (2, 5, 10, 20, 30)
METHODS = (
    "always_hard",
    "fixed_easy_cascade",
    "counterfactual_cghik_v4",
    "temporal_event_cghik",
)
CALIBRATION_ROLE = "trajectory_calibration"
POLICY_VALIDATION_ROLE = "trajectory_policy_validation"
MODE_CODE = {TemporalMode.INIT: 0, TemporalMode.LOCAL: 1, TemporalMode.ROBUST: 2}
CODE_MODE = {0: "init", 1: "local", 2: "robust"}
CALIBRATION_ARMS = ("always_hard",) + tuple(f"h{value}" for value in H_VALUES)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _forbid_formal_path(value: str | Path, *, name: str) -> None:
    text = str(value).casefold()
    if "test_v3" in text or "test_v4" in text:
        raise ValueError(f"{name} must not reference test_v3/test_v4: {value}")


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_safe_json(row) for row in rows)


def _tree_descriptors(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): _artifact(path, relative_to=root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def validate_config(config: Mapping[str, Any], *, workspace: Path) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected Temporal Event V6 protocol")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("Temporal Event V6 requires Panda and UR5e")
    if int(config.get("release_seed", -1)) != 17:
        raise ValueError("release_seed must remain 17")
    roles = config.get("roles", {})
    if (roles.get("calibration"), roles.get("policy_validation")) != (
        CALIBRATION_ROLE,
        POLICY_VALIDATION_ROLE,
    ):
        raise ValueError("development roles changed")
    boundary = config.get("data_boundary", {})
    if tuple(boundary.get("allowed_roles", ())) != (
        CALIBRATION_ROLE,
        POLICY_VALIDATION_ROLE,
    ) or not all(
        boundary.get(key) is True
        for key in (
            "formal_test_data_forbidden",
            "reject_test_v3_v4_paths",
            "split_before_outcome_collection",
            "policy_validation_after_calibration_seal_only",
        )
    ):
        raise ValueError("development data boundary changed")
    data = config.get("trajectory_data", {})
    if (
        data.get("generator") != "fresh_temporal_event_trajectories"
        or tuple(data.get("families", ())) != FAMILIES
        or int(data.get("paths_per_family_pool", -1)) != 20
        or int(data.get("paths_per_family_per_role", -1)) != 10
        or int(data.get("steps_per_trajectory", -1)) != 150
        or float(data.get("dt", np.nan)) != 0.02
        or data.get("split_unit") != "complete_trajectory"
        or data.get("preserve_time_order") is not True
        or data.get("require_seed_isolation") is not True
        or data.get("require_query_hash_isolation") is not True
        or dict(data.get("pool_seed", {})) != {"panda": 861601, "ur5e": 861602}
        or dict(data.get("split_seed", {})) != {"panda": 861611, "ur5e": 861612}
    ):
        raise ValueError("fresh 40+40 by 150 trajectory contract changed")
    temporal_policy = config.get("temporal_policy", {})
    if set(temporal_policy) != {"hold_frames"} or tuple(
        temporal_policy.get("hold_frames", ())
    ) != H_VALUES:
        raise ValueError(f"H grid must remain exactly {H_VALUES}")
    selection = config.get("calibration_selection", {})
    if (
        selection.get("required_role") != CALIBRATION_ROLE
        or selection.get("eligibility")
        != "whole_trajectory_completion_count_equal_always_hard"
        or tuple(selection.get("objective_order", ()))
        != (
            "minimum_p95_end_to_end_latency",
            "minimum_learned_seed_invocation_rate",
            "minimum_p50_end_to_end_latency",
        )
        or selection.get("exact_tie_break") != "higher_hold_frames"
        or selection.get("policy_validation_used_for_selection") is not False
    ):
        raise ValueError("calibration selection contract changed")
    if tuple(config.get("strategies", ())) != METHODS:
        raise ValueError(f"strategies must be exactly {METHODS}")
    timing = config.get("timing", {})
    if (
        timing.get("clock") != "perf_counter_ns"
        or int(timing.get("trajectory_repeats", -1)) != 1
        or timing.get("method_order") != "seeded_four_by_four_latin_blocks"
        or timing.get("disk_io_inside_timed_interval") is not False
        or timing.get("logging_serialization_inside_timed_interval") is not False
    ):
        raise ValueError("timing contract changed")
    runtime = config.get("runtime", {})
    if set(runtime) != {
        "device",
        "intra_op_threads",
        "inter_op_threads",
        "deterministic_algorithms",
        "progress_every_trajectories",
    }:
        raise ValueError("runtime protocol contains undeclared controls")
    goals = config.get("pilot_goals", {})
    if (
        goals.get("whole_trajectory_completion_equal_always_hard") is not True
        or float(goals.get("p95_ratio_vs_always_hard_max", np.nan)) != 1.0
        or float(goals.get("p50_ratio_vs_always_hard_max_exclusive", np.nan)) != 1.0
        or float(goals.get("learned_seed_invocation_rate_max", np.nan)) != 0.60
    ):
        raise ValueError("pilot goals changed")
    reporting = config.get("reporting", {})
    if (
        tuple(reporting.get("latency_quantiles", ())) != (0.50, 0.95, 0.99)
        or tuple(reporting.get("run_length_quantiles", ()))
        != (0.25, 0.50, 0.75, 0.95)
        or reporting.get("representative_scope") != "per_robot"
        or reporting.get("representative_selection")
        != "max_mode_switch_then_robust_occupancy_then_uid"
        or tuple(reporting.get("figure_formats", ())) != ("png", "pdf")
        or int(reporting.get("png_dpi", -1)) != 220
    ):
        raise ValueError("reporting contract changed")
    for key in ("source_config", "release_v3_root", "release_v4_root", "output_root"):
        _forbid_formal_path(config.get(key, ""), name=key)
    expected = (workspace / "outputs" / "temporal_event_v6_pilot").resolve()
    if resolve_path(dict(config), str(config["output_root"])) != expected:
        raise ValueError(f"output_root must resolve to {expected}")


@dataclass(frozen=True)
class TrajectoryView:
    robot: str
    role: str
    dataset: QueryDataset
    source_query_hash: np.ndarray
    trajectory_uid: np.ndarray
    trajectory_order: tuple[str, ...]
    dt: float

    @property
    def count(self) -> int:
        return len(self.dataset)

    def groups(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple(
            (uid, np.flatnonzero(self.trajectory_uid == uid).astype(np.int64))
            for uid in self.trajectory_order
        )

    def query(self, index: int, previous_q: np.ndarray | None = None) -> IKQuery:
        return IKQuery(
            Pose(self.dataset.target_position[index], self.dataset.target_rotation[index]),
            self.dataset.previous_q[index] if previous_q is None else previous_q,
            dt=self.dt,
        )


def _subset_dataset(dataset: QueryDataset, indices: np.ndarray) -> QueryDataset:
    return QueryDataset(
        previous_q=dataset.previous_q[indices],
        target_position=dataset.target_position[indices],
        target_rotation=dataset.target_rotation[indices],
        reference_q=dataset.reference_q[indices],
        category=dataset.category[indices],
        expected_reachable=dataset.expected_reachable[indices],
        continuity_feasible=dataset.continuity_feasible[indices],
        trajectory_id=dataset.trajectory_id[indices],
        time_index=dataset.time_index[indices],
    )


def _view(role: FreshTrajectoryRole, *, smoke: bool) -> TrajectoryView:
    if not smoke:
        indices = np.arange(role.count, dtype=np.int64)
        order = role.trajectory_order
    else:
        selected_uids: list[str] = []
        for family in FAMILIES:
            for uid in role.trajectory_order:
                rows = np.flatnonzero(role.trajectory_uid == uid)
                if str(role.dataset.category[int(rows[0])]) == family:
                    selected_uids.append(uid)
                    break
        indices = np.concatenate(
            [np.flatnonzero(role.trajectory_uid == uid) for uid in selected_uids]
        ).astype(np.int64)
        order = tuple(selected_uids)
    return TrajectoryView(
        role.robot,
        role.role,
        _subset_dataset(role.dataset, indices),
        role.source_query_hash[indices].copy(),
        role.trajectory_uid[indices].copy(),
        order,
        role.dt,
    )


def _kinematics_identity(source_config: Mapping[str, Any], robot: str) -> tuple[str, Path]:
    config = dict(source_config)
    path = resolve_path(config, str(config["robots"][robot]["urdf"]))
    return _sha256_file(path), path


def _load_prior_query_hashes(
    path: Path,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> tuple[str, ...]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    # TransitionDataset and QueryDataset artifacts share exactly these three
    # runtime inputs.  Labels, references, outcomes and role metadata are not
    # read and cannot influence the identity audit.
    _forbid_formal_path(path, name="prior development source")
    required = {"previous_q", "target_position", "target_rotation"}
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(
                f"prior source is missing runtime query fields: {sorted(missing)}"
            )
        hashes = runtime_query_hashes(
            payload["previous_q"],
            payload["target_position"],
            payload["target_rotation"],
            robot=robot,
            dt=dt,
            kinematics_identity=kinematics_identity,
        )
    return tuple(hashes.astype(str).tolist())


def _prior_evaluation_trajectory_seeds(
    workspace: Path, robot: str
) -> tuple[int, ...]:
    """Derive prior formal trajectory seeds without opening formal data/results."""

    v3_manifest_path = workspace / "outputs" / "release_v3_locked" / "release_manifest.json"
    v4_manifest_path = workspace / "outputs" / "release_v4_locked" / "release_manifest.json"
    v3_manifest = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    v4_manifest = json.loads(v4_manifest_path.read_text(encoding="utf-8"))
    v3_release = str(v3_manifest["git_commit"])
    v4_release = str(v4_manifest["release_digest"])

    def derive(material: str) -> int:
        return int.from_bytes(sha256(material.encode("utf-8")).digest()[:4], "big")

    return (
        derive(f"test_v3_locked|{v3_release}|{robot}|trajectories"),
        derive(f"test_v4_locked|{v4_release}|{robot}|id_trajectories"),
        derive(f"test_v4_locked|{v4_release}|{robot}|ood_trajectories"),
    )


def _prior_registries(
    *, workspace: Path, robot: str, kinematics: object, identity: str, dt: float
) -> tuple[
    tuple[AllowedSourceHashRegistry, ...],
    tuple[AllowedSeedRegistry, ...],
    dict[str, Any],
]:
    """Build explicit, non-formal row-hash registries; never discover files."""

    input_files: dict[str, Any] = {}
    for release_manifest in (
        workspace / "outputs" / "release_v3_locked" / "release_manifest.json",
        workspace / "outputs" / "release_v4_locked" / "release_manifest.json",
    ):
        input_files[str(release_manifest.relative_to(workspace))] = _artifact(
            release_manifest, relative_to=workspace
        )

    def collect(paths: Sequence[Path]) -> tuple[str, ...]:
        hashes: set[str] = set()
        for path in paths:
            hashes.update(
                _load_prior_query_hashes(
                    path,
                    robot=robot,
                    dt=dt,
                    kinematics_identity=identity,
                )
            )
            input_files[str(path.relative_to(workspace))] = _artifact(
                path, relative_to=workspace
            )
        return tuple(sorted(hashes))

    training_paths = tuple(
        workspace
        / "outputs"
        / f"paper_v2_seed{seed}"
        / robot
        / "datasets"
        / filename
        for seed in (17, 29, 43)
        for filename in ("seed_train.npz", "risk_train_queries.npz")
    )
    development_paths = tuple(
        workspace
        / "outputs"
        / f"paper_v2_seed{seed}"
        / robot
        / "datasets"
        / filename
        for seed in (17, 29, 43)
        for filename in (
            "seed_validation.npz",
            "risk_validation_queries.npz",
            "calibration_queries.npz",
            "policy_validation_queries.npz",
        )
    )
    old_temporal_paths = (
        workspace / "outputs" / "temporal_v6_pilot" / f"{robot}_trajectory_calibration.npz",
        workspace
        / "outputs"
        / "temporal_v6_pilot"
        / f"{robot}_trajectory_policy_validation.npz",
    )
    old_temporal_hashes = set(collect(old_temporal_paths))
    development_hashes = set(collect(development_paths)) | old_temporal_hashes
    reference_hashes: set[str] = set()
    regenerated_reference = generate_reference_trajectory_tests(
        kinematics,  # type: ignore[arg-type]
        paths_per_type=10,
        steps=150,
        seed=9020,
        dt=dt,
    )
    reference_hashes.update(
        source_query_hashes(
            regenerated_reference,
            robot=robot,
            dt=dt,
            kinematics_identity=identity,
        ).astype(str).tolist()
    )
    sources = (
        AllowedSourceHashRegistry(
            f"{robot}_paper_v2_training_rows", "training", collect(training_paths)
        ),
        AllowedSourceHashRegistry(
            f"{robot}_paper_v2_development_rows",
            "development",
            tuple(sorted(development_hashes)),
        ),
        AllowedSourceHashRegistry(
            f"{robot}_prior_reference_rows",
            "reference",
            tuple(sorted(reference_hashes)),
        ),
    )
    seeds = (
        AllowedSeedRegistry(
            f"{robot}_prior_training_seeds",
            "training",
            pool_seeds=(101, 201),
            split_seeds=(17, 29, 43),
        ),
        AllowedSeedRegistry(
            f"{robot}_prior_development_seeds",
            "development",
            pool_seeds=(102, 202, 203, 204, 860601, 860602),
            split_seeds=(7318, 9418, 860611, 860612),
        ),
        AllowedSeedRegistry(
            f"{robot}_prior_reference_seeds",
            "reference",
            pool_seeds=(9020,),
        ),
        AllowedSeedRegistry(
            f"{robot}_prior_evaluation_seed_lineage",
            "evaluation",
            pool_seeds=(
                205,
                301,
                9001,
                9020,
                *_prior_evaluation_trajectory_seeds(workspace, robot),
            ),
        ),
    )
    return sources, seeds, {
        "input_files": input_files,
        "regenerated_reference": {
            "generator": "generate_reference_trajectory_tests",
            "seed": 9020,
            "paths_per_family": 10,
            "steps": 150,
            "query_hash_count": len(reference_hashes),
        },
        "test_v3_test_v4_files_opened": 0,
        "formal_evaluation_query_rows_opened": 0,
        "formal_evaluation_query_hash_comparison": (
            "not performed because the formal-data no-read boundary takes precedence; "
            "formal trajectory generator seeds are checked from locked release lineage"
        ),
    }


def _fresh_fixed_easy(
    *, source_config: dict[str, Any], release_root: Path, robot: str,
    kinematics: object, device: str,
) -> object:
    _, runtimes, _ = _build_runtimes(
        source_config=source_config,
        release_root=release_root,
        robot=robot,
        training_seed=17,
        kinematics=kinematics,
        device=device,
    )
    runtime = runtimes["easy"]
    runtime.name = "fixed_easy_cascade"
    return runtime


def _fresh_temporal(
    *, source_config: dict[str, Any], release_root: Path, robot: str,
    kinematics: object, device: str, hold_frames: int,
) -> TemporalCGHIKRuntime:
    hard, dls, verifier = _fresh_shared_fixed_hard(
        source_config=source_config,
        release_root=release_root,
        robot=robot,
        kinematics=kinematics,
        device=device,
    )
    return TemporalCGHIKRuntime(
        kinematics=kinematics,
        dls=dls,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        always_hard_runtime=hard,  # type: ignore[arg-type]
        policy_config=TemporalPolicyConfig(hold_frames),
        local_iterations=1,
        name="temporal_event_cghik",
    )


def _warm_temporal(runtime: TemporalCGHIKRuntime, role: TrajectoryView, frames: int) -> None:
    if frames <= 0:
        return
    state = runtime.initial_state()
    previous = role.dataset.previous_q[0].copy()
    for warm_index in range(frames):
        index = warm_index % role.count
        if int(role.dataset.time_index[index]) == 0:
            state = runtime.initial_state()
            previous = role.dataset.previous_q[index].copy()
        outcome = runtime.step(role.query(index, previous), state)
        state = outcome.state_after
        if outcome.accepted and outcome.q is not None:
            previous = np.asarray(outcome.q, dtype=np.float64).copy()


@dataclass(frozen=True)
class CalibrationData:
    arm_names: tuple[str, ...]
    hold_frames: np.ndarray
    latency_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    hard_attempted: np.ndarray
    hard_accepted: np.ndarray
    state_before: np.ndarray
    state_after: np.ndarray
    hard_count_before: np.ndarray
    hard_count_after: np.ndarray
    mode_switched: np.ndarray
    local_probe: np.ndarray
    route: np.ndarray
    command_q: np.ndarray
    executed_query_hash: np.ndarray
    method_order_position: np.ndarray


def collect_calibration(
    role: TrajectoryView,
    *, hard_runtime: object,
    temporal_runtimes: Mapping[int, TemporalCGHIKRuntime],
    order_seed: int,
    warmup_frames: int,
    progress_every: int,
) -> CalibrationData:
    if tuple(temporal_runtimes) != H_VALUES:
        raise ValueError("calibration must contain the exact H grid")
    for runtime in temporal_runtimes.values():
        _warm_temporal(runtime, role, warmup_frames)
    for index in range(warmup_frames):
        hard_runtime.solve(role.query(index % role.count))  # type: ignore[attr-defined]
    arm_index = {name: index for index, name in enumerate(CALIBRATION_ARMS)}
    count, arms, nq = role.count, len(CALIBRATION_ARMS), role.dataset.previous_q.shape[1]
    latency = np.zeros((arms, count), dtype=np.int64)
    accepted = np.zeros((arms, count), dtype=bool)
    fev = np.zeros((arms, count), dtype=np.int64)
    seed = np.zeros((arms, count), dtype=bool)
    local_attempted = np.zeros((arms, count), dtype=bool)
    local_accepted = np.zeros((arms, count), dtype=bool)
    hard_attempted = np.zeros((arms, count), dtype=bool)
    hard_accepted = np.zeros((arms, count), dtype=bool)
    state_before = np.full((arms, count), -1, dtype=np.int8)
    state_after = np.full((arms, count), -1, dtype=np.int8)
    hard_before = np.full((arms, count), -1, dtype=np.int16)
    hard_after = np.full((arms, count), -1, dtype=np.int16)
    switched = np.zeros((arms, count), dtype=bool)
    probe = np.zeros((arms, count), dtype=bool)
    route = np.full((arms, count), "", dtype="U64")
    command = np.full((arms, count, nq), np.nan, dtype=np.float64)
    query_hash = np.full((arms, count), "", dtype="U64")
    order_position = np.full((arms, count), -1, dtype=np.int8)
    latin = latin_method_orders(CALIBRATION_ARMS, order_seed)

    for trajectory_number, (uid, indices) in enumerate(role.groups()):
        previous = {name: role.dataset.previous_q[int(indices[0])].copy() for name in CALIBRATION_ARMS}
        states = {h: temporal_runtimes[h].initial_state() for h in H_VALUES}
        for offset, raw_index in enumerate(indices):
            index = int(raw_index)
            order = latin[(trajectory_number + offset) % len(latin)]
            if set(order) != set(CALIBRATION_ARMS):
                raise RuntimeError("calibration order is not a complete permutation")
            for position, name in enumerate(order):
                arm = arm_index[name]
                query = role.query(index, previous[name])
                started = perf_counter_ns()
                if name == "always_hard":
                    outcome = hard_runtime.solve(query)  # type: ignore[attr-defined]
                else:
                    h = int(name[1:])
                    outcome = temporal_runtimes[h].step(query, states[h])
                    states[h] = outcome.state_after
                elapsed = perf_counter_ns() - started
                latency[arm, index] = elapsed
                accepted[arm, index] = bool(outcome.accepted)
                fev[arm, index] = int(outcome.function_evaluations)
                order_position[arm, index] = position
                query_hash[arm, index] = query_digest(query)
                route[arm, index] = str(getattr(outcome, "route", getattr(outcome, "entry_action", "")))
                if name == "always_hard":
                    seed[arm, index] = True
                    hard_attempted[arm, index] = True
                    hard_accepted[arm, index] = bool(outcome.accepted)
                else:
                    seed[arm, index] = bool(outcome.learned_seed_ensemble_invoked)
                    local_attempted[arm, index] = bool(outcome.local_attempted)
                    local_accepted[arm, index] = bool(outcome.local_accepted)
                    hard_attempted[arm, index] = bool(outcome.hard_attempted)
                    hard_accepted[arm, index] = bool(outcome.hard_accepted)
                    state_before[arm, index] = MODE_CODE[outcome.state_before.mode]
                    state_after[arm, index] = MODE_CODE[outcome.state_after.mode]
                    hard_before[arm, index] = int(outcome.hard_calls_since_local_attempt_before)
                    hard_after[arm, index] = int(outcome.hard_calls_since_local_attempt_after)
                    switched[arm, index] = bool(outcome.mode_switched)
                    probe[arm, index] = bool(outcome.local_probe_executed)
                if outcome.accepted and outcome.q is not None:
                    value = np.asarray(outcome.q, dtype=np.float64)
                    command[arm, index] = value
                    previous[name] = value.copy()
        if progress_every and (trajectory_number + 1) % progress_every == 0:
            print(
                f"[temporal-event-v6] {role.robot} calibration "
                f"{trajectory_number + 1}/{len(role.trajectory_order)}",
                flush=True,
            )
    if np.any(latency <= 0) or np.any(order_position < 0) or np.any(query_hash == ""):
        raise RuntimeError("calibration collection is incomplete")
    return CalibrationData(
        CALIBRATION_ARMS,
        np.asarray((-1,) + H_VALUES, dtype=np.int16),
        latency,
        accepted,
        fev,
        seed,
        local_attempted,
        local_accepted,
        hard_attempted,
        hard_accepted,
        state_before,
        state_after,
        hard_before,
        hard_after,
        switched,
        probe,
        route,
        command,
        query_hash,
        order_position,
    )


def _trajectory_completion(accepted: np.ndarray, role: TrajectoryView) -> np.ndarray:
    values = np.asarray(accepted, dtype=bool)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != role.count:
        raise ValueError("accepted data does not match the trajectory role")
    return np.asarray(
        [
            [bool(np.all(values[row, indices])) for _, indices in role.groups()]
            for row in range(values.shape[0])
        ],
        dtype=bool,
    )


def select_hold_frames(
    data: CalibrationData, role: TrajectoryView
) -> tuple[int, dict[str, Any]]:
    """Apply the frozen scalar-completion, P95, seed, P50 ordering."""

    completion = _trajectory_completion(data.accepted, role)
    hard_vector = completion[0]
    hard_count = int(np.sum(hard_vector))
    rows: list[dict[str, Any]] = []
    eligible_indices: list[int] = []
    for arm, h in enumerate(H_VALUES, start=1):
        vector = completion[arm]
        count = int(np.sum(vector))
        gained = [
            uid
            for uid, hard_ok, candidate_ok in zip(
                role.trajectory_order, hard_vector, vector, strict=True
            )
            if candidate_ok and not hard_ok
        ]
        lost = [
            uid
            for uid, hard_ok, candidate_ok in zip(
                role.trajectory_order, hard_vector, vector, strict=True
            )
            if hard_ok and not candidate_ok
        ]
        eligible = count == hard_count
        if eligible:
            eligible_indices.append(arm)
        rows.append(
            {
                "candidate_index": arm - 1,
                "hold_frames": h,
                "eligible": eligible,
                "whole_trajectory_completion_count": count,
                "whole_trajectory_completion_rate": count / len(role.trajectory_order),
                "always_hard_completion_count": hard_count,
                "completion_count_difference": count - hard_count,
                "completion_vector_hamming_count": int(np.sum(vector != hard_vector)),
                "gained_trajectory_uids": gained,
                "lost_trajectory_uids": lost,
                "p50_latency_ms": float(np.quantile(data.latency_ns[arm], 0.50) / 1e6),
                "p95_latency_ms": float(np.quantile(data.latency_ns[arm], 0.95) / 1e6),
                "p99_latency_ms": float(np.quantile(data.latency_ns[arm], 0.99) / 1e6),
                "learned_seed_invocation_rate": float(np.mean(data.seed_invoked[arm])),
                "frame_verified_success": float(np.mean(data.accepted[arm])),
                "mean_fev": float(np.mean(data.function_evaluations[arm])),
            }
        )
    hard_summary = {
        "whole_trajectory_completion_count": hard_count,
        "whole_trajectory_completion_rate": hard_count / len(role.trajectory_order),
        "p50_latency_ms": float(np.quantile(data.latency_ns[0], 0.50) / 1e6),
        "p95_latency_ms": float(np.quantile(data.latency_ns[0], 0.95) / 1e6),
        "p99_latency_ms": float(np.quantile(data.latency_ns[0], 0.99) / 1e6),
        "frame_verified_success": float(np.mean(data.accepted[0])),
        "mean_fev": float(np.mean(data.function_evaluations[0])),
        "learned_seed_invocation_rate": 1.0,
    }
    if not eligible_indices:
        raise RuntimeError(
            f"{role.robot}: no H preserves the scalar whole-trajectory completion count"
        )
    selected_arm = min(
        eligible_indices,
        key=lambda arm: (
            float(np.quantile(data.latency_ns[arm], 0.95)),
            float(np.mean(data.seed_invoked[arm])),
            float(np.quantile(data.latency_ns[arm], 0.50)),
            -int(data.hold_frames[arm]),
        ),
    )
    selected_h = int(data.hold_frames[selected_arm])
    selected = dict(rows[selected_arm - 1])
    selected["arm_index"] = selected_arm
    return selected_h, {
        "robot": role.robot,
        "selection_role": role.role,
        "policy_validation_used_for_selection": False,
        "completion_eligibility_definition": (
            "integer count of complete trajectories exactly equals always-hard; "
            "trajectory identity swaps are diagnostic only"
        ),
        "objective_order": ["p95_latency", "seed_invocation", "p50_latency"],
        "exact_tie_break": "higher_hold_frames",
        "always_hard": hard_summary,
        "candidate_metrics": rows,
        "eligible_candidate_count": len(eligible_indices),
        "selected": selected,
    }


def _save_calibration(path: Path, data: CalibrationData, role: TrajectoryView) -> None:
    np.savez_compressed(
        path,
        arm_names=np.asarray(data.arm_names, dtype="U32"),
        hold_frames=data.hold_frames,
        latency_ns=data.latency_ns,
        accepted=data.accepted,
        function_evaluations=data.function_evaluations,
        seed_invoked=data.seed_invoked,
        local_attempted=data.local_attempted,
        local_accepted=data.local_accepted,
        hard_attempted=data.hard_attempted,
        hard_accepted=data.hard_accepted,
        state_before=data.state_before,
        state_after=data.state_after,
        hard_count_before=data.hard_count_before,
        hard_count_after=data.hard_count_after,
        mode_switched=data.mode_switched,
        local_probe=data.local_probe,
        route=data.route,
        command_q=data.command_q,
        executed_query_hash=data.executed_query_hash,
        method_order_position=data.method_order_position,
        source_query_hash=role.source_query_hash,
        trajectory_uid=role.trajectory_uid,
        trajectory_order=np.asarray(role.trajectory_order, dtype="U64"),
        category=role.dataset.category,
        time_index=role.dataset.time_index,
    )


@dataclass(frozen=True)
class BenchmarkData:
    robot: str
    role: TrajectoryView
    latency_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    hard_attempted: np.ndarray
    hard_accepted: np.ndarray
    same_frame_hard_recovery: np.ndarray
    occupancy_mode: np.ndarray
    state_before: np.ndarray
    state_after: np.ndarray
    mode_switched: np.ndarray
    local_probe: np.ndarray
    hard_count_before: np.ndarray
    hard_count_after: np.ndarray
    recovery_delay: np.ndarray
    route: np.ndarray
    executed_stages: np.ndarray
    command_q: np.ndarray
    executed_query_hash: np.ndarray
    method_order_position: np.ndarray
    stage_latency_ns: np.ndarray


def _build_policy_validation_methods(
    *, source_config: dict[str, Any], release_v3_root: Path,
    release_v4_root: Path, robot: str, kinematics: object,
    device: str, hold_frames: int,
) -> dict[str, object]:
    methods = {
        "always_hard": _fresh_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        ),
        "fixed_easy_cascade": _fresh_fixed_easy(
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
        "temporal_event_cghik": _fresh_temporal(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
            hold_frames=hold_frames,
        ),
    }
    if tuple(methods) != METHODS:
        raise RuntimeError("policy-validation method registry changed")
    return methods


def _warm_methods(
    methods: Mapping[str, object], role: TrajectoryView, *, frames: int
) -> None:
    previous = {name: role.dataset.previous_q[0].copy() for name in METHODS}
    temporal = methods["temporal_event_cghik"]
    state = temporal.initial_state()  # type: ignore[attr-defined]
    for warm_index in range(frames):
        index = warm_index % role.count
        if int(role.dataset.time_index[index]) == 0:
            previous = {name: role.dataset.previous_q[index].copy() for name in METHODS}
            state = temporal.initial_state()  # type: ignore[attr-defined]
        for name in METHODS:
            query = role.query(index, previous[name])
            if name == "temporal_event_cghik":
                outcome = temporal.step(query, state)  # type: ignore[attr-defined]
                state = outcome.state_after
            else:
                outcome = methods[name].solve(query)  # type: ignore[attr-defined]
            if outcome.accepted and outcome.q is not None:
                previous[name] = np.asarray(outcome.q, dtype=np.float64).copy()


def benchmark_policy_validation(
    role: TrajectoryView,
    *, methods: Mapping[str, object], warmup_role: TrajectoryView, order_seed: int,
    warmup_frames: int, progress_every: int,
) -> BenchmarkData:
    if warmup_frames > 0:
        overlap = set(role.source_query_hash.astype(str).tolist()) & set(
            warmup_role.source_query_hash.astype(str).tolist()
        )
        if overlap:
            raise RuntimeError(
                "policy-validation warmup must use disjoint calibration queries"
            )
        _warm_methods(methods, warmup_role, frames=warmup_frames)
    count, method_count = role.count, len(METHODS)
    nq = role.dataset.previous_q.shape[1]
    latency = np.zeros((count, method_count), dtype=np.int64)
    accepted = np.zeros((count, method_count), dtype=bool)
    fev = np.zeros((count, method_count), dtype=np.int64)
    seed = np.ones((count, method_count), dtype=bool)
    local_attempted = np.zeros((count, method_count), dtype=bool)
    local_accepted = np.zeros((count, method_count), dtype=bool)
    hard_attempted = np.zeros((count, method_count), dtype=bool)
    hard_accepted = np.zeros((count, method_count), dtype=bool)
    same_frame = np.zeros((count, method_count), dtype=bool)
    occupancy = np.full((count, method_count), -1, dtype=np.int8)
    state_before = np.full((count, method_count), -1, dtype=np.int8)
    state_after = np.full((count, method_count), -1, dtype=np.int8)
    switched = np.zeros((count, method_count), dtype=bool)
    probe = np.zeros((count, method_count), dtype=bool)
    hard_before = np.full((count, method_count), -1, dtype=np.int16)
    hard_after = np.full((count, method_count), -1, dtype=np.int16)
    recovery_delay = np.full((count, method_count), -1, dtype=np.int16)
    route = np.full((count, method_count), "", dtype="U64")
    stages = np.full((count, method_count), "", dtype="U128")
    command = np.full((count, method_count, nq), np.nan, dtype=np.float64)
    query_hash = np.full((count, method_count), "", dtype="U64")
    order_position = np.full((count, method_count), -1, dtype=np.int8)
    # state-policy, local, hard, unattributed; all non-temporal work is hard-path cost.
    stage_latency = np.zeros((count, method_count, 4), dtype=np.int64)
    method_index = {name: index for index, name in enumerate(METHODS)}
    latin = latin_method_orders(METHODS, order_seed)
    temporal = methods["temporal_event_cghik"]

    for trajectory_number, (uid, indices) in enumerate(role.groups()):
        del uid
        previous = {name: role.dataset.previous_q[int(indices[0])].copy() for name in METHODS}
        temporal_state = temporal.initial_state()  # type: ignore[attr-defined]
        for offset, raw_index in enumerate(indices):
            index = int(raw_index)
            order = latin[(trajectory_number + offset) % len(latin)]
            if set(order) != set(METHODS):
                raise RuntimeError("policy-validation order is not a permutation")
            for position, name in enumerate(order):
                column = method_index[name]
                query = role.query(index, previous[name])
                started = perf_counter_ns()
                if name == "temporal_event_cghik":
                    outcome = temporal.step(query, temporal_state)  # type: ignore[attr-defined]
                    temporal_state = outcome.state_after
                else:
                    outcome = methods[name].solve(query)  # type: ignore[attr-defined]
                elapsed = perf_counter_ns() - started
                latency[index, column] = elapsed
                accepted[index, column] = bool(outcome.accepted)
                fev[index, column] = int(outcome.function_evaluations)
                query_hash[index, column] = query_digest(query)
                order_position[index, column] = position
                route[index, column] = str(
                    getattr(outcome, "route", getattr(outcome, "entry_action", ""))
                )
                stages[index, column] = ",".join(
                    str(value) for value in getattr(outcome, "executed_stages", ())
                )
                if name == "temporal_event_cghik":
                    value: TemporalOutcome = outcome
                    seed[index, column] = bool(value.learned_seed_ensemble_invoked)
                    local_attempted[index, column] = bool(value.local_attempted)
                    local_accepted[index, column] = bool(value.local_accepted)
                    hard_attempted[index, column] = bool(value.hard_attempted)
                    hard_accepted[index, column] = bool(value.hard_accepted)
                    same_frame[index, column] = bool(value.same_frame_hard_recovery)
                    occupancy[index, column] = (
                        1 if value.occupancy_mode == "local" else 2
                    )
                    state_before[index, column] = MODE_CODE[value.state_before.mode]
                    state_after[index, column] = MODE_CODE[value.state_after.mode]
                    switched[index, column] = bool(value.mode_switched)
                    probe[index, column] = bool(value.local_probe_executed)
                    hard_before[index, column] = int(
                        value.hard_calls_since_local_attempt_before
                    )
                    hard_after[index, column] = int(
                        value.hard_calls_since_local_attempt_after
                    )
                    delay = value.robust_to_local_recovery_delay_frames
                    recovery_delay[index, column] = -1 if delay is None else int(delay)
                    timing = value.timings_ns
                    stage_latency[index, column, 0] = int(timing["state_policy_ns"])
                    stage_latency[index, column, 1] = int(timing["local_path_ns"])
                    stage_latency[index, column, 2] = int(timing["hard_path_ns"])
                    attributed = int(np.sum(stage_latency[index, column, :3]))
                    if attributed > elapsed:
                        raise RuntimeError("temporal stages exceed the outer API latency")
                    stage_latency[index, column, 3] = elapsed - attributed
                else:
                    stage_latency[index, column, 2] = elapsed
                    hard_attempted[index, column] = "hard" in tuple(
                        str(item).lower()
                        for item in getattr(outcome, "executed_stages", ())
                    )
                    hard_accepted[index, column] = bool(
                        outcome.accepted and hard_attempted[index, column]
                    )
                if outcome.accepted and outcome.q is not None:
                    q = np.asarray(outcome.q, dtype=np.float64)
                    command[index, column] = q
                    previous[name] = q.copy()
        if progress_every and (trajectory_number + 1) % progress_every == 0:
            print(
                f"[temporal-event-v6] {role.robot} policy-validation "
                f"{trajectory_number + 1}/{len(role.trajectory_order)}",
                flush=True,
            )
    if (
        np.any(latency <= 0)
        or np.any(order_position < 0)
        or np.any(query_hash == "")
        or not np.array_equal(np.sum(stage_latency, axis=2), latency)
    ):
        raise RuntimeError("policy-validation record contract is incomplete")
    return BenchmarkData(
        role.robot,
        role,
        latency,
        accepted,
        fev,
        seed,
        local_attempted,
        local_accepted,
        hard_attempted,
        hard_accepted,
        same_frame,
        occupancy,
        state_before,
        state_after,
        switched,
        probe,
        hard_before,
        hard_after,
        recovery_delay,
        route,
        stages,
        command,
        query_hash,
        order_position,
        stage_latency,
    )


def _save_benchmark(path: Path, data: BenchmarkData) -> None:
    role = data.role
    np.savez_compressed(
        path,
        method_names=np.asarray(METHODS, dtype="U40"),
        latency_ns=data.latency_ns,
        accepted=data.accepted,
        function_evaluations=data.function_evaluations,
        seed_invoked=data.seed_invoked,
        local_attempted=data.local_attempted,
        local_accepted=data.local_accepted,
        hard_attempted=data.hard_attempted,
        hard_accepted=data.hard_accepted,
        same_frame_hard_recovery=data.same_frame_hard_recovery,
        occupancy_mode=data.occupancy_mode,
        state_before=data.state_before,
        state_after=data.state_after,
        mode_switched=data.mode_switched,
        local_probe=data.local_probe,
        hard_count_before=data.hard_count_before,
        hard_count_after=data.hard_count_after,
        recovery_delay=data.recovery_delay,
        route=data.route,
        executed_stages=data.executed_stages,
        command_q=data.command_q,
        executed_query_hash=data.executed_query_hash,
        method_order_position=data.method_order_position,
        stage_names=np.asarray(("state_policy", "local", "hard", "unattributed"), dtype="U32"),
        stage_latency_ns=data.stage_latency_ns,
        source_query_hash=role.source_query_hash,
        trajectory_uid=role.trajectory_uid,
        trajectory_order=np.asarray(role.trajectory_order, dtype="U64"),
        category=role.dataset.category,
        time_index=role.dataset.time_index,
    )


def _runs(mask: np.ndarray, groups: Sequence[tuple[str, np.ndarray]]) -> list[int]:
    result: list[int] = []
    values = np.asarray(mask, dtype=bool)
    for _, indices in groups:
        current = 0
        for value in values[indices]:
            if value:
                current += 1
            elif current:
                result.append(current)
                current = 0
        if current:
            result.append(current)
    return result


def _run_summary(values: Sequence[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return {
            "run_count": 0,
            "mean": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "histogram": {},
        }
    unique, counts = np.unique(array, return_counts=True)
    return {
        "run_count": int(array.size),
        "mean": float(np.mean(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(np.max(array)),
        "histogram": {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)},
    }


def summarize_benchmark(data: BenchmarkData) -> list[dict[str, Any]]:
    completion = _trajectory_completion(data.accepted.T, data.role)
    rows: list[dict[str, Any]] = []
    temporal_column = METHODS.index("temporal_event_cghik")
    local_failure = data.local_attempted[:, temporal_column] & ~data.local_accepted[:, temporal_column]
    recovered = local_failure & data.hard_accepted[:, temporal_column]
    for column, method in enumerate(METHODS):
        latency_ms = data.latency_ns[:, column].astype(np.float64) / 1e6
        row: dict[str, Any] = {
            "robot": data.robot,
            "method": method,
            "trajectory_count": len(data.role.trajectory_order),
            "whole_trajectory_completion_count": int(np.sum(completion[column])),
            "whole_trajectory_completion": float(np.mean(completion[column])),
            "frame_verified_success": float(np.mean(data.accepted[:, column])),
            "p50_latency_ms": float(np.quantile(latency_ms, 0.50)),
            "p95_latency_ms": float(np.quantile(latency_ms, 0.95)),
            "p99_latency_ms": float(np.quantile(latency_ms, 0.99)),
            "mean_fev": float(np.mean(data.function_evaluations[:, column])),
            "learned_seed_invocation_rate": float(np.mean(data.seed_invoked[:, column])),
            "local_occupancy": None,
            "robust_occupancy": None,
            "mode_switches_per_trajectory": None,
            "transition_frame_rate": None,
            "local_failure_hard_recovery_rate": None,
        }
        if method == "temporal_event_cghik":
            local_mode = data.occupancy_mode[:, column] == MODE_CODE[TemporalMode.LOCAL]
            robust_mode = data.occupancy_mode[:, column] == MODE_CODE[TemporalMode.ROBUST]
            if not np.all(local_mode | robust_mode):
                raise RuntimeError("temporal occupancy must be LOCAL or ROBUST on every frame")
            row.update(
                {
                    "local_occupancy": float(np.mean(local_mode)),
                    "robust_occupancy": float(np.mean(robust_mode)),
                    "mode_switches_per_trajectory": float(
                        np.sum(data.mode_switched[:, column])
                        / len(data.role.trajectory_order)
                    ),
                    "transition_frame_rate": float(np.mean(data.mode_switched[:, column])),
                    "local_failure_hard_recovery_rate": (
                        None
                        if not np.any(local_failure)
                        else float(np.sum(recovered) / np.sum(local_failure))
                    ),
                }
            )
        rows.append(row)
    return rows


def temporal_run_lengths(data: BenchmarkData) -> dict[str, Any]:
    column = METHODS.index("temporal_event_cghik")
    groups = data.role.groups()
    local_success = data.local_attempted[:, column] & data.local_accepted[:, column]
    bootstrap = np.char.startswith(data.route[:, column].astype(str), "bootstrap_")
    all_hard_invocation = data.hard_attempted[:, column]
    robust_required = all_hard_invocation & ~bootstrap
    return {
        "robot": data.robot,
        "definition": {
            "local_success": "consecutive frames accepted by one-step LOCAL, including successful probes",
            "robust_required": (
                "consecutive non-bootstrap frames on which fixed HARD was actually "
                "invoked; mandatory first-frame branch establishment is excluded"
            ),
            "all_hard_invocation": (
                "all consecutive fixed-HARD calls including mandatory bootstrap"
            ),
            "trajectory_boundaries_crossed": False,
        },
        "local_success": _run_summary(_runs(local_success, groups)),
        "robust_required": _run_summary(_runs(robust_required, groups)),
        "all_hard_invocation": _run_summary(_runs(all_hard_invocation, groups)),
    }


def family_mode_rows(data: BenchmarkData) -> list[dict[str, Any]]:
    column = METHODS.index("temporal_event_cghik")
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        mask = data.role.dataset.category == family
        local = data.occupancy_mode[mask, column] == MODE_CODE[TemporalMode.LOCAL]
        robust = data.occupancy_mode[mask, column] == MODE_CODE[TemporalMode.ROBUST]
        if not np.all(local | robust):
            raise RuntimeError("family occupancy contains an unclassified frame")
        rows.append(
            {
                "robot": data.robot,
                "family": family,
                "frame_count": int(np.sum(mask)),
                "local_occupancy": float(np.mean(local)),
                "robust_occupancy": float(np.mean(robust)),
                "frame_verified_success": float(np.mean(data.accepted[mask, column])),
                "learned_seed_invocation_rate": float(np.mean(data.seed_invoked[mask, column])),
                "transition_frame_rate": float(np.mean(data.mode_switched[mask, column])),
                "local_probe_rate": float(np.mean(data.local_probe[mask, column])),
            }
        )
    return rows


def paired_latency_rows(data: BenchmarkData) -> list[dict[str, Any]]:
    temporal = METHODS.index("temporal_event_cghik")
    rows: list[dict[str, Any]] = []
    for comparator in ("always_hard", "counterfactual_cghik_v4"):
        other = METHODS.index(comparator)
        delta = (data.latency_ns[:, temporal] - data.latency_ns[:, other]).astype(np.float64) / 1e6
        query_hash_match = data.executed_query_hash[:, temporal] == data.executed_query_hash[:, other]
        rows.append(
            {
                "robot": data.robot,
                "method": "temporal_event_cghik",
                "comparator": comparator,
                "paired_frame_count": int(delta.size),
                "pairing_unit": (
                    "same exogenous target and trajectory frame; previous_q is "
                    "method-specific closed-loop state"
                ),
                "executed_full_query_hash_match_count": int(np.sum(query_hash_match)),
                "executed_full_query_hash_match_rate": float(np.mean(query_hash_match)),
                "mean_difference_ms": float(np.mean(delta)),
                "median_difference_ms": float(np.median(delta)),
                "p05_difference_ms": float(np.quantile(delta, 0.05)),
                "p95_difference_ms": float(np.quantile(delta, 0.95)),
                "fraction_temporal_faster": float(np.mean(delta < 0.0)),
            }
        )
    return rows


def completion_identity_diagnostic(data: BenchmarkData) -> dict[str, Any]:
    """Report trajectory identity swaps without strengthening the scalar gate."""

    completion = _trajectory_completion(data.accepted.T, data.role)
    hard = completion[METHODS.index("always_hard")]
    temporal = completion[METHODS.index("temporal_event_cghik")]
    gained = [
        uid
        for uid, hard_ok, temporal_ok in zip(
            data.role.trajectory_order, hard, temporal, strict=True
        )
        if temporal_ok and not hard_ok
    ]
    lost = [
        uid
        for uid, hard_ok, temporal_ok in zip(
            data.role.trajectory_order, hard, temporal, strict=True
        )
        if hard_ok and not temporal_ok
    ]
    return {
        "robot": data.robot,
        "gate_uses_scalar_completion_count_only": True,
        "always_hard_completion_count": int(np.sum(hard)),
        "temporal_completion_count": int(np.sum(temporal)),
        "completion_vector_hamming_count": int(np.sum(hard != temporal)),
        "gained_trajectory_uids": gained,
        "lost_trajectory_uids": lost,
    }


def _select_representative(data: BenchmarkData) -> tuple[str, np.ndarray]:
    column = METHODS.index("temporal_event_cghik")
    ranked: list[tuple[int, int, str, np.ndarray]] = []
    for uid, indices in data.role.groups():
        switches = int(np.sum(data.mode_switched[indices, column]))
        robust = int(
            np.sum(data.occupancy_mode[indices, column] == MODE_CODE[TemporalMode.ROBUST])
        )
        ranked.append((-switches, -robust, uid, indices))
    _, _, uid, indices = min(ranked, key=lambda row: row[:3])
    return uid, indices


def _plot_representative(data: BenchmarkData, output: Path, *, dpi: int) -> dict[str, Any]:
    uid, indices = _select_representative(data)
    temporal = METHODS.index("temporal_event_cghik")
    hard = METHODS.index("always_hard")
    x = np.arange(len(indices))
    mode = np.where(
        data.occupancy_mode[indices, temporal] == MODE_CODE[TemporalMode.LOCAL],
        0,
        1,
    )
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 6.6), sharex=True)
    axes[0].step(x, mode, where="post", color="#2f5597")
    axes[0].set_yticks([0, 1], ["LOCAL", "ROBUST"])
    axes[0].set_ylabel("Mode")
    axes[1].plot(x, data.latency_ns[indices, temporal] / 1e6, label="Temporal", lw=1.2)
    axes[1].plot(x, data.latency_ns[indices, hard] / 1e6, label="Always hard", lw=1.0, alpha=0.75)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].step(x, data.function_evaluations[indices, temporal], where="mid", label="Temporal")
    axes[2].step(x, data.function_evaluations[indices, hard], where="mid", label="Always hard", alpha=0.75)
    axes[2].set_ylabel("FEV")
    axes[2].set_xlabel("Frame")
    figure.suptitle(f"{data.robot}: representative temporal trajectory")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    rows = indices
    family = str(data.role.dataset.category[int(rows[0])])
    return {
        "robot": data.robot,
        "trajectory_uid": uid,
        "family": family,
        "selection_rule": "maximum mode switches, then maximum ROBUST occupancy, then UID",
        "mode_switch_count": int(np.sum(data.mode_switched[indices, temporal])),
        "robust_frame_count": int(
            np.sum(data.occupancy_mode[indices, temporal] == MODE_CODE[TemporalMode.ROBUST])
        ),
    }


def _main_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Robot | Method | Completion | Frame success | P50 ms | P95 ms | P99 ms | Mean FEV | Seed rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {robot} | {method} | {whole_trajectory_completion:.4f} | "
            "{frame_verified_success:.4f} | {p50_latency_ms:.4f} | "
            "{p95_latency_ms:.4f} | {p99_latency_ms:.4f} | {mean_fev:.3f} | "
            "{learned_seed_invocation_rate:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def pilot_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_robot = {
        robot: {str(row["method"]): row for row in rows if row["robot"] == robot}
        for robot in ("panda", "ur5e")
    }
    robots: dict[str, Any] = {}
    for robot, methods in by_robot.items():
        hard = methods["always_hard"]
        temporal = methods["temporal_event_cghik"]
        checks = {
            "completion_equal_always_hard": int(temporal["whole_trajectory_completion_count"])
            == int(hard["whole_trajectory_completion_count"]),
            "p95_not_above_always_hard": float(temporal["p95_latency_ms"])
            <= float(hard["p95_latency_ms"]),
            "p50_below_always_hard": float(temporal["p50_latency_ms"])
            < float(hard["p50_latency_ms"]),
            "learned_seed_invocation_rate_at_most_0_60": float(
                temporal["learned_seed_invocation_rate"]
            )
            <= 0.60,
        }
        robots[robot] = {
            "pass": all(checks.values()),
            "checks": checks,
            "p50_ratio_vs_always_hard": float(temporal["p50_latency_ms"])
            / float(hard["p50_latency_ms"]),
            "p95_ratio_vs_always_hard": float(temporal["p95_latency_ms"])
            / float(hard["p95_latency_ms"]),
        }
    return {
        "status": "pass" if all(item["pass"] for item in robots.values()) else "fail",
        "all_robots_pass": all(item["pass"] for item in robots.values()),
        "robots": robots,
        "development_only": True,
        "fresh_evaluation_authorized": all(item["pass"] for item in robots.values()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one complete 150-frame trajectory per family and a separate output.",
    )
    return parser


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    release_v3_root = resolve_path(config, str(config["release_v3_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    output_root = resolve_path(config, str(config["output_root"]))
    if smoke:
        output_root = workspace / "outputs" / "temporal_event_v6_smoke"
    for name, value in (
        ("source_config", source_config_path),
        ("release_v3_root", release_v3_root),
        ("release_v4_root", release_v4_root),
        ("output_root", output_root),
    ):
        _forbid_formal_path(value, name=name)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"overwrite forbidden: {output_root}")
    stale = sorted(output_root.parent.glob(f".{output_root.name}.incomplete.*"))
    if stale:
        raise FileExistsError(
            "stale staging requires inspection: " + ", ".join(str(path) for path in stale)
        )
    git_commit_start = _git(workspace, "rev-parse", "HEAD")
    git_status_start = _git(workspace, "status", "--short").splitlines()
    if not smoke and git_status_start:
        raise RuntimeError("full Temporal Event V6 pilot requires a clean committed worktree")

    protected_roots = {
        "hierarchical_v5_pilot": workspace / "outputs" / "hierarchical_v5_pilot",
        "hierarchical_v5_lite_pilot": workspace / "outputs" / "hierarchical_v5_lite_pilot",
        "prior_temporal_v6_pilot": workspace / "outputs" / "temporal_v6_pilot",
        "release_v3_locked": release_v3_root,
        "release_v4_locked": release_v4_root,
    }
    protected_before = {name: _tree_snapshot(path) for name, path in protected_roots.items()}
    protected_digest_before = {
        name: _snapshot_digest(value) for name, value in protected_before.items()
    }
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("state.py").resolve(),
        Path(__file__).with_name("runtime.py").resolve(),
        Path(__file__).with_name("trajectories.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
        Path(config["_config_path"]).resolve(),
        source_config_path.resolve(),
        (workspace / "scripts" / "run_temporal_v6_pilot.sh").resolve(),
        (workspace / "tests" / "test_temporal_v6.py").resolve(),
    )
    implementation_before = {
        str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
        for path in implementation_paths
    }
    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    started_at = _utc()
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
    device = str(config["runtime"]["device"])
    dt = float(config["trajectory_data"]["dt"])
    warmup = 8 if smoke else int(config["timing"]["warmup_frames"])
    progress = 0 if smoke else int(config["runtime"]["progress_every_trajectories"])

    split_reports: dict[str, Any] = {}
    calibration_reports: dict[str, Any] = {}
    selected_h: dict[str, int] = {}
    contexts: dict[str, dict[str, Any]] = {}
    release_inputs: dict[str, Any] = {}
    prior_registry_inputs: dict[str, Any] = {}

    # Phase A: generate and seal outcome-blind roles, then collect calibration only.
    for robot in ("panda", "ur5e"):
        kinematics = load_robot(source_config, robot)
        identity, urdf_path = _kinematics_identity(source_config, robot)
        # Verify every frozen learned/solver artifact before constructing a
        # runtime or spending calibration compute with it.
        release_inputs[robot] = _verified_release_inputs(
            workspace=workspace,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
        )
        sources, seed_sources, prior_audit = _prior_registries(
            workspace=workspace,
            robot=robot,
            kinematics=kinematics,
            identity=identity,
            dt=dt,
        )
        spec = FreshTrajectorySpec.frozen(robot, kinematics_identity=identity)
        calibration_role, policy_role, split = generate_fresh_development_roles(
            kinematics,
            spec,
            source_registries=sources,
            seed_registries=seed_sources,
        )
        if not split["source_isolation"][
            "declared_nonformal_row_hash_coverage_complete"
        ]:
            raise RuntimeError("declared nonformal query-hash coverage is incomplete")
        if not split["seed_isolation"]["prior_seed_class_coverage_complete"]:
            raise RuntimeError("seed prior-source coverage is incomplete")
        split_reports[robot] = split
        prior_registry_inputs[robot] = {
            **prior_audit,
            "urdf": _artifact(urdf_path, relative_to=workspace if urdf_path.is_relative_to(workspace) else None),
        }
        save_trajectory_role(staging / f"{robot}_trajectory_calibration.npz", calibration_role)
        save_trajectory_role(
            staging / f"{robot}_trajectory_policy_validation.npz", policy_role
        )
        calibration_view = _view(calibration_role, smoke=smoke)
        hard = _fresh_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        temporal = {
            h: _fresh_temporal(
                source_config=source_config,
                release_root=release_v3_root,
                robot=robot,
                kinematics=kinematics,
                device=device,
                hold_frames=h,
            )
            for h in H_VALUES
        }
        calibration_data = collect_calibration(
            calibration_view,
            hard_runtime=hard,
            temporal_runtimes=temporal,
            order_seed=910_000 + int(config["trajectory_data"]["split_seed"][robot]),
            warmup_frames=warmup,
            progress_every=progress,
        )
        _save_calibration(staging / f"{robot}_calibration_records.npz", calibration_data, calibration_view)
        h, report = select_hold_frames(calibration_data, calibration_view)
        selected_h[robot] = h
        calibration_reports[robot] = report
        contexts[robot] = {
            "kinematics": kinematics,
            "policy_role_path": staging / f"{robot}_trajectory_policy_validation.npz",
            "calibration_warmup_role": calibration_view,
        }
        print(
            f"[temporal-event-v6] {robot} selected H={h} on calibration",
            flush=True,
        )

    save_split_audit_manifest(
        staging / "trajectory_split_manifest.json",
        {
            "protocol": TRAJECTORY_PROTOCOL,
            "pilot_protocol": PROTOCOL,
            "robots": split_reports,
            "formal_test_data_opened": False,
            "test_v3_test_v4_files_opened": 0,
        },
    )
    _write_json(staging / "calibration_candidate_metrics.json", calibration_reports)
    selection_payload = {
        "status": "selected_on_calibration",
        "selected_hold_frames": selected_h,
        "selection_role": CALIBRATION_ROLE,
        "policy_validation_outcomes_computed": False,
        "policy_validation_used_for_selection": False,
        "completion_rule": "scalar whole-trajectory completion count exactly equal always-hard",
        "reports": calibration_reports,
    }
    _write_json(staging / "calibration_selection.json", selection_payload)
    effective_config = json.loads(json.dumps({key: value for key, value in config.items() if key != "_config_path"}))
    effective_config["execution"] = {
        "smoke": smoke,
        "trajectories_per_family_per_role": 1 if smoke else 10,
        "frames_per_trajectory": 150,
        "warmup_frames": warmup,
    }
    (staging / "temporal_event_v6_pilot.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
    )
    seal_payload = {
        "protocol": PROTOCOL,
        "sealed_at": _utc(),
        "selected_hold_frames": selected_h,
        "selection": _artifact(staging / "calibration_selection.json", relative_to=staging),
        "candidate_metrics": _artifact(staging / "calibration_candidate_metrics.json", relative_to=staging),
        "split_manifest": _artifact(staging / "trajectory_split_manifest.json", relative_to=staging),
        "roles": {
            robot: {
                "calibration": _artifact(staging / f"{robot}_trajectory_calibration.npz", relative_to=staging),
                "policy_validation": _artifact(staging / f"{robot}_trajectory_policy_validation.npz", relative_to=staging),
                "calibration_records": _artifact(staging / f"{robot}_calibration_records.npz", relative_to=staging),
            }
            for robot in ("panda", "ur5e")
        },
        "release_inputs": release_inputs,
        "policy_validation_outcomes_computed_before_seal": False,
    }
    _write_json(staging / "calibration_seal.json", seal_payload)
    seal_sha = _sha256_file(staging / "calibration_seal.json")

    # Phase B: reload the sealed PV roles and run each robot exactly once.
    benchmark_data: dict[str, BenchmarkData] = {}
    main_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    run_lengths: dict[str, Any] = {}
    completion_diagnostics: dict[str, Any] = {}
    representative: dict[str, Any] = {}
    for robot in ("panda", "ur5e"):
        if _sha256_file(staging / "calibration_seal.json") != seal_sha:
            raise RuntimeError("calibration seal changed before policy validation")
        full_role = load_trajectory_role(
            contexts[robot]["policy_role_path"],
            robot=robot,
            expected_role=POLICY_VALIDATION_ROLE,
            expected_artifact=seal_payload["roles"][robot]["policy_validation"],
        )
        role = _view(full_role, smoke=smoke)
        kinematics = contexts[robot]["kinematics"]
        methods = _build_policy_validation_methods(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
            hold_frames=selected_h[robot],
        )
        result = benchmark_policy_validation(
            role,
            methods=methods,
            warmup_role=contexts[robot]["calibration_warmup_role"],
            order_seed=920_000 + int(config["trajectory_data"]["split_seed"][robot]),
            warmup_frames=warmup,
            progress_every=progress,
        )
        benchmark_data[robot] = result
        _save_benchmark(staging / f"{robot}_policy_validation_records.npz", result)
        main_rows.extend(summarize_benchmark(result))
        family_rows.extend(family_mode_rows(result))
        paired_rows.extend(paired_latency_rows(result))
        run_lengths[robot] = temporal_run_lengths(result)
        completion_diagnostics[robot] = completion_identity_diagnostic(result)
        representative[robot] = _plot_representative(
            result,
            staging / f"{robot}_representative_timeline",
            dpi=int(config["reporting"]["png_dpi"]),
        )
    if _sha256_file(staging / "calibration_seal.json") != seal_sha:
        raise RuntimeError("calibration seal changed after policy validation")

    gate = pilot_gate(main_rows)
    _write_json(staging / "main_table.json", main_rows)
    _write_csv(staging / "main_table.csv", main_rows)
    (staging / "main_table.md").write_text(_main_markdown(main_rows), encoding="utf-8")
    _write_json(staging / "query_family_mode_distribution.json", family_rows)
    _write_csv(staging / "query_family_mode_distribution.csv", family_rows)
    _write_json(staging / "paired_latency_summary.json", paired_rows)
    _write_json(staging / "run_length_distribution.json", run_lengths)
    _write_json(
        staging / "completion_identity_diagnostics.json", completion_diagnostics
    )
    _write_json(staging / "representative_trajectory.json", representative)
    _write_json(staging / "pilot_gate.json", gate)
    environment = environment_payload()
    environment.update(
        {
            "development_only": True,
            "formal_test_data_opened": False,
            "test_v3_test_v4_files_opened": 0,
            "timing_clock": "perf_counter_ns",
            "method_interleaving": "seeded four-by-four Latin blocks",
        }
    )
    _write_json(staging / "environment.json", environment)

    protected_after = {name: _tree_snapshot(path) for name, path in protected_roots.items()}
    protected_digest_after = {
        name: _snapshot_digest(value) for name, value in protected_after.items()
    }
    if protected_digest_after != protected_digest_before:
        raise RuntimeError("a protected V5/V5-Lite/V6/release tree changed")
    git_commit_end = _git(workspace, "rev-parse", "HEAD")
    if git_commit_end != git_commit_start:
        raise RuntimeError("git HEAD changed during the pilot")
    implementation_after = {
        str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
        for path in implementation_paths
    }
    if implementation_after != implementation_before:
        raise RuntimeError("Temporal Event V6 implementation changed during the run")
    finished_at = _utc()
    artifacts = _tree_descriptors(staging)
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_smoke" if smoke else "complete_policy_validation_pilot",
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit_start": git_commit_start,
        "git_commit_end": git_commit_end,
        "git_status_start": git_status_start,
        "implementation_sources": implementation_before,
        "artifacts": artifacts,
        "selected_hold_frames": selected_h,
        "calibration_seal_sha256": seal_sha,
        "calibration_role": CALIBRATION_ROLE,
        "policy_validation_role": POLICY_VALIDATION_ROLE,
        "policy_frozen_before_policy_validation": True,
        "policy_validation_run_count_per_robot": 1,
        "policy_validation_used_for_selection": False,
        "policy_validation_used_for_retuning": False,
        "formal_test_data_opened": False,
        "test_v3_test_v4_files_opened": 0,
        "formal_test_started": False,
        "fresh_evaluation_started": False,
        "protected_tree_digest_before": protected_digest_before,
        "protected_tree_digest_after": protected_digest_after,
        "protected_trees_unchanged": True,
        "prior_registry_inputs": prior_registry_inputs,
        "trajectory_counts_per_robot": {
            "calibration": 4 if smoke else 40,
            "policy_validation": 4 if smoke else 40,
        },
        "frames_per_trajectory": 150,
        "methods": list(METHODS),
        "pilot_gate": gate,
    }
    _write_json(staging / "run_manifest.json", manifest)
    os.replace(staging, output_root)
    print(
        f"[temporal-event-v6] complete: output={output_root}, gate={gate['status']}",
        flush=True,
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    run(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkData",
    "CalibrationData",
    "H_VALUES",
    "METHODS",
    "TrajectoryView",
    "benchmark_policy_validation",
    "collect_calibration",
    "family_mode_rows",
    "pilot_gate",
    "run",
    "select_hold_frames",
    "summarize_benchmark",
    "temporal_run_lengths",
    "validate_config",
]
