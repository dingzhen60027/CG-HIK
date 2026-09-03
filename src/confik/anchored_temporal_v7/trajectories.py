"""Fresh transition-rich development trajectories for Anchored Temporal V7.

The generator in this module is deliberately independent from the Temporal V6
reference-trajectory generator.  It creates four explicitly staged joint-space
reference families, converts them to known-feasible FK targets, and splits only
at complete-trajectory grain before any solver, verifier, or policy is run.

Formal-evaluation data is outside this module's trust boundary.  Files, roles,
and identities containing ``test`` are rejected before opening.  Isolation from
older non-formal sources is checked only through registries supplied explicitly
by the caller.  Runtime-query hashes reuse the canonical V6 implementation so
that row-level comparisons remain meaningful across versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..data.datasets import QueryDataset
from ..geometry import pose_distance
from ..kinematics.base import KinematicsModel
from ..temporal_v6.trajectories import (
    ALLOWED_SEED_SOURCE_CLASSES,
    ALLOWED_SOURCE_CLASSES,
    HASH_SCHEMA,
    AllowedManifestIdentity,
    AllowedSeedRegistry,
    AllowedSourceHashRegistry,
    audit_seed_isolation,
    audit_source_isolation,
    load_allowed_source_hash_registry,
    runtime_query_hashes,
)


PROTOCOL = "anchored_temporal_v7_transition_trajectories_v1"
TRAJECTORY_HASH_SCHEMA = "confik_ordered_transition_trajectory_v2"

CALIBRATION_ROLE = "anchored_trajectory_calibration"
POLICY_VALIDATION_ROLE = "anchored_trajectory_policy_validation"
ROLES = (CALIBRATION_ROLE, POLICY_VALIDATION_ROLE)

SMOOTH_ORIENTATION_FAMILY = "smooth_fast_orientation_smooth"
SINGULAR_TRANSITION_FAMILY = "regular_near_singular_regular"
LIMIT_TRANSITION_FAMILY = "central_joint_limit_skim_return"
CURVATURE_TRANSITION_FAMILY = "slow_high_curvature_high_speed_slow"
FAMILIES = (
    SMOOTH_ORIENTATION_FAMILY,
    SINGULAR_TRANSITION_FAMILY,
    LIMIT_TRANSITION_FAMILY,
    CURVATURE_TRANSITION_FAMILY,
)

# Half-open phase intervals.  Keeping them frozen makes the intended transition
# structure inspectable without consulting method outcomes.
FAMILY_PHASES: dict[str, tuple[tuple[int, int, str], ...]] = {
    SMOOTH_ORIENTATION_FAMILY: (
        (0, 45, "smooth_pre"),
        (45, 105, "fast_orientation"),
        (105, 150, "smooth_post"),
    ),
    SINGULAR_TRANSITION_FAMILY: (
        (0, 20, "regular_pre"),
        (20, 65, "approach_singular"),
        (65, 85, "near_singular"),
        (85, 130, "return_regular"),
        (130, 150, "regular_post"),
    ),
    LIMIT_TRANSITION_FAMILY: (
        (0, 20, "central_pre"),
        (20, 70, "approach_limit"),
        (70, 80, "joint_limit_skim"),
        (80, 130, "return_central"),
        (130, 150, "central_post"),
    ),
    CURVATURE_TRANSITION_FAMILY: (
        (0, 40, "slow_pre"),
        (40, 110, "high_curvature_high_speed"),
        (110, 150, "slow_post"),
    ),
}

FROZEN_POOL_SEEDS = {"panda": 862_701, "ur5e": 862_702}
FROZEN_SPLIT_SEEDS = {"panda": 862_711, "ur5e": 862_712}

PATHS_PER_FAMILY_POOL = 20
PATHS_PER_FAMILY_PER_ROLE = 10
STEPS_PER_TRAJECTORY = 150
DT = 0.02

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LIMIT_EPS = 1e-10
_VELOCITY_EPS = 1e-9


def _reject_test_named(value: str | Path, *, name: str) -> None:
    if "test" in str(value).casefold():
        raise ValueError(f"{name} must not contain 'test': {value}")


def _require_hash(value: str, *, name: str) -> str:
    normalized = str(value).casefold()
    if _HEX64.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def _hash_field(digest: Any, name: str, value: bytes) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(4, "little", signed=False))
    digest.update(encoded_name)
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(value)


def _canonical_int_bytes(values: Any) -> bytes:
    array = np.asarray(values, dtype="<i8", order="C")
    shape = np.asarray(array.shape, dtype="<i8").tobytes()
    return len(array.shape).to_bytes(4, "little") + shape + array.tobytes(order="C")


def _digest_strings(values: Sequence[str], *, ordered: bool) -> str:
    sequence = list(values) if ordered else sorted(set(values))
    digest = sha256()
    _hash_field(digest, "domain", b"ordered" if ordered else b"set")
    for value in sequence:
        _hash_field(digest, "value", str(value).encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_arrays(family: str) -> tuple[NDArray[np.str_], NDArray[np.int64], NDArray[np.bool_]]:
    if family not in FAMILY_PHASES:
        raise ValueError(f"unknown transition family: {family!r}")
    phase = np.empty(STEPS_PER_TRAJECTORY, dtype="U32")
    phase_index = np.empty(STEPS_PER_TRAJECTORY, dtype=np.int64)
    boundary = np.zeros(STEPS_PER_TRAJECTORY, dtype=bool)
    cursor = 0
    for index, (start, stop, name) in enumerate(FAMILY_PHASES[family]):
        if start != cursor or stop <= start:
            raise RuntimeError(f"non-contiguous phase contract for {family}")
        phase[start:stop] = name
        phase_index[start:stop] = index
        if start:
            boundary[start] = True
        cursor = stop
    if cursor != STEPS_PER_TRAJECTORY:
        raise RuntimeError(f"phase contract for {family} does not cover 150 frames")
    return phase, phase_index, boundary


@dataclass(frozen=True)
class FreshTrajectorySpec:
    """Frozen V7 generation and whole-trajectory split contract."""

    robot: str
    pool_seed: int
    split_seed: int
    kinematics_identity: str
    paths_per_family_pool: int = PATHS_PER_FAMILY_POOL
    paths_per_family_per_role: int = PATHS_PER_FAMILY_PER_ROLE
    steps: int = STEPS_PER_TRAJECTORY
    dt: float = DT

    def __post_init__(self) -> None:
        robot = str(self.robot).casefold()
        if robot not in FROZEN_POOL_SEEDS:
            raise ValueError(f"unsupported Anchored Temporal V7 robot: {self.robot!r}")
        if int(self.pool_seed) != FROZEN_POOL_SEEDS[robot]:
            raise ValueError(
                f"{robot} pool_seed must remain frozen at {FROZEN_POOL_SEEDS[robot]}"
            )
        if int(self.split_seed) != FROZEN_SPLIT_SEEDS[robot]:
            raise ValueError(
                f"{robot} split_seed must remain frozen at {FROZEN_SPLIT_SEEDS[robot]}"
            )
        if (
            int(self.paths_per_family_pool) != PATHS_PER_FAMILY_POOL
            or int(self.paths_per_family_per_role) != PATHS_PER_FAMILY_PER_ROLE
            or int(self.steps) != STEPS_PER_TRAJECTORY
            or float(self.dt) != DT
        ):
            raise ValueError("the frozen 20-pool/10+10-role/150-frame contract changed")
        identity = str(self.kinematics_identity).casefold()
        _reject_test_named(identity, name="kinematics_identity")
        _require_hash(identity, name="kinematics_identity")
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "kinematics_identity", identity)

    @classmethod
    def frozen(cls, robot: str, *, kinematics_identity: str) -> "FreshTrajectorySpec":
        normalized = str(robot).casefold()
        if normalized not in FROZEN_POOL_SEEDS:
            raise ValueError(f"unsupported Anchored Temporal V7 robot: {robot!r}")
        return cls(
            robot=normalized,
            pool_seed=FROZEN_POOL_SEEDS[normalized],
            split_seed=FROZEN_SPLIT_SEEDS[normalized],
            kinematics_identity=kinematics_identity,
        )


# More explicit alias for callers that do not need V6-compatible naming.
FreshTransitionTrajectorySpec = FreshTrajectorySpec


@dataclass(frozen=True)
class FreshTrajectoryRole:
    robot: str
    role: str
    kinematics_identity: str
    dt: float
    dataset: QueryDataset
    source_query_hash: NDArray[np.str_]
    trajectory_uid: NDArray[np.str_]
    trajectory_order: tuple[str, ...]
    phase: NDArray[np.str_]
    phase_index: NDArray[np.int64]
    transition_boundary: NDArray[np.bool_]

    def __post_init__(self) -> None:
        robot = str(self.robot).casefold()
        role = str(self.role)
        if robot not in FROZEN_POOL_SEEDS:
            raise ValueError(f"unsupported robot: {self.robot!r}")
        _reject_test_named(role, name="role")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        identity = _require_hash(
            str(self.kinematics_identity).casefold(), name="kinematics_identity"
        )
        if float(self.dt) != DT:
            raise ValueError(f"dt must remain {DT}")
        count = len(self.dataset)
        query_hash = np.asarray(self.source_query_hash, dtype="U64")
        trajectory_uid = np.asarray(self.trajectory_uid, dtype="U64")
        phase = np.asarray(self.phase, dtype="U32")
        phase_index = np.asarray(self.phase_index, dtype=np.int64)
        boundary = np.asarray(self.transition_boundary, dtype=bool)
        if any(array.shape != (count,) for array in (
            query_hash, trajectory_uid, phase, phase_index, boundary
        )):
            raise ValueError("role metadata must contain exactly one value per frame")
        if any(_HEX64.fullmatch(value) is None for value in query_hash.tolist()):
            raise ValueError("role contains an invalid source query hash")
        if any(_HEX64.fullmatch(value) is None for value in trajectory_uid.tolist()):
            raise ValueError("role contains an invalid trajectory UID")
        if len(set(query_hash.tolist())) != count:
            raise ValueError("source query hashes are not unique within the role")
        order = tuple(str(value) for value in self.trajectory_order)
        expected_trajectories = PATHS_PER_FAMILY_PER_ROLE * len(FAMILIES)
        if len(order) != expected_trajectories or len(set(order)) != len(order):
            raise ValueError("trajectory_order must contain exactly 40 unique UIDs")
        if set(order) != set(trajectory_uid.tolist()):
            raise ValueError("trajectory_order and per-frame trajectory UIDs differ")
        if count != expected_trajectories * STEPS_PER_TRAJECTORY:
            raise ValueError("role does not contain the frozen 6000-frame workload")
        if set(self.dataset.category.tolist()) != set(FAMILIES):
            raise ValueError("role does not contain the four V7 transition families")
        if not np.all(self.dataset.expected_reachable) or not np.all(
            self.dataset.continuity_feasible
        ):
            raise ValueError("V7 reference roles must be known feasible")

        for family in FAMILIES:
            selected = self.dataset.category == family
            if int(np.sum(selected)) != PATHS_PER_FAMILY_PER_ROLE * STEPS_PER_TRAJECTORY:
                raise ValueError(f"family {family} does not contain 1500 frames")
            if len(set(trajectory_uid[selected].tolist())) != PATHS_PER_FAMILY_PER_ROLE:
                raise ValueError(f"family {family} does not contain 10 trajectories")

        for uid in order:
            rows = np.flatnonzero(trajectory_uid == uid).astype(np.int64)
            if len(rows) != STEPS_PER_TRAJECTORY or not np.array_equal(
                self.dataset.time_index[rows],
                np.arange(STEPS_PER_TRAJECTORY, dtype=np.int64),
            ):
                raise ValueError("role contains an incomplete or out-of-order trajectory")
            families = set(self.dataset.category[rows].tolist())
            if len(families) != 1:
                raise ValueError("role trajectory crosses family boundaries")
            family = next(iter(families))
            expected_phase, expected_index, expected_boundary = _phase_arrays(family)
            if (
                not np.array_equal(phase[rows], expected_phase)
                or not np.array_equal(phase_index[rows], expected_index)
                or not np.array_equal(boundary[rows], expected_boundary)
            ):
                raise ValueError(f"trajectory phase schema changed for {family}")

        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "kinematics_identity", identity)
        object.__setattr__(self, "source_query_hash", query_hash)
        object.__setattr__(self, "trajectory_uid", trajectory_uid)
        object.__setattr__(self, "trajectory_order", order)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "phase_index", phase_index)
        object.__setattr__(self, "transition_boundary", boundary)

    @property
    def count(self) -> int:
        return len(self.dataset)

    def groups(self) -> tuple[tuple[str, NDArray[np.int64]], ...]:
        return tuple(
            (uid, np.flatnonzero(self.trajectory_uid == uid).astype(np.int64))
            for uid in self.trajectory_order
        )


FreshTransitionTrajectoryRole = FreshTrajectoryRole


def source_query_hashes(
    dataset: QueryDataset,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> NDArray[np.str_]:
    """Hash source queries with the exact canonical V6 runtime schema."""

    return runtime_query_hashes(
        dataset.previous_q,
        dataset.target_position,
        dataset.target_rotation,
        robot=robot,
        dt=dt,
        kinematics_identity=kinematics_identity,
    )


def _trajectory_hash(
    *,
    robot: str,
    kinematics_identity: str,
    public_family: str,
    query_hashes: Sequence[str],
    time_index: Any,
    phase: Sequence[str],
) -> str:
    digest = sha256()
    _hash_field(digest, "schema", TRAJECTORY_HASH_SCHEMA.encode("utf-8"))
    _hash_field(digest, "robot", robot.encode("utf-8"))
    _hash_field(digest, "kinematics", kinematics_identity.encode("ascii"))
    _hash_field(digest, "family", public_family.encode("utf-8"))
    _hash_field(digest, "time_index", _canonical_int_bytes(time_index))
    for query_hash, phase_name in zip(query_hashes, phase, strict=True):
        _hash_field(digest, "source_query_hash", str(query_hash).encode("ascii"))
        _hash_field(digest, "phase", str(phase_name).encode("utf-8"))
    return digest.hexdigest()


def source_hash_registry_from_role(
    role: FreshTrajectoryRole,
    *,
    identity: str,
    source_class: str,
) -> AllowedSourceHashRegistry:
    payload = {
        "identity": str(identity),
        "source_class": str(source_class),
        "query_hashes": sorted(set(role.source_query_hash.astype(str).tolist())),
        "trajectory_uids": sorted(set(role.trajectory_uid.astype(str).tolist())),
    }
    registry_sha = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AllowedSourceHashRegistry(
        identity=str(identity),
        source_class=source_class,
        query_hashes=tuple(payload["query_hashes"]),
        trajectory_uids=tuple(payload["trajectory_uids"]),
        registry_sha256=registry_sha,
    )


def source_hash_registry_from_dataset(
    dataset: QueryDataset,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
    identity: str,
    source_class: str,
) -> AllowedSourceHashRegistry:
    _reject_test_named(identity, name="registry identity")
    hashes = tuple(sorted(source_query_hashes(
        dataset,
        robot=robot,
        dt=dt,
        kinematics_identity=kinematics_identity,
    ).astype(str).tolist()))
    if len(set(hashes)) != len(hashes):
        raise ValueError("authorized source dataset contains duplicate runtime queries")
    payload = {
        "identity": str(identity),
        "source_class": str(source_class),
        "query_hashes": list(hashes),
        "trajectory_uids": [],
    }
    registry_sha = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AllowedSourceHashRegistry(
        identity=str(identity),
        source_class=source_class,
        query_hashes=hashes,
        registry_sha256=registry_sha,
    )


def _closed_loop(
    anchor: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    direction: NDArray[np.float64],
    count: int,
    *,
    include_start: bool,
) -> NDArray[np.float64]:
    if include_start:
        angles = np.linspace(0.0, 2.0 * np.pi, count)
    else:
        angles = 2.0 * np.pi * np.arange(1, count + 1, dtype=np.float64) / count
    return anchor[None, :] + np.sin(angles)[:, None] * (
        amplitude * direction
    )[None, :]


def _unit_direction(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    direction = rng.normal(size=count)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction[0] = 1.0
        norm = 1.0
    return np.asarray(direction / norm, dtype=np.float64)


def _harmonic_reference(
    kinematics: KinematicsModel,
    rng: np.random.Generator,
    *,
    orientation_only: bool,
) -> NDArray[np.float64]:
    count = STEPS_PER_TRAJECTORY
    center = kinematics.random_configuration(rng, margin=0.18)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=kinematics.nq)
    if orientation_only:
        challenge = np.zeros(count, dtype=bool)
        challenge[45:105] = True
        wrist = np.arange(kinematics.nq) >= kinematics.nq - min(3, kinematics.nq)
        low_rate = np.where(wrist, 0.05, 0.035)
        high_rate = np.where(wrist, rng.uniform(0.38, 0.52, kinematics.nq), 0.05)
        weights = np.where(wrist, 1.0, 0.15)
    else:
        challenge = np.zeros(count, dtype=bool)
        challenge[40:110] = True
        low_rate = np.full(kinematics.nq, 0.035, dtype=np.float64)
        high_rate = rng.uniform(0.32, 0.55, size=kinematics.nq)
        weights = np.ones(kinematics.nq, dtype=np.float64)
    rates = low_rate[None, :] + challenge[:, None] * (high_rate - low_rate)[None, :]
    theta = phase[None, :] + np.cumsum(rates, axis=0)
    amplitude = (
        0.90
        * kinematics.limits.velocity
        * DT
        / (2.0 * np.sin(high_rate / 2.0))
        * weights
    )
    return np.asarray(center[None, :] + np.sin(theta) * amplitude[None, :])


def _singular_transition_reference(
    kinematics: KinematicsModel,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    singular_pool = np.stack(
        [kinematics.random_configuration(rng, margin=0.08) for _ in range(256)]
    )
    singular_sigma = np.asarray(
        [kinematics.min_singular_value(row) for row in singular_pool],
        dtype=np.float64,
    )
    singular = singular_pool[int(np.argmin(singular_sigma))]

    travel_cap = 0.78 * kinematics.limits.velocity * DT * 45.0
    regular_pool = np.stack(
        [
            kinematics.clip(
                singular + rng.uniform(-travel_cap, travel_cap), margin=0.08
            )
            for _ in range(256)
        ]
    )
    regular_sigma = np.asarray(
        [kinematics.min_singular_value(row) for row in regular_pool],
        dtype=np.float64,
    )
    regular = regular_pool[int(np.argmax(regular_sigma))]

    amplitude = np.minimum(
        0.015 * (kinematics.limits.upper - kinematics.limits.lower),
        0.10 * kinematics.limits.velocity * DT,
    )
    pre_direction = _unit_direction(rng, kinematics.nq)
    singular_direction = _unit_direction(rng, kinematics.nq)
    post_direction = _unit_direction(rng, kinematics.nq)
    reference = np.empty((STEPS_PER_TRAJECTORY, kinematics.nq), dtype=np.float64)
    reference[:20] = _closed_loop(
        regular, amplitude, pre_direction, 20, include_start=True
    )
    for frame in range(20, 65):
        reference[frame] = regular + (frame - 19) / 45.0 * (singular - regular)
    reference[65:85] = _closed_loop(
        singular, 0.25 * amplitude, singular_direction, 20, include_start=True
    )
    for frame in range(85, 130):
        reference[frame] = singular + (frame - 84) / 45.0 * (regular - singular)
    reference[130:] = _closed_loop(
        regular, 1.13 * amplitude, post_direction, 20, include_start=False
    )
    return reference, {
        "singular_candidate_count": 256,
        "regular_candidate_count": 256,
        "selected_singular_sigma": float(np.min(singular_sigma)),
        "selected_regular_sigma": float(np.max(regular_sigma)),
        "regular_search_radius_velocity_fraction": 0.78,
    }


def _limit_transition_reference(
    kinematics: KinematicsModel,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    span = kinematics.limits.upper - kinematics.limits.lower
    normalized_velocity = kinematics.limits.velocity * DT / span
    joint = int(np.argmax(normalized_velocity))
    central = kinematics.random_configuration(rng, margin=0.30)
    central[joint] = 0.5 * (
        kinematics.limits.lower[joint] + kinematics.limits.upper[joint]
    )
    side = int(rng.integers(0, 2))
    limit = central.copy()
    normalized_limit = 0.025 if side == 0 else 0.975
    limit[joint] = kinematics.limits.lower[joint] + normalized_limit * span[joint]

    amplitude = np.minimum(0.01 * span, 0.08 * kinematics.limits.velocity * DT)
    pre_direction = _unit_direction(rng, kinematics.nq)
    pre_direction[joint] = 0.0
    pre_norm = float(np.linalg.norm(pre_direction))
    if pre_norm > 0.0:
        pre_direction /= pre_norm
    post_direction = _unit_direction(rng, kinematics.nq)
    post_direction[joint] = 0.0
    post_norm = float(np.linalg.norm(post_direction))
    if post_norm > 0.0:
        post_direction /= post_norm

    reference = np.empty((STEPS_PER_TRAJECTORY, kinematics.nq), dtype=np.float64)
    reference[:20] = _closed_loop(
        central, amplitude, pre_direction, 20, include_start=True
    )
    for frame in range(20, 70):
        reference[frame] = central + (frame - 19) / 50.0 * (limit - central)
    reference[70:80] = limit
    skim_angle = np.linspace(0.0, 2.0 * np.pi, 10)
    inward = 1.0 if side == 0 else -1.0
    reference[70:80, joint] += (
        inward * 0.004 * span[joint] * (1.0 - np.cos(skim_angle)) / 2.0
    )
    for frame in range(80, 130):
        reference[frame] = limit + (frame - 79) / 50.0 * (central - limit)
    reference[130:] = _closed_loop(
        central, 1.17 * amplitude, post_direction, 20, include_start=False
    )
    return reference, {
        "skim_joint_index": joint,
        "skim_joint_name": kinematics.joint_names[joint],
        "skim_side": "lower" if side == 0 else "upper",
        "skim_normalized_position": normalized_limit,
    }


def _velocity_utilization(
    kinematics: KinematicsModel,
    reference: NDArray[np.float64],
    *,
    dt: float,
) -> NDArray[np.float64]:
    delta = np.stack(
        [
            np.abs(kinematics.difference(reference[index], reference[index - 1]))
            for index in range(1, len(reference))
        ]
    )
    return np.max(delta / (kinematics.limits.velocity * dt)[None, :], axis=1)


def _geometric_audit(
    kinematics: KinematicsModel,
    reference: NDArray[np.float64],
    family: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if reference.shape != (STEPS_PER_TRAJECTORY, kinematics.nq):
        raise RuntimeError("transition generator returned an invalid reference shape")
    if not np.all(np.isfinite(reference)):
        raise RuntimeError("transition generator returned non-finite joints")
    if np.any(reference < kinematics.limits.lower - _LIMIT_EPS) or np.any(
        reference > kinematics.limits.upper + _LIMIT_EPS
    ):
        raise RuntimeError("transition generator exceeded a joint limit")
    utilization = _velocity_utilization(kinematics, reference, dt=DT)
    max_utilization = float(np.max(utilization))
    if max_utilization > 1.0 + _VELOCITY_EPS:
        raise RuntimeError(
            f"{family} reference exceeds velocity contract: {max_utilization}"
        )
    joint_margin = np.asarray(
        [np.min(kinematics.joint_margin(row)) for row in reference],
        dtype=np.float64,
    )
    poses = [kinematics.forward(row) for row in reference]
    position_step = np.asarray(
        [pose_distance(poses[index], poses[index - 1])[0] for index in range(1, 150)]
    )
    orientation_step = np.asarray(
        [pose_distance(poses[index], poses[index - 1])[1] for index in range(1, 150)]
    )
    phase, _, _ = _phase_arrays(family)
    result: dict[str, Any] = {
        **dict(metadata),
        "max_reference_velocity_utilization": max_utilization,
        "p50_reference_velocity_utilization": float(np.quantile(utilization, 0.50)),
        "p95_reference_velocity_utilization": float(np.quantile(utilization, 0.95)),
        "minimum_normalized_joint_limit_margin": float(np.min(joint_margin)),
        "maximum_cartesian_position_step": float(np.max(position_step)),
        "maximum_cartesian_orientation_step": float(np.max(orientation_step)),
        "phase_boundaries": [
            {"start": start, "stop": stop, "phase": name}
            for start, stop, name in FAMILY_PHASES[family]
        ],
    }

    if family == SMOOTH_ORIENTATION_FAMILY:
        challenge = phase[1:] == "fast_orientation"
        challenge_median = float(np.median(orientation_step[challenge]))
        outer_median = float(np.median(orientation_step[~challenge]))
        ratio = challenge_median / max(outer_median, 1e-12)
        if challenge_median < 0.02 or ratio < 3.0:
            raise RuntimeError("fast-orientation phase lacks its preregistered contrast")
        result.update(
            challenge_orientation_step_median=challenge_median,
            outer_orientation_step_median=outer_median,
            challenge_to_outer_orientation_step_ratio=ratio,
        )
    elif family == SINGULAR_TRANSITION_FAMILY:
        sigma = np.asarray(
            [kinematics.min_singular_value(row) for row in reference], dtype=np.float64
        )
        challenge = phase == "near_singular"
        regular = np.isin(phase, ("regular_pre", "regular_post"))
        challenge_min = float(np.min(sigma[challenge]))
        regular_min = float(np.min(sigma[regular]))
        contrast = regular_min / max(challenge_min, 1e-12)
        if challenge_min > 0.01 or regular_min < 0.02 or contrast < 10.0:
            raise RuntimeError("near-singular phase lacks its preregistered contrast")
        result.update(
            near_singular_sigma_min=challenge_min,
            regular_sigma_min=regular_min,
            regular_to_near_singular_sigma_ratio=contrast,
        )
    elif family == LIMIT_TRANSITION_FAMILY:
        challenge = phase == "joint_limit_skim"
        outer = np.isin(phase, ("central_pre", "central_post"))
        challenge_min = float(np.min(joint_margin[challenge]))
        outer_min = float(np.min(joint_margin[outer]))
        if challenge_min > 0.03 or outer_min < 0.25:
            raise RuntimeError("joint-limit phase lacks its preregistered contrast")
        result.update(
            skim_joint_margin_min=challenge_min,
            central_joint_margin_min=outer_min,
        )
    elif family == CURVATURE_TRANSITION_FAMILY:
        challenge = phase[1:] == "high_curvature_high_speed"
        challenge_speed = float(np.median(utilization[challenge]))
        outer_speed = float(np.median(utilization[~challenge]))
        curvature = np.linalg.norm(np.diff(reference, n=2, axis=0), axis=1)
        curvature_challenge = phase[2:] == "high_curvature_high_speed"
        challenge_curvature = float(np.median(curvature[curvature_challenge]))
        outer_curvature = float(np.median(curvature[~curvature_challenge]))
        curvature_ratio = challenge_curvature / max(outer_curvature, 1e-12)
        if challenge_speed < 0.75 or outer_speed > 0.20 or curvature_ratio < 10.0:
            raise RuntimeError("high-curvature/high-speed phase lacks contrast")
        result.update(
            challenge_velocity_utilization_median=challenge_speed,
            outer_velocity_utilization_median=outer_speed,
            challenge_joint_curvature_median=challenge_curvature,
            outer_joint_curvature_median=outer_curvature,
            challenge_to_outer_curvature_ratio=curvature_ratio,
        )
    else:  # pragma: no cover - guarded by the public family contract
        raise RuntimeError(f"unsupported transition family: {family}")
    return result


def _generate_reference(
    kinematics: KinematicsModel,
    family: str,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if family == SMOOTH_ORIENTATION_FAMILY:
        reference = _harmonic_reference(kinematics, rng, orientation_only=True)
        metadata: dict[str, Any] = {}
    elif family == SINGULAR_TRANSITION_FAMILY:
        reference, metadata = _singular_transition_reference(kinematics, rng)
    elif family == LIMIT_TRANSITION_FAMILY:
        reference, metadata = _limit_transition_reference(kinematics, rng)
    elif family == CURVATURE_TRANSITION_FAMILY:
        reference = _harmonic_reference(kinematics, rng, orientation_only=False)
        metadata = {}
    else:
        raise ValueError(f"unknown transition family: {family!r}")
    audit = _geometric_audit(kinematics, reference, family, metadata)
    return reference, audit


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


def _generate_pool(
    kinematics: KinematicsModel,
    spec: FreshTrajectorySpec,
) -> tuple[
    QueryDataset,
    NDArray[np.str_],
    NDArray[np.int64],
    NDArray[np.bool_],
    dict[int, dict[str, Any]],
]:
    previous_rows: list[NDArray[np.float64]] = []
    reference_rows: list[NDArray[np.float64]] = []
    positions: list[NDArray[np.float64]] = []
    rotations: list[NDArray[np.float64]] = []
    categories: list[str] = []
    trajectory_ids: list[int] = []
    time_indices: list[int] = []
    phase_rows: list[str] = []
    phase_index_rows: list[int] = []
    boundary_rows: list[bool] = []
    generation: dict[int, dict[str, Any]] = {}

    for family_index, family in enumerate(FAMILIES):
        phase, phase_index, boundary = _phase_arrays(family)
        for path_index in range(spec.paths_per_family_pool):
            seed_material = [spec.pool_seed, family_index, path_index]
            rng = np.random.default_rng(np.random.SeedSequence(seed_material))
            reference, geometry = _generate_reference(kinematics, family, rng)
            previous = np.vstack([reference[0], reference[:-1]])
            poses = [kinematics.forward(row) for row in reference]
            trajectory_id = 300_000_000 + family_index * 1_000 + path_index
            previous_rows.extend(previous)
            reference_rows.extend(reference)
            positions.extend(pose.position for pose in poses)
            rotations.extend(pose.rotation for pose in poses)
            categories.extend([family] * spec.steps)
            trajectory_ids.extend([trajectory_id] * spec.steps)
            time_indices.extend(range(spec.steps))
            phase_rows.extend(phase.tolist())
            phase_index_rows.extend(phase_index.tolist())
            boundary_rows.extend(boundary.tolist())
            generation[trajectory_id] = {
                "family": family,
                "family_index": family_index,
                "path_index": path_index,
                "seed_sequence_material": seed_material,
                "geometry": geometry,
            }
    count = len(previous_rows)
    dataset = QueryDataset(
        previous_q=np.stack(previous_rows),
        target_position=np.stack(positions),
        target_rotation=np.stack(rotations),
        reference_q=np.stack(reference_rows),
        category=np.asarray(categories),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
        time_index=np.asarray(time_indices, dtype=np.int64),
    )
    return (
        dataset,
        np.asarray(phase_rows, dtype="U32"),
        np.asarray(phase_index_rows, dtype=np.int64),
        np.asarray(boundary_rows, dtype=bool),
        generation,
    )


def _build_role(
    *,
    spec: FreshTrajectorySpec,
    role: str,
    pool: QueryDataset,
    pool_phase: NDArray[np.str_],
    pool_phase_index: NDArray[np.int64],
    pool_boundary: NDArray[np.bool_],
    selected_ids: Sequence[int],
    query_hash_by_row: NDArray[np.str_],
    trajectory_uid_by_id: Mapping[int, str],
) -> FreshTrajectoryRole:
    _reject_test_named(role, name="role")
    indices: list[int] = []
    order: list[str] = []
    for trajectory_id in selected_ids:
        rows = np.flatnonzero(pool.trajectory_id == int(trajectory_id)).astype(np.int64)
        rows = rows[np.argsort(pool.time_index[rows], kind="stable")]
        if len(rows) != spec.steps or not np.array_equal(
            pool.time_index[rows], np.arange(spec.steps, dtype=np.int64)
        ):
            raise RuntimeError("selected trajectory is incomplete or out of order")
        indices.extend(rows.tolist())
        order.append(trajectory_uid_by_id[int(trajectory_id)])
    selected = np.asarray(indices, dtype=np.int64)
    subset = _subset(pool, selected)
    uid = np.asarray(
        [trajectory_uid_by_id[int(value)] for value in subset.trajectory_id],
        dtype="U64",
    )
    return FreshTrajectoryRole(
        robot=spec.robot,
        role=role,
        kinematics_identity=spec.kinematics_identity,
        dt=spec.dt,
        dataset=subset,
        source_query_hash=query_hash_by_row[selected],
        trajectory_uid=uid,
        trajectory_order=tuple(order),
        phase=pool_phase[selected],
        phase_index=pool_phase_index[selected],
        transition_boundary=pool_boundary[selected],
    )


def _role_audit(role: FreshTrajectoryRole) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        selected = role.dataset.category == family
        by_family[family] = {
            "trajectory_count": len(set(role.trajectory_uid[selected].tolist())),
            "frame_count": int(np.sum(selected)),
            "transition_boundary_frame_count": int(
                np.sum(role.transition_boundary[selected])
            ),
            "phase_frame_counts": {
                phase_name: int(np.sum(role.phase[selected] == phase_name))
                for _, _, phase_name in FAMILY_PHASES[family]
            },
        }
    query_values = role.source_query_hash.astype(str).tolist()
    trajectory_values = list(role.trajectory_order)
    return {
        "role": role.role,
        "frame_count": role.count,
        "unique_source_query_hash_count": len(set(query_values)),
        "trajectory_count": len(trajectory_values),
        "unique_trajectory_uid_count": len(set(trajectory_values)),
        "ordered_query_digest": _digest_strings(query_values, ordered=True),
        "query_set_digest": _digest_strings(query_values, ordered=False),
        "ordered_trajectory_digest": _digest_strings(trajectory_values, ordered=True),
        "trajectory_set_digest": _digest_strings(trajectory_values, ordered=False),
        "ordered_phase_digest": _digest_strings(role.phase.astype(str).tolist(), ordered=True),
        "by_family": by_family,
    }


def _pool_identity(
    spec: FreshTrajectorySpec,
    query_hashes: Sequence[str],
    trajectory_uids: Sequence[str],
    phases: Sequence[str],
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "hash_schema": HASH_SCHEMA,
        "trajectory_hash_schema": TRAJECTORY_HASH_SCHEMA,
        "robot": spec.robot,
        "kinematics_identity": spec.kinematics_identity,
        "pool_seed": spec.pool_seed,
        "split_seed": spec.split_seed,
        "paths_per_family_pool": spec.paths_per_family_pool,
        "paths_per_family_per_role": spec.paths_per_family_per_role,
        "steps": spec.steps,
        "dt": spec.dt,
        "query_set_digest": _digest_strings(query_hashes, ordered=False),
        "trajectory_set_digest": _digest_strings(trajectory_uids, ordered=False),
        "ordered_phase_digest": _digest_strings(phases, ordered=True),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generate_fresh_development_roles(
    kinematics: KinematicsModel,
    spec: FreshTrajectorySpec,
    *,
    source_registries: Sequence[AllowedSourceHashRegistry] = (),
    manifest_identities: Sequence[AllowedManifestIdentity] = (),
    seed_registries: Sequence[AllowedSeedRegistry] = (),
) -> tuple[FreshTrajectoryRole, FreshTrajectoryRole, dict[str, Any]]:
    """Generate, geometrically validate, and outcome-blind split a V7 pool."""

    seed_isolation = audit_seed_isolation(spec, seed_registries=seed_registries)
    pool, phase, phase_index, boundary, generation = _generate_pool(kinematics, spec)
    expected_frames = len(FAMILIES) * spec.paths_per_family_pool * spec.steps
    if len(pool) != expected_frames:
        raise RuntimeError(f"fresh pool has {len(pool)} frames, expected {expected_frames}")

    query_hash = source_query_hashes(
        pool,
        robot=spec.robot,
        dt=spec.dt,
        kinematics_identity=spec.kinematics_identity,
    )
    if len(set(query_hash.tolist())) != len(query_hash):
        raise RuntimeError("fresh V7 pool contains duplicate runtime-query hashes")

    trajectory_uid_by_id: dict[int, str] = {}
    family_by_id: dict[int, str] = {}
    for trajectory_id in np.unique(pool.trajectory_id).astype(np.int64):
        rows = np.flatnonzero(pool.trajectory_id == trajectory_id).astype(np.int64)
        rows = rows[np.argsort(pool.time_index[rows], kind="stable")]
        if len(rows) != spec.steps or not np.array_equal(
            pool.time_index[rows], np.arange(spec.steps, dtype=np.int64)
        ):
            raise RuntimeError("fresh V7 pool trajectory is incomplete")
        families = set(pool.category[rows].tolist())
        if len(families) != 1:
            raise RuntimeError("a fresh V7 trajectory crosses family boundaries")
        family = next(iter(families))
        family_by_id[int(trajectory_id)] = family
        trajectory_uid_by_id[int(trajectory_id)] = _trajectory_hash(
            robot=spec.robot,
            kinematics_identity=spec.kinematics_identity,
            public_family=family,
            query_hashes=query_hash[rows].tolist(),
            time_index=pool.time_index[rows],
            phase=phase[rows].tolist(),
        )
    if len(set(trajectory_uid_by_id.values())) != len(trajectory_uid_by_id):
        raise RuntimeError("fresh V7 pool contains duplicate trajectory UIDs")

    assignment: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ids_by_role: dict[str, list[int]] = {role: [] for role in ROLES}
    for family_index, family in enumerate(FAMILIES):
        family_ids = np.asarray(
            sorted(
                trajectory_id
                for trajectory_id, observed_family in family_by_id.items()
                if observed_family == family
            ),
            dtype=np.int64,
        )
        if len(family_ids) != spec.paths_per_family_pool:
            raise RuntimeError(f"{family} does not contain the frozen 20-path pool")
        rng = np.random.default_rng(
            np.random.SeedSequence([spec.split_seed, family_index])
        )
        shuffled = family_ids.copy()
        rng.shuffle(shuffled)
        selected_by_role = (
            shuffled[: spec.paths_per_family_per_role],
            shuffled[spec.paths_per_family_per_role :],
        )
        assignment[family] = {}
        for role, selected in zip(ROLES, selected_by_role, strict=True):
            if len(selected) != spec.paths_per_family_per_role:
                raise RuntimeError("fresh V7 pool cannot support a complete 10+10 split")
            ids_by_role[role].extend(int(value) for value in selected)
            assignment[family][role] = [
                {
                    "source_trajectory_id": int(value),
                    "trajectory_uid": trajectory_uid_by_id[int(value)],
                    **generation[int(value)],
                }
                for value in selected
            ]

    # Interleave families for timing while leaving membership unchanged.
    for role_index, role in enumerate(ROLES):
        per_family = {
            family: [
                int(record["source_trajectory_id"])
                for record in assignment[family][role]
            ]
            for family in FAMILIES
        }
        rng = np.random.default_rng(
            np.random.SeedSequence([spec.split_seed, 90 + role_index])
        )
        for values in per_family.values():
            rng.shuffle(values)
        ordered: list[int] = []
        for rank in range(spec.paths_per_family_per_role):
            family_order = list(FAMILIES)
            rng.shuffle(family_order)
            ordered.extend(per_family[family][rank] for family in family_order)
        ids_by_role[role] = ordered

    calibration = _build_role(
        spec=spec,
        role=CALIBRATION_ROLE,
        pool=pool,
        pool_phase=phase,
        pool_phase_index=phase_index,
        pool_boundary=boundary,
        selected_ids=ids_by_role[CALIBRATION_ROLE],
        query_hash_by_row=query_hash,
        trajectory_uid_by_id=trajectory_uid_by_id,
    )
    policy_validation = _build_role(
        spec=spec,
        role=POLICY_VALIDATION_ROLE,
        pool=pool,
        pool_phase=phase,
        pool_phase_index=phase_index,
        pool_boundary=boundary,
        selected_ids=ids_by_role[POLICY_VALIDATION_ROLE],
        query_hash_by_row=query_hash,
        trajectory_uid_by_id=trajectory_uid_by_id,
    )
    query_overlap = set(calibration.source_query_hash.tolist()) & set(
        policy_validation.source_query_hash.tolist()
    )
    trajectory_overlap = set(calibration.trajectory_order) & set(
        policy_validation.trajectory_order
    )
    if query_overlap or trajectory_overlap:
        raise RuntimeError(
            "fresh V7 roles overlap: "
            f"query={len(query_overlap)}, trajectory={len(trajectory_overlap)}"
        )

    pool_identity = _pool_identity(
        spec,
        query_hash.tolist(),
        tuple(trajectory_uid_by_id.values()),
        phase.tolist(),
    )
    isolation = audit_source_isolation(
        new_query_hashes=query_hash.tolist(),
        new_trajectory_uids=tuple(trajectory_uid_by_id.values()),
        new_manifest_identity=pool_identity,
        source_registries=source_registries,
        manifest_identities=manifest_identities,
    )
    audit = {
        "protocol": PROTOCOL,
        "generator": "generate_transition_rich_reference_trajectories",
        "generator_independent_from_temporal_v6": True,
        "generator_outcome_blind": True,
        "geometric_screening_only": True,
        "solver_or_verifier_screening_performed": False,
        "split_before_outcome_collection": True,
        "formal_test_data_opened": False,
        "filesystem_discovery_performed": False,
        "robot": spec.robot,
        "kinematics_identity": spec.kinematics_identity,
        "hash_schema": HASH_SCHEMA,
        "runtime_query_hash_implementation": "temporal_v6.runtime_query_hashes",
        "trajectory_hash_schema": TRAJECTORY_HASH_SCHEMA,
        "pool_seed": spec.pool_seed,
        "split_seed": spec.split_seed,
        "seed_registry": {
            "status": "frozen",
            "pool_seed": spec.pool_seed,
            "split_seed": spec.split_seed,
            "all_frozen_pool_seeds": dict(FROZEN_POOL_SEEDS),
            "all_frozen_split_seeds": dict(FROZEN_SPLIT_SEEDS),
        },
        "seed_isolation": seed_isolation,
        "families": list(FAMILIES),
        "family_phase_contract": {
            family: [
                {"start": start, "stop": stop, "phase": name}
                for start, stop, name in FAMILY_PHASES[family]
            ]
            for family in FAMILIES
        },
        "paths_per_family_pool": spec.paths_per_family_pool,
        "paths_per_family_per_role": spec.paths_per_family_per_role,
        "steps_per_trajectory": spec.steps,
        "dt": spec.dt,
        "pool_frame_count": len(pool),
        "pool_trajectory_count": len(trajectory_uid_by_id),
        "pool_unique_source_query_hash_count": len(set(query_hash.tolist())),
        "pool_unique_trajectory_uid_count": len(set(trajectory_uid_by_id.values())),
        "pool_manifest_identity_sha256": pool_identity,
        "roles": {
            CALIBRATION_ROLE: _role_audit(calibration),
            POLICY_VALIDATION_ROLE: _role_audit(policy_validation),
        },
        "family_assignment": assignment,
        "calibration_policy_validation_query_overlap_count": 0,
        "calibration_policy_validation_trajectory_overlap_count": 0,
        "policy_validation_outcomes_computed_during_split": False,
        "source_isolation": isolation,
    }
    return calibration, policy_validation, audit


def _role_payload(role: FreshTrajectoryRole) -> dict[str, Any]:
    return {
        "protocol": np.asarray([PROTOCOL], dtype="U64"),
        "robot": np.asarray([role.robot], dtype="U16"),
        "role": np.asarray([role.role], dtype="U48"),
        "kinematics_identity": np.asarray([role.kinematics_identity], dtype="U64"),
        "dt": np.asarray([role.dt], dtype=np.float64),
        "previous_q": role.dataset.previous_q,
        "target_position": role.dataset.target_position,
        "target_rotation": role.dataset.target_rotation,
        "reference_q": role.dataset.reference_q,
        "category": role.dataset.category,
        "expected_reachable": role.dataset.expected_reachable,
        "continuity_feasible": role.dataset.continuity_feasible,
        "trajectory_id": role.dataset.trajectory_id,
        "time_index": role.dataset.time_index,
        "source_query_hash": role.source_query_hash,
        "trajectory_uid": role.trajectory_uid,
        "trajectory_order": np.asarray(role.trajectory_order, dtype="U64"),
        "phase": role.phase,
        "phase_index": role.phase_index,
        "transition_boundary": role.transition_boundary,
    }


def save_trajectory_role(path: str | Path, role: FreshTrajectoryRole) -> dict[str, Any]:
    """Exclusively create one immutable role NPZ."""

    destination = Path(path)
    _reject_test_named(destination, name="trajectory role path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"trajectory role already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            np.savez_compressed(handle, **_role_payload(role))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _semantic_revalidation(
    role: FreshTrajectoryRole,
    kinematics: KinematicsModel | None,
) -> None:
    for uid, rows in role.groups():
        reference = role.dataset.reference_q[rows]
        expected_previous = np.vstack([reference[0], reference[:-1]])
        if not np.array_equal(role.dataset.previous_q[rows], expected_previous):
            raise RuntimeError(f"stored previous_q lineage changed for {uid}")
        if kinematics is not None:
            utilization = _velocity_utilization(kinematics, reference, dt=role.dt)
            if float(np.max(utilization)) > 1.0 + _VELOCITY_EPS:
                raise RuntimeError(f"stored trajectory {uid} exceeds velocity limits")
            family_values = set(role.dataset.category[rows].tolist())
            if len(family_values) != 1:
                raise RuntimeError(f"stored trajectory {uid} crosses family boundaries")
            _geometric_audit(
                kinematics,
                reference,
                next(iter(family_values)),
                {},
            )
            expected_positions: list[NDArray[np.float64]] = []
            expected_rotations: list[NDArray[np.float64]] = []
            for q in reference:
                pose = kinematics.forward(q)
                expected_positions.append(pose.position)
                expected_rotations.append(pose.rotation)
            if not np.allclose(
                role.dataset.target_position[rows],
                np.stack(expected_positions),
                rtol=0.0,
                atol=1e-12,
            ) or not np.allclose(
                role.dataset.target_rotation[rows],
                np.stack(expected_rotations),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"stored FK targets changed for {uid}")


def load_trajectory_role(
    path: str | Path,
    *,
    robot: str,
    expected_role: str,
    expected_artifact: Mapping[str, Any] | None = None,
    kinematics: KinematicsModel | None = None,
) -> FreshTrajectoryRole:
    """Hash-check and semantically revalidate one explicit development role."""

    source = Path(path)
    _reject_test_named(source, name="trajectory role path")
    _reject_test_named(expected_role, name="expected role")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"trajectory role is unavailable: {source}")
    if expected_artifact is not None and (
        source.stat().st_size != int(expected_artifact.get("size", -1))
        or _sha256_file(source) != str(expected_artifact.get("sha256", "")).casefold()
    ):
        raise RuntimeError("trajectory role artifact differs from its descriptor")
    required = {
        "protocol", "robot", "role", "kinematics_identity", "dt",
        "previous_q", "target_position", "target_rotation", "reference_q",
        "category", "expected_reachable", "continuity_feasible", "trajectory_id",
        "time_index", "source_query_hash", "trajectory_uid", "trajectory_order",
        "phase", "phase_index", "transition_boundary",
    }
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise RuntimeError("V7 trajectory role NPZ schema changed")
        stored_protocol = str(np.asarray(payload["protocol"]).reshape(-1)[0])
        stored_robot = str(np.asarray(payload["robot"]).reshape(-1)[0]).casefold()
        stored_role = str(np.asarray(payload["role"]).reshape(-1)[0])
        identity = str(np.asarray(payload["kinematics_identity"]).reshape(-1)[0])
        dt = float(np.asarray(payload["dt"]).reshape(-1)[0])
        if stored_protocol != PROTOCOL:
            raise RuntimeError("V7 trajectory role protocol changed")
        if stored_robot != str(robot).casefold() or stored_role != expected_role:
            raise RuntimeError("V7 trajectory role robot/role binding changed")
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
        role = FreshTrajectoryRole(
            robot=stored_robot,
            role=stored_role,
            kinematics_identity=identity,
            dt=dt,
            dataset=dataset,
            source_query_hash=payload["source_query_hash"].astype("U64", copy=True),
            trajectory_uid=payload["trajectory_uid"].astype("U64", copy=True),
            trajectory_order=tuple(payload["trajectory_order"].astype(str).tolist()),
            phase=payload["phase"].astype("U32", copy=True),
            phase_index=payload["phase_index"].astype(np.int64, copy=True),
            transition_boundary=payload["transition_boundary"].astype(bool, copy=True),
        )
    recomputed_query_hash = source_query_hashes(
        role.dataset,
        robot=role.robot,
        dt=role.dt,
        kinematics_identity=role.kinematics_identity,
    )
    if not np.array_equal(recomputed_query_hash, role.source_query_hash):
        raise RuntimeError("stored V7 source query hashes do not match role inputs")
    recomputed_uid = np.empty(role.count, dtype="U64")
    recomputed_order: list[str] = []
    for _, rows in role.groups():
        family_values = set(role.dataset.category[rows].tolist())
        if len(family_values) != 1:
            raise RuntimeError("loaded V7 trajectory crosses family boundaries")
        observed = _trajectory_hash(
            robot=role.robot,
            kinematics_identity=role.kinematics_identity,
            public_family=next(iter(family_values)),
            query_hashes=role.source_query_hash[rows].tolist(),
            time_index=role.dataset.time_index[rows],
            phase=role.phase[rows].tolist(),
        )
        recomputed_uid[rows] = observed
        recomputed_order.append(observed)
    if not np.array_equal(recomputed_uid, role.trajectory_uid) or tuple(
        recomputed_order
    ) != role.trajectory_order:
        raise RuntimeError("stored V7 trajectory UIDs do not match ordered inputs")
    _semantic_revalidation(role, kinematics)
    return role


def save_split_audit_manifest(path: str | Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    _reject_test_named(destination, name="split audit manifest path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"split audit manifest already exists: {destination}")
    if audit.get("protocol") != PROTOCOL:
        raise ValueError("V7 split audit protocol changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def load_split_audit_manifest(
    path: str | Path,
    *,
    expected_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    _reject_test_named(source, name="split audit manifest path")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"split audit manifest is unavailable: {source}")
    if expected_artifact is not None and (
        source.stat().st_size != int(expected_artifact.get("size", -1))
        or _sha256_file(source) != str(expected_artifact.get("sha256", "")).casefold()
    ):
        raise RuntimeError("V7 split audit manifest differs from its descriptor")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise RuntimeError("V7 split audit manifest protocol changed")
    if payload.get("formal_test_data_opened") is not False:
        raise RuntimeError("V7 split audit does not preserve the development boundary")
    return payload


__all__ = [
    "ALLOWED_SEED_SOURCE_CLASSES",
    "ALLOWED_SOURCE_CLASSES",
    "AllowedManifestIdentity",
    "AllowedSeedRegistry",
    "AllowedSourceHashRegistry",
    "CALIBRATION_ROLE",
    "CURVATURE_TRANSITION_FAMILY",
    "DT",
    "FAMILIES",
    "FAMILY_PHASES",
    "FROZEN_POOL_SEEDS",
    "FROZEN_SPLIT_SEEDS",
    "FreshTrajectoryRole",
    "FreshTrajectorySpec",
    "FreshTransitionTrajectoryRole",
    "FreshTransitionTrajectorySpec",
    "HASH_SCHEMA",
    "LIMIT_TRANSITION_FAMILY",
    "PATHS_PER_FAMILY_PER_ROLE",
    "PATHS_PER_FAMILY_POOL",
    "POLICY_VALIDATION_ROLE",
    "PROTOCOL",
    "ROLES",
    "SINGULAR_TRANSITION_FAMILY",
    "SMOOTH_ORIENTATION_FAMILY",
    "STEPS_PER_TRAJECTORY",
    "TRAJECTORY_HASH_SCHEMA",
    "audit_seed_isolation",
    "audit_source_isolation",
    "generate_fresh_development_roles",
    "load_allowed_source_hash_registry",
    "load_split_audit_manifest",
    "load_trajectory_role",
    "runtime_query_hashes",
    "save_split_audit_manifest",
    "save_trajectory_role",
    "source_hash_registry_from_dataset",
    "source_hash_registry_from_role",
    "source_query_hashes",
]
