"""Frozen, outcome-blind trajectories for the final fresh V4 evaluation.

This module owns only data identity.  It neither imports nor calls an IK
solver, verifier, learned model, or V5--V7 runtime.  The fresh workload is
generated from fixed seeds and is checked against explicitly enumerated V6/V7
development roles and the identity arrays of the old test_v3/test_v4 data.
Formal result records are never opened.

Two query digests are retained deliberately:

* ``source_query_hash`` reproduces the V6/V7 runtime-input hash schema;
* ``formal_query_sha256`` reproduces the test_v3/test_v4 byte contract.

The content trajectory UID is built from the ordered formal-query digests and
time indices, so trajectories remain comparable even when an older experiment
used a different protocol-specific UID schema.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import io
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


PROTOCOL = "fresh_transition_v4_final_evaluation_v1"
ROLE = "fresh_transition_evaluation"
IDENTITY_MANIFEST_SCHEMA = "fresh_transition_v4_identity_manifest_v1"
RUNTIME_QUERY_HASH_SCHEMA = "confik_source_query_runtime_inputs_v1"
FORMAL_QUERY_HASH_SCHEMA = "test_v3_v4_exact_query_bytes_v1"
TRAJECTORY_UID_SCHEMA = "confik_ordered_runtime_trajectory_content_v1"

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

ROBOTS = ("panda", "ur5e")
FROZEN_POOL_SEEDS = {"panda": 864_901, "ur5e": 864_902}
TRAJECTORIES_PER_FAMILY = 20
STEPS_PER_TRAJECTORY = 150
DT = 0.02
TRAJECTORY_ID_BASE = 400_000_000
EXPECTED_TRAJECTORIES = len(FAMILIES) * TRAJECTORIES_PER_FAMILY
EXPECTED_FRAMES = EXPECTED_TRAJECTORIES * STEPS_PER_TRAJECTORY

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LIMIT_EPS = 1e-10
_VELOCITY_EPS = 1e-9

_QUERY_IDENTITY_FIELDS = (
    "previous_q",
    "target_position",
    "target_rotation",
)
_TRAJECTORY_IDENTITY_FIELDS = ("category", "trajectory_id", "time_index")
_FRESH_NPZ_FIELDS = {
    "protocol",
    "robot",
    "role",
    "kinematics_identity",
    "dt",
    "pool_seed",
    "previous_q",
    "target_position",
    "target_rotation",
    "reference_q",
    "category",
    "expected_reachable",
    "continuity_feasible",
    "trajectory_id",
    "time_index",
    "source_query_hash",
    "formal_query_sha256",
    "trajectory_uid",
    "trajectory_order",
    "trajectory_seed",
    "phase",
    "phase_index",
    "transition_boundary",
}


def _require_hex64(value: str, *, name: str) -> str:
    normalized = str(value).casefold()
    if _HEX64.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"identity artifact is missing or a symlink: {path}")
    shown = path if relative_to is None else path.relative_to(relative_to)
    return {
        "path": str(shown),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _verify_artifact(path: Path, descriptor: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"frozen identity artifact is unavailable: {path}")
    if (
        path.stat().st_size != int(descriptor.get("size", -1))
        or _sha256_file(path) != str(descriptor.get("sha256", "")).casefold()
    ):
        raise RuntimeError(f"frozen identity artifact changed: {path}")


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"identity manifest is unavailable: {path}")

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(payload, dict):
        raise RuntimeError(f"identity manifest is not a JSON object: {path}")
    return payload


def _hash_field(digest: Any, name: str, value: bytes) -> None:
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "little", signed=False))
    digest.update(encoded)
    digest.update(len(value).to_bytes(8, "little", signed=False))
    digest.update(value)


def _canonical_float_bytes(values: Any) -> bytes:
    array = np.asarray(values, dtype="<f8", order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError("query identity cannot contain non-finite floats")
    shape = np.asarray(array.shape, dtype="<i8").tobytes()
    return len(array.shape).to_bytes(4, "little") + shape + array.tobytes(order="C")


def _canonical_int_bytes(values: Any) -> bytes:
    array = np.asarray(values, dtype="<i8", order="C")
    shape = np.asarray(array.shape, dtype="<i8").tobytes()
    return len(array.shape).to_bytes(4, "little") + shape + array.tobytes(order="C")


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest_strings(values: Sequence[str], *, ordered: bool) -> str:
    sequence = list(values) if ordered else sorted(set(values))
    digest = sha256()
    _hash_field(digest, "domain", b"ordered" if ordered else b"set")
    for value in sequence:
        _hash_field(digest, "value", str(value).encode("ascii"))
    return digest.hexdigest()


def runtime_query_hashes(
    previous_q: Any,
    target_position: Any,
    target_rotation: Any,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> NDArray[np.str_]:
    """Reproduce the V6/V7 canonical runtime-query identity contract."""

    normalized_robot = str(robot).casefold()
    if normalized_robot not in ROBOTS:
        raise ValueError(f"unsupported robot: {robot!r}")
    identity = _require_hex64(kinematics_identity, name="kinematics_identity")
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    previous = np.asarray(previous_q, dtype=np.float64)
    positions = np.asarray(target_position, dtype=np.float64)
    rotations = np.asarray(target_rotation, dtype=np.float64)
    count = previous.shape[0] if previous.ndim == 2 else -1
    if (
        previous.ndim != 2
        or positions.shape != (count, 3)
        or rotations.shape != (count, 3, 3)
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rotations))
    ):
        raise ValueError("runtime query arrays have invalid shapes or values")
    values: list[str] = []
    for index in range(count):
        digest = sha256()
        _hash_field(digest, "schema", RUNTIME_QUERY_HASH_SCHEMA.encode("utf-8"))
        _hash_field(digest, "robot", normalized_robot.encode("utf-8"))
        _hash_field(digest, "kinematics", identity.encode("ascii"))
        _hash_field(digest, "dt", _canonical_float_bytes(float(dt)))
        _hash_field(digest, "previous_q", _canonical_float_bytes(previous[index]))
        _hash_field(
            digest, "target_position", _canonical_float_bytes(positions[index])
        )
        _hash_field(
            digest, "target_rotation", _canonical_float_bytes(rotations[index])
        )
        values.append(digest.hexdigest())
    return np.asarray(values, dtype="U64")


def formal_query_hashes(
    previous_q: Any,
    target_position: Any,
    target_rotation: Any,
    *,
    dt: float,
) -> NDArray[np.str_]:
    """Reproduce the exact test_v3/test_v4 query-byte identity contract."""

    previous = np.asarray(previous_q, dtype=np.float64)
    positions = np.asarray(target_position, dtype=np.float64)
    rotations = np.asarray(target_rotation, dtype=np.float64)
    count = previous.shape[0] if previous.ndim == 2 else -1
    if (
        previous.ndim != 2
        or positions.shape != (count, 3)
        or rotations.shape != (count, 3, 3)
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rotations))
        or not np.isfinite(float(dt))
        or float(dt) <= 0.0
    ):
        raise ValueError("formal query identity arrays have invalid shapes or values")
    dt_bytes = np.asarray([dt], dtype=np.float64).tobytes()
    result: list[str] = []
    for index in range(count):
        digest = sha256()
        digest.update(np.ascontiguousarray(previous[index], dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(positions[index], dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(rotations[index], dtype=np.float64).tobytes())
        digest.update(dt_bytes)
        result.append(digest.hexdigest())
    return np.asarray(result, dtype="U64")


def trajectory_content_uid(
    *, robot: str, query_hashes: Sequence[str], time_index: Any
) -> str:
    """Return a protocol-independent UID for one ordered runtime trajectory."""

    normalized_robot = str(robot).casefold()
    if normalized_robot not in ROBOTS:
        raise ValueError(f"unsupported robot: {robot!r}")
    hashes = tuple(_require_hex64(value, name="query hash") for value in query_hashes)
    indices = np.asarray(time_index, dtype=np.int64)
    if len(hashes) != len(indices) or indices.ndim != 1 or not hashes:
        raise ValueError("trajectory UID inputs have incompatible shapes")
    digest = sha256()
    _hash_field(digest, "schema", TRAJECTORY_UID_SCHEMA.encode("utf-8"))
    _hash_field(digest, "robot", normalized_robot.encode("utf-8"))
    _hash_field(digest, "time_index", _canonical_int_bytes(indices))
    for value in hashes:
        _hash_field(digest, "formal_query_sha256", value.encode("ascii"))
    return digest.hexdigest()


def _phase_arrays(
    family: str,
) -> tuple[NDArray[np.str_], NDArray[np.int64], NDArray[np.bool_]]:
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
class FreshTransitionSpec:
    robot: str
    pool_seed: int
    kinematics_identity: str
    trajectories_per_family: int = TRAJECTORIES_PER_FAMILY
    steps: int = STEPS_PER_TRAJECTORY
    dt: float = DT

    def __post_init__(self) -> None:
        robot = str(self.robot).casefold()
        if robot not in ROBOTS:
            raise ValueError(f"unsupported fresh-transition robot: {self.robot!r}")
        if int(self.pool_seed) != FROZEN_POOL_SEEDS[robot]:
            raise ValueError(
                f"{robot} pool seed must remain frozen at {FROZEN_POOL_SEEDS[robot]}"
            )
        if (
            int(self.trajectories_per_family) != TRAJECTORIES_PER_FAMILY
            or int(self.steps) != STEPS_PER_TRAJECTORY
            or float(self.dt) != DT
        ):
            raise ValueError("the frozen 4x20 trajectory/150-frame/dt=0.02 contract changed")
        object.__setattr__(self, "robot", robot)
        object.__setattr__(
            self,
            "kinematics_identity",
            _require_hex64(self.kinematics_identity, name="kinematics_identity"),
        )

    @classmethod
    def frozen(
        cls, robot: str, *, kinematics_identity: str
    ) -> "FreshTransitionSpec":
        normalized = str(robot).casefold()
        if normalized not in FROZEN_POOL_SEEDS:
            raise ValueError(f"unsupported fresh-transition robot: {robot!r}")
        return cls(
            robot=normalized,
            pool_seed=FROZEN_POOL_SEEDS[normalized],
            kinematics_identity=kinematics_identity,
        )


@dataclass(frozen=True)
class PriorIdentityRegistry:
    name: str
    source_class: str
    artifact: Mapping[str, Any]
    arrays_read: tuple[str, ...]
    formal_query_hashes: tuple[str, ...]
    runtime_query_hashes: tuple[str, ...]
    trajectory_uids: tuple[str, ...]
    trajectory_ids: tuple[int, ...]
    seed_values: tuple[int, ...]
    performance_arrays_read: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("prior identity registry name cannot be empty")
        if self.source_class not in {"development", "formal_evaluation"}:
            raise ValueError("invalid prior identity source class")
        for field_name, values in (
            ("formal_query_hash", self.formal_query_hashes),
            ("runtime_query_hash", self.runtime_query_hashes),
            ("trajectory_uid", self.trajectory_uids),
        ):
            if any(_HEX64.fullmatch(str(value)) is None for value in values):
                raise ValueError(f"registry contains malformed {field_name}")
        if len(self.formal_query_hashes) != len(self.runtime_query_hashes):
            raise ValueError("registry query hash contracts cover different rows")
        if self.performance_arrays_read:
            raise ValueError("performance arrays are forbidden in an identity registry")


@dataclass(frozen=True)
class FreshTransitionDataset:
    robot: str
    kinematics_identity: str
    dt: float
    pool_seed: int
    dataset: QueryDataset
    source_query_hash: NDArray[np.str_]
    formal_query_sha256: NDArray[np.str_]
    trajectory_uid: NDArray[np.str_]
    trajectory_order: tuple[str, ...]
    trajectory_seed: NDArray[np.int64]
    phase: NDArray[np.str_]
    phase_index: NDArray[np.int64]
    transition_boundary: NDArray[np.bool_]
    role: str = ROLE

    def __post_init__(self) -> None:
        robot = str(self.robot).casefold()
        if robot not in ROBOTS:
            raise ValueError(f"unsupported robot: {self.robot!r}")
        if self.role != ROLE:
            raise ValueError(f"fresh dataset role must remain {ROLE!r}")
        if float(self.dt) != DT or int(self.pool_seed) != FROZEN_POOL_SEEDS[robot]:
            raise ValueError("fresh dataset timing or seed contract changed")
        identity = _require_hex64(
            self.kinematics_identity, name="kinematics_identity"
        )
        count = len(self.dataset)
        if count != EXPECTED_FRAMES:
            raise ValueError(f"fresh dataset has {count} frames, expected {EXPECTED_FRAMES}")
        source_hash = np.asarray(self.source_query_hash, dtype="U64")
        formal_hash = np.asarray(self.formal_query_sha256, dtype="U64")
        uid = np.asarray(self.trajectory_uid, dtype="U64")
        seed = np.asarray(self.trajectory_seed, dtype=np.int64)
        phase = np.asarray(self.phase, dtype="U32")
        phase_index = np.asarray(self.phase_index, dtype=np.int64)
        boundary = np.asarray(self.transition_boundary, dtype=bool)
        if any(
            value.shape != (count,)
            for value in (source_hash, formal_hash, uid, seed, phase, phase_index, boundary)
        ):
            raise ValueError("fresh trajectory metadata must have one value per frame")
        if any(_HEX64.fullmatch(value) is None for value in source_hash.tolist()):
            raise ValueError("fresh source query hashes are malformed")
        if any(_HEX64.fullmatch(value) is None for value in formal_hash.tolist()):
            raise ValueError("fresh formal query hashes are malformed")
        if any(_HEX64.fullmatch(value) is None for value in uid.tolist()):
            raise ValueError("fresh trajectory UIDs are malformed")
        if len(set(source_hash.tolist())) != count or len(set(formal_hash.tolist())) != count:
            raise ValueError("fresh query identities are not unique")
        order = tuple(str(value) for value in self.trajectory_order)
        if len(order) != EXPECTED_TRAJECTORIES or len(set(order)) != len(order):
            raise ValueError("trajectory_order must contain exactly 80 unique UIDs")
        if set(order) != set(uid.tolist()):
            raise ValueError("trajectory_order does not match per-frame UIDs")
        if set(self.dataset.category.tolist()) != set(FAMILIES):
            raise ValueError("fresh dataset family contract changed")
        if not np.all(self.dataset.expected_reachable) or not np.all(
            self.dataset.continuity_feasible
        ):
            raise ValueError("fresh reference trajectories must be known feasible")

        unique_numeric_ids: set[int] = set()
        unique_seeds: set[int] = set()
        for family in FAMILIES:
            selected = self.dataset.category == family
            if int(np.sum(selected)) != TRAJECTORIES_PER_FAMILY * STEPS_PER_TRAJECTORY:
                raise ValueError(f"family {family} does not contain 3000 frames")
            if len(set(uid[selected].tolist())) != TRAJECTORIES_PER_FAMILY:
                raise ValueError(f"family {family} does not contain 20 trajectories")

        for trajectory_uid_value in order:
            rows = np.flatnonzero(uid == trajectory_uid_value).astype(np.int64)
            if len(rows) != STEPS_PER_TRAJECTORY or not np.array_equal(
                self.dataset.time_index[rows],
                np.arange(STEPS_PER_TRAJECTORY, dtype=np.int64),
            ):
                raise ValueError("fresh trajectory is incomplete or out of order")
            families = set(self.dataset.category[rows].tolist())
            numeric_ids = set(self.dataset.trajectory_id[rows].tolist())
            seeds = set(seed[rows].tolist())
            if len(families) != 1 or len(numeric_ids) != 1 or len(seeds) != 1:
                raise ValueError("fresh trajectory identity changes within a trajectory")
            expected_phase, expected_index, expected_boundary = _phase_arrays(
                next(iter(families))
            )
            if (
                not np.array_equal(phase[rows], expected_phase)
                or not np.array_equal(phase_index[rows], expected_index)
                or not np.array_equal(boundary[rows], expected_boundary)
            ):
                raise ValueError("fresh trajectory phase contract changed")
            unique_numeric_ids.update(int(value) for value in numeric_ids)
            unique_seeds.update(int(value) for value in seeds)
        if len(unique_numeric_ids) != EXPECTED_TRAJECTORIES:
            raise ValueError("fresh numeric trajectory IDs are not unique")
        if len(unique_seeds) != EXPECTED_TRAJECTORIES:
            raise ValueError("fresh per-trajectory seeds are not unique")

        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "kinematics_identity", identity)
        object.__setattr__(self, "source_query_hash", source_hash)
        object.__setattr__(self, "formal_query_sha256", formal_hash)
        object.__setattr__(self, "trajectory_uid", uid)
        object.__setattr__(self, "trajectory_order", order)
        object.__setattr__(self, "trajectory_seed", seed)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "phase_index", phase_index)
        object.__setattr__(self, "transition_boundary", boundary)

    @property
    def count(self) -> int:
        return len(self.dataset)

    def groups(self) -> tuple[tuple[str, NDArray[np.int64]], ...]:
        return tuple(
            (
                uid,
                np.flatnonzero(self.trajectory_uid == uid).astype(np.int64),
            )
            for uid in self.trajectory_order
        )


# Short aliases used by the benchmark/runner modules.
FreshDataset = FreshTransitionDataset
FreshSpec = FreshTransitionSpec


def _closed_loop(
    anchor: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    direction: NDArray[np.float64],
    count: int,
    *,
    include_start: bool,
) -> NDArray[np.float64]:
    angles = (
        np.linspace(0.0, 2.0 * np.pi, count)
        if include_start
        else 2.0 * np.pi * np.arange(1, count + 1, dtype=np.float64) / count
    )
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
    center = kinematics.random_configuration(rng, margin=0.18)
    offset = rng.uniform(0.0, 2.0 * np.pi, size=kinematics.nq)
    challenge = np.zeros(STEPS_PER_TRAJECTORY, dtype=bool)
    if orientation_only:
        challenge[45:105] = True
        wrist = np.arange(kinematics.nq) >= kinematics.nq - min(3, kinematics.nq)
        low_rate = np.where(wrist, 0.05, 0.035)
        high_rate = np.where(wrist, rng.uniform(0.38, 0.52, kinematics.nq), 0.05)
        weights = np.where(wrist, 1.0, 0.15)
    else:
        challenge[40:110] = True
        low_rate = np.full(kinematics.nq, 0.035, dtype=np.float64)
        high_rate = rng.uniform(0.32, 0.55, size=kinematics.nq)
        weights = np.ones(kinematics.nq, dtype=np.float64)
    rates = low_rate[None, :] + challenge[:, None] * (
        high_rate - low_rate
    )[None, :]
    theta = offset[None, :] + np.cumsum(rates, axis=0)
    amplitude = (
        0.90
        * kinematics.limits.velocity
        * DT
        / (2.0 * np.sin(high_rate / 2.0))
        * weights
    )
    return np.asarray(center[None, :] + np.sin(theta) * amplitude[None, :])


def _singular_transition_reference(
    kinematics: KinematicsModel, rng: np.random.Generator
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
    kinematics: KinematicsModel, rng: np.random.Generator
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
        [np.min(kinematics.joint_margin(row)) for row in reference], dtype=np.float64
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
    }
    if family == SMOOTH_ORIENTATION_FAMILY:
        challenge = phase[1:] == "fast_orientation"
        challenge_median = float(np.median(orientation_step[challenge]))
        outer_median = float(np.median(orientation_step[~challenge]))
        ratio = challenge_median / max(outer_median, 1e-12)
        if challenge_median < 0.02 or ratio < 3.0:
            raise RuntimeError("fast-orientation phase lacks its frozen contrast")
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
            raise RuntimeError("near-singular phase lacks its frozen contrast")
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
            raise RuntimeError("joint-limit phase lacks its frozen contrast")
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
            raise RuntimeError("high-curvature/high-speed phase lacks its frozen contrast")
        result.update(
            challenge_velocity_utilization_median=challenge_speed,
            outer_velocity_utilization_median=outer_speed,
            challenge_joint_curvature_median=challenge_curvature,
            outer_joint_curvature_median=outer_curvature,
            challenge_to_outer_curvature_ratio=curvature_ratio,
        )
    else:  # pragma: no cover - guarded by constants
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
    return reference, _geometric_audit(kinematics, reference, family, metadata)


def generate_fresh_transition_dataset(
    kinematics: KinematicsModel,
    spec: FreshTransitionSpec,
) -> tuple[FreshTransitionDataset, dict[str, Any]]:
    """Generate the fixed 80-trajectory workload without method information.

    Generation and ordering consume only the frozen seed, robot kinematics and
    geometric family contract.  No IK solver, verifier, learned model, method
    output, or prior outcome is accepted by this API.
    """

    if int(getattr(kinematics, "nq")) <= 0:
        raise ValueError("kinematics has no active joints")
    records: dict[int, dict[str, Any]] = {}
    by_family: dict[str, list[int]] = {family: [] for family in FAMILIES}
    trajectory_seed_values: set[int] = set()
    for family_index, family in enumerate(FAMILIES):
        phase, phase_index, boundary = _phase_arrays(family)
        for path_index in range(TRAJECTORIES_PER_FAMILY):
            seed_material = [spec.pool_seed, family_index, path_index]
            seed_sequence = np.random.SeedSequence(seed_material)
            trajectory_seed = int(
                seed_sequence.generate_state(1, dtype=np.uint32)[0]
            )
            if trajectory_seed in trajectory_seed_values:
                raise RuntimeError("derived per-trajectory seed collision")
            trajectory_seed_values.add(trajectory_seed)
            rng = np.random.default_rng(seed_sequence)
            reference, geometry = _generate_reference(kinematics, family, rng)
            poses = [kinematics.forward(q) for q in reference]
            trajectory_id = (
                TRAJECTORY_ID_BASE + family_index * 1_000 + path_index
            )
            records[trajectory_id] = {
                "family": family,
                "family_index": family_index,
                "path_index": path_index,
                "trajectory_id": trajectory_id,
                "trajectory_seed": trajectory_seed,
                "seed_sequence_material": seed_material,
                "reference": reference,
                "previous": np.vstack([reference[0], reference[:-1]]),
                "positions": np.stack([pose.position for pose in poses]),
                "rotations": np.stack([pose.rotation for pose in poses]),
                "phase": phase,
                "phase_index": phase_index,
                "boundary": boundary,
                "geometry": geometry,
            }
            by_family[family].append(trajectory_id)

    # Deterministically interleave the four families without splitting or
    # selecting trajectories.  All 80 identities remain in the final role.
    order_rng = np.random.default_rng(
        np.random.SeedSequence([spec.pool_seed, 90])
    )
    for family in FAMILIES:
        order_rng.shuffle(by_family[family])
    ordered_ids: list[int] = []
    for rank in range(TRAJECTORIES_PER_FAMILY):
        family_order = list(FAMILIES)
        order_rng.shuffle(family_order)
        ordered_ids.extend(by_family[family][rank] for family in family_order)
    if len(ordered_ids) != EXPECTED_TRAJECTORIES or len(set(ordered_ids)) != len(
        ordered_ids
    ):
        raise RuntimeError("fresh trajectory order is incomplete or duplicated")

    previous_rows: list[NDArray[np.float64]] = []
    reference_rows: list[NDArray[np.float64]] = []
    position_rows: list[NDArray[np.float64]] = []
    rotation_rows: list[NDArray[np.float64]] = []
    categories: list[str] = []
    trajectory_ids: list[int] = []
    time_indices: list[int] = []
    trajectory_seeds: list[int] = []
    phases: list[str] = []
    phase_indices: list[int] = []
    boundaries: list[bool] = []
    for trajectory_id in ordered_ids:
        record = records[trajectory_id]
        previous_rows.extend(record["previous"])
        reference_rows.extend(record["reference"])
        position_rows.extend(record["positions"])
        rotation_rows.extend(record["rotations"])
        categories.extend([record["family"]] * STEPS_PER_TRAJECTORY)
        trajectory_ids.extend([trajectory_id] * STEPS_PER_TRAJECTORY)
        time_indices.extend(range(STEPS_PER_TRAJECTORY))
        trajectory_seeds.extend(
            [record["trajectory_seed"]] * STEPS_PER_TRAJECTORY
        )
        phases.extend(record["phase"].tolist())
        phase_indices.extend(record["phase_index"].tolist())
        boundaries.extend(record["boundary"].tolist())
    dataset = QueryDataset(
        previous_q=np.stack(previous_rows),
        target_position=np.stack(position_rows),
        target_rotation=np.stack(rotation_rows),
        reference_q=np.stack(reference_rows),
        category=np.asarray(categories),
        expected_reachable=np.ones(EXPECTED_FRAMES, dtype=bool),
        continuity_feasible=np.ones(EXPECTED_FRAMES, dtype=bool),
        trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
        time_index=np.asarray(time_indices, dtype=np.int64),
    )
    runtime_hash = runtime_query_hashes(
        dataset.previous_q,
        dataset.target_position,
        dataset.target_rotation,
        robot=spec.robot,
        dt=spec.dt,
        kinematics_identity=spec.kinematics_identity,
    )
    formal_hash = formal_query_hashes(
        dataset.previous_q,
        dataset.target_position,
        dataset.target_rotation,
        dt=spec.dt,
    )
    if len(set(runtime_hash.tolist())) != EXPECTED_FRAMES or len(
        set(formal_hash.tolist())
    ) != EXPECTED_FRAMES:
        raise RuntimeError("fresh workload contains duplicate query identities")
    uid_by_id: dict[int, str] = {}
    uid_rows = np.empty(EXPECTED_FRAMES, dtype="U64")
    trajectory_order: list[str] = []
    for trajectory_id in ordered_ids:
        rows = np.flatnonzero(dataset.trajectory_id == trajectory_id).astype(np.int64)
        uid = trajectory_content_uid(
            robot=spec.robot,
            query_hashes=formal_hash[rows].tolist(),
            time_index=dataset.time_index[rows],
        )
        uid_by_id[trajectory_id] = uid
        uid_rows[rows] = uid
        trajectory_order.append(uid)
    if len(set(trajectory_order)) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("fresh workload contains duplicate trajectory UIDs")

    fresh = FreshTransitionDataset(
        robot=spec.robot,
        kinematics_identity=spec.kinematics_identity,
        dt=spec.dt,
        pool_seed=spec.pool_seed,
        dataset=dataset,
        source_query_hash=runtime_hash,
        formal_query_sha256=formal_hash,
        trajectory_uid=uid_rows,
        trajectory_order=tuple(trajectory_order),
        trajectory_seed=np.asarray(trajectory_seeds, dtype=np.int64),
        phase=np.asarray(phases, dtype="U32"),
        phase_index=np.asarray(phase_indices, dtype=np.int64),
        transition_boundary=np.asarray(boundaries, dtype=bool),
    )
    generation_records = []
    for trajectory_id in ordered_ids:
        record = records[trajectory_id]
        generation_records.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_uid": uid_by_id[trajectory_id],
                "family": record["family"],
                "family_index": record["family_index"],
                "path_index": record["path_index"],
                "trajectory_seed": record["trajectory_seed"],
                "seed_sequence_material": record["seed_sequence_material"],
                "geometry": record["geometry"],
            }
        )
    audit = {
        "protocol": PROTOCOL,
        "role": ROLE,
        "robot": spec.robot,
        "kinematics_identity": spec.kinematics_identity,
        "generator": "fresh_transition_v4_test.data.generate_fresh_transition_dataset",
        "generator_outcome_blind": True,
        "geometric_screening_only": True,
        "solver_calls": 0,
        "verifier_calls": 0,
        "learned_model_calls": 0,
        "prior_performance_fields_read": 0,
        "pool_seed": spec.pool_seed,
        "trajectory_seed_derivation": "numpy.SeedSequence([pool_seed,family_index,path_index])",
        "ordering_seed_material": [spec.pool_seed, 90],
        "dt": spec.dt,
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "frame_count": EXPECTED_FRAMES,
        "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        "families": {
            family: {
                "trajectory_count": TRAJECTORIES_PER_FAMILY,
                "frame_count": TRAJECTORIES_PER_FAMILY * STEPS_PER_TRAJECTORY,
                "phase_contract": [
                    {"start": start, "stop": stop, "phase": name}
                    for start, stop, name in FAMILY_PHASES[family]
                ],
            }
            for family in FAMILIES
        },
        "ordered_trajectory_ids": ordered_ids,
        "generation_records": generation_records,
    }
    return fresh, audit


def _semantic_revalidation(
    fresh: FreshTransitionDataset, kinematics: KinematicsModel | None
) -> None:
    recomputed_runtime = runtime_query_hashes(
        fresh.dataset.previous_q,
        fresh.dataset.target_position,
        fresh.dataset.target_rotation,
        robot=fresh.robot,
        dt=fresh.dt,
        kinematics_identity=fresh.kinematics_identity,
    )
    recomputed_formal = formal_query_hashes(
        fresh.dataset.previous_q,
        fresh.dataset.target_position,
        fresh.dataset.target_rotation,
        dt=fresh.dt,
    )
    if not np.array_equal(recomputed_runtime, fresh.source_query_hash):
        raise RuntimeError("stored runtime query hashes do not match fresh inputs")
    if not np.array_equal(recomputed_formal, fresh.formal_query_sha256):
        raise RuntimeError("stored formal query hashes do not match fresh inputs")
    recomputed_uid = np.empty(fresh.count, dtype="U64")
    recomputed_order: list[str] = []
    for _, rows in fresh.groups():
        uid = trajectory_content_uid(
            robot=fresh.robot,
            query_hashes=recomputed_formal[rows].tolist(),
            time_index=fresh.dataset.time_index[rows],
        )
        recomputed_uid[rows] = uid
        recomputed_order.append(uid)
        reference = fresh.dataset.reference_q[rows]
        if not np.array_equal(
            fresh.dataset.previous_q[rows], np.vstack([reference[0], reference[:-1]])
        ):
            raise RuntimeError(f"stored previous-state lineage changed for {uid}")
        if kinematics is not None:
            family_values = set(fresh.dataset.category[rows].tolist())
            if len(family_values) != 1:
                raise RuntimeError(f"stored trajectory {uid} crosses families")
            _geometric_audit(
                kinematics, reference, next(iter(family_values)), {}
            )
            poses = [kinematics.forward(q) for q in reference]
            if not np.allclose(
                fresh.dataset.target_position[rows],
                np.stack([pose.position for pose in poses]),
                rtol=0.0,
                atol=1e-12,
            ) or not np.allclose(
                fresh.dataset.target_rotation[rows],
                np.stack([pose.rotation for pose in poses]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"stored FK targets changed for {uid}")
    if not np.array_equal(recomputed_uid, fresh.trajectory_uid) or tuple(
        recomputed_order
    ) != fresh.trajectory_order:
        raise RuntimeError("stored trajectory UIDs do not match fresh inputs")


_QUERY_DATASET_SCHEMA = {
    "previous_q",
    "target_position",
    "target_rotation",
    "reference_q",
    "category",
    "expected_reachable",
    "continuity_feasible",
    "trajectory_id",
    "time_index",
}
_LEGACY_V6_ROLE_SCHEMA = _QUERY_DATASET_SCHEMA | {
    "trajectory_uid",
    "trajectory_order",
    "role",
}
_EVENT_V6_ROLE_SCHEMA = _QUERY_DATASET_SCHEMA | {
    "protocol",
    "robot",
    "role",
    "kinematics_identity",
    "dt",
    "source_query_hash",
    "trajectory_uid",
    "trajectory_order",
}
_V7_ROLE_SCHEMA = _EVENT_V6_ROLE_SCHEMA | {
    "phase",
    "phase_index",
    "transition_boundary",
}


def _registry_from_npz(
    *,
    path: Path,
    descriptor: Mapping[str, Any],
    name: str,
    source_class: str,
    robot: str,
    dt: float,
    kinematics_identity: str,
    expected_schema: set[str],
    expected_count: int,
    trajectory_selector: str,
    seed_values: Sequence[int],
    verify_stored_runtime_hash: bool,
) -> PriorIdentityRegistry:
    """Read identity fields only from one hash-bound NPZ.

    ``trajectory_selector`` is one of ``all``, ``category_prefix`` or
    ``none``.  No reference joints, labels, method outcomes, latency, FEV, or
    success arrays are materialized.
    """

    _verify_artifact(path, descriptor)
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected_schema:
            raise RuntimeError(f"prior identity NPZ schema changed: {path}")
        previous = np.asarray(payload["previous_q"], dtype=np.float64)
        position = np.asarray(payload["target_position"], dtype=np.float64)
        rotation = np.asarray(payload["target_rotation"], dtype=np.float64)
        category = np.asarray(payload["category"]).astype(str)
        trajectory_id = np.asarray(payload["trajectory_id"], dtype=np.int64)
        time_index = np.asarray(payload["time_index"], dtype=np.int64)
        stored_runtime_hash = (
            np.asarray(payload["source_query_hash"]).astype(str)
            if verify_stored_runtime_hash
            else None
        )
        stored_trajectory_uid = (
            np.asarray(payload["trajectory_uid"]).astype(str)
            if "trajectory_uid" in expected_schema
            else None
        )
        stored_trajectory_order = (
            np.asarray(payload["trajectory_order"]).astype(str)
            if "trajectory_order" in expected_schema
            else None
        )
    count = len(previous)
    if (
        count != expected_count
        or previous.ndim != 2
        or position.shape != (count, 3)
        or rotation.shape != (count, 3, 3)
        or category.shape != (count,)
        or trajectory_id.shape != (count,)
        or time_index.shape != (count,)
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(rotation))
    ):
        raise RuntimeError(f"prior identity arrays have invalid shapes: {path}")
    formal_hash = formal_query_hashes(previous, position, rotation, dt=dt)
    runtime_hash = runtime_query_hashes(
        previous,
        position,
        rotation,
        robot=robot,
        dt=dt,
        kinematics_identity=kinematics_identity,
    )
    if stored_runtime_hash is not None and not np.array_equal(
        stored_runtime_hash, runtime_hash
    ):
        raise RuntimeError(f"stored V6/V7 query hashes changed: {path}")
    if stored_trajectory_uid is not None:
        if (
            stored_trajectory_uid.shape != (count,)
            or any(_HEX64.fullmatch(value) is None for value in stored_trajectory_uid)
            or stored_trajectory_order is None
            or len(set(stored_trajectory_order.tolist()))
            != len(stored_trajectory_order)
            or set(stored_trajectory_order.tolist())
            != set(stored_trajectory_uid.tolist())
        ):
            raise RuntimeError(f"stored V6/V7 trajectory identity changed: {path}")

    if trajectory_selector == "all":
        selected = np.ones(count, dtype=bool)
    elif trajectory_selector == "category_prefix":
        selected = np.char.startswith(category, "trajectory_")
    elif trajectory_selector == "none":
        selected = np.zeros(count, dtype=bool)
    else:
        raise ValueError(f"unsupported trajectory selector: {trajectory_selector}")
    selected_ids = sorted(set(trajectory_id[selected].tolist()))
    trajectory_uids: list[str] = []
    for numeric_id in selected_ids:
        rows = np.flatnonzero(selected & (trajectory_id == numeric_id)).astype(np.int64)
        rows = rows[np.argsort(time_index[rows], kind="stable")]
        if len(rows) != STEPS_PER_TRAJECTORY or not np.array_equal(
            time_index[rows], np.arange(STEPS_PER_TRAJECTORY, dtype=np.int64)
        ):
            raise RuntimeError(f"prior trajectory is incomplete in {path}: {numeric_id}")
        trajectory_uids.append(
            trajectory_content_uid(
                robot=robot,
                query_hashes=formal_hash[rows].tolist(),
                time_index=time_index[rows],
            )
        )
    if len(set(trajectory_uids)) != len(trajectory_uids):
        raise RuntimeError(f"prior source contains duplicate trajectory content: {path}")
    arrays_read = [*_QUERY_IDENTITY_FIELDS, *_TRAJECTORY_IDENTITY_FIELDS]
    if verify_stored_runtime_hash:
        arrays_read.append("source_query_hash")
    if stored_trajectory_uid is not None:
        arrays_read.extend(("trajectory_uid", "trajectory_order"))
    return PriorIdentityRegistry(
        name=name,
        source_class=source_class,
        artifact=_artifact(path),
        arrays_read=tuple(arrays_read),
        formal_query_hashes=tuple(formal_hash.tolist()),
        runtime_query_hashes=tuple(runtime_hash.tolist()),
        trajectory_uids=tuple(trajectory_uids),
        trajectory_ids=tuple(int(value) for value in selected_ids),
        seed_values=tuple(sorted(set(int(value) for value in seed_values))),
        performance_arrays_read=False,
    )


def _flatten_ints(value: Any) -> tuple[int, ...]:
    result: list[int] = []
    if isinstance(value, Mapping):
        for child in value.values():
            result.extend(_flatten_ints(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_flatten_ints(child))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        result.append(int(value))
    return tuple(result)


def _relative_descriptor(
    workspace: Path, path: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace)),
        "sha256": str(descriptor["sha256"]).casefold(),
        "size": int(descriptor["size"]),
    }


def build_prior_identity_registries(
    workspace: str | Path,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> tuple[tuple[PriorIdentityRegistry, ...], dict[str, Any]]:
    """Load the exhaustive, explicit prior identity boundary for one robot.

    Exactly six V6/V7 development role files, the old test_v3 identity data,
    and all four old test_v4 role datasets are consumed.  Formal NPZ access is
    restricted to the six identity arrays named in
    ``_QUERY_IDENTITY_FIELDS + _TRAJECTORY_IDENTITY_FIELDS``.  Formal
    performance/checkpoint/result files are not opened.
    """

    root = Path(workspace).resolve()
    normalized_robot = str(robot).casefold()
    if normalized_robot not in ROBOTS:
        raise ValueError(f"unsupported robot: {robot!r}")
    identity = _require_hex64(kinematics_identity, name="kinematics_identity")
    if float(dt) != DT:
        raise ValueError(f"fresh-transition dt must remain {DT}")
    registries: list[PriorIdentityRegistry] = []
    manifest_artifacts: dict[str, Any] = {}

    development_specs = (
        (
            "temporal_v6_pilot",
            "temporal_v6_development_pilot_v1",
            _LEGACY_V6_ROLE_SCHEMA,
            False,
            "legacy",
        ),
        (
            "temporal_event_v6_pilot",
            "temporal_event_v6_development_pilot_v1",
            _EVENT_V6_ROLE_SCHEMA,
            True,
            "robots",
        ),
        (
            "anchored_temporal_v7_pilot",
            "anchored_temporal_v7_development_pilot_v1",
            _V7_ROLE_SCHEMA,
            True,
            "robots",
        ),
    )
    for directory, expected_protocol, schema, stored_hash, split_shape in development_specs:
        source_root = root / "outputs" / directory
        run_path = source_root / "run_manifest.json"
        split_path = source_root / "trajectory_split_manifest.json"
        run_manifest = _strict_json(run_path)
        split_manifest = _strict_json(split_path)
        if run_manifest.get("protocol") != expected_protocol:
            raise RuntimeError(f"{directory} run-manifest protocol changed")
        if run_manifest.get("formal_test_started", False) is not False:
            raise RuntimeError(f"{directory} crossed its development boundary")
        if run_manifest.get("formal_test_data_opened", False) is not False:
            raise RuntimeError(f"{directory} opened formal-test data")
        if directory == "temporal_v6_pilot" and run_manifest.get(
            "test_data_loaded"
        ) is not False:
            raise RuntimeError("legacy Temporal V6 test-data boundary changed")
        robot_split = (
            split_manifest[normalized_robot]
            if split_shape == "legacy"
            else split_manifest["robots"][normalized_robot]
        )
        pool_seed = int(robot_split["pool_seed"])
        split_seed = int(robot_split["split_seed"])
        if int(robot_split["steps_per_trajectory"]) != STEPS_PER_TRAJECTORY:
            raise RuntimeError(f"{directory} trajectory length changed")
        artifacts = run_manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"{directory} lacks an artifact manifest")
        for short_role in ("calibration", "policy_validation"):
            filename = f"{normalized_robot}_trajectory_{short_role}.npz"
            descriptor = artifacts.get(filename)
            if not isinstance(descriptor, Mapping):
                raise RuntimeError(f"{directory} lacks descriptor {filename}")
            path = source_root / filename
            registries.append(
                _registry_from_npz(
                    path=path,
                    descriptor=descriptor,
                    name=f"{directory}/{normalized_robot}/{short_role}",
                    source_class="development",
                    robot=normalized_robot,
                    dt=dt,
                    kinematics_identity=identity,
                    expected_schema=schema,
                    expected_count=40 * STEPS_PER_TRAJECTORY,
                    trajectory_selector="all",
                    seed_values=(pool_seed, split_seed),
                    verify_stored_runtime_hash=stored_hash,
                )
            )
        manifest_artifacts[str(run_path.relative_to(root))] = _artifact(
            run_path, relative_to=root
        )
        manifest_artifacts[str(split_path.relative_to(root))] = _artifact(
            split_path, relative_to=root
        )

    # test_v3: its dataset manifest is identity-only and contains the frozen
    # dataset descriptor, generation seeds and a declaration that no solver
    # performance was inspected before the data were frozen.
    v3_root = root / "outputs" / "test_v3_aggregate"
    v3_manifest_path = v3_root / "test_v3_dataset_manifest.json"
    v3_manifest = _strict_json(v3_manifest_path)
    if (
        v3_manifest.get("protocol") != "test_v3 locked dataset manifest"
        or v3_manifest.get("method_outputs_inspected_before_freeze") is not False
        or v3_manifest.get("solver_performance_inspected_before_freeze") is not False
    ):
        raise RuntimeError("old test_v3 identity manifest boundary changed")
    v3_record = v3_manifest["datasets"][normalized_robot]
    v3_path = Path(str(v3_record["path"]))
    if not v3_path.is_absolute():
        v3_path = root / v3_path
    v3_path = v3_path.resolve()
    expected_v3_path = (
        v3_root / "datasets" / f"{normalized_robot}_test_v3_queries.npz"
    ).resolve()
    if v3_path != expected_v3_path:
        raise RuntimeError("old test_v3 identity path changed")
    registries.append(
        _registry_from_npz(
            path=v3_path,
            descriptor=v3_record,
            name=f"test_v3/{normalized_robot}/all_queries_identity_only",
            source_class="formal_evaluation",
            robot=normalized_robot,
            dt=dt,
            kinematics_identity=identity,
            expected_schema=_QUERY_DATASET_SCHEMA,
            expected_count=18_000,
            trajectory_selector="category_prefix",
            seed_values=_flatten_ints(v3_record["generation_seeds"]),
            verify_stored_runtime_hash=False,
        )
    )
    manifest_artifacts[str(v3_manifest_path.relative_to(root))] = _artifact(
        v3_manifest_path, relative_to=root
    )

    # test_v4: read only its identity-only dataset manifest and the four frozen
    # role NPZs.  No formal checkpoint or result record is reachable here.
    v4_root = root / "outputs" / ".test_v4_aggregate.incomplete"
    v4_manifest_path = v4_root / "test_v4_dataset_manifest.json"
    v4_manifest = _strict_json(v4_manifest_path)
    if (
        v4_manifest.get("protocol") != "test_v4 locked fresh dataset manifest"
        or v4_manifest.get("method_outputs_inspected_before_freeze") is not False
        or v4_manifest.get("old_test_performance_results_read") is not False
    ):
        raise RuntimeError("old test_v4 identity manifest boundary changed")
    v4_robot = v4_manifest["robots"][normalized_robot]
    v4_seeds = v4_robot["seeds"]
    for role_name in ("id_points", "id_trajectories", "ood_points", "ood_trajectories"):
        descriptor = v4_robot["roles"][role_name]
        path = (v4_root / str(descriptor["path"])).resolve()
        expected_path = (v4_root / "datasets" / f"{normalized_robot}_{role_name}.npz").resolve()
        if path != expected_path:
            raise RuntimeError(f"old test_v4 {role_name} identity path changed")
        expected_count = int(descriptor["query_count"])
        registry = _registry_from_npz(
            path=path,
            descriptor=descriptor,
            name=f"test_v4/{normalized_robot}/{role_name}_identity_only",
            source_class="formal_evaluation",
            robot=normalized_robot,
            dt=dt,
            kinematics_identity=identity,
            expected_schema=_QUERY_DATASET_SCHEMA,
            expected_count=expected_count,
            trajectory_selector=("all" if role_name.endswith("trajectories") else "none"),
            seed_values=_flatten_ints(v4_seeds[role_name]),
            verify_stored_runtime_hash=False,
        )
        # Bind the identity hashes to the exact digests frozen by test_v4.
        formal_values = list(registry.formal_query_hashes)
        if (
            _json_digest(formal_values)
            != str(descriptor["ordered_query_sha256_digest"])
            or _json_digest(sorted(formal_values))
            != str(descriptor["query_sha256_set_digest"])
        ):
            raise RuntimeError(f"old test_v4 {role_name} query identity changed")
        registries.append(registry)
    manifest_artifacts[str(v4_manifest_path.relative_to(root))] = _artifact(
        v4_manifest_path, relative_to=root
    )

    names = [registry.name for registry in registries]
    source_classes = Counter(registry.source_class for registry in registries)
    if (
        len(registries) != 11
        or len(set(names)) != 11
        or source_classes != Counter({"development": 6, "formal_evaluation": 5})
    ):
        raise RuntimeError("prior identity registry coverage is incomplete")
    formal = [r for r in registries if r.source_class == "formal_evaluation"]
    ledger = {
        "protocol": PROTOCOL,
        "robot": normalized_robot,
        "identity_only_access": True,
        "formal_identity_files_opened": len(formal),
        "development_identity_files_opened": len(registries) - len(formal),
        "performance_files_opened": 0,
        "performance_arrays_read": 0,
        "formal_allowed_arrays": [
            *_QUERY_IDENTITY_FIELDS,
            *_TRAJECTORY_IDENTITY_FIELDS,
        ],
        "formal_arrays_read_by_source": {
            registry.name: list(registry.arrays_read) for registry in formal
        },
        "source_count": len(registries),
        "source_class_counts": dict(sorted(source_classes.items())),
        "source_manifests": manifest_artifacts,
        "formal_result_roots_opened": [],
        "formal_checkpoint_files_opened": 0,
        "claim_boundary": (
            "old formal access is limited to frozen dataset identity manifests and "
            "the previous_q/target pose/category/trajectory_id/time_index arrays; "
            "no success, latency, FEV, command, probability, or method output is read"
        ),
    }
    return tuple(registries), ledger


def audit_freshness(
    fresh: FreshTransitionDataset,
    registries: Sequence[PriorIdentityRegistry],
    *,
    access_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every declared prior identity is disjoint."""

    expected_names = {
        f"{directory}/{fresh.robot}/{role}"
        for directory in (
            "temporal_v6_pilot",
            "temporal_event_v6_pilot",
            "anchored_temporal_v7_pilot",
        )
        for role in ("calibration", "policy_validation")
    } | {
        f"test_v3/{fresh.robot}/all_queries_identity_only",
        *(
            f"test_v4/{fresh.robot}/{role}_identity_only"
            for role in (
                "id_points",
                "id_trajectories",
                "ood_points",
                "ood_trajectories",
            )
        ),
    }
    observed_names = [registry.name for registry in registries]
    if len(registries) != 11 or set(observed_names) != expected_names or len(
        observed_names
    ) != len(set(observed_names)):
        raise RuntimeError("freshness audit did not receive the exhaustive prior registry")
    if (
        access_ledger.get("protocol") != PROTOCOL
        or access_ledger.get("robot") != fresh.robot
        or access_ledger.get("identity_only_access") is not True
        or int(access_ledger.get("formal_identity_files_opened", -1)) != 5
        or int(access_ledger.get("development_identity_files_opened", -1)) != 6
        or int(access_ledger.get("performance_files_opened", -1)) != 0
        or int(access_ledger.get("performance_arrays_read", -1)) != 0
    ):
        raise RuntimeError("prior identity access ledger is incomplete")

    fresh_formal = set(fresh.formal_query_sha256.tolist())
    fresh_runtime = set(fresh.source_query_hash.tolist())
    fresh_uids = set(fresh.trajectory_order)
    fresh_numeric_ids = set(int(value) for value in fresh.dataset.trajectory_id.tolist())
    fresh_seed_values = {int(fresh.pool_seed)} | set(
        int(value) for value in fresh.trajectory_seed.tolist()
    )
    if (
        len(fresh_formal) != fresh.count
        or len(fresh_runtime) != fresh.count
        or len(fresh_uids) != EXPECTED_TRAJECTORIES
        or len(fresh_numeric_ids) != EXPECTED_TRAJECTORIES
        or len(fresh_seed_values) != EXPECTED_TRAJECTORIES + 1
    ):
        raise RuntimeError("fresh workload contains an internal identity collision")

    records: list[dict[str, Any]] = []
    for registry in registries:
        formal_overlap = sorted(fresh_formal & set(registry.formal_query_hashes))
        runtime_overlap = sorted(fresh_runtime & set(registry.runtime_query_hashes))
        uid_overlap = sorted(fresh_uids & set(registry.trajectory_uids))
        numeric_id_overlap = sorted(fresh_numeric_ids & set(registry.trajectory_ids))
        seed_overlap = sorted(fresh_seed_values & set(registry.seed_values))
        if (
            formal_overlap
            or runtime_overlap
            or uid_overlap
            or numeric_id_overlap
            or seed_overlap
        ):
            raise RuntimeError(
                f"fresh workload overlaps {registry.name}: "
                f"formal_query={len(formal_overlap)}, runtime_query={len(runtime_overlap)}, "
                f"trajectory_uid={len(uid_overlap)}, trajectory_id={len(numeric_id_overlap)}, "
                f"seed={len(seed_overlap)}"
            )
        records.append(
            {
                "name": registry.name,
                "source_class": registry.source_class,
                "artifact": dict(registry.artifact),
                "arrays_read": list(registry.arrays_read),
                "performance_arrays_read": False,
                "query_count": len(registry.formal_query_hashes),
                "unique_formal_query_hash_count": len(
                    set(registry.formal_query_hashes)
                ),
                "unique_runtime_query_hash_count": len(
                    set(registry.runtime_query_hashes)
                ),
                "trajectory_uid_count": len(registry.trajectory_uids),
                "trajectory_id_count": len(registry.trajectory_ids),
                "declared_seed_count": len(registry.seed_values),
                "formal_query_hash_set_digest": _digest_strings(
                    registry.formal_query_hashes, ordered=False
                ),
                "runtime_query_hash_set_digest": _digest_strings(
                    registry.runtime_query_hashes, ordered=False
                ),
                "trajectory_uid_set_digest": _digest_strings(
                    registry.trajectory_uids, ordered=False
                ),
                "formal_query_overlap_count": 0,
                "runtime_query_overlap_count": 0,
                "trajectory_uid_overlap_count": 0,
                "trajectory_id_overlap_count": 0,
                "seed_overlap_count": 0,
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "pass",
        "all_isolation_checks_pass": True,
        "robot": fresh.robot,
        "fresh_frame_count": fresh.count,
        "fresh_trajectory_count": len(fresh.trajectory_order),
        "fresh_pool_seed": fresh.pool_seed,
        "fresh_trajectory_seed_count": len(set(fresh.trajectory_seed.tolist())),
        "source_count": len(registries),
        "source_class_counts": dict(
            sorted(Counter(r.source_class for r in registries).items())
        ),
        "formal_identity_files_opened": 5,
        "development_identity_files_opened": 6,
        "performance_files_opened": 0,
        "performance_arrays_read": 0,
        "query_hash_contracts": [
            RUNTIME_QUERY_HASH_SCHEMA,
            FORMAL_QUERY_HASH_SCHEMA,
        ],
        "trajectory_uid_schema": TRAJECTORY_UID_SCHEMA,
        "internal_query_duplicate_count": 0,
        "internal_trajectory_uid_duplicate_count": 0,
        "internal_trajectory_id_duplicate_count": 0,
        "internal_trajectory_seed_duplicate_count": 0,
        "prior_overlap_counts": {
            "formal_query_hash": 0,
            "runtime_query_hash": 0,
            "trajectory_uid": 0,
            "trajectory_id": 0,
            "seed": 0,
        },
        "access_ledger": dict(access_ledger),
        "sources": records,
        "formal_performance_consumed": False,
        "method_outcomes_used_for_generation_or_identity": False,
    }


def _fresh_payload(fresh: FreshTransitionDataset) -> dict[str, Any]:
    return {
        "protocol": np.asarray([PROTOCOL], dtype="U64"),
        "robot": np.asarray([fresh.robot], dtype="U16"),
        "role": np.asarray([ROLE], dtype="U64"),
        "kinematics_identity": np.asarray([fresh.kinematics_identity], dtype="U64"),
        "dt": np.asarray([fresh.dt], dtype=np.float64),
        "pool_seed": np.asarray([fresh.pool_seed], dtype=np.int64),
        "previous_q": fresh.dataset.previous_q,
        "target_position": fresh.dataset.target_position,
        "target_rotation": fresh.dataset.target_rotation,
        "reference_q": fresh.dataset.reference_q,
        "category": fresh.dataset.category,
        "expected_reachable": fresh.dataset.expected_reachable,
        "continuity_feasible": fresh.dataset.continuity_feasible,
        "trajectory_id": fresh.dataset.trajectory_id,
        "time_index": fresh.dataset.time_index,
        "source_query_hash": fresh.source_query_hash,
        "formal_query_sha256": fresh.formal_query_sha256,
        "trajectory_uid": fresh.trajectory_uid,
        "trajectory_order": np.asarray(fresh.trajectory_order, dtype="U64"),
        "trajectory_seed": fresh.trajectory_seed,
        "phase": fresh.phase,
        "phase_index": fresh.phase_index,
        "transition_boundary": fresh.transition_boundary,
    }


def save_fresh_dataset(
    path: str | Path, fresh: FreshTransitionDataset
) -> dict[str, Any]:
    """Exclusively create the immutable fresh identity/data NPZ."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"fresh dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            np.savez_compressed(handle, **_fresh_payload(fresh))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    return _artifact(destination)


def load_fresh_dataset(
    path: str | Path,
    *,
    robot: str,
    expected_artifact: Mapping[str, Any] | None = None,
    identity_manifest: Mapping[str, Any] | None = None,
    kinematics: KinematicsModel | None = None,
) -> FreshTransitionDataset:
    """Hash-check, schema-check and semantically revalidate a fresh NPZ."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"fresh dataset is unavailable: {source}")
    raw = source.read_bytes()
    if expected_artifact is not None and (
        len(raw) != int(expected_artifact.get("size", -1))
        or sha256(raw).hexdigest()
        != str(expected_artifact.get("sha256", "")).casefold()
    ):
        raise RuntimeError("fresh dataset differs from its frozen descriptor")
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        if set(payload.files) != _FRESH_NPZ_FIELDS:
            raise RuntimeError("fresh dataset NPZ schema changed")
        protocol = np.asarray(payload["protocol"])
        stored_robot = np.asarray(payload["robot"])
        role = np.asarray(payload["role"])
        stored_identity = np.asarray(payload["kinematics_identity"])
        stored_dt = np.asarray(payload["dt"])
        pool_seed = np.asarray(payload["pool_seed"])
        if (
            protocol.shape != (1,)
            or str(protocol[0]) != PROTOCOL
            or stored_robot.shape != (1,)
            or str(stored_robot[0]).casefold() != str(robot).casefold()
            or role.shape != (1,)
            or str(role[0]) != ROLE
            or stored_identity.shape != (1,)
            or stored_dt.shape != (1,)
            or stored_dt.dtype != np.dtype(np.float64)
            or float(stored_dt[0]) != DT
            or pool_seed.shape != (1,)
            or pool_seed.dtype != np.dtype(np.int64)
        ):
            raise RuntimeError("fresh dataset scalar contract changed")
        float_fields = ("previous_q", "target_position", "target_rotation", "reference_q")
        int_fields = ("trajectory_id", "time_index", "trajectory_seed", "phase_index")
        bool_fields = (
            "expected_reachable",
            "continuity_feasible",
            "transition_boundary",
        )
        if any(payload[name].dtype != np.dtype(np.float64) for name in float_fields):
            raise RuntimeError("fresh dataset floating dtype changed")
        if any(payload[name].dtype != np.dtype(np.int64) for name in int_fields):
            raise RuntimeError("fresh dataset integer dtype changed")
        if any(payload[name].dtype != np.dtype(bool) for name in bool_fields):
            raise RuntimeError("fresh dataset boolean dtype changed")
        if any(
            payload[name].dtype.kind != "U"
            for name in (
                "category",
                "source_query_hash",
                "formal_query_sha256",
                "trajectory_uid",
                "trajectory_order",
                "phase",
            )
        ):
            raise RuntimeError("fresh dataset string dtype changed")
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
        fresh = FreshTransitionDataset(
            robot=str(stored_robot[0]),
            kinematics_identity=str(stored_identity[0]),
            dt=float(stored_dt[0]),
            pool_seed=int(pool_seed[0]),
            dataset=dataset,
            source_query_hash=payload["source_query_hash"].astype("U64", copy=True),
            formal_query_sha256=payload["formal_query_sha256"].astype("U64", copy=True),
            trajectory_uid=payload["trajectory_uid"].astype("U64", copy=True),
            trajectory_order=tuple(payload["trajectory_order"].astype(str).tolist()),
            trajectory_seed=payload["trajectory_seed"].copy(),
            phase=payload["phase"].astype("U32", copy=True),
            phase_index=payload["phase_index"].copy(),
            transition_boundary=payload["transition_boundary"].copy(),
        )
    _semantic_revalidation(fresh, kinematics)
    if identity_manifest is not None:
        assert_identity_manifest_matches(fresh, identity_manifest)
        descriptor = identity_manifest.get("dataset_artifact")
        if descriptor is not None and (
            len(raw) != int(descriptor.get("size", -1))
            or sha256(raw).hexdigest()
            != str(descriptor.get("sha256", "")).casefold()
        ):
            raise RuntimeError("fresh dataset is not bound to the identity manifest")
    return fresh


def _identity_core(fresh: FreshTransitionDataset) -> dict[str, Any]:
    family_uids: dict[str, list[str]] = {family: [] for family in FAMILIES}
    trajectories: list[dict[str, Any]] = []
    for uid, rows in fresh.groups():
        family = str(fresh.dataset.category[int(rows[0])])
        numeric_id = int(fresh.dataset.trajectory_id[int(rows[0])])
        seed = int(fresh.trajectory_seed[int(rows[0])])
        family_uids[family].append(uid)
        trajectories.append(
            {
                "trajectory_uid": uid,
                "trajectory_id": numeric_id,
                "trajectory_seed": seed,
                "family": family,
                "frame_count": len(rows),
                "first_frame_index": int(rows[0]),
            }
        )
    source_values = fresh.source_query_hash.astype(str).tolist()
    formal_values = fresh.formal_query_sha256.astype(str).tolist()
    uid_values = list(fresh.trajectory_order)
    numeric_ids = [record["trajectory_id"] for record in trajectories]
    seed_values = [record["trajectory_seed"] for record in trajectories]
    return {
        "protocol": PROTOCOL,
        "identity_manifest_schema": IDENTITY_MANIFEST_SCHEMA,
        "role": ROLE,
        "robot": fresh.robot,
        "kinematics_identity": fresh.kinematics_identity,
        "dt": fresh.dt,
        "pool_seed": fresh.pool_seed,
        "generator_outcome_blind": True,
        "solver_calls_before_identity_freeze": 0,
        "verifier_calls_before_identity_freeze": 0,
        "learned_model_calls_before_identity_freeze": 0,
        "method_outcomes_used_for_identity": False,
        "frame_count": fresh.count,
        "trajectory_count": len(uid_values),
        "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        "families": {
            family: {
                "trajectory_count": len(family_uids[family]),
                "frame_count": len(family_uids[family]) * STEPS_PER_TRAJECTORY,
                "trajectory_uids": family_uids[family],
            }
            for family in FAMILIES
        },
        "query_identity": {
            "runtime_schema": RUNTIME_QUERY_HASH_SCHEMA,
            "formal_schema": FORMAL_QUERY_HASH_SCHEMA,
            "ordered_runtime_query_digest": _digest_strings(
                source_values, ordered=True
            ),
            "runtime_query_set_digest": _digest_strings(
                source_values, ordered=False
            ),
            "ordered_formal_query_digest": _digest_strings(
                formal_values, ordered=True
            ),
            "formal_query_set_digest": _digest_strings(
                formal_values, ordered=False
            ),
            "unique_runtime_query_count": len(set(source_values)),
            "unique_formal_query_count": len(set(formal_values)),
        },
        "trajectory_identity": {
            "schema": TRAJECTORY_UID_SCHEMA,
            "trajectory_order": uid_values,
            "ordered_trajectory_uid_digest": _digest_strings(
                uid_values, ordered=True
            ),
            "trajectory_uid_set_digest": _digest_strings(
                uid_values, ordered=False
            ),
            "ordered_numeric_trajectory_id_digest": _json_digest(numeric_ids),
            "trajectories": trajectories,
        },
        "seed_identity": {
            "pool_seed": fresh.pool_seed,
            "trajectory_seed_derivation": (
                "numpy.SeedSequence([pool_seed,family_index,path_index])"
            ),
            "ordered_trajectory_seed_digest": _json_digest(seed_values),
            "unique_trajectory_seed_count": len(set(seed_values)),
        },
    }


def _verify_manifest_self_hash(payload: Mapping[str, Any]) -> None:
    recorded = _require_hex64(
        str(payload.get("manifest_sha256", "")), name="identity manifest SHA-256"
    )
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    if _json_digest(canonical) != recorded:
        raise RuntimeError("identity manifest self-hash changed")


def build_identity_manifest(
    fresh: FreshTransitionDataset,
    freshness: Mapping[str, Any],
    *,
    dataset_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity seal that must precede solver execution."""

    if (
        freshness.get("protocol") != PROTOCOL
        or freshness.get("status") != "pass"
        or freshness.get("robot") != fresh.robot
        or freshness.get("formal_performance_consumed") is not False
        or freshness.get("method_outcomes_used_for_generation_or_identity") is not False
        or int(freshness.get("performance_files_opened", -1)) != 0
        or int(freshness.get("performance_arrays_read", -1)) != 0
        or any(int(value) != 0 for value in freshness["prior_overlap_counts"].values())
    ):
        raise RuntimeError("a passing, outcome-blind freshness audit is required")
    normalized_artifact: dict[str, Any] | None = None
    if dataset_artifact is not None:
        if set(dataset_artifact) != {"path", "sha256", "size"}:
            raise ValueError("dataset artifact descriptor schema changed")
        normalized_artifact = {
            "path": str(dataset_artifact["path"]),
            "sha256": _require_hex64(
                str(dataset_artifact["sha256"]), name="dataset artifact SHA-256"
            ),
            "size": int(dataset_artifact["size"]),
        }
        if normalized_artifact["size"] <= 0:
            raise ValueError("dataset artifact size must be positive")
    payload = {
        **_identity_core(fresh),
        "dataset_artifact": normalized_artifact,
        "freshness": dict(freshness),
        "identity_frozen_before_any_solver_call": True,
        "formal_identity_files_opened": int(
            freshness["formal_identity_files_opened"]
        ),
        "performance_files_opened": 0,
    }
    payload["manifest_sha256"] = _json_digest(payload)
    _verify_manifest_self_hash(payload)
    return payload


def assert_identity_manifest_matches(
    fresh: FreshTransitionDataset, manifest: Mapping[str, Any]
) -> None:
    required = set(_identity_core(fresh)) | {
        "dataset_artifact",
        "freshness",
        "identity_frozen_before_any_solver_call",
        "formal_identity_files_opened",
        "performance_files_opened",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise RuntimeError("identity manifest schema changed")
    _verify_manifest_self_hash(manifest)
    core = _identity_core(fresh)
    observed_core = {key: manifest[key] for key in core}
    if observed_core != core:
        raise RuntimeError("identity manifest does not match the fresh workload")
    freshness = manifest["freshness"]
    if (
        not isinstance(freshness, Mapping)
        or freshness.get("status") != "pass"
        or freshness.get("robot") != fresh.robot
        or freshness.get("formal_performance_consumed") is not False
        or freshness.get("method_outcomes_used_for_generation_or_identity") is not False
        or manifest.get("identity_frozen_before_any_solver_call") is not True
        or int(manifest.get("formal_identity_files_opened", -1)) != 5
        or int(manifest.get("performance_files_opened", -1)) != 0
    ):
        raise RuntimeError("identity manifest provenance boundary changed")


def save_identity_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Exclusively save one self-hashed immutable identity manifest."""

    _verify_manifest_self_hash(manifest)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"identity manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if destination.exists() and not destination.is_symlink():
            destination.unlink()
        raise
    return _artifact(destination)


def load_identity_manifest(
    path: str | Path,
    *,
    expected_artifact: Mapping[str, Any] | None = None,
    expected_robot: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"identity manifest is unavailable: {source}")
    raw = source.read_bytes()
    if expected_artifact is not None and (
        len(raw) != int(expected_artifact.get("size", -1))
        or sha256(raw).hexdigest()
        != str(expected_artifact.get("sha256", "")).casefold()
    ):
        raise RuntimeError("identity manifest differs from its artifact descriptor")

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {source}")

    payload = json.loads(raw.decode("utf-8"), parse_constant=reject)
    if not isinstance(payload, dict):
        raise RuntimeError("identity manifest is not a JSON object")
    _verify_manifest_self_hash(payload)
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("identity_manifest_schema") != IDENTITY_MANIFEST_SCHEMA
        or payload.get("role") != ROLE
        or payload.get("identity_frozen_before_any_solver_call") is not True
        or payload.get("method_outcomes_used_for_identity") is not False
        or int(payload.get("solver_calls_before_identity_freeze", -1)) != 0
        or int(payload.get("verifier_calls_before_identity_freeze", -1)) != 0
        or int(payload.get("learned_model_calls_before_identity_freeze", -1)) != 0
        or int(payload.get("performance_files_opened", -1)) != 0
    ):
        raise RuntimeError("identity manifest protocol boundary changed")
    if expected_robot is not None and payload.get("robot") != str(
        expected_robot
    ).casefold():
        raise RuntimeError("identity manifest robot binding changed")
    if (
        int(payload.get("frame_count", -1)) != EXPECTED_FRAMES
        or int(payload.get("trajectory_count", -1)) != EXPECTED_TRAJECTORIES
        or int(payload.get("steps_per_trajectory", -1)) != STEPS_PER_TRAJECTORY
        or float(payload.get("dt", -1.0)) != DT
        or int(payload.get("pool_seed", -1))
        != FROZEN_POOL_SEEDS[str(payload.get("robot", "")).casefold()]
    ):
        raise RuntimeError("identity manifest fixed workload contract changed")
    return payload


# Compatibility aliases kept intentionally small for the runner.
save_dataset = save_fresh_dataset
load_dataset = load_fresh_dataset
generate_dataset = generate_fresh_transition_dataset


__all__ = [
    "CURVATURE_TRANSITION_FAMILY",
    "DT",
    "EXPECTED_FRAMES",
    "EXPECTED_TRAJECTORIES",
    "FAMILIES",
    "FAMILY_PHASES",
    "FORMAL_QUERY_HASH_SCHEMA",
    "FROZEN_POOL_SEEDS",
    "FreshDataset",
    "FreshSpec",
    "FreshTransitionDataset",
    "FreshTransitionSpec",
    "IDENTITY_MANIFEST_SCHEMA",
    "LIMIT_TRANSITION_FAMILY",
    "PROTOCOL",
    "PriorIdentityRegistry",
    "ROLE",
    "RUNTIME_QUERY_HASH_SCHEMA",
    "SINGULAR_TRANSITION_FAMILY",
    "SMOOTH_ORIENTATION_FAMILY",
    "STEPS_PER_TRAJECTORY",
    "TRAJECTORIES_PER_FAMILY",
    "TRAJECTORY_ID_BASE",
    "TRAJECTORY_UID_SCHEMA",
    "assert_identity_manifest_matches",
    "audit_freshness",
    "build_identity_manifest",
    "build_prior_identity_registries",
    "formal_query_hashes",
    "generate_dataset",
    "generate_fresh_transition_dataset",
    "load_dataset",
    "load_fresh_dataset",
    "load_identity_manifest",
    "runtime_query_hashes",
    "save_dataset",
    "save_fresh_dataset",
    "save_identity_manifest",
    "trajectory_content_uid",
]
