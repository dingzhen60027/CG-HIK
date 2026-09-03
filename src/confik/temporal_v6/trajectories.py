"""Fresh, development-only trajectory roles for the Temporal V6 pilot.

This module deliberately has no formal-test loader and performs no filesystem
discovery.  Prior-source isolation is possible only through registries or
manifest identities passed explicitly by the caller.  In particular, a path or
role containing the substring ``test`` is rejected before it can be opened.

The generator is shared with earlier work so that the trajectory distribution
does not change, but the pool and split seeds below are new and frozen.  Query
identity is based only on runtime inputs (previous joint state, target pose and
``dt``), never on a label, solver outcome, or reference joint solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Collection, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..data.datasets import QueryDataset
from ..data.generate_v2 import generate_reference_trajectory_tests
from ..kinematics.base import KinematicsModel


PROTOCOL = "temporal_v6_fresh_development_trajectories_v1"
HASH_SCHEMA = "confik_source_query_runtime_inputs_v1"
TRAJECTORY_HASH_SCHEMA = "confik_ordered_source_query_trajectory_v1"

CALIBRATION_ROLE = "trajectory_calibration"
POLICY_VALIDATION_ROLE = "trajectory_policy_validation"
ROLES = (CALIBRATION_ROLE, POLICY_VALIDATION_ROLE)

SOURCE_FAMILY_TO_PUBLIC = {
    "trajectory_smooth": "smooth_multi_joint",
    "trajectory_orientation": "wrist_orientation",
    "trajectory_singular": "low_manipulability",
    "trajectory_limit": "joint_limit_skimming",
}
SOURCE_FAMILIES = tuple(SOURCE_FAMILY_TO_PUBLIC)
FAMILIES = tuple(SOURCE_FAMILY_TO_PUBLIC.values())

FROZEN_POOL_SEEDS = {"panda": 861_601, "ur5e": 861_602}
FROZEN_SPLIT_SEEDS = {"panda": 861_611, "ur5e": 861_612}

PATHS_PER_FAMILY_POOL = 20
PATHS_PER_FAMILY_PER_ROLE = 10
STEPS_PER_TRAJECTORY = 150
DT = 0.02

ALLOWED_SOURCE_CLASSES = frozenset({"development", "training", "reference"})
ALLOWED_SEED_SOURCE_CLASSES = frozenset(
    {"development", "training", "reference", "evaluation"}
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _reject_test_named(value: str | Path, *, name: str) -> None:
    """Reject every role, identity, or path containing ``test``.

    This intentionally uses a substring check rather than trying to infer
    whether a particular test-looking path is formal.  It makes the data
    boundary fail closed and auditable.
    """

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


def _validate_source_class(value: str) -> str:
    normalized = str(value).casefold()
    _reject_test_named(normalized, name="source_class")
    if normalized not in ALLOWED_SOURCE_CLASSES:
        raise ValueError(
            f"source_class must be one of {sorted(ALLOWED_SOURCE_CLASSES)}, got {value!r}"
        )
    return normalized


def _validate_seed_source_class(value: str) -> str:
    normalized = str(value).casefold()
    _reject_test_named(normalized, name="seed source_class")
    if normalized not in ALLOWED_SEED_SOURCE_CLASSES:
        raise ValueError(
            "seed source_class must be one of "
            f"{sorted(ALLOWED_SEED_SOURCE_CLASSES)}, got {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class FreshTrajectorySpec:
    """The exact, frozen data-generation contract for one robot."""

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
            raise ValueError(f"unsupported Temporal V6 robot: {self.robot!r}")
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
            raise ValueError(f"unsupported Temporal V6 robot: {robot!r}")
        return cls(
            robot=normalized,
            pool_seed=FROZEN_POOL_SEEDS[normalized],
            split_seed=FROZEN_SPLIT_SEEDS[normalized],
            kinematics_identity=kinematics_identity,
        )


@dataclass(frozen=True)
class AllowedSourceHashRegistry:
    """Explicitly authorized hashes from a non-test prior source.

    A registry with query hashes proves exact runtime-query non-overlap.  A
    registry may additionally contain trajectory hashes when the old source had
    a trajectory grain.  The object contains no filesystem path and therefore
    cannot discover or open an undeclared data source.
    """

    identity: str
    source_class: str
    query_hashes: tuple[str, ...]
    trajectory_uids: tuple[str, ...] = ()
    registry_sha256: str | None = None

    def __post_init__(self) -> None:
        identity = str(self.identity)
        _reject_test_named(identity, name="registry identity")
        if not identity.strip():
            raise ValueError("registry identity cannot be empty")
        source_class = _validate_source_class(self.source_class)
        query_hashes = tuple(
            _require_hash(value, name="query_hash") for value in self.query_hashes
        )
        trajectory_uids = tuple(
            _require_hash(value, name="trajectory_uid") for value in self.trajectory_uids
        )
        if len(set(query_hashes)) != len(query_hashes):
            raise ValueError("source registry contains duplicate query hashes")
        if len(set(trajectory_uids)) != len(trajectory_uids):
            raise ValueError("source registry contains duplicate trajectory UIDs")
        if not query_hashes and not trajectory_uids:
            raise ValueError("a source hash registry cannot be empty")
        registry_sha = self.registry_sha256
        if registry_sha is not None:
            registry_sha = _require_hash(registry_sha, name="registry_sha256")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "source_class", source_class)
        object.__setattr__(self, "query_hashes", query_hashes)
        object.__setattr__(self, "trajectory_uids", trajectory_uids)
        object.__setattr__(self, "registry_sha256", registry_sha)


@dataclass(frozen=True)
class AllowedManifestIdentity:
    """An explicit old-source identity when row-level hashes are unavailable.

    This detects whole-source reuse only.  It must never be reported as proof of
    row-level query disjointness; the audit output distinguishes the two modes.
    """

    name: str
    source_class: str
    identity_sha256: str

    def __post_init__(self) -> None:
        name = str(self.name)
        _reject_test_named(name, name="manifest identity name")
        if not name.strip():
            raise ValueError("manifest identity name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_class", _validate_source_class(self.source_class))
        object.__setattr__(
            self,
            "identity_sha256",
            _require_hash(self.identity_sha256, name="manifest identity SHA-256"),
        )


@dataclass(frozen=True)
class AllowedSeedRegistry:
    """Explicit non-test lineage seeds against which freshness is checked."""

    identity: str
    source_class: str
    pool_seeds: tuple[int, ...] = ()
    split_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        identity = str(self.identity)
        _reject_test_named(identity, name="seed registry identity")
        if not identity.strip():
            raise ValueError("seed registry identity cannot be empty")
        pool = tuple(int(value) for value in self.pool_seeds)
        split = tuple(int(value) for value in self.split_seeds)
        if not pool and not split:
            raise ValueError("an explicit seed registry cannot be empty")
        if len(set(pool)) != len(pool) or len(set(split)) != len(split):
            raise ValueError("seed registry contains duplicate values")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self, "source_class", _validate_seed_source_class(self.source_class)
        )
        object.__setattr__(self, "pool_seeds", pool)
        object.__setattr__(self, "split_seeds", split)


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
        query_hash = np.asarray(self.source_query_hash, dtype="U64")
        trajectory_uid = np.asarray(self.trajectory_uid, dtype="U64")
        count = len(self.dataset)
        if query_hash.shape != (count,) or trajectory_uid.shape != (count,):
            raise ValueError("role hashes must contain one value per frame")
        if any(_HEX64.fullmatch(value) is None for value in query_hash.tolist()):
            raise ValueError("role contains an invalid source query hash")
        if any(_HEX64.fullmatch(value) is None for value in trajectory_uid.tolist()):
            raise ValueError("role contains an invalid trajectory UID")
        order = tuple(str(value) for value in self.trajectory_order)
        if len(order) != PATHS_PER_FAMILY_PER_ROLE * len(FAMILIES):
            raise ValueError("role must contain exactly 40 ordered trajectories")
        if len(set(order)) != len(order):
            raise ValueError("trajectory_order contains duplicates")
        if set(order) != set(trajectory_uid.tolist()):
            raise ValueError("trajectory_order and per-frame trajectory UIDs differ")
        if len(set(query_hash.tolist())) != count:
            raise ValueError("source query hashes are not unique within the role")
        if set(self.dataset.category.tolist()) != set(FAMILIES):
            raise ValueError("role does not contain the four public trajectory families")
        if count != len(FAMILIES) * PATHS_PER_FAMILY_PER_ROLE * STEPS_PER_TRAJECTORY:
            raise ValueError("role does not contain the frozen 6000-frame workload")
        family_trajectory_counts: dict[str, int] = {}
        for family in FAMILIES:
            selected = self.dataset.category == family
            family_trajectory_counts[family] = len(set(trajectory_uid[selected].tolist()))
            if int(np.sum(selected)) != PATHS_PER_FAMILY_PER_ROLE * STEPS_PER_TRAJECTORY:
                raise ValueError(f"family {family} does not contain 1500 frames")
        if any(
            value != PATHS_PER_FAMILY_PER_ROLE
            for value in family_trajectory_counts.values()
        ):
            raise ValueError("each role family must contain exactly 10 trajectories")
        for uid in order:
            rows = np.flatnonzero(trajectory_uid == uid).astype(np.int64)
            if len(rows) != STEPS_PER_TRAJECTORY or not np.array_equal(
                self.dataset.time_index[rows],
                np.arange(STEPS_PER_TRAJECTORY, dtype=np.int64),
            ):
                raise ValueError("role contains an incomplete or out-of-order trajectory")
            if len(set(self.dataset.category[rows].tolist())) != 1:
                raise ValueError("role trajectory crosses public family boundaries")
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "kinematics_identity", identity)
        object.__setattr__(self, "source_query_hash", query_hash)
        object.__setattr__(self, "trajectory_uid", trajectory_uid)
        object.__setattr__(self, "trajectory_order", order)

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


def runtime_query_hashes(
    previous_q: Any,
    target_position: Any,
    target_rotation: Any,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> NDArray[np.str_]:
    """Return canonical hashes from the three runtime query arrays.

    ``reference_q``, category, split role, trajectory ID and all outcomes are
    deliberately excluded.  Two rows that present the same runtime IK query to
    the same frozen kinematics therefore have the same hash regardless of how
    they were labelled or stored.
    """

    robot = str(robot).casefold()
    if robot not in FROZEN_POOL_SEEDS:
        raise ValueError(f"unsupported robot: {robot!r}")
    identity = _require_hash(
        str(kinematics_identity).casefold(), name="kinematics_identity"
    )
    if float(dt) <= 0.0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be finite and positive")
    previous = np.asarray(previous_q, dtype=np.float64)
    positions = np.asarray(target_position, dtype=np.float64)
    rotations = np.asarray(target_rotation, dtype=np.float64)
    if (
        previous.ndim != 2
        or positions.shape != (previous.shape[0], 3)
        or rotations.shape != (previous.shape[0], 3, 3)
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rotations))
    ):
        raise ValueError("runtime query arrays have invalid shape or non-finite values")
    result: list[str] = []
    for index in range(previous.shape[0]):
        digest = sha256()
        _hash_field(digest, "schema", HASH_SCHEMA.encode("utf-8"))
        _hash_field(digest, "robot", robot.encode("utf-8"))
        _hash_field(digest, "kinematics", identity.encode("ascii"))
        _hash_field(digest, "dt", _canonical_float_bytes(float(dt)))
        _hash_field(
            digest,
            "previous_q",
            _canonical_float_bytes(previous[index]),
        )
        _hash_field(
            digest,
            "target_position",
            _canonical_float_bytes(positions[index]),
        )
        _hash_field(
            digest,
            "target_rotation",
            _canonical_float_bytes(rotations[index]),
        )
        result.append(digest.hexdigest())
    return np.asarray(result, dtype="U64")


def source_query_hashes(
    dataset: QueryDataset,
    *,
    robot: str,
    dt: float,
    kinematics_identity: str,
) -> NDArray[np.str_]:
    """Return canonical hashes for a QueryDataset's runtime inputs."""

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
) -> str:
    digest = sha256()
    _hash_field(digest, "schema", TRAJECTORY_HASH_SCHEMA.encode("utf-8"))
    _hash_field(digest, "robot", robot.encode("utf-8"))
    _hash_field(digest, "kinematics", kinematics_identity.encode("ascii"))
    _hash_field(digest, "family", public_family.encode("utf-8"))
    _hash_field(digest, "time_index", _canonical_int_bytes(time_index))
    for value in query_hashes:
        _hash_field(digest, "source_query_hash", str(value).encode("ascii"))
    return digest.hexdigest()


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


def _pool_identity(
    spec: FreshTrajectorySpec,
    query_hashes: Sequence[str],
    trajectory_uids: Sequence[str],
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
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_hash_registry_from_role(
    role: FreshTrajectoryRole,
    *,
    identity: str,
    source_class: str,
) -> AllowedSourceHashRegistry:
    """Build an in-memory explicit registry from an allowed non-test role."""

    registry_payload = {
        "identity": str(identity),
        "source_class": str(source_class),
        "query_hashes": sorted(set(role.source_query_hash.astype(str).tolist())),
        "trajectory_uids": sorted(set(role.trajectory_uid.astype(str).tolist())),
    }
    registry_sha = sha256(
        json.dumps(registry_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return AllowedSourceHashRegistry(
        identity=str(identity),
        source_class=source_class,
        query_hashes=tuple(registry_payload["query_hashes"]),
        trajectory_uids=tuple(registry_payload["trajectory_uids"]),
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
    """Build a registry from an already-authorized in-memory old source.

    The helper does not open or discover a path.  It therefore supports prior
    training, development, and reference datasets without creating a route to
    formal-test data.  A non-trajectory dataset contributes query hashes only.
    """

    _reject_test_named(identity, name="registry identity")
    hashes = tuple(
        sorted(
            source_query_hashes(
                dataset,
                robot=robot,
                dt=dt,
                kinematics_identity=kinematics_identity,
            ).tolist()
        )
    )
    if len(set(hashes)) != len(hashes):
        raise ValueError("authorized source dataset contains duplicate runtime queries")
    registry_payload = {
        "identity": str(identity),
        "source_class": _validate_source_class(source_class),
        "query_hashes": list(hashes),
        "trajectory_uids": [],
    }
    registry_sha = sha256(
        json.dumps(registry_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return AllowedSourceHashRegistry(
        identity=str(identity),
        source_class=source_class,
        query_hashes=hashes,
        registry_sha256=registry_sha,
    )


def load_allowed_source_hash_registry(
    path: str | Path,
    *,
    expected_identity: str,
    expected_source_class: str,
) -> AllowedSourceHashRegistry:
    """Load one explicitly named non-test registry; never searches for one."""

    source = Path(path)
    _reject_test_named(source, name="source registry path")
    _reject_test_named(expected_identity, name="expected registry identity")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"explicit source registry is unavailable: {source}")
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source registry must be a JSON object")
    allowed_keys = {
        "identity",
        "source_class",
        "query_hashes",
        "trajectory_uids",
        "registry_sha256",
    }
    if set(payload) - allowed_keys:
        raise ValueError("source registry contains unknown fields")
    if str(payload.get("identity", "")) != str(expected_identity):
        raise RuntimeError("source registry identity changed")
    if str(payload.get("source_class", "")).casefold() != str(
        expected_source_class
    ).casefold():
        raise RuntimeError("source registry class changed")
    recorded_sha = payload.get("registry_sha256")
    if recorded_sha is not None:
        # The self-identity is over the object with the self field omitted.
        canonical = dict(payload)
        canonical.pop("registry_sha256", None)
        observed = sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if observed != str(recorded_sha).casefold():
            raise RuntimeError("source registry content hash changed")
    return AllowedSourceHashRegistry(
        identity=str(payload.get("identity", "")),
        source_class=str(payload.get("source_class", "")),
        query_hashes=tuple(str(value) for value in payload.get("query_hashes", ())),
        trajectory_uids=tuple(
            str(value) for value in payload.get("trajectory_uids", ())
        ),
        registry_sha256=(None if recorded_sha is None else str(recorded_sha)),
    )


def audit_source_isolation(
    *,
    new_query_hashes: Collection[str],
    new_trajectory_uids: Collection[str],
    new_manifest_identity: str,
    source_registries: Sequence[AllowedSourceHashRegistry] = (),
    manifest_identities: Sequence[AllowedManifestIdentity] = (),
) -> dict[str, Any]:
    """Audit exact overlap against only explicitly supplied old sources."""

    new_queries = set(str(value) for value in new_query_hashes)
    new_trajectories = set(str(value) for value in new_trajectory_uids)
    new_identity = _require_hash(new_manifest_identity, name="new manifest identity")
    records: list[dict[str, Any]] = []
    identity_names: set[str] = set()
    for registry in source_registries:
        if registry.identity in identity_names:
            raise ValueError(f"duplicate declared prior-source identity: {registry.identity}")
        identity_names.add(registry.identity)
        query_overlap = sorted(new_queries & set(registry.query_hashes))
        trajectory_overlap = sorted(new_trajectories & set(registry.trajectory_uids))
        if query_overlap or trajectory_overlap:
            raise RuntimeError(
                f"fresh source overlaps explicit registry {registry.identity}: "
                f"query={len(query_overlap)}, trajectory={len(trajectory_overlap)}"
            )
        records.append(
            {
                "identity": registry.identity,
                "source_class": registry.source_class,
                "evidence_mode": "row_hash_registry",
                "registry_sha256": registry.registry_sha256,
                "declared_query_hash_count": len(registry.query_hashes),
                "declared_trajectory_uid_count": len(registry.trajectory_uids),
                "query_overlap_count": 0,
                "trajectory_overlap_count": 0,
            }
        )
    for manifest in manifest_identities:
        if manifest.name in identity_names:
            raise ValueError(f"duplicate declared prior-source identity: {manifest.name}")
        identity_names.add(manifest.name)
        reused = manifest.identity_sha256 == new_identity
        if reused:
            raise RuntimeError(f"fresh source reuses manifest identity {manifest.name}")
        records.append(
            {
                "identity": manifest.name,
                "source_class": manifest.source_class,
                "evidence_mode": "manifest_identity_only",
                "manifest_identity_sha256": manifest.identity_sha256,
                "whole_source_identity_equal": False,
                "query_overlap_count": None,
                "trajectory_overlap_count": None,
                "limitation": "manifest identity alone does not prove row-level disjointness",
            }
        )
    row_hash_classes = sorted(
        {
            record["source_class"]
            for record in records
            if record["evidence_mode"] == "row_hash_registry"
        }
    )
    return {
        "status": "pass",
        "discovery_performed": False,
        "formal_test_source_opened": False,
        "allowed_source_classes": sorted(ALLOWED_SOURCE_CLASSES),
        "declared_source_count": len(records),
        "new_query_hash_count": len(new_queries),
        "new_trajectory_uid_count": len(new_trajectories),
        "new_manifest_identity_sha256": new_identity,
        "row_hash_covered_source_classes": row_hash_classes,
        "row_level_isolation_proven_for_every_declared_source": bool(records)
        and all(record["evidence_mode"] == "row_hash_registry" for record in records),
        "declared_nonformal_row_hash_coverage_complete": bool(records)
        and ALLOWED_SOURCE_CLASSES.issubset(set(row_hash_classes)),
        "formal_evaluation_row_hash_comparison_performed": False,
        "formal_evaluation_row_hash_isolation_proven": False,
        "all_prior_source_row_hash_coverage_complete": False,
        "records": records,
        "claim_boundary": (
            "absence of overlap is established only for explicitly declared row-hash "
            "registries; identity-only sources have no row-level non-overlap claim"
        ),
    }


def audit_seed_isolation(
    spec: FreshTrajectorySpec,
    *,
    seed_registries: Sequence[AllowedSeedRegistry] = (),
) -> dict[str, Any]:
    """Check frozen seeds against explicitly declared non-test seed registries."""

    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for registry in seed_registries:
        if registry.identity in names:
            raise ValueError(f"duplicate declared seed registry: {registry.identity}")
        names.add(registry.identity)
        declared = set(registry.pool_seeds) | set(registry.split_seeds)
        pool_collision = spec.pool_seed in declared
        split_collision = spec.split_seed in declared
        if pool_collision or split_collision:
            raise RuntimeError(
                f"fresh seeds overlap explicit registry {registry.identity}: "
                f"pool={pool_collision}, split={split_collision}"
            )
        records.append(
            {
                "identity": registry.identity,
                "source_class": registry.source_class,
                "declared_pool_seed_count": len(registry.pool_seeds),
                "declared_split_seed_count": len(registry.split_seeds),
                "cross_role_seed_comparison": True,
                "pool_seed_overlap": False,
                "split_seed_overlap": False,
            }
        )
    covered_classes = sorted({record["source_class"] for record in records})
    return {
        "status": "pass",
        "discovery_performed": False,
        "formal_test_source_opened": False,
        "pool_seed": spec.pool_seed,
        "split_seed": spec.split_seed,
        "declared_registry_count": len(records),
        "required_source_classes": sorted(ALLOWED_SEED_SOURCE_CLASSES),
        "covered_source_classes": covered_classes,
        "prior_seed_class_coverage_complete": bool(records)
        and ALLOWED_SEED_SOURCE_CLASSES.issubset(set(covered_classes)),
        "records": records,
        "claim_boundary": (
            "seed non-overlap is established only for explicitly declared registries; "
            "query-hash isolation remains the authoritative content check"
        ),
    }


def _build_role(
    *,
    spec: FreshTrajectorySpec,
    role: str,
    pool: QueryDataset,
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
    )


def _role_audit(role: FreshTrajectoryRole) -> dict[str, Any]:
    family: dict[str, dict[str, int]] = {}
    for name in FAMILIES:
        selected = role.dataset.category == name
        family[name] = {
            "trajectory_count": len(set(role.trajectory_uid[selected].tolist())),
            "frame_count": int(np.sum(selected)),
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
        "ordered_trajectory_digest": _digest_strings(
            trajectory_values, ordered=True
        ),
        "trajectory_set_digest": _digest_strings(trajectory_values, ordered=False),
        "by_family": family,
    }


def generate_fresh_development_roles(
    kinematics: KinematicsModel,
    spec: FreshTrajectorySpec,
    *,
    source_registries: Sequence[AllowedSourceHashRegistry] = (),
    manifest_identities: Sequence[AllowedManifestIdentity] = (),
    seed_registries: Sequence[AllowedSeedRegistry] = (),
) -> tuple[FreshTrajectoryRole, FreshTrajectoryRole, dict[str, Any]]:
    """Generate and outcome-blind split one fresh development pool.

    The complete pool is generated first.  Whole trajectories are then split
    within each source family using only the frozen split seed.  No solver,
    learned model, verifier, outcome, or policy is consulted during generation
    or assignment.
    """

    seed_isolation = audit_seed_isolation(spec, seed_registries=seed_registries)
    pool = generate_reference_trajectory_tests(
        kinematics,
        paths_per_type=spec.paths_per_family_pool,
        steps=spec.steps,
        seed=spec.pool_seed,
        dt=spec.dt,
    )
    expected_frames = len(SOURCE_FAMILIES) * spec.paths_per_family_pool * spec.steps
    if len(pool) != expected_frames:
        raise RuntimeError(
            f"fresh pool has {len(pool)} frames, expected {expected_frames}"
        )
    if not np.all(pool.expected_reachable) or not np.all(pool.continuity_feasible):
        raise RuntimeError("reference trajectory generator returned an infeasible label")
    source_categories = pool.category.copy()
    if set(source_categories.tolist()) != set(SOURCE_FAMILIES):
        raise RuntimeError("reference trajectory generator family schema changed")
    pool.category = np.asarray(
        [SOURCE_FAMILY_TO_PUBLIC[value] for value in source_categories], dtype=str
    )

    query_hash = source_query_hashes(
        pool,
        robot=spec.robot,
        dt=spec.dt,
        kinematics_identity=spec.kinematics_identity,
    )
    if len(set(query_hash.tolist())) != len(query_hash):
        raise RuntimeError("fresh pool contains duplicate runtime-query hashes")

    trajectory_uid_by_id: dict[int, str] = {}
    family_by_id: dict[int, str] = {}
    for trajectory_id in np.unique(pool.trajectory_id).astype(np.int64):
        rows = np.flatnonzero(pool.trajectory_id == trajectory_id).astype(np.int64)
        rows = rows[np.argsort(pool.time_index[rows], kind="stable")]
        if len(rows) != spec.steps or not np.array_equal(
            pool.time_index[rows], np.arange(spec.steps, dtype=np.int64)
        ):
            raise RuntimeError("fresh pool trajectory is incomplete")
        families = set(pool.category[rows].tolist())
        if len(families) != 1:
            raise RuntimeError("a fresh trajectory crosses family boundaries")
        public_family = next(iter(families))
        family_by_id[int(trajectory_id)] = public_family
        trajectory_uid_by_id[int(trajectory_id)] = _trajectory_hash(
            robot=spec.robot,
            kinematics_identity=spec.kinematics_identity,
            public_family=public_family,
            query_hashes=query_hash[rows].tolist(),
            time_index=pool.time_index[rows],
        )
    if len(set(trajectory_uid_by_id.values())) != len(trajectory_uid_by_id):
        raise RuntimeError("fresh pool contains duplicate trajectory UIDs")

    assignment: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ids_by_role: dict[str, list[int]] = {role: [] for role in ROLES}
    for family_index, public_family in enumerate(FAMILIES):
        family_ids = np.asarray(
            sorted(
                trajectory_id
                for trajectory_id, family in family_by_id.items()
                if family == public_family
            ),
            dtype=np.int64,
        )
        if len(family_ids) != spec.paths_per_family_pool:
            raise RuntimeError(
                f"{public_family} has {len(family_ids)} paths, expected "
                f"{spec.paths_per_family_pool}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([spec.split_seed, family_index])
        )
        shuffled = family_ids.copy()
        rng.shuffle(shuffled)
        per_role = (
            shuffled[: spec.paths_per_family_per_role],
            shuffled[
                spec.paths_per_family_per_role : 2
                * spec.paths_per_family_per_role
            ],
        )
        assignment[public_family] = {}
        source_family = SOURCE_FAMILIES[family_index]
        for role, selected in zip(ROLES, per_role, strict=True):
            if len(selected) != spec.paths_per_family_per_role:
                raise RuntimeError("fresh pool is too small for a complete 10+10 split")
            ids_by_role[role].extend(int(value) for value in selected)
            assignment[public_family][role] = [
                {
                    "source_category": source_family,
                    "source_trajectory_id": int(value),
                    "trajectory_uid": trajectory_uid_by_id[int(value)],
                }
                for value in selected
            ]

    # Interleave families for measurement without changing role membership.
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
        selected_ids=ids_by_role[CALIBRATION_ROLE],
        query_hash_by_row=query_hash,
        trajectory_uid_by_id=trajectory_uid_by_id,
    )
    policy_validation = _build_role(
        spec=spec,
        role=POLICY_VALIDATION_ROLE,
        pool=pool,
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
            "fresh calibration/policy-validation roles overlap: "
            f"query={len(query_overlap)}, trajectory={len(trajectory_overlap)}"
        )

    pool_identity = _pool_identity(
        spec,
        query_hash.tolist(),
        tuple(trajectory_uid_by_id.values()),
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
        "generator": "generate_reference_trajectory_tests",
        "generator_outcome_blind": True,
        "split_before_outcome_collection": True,
        "formal_test_data_opened": False,
        "filesystem_discovery_performed": False,
        "robot": spec.robot,
        "kinematics_identity": spec.kinematics_identity,
        "hash_schema": HASH_SCHEMA,
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
        "source_family_to_public_family": dict(SOURCE_FAMILY_TO_PUBLIC),
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
        "role": np.asarray([role.role], dtype="U32"),
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
    }


def save_trajectory_role(path: str | Path, role: FreshTrajectoryRole) -> dict[str, Any]:
    """Exclusively create a role NPZ and return its artifact descriptor."""

    destination = Path(path)
    _reject_test_named(destination, name="trajectory role path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"trajectory role already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o644)
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


def load_trajectory_role(
    path: str | Path,
    *,
    robot: str,
    expected_role: str,
    expected_artifact: Mapping[str, Any] | None = None,
) -> FreshTrajectoryRole:
    """Load, hash-check and semantically revalidate one explicitly named role."""

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
        "protocol",
        "robot",
        "role",
        "kinematics_identity",
        "dt",
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
        "trajectory_uid",
        "trajectory_order",
    }
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise RuntimeError("trajectory role NPZ schema changed")
        stored_protocol = str(np.asarray(payload["protocol"]).reshape(-1)[0])
        stored_robot = str(np.asarray(payload["robot"]).reshape(-1)[0]).casefold()
        stored_role = str(np.asarray(payload["role"]).reshape(-1)[0])
        identity = str(np.asarray(payload["kinematics_identity"]).reshape(-1)[0])
        dt = float(np.asarray(payload["dt"]).reshape(-1)[0])
        if stored_protocol != PROTOCOL:
            raise RuntimeError("trajectory role protocol changed")
        if stored_robot != str(robot).casefold() or stored_role != expected_role:
            raise RuntimeError("trajectory role robot/role binding changed")
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
        stored_query_hash = payload["source_query_hash"].astype("U64", copy=True)
        stored_trajectory_uid = payload["trajectory_uid"].astype("U64", copy=True)
        order = tuple(payload["trajectory_order"].astype(str).tolist())
    role = FreshTrajectoryRole(
        robot=stored_robot,
        role=stored_role,
        kinematics_identity=identity,
        dt=dt,
        dataset=dataset,
        source_query_hash=stored_query_hash,
        trajectory_uid=stored_trajectory_uid,
        trajectory_order=order,
    )
    recomputed_query_hash = source_query_hashes(
        role.dataset,
        robot=role.robot,
        dt=role.dt,
        kinematics_identity=role.kinematics_identity,
    )
    if not np.array_equal(recomputed_query_hash, role.source_query_hash):
        raise RuntimeError("stored source query hashes do not match role inputs")
    recomputed_uid = np.empty(role.count, dtype="U64")
    recomputed_order: list[str] = []
    for _, rows in role.groups():
        families = set(role.dataset.category[rows].tolist())
        if len(families) != 1:
            raise RuntimeError("loaded trajectory crosses family boundaries")
        observed = _trajectory_hash(
            robot=role.robot,
            kinematics_identity=role.kinematics_identity,
            public_family=next(iter(families)),
            query_hashes=role.source_query_hash[rows].tolist(),
            time_index=role.dataset.time_index[rows],
        )
        recomputed_uid[rows] = observed
        recomputed_order.append(observed)
    if not np.array_equal(recomputed_uid, role.trajectory_uid) or tuple(
        recomputed_order
    ) != role.trajectory_order:
        raise RuntimeError("stored trajectory UIDs do not match ordered role inputs")
    return role


def save_split_audit_manifest(path: str | Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Exclusively save the outcome-blind split/isolation audit."""

    destination = Path(path)
    _reject_test_named(destination, name="split audit manifest path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"split audit manifest already exists: {destination}")
    if audit.get("protocol") != PROTOCOL:
        raise ValueError("split audit protocol changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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
    """Load one explicit development manifest without filesystem discovery."""

    source = Path(path)
    _reject_test_named(source, name="split audit manifest path")
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"split audit manifest is unavailable: {source}")
    if expected_artifact is not None and (
        source.stat().st_size != int(expected_artifact.get("size", -1))
        or _sha256_file(source) != str(expected_artifact.get("sha256", "")).casefold()
    ):
        raise RuntimeError("split audit manifest differs from its descriptor")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise RuntimeError("split audit manifest protocol changed")
    if payload.get("formal_test_data_opened") is not False:
        raise RuntimeError("split audit does not prove the development boundary")
    return payload


__all__ = [
    "ALLOWED_SOURCE_CLASSES",
    "ALLOWED_SEED_SOURCE_CLASSES",
    "AllowedManifestIdentity",
    "AllowedSeedRegistry",
    "AllowedSourceHashRegistry",
    "CALIBRATION_ROLE",
    "DT",
    "FAMILIES",
    "FROZEN_POOL_SEEDS",
    "FROZEN_SPLIT_SEEDS",
    "FreshTrajectoryRole",
    "FreshTrajectorySpec",
    "HASH_SCHEMA",
    "PATHS_PER_FAMILY_PER_ROLE",
    "PATHS_PER_FAMILY_POOL",
    "POLICY_VALIDATION_ROLE",
    "PROTOCOL",
    "ROLES",
    "SOURCE_FAMILIES",
    "SOURCE_FAMILY_TO_PUBLIC",
    "STEPS_PER_TRAJECTORY",
    "TRAJECTORY_HASH_SCHEMA",
    "audit_source_isolation",
    "audit_seed_isolation",
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
