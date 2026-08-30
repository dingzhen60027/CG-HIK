from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..counterfactual_v4.data_v4 import (
    generate_prespecified_ood_points,
    generate_prespecified_ood_trajectories,
)
from ..data.datasets import QueryDataset
from ..data.generate import generate_point_test_set, generate_smooth_transitions
from ..data.generate_v2 import (
    generate_hard_valid_queries,
    generate_reference_trajectory_tests,
)
from ..kinematics.base import KinematicsModel
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier


TEST_V4_ROLES = (
    "id_points",
    "id_trajectories",
    "ood_points",
    "ood_trajectories",
)
ROLE_DOMAIN = {
    "id_points": "id",
    "id_trajectories": "id",
    "ood_points": "ood",
    "ood_trajectories": "ood",
}
IDENTITY_ARRAYS = ("previous_q", "target_position", "target_rotation")


@dataclass(frozen=True)
class ComparisonSource:
    """A prior-query identity source allowed in the freshness audit.

    ``query_hash_npz`` reads only a precomputed ``query_sha256`` array.
    ``identity_npz`` reads only the three arrays in :data:`IDENTITY_ARRAYS`.
    In particular, the old formal-test source is never opened as a result
    record and no performance metric is consumed.
    """

    name: str
    path: Path
    kind: str


def derive_seed(release_v4_digest: str, robot: str, role: str) -> int:
    """Derive the exact unsigned 32-bit seed frozen by the v4 protocol."""

    if not release_v4_digest or not robot or not role:
        raise ValueError("release digest, robot, and role must be non-empty")
    material = f"test_v4_locked|{release_v4_digest}|{robot}|{role}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:4], "big", signed=False)


def _subseeds(root_seed: int, count: int) -> list[int]:
    """Create deterministic component seeds without adding hidden test knobs."""

    children = np.random.SeedSequence(int(root_seed)).spawn(count)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def query_sha256(
    previous_q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    dt: float,
) -> str:
    """Hash the exact solver query using the established v3 byte contract."""

    digest = sha256()
    digest.update(np.ascontiguousarray(previous_q, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(target_position, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(target_rotation, dtype=np.float64).tobytes())
    digest.update(np.asarray([dt], dtype=np.float64).tobytes())
    return digest.hexdigest()


def dataset_query_hashes(dataset: QueryDataset, *, dt: float) -> np.ndarray:
    return np.asarray(
        [
            query_sha256(
                dataset.previous_q[index],
                dataset.target_position[index],
                dataset.target_rotation[index],
                dt=dt,
            )
            for index in range(len(dataset))
        ],
        dtype="U64",
    )


def query_keys(
    robot: str,
    role: str,
    dataset: QueryDataset,
    *,
    dt: float,
) -> list[dict[str, str]]:
    """Return the preregistered four-field identity key for every query."""

    if role not in ROLE_DOMAIN:
        raise ValueError(f"unknown test_v4 role: {role}")
    hashes = dataset_query_hashes(dataset, dt=dt)
    return [
        {
            "robot": robot,
            "domain": ROLE_DOMAIN[role],
            "family": str(dataset.category[index]),
            "query_sha256": str(hashes[index]),
        }
        for index in range(len(dataset))
    ]


def _id_points(
    kinematics: KinematicsModel,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    *,
    role_config: Mapping[str, Any],
    seed: int,
    dt: float,
) -> tuple[QueryDataset, dict[str, int]]:
    transition_seed, stress_seed, hard_seed = _subseeds(seed, 3)
    id_count = int(role_config["id_count"])
    local = generate_smooth_transitions(
        kinematics,
        trajectories=id_count,
        steps_per_trajectory=1,
        seed=transition_seed,
        dt=dt,
        margin=0.1,
    )
    points = generate_point_test_set(
        kinematics,
        local,
        per_category=int(role_config["per_stress_category"]),
        id_count=id_count,
        seed=stress_seed,
        dt=dt,
    )
    hard = generate_hard_valid_queries(
        kinematics,
        dls,
        verifier,
        count=int(role_config["hard_valid_count"]),
        seed=hard_seed,
        dt=dt,
        easy_iterations=int(role_config["hard_screening_easy_iterations"]),
        hard_iterations=int(role_config["hard_screening_robust_iterations"]),
    )
    return QueryDataset.concatenate([points, hard]), {
        "root": int(seed),
        "local_transitions": transition_seed,
        "point_stress": stress_seed,
        "hard_valid": hard_seed,
    }


def generate_locked_datasets(
    *,
    kinematics: KinematicsModel,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    release_v4_digest: str,
    robot: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, QueryDataset], dict[str, dict[str, Any]]]:
    """Generate all four explicit test roles from the frozen release digest.

    This function is deliberately side-effect free: it returns arrays in
    memory and never creates ``outputs/test_v4_*``.  The one-shot formal runner
    is responsible for freezing a preregistration before persisting them.
    """

    data = config["data"]
    roles = data["roles"]
    if tuple(roles) != TEST_V4_ROLES:
        raise ValueError(
            f"data roles must appear exactly as {TEST_V4_ROLES}, got {tuple(roles)}"
        )
    dt = float(data["dt"])
    root_seeds = {
        role: derive_seed(release_v4_digest, robot, role) for role in TEST_V4_ROLES
    }

    id_points, point_seed_details = _id_points(
        kinematics,
        dls,
        verifier,
        role_config=roles["id_points"],
        seed=root_seeds["id_points"],
        dt=dt,
    )
    id_trajectories = generate_reference_trajectory_tests(
        kinematics,
        paths_per_type=int(roles["id_trajectories"]["paths_per_family"]),
        steps=int(roles["id_trajectories"]["steps"]),
        seed=root_seeds["id_trajectories"],
        dt=dt,
    )
    ood_points = generate_prespecified_ood_points(
        kinematics,
        per_family=int(roles["ood_points"]["per_family"]),
        seed=root_seeds["ood_points"],
        dt=dt,
    )
    ood_trajectories = generate_prespecified_ood_trajectories(
        kinematics,
        paths_per_family=int(roles["ood_trajectories"]["paths_per_family"]),
        steps=int(roles["ood_trajectories"]["steps"]),
        seed=root_seeds["ood_trajectories"],
        dt=dt,
    )
    datasets = {
        "id_points": id_points,
        "id_trajectories": id_trajectories,
        "ood_points": ood_points,
        "ood_trajectories": ood_trajectories,
    }
    seed_manifest: dict[str, dict[str, Any]] = {
        role: {
            "root": root_seeds[role],
            "derivation_material": (
                f"test_v4_locked|{release_v4_digest}|{robot}|{role}"
            ),
        }
        for role in TEST_V4_ROLES
    }
    seed_manifest["id_points"]["components"] = point_seed_details
    return datasets, seed_manifest


def dataset_contract(
    datasets: Mapping[str, QueryDataset],
    *,
    robot: str,
    dt: float,
) -> dict[str, Any]:
    if set(datasets) != set(TEST_V4_ROLES):
        raise ValueError(f"datasets must contain exactly {TEST_V4_ROLES}")
    role_payload: dict[str, Any] = {}
    all_keys: list[tuple[str, str, str, str]] = []
    for role in TEST_V4_ROLES:
        dataset = datasets[role]
        keys = query_keys(robot, role, dataset, dt=dt)
        categories = Counter(dataset.category.astype(str).tolist())
        hashes = [item["query_sha256"] for item in keys]
        trajectories = (
            len(np.unique(dataset.trajectory_id))
            if role.endswith("trajectories")
            else 0
        )
        role_payload[role] = {
            "domain": ROLE_DOMAIN[role],
            "query_count": len(dataset),
            "trajectory_count": int(trajectories),
            "category_counts": dict(sorted(categories.items())),
            "unique_query_sha256": len(set(hashes)),
            "within_role_exact_duplicate_count": len(hashes) - len(set(hashes)),
            "finite_identity_arrays": bool(
                np.all(np.isfinite(dataset.previous_q))
                and np.all(np.isfinite(dataset.target_position))
                and np.all(np.isfinite(dataset.target_rotation))
            ),
        }
        all_keys.extend(
            (
                item["robot"],
                item["domain"],
                item["family"],
                item["query_sha256"],
            )
            for item in keys
        )
    return {
        "robot": robot,
        "query_key_fields": ["robot", "domain", "family", "query_sha256"],
        "roles": role_payload,
        "total_queries": len(all_keys),
        "unique_query_keys": len(set(all_keys)),
        "internal_duplicate_key_count": len(all_keys) - len(set(all_keys)),
    }


def validate_dataset_contract(contract: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    data = config["data"]
    expected_roles = data["roles"]
    if tuple(expected_roles) != TEST_V4_ROLES:
        raise RuntimeError("test_v4 role order or membership differs from preregistration")
    for role in TEST_V4_ROLES:
        observed = contract["roles"][role]
        expected = expected_roles[role]
        checks = {
            "domain": str(expected["domain"]),
            "query_count": int(expected["expected_queries"]),
            "trajectory_count": int(expected["expected_trajectories"]),
            "category_counts": {
                str(key): int(value)
                for key, value in expected["expected_category_counts"].items()
            },
            "within_role_exact_duplicate_count": 0,
            "finite_identity_arrays": True,
        }
        actual = {key: observed[key] for key in checks}
        if actual != checks:
            raise RuntimeError(
                f"test_v4 {role} contract mismatch: expected={checks}, actual={actual}"
            )
    if int(contract["total_queries"]) != int(data["expected_queries_per_robot"]):
        raise RuntimeError("test_v4 total query count differs from preregistration")
    if int(contract["internal_duplicate_key_count"]) != 0:
        raise RuntimeError("test_v4 contains duplicate query keys")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_hashes_from_npz(path: Path, *, dt: float) -> set[str]:
    """Read identity arrays only; extra arrays (including metrics) are ignored."""

    with np.load(path, allow_pickle=False) as arrays:
        missing = set(IDENTITY_ARRAYS) - set(arrays.files)
        if missing:
            raise KeyError(f"identity source {path} lacks arrays: {sorted(missing)}")
        previous_q = np.asarray(arrays["previous_q"], dtype=np.float64)
        target_position = np.asarray(arrays["target_position"], dtype=np.float64)
        target_rotation = np.asarray(arrays["target_rotation"], dtype=np.float64)
    count = len(previous_q)
    if target_position.shape != (count, 3) or target_rotation.shape != (count, 3, 3):
        raise ValueError(f"identity source {path} has incompatible query shapes")
    return {
        query_sha256(
            previous_q[index], target_position[index], target_rotation[index], dt=dt
        )
        for index in range(count)
    }


def _precomputed_hashes_from_npz(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as arrays:
        if "query_sha256" not in arrays.files:
            raise KeyError(f"query-hash source {path} lacks query_sha256")
        values = np.asarray(arrays["query_sha256"]).astype(str).reshape(-1)
    if any(len(value) != 64 for value in values):
        raise ValueError(f"query-hash source {path} contains a malformed digest")
    return set(values.tolist())


def default_comparison_sources(workspace: Path, robot: str) -> list[ComparisonSource]:
    """List all training/validation pilots and the only allowed old-test source."""

    root = workspace.resolve()
    bulk = root / "outputs" / "counterfactual_v4_bulk" / robot / "seed17"
    old_v2 = root / "outputs" / "paper_v2_seed17" / robot / "datasets"
    return [
        *[
            ComparisonSource(
                f"bulk/{role}", bulk / role / "selection.npz", "query_hash_npz"
            )
            for role in (
                "risk_train_queries",
                "calibration_queries",
                "policy_validation_queries",
            )
        ],
        ComparisonSource(
            "counterfactual_v4_pilot",
            root
            / "outputs"
            / "counterfactual_v4_pilot"
            / robot
            / "seed17"
            / "counterfactual_labels.npz",
            "query_hash_npz",
        ),
        # latency_pilot_v3 drew from these two complete validation sources.
        ComparisonSource(
            "latency_pilot_v3/risk_validation_source",
            old_v2 / "risk_validation_queries.npz",
            "identity_npz",
        ),
        ComparisonSource(
            "latency_pilot_v3/trajectory_validation_source",
            old_v2 / "seed_validation.npz",
            "identity_npz",
        ),
        # Evidence boundary: only these identity arrays are read from old test.
        ComparisonSource(
            "old_formal_test_v3_identity",
            root
            / "outputs"
            / "test_v3_aggregate"
            / "datasets"
            / f"{robot}_test_v3_queries.npz",
            "identity_npz",
        ),
    ]


def audit_freshness(
    datasets: Mapping[str, QueryDataset],
    *,
    robot: str,
    dt: float,
    comparison_sources: Sequence[ComparisonSource],
) -> dict[str, Any]:
    """Prove internal and prior-data exact-query separation before testing."""

    if set(datasets) != set(TEST_V4_ROLES):
        raise ValueError(f"datasets must contain exactly {TEST_V4_ROLES}")
    role_hashes = {
        role: dataset_query_hashes(datasets[role], dt=dt).astype(str).tolist()
        for role in TEST_V4_ROLES
    }
    duplicate_counts = {
        role: len(values) - len(set(values)) for role, values in role_hashes.items()
    }
    cross_role_overlap: dict[str, int] = {}
    for left_index, left in enumerate(TEST_V4_ROLES):
        for right in TEST_V4_ROLES[left_index + 1 :]:
            cross_role_overlap[f"{left}__{right}"] = len(
                set(role_hashes[left]) & set(role_hashes[right])
            )
    fresh_hashes = set().union(*(set(values) for values in role_hashes.values()))
    source_overlap: dict[str, int] = {}
    source_manifest: dict[str, Any] = {}
    for source in comparison_sources:
        path = source.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if source.kind == "query_hash_npz":
            prior = _precomputed_hashes_from_npz(path)
            arrays_read = ["query_sha256"]
        elif source.kind == "identity_npz":
            prior = _identity_hashes_from_npz(path, dt=dt)
            arrays_read = list(IDENTITY_ARRAYS)
        else:
            raise ValueError(f"unsupported freshness-source kind: {source.kind}")
        source_overlap[source.name] = len(fresh_hashes & prior)
        source_manifest[source.name] = {
            "path": str(path),
            "kind": source.kind,
            "arrays_read": arrays_read,
            "performance_arrays_read": False,
            "query_count": len(prior),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    passed = (
        all(value == 0 for value in duplicate_counts.values())
        and all(value == 0 for value in cross_role_overlap.values())
        and all(value == 0 for value in source_overlap.values())
    )
    return {
        "robot": robot,
        "fresh_query_count": sum(len(values) for values in role_hashes.values()),
        "fresh_unique_query_sha256": len(fresh_hashes),
        "within_role_exact_duplicate_counts": duplicate_counts,
        "cross_role_exact_overlap_counts": cross_role_overlap,
        "prior_source_exact_overlap_counts": source_overlap,
        "comparison_sources": source_manifest,
        "old_test_evidence_boundary": (
            "Only previous_q, target_position, and target_rotation are read from "
            "the old test dataset; no old-test performance result is inspected."
        ),
        "passed": passed,
    }


__all__ = [
    "IDENTITY_ARRAYS",
    "ROLE_DOMAIN",
    "TEST_V4_ROLES",
    "ComparisonSource",
    "audit_freshness",
    "dataset_contract",
    "dataset_query_hashes",
    "default_comparison_sources",
    "derive_seed",
    "generate_locked_datasets",
    "query_keys",
    "query_sha256",
    "validate_dataset_contract",
]
