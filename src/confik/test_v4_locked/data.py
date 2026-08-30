from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
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
PAPER_V2_SEEDS = (17, 29, 43)
PAPER_V2_QUERY_ROLES = (
    "seed_train",
    "seed_validation",
    "risk_train_queries",
    "risk_validation_queries",
    "risk_test_queries",
    "calibration_queries",
    "policy_validation_queries",
    "test_id",
    "test_queries",
)
COUNTERFACTUAL_V4_QUERY_ROOTS = (
    "counterfactual_v4_pilot",
    "counterfactual_v4_smoke",
    "counterfactual_v4_readiness_smoke",
    "counterfactual_v4_readiness_smoke_r2",
)
BULK_QUERY_ROLES = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)


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
    group: str = "custom"
    role: str = "unspecified"
    source_indices: tuple[int, ...] = ()
    trajectory_ids: tuple[int, ...] = ()
    provenance_path: Path | None = None


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


def _identity_hashes_from_npz(
    path: Path,
    *,
    dt: float,
    source_indices: Sequence[int] = (),
    trajectory_ids: Sequence[int] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Read identity arrays only; extra arrays (including metrics) are ignored.

    Optional selectors reproduce the exact validation subset used by
    ``latency_pilot_v3``.  They are mutually exclusive and validated rather
    than clipped, deduplicated, or silently ignored.
    """

    source_index_values = tuple(int(value) for value in source_indices)
    trajectory_id_values = tuple(int(value) for value in trajectory_ids)
    if source_index_values and trajectory_id_values:
        raise ValueError(
            "identity source cannot use index and trajectory selectors together"
        )

    with np.load(path, allow_pickle=False) as arrays:
        missing = set(IDENTITY_ARRAYS) - set(arrays.files)
        if missing:
            raise KeyError(f"identity source {path} lacks arrays: {sorted(missing)}")
        previous_q = np.asarray(arrays["previous_q"], dtype=np.float64)
        target_position = np.asarray(arrays["target_position"], dtype=np.float64)
        target_rotation = np.asarray(arrays["target_rotation"], dtype=np.float64)
        trajectory_id = (
            np.asarray(arrays["trajectory_id"], dtype=np.int64)
            if trajectory_id_values
            else None
        )
    count = len(previous_q)
    if (
        previous_q.ndim != 2
        or target_position.shape != (count, 3)
        or target_rotation.shape != (count, 3, 3)
        or not np.all(np.isfinite(previous_q))
        or not np.all(np.isfinite(target_position))
        or not np.all(np.isfinite(target_rotation))
    ):
        raise ValueError(f"identity source {path} has incompatible query shapes")
    selected = np.arange(count, dtype=np.int64)
    selector: dict[str, Any] = {"type": "all_rows", "selected_count": count}
    if source_index_values:
        selected = np.asarray(source_index_values, dtype=np.int64)
        if (
            selected.ndim != 1
            or len(selected) != len(np.unique(selected))
            or np.any(selected < 0)
            or np.any(selected >= count)
        ):
            raise ValueError(f"identity source {path} has invalid source indices")
        selector = {
            "type": "source_indices",
            "selected_count": len(selected),
            "selector_sha256": sha256(
                np.ascontiguousarray(selected, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
    elif trajectory_id_values:
        if trajectory_id is None or trajectory_id.shape != (count,):
            raise KeyError(f"trajectory identity source {path} lacks trajectory_id")
        requested = np.asarray(trajectory_id_values, dtype=np.int64)
        if requested.ndim != 1 or len(requested) != len(np.unique(requested)):
            raise ValueError(f"identity source {path} has invalid trajectory ids")
        missing_ids = sorted(set(requested.tolist()) - set(trajectory_id.tolist()))
        if missing_ids:
            raise ValueError(
                f"identity source {path} lacks requested trajectory ids: {missing_ids}"
            )
        selected = np.flatnonzero(np.isin(trajectory_id, requested))
        selector = {
            "type": "trajectory_ids",
            "trajectory_count": len(requested),
            "selected_count": len(selected),
            "selector_sha256": sha256(
                np.ascontiguousarray(requested, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
    hashes = [
        query_sha256(
            previous_q[index], target_position[index], target_rotation[index], dt=dt
        )
        for index in selected.tolist()
    ]
    return hashes, selector


def _precomputed_hashes_from_npz(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as arrays:
        if "query_sha256" not in arrays.files:
            raise KeyError(f"query-hash source {path} lacks query_sha256")
        values = np.asarray(arrays["query_sha256"]).astype(str).reshape(-1)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError(f"query-hash source {path} contains a malformed digest")
    return values.tolist()


def _precomputed_hashes_from_jsonl_gz(path: Path) -> list[str]:
    """Read only the query identity field from an incomplete pilot record."""

    values: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            value = str(payload.get("source_query_sha256", payload.get("query_sha256", "")))
            if (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"query-hash source {path}:{line_number} contains a malformed digest"
                )
            values.append(value)
    if not values:
        raise ValueError(f"query-hash source {path} is empty")
    return values


def _czy_closed_loop_identity_hashes(path: Path, *, dt: float) -> list[str]:
    """Hash only the IK identity columns from the prior Panda closed-loop CSV.

    The CSV contains outcome and timing columns, but they are neither converted
    nor inspected here.  The prior controller query is reconstructed from the
    logged pre-command actual joint state and target end-effector pose.
    """

    required = (
        "actual_joint_state",
        "target_ee_position",
        "target_ee_rotation",
        "dt_s",
    )
    values: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"closed-loop identity source {path} is empty") from error
        if len(header) != len(set(header)) or any(name not in header for name in required):
            raise KeyError(f"closed-loop identity source {path} lacks identity columns")
        indices = {name: header.index(name) for name in required}
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            observed_dt = float(row[indices["dt_s"]])
            if not np.isclose(observed_dt, dt, rtol=0.0, atol=1.0e-15):
                raise ValueError(
                    f"closed-loop identity source {path}:{line_number} changed dt"
                )
            previous_q = np.asarray(
                json.loads(row[indices["actual_joint_state"]]), dtype=np.float64
            )
            target_position = np.asarray(
                json.loads(row[indices["target_ee_position"]]), dtype=np.float64
            )
            target_rotation = np.asarray(
                json.loads(row[indices["target_ee_rotation"]]), dtype=np.float64
            )
            if (
                previous_q.ndim != 1
                or target_position.shape != (3,)
                or target_rotation.shape != (3, 3)
                or not np.all(np.isfinite(previous_q))
                or not np.all(np.isfinite(target_position))
                or not np.all(np.isfinite(target_rotation))
            ):
                raise ValueError(
                    f"closed-loop identity source {path}:{line_number} has invalid shapes"
                )
            values.append(
                query_sha256(previous_q, target_position, target_rotation, dt=dt)
            )
    if not values:
        raise ValueError(f"closed-loop identity source {path} has no query rows")
    return values


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON source is not a mapping: {path}")
    return payload


def default_comparison_sources(workspace: Path, robot: str) -> list[ComparisonSource]:
    """Return the exhaustive, fail-closed freshness comparison contract.

    The list intentionally contains supersets as well as selected subsets.
    Every paper-v2 query role for all three seeds is covered; the selected bulk
    and pilot identities preserve direct provenance; and latency-pilot entries
    reproduce its exact point/trajectory validation selections.  Constructing
    this list requires the frozen latency manifest, so a missing expected input
    fails before fresh test generation.
    """

    root = workspace.resolve()
    bulk = root / "outputs" / "counterfactual_v4_bulk" / robot / "seed17"
    sources: list[ComparisonSource] = []
    for seed in PAPER_V2_SEEDS:
        datasets = root / "outputs" / f"paper_v2_seed{seed}" / robot / "datasets"
        sources.extend(
            ComparisonSource(
                name=f"paper_v2/seed{seed}/{role}",
                path=datasets / f"{role}.npz",
                kind="identity_npz",
                group="paper_v2",
                role=role,
            )
            for role in PAPER_V2_QUERY_ROLES
        )
    sources.extend(
        [
            ComparisonSource(
                name=f"bulk/{role}",
                path=bulk / role / "selection.npz",
                kind="query_hash_npz",
                group="counterfactual_v4_bulk",
                role=role,
            )
            for role in BULK_QUERY_ROLES
        ]
    )
    sources.extend(
        ComparisonSource(
            name=f"{directory}/seed17",
            path=(
                root
                / "outputs"
                / directory
                / robot
                / "seed17"
                / "counterfactual_labels.npz"
            ),
            kind="query_hash_npz",
            group=directory,
            role="selected_queries",
        )
        for directory in COUNTERFACTUAL_V4_QUERY_ROOTS
    )

    latency_manifest_path = root / "outputs/latency_pilot_v3/run_manifest.json"
    if not latency_manifest_path.is_file() or latency_manifest_path.is_symlink():
        raise FileNotFoundError(latency_manifest_path)
    latency_manifest = _strict_json(latency_manifest_path)
    if (
        latency_manifest.get("protocol_version") != "latency_pilot_v3"
        or latency_manifest.get("formal_test_v3_started") is not False
    ):
        raise RuntimeError("latency-pilot manifest is not validation-only v3 evidence")
    try:
        selection = latency_manifest["selection_inputs"][robot]
    except KeyError as error:
        raise KeyError(f"latency-pilot manifest lacks selection for {robot}") from error
    if (
        selection.get("test_queries_loaded") is not False
        or selection.get("point_source_split") != "risk_validation_queries"
        or selection.get("trajectory_source_split") != "seed_validation"
    ):
        raise RuntimeError(f"latency-pilot selection contract changed for {robot}")
    point_indices = tuple(
        int(value) for value in selection["point_selected_source_indices"]
    )
    selected_trajectories = tuple(
        int(value) for value in selection["selected_trajectory_ids"]
    )
    if not point_indices or not selected_trajectories:
        raise RuntimeError(f"latency-pilot selection is empty for {robot}")
    old_v2 = root / "outputs/paper_v2_seed17" / robot / "datasets"
    sources.extend(
        [
            ComparisonSource(
                name="latency_pilot_v3/point_validation_selection",
                path=old_v2 / "risk_validation_queries.npz",
                kind="indexed_identity_npz",
                group="latency_pilot_v3",
                role="risk_validation_queries",
                source_indices=point_indices,
                provenance_path=latency_manifest_path,
            ),
            ComparisonSource(
                name="latency_pilot_v3/trajectory_validation_selection",
                path=old_v2 / "seed_validation.npz",
                kind="trajectory_identity_npz",
                group="latency_pilot_v3",
                role="seed_validation",
                trajectory_ids=selected_trajectories,
                provenance_path=latency_manifest_path,
            ),
        ]
    )
    sources.append(
        ComparisonSource(
            name="old_formal_test_v3_identity",
            path=(
                root
                / "outputs"
                / "test_v3_aggregate"
                / "datasets"
                / f"{robot}_test_v3_queries.npz"
            ),
            kind="identity_npz",
            group="old_formal_test_v3",
            role="query_identity_only",
        )
    )
    # A prior Panda closed-loop study and one interrupted readiness smoke were
    # inspected during development.  They are therefore included whenever the
    # corresponding identity evidence exists, even though neither is a model
    # training source.  No performance field is consumed by the freshness test.
    if robot == "panda":
        czy_path = root / "czy" / "closed_loop_v3_raw_frame_records.csv"
        if czy_path.is_file() and not czy_path.is_symlink():
            sources.append(
                ComparisonSource(
                    name="czy/closed_loop_v3_query_identity",
                    path=czy_path,
                    kind="czy_identity_csv",
                    group="prior_closed_loop_czy",
                    role="query_identity_only",
                )
            )
    incomplete_readiness = (
        root
        / "outputs/.counterfactual_v4_readiness_smoke.incomplete.1313949"
        / robot
        / "seed17/counterfactual_records.jsonl.gz"
    )
    if incomplete_readiness.is_file() and not incomplete_readiness.is_symlink():
        sources.append(
            ComparisonSource(
                name="counterfactual_v4_readiness_smoke_incomplete/seed17",
                path=incomplete_readiness,
                kind="query_hash_jsonl_gz",
                group="counterfactual_v4_readiness_smoke_incomplete",
                role="selected_queries",
            )
        )
    expected_count = (
        len(PAPER_V2_SEEDS) * len(PAPER_V2_QUERY_ROLES)
        + len(BULK_QUERY_ROLES)
        + len(COUNTERFACTUAL_V4_QUERY_ROOTS)
        + 2
        + 1
    )
    names = [source.name for source in sources]
    if len(sources) < expected_count or len(names) != len(set(names)):
        raise RuntimeError("freshness comparison-source contract is incomplete or ambiguous")
    missing = [
        str(source.path.resolve())
        for source in sources
        if not source.path.is_file() or source.path.is_symlink()
    ]
    if missing:
        raise FileNotFoundError(
            "required freshness sources are missing or symlinks:\n" + "\n".join(missing)
        )
    return sources


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
    if not comparison_sources:
        raise ValueError("freshness comparison sources must not be empty")
    source_names = [source.name for source in comparison_sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError("freshness comparison-source names must be unique")
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
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if source.kind == "query_hash_npz":
            prior_values = _precomputed_hashes_from_npz(path)
            arrays_read = ["query_sha256"]
            selector = {
                "type": "all_precomputed_hashes",
                "selected_count": len(prior_values),
            }
        elif source.kind == "query_hash_jsonl_gz":
            prior_values = _precomputed_hashes_from_jsonl_gz(path)
            arrays_read = ["source_query_sha256"]
            selector = {
                "type": "all_precomputed_hashes",
                "selected_count": len(prior_values),
            }
        elif source.kind == "czy_identity_csv":
            prior_values = _czy_closed_loop_identity_hashes(path, dt=dt)
            arrays_read = [
                "actual_joint_state",
                "target_ee_position",
                "target_ee_rotation",
                "dt_s",
            ]
            selector = {
                "type": "all_identity_rows",
                "selected_count": len(prior_values),
            }
        elif source.kind == "identity_npz":
            prior_values, selector = _identity_hashes_from_npz(path, dt=dt)
            arrays_read = list(IDENTITY_ARRAYS)
        elif source.kind == "indexed_identity_npz":
            prior_values, selector = _identity_hashes_from_npz(
                path, dt=dt, source_indices=source.source_indices
            )
            arrays_read = list(IDENTITY_ARRAYS)
        elif source.kind == "trajectory_identity_npz":
            prior_values, selector = _identity_hashes_from_npz(
                path, dt=dt, trajectory_ids=source.trajectory_ids
            )
            arrays_read = [*IDENTITY_ARRAYS, "trajectory_id"]
        else:
            raise ValueError(f"unsupported freshness-source kind: {source.kind}")
        prior = set(prior_values)
        source_overlap[source.name] = len(fresh_hashes & prior)
        provenance: dict[str, Any] | None = None
        if source.provenance_path is not None:
            provenance_path = source.provenance_path.resolve()
            if not provenance_path.is_file() or provenance_path.is_symlink():
                raise FileNotFoundError(provenance_path)
            provenance = {
                "path": str(provenance_path),
                "sha256": _sha256_file(provenance_path),
                "size": provenance_path.stat().st_size,
            }
        source_manifest[source.name] = {
            "path": str(path),
            "kind": source.kind,
            "group": source.group,
            "role": source.role,
            "arrays_read": arrays_read,
            "performance_arrays_read": False,
            "query_count": len(prior_values),
            "unique_query_sha256": len(prior),
            "within_source_exact_duplicate_count": len(prior_values) - len(prior),
            "query_sha256_set_digest": sha256(
                "".join(sorted(prior)).encode("ascii")
            ).hexdigest(),
            "selector": selector,
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
            "provenance": provenance,
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
        "comparison_source_count": len(comparison_sources),
        "comparison_source_group_counts": dict(
            sorted(Counter(source.group for source in comparison_sources).items())
        ),
        "comparison_source_contract": {
            "paper_v2_seeds": list(PAPER_V2_SEEDS),
            "paper_v2_query_roles": list(PAPER_V2_QUERY_ROLES),
            "counterfactual_v4_query_roots": list(COUNTERFACTUAL_V4_QUERY_ROOTS),
            "bulk_query_roles": list(BULK_QUERY_ROLES),
            "latency_pilot_exact_validation_selections": True,
            "old_formal_test_identity_only": True,
            "missing_expected_source_policy": "fail_closed",
        },
        "old_test_evidence_boundary": (
            "Only previous_q, target_position, and target_rotation are read from "
            "the old test dataset; no old-test performance result is inspected."
        ),
        "passed": passed,
    }


__all__ = [
    "BULK_QUERY_ROLES",
    "COUNTERFACTUAL_V4_QUERY_ROOTS",
    "IDENTITY_ARRAYS",
    "PAPER_V2_QUERY_ROLES",
    "PAPER_V2_SEEDS",
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
