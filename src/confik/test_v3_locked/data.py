from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data.datasets import QueryDataset, TransitionDataset
from ..data.generate import generate_point_test_set, generate_smooth_transitions
from ..data.generate_v2 import (
    generate_hard_valid_queries,
    generate_reference_trajectory_tests,
)
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier


def derive_seed(release_commit: str, robot: str, role: str) -> int:
    """Derive an unsigned 32-bit generation seed from the frozen release."""

    material = f"test_v3_locked|{release_commit}|{robot}|{role}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:4], "big", signed=False)


def query_rows(dataset: QueryDataset) -> set[bytes]:
    matrices = (
        dataset.previous_q.reshape(len(dataset), -1),
        dataset.target_position.reshape(len(dataset), -1),
        dataset.target_rotation.reshape(len(dataset), -1),
    )
    numeric = np.ascontiguousarray(np.concatenate(matrices, axis=1), dtype=np.float64)
    width = numeric.dtype.itemsize * numeric.shape[1]
    return set(numeric.view(np.dtype((np.void, width))).reshape(-1).tolist())


def transition_rows(dataset: TransitionDataset) -> set[bytes]:
    matrices = (
        dataset.previous_q.reshape(len(dataset), -1),
        dataset.target_position.reshape(len(dataset), -1),
        dataset.target_rotation.reshape(len(dataset), -1),
    )
    numeric = np.ascontiguousarray(np.concatenate(matrices, axis=1), dtype=np.float64)
    width = numeric.dtype.itemsize * numeric.shape[1]
    return set(numeric.view(np.dtype((np.void, width))).reshape(-1).tolist())


def generate_locked_dataset(
    *,
    kinematics: object,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    release_commit: str,
    robot: str,
    config: dict[str, Any],
) -> tuple[QueryDataset, dict[str, int]]:
    data = config["data"]
    dt = float(data["dt"])
    seeds = {
        role: derive_seed(release_commit, robot, role)
        for role in ("id_transitions", "point_stress", "hard_valid", "trajectories")
    }
    id_transitions = generate_smooth_transitions(
        kinematics,  # type: ignore[arg-type]
        trajectories=int(data["id_count"]),
        steps_per_trajectory=1,
        seed=seeds["id_transitions"],
        dt=dt,
        margin=0.1,
    )
    points = generate_point_test_set(
        kinematics,  # type: ignore[arg-type]
        id_transitions,
        per_category=int(data["per_stress_category"]),
        id_count=int(data["id_count"]),
        seed=seeds["point_stress"],
        dt=dt,
    )
    hard = generate_hard_valid_queries(
        kinematics,  # type: ignore[arg-type]
        dls,
        verifier,
        count=int(data["hard_valid_count"]),
        seed=seeds["hard_valid"],
        dt=dt,
        easy_iterations=int(data["hard_screening_easy_iterations"]),
        hard_iterations=int(data["hard_screening_robust_iterations"]),
    )
    trajectories = generate_reference_trajectory_tests(
        kinematics,  # type: ignore[arg-type]
        paths_per_type=int(data["trajectory_paths_per_type"]),
        steps=int(data["trajectory_steps"]),
        seed=seeds["trajectories"],
        dt=dt,
    )
    return QueryDataset.concatenate([points, hard, trajectories]), seeds


def dataset_schema(dataset: QueryDataset) -> dict[str, Any]:
    closed_loop = np.char.startswith(dataset.category.astype(str), "trajectory_")
    point = ~closed_loop
    feasible = dataset.expected_reachable & dataset.continuity_feasible & point
    rejectable = (~(dataset.expected_reachable & dataset.continuity_feasible)) & point
    trajectory_ids = np.unique(dataset.trajectory_id[closed_loop])
    categories = Counter(dataset.category.astype(str).tolist())
    return {
        "total_queries": int(len(dataset)),
        "point_queries": int(np.sum(point)),
        "known_feasible_point_queries": int(np.sum(feasible)),
        "rejectable_point_queries": int(np.sum(rejectable)),
        "trajectory_frames": int(np.sum(closed_loop)),
        "trajectory_count": int(len(trajectory_ids)),
        "category_counts": dict(sorted(categories.items())),
        "array_shapes": {
            name: list(getattr(dataset, name).shape)
            for name in dataset.__dataclass_fields__
        },
        "finite_required_arrays": bool(
            np.all(np.isfinite(dataset.previous_q))
            and np.all(np.isfinite(dataset.target_position))
            and np.all(np.isfinite(dataset.target_rotation))
        ),
    }


def validate_schema(schema: dict[str, Any], config: dict[str, Any]) -> None:
    data = config["data"]
    expected = {
        "total_queries": int(data["expected_total_queries"]),
        "point_queries": int(data["expected_point_queries"]),
        "known_feasible_point_queries": int(data["expected_feasible_points"]),
        "rejectable_point_queries": int(data["expected_rejectable_points"]),
        "trajectory_count": int(data["expected_trajectories"]),
    }
    actual = {key: schema[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"test_v3 schema count mismatch: expected={expected}, actual={actual}")
    expected_categories = {
        "id": 5000,
        "near_singular": 1000,
        "near_limit": 1000,
        "workspace_boundary": 1000,
        "large_step": 1000,
        "unreachable": 1000,
        "hard_valid": 2000,
        "trajectory_smooth": 1500,
        "trajectory_orientation": 1500,
        "trajectory_singular": 1500,
        "trajectory_limit": 1500,
    }
    if schema["category_counts"] != expected_categories:
        raise RuntimeError("test_v3 category composition differs from the preregistration")
    if not schema["finite_required_arrays"]:
        raise RuntimeError("test_v3 contains a non-finite required query array")


def split_audit(
    dataset: QueryDataset,
    *,
    comparison_files: Iterable[tuple[str, Path, str]],
) -> dict[str, Any]:
    new_rows = query_rows(dataset)
    overlaps: dict[str, int] = {}
    source_hashes: dict[str, dict[str, Any]] = {}
    for role, path, kind in comparison_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        if kind == "query":
            rows = query_rows(QueryDataset.load(path))
        elif kind == "transition":
            rows = transition_rows(TransitionDataset.load(path))
        else:
            raise ValueError(f"unsupported split-audit kind: {kind}")
        overlaps[role] = len(new_rows & rows)
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        source_hashes[role] = {
            "path": str(path),
            "kind": kind,
            "row_count": len(rows),
            "sha256": digest.hexdigest(),
            "size": path.stat().st_size,
        }
    duplicate_count = len(dataset) - len(new_rows)
    passed = duplicate_count == 0 and all(value == 0 for value in overlaps.values())
    return {
        "test_v3_row_count": len(dataset),
        "test_v3_unique_row_count": len(new_rows),
        "within_test_v3_exact_duplicate_count": duplicate_count,
        "exact_overlap_counts": overlaps,
        "comparison_sources": source_hashes,
        "latency_pilot_coverage": (
            "latency_pilot_v3 used subsets of seed_validation and risk_validation_queries; "
            "zero overlap with those full source splits proves zero pilot-query reuse"
        ),
        "passed": passed,
    }
