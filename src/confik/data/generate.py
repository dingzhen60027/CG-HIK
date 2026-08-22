from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..kinematics.base import KinematicsModel
from ..runtime.hybrid import risk_features
from ..solvers.dls import AdaptiveDLS
from ..types import IKQuery
from .datasets import QueryDataset, RiskDataset, TransitionDataset


def generate_smooth_transitions(
    kinematics: KinematicsModel,
    *,
    trajectories: int,
    steps_per_trajectory: int,
    seed: int,
    dt: float = 0.02,
    margin: float = 0.1,
    velocity_fraction: float = 0.8,
    momentum: float = 0.85,
) -> TransitionDataset:
    if trajectories <= 0 or steps_per_trajectory <= 0:
        raise ValueError("trajectory and step counts must be positive")
    rng = np.random.default_rng(seed)
    previous_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    trajectory_ids: list[int] = []
    max_step = kinematics.limits.velocity * dt * velocity_fraction
    span = kinematics.limits.upper - kinematics.limits.lower
    low = kinematics.limits.lower + margin * span
    high = kinematics.limits.upper - margin * span

    for trajectory_id in range(trajectories):
        q = kinematics.random_configuration(rng, margin=margin)
        velocity = rng.normal(size=kinematics.nq)
        velocity = velocity / max(np.max(np.abs(velocity)), 1e-12) * max_step
        for _ in range(steps_per_trajectory):
            innovation = rng.normal(size=kinematics.nq)
            innovation = innovation / max(np.max(np.abs(innovation)), 1e-12) * max_step
            velocity = momentum * velocity + (1.0 - momentum) * innovation
            velocity = np.clip(velocity, -max_step, max_step)
            target_q = np.clip(q + velocity, low, high)
            velocity = target_q - q
            target_pose = kinematics.forward(target_q)
            previous_rows.append(q.copy())
            target_rows.append(target_q.copy())
            positions.append(target_pose.position)
            rotations.append(target_pose.rotation)
            trajectory_ids.append(trajectory_id)
            q = target_q
    return TransitionDataset(
        np.stack(previous_rows),
        np.stack(target_rows),
        np.stack(positions),
        np.stack(rotations),
        np.asarray(trajectory_ids),
    )


def _transitions_from_pairs(
    kinematics: KinematicsModel,
    previous_q: np.ndarray,
    target_q: np.ndarray,
    trajectory_offset: int = 0,
) -> TransitionDataset:
    poses = [kinematics.forward(q) for q in target_q]
    return TransitionDataset(
        previous_q,
        target_q,
        np.stack([pose.position for pose in poses]),
        np.stack([pose.rotation for pose in poses]),
        np.arange(len(previous_q), dtype=np.int64) + trajectory_offset,
    )


def generate_mixed_transitions(
    kinematics: KinematicsModel,
    *,
    samples: int,
    seed: int,
    challenge_fraction: float = 0.5,
    dt: float = 0.02,
) -> TransitionDataset:
    """Mix local trajectory transitions with large-step, limit, and singular queries."""
    if not 0.0 <= challenge_fraction <= 1.0:
        raise ValueError("challenge_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    challenging = int(round(samples * challenge_fraction))
    local_count = samples - challenging
    local = generate_smooth_transitions(
        kinematics,
        trajectories=max(1, local_count // 20),
        steps_per_trajectory=max(1, int(np.ceil(local_count / max(1, local_count // 20)))),
        seed=seed,
        dt=dt,
    )
    local = local.subset(np.arange(min(local_count, len(local)))) if local_count else None
    if challenging == 0:
        assert local is not None
        return local

    previous = np.stack([kinematics.random_configuration(rng, margin=0.05) for _ in range(challenging)])
    target = np.stack([kinematics.random_configuration(rng, margin=0.0) for _ in range(challenging)])
    limit_count = challenging // 3
    for index in range(limit_count):
        joint = index % kinematics.nq
        if index % 2:
            target[index, joint] = kinematics.limits.upper[joint] - 0.01 * (
                kinematics.limits.upper[joint] - kinematics.limits.lower[joint]
            )
        else:
            target[index, joint] = kinematics.limits.lower[joint] + 0.01 * (
                kinematics.limits.upper[joint] - kinematics.limits.lower[joint]
            )
    hard = _transitions_from_pairs(kinematics, previous, target, trajectory_offset=10_000_000)
    if local is None:
        return hard
    return TransitionDataset(
        np.concatenate([local.previous_q, hard.previous_q]),
        np.concatenate([local.target_q, hard.target_q]),
        np.concatenate([local.target_position, hard.target_position]),
        np.concatenate([local.target_rotation, hard.target_rotation]),
        np.concatenate([local.trajectory_id, hard.trajectory_id]),
    )


def _query_from_q(
    kinematics: KinematicsModel,
    previous: np.ndarray,
    target: np.ndarray,
    category: str,
    *,
    continuity_feasible: bool,
    trajectory_offset: int,
) -> QueryDataset:
    poses = [kinematics.forward(q) for q in target]
    count = len(previous)
    return QueryDataset(
        previous_q=previous,
        target_position=np.stack([pose.position for pose in poses]),
        target_rotation=np.stack([pose.rotation for pose in poses]),
        reference_q=target,
        category=np.full(count, category),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.full(count, continuity_feasible, dtype=bool),
        trajectory_id=np.arange(count, dtype=np.int64) + trajectory_offset,
        time_index=np.zeros(count, dtype=np.int64),
    )


def _reach_upper_bound(kinematics: KinematicsModel) -> float:
    chain = getattr(kinematics, "chain", None)
    if chain is None:
        samples = np.stack(
            [kinematics.forward(kinematics.random_configuration(np.random.default_rng(i), margin=0.0)).position for i in range(256)]
        )
        return float(np.max(np.linalg.norm(samples, axis=1)) * 1.25)
    translation_sum = sum(float(np.linalg.norm(joint.origin[:3, 3])) for joint in chain)
    prismatic_sum = sum(
        max(abs(joint.lower), abs(joint.upper)) for joint in chain if joint.kind == "prismatic"
    )
    return translation_sum + prismatic_sum


def generate_point_test_set(
    kinematics: KinematicsModel,
    id_dataset: TransitionDataset,
    *,
    per_category: int,
    id_count: int | None = None,
    seed: int,
    dt: float = 0.02,
) -> QueryDataset:
    rng = np.random.default_rng(seed)
    selected_id_count = min(id_count if id_count is not None else per_category, len(id_dataset))
    id_queries = QueryDataset.from_transitions(id_dataset.subset(np.arange(selected_id_count)), "id")
    pool_count = max(per_category * 6, per_category)
    pool = np.stack([kinematics.random_configuration(rng, margin=0.0) for _ in range(pool_count)])
    sigmas = np.array([kinematics.min_singular_value(q) for q in pool])
    singular_target = pool[np.argsort(sigmas)[:per_category]]
    max_step = kinematics.limits.velocity * dt * 0.6
    singular_previous = kinematics.clip(singular_target - rng.uniform(-max_step, max_step, singular_target.shape))
    singular = _query_from_q(
        kinematics,
        singular_previous,
        singular_target,
        "near_singular",
        continuity_feasible=True,
        trajectory_offset=20_000_000,
    )

    limit_target = np.stack([kinematics.random_configuration(rng, margin=0.0) for _ in range(per_category)])
    for index in range(per_category):
        joint = index % kinematics.nq
        span = kinematics.limits.upper[joint] - kinematics.limits.lower[joint]
        limit_target[index, joint] = (
            kinematics.limits.lower[joint] + 0.01 * span
            if index % 2 == 0
            else kinematics.limits.upper[joint] - 0.01 * span
        )
    direction = np.sign((kinematics.limits.lower + kinematics.limits.upper) / 2.0 - limit_target)
    limit_previous = kinematics.clip(limit_target + direction * max_step * 0.5)
    near_limit = _query_from_q(
        kinematics,
        limit_previous,
        limit_target,
        "near_limit",
        continuity_feasible=True,
        trajectory_offset=30_000_000,
    )

    radii = np.array([np.linalg.norm(kinematics.forward(q).position) for q in pool])
    boundary_target = pool[np.argsort(radii)[-per_category:]]
    boundary_previous = kinematics.clip(boundary_target - rng.uniform(-max_step, max_step, boundary_target.shape))
    boundary = _query_from_q(
        kinematics,
        boundary_previous,
        boundary_target,
        "workspace_boundary",
        continuity_feasible=True,
        trajectory_offset=40_000_000,
    )

    large_previous = np.stack([kinematics.random_configuration(rng, margin=0.1) for _ in range(per_category)])
    large_target = np.stack([kinematics.random_configuration(rng, margin=0.1) for _ in range(per_category)])
    large_step = _query_from_q(
        kinematics,
        large_previous,
        large_target,
        "large_step",
        continuity_feasible=False,
        trajectory_offset=50_000_000,
    )

    reach = _reach_upper_bound(kinematics)
    directions = rng.normal(size=(per_category, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    unreachable_position = directions * rng.uniform(reach + 0.05, reach + 0.25, size=(per_category, 1))
    unreachable_previous = np.stack(
        [kinematics.random_configuration(rng, margin=0.1) for _ in range(per_category)]
    )
    unreachable_rotation = np.stack([kinematics.forward(q).rotation for q in unreachable_previous])
    unreachable = QueryDataset(
        previous_q=unreachable_previous,
        target_position=unreachable_position,
        target_rotation=unreachable_rotation,
        reference_q=np.full_like(unreachable_previous, np.nan),
        category=np.full(per_category, "unreachable"),
        expected_reachable=np.zeros(per_category, dtype=bool),
        continuity_feasible=np.zeros(per_category, dtype=bool),
        trajectory_id=np.arange(per_category, dtype=np.int64) + 60_000_000,
        time_index=np.zeros(per_category, dtype=np.int64),
    )
    return QueryDataset.concatenate([id_queries, singular, near_limit, boundary, large_step, unreachable])


def _orthonormal_plane(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    first = rng.normal(size=3)
    first /= np.linalg.norm(first)
    second = rng.normal(size=3)
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second)
    return first, second


def _path_offsets(
    kind: str,
    steps: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    u, v = _orthonormal_plane(rng)
    phase = np.linspace(0.0, 1.0, steps)
    if kind == "line":
        scalar = amplitude * np.sin(np.pi * phase)
        return scalar[:, None] * u
    theta = 2.0 * np.pi * phase
    if kind == "circle":
        return amplitude * ((np.cos(theta) - 1.0)[:, None] * u + np.sin(theta)[:, None] * v)
    if kind == "figure8":
        return amplitude * (np.sin(theta)[:, None] * u + 0.5 * np.sin(2.0 * theta)[:, None] * v)
    if kind == "spline":
        knots = np.linspace(0.0, 1.0, 6)
        controls = rng.normal(size=(6, 2))
        controls[0] = 0.0
        controls[-1] = 0.0
        controls *= amplitude / max(np.max(np.linalg.norm(controls, axis=1)), 1e-12)
        values = CubicSpline(knots, controls, bc_type="natural")(phase)
        return values[:, :1] * u + values[:, 1:] * v
    raise ValueError(f"unknown path kind {kind}")


def generate_cartesian_path_tests(
    kinematics: KinematicsModel,
    dls: AdaptiveDLS,
    *,
    paths_per_type: int,
    steps: int,
    seed: int,
    dt: float = 0.02,
    amplitude: float = 0.03,
    max_attempts_per_path: int = 30,
) -> QueryDataset:
    """Generate verified line, circle, figure-eight, and spline Cartesian paths."""
    rng = np.random.default_rng(seed)
    previous_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    categories: list[str] = []
    trajectory_ids: list[int] = []
    time_indices: list[int] = []
    next_trajectory_id = 100_000_000
    velocity_bound = kinematics.limits.velocity * dt + 1e-4

    for kind in ("line", "circle", "figure8", "spline"):
        completed = 0
        attempts = 0
        while completed < paths_per_type and attempts < paths_per_type * max_attempts_per_path:
            attempts += 1
            start_q = kinematics.random_configuration(rng, margin=0.15)
            start_pose = kinematics.forward(start_q)
            offsets = _path_offsets(kind, steps, amplitude, rng)
            path_previous: list[np.ndarray] = []
            path_reference: list[np.ndarray] = []
            path_positions: list[np.ndarray] = []
            path_rotations: list[np.ndarray] = []
            q = start_q
            valid = True
            for offset in offsets:
                target_pose = type(start_pose)(start_pose.position + offset, start_pose.rotation)
                trace = dls.solve(target_pose, q, 100, seed_source="path_oracle")
                if trace.q is None or not trace.converged:
                    valid = False
                    break
                if np.any(np.abs(trace.q - q) > velocity_bound):
                    valid = False
                    break
                path_previous.append(q.copy())
                path_reference.append(trace.q.copy())
                path_positions.append(target_pose.position)
                path_rotations.append(target_pose.rotation)
                q = trace.q
            if not valid or len(path_previous) != steps:
                continue
            previous_rows.extend(path_previous)
            reference_rows.extend(path_reference)
            positions.extend(path_positions)
            rotations.extend(path_rotations)
            categories.extend([f"trajectory_{kind}"] * steps)
            trajectory_ids.extend([next_trajectory_id] * steps)
            time_indices.extend(range(steps))
            next_trajectory_id += 1
            completed += 1
        if completed < paths_per_type:
            raise RuntimeError(
                f"generated only {completed}/{paths_per_type} valid {kind} paths; "
                "reduce amplitude or increase attempts"
            )
    count = len(previous_rows)
    return QueryDataset(
        previous_q=np.stack(previous_rows),
        target_position=np.stack(positions),
        target_rotation=np.stack(rotations),
        reference_q=np.stack(reference_rows),
        category=np.asarray(categories),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.asarray(trajectory_ids),
        time_index=np.asarray(time_indices),
    )


def label_solver_risk(
    kinematics: KinematicsModel,
    candidate_provider: object,
    dls: AdaptiveDLS,
    dataset: TransitionDataset,
    *,
    max_iterations: int = 50,
    dt: float = 0.02,
) -> RiskDataset:
    features: list[np.ndarray] = []
    labels: list[int] = []
    iterations: list[int] = []
    converged: list[bool] = []
    for index in range(len(dataset)):
        target = kinematics.forward(dataset.target_q[index])
        query = IKQuery(target, dataset.previous_q[index], dt=dt)
        candidates = candidate_provider.candidates(query)  # type: ignore[attr-defined]
        trace = dls.solve(target, candidates.joints[0], max_iterations, seed_source="risk_label")
        features.append(risk_features(kinematics, query, candidates))
        iterations.append(trace.iterations)
        converged.append(trace.converged)
        if not trace.converged:
            labels.append(3)
        elif trace.iterations <= 8:
            labels.append(0)
        elif trace.iterations <= 25:
            labels.append(1)
        else:
            labels.append(2)
    return RiskDataset(
        np.stack(features),
        np.asarray(labels),
        np.asarray(iterations),
        np.asarray(converged),
    )
