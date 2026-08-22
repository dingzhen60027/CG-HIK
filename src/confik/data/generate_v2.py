from __future__ import annotations

import numpy as np

from ..kinematics.base import KinematicsModel
from ..runtime.cascade import CascadedHybridIK, EntryAction
from ..runtime.hybrid import risk_features
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier
from ..types import IKQuery, Pose
from .datasets import QueryDataset, RiskDataset, TransitionDataset
from .generate import generate_point_test_set, generate_smooth_transitions


def generate_mixed_query_set(
    kinematics: KinematicsModel,
    *,
    samples: int,
    seed: int,
    challenge_fraction: float = 0.6,
    dt: float = 0.02,
) -> QueryDataset:
    """Generate action-label queries with reachable, discontinuous, and unreachable cases."""
    if samples < 6:
        raise ValueError("mixed query set requires at least six samples")
    if not 0.0 <= challenge_fraction <= 1.0:
        raise ValueError("challenge_fraction must lie in [0, 1]")
    challenging = max(5, int(round(samples * challenge_fraction)))
    challenging = min(challenging, samples)
    per_category = max(1, challenging // 5)
    stress_total = 5 * per_category
    local_count = samples - stress_total
    if local_count <= 0:
        per_category = max(1, (samples - 1) // 5)
        stress_total = 5 * per_category
        local_count = samples - stress_total
    # Point-query inference treats one generated transition as one independent
    # unit. Sequential correlation is evaluated only in the dedicated path set.
    trajectories = local_count
    local = generate_smooth_transitions(
        kinematics,
        trajectories=trajectories,
        steps_per_trajectory=1,
        seed=seed + 1,
        dt=dt,
        velocity_fraction=0.9,
    )
    return generate_point_test_set(
        kinematics,
        local,
        per_category=per_category,
        id_count=local_count,
        seed=seed,
        dt=dt,
    )


def generate_hard_valid_queries(
    kinematics: KinematicsModel,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    *,
    count: int,
    seed: int,
    dt: float = 0.02,
    easy_iterations: int = 1,
    hard_iterations: int = 25,
    max_attempts_per_query: int = 120,
) -> QueryDataset:
    """Screen known-feasible local queries that exceed the easy DLS budget.

    Selection uses only the previous-state numerical baseline and a known joint
    reference. It does not use the learned gate or learned seed, so the locked
    challenge set can be shared across training seeds.
    """
    if count <= 0:
        raise ValueError("hard-valid query count must be positive")
    rng = np.random.default_rng(seed)
    previous_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    attempts = 0
    max_attempts = count * max_attempts_per_query
    max_delta = kinematics.limits.velocity * dt * 0.98

    while len(previous_rows) < count and attempts < max_attempts:
        attempts += 1
        mode = attempts % 3
        if mode == 0:
            pool = np.stack([kinematics.random_configuration(rng, margin=0.0) for _ in range(8)])
            sigmas = np.array([kinematics.min_singular_value(q) for q in pool])
            target_q = pool[int(np.argmin(sigmas))]
        else:
            target_q = kinematics.random_configuration(rng, margin=0.02)
            if mode == 1:
                joint = attempts % kinematics.nq
                span = kinematics.limits.upper[joint] - kinematics.limits.lower[joint]
                target_q[joint] = (
                    kinematics.limits.lower[joint] + 0.015 * span
                    if attempts % 2
                    else kinematics.limits.upper[joint] - 0.015 * span
                )
        direction = rng.normal(size=kinematics.nq)
        direction /= max(np.max(np.abs(direction)), 1e-12)
        magnitude = rng.uniform(0.70, 0.98)
        previous_q = kinematics.clip(target_q - direction * max_delta * magnitude)
        target = kinematics.forward(target_q)
        query = IKQuery(target, previous_q, dt=dt)
        if not verifier.check(target_q, query).accepted:
            continue
        easy = dls.solve(target, previous_q, easy_iterations, seed_source="hard_screen:easy")
        easy_ok = easy.converged and easy.q is not None and verifier.check(easy.q, query).accepted
        if easy_ok:
            continue
        robust = dls.solve(target, previous_q, hard_iterations, seed_source="hard_screen:robust")
        robust_ok = robust.converged and robust.q is not None and verifier.check(robust.q, query).accepted
        if not robust_ok:
            continue
        previous_rows.append(previous_q)
        reference_rows.append(target_q)
        positions.append(target.position)
        rotations.append(target.rotation)

    if len(previous_rows) < count:
        raise RuntimeError(
            f"generated only {len(previous_rows)}/{count} hard-valid queries after {attempts} attempts; "
            "increase the easy/robust budget gap or max_attempts_per_query"
        )
    return QueryDataset(
        previous_q=np.stack(previous_rows),
        target_position=np.stack(positions),
        target_rotation=np.stack(rotations),
        reference_q=np.stack(reference_rows),
        category=np.full(count, "hard_valid"),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.arange(count, dtype=np.int64) + 70_000_000,
        time_index=np.zeros(count, dtype=np.int64),
    )


def _trajectory_reference(
    kinematics: KinematicsModel,
    kind: str,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, steps)
    if kind == "singular":
        pool = np.stack([kinematics.random_configuration(rng, margin=0.08) for _ in range(256)])
        sigmas = np.array([kinematics.min_singular_value(q) for q in pool])
        start = pool[int(np.argmin(sigmas))]
    else:
        start = kinematics.random_configuration(rng, margin=0.12)
    span = kinematics.limits.upper - kinematics.limits.lower
    velocity_amplitude = kinematics.limits.velocity * 0.02 * max(steps - 1, 1) / (2.0 * np.pi) * 0.55
    amplitude = np.minimum(0.04 * span, velocity_amplitude)
    weights = rng.uniform(0.25, 1.0, size=kinematics.nq)

    if kind == "orientation":
        weights[: max(0, kinematics.nq - 3)] *= 0.15
        weights[-min(3, kinematics.nq) :] = 1.0
    elif kind == "singular":
        amplitude *= 0.35
    elif kind == "limit":
        joint = int(rng.integers(0, kinematics.nq))
        start[joint] = kinematics.limits.lower[joint] + 0.025 * span[joint]
        weights *= 0.35
        weights[joint] = 1.0

    phases = rng.uniform(0.0, 2.0 * np.pi, size=kinematics.nq)
    offsets = np.sin(phase[:, None] + phases[None, :]) - np.sin(phases)[None, :]
    if kind == "limit":
        offsets[:, joint] = 0.5 * (1.0 - np.cos(phase))
    q = start[None, :] + offsets * (amplitude * weights)[None, :]

    lower_slack = np.min(q - kinematics.limits.lower, axis=0)
    upper_slack = np.min(kinematics.limits.upper - q, axis=0)
    shift = np.where(lower_slack < 0.0, -lower_slack, 0.0) - np.where(upper_slack < 0.0, -upper_slack, 0.0)
    q = np.stack([kinematics.clip(row + shift) for row in q])
    return q


def generate_reference_trajectory_tests(
    kinematics: KinematicsModel,
    *,
    paths_per_type: int,
    steps: int,
    seed: int,
    dt: float = 0.02,
) -> QueryDataset:
    """Generate FK targets from independent, known-feasible joint references."""
    if paths_per_type <= 0 or steps < 2:
        raise ValueError("reference trajectories require positive paths and at least two steps")
    rng = np.random.default_rng(seed)
    previous_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    categories: list[str] = []
    trajectory_ids: list[int] = []
    time_indices: list[int] = []
    next_id = 200_000_000
    for kind in ("smooth", "orientation", "singular", "limit"):
        for _ in range(paths_per_type):
            reference = _trajectory_reference(kinematics, kind, steps, rng)
            previous = np.vstack([reference[0], reference[:-1]])
            delta = np.stack(
                [np.abs(kinematics.difference(q, p)) for q, p in zip(reference, previous, strict=True)]
            )
            if np.any(delta > kinematics.limits.velocity * dt + 1e-9):
                raise RuntimeError(f"generated {kind} reference exceeds the velocity contract")
            poses = [kinematics.forward(q) for q in reference]
            previous_rows.extend(previous)
            reference_rows.extend(reference)
            positions.extend(pose.position for pose in poses)
            rotations.extend(pose.rotation for pose in poses)
            categories.extend([f"trajectory_{kind}"] * steps)
            trajectory_ids.extend([next_id] * steps)
            time_indices.extend(range(steps))
            next_id += 1
    count = len(previous_rows)
    return QueryDataset(
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


def label_cascade_actions(
    kinematics: KinematicsModel,
    cascade: CascadedHybridIK,
    dataset: QueryDataset,
    *,
    dt: float = 0.02,
) -> RiskDataset:
    """Label the first verified cascade stage, or reject when none is valid."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    effort: list[int] = []
    solved: list[bool] = []
    for index in range(len(dataset)):
        query = IKQuery(
            Pose(dataset.target_position[index], dataset.target_rotation[index]),
            dataset.previous_q[index],
            dt=dt,
        )
        candidates = cascade.candidate_provider.candidates(query)
        features.append(risk_features(kinematics, query, candidates))
        if not dataset.expected_reachable[index] or not dataset.continuity_feasible[index]:
            labels.append(int(EntryAction.REJECT))
            effort.append(0)
            solved.append(False)
            continue
        action, outcomes = cascade.oracle_action(query, candidates)
        labels.append(int(action))
        effort.append(int(sum(trace.function_evaluations for outcome in outcomes for trace in outcome.traces)))
        solved.append(action != EntryAction.REJECT)
    return RiskDataset(
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.int64),
        iterations=np.asarray(effort, dtype=np.int64),
        converged=np.asarray(solved, dtype=bool),
    )
