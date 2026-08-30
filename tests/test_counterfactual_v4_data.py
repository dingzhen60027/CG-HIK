from pathlib import Path

import numpy as np

from confik.counterfactual_v4.data_v4 import (
    OOD_POINT_FAMILIES,
    OOD_TRAJECTORY_FAMILIES,
    derive_v4_seed,
    exact_query_rows,
    generate_prespecified_ood_points,
    generate_prespecified_ood_trajectories,
)
from confik.kinematics.urdf import URDFKinematics


ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_v4_seed_derivation_is_deterministic_and_role_separated() -> None:
    first = derive_v4_seed("abc", "panda", "ood_points")
    assert first == derive_v4_seed("abc", "panda", "ood_points")
    assert first != derive_v4_seed("abc", "panda", "ood_trajectories")


def test_prespecified_ood_points_are_known_feasible_and_unique() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    first = generate_prespecified_ood_points(model, per_family=2, seed=19)
    second = generate_prespecified_ood_points(model, per_family=2, seed=19)
    assert len(first) == 2 * len(OOD_POINT_FAMILIES)
    assert set(first.category) == set(OOD_POINT_FAMILIES)
    assert np.all(first.expected_reachable & first.continuity_feasible)
    np.testing.assert_allclose(first.previous_q, second.previous_q)
    assert len(exact_query_rows([first])) == len(first)
    for q, previous in zip(first.reference_q, first.previous_q, strict=True):
        assert np.all(
            np.abs(model.difference(q, previous))
            <= model.limits.velocity * 0.02 + 1e-12
        )


def test_prespecified_ood_trajectories_obey_velocity_contract() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    dataset = generate_prespecified_ood_trajectories(
        model, paths_per_family=1, steps=16, seed=23
    )
    assert len(dataset) == len(OOD_TRAJECTORY_FAMILIES) * 16
    assert set(dataset.category) == set(OOD_TRAJECTORY_FAMILIES)
    assert len(np.unique(dataset.trajectory_id)) == len(OOD_TRAJECTORY_FAMILIES)
    for q, previous in zip(dataset.reference_q, dataset.previous_q, strict=True):
        assert np.all(
            np.abs(model.difference(q, previous))
            <= model.limits.velocity * 0.02 + 1e-9
        )
