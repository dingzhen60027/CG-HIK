from __future__ import annotations

from pathlib import Path

import numpy as np

from confik.data.datasets import QueryDataset, TransitionDataset
from confik.test_v3_locked.data import derive_seed, query_rows, split_audit


def _query(offset: float = 0.0) -> QueryDataset:
    previous = np.asarray([[offset, offset + 0.1]], dtype=np.float64)
    position = np.asarray([[offset + 0.2, 0.0, 0.0]], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)[None, ...]
    return QueryDataset(
        previous_q=previous,
        target_position=position,
        target_rotation=rotation,
        reference_q=previous.copy(),
        category=np.asarray(["id"]),
        expected_reachable=np.asarray([True]),
        continuity_feasible=np.asarray([True]),
        trajectory_id=np.asarray([0]),
        time_index=np.asarray([0]),
    )


def test_generation_seed_is_stable_and_role_separated() -> None:
    commit = "6e656a9af55e260973121d6c85a06283e4822b81"
    assert derive_seed(commit, "panda", "point_stress") == derive_seed(
        commit, "panda", "point_stress"
    )
    assert derive_seed(commit, "panda", "point_stress") != derive_seed(
        commit, "panda", "hard_valid"
    )
    assert 0 <= derive_seed(commit, "ur5e", "trajectories") <= 2**32 - 1


def test_exact_query_rows_detect_duplicates() -> None:
    first = _query(0.0)
    duplicate = QueryDataset.concatenate([first, first])
    distinct = QueryDataset.concatenate([first, _query(1.0)])
    assert len(query_rows(duplicate)) == 1
    assert len(query_rows(distinct)) == 2


def test_split_audit_accepts_disjoint_sources(tmp_path: Path) -> None:
    source_query = tmp_path / "source_query.npz"
    _query(2.0).save(source_query)
    transition = TransitionDataset(
        previous_q=np.asarray([[3.0, 3.1]]),
        target_q=np.asarray([[3.2, 3.3]]),
        target_position=np.asarray([[3.4, 0.0, 0.0]]),
        target_rotation=np.eye(3)[None, ...],
        trajectory_id=np.asarray([0]),
    )
    source_transition = tmp_path / "source_transition.npz"
    transition.save(source_transition)
    audit = split_audit(
        _query(0.0),
        comparison_files=[
            ("query", source_query, "query"),
            ("transition", source_transition, "transition"),
        ],
    )
    assert audit["passed"]
    assert audit["within_test_v3_exact_duplicate_count"] == 0
    assert audit["exact_overlap_counts"] == {"query": 0, "transition": 0}
