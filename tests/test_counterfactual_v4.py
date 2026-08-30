from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from confik.counterfactual_v4.collector import (
    ACTIONS,
    COLLECTED_ACTIONS,
    select_pilot_indices,
    summarize_records,
    validate_source_role,
)
from confik.data.datasets import QueryDataset
from confik.counterfactual_v4 import runner as v4_runner


def _dataset(count: int = 20) -> QueryDataset:
    return QueryDataset(
        previous_q=np.zeros((count, 2)),
        target_position=np.zeros((count, 3)),
        target_rotation=np.repeat(np.eye(3)[None], count, axis=0),
        reference_q=np.zeros((count, 2)),
        category=np.asarray(["id", "near_limit"] * (count // 2)),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.arange(count),
        time_index=np.arange(count),
    )


def test_source_roles_exclude_every_test_role() -> None:
    for role in (
        "risk_train_queries",
        "risk_validation_queries",
        "calibration_queries",
        "policy_validation_queries",
    ):
        assert validate_source_role(role) == role
    for forbidden in ("risk_test_queries", "test_queries", "test_v3"):
        with pytest.raises(ValueError):
            validate_source_role(forbidden)


def test_pilot_selection_is_deterministic_unique_and_not_a_prefix() -> None:
    dataset = _dataset()
    first = select_pilot_indices(dataset, count=8, seed=123)
    second = select_pilot_indices(dataset, count=8, seed=123)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 8
    assert not np.array_equal(first, np.arange(8))
    assert not np.array_equal(first, np.sort(first))


def test_summary_selects_fastest_successful_p95_action() -> None:
    records = []
    for query in range(2):
        for action, success, latency in (
            ("easy", True, 5.0 + query),
            ("medium", True, 3.0 + query),
            ("hard", query == 0, 2.0 + query),
            ("fixed_robust", True, 5.1 + query),
        ):
            records.append(
                {
                    "query_index": query,
                    "entry_action": action,
                    "verified_success": success,
                    "verified_success_before_deadline": success,
                    "latency_p50_ns": latency * 1e6,
                    "latency_p95_ns": latency * 1e6,
                    "function_evaluations": 1,
                    "fallback_used": False,
                    "failure_reason": "" if success else "failed",
                }
            )
    summary = summarize_records(records)
    assert set(summary["actions"]) == set(COLLECTED_ACTIONS)
    assert summary["oracle_min_p95_action_counts"] == {"hard": 1, "medium": 1}
    assert summary["action_success_disagreement_rate"] == 0.5


def test_fixed_robust_is_collected_but_not_a_decision_head() -> None:
    assert ACTIONS == ("easy", "medium", "hard")
    assert COLLECTED_ACTIONS == ("easy", "medium", "hard", "fixed_robust")


def test_quiet_environment_fast_path_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v4_runner, "_busy_unrelated_processes", lambda **_: [])
    monkeypatch.setattr(
        v4_runner.time,
        "sleep",
        lambda _: pytest.fail("quiet fast path unexpectedly slept"),
    )
    result = v4_runner._wait_for_quiet_environment(
        {
            "runtime": {
                "max_unrelated_cpu_percent": 50.0,
                "quiet_stable_checks": 2,
                "quiet_poll_seconds": 1.0,
                "max_quiet_wait_seconds": 10.0,
            }
        },
        context="unit-test",
    )
    assert result["had_busy_process"] is False
