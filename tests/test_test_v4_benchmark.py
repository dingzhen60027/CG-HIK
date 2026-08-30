from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from confik.counterfactual_v4.policy import V4Decision
from confik.data.datasets import QueryDataset
from confik.latency_pilot_v3.benchmark import ProfiledOutcome
from confik.test_v4_locked.benchmark import benchmark_role


class _Kinematics:
    def __init__(self) -> None:
        self.limits = SimpleNamespace(velocity=np.asarray([100.0, 100.0]))

    def difference(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.asarray(q) - np.asarray(reference)


class _Method:
    def __init__(self, *, action: str = "easy", fev: int = 1) -> None:
        self.kinematics = _Kinematics()
        self.action = action
        self.fev = fev
        self.calls = 0
        self.last_decision = None

    def solve(self, query):
        self.calls += 1
        if self.action in {"defer", "reject"}:
            self.last_decision = V4Decision(
                action=self.action,
                reason="unit_test",
                ood_score=2.0 if self.action == "defer" else 0.0,
                is_ood=self.action == "defer",
                eligible_actions=(),
                predicted_success=(0.9, 0.8, 0.7),
                predicted_p50_ms=(1.0, 2.0, 3.0),
                predicted_p95_ms=(2.0, 3.0, 4.0),
                fail_all_probability=0.99 if self.action == "reject" else 0.1,
            )
        accepted = self.action != "reject"
        stages = () if self.action == "reject" else ("easy",)
        return ProfiledOutcome(
            q=query.previous_q.copy() if accepted else None,
            accepted=accepted,
            entry_action=self.action,
            executed_stages=stages,
            risk_probabilities=np.asarray([1.0, 0.0, 0.0, 0.0]),
            risk_score=0.0,
            function_evaluations=self.fev,
            iterations=self.fev,
            fallback_used=False,
            verification_reasons=(),
            reject_reason="" if accepted else "command_reject",
            candidate_count=1,
            timings_ns={
                "feature_preparation_ns": 1,
                "numpy_torch_conversion_ns": 1,
                "learned_seed_inference_ns": 1,
                "uncertainty_risk_inference_ns": 1,
                "routing_decision_ns": 1,
                "numerical_solver_ns": 1,
                "verification_ns": 1,
            },
        )


def _dataset(*, trajectory: bool = False) -> QueryDataset:
    count = 2
    return QueryDataset(
        previous_q=np.zeros((count, 2), dtype=np.float64),
        target_position=np.zeros((count, 3), dtype=np.float64),
        target_rotation=np.repeat(np.eye(3)[None], count, axis=0),
        reference_q=np.zeros((count, 2), dtype=np.float64),
        category=np.asarray(["trajectory_smooth" if trajectory else "id"] * count),
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.asarray([7, 7] if trajectory else [1, 2]),
        time_index=np.asarray([0, 1] if trajectory else [0, 0]),
    )


def test_point_benchmark_interleaves_and_retains_raw_repeats() -> None:
    methods = {
        "fixed_robust_cascade": _Method(),
        "proposed_v4": _Method(action="defer"),
    }
    records = benchmark_role(
        robot="panda",
        training_seed=17,
        role="id_points",
        methods=methods,
        dataset=_dataset(),
        repeats_by_method={"fixed_robust_cascade": 2, "proposed_v4": 3},
        dt=0.02,
        order_seed=9,
        synchronize_cuda=False,
    )
    assert len(records) == 4
    proposed = [row for row in records if row["method"] == "proposed_v4"]
    assert all(row["latency_repeat_count"] == 3 for row in proposed)
    assert all(len(row["latency_repeats_ns"]) == 3 for row in proposed)
    assert all(len(row["timing_repeats_ns"]) == 3 for row in proposed)
    assert all(len(row["method_order_indices_by_repeat"]) == 3 for row in proposed)
    assert all(row["role"] == "id_points" and row["domain"] == "id" for row in records)
    assert all(row["decision_action"] == "defer" for row in proposed)
    assert all(row["predicted_success"] == [0.9, 0.8, 0.7] for row in proposed)


def test_trajectory_benchmark_forces_one_closed_loop_call() -> None:
    method = _Method()
    records = benchmark_role(
        robot="ur5e",
        training_seed=29,
        role="ood_trajectories",
        methods={"fixed_robust_cascade": method},
        dataset=_dataset(trajectory=True),
        repeats_by_method={"fixed_robust_cascade": 7},
        dt=0.02,
        order_seed=10,
        synchronize_cuda=False,
    )
    assert method.calls == 2
    assert all(row["latency_repeat_count"] == 1 for row in records)
    assert all(row["is_trajectory"] and row["domain"] == "ood" for row in records)


def test_command_reject_contract_is_enforced_during_benchmark() -> None:
    with pytest.raises(RuntimeError, match="command_reject_nonzero_fev"):
        benchmark_role(
            robot="panda",
            training_seed=17,
            role="id_points",
            methods={"proposed_v4": _Method(action="reject", fev=1)},
            dataset=_dataset(),
            repeats_by_method={"proposed_v4": 1},
            dt=0.02,
            order_seed=11,
            synchronize_cuda=False,
        )


def test_zero_budget_reject_has_no_solver_stage() -> None:
    records = benchmark_role(
        robot="panda",
        training_seed=17,
        role="id_points",
        methods={"proposed_v4": _Method(action="reject", fev=0)},
        dataset=_dataset(),
        repeats_by_method={"proposed_v4": 2},
        dt=0.02,
        order_seed=12,
        synchronize_cuda=False,
    )
    assert all(row["function_evaluations"] == 0 for row in records)
    assert all(row["executed_stages"] == [] for row in records)
    assert all(row["command_q"] is None for row in records)
