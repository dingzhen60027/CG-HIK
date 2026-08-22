from pathlib import Path

import numpy as np

from confik.data.datasets import QueryDataset
from confik.experiments.evaluate import evaluate_methods, summarize_records
from confik.kinematics.urdf import URDFKinematics
from confik.types import (
    CalibratedRisk,
    IKResult,
    RiskLevel,
    SolveTrace,
    SolverPolicy,
    VerificationResult,
)

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


class EchoMethod:
    def __init__(self, kinematics: URDFKinematics):
        self.kinematics = kinematics
        self.seen_previous: list[np.ndarray] = []

    def solve(self, query):
        self.seen_previous.append(query.previous_q.copy())
        q = query.previous_q.copy()
        trace = SolveTrace(q, True, 0, 0.0, 0.0)
        verification = VerificationResult(True, 0.0, 0.0, True, True, True)
        return IKResult(
            q=q,
            accepted=True,
            risk=CalibratedRisk(np.array([1.0, 0.0, 0.0, 0.0])),
            policy=SolverPolicy(RiskLevel.EASY, 1, 1),
            verification=verification,
            traces=[trace],
            fallback_used=False,
        )


def test_trajectory_evaluation_uses_method_closed_loop_state() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    q0 = np.array([0.1, -0.1])
    q1 = np.array([1.0, 1.0])
    pose0 = model.forward(q0)
    pose1 = model.forward(q1)
    dataset = QueryDataset(
        previous_q=np.stack([q0, q1]),
        target_position=np.stack([pose0.position, pose1.position]),
        target_rotation=np.stack([pose0.rotation, pose1.rotation]),
        reference_q=np.stack([q0, q1]),
        category=np.array(["trajectory_line", "trajectory_line"]),
        expected_reachable=np.ones(2, dtype=bool),
        continuity_feasible=np.ones(2, dtype=bool),
        trajectory_id=np.array([7, 7]),
        time_index=np.array([0, 1]),
    )
    method = EchoMethod(model)
    records = evaluate_methods({"echo": method}, dataset, timing_repeats=2)
    np.testing.assert_allclose(method.seen_previous[0], q0)
    np.testing.assert_allclose(method.seen_previous[2], q0)
    assert all(record["closed_loop"] for record in records)
    summary = summarize_records(records)
    assert summary["echo"]["trajectory_line"]["trajectory_completion_rate"] == 1.0
