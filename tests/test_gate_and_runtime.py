from pathlib import Path

import numpy as np

from confik.kinematics.urdf import URDFKinematics
from confik.models.risk import ConstantRiskProvider
from confik.models.seed import PreviousStateCandidates
from confik.runtime.gate import ConfidenceGate
from confik.runtime.hybrid import HybridIK
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.types import CalibratedRisk, IKQuery, RiskLevel

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_gate_thresholds() -> None:
    gate = ConfidenceGate()
    assert gate.make_policy(CalibratedRisk(np.array([0.9, 0.05, 0.03, 0.02]))).level == RiskLevel.EASY
    assert gate.make_policy(CalibratedRisk(np.array([0.4, 0.4, 0.15, 0.05]))).level == RiskLevel.MEDIUM
    assert gate.make_policy(CalibratedRisk(np.array([0.1, 0.1, 0.4, 0.4]))).level == RiskLevel.HARD


def test_hybrid_runtime_accepts_verified_solution() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    runtime = HybridIK(
        model,
        PreviousStateCandidates(),
        ConstantRiskProvider(),
        AdaptiveDLS(model),
        SolutionVerifier(model),
    )
    target_q = np.array([0.015, -0.01])
    result = runtime.solve(IKQuery(model.forward(target_q), np.zeros(2), dt=0.02))
    assert result.accepted
    assert result.q is not None
    assert result.verification is not None and result.verification.accepted
    assert result.metadata["total_iterations"] <= 8

