from pathlib import Path

import numpy as np

from confik.kinematics.urdf import URDFKinematics
from confik.solvers.dls import AdaptiveDLS, DLSConfig
from confik.solvers.fallback import KDTreeSeedBank, TRFFallbackSolver
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.types import IKQuery

ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_dls_reaches_a_nearby_reachable_pose() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    target_q = np.array([0.025, -0.015])
    target = model.forward(target_q)
    solver = AdaptiveDLS(model, DLSConfig(sigma_threshold=0.1))
    trace = solver.solve(target, np.zeros(2), 40, seed_source="test")
    assert trace.converged
    assert trace.position_error <= 1e-3
    assert trace.orientation_error <= np.deg2rad(0.5)
    verifier = SolutionVerifier(model)
    verified = verifier.check(trace.q, IKQuery(target, np.zeros(2), dt=0.02))
    assert verified.accepted


def test_kdtree_and_trf_fallback() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    samples = np.array([[a, b] for a in np.linspace(-1, 1, 8) for b in np.linspace(-1, 1, 8)])
    bank = KDTreeSeedBank(model).fit(samples)
    target_q = np.array([0.37, -0.42])
    seeds = bank.query(model.forward(target_q), np.zeros(2), k=3)
    assert seeds.shape == (3, 2)
    trace = TRFFallbackSolver(model).solve(model.forward(target_q), seeds[0])
    assert trace.converged


def test_verifier_can_disable_velocity_for_point_queries() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    target_q = np.array([0.5, -0.3])
    query = IKQuery(model.forward(target_q), np.zeros(2))
    strict = SolutionVerifier(model).check(target_q, query)
    assert not strict.velocity_ok
    point = SolutionVerifier(model, VerifierConfig(enforce_velocity=False)).check(target_q, query)
    assert point.accepted

