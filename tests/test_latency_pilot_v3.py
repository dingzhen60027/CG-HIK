from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from confik.kinematics.urdf import URDFKinematics
from confik.latency_pilot_v3.benchmark import (
    CORE_STAGE_KEYS,
    ConstantRiskEngine,
    ProfiledCascadeRuntime,
    STAGE_KEYS,
    compare_backends,
    distribution_summary,
    paired_latency_summary,
)
from confik.latency_pilot_v3.optimized_inference import (
    ExactSingleCallSeedEnsemble,
    OptimizedSeedEngine,
    VectorizedHGBRiskModel,
    VectorizedSeedMLP,
)
from confik.latency_pilot_v3.runner import _assert_allowed
from confik.latency_pilot_v3.validation import record_equivalence
from confik.models.risk import RiskModel
from confik.models.seed import SeedTrainingConfig, TorchSeedEnsemble
from confik.runtime.cascade import CascadeConfig, EntryAction, FixedEntryGate
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.types import IKQuery


def _toy_ensemble() -> TorchSeedEnsemble:
    model = URDFKinematics.from_file(Path(__file__).parent / "assets" / "toy_arm.urdf")
    ensemble = TorchSeedEnsemble(
        model,
        SeedTrainingConfig(members=3, hidden_sizes=(8, 8), epochs=1, seed=7),
        device="cpu",
    )
    ensemble.members.eval()
    ensemble.fitted = True
    return ensemble


def test_vectorized_seed_forward_matches_member_eager_and_torch_export() -> None:
    ensemble = _toy_ensemble()
    inputs = torch.randn(11, ensemble.kinematics.nq + 9, dtype=torch.float32)
    vectorized = VectorizedSeedMLP.from_ensemble(ensemble, device="cpu")
    exported = torch.export.export(vectorized, (inputs[:1],), strict=True).module()
    with torch.inference_mode():
        reference = torch.stack([member(inputs) for member in ensemble.members], dim=1)
        actual = vectorized(inputs)
        exported_actual = torch.cat([exported(row) for row in inputs.split(1)], dim=0)
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(exported_actual, reference, rtol=0.0, atol=1e-7)


def test_exact_single_call_trace_matches_member_outputs_bitwise() -> None:
    ensemble = _toy_ensemble()
    inputs = torch.randn(11, ensemble.kinematics.nq + 9, dtype=torch.float32)
    exact = ExactSingleCallSeedEnsemble(ensemble).eval()
    traced = torch.jit.trace(exact, inputs[:1], strict=True).eval()
    with torch.inference_mode():
        reference = torch.stack([member(inputs) for member in ensemble.members], dim=1)
        actual = traced(inputs)
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_vectorized_hgb_retains_probabilities_and_actions() -> None:
    rng = np.random.default_rng(19)
    train_x = rng.normal(size=(500, 9))
    train_y = np.arange(500, dtype=np.int64) % 4
    calibration_x = rng.normal(size=(300, 9))
    calibration_y = np.arange(300, dtype=np.int64) % 4
    validation_x = rng.normal(size=(200, 9))
    model = RiskModel("gradient_boosting", seed=11).fit(train_x, train_y)
    model.calibrate(calibration_x, calibration_y)
    optimized = VectorizedHGBRiskModel(model)
    reference = model.predict_proba(validation_x)
    actual = optimized.predict_proba(validation_x)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(reference, axis=1))


def test_profiled_runtime_has_all_required_nonnegative_stages() -> None:
    ensemble = _toy_ensemble()
    model = ensemble.kinematics
    vectorized = VectorizedSeedMLP.from_ensemble(ensemble, device="cpu")
    seed_engine = OptimizedSeedEngine(ensemble, vectorized)
    runtime = ProfiledCascadeRuntime(
        name="baseline",
        kinematics=model,
        seed_engine=seed_engine,
        risk_engine=ConstantRiskEngine(),
        gate=FixedEntryGate(EntryAction.EASY),
        dls=AdaptiveDLS(model),
        verifier=SolutionVerifier(model),
        seed_bank=None,
        fallback=None,
        cascade_config=CascadeConfig(easy_iterations=1),
        reuse_candidate_features=True,
    )
    q = np.zeros(model.nq, dtype=np.float64)
    outcome = runtime.solve(IKQuery(model.forward(q), q, dt=0.02))
    assert outcome.accepted
    assert set(STAGE_KEYS).issubset(outcome.timings_ns)
    assert all(outcome.timings_ns[key] >= 0 for key in STAGE_KEYS)
    assert outcome.timings_ns["total_end_to_end_ns"] >= sum(
        outcome.timings_ns[key] for key in CORE_STAGE_KEYS
    )


def test_distribution_summary_contract() -> None:
    summary = distribution_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == 2.5
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)
    assert summary["mean"] == 2.5
    assert summary["max"] == 4.0


def test_validation_allowlist_rejects_test_named_input(tmp_path: Path) -> None:
    source = tmp_path / "paper_v2_seed17"
    source.mkdir()
    forbidden = source / "test_queries.npz"
    forbidden.touch()
    with pytest.raises(RuntimeError, match="test-named"):
        _assert_allowed(forbidden, source, "forbidden")


def _minimal_record(*, backend: str, method: str, digest: str, closed_loop: bool) -> dict[str, object]:
    return {
        "robot": "toy",
        "backend": backend,
        "method": method,
        "split": "trajectory" if closed_loop else "points",
        "query_index": 0,
        "query_sha256": digest,
        "closed_loop": closed_loop,
        "expected_reachable": True,
        "continuity_feasible": True,
        "accepted": True,
        "entry_action": "easy",
        "function_evaluations": 1,
        "command_q": [0.0],
        "timings_ns": {"total_end_to_end_ns": 1_000_000},
    }


def test_paired_latency_rejects_different_query_hashes() -> None:
    records = [
        _minimal_record(backend="candidate", method="baseline", digest="a", closed_loop=False),
        _minimal_record(backend="candidate", method="proposed", digest="b", closed_loop=False),
    ]
    with pytest.raises(RuntimeError, match="query hashes"):
        paired_latency_summary(records)


def test_backend_comparison_rejects_different_query_hashes() -> None:
    records = [
        _minimal_record(backend="reference", method="proposed", digest="a", closed_loop=False),
        _minimal_record(backend="candidate", method="proposed", digest="b", closed_loop=False),
    ]
    with pytest.raises(RuntimeError, match="query hash differs"):
        compare_backends(
            records,
            reference_backend="reference",
            candidate_backend="candidate",
        )


def test_record_equivalence_checks_point_and_trajectory_routes() -> None:
    records: list[dict[str, object]] = []
    for closed_loop in (False, True):
        for backend in ("reference", "candidate"):
            record = _minimal_record(
                backend=backend,
                method="proposed",
                digest="same",
                closed_loop=closed_loop,
            )
            if closed_loop and backend == "candidate":
                record["entry_action"] = "hard"
            records.append(record)
    result = record_equivalence(
        records,
        robot="toy",
        reference_backend="reference",
        candidate_backend="candidate",
    )
    assert result["point_route_action_agreement"] == 1.0
    assert result["trajectory_route_action_agreement"] == 0.0
    assert result["all_route_action_agreement"] == 0.5
