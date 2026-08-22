import json
from pathlib import Path

import numpy as np

from confik.data.generate_v2 import generate_mixed_query_set, generate_reference_trajectory_tests
from confik.data.datasets import RiskDataset
from confik.experiments.baselines_v2 import ThresholdGuardRiskProvider
from confik.experiments.policy_selection import tune_action_gate, tune_threshold_guard
from confik.experiments.provenance import ensure_protocol_manifest
from confik.experiments.statistics import paired_cluster_bootstrap_difference
from confik.kinematics.urdf import URDFKinematics
from confik.models.risk import ConstantRiskProvider
from confik.models.seed import PreviousStateCandidates
from confik.pipeline_v2 import aggregate_v2
from confik.runtime.cascade import (
    ActionGateConfig,
    CalibratedActionGate,
    CascadeConfig,
    CascadedHybridIK,
    EntryAction,
    FixedEntryGate,
)
from confik.solvers.dls import AdaptiveDLS
from confik.solvers.verifier import SolutionVerifier
from confik.types import CalibratedRisk, IKQuery


ASSET = Path(__file__).parent / "assets" / "toy_arm.urdf"


def test_action_gate_has_explicit_reject_and_three_solver_entries() -> None:
    gate = CalibratedActionGate(ActionGateConfig())
    assert gate.choose(CalibratedRisk(np.array([0.8, 0.1, 0.05, 0.05]))) == EntryAction.EASY
    assert gate.choose(CalibratedRisk(np.array([0.2, 0.6, 0.1, 0.1]))) == EntryAction.MEDIUM
    assert gate.choose(CalibratedRisk(np.array([0.1, 0.2, 0.6, 0.1]))) == EntryAction.HARD
    assert gate.choose(CalibratedRisk(np.array([0.02, 0.03, 0.05, 0.9]))) == EntryAction.REJECT


def test_reject_entry_runs_no_numerical_solver() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    q = np.array([0.1, -0.1])
    query = IKQuery(model.forward(q), q)
    method = CascadedHybridIK(
        model,
        PreviousStateCandidates(),
        ConstantRiskProvider(np.array([0.0, 0.0, 0.0, 1.0])),
        AdaptiveDLS(model),
        SolutionVerifier(model),
        gate=CalibratedActionGate(),
        config=CascadeConfig(),
    )
    result = method.solve(query)
    assert not result.accepted
    assert result.policy.level.value == "reject"
    assert result.traces == []
    assert result.metadata["executed_stages"] == []


def test_fixed_cascade_accepts_trivial_easy_query() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    q = np.array([0.2, -0.2])
    method = CascadedHybridIK(
        model,
        PreviousStateCandidates(),
        ConstantRiskProvider(np.array([1.0, 0.0, 0.0, 0.0])),
        AdaptiveDLS(model),
        SolutionVerifier(model),
        gate=FixedEntryGate(EntryAction.EASY),
        config=CascadeConfig(easy_iterations=1),
    )
    result = method.solve(IKQuery(model.forward(q), q))
    assert result.accepted
    assert result.metadata["executed_stages"] == ["easy"]


def test_reference_trajectories_are_known_and_velocity_feasible() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    dataset = generate_reference_trajectory_tests(model, paths_per_type=1, steps=12, seed=7)
    assert len(dataset) == 4 * 12
    for q, previous in zip(dataset.reference_q, dataset.previous_q, strict=True):
        assert np.all(np.abs(model.difference(q, previous)) <= model.limits.velocity * 0.02 + 1e-9)
    assert set(dataset.category) == {
        "trajectory_smooth",
        "trajectory_orientation",
        "trajectory_singular",
        "trajectory_limit",
    }


def test_mixed_v2_queries_include_rejectable_and_reachable_cases() -> None:
    model = URDFKinematics.from_file(ASSET, end_link="tool")
    dataset = generate_mixed_query_set(model, samples=60, seed=11)
    assert len(dataset) == 60
    assert np.any(dataset.expected_reachable)
    assert np.any(~dataset.expected_reachable)
    assert "large_step" in set(dataset.category)
    assert len(np.unique(dataset.trajectory_id)) == len(dataset)


def test_cluster_bootstrap_uses_cluster_as_independent_unit() -> None:
    baseline = np.zeros(6)
    proposed = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    clusters = np.array([10, 10, 10, 20, 20, 20])
    result = paired_cluster_bootstrap_difference(
        baseline,
        proposed,
        clusters,
        samples=200,
        seed=3,
    )
    assert result["cluster_count"] == 2.0
    assert abs(result["mean_difference"]) < 1e-12


def test_threshold_guard_is_fitted_only_to_nonreject_training_actions() -> None:
    features = np.zeros((5, 9), dtype=np.float64)
    features[:4, 7] = [0.01, 0.02, 0.03, 0.04]
    features[:4, 8] = [0.1, 0.2, 0.3, 0.4]
    features[4, 7:9] = [100.0, 100.0]
    dataset = RiskDataset(
        features=features,
        labels=np.array([0, 1, 2, 0, 3], dtype=np.int64),
        iterations=np.array([1, 1, 25, 1, 0], dtype=np.int64),
        converged=np.array([True, True, True, True, False]),
    )
    guard = ThresholdGuardRiskProvider.fit(dataset, quantile=0.75)
    assert guard.config.position_step_threshold < 1.0
    assert guard.config.orientation_step_threshold < 1.0
    assert int(np.argmax(guard.predict(np.array([0, 0, 0, 0, 0, 0, 0, 0.02, 0.2])).probabilities)) == 0
    assert int(np.argmax(guard.predict(np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 0.2])).probabilities)) == 3


def test_policy_selection_uses_locked_validation_constraints() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.08, 0.02],
            [0.1, 0.1, 0.75, 0.05],
            [0.01, 0.01, 0.01, 0.97],
        ]
    )

    class StubModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert len(features) == len(probabilities)
            return probabilities

    dataset = RiskDataset(
        features=np.zeros((4, 9)),
        labels=labels,
        iterations=np.array([1, 2, 10, 0]),
        converged=np.array([True, True, True, False]),
    )
    selected, report = tune_action_gate(
        StubModel(),  # type: ignore[arg-type]
        dataset,
        easy_grid=[0.7],
        hard_grid=[0.45],
        reject_grid=[0.85],
        max_false_reject_rate=0.01,
        min_reject_recall=0.95,
    )
    assert selected.reject_probability == 0.85
    assert report["constraints_satisfied"]
    assert report["validation_metrics"]["action_accuracy"] == 1.0


def test_threshold_policy_is_selected_under_same_false_reject_constraint() -> None:
    train_features = np.zeros((8, 9))
    train_features[:, 7] = np.linspace(0.01, 0.08, 8)
    train_features[:, 8] = np.linspace(0.02, 0.16, 8)
    train = RiskDataset(
        train_features,
        np.zeros(8, dtype=np.int64),
        np.ones(8, dtype=np.int64),
        np.ones(8, dtype=bool),
    )
    validation_features = np.zeros((6, 9))
    validation_features[:4, 7:9] = [0.02, 0.04]
    validation_features[4:, 7:9] = [1.0, 1.0]
    validation = RiskDataset(
        validation_features,
        np.array([0, 1, 2, 0, 3, 3]),
        np.array([1, 2, 20, 1, 0, 0]),
        np.array([True, True, True, True, False, False]),
    )
    guard, report = tune_threshold_guard(
        train,
        validation,
        quantiles=[0.99, 1.0],
        max_false_reject_rate=0.01,
        min_reject_recall=0.95,
    )
    assert report["constraints_satisfied"]
    assert report["validation_metrics"]["false_reject_rate"] == 0.0
    assert report["validation_metrics"]["reject_recall"] == 1.0
    assert guard.config.quantile == 0.99


def test_protocol_manifest_refuses_stale_artifact_reuse(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    config = {"protocol_version": 2, "experiment_name": "locked", "seed": 17}
    first = ensure_protocol_manifest(path, config, "toy")
    second = ensure_protocol_manifest(path, config, "toy")
    assert first["run_fingerprint"] == second["run_fingerprint"]
    changed = {**config, "seed": 29}
    try:
        ensure_protocol_manifest(path, changed, "toy")
    except RuntimeError as error:
        assert "new experiment_name" in str(error)
    else:  # pragma: no cover
        raise AssertionError("stale manifest reuse was not rejected")


def test_paper_aggregation_requires_complete_consistent_locked_runs(tmp_path: Path) -> None:
    config_path = tmp_path / "paper.yaml"
    config_path.write_text(
        "protocol_version: 2\nexperiment_name: confirm\noutput_root: outputs\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "outputs" / "confirm_seed17" / "toy" / "results"
    result_dir.mkdir(parents=True)
    claim = {
        "pilot_gate_pass": True,
        "point_feasible_success_gap": 0.0,
        "point_feasible_evaluation_reduction": 0.15,
        "point_feasible_p95_latency_reduction": -0.05,
        "point_rejectable_evaluation_reduction": 0.9,
        "point_rejectable_p95_latency_reduction": 0.8,
        "trajectory_completion_gap": 0.0,
        "threshold_guard_point_evaluation_reduction": 0.08,
    }
    (result_dir / "claim_gate_v2.json").write_text(json.dumps(claim), encoding="utf-8")
    aggregate = aggregate_v2(config_path, ["toy"], [17])
    assert aggregate["paper_gate_pass"]
    incomplete = aggregate_v2(config_path, ["toy"], [17, 29])
    assert not incomplete["paper_gate_pass"]
    assert len(incomplete["missing"]) == 1
