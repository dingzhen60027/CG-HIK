from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from confik.counterfactual_v4.model import (
    LABEL_CONTRACT,
    CounterfactualTrainingConfig,
    CounterfactualV4Predictor,
)
from confik.counterfactual_v4.policy import V4Decision, V4PolicyConfig
from confik.release_v4_locked.artifacts import (
    EagerV4Inference,
    FrozenV4Policy,
    TorchScriptV4Inference,
    V4InferenceOutput,
    decision_record,
    export_exact_v4_predictor,
    load_exact_v4_predictor,
    load_policy_config,
)
from confik.release_v4_locked.runner import (
    _formal_source_manifest,
    _module_batch,
    _numerical_equivalence,
    _runtime_equivalence,
)


@pytest.fixture()
def frozen_candidate(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(4107)
    features = rng.normal(size=(96, 9)).astype(np.float32)
    semantic_success = (features[:, 0] + 0.3 * features[:, 1] > 0.0).astype(
        np.float32
    )
    success = np.repeat(semantic_success[:, None], 3, axis=1)
    fail_all = 1.0 - semantic_success
    base = 1.0 + np.abs(features[:, :3])
    latency = np.stack(
        [base + 0.01 * rng.normal(size=base.shape) for _ in range(5)], axis=2
    ).astype(np.float32)
    predictor = CounterfactualV4Predictor(
        CounterfactualTrainingConfig(
            hidden_sizes=(8, 8),
            epochs=2,
            batch_size=32,
            learning_rate=0.002,
            seed=4108,
        )
    )
    predictor.fit(features[:64], success[:64], latency[:64], fail_all[:64])
    predictor.calibrate(features[64:80], success[64:80], fail_all[64:80])
    predictor.fit_ood_detector(
        features[:64], features[64:80], target_id_coverage=0.99
    )
    candidate = tmp_path / "candidate.pt"
    predictor.save(candidate)
    return candidate, features[80:]


def test_exact_torchscript_matches_disk_loaded_candidate(
    tmp_path: Path, frozen_candidate: tuple[Path, np.ndarray]
) -> None:
    candidate, features = frozen_candidate
    artifact = tmp_path / "exact.ts"
    metadata = export_exact_v4_predictor(candidate, artifact)
    module = load_exact_v4_predictor(artifact)
    result = _numerical_equivalence(
        candidate_path=candidate,
        module=module,
        features=features,
        policy_config=V4PolicyConfig(
            minimum_success_probability=0.6,
            reject_probability=0.6,
            deadline_ms=20.0,
            latency_tie_margin_ms=0.15,
        ),
        tolerance={
            "probability_max_abs": 1e-12,
            "latency_max_abs_ms": 1e-6,
            "embedding_max_abs": 1e-6,
            "ood_score_max_abs": 1e-8,
        },
    )
    assert metadata["label_contract"] == LABEL_CONTRACT
    assert metadata["torchscript_load_only"] is True
    assert result["pass"] is True
    assert result["route_action_agreement"] == 1.0
    assert result["ood_decision_agreement"] == 1.0
    assert set(result["exact_route_counts"]) == {
        "easy",
        "medium",
        "hard",
        "reject",
        "defer",
    }


def test_loaded_deployment_does_not_reopen_candidate(
    tmp_path: Path, frozen_candidate: tuple[Path, np.ndarray]
) -> None:
    candidate, features = frozen_candidate
    artifact = tmp_path / "exact.ts"
    export_exact_v4_predictor(candidate, artifact)
    with patch.object(
        CounterfactualV4Predictor,
        "load",
        side_effect=AssertionError("deployment must not load the candidate"),
    ):
        backend = TorchScriptV4Inference(load_exact_v4_predictor(artifact))
        output = backend.infer(features[0])
    assert output.success_probabilities.shape == (3,)
    assert np.all(output.latency_p95_ms >= output.latency_p50_ms)


def test_frozen_policy_keeps_ood_reject_and_defer_distinct() -> None:
    class Backend:
        def __init__(self, output: V4InferenceOutput):
            self.output = output

        def infer(self, features: np.ndarray) -> V4InferenceOutput:
            assert features.shape == (9,)
            return self.output

    config = V4PolicyConfig(
        minimum_success_probability=0.9,
        reject_probability=0.7,
        deadline_ms=20.0,
        latency_tie_margin_ms=0.15,
    )

    def output(*, success: float, fail: float, ood: bool) -> V4InferenceOutput:
        return V4InferenceOutput(
            np.repeat(success, 3),
            np.array([1.0, 1.1, 1.2]),
            np.array([2.0, 2.1, 2.2]),
            fail,
            np.zeros(8),
            100.0 if ood else 0.0,
            ood,
        )

    ood = FrozenV4Policy(Backend(output(success=0.0, fail=1.0, ood=True)), config)
    reject = FrozenV4Policy(
        Backend(output(success=0.0, fail=1.0, ood=False)), config
    )
    uncertain = FrozenV4Policy(
        Backend(output(success=0.0, fail=0.2, ood=False)), config
    )
    assert ood.decide(np.zeros(9)).action == "defer"
    assert reject.decide(np.zeros(9)).action == "reject"
    assert uncertain.decide(np.zeros(9)).reason == "uncertain_no_eligible_action"


def test_raw_decision_record_is_not_probability_normalized() -> None:
    decision = V4Decision(
        action="hard",
        reason="minimum_predicted_p95",
        ood_score=0.25,
        is_ood=False,
        eligible_actions=("easy", "medium", "hard"),
        predicted_success=(0.95, 0.95, 0.95),
        predicted_p50_ms=(1.0, 1.1, 1.2),
        predicted_p95_ms=(2.0, 2.1, 1.5),
        fail_all_probability=0.03,
    )
    record = decision_record(decision)
    assert record["predicted_success"] == [0.95, 0.95, 0.95]
    assert sum(record["predicted_success"]) + record["fail_all_probability"] > 1.0
    assert "risk_probabilities" not in record


def test_policy_artifact_requires_validation_only_contract(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "label_contract": LABEL_CONTRACT,
                "policy_config": {
                    "minimum_success_probability": 0.95,
                    "reject_probability": 0.7,
                    "deadline_ms": 20.0,
                    "latency_tie_margin_ms": 0.15,
                },
                "test_data_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    config, _ = load_policy_config(path)
    assert config.reject_probability == 0.7
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_data_loaded"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="test access"):
        load_policy_config(path)


def test_runtime_equivalence_checks_reject_and_defer_contracts() -> None:
    reject = V4Decision(
        "reject",
        "high_confidence_fail_all",
        0.0,
        False,
        (),
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        0.9,
    )
    defer = V4Decision(
        "defer",
        "ood_defer",
        10.0,
        True,
        (),
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        0.9,
    )
    rows = [
        {
            "query_sha256": "a",
            "accepted": False,
            "command_q": None,
            "function_evaluations": 0,
            "iterations": 0,
            "fallback_used": False,
            "verification_reasons": (),
            "executed_stages": (),
            "reject_reason": "command_reject_high_confidence_fail_all",
            "decision": reject,
        },
        {
            "query_sha256": "b",
            "accepted": True,
            "command_q": np.zeros(2),
            "function_evaluations": 1,
            "iterations": 1,
            "fallback_used": False,
            "verification_reasons": (),
            "executed_stages": ("easy",),
            "reject_reason": "",
            "decision": defer,
        },
    ]
    result = _runtime_equivalence(
        rows,
        rows,
        tolerance={
            "probability_max_abs": 1e-12,
            "latency_max_abs_ms": 1e-6,
            "ood_score_max_abs": 1e-8,
            "accepted_command_max_abs_rad": 1e-6,
        },
    )
    assert result["pass"] is True
    assert result["command_reject_zero_solver_rate"] == 1.0
    assert result["defer_enters_fixed_easy_stage_rate"] == 1.0


def test_formal_source_manifest_allows_only_out_of_scope_user_changes(
    tmp_path: Path,
) -> None:
    def fake_git(workspace: Path, *arguments: str) -> str:
        assert workspace == tmp_path
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "HEAD"):
            return "commit-sha"
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "tree-sha"
        if arguments[:3] == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return "" if "--" in arguments else " M docs/HANDOVER.md"
        raise AssertionError(arguments)

    with patch("confik.release_v4_locked.runner._git", side_effect=fake_git):
        manifest = _formal_source_manifest(tmp_path)
    assert manifest["release_source_scope_clean"] is True
    assert manifest["git_worktree_clean"] is False
    assert manifest["out_of_scope_changes_present"] is True
    assert manifest["out_of_scope_change_count"] == 1
