from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from confik.counterfactual_v4.model import CounterfactualPrediction
from confik.counterfactual_v4.training_runner import (
    COLLECTED_ACTIONS,
    DECISION_ENTRIES,
    RoleArrays,
    _load_role,
    _select_policy_configuration,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_synthetic_role(root: Path, *, count: int = 8, repeats: int = 5) -> Path:
    role_root = root / "panda" / "seed17" / "risk_train_queries"
    chunk = role_root / "chunks" / f"chunk_000000_{count - 1:06d}"
    chunk.mkdir(parents=True)
    query_hash = np.asarray([f"{index:064x}" for index in range(count)], dtype="U64")
    with (role_root / "selection.npz").open("wb") as handle:
        np.savez_compressed(handle, query_sha256=query_hash)
    (role_root / "selection_manifest.json").write_text(
        json.dumps(
            {
                "source_role": "risk_train_queries",
                "selected_query_count": count,
                "test_named_dataset_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    success = np.ones((count, 4), dtype=bool)
    success[-2:] = False
    deadline = success.copy()
    latency = np.full((count, 4, repeats), 2_000_000, dtype=np.int64)
    latency[:, 1] += 100_000
    latency[:, 2] += 200_000
    latency[:, 3] = latency[:, 0]
    fev = np.tile(np.asarray([2, 3, 4, 2]), (count, 1))
    fallback = np.zeros((count, 4), dtype=bool)
    labels_path = chunk / "counterfactual_labels.npz"
    with labels_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            feature_names=np.asarray([f"f{index}" for index in range(9)]),
            action_names=np.asarray(COLLECTED_ACTIONS),
            decision_action_names=np.asarray(DECISION_ENTRIES),
            features=np.arange(count * 9, dtype=np.float64).reshape(count, 9),
            query_sha256=query_hash,
            category=np.asarray(["id"] * count),
            expected_reachable=np.ones(count, dtype=bool),
            continuity_feasible=np.ones(count, dtype=bool),
            verified_success=success,
            verified_success_before_deadline=deadline,
            latency_samples_ns=latency,
            function_evaluations=fev,
            fallback_used=fallback,
        )
    manifest = {
        "robot": "panda",
        "training_seed": 17,
        "source_role": "risk_train_queries",
        "query_start": 0,
        "query_stop_exclusive": count,
        "environment_contaminated": False,
        "test_data_loaded": False,
        "artifacts": {
            labels_path.name: {
                "sha256": _hash(labels_path),
                "size": labels_path.stat().st_size,
            }
        },
    }
    (chunk / "chunk_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return role_root


def test_role_loader_requires_and_preserves_raw_five_repeat_ns(tmp_path: Path) -> None:
    _write_synthetic_role(tmp_path)
    role = _load_role(tmp_path, "panda", 17, "risk_train_queries")
    assert role.latency_samples_ns.shape == (8, 4, 5)
    assert role.latency_samples_ns.dtype == np.int64
    assert role.decision_latency_samples_ms.shape == (8, 3, 5)
    assert role.fail_all.tolist() == [0.0] * 6 + [1.0, 1.0]
    assert np.array_equal(role.latency_samples_ns[:, 0], role.latency_samples_ns[:, 3])


def test_role_loader_rejects_aggregated_or_wrong_repeat_latency(tmp_path: Path) -> None:
    role_root = _write_synthetic_role(tmp_path, repeats=4)
    labels = next((role_root / "chunks").glob("chunk_*/counterfactual_labels.npz"))
    manifest_path = labels.parent / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][labels.name] = {
        "sha256": _hash(labels),
        "size": labels.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="raw latency_samples_ns"):
        _load_role(tmp_path, "panda", 17, "risk_train_queries")


def _policy_role() -> tuple[RoleArrays, CounterfactualPrediction]:
    count = 12
    success = np.ones((count, 4), dtype=bool)
    success[-3:] = False
    deadline = success.copy()
    latency = np.empty((count, 4, 5), dtype=np.int64)
    latency[:, 0] = 3_000_000
    latency[:, 1] = 2_000_000
    latency[:, 2] = 1_000_000
    latency[:, 3] = latency[:, 0]
    role = RoleArrays(
        features=np.zeros((count, 9), dtype=np.float32),
        query_sha256=np.asarray([f"{index:064x}" for index in range(count)]),
        category=np.asarray(["id"] * count),
        expected_reachable=np.asarray([True] * 9 + [False] * 3),
        continuity_feasible=np.ones(count, dtype=bool),
        verified_success=success,
        verified_success_before_deadline=deadline,
        latency_samples_ns=latency,
        function_evaluations=np.tile(np.asarray([8, 6, 4, 8]), (count, 1)),
        fallback_used=np.zeros((count, 4), dtype=bool),
        source_files=(),
    )
    probabilities = np.tile(np.asarray([0.98, 0.98, 0.98]), (count, 1))
    probabilities[-3:] = 0.05
    prediction = CounterfactualPrediction(
        deadline_success_logits=np.zeros((count, 3)),
        deadline_success_probability=probabilities,
        latency_p50_ms=np.tile(np.asarray([2.5, 1.5, 0.8]), (count, 1)),
        latency_p95_ms=np.tile(np.asarray([3.0, 2.0, 1.0]), (count, 1)),
        fail_all_logit=np.zeros(count),
        fail_all_probability=np.asarray([0.05] * 9 + [0.99] * 3),
        embedding=np.zeros((count, 4)),
        ood_score=np.zeros(count),
        is_ood=np.zeros(count, dtype=bool),
    )
    return role, prediction


def test_policy_selection_uses_policy_role_and_enforces_hard_gates() -> None:
    role, prediction = _policy_role()
    selected, candidates = _select_policy_configuration(
        role,
        prediction,
        minimum_success_probabilities=[0.9],
        reject_probabilities=[0.9],
        latency_tie_margins_ms=[0.0, 3.0],
        deadline_ms=20.0,
    )
    assert len(candidates) == 2
    assert selected["hard_gate_pass"]
    assert selected["route_counts"]["reject"] == 3
    assert selected["fixed_success_false_reject_rate"] == 0.0
    assert selected["operational_feasible_false_reject_rate"] == 0.0
    # Zero margin permits the genuinely faster hard entry.
    assert selected["config"]["latency_tie_margin_ms"] == 0.0
    assert selected["route_counts"]["hard"] == 9
