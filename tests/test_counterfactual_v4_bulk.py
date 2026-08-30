from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from confik.counterfactual_v4 import bulk_runner
from confik.data.datasets import QueryDataset


def _config() -> dict:
    return {
        "protocol_version": 4,
        "robots": ["panda", "ur5e"],
        "training_seeds": [17],
        "data": {
            "role_counts": {
                "risk_train_queries": 15_000,
                "calibration_queries": 2_500,
                "policy_validation_queries": 2_500,
            },
            "dt": 0.02,
        },
        "timing": {"repeats": 5, "deadline_ms": 20.0},
        "bulk": {"chunk_size": 250},
        "runtime": {"environment_check_every_queries": 1},
    }


def test_bulk_config_is_exact_and_has_no_test_role() -> None:
    config = _config()
    bulk_runner._validate_config(config)
    config["data"]["role_counts"]["test_v3"] = 1
    with pytest.raises(ValueError):
        bulk_runner._validate_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    (("repeats", 4), ("deadline_ms", 25.0)),
)
def test_bulk_timing_contract_is_frozen(field: str, value: float) -> None:
    config = _config()
    config["timing"][field] = value
    with pytest.raises(ValueError):
        bulk_runner._validate_config(config)


def test_per_query_environment_check_is_mandatory() -> None:
    config = _config()
    config["runtime"]["environment_check_every_queries"] = 10
    with pytest.raises(ValueError, match="environment_check_every_queries"):
        bulk_runner._validate_config(config)


def test_fixed_robust_is_an_unexecuted_deep_alias() -> None:
    easy = {
        "entry_action": "easy",
        "measurement_mode": "executed",
        "measurement_executed": True,
        "latency_samples_ns": [1, 2, 3, 4, 5],
        "command_q": [0.1, 0.2],
    }
    fixed = bulk_runner._fixed_alias_record(easy)
    assert fixed["entry_action"] == "fixed_robust"
    assert fixed["measurement_mode"] == "semantic_alias"
    assert fixed["measurement_executed"] is False
    assert fixed["aliased_from_action"] == "easy"
    assert fixed["latency_samples_ns"] == easy["latency_samples_ns"]
    fixed["latency_samples_ns"][0] = 99
    assert easy["latency_samples_ns"][0] == 1


def test_chunk_names_use_half_open_intervals() -> None:
    assert bulk_runner._chunk_name(0, 250) == "chunk_000000_000249"
    assert bulk_runner._chunk_intervals(501, 250) == [(0, 250), (250, 500), (500, 501)]


def test_resume_provenance_rejects_any_change() -> None:
    frozen = {"code": {"sha256": "a"}, "config": {"sha256": "b"}}
    bulk_runner._validate_frozen_provenance(frozen, frozen.copy())
    changed = {"code": {"sha256": "x"}, "config": {"sha256": "b"}}
    with pytest.raises(RuntimeError, match="refusing to mix"):
        bulk_runner._validate_frozen_provenance(frozen, changed)


def test_committed_chunk_hash_is_verified(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_000000_000001"
    chunk.mkdir()
    artifact = chunk / "counterfactual_labels.npz"
    artifact.write_bytes(b"labels")
    artifacts = {
        artifact.name: {
            "sha256": bulk_runner._sha256_file(artifact),
            "size": artifact.stat().st_size,
        }
    }
    manifest = {
        "robot": "panda",
        "training_seed": 17,
        "source_role": "risk_train_queries",
        "query_start": 0,
        "query_stop_exclusive": 2,
        "artifacts": artifacts,
    }
    manifest["chunk_payload_sha256"] = bulk_runner._digest_mapping(manifest)
    (chunk / "chunk_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checked = bulk_runner._validate_chunk_directory(
        chunk,
        robot="panda",
        training_seed=17,
        role="risk_train_queries",
        start=0,
        stop=2,
    )
    assert checked["chunk_payload_sha256"] == manifest["chunk_payload_sha256"]
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="mismatch"):
        bulk_runner._validate_chunk_directory(
            chunk,
            robot="panda",
            training_seed=17,
            role="risk_train_queries",
            start=0,
            stop=2,
        )


def test_atomic_npz_does_not_leave_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "selection.npz"
    bulk_runner._write_npz_atomic(path, source_indices=np.asarray([2, 1]))
    with np.load(path, allow_pickle=False) as data:
        assert data["source_indices"].tolist() == [2, 1]
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_contaminated_query_attempt_is_discarded_and_recollected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role_root = tmp_path / "panda" / "seed17" / "risk_train_queries"
    (role_root / "chunks").mkdir(parents=True)
    dataset = QueryDataset(
        previous_q=np.zeros((1, 2)),
        target_position=np.zeros((1, 3)),
        target_rotation=np.eye(3)[None],
        reference_q=np.zeros((1, 2)),
        category=np.asarray(["id"]),
        expected_reachable=np.ones(1, dtype=bool),
        continuity_feasible=np.ones(1, dtype=bool),
        trajectory_id=np.zeros(1, dtype=np.int64),
        time_index=np.zeros(1, dtype=np.int64),
    )
    calls = {"collect": 0, "quiet": 0}

    def fake_quiet(*_args, **kwargs):
        calls["quiet"] += 1
        return {
            "context": kwargs["context"],
            "wait_seconds": 0.0,
            "had_busy_process": False,
            "last_busy_processes": [],
        }

    def fake_collect(**_kwargs):
        calls["collect"] += 1
        marker = calls["collect"]
        rows = []
        for action in ("easy", "medium", "hard"):
            rows.append(
                {
                    "query_index": 0,
                    "entry_action": action,
                    "verified_success": True,
                    "verified_success_before_deadline": True,
                    "latency_samples_ns": [marker] * 5,
                    "latency_p50_ns": float(marker),
                    "latency_p95_ns": float(marker),
                    "function_evaluations": 1,
                    "fallback_used": False,
                    "failure_reason": "",
                    "measurement_executed": True,
                    "max_joint_step_rad": 0.0,
                    "max_joint_velocity_rad_s": 0.0,
                    "max_velocity_limit_utilization": 0.0,
                    "command_q": [0.0, 0.0],
                }
            )
        rows.append(bulk_runner._fixed_alias_record(rows[0]))
        return np.full(9, marker, dtype=np.float64), rows

    busy_responses = iter(
        [
            [{"pid": 999, "cpu_percent": 90.0, "stat": "R", "args": "noise"}],
            [],
            [],
        ]
    )
    monkeypatch.setattr(bulk_runner, "_wait_for_quiet_environment", fake_quiet)
    monkeypatch.setattr(
        bulk_runner, "_busy_unrelated_processes", lambda **_kwargs: next(busy_responses)
    )
    monkeypatch.setattr(bulk_runner, "_collect_query", fake_collect)
    runtime = SimpleNamespace(kinematics=SimpleNamespace(nq=2))
    manifest = bulk_runner._commit_chunk(
        role_root=role_root,
        robot="panda",
        training_seed=17,
        role="risk_train_queries",
        dataset=dataset,
        selected=np.asarray([0], dtype=np.int64),
        query_start=0,
        query_stop=1,
        runtimes={action: runtime for action in ("easy", "medium", "hard")},
        seed_engine=object(),
        config={
            "timing": {"repeats": 5, "deadline_ms": 20.0},
            "data": {"dt": 0.02},
            "runtime": {"max_unrelated_cpu_percent": 50.0},
        },
        selection_seed=123,
    )
    assert calls == {"collect": 2, "quiet": 2}
    assert manifest["contaminated_query_retries"] == 1
    assert manifest["environment_contaminated"] is False
    assert manifest["contaminated_attempt_events"][0]["feature_and_rows_discarded"] is True
    labels = role_root / "chunks" / "chunk_000000_000000" / "counterfactual_labels.npz"
    with np.load(labels, allow_pickle=False) as data:
        assert np.all(data["features"] == 2.0)
        assert np.all(data["latency_samples_ns"] == 2)
