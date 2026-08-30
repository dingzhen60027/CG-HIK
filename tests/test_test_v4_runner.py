from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from confik.test_v4_locked.benchmark import PRIMARY_METHODS, SENSITIVITY_METHODS
from confik.test_v4_locked.runner import (
    _baseline_availability,
    _safe,
    _verify_release,
    _write_json,
)


def test_formal_config_locks_seven_primary_and_three_sensitivity_methods() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "test_v4_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert tuple(config["methods"]["primary"]) == PRIMARY_METHODS
    assert tuple(config["methods"]["sensitivity_only"]) == SENSITIVITY_METHODS
    assert config["timing"]["clock"] == "perf_counter_ns"
    assert config["timing"]["interleaved_same_query_methods"]
    assert not config["timing"]["disk_writes_inside_timed_interval"]
    assert config["statistics"]["bootstrap_samples"] == 10_000
    assert config["statistics"]["multiplicity_correction"] == "holm"


def test_baseline_manifest_does_not_mislabel_trf_as_trac_ik() -> None:
    availability = _baseline_availability()
    assert availability["trf_previous"]["available"]
    assert not availability["trf_previous"]["is_trac_ik"]
    assert not availability["trac_ik"]["available"]
    assert not availability["trac_ik"]["substitution_claimed"]


def test_json_writer_is_atomic_and_converts_nonfinite_values(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    _write_json(destination, {"finite": 1.0, "nan": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "nan": None,
    }
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert _safe(float("inf")) is None


def test_release_verifier_rejects_a_release_that_authorized_test(tmp_path: Path) -> None:
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps({"files": {}}), encoding="utf-8"
    )
    import hashlib

    digest = hashlib.sha256((tmp_path / "artifact_manifest.json").read_bytes()).hexdigest()
    (tmp_path / "release_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "release_v4_locked",
                "release_status": "sealed",
                "backend": "torchscript_exact_v4",
                "all_six_validation_runtime_equivalence_pass": True,
                "formal_test_authorized_or_started": True,
                "test_v4_started": False,
                "artifact_manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "release_equivalence.json").write_text(
        json.dumps({"all_pass": True}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not eligible"):
        _verify_release(tmp_path)
