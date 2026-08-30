from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from confik.data.datasets import QueryDataset
from confik.test_v4_locked.data import (
    ComparisonSource,
    TEST_V4_ROLES,
    audit_freshness,
    dataset_contract,
    default_comparison_sources,
    derive_seed,
    query_keys,
    validate_dataset_contract,
)


def _query(offset: float, *, category: str) -> QueryDataset:
    previous = np.asarray([[offset, offset + 0.1]], dtype=np.float64)
    position = np.asarray([[offset + 0.2, 0.0, 0.0]], dtype=np.float64)
    return QueryDataset(
        previous_q=previous,
        target_position=position,
        target_rotation=np.eye(3, dtype=np.float64)[None, ...],
        reference_q=previous.copy(),
        category=np.asarray([category]),
        expected_reachable=np.asarray([True]),
        continuity_feasible=np.asarray([True]),
        trajectory_id=np.asarray([int(offset * 10) + 1]),
        time_index=np.asarray([0]),
    )


def _roles() -> dict[str, QueryDataset]:
    return {
        "id_points": _query(0.0, category="id"),
        "id_trajectories": _query(1.0, category="trajectory_smooth"),
        "ood_points": _query(2.0, category="ood_workspace_sector"),
        "ood_trajectories": _query(
            3.0, category="ood_trajectory_high_frequency"
        ),
    }


def test_seed_uses_exact_frozen_material() -> None:
    digest = "a" * 64
    material = f"test_v4_locked|{digest}|panda|ood_points".encode("utf-8")
    expected = int.from_bytes(sha256(material).digest()[:4], "big", signed=False)
    assert derive_seed(digest, "panda", "ood_points") == expected
    assert derive_seed(digest, "panda", "ood_points") != derive_seed(
        digest, "panda", "id_points"
    )


def test_query_key_is_explicit_and_hash_matches_established_contract() -> None:
    dataset = _query(0.0, category="id")
    [key] = query_keys("panda", "id_points", dataset, dt=0.02)
    expected_digest = sha256()
    expected_digest.update(np.ascontiguousarray(dataset.previous_q[0]).tobytes())
    expected_digest.update(np.ascontiguousarray(dataset.target_position[0]).tobytes())
    expected_digest.update(np.ascontiguousarray(dataset.target_rotation[0]).tobytes())
    expected_digest.update(np.asarray([0.02], dtype=np.float64).tobytes())
    assert list(key) == ["robot", "domain", "family", "query_sha256"]
    assert key == {
        "robot": "panda",
        "domain": "id",
        "family": "id",
        "query_sha256": expected_digest.hexdigest(),
    }


def test_contract_uses_four_explicit_roles_not_category_prefixes() -> None:
    contract = dataset_contract(_roles(), robot="panda", dt=0.02)
    assert tuple(contract["roles"]) == TEST_V4_ROLES
    assert contract["roles"]["id_trajectories"]["domain"] == "id"
    assert contract["roles"]["ood_trajectories"]["domain"] == "ood"
    assert contract["roles"]["id_points"]["trajectory_count"] == 0
    assert contract["total_queries"] == 4
    assert contract["internal_duplicate_key_count"] == 0


def test_freshness_audit_reads_identity_not_old_performance(tmp_path: Path) -> None:
    roles = _roles()
    hash_source = tmp_path / "pilot_hashes.npz"
    np.savez_compressed(hash_source, query_sha256=np.asarray(["f" * 64]))

    old_test = tmp_path / "old_test.npz"
    old = _query(9.0, category="id")
    # Object data would require pickle if accessed. The freshness reader must
    # succeed because it loads only the three preregistered identity arrays.
    np.savez_compressed(
        old_test,
        previous_q=old.previous_q,
        target_position=old.target_position,
        target_rotation=old.target_rotation,
        forbidden_performance_result=np.asarray([{"latency_ms": 1.0}], dtype=object),
    )
    audit = audit_freshness(
        roles,
        robot="panda",
        dt=0.02,
        comparison_sources=[
            ComparisonSource("pilot", hash_source, "query_hash_npz"),
            ComparisonSource("old_test", old_test, "identity_npz"),
        ],
    )
    assert audit["passed"]
    assert audit["comparison_sources"]["old_test"]["arrays_read"] == [
        "previous_q",
        "target_position",
        "target_rotation",
    ]
    assert not audit["comparison_sources"]["old_test"][
        "performance_arrays_read"
    ]


def test_freshness_audit_reports_prior_overlap(tmp_path: Path) -> None:
    roles = _roles()
    overlap = query_keys("panda", "id_points", roles["id_points"], dt=0.02)[0][
        "query_sha256"
    ]
    source = tmp_path / "bulk.npz"
    np.savez_compressed(source, query_sha256=np.asarray([overlap]))
    audit = audit_freshness(
        roles,
        robot="panda",
        dt=0.02,
        comparison_sources=[ComparisonSource("bulk", source, "query_hash_npz")],
    )
    assert not audit["passed"]
    assert audit["prior_source_exact_overlap_counts"] == {"bulk": 1}


def test_default_sources_cover_bulk_both_pilots_and_old_test(tmp_path: Path) -> None:
    names = {source.name for source in default_comparison_sources(tmp_path, "ur5e")}
    assert {
        "bulk/risk_train_queries",
        "bulk/calibration_queries",
        "bulk/policy_validation_queries",
        "counterfactual_v4_pilot",
        "latency_pilot_v3/risk_validation_source",
        "latency_pilot_v3/trajectory_validation_source",
        "old_formal_test_v3_identity",
    } == names


def test_preregistration_locks_data_statistics_and_claim_contract() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "test_v4_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert tuple(config["data"]["roles"]) == TEST_V4_ROLES
    assert [config["data"]["roles"][role]["expected_queries"] for role in TEST_V4_ROLES] == [
        12_000,
        6_000,
        4_000,
        3_000,
    ]
    assert config["data"]["expected_queries_per_robot"] == 25_000
    assert config["statistics"]["bootstrap_samples"] == 10_000
    assert config["statistics"]["multiplicity_correction"] == "holm"
    assert config["claim_gate"]["trajectory_completion_gap_min"] == -0.10
    assert config["runtime"]["hysteresis"] == {
        "enabled": False,
        "method": "disabled_by_preregistration",
    }
    assert not config["data"]["freshness"]["old_test_performance_results_allowed"]


def test_validate_dataset_contract_rejects_changed_counts() -> None:
    contract = dataset_contract(_roles(), robot="panda", dt=0.02)
    roles_config: dict[str, dict[str, object]] = {}
    for role in TEST_V4_ROLES:
        observed = contract["roles"][role]
        roles_config[role] = {
            "domain": observed["domain"],
            "expected_queries": observed["query_count"],
            "expected_trajectories": observed["trajectory_count"],
            "expected_category_counts": observed["category_counts"],
        }
    config = {"data": {"roles": roles_config, "expected_queries_per_robot": 4}}
    validate_dataset_contract(contract, config)
    config["data"]["roles"]["ood_points"]["expected_queries"] = 2
    try:
        validate_dataset_contract(contract, config)
    except RuntimeError as error:
        assert "ood_points" in str(error)
    else:
        raise AssertionError("changed preregistered count was accepted")
