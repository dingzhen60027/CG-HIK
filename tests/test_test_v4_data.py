from __future__ import annotations

from hashlib import sha256
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from confik.data.datasets import QueryDataset
from confik.test_v4_locked.data import (
    BULK_QUERY_ROLES,
    COUNTERFACTUAL_V4_QUERY_ROOTS,
    ComparisonSource,
    PAPER_V2_QUERY_ROLES,
    PAPER_V2_SEEDS,
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


def _save_identity(
    path: Path,
    offsets: list[float],
    *,
    trajectory_ids: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = np.asarray(
        [[value, value + 0.1] for value in offsets], dtype=np.float64
    )
    position = np.asarray(
        [[value + 0.2, 0.0, 0.0] for value in offsets], dtype=np.float64
    )
    np.savez_compressed(
        path,
        previous_q=previous,
        target_position=position,
        target_rotation=np.repeat(
            np.eye(3, dtype=np.float64)[None, ...], len(offsets), axis=0
        ),
        trajectory_id=np.asarray(
            trajectory_ids if trajectory_ids is not None else list(range(len(offsets))),
            dtype=np.int64,
        ),
    )


def _comparison_workspace(root: Path, robot: str) -> None:
    for seed in PAPER_V2_SEEDS:
        for role_index, role in enumerate(PAPER_V2_QUERY_ROLES):
            _save_identity(
                root
                / "outputs"
                / f"paper_v2_seed{seed}"
                / robot
                / "datasets"
                / f"{role}.npz",
                [
                    float(seed * 100 + role_index),
                    float(seed * 100 + role_index) + 0.5,
                ],
                trajectory_ids=[10, 11],
            )
    for role_index, role in enumerate(BULK_QUERY_ROLES):
        path = (
            root
            / "outputs/counterfactual_v4_bulk"
            / robot
            / "seed17"
            / role
            / "selection.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, query_sha256=np.asarray([f"{role_index + 1:064x}"]))
    for root_index, directory in enumerate(COUNTERFACTUAL_V4_QUERY_ROOTS):
        path = root / "outputs" / directory / robot / "seed17/counterfactual_labels.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, query_sha256=np.asarray([f"{root_index + 10:064x}"]))
    _save_identity(
        root / "outputs/test_v3_aggregate/datasets" / f"{robot}_test_v3_queries.npz",
        [999.0],
    )
    latency = root / "outputs/latency_pilot_v3/run_manifest.json"
    latency.parent.mkdir(parents=True, exist_ok=True)
    latency.write_text(
        json.dumps(
            {
                "protocol_version": "latency_pilot_v3",
                "formal_test_v3_started": False,
                "selection_inputs": {
                    robot: {
                        "test_queries_loaded": False,
                        "point_source_split": "risk_validation_queries",
                        "trajectory_source_split": "seed_validation",
                        "point_selected_source_indices": [0],
                        "selected_trajectory_ids": [10],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_seed_uses_exact_frozen_material() -> None:
    digest = "a" * 64
    material = f"test_v4_locked|{digest}|panda|ood_points".encode("utf-8")
    expected = int.from_bytes(sha256(material).digest()[:4], "big", signed=False)
    assert derive_seed(digest, "panda", "ood_points") == expected
    assert derive_seed(digest, "panda", "ood_points") != derive_seed(
        digest, "panda", "id_points"
    )


def test_optional_inspected_czy_and_incomplete_smoke_are_identity_sources(
    tmp_path: Path,
) -> None:
    robot = "panda"
    _comparison_workspace(tmp_path, robot)
    czy = tmp_path / "czy/closed_loop_v3_raw_frame_records.csv"
    czy.parent.mkdir(parents=True)
    with czy.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "actual_joint_state",
                "target_ee_position",
                "target_ee_rotation",
                "dt_s",
                "latency_ms",
            ]
        )
        writer.writerow(
            [
                json.dumps([40.0, 40.1]),
                json.dumps([40.2, 0.0, 0.0]),
                json.dumps(np.eye(3).tolist()),
                "0.02",
                "999.0",
            ]
        )
    incomplete = (
        tmp_path
        / "outputs/.counterfactual_v4_readiness_smoke.incomplete.1313949"
        / robot
        / "seed17/counterfactual_records.jsonl.gz"
    )
    incomplete.parent.mkdir(parents=True)
    with gzip.open(incomplete, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "source_query_sha256": "e" * 64,
                    "latency_samples_ns": [999999999],
                }
            )
            + "\n"
        )
    sources = default_comparison_sources(tmp_path, robot)
    by_name = {source.name: source for source in sources}
    assert by_name["czy/closed_loop_v3_query_identity"].kind == "czy_identity_csv"
    assert (
        by_name["counterfactual_v4_readiness_smoke_incomplete/seed17"].kind
        == "query_hash_jsonl_gz"
    )
    audit = audit_freshness(
        _roles(), robot=robot, dt=0.02, comparison_sources=sources
    )
    czy_manifest = audit["comparison_sources"]["czy/closed_loop_v3_query_identity"]
    assert czy_manifest["arrays_read"] == [
        "actual_joint_state",
        "target_ee_position",
        "target_ee_rotation",
        "dt_s",
    ]
    assert not czy_manifest["performance_arrays_read"]


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
    assert audit["comparison_sources"]["old_test"]["role"] == "unspecified"
    assert audit["comparison_sources"]["old_test"]["query_count"] == 1
    assert audit["comparison_sources"]["old_test"]["unique_query_sha256"] == 1
    assert len(audit["comparison_sources"]["old_test"]["sha256"]) == 64


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


def test_freshness_audit_rejects_empty_or_duplicate_source_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        audit_freshness(
            _roles(), robot="panda", dt=0.02, comparison_sources=[]
        )
    source = tmp_path / "hashes.npz"
    np.savez_compressed(source, query_sha256=np.asarray(["f" * 64]))
    duplicate = ComparisonSource("duplicate", source, "query_hash_npz")
    with pytest.raises(ValueError, match="names must be unique"):
        audit_freshness(
            _roles(),
            robot="panda",
            dt=0.02,
            comparison_sources=[duplicate, duplicate],
        )


def test_default_sources_are_exhaustive_and_fail_closed(tmp_path: Path) -> None:
    robot = "ur5e"
    _comparison_workspace(tmp_path, robot)
    sources = default_comparison_sources(tmp_path, robot)
    names = {source.name for source in sources}
    expected_paper = {
        f"paper_v2/seed{seed}/{role}"
        for seed in PAPER_V2_SEEDS
        for role in PAPER_V2_QUERY_ROLES
    }
    assert len(sources) == 37
    assert names == {
        *expected_paper,
        *(f"bulk/{role}" for role in BULK_QUERY_ROLES),
        *(f"{directory}/seed17" for directory in COUNTERFACTUAL_V4_QUERY_ROOTS),
        "latency_pilot_v3/point_validation_selection",
        "latency_pilot_v3/trajectory_validation_selection",
        "old_formal_test_v3_identity",
    }
    assert {source.group for source in sources if source.name in expected_paper} == {
        "paper_v2"
    }
    assert next(
        source
        for source in sources
        if source.name == "latency_pilot_v3/point_validation_selection"
    ).source_indices == (0,)
    assert next(
        source
        for source in sources
        if source.name == "latency_pilot_v3/trajectory_validation_selection"
    ).trajectory_ids == (10,)

    missing = (
        tmp_path
        / "outputs/paper_v2_seed43"
        / robot
        / "datasets/policy_validation_queries.npz"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="policy_validation_queries"):
        default_comparison_sources(tmp_path, robot)


def test_freshness_manifest_records_selected_role_count_and_hash(tmp_path: Path) -> None:
    indexed = tmp_path / "risk_validation_queries.npz"
    _save_identity(indexed, [10.0, 11.0, 12.0], trajectory_ids=[1, 1, 2])
    trajectory = tmp_path / "seed_validation.npz"
    _save_identity(trajectory, [20.0, 21.0, 22.0], trajectory_ids=[5, 6, 6])
    provenance = tmp_path / "run_manifest.json"
    provenance.write_text('{"validation_only": true}\n', encoding="utf-8")
    audit = audit_freshness(
        _roles(),
        robot="panda",
        dt=0.02,
        comparison_sources=[
            ComparisonSource(
                "latency/point",
                indexed,
                "indexed_identity_npz",
                group="latency_pilot_v3",
                role="risk_validation_queries",
                source_indices=(0, 2),
                provenance_path=provenance,
            ),
            ComparisonSource(
                "latency/trajectory",
                trajectory,
                "trajectory_identity_npz",
                group="latency_pilot_v3",
                role="seed_validation",
                trajectory_ids=(6,),
                provenance_path=provenance,
            ),
        ],
    )
    assert audit["passed"]
    point = audit["comparison_sources"]["latency/point"]
    assert point["group"] == "latency_pilot_v3"
    assert point["role"] == "risk_validation_queries"
    assert point["query_count"] == point["unique_query_sha256"] == 2
    assert point["selector"]["type"] == "source_indices"
    assert len(point["selector"]["selector_sha256"]) == 64
    assert len(point["sha256"]) == 64
    assert len(point["query_sha256_set_digest"]) == 64
    assert point["provenance"]["path"] == str(provenance.resolve())
    trajectory_row = audit["comparison_sources"]["latency/trajectory"]
    assert trajectory_row["query_count"] == 2
    assert trajectory_row["selector"] == {
        "type": "trajectory_ids",
        "trajectory_count": 1,
        "selected_count": 2,
        "selector_sha256": trajectory_row["selector"]["selector_sha256"],
    }
    assert audit["comparison_source_group_counts"] == {"latency_pilot_v3": 2}


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
