from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from confik.anchored_temporal_v7 import dominance_pilot
from confik.anchored_temporal_v7.dominance_selection import (
    ARM_NAMES,
    FRAME_COUNT,
    FRAMES_PER_TRAJECTORY,
    R_VALUES,
    TRAJECTORIES,
    FrozenCalibrationRecords,
    NoDominanceEligibleCandidate,
    load_frozen_v7_calibration_records,
    select_dominance_reanchor_interval,
)
from confik.anchored_temporal_v7.pilot import BenchmarkData, TrajectoryView
from confik.anchored_temporal_v7.trajectories import FAMILIES
from confik.config import load_config, resolve_path
from confik.data.datasets import QueryDataset


WORKSPACE = Path(__file__).resolve().parents[1]


def _records() -> FrozenCalibrationRecords:
    order = tuple(f"{index:064x}" for index in range(TRAJECTORIES))
    accepted = np.zeros((len(ARM_NAMES), FRAME_COUNT), dtype=bool)
    latency = np.full((len(ARM_NAMES), FRAME_COUNT), 10_000, dtype=np.int64)
    fev = np.full((len(ARM_NAMES), FRAME_COUNT), 10, dtype=np.int64)
    seed = np.ones((len(ARM_NAMES), FRAME_COUNT), dtype=bool)
    records = FrozenCalibrationRecords(
        robot="panda",
        source_path="/synthetic/panda_calibration_records.npz",
        source_sha256="a" * 64,
        source_size_bytes=1,
        arm_names=ARM_NAMES,
        reanchor_interval=np.asarray((-1,) + R_VALUES, dtype=np.int16),
        latency_ns=latency,
        accepted=accepted,
        function_evaluations=fev,
        seed_invoked=seed,
        trajectory_uid=np.repeat(np.asarray(order, dtype="U64"), FRAMES_PER_TRAJECTORY),
        trajectory_order=order,
        category=np.repeat(
            np.asarray(
                [f"family_{index % 4}" for index in range(TRAJECTORIES)],
                dtype="U16",
            ),
            FRAMES_PER_TRAJECTORY,
        ),
        time_index=np.tile(np.arange(FRAMES_PER_TRAJECTORY), TRAJECTORIES),
    )
    return records


def _set_completed(
    accepted: np.ndarray, arm: int, trajectory_indices: set[int]
) -> None:
    accepted[arm] = False
    for trajectory_index in trajectory_indices:
        start = trajectory_index * FRAMES_PER_TRAJECTORY
        accepted[arm, start : start + FRAMES_PER_TRAJECTORY] = True


def _candidate(report: dict[str, object], reanchor_interval: int) -> dict[str, object]:
    rows = report["candidate_metrics"]
    assert isinstance(rows, list)
    return next(
        row for row in rows if row["reanchor_interval"] == reanchor_interval
    )


def test_dominance_rejects_loss_and_maximizes_completion_before_latency() -> None:
    records = _records()
    _set_completed(records.accepted, 0, {0, 1})
    _set_completed(records.accepted, 1, {1, 2, 3, 4, 5})  # gain four, lose one
    _set_completed(records.accepted, 2, {0, 1, 2})
    _set_completed(records.accepted, 3, {0, 1, 2, 3})
    _set_completed(records.accepted, 4, {0, 1})
    _set_completed(records.accepted, 5, {0, 1})
    records.latency_ns[1] = 1  # Ineligible candidate must never win.
    records.latency_ns[2] = 2
    records.latency_ns[3] = 50_000  # More completions outrank cumulative time.

    selected, report = select_dominance_reanchor_interval(records)

    assert selected == 30
    rejected = _candidate(report, 20)
    assert rejected["eligible"] is False
    assert rejected["lost_trajectory_uids"] == [records.trajectory_order[0]]
    assert rejected["gained_trajectory_uids"] == list(records.trajectory_order[2:6])
    winner = report["selected"]
    assert winner["whole_trajectory_completion_count"] == 4
    assert winner["lost_trajectory_uids"] == []
    assert winner["gained_trajectory_uids"] == list(records.trajectory_order[2:4])


def test_aggregate_cumulative_latency_precedes_frame_p99() -> None:
    records = _records()
    for arm in range(len(ARM_NAMES)):
        _set_completed(records.accepted, arm, {0, 1})
    records.latency_ns[1] = 100
    # Lower total than R20, but a deliberately worse frame P99.
    records.latency_ns[2] = 1
    records.latency_ns[2, :120] = 1_000
    records.latency_ns[3:] = 1_000

    selected, report = select_dominance_reanchor_interval(records)

    assert selected == 25
    r20 = _candidate(report, 20)
    r25 = _candidate(report, 25)
    assert r25["aggregate_cumulative_latency_ns"] < r20[
        "aggregate_cumulative_latency_ns"
    ]
    assert r25["p99_latency_ms"] > r20["p99_latency_ms"]


def test_frame_p99_seed_fev_and_lower_r_are_ordered_tie_breaks() -> None:
    records = _records()
    for arm in range(len(ARM_NAMES)):
        _set_completed(records.accepted, arm, {0, 1})

    # R20 and R25 have equal totals; R20 wins their comparison on P99.
    records.latency_ns[1] = 100
    records.latency_ns[2, : FRAME_COUNT // 2] = 50
    records.latency_ns[2, FRAME_COUNT // 2 :] = 150
    # R30/R40/R50 tie on total and P99, then seed, FEV and lower R decide.
    records.latency_ns[3:] = 90
    records.seed_invoked[3:] = False
    records.function_evaluations[3] = 2
    records.function_evaluations[4] = 1
    records.function_evaluations[5] = 1

    selected, report = select_dominance_reanchor_interval(records)

    assert selected == 40
    assert _candidate(report, 20)["p99_latency_ms"] < _candidate(report, 25)[
        "p99_latency_ms"
    ]
    assert report["selection_sort_key"]["mean_fev"] == 1.0
    assert report["selection_sort_key"]["reanchor_interval"] == 40


def test_reports_exact_trajectory_cumulative_latency_statistics() -> None:
    records = _records()
    for arm in range(len(ARM_NAMES)):
        _set_completed(records.accepted, arm, {0})
    for trajectory_index in range(TRAJECTORIES):
        start = trajectory_index * FRAMES_PER_TRAJECTORY
        records.latency_ns[1, start : start + FRAMES_PER_TRAJECTORY] = (
            trajectory_index + 1
        )

    _, report = select_dominance_reanchor_interval(records)
    r20 = _candidate(report, 20)
    trajectory_totals = 150 * np.arange(1, TRAJECTORIES + 1, dtype=np.int64)
    assert r20["aggregate_cumulative_latency_ns"] == int(
        np.sum(trajectory_totals)
    )
    assert r20["trajectory_cumulative_latency_mean_ms"] == pytest.approx(
        float(np.mean(trajectory_totals) / 1e6)
    )
    assert r20["trajectory_cumulative_latency_median_ms"] == pytest.approx(
        float(np.median(trajectory_totals) / 1e6)
    )
    assert r20["trajectory_cumulative_latency_p95_ms"] == pytest.approx(
        float(np.quantile(trajectory_totals, 0.95) / 1e6)
    )
    by_uid = r20["trajectory_cumulative_latency_ns_by_uid"]
    assert by_uid[records.trajectory_order[0]] == 150
    assert by_uid[records.trajectory_order[-1]] == 6_000


def test_no_dominance_candidate_fails_closed() -> None:
    records = _records()
    _set_completed(records.accepted, 0, {0, 1})
    for arm in range(1, len(ARM_NAMES)):
        _set_completed(records.accepted, arm, {1, 2})

    with pytest.raises(NoDominanceEligibleCandidate) as error:
        select_dominance_reanchor_interval(records)

    assert error.value.report["eligible_candidate_count"] == 0
    assert error.value.report["selected"] is None
    assert error.value.report["policy_validation_outcomes_opened"] is False


def test_selection_rejects_duplicate_trajectory_uids() -> None:
    records = _records()
    duplicate_order = (
        records.trajectory_order[0],
        records.trajectory_order[0],
        *records.trajectory_order[2:],
    )
    with pytest.raises(ValueError, match="40 unique UIDs"):
        select_dominance_reanchor_interval(
            replace(records, trajectory_order=duplicate_order)
        )


def test_loader_accepts_sha_pinned_original_v7_calibration_records() -> None:
    root = WORKSPACE / "outputs" / "anchored_temporal_v7_pilot"
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    path = root / "panda_calibration_records.npz"

    records = load_frozen_v7_calibration_records(
        path,
        robot="panda",
        expected_artifact=manifest["artifacts"][path.name],
    )
    selected, report = select_dominance_reanchor_interval(records)

    assert selected == 50
    assert report["selected"]["lost_trajectory_uids"] == []
    assert len(report["selected"]["gained_trajectory_uids"]) == 2
    assert records.latency_ns.flags.writeable is False


def test_loader_rejects_hash_mismatch_strict_schema_and_pv_path(
    tmp_path: Path,
) -> None:
    root = WORKSPACE / "outputs" / "anchored_temporal_v7_pilot"
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    original = root / "panda_calibration_records.npz"
    bad_descriptor = dict(manifest["artifacts"][original.name])
    bad_descriptor["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/size mismatch"):
        load_frozen_v7_calibration_records(
            original, robot="panda", expected_artifact=bad_descriptor
        )

    incomplete = tmp_path / "panda_calibration_records.npz"
    np.savez(incomplete, latency_ns=np.ones((6, FRAME_COUNT), dtype=np.int64))
    raw = incomplete.read_bytes()
    descriptor = {
        "path": incomplete.name,
        "sha256": sha256(raw).hexdigest(),
        "size": len(raw),
    }
    with pytest.raises(ValueError, match="schema changed"):
        load_frozen_v7_calibration_records(
            incomplete, robot="panda", expected_artifact=descriptor
        )

    forbidden = tmp_path / "panda_trajectory_policy_validation.npz"
    forbidden.write_bytes(b"not opened")
    forbidden_raw = forbidden.read_bytes()
    with pytest.raises(ValueError, match="calibration records only"):
        load_frozen_v7_calibration_records(
            forbidden,
            robot="panda",
            expected_artifact={
                "path": forbidden.name,
                "sha256": sha256(forbidden_raw).hexdigest(),
                "size": len(forbidden_raw),
            },
        )


def test_dominance_config_is_exact_rejects_formal_paths_and_has_no_smoke() -> None:
    path = WORKSPACE / "configs" / "anchored_temporal_v7_dominance.yaml"
    config = load_config(path)
    workspace = resolve_path(config, str(config["workspace"]))
    dominance_pilot.validate_config(config, workspace=workspace)

    changed = deepcopy(config)
    changed["dominance_selection"]["objective_order"] = list(
        reversed(changed["dominance_selection"]["objective_order"])
    )
    with pytest.raises(ValueError, match="selection contract changed"):
        dominance_pilot.validate_config(changed, workspace=workspace)

    formal = deepcopy(config)
    formal["source_config"] = "../outputs/test_v4/paper_v2.yaml"
    with pytest.raises(ValueError, match="formal test data"):
        dominance_pilot.validate_config(formal, workspace=workspace)

    parser = dominance_pilot._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", str(path), "--smoke"])
    launcher = (
        WORKSPACE / "scripts" / "run_anchored_temporal_v7_dominance.sh"
    ).read_text(encoding="utf-8")
    assert "--smoke" not in launcher


def _gate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for robot in ("panda", "ur5e"):
        rows.extend(
            [
                {
                    "robot": robot,
                    "method": "always_hard",
                    "completion_trajectory_uids": ["hard-ok"],
                    "whole_trajectory_completion_count": 1,
                    "total_latency_over_all_frames_ns": 1_000,
                    "p50_frame_latency_ms": 1.0,
                    "p95_frame_latency_ms": 2.0,
                    "p99_frame_latency_ms": 10.0,
                    "mean_fev": 10.0,
                    "learned_seed_invocation_rate": 1.0,
                },
                {
                    "robot": robot,
                    "method": "anchored_temporal_cghik_v7",
                    "completion_trajectory_uids": ["hard-ok", "gained"],
                    "whole_trajectory_completion_count": 2,
                    "total_latency_over_all_frames_ns": 800,
                    "p50_frame_latency_ms": 0.5,
                    "p95_frame_latency_ms": 1.5,
                    "p99_frame_latency_ms": 10.5,
                    "mean_fev": 10.5,
                    "learned_seed_invocation_rate": 0.5,
                },
            ]
        )
    return rows


def test_dominance_gate_allows_gains_and_checks_all_six_boundaries() -> None:
    passing = dominance_pilot.dominance_gate(_gate_rows())
    assert passing["status"] == "pass"
    assert passing["fresh_trajectory_evaluation_authorized"] is True
    for robot in ("panda", "ur5e"):
        assert passing["robots"][robot]["lost_trajectory_uids"] == []
        assert passing["robots"][robot]["gained_trajectory_uids"] == ["gained"]
        assert len(passing["robots"][robot]["checks"]) == 6
        assert all(passing["robots"][robot]["checks"].values())


@pytest.mark.parametrize(
    ("failed_check", "field", "value"),
    (
        (
            "always_hard_completion_set_subset_of_anchor",
            "completion_trajectory_uids",
            ["gained", "another-gain"],
        ),
        (
            "anchored_completion_count_not_below_always_hard",
            "whole_trajectory_completion_count",
            0,
        ),
        (
            "aggregate_cumulative_latency_ratio_at_most_0_80",
            "total_latency_over_all_frames_ns",
            801,
        ),
        (
            "learned_seed_invocation_rate_at_most_0_50",
            "learned_seed_invocation_rate",
            0.500_001,
        ),
        ("p99_ratio_at_most_1_05", "p99_frame_latency_ms", 10.500_001),
        ("mean_fev_ratio_at_most_1_05", "mean_fev", 10.500_001),
    ),
)
def test_dominance_gate_rejects_each_failed_condition(
    failed_check: str, field: str, value: object
) -> None:
    rows = _gate_rows()
    panda_anchor = next(
        row
        for row in rows
        if row["robot"] == "panda"
        and row["method"] == "anchored_temporal_cghik_v7"
    )
    panda_anchor[field] = value

    gate = dominance_pilot.dominance_gate(rows)

    assert gate["status"] == "fail"
    assert gate["fresh_trajectory_evaluation_authorized"] is False
    assert gate["robots"]["panda"]["checks"][failed_check] is False


def _synthetic_benchmark() -> BenchmarkData:
    count = FRAME_COUNT
    methods = len(dominance_pilot.METHODS)
    nq = 7
    order = tuple(f"{index + 100:064x}" for index in range(TRAJECTORIES))
    trajectory_uid = np.repeat(np.asarray(order, dtype="U64"), FRAMES_PER_TRAJECTORY)
    categories_by_trajectory = np.repeat(np.asarray(FAMILIES, dtype="U35"), 10)
    category = np.repeat(categories_by_trajectory, FRAMES_PER_TRAJECTORY)
    time_index = np.tile(np.arange(FRAMES_PER_TRAJECTORY), TRAJECTORIES)
    dataset = QueryDataset(
        previous_q=np.zeros((count, nq)),
        target_position=np.zeros((count, 3)),
        target_rotation=np.broadcast_to(np.eye(3), (count, 3, 3)).copy(),
        reference_q=np.zeros((count, nq)),
        category=category,
        expected_reachable=np.ones(count, dtype=bool),
        continuity_feasible=np.ones(count, dtype=bool),
        trajectory_id=np.repeat(np.arange(TRAJECTORIES), FRAMES_PER_TRAJECTORY),
        time_index=time_index,
    )
    role = TrajectoryView(
        robot="panda",
        role="anchored_trajectory_policy_validation",
        dataset=dataset,
        source_query_hash=np.asarray(
            [f"{index + 10_000:064x}" for index in range(count)], dtype="U64"
        ),
        trajectory_uid=trajectory_uid,
        trajectory_order=order,
        phase=np.full(count, "phase", dtype="U32"),
        phase_index=time_index.copy(),
        transition_boundary=np.zeros(count, dtype=bool),
        dt=0.02,
    )
    latency = np.empty((count, methods), dtype=np.int64)
    for column in range(methods):
        for trajectory_index in range(TRAJECTORIES):
            start = trajectory_index * FRAMES_PER_TRAJECTORY
            latency[start : start + FRAMES_PER_TRAJECTORY, column] = (
                (column + 1) * 1_000_000 + trajectory_index * 10_000
            )
    accepted = np.ones((count, methods), dtype=bool)
    fev = np.broadcast_to(
        np.arange(1, methods + 1, dtype=np.int64), (count, methods)
    ).copy()
    seed = np.ones((count, methods), dtype=bool)
    seed[::2, 3] = False
    occupancy = np.full((count, methods), -1, dtype=np.int8)
    occupancy[:, 2:] = dominance_pilot.OCCUPANCY_CODE["local"]
    anchor_kind = np.full((count, methods), "", dtype="U32")
    for trajectory_index in range(TRAJECTORIES):
        start = trajectory_index * FRAMES_PER_TRAJECTORY
        occupancy[start, 2:] = dominance_pilot.OCCUPANCY_CODE["anchor"]
        occupancy[start + 10 : start + 15, 2:] = dominance_pilot.OCCUPANCY_CODE[
            "robust"
        ]
        anchor_kind[start + 20, 3] = "periodic"
    bools = np.zeros((count, methods), dtype=bool)
    ints = np.zeros((count, methods), dtype=np.int16)
    strings = np.full((count, methods), "", dtype="U64")
    return BenchmarkData(
        robot="panda",
        role=role,
        latency_ns=latency,
        accepted=accepted,
        function_evaluations=fev,
        seed_invoked=seed,
        local_attempted=bools.copy(),
        local_accepted=bools.copy(),
        hard_attempted=bools.copy(),
        hard_accepted=bools.copy(),
        same_frame_hard_recovery_attempted=bools.copy(),
        same_frame_hard_recovery=bools.copy(),
        occupancy_mode=occupancy,
        state_before=ints.copy(),
        state_after=ints.copy(),
        mode_switched=bools.copy(),
        switch_kind=strings.copy(),
        anchor_scheduled=bools.copy(),
        anchor_attempted=bools.copy(),
        anchor_accepted=bools.copy(),
        anchor_kind=anchor_kind,
        local_probe_scheduled=bools.copy(),
        local_probe=bools.copy(),
        hard_count_before=ints.copy(),
        hard_count_after=ints.copy(),
        local_streak_before=ints.copy(),
        local_streak_after=ints.copy(),
        recovery_delay=ints.copy(),
        route=strings.copy(),
        executed_stages=strings.copy(),
        command_q=np.zeros((count, methods, nq), dtype=np.float64),
        executed_query_hash=np.full((count, methods), "a" * 64, dtype="U64"),
        method_order_position=np.zeros((count, methods), dtype=np.int8),
        stage_latency_ns=np.zeros((count, methods, 4), dtype=np.int64),
    )


def test_holdout_trajectory_main_and_family_latency_recompute_exactly() -> None:
    data = _synthetic_benchmark()
    trajectory_rows = dominance_pilot.trajectory_metric_rows(data)
    main_rows = dominance_pilot.summarize_holdout(data, trajectory_rows)
    family = dominance_pilot.family_rows(data, trajectory_rows)

    assert len(trajectory_rows) == len(dominance_pilot.METHODS) * TRAJECTORIES
    hard_first = next(
        row
        for row in trajectory_rows
        if row["method"] == "always_hard"
        and row["trajectory_uid"] == data.role.trajectory_order[0]
    )
    assert hard_first["cumulative_latency_ns"] == 150_000_000
    anchored_first = next(
        row
        for row in trajectory_rows
        if row["method"] == "anchored_temporal_cghik_v7"
        and row["trajectory_uid"] == data.role.trajectory_order[0]
    )
    assert anchored_first["cumulative_latency_ns"] == 600_000_000
    assert anchored_first["periodic_reanchor_count"] == 1
    assert anchored_first["longest_robust_run"] == 5

    hard_totals = 150 * (
        1_000_000 + np.arange(TRAJECTORIES, dtype=np.int64) * 10_000
    )
    hard_main = next(row for row in main_rows if row["method"] == "always_hard")
    assert hard_main["total_latency_over_all_frames_ns"] == int(
        np.sum(hard_totals)
    )
    assert hard_main["trajectory_cumulative_latency_mean_ms"] == pytest.approx(
        float(np.mean(hard_totals) / 1e6)
    )
    assert hard_main["trajectory_cumulative_latency_median_ms"] == pytest.approx(
        float(np.median(hard_totals) / 1e6)
    )
    assert hard_main["trajectory_cumulative_latency_p95_ms"] == pytest.approx(
        float(np.quantile(hard_totals, 0.95) / 1e6)
    )

    first_family = next(
        row
        for row in family
        if row["method"] == "always_hard" and row["family"] == FAMILIES[0]
    )
    expected_family = hard_totals[:10]
    assert first_family["total_cumulative_latency_ns"] == int(
        np.sum(expected_family)
    )
    assert first_family["trajectory_cumulative_latency_p95_ms"] == pytest.approx(
        float(np.quantile(expected_family, 0.95) / 1e6)
    )


def test_verify_frozen_v7_accepts_current_real_no_go_tree() -> None:
    config = load_config(
        WORKSPACE / "configs" / "anchored_temporal_v7_dominance.yaml"
    )
    workspace = resolve_path(config, str(config["workspace"]))
    root = resolve_path(config, str(config["frozen_v7_root"]))

    manifest, evidence = dominance_pilot.verify_frozen_v7(
        workspace=workspace,
        root=root,
        required_sha256=config["frozen_inputs"]["required_sha256"],
    )

    assert manifest["status"] == "calibration_no_go"
    assert evidence["old_exact_vector_no_go_preserved"] is True
    assert evidence["old_policy_validation_outcomes_computed"] is False
    assert evidence["policy_validation_npz_parsed_during_verification"] is False


def _descriptor(path: Path, *, root: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(raw).hexdigest(),
        "size": len(raw),
    }


def test_selection_seal_rejects_selected_r_semantic_mismatch(tmp_path: Path) -> None:
    artifact_names = (
        "dominance_selection.json",
        "dominance_candidate_metrics.json",
        "anchored_temporal_v7_dominance.yaml",
    )
    for name in artifact_names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    selected = {"panda": 50, "ur5e": 50}
    seal = {
        "protocol": dominance_pilot.PROTOCOL,
        "selected_reanchor_interval": {"panda": 20, "ur5e": 50},
        "dominance_rule": "S_hard is a subset of S_anchor",
        "policy_validation_npz_semantically_opened_before_seal": False,
        "policy_validation_outcomes_computed_before_seal": False,
        "selection_solver_call_count": 0,
        "calibration_outcome_collection_solver_call_count": 0,
        "trajectory_generator_call_count": 0,
        "selection_report": _descriptor(tmp_path / artifact_names[0], root=tmp_path),
        "candidate_metrics": _descriptor(tmp_path / artifact_names[1], root=tmp_path),
        "protocol_config": _descriptor(tmp_path / artifact_names[2], root=tmp_path),
    }
    seal_path = tmp_path / "selection_seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="semantic contract changed"):
        dominance_pilot._verify_selection_seal(
            tmp_path,
            expected_sha256=sha256(seal_path.read_bytes()).hexdigest(),
            selected_r=selected,
            workspace=tmp_path,
            frozen_v7_root=tmp_path,
            expected_release_inputs={},
        )
