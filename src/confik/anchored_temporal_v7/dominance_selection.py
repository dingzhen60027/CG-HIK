"""Calibration-only selector for the Anchored Temporal V7 dominance holdout.

This module deliberately has no policy-validation loader.  It accepts only the
frozen ``*_calibration_records.npz`` artifact produced by the original V7 run,
checks that artifact against its recorded SHA-256 descriptor, and selects the
re-anchor interval using the preregistered trajectory-dominance ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


R_VALUES = (20, 25, 30, 40, 50)
ARM_NAMES = ("always_hard",) + tuple(f"r{value}" for value in R_VALUES)
CALIBRATION_ROLE = "anchored_trajectory_calibration"
TRAJECTORIES = 40
FRAMES_PER_TRAJECTORY = 150
FRAME_COUNT = TRAJECTORIES * FRAMES_PER_TRAJECTORY
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_NPZ_FIELDS = frozenset(
    {
        "arm_names",
        "reanchor_interval",
        "latency_ns",
        "accepted",
        "function_evaluations",
        "seed_invoked",
        "local_attempted",
        "local_accepted",
        "hard_attempted",
        "hard_accepted",
        "same_frame_hard_recovery_attempted",
        "same_frame_hard_recovered",
        "occupancy_mode",
        "state_before",
        "state_after",
        "hard_count_before",
        "hard_count_after",
        "local_streak_before",
        "local_streak_after",
        "mode_switched",
        "switch_kind",
        "anchor_scheduled",
        "anchor_attempted",
        "anchor_accepted",
        "anchor_kind",
        "local_probe",
        "route",
        "executed_stages",
        "command_q",
        "executed_query_hash",
        "method_order_position",
        "stage_names",
        "stage_latency_ns",
        "source_query_hash",
        "trajectory_uid",
        "trajectory_order",
        "category",
        "time_index",
        "phase",
        "phase_index",
        "transition_boundary",
    }
)

_ARM_FRAME_BOOL_FIELDS = (
    "accepted",
    "seed_invoked",
    "local_attempted",
    "local_accepted",
    "hard_attempted",
    "hard_accepted",
    "same_frame_hard_recovery_attempted",
    "same_frame_hard_recovered",
    "mode_switched",
    "anchor_scheduled",
    "anchor_attempted",
    "anchor_accepted",
    "local_probe",
)
_ARM_FRAME_STRING_DTYPES = {
    "switch_kind": np.dtype("<U32"),
    "anchor_kind": np.dtype("<U32"),
    "route": np.dtype("<U64"),
    "executed_stages": np.dtype("<U128"),
    "executed_query_hash": np.dtype("<U64"),
}
_ARM_FRAME_INTEGER_DTYPES = {
    "latency_ns": np.dtype("int64"),
    "function_evaluations": np.dtype("int64"),
    "occupancy_mode": np.dtype("int8"),
    "state_before": np.dtype("int8"),
    "state_after": np.dtype("int8"),
    "hard_count_before": np.dtype("int16"),
    "hard_count_after": np.dtype("int16"),
    "local_streak_before": np.dtype("int16"),
    "local_streak_after": np.dtype("int16"),
    "method_order_position": np.dtype("int8"),
}


@dataclass(frozen=True)
class FrozenCalibrationRecords:
    """Validated selection fields from one immutable V7 calibration artifact."""

    robot: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    arm_names: tuple[str, ...]
    reanchor_interval: np.ndarray
    latency_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    trajectory_uid: np.ndarray
    trajectory_order: tuple[str, ...]
    category: np.ndarray
    time_index: np.ndarray


class NoDominanceEligibleCandidate(RuntimeError):
    """No anchored candidate retained every hard-completed trajectory."""

    def __init__(self, robot: str, report: Mapping[str, Any]) -> None:
        super().__init__(
            f"{robot}: no R satisfies always-hard trajectory-set dominance"
        )
        self.robot = robot
        self.report = dict(report)


def _validate_robot(robot: str) -> None:
    if robot not in {"panda", "ur5e"}:
        raise ValueError("robot must be panda or ur5e")


def _validate_hex_digest(value: str, *, name: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _require_array(
    payload: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    value = np.asarray(payload[name])
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}, expected {dtype}")
    return value


def _validate_artifact_descriptor(
    path: Path, expected_artifact: Mapping[str, Any]
) -> tuple[int, str]:
    if set(expected_artifact) != {"path", "sha256", "size"}:
        raise ValueError("calibration artifact descriptor schema changed")
    if str(expected_artifact["path"]) != path.name:
        raise ValueError("calibration artifact descriptor path does not match")
    try:
        expected_size = int(expected_artifact["size"])
    except (TypeError, ValueError) as error:
        raise ValueError("calibration artifact size is invalid") from error
    if isinstance(expected_artifact["size"], bool) or expected_size <= 0:
        raise ValueError("calibration artifact size is invalid")
    expected_sha256 = str(expected_artifact["sha256"])
    _validate_hex_digest(expected_sha256, name="expected artifact sha256")
    return expected_size, expected_sha256


def _validate_calibration_path(path: Path, *, robot: str) -> None:
    lowered_parts = tuple(part.casefold() for part in path.parts)
    if any(
        forbidden in part
        for part in lowered_parts
        for forbidden in ("policy_validation", "test_v3", "test_v4")
    ):
        raise ValueError("selector may open calibration records only")
    if path.name != f"{robot}_calibration_records.npz":
        raise ValueError("unexpected V7 calibration-record filename")
    if path.is_symlink():
        raise ValueError("calibration record must not be a symlink")
    if not path.is_file():
        raise FileNotFoundError(path)


def _validate_npz_schema(
    payload: Mapping[str, np.ndarray], *, robot: str
) -> None:
    fields = set(payload)
    if fields != _NPZ_FIELDS:
        missing = sorted(_NPZ_FIELDS - fields)
        extra = sorted(fields - _NPZ_FIELDS)
        raise ValueError(
            f"calibration record schema changed: missing={missing}, extra={extra}"
        )

    arms = len(ARM_NAMES)
    arm_shape = (arms, FRAME_COUNT)
    _require_array(
        payload, "arm_names", shape=(arms,), dtype=np.dtype("<U32")
    )
    _require_array(
        payload, "reanchor_interval", shape=(arms,), dtype=np.dtype("int16")
    )
    for name in _ARM_FRAME_BOOL_FIELDS:
        _require_array(payload, name, shape=arm_shape, dtype=np.dtype("bool"))
    for name, dtype in _ARM_FRAME_INTEGER_DTYPES.items():
        _require_array(payload, name, shape=arm_shape, dtype=dtype)
    for name, dtype in _ARM_FRAME_STRING_DTYPES.items():
        _require_array(payload, name, shape=arm_shape, dtype=dtype)

    nq = 7 if robot == "panda" else 6
    _require_array(
        payload,
        "command_q",
        shape=(arms, FRAME_COUNT, nq),
        dtype=np.dtype("float64"),
    )
    _require_array(
        payload,
        "stage_names",
        shape=(4,),
        dtype=np.dtype("<U32"),
    )
    _require_array(
        payload,
        "stage_latency_ns",
        shape=(arms, FRAME_COUNT, 4),
        dtype=np.dtype("int64"),
    )
    for name, dtype in {
        "source_query_hash": np.dtype("<U64"),
        "trajectory_uid": np.dtype("<U64"),
        "category": np.dtype("<U35"),
        "phase": np.dtype("<U32"),
    }.items():
        _require_array(payload, name, shape=(FRAME_COUNT,), dtype=dtype)
    _require_array(
        payload,
        "trajectory_order",
        shape=(TRAJECTORIES,),
        dtype=np.dtype("<U64"),
    )
    for name in ("time_index", "phase_index"):
        _require_array(
            payload, name, shape=(FRAME_COUNT,), dtype=np.dtype("int64")
        )
    _require_array(
        payload,
        "transition_boundary",
        shape=(FRAME_COUNT,),
        dtype=np.dtype("bool"),
    )


def _validate_npz_semantics(payload: Mapping[str, np.ndarray]) -> None:
    arm_names = tuple(np.asarray(payload["arm_names"]).astype(str).tolist())
    if arm_names != ARM_NAMES or len(set(arm_names)) != len(arm_names):
        raise ValueError("calibration arm names/order changed or contain duplicates")
    intervals = tuple(
        int(value) for value in np.asarray(payload["reanchor_interval"]).tolist()
    )
    if intervals != (-1,) + R_VALUES or len(set(intervals)) != len(intervals):
        raise ValueError("calibration R grid/order changed or contains duplicates")
    if tuple(np.asarray(payload["stage_names"]).astype(str).tolist()) != (
        "state_policy",
        "local",
        "hard",
        "unattributed",
    ):
        raise ValueError("calibration stage schema changed")

    latency = np.asarray(payload["latency_ns"])
    stage_latency = np.asarray(payload["stage_latency_ns"])
    if np.any(latency <= 0) or np.any(stage_latency < 0):
        raise ValueError("latencies must be positive with nonnegative stages")
    if not np.array_equal(np.sum(stage_latency, axis=2), latency):
        raise ValueError("stage latencies do not close to total latency")
    if np.any(np.asarray(payload["function_evaluations"]) < 0):
        raise ValueError("function evaluations must be nonnegative")
    expected_positions = np.arange(len(ARM_NAMES), dtype=np.int8)
    positions = np.sort(np.asarray(payload["method_order_position"]), axis=0)
    if not np.array_equal(
        positions,
        np.broadcast_to(expected_positions[:, None], positions.shape),
    ):
        raise ValueError("method order is not a complete per-frame permutation")

    order = np.asarray(payload["trajectory_order"]).astype(str)
    uids = np.asarray(payload["trajectory_uid"]).astype(str)
    if len(set(order.tolist())) != TRAJECTORIES:
        raise ValueError("trajectory_order contains duplicate UIDs")
    for name, values, require_unique in (
        ("trajectory_order", order, True),
        (
            "source_query_hash",
            np.asarray(payload["source_query_hash"]).astype(str),
            True,
        ),
        (
            "executed_query_hash",
            np.asarray(payload["executed_query_hash"]).astype(str).ravel(),
            False,
        ),
    ):
        if any(_SHA256_PATTERN.fullmatch(value) is None for value in values):
            raise ValueError(f"{name} contains a malformed digest")
        if require_unique and len(set(values.tolist())) != len(values):
            raise ValueError(f"{name} contains duplicate hashes")

    time_index = np.asarray(payload["time_index"])
    category = np.asarray(payload["category"]).astype(str)
    for trajectory_index, uid in enumerate(order):
        start = trajectory_index * FRAMES_PER_TRAJECTORY
        stop = start + FRAMES_PER_TRAJECTORY
        if not np.all(uids[start:stop] == uid):
            raise ValueError("trajectory rows are not complete ordered UID blocks")
        if not np.array_equal(
            time_index[start:stop], np.arange(FRAMES_PER_TRAJECTORY)
        ):
            raise ValueError("trajectory time indices are not exactly 0..149")
        if len(set(category[start:stop].tolist())) != 1 or not category[start]:
            raise ValueError("trajectory category must be nonempty and constant")
    if set(uids.tolist()) != set(order.tolist()):
        raise ValueError("trajectory UID rows do not match trajectory_order")


def load_frozen_v7_calibration_records(
    path: str | Path,
    *,
    robot: str,
    expected_artifact: Mapping[str, Any],
) -> FrozenCalibrationRecords:
    """Load exactly one SHA-pinned V7 calibration record, never PV outcomes.

    ``expected_artifact`` is the three-field descriptor from the immutable
    original V7 ``run_manifest.json``.  Its SHA and byte size are checked on an
    in-memory snapshot before NumPy parses the archive.
    """

    _validate_robot(robot)
    source = Path(path)
    _validate_calibration_path(source, robot=robot)
    expected_size, expected_sha256 = _validate_artifact_descriptor(
        source, expected_artifact
    )
    raw = source.read_bytes()
    actual_sha256 = sha256(raw).hexdigest()
    if len(raw) != expected_size or actual_sha256 != expected_sha256:
        raise ValueError("frozen V7 calibration artifact hash/size mismatch")

    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("invalid frozen V7 calibration NPZ") from error
    _validate_npz_schema(payload, robot=robot)
    _validate_npz_semantics(payload)

    records = FrozenCalibrationRecords(
        robot=robot,
        source_path=str(source.resolve()),
        source_sha256=actual_sha256,
        source_size_bytes=len(raw),
        arm_names=tuple(payload["arm_names"].astype(str).tolist()),
        reanchor_interval=_readonly_copy(payload["reanchor_interval"]),
        latency_ns=_readonly_copy(payload["latency_ns"]),
        accepted=_readonly_copy(payload["accepted"]),
        function_evaluations=_readonly_copy(payload["function_evaluations"]),
        seed_invoked=_readonly_copy(payload["seed_invoked"]),
        trajectory_uid=_readonly_copy(payload["trajectory_uid"].astype(str)),
        trajectory_order=tuple(payload["trajectory_order"].astype(str).tolist()),
        category=_readonly_copy(payload["category"].astype(str)),
        time_index=_readonly_copy(payload["time_index"]),
    )
    _validate_selection_records(records)
    return records


def _validate_selection_records(records: FrozenCalibrationRecords) -> None:
    _validate_robot(records.robot)
    _validate_hex_digest(records.source_sha256, name="source sha256")
    if records.source_size_bytes <= 0:
        raise ValueError("source size must be positive")
    if records.arm_names != ARM_NAMES or len(set(records.arm_names)) != len(ARM_NAMES):
        raise ValueError("selection arm names/order changed or contain duplicates")
    intervals = tuple(int(value) for value in np.asarray(records.reanchor_interval))
    if intervals != (-1,) + R_VALUES or len(set(intervals)) != len(intervals):
        raise ValueError("selection R grid/order changed or contains duplicates")
    arms = len(ARM_NAMES)
    shape = (arms, FRAME_COUNT)
    for name, values in (
        ("latency_ns", records.latency_ns),
        ("accepted", records.accepted),
        ("function_evaluations", records.function_evaluations),
        ("seed_invoked", records.seed_invoked),
    ):
        if np.asarray(values).shape != shape:
            raise ValueError(f"{name} does not have the frozen 6x6000 shape")
    if np.asarray(records.accepted).dtype != np.dtype("bool"):
        raise ValueError("accepted must be Boolean")
    if np.asarray(records.seed_invoked).dtype != np.dtype("bool"):
        raise ValueError("seed_invoked must be Boolean")
    if not np.issubdtype(np.asarray(records.latency_ns).dtype, np.integer):
        raise ValueError("latency_ns must be integral")
    if not np.issubdtype(
        np.asarray(records.function_evaluations).dtype, np.integer
    ):
        raise ValueError("function_evaluations must be integral")
    if np.any(np.asarray(records.latency_ns) <= 0):
        raise ValueError("latency_ns must be positive")
    if np.any(np.asarray(records.function_evaluations) < 0):
        raise ValueError("function_evaluations must be nonnegative")
    if len(records.trajectory_order) != TRAJECTORIES or len(
        set(records.trajectory_order)
    ) != TRAJECTORIES:
        raise ValueError("trajectory order must contain 40 unique UIDs")
    if any(
        _SHA256_PATTERN.fullmatch(uid) is None for uid in records.trajectory_order
    ):
        raise ValueError("trajectory order contains malformed UIDs")
    uids = np.asarray(records.trajectory_uid).astype(str)
    categories = np.asarray(records.category).astype(str)
    times = np.asarray(records.time_index)
    if uids.shape != (FRAME_COUNT,) or categories.shape != (FRAME_COUNT,):
        raise ValueError("trajectory metadata does not contain 6000 frames")
    if times.shape != (FRAME_COUNT,):
        raise ValueError("time_index does not contain 6000 frames")
    for index, uid in enumerate(records.trajectory_order):
        start = index * FRAMES_PER_TRAJECTORY
        stop = start + FRAMES_PER_TRAJECTORY
        if not np.all(uids[start:stop] == uid):
            raise ValueError("trajectory UID rows are incomplete or out of order")
        if not np.array_equal(times[start:stop], np.arange(FRAMES_PER_TRAJECTORY)):
            raise ValueError("each trajectory must contain exactly frames 0..149")
        if len(set(categories[start:stop].tolist())) != 1 or not categories[start]:
            raise ValueError("trajectory category must be nonempty and constant")


def _trajectory_completion(records: FrozenCalibrationRecords) -> np.ndarray:
    accepted = np.asarray(records.accepted, dtype=bool)
    return np.all(
        accepted.reshape(len(ARM_NAMES), TRAJECTORIES, FRAMES_PER_TRAJECTORY),
        axis=2,
    )


def _latency_summary(
    latency_ns: np.ndarray, records: FrozenCalibrationRecords
) -> dict[str, Any]:
    values = np.asarray(latency_ns, dtype=np.int64)
    trajectory_ns = np.sum(
        values.reshape(TRAJECTORIES, FRAMES_PER_TRAJECTORY),
        axis=1,
        dtype=np.int64,
    )
    aggregate_ns = sum(int(value) for value in trajectory_ns)
    return {
        "aggregate_cumulative_latency_ns": aggregate_ns,
        "aggregate_cumulative_latency_ms": aggregate_ns / 1e6,
        "trajectory_cumulative_latency_mean_ms": float(np.mean(trajectory_ns) / 1e6),
        "trajectory_cumulative_latency_median_ms": float(
            np.median(trajectory_ns) / 1e6
        ),
        "trajectory_cumulative_latency_p95_ms": float(
            np.quantile(trajectory_ns, 0.95) / 1e6
        ),
        "trajectory_cumulative_latency_ns_by_uid": {
            uid: int(value)
            for uid, value in zip(
                records.trajectory_order, trajectory_ns.tolist(), strict=True
            )
        },
        "p50_latency_ms": float(np.quantile(values, 0.50) / 1e6),
        "p95_latency_ms": float(np.quantile(values, 0.95) / 1e6),
        "p99_latency_ms": float(np.quantile(values, 0.99) / 1e6),
    }


def _arm_summary(
    records: FrozenCalibrationRecords,
    *,
    arm: int,
    completion: np.ndarray,
    hard_completion: np.ndarray,
) -> dict[str, Any]:
    vector = completion[arm]
    completed = [
        uid
        for uid, ok in zip(records.trajectory_order, vector, strict=True)
        if ok
    ]
    hard_set = {
        uid
        for uid, ok in zip(records.trajectory_order, hard_completion, strict=True)
        if ok
    }
    candidate_set = set(completed)
    count = len(completed)
    summary = {
        "arm_index": arm,
        "arm_name": records.arm_names[arm],
        "reanchor_interval": (
            None if arm == 0 else int(records.reanchor_interval[arm])
        ),
        "eligible": hard_set.issubset(candidate_set),
        "whole_trajectory_completion_count": count,
        "whole_trajectory_completion_rate": count / TRAJECTORIES,
        "completion_vector": vector.tolist(),
        "completion_trajectory_uids": completed,
        "lost_trajectory_uids": [
            uid
            for uid in records.trajectory_order
            if uid in hard_set and uid not in candidate_set
        ],
        "gained_trajectory_uids": [
            uid
            for uid in records.trajectory_order
            if uid in candidate_set and uid not in hard_set
        ],
        "frame_verified_success": float(np.mean(records.accepted[arm])),
        "learned_seed_invocation_rate": float(np.mean(records.seed_invoked[arm])),
        "mean_fev": float(np.mean(records.function_evaluations[arm])),
        **_latency_summary(records.latency_ns[arm], records),
    }
    return summary


def select_dominance_reanchor_interval(
    records: FrozenCalibrationRecords,
) -> tuple[int, dict[str, Any]]:
    """Select R from frozen calibration under trajectory-set dominance.

    Eligible candidates retain every trajectory completed by always-hard.
    Eligible candidates are then ordered by maximum completion count, exact
    total latency over all 40x150 frames, frame P99, seed rate, mean FEV, and R.
    No policy-validation value is accepted by this API.
    """

    _validate_selection_records(records)
    completion = _trajectory_completion(records)
    hard = _arm_summary(
        records,
        arm=0,
        completion=completion,
        hard_completion=completion[0],
    )
    candidates = [
        _arm_summary(
            records,
            arm=arm,
            completion=completion,
            hard_completion=completion[0],
        )
        for arm in range(1, len(ARM_NAMES))
    ]
    eligible = [row for row in candidates if row["eligible"]]
    report: dict[str, Any] = {
        "robot": records.robot,
        "selection_role": CALIBRATION_ROLE,
        "source_calibration_artifact": {
            "path": records.source_path,
            "sha256": records.source_sha256,
            "size": records.source_size_bytes,
        },
        "policy_validation_outcomes_opened": False,
        "policy_validation_used_for_selection": False,
        "completion_eligibility_definition": (
            "S_hard is a subset of S_anchor; lost_trajectory_uids must be empty"
        ),
        "trajectory_uid_order": list(records.trajectory_order),
        "objective_order": [
            "empty_lost_trajectory_uids",
            "maximum_whole_trajectory_completion_count",
            "minimum_aggregate_cumulative_latency_ns_over_40x150_frames",
            "minimum_frame_p99_latency",
            "minimum_learned_seed_invocation_rate",
            "minimum_mean_fev",
            "minimum_reanchor_interval",
        ],
        "always_hard": hard,
        "candidate_metrics": candidates,
        "eligible_candidate_count": len(eligible),
    }
    if not eligible:
        report["selected"] = None
        raise NoDominanceEligibleCandidate(records.robot, report)

    selected = min(
        eligible,
        key=lambda row: (
            -int(row["whole_trajectory_completion_count"]),
            int(row["aggregate_cumulative_latency_ns"]),
            float(row["p99_latency_ms"]),
            float(row["learned_seed_invocation_rate"]),
            float(row["mean_fev"]),
            int(row["reanchor_interval"]),
        ),
    )
    selected_r = int(selected["reanchor_interval"])
    report["selected"] = dict(selected)
    report["selection_sort_key"] = {
        "negated_completion_count": -int(
            selected["whole_trajectory_completion_count"]
        ),
        "aggregate_cumulative_latency_ns": int(
            selected["aggregate_cumulative_latency_ns"]
        ),
        "frame_p99_latency_ms": float(selected["p99_latency_ms"]),
        "learned_seed_invocation_rate": float(
            selected["learned_seed_invocation_rate"]
        ),
        "mean_fev": float(selected["mean_fev"]),
        "reanchor_interval": selected_r,
    }
    return selected_r, report


__all__ = [
    "ARM_NAMES",
    "CALIBRATION_ROLE",
    "FRAME_COUNT",
    "FRAMES_PER_TRAJECTORY",
    "FrozenCalibrationRecords",
    "NoDominanceEligibleCandidate",
    "R_VALUES",
    "TRAJECTORIES",
    "load_frozen_v7_calibration_records",
    "select_dominance_reanchor_interval",
]
