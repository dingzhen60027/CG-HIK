"""One-shot final evaluation of sealed counterfactual CG-HIK V4.

The fresh workload is identified and sealed before any outcome is computed.
There is deliberately no smoke, calibration, resume, or rerun entry point.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..experiments.provenance import environment_payload
from .benchmark import (
    METHODS,
    BenchmarkData,
    benchmark_trajectories,
    build_frozen_methods,
    verify_frozen_releases,
    warmup_methods,
)
from .data import (
    DT,
    EXPECTED_FRAMES,
    EXPECTED_TRAJECTORIES,
    FAMILIES,
    FROZEN_POOL_SEEDS,
    ROLE,
    FreshTransitionDataset,
    FreshTransitionSpec,
    audit_freshness,
    build_identity_manifest,
    build_prior_identity_registries,
    generate_fresh_transition_dataset,
    load_fresh_dataset,
    load_identity_manifest,
    save_fresh_dataset,
)
from .reporting import (
    completion_identity,
    family_rows,
    final_gate,
    main_markdown,
    main_rows,
    trajectory_rows,
)


PROTOCOL = "fresh_transition_v4_final_evaluation_v1"
ROBOTS = ("panda", "ur5e")
OUTPUT_NAME = "fresh_transition_v4_test"
EXPECTED_RELEASE_V3_MANIFEST = (
    "9cd72b1fa1abfaa5e8e8e3f769987996dec0877b35be7fda58eddd813bbe05d6"
)
EXPECTED_RELEASE_V4_DIGEST = (
    "01bd104c96b6e33be1a5ac29dd2bbedad1369156a7e9beb86f551eabe9db541a"
)
EXPECTED_RELEASE_V4_MANIFEST = (
    "816f6fa7fba2a015eb0f3dd146f76d469dd3ef13d8e7440b8381fcdcc0efee10"
)
EXPECTED_RELEASE_V4_ARTIFACT_MANIFEST = (
    "411921ca349bfbcff5c7e7c8c372d750ca9df499677c0d15f84dc4a98155e288"
)
EXPECTED_SOURCE_CONFIG = (
    "c579c282ac3b19e10835b2dae1dc9820c3df195daa0c4e446d49c92a323df70a"
)
EXPECTED_TEST_V3_IDENTITY_MANIFEST = (
    "d16a21fed8c22fc34f548d3c4d403cbc8e5aa3671d0cb3a05cace1e2375be09b"
)
EXPECTED_TEST_V4_IDENTITY_MANIFEST = (
    "77fa670e5de1bf40301f0bcd0292ebcc7860f412ac8bd946a485dc782458d51e"
)
EXPECTED_PROTECTED_RELATIVE = (
    "outputs/anchored_temporal_v7_dominance_pilot",
    "outputs/temporal_event_v6_pilot",
    "outputs/anchored_temporal_v7_pilot",
    "outputs/temporal_v6_pilot",
    "outputs/hierarchical_v5_pilot",
    "outputs/hierarchical_v5_lite_pilot",
    "outputs/release_v3_locked",
    "outputs/release_v4_locked",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    path = path.resolve()
    root = relative_to.resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"artifact is missing or a symlink: {path}")
    try:
        shown = path.relative_to(root)
    except ValueError:
        shown = path
    return {
        "path": str(shown),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _verify_local_artifact(
    root: Path, descriptor: Mapping[str, Any], *, label: str
) -> None:
    """Verify one hash-bound artifact that must remain inside ``root``."""

    if set(descriptor) != {"path", "sha256", "size"}:
        raise RuntimeError(f"{label} artifact descriptor schema changed")
    relative = Path(str(descriptor["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} artifact path escapes the sealed namespace")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} artifact path escapes the sealed namespace") from exc
    actual = _artifact(candidate, relative_to=root)
    if actual != dict(descriptor):
        raise RuntimeError(f"{label} artifact changed after preregistration")


def _verify_sealed_stage_inputs(
    staging: Path, seal: Mapping[str, Any]
) -> None:
    """Recheck every staging-local input bound by the preregistration seal."""

    datasets = seal.get("fresh_dataset_artifacts", {})
    identities = seal.get("identity_manifest_artifacts", {})
    if set(datasets) != set(ROBOTS) or set(identities) != set(ROBOTS):
        raise RuntimeError("preregistration seal does not bind both robot inputs")
    for robot in ROBOTS:
        _verify_local_artifact(
            staging, datasets[robot], label=f"{robot} fresh dataset"
        )
        _verify_local_artifact(
            staging, identities[robot], label=f"{robot} identity manifest"
        )
    _verify_local_artifact(
        staging, seal.get("freshness_audit", {}), label="freshness audit"
    )
    _verify_local_artifact(
        staging, seal.get("protocol_config", {}), label="effective protocol config"
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(_safe_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        for row in values:
            serialized = {
                key: (
                    json.dumps(_safe_json(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else _safe_json(value)
                )
                for key, value in row.items()
            }
            writer.writerow(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _save_npz_exclusive(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def _tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"protected root is unavailable: {root}")
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"protected tree contains symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(root))
            snapshot[relative] = {
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
    if not snapshot:
        raise RuntimeError(f"protected root is empty: {root}")
    return snapshot


def _snapshot_digest(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    digest = sha256()
    for relative, descriptor in sorted(snapshot.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(descriptor["size"])).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(descriptor["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _digest_strings(values: Sequence[str], *, ordered: bool) -> str:
    sequence = list(values) if ordered else sorted(set(values))
    digest = sha256()
    domain = b"ordered" if ordered else b"set"
    digest.update(len(b"domain").to_bytes(4, "little"))
    digest.update(b"domain")
    digest.update(len(domain).to_bytes(8, "little"))
    digest.update(domain)
    for value in sequence:
        name = b"value"
        encoded = str(value).encode("ascii")
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def identity_anchors(fresh: FreshTransitionDataset) -> dict[str, str]:
    """Return the four control-plane anchors committed in the YAML."""

    return {
        "ordered_query_sha256_digest": _digest_strings(
            fresh.formal_query_sha256.astype(str).tolist(), ordered=True
        ),
        "query_sha256_set_digest": _digest_strings(
            fresh.formal_query_sha256.astype(str).tolist(), ordered=False
        ),
        "ordered_trajectory_uid_digest": _digest_strings(
            list(fresh.trajectory_order), ordered=True
        ),
        "trajectory_uid_set_digest": _digest_strings(
            list(fresh.trajectory_order), ordered=False
        ),
    }


def validate_config(config: Mapping[str, Any], *, workspace: Path) -> None:
    """Fail closed on every scientific degree of freedom in the final test."""

    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected fresh-transition V4 protocol")
    if tuple(config.get("robots", ())) != ROBOTS:
        raise ValueError("the final evaluation requires Panda and UR5e")
    if int(config.get("release_seed", -1)) != 17:
        raise ValueError("the sealed V4 lineage must use release seed 17")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"method registry must be exactly {METHODS}")
    scope = config.get("evaluation_scope", {})
    if (
        scope.get("final_main_method") != "counterfactual_cghik_v4"
        or tuple(scope.get("development_only_routes_retired", ()))
        != ("v5", "v5_lite", "temporal_v6", "anchored_temporal_v7")
        or any(
            scope.get(key) is not expected
            for key, expected in (
                ("v8_v9_forbidden", True),
                ("fresh_results_used_for_tuning", False),
                ("exactly_one_retained_run_per_robot", True),
                ("resume_allowed", False),
                ("scientific_rerun_allowed", False),
            )
        )
    ):
        raise ValueError("final evaluation scope changed")
    release = config.get("sealed_release", {})
    exact_release = {
        "release_v3_manifest_sha256": EXPECTED_RELEASE_V3_MANIFEST,
        "release_v4_digest": EXPECTED_RELEASE_V4_DIGEST,
        "release_v4_manifest_sha256": EXPECTED_RELEASE_V4_MANIFEST,
        "release_v4_artifact_manifest_sha256": EXPECTED_RELEASE_V4_ARTIFACT_MANIFEST,
        "source_config_sha256": EXPECTED_SOURCE_CONFIG,
    }
    if any(str(release.get(key, "")) != value for key, value in exact_release.items()):
        raise ValueError("sealed release anchors changed")
    for key in (
        "retraining_forbidden", "policy_selection_forbidden",
        "threshold_changes_forbidden", "solver_budget_changes_forbidden",
        "fallback_changes_forbidden", "verifier_changes_forbidden",
    ):
        if release.get(key) is not True:
            raise ValueError(f"missing release prohibition: {key}")
    data = config.get("fresh_data", {})
    if (
        data.get("role") != ROLE
        or data.get("split_unit") != "complete_trajectory"
        or int(data.get("trajectories_per_robot", -1)) != EXPECTED_TRAJECTORIES
        or int(data.get("trajectories_per_family", -1)) != 20
        or int(data.get("frames_per_trajectory", -1)) != 150
        or float(data.get("dt", np.nan)) != DT
        or tuple(data.get("families", ())) != FAMILIES
        or dict(data.get("pool_seed", {})) != FROZEN_POOL_SEEDS
        or data.get("identity_frozen_before_solver_calls") is not True
    ):
        raise ValueError("fresh trajectory contract changed")
    identity = data.get("frozen_identity", {})
    expected_anchor_keys = {
        "ordered_query_sha256_digest", "query_sha256_set_digest",
        "ordered_trajectory_uid_digest", "trajectory_uid_set_digest",
    }
    for robot in ROBOTS:
        anchors = identity.get(robot, {})
        if set(anchors) != expected_anchor_keys or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in anchors.values()
        ):
            raise ValueError(f"{robot} fresh identity is not frozen in the config")
    freshness = config.get("freshness", {})
    for key in (
        "require_query_hash_isolation", "require_trajectory_uid_isolation",
        "require_seed_isolation", "require_internal_unique_query_hashes",
        "require_internal_unique_trajectory_uids",
        "compare_all_v6_v7_calibration_and_policy_validation_roles",
        "compare_old_formal_identity_arrays_only",
    ):
        if freshness.get(key) is not True:
            raise ValueError(f"freshness contract missing {key}")
    if (
        freshness.get("formal_performance_results_read") is not False
        or freshness.get("old_test_v3_dataset_manifest_sha256")
        != EXPECTED_TEST_V3_IDENTITY_MANIFEST
        or freshness.get("old_test_v4_dataset_manifest_sha256")
        != EXPECTED_TEST_V4_IDENTITY_MANIFEST
    ):
        raise ValueError("formal identity-only boundary changed")
    timing = config.get("timing", {})
    if (
        timing.get("clock") != "perf_counter_ns"
        or int(timing.get("warmup_frames", -1)) != 200
        or int(timing.get("trajectory_repeats", -1)) != 1
        or timing.get("method_order") != "seeded_three_by_three_latin_blocks"
        or dict(timing.get("order_seed", {})) != {"panda": 940901, "ur5e": 940902}
        or timing.get("synchronize_cuda") is not True
        or timing.get("disk_io_inside_timed_interval") is not False
        or timing.get("logging_serialization_inside_timed_interval") is not False
        or tuple(float(value) for value in timing.get("quantiles", ()))
        != (0.5, 0.95, 0.99)
    ):
        raise ValueError("timing contract changed")
    runtime = config.get("runtime", {})
    if (
        runtime.get("device") != "cuda:0"
        or runtime.get("cuda_visible_devices") != "0"
        or int(runtime.get("intra_op_threads", -1)) != 8
        or int(runtime.get("inter_op_threads", -1)) != 1
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise ValueError("runtime contract changed")
    gate = config.get("final_gate", {})
    if (
        gate.get("v4_completion_count_not_below_always_hard") is not True
        or float(gate.get("aggregate_cumulative_latency_ratio_max", np.nan)) != 0.85
        or float(gate.get("mean_fev_ratio_max", np.nan)) != 0.85
        or float(gate.get("p95_latency_ratio_max", np.nan)) != 1.0
        or float(gate.get("p99_latency_ratio_max", np.nan)) != 1.05
        or int(gate.get("accepted_contract_violation_count_max", -1)) != 0
        or gate.get("accepted_contract_violation_scope") != "all_three_methods"
        or gate.get("p50_is_gate") is not False
    ):
        raise ValueError("final gate changed")
    reporting = config.get("reporting", {})
    if reporting.get("accepted_contract_violation_scope") != "all_three_methods":
        raise ValueError("accepted contract reporting scope changed")
    expected_paths = tuple((workspace / value).resolve() for value in EXPECTED_PROTECTED_RELATIVE)
    actual_paths = tuple(
        resolve_path(dict(config), value) for value in config.get("protected_outputs", ())
    )
    if actual_paths != expected_paths:
        raise ValueError("protected output roots changed")
    expected_resolved = {
        "source_config": workspace / "configs" / "paper_v2.yaml",
        "release_v3_root": workspace / "outputs" / "release_v3_locked",
        "release_v4_root": workspace / "outputs" / "release_v4_locked",
        "output_root": workspace / "outputs" / OUTPUT_NAME,
    }
    for key, expected in expected_resolved.items():
        if resolve_path(dict(config), str(config[key])) != expected.resolve():
            raise ValueError(f"{key} must resolve to {expected}")


def _kinematics_identity(source_config: Mapping[str, Any], robot: str) -> tuple[str, Path]:
    config = dict(source_config)
    path = resolve_path(config, str(config["robots"][robot]["urdf"]))
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"URDF identity file unavailable: {path}")
    return _sha256_file(path), path


def _load_warmup_role(path: Path) -> object:
    """Load only an old development trajectory's query arrays for warmup."""

    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"warmup role unavailable: {path}")
    allowed = {
        "previous_q", "target_position", "target_rotation", "reference_q",
        "category", "expected_reachable", "continuity_feasible",
        "trajectory_id", "time_index", "dt",
    }
    required = allowed - {"dt"}
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise RuntimeError("warmup development role has an unexpected schema")
        values = {name: np.asarray(archive[name]).copy() for name in required}
        dt = float(np.asarray(archive["dt"]).item()) if "dt" in archive.files else DT
    dataset = QueryDataset(**values)
    return SimpleNamespace(dataset=dataset, dt=dt)


def _verify_release_anchors(
    *, workspace: Path, release_v3_root: Path, release_v4_root: Path,
    source_config_path: Path,
) -> dict[str, Any]:
    if _sha256_file(source_config_path) != EXPECTED_SOURCE_CONFIG:
        raise RuntimeError("paper_v2 solver/verifier configuration changed")
    if _sha256_file(
        workspace / "outputs" / "test_v3_aggregate" / "test_v3_dataset_manifest.json"
    ) != EXPECTED_TEST_V3_IDENTITY_MANIFEST:
        raise RuntimeError("old test_v3 identity manifest changed")
    if _sha256_file(
        workspace / "outputs" / ".test_v4_aggregate.incomplete"
        / "test_v4_dataset_manifest.json"
    ) != EXPECTED_TEST_V4_IDENTITY_MANIFEST:
        raise RuntimeError("old test_v4 identity manifest changed")
    if _sha256_file(release_v3_root / "release_manifest.json") != EXPECTED_RELEASE_V3_MANIFEST:
        raise RuntimeError("release_v3 manifest changed")
    if _sha256_file(release_v4_root / "release_manifest.json") != EXPECTED_RELEASE_V4_MANIFEST:
        raise RuntimeError("release_v4 manifest changed")
    if _sha256_file(release_v4_root / "artifact_manifest.json") != EXPECTED_RELEASE_V4_ARTIFACT_MANIFEST:
        raise RuntimeError("release_v4 artifact manifest changed")
    manifest = json.loads((release_v4_root / "release_manifest.json").read_text())
    if manifest.get("release_digest") != EXPECTED_RELEASE_V4_DIGEST:
        raise RuntimeError("release_v4 digest changed")
    return {
        robot: verify_frozen_releases(
            workspace=workspace,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=robot,
        )
        for robot in ROBOTS
    }


def _postrun_query_isolation(
    data: BenchmarkData, registries: Sequence[Any]
) -> dict[str, Any]:
    prior_formal = {
        str(value)
        for registry in registries
        for value in getattr(registry, "formal_query_hashes")
    }
    methods: dict[str, Any] = {}
    for column, method in enumerate(METHODS):
        current = set(np.asarray(data.executed_query_hash[:, column]).astype(str).tolist())
        overlap = sorted(current & prior_formal)
        if overlap:
            raise RuntimeError(
                f"{data.robot}/{method} executed queries overlap prior identities"
            )
        methods[method] = {
            "executed_query_count": EXPECTED_FRAMES,
            "unique_executed_query_count": len(current),
            "prior_formal_query_hash_overlap_count": 0,
        }
    return {
        "robot": data.robot,
        "comparison_scope": "method-specific closed-loop executed query hashes",
        "prior_identity_count": len(prior_formal),
        "methods": methods,
        "all_zero_overlap": True,
    }


def _global_identity_isolation(
    fresh_by_robot: Mapping[str, FreshTransitionDataset],
    registries_by_robot: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Check fresh identities against both robots and every prior registry.

    Query hashes and trajectory UIDs are content identities, so the isolation
    claim is global rather than silently scoped to a robot.  Numeric trajectory
    IDs are deliberately excluded because they are only robot-local indices;
    the protocol requires UID, query-hash, and seed isolation.
    """

    if set(fresh_by_robot) != set(ROBOTS) or set(registries_by_robot) != set(ROBOTS):
        raise RuntimeError("global identity audit requires both robots")
    all_registries = tuple(
        registry for robot in ROBOTS for registry in registries_by_robot[robot]
    )
    records: dict[str, Any] = {}
    fresh_sets: dict[str, dict[str, set[Any]]] = {}
    for robot in ROBOTS:
        fresh = fresh_by_robot[robot]
        sets = {
            "formal_query": set(fresh.formal_query_sha256.astype(str).tolist()),
            "runtime_query": set(fresh.source_query_hash.astype(str).tolist()),
            "trajectory_uid": set(fresh.trajectory_order),
            "seed": {int(fresh.pool_seed)}
            | set(int(value) for value in fresh.trajectory_seed.tolist()),
        }
        fresh_sets[robot] = sets
        prior_overlap = {
            "formal_query": sum(
                len(sets["formal_query"] & set(registry.formal_query_hashes))
                for registry in all_registries
            ),
            "runtime_query": sum(
                len(sets["runtime_query"] & set(registry.runtime_query_hashes))
                for registry in all_registries
            ),
            "trajectory_uid": sum(
                len(sets["trajectory_uid"] & set(registry.trajectory_uids))
                for registry in all_registries
            ),
            "seed": sum(
                len(sets["seed"] & set(int(value) for value in registry.seed_values))
                for registry in all_registries
            ),
        }
        if any(prior_overlap.values()):
            raise RuntimeError(f"{robot} fresh identities overlap a prior robot registry")
        records[robot] = {
            "fresh_counts": {name: len(values) for name, values in sets.items()},
            "prior_registry_count_compared": len(all_registries),
            "prior_overlap_counts": prior_overlap,
        }
    cross_robot_overlap = {
        name: len(fresh_sets["panda"][name] & fresh_sets["ur5e"][name])
        for name in ("formal_query", "runtime_query", "trajectory_uid", "seed")
    }
    if any(cross_robot_overlap.values()):
        raise RuntimeError("Panda and UR5e fresh identity lineages are not disjoint")
    return {
        "scope": "both_fresh_robots_against_both_robots_all_prior_registries",
        "prior_registry_count": len(all_registries),
        "robots": records,
        "cross_robot_fresh_overlap_counts": cross_robot_overlap,
        "all_zero_overlap": True,
    }


def _source_descriptors(workspace: Path, config_path: Path) -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("data.py").resolve(),
        Path(__file__).with_name("benchmark.py").resolve(),
        Path(__file__).with_name("reporting.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
        config_path.resolve(),
        (workspace / "scripts" / "run_fresh_transition_v4_test.sh").resolve(),
        (workspace / "tests" / "test_fresh_transition_v4_test.py").resolve(),
    )
    return {
        str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
        for path in paths
    }


def _verify_sources(workspace: Path, expected: Mapping[str, Mapping[str, Any]]) -> None:
    for relative, descriptor in expected.items():
        actual = _artifact(workspace / relative, relative_to=workspace)
        if actual != dict(descriptor):
            raise RuntimeError(f"implementation source changed during run: {relative}")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    config_path = Path(config["_config_path"]).resolve()
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    release_v3_root = resolve_path(config, str(config["release_v3_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    output_root = resolve_path(config, str(config["output_root"]))
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"overwrite forbidden: {output_root}")
    stale = sorted(output_root.parent.glob(f".{output_root.name}.incomplete.*"))
    if stale:
        raise FileExistsError(
            "stale one-shot staging requires audit: " + ", ".join(map(str, stale))
        )
    lock = output_root.with_name(f".{output_root.name}.lock")
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"one-shot lock already exists: {lock}")

    git_commit_start = _git(workspace, "rev-parse", "HEAD")
    git_status_start = _git(workspace, "status", "--short").splitlines()
    if git_status_start:
        raise RuntimeError("final fresh evaluation requires a clean committed worktree")
    release_inputs = _verify_release_anchors(
        workspace=workspace,
        release_v3_root=release_v3_root,
        release_v4_root=release_v4_root,
        source_config_path=source_config_path,
    )
    source_before = _source_descriptors(workspace, config_path)
    protected_roots = {
        relative: (workspace / relative).resolve() for relative in EXPECTED_PROTECTED_RELATIVE
    }
    protected_before = {name: _tree_snapshot(path) for name, path in protected_roots.items()}
    protected_digest_before = {
        name: _snapshot_digest(snapshot) for name, snapshot in protected_before.items()
    }

    # Generate identities and compare all explicitly enumerated prior identity
    # sources before the one-shot lock or any solver call is made.
    fresh_by_robot: dict[str, FreshTransitionDataset] = {}
    generation: dict[str, Any] = {}
    freshness: dict[str, Any] = {}
    registries_by_robot: dict[str, Sequence[Any]] = {}
    access_ledger: dict[str, Any] = {}
    urdf_descriptors: dict[str, Any] = {}
    for robot in ROBOTS:
        kinematics = load_robot(source_config, robot)
        identity, urdf_path = _kinematics_identity(source_config, robot)
        spec = FreshTransitionSpec.frozen(robot, kinematics_identity=identity)
        fresh, generated = generate_fresh_transition_dataset(kinematics, spec)
        anchors = identity_anchors(fresh)
        if anchors != dict(config["fresh_data"]["frozen_identity"][robot]):
            raise RuntimeError(f"{robot} generated identities differ from committed anchors")
        registries, ledger = build_prior_identity_registries(
            workspace, robot=robot, dt=DT, kinematics_identity=identity
        )
        audited = audit_freshness(fresh, registries, access_ledger=ledger)
        if audited.get("status") != "pass":
            raise RuntimeError(f"{robot} fresh identity isolation failed")
        fresh_by_robot[robot] = fresh
        generation[robot] = generated
        freshness[robot] = audited
        registries_by_robot[robot] = registries
        access_ledger[robot] = ledger
        urdf_descriptors[robot] = _artifact(urdf_path, relative_to=workspace)
    global_identity_isolation = _global_identity_isolation(
        fresh_by_robot, registries_by_robot
    )

    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\ncommit={git_commit_start}\n".encode("ascii"))
        os.fsync(lock_fd)
    finally:
        os.close(lock_fd)
    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        started_utc = _utc()
        effective_config = {key: value for key, value in config.items() if key != "_config_path"}
        (staging / "fresh_transition_v4_test.yaml").write_text(
            yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
        )
        identity_manifests: dict[str, Any] = {}
        dataset_descriptors: dict[str, Any] = {}
        for robot in ROBOTS:
            dataset_path = staging / f"{robot}_fresh_trajectories.npz"
            save_fresh_dataset(dataset_path, fresh_by_robot[robot])
            descriptor = _artifact(dataset_path, relative_to=staging)
            dataset_descriptors[robot] = descriptor
            identity_payload = build_identity_manifest(
                fresh_by_robot[robot],
                freshness[robot],
                dataset_artifact=descriptor,
            )
            _write_json(staging / f"{robot}_identity_manifest.json", identity_payload)
            identity_manifests[robot] = identity_payload
        _write_json(staging / "freshness_audit.json", {
            "protocol": PROTOCOL,
            "all_robots_pass": True,
            "robots": freshness,
            "generation_audits": generation,
            "formal_identity_access_ledgers": access_ledger,
            "formal_performance_result_files_opened": 0,
            "fresh_results_available_at_audit_time": False,
            "global_identity_isolation": global_identity_isolation,
        })
        seal_payload = {
            "protocol": PROTOCOL,
            "sealed_utc": _utc(),
            "git_commit": git_commit_start,
            "git_status": [],
            "method_registry": list(METHODS),
            "final_main_method": "counterfactual_cghik_v4",
            "fresh_results_used_for_tuning": False,
            "model_retrained": False,
            "policy_or_threshold_selected": False,
            "solver_verifier_fallback_changed": False,
            "v5_v6_v7_runtime_invocation_count": 0,
            "v8_v9_implemented": False,
            "fresh_outcomes_computed_before_seal": False,
            "fresh_evaluation_run_count_per_robot_before_seal": 0,
            "fresh_dataset_artifacts": dataset_descriptors,
            "identity_manifest_artifacts": {
                robot: _artifact(
                    staging / f"{robot}_identity_manifest.json", relative_to=staging
                ) for robot in ROBOTS
            },
            "freshness_audit": _artifact(
                staging / "freshness_audit.json", relative_to=staging
            ),
            "protocol_config": _artifact(
                staging / "fresh_transition_v4_test.yaml", relative_to=staging
            ),
            "source_config": _artifact(source_config_path, relative_to=workspace),
            "urdf_artifacts": urdf_descriptors,
            "sealed_release_inputs": release_inputs,
            "implementation_sources": source_before,
            "protected_tree_digests_before": protected_digest_before,
            "one_shot_no_smoke_no_resume": True,
        }
        _write_exclusive_json(staging / "preregistration_seal.json", seal_payload)
        seal_sha = _sha256_file(staging / "preregistration_seal.json")
        _verify_sealed_stage_inputs(staging, seal_payload)

        # Phase boundary: discard the generation objects and reconstruct the
        # exact evaluation inputs from the sealed, hash-bound bytes.
        sealed_fresh: dict[str, FreshTransitionDataset] = {}
        for robot in ROBOTS:
            identity_descriptor = seal_payload["identity_manifest_artifacts"][robot]
            identity_payload = load_identity_manifest(
                staging / str(identity_descriptor["path"]),
                expected_artifact=identity_descriptor,
                expected_robot=robot,
            )
            kinematics = load_robot(source_config, robot)
            sealed_fresh[robot] = load_fresh_dataset(
                staging / str(dataset_descriptors[robot]["path"]),
                robot=robot,
                expected_artifact=dataset_descriptors[robot],
                identity_manifest=identity_payload,
                kinematics=kinematics,
            )
            if identity_anchors(sealed_fresh[robot]) != dict(
                config["fresh_data"]["frozen_identity"][robot]
            ):
                raise RuntimeError(f"{robot} sealed fresh identity anchor changed")
        fresh_by_robot = sealed_fresh

        torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
        torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
        torch.use_deterministic_algorithms(
            bool(config["runtime"]["deterministic_algorithms"])
        )
        methods_by_robot: dict[str, Mapping[str, Any]] = {}
        for robot in ROBOTS:
            kinematics = load_robot(source_config, robot)
            methods = build_frozen_methods(
                workspace=workspace,
                source_config=source_config,
                release_v3_root=release_v3_root,
                release_v4_root=release_v4_root,
                robot=robot,
                kinematics=kinematics,
                device=str(config["runtime"]["device"]),
                release_seed=17,
            )
            warmup_path = (
                workspace / "outputs" / "anchored_temporal_v7_pilot"
                / f"{robot}_trajectory_calibration.npz"
            )
            warmup_role = _load_warmup_role(warmup_path)
            warmup_methods(
                methods,
                warmup_role,
                frames=int(config["timing"]["warmup_frames"]),
                synchronize_cuda=bool(config["timing"]["synchronize_cuda"]),
            )
            methods_by_robot[robot] = methods

        benchmark_by_robot: dict[str, BenchmarkData] = {}
        postrun_isolation: dict[str, Any] = {}
        all_prior_registries = tuple(
            registry
            for prior_robot in ROBOTS
            for registry in registries_by_robot[prior_robot]
        )
        for ordinal, robot in enumerate(ROBOTS, start=1):
            if _sha256_file(staging / "preregistration_seal.json") != seal_sha:
                raise RuntimeError("preregistration seal changed before fresh evaluation")
            _verify_sealed_stage_inputs(staging, seal_payload)
            marker = {
                "protocol": PROTOCOL,
                "robot": robot,
                "robot_ordinal": ordinal,
                "started_utc": _utc(),
                "preregistration_seal_sha256": seal_sha,
                "fresh_identity_frozen": True,
                "retained_run_ordinal": 1,
                "expected_trajectory_count": EXPECTED_TRAJECTORIES,
                "expected_frame_count": EXPECTED_FRAMES,
                "method_calls_per_frame": 1,
            }
            _write_exclusive_json(staging / f"{robot}_evaluation_attempt_started.json", marker)
            result = benchmark_trajectories(
                fresh_by_robot[robot],
                methods_by_robot[robot],
                order_seed=int(config["timing"]["order_seed"][robot]),
                progress_every=int(config["runtime"]["progress_every_trajectories"]),
                synchronize_cuda=bool(config["timing"]["synchronize_cuda"]),
            )
            postrun_isolation[robot] = _postrun_query_isolation(
                result, all_prior_registries
            )
            raw_path = staging / f"{robot}_raw_records.npz"
            _save_npz_exclusive(raw_path, result.npz_payload())
            benchmark_by_robot[robot] = result
            _write_exclusive_json(staging / f"{robot}_evaluation_completed.json", {
                "protocol": PROTOCOL,
                "robot": robot,
                "completed_utc": _utc(),
                "preregistration_seal_sha256": seal_sha,
                "retained_run_count": 1,
                "trajectory_count": EXPECTED_TRAJECTORIES,
                "frame_count": EXPECTED_FRAMES,
                "method_count": len(METHODS),
                "solver_calls": EXPECTED_FRAMES * len(METHODS),
                "raw_records": _artifact(raw_path, relative_to=staging),
            })

        _write_json(staging / "executed_query_isolation.json", {
            "protocol": PROTOCOL,
            "all_zero_overlap": True,
            "robots": postrun_isolation,
        })
        main = [row for robot in ROBOTS for row in main_rows(benchmark_by_robot[robot])]
        trajectories = [
            row for robot in ROBOTS for row in trajectory_rows(benchmark_by_robot[robot])
        ]
        families = [
            row for robot in ROBOTS for row in family_rows(benchmark_by_robot[robot])
        ]
        completion = {
            "protocol": PROTOCOL,
            "robots": {
                robot: completion_identity(benchmark_by_robot[robot]) for robot in ROBOTS
            },
        }
        gate = final_gate(main)
        _write_json(staging / "main_table.json", main)
        _write_csv(staging / "main_table.csv", main)
        (staging / "main_table.md").write_text(main_markdown(main), encoding="utf-8")
        _write_json(staging / "trajectory_table.json", trajectories)
        _write_csv(staging / "trajectory_table.csv", trajectories)
        _write_json(staging / "family_table.json", families)
        _write_csv(staging / "family_table.csv", families)
        _write_json(staging / "completion_uids.json", completion)
        _write_json(staging / "final_gate.json", gate)
        _write_json(staging / "environment.json", {
            **environment_payload(),
            "protocol": PROTOCOL,
            "git_commit": git_commit_start,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "fresh_result_used_for_tuning": False,
        })

        # Final content/provenance recheck before atomic publication.
        if _sha256_file(staging / "preregistration_seal.json") != seal_sha:
            raise RuntimeError("preregistration seal changed during fresh evaluation")
        _verify_sealed_stage_inputs(staging, seal_payload)
        _verify_release_anchors(
            workspace=workspace,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            source_config_path=source_config_path,
        )
        _verify_sources(workspace, source_before)
        if _git(workspace, "rev-parse", "HEAD") != git_commit_start:
            raise RuntimeError("git HEAD changed during fresh evaluation")
        git_status_end = _git(workspace, "status", "--short").splitlines()
        if git_status_end != git_status_start:
            raise RuntimeError("worktree status changed during fresh evaluation")
        protected_after = {
            name: _tree_snapshot(path) for name, path in protected_roots.items()
        }
        protected_digest_after = {
            name: _snapshot_digest(snapshot) for name, snapshot in protected_after.items()
        }
        if protected_after != protected_before:
            raise RuntimeError("a protected historical evidence tree changed")

        expected_files = {
            "fresh_transition_v4_test.yaml", "freshness_audit.json",
            "preregistration_seal.json", "executed_query_isolation.json",
            "main_table.json", "main_table.csv", "main_table.md",
            "trajectory_table.json", "trajectory_table.csv",
            "family_table.json", "family_table.csv", "completion_uids.json",
            "final_gate.json", "environment.json",
        }
        for robot in ROBOTS:
            expected_files.update({
                f"{robot}_fresh_trajectories.npz",
                f"{robot}_identity_manifest.json",
                f"{robot}_evaluation_attempt_started.json",
                f"{robot}_evaluation_completed.json",
                f"{robot}_raw_records.npz",
            })
        actual_files = {
            path.name for path in staging.iterdir() if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files or any(path.is_symlink() for path in staging.iterdir()):
            raise RuntimeError(
                f"final output set differs from the preregistered allowlist: "
                f"missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}"
            )
        artifacts = {
            name: _artifact(staging / name, relative_to=staging)
            for name in sorted(expected_files)
        }
        manifest = {
            "protocol": PROTOCOL,
            "status": "complete",
            "started_utc": started_utc,
            "completed_utc": _utc(),
            "git_commit_start": git_commit_start,
            "git_commit_end": git_commit_start,
            "git_status_start": git_status_start,
            "git_status_end": git_status_end,
            "method_registry": list(METHODS),
            "final_main_method": "counterfactual_cghik_v4",
            "preregistration_seal_sha256": seal_sha,
            "fresh_identity_frozen_before_solver_calls": True,
            "fresh_results_used_for_tuning": False,
            "model_retrained": False,
            "threshold_policy_solver_fallback_verifier_changed": False,
            "v5_v6_v7_runtime_invocation_count": 0,
            "v8_v9_implemented": False,
            "fresh_evaluation_run_count_per_robot": {robot: 1 for robot in ROBOTS},
            "trajectory_count_per_robot": EXPECTED_TRAJECTORIES,
            "frame_count_per_robot": EXPECTED_FRAMES,
            "method_calls_per_frame": 1,
            "formal_identity_arrays_opened_before_run": True,
            "formal_performance_result_files_opened": 0,
            "old_formal_query_and_trajectory_identity_overlap_count": 0,
            "protected_tree_digests_before": protected_digest_before,
            "protected_tree_digests_after": protected_digest_after,
            "protected_trees_unchanged": True,
            "release_inputs": release_inputs,
            "implementation_sources": source_before,
            "final_gate": gate,
            "next_stage": "frozen_v4_final_paper_preparation",
            "v8_v9_development_forbidden": True,
            "artifacts": artifacts,
        }
        _write_exclusive_json(staging / "run_manifest.json", manifest)
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.replace(staging, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        lock.unlink()
        return manifest
    except BaseException:
        # Deliberately retain lock + staging as a one-shot audit trail.  There
        # is no automatic cleanup, resume, or second scientific run.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run(args.config)
    print(json.dumps({
        "status": manifest["status"],
        "output": OUTPUT_NAME,
        "gate": manifest["final_gate"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["identity_anchors", "main", "run", "validate_config"]
