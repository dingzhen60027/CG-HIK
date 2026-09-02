"""Development-only Hierarchical V5-Lite training and policy-validation pilot.

The runner has an explicit calibration seal.  It cannot open the
``policy_validation_queries`` role until both robot-specific models and
cost-sensitive thresholds have been frozen and hashed.  Formal-test roles and
paths are rejected, and every existing output tree is treated as read-only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..counterfactual_v4.runner import _build_runtimes, _wait_for_quiet_environment
from ..counterfactual_v4.runtime_v4 import wrap_profiled_runtime
from ..hierarchical_v5.model import (
    TorchScriptFastGateInference as TorchScriptV5Inference,
    load_exact_torchscript as load_exact_v5_torchscript,
)
from ..hierarchical_v5.pilot import (
    ROLE_ORDER,
    _artifact,
    _load_development_role,
    _sha256_file,
    _solver_components,
    _verified_release_inputs,
    _write_json,
    latin_method_orders,
)
from ..hierarchical_v5.policy import FastGatePolicy as FrozenV5FastGatePolicy
from ..hierarchical_v5.policy import load_policy as load_v5_policy
from ..hierarchical_v5.runtime import (
    AlwaysLocalRuntime,
    HierarchicalOutcome,
    HierarchicalRuntime,
)
from ..latency_pilot_v3.benchmark import (
    ConstantRiskEngine,
    ProfiledCascadeRuntime,
    ProfiledOutcome,
)
from ..experiments.provenance import environment_payload
from ..release_v3_locked.artifacts import load_locked_seed_engine
from ..release_v4_locked.artifacts import (
    FrozenV4Policy,
    TorchScriptV4Inference,
    load_exact_v4_predictor,
    load_policy_config,
)
from ..test_v3_locked.runner import _release_paths
from ..runtime.cascade import EntryAction, FixedEntryGate
from .features import lite_feature_dim, lite_feature_names, prepare_lite_features
from .model import (
    LiteGatePredictor,
    LiteGateTrainingConfig,
    TorchScriptLiteGateInference,
    export_exact_torchscript,
    load_exact_torchscript,
    numerical_equivalence,
)
from .policy import (
    LiteFastGatePolicy,
    LiteGatePolicyConfig,
    ThresholdSelectionConfig,
    load_policy,
    save_policy,
    select_thresholds,
)
from .runtime import HierarchicalLiteOutcome, HierarchicalLiteRuntime


PROTOCOL = "hierarchical_v5_lite_policy_validation_pilot_v1"
METHODS = (
    "always_local",
    "always_hard",
    "counterfactual_cghik_v4",
    "hierarchical_cghik_v5",
    "hierarchical_cghik_v5_lite",
)
STAGE_NAMES = ("feature", "gate", "local", "robust", "unattributed")
ROUTE_STATES = ("fast_verified", "fast_failed_then_hard", "direct_hard")
EXPECTED_THRESHOLD_GRID = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925,
    0.95, 0.975, 0.99, 0.995, 1.00,
)
STAGE_DEFINITIONS = {
    "feature": "first-level feature preparation attributed by hierarchical runtimes",
    "gate": "first-level gate inference and threshold decision",
    "local": "one-step previous-state local solver plus deterministic verification",
    "robust": "complete invoked robust runtime, including its internal learned front end, solvers, and verification",
    "unattributed": "outer API latency not included in the four explicitly attributed stages",
}
METHOD_STAGE_CONTRACT = {
    "always_local": "the entire outer call is attributed to local",
    "always_hard": "the entire outer call is attributed to robust; feature/gate are zero by taxonomy, not absent internally",
    "counterfactual_cghik_v4": "the entire outer call is attributed to robust; feature/gate are zero by taxonomy, not absent internally",
    "hierarchical_cghik_v5": "the runtime's cheap-feature, gate, local, and slow timings are mapped to the common stages",
    "hierarchical_cghik_v5_lite": "the runtime's Lite feature, gate, local, and fixed-HARD timings are mapped to the common stages",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _forbid_test(value: str | Path, *, name: str) -> None:
    if "test" in str(value).lower():
        raise ValueError(f"{name} must not name formal-test data: {value}")


def validate_config(config: Mapping[str, Any], *, workspace: Path) -> None:
    """Fail closed on method, split, compute-boundary, and goal drift."""

    if config.get("protocol_version") != "hierarchical_v5_lite_pilot_v1":
        raise ValueError("unexpected V5-Lite protocol")
    if tuple(config.get("robots", ())) != ("panda", "ur5e"):
        raise ValueError("the pilot requires Panda and UR5e")
    if int(config.get("training_seed", -1)) != 17:
        raise ValueError("the development seed must remain 17")
    roles = config.get("roles", {})
    actual_roles = (
        str(roles.get("train", "")),
        str(roles.get("calibration", "")),
        str(roles.get("policy_validation", "")),
    )
    if actual_roles != ROLE_ORDER:
        raise ValueError(f"development roles must be exactly {ROLE_ORDER}")
    boundary = config.get("data_boundary", {})
    if tuple(boundary.get("allowed_roles", ())) != ROLE_ORDER:
        raise ValueError("development role allowlist changed")
    if not all(
        bool(boundary.get(key, False))
        for key in (
            "test_data_forbidden",
            "reject_test_named_paths",
            "load_policy_validation_only_after_policy_seal",
        )
    ):
        raise ValueError("data-boundary assertions are incomplete")
    for role in actual_roles:
        _forbid_test(role, name="development role")
    expected_dims = {"panda": 16, "ur5e": 15}
    if dict(config.get("lite_features", {}).get("dimension_by_robot", {})) != expected_dims:
        raise ValueError("V5-Lite per-robot feature dimensions changed")
    feature = config.get("lite_features", {})
    forbidden = (
        "jacobian_forbidden",
        "svd_forbidden",
        "dls_preview_forbidden",
        "linear_solve_forbidden",
        "learned_seed_ensemble_forbidden",
    )
    if not all(bool(feature.get(key, False)) for key in forbidden):
        raise ValueError("the lightweight compute boundary is not explicit")
    if tuple(config.get("model", {}).get("hidden_sizes", ())) != (16, 16):
        raise ValueError("the V5-Lite gate must remain 16x16")
    if tuple(config.get("model", {}).get("output_heads", ())) != (
        "local_verified_success",
    ):
        raise ValueError("the V5-Lite gate must have one output")
    if int(config.get("fast_path", {}).get("fast_iterations", -1)) != 1:
        raise ValueError("the local budget must remain one iteration")
    if config.get("fast_path", {}).get("direct_robust_action") != "always_hard":
        raise ValueError("the V5-Lite slow path must be fixed HARD")
    if tuple(config.get("strategies", ())) != METHODS:
        raise ValueError(f"strategies must be exactly {METHODS}")
    timing = config.get("timing", {})
    if (
        timing.get("clock") != "perf_counter_ns"
        or int(timing.get("repeats", -1)) != 5
        or tuple(timing.get("stage_names", ())) != STAGE_NAMES
    ):
        raise ValueError("timing contract changed")
    if int(config.get("calibration", {}).get("exact_arm_timing_repeats", -1)) != 5:
        raise ValueError("calibration requires five paired exact-arm timing repeats")
    calibration = config.get("calibration", {})
    if (
        calibration.get("fit_role") != "calibration_queries"
        or calibration.get("threshold_selection_role") != "calibration_queries"
        or tuple(float(value) for value in calibration.get("threshold_probability_grid", ()))
        != EXPECTED_THRESHOLD_GRID
        or calibration.get("include_always_robust_sentinel") is not True
        or calibration.get("selected_policy_exact_runtime_check") is not True
        or calibration.get("retune_after_exact_runtime_check") is not False
        or calibration.get("selection_constraints", {}).get(
            "per_query_verified_success_equal_always_hard"
        ) is not True
        or tuple(calibration.get("selection_objective_order", ()))
        != (
            "minimum_p95_end_to_end_latency",
            "minimum_p50_end_to_end_latency",
            "minimum_learned_seed_invocation_rate",
            "conservative_threshold_tiebreak",
        )
    ):
        raise ValueError("calibration selection contract changed")
    goals = config.get("pilot_goals", {})
    if goals.get("per_query_verified_success_equal_always_hard") is not True:
        raise ValueError("per-query success equality goal must be preregistered")
    if float(goals.get("p95_ratio_vs_always_hard_max", np.nan)) != 1.0:
        raise ValueError("P95 goal must be preregistered at 1.0")
    if not float(
        goals.get("learned_seed_ensemble_invocation_rate_max_exclusive", np.nan)
    ) < 1.0:
        raise ValueError("seed-invocation reduction goal must be preregistered")
    for key in (
        "bulk_root",
        "release_v3_root",
        "release_v4_root",
        "frozen_v5_root",
        "output_root",
    ):
        _forbid_test(str(config.get(key, "")), name=key)
    expected_output = (workspace / "outputs" / "hierarchical_v5_lite_pilot").resolve()
    if resolve_path(dict(config), str(config["output_root"])) != expected_output:
        raise ValueError(f"output_root must resolve to {expected_output}")


def _tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _snapshot_digest(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _query_role_audit(values: np.ndarray) -> dict[str, Any]:
    hashes = [str(value) for value in np.asarray(values).reshape(-1)]
    unique = sorted(set(hashes))
    return {
        "query_count": len(hashes),
        "unique_query_count": len(unique),
        "duplicate_query_count": len(hashes) - len(unique),
        "ordered_query_digest": sha256(
            json.dumps(hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "query_set_digest": sha256("\n".join(unique).encode("utf-8")).hexdigest(),
    }


def _verify_recorded_artifact(path: Path, descriptor: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"recorded artifact is missing or unsafe: {path}")
    if (
        path.stat().st_size != int(descriptor.get("size", -1))
        or _sha256_file(path) != str(descriptor.get("sha256", ""))
    ):
        raise RuntimeError(f"recorded artifact changed: {path}")


def _verify_calibration_seal(
    staging: Path,
    seal_payload: Mapping[str, Any],
    seal_hash: str,
) -> None:
    seal_path = staging / "calibration_seal.json"
    if _sha256_file(seal_path) != seal_hash:
        raise RuntimeError("calibration seal changed")
    for robot, descriptors in dict(seal_payload["robots"]).items():
        for key in ("model", "policy"):
            descriptor = dict(descriptors[key])
            relative = Path(str(descriptor.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"unsafe sealed {robot}/{key} artifact path: {relative}"
                )
            _verify_recorded_artifact(staging / relative, descriptor)


@dataclass(frozen=True)
class LabelData:
    features: np.ndarray
    local_success: np.ndarray
    local_fev: np.ndarray


@dataclass(frozen=True)
class CalibrationComponents:
    features: np.ndarray
    probability: np.ndarray
    forced_fast_total_samples_ns: np.ndarray
    forced_robust_total_samples_ns: np.ndarray
    forced_fast_stage_samples_ns: np.ndarray
    forced_robust_stage_samples_ns: np.ndarray
    forced_fast_success: np.ndarray
    forced_robust_success: np.ndarray
    forced_fast_local_success: np.ndarray
    forced_fast_seed_invoked: np.ndarray
    forced_robust_seed_invoked: np.ndarray
    forced_fast_fev: np.ndarray
    forced_robust_fev: np.ndarray


def _exact_selected_calibration_check(
    role: object,
    *,
    runtime: HierarchicalLiteRuntime,
    components: CalibrationComponents,
    threshold: float,
    dt: float,
    repeats: int,
    warmup: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Measure the frozen selected policy exactly; never return a new threshold."""

    count = int(role.count)
    for index in range(warmup):
        runtime.solve(role.query(index % count, dt=dt))
    latency = np.zeros((count, repeats), dtype=np.int64)
    accepted = np.zeros(count, dtype=bool)
    seed_invoked = np.zeros(count, dtype=bool)
    local_attempted = np.zeros(count, dtype=bool)
    route = np.full(count, "", dtype="U64")
    for query_index in range(count):
        query = role.query(query_index, dt=dt)
        signature: tuple[Any, ...] | None = None
        for repeat in range(repeats):
            started = perf_counter_ns()
            outcome = runtime.solve(query)
            latency[query_index, repeat] = perf_counter_ns() - started
            current = _signature(outcome)
            if signature is None:
                signature = current
                accepted[query_index] = bool(outcome.accepted)
                seed_invoked[query_index] = bool(
                    outcome.learned_seed_ensemble_invoked
                )
                local_attempted[query_index] = bool(outcome.local_attempted)
                route[query_index] = str(outcome.route)
            elif signature != current:
                raise RuntimeError(
                    "selected calibration runtime semantics changed across repeats"
                )
    expected_fast = (float(threshold) < 1.0) & (
        components.probability >= float(threshold)
    )
    expected_success = np.where(
        expected_fast,
        components.forced_fast_success,
        components.forced_robust_success,
    )
    expected_seed = np.where(
        expected_fast,
        components.forced_fast_seed_invoked,
        components.forced_robust_seed_invoked,
    )
    if not np.array_equal(local_attempted, expected_fast):
        raise RuntimeError("exact calibration route differs from frozen threshold")
    if not np.array_equal(accepted, expected_success):
        raise RuntimeError("exact calibration success differs from counterfactual simulation")
    if not np.array_equal(seed_invoked, expected_seed):
        raise RuntimeError("exact calibration seed invocation differs from simulation")
    primary = role.expected_reachable & role.continuity_feasible
    query_latency = np.median(latency, axis=1)
    report = {
        "query_count": count,
        "timing_repeats": repeats,
        "selected_threshold": float(threshold),
        "route_match_rate": 1.0,
        "success_match_rate": 1.0,
        "seed_invocation_match_rate": 1.0,
        "primary_p50_ns": float(np.percentile(query_latency[primary], 50)),
        "primary_p95_ns": float(np.percentile(query_latency[primary], 95)),
        "primary_p99_ns": float(np.percentile(query_latency[primary], 99)),
        "retuned_after_check": False,
    }
    raw = {
        "accepted": accepted,
        "seed_invoked": seed_invoked,
        "local_attempted": local_attempted,
        "route": route,
    }
    return latency, report, raw


def _signature(outcome: object) -> tuple[Any, ...]:
    q = getattr(outcome, "q", None)
    return (
        bool(getattr(outcome, "accepted")),
        int(getattr(outcome, "function_evaluations")),
        int(getattr(outcome, "iterations")),
        bool(getattr(outcome, "fallback_used", False)),
        tuple(getattr(outcome, "verification_reasons", ())),
        str(getattr(outcome, "route", getattr(outcome, "entry_action", ""))),
        tuple(str(stage) for stage in getattr(outcome, "executed_stages", ())),
        None if q is None else np.asarray(q, dtype=np.float64).tobytes(),
    )


def _collect_labels(
    role: object,
    *,
    kinematics: object,
    local_runtime: AlwaysLocalRuntime,
    dt: float,
    progress_every: int,
) -> LabelData:
    count = int(role.count)
    dimension = lite_feature_dim(int(kinematics.nq))
    features = np.empty((count, dimension), dtype=np.float32)
    success = np.zeros(count, dtype=bool)
    fev = np.zeros(count, dtype=np.int64)
    for index in range(count):
        query = role.query(index, dt=dt)
        prepared = prepare_lite_features(kinematics, query)
        outcome = local_runtime.solve(query)
        features[index] = prepared.features
        success[index] = bool(outcome.accepted)
        fev[index] = int(outcome.function_evaluations)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"[v5-lite] {role.robot}/{role.role} labels {index + 1}/{count}", flush=True)
    return LabelData(np.ascontiguousarray(features), success, fev)


def _collect_calibration_components(
    role: object,
    *,
    forced_fast_runtime: HierarchicalLiteRuntime,
    forced_robust_runtime: HierarchicalLiteRuntime,
    expected: LabelData,
    dt: float,
    repeats: int,
    warmup: int,
    progress_every: int,
) -> CalibrationComponents:
    count = int(role.count)
    for index in range(warmup):
        query = role.query(index % count, dt=dt)
        forced_fast_runtime.solve(query)
        forced_robust_runtime.solve(query)
    fast_total = np.zeros((count, repeats), dtype=np.int64)
    robust_total = np.zeros_like(fast_total)
    fast_stages = np.zeros((count, repeats, len(STAGE_NAMES)), dtype=np.int64)
    robust_stages = np.zeros_like(fast_stages)
    probability = np.full(count, np.nan, dtype=np.float64)
    fast_success = np.zeros(count, dtype=bool)
    robust_success = np.zeros(count, dtype=bool)
    fast_local_success = np.zeros(count, dtype=bool)
    fast_seed = np.zeros(count, dtype=bool)
    robust_seed = np.zeros(count, dtype=bool)
    fast_fev = np.zeros(count, dtype=np.int64)
    robust_fev = np.zeros(count, dtype=np.int64)
    base = ("forced_fast", "forced_robust")
    for query_index in range(count):
        query = role.query(query_index, dt=dt)
        signatures: dict[str, tuple[Any, ...]] = {}
        for repeat in range(repeats):
            offset = (query_index + repeat) % len(base)
            order = base[offset:] + base[:offset]
            for action in order:
                runtime = (
                    forced_fast_runtime
                    if action == "forced_fast"
                    else forced_robust_runtime
                )
                started = perf_counter_ns()
                outcome = runtime.solve(query)
                elapsed = perf_counter_ns() - started
                totals = fast_total if action == "forced_fast" else robust_total
                stages = fast_stages if action == "forced_fast" else robust_stages
                totals[query_index, repeat] = elapsed
                stages[query_index, repeat] = _stage_timings(
                    outcome, "hierarchical_cghik_v5_lite", elapsed
                )
                signature = _signature(outcome)
                if action not in signatures:
                    signatures[action] = signature
                    value = float(outcome.gate_local_success_probability)
                    if np.isnan(probability[query_index]):
                        probability[query_index] = value
                    elif value != probability[query_index]:
                        raise RuntimeError("gate probability differs between exact arms")
                    if not np.array_equal(
                        outcome.lite_features, expected.features[query_index]
                    ):
                        raise RuntimeError(
                            "calibration features changed after model fitting"
                        )
                    if action == "forced_fast":
                        if not bool(outcome.local_attempted):
                            raise RuntimeError("forced FAST arm did not attempt local DLS")
                        fast_success[query_index] = bool(outcome.accepted)
                        fast_local_success[query_index] = bool(outcome.local_accepted)
                        fast_seed[query_index] = bool(
                            outcome.learned_seed_ensemble_invoked
                        )
                        fast_fev[query_index] = int(outcome.function_evaluations)
                    else:
                        if bool(outcome.local_attempted):
                            raise RuntimeError("forced ROBUST arm attempted local DLS")
                        if not bool(outcome.learned_seed_ensemble_invoked):
                            raise RuntimeError("forced ROBUST arm skipped learned HARD")
                        robust_success[query_index] = bool(outcome.accepted)
                        robust_seed[query_index] = True
                        robust_fev[query_index] = int(outcome.function_evaluations)
                elif signature != signatures[action]:
                    raise RuntimeError(f"{action} semantics changed across repeats")
        if progress_every and (query_index + 1) % progress_every == 0:
            print(f"[v5-lite] {role.robot}/calibration timing {query_index + 1}/{count}", flush=True)
    if not np.array_equal(fast_local_success, expected.local_success):
        raise RuntimeError("local labels changed during paired calibration timing")
    return CalibrationComponents(
        expected.features.copy(),
        probability,
        fast_total,
        robust_total,
        fast_stages,
        robust_stages,
        fast_success,
        robust_success,
        fast_local_success,
        fast_seed,
        robust_seed,
        fast_fev,
        robust_fev,
    )


def _fresh_fixed_hard(
    *, source_config: dict[str, Any], release_root: Path, robot: str,
    kinematics: object, device: str,
) -> ProfiledOutcome | object:
    _, fixed, _ = _build_runtimes(
        source_config=source_config,
        release_root=release_root,
        robot=robot,
        training_seed=17,
        kinematics=kinematics,
        device=device,
    )
    return fixed["hard"]


def _fresh_shared_fixed_hard(
    *,
    source_config: dict[str, Any],
    release_root: Path,
    robot: str,
    kinematics: object,
    device: str,
) -> tuple[object, object, object]:
    """Build HARD with the exact DLS/verifier objects used by Lite local."""

    paths = _release_paths(release_root, robot, 17)
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    hard = ProfiledCascadeRuntime(
        name="hierarchical_v5_lite_always_hard",
        kinematics=kinematics,
        seed_engine=load_locked_seed_engine(
            kinematics=kinematics,
            torchscript_path=paths["torchscript"],
            normalization_path=paths["normalization"],
            runtime_spec_path=paths["runtime_spec"],
            device=device,
        ),
        risk_engine=ConstantRiskEngine(),
        gate=FixedEntryGate(EntryAction.HARD),
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
        reuse_candidate_features=True,
    )
    return hard, dls, verifier


def _fresh_local(
    *, source_config: dict[str, Any], release_root: Path, robot: str,
    kinematics: object,
) -> tuple[AlwaysLocalRuntime, object, object]:
    paths = _release_paths(release_root, robot, 17)
    dls, verifier, _, _, _ = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    return AlwaysLocalRuntime(dls, verifier, iterations=1), dls, verifier


def _build_current_v4(
    *, source_config: dict[str, Any], release_v3_root: Path,
    release_v4_root: Path, robot: str, kinematics: object, device: str,
) -> object:
    release_paths = _release_paths(release_v3_root, robot, 17)
    dls, verifier, fallback, bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": release_paths["solver_metadata"], "seed_bank": release_paths["seed_bank"]},
        kinematics,
    )
    policy_config, _ = load_policy_config(release_v4_root / robot / "v4_policy.json")
    policy = FrozenV4Policy(
        TorchScriptV4Inference(
            load_exact_v4_predictor(release_v4_root / robot / "exact_v4_predictor.ts", device="cpu")
        ),
        policy_config,
    )
    return wrap_profiled_runtime(
        name="counterfactual_cghik_v4",
        policy=policy,
        kinematics=kinematics,
        seed_engine=load_locked_seed_engine(
            kinematics=kinematics,
            torchscript_path=release_paths["torchscript"],
            normalization_path=release_paths["normalization"],
            runtime_spec_path=release_paths["runtime_spec"],
            device=device,
        ),
        dls=dls,
        verifier=verifier,
        seed_bank=bank,
        fallback=fallback,
        cascade_config=cascade,
    )


def _verify_frozen_v5_artifacts(root: Path, robot: str) -> tuple[Path, Path]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_policy_validation_pilot"
        or bool(manifest.get("test_data_loaded", True))
        or bool(manifest.get("formal_test_started", True))
    ):
        raise RuntimeError("frozen V5 pilot is not an eligible development artifact")
    workspace = root.resolve().parents[1]
    implementation_sources = manifest.get("implementation_sources")
    if not isinstance(implementation_sources, dict) or not implementation_sources:
        raise RuntimeError("frozen V5 implementation source manifest is missing")
    for key, descriptor in implementation_sources.items():
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"invalid frozen V5 source descriptor: {key}")
        relative = Path(str(descriptor.get("path", key)))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe frozen V5 source path: {relative}")
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"frozen V5 implementation source is missing: {path}")
        if (
            path.stat().st_size != int(descriptor.get("size", -1))
            or _sha256_file(path) != str(descriptor.get("sha256", ""))
        ):
            raise RuntimeError(f"frozen V5 implementation source changed: {path}")
    files = manifest.get("artifacts", {})
    model = root / f"{robot}_exact_fast_gate.ts"
    policy = root / f"{robot}_fast_gate_policy.json"
    for path in (model, policy):
        expected = files.get(path.name)
        if expected is None or _sha256_file(path) != expected.get("sha256"):
            raise RuntimeError(f"frozen V5 artifact changed: {path}")
    return model, policy


def _build_current_v5(
    *, source_config: dict[str, Any], release_v3_root: Path,
    release_v4_root: Path, frozen_v5_root: Path, robot: str,
    kinematics: object, device: str,
) -> object:
    from ..hierarchical_v5.pilot import _build_no_easy_slow_runtime

    model_path, policy_path = _verify_frozen_v5_artifacts(frozen_v5_root, robot)
    policy_config, _ = load_v5_policy(policy_path)
    gate = FrozenV5FastGatePolicy(
        TorchScriptV5Inference(load_exact_v5_torchscript(model_path, device="cpu")),
        policy_config,
    )
    slow, dls, verifier = _build_no_easy_slow_runtime(
        source_config=source_config,
        release_v3_root=release_v3_root,
        release_v4_root=release_v4_root,
        robot=robot,
        kinematics=kinematics,
        device=device,
    )
    return HierarchicalRuntime(
        kinematics=kinematics, dls=dls, verifier=verifier,
        fast_gate=gate, slow_runtime=slow, fast_iterations=1,
    )


def _build_methods(
    *, source_config: dict[str, Any], release_v3_root: Path,
    release_v4_root: Path, frozen_v5_root: Path, robot: str,
    kinematics: object, device: str, lite_policy: LiteFastGatePolicy,
) -> dict[str, object]:
    local, _, _ = _fresh_local(
        source_config=source_config, release_root=release_v3_root,
        robot=robot, kinematics=kinematics,
    )
    lite_hard, lite_dls, lite_verifier = _fresh_shared_fixed_hard(
        source_config=source_config,
        release_root=release_v3_root,
        robot=robot,
        kinematics=kinematics,
        device=device,
    )
    methods = {
        "always_local": local,
        "always_hard": _fresh_fixed_hard(
            source_config=source_config, release_root=release_v3_root,
            robot=robot, kinematics=kinematics, device=device,
        ),
        "counterfactual_cghik_v4": _build_current_v4(
            source_config=source_config, release_v3_root=release_v3_root,
            release_v4_root=release_v4_root, robot=robot,
            kinematics=kinematics, device=device,
        ),
        "hierarchical_cghik_v5": _build_current_v5(
            source_config=source_config, release_v3_root=release_v3_root,
            release_v4_root=release_v4_root, frozen_v5_root=frozen_v5_root,
            robot=robot, kinematics=kinematics, device=device,
        ),
        "hierarchical_cghik_v5_lite": HierarchicalLiteRuntime(
            kinematics=kinematics,
            dls=lite_dls,
            verifier=lite_verifier,
            fast_gate=lite_policy,
            always_hard_runtime=lite_hard,
            fast_iterations=1,
        ),
    }
    if tuple(methods) != METHODS:
        raise RuntimeError("five-method registry drifted")
    return methods


def _stage_timings(outcome: object, method: str, outer_ns: int) -> np.ndarray:
    values = {name: 0 for name in STAGE_NAMES}
    timing = dict(getattr(outcome, "timings_ns", {}))
    if method == "always_local":
        values["local"] = int(outer_ns)
    elif method in {"always_hard", "counterfactual_cghik_v4"}:
        values["robust"] = int(outer_ns)
    elif method == "hierarchical_cghik_v5":
        values["feature"] = int(timing.get("cheap_feature_ns", 0))
        values["gate"] = int(timing.get("gate_ns", 0))
        values["local"] = int(timing.get("local_solver_ns", 0)) + int(
            timing.get("local_verifier_ns", 0)
        )
        values["robust"] = int(timing.get("slow_ns", 0))
    elif method == "hierarchical_cghik_v5_lite":
        values["feature"] = int(timing.get("feature_extraction_ns", 0))
        values["gate"] = int(timing.get("gate_ns", 0))
        values["local"] = int(timing.get("local_path_ns", 0))
        values["robust"] = int(timing.get("robust_path_ns", 0))
    else:
        raise ValueError(method)
    attributed = sum(values[name] for name in STAGE_NAMES[:-1])
    values["unattributed"] = max(int(outer_ns) - attributed, 0)
    return np.asarray([values[name] for name in STAGE_NAMES], dtype=np.int64)


@dataclass(frozen=True)
class BenchmarkData:
    robot: str
    query_sha256: np.ndarray
    category: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    latency_samples_ns: np.ndarray
    stage_latency_samples_ns: np.ndarray
    accepted: np.ndarray
    function_evaluations: np.ndarray
    seed_invoked: np.ndarray
    local_attempted: np.ndarray
    local_accepted: np.ndarray
    route: np.ndarray
    gate_probability: np.ndarray
    executed_stages: np.ndarray
    command_q: np.ndarray


def _instrument(outcome: object, method: str) -> tuple[bool, bool, bool, str, float]:
    if isinstance(outcome, (HierarchicalLiteOutcome, HierarchicalOutcome)):
        probability = getattr(
            outcome,
            "gate_local_success_probability",
            np.nan,
        )
        return (
            bool(outcome.learned_seed_ensemble_invoked),
            bool(outcome.local_attempted),
            bool(outcome.local_accepted),
            str(outcome.route),
            np.nan if probability is None else float(probability),
        )
    if method == "always_local":
        return False, True, bool(outcome.accepted), (
            "always_local_accept" if outcome.accepted else "always_local_failure"
        ), np.nan
    return True, False, False, str(getattr(outcome, "entry_action", method)), np.nan


def benchmark_policy_validation(
    role: object,
    methods: Mapping[str, object],
    *, config: Mapping[str, Any], dt: float,
) -> tuple[BenchmarkData, list[dict[str, Any]]]:
    if tuple(methods) != METHODS:
        raise ValueError("method registry differs from the frozen contract")
    warmup = int(config["timing"]["warmup_iterations"])
    for index in range(warmup):
        query = role.query(index % role.count, dt=dt)
        order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
        for name in order:
            methods[name].solve(query)
    count, method_count, repeats = int(role.count), len(METHODS), 5
    nq = int(role.dataset.previous_q.shape[1])
    latency = np.zeros((count, method_count, repeats), dtype=np.int64)
    stages = np.zeros((count, method_count, repeats, len(STAGE_NAMES)), dtype=np.int64)
    accepted = np.zeros((count, method_count), dtype=bool)
    fev = np.zeros((count, method_count), dtype=np.int64)
    seed = np.zeros((count, method_count), dtype=bool)
    attempted = np.zeros((count, method_count), dtype=bool)
    local_ok = np.zeros((count, method_count), dtype=bool)
    route = np.full((count, method_count), "", dtype="U64")
    probability = np.full((count, method_count), np.nan, dtype=np.float64)
    executed = np.full((count, method_count), "", dtype="U256")
    commands = np.full((count, method_count, nq), np.nan, dtype=np.float64)
    positions = {name: index for index, name in enumerate(METHODS)}
    quiet_events: list[dict[str, Any]] = []
    for query_index in range(count):
        if query_index % max(int(config["runtime"]["environment_check_every_queries"]), 1) == 0:
            event = _wait_for_quiet_environment(
                dict(config), context=f"hierarchical-v5-lite/{role.robot}/query{query_index}"
            )
            if event["had_busy_process"]:
                quiet_events.append(event)
        query = role.query(query_index, dt=dt)
        signatures: dict[str, tuple[Any, ...]] = {}
        orders = latin_method_orders(METHODS, int(config["model"]["seed"]) + query_index)
        for repeat, order in enumerate(orders):
            for name in order:
                started = perf_counter_ns()
                outcome = methods[name].solve(query)
                outer = perf_counter_ns() - started
                method_index = positions[name]
                latency[query_index, method_index, repeat] = outer
                stages[query_index, method_index, repeat] = _stage_timings(outcome, name, outer)
                signature = _signature(outcome)
                if name in signatures and signature != signatures[name]:
                    raise RuntimeError(f"{role.robot}/{name} semantics changed across repeats")
                if name not in signatures:
                    signatures[name] = signature
                    accepted[query_index, method_index] = bool(outcome.accepted)
                    fev[query_index, method_index] = int(outcome.function_evaluations)
                    seed_value, attempt_value, local_value, route_value, probability_value = _instrument(outcome, name)
                    seed[query_index, method_index] = seed_value
                    attempted[query_index, method_index] = attempt_value
                    local_ok[query_index, method_index] = local_value
                    route[query_index, method_index] = route_value
                    probability[query_index, method_index] = probability_value
                    executed[query_index, method_index] = "|".join(
                        str(stage) for stage in getattr(outcome, "executed_stages", ())
                    )
                    if outcome.q is not None:
                        commands[query_index, method_index] = np.asarray(outcome.q, dtype=np.float64)
        if (query_index + 1) % 100 == 0:
            print(f"[v5-lite] {role.robot} policy validation {query_index + 1}/{count}", flush=True)
    return BenchmarkData(
        str(role.robot), role.query_sha256.copy(), role.category.copy(),
        role.expected_reachable.copy(), role.continuity_feasible.copy(),
        latency, stages, accepted, fev, seed, attempted, local_ok, route,
        probability, executed, commands,
    ), quiet_events


def _primary(data: BenchmarkData) -> np.ndarray:
    mask = data.expected_reachable & data.continuity_feasible
    if not np.any(mask):
        raise ValueError("no operational-feasible policy-validation queries")
    return mask


def summarize(data: BenchmarkData) -> list[dict[str, Any]]:
    feasible = _primary(data)
    query_ms = np.median(data.latency_samples_ns, axis=2) / 1e6
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        attempts = data.local_attempted[feasible, method_index]
        hits = data.local_accepted[feasible, method_index]
        failed = attempts & ~hits
        rows.append({
            "robot": data.robot,
            "method": method,
            "feasible_queries": int(np.sum(feasible)),
            "verified_success": float(np.mean(data.accepted[feasible, method_index])),
            "p50_ms": float(np.percentile(query_ms[feasible, method_index], 50)),
            "p95_ms": float(np.percentile(query_ms[feasible, method_index], 95)),
            "p99_ms": float(np.percentile(query_ms[feasible, method_index], 99)),
            "mean_fev": float(np.mean(data.function_evaluations[feasible, method_index])),
            "learned_seed_ensemble_invocation_rate": float(np.mean(data.seed_invoked[feasible, method_index])),
            "fast_path_attempt_rate": float(np.mean(attempts)),
            "fast_path_hit_rate": float(np.mean(hits)),
            "fast_path_precision": float(np.sum(hits) / np.sum(attempts)) if np.any(attempts) else None,
            "fast_failure_recovery_rate": (
                float(np.mean(data.accepted[feasible, method_index][failed]))
                if np.any(failed) else None
            ),
        })
    return rows


def _pilot_gate(
    data: BenchmarkData,
    rows: Sequence[Mapping[str, Any]],
    goals: Mapping[str, Any],
) -> dict[str, Any]:
    by_method = {str(row["method"]): row for row in rows}
    lite = by_method["hierarchical_cghik_v5_lite"]
    hard = by_method["always_hard"]
    v4 = by_method["counterfactual_cghik_v4"]
    feasible = _primary(data)
    lite_index = METHODS.index("hierarchical_cghik_v5_lite")
    hard_index = METHODS.index("always_hard")
    success_mismatch_count = int(
        np.sum(
            data.accepted[feasible, lite_index]
            != data.accepted[feasible, hard_index]
        )
    )
    checks = {
        "per_query_verified_success_equal_always_hard": (
            success_mismatch_count == 0
        ),
        "p95_not_above_always_hard": (
            float(lite["p95_ms"]) / float(hard["p95_ms"])
            <= float(goals["p95_ratio_vs_always_hard_max"])
        ),
        "p50_below_current_v4": (
            float(lite["p50_ms"]) / float(v4["p50_ms"])
            < float(
                goals["p50_ratio_vs_counterfactual_cghik_v4_max_exclusive"]
            )
        ),
        "learned_seed_invocation_reduced": (
            float(lite["learned_seed_ensemble_invocation_rate"])
            < float(
                goals[
                    "learned_seed_ensemble_invocation_rate_max_exclusive"
                ]
            )
        ),
    }
    return {
        "checks": checks,
        "operational_feasible_query_count": int(np.sum(feasible)),
        "success_mismatch_count_vs_always_hard": success_mismatch_count,
        "aggregate_verified_success_lite": float(lite["verified_success"]),
        "aggregate_verified_success_always_hard": float(
            hard["verified_success"]
        ),
        "all_pass": all(checks.values()),
    }


def latency_breakdown(data: BenchmarkData) -> list[dict[str, Any]]:
    feasible = _primary(data)
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        for stage_index, stage in enumerate(STAGE_NAMES):
            per_query = np.median(
                data.stage_latency_samples_ns[feasible, method_index, :, stage_index], axis=1
            ) / 1e6
            invoked = per_query > 0.0
            for scope, values in (
                ("all_feasible_zero_when_not_invoked", per_query),
                ("invoked_only", per_query[invoked]),
            ):
                rows.append({
                    "robot": data.robot,
                    "method": method,
                    "stage": stage,
                    "scope": scope,
                    "query_count": int(values.size),
                    "invocation_rate": float(np.mean(invoked)),
                    "mean_ms": float(np.mean(values)) if values.size else None,
                    "p50_ms": float(np.percentile(values, 50)) if values.size else None,
                    "p95_ms": float(np.percentile(values, 95)) if values.size else None,
                    "p99_ms": float(np.percentile(values, 99)) if values.size else None,
                })
    return rows


def paired_summary(data: BenchmarkData, comparator: str) -> dict[str, Any]:
    feasible = _primary(data)
    per_query = np.median(data.latency_samples_ns, axis=2) / 1e6
    lite = METHODS.index("hierarchical_cghik_v5_lite")
    other = METHODS.index(comparator)
    delta = per_query[feasible, lite] - per_query[feasible, other]
    return {
        "robot": data.robot,
        "comparator": comparator,
        "paired_query_count": int(delta.size),
        "query_hash_digest": sha256(np.ascontiguousarray(data.query_sha256[feasible].astype("S64")).tobytes()).hexdigest(),
        "lite_minus_comparator_ms": {
            "mean": float(np.mean(delta)), "median": float(np.median(delta)),
            "p05": float(np.percentile(delta, 5)), "p95": float(np.percentile(delta, 95)),
            "fraction_lite_faster": float(np.mean(delta < 0.0)),
        },
    }


def family_routes(data: BenchmarkData) -> list[dict[str, Any]]:
    lite = METHODS.index("hierarchical_cghik_v5_lite")
    rows: list[dict[str, Any]] = []
    for family in sorted(set(data.category.astype(str))):
        selected = data.category == family
        attempts = data.local_attempted[selected, lite]
        hits = data.local_accepted[selected, lite]
        states = np.where(hits, "fast_verified", np.where(attempts, "fast_failed_then_hard", "direct_hard"))
        total = int(np.sum(selected))
        for state in ROUTE_STATES:
            count = int(np.sum(states == state))
            rows.append({
                "robot": data.robot, "query_family": family,
                "route_state": state, "count": count,
                "rate": count / total, "family_queries": total,
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _save_benchmark(path: Path, data: BenchmarkData) -> None:
    np.savez_compressed(
        path,
        method_names=np.asarray(METHODS, dtype=np.str_),
        stage_names=np.asarray(STAGE_NAMES, dtype=np.str_),
        query_sha256=data.query_sha256, category=data.category,
        expected_reachable=data.expected_reachable,
        continuity_feasible=data.continuity_feasible,
        latency_samples_ns=data.latency_samples_ns,
        stage_latency_samples_ns=data.stage_latency_samples_ns,
        accepted=data.accepted, function_evaluations=data.function_evaluations,
        learned_seed_ensemble_invoked=data.seed_invoked,
        local_attempted=data.local_attempted, local_accepted=data.local_accepted,
        route=data.route, gate_local_success_probability=data.gate_probability,
        executed_stages=data.executed_stages, command_q=data.command_q,
    )


def _plot_outputs(
    main_rows: Sequence[Mapping[str, Any]], family_rows: Sequence[Mapping[str, Any]],
    output: Path, dpi: int,
) -> None:
    lite = [row for row in main_rows if row["method"] == "hierarchical_cghik_v5_lite"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(3)
    for index, row in enumerate(lite):
        values = [row["fast_path_attempt_rate"], row["fast_path_hit_rate"], row["fast_path_precision"] or 0.0]
        ax.bar(x + (index - 0.5) * 0.34, values, 0.34, label=str(row["robot"]))
    ax.set_xticks(x, ["Attempt", "Hit", "Precision"]); ax.set_ylim(0, 1.05); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "fig1_fast_path_use_success.png", dpi=dpi); fig.savefig(output / "fig1_fast_path_use_success.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["Local", "Hard", "V4", "V5", "V5-Lite"]
    for ax, robot in zip(axes, ("panda", "ur5e"), strict=True):
        rows = [row for row in main_rows if row["robot"] == robot]
        x = np.arange(5)
        for offset, metric in enumerate(("p50_ms", "p95_ms", "p99_ms")):
            ax.bar(x + (offset - 1) * .24, [row[metric] for row in rows], .24, label=metric.upper())
        ax.set_xticks(x, labels, rotation=20); ax.set_ylabel("Latency (ms)"); ax.set_title(robot.upper()); ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(output / "fig2_five_strategy_latency_quantiles.png", dpi=dpi); fig.savefig(output / "fig2_five_strategy_latency_quantiles.pdf"); plt.close(fig)

    colors = {"fast_verified": "#2a9d8f", "fast_failed_then_hard": "#e9c46a", "direct_hard": "#264653"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, robot in zip(axes, ("panda", "ur5e"), strict=True):
        rows = [row for row in family_rows if row["robot"] == robot]
        families = sorted(set(str(row["query_family"]) for row in rows)); y = np.arange(len(families)); left = np.zeros(len(families))
        for state in ROUTE_STATES:
            values = np.asarray([next(float(row["rate"]) for row in rows if row["query_family"] == family and row["route_state"] == state) for family in families])
            ax.barh(y, values, left=left, color=colors[state], label=state); left += values
        ax.set_yticks(y, families); ax.set_xlim(0, 1); ax.set_title(robot.upper()); ax.set_xlabel("Route fraction")
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(output / "fig3_query_family_routes.png", dpi=dpi); fig.savefig(output / "fig3_query_family_routes.pdf"); plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run reduced development roles in outputs/hierarchical_v5_lite_smoke.",
    )
    return parser


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    workspace = resolve_path(config, str(config["workspace"]))
    validate_config(config, workspace=workspace)
    if smoke:
        config = deepcopy(config)
        config["model"]["epochs"] = 3
        config["timing"]["warmup_iterations"] = 5
        config["runtime"]["environment_check_every_queries"] = 32
    source_config_path = resolve_path(config, str(config["source_config"]))
    source_config = load_config(source_config_path)
    bulk_root = resolve_path(config, str(config["bulk_root"]))
    release_v3_root = resolve_path(config, str(config["release_v3_root"]))
    release_v4_root = resolve_path(config, str(config["release_v4_root"]))
    frozen_v5_root = resolve_path(config, str(config["frozen_v5_root"]))
    output_root = resolve_path(config, str(config["output_root"]))
    if smoke:
        output_root = (workspace / "outputs" / "hierarchical_v5_lite_smoke").resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"overwrite forbidden: {output_root}")
    stale_staging = sorted(
        output_root.parent.glob(f".{output_root.name}.incomplete.*")
    )
    if stale_staging:
        raise FileExistsError(
            "stale V5-Lite staging namespace requires explicit inspection: "
            + ", ".join(str(path) for path in stale_staging)
        )
    git_status = subprocess.run(["git", "status", "--short"], cwd=workspace, check=True, capture_output=True, text=True).stdout.splitlines()
    if not smoke and git_status:
        raise RuntimeError("full V5-Lite pilot requires a clean committed worktree")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()
    protected_before = _tree_snapshot(frozen_v5_root)
    protected_before_digest = _snapshot_digest(protected_before)
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("features.py").resolve(),
        Path(__file__).with_name("model.py").resolve(),
        Path(__file__).with_name("policy.py").resolve(),
        Path(__file__).with_name("runtime.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
        Path(config["_config_path"]).resolve(),
        (workspace / "scripts" / "run_hierarchical_v5_lite_pilot.sh").resolve(),
        (workspace / "tests" / "test_hierarchical_v5_lite.py").resolve(),
    )
    implementation_sources = {
        str(path.relative_to(workspace)): _artifact(path, relative_to=workspace)
        for path in implementation_paths
    }
    staging = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))
    torch.use_deterministic_algorithms(bool(config["runtime"]["deterministic_algorithms"]))
    dt = float(source_config["data"]["dt"])
    device = str(config["runtime"]["device"])
    started = _utc()
    contexts: dict[str, dict[str, Any]] = {}
    training_report: dict[str, Any] = {}
    threshold_report: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    role_audit: dict[str, Any] = {}
    quiet_events: dict[str, Any] = {
        str(robot): {"calibration_components": None, "exact_calibration": None, "policy_validation": []}
        for robot in config["robots"]
    }

    # Phase A: train + calibration only.  Policy-validation is not opened here.
    for robot in config["robots"]:
        robot = str(robot); kinematics = load_robot(source_config, robot)
        source_records[f"{robot}/sealed_releases"] = _verified_release_inputs(
            workspace=workspace, release_v3_root=release_v3_root,
            release_v4_root=release_v4_root, robot=robot,
        )
        train_role = _load_development_role(
            workspace=workspace, bulk_root=bulk_root, robot=robot,
            role="risk_train_queries", training_seed=17, dt=dt,
        )
        calibration_role = _load_development_role(
            workspace=workspace, bulk_root=bulk_root, robot=robot,
            role="calibration_queries", training_seed=17, dt=dt,
        )
        if smoke:
            train_role = train_role.subset(min(128, train_role.count))
            calibration_role = calibration_role.subset(min(64, calibration_role.count))
        if set(train_role.query_sha256.astype(str)) & set(calibration_role.query_sha256.astype(str)):
            raise RuntimeError(f"{robot} train/calibration overlap")
        train_hashes = set(train_role.query_sha256.astype(str))
        calibration_hashes = set(calibration_role.query_sha256.astype(str))
        role_audit[robot] = {
            "roles": {
                "risk_train_queries": _query_role_audit(train_role.query_sha256),
                "calibration_queries": _query_role_audit(
                    calibration_role.query_sha256
                ),
            },
            "pairwise_overlap_counts": {
                "risk_train_queries__calibration_queries": len(
                    train_hashes & calibration_hashes
                )
            },
        }
        source_records[f"{robot}/risk_train_queries"] = train_role.source_manifest
        source_records[f"{robot}/calibration_queries"] = calibration_role.source_manifest
        local, _, _ = _fresh_local(source_config=source_config, release_root=release_v3_root, robot=robot, kinematics=kinematics)
        progress = 0 if smoke else 1000
        train_labels = _collect_labels(train_role, kinematics=kinematics, local_runtime=local, dt=dt, progress_every=progress)
        calibration_labels = _collect_labels(calibration_role, kinematics=kinematics, local_runtime=local, dt=dt, progress_every=progress)
        training_config = LiteGateTrainingConfig(
            epochs=int(config["model"]["epochs"]), batch_size=int(config["model"]["batch_size"]),
            learning_rate=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"]),
            gradient_clip_norm=float(config["model"]["gradient_clip_norm"]), seed=int(config["model"]["seed"]),
        )
        predictor = LiteGatePredictor(
            train_labels.features.shape[1],
            training_config,
            device="cpu",
            feature_names=lite_feature_names(kinematics),
        )
        predictor.fit(train_labels.features, train_labels.local_success, role="risk_train_queries", provenance={"robot": robot, "test_data_loaded": False})
        predictor.calibrate(calibration_labels.features, calibration_labels.local_success, role="calibration_queries", provenance={"robot": robot, "test_data_loaded": False})
        eager_path = staging / f"{robot}_lite_gate_predictor.pt"
        exact_path = staging / f"{robot}_exact_lite_gate.ts"
        predictor.save(eager_path, provenance={"selection_role": "calibration_queries"})
        export_exact_torchscript(predictor, exact_path)
        exact_module = load_exact_torchscript(exact_path, device="cpu")
        backend = TorchScriptLiteGateInference(exact_module, train_labels.features.shape[1])
        fast_hard, fast_dls, fast_verifier = _fresh_shared_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        robust_hard, robust_dls, robust_verifier = _fresh_shared_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        forced_fast_runtime = HierarchicalLiteRuntime(
            kinematics=kinematics,
            dls=fast_dls,
            verifier=fast_verifier,
            fast_gate=LiteFastGatePolicy(
                TorchScriptLiteGateInference(
                    load_exact_torchscript(exact_path, device="cpu"),
                    train_labels.features.shape[1],
                ),
                LiteGatePolicyConfig(
                    local_success_threshold=0.0,
                    calibration_count=int(calibration_role.count),
                ),
            ),
            always_hard_runtime=fast_hard,
            fast_iterations=1,
            name="hierarchical_v5_lite_forced_fast_calibration",
        )
        forced_robust_runtime = HierarchicalLiteRuntime(
            kinematics=kinematics,
            dls=robust_dls,
            verifier=robust_verifier,
            fast_gate=LiteFastGatePolicy(
                TorchScriptLiteGateInference(
                    load_exact_torchscript(exact_path, device="cpu"),
                    train_labels.features.shape[1],
                ),
                LiteGatePolicyConfig(
                    local_success_threshold=1.0,
                    calibration_count=int(calibration_role.count),
                ),
            ),
            always_hard_runtime=robust_hard,
            fast_iterations=1,
            name="hierarchical_v5_lite_forced_robust_calibration",
        )
        quiet_events[robot]["calibration_components"] = _wait_for_quiet_environment(
            dict(config), context=f"hierarchical-v5-lite/{robot}/calibration-components"
        )
        components = _collect_calibration_components(
            calibration_role,
            forced_fast_runtime=forced_fast_runtime,
            forced_robust_runtime=forced_robust_runtime,
            expected=calibration_labels,
            dt=dt,
            repeats=5,
            warmup=int(config["timing"]["warmup_iterations"]),
            progress_every=0 if smoke else 500,
        )
        primary_mask = calibration_role.expected_reachable & calibration_role.continuity_feasible
        selection_config = ThresholdSelectionConfig(
            threshold_grid=tuple(float(value) for value in config["calibration"]["threshold_probability_grid"])
        )
        policy_config, selection = select_thresholds(
            components.probability,
            components.forced_fast_total_samples_ns,
            components.forced_robust_total_samples_ns,
            components.forced_fast_success,
            components.forced_robust_success,
            components.forced_fast_local_success,
            components.forced_fast_seed_invoked,
            components.forced_robust_seed_invoked,
            primary_mask,
            role="calibration_queries",
            config=selection_config,
        )
        policy_path = staging / f"{robot}_lite_gate_policy.json"
        save_policy(policy_path, policy_config, selection, provenance={"robot": robot, "test_data_loaded": False})
        equivalence = numerical_equivalence(
            predictor, exact_module, calibration_labels.features[:1000],
            threshold=float(policy_config.local_success_threshold), atol=1e-10,
        )
        if not equivalence["passed"]:
            raise RuntimeError(f"{robot} exact Lite gate equivalence failed")
        frozen_policy = LiteFastGatePolicy(backend, policy_config)
        exact_hard, exact_dls, exact_verifier = _fresh_shared_fixed_hard(
            source_config=source_config,
            release_root=release_v3_root,
            robot=robot,
            kinematics=kinematics,
            device=device,
        )
        quiet_events[robot]["exact_calibration"] = _wait_for_quiet_environment(
            dict(config), context=f"hierarchical-v5-lite/{robot}/exact-calibration"
        )
        exact_latency, exact_check, exact_raw = _exact_selected_calibration_check(
            calibration_role,
            runtime=HierarchicalLiteRuntime(
                kinematics=kinematics,
                dls=exact_dls,
                verifier=exact_verifier,
                fast_gate=frozen_policy,
                always_hard_runtime=exact_hard,
                fast_iterations=1,
            ),
            components=components,
            threshold=float(policy_config.local_success_threshold),
            dt=dt,
            repeats=5,
            warmup=int(config["timing"]["warmup_iterations"]),
        )
        np.savez_compressed(
            staging / f"{robot}_risk_train_lite_labels.npz",
            feature_names=np.asarray(lite_feature_names(kinematics), dtype=np.str_),
            query_sha256=train_role.query_sha256, features=train_labels.features,
            local_verified_success=train_labels.local_success,
            local_function_evaluations=train_labels.local_fev,
        )
        np.savez_compressed(
            staging / f"{robot}_calibration_lite_components.npz",
            feature_names=np.asarray(lite_feature_names(kinematics), dtype=np.str_),
            query_sha256=calibration_role.query_sha256, features=components.features,
            local_verified_success=components.forced_fast_local_success,
            forced_fast_verified_success=components.forced_fast_success,
            forced_robust_verified_success=components.forced_robust_success,
            forced_fast_function_evaluations=components.forced_fast_fev,
            forced_robust_function_evaluations=components.forced_robust_fev,
            forced_fast_seed_invoked=components.forced_fast_seed_invoked,
            forced_robust_seed_invoked=components.forced_robust_seed_invoked,
            local_success_probability=components.probability,
            forced_fast_total_samples_ns=components.forced_fast_total_samples_ns,
            forced_robust_total_samples_ns=components.forced_robust_total_samples_ns,
            forced_fast_stage_samples_ns=components.forced_fast_stage_samples_ns,
            forced_robust_stage_samples_ns=components.forced_robust_stage_samples_ns,
            stage_names=np.asarray(STAGE_NAMES, dtype=np.str_),
            selected_exact_runtime_latency_samples_ns=exact_latency,
            selected_exact_runtime_accepted=exact_raw["accepted"],
            selected_exact_runtime_seed_invoked=exact_raw["seed_invoked"],
            selected_exact_runtime_local_attempted=exact_raw["local_attempted"],
            selected_exact_runtime_route=exact_raw["route"],
            primary_mask=primary_mask,
        )
        training_report[robot] = {
            "input_dimension": int(train_labels.features.shape[1]),
            "architecture": [int(train_labels.features.shape[1]), 16, 16, 1],
            "train_count": int(train_role.count), "calibration_count": int(calibration_role.count),
            "train_local_success_rate": float(np.mean(train_labels.local_success)),
            "calibration_local_success_rate": float(np.mean(calibration_labels.local_success)),
            "calibration_metrics": predictor.calibration_metrics(calibration_labels.features, calibration_labels.local_success),
            "numerical_equivalence": equivalence, "test_data_loaded": False,
        }
        threshold_report[robot] = {
            **selection,
            "selected_policy_exact_runtime_check": exact_check,
            "threshold_changed_after_exact_runtime_check": False,
        }
        contexts[robot] = {
            "kinematics": kinematics,
            "policy": frozen_policy,
            "development_hashes": train_hashes | calibration_hashes,
            "train_hashes": train_hashes,
            "calibration_hashes": calibration_hashes,
        }

    seal_payload = {
        "protocol": PROTOCOL, "sealed_utc": _utc(),
        "git_commit": git_commit,
        "git_status_at_start": git_status,
        "roles_loaded_before_seal": ["risk_train_queries", "calibration_queries"],
        "policy_validation_loaded_before_seal": False,
        "test_data_loaded": False,
        "implementation_sources": implementation_sources,
        "source_config": _artifact(source_config_path, relative_to=workspace),
        "pilot_config": _artifact(Path(config["_config_path"]), relative_to=workspace),
        "robots": {
            robot: {
                "model": _artifact(staging / f"{robot}_exact_lite_gate.ts", relative_to=staging),
                "policy": _artifact(staging / f"{robot}_lite_gate_policy.json", relative_to=staging),
                "selected_threshold": float(contexts[robot]["policy"].config.local_success_threshold),
            }
            for robot in config["robots"]
        },
    }
    _write_json(staging / "calibration_seal.json", seal_payload)
    seal_hash = _sha256_file(staging / "calibration_seal.json")

    # Phase B: open policy-validation once, after the immutable seal exists.
    main_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    policy_validation_opened_utc = _utc()
    for robot in config["robots"]:
        robot = str(robot)
        _verify_calibration_seal(staging, seal_payload, seal_hash)
        sealed_model = staging / str(
            seal_payload["robots"][robot]["model"]["path"]
        )
        sealed_policy = staging / str(
            seal_payload["robots"][robot]["policy"]["path"]
        )
        reloaded_policy_config, _ = load_policy(sealed_policy)
        if (
            float(reloaded_policy_config.local_success_threshold)
            != float(seal_payload["robots"][robot]["selected_threshold"])
        ):
            raise RuntimeError(f"{robot} sealed threshold changed")
        reloaded_backend = TorchScriptLiteGateInference(
            load_exact_torchscript(sealed_model, device="cpu"),
            int(training_report[robot]["input_dimension"]),
        )
        contexts[robot]["policy"] = LiteFastGatePolicy(
            reloaded_backend, reloaded_policy_config
        )
        policy_role = _load_development_role(
            workspace=workspace, bulk_root=bulk_root, robot=robot,
            role="policy_validation_queries", training_seed=17, dt=dt,
        )
        if smoke:
            policy_role = policy_role.subset(min(32, policy_role.count))
        source_records[f"{robot}/policy_validation_queries"] = policy_role.source_manifest
        if contexts[robot]["development_hashes"] & set(
            policy_role.query_sha256.astype(str)
        ):
            raise RuntimeError(f"{robot} policy-validation overlaps train/calibration")
        policy_hashes = set(policy_role.query_sha256.astype(str))
        role_audit[robot]["roles"]["policy_validation_queries"] = (
            _query_role_audit(policy_role.query_sha256)
        )
        role_audit[robot]["pairwise_overlap_counts"].update(
            {
                "risk_train_queries__policy_validation_queries": len(
                    contexts[robot]["train_hashes"] & policy_hashes
                ),
                "calibration_queries__policy_validation_queries": len(
                    contexts[robot]["calibration_hashes"] & policy_hashes
                ),
            }
        )
        role_audit[robot]["all_pairwise_disjoint"] = all(
            value == 0
            for value in role_audit[robot]["pairwise_overlap_counts"].values()
        )
        methods = _build_methods(
            source_config=source_config, release_v3_root=release_v3_root,
            release_v4_root=release_v4_root, frozen_v5_root=frozen_v5_root,
            robot=robot, kinematics=contexts[robot]["kinematics"], device=device,
            lite_policy=contexts[robot]["policy"],
        )
        benchmark, events = benchmark_policy_validation(policy_role, methods, config=config, dt=dt)
        quiet_events[robot]["policy_validation"] = events
        _save_benchmark(staging / f"{robot}_policy_validation_records.npz", benchmark)
        rows = summarize(benchmark); main_rows.extend(rows)
        breakdown_rows.extend(latency_breakdown(benchmark))
        family_rows.extend(family_routes(benchmark))
        paired_rows.extend(paired_summary(benchmark, name) for name in config["reporting"]["paired_comparators"])
        goals = config["pilot_goals"]
        gates[robot] = _pilot_gate(benchmark, rows, goals)

    _write_csv(staging / "main_table.csv", main_rows); _write_json(staging / "main_table.json", main_rows)
    _write_csv(staging / "latency_breakdown.csv", breakdown_rows); _write_json(staging / "latency_breakdown.json", breakdown_rows)
    _write_csv(staging / "query_family_routes.csv", family_rows); _write_json(staging / "query_family_routes.json", family_rows)
    _write_json(staging / "paired_latency_summary.json", paired_rows)
    _write_json(staging / "training_report.json", training_report)
    _write_json(staging / "threshold_selection.json", threshold_report)
    _write_json(
        staging / "latency_breakdown_contract.json",
        {
            "stage_names": list(STAGE_NAMES),
            "stage_definitions": STAGE_DEFINITIONS,
            "method_stage_contract": METHOD_STAGE_CONTRACT,
            "outer_timer": (
                "perf_counter_ns from method solve entry through materialized "
                "verified outcome; I/O and serialization excluded"
            ),
            "primary_aggregation": (
                "median across five repeats per query, then cross-query quantile"
            ),
        },
    )
    _write_json(staging / "development_role_audit.json", role_audit)
    gate_payload = {
        "robots": gates, "all_robots_pass": all(row["all_pass"] for row in gates.values()),
        "development_only": True, "fresh_evaluation_started": False,
        "test_data_loaded": False, "policy_validation_used_for_retuning": False,
    }
    _write_json(staging / "pilot_gate.json", gate_payload)
    lines = ["# Hierarchical V5-Lite policy-validation pilot", "", "| robot | method | success | P50 ms | P95 ms | P99 ms | mean FEV | seed invocation |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in main_rows:
        lines.append(f"| {row['robot']} | {row['method']} | {row['verified_success']:.6f} | {row['p50_ms']:.6f} | {row['p95_ms']:.6f} | {row['p99_ms']:.6f} | {row['mean_fev']:.6f} | {row['learned_seed_ensemble_invocation_rate']:.6f} |")
    (staging / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_outputs(main_rows, family_rows, staging, int(config["reporting"]["png_dpi"]))
    _write_json(staging / "environment.json", {
        **environment_payload(),
        "started_utc": started, "completed_utc": _utc(),
        "policy_sealed_utc": seal_payload["sealed_utc"],
        "policy_validation_opened_utc": policy_validation_opened_utc,
        "quiet_events": quiet_events,
    })
    (staging / "hierarchical_v5_lite_pilot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _verify_calibration_seal(staging, seal_payload, seal_hash)
    for relative, descriptor in implementation_sources.items():
        _verify_recorded_artifact(workspace / relative, descriptor)
    git_commit_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status_after = subprocess.run(
        ["git", "status", "--short"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if git_commit_after != git_commit or git_status_after != git_status:
        raise RuntimeError("Git source state changed during V5-Lite pilot")
    protected_after = _tree_snapshot(frozen_v5_root)
    if protected_after != protected_before:
        raise RuntimeError("frozen hierarchical_v5_pilot output changed")
    generated = sorted(path for path in staging.iterdir() if path.is_file())
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete_smoke" if smoke else "complete_policy_validation_pilot",
        "git_commit": git_commit,
        "git_commit_after": git_commit_after,
        "git_status_at_start": git_status,
        "git_status_after": git_status_after,
        "source_state_unchanged_during_run": True,
        "methods": list(METHODS), "stage_names": list(STAGE_NAMES),
        "development_roles": list(ROLE_ORDER), "training_seed": 17,
        "policy_frozen_before_policy_validation": True,
        "policy_validation_used_for_retuning": False,
        "test_data_loaded": False, "formal_test_started": False,
        "calibration_seal_sha256": seal_hash,
        "protected_v5_tree_unchanged": True,
        "protected_v5_tree_file_count": len(protected_before),
        "protected_v5_tree_digest_before": protected_before_digest,
        "protected_v5_tree_digest_after": _snapshot_digest(protected_after),
        "source_config": _artifact(source_config_path, relative_to=workspace),
        "pilot_config": _artifact(Path(config["_config_path"]), relative_to=workspace),
        "implementation_sources": implementation_sources,
        "source_records": source_records,
        "development_role_audit": role_audit,
        "stage_definitions": STAGE_DEFINITIONS,
        "method_stage_contract": METHOD_STAGE_CONTRACT,
        "pilot_gate": gate_payload,
        "artifacts": {path.name: _artifact(path, relative_to=staging) for path in generated},
    }
    _write_json(staging / "run_manifest.json", manifest)
    os.replace(staging, output_root)
    print(f"[v5-lite] completed: {output_root}", flush=True)
    return gate_payload


def main() -> None:
    args = _parser().parse_args()
    run(args.config, smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
