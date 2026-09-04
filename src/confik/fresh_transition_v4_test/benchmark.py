"""Frozen-method benchmark for the fresh transition-rich V4 evaluation.

This module has deliberately narrow responsibilities:

* construct exactly the fixed robust cascade, fixed HARD entry, and sealed
  counterfactual CG-HIK V4 runtimes;
* give every method its own numerical solver, verifier, learned seed engine,
  seed bank, fallback solver, and runtime instance;
* execute one method call per frame in a balanced Latin interleaving while
  maintaining an independent closed-loop joint state for every method; and
* retain an NPZ-safe raw record, including an out-of-timing verifier audit of
  every command that a runtime marks as accepted.

No V5/V6/V7 module is imported here.  In particular, this benchmark contains
no temporal state machine, local shortcut, trainable gate, or parameter
selection path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

from ..counterfactual_v4.runtime_v4 import wrap_profiled_runtime
from ..latency_pilot_v3.benchmark import (
    CORE_STAGE_KEYS,
    ConstantRiskEngine,
    ProfiledCascadeRuntime,
    ProfiledOutcome,
    query_digest,
)
from ..latency_pilot_v3.runner import _solver_components
from ..release_v3_locked.artifacts import load_locked_seed_engine
from ..release_v4_locked.artifacts import (
    FrozenV4Policy,
    TorchScriptV4Inference,
    load_exact_v4_predictor,
    load_policy_config,
)
from ..runtime.cascade import EntryAction, FixedEntryGate
from ..solvers.verifier import SolutionVerifier
from ..types import IKQuery, Pose, VerificationResult


METHODS = (
    "fixed_robust_cascade",
    "always_hard",
    "counterfactual_cghik_v4",
)
"""The complete and immutable method registry for the fresh evaluation."""

STAGE_NAMES = (*CORE_STAGE_KEYS, "unattributed_framework_ns")
"""Mutually exhaustive components of the measured outer call latency."""

_V4_METHOD = "counterfactual_cghik_v4"
_HEX64 = frozenset("0123456789abcdef")
_FROZEN_SOURCE_CONFIG_SHA256 = (
    "c579c282ac3b19e10835b2dae1dc9820c3df195daa0c4e446d49c92a323df70a"
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        raise FileNotFoundError(f"{description} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} must be a regular non-symlink file: {resolved}")
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _release_paths(release_root: Path, robot: str, seed: int) -> dict[str, Path]:
    root = Path(release_root) / robot / f"seed{int(seed)}"
    paths = {
        "torchscript": root / "exact_seed_ensemble.ts",
        "forest": root / "hgb_vectorized_parameters.npz",
        "calibration": root / "isotonic_calibration_parameters.npz",
        "normalization": root / "normalization_parameters.npz",
        "solver_metadata": root / "solver_metadata.json",
        "route_thresholds": root / "route_thresholds.json",
        "seed_bank": root / "seed_bank.npz",
        "runtime_spec": root / "runtime_spec.json",
    }
    for name, path in paths.items():
        _require_regular_file(path, description=f"V3 release artifact {name}")
    return paths


def _sync_cuda(enabled: bool) -> None:
    if not enabled:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def latin_method_orders(
    methods: Sequence[str], seed: int
) -> tuple[tuple[str, ...], ...]:
    """Return one seeded cyclic Latin square without executing any method."""

    names = tuple(str(item) for item in methods)
    if not names or len(set(names)) != len(names):
        raise ValueError("Latin methods must be non-empty and unique")
    base = list(names)
    np.random.default_rng(int(seed)).shuffle(base)
    return tuple(
        tuple(base[offset:] + base[:offset]) for offset in range(len(base))
    )


def _verify_v4_runtime_spec(
    release_v4_root: Path,
    *,
    robot: str,
) -> tuple[Path, Path, Path]:
    """Verify and bind the three robot-specific V4 deployment artifacts.

    The artifact manifest binds the predictor, policy, and runtime
    specification so device roles, policy thresholds, reject semantics, and
    the no-retrace requirement cannot silently drift at benchmark time.
    """

    artifact_manifest_path = _require_regular_file(
        release_v4_root / "artifact_manifest.json",
        description="V4 artifact manifest",
    )
    manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("release_status") != "sealed"
        or bool(manifest.get("test_data_loaded", True))
        or not isinstance(manifest.get("files"), Mapping)
    ):
        raise RuntimeError("V4 artifact manifest is not an eligible sealed release")

    relative_paths = (
        f"{robot}/exact_v4_predictor.ts",
        f"{robot}/v4_policy.json",
        f"{robot}/v4_runtime_spec.json",
    )
    resolved: list[Path] = []
    for relative in relative_paths:
        path = _require_regular_file(
            release_v4_root / relative,
            description=f"sealed V4 artifact {relative}",
        )
        descriptor = manifest["files"].get(relative)
        if not isinstance(descriptor, Mapping):
            raise RuntimeError(f"V4 artifact manifest omits {relative}")
        if (
            path.stat().st_size != int(descriptor.get("size", -1))
            or _sha256_file(path) != str(descriptor.get("sha256", ""))
        ):
            raise RuntimeError(f"sealed V4 artifact changed: {path}")
        resolved.append(path)

    predictor_path, policy_path, runtime_spec_path = resolved
    runtime_spec = json.loads(runtime_spec_path.read_text(encoding="utf-8"))
    policy_config, policy_payload = load_policy_config(policy_path)
    expected_runtime_fields = {
        "protocol": "release_v4_locked",
        "robot": robot,
        "backend": "torchscript_exact_v4",
        "gate_device": "cpu",
        "ood_before_command_reject": True,
        "defer_entry": "complete_fixed_robust_cascade_from_easy",
        "command_reject_numerical_solver_budget": 0,
        "torchscript_load_only": True,
        "retrace_allowed": False,
    }
    for key, expected in expected_runtime_fields.items():
        if runtime_spec.get(key) != expected:
            raise RuntimeError(f"sealed V4 runtime field changed: {key}")
    if dict(runtime_spec.get("policy_config", {})) != dict(policy_config.__dict__):
        raise RuntimeError("V4 runtime specification and policy thresholds differ")
    if dict(policy_payload.get("policy_config", {})) != dict(policy_config.__dict__):
        raise RuntimeError("V4 policy payload and loaded thresholds differ")
    return predictor_path, policy_path, runtime_spec_path


def verify_frozen_releases(
    *,
    workspace: Path,
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
) -> dict[str, Any]:
    """Fail closed unless both numerical and V4 deployment releases are sealed."""

    normalized_robot = str(robot).casefold()
    if normalized_robot not in {"panda", "ur5e"}:
        raise ValueError(f"unsupported fresh-evaluation robot: {robot!r}")
    workspace = Path(workspace).resolve()
    release_v3_root = Path(release_v3_root).resolve()
    release_v4_root = Path(release_v4_root).resolve()
    expected_v3_root = (workspace / "outputs" / "release_v3_locked").resolve()
    expected_v4_root = (workspace / "outputs" / "release_v4_locked").resolve()
    if release_v3_root != expected_v3_root:
        raise ValueError(f"release_v3_root must resolve exactly to {expected_v3_root}")
    if release_v4_root != expected_v4_root:
        raise ValueError(f"release_v4_root must resolve exactly to {expected_v4_root}")
    v3_manifest_path = _require_regular_file(
        release_v3_root / "release_manifest.json",
        description="V3 release manifest",
    )
    v3_manifest = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    if (
        v3_manifest.get("release_status") != "sealed"
        or not bool(v3_manifest.get("release_equivalence_all_six_pass", False))
        or bool(v3_manifest.get("test_named_dataset_loaded", True))
    ):
        raise RuntimeError("release_v3_locked is not an eligible sealed release")
    v3_index = {
        (workspace / str(row["path"])).resolve(): row
        for row in v3_manifest.get("artifacts", ())
        if isinstance(row, Mapping) and "path" in row
    }
    v3_paths = _release_paths(release_v3_root, normalized_robot, 17)
    v3_used: dict[str, Any] = {}
    for key in ("torchscript", "normalization", "runtime_spec", "solver_metadata", "seed_bank"):
        original = Path(v3_paths[key])
        path = _require_regular_file(original, description=f"sealed V3 artifact {key}")
        expected = v3_index.get(path)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"V3 release manifest omits {path}")
        actual = _artifact(path)
        if (
            actual["sha256"] != str(expected.get("sha256", ""))
            or actual["size"] != int(expected.get("size", -1))
        ):
            raise RuntimeError(f"sealed V3 artifact changed: {path}")
        v3_used[key] = actual

    v4_manifest_path = _require_regular_file(
        release_v4_root / "release_manifest.json",
        description="V4 release manifest",
    )
    v4_manifest = json.loads(v4_manifest_path.read_text(encoding="utf-8"))
    if (
        v4_manifest.get("release_status") != "sealed"
        or not bool(v4_manifest.get("all_six_validation_runtime_equivalence_pass", False))
        or bool(v4_manifest.get("test_named_dataset_loaded", True))
    ):
        raise RuntimeError("release_v4_locked is not an eligible sealed release")
    v4_artifact_manifest_path = _require_regular_file(
        release_v4_root / "artifact_manifest.json",
        description="V4 artifact manifest",
    )
    if _sha256_file(v4_artifact_manifest_path) != str(
        v4_manifest.get("artifact_manifest_sha256", "")
    ):
        raise RuntimeError("V4 release manifest does not bind its artifact manifest")
    if _sha256_file(v3_manifest_path) != str(
        v4_manifest.get("upstream_v3_release_manifest_sha256", "")
    ):
        raise RuntimeError("V4 release is not bound to the loaded V3 numerical release")
    predictor, policy, runtime_spec = _verify_v4_runtime_spec(
        release_v4_root,
        robot=normalized_robot,
    )
    evidence: dict[str, Any] = {
        "release_v3_manifest": _artifact(v3_manifest_path),
        "release_v3_loaded_artifacts": v3_used,
        "release_v4_manifest": _artifact(v4_manifest_path),
        "release_v4_artifact_manifest": _artifact(v4_artifact_manifest_path),
        "release_v4_loaded_artifacts": {
            "exact_predictor": _artifact(predictor),
            "policy": _artifact(policy),
        },
    }
    evidence["release_v4_loaded_artifacts"] = {
        **evidence["release_v4_loaded_artifacts"],
        "runtime_spec": {
            "path": str(runtime_spec),
            "sha256": _sha256_file(runtime_spec),
            "size": runtime_spec.stat().st_size,
        },
    }
    # Make the three exact deployment paths explicit for a later run seal.
    evidence["fresh_v4_exact_artifacts"] = {
        "predictor": str(predictor),
        "policy": str(policy),
        "runtime_spec": str(runtime_spec),
    }
    return evidence


@dataclass(frozen=True)
class FrozenMethod:
    """One independently constructed verifier-governed method runtime."""

    name: str
    runtime: object
    verifier: SolutionVerifier
    kinematics: object
    dls: object
    seed_engine: object
    seed_bank: object | None = None
    fallback: object | None = None

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        outcome = self.runtime.solve(query)  # type: ignore[attr-defined]
        if not isinstance(outcome, ProfiledOutcome):
            raise TypeError(f"{self.name} returned {type(outcome).__name__}, not ProfiledOutcome")
        return outcome


def _fixed_method(
    *,
    name: str,
    action: EntryAction,
    source_config: dict[str, Any],
    release_v3_root: Path,
    robot: str,
    release_seed: int,
    kinematics: object,
    device: str,
) -> FrozenMethod:
    paths = _release_paths(release_v3_root, robot, release_seed)
    seed_engine = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=paths["torchscript"],
        normalization_path=paths["normalization"],
        runtime_spec_path=paths["runtime_spec"],
        device=device,
    )
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    runtime = ProfiledCascadeRuntime(
        name=name,
        kinematics=kinematics,
        seed_engine=seed_engine,
        risk_engine=ConstantRiskEngine(),
        gate=FixedEntryGate(action),
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
        reuse_candidate_features=True,
    )
    return FrozenMethod(
        name, runtime, verifier, kinematics, dls, seed_engine, seed_bank, fallback
    )


def _v4_method(
    *,
    source_config: dict[str, Any],
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
    release_seed: int,
    kinematics: object,
    device: str,
) -> FrozenMethod:
    predictor_path, policy_path, _ = _verify_v4_runtime_spec(
        release_v4_root,
        robot=robot,
    )
    paths = _release_paths(release_v3_root, robot, release_seed)
    seed_engine = load_locked_seed_engine(
        kinematics=kinematics,
        torchscript_path=paths["torchscript"],
        normalization_path=paths["normalization"],
        runtime_spec_path=paths["runtime_spec"],
        device=device,
    )
    dls, verifier, fallback, seed_bank, cascade = _solver_components(
        source_config,
        {"solver_metadata": paths["solver_metadata"], "seed_bank": paths["seed_bank"]},
        kinematics,
    )
    policy_config, _ = load_policy_config(policy_path)
    policy = FrozenV4Policy(
        TorchScriptV4Inference(load_exact_v4_predictor(predictor_path, device="cpu")),
        policy_config,
    )
    runtime = wrap_profiled_runtime(
        name=_V4_METHOD,
        policy=policy,
        kinematics=kinematics,
        seed_engine=seed_engine,
        dls=dls,
        verifier=verifier,
        seed_bank=seed_bank,
        fallback=fallback,
        cascade_config=cascade,
    )
    return FrozenMethod(
        _V4_METHOD,
        runtime,
        verifier,
        kinematics,
        dls,
        seed_engine,
        seed_bank,
        fallback,
    )


def _assert_independent_methods(methods: Mapping[str, FrozenMethod]) -> None:
    if set(methods) != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError(f"method registry must be exactly {METHODS}")
    if any(methods[name].name != name for name in METHODS):
        raise ValueError("method mapping keys and immutable runtime names differ")
    for field in ("runtime", "verifier", "dls", "seed_engine"):
        identities = [id(getattr(methods[name], field)) for name in METHODS]
        if len(set(identities)) != len(METHODS):
            raise RuntimeError(f"fresh methods unexpectedly share {field} instances")
    for field in ("seed_bank", "fallback"):
        values = [getattr(methods[name], field) for name in METHODS]
        if all(value is not None for value in values) and len(
            {id(value) for value in values}
        ) != len(METHODS):
            raise RuntimeError(f"fresh methods unexpectedly share {field} instances")
    reference_config = methods[METHODS[0]].verifier.config
    if any(methods[name].verifier.config != reference_config for name in METHODS[1:]):
        raise RuntimeError("fresh methods do not share the deterministic verifier contract")

    for name in METHODS:
        method = methods[name]
        base = (
            getattr(method.runtime, "base", method.runtime)
            if name == _V4_METHOD
            else method.runtime
        )
        timed_verifier = getattr(base, "_timed_verifier", None)
        internal = getattr(timed_verifier, "verifier", None)
        if timed_verifier is not None and internal is not method.verifier:
            raise RuntimeError(f"{name} is not governed by its recorded verifier instance")
        cascade = getattr(base, "_cascade", None)
        if cascade is not None:
            bindings = {
                "dls": method.dls,
                "seed_bank": method.seed_bank,
                "fallback": method.fallback,
            }
            for field, expected in bindings.items():
                if getattr(cascade, field, None) is not expected:
                    raise RuntimeError(f"{name} does not own its recorded {field} instance")


def build_frozen_methods(
    *,
    workspace: Path,
    source_config: dict[str, Any],
    release_v3_root: Path,
    release_v4_root: Path,
    robot: str,
    kinematics: object,
    device: str,
    release_seed: int = 17,
) -> dict[str, FrozenMethod]:
    """Build the exact three-method registry from sealed disk artifacts only."""

    normalized_robot = str(robot).casefold()
    if int(release_seed) != 17:
        raise ValueError("the final V4 method must use frozen upstream seed 17")
    if str(device) != "cuda:0":
        raise ValueError("the final sealed seed runtime device must remain cuda:0")
    expected_source = (Path(workspace).resolve() / "configs" / "paper_v2.yaml").resolve()
    configured_source = Path(str(source_config.get("_config_path", ""))).resolve()
    if configured_source != expected_source:
        raise ValueError(f"source_config must be loaded exactly from {expected_source}")
    _require_regular_file(expected_source, description="frozen numerical source config")
    if _sha256_file(expected_source) != _FROZEN_SOURCE_CONFIG_SHA256:
        raise RuntimeError("the frozen paper_v2 numerical/verifier configuration changed")
    verify_frozen_releases(
        workspace=workspace,
        release_v3_root=release_v3_root,
        release_v4_root=release_v4_root,
        robot=normalized_robot,
    )
    methods = {
        "fixed_robust_cascade": _fixed_method(
            name="fixed_robust_cascade",
            action=EntryAction.EASY,
            source_config=source_config,
            release_v3_root=release_v3_root,
            robot=normalized_robot,
            release_seed=release_seed,
            kinematics=kinematics,
            device=device,
        ),
        "always_hard": _fixed_method(
            name="always_hard",
            action=EntryAction.HARD,
            source_config=source_config,
            release_v3_root=release_v3_root,
            robot=normalized_robot,
            release_seed=release_seed,
            kinematics=kinematics,
            device=device,
        ),
        _V4_METHOD: _v4_method(
            source_config=source_config,
            release_v3_root=release_v3_root,
            release_v4_root=release_v4_root,
            robot=normalized_robot,
            release_seed=release_seed,
            kinematics=kinematics,
            device=device,
        ),
    }
    _assert_independent_methods(methods)
    return methods


# A short compatibility alias for runners that use the older pilot naming.
build_methods = build_frozen_methods


def _valid_hash(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX64 for character in value)


@dataclass(frozen=True)
class BenchmarkData:
    """Array-only raw telemetry for one robot's full fresh trajectory suite."""

    robot: str
    method_names: tuple[str, ...]
    stage_names: tuple[str, ...]
    trajectory_order: np.ndarray
    source_query_hash: np.ndarray
    trajectory_uid: np.ndarray
    category: np.ndarray
    time_index: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    latency_ns: np.ndarray
    stage_latency_ns: np.ndarray
    accepted: np.ndarray
    accepted_contract_violation: np.ndarray
    verifier_checked: np.ndarray
    verifier_accepted: np.ndarray
    verifier_position_error: np.ndarray
    verifier_orientation_error: np.ndarray
    verifier_joint_limit_ok: np.ndarray
    verifier_velocity_ok: np.ndarray
    verifier_finite_ok: np.ndarray
    verifier_reasons: np.ndarray
    function_evaluations: np.ndarray
    iterations: np.ndarray
    fallback_used: np.ndarray
    learned_seed_invoked: np.ndarray
    candidate_count: np.ndarray
    entry_action: np.ndarray
    executed_stages: np.ndarray
    reject_reason: np.ndarray
    risk_score: np.ndarray
    risk_probabilities: np.ndarray
    v4_decision_reason: np.ndarray
    v4_eligible_actions: np.ndarray
    v4_predicted_success: np.ndarray
    v4_predicted_p50_ms: np.ndarray
    v4_predicted_p95_ms: np.ndarray
    v4_fail_all_probability: np.ndarray
    v4_ood_score: np.ndarray
    v4_is_ood: np.ndarray
    command_q: np.ndarray
    executed_query_hash: np.ndarray
    method_order_position: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.method_names) != METHODS:
            raise ValueError(f"raw benchmark method schema must be exactly {METHODS}")
        if tuple(self.stage_names) != STAGE_NAMES:
            raise ValueError("raw benchmark stage schema changed")
        count = int(np.asarray(self.source_query_hash).shape[0])
        methods = len(METHODS)
        nq = int(np.asarray(self.command_q).shape[-1])
        vector_fields = (
            "source_query_hash", "trajectory_uid", "category", "time_index",
            "expected_reachable", "continuity_feasible",
        )
        matrix_fields = (
            "latency_ns", "accepted", "accepted_contract_violation",
            "verifier_checked", "verifier_accepted", "verifier_position_error",
            "verifier_orientation_error", "verifier_joint_limit_ok",
            "verifier_velocity_ok", "verifier_finite_ok", "verifier_reasons",
            "function_evaluations", "iterations", "fallback_used",
            "learned_seed_invoked", "candidate_count", "entry_action",
            "executed_stages", "reject_reason", "risk_score",
            "v4_decision_reason", "v4_eligible_actions",
            "v4_fail_all_probability", "v4_ood_score", "v4_is_ood",
            "executed_query_hash", "method_order_position",
        )
        for name in vector_fields:
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have one value per frame")
        for name in matrix_fields:
            if np.asarray(getattr(self, name)).shape != (count, methods):
                raise ValueError(f"{name} must have shape {(count, methods)}")
        if np.asarray(self.stage_latency_ns).shape != (count, methods, len(STAGE_NAMES)):
            raise ValueError("stage_latency_ns has an invalid shape")
        if np.asarray(self.risk_probabilities).shape != (count, methods, 4):
            raise ValueError("risk_probabilities has an invalid shape")
        for name in ("v4_predicted_success", "v4_predicted_p50_ms", "v4_predicted_p95_ms"):
            if np.asarray(getattr(self, name)).shape != (count, methods, 3):
                raise ValueError(f"{name} has an invalid shape")
        if np.asarray(self.command_q).shape != (count, methods, nq) or nq <= 0:
            raise ValueError("command_q has an invalid shape")
        if np.any(np.asarray(self.latency_ns) <= 0):
            raise ValueError("every formal method call must have positive measured latency")
        if np.any(np.asarray(self.stage_latency_ns) < 0) or not np.array_equal(
            np.sum(self.stage_latency_ns, axis=2), self.latency_ns
        ):
            raise ValueError("stage latency must be nonnegative and sum to outer latency")
        positions = np.asarray(self.method_order_position)
        expected_positions = np.arange(methods)
        if any(not np.array_equal(np.sort(row), expected_positions) for row in positions):
            raise ValueError("each frame must contain one complete Latin method order")
        expected_violation = np.asarray(self.accepted) & ~(
            np.asarray(self.verifier_checked) & np.asarray(self.verifier_accepted)
        )
        if not np.array_equal(self.accepted_contract_violation, expected_violation):
            raise ValueError("accepted contract-violation field is inconsistent")
        if any(not _valid_hash(str(value)) for value in self.executed_query_hash.flat):
            raise ValueError("executed query hashes must be lowercase SHA-256 values")
        if any(not str(value) for value in self.source_query_hash.tolist()):
            raise ValueError("source query hashes cannot be empty")
        if len(set(self.trajectory_order.astype(str).tolist())) != len(self.trajectory_order):
            raise ValueError("trajectory_order contains duplicate UIDs")
        if set(self.trajectory_order.astype(str).tolist()) != set(
            self.trajectory_uid.astype(str).tolist()
        ):
            raise ValueError("trajectory_order and frame trajectory UIDs differ")

    def npz_payload(self) -> dict[str, np.ndarray]:
        """Return an ``allow_pickle=False`` compatible serialization payload."""

        payload: dict[str, np.ndarray] = {
            "robot": np.asarray(self.robot, dtype="U16"),
            "method_names": np.asarray(self.method_names, dtype="U64"),
            "stage_names": np.asarray(self.stage_names, dtype="U64"),
        }
        for name in self.__dataclass_fields__:
            if name in {"robot", "method_names", "stage_names"}:
                continue
            payload[name] = np.asarray(getattr(self, name))
        if any(value.dtype == object for value in payload.values()):
            raise TypeError("raw benchmark payload must never require pickle")
        return payload


def _role_groups(role: object) -> tuple[tuple[str, np.ndarray], ...]:
    if hasattr(role, "groups"):
        groups = tuple(
            (str(uid), np.asarray(indices, dtype=np.int64))
            for uid, indices in role.groups()  # type: ignore[attr-defined]
        )
    else:
        order = tuple(str(value) for value in getattr(role, "trajectory_order"))
        uids = np.asarray(getattr(role, "trajectory_uid")).astype(str)
        groups = tuple((uid, np.flatnonzero(uids == uid).astype(np.int64)) for uid in order)
    count = len(role.dataset)  # type: ignore[attr-defined]
    flattened = np.concatenate([indices for _, indices in groups]) if groups else np.empty(0, dtype=np.int64)
    if not groups or not np.array_equal(np.sort(flattened), np.arange(count, dtype=np.int64)):
        raise ValueError("trajectory groups must partition every frame exactly once")
    if len(np.unique(flattened)) != count:
        raise ValueError("trajectory groups contain duplicate frame indices")
    return groups


def _source_query_hashes(role: object, *, dt: float) -> np.ndarray:
    supplied = getattr(role, "source_query_hash", None)
    count = len(role.dataset)  # type: ignore[attr-defined]
    if supplied is not None:
        values = np.asarray(supplied).astype("U128")
        if values.shape != (count,) or np.any(values == ""):
            raise ValueError("role source_query_hash must contain one non-empty value per frame")
        return values
    dataset = role.dataset  # type: ignore[attr-defined]
    return np.asarray(
        [
            query_digest(
                IKQuery(
                    Pose(dataset.target_position[index], dataset.target_rotation[index]),
                    dataset.previous_q[index],
                    dt=dt,
                )
            )
            for index in range(count)
        ],
        dtype="U64",
    )


def _query(role: object, index: int, previous_q: np.ndarray, *, dt: float) -> IKQuery:
    dataset = role.dataset  # type: ignore[attr-defined]
    return IKQuery(
        Pose(dataset.target_position[index], dataset.target_rotation[index]),
        np.asarray(previous_q, dtype=np.float64),
        dt=dt,
    )


def _audit_verifier(
    method: FrozenMethod,
    outcome: ProfiledOutcome,
    query: IKQuery,
) -> tuple[bool, VerificationResult | None]:
    if outcome.q is None:
        return False, None
    return True, method.verifier.check(np.asarray(outcome.q, dtype=np.float64), query)


def _stage_latencies(outcome: ProfiledOutcome, outer_ns: int) -> np.ndarray:
    core = np.asarray(
        [max(int(outcome.timings_ns.get(name, 0)), 0) for name in CORE_STAGE_KEYS],
        dtype=np.int64,
    )
    attributed = int(np.sum(core))
    if attributed > int(outer_ns):
        raise RuntimeError("profiled core stages exceed the measured outer call latency")
    return np.concatenate(
        [core, np.asarray([int(outer_ns) - attributed], dtype=np.int64)]
    )


def _v4_decision(method: FrozenMethod) -> object | None:
    return getattr(method.runtime, "last_decision", None)


def warmup_methods(
    methods: Mapping[str, FrozenMethod],
    role: object,
    *,
    frames: int,
    synchronize_cuda: bool = True,
) -> None:
    """Warm stateless runtimes on a caller-provided, non-fresh role.

    The formal runner is responsible for proving this role is disjoint from
    the fresh identities.  This function is intentionally separate from
    :func:`benchmark_trajectories`, which never performs hidden warmup calls.
    """

    _assert_independent_methods(methods)
    if frames < 0:
        raise ValueError("warmup frame count cannot be negative")
    if frames == 0:
        return
    count = len(role.dataset)  # type: ignore[attr-defined]
    if count <= 0:
        raise ValueError("cannot warm methods on an empty role")
    dt = float(getattr(role, "dt", 0.02))
    for warm_index in range(frames):
        index = warm_index % count
        previous = np.asarray(role.dataset.previous_q[index], dtype=np.float64)  # type: ignore[attr-defined]
        query = _query(role, index, previous, dt=dt)
        order = METHODS if warm_index % 2 == 0 else tuple(reversed(METHODS))
        for name in order:
            _sync_cuda(synchronize_cuda)
            methods[name].solve(query)
            _sync_cuda(synchronize_cuda)


def benchmark_trajectories(
    role: object,
    methods: Mapping[str, FrozenMethod],
    *,
    order_seed: int,
    progress_every: int = 0,
    synchronize_cuda: bool = True,
) -> BenchmarkData:
    """Execute one closed-loop call per method and fresh frame.

    The call ordering is a seeded three-period Latin rotation.  Each method's
    accepted command becomes only that method's next ``previous_q``; a failed
    method holds its own previous state.  There are no repeats, retries,
    previews, or post-outcome method changes.
    """

    _assert_independent_methods(methods)
    dataset = role.dataset  # type: ignore[attr-defined]
    count = len(dataset)
    if count <= 0:
        raise ValueError("fresh trajectory benchmark cannot be empty")
    robot = str(getattr(role, "robot", "")).casefold()
    if not robot:
        raise ValueError("fresh trajectory role must identify its robot")
    dt = float(getattr(role, "dt", 0.02))
    if dt != 0.02:
        raise ValueError("fresh transition evaluation requires dt=0.02 seconds")
    groups = _role_groups(role)
    trajectory_order = np.asarray([uid for uid, _ in groups], dtype="U128")
    source_hash = _source_query_hashes(role, dt=dt)
    trajectory_uid = np.asarray(getattr(role, "trajectory_uid")).astype("U128")
    category = np.asarray(dataset.category).astype("U128")
    time_index = np.asarray(dataset.time_index, dtype=np.int64)
    expected_reachable = np.asarray(dataset.expected_reachable, dtype=bool)
    continuity_feasible = np.asarray(dataset.continuity_feasible, dtype=bool)
    method_count = len(METHODS)
    nq = int(dataset.previous_q.shape[1])

    latency = np.zeros((count, method_count), dtype=np.int64)
    stages = np.zeros((count, method_count, len(STAGE_NAMES)), dtype=np.int64)
    accepted = np.zeros((count, method_count), dtype=bool)
    violation = np.zeros((count, method_count), dtype=bool)
    verifier_checked = np.zeros((count, method_count), dtype=bool)
    verifier_accepted = np.zeros((count, method_count), dtype=bool)
    verifier_position = np.full((count, method_count), np.nan, dtype=np.float64)
    verifier_orientation = np.full((count, method_count), np.nan, dtype=np.float64)
    verifier_joint_limit = np.zeros((count, method_count), dtype=bool)
    verifier_velocity = np.zeros((count, method_count), dtype=bool)
    verifier_finite = np.zeros((count, method_count), dtype=bool)
    verifier_reasons = np.full((count, method_count), "", dtype="U256")
    fev = np.zeros((count, method_count), dtype=np.int64)
    iterations = np.zeros((count, method_count), dtype=np.int64)
    fallback = np.zeros((count, method_count), dtype=bool)
    seed_invoked = np.ones((count, method_count), dtype=bool)
    candidate_count = np.zeros((count, method_count), dtype=np.int64)
    entry = np.full((count, method_count), "", dtype="U32")
    executed = np.full((count, method_count), "", dtype="U128")
    reject = np.full((count, method_count), "", dtype="U128")
    risk_score = np.full((count, method_count), np.nan, dtype=np.float64)
    risk_probabilities = np.full((count, method_count, 4), np.nan, dtype=np.float64)
    decision_reason = np.full((count, method_count), "", dtype="U64")
    eligible_actions = np.full((count, method_count), "", dtype="U64")
    predicted_success = np.full((count, method_count, 3), np.nan, dtype=np.float64)
    predicted_p50 = np.full((count, method_count, 3), np.nan, dtype=np.float64)
    predicted_p95 = np.full((count, method_count, 3), np.nan, dtype=np.float64)
    fail_all = np.full((count, method_count), np.nan, dtype=np.float64)
    ood_score = np.full((count, method_count), np.nan, dtype=np.float64)
    is_ood = np.zeros((count, method_count), dtype=bool)
    command = np.full((count, method_count, nq), np.nan, dtype=np.float64)
    executed_hash = np.full((count, method_count), "", dtype="U64")
    order_position = np.full((count, method_count), -1, dtype=np.int8)
    column_for = {name: index for index, name in enumerate(METHODS)}
    latin = latin_method_orders(METHODS, int(order_seed))

    for trajectory_number, (uid, indices) in enumerate(groups):
        if indices.size == 0:
            raise ValueError(f"trajectory {uid} contains no frames")
        local_time = time_index[indices]
        if not np.array_equal(local_time, np.arange(indices.size, dtype=np.int64)):
            raise ValueError(f"trajectory {uid} is incomplete or out of frame order")
        previous = {
            name: np.asarray(dataset.previous_q[int(indices[0])], dtype=np.float64).copy()
            for name in METHODS
        }
        for offset, raw_index in enumerate(indices):
            index = int(raw_index)
            order = latin[(trajectory_number + offset) % len(latin)]
            if set(order) != set(METHODS):
                raise RuntimeError("Latin schedule is not a complete method permutation")
            for position, name in enumerate(order):
                column = column_for[name]
                method = methods[name]
                query = _query(role, index, previous[name], dt=dt)
                executed_hash[index, column] = query_digest(query)
                order_position[index, column] = position

                _sync_cuda(synchronize_cuda)
                started = perf_counter_ns()
                outcome = method.solve(query)
                _sync_cuda(synchronize_cuda)
                outer_ns = perf_counter_ns() - started
                if outer_ns <= 0:
                    raise RuntimeError("perf_counter_ns returned a non-positive call latency")
                latency[index, column] = outer_ns
                stages[index, column] = _stage_latencies(outcome, outer_ns)
                accepted[index, column] = bool(outcome.accepted)
                fev[index, column] = int(outcome.function_evaluations)
                iterations[index, column] = int(outcome.iterations)
                fallback[index, column] = bool(outcome.fallback_used)
                candidate_count[index, column] = int(outcome.candidate_count)
                entry[index, column] = str(outcome.entry_action)
                executed[index, column] = "|".join(str(value) for value in outcome.executed_stages)
                reject[index, column] = str(outcome.reject_reason)
                risk_score[index, column] = float(outcome.risk_score)
                risk_probabilities[index, column] = np.asarray(
                    outcome.risk_probabilities, dtype=np.float64
                )

                checked, verification = _audit_verifier(method, outcome, query)
                verifier_checked[index, column] = checked
                if verification is not None:
                    verifier_accepted[index, column] = bool(verification.accepted)
                    verifier_position[index, column] = float(verification.position_error)
                    verifier_orientation[index, column] = float(verification.orientation_error)
                    verifier_joint_limit[index, column] = bool(verification.joint_limit_ok)
                    verifier_velocity[index, column] = bool(verification.velocity_ok)
                    verifier_finite[index, column] = bool(verification.finite_ok)
                    verifier_reasons[index, column] = "|".join(verification.reasons)
                violation[index, column] = bool(
                    outcome.accepted
                    and (verification is None or not verification.accepted)
                )

                decision = _v4_decision(method) if name == _V4_METHOD else None
                if decision is not None:
                    decision_reason[index, column] = str(decision.reason)
                    eligible_actions[index, column] = "|".join(decision.eligible_actions)
                    predicted_success[index, column] = np.asarray(
                        decision.predicted_success, dtype=np.float64
                    )
                    predicted_p50[index, column] = np.asarray(
                        decision.predicted_p50_ms, dtype=np.float64
                    )
                    predicted_p95[index, column] = np.asarray(
                        decision.predicted_p95_ms, dtype=np.float64
                    )
                    fail_all[index, column] = float(decision.fail_all_probability)
                    ood_score[index, column] = float(decision.ood_score)
                    is_ood[index, column] = bool(decision.is_ood)

                if outcome.q is not None:
                    q = np.asarray(outcome.q, dtype=np.float64)
                    if q.shape == (nq,):
                        command[index, column] = q
                # A command advances the branch only when the runtime marked
                # it accepted and the independent audit confirms the exact
                # same verifier contract.  Any discrepancy remains recorded
                # as a hard final-gate failure instead of contaminating later
                # frames with an unverified command.
                if (
                    outcome.accepted
                    and outcome.q is not None
                    and verification is not None
                    and verification.accepted
                ):
                    previous[name] = np.asarray(outcome.q, dtype=np.float64).copy()

        if progress_every and (trajectory_number + 1) % progress_every == 0:
            print(
                f"[fresh-transition-v4] {robot} "
                f"{trajectory_number + 1}/{len(groups)} trajectories",
                flush=True,
            )

    if (
        np.any(latency <= 0)
        or np.any(order_position < 0)
        or np.any(executed_hash == "")
        or np.any(entry == "")
    ):
        raise RuntimeError("fresh benchmark raw-record contract is incomplete")

    return BenchmarkData(
        robot=robot,
        method_names=METHODS,
        stage_names=STAGE_NAMES,
        trajectory_order=trajectory_order,
        source_query_hash=source_hash,
        trajectory_uid=trajectory_uid,
        category=category,
        time_index=time_index,
        expected_reachable=expected_reachable,
        continuity_feasible=continuity_feasible,
        latency_ns=latency,
        stage_latency_ns=stages,
        accepted=accepted,
        accepted_contract_violation=violation,
        verifier_checked=verifier_checked,
        verifier_accepted=verifier_accepted,
        verifier_position_error=verifier_position,
        verifier_orientation_error=verifier_orientation,
        verifier_joint_limit_ok=verifier_joint_limit,
        verifier_velocity_ok=verifier_velocity,
        verifier_finite_ok=verifier_finite,
        verifier_reasons=verifier_reasons,
        function_evaluations=fev,
        iterations=iterations,
        fallback_used=fallback,
        learned_seed_invoked=seed_invoked,
        candidate_count=candidate_count,
        entry_action=entry,
        executed_stages=executed,
        reject_reason=reject,
        risk_score=risk_score,
        risk_probabilities=risk_probabilities,
        v4_decision_reason=decision_reason,
        v4_eligible_actions=eligible_actions,
        v4_predicted_success=predicted_success,
        v4_predicted_p50_ms=predicted_p50,
        v4_predicted_p95_ms=predicted_p95,
        v4_fail_all_probability=fail_all,
        v4_ood_score=ood_score,
        v4_is_ood=is_ood,
        command_q=command,
        executed_query_hash=executed_hash,
        method_order_position=order_position,
    )


__all__ = [
    "METHODS",
    "STAGE_NAMES",
    "BenchmarkData",
    "FrozenMethod",
    "benchmark_trajectories",
    "build_frozen_methods",
    "build_methods",
    "verify_frozen_releases",
    "warmup_methods",
]
