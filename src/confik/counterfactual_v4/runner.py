from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..experiments.provenance import environment_payload
from ..latency_pilot_v3.benchmark import (
    ConstantRiskEngine,
    ProfiledCascadeRuntime,
    query_digest,
    query_from_dataset,
)
from ..latency_pilot_v3.runner import _solver_components
from ..release_v3_locked.artifacts import load_locked_seed_engine
from ..runtime.cascade import EntryAction, FixedEntryGate
from ..test_v3_locked.runner import _release_paths, _sha256_file, _snapshot
from .collector import (
    ACTIONS,
    COLLECTED_ACTIONS,
    collect_query_actions,
    select_pilot_indices,
    summarize_records,
    validate_source_role,
)


FEATURE_NAMES = (
    "learned_seed_position_error",
    "learned_seed_orientation_error",
    "ensemble_uncertainty_mean",
    "ensemble_uncertainty_max",
    "learned_seed_min_singular_value",
    "learned_seed_joint_limit_margin",
    "learned_seed_joint_step_l2",
    "current_pose_position_step",
    "current_pose_orientation_step",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect training/validation-only v4 counterfactual action labels"
    )
    parser.add_argument("--config", required=True)
    return parser


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _safe(record), sort_keys=True, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )


def _selection_seed(release_commit: str, robot: str, training_seed: int, role: str) -> int:
    material = (
        f"counterfactual_v4_pilot|{release_commit}|{robot}|{training_seed}|{role}"
    ).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:4], "big", signed=False)


def _source_dataset_path(
    workspace: Path,
    *,
    robot: str,
    training_seed: int,
    role: str,
) -> Path:
    validated = validate_source_role(role)
    path = (
        workspace
        / "outputs"
        / f"paper_v2_seed{training_seed}"
        / robot
        / "datasets"
        / f"{validated}.npz"
    )
    if path.name != f"{validated}.npz" or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _busy_unrelated_processes(
    *,
    cpu_threshold_percent: float,
    excluded_pids: set[int] | None = None,
) -> list[dict[str, Any]]:
    excluded = {os.getpid(), os.getppid()} | (excluded_pids or set())
    completed = subprocess.run(
        ["ps", "-eo", "pid=,pcpu=,stat=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    busy: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 4:
            continue
        pid = int(fields[0])
        cpu = float(fields[1])
        if pid in excluded or cpu < cpu_threshold_percent:
            continue
        busy.append({"pid": pid, "cpu_percent": cpu, "stat": fields[2], "args": fields[3]})
    return busy


def _wait_for_quiet_environment(
    config: dict[str, Any], *, context: str
) -> dict[str, Any]:
    threshold = float(config["runtime"]["max_unrelated_cpu_percent"])
    stable_required = int(config["runtime"].get("quiet_stable_checks", 2))
    poll_seconds = float(config["runtime"].get("quiet_poll_seconds", 1.0))
    max_wait_seconds = float(config["runtime"].get("max_quiet_wait_seconds", 600.0))
    started = time.monotonic()
    stable = 0
    observed: list[dict[str, Any]] = []
    announced = False
    while stable < stable_required:
        busy = _busy_unrelated_processes(cpu_threshold_percent=threshold)
        if busy:
            stable = 0
            observed = busy
            if not announced:
                print(
                    f"[counterfactual-v4] waiting for quiet host at {context}: {busy}",
                    flush=True,
                )
                announced = True
        else:
            # The fast path must not add a one-second sleep to every query.
            # Consecutive quiet samples are required only after actual
            # interference has been observed.
            if not announced:
                return {
                    "context": context,
                    "wait_seconds": time.monotonic() - started,
                    "had_busy_process": False,
                    "last_busy_processes": [],
                }
            stable += 1
        elapsed = time.monotonic() - started
        if elapsed > max_wait_seconds:
            raise RuntimeError(
                "counterfactual latency collection did not regain a quiet host; "
                f"context={context!r}, waited={elapsed:.1f}s, last_busy={observed}"
            )
        if stable < stable_required:
            time.sleep(poll_seconds)
    elapsed = time.monotonic() - started
    if announced:
        print(
            f"[counterfactual-v4] quiet host restored at {context} after {elapsed:.1f}s",
            flush=True,
        )
    return {
        "context": context,
        "wait_seconds": elapsed,
        "had_busy_process": announced,
        "last_busy_processes": observed,
    }


def _build_runtimes(
    *,
    source_config: dict[str, Any],
    release_root: Path,
    robot: str,
    training_seed: int,
    kinematics: object,
    device: str,
) -> tuple[object, dict[str, ProfiledCascadeRuntime], dict[str, Path]]:
    paths = _release_paths(release_root, robot, training_seed)
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
    risk = ConstantRiskEngine()
    action_values = {
        "easy": EntryAction.EASY,
        "medium": EntryAction.MEDIUM,
        "hard": EntryAction.HARD,
        "fixed_robust": EntryAction.EASY,
    }
    runtimes = {
        name: ProfiledCascadeRuntime(
            name=f"counterfactual_{name}",
            kinematics=kinematics,
            seed_engine=seed_engine,  # type: ignore[arg-type]
            risk_engine=risk,
            gate=FixedEntryGate(action),
            dls=dls,
            verifier=verifier,
            seed_bank=seed_bank,
            fallback=fallback,
            cascade_config=cascade,
            reuse_candidate_features=True,
        )
        for name, action in action_values.items()
    }
    return seed_engine, runtimes, paths


def _warmup(
    runtimes: dict[str, ProfiledCascadeRuntime],
    dataset: QueryDataset,
    indices: np.ndarray,
    *,
    iterations: int,
    dt: float,
) -> None:
    for warmup_index in range(iterations):
        source_index = int(indices[warmup_index % len(indices)])
        query = query_from_dataset(dataset, source_index, dt=dt)
        actions = list(COLLECTED_ACTIONS)
        if warmup_index % 2:
            actions.reverse()
        for action in actions:
            runtimes[action].solve(query)


def _matrix_artifact(
    *,
    path: Path,
    features: list[np.ndarray],
    records: list[dict[str, Any]],
    selected_indices: np.ndarray,
    dataset: QueryDataset,
    nq: int,
) -> None:
    count = len(selected_indices)
    action_index = {action: index for index, action in enumerate(COLLECTED_ACTIONS)}
    success = np.zeros((count, len(COLLECTED_ACTIONS)), dtype=bool)
    deadline = np.zeros_like(success)
    latency_p50 = np.zeros((count, len(COLLECTED_ACTIONS)), dtype=np.float64)
    latency_p95 = np.zeros_like(latency_p50)
    evaluations = np.zeros((count, len(COLLECTED_ACTIONS)), dtype=np.int64)
    fallback = np.zeros_like(success)
    commands = np.full((count, len(COLLECTED_ACTIONS), nq), np.nan, dtype=np.float64)
    failure = np.full((count, len(COLLECTED_ACTIONS)), "", dtype="U256")
    joint_step = np.full((count, len(COLLECTED_ACTIONS)), np.nan, dtype=np.float64)
    joint_velocity = np.full_like(joint_step, np.nan)
    velocity_utilization = np.full_like(joint_step, np.nan)
    query_hashes = np.full(count, "", dtype="U64")
    for row in records:
        query_index = int(row["query_index"])
        action = action_index[str(row["entry_action"])]
        success[query_index, action] = bool(row["verified_success"])
        deadline[query_index, action] = bool(row["verified_success_before_deadline"])
        latency_p50[query_index, action] = float(row["latency_p50_ns"]) / 1e6
        latency_p95[query_index, action] = float(row["latency_p95_ns"]) / 1e6
        evaluations[query_index, action] = int(row["function_evaluations"])
        fallback[query_index, action] = bool(row["fallback_used"])
        failure[query_index, action] = str(row["failure_reason"])
        if row["max_joint_step_rad"] is not None:
            joint_step[query_index, action] = float(row["max_joint_step_rad"])
            joint_velocity[query_index, action] = float(
                row["max_joint_velocity_rad_s"]
            )
            velocity_utilization[query_index, action] = float(
                row["max_velocity_limit_utilization"]
            )
        query_hashes[query_index] = str(row["query_sha256"])
        if row["command_q"] is not None:
            commands[query_index, action] = np.asarray(row["command_q"], dtype=np.float64)
    np.savez_compressed(
        path,
        feature_names=np.asarray(FEATURE_NAMES, dtype=np.str_),
        action_names=np.asarray(COLLECTED_ACTIONS, dtype=np.str_),
        decision_action_names=np.asarray(ACTIONS, dtype=np.str_),
        features=np.asarray(features, dtype=np.float64),
        source_indices=np.asarray(selected_indices, dtype=np.int64),
        query_sha256=query_hashes,
        category=dataset.category[selected_indices],
        expected_reachable=dataset.expected_reachable[selected_indices],
        continuity_feasible=dataset.continuity_feasible[selected_indices],
        verified_success=success,
        verified_success_before_deadline=deadline,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        function_evaluations=evaluations,
        fallback_used=fallback,
        failure_reason=failure,
        command_q=commands,
        max_joint_step_rad=joint_step,
        max_joint_velocity_rad_s=joint_velocity,
        max_velocity_limit_utilization=velocity_utilization,
        max_joint_acceleration_rad_s2=np.full_like(joint_step, np.nan),
        max_joint_jerk_rad_s3=np.full_like(joint_step, np.nan),
        dynamic_history_available=np.zeros_like(success),
    )


def _run_combination(
    *,
    workspace: Path,
    destination: Path,
    source_config: dict[str, Any],
    config: dict[str, Any],
    release_root: Path,
    release_commit: str,
    robot: str,
    training_seed: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    role = validate_source_role(str(config["data"]["source_role"]))
    source_path = _source_dataset_path(
        workspace, robot=robot, training_seed=training_seed, role=role
    )
    dataset = QueryDataset.load(source_path)
    selection_seed = _selection_seed(release_commit, robot, training_seed, role)
    selected = select_pilot_indices(
        dataset, count=int(config["data"]["pilot_count"]), seed=selection_seed
    )
    kinematics = load_robot(source_config, robot)
    seed_engine, runtimes, release_paths = _build_runtimes(
        source_config=source_config,
        release_root=release_root,
        robot=robot,
        training_seed=training_seed,
        kinematics=kinematics,
        device=str(config["runtime"]["device"]),
    )
    dt = float(config["data"]["dt"])
    repeats = int(config["timing"]["repeats"])
    deadline_ms = float(config["timing"]["deadline_ms"])
    _warmup(
        runtimes,
        dataset,
        selected,
        iterations=int(config["timing"]["warmup_iterations"]),
        dt=dt,
    )

    features: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    environment_wait_events: list[dict[str, Any]] = []
    contaminated_query_retries = 0
    started = time.perf_counter()
    for query_index, source_index_value in enumerate(selected):
        check_every = int(config["runtime"]["environment_check_every_queries"])
        if query_index % max(check_every, 1) == 0:
            event = _wait_for_quiet_environment(
                config, context=f"{robot}/seed{training_seed}/query{query_index}/before"
            )
            if event["had_busy_process"]:
                environment_wait_events.append(event)
        source_index = int(source_index_value)
        query = query_from_dataset(dataset, source_index, dt=dt)
        while True:
            feature, rows = collect_query_actions(
                query=query,
                query_index=query_index,
                source_index=source_index,
                dataset=dataset,
                runtimes=runtimes,
                seed_engine=seed_engine,  # type: ignore[arg-type]
                repeats=repeats,
                deadline_ms=deadline_ms,
                order_seed=selection_seed + query_index,
            )
            if query_index % max(check_every, 1) != 0:
                break
            busy_after = _busy_unrelated_processes(
                cpu_threshold_percent=float(
                    config["runtime"]["max_unrelated_cpu_percent"]
                )
            )
            if not busy_after:
                break
            contaminated_query_retries += 1
            print(
                f"[counterfactual-v4] discarding and recollecting {robot}/seed{training_seed} "
                f"query {query_index} after concurrent CPU activity: {busy_after}",
                flush=True,
            )
            event = _wait_for_quiet_environment(
                config, context=f"{robot}/seed{training_seed}/query{query_index}/retry"
            )
            environment_wait_events.append(event)
        query_hash = query_digest(query)
        for row in rows:
            row.update(
                {
                    "robot": robot,
                    "training_seed": training_seed,
                    "source_role": role,
                    "source_query_sha256": query_hash,
                    "query_sha256": query_hash,
                    "risk_features": feature.tolist(),
                }
            )
        features.append(feature)
        records.extend(rows)
        completed = query_index + 1
        if completed % max(int(config["logging"]["progress_every"]), 1) == 0:
            print(
                f"[counterfactual-v4] {robot}/seed{training_seed} "
                f"{completed}/{len(selected)} queries",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    summary = summarize_records(records)
    summary.update(
        {
            "robot": robot,
            "training_seed": training_seed,
            "source_role": role,
            "query_count": len(selected),
            "wall_time_seconds_excluding_warmup_and_writes": elapsed,
            "seconds_per_query_four_collected_actions": elapsed / len(selected),
            "projected_hours_for_120k_queries": elapsed / len(selected) * 120_000 / 3600,
            "contaminated_query_retries": contaminated_query_retries,
            "quiet_wait_event_count": len(environment_wait_events),
            "quiet_wait_seconds": float(
                sum(float(event["wait_seconds"]) for event in environment_wait_events)
            ),
            "quiet_wait_events": environment_wait_events,
            "pilot_only": True,
            "eligible_for_test_claims": False,
        }
    )
    selection_digest = sha256(
        np.ascontiguousarray(selected, dtype=np.int64).tobytes()
    ).hexdigest()
    selection = {
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_role": role,
        "source_query_count": len(dataset),
        "selected_query_count": len(selected),
        "selection_seed": selection_seed,
        "selection_indices_sha256": selection_digest,
        "selected_category_counts": dict(
            sorted(Counter(dataset.category[selected].astype(str).tolist()).items())
        ),
        "test_named_dataset_loaded": False,
        "allowed_role_check_pass": True,
    }
    _write_records(destination / "counterfactual_records.jsonl.gz", records)
    _matrix_artifact(
        path=destination / "counterfactual_labels.npz",
        features=features,
        records=records,
        selected_indices=selected,
        dataset=dataset,
        nq=int(kinematics.nq),
    )
    _write_json(destination / "pilot_summary.json", summary)
    _write_json(destination / "selection_manifest.json", selection)
    _write_json(
        destination / "artifact_manifest.json",
        {
            "release_artifacts": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in release_paths.items()
            },
            "generated_files": {
                path.name: {"sha256": _sha256_file(path), "size": path.stat().st_size}
                for path in sorted(destination.iterdir())
                if path.is_file()
            },
        },
    )
    return summary


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("counterfactual v4 config must be a mapping")
    config["_config_path"] = str(config_path)
    workspace = resolve_path(config, config.get("workspace", ".."))
    source_config = load_config(resolve_path(config, config["source_config"]))
    release_root = resolve_path(config, config["release_root"])
    output_root = resolve_path(config, config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual output: {output_root}")
    incomplete = output_root.with_name(f".{output_root.name}.incomplete.{os.getpid()}")
    if incomplete.exists():
        raise FileExistsError(incomplete)

    release_manifest = json.loads(
        (release_root / "release_manifest.json").read_text(encoding="utf-8")
    )
    equivalence = json.loads(
        (release_root / "release_equivalence.json").read_text(encoding="utf-8")
    )
    if release_manifest["release_status"] != "sealed" or not equivalence["all_six_pass"]:
        raise RuntimeError("v4 labels require the sealed, six-pass exact v3 release")
    validate_source_role(str(config["data"]["source_role"]))
    preflight_environment = _wait_for_quiet_environment(config, context="run_preflight")
    torch.set_num_threads(int(config["runtime"]["intra_op_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["inter_op_threads"]))

    protected_patterns = [
        "paper_v2_*",
        "paper_v2_aggregate",
        "latency_pilot_v3",
        "release_v3_locked",
        "test_v3_seed*",
        "test_v3_aggregate",
    ]
    protected_before = _snapshot(workspace / "outputs", protected_patterns)
    incomplete.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    try:
        for robot in config["robots"]:
            for training_seed in config["training_seeds"]:
                key = f"{robot}/seed{int(training_seed)}"
                print(f"[counterfactual-v4] starting {key}", flush=True)
                summaries[key] = _run_combination(
                    workspace=workspace,
                    destination=incomplete / robot / f"seed{int(training_seed)}",
                    source_config=source_config,
                    config=config,
                    release_root=release_root,
                    release_commit=str(release_manifest["git_commit"]),
                    robot=str(robot),
                    training_seed=int(training_seed),
                )
        protected_after = _snapshot(workspace / "outputs", protected_patterns)
        if protected_before != protected_after:
            raise RuntimeError("a frozen v2/v3 output changed during v4 pilot collection")
        environment = environment_payload()
        environment.update(
            {
                "captured_utc": _utc(),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "python_executable": os.sys.executable,
            }
        )
        _write_json(incomplete / "environment.json", environment)
        _write_json(incomplete / "pilot_aggregate_summary.json", summaries)
        _write_json(
            incomplete / "run_manifest.json",
            {
                "protocol": "counterfactual_v4_training_validation_pilot_v2",
                "label_schema_version": 2,
                "created_utc": _utc(),
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "runner_sha256": _sha256_file(Path(__file__)),
                "release_root": str(release_root),
                "release_manifest_sha256": _sha256_file(
                    release_root / "release_manifest.json"
                ),
                "release_equivalence_sha256": _sha256_file(
                    release_root / "release_equivalence.json"
                ),
                "release_commit": release_manifest["git_commit"],
                "source_role": config["data"]["source_role"],
                "decision_actions": list(ACTIONS),
                "collected_actions": list(COLLECTED_ACTIONS),
                "action_semantics": {
                    "easy": "enter easy and escalate medium then hard on failure",
                    "medium": "enter medium and escalate hard on failure",
                    "hard": "enter frozen robust hard stage",
                    "fixed_robust": "audit reference; enter easy and execute the full frozen escalation semantics",
                },
                "point_dynamic_diagnostic_contract": {
                    "joint_step": "max absolute wrapped joint displacement from q_(t-1) to accepted command",
                    "joint_velocity": "joint displacement divided by dt",
                    "acceleration_and_jerk": "null for independent point queries without q_(t-2)/q_(t-3); computed only in sequential trajectory evaluation",
                },
                "test_data_loaded": False,
                "test_v3_used_for_parameter_selection": False,
                "pilot_only": True,
                "eligible_for_formal_claims": False,
                "protected_outputs_unchanged": True,
                "preflight_environment": preflight_environment,
                "protected_before": protected_before,
                "protected_after": protected_after,
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "git_status_at_collection": subprocess.run(
                    ["git", "status", "--short"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines(),
            },
        )
        os.replace(incomplete, output_root)
    except BaseException as exc:
        _write_json(
            incomplete / "PILOT_INCOMPLETE.json",
            {
                "failed_utc": _utc(),
                "test_data_loaded": False,
                "protected_outputs_before": protected_before,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"[counterfactual-v4] complete output={output_root}", flush=True)


if __name__ == "__main__":
    main()
