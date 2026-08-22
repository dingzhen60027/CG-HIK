from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from time import perf_counter_ns
from typing import Iterable, Mapping

import numpy as np

from ..data.datasets import QueryDataset
from ..models.risk import ConstantRiskProvider
from ..models.seed import PreviousStateCandidates
from ..runtime.cascade import (
    CascadedHybridIK,
    EntryAction,
    EntryGate,
)
from ..solvers.dls import AdaptiveDLS
from ..solvers.fallback import KDTreeSeedBank, TRFFallbackSolver
from ..solvers.verifier import SolutionVerifier
from ..types import CalibratedRisk, FloatArray, IKQuery, Pose, RiskLevel, SolverPolicy
from .optimized_inference import (
    PreparedCandidates,
    RiskEngine,
    SeedEngine,
    cached_risk_features,
)


STAGE_KEYS = (
    "feature_preparation_ns",
    "numpy_torch_conversion_ns",
    "learned_seed_inference_ns",
    "uncertainty_risk_inference_ns",
    "routing_decision_ns",
    "numerical_solver_ns",
    "verification_ns",
    "logging_serialization_ns",
    "total_end_to_end_ns",
)

CORE_STAGE_KEYS = tuple(
    key for key in STAGE_KEYS if key not in {"logging_serialization_ns", "total_end_to_end_ns"}
)


class ConstantRiskEngine:
    name = "constant_easy"

    def __init__(self):
        self.provider = ConstantRiskProvider(np.array([1.0, 0.0, 0.0, 0.0]))

    def predict(self, features: FloatArray) -> CalibratedRisk:
        return self.provider.predict(features)


class _TimedVerifier:
    def __init__(self, verifier: SolutionVerifier):
        self.verifier = verifier
        self.elapsed_ns = 0

    def reset(self) -> None:
        self.elapsed_ns = 0

    def check(self, q: FloatArray, query: IKQuery):
        started = perf_counter_ns()
        result = self.verifier.check(q, query)
        self.elapsed_ns += perf_counter_ns() - started
        return result


@dataclass
class ProfiledOutcome:
    q: FloatArray | None
    accepted: bool
    entry_action: str
    executed_stages: tuple[str, ...]
    risk_probabilities: FloatArray
    risk_score: float
    function_evaluations: int
    iterations: int
    fallback_used: bool
    verification_reasons: tuple[str, ...]
    reject_reason: str
    candidate_count: int
    timings_ns: dict[str, int]


class ProfiledCascadeRuntime:
    """A timing-only outer shell around the frozen v2 cascade stages."""

    def __init__(
        self,
        *,
        name: str,
        kinematics: object,
        seed_engine: SeedEngine,
        risk_engine: RiskEngine,
        gate: EntryGate,
        dls: AdaptiveDLS,
        verifier: SolutionVerifier,
        seed_bank: KDTreeSeedBank | None,
        fallback: TRFFallbackSolver | None,
        cascade_config: object,
        reuse_candidate_features: bool,
    ):
        self.name = name
        self.kinematics = kinematics
        self.seed_engine = seed_engine
        self.risk_engine = risk_engine
        self.gate = gate
        self.reuse_candidate_features = bool(reuse_candidate_features)
        self._timed_verifier = _TimedVerifier(verifier)
        self._cascade = CascadedHybridIK(
            kinematics,  # type: ignore[arg-type]
            PreviousStateCandidates(),
            ConstantRiskProvider(),
            dls,
            self._timed_verifier,  # type: ignore[arg-type]
            gate=gate,
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade_config,  # type: ignore[arg-type]
        )

    def solve(self, query: IKQuery) -> ProfiledOutcome:
        total_started = perf_counter_ns()
        self._timed_verifier.reset()
        prepared = self.seed_engine.prepare(query)
        timings = dict(prepared.timings_ns)

        started = perf_counter_ns()
        features = cached_risk_features(
            query,
            prepared,
            reuse_best_pose=self.reuse_candidate_features,
        )
        timings["feature_preparation_ns"] += perf_counter_ns() - started

        started = perf_counter_ns()
        risk = self.risk_engine.predict(features)
        timings["uncertainty_risk_inference_ns"] += perf_counter_ns() - started

        started = perf_counter_ns()
        entry = self.gate.choose(risk)
        policy = SolverPolicy(
            entry.risk_level,
            learned_candidates=(0 if entry in {EntryAction.EASY, EntryAction.REJECT} else 1),
            dls_iterations_per_candidate={
                EntryAction.EASY: self._cascade.config.easy_iterations,
                EntryAction.MEDIUM: self._cascade.config.medium_iterations,
                EntryAction.HARD: self._cascade.config.hard_iterations,
                EntryAction.REJECT: 0,
            }[entry],
            include_previous=entry in {EntryAction.EASY, EntryAction.HARD},
            use_fallback=entry == EntryAction.HARD,
        )
        del policy  # Construction is part of the locked routing path, but is not logged in-core.
        timings["routing_decision_ns"] = perf_counter_ns() - started
        timings["numerical_solver_ns"] = 0
        timings["verification_ns"] = 0

        traces = []
        executed: list[str] = []
        fallback_used = False
        accepted = False
        result_q = None
        verification = None
        reject_reason = "confidence_reject" if entry == EntryAction.REJECT else ""
        if entry != EntryAction.REJECT:
            stages = list(range(int(entry), int(EntryAction.HARD) + 1))
            if not self._cascade.config.escalate_on_failure:
                stages = stages[:1]
            for stage_index in stages:
                action = EntryAction(stage_index)
                executed.append(action.name.lower())
                verification_before = self._timed_verifier.elapsed_ns
                started = perf_counter_ns()
                stage = self._cascade.run_stage(query, prepared.candidates, action)
                stage_elapsed = perf_counter_ns() - started
                verification_delta = self._timed_verifier.elapsed_ns - verification_before
                timings["verification_ns"] += verification_delta
                timings["numerical_solver_ns"] += max(stage_elapsed - verification_delta, 0)
                traces.extend(stage.traces)
                fallback_used = fallback_used or stage.fallback_used
                verification = stage.verification
                if stage.accepted:
                    accepted = True
                    result_q = None if stage.q is None else np.asarray(stage.q, dtype=np.float64).copy()
                    break
            if not accepted:
                reject_reason = "all_cascade_stages_failed"

        timings["logging_serialization_ns"] = 0
        timings["total_end_to_end_ns"] = 0
        timings["unattributed_framework_ns"] = 0
        outcome = ProfiledOutcome(
            q=result_q,
            accepted=accepted,
            entry_action=entry.name.lower(),
            executed_stages=tuple(executed),
            risk_probabilities=risk.probabilities.copy(),
            risk_score=risk.score,
            function_evaluations=int(sum(trace.function_evaluations for trace in traces)),
            iterations=int(sum(trace.iterations for trace in traces)),
            fallback_used=fallback_used,
            verification_reasons=tuple(verification.reasons) if verification is not None else (),
            reject_reason=reject_reason,
            candidate_count=len(prepared.candidates.joints),
            timings_ns=timings,
        )
        # Stop only after the core API result is materialized.  Logging record
        # construction and JSON serialization remain deliberately outside.
        total_elapsed = perf_counter_ns() - total_started
        timings["total_end_to_end_ns"] = total_elapsed
        timings["unattributed_framework_ns"] = total_elapsed - sum(
            timings.get(key, 0) for key in CORE_STAGE_KEYS
        )
        return outcome


def query_from_dataset(dataset: QueryDataset, index: int, previous_q: FloatArray | None = None, dt: float = 0.02) -> IKQuery:
    return IKQuery(
        Pose(dataset.target_position[index], dataset.target_rotation[index]),
        dataset.previous_q[index] if previous_q is None else previous_q,
        dt=dt,
    )


def query_digest(query: IKQuery) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(query.previous_q, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(query.target.position, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(query.target.rotation, dtype=np.float64).tobytes())
    digest.update(np.asarray([query.dt], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _solve_with_call_e2e(
    runtime: ProfiledCascadeRuntime, query: IKQuery
) -> ProfiledOutcome:
    """Measure the complete Python call boundary, including result return."""

    started = perf_counter_ns()
    outcome = runtime.solve(query)
    total_elapsed = perf_counter_ns() - started
    outcome.timings_ns["total_end_to_end_ns"] = total_elapsed
    outcome.timings_ns["unattributed_framework_ns"] = total_elapsed - sum(
        outcome.timings_ns.get(key, 0) for key in CORE_STAGE_KEYS
    )
    return outcome


def _median_outcome(outcomes: list[ProfiledOutcome]) -> ProfiledOutcome:
    if not outcomes:
        raise ValueError("at least one timing outcome is required")
    reference = outcomes[-1]
    deterministic = all(
        (
            outcome.accepted,
            outcome.entry_action,
            outcome.function_evaluations,
            outcome.verification_reasons,
        )
        == (
            reference.accepted,
            reference.entry_action,
            reference.function_evaluations,
            reference.verification_reasons,
        )
        for outcome in outcomes
    )
    if not deterministic:
        raise RuntimeError("repeated validation timing changed solver semantics")
    keys = set().union(*(outcome.timings_ns for outcome in outcomes))
    medians = {
        key: int(np.median([outcome.timings_ns.get(key, 0) for outcome in outcomes]))
        for key in keys
    }
    return replace(reference, timings_ns=medians)


def _record_outside_core(
    *,
    robot: str,
    backend: str,
    method: str,
    split: str,
    query_index: int,
    category: str,
    trajectory_id: int,
    time_index: int,
    query: IKQuery,
    expected_reachable: bool,
    continuity_feasible: bool,
    closed_loop: bool,
    order_index: int,
    outcome: ProfiledOutcome,
    kinematics: object,
) -> dict[str, object]:
    started = perf_counter_ns()
    timings = dict(outcome.timings_ns)
    if outcome.accepted and outcome.q is not None:
        joint_delta = np.abs(kinematics.difference(outcome.q, query.previous_q))  # type: ignore[attr-defined]
        joint_step_max = float(np.max(joint_delta))
        command_spike = bool(
            np.any(joint_delta > kinematics.limits.velocity * query.dt + 1e-4)  # type: ignore[attr-defined]
        )
    else:
        joint_step_max = float("nan")
        command_spike = False
    record: dict[str, object] = {
        "robot": robot,
        "backend": backend,
        "method": method,
        "split": split,
        "query_index": int(query_index),
        "query_sha256": query_digest(query),
        "category": category,
        "trajectory_id": int(trajectory_id),
        "time_index": int(time_index),
        "closed_loop": bool(closed_loop),
        "method_order_index": int(order_index),
        "expected_reachable": bool(expected_reachable),
        "continuity_feasible": bool(continuity_feasible),
        "accepted": bool(outcome.accepted),
        "entry_action": outcome.entry_action,
        "executed_stages": list(outcome.executed_stages),
        "risk_probabilities": outcome.risk_probabilities.tolist(),
        "risk_score": float(outcome.risk_score),
        "function_evaluations": int(outcome.function_evaluations),
        "iterations": int(outcome.iterations),
        "fallback_used": bool(outcome.fallback_used),
        "verification_reasons": list(outcome.verification_reasons),
        "reject_reason": outcome.reject_reason,
        "candidate_count": int(outcome.candidate_count),
        "command_q": None if outcome.q is None else outcome.q.tolist(),
        "joint_step_max": joint_step_max,
        "trajectory_command_spike": command_spike,
        "timings_ns": timings,
    }
    # Deliberately exercise object creation and JSON serialization after the
    # core timing interval.  There is still no disk I/O here.
    json.dumps(record, allow_nan=True, separators=(",", ":"))
    logging_ns = perf_counter_ns() - started
    timings["logging_serialization_ns"] = logging_ns
    timings["total_including_serialization_ns"] = (
        timings["total_end_to_end_ns"] + logging_ns
    )
    return record


def warmup_runtimes(
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    iterations: int,
    dt: float,
) -> None:
    if iterations <= 0:
        return
    categories = np.unique(dataset.category)
    per_category = {category: np.flatnonzero(dataset.category == category) for category in categories}
    for warmup_index in range(iterations):
        category = categories[warmup_index % len(categories)]
        indices = per_category[category]
        index = int(indices[(warmup_index // len(categories)) % len(indices)])
        query = query_from_dataset(dataset, index, dt=dt)
        for baseline, proposed in runtimes.values():
            if warmup_index % 2:
                proposed.solve(query)
                baseline.solve(query)
            else:
                baseline.solve(query)
                proposed.solve(query)


def benchmark_points(
    robot: str,
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    repeats: int,
    dt: float,
    order_seed: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    if repeats <= 0:
        raise ValueError("point timing repeats must be positive")
    records: list[dict[str, object]] = []
    order_counts = {
        backend: {"baseline_first": 0, "proposed_first": 0} for backend in runtimes
    }
    backend_names = list(runtimes)
    for index in range(len(dataset)):
        query = query_from_dataset(dataset, index, dt=dt)
        rng = np.random.default_rng(order_seed + index)
        backend_order = list(backend_names)
        rng.shuffle(backend_order)
        for backend in backend_order:
            baseline, proposed = runtimes[backend]
            collected = {"baseline": [], "proposed": []}
            first_order = None
            for repeat in range(repeats):
                proposed_first = bool((index + repeat) % 2)
                if proposed_first:
                    pair = (("proposed", proposed), ("baseline", baseline))
                    order_counts[backend]["proposed_first"] += 1
                else:
                    pair = (("baseline", baseline), ("proposed", proposed))
                    order_counts[backend]["baseline_first"] += 1
                if first_order is None:
                    first_order = {name: order for order, (name, _) in enumerate(pair)}
                for method, runtime in pair:
                    collected[method].append(_solve_with_call_e2e(runtime, query))
            for method in ("baseline", "proposed"):
                outcome = _median_outcome(collected[method])
                records.append(
                    _record_outside_core(
                        robot=robot,
                        backend=backend,
                        method=method,
                        split="risk_validation_points",
                        query_index=index,
                        category=str(dataset.category[index]),
                        trajectory_id=int(dataset.trajectory_id[index]),
                        time_index=int(dataset.time_index[index]),
                        query=query,
                        expected_reachable=bool(dataset.expected_reachable[index]),
                        continuity_feasible=bool(dataset.continuity_feasible[index]),
                        closed_loop=False,
                        order_index=int(first_order[method]),
                        outcome=outcome,
                        kinematics=(baseline if method == "baseline" else proposed).kinematics,
                    )
                )
    return records, order_counts


def benchmark_trajectories(
    robot: str,
    runtimes: Mapping[str, tuple[ProfiledCascadeRuntime, ProfiledCascadeRuntime]],
    dataset: QueryDataset,
    *,
    dt: float,
    order_seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    states: dict[tuple[str, str, int], np.ndarray] = {}
    backend_names = list(runtimes)
    for index in range(len(dataset)):
        trajectory_id = int(dataset.trajectory_id[index])
        rng = np.random.default_rng(order_seed + 100_000 + index)
        backend_order = list(backend_names)
        rng.shuffle(backend_order)
        for backend in backend_order:
            baseline, proposed = runtimes[backend]
            proposed_first = bool(index % 2)
            pair = (
                (("proposed", proposed), ("baseline", baseline))
                if proposed_first
                else (("baseline", baseline), ("proposed", proposed))
            )
            for order_index, (method, runtime) in enumerate(pair):
                state_key = (backend, method, trajectory_id)
                previous = states.get(state_key, dataset.previous_q[index])
                query = query_from_dataset(dataset, index, previous, dt=dt)
                outcome = _solve_with_call_e2e(runtime, query)
                if outcome.accepted and outcome.q is not None:
                    states[state_key] = outcome.q.copy()
                else:
                    states[state_key] = np.asarray(previous, dtype=np.float64).copy()
                records.append(
                    _record_outside_core(
                        robot=robot,
                        backend=backend,
                        method=method,
                        split="seed_validation_trajectories",
                        query_index=index,
                        category=str(dataset.category[index]),
                        trajectory_id=trajectory_id,
                        time_index=int(dataset.time_index[index]),
                        query=query,
                        expected_reachable=True,
                        continuity_feasible=True,
                        closed_loop=True,
                        order_index=order_index,
                        outcome=outcome,
                        kinematics=runtime.kinematics,
                    )
                )
    return records


def distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def latency_breakdown_summary(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for record in records:
        feasible = bool(record["expected_reachable"]) and bool(record["continuity_feasible"])
        subset = "trajectory" if bool(record["closed_loop"]) else ("point_feasible" if feasible else "point_rejectable")
        keys = [
            (str(record["robot"]), str(record["backend"]), str(record["method"]), subset, "all"),
            (
                str(record["robot"]),
                str(record["backend"]),
                str(record["method"]),
                subset,
                str(record["category"]),
            ),
        ]
        for key in keys:
            grouped.setdefault(key, []).append(record)
    result: dict[str, object] = {}
    for (robot, backend, method, subset, category), rows in sorted(grouped.items()):
        group_key = f"{robot}/{backend}/{method}/{subset}/{category}"
        stage_payload: dict[str, object] = {}
        timing_names = tuple(STAGE_KEYS) + ("total_including_serialization_ns", "unattributed_framework_ns")
        for timing_name in timing_names:
            stage_payload[timing_name.removesuffix("_ns") + "_ms"] = distribution_summary(
                float(row["timings_ns"].get(timing_name, 0)) / 1e6  # type: ignore[union-attr]
                for row in rows
            )
        result[group_key] = stage_payload
    return result


def paired_latency_summary(records: list[dict[str, object]]) -> dict[str, object]:
    point_rows = [record for record in records if not bool(record["closed_loop"])]
    paired: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for record in point_rows:
        feasible = bool(record["expected_reachable"]) and bool(record["continuity_feasible"])
        subset = "point_feasible" if feasible else "point_rejectable"
        key = (str(record["robot"]), str(record["backend"]), subset)
        query_key = str(record["query_index"])
        paired.setdefault(key, {}).setdefault(query_key, {})[str(record["method"])] = record
    summaries: dict[str, object] = {}
    for (robot, backend, subset), query_rows in sorted(paired.items()):
        baseline: list[float] = []
        proposed: list[float] = []
        differences: list[float] = []
        mismatched_hashes = 0
        for pair in query_rows.values():
            if set(pair) != {"baseline", "proposed"}:
                raise RuntimeError("paired latency record is missing baseline or proposed")
            b = pair["baseline"]
            p = pair["proposed"]
            if b["query_sha256"] != p["query_sha256"]:
                mismatched_hashes += 1
            b_ms = float(b["timings_ns"]["total_end_to_end_ns"]) / 1e6  # type: ignore[index]
            p_ms = float(p["timings_ns"]["total_end_to_end_ns"]) / 1e6  # type: ignore[index]
            baseline.append(b_ms)
            proposed.append(p_ms)
            differences.append(p_ms - b_ms)
        if mismatched_hashes:
            raise RuntimeError(
                f"paired latency inputs differ for {robot}/{backend}/{subset}: "
                f"{mismatched_hashes} query hashes"
            )
        b_summary = distribution_summary(baseline)
        p_summary = distribution_summary(proposed)
        summaries[f"{robot}/{backend}/{subset}"] = {
            "baseline_ms": b_summary,
            "proposed_ms": p_summary,
            "paired_difference_ms_proposed_minus_baseline": distribution_summary(differences),
            "p95_ratio_proposed_over_baseline": float(p_summary["p95"]) / max(float(b_summary["p95"]), 1e-12),
            "query_hash_mismatch_count": mismatched_hashes,
        }
    return summaries


def compare_backends(
    records: list[dict[str, object]],
    *,
    reference_backend: str,
    candidate_backend: str,
    method: str = "proposed",
) -> dict[str, object]:
    result: dict[str, object] = {}
    robots = sorted({str(record["robot"]) for record in records})
    for robot in robots:
        rows = [
            record
            for record in records
            if str(record["robot"]) == robot
            and str(record["method"]) == method
            and not bool(record["closed_loop"])
            and bool(record["expected_reachable"])
            and bool(record["continuity_feasible"])
        ]
        by_key = {
            (str(record["backend"]), int(record["query_index"])): record for record in rows
        }
        reference_ids = {
            query_index for backend, query_index in by_key if backend == reference_backend
        }
        candidate_ids = {
            query_index for backend, query_index in by_key if backend == candidate_backend
        }
        if reference_ids != candidate_ids:
            raise RuntimeError(
                f"backend latency inputs are incomplete for {robot}: "
                f"reference_only={len(reference_ids - candidate_ids)}, "
                f"candidate_only={len(candidate_ids - reference_ids)}"
            )
        reference: list[float] = []
        candidate: list[float] = []
        differences: list[float] = []
        ids = sorted(reference_ids)
        for query_index in ids:
            ref = by_key[(reference_backend, query_index)]
            cand = by_key[(candidate_backend, query_index)]
            if ref["query_sha256"] != cand["query_sha256"]:
                raise RuntimeError(
                    f"backend latency query hash differs for {robot} query {query_index}"
                )
            ref_ms = float(ref["timings_ns"]["total_end_to_end_ns"]) / 1e6  # type: ignore[index]
            cand_ms = float(cand["timings_ns"]["total_end_to_end_ns"]) / 1e6  # type: ignore[index]
            reference.append(ref_ms)
            candidate.append(cand_ms)
            differences.append(cand_ms - ref_ms)
        ref_summary = distribution_summary(reference)
        cand_summary = distribution_summary(candidate)
        result[robot] = {
            "reference_backend": reference_backend,
            "candidate_backend": candidate_backend,
            "reference_ms": ref_summary,
            "candidate_ms": cand_summary,
            "paired_difference_ms_candidate_minus_reference": distribution_summary(differences),
            "p95_reduction": 1.0 - float(cand_summary["p95"]) / max(float(ref_summary["p95"]), 1e-12),
            "paired_query_count": len(ids),
            "query_hash_mismatch_count": 0,
        }
    return result
