#!/usr/bin/env python3
"""Independent, read-only audit for the locked one-shot test_v3 evidence.

The script intentionally does not import the test_v3 reporting implementation.
It recomputes the published summaries from raw JSONL records and writes only to
``docs/audits/test_v3_locked``. Frozen release and test output directories are
never opened for writing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

# Keep the audit runnable without installing the local package.
WORKSPACE_FROM_SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_FROM_SCRIPT / "src"))

from confik.data.datasets import QueryDataset, TransitionDataset


ROBOTS = ("panda", "ur5e")
SEEDS = (17, 29, 43)
PRIMARY_BACKEND = "torchscript_exact"


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def strict_line(line: str, path: Path, line_number: int) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(
            f"non-finite JSON constant {value!r} in {path}:{line_number}"
        )

    value = json.loads(line, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"record is not an object in {path}:{line_number}")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("audit encountered an empty metric population")
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def subset_name(row: dict[str, Any]) -> str:
    if bool(row["closed_loop"]):
        return "trajectory"
    if bool(row["expected_reachable"]) and bool(row["continuity_feasible"]):
        return "point_feasible"
    return "point_rejectable"


def independent_method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    methods = sorted({str(row["method"]) for row in records})
    for method in methods:
        method_rows = [row for row in records if row["method"] == method]
        payload: dict[str, Any] = {}
        for subset in ("point_feasible", "point_rejectable", "trajectory"):
            rows = [row for row in method_rows if subset_name(row) == subset]
            accepted = np.asarray([bool(row["accepted"]) for row in rows], dtype=bool)
            evaluations = np.asarray(
                [float(row["function_evaluations"]) for row in rows], dtype=np.float64
            )
            latencies = np.asarray(
                [float(row["latency_ns"]) / 1e6 for row in rows], dtype=np.float64
            )
            grouped: dict[int, list[bool]] = defaultdict(list)
            route_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
            for row in rows:
                if row["closed_loop"]:
                    trajectory = int(row["trajectory_id"])
                    grouped[trajectory].append(bool(row["accepted"]))
                    route_groups[trajectory].append(
                        (int(row["time_index"]), str(row["entry_action"]))
                    )
            route_switches = 0
            route_transitions = 0
            for values in route_groups.values():
                actions = [action for _, action in sorted(values)]
                route_switches += sum(
                    left != right for left, right in zip(actions[:-1], actions[1:])
                )
                route_transitions += max(len(actions) - 1, 0)
            payload[subset] = {
                "count": len(rows),
                "acceptance_rate": float(np.mean(accepted)),
                "rejection_rate": float(np.mean(~accepted)),
                "mean_function_evaluations": float(np.mean(evaluations)),
                "function_evaluations": distribution(evaluations),
                "latency_ms": distribution(latencies),
                "deadline_miss_rate_20ms": float(np.mean(latencies > 20.0)),
                "trajectory_completion": (
                    float(np.mean([all(values) for values in grouped.values()]))
                    if grouped
                    else None
                ),
                "trajectory_command_spike": (
                    float(np.mean([row["trajectory_command_spike"] for row in rows]))
                    if rows
                    else None
                ),
                "route_switch_count": route_switches if route_groups else None,
                "route_switch_rate": (
                    route_switches / route_transitions if route_transitions else None
                ),
            }
        payload["entry_action_counts"] = dict(
            Counter(str(row["entry_action"]) for row in method_rows)
        )
        result[method] = payload
    return result


def compare_nested(
    observed: Any,
    expected: Any,
    *,
    path: str,
    differences: list[str],
    atol: float = 1e-12,
) -> None:
    if expected is None or observed is None:
        if observed is not expected:
            differences.append(f"{path}: observed={observed!r}, expected={expected!r}")
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            differences.append(f"{path}: observed is not a dict")
            return
        if set(observed) != set(expected):
            differences.append(
                f"{path}: key mismatch observed={sorted(observed)} expected={sorted(expected)}"
            )
            return
        for key in sorted(expected):
            compare_nested(
                observed[key], expected[key], path=f"{path}.{key}",
                differences=differences, atol=atol
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            differences.append(f"{path}: list shape mismatch")
            return
        for index, (left, right) in enumerate(zip(observed, expected)):
            compare_nested(
                left, right, path=f"{path}[{index}]", differences=differences, atol=atol
            )
        return
    if isinstance(expected, bool) or isinstance(expected, str) or isinstance(expected, int):
        if observed != expected:
            differences.append(f"{path}: observed={observed!r}, expected={expected!r}")
        return
    if isinstance(expected, float):
        if not np.isclose(float(observed), expected, rtol=0.0, atol=atol):
            differences.append(f"{path}: observed={observed!r}, expected={expected!r}")
        return
    if observed != expected:
        differences.append(f"{path}: observed={observed!r}, expected={expected!r}")


def audit_manifest(workspace: Path) -> dict[str, Any]:
    root = workspace / "outputs" / "test_v3_aggregate"
    manifest_path = root / "test_v3_final_manifest.json"
    manifest = strict_load(manifest_path)
    mismatches: list[str] = []
    listed: set[str] = set()
    for artifact in manifest["files"]:
        relative = str(artifact["path"])
        listed.add(relative)
        path = workspace / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        if path.stat().st_size != int(artifact["size"]):
            mismatches.append(f"size:{relative}")
        if file_sha256(path) != str(artifact["sha256"]):
            mismatches.append(f"sha256:{relative}")

    roots = [root] + [workspace / "outputs" / f"test_v3_seed{seed}" for seed in SEEDS]
    actual = {
        str(path.relative_to(workspace))
        for directory in roots
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    unlisted = sorted(actual - listed)
    missing_from_disk = sorted(listed - actual)
    prereg = workspace / "outputs/test_v3_aggregate/test_v3_preregistration.json"
    dataset_manifest = workspace / "outputs/test_v3_aggregate/test_v3_dataset_manifest.json"
    return {
        "listed_artifact_count": len(listed),
        "actual_artifact_count_excluding_final_manifest": len(actual),
        "hash_or_size_mismatches": mismatches,
        "unlisted_files": unlisted,
        "listed_but_missing_files": missing_from_disk,
        "preregistration_sha256_matches": (
            file_sha256(prereg) == manifest["preregistration_sha256"]
        ),
        "dataset_manifest_sha256_matches": (
            file_sha256(dataset_manifest) == manifest["dataset_manifest_sha256"]
        ),
        "all_six_natural_exits": bool(manifest["all_six_natural_exits"]),
        "technical_retry_count": int(manifest["technical_retry_count"]),
        "test_set_retuning_performed": bool(manifest["test_set_retuning_performed"]),
        "threshold_or_gate_changes_after_test": bool(
            manifest["threshold_or_gate_changes_after_test"]
        ),
        "outliers_removed": bool(manifest["outliers_removed"]),
        "winsorization_performed": bool(manifest["winsorization_performed"]),
    }


def raw_query_rows(dataset: QueryDataset | TransitionDataset) -> set[bytes]:
    numeric = np.ascontiguousarray(
        np.concatenate(
            (
                dataset.previous_q.reshape(len(dataset), -1),
                dataset.target_position.reshape(len(dataset), -1),
                dataset.target_rotation.reshape(len(dataset), -1),
            ),
            axis=1,
        ),
        dtype=np.float64,
    )
    width = numeric.dtype.itemsize * numeric.shape[1]
    return set(numeric.view(np.dtype((np.void, width))).reshape(-1).tolist())


def audit_datasets(workspace: Path) -> dict[str, Any]:
    manifest = strict_load(
        workspace / "outputs/test_v3_aggregate/test_v3_dataset_manifest.json"
    )
    result: dict[str, Any] = {}
    for robot in ROBOTS:
        declared = manifest["datasets"][robot]
        path = Path(declared["path"])
        dataset = QueryDataset.load(path)
        new_rows = raw_query_rows(dataset)
        overlaps: dict[str, int] = {}
        source_hash_mismatches: list[str] = []
        for role, source in declared["overlap_audit"]["comparison_sources"].items():
            source_path = Path(source["path"])
            if file_sha256(source_path) != source["sha256"]:
                source_hash_mismatches.append(role)
            old = (
                QueryDataset.load(source_path)
                if source["kind"] == "query"
                else TransitionDataset.load(source_path)
            )
            overlaps[role] = len(new_rows & raw_query_rows(old))
            del old
            gc.collect()
        result[robot] = {
            "dataset_sha256_matches": file_sha256(path) == declared["sha256"],
            "row_count": len(dataset),
            "unique_row_count": len(new_rows),
            "within_dataset_duplicates": len(dataset) - len(new_rows),
            "source_hash_mismatches": source_hash_mismatches,
            "nonzero_overlap_counts": {
                role: count for role, count in overlaps.items() if count != 0
            },
            "finite_required_arrays": bool(
                np.all(np.isfinite(dataset.previous_q))
                and np.all(np.isfinite(dataset.target_position))
                and np.all(np.isfinite(dataset.target_rotation))
            ),
            "category_counts": dict(
                sorted(Counter(dataset.category.astype(str).tolist()).items())
            ),
        }
    return result


def selected_run_metrics(
    recomputed: dict[str, Any],
    saved_latency: dict[str, Any],
) -> dict[str, Any]:
    baseline = recomputed["fixed_robust_cascade"]
    proposed = recomputed["proposed_v2"]
    feasible_b = baseline["point_feasible"]
    feasible_p = proposed["point_feasible"]
    reject_p = proposed["point_rejectable"]
    trajectory_p = proposed["trajectory"]
    paired = saved_latency["paired"][f"{PRIMARY_BACKEND}/point_feasible"]
    return {
        "baseline_feasible_latency_ms": {
            key: feasible_b["latency_ms"][key] for key in ("p50", "p95", "p99", "max")
        },
        "proposed_feasible_latency_ms": {
            key: feasible_p["latency_ms"][key] for key in ("p50", "p95", "p99", "max")
        },
        "feasible_p95_ratio": (
            feasible_p["latency_ms"]["p95"] / feasible_b["latency_ms"]["p95"]
        ),
        "paired_feasible_latency_difference_ms": {
            "mean": paired["paired_mean_difference_ms"],
            "median": paired["paired_median_difference_ms"],
            "p95": paired["paired_p95_difference_ms"],
        },
        "baseline_feasible_mean_fev": feasible_b["mean_function_evaluations"],
        "proposed_feasible_mean_fev": feasible_p["mean_function_evaluations"],
        "baseline_feasible_success": feasible_b["acceptance_rate"],
        "proposed_feasible_success": feasible_p["acceptance_rate"],
        "proposed_feasible_deadline_miss_rate_20ms": feasible_p[
            "deadline_miss_rate_20ms"
        ],
        "proposed_rejectable_rejection": reject_p["rejection_rate"],
        "proposed_rejectable_mean_fev": reject_p["mean_function_evaluations"],
        "proposed_rejectable_latency_ms": {
            key: reject_p["latency_ms"][key] for key in ("p50", "p95", "p99", "max")
        },
        "proposed_trajectory_completion": trajectory_p["trajectory_completion"],
        "proposed_trajectory_command_spike": trajectory_p[
            "trajectory_command_spike"
        ],
        "proposed_trajectory_route_switch_rate": trajectory_p["route_switch_rate"],
    }


def audit_run(workspace: Path, robot: str, seed: int) -> dict[str, Any]:
    root = workspace / "outputs" / f"test_v3_seed{seed}" / robot
    raw_path = root / "query_records_v3.jsonl.gz"
    records: list[dict[str, Any]] = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            records.append(strict_line(line, raw_path, line_number))

    protocol = strict_load(root / "protocol_manifest_v3.json")
    saved_summary = strict_load(root / "summary_v3.json")
    saved_latency = strict_load(root / "latency_breakdown_v3.json")
    saved_claim = strict_load(root / "claim_gate_v3.json")
    formal = [
        row
        for row in records
        if row["backend"] in {PRIMARY_BACKEND, "production_comparator"}
    ]
    recomputed = independent_method_summary(formal)
    differences: list[str] = []
    compare_nested(
        recomputed,
        saved_summary["method_metrics"],
        path="method_metrics",
        differences=differences,
    )

    exact = [row for row in records if row["backend"] == PRIMARY_BACKEND]
    # Point queries are identical paired estimands. Closed-loop trajectory queries
    # intentionally inherit each method's own previous command, so their query
    # hashes may diverge after the first differing command and are not pairable.
    exact_points = [row for row in exact if not row["closed_loop"]]
    baseline = {
        (row["split"], int(row["query_index"])): row
        for row in exact_points
        if row["method"] == "fixed_robust_cascade"
    }
    proposed = {
        (row["split"], int(row["query_index"])): row
        for row in exact_points
        if row["method"] == "proposed_v2"
    }
    pair_keys_match = set(baseline) == set(proposed)
    hash_mismatch_count = sum(
        baseline[key]["query_sha256"] != proposed[key]["query_sha256"]
        for key in set(baseline) & set(proposed)
    )
    semantic_mismatch_count = sum(
        (
            baseline[key]["expected_reachable"],
            baseline[key]["continuity_feasible"],
            baseline[key]["category"],
        )
        != (
            proposed[key]["expected_reachable"],
            proposed[key]["continuity_feasible"],
            proposed[key]["category"],
        )
        for key in set(baseline) & set(proposed)
    )
    trajectory_baseline = {
        int(row["query_index"]): row
        for row in exact
        if row["closed_loop"] and row["method"] == "fixed_robust_cascade"
    }
    trajectory_proposed = {
        int(row["query_index"]): row
        for row in exact
        if row["closed_loop"] and row["method"] == "proposed_v2"
    }
    trajectory_hash_difference_count = sum(
        trajectory_baseline[key]["query_sha256"]
        != trajectory_proposed[key]["query_sha256"]
        for key in set(trajectory_baseline) & set(trajectory_proposed)
    )
    return {
        "raw_record_count": len(records),
        "declared_raw_record_count": int(protocol["raw_record_count"]),
        "method_summary_difference_count": len(differences),
        "method_summary_differences_first_20": differences[:20],
        "exact_pair_key_sets_match": pair_keys_match,
        "exact_pair_query_hash_mismatch_count": hash_mismatch_count,
        "exact_pair_semantic_mismatch_count": semantic_mismatch_count,
        "closed_loop_query_hash_difference_count_expected_unpaired": (
            trajectory_hash_difference_count
        ),
        "all_locked_claim_gates_pass": all(
            bool(saved_claim[key])
            for key in (
                "success_gate_pass",
                "rejection_gate_pass",
                "routing_efficiency_gate_pass",
                "reject_efficiency_gate_pass",
                "risk_gate_pass",
                "trajectory_gate_pass",
                "nontriviality_gate_pass",
            )
        ),
        "test_set_retuning_performed": bool(
            saved_claim["test_set_retuning_performed"]
        ),
        "selected_metrics": selected_run_metrics(recomputed, saved_latency),
        "risk_metrics": {
            "fail_auroc": saved_claim["risk"]["fail_auroc"],
            "fail_ece": saved_claim["risk"]["fail_ece"],
            "false_reject_rate": saved_claim["risk"]["policy_test_metrics"][
                "false_reject_rate"
            ],
            "reject_recall": saved_claim["risk"]["policy_test_metrics"][
                "reject_recall"
            ],
        },
        "formal_feasible_latency_gate_pass": (
            saved_claim["point_feasible_p95_latency_ratio"] <= 1.25
        ),
    }


def audit_release(workspace: Path) -> dict[str, Any]:
    root = workspace / "outputs/release_v3_locked"
    manifest = strict_load(root / "release_manifest.json")
    equivalence = strict_load(root / "release_equivalence.json")
    mismatches: list[str] = []
    for artifact in manifest["artifacts"]:
        path = workspace / artifact["path"]
        if not path.is_file():
            mismatches.append(f"missing:{artifact['path']}")
        elif path.stat().st_size != int(artifact["size"]):
            mismatches.append(f"size:{artifact['path']}")
        elif file_sha256(path) != artifact["sha256"]:
            mismatches.append(f"sha256:{artifact['path']}")
    combinations = equivalence["combinations"]
    return {
        "artifact_count": len(manifest["artifacts"]),
        "artifact_mismatches": mismatches,
        "backend": manifest["backend"],
        "release_status": manifest["release_status"],
        "all_six_equivalence_pass": bool(equivalence["all_six_pass"]),
        "combination_count": len(combinations),
        "combination_pass": {
            name: bool(payload["pass"]) for name, payload in combinations.items()
        },
        "max_abs_differences": {
            name: {
                "seed_output": payload["seed_output_max_absolute_error"],
                "risk_probability": payload[
                    "risk_probability_max_absolute_error"
                ],
                "risk_score": payload["risk_score_max_absolute_error"],
                "accepted_command": payload["runtime_records"][
                    "accepted_command_max_abs_error_rad"
                ],
            }
            for name, payload in combinations.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/audits/test_v3_locked/audit_report.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    # Parse all JSON deliverables strictly before interpreting any metric.
    json_paths = sorted(
        list((workspace / "outputs/test_v3_aggregate").glob("*.json"))
        + [
            path
            for seed in SEEDS
            for robot in ROBOTS
            for path in (workspace / "outputs" / f"test_v3_seed{seed}" / robot).glob(
                "*.json"
            )
        ]
        + list((workspace / "outputs/release_v3_locked").glob("*.json"))
    )
    for path in json_paths:
        strict_load(path)

    report: dict[str, Any] = {
        "protocol": "independent_read_only_test_v3_audit_v1",
        "workspace": str(workspace),
        "strict_json_file_count": len(json_paths),
        "audit_scope": {
            "verified": [
                "artifact integrity",
                "dataset uniqueness and exact split overlap",
                "raw-record arithmetic and saved summaries",
                "locked gate calculations",
            ],
            "not_reconstructable_from_saved_artifacts": [
                "historical unrelated CPU/GPU workload during each timed run",
                "historical thermal and frequency state",
            ],
        },
        "release": audit_release(workspace),
        "manifest": audit_manifest(workspace),
        "datasets": audit_datasets(workspace),
        "runs": {},
    }
    for robot in ROBOTS:
        for seed in SEEDS:
            key = f"{robot}/seed{seed}"
            print(f"[audit] recomputing {key}", flush=True)
            report["runs"][key] = audit_run(workspace, robot, seed)
            gc.collect()

    paper_gate = strict_load(
        workspace / "outputs/test_v3_aggregate/paper_gate_v3.json"
    )
    report["paper_gate"] = {
        "observed_run_count": paper_gate["observed_run_count"],
        "all_run_gates_pass": paper_gate["all_run_gates_pass"],
        "effect_direction_consistent": paper_gate["effect_direction_consistent"],
        "paper_gate_pass": paper_gate["paper_gate_pass"],
        "formal_feasible_latency_gate": paper_gate["formal_feasible_latency_gate"],
        "validation_readiness_1_15_used_as_paper_gate": paper_gate[
            "validation_readiness_1_15_used_as_paper_gate"
        ],
    }

    run_values = list(report["runs"].values())
    report["audit_pass"] = bool(
        not report["release"]["artifact_mismatches"]
        and report["release"]["all_six_equivalence_pass"]
        and not report["manifest"]["hash_or_size_mismatches"]
        and not report["manifest"]["unlisted_files"]
        and not report["manifest"]["listed_but_missing_files"]
        and report["manifest"]["preregistration_sha256_matches"]
        and report["manifest"]["dataset_manifest_sha256_matches"]
        and all(
            item["dataset_sha256_matches"]
            and item["within_dataset_duplicates"] == 0
            and not item["source_hash_mismatches"]
            and not item["nonzero_overlap_counts"]
            and item["finite_required_arrays"]
            for item in report["datasets"].values()
        )
        and all(
            run["raw_record_count"] == run["declared_raw_record_count"]
            and run["method_summary_difference_count"] == 0
            and run["exact_pair_key_sets_match"]
            and run["exact_pair_query_hash_mismatch_count"] == 0
            and run["exact_pair_semantic_mismatch_count"] == 0
            and run["all_locked_claim_gates_pass"]
            and not run["test_set_retuning_performed"]
            and run["formal_feasible_latency_gate_pass"]
            for run in run_values
        )
        and report["paper_gate"]["paper_gate_pass"]
    )
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[audit] pass={report['audit_pass']} output={output}")
    if not report["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
