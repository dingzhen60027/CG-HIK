#!/usr/bin/env python3
"""Extract manuscript evidence without mutating frozen experiment outputs."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
OUTPUTS = ROOT / "outputs"
SOURCE_DATA = PAPER_DIR / "source_data"
GENERATED = PAPER_DIR / "generated"
SEEDS = (17, 29, 43)
ROBOTS = ("panda", "ur5e")

METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "threshold_guard_cascade",
    "ablation_no_history",
    "ablation_single_member",
    "ablation_no_uncertainty",
    "ablation_uncalibrated",
    "ablation_no_reject",
    "ablation_no_fallback",
    "ablation_fixed_damping",
)
COMPARATOR_METHODS = (
    "dls_previous_1x50",
    "trf_previous",
    "learned_1x25",
    "threshold_guard_cascade",
    "fixed_robust_cascade",
    "proposed_v2",
)
COMPARATOR_LABELS = {
    "dls_previous_1x50": "Previous-state DLS",
    "trf_previous": "Previous-state TRF",
    "learned_1x25": "Learned seed + fixed DLS",
    "threshold_guard_cascade": "Cartesian-step guard",
    "fixed_robust_cascade": "Fixed robust cascade",
    "proposed_v2": "CG-HIK",
}
FEASIBLE_CATEGORIES = (
    "id",
    "near_singular",
    "near_limit",
    "workspace_boundary",
    "hard_valid",
)
REJECTABLE_CATEGORIES = ("large_step", "unreachable")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted(method: dict[str, Any], categories: Iterable[str], key: str) -> float:
    rows = [method[name] for name in categories]
    count = sum(int(row["count"]) for row in rows)
    return sum(float(row[key]) * int(row["count"]) for row in rows) / count


def method_summary(method: dict[str, Any]) -> dict[str, float]:
    feasible_success = weighted(method, FEASIBLE_CATEGORIES, "acceptance_rate")
    feasible_fev = weighted(method, FEASIBLE_CATEGORIES, "mean_function_evaluations")
    reject_acceptance = weighted(method, REJECTABLE_CATEGORIES, "acceptance_rate")
    reject_fev = weighted(method, REJECTABLE_CATEGORIES, "mean_function_evaluations")
    return {
        "feasible_success": feasible_success,
        "feasible_mean_fev": feasible_fev,
        "rejectable_rejection": 1.0 - reject_acceptance,
        "rejectable_mean_fev": reject_fev,
        "trajectory_completion": float(method["all"]["trajectory_completion_rate"]),
        "trajectory_command_spike": float(method["all"]["trajectory_spike_rate"]),
    }


def comparator_latency(path: Path) -> dict[str, dict[str, float]]:
    """Stream the frozen query records and recover point-stratum tail latency."""
    values: dict[str, dict[str, list[float]]] = {
        method: {"feasible": [], "rejectable": []} for method in COMPARATOR_METHODS
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            method = str(row["method"])
            if method not in values or bool(row["closed_loop"]):
                continue
            feasible = bool(row["expected_reachable"]) and bool(row["continuity_feasible"])
            stratum = "feasible" if feasible else "rejectable"
            values[method][stratum].append(1000.0 * float(row["latency_seconds"]))
    result: dict[str, dict[str, float]] = {}
    for method, strata in values.items():
        if not strata["feasible"] or not strata["rejectable"]:
            raise RuntimeError(f"Missing comparator latency stratum for {method}: {path}")
        result[method] = {
            "feasible_p95_ms": float(np.percentile(strata["feasible"], 95)),
            "rejectable_p95_ms": float(np.percentile(strata["rejectable"], 95)),
        }
    return result


def tex_escape_percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}\\%"


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    source_manifest: dict[str, dict[str, Any]] = {}

    primary_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    ablation_run_rows: list[dict[str, Any]] = []
    comparator_run_rows: list[dict[str, Any]] = []

    def register(path: Path) -> None:
        resolved = path.resolve()
        if OUTPUTS.resolve() not in resolved.parents:
            raise RuntimeError(f"Evidence source is outside frozen outputs: {path}")
        relative = str(path.relative_to(ROOT))
        source_manifest[relative] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    for seed in SEEDS:
        for robot in ROBOTS:
            result_dir = OUTPUTS / f"paper_v2_seed{seed}" / robot / "results"
            gate_path = result_dir / "claim_gate_v2.json"
            risk_path = result_dir / "risk_metrics.json"
            summary_path = result_dir / "summary_v2.json"
            cluster_path = result_dir / "cluster_statistics_v2.json"
            query_path = result_dir / "query_results_v2.jsonl"
            for path in (gate_path, risk_path, summary_path, cluster_path, query_path):
                register(path)

            gate = load_json(gate_path)
            for label, key in (("Fixed robust cascade", "baseline"), ("Proposed", "proposed")):
                row = gate[key]
                primary_rows.append(
                    {
                        "robot": robot,
                        "training_seed": seed,
                        "method": label,
                        "feasible_count": row["point_feasible_count"],
                        "feasible_success": row["point_feasible_success"],
                        "feasible_mean_fev": row["point_feasible_mean_function_evaluations"],
                        "feasible_p95_ms": 1000.0 * row["point_feasible_p95_latency_seconds"],
                        "rejectable_count": row["point_rejectable_count"],
                        "rejectable_rejection": row["point_rejectable_rejection"],
                        "rejectable_mean_fev": row["point_rejectable_mean_function_evaluations"],
                        "rejectable_p95_ms": 1000.0 * row["point_rejectable_p95_latency_seconds"],
                        "trajectory_count": row["trajectory_count"],
                        "trajectory_completion": row["trajectory_completion_rate"],
                        "trajectory_command_spike": row["trajectory_command_spike_rate"],
                        "verification_interceptions": row["verification_interception_count"],
                    }
                )
            policy = gate["policy_test_metrics"]
            risk_rows.append(
                {
                    "robot": robot,
                    "training_seed": seed,
                    "reject_auroc": gate["risk_reject_auroc"],
                    "reject_ece": gate["risk_reject_ece"],
                    "false_reject_rate": policy["false_reject_rate"],
                    "reject_recall": policy["reject_recall"],
                    "nonreject_macro_f1": policy["nonreject_macro_f1"],
                }
            )

            summaries = load_json(summary_path)
            comparator_p95 = comparator_latency(query_path)
            for method in COMPARATOR_METHODS:
                values = method_summary(summaries[method])
                comparator_run_rows.append(
                    {
                        "robot": robot,
                        "training_seed": seed,
                        "method": method,
                        "method_label": COMPARATOR_LABELS[method],
                        **values,
                        "hard_valid_success": float(
                            summaries[method]["hard_valid"]["acceptance_rate"]
                        ),
                        **comparator_p95[method],
                    }
                )
            for method in METHODS:
                values = method_summary(summaries[method])
                ablation_run_rows.append(
                    {"robot": robot, "training_seed": seed, "method": method, **values}
                )

    aggregate_path = OUTPUTS / "paper_v2_aggregate" / "paper_gate_v2.json"
    register(aggregate_path)
    aggregate = load_json(aggregate_path)

    latency_gate_path = OUTPUTS / "latency_pilot_v3" / "validation_gate_v3.json"
    paired_path = OUTPUTS / "latency_pilot_v3" / "paired_latency_summary.json"
    equivalence_path = OUTPUTS / "latency_pilot_v3" / "numerical_equivalence.json"
    breakdown_path = OUTPUTS / "latency_pilot_v3" / "latency_breakdown.json"
    run_manifest_path = OUTPUTS / "latency_pilot_v3" / "run_manifest.json"
    for path in (latency_gate_path, paired_path, equivalence_path, breakdown_path, run_manifest_path):
        register(path)
    latency_gate = load_json(latency_gate_path)
    paired = load_json(paired_path)
    equivalence = load_json(equivalence_path)
    breakdown = load_json(breakdown_path)
    run_manifest = load_json(run_manifest_path)

    latency_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for robot in ROBOTS:
        gate = latency_gate["robots"][robot]
        gate_metrics = gate["metrics"]
        summary = paired["summaries"][f"{robot}/torchscript_exact/point_feasible"]
        latency_rows.append(
            {
                "robot": robot,
                "split": "validation_only",
                "training_seed": 17,
                "backend": run_manifest["selected_backend"],
                "baseline_p50_ms": summary["baseline_ms"]["p50"],
                "baseline_p95_ms": summary["baseline_ms"]["p95"],
                "baseline_p99_ms": summary["baseline_ms"]["p99"],
                "proposed_p50_ms": summary["proposed_ms"]["p50"],
                "proposed_p95_ms": summary["proposed_ms"]["p95"],
                "proposed_p99_ms": summary["proposed_ms"]["p99"],
                "p95_ratio": gate_metrics["feasible_p95_ratio"],
                "p95_reduction_vs_eager": gate_metrics["feasible_p95_reduction_vs_eager"],
                "equivalence_all_pass": gate["numerical_equivalence_pass"],
            }
        )
        stage = breakdown["summaries"][f"{robot}/torchscript_exact/proposed/point_feasible/all"]
        for key in (
            "feature_preparation_ms",
            "numpy_torch_conversion_ms",
            "learned_seed_inference_ms",
            "uncertainty_risk_inference_ms",
            "routing_decision_ms",
            "numerical_solver_ms",
            "verification_ms",
            "unattributed_framework_ms",
        ):
            stage_rows.append(
                {
                    "robot": robot,
                    "stage": key.removesuffix("_ms"),
                    "p50_ms": stage[key]["p50"],
                    "p95_ms": stage[key]["p95"],
                    "mean_ms": stage[key]["mean"],
                    "max_ms": stage[key]["max"],
                }
            )

    ablation_summary_rows: list[dict[str, Any]] = []
    for robot in ROBOTS:
        for method in METHODS:
            selected = [
                row
                for row in ablation_run_rows
                if row["robot"] == robot and row["method"] == method
            ]
            ablation_summary_rows.append(
                {
                    "robot": robot,
                    "method": method,
                    **{
                        key: mean(float(row[key]) for row in selected)
                        for key in (
                            "feasible_success",
                            "feasible_mean_fev",
                            "rejectable_rejection",
                            "rejectable_mean_fev",
                            "trajectory_completion",
                            "trajectory_command_spike",
                        )
                    },
                }
            )

    comparator_summary_rows: list[dict[str, Any]] = []
    for robot in ROBOTS:
        for method in COMPARATOR_METHODS:
            selected = [
                row
                for row in comparator_run_rows
                if row["robot"] == robot and row["method"] == method
            ]
            comparator_summary_rows.append(
                {
                    "robot": robot,
                    "method": method,
                    "method_label": COMPARATOR_LABELS[method],
                    **{
                        key: mean(float(row[key]) for row in selected)
                        for key in (
                            "feasible_success",
                            "hard_valid_success",
                            "feasible_mean_fev",
                            "feasible_p95_ms",
                            "rejectable_rejection",
                            "rejectable_mean_fev",
                            "rejectable_p95_ms",
                            "trajectory_completion",
                        )
                    },
                }
            )

    primary_reduction_rows: list[dict[str, Any]] = []
    for robot in ROBOTS:
        for seed in SEEDS:
            base = next(
                row for row in primary_rows
                if row["robot"] == robot and row["training_seed"] == seed and row["method"] == "Fixed robust cascade"
            )
            prop = next(
                row for row in primary_rows
                if row["robot"] == robot and row["training_seed"] == seed and row["method"] == "Proposed"
            )
            primary_reduction_rows.append(
                {
                    "robot": robot,
                    "training_seed": seed,
                    "feasible_success_difference_pp": 100.0 * (prop["feasible_success"] - base["feasible_success"]),
                    "feasible_fev_reduction": 1.0 - prop["feasible_mean_fev"] / base["feasible_mean_fev"],
                    "feasible_p95_reduction": 1.0 - prop["feasible_p95_ms"] / base["feasible_p95_ms"],
                    "rejectable_fev_reduction": 1.0 - prop["rejectable_mean_fev"] / base["rejectable_mean_fev"],
                    "rejectable_p95_reduction": 1.0 - prop["rejectable_p95_ms"] / base["rejectable_p95_ms"],
                    "trajectory_completion_difference_pp": 100.0 * (prop["trajectory_completion"] - base["trajectory_completion"]),
                }
            )

    write_csv(SOURCE_DATA / "formal_primary_results.csv", primary_rows)
    write_csv(SOURCE_DATA / "formal_primary_reductions.csv", primary_reduction_rows)
    write_csv(SOURCE_DATA / "formal_risk_results.csv", risk_rows)
    write_csv(SOURCE_DATA / "formal_ablation_runs.csv", ablation_run_rows)
    write_csv(SOURCE_DATA / "formal_ablation_summary.csv", ablation_summary_rows)
    write_csv(SOURCE_DATA / "formal_comparator_runs.csv", comparator_run_rows)
    write_csv(SOURCE_DATA / "formal_comparator_summary.csv", comparator_summary_rows)
    write_csv(SOURCE_DATA / "validation_latency_results.csv", latency_rows)
    write_csv(SOURCE_DATA / "validation_latency_stages.csv", stage_rows)

    robot_summary: dict[str, Any] = {}
    for robot in ROBOTS:
        reductions = [row for row in primary_reduction_rows if row["robot"] == robot]
        risks = [row for row in risk_rows if row["robot"] == robot]
        robot_summary[robot] = {
            "feasible_fev_reduction_mean": mean(row["feasible_fev_reduction"] for row in reductions),
            "feasible_p95_reduction_mean": mean(row["feasible_p95_reduction"] for row in reductions),
            "rejectable_fev_reduction_mean": mean(row["rejectable_fev_reduction"] for row in reductions),
            "rejectable_p95_reduction_mean": mean(row["rejectable_p95_reduction"] for row in reductions),
            "risk_auroc_range": [min(row["reject_auroc"] for row in risks), max(row["reject_auroc"] for row in risks)],
            "risk_ece_range": [min(row["reject_ece"] for row in risks), max(row["reject_ece"] for row in risks)],
            "reject_recall_range": [min(row["reject_recall"] for row in risks), max(row["reject_recall"] for row in risks)],
        }

    snapshot = {
        "schema_version": 1,
        "title": "Confidence-Gated Solver Budgeting for Verified Hybrid Inverse Kinematics",
        "formal_test": {
            "robots": list(ROBOTS),
            "training_seeds": list(SEEDS),
            "runtime_test_reused_across_seeds_within_robot": True,
            "per_run_point_queries": 12000,
            "per_run_feasible_point_queries": 10000,
            "per_run_rejectable_point_queries": 2000,
            "per_run_trajectories": 40,
            "per_trajectory_frames": 150,
            "paper_gate_pass": aggregate.get("paper_gate_pass"),
            "robot_summary": robot_summary,
        },
        "latency_pilot_v3": {
            "split": "validation_only",
            "training_seed": 17,
            "selected_backend": run_manifest["selected_backend"],
            "all_robots_pass": latency_gate["all_robots_pass"],
            "numerical_equivalence_all_pass": equivalence["all_pass"],
            "test_v3_started": False,
            "rows": latency_rows,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
    }
    (GENERATED / "evidence_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    panda = robot_summary["panda"]
    ur5e = robot_summary["ur5e"]
    macros = [
        "% Generated from frozen evidence; do not edit by hand.",
        f"\\newcommand{{\\PandaFEVReduction}}{{{tex_escape_percent(100*panda['feasible_fev_reduction_mean'])}}}",
        f"\\newcommand{{\\URFEVReduction}}{{{tex_escape_percent(100*ur5e['feasible_fev_reduction_mean'])}}}",
        f"\\newcommand{{\\PandaRejectLatencyReduction}}{{{tex_escape_percent(100*panda['rejectable_p95_reduction_mean'])}}}",
        f"\\newcommand{{\\URRejectLatencyReduction}}{{{tex_escape_percent(100*ur5e['rejectable_p95_reduction_mean'])}}}",
        f"\\newcommand{{\\PandaFormalLatencyIncrease}}{{{tex_escape_percent(-100*panda['feasible_p95_reduction_mean'])}}}",
        f"\\newcommand{{\\URFormalLatencyIncrease}}{{{tex_escape_percent(-100*ur5e['feasible_p95_reduction_mean'])}}}",
        f"\\newcommand{{\\PandaPilotRatio}}{{{latency_gate['robots']['panda']['metrics']['feasible_p95_ratio']:.3f}}}",
        f"\\newcommand{{\\URPilotRatio}}{{{latency_gate['robots']['ur5e']['metrics']['feasible_p95_ratio']:.3f}}}",
        f"\\newcommand{{\\PandaPilotReduction}}{{{tex_escape_percent(100*latency_gate['robots']['panda']['metrics']['feasible_p95_reduction_vs_eager'])}}}",
        f"\\newcommand{{\\URPilotReduction}}{{{tex_escape_percent(100*latency_gate['robots']['ur5e']['metrics']['feasible_p95_reduction_vs_eager'])}}}",
    ]
    (GENERATED / "paper_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    print(f"Wrote evidence snapshot and {len(source_manifest)} source hashes.")


if __name__ == "__main__":
    main()
