from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "confik-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.datasets import RiskDataset
from ..models.risk import RiskModel

MAIN_METHODS = (
    "dls_previous_1x50",
    "random_multistart_5x25",
    "kdtree_3x25",
    "learned_1x25",
    "learned_3x15",
    "trf_previous",
    "proposed",
)


def _save_figure(figure: plt.Figure, base_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _eligible_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if bool(record["expected_reachable"]) and bool(record["continuity_feasible"])
    ]


def plot_success_computation(records: list[dict[str, object]], output_dir: Path) -> None:
    eligible = _eligible_records(records)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in MAIN_METHODS:
        rows = [record for record in eligible if record["method"] == method]
        if not rows:
            continue
        x = np.mean([float(record["function_evaluations"]) for record in rows])
        y = np.mean([bool(record["accepted"]) for record in rows])
        axis.scatter(x, y, s=55, label=method)
    axis.set_xlabel("Mean function evaluations")
    axis.set_ylabel("Verified success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False)
    _save_figure(figure, output_dir / "success_computation")


def plot_latency_cdf(records: list[dict[str, object]], output_dir: Path) -> None:
    eligible = _eligible_records(records)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in ("dls_previous_1x50", "learned_3x15", "proposed"):
        values = np.sort(
            [1000.0 * float(record["latency_seconds"]) for record in eligible if record["method"] == method]
        )
        if values.size:
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=method)
    axis.set_xlabel("End-to-end latency (ms)")
    axis.set_ylabel("Empirical CDF")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(figure, output_dir / "latency_cdf")


def plot_category_success(records: list[dict[str, object]], output_dir: Path) -> None:
    categories = sorted({str(record["category"]) for record in records})
    matrix = np.full((len(MAIN_METHODS), len(categories)), np.nan)
    for method_index, method in enumerate(MAIN_METHODS):
        for category_index, category in enumerate(categories):
            rows = [
                record
                for record in records
                if record["method"] == method and record["category"] == category
            ]
            if rows:
                matrix[method_index, category_index] = np.mean([bool(record["accepted"]) for record in rows])
    figure, axis = plt.subplots(figsize=(max(7.0, 0.8 * len(categories)), 4.8))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(categories)), categories, rotation=40, ha="right", fontsize=8)
    axis.set_yticks(range(len(MAIN_METHODS)), MAIN_METHODS, fontsize=8)
    figure.colorbar(image, ax=axis, label="Verified success rate")
    _save_figure(figure, output_dir / "category_success")


def plot_reliability(model: RiskModel, dataset: RiskDataset, output_dir: Path, bins: int = 10) -> None:
    probabilities = model.predict_proba(dataset.features)
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    x_values: list[float] = []
    y_values: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            x_values.append(float(np.mean(confidence[mask])))
            y_values.append(float(np.mean(predictions[mask] == dataset.labels[mask])))
    figure, axis = plt.subplots(figsize=(4.6, 4.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="ideal")
    axis.plot(x_values, y_values, marker="o", label="risk model")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Empirical accuracy")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(figure, output_dir / "risk_reliability")


def write_summary_csv(summary: dict[str, dict[str, dict[str, float]]], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for method, category_data in summary.items():
        for category, metrics in category_data.items():
            rows.append({"method": method, "category": category, **metrics})
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_metrics(records: list[dict[str, object]], method: str) -> dict[str, float]:
    rows = [record for record in _eligible_records(records) if record["method"] == method]
    latency = np.array([float(record["latency_seconds"]) for record in rows])
    return {
        "success": float(np.mean([bool(record["accepted"]) for record in rows])),
        "mean_evaluations": float(np.mean([float(record["function_evaluations"]) for record in rows])),
        "p95_latency": float(np.percentile(latency, 95)),
    }


def write_claim_gate(
    records: list[dict[str, object]],
    risk_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    baseline = _method_metrics(records, "learned_3x15")
    proposed = _method_metrics(records, "proposed")
    evaluation_reduction = 1.0 - proposed["mean_evaluations"] / max(baseline["mean_evaluations"], 1e-12)
    latency_reduction = 1.0 - proposed["p95_latency"] / max(baseline["p95_latency"], 1e-12)
    risk = risk_payload["test_metrics"]
    payload = {
        "risk_auroc_pass": bool(risk.get("fail_auroc", float("nan")) >= 0.75),
        "risk_ece_pass": bool(risk.get("ece", float("inf")) <= 0.10),
        "matched_success_gap": proposed["success"] - baseline["success"],
        "mean_evaluation_reduction": evaluation_reduction,
        "p95_latency_reduction": latency_reduction,
        "efficiency_gate_pass": bool(
            abs(proposed["success"] - baseline["success"]) <= 0.01
            and evaluation_reduction >= 0.15
            and latency_reduction >= 0.10
        ),
        "note": "This per-run diagnostic is not a cross-robot or three-repetition paper conclusion.",
    }
    (output_dir / "claim_gate.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    return payload


def generate_report(
    records: list[dict[str, object]],
    summary: dict[str, dict[str, dict[str, float]]],
    risk_model: RiskModel,
    risk_test: RiskDataset,
    risk_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_success_computation(records, output_dir)
    plot_latency_cdf(records, output_dir)
    plot_category_success(records, output_dir)
    plot_reliability(risk_model, risk_test, output_dir)
    write_summary_csv(summary, output_dir)
    return write_claim_gate(records, risk_payload, output_dir)
