#!/usr/bin/env python3
"""Create publication figures from the extracted manuscript source data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
SOURCE = PAPER_DIR / "source_data"
FIGURES = PAPER_DIR / "figures"

COLORS = {
    "baseline": "#7A8793",
    "proposed": "#2A6FBB",
    "panda": "#6B4C9A",
    "ur5e": "#D66B3D",
    "good": "#2E8B74",
    "warning": "#C9862C",
    "reject": "#B64E5A",
    "ink": "#263238",
    "light": "#EDF2F5",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def rows(name: str) -> list[dict[str, Any]]:
    with (SOURCE / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def panel(ax: mpl.axes.Axes, label: str, title: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")


def save(fig: mpl.figure.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax: mpl.axes.Axes, xy: tuple[float, float], width: float, height: float,
             title: str, body: str, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor=COLORS["ink"], linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=COLORS["ink"])
    ax.text(x + width / 2, y + height * 0.29, body, ha="center", va="center",
            fontsize=6.3, color=COLORS["ink"], linespacing=1.2)


def arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color=COLORS["ink"]))


def method_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.02, 0.60), 0.17, 0.24, "Online query",
             "target pose\nprevious joint state", "#E8EEF7")
    draw_box(ax, (0.235, 0.60), 0.18, 0.24, "Proposal",
             "five history-conditioned\nseed members", "#E7E2F2")
    draw_box(ax, (0.46, 0.60), 0.18, 0.24, "Calibrated gate",
             "risk + uncertainty\nroute budget or reject", "#DDEFEA")
    draw_box(ax, (0.69, 0.60), 0.28, 0.24, "Verified solver cascade",
             "DLS entry + escalation\nKD-tree / TRF fallback", "#F5E7D4")
    for x0, x1 in ((0.19, 0.235), (0.415, 0.46), (0.64, 0.69)):
        arrow(ax, (x0, 0.72), (x1, 0.72))

    draw_box(ax, (0.29, 0.16), 0.21, 0.22, "Reject action",
             "zero numerical solve\nhold / refuse command", "#F6DEE2")
    draw_box(ax, (0.60, 0.16), 0.25, 0.22, "Common verifier",
             "pose error + joint limits\nvelocity continuity", "#DEE8ED")
    arrow(ax, (0.55, 0.60), (0.395, 0.38))
    arrow(ax, (0.83, 0.60), (0.725, 0.38))
    ax.text(0.775, 0.46, "candidate", ha="center", va="center", fontsize=6.2)
    ax.text(0.71, 0.09, "accept only verified commands; otherwise escalate or reject",
            ha="center", va="center", fontsize=6.5, color=COLORS["ink"])
    ax.text(0.02, 0.96, "Learned models allocate computation; geometry certifies admissibility",
            fontsize=8.5, fontweight="bold", color=COLORS["ink"], va="top")
    save(fig, "figure1_architecture")


def formal_results_figure() -> None:
    primary = rows("formal_primary_results.csv")
    reductions = rows("formal_primary_reductions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    axes = axes.ravel()
    robot_offsets = {"panda": -0.08, "ur5e": 0.08}
    robot_colors = {"panda": COLORS["panda"], "ur5e": COLORS["ur5e"]}

    ax = axes[0]
    panel(ax, "a", "Feasible-query numerical effort")
    for robot in ("panda", "ur5e"):
        for seed in (17, 29, 43):
            pair = [r for r in primary if r["robot"] == robot and int(r["training_seed"]) == seed]
            pair.sort(key=lambda r: 0 if r["method"] == "Fixed robust cascade" else 1)
            xs = np.array([0, 1], dtype=float) + robot_offsets[robot]
            ys = [f(r, "feasible_mean_fev") for r in pair]
            ax.plot(xs, ys, "o-", color=robot_colors[robot], alpha=0.78, markersize=3.2)
    ax.set_xticks([0, 1], ["Fixed", "Proposed"])
    ax.set_ylabel("Mean function evaluations")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.55)
    ax.plot([], [], "o-", color=COLORS["panda"], label="Panda")
    ax.plot([], [], "o-", color=COLORS["ur5e"], label="UR5e")
    ax.legend(loc="upper right")

    ax = axes[1]
    panel(ax, "b", "Paired feasible-success difference")
    for i, robot in enumerate(("panda", "ur5e")):
        selected = [r for r in reductions if r["robot"] == robot]
        x = np.full(len(selected), i) + np.array([-0.07, 0, 0.07])
        y = [f(r, "feasible_success_difference_pp") for r in selected]
        ax.scatter(x, y, s=24, color=robot_colors[robot], edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(-1.0, color=COLORS["reject"], linestyle="--", linewidth=0.9,
               label="Non-inferiority margin (-1 pp)")
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks([0, 1], ["Panda", "UR5e"])
    ax.set_ylabel("Proposed - fixed (percentage points)")
    ax.set_ylim(-1.15, 0.35)
    ax.legend(loc="lower right")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.55)

    ax = axes[2]
    panel(ax, "c", "Rejectable-query P95 latency reduction")
    for i, robot in enumerate(("panda", "ur5e")):
        selected = [r for r in reductions if r["robot"] == robot]
        x = np.full(len(selected), i) + np.array([-0.07, 0, 0.07])
        y = [100.0 * f(r, "rejectable_p95_reduction") for r in selected]
        ax.scatter(x, y, s=24, color=robot_colors[robot], edgecolor="white", linewidth=0.5)
    ax.set_xticks([0, 1], ["Panda", "UR5e"])
    ax.set_ylabel("Reduction relative to fixed (%)")
    ax.set_ylim(96.5, 100)
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.5)

    ax = axes[3]
    panel(ax, "d", "Formal-test feasible P95 latency (negative result)")
    for i, robot in enumerate(("panda", "ur5e")):
        selected = [r for r in reductions if r["robot"] == robot]
        x = np.full(len(selected), i) + np.array([-0.07, 0, 0.07])
        overhead = [-100.0 * f(r, "feasible_p95_reduction") for r in selected]
        ax.scatter(x, overhead, s=24, color=robot_colors[robot], edgecolor="white", linewidth=0.5)
    ax.axhline(25, color=COLORS["reject"], linestyle="--", linewidth=0.9,
               label="Prespecified +25% limit")
    ax.set_xticks([0, 1], ["Panda", "UR5e"])
    ax.set_ylabel("P95 latency increase over fixed (%)")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.55)

    fig.text(0.5, 0.005,
             "Points are three training-seed sensitivity runs per robot on one locked test set per robot; seeds are not independent test datasets.",
             ha="center", va="bottom", fontsize=6.2, color="#455A64")
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.55, w_pad=1.3)
    save(fig, "figure2_formal_results")


def ablation_figure() -> None:
    run_rows = rows("formal_ablation_runs.csv")
    methods = [
        "fixed_robust_cascade", "proposed_v2", "threshold_guard_cascade",
        "ablation_no_history", "ablation_single_member", "ablation_no_uncertainty",
        "ablation_uncalibrated", "ablation_no_reject", "ablation_no_fallback",
        "ablation_fixed_damping",
    ]
    labels = ["Fixed", "Proposed", "Threshold", "No history", "Single member",
              "No uncertainty", "Uncalibrated", "No reject", "No fallback", "Fixed damping"]
    colors = [COLORS["baseline"], COLORS["proposed"], "#9A8F60"] + ["#B6C1C8"] * 7
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey="row")
    for ax, robot in zip(axes[0], ("panda", "ur5e")):
        for i, method in enumerate(methods):
            selected = [r for r in run_rows if r["robot"] == robot and r["method"] == method]
            values = np.array([f(r, "feasible_mean_fev") for r in selected])
            ax.barh(i, values.mean(), xerr=values.std(ddof=1), color=colors[i],
                    height=0.65, edgecolor="white", error_kw={"elinewidth": 0.7, "capsize": 2})
            ax.scatter(values, np.full(3, i) + np.array([-0.11, 0, 0.11]),
                       s=11, color=COLORS["ink"], alpha=0.75, zorder=3)
        ax.set_title(robot.upper() if robot == "ur5e" else "Panda", fontweight="bold")
        ax.set_xlabel("Feasible-query mean function evaluations")
        ax.grid(axis="x", color="#D8DEE3", linewidth=0.5)
        ax.invert_yaxis()
    axes[0, 0].set_yticks(np.arange(len(labels)), labels)
    axes[0, 1].tick_params(labelleft=False)

    for ax, robot in zip(axes[1], ("panda", "ur5e")):
        for i, method in enumerate(methods):
            selected = [r for r in run_rows if r["robot"] == robot and r["method"] == method]
            values = np.array([f(r, "rejectable_mean_fev") for r in selected])
            ax.barh(i, values.mean(), xerr=values.std(ddof=1), color=colors[i],
                    height=0.65, edgecolor="white", error_kw={"elinewidth": 0.7, "capsize": 2})
            ax.scatter(values, np.full(3, i) + np.array([-0.11, 0, 0.11]),
                       s=11, color=COLORS["ink"], alpha=0.75, zorder=3)
        ax.set_title(robot.upper() if robot == "ur5e" else "Panda", fontweight="bold")
        ax.set_xlabel("Rejectable-query mean function evaluations")
        ax.grid(axis="x", color="#D8DEE3", linewidth=0.5)
        ax.invert_yaxis()
    axes[1, 0].set_yticks(np.arange(len(labels)), labels)
    axes[1, 1].tick_params(labelleft=False)
    for ax, label in zip(axes.ravel(), ("a", "b", "c", "d")):
        ax.text(-0.18 if ax in axes[:, 0] else -0.10, 1.04, label,
                transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.suptitle("Ablations separate routing efficiency from mostly null learned-component removals",
                 x=0.08, ha="left", fontsize=8.3, fontweight="bold")
    fig.text(0.5, 0.008,
             "Bars are means and dots are the three locked training-seed sensitivity runs; no seed-level hypothesis test is asserted.",
             ha="center", fontsize=6.2, color="#455A64")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), w_pad=1.5, h_pad=1.6)
    save(fig, "figure3_ablations")


def latency_figure() -> None:
    latency = rows("validation_latency_results.csv")
    stages = rows("validation_latency_stages.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    axes = axes.ravel()
    robot_colors = {"panda": COLORS["panda"], "ur5e": COLORS["ur5e"]}

    ax = axes[0]
    panel(ax, "a", "Feasible P95 ratio after exact export")
    x = np.arange(2)
    ratios = [f(next(r for r in latency if r["robot"] == robot), "p95_ratio") for robot in ("panda", "ur5e")]
    ax.bar(x, ratios, color=[COLORS["panda"], COLORS["ur5e"]], width=0.58)
    ax.axhline(1.15, color=COLORS["reject"], linestyle="--", label="Validation target 1.15")
    ax.axhline(1.0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks(x, ["Panda", "UR5e"])
    ax.set_ylabel("Proposed / fixed P95")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.5)

    ax = axes[1]
    panel(ax, "b", "P95 reduction versus eager proposed implementation")
    reduction = [100.0 * f(next(r for r in latency if r["robot"] == robot), "p95_reduction_vs_eager")
                 for robot in ("panda", "ur5e")]
    ax.bar(x, reduction, color=[COLORS["panda"], COLORS["ur5e"]], width=0.58)
    ax.axhline(30, color=COLORS["reject"], linestyle="--", label="Validation target 30%")
    ax.set_xticks(x, ["Panda", "UR5e"])
    ax.set_ylabel("P95 reduction (%)")
    ax.set_ylim(0, 55)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.5)

    ax = axes[2]
    panel(ax, "c", "Stage-level P95 profile (not additive)")
    stage_order = ["feature_preparation", "numpy_torch_conversion", "learned_seed_inference",
                   "uncertainty_risk_inference", "routing_decision", "numerical_solver",
                   "verification", "unattributed_framework"]
    display = ["Features", "Conversion", "Seed", "Risk", "Route", "Solver", "Verify", "Framework"]
    y = np.arange(len(stage_order))
    for robot, offset in (("panda", -0.11), ("ur5e", 0.11)):
        selected = {r["stage"]: f(r, "p95_ms") for r in stages if r["robot"] == robot}
        ax.scatter([selected[name] for name in stage_order], y + offset, s=24,
                   color=robot_colors[robot], label=robot.upper() if robot == "ur5e" else "Panda")
    ax.set_yticks(y, display)
    ax.invert_yaxis()
    ax.set_xlabel("Stage P95 (ms)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#D8DEE3", linewidth=0.5)

    ax = axes[3]
    panel(ax, "d", "End-to-end latency percentiles")
    percentiles = ("p50", "p95", "p99")
    marker_map = {"p50": "o", "p95": "s", "p99": "^"}
    positions = {"panda": 0, "ur5e": 1}
    for robot in ("panda", "ur5e"):
        row = next(r for r in latency if r["robot"] == robot)
        for j, pct in enumerate(percentiles):
            dx = (j - 1) * 0.08
            ax.plot([positions[robot] + dx - 0.02, positions[robot] + dx + 0.02],
                    [f(row, f"baseline_{pct}_ms"), f(row, f"proposed_{pct}_ms")],
                    color="#AEB8BF", linewidth=0.8)
            ax.scatter(positions[robot] + dx - 0.02, f(row, f"baseline_{pct}_ms"),
                       marker=marker_map[pct], s=22, color=COLORS["baseline"])
            ax.scatter(positions[robot] + dx + 0.02, f(row, f"proposed_{pct}_ms"),
                       marker=marker_map[pct], s=22, color=COLORS["proposed"])
    ax.set_xticks([0, 1], ["Panda", "UR5e"])
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.5)
    ax.scatter([], [], color=COLORS["baseline"], label="Fixed")
    ax.scatter([], [], color=COLORS["proposed"], label="Proposed")
    for pct in percentiles:
        ax.scatter([], [], color=COLORS["ink"], marker=marker_map[pct], label=pct.upper())
    ax.legend(ncol=2, loc="upper left")

    fig.text(0.5, 0.006,
             "Validation only: seed 17, 750 paired feasible queries per robot, identical query order, no test_v3 inference.",
             ha="center", fontsize=6.2, color=COLORS["reject"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.55, w_pad=1.3)
    save(fig, "figure4_latency_validation")


def main() -> None:
    method_figure()
    formal_results_figure()
    ablation_figure()
    latency_figure()
    print("Created four manuscript figures in PDF/SVG/PNG/TIFF formats.")


if __name__ == "__main__":
    main()
