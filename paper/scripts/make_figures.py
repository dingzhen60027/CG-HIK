#!/usr/bin/env python3
"""Render the five main-paper figures from immutable source-data tables.

The figures deliberately distinguish development diagnostics (Figure 2) from
fresh point and trajectory evaluations (Figures 3--5). Complete frozen-role
aggregates are shown without invented uncertainty intervals.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
SOURCE = PAPER_DIR / "source_data"
FIGURES = PAPER_DIR / "figures"

FONT_PT = 7.0

# Okabe--Ito-derived categorical colours plus neutral baselines. Colour is
# always reinforced by labels, markers, line style, or position.
COLORS = {
    "easy": "#56B4E9",
    "medium": "#E69F00",
    "hard": "#4D4D4D",
    "fixed": "#8A9299",
    "fixed_dark": "#606970",
    "proposed": "#0072B2",
    "proposed_light": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#25313A",
    "grid": "#D9E0E4",
    "pale_blue": "#EAF3F8",
    "pale_green": "#EAF5F1",
    "pale_orange": "#FBF2E2",
    "pale_red": "#F9ECE8",
    "pale_gray": "#F3F5F6",
}

METHOD_ORDER = (
    "fixed_robust_cascade",
    "always_hard",
    "counterfactual_cghik_v4",
)
METHOD_LABELS = {
    "fixed_robust_cascade": "Fixed robust cascade",
    "always_hard": "Fixed hard-entry cascade",
    "counterfactual_cghik_v4": "CG-HIK",
    "proposed_v4": "CG-HIK",
}
METHOD_COLORS = {
    "fixed_robust_cascade": COLORS["fixed"],
    "always_hard": COLORS["fixed_dark"],
    "counterfactual_cghik_v4": COLORS["proposed"],
    "proposed_v4": COLORS["proposed"],
}
ROBOT_LABELS = {"panda": "Panda", "ur5e": "UR5e"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.2,
        "figure.titlesize": 8.0,
        "svg.fonttype": "none",
        "svg.hashsalt": "cghik-final-paper",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.15,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)


def load_rows(
    name: str,
    required: Iterable[str],
    *,
    expected_count: int | None = None,
) -> list[dict[str, str]]:
    """Load one source-data file and fail closed on schema/count drift."""
    path = SOURCE / name
    if not path.is_file():
        raise FileNotFoundError(f"missing source data: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(required) - fieldnames
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        data = list(reader)
    if expected_count is not None and len(data) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} rows, found {len(data)}")
    return data


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]}")
    return value


def one(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    selected = [
        row for row in rows if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(selected)}")
    return selected[0]


def panel(ax: mpl.axes.Axes, label: str, title: str, *, x: float = -0.13) -> None:
    ax.text(
        x,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.set_title(title, loc="left", pad=5.0, fontweight="bold")


def grid_y(ax: mpl.axes.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)


def grid_x(ax: mpl.axes.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)


def save(fig: mpl.figure.Figure, name: str) -> None:
    """Export editable vectors plus a 600-dpi inspection raster."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Preserve the declared 182.9-mm final width; tight bounding boxes can
    # silently expand wide y labels beyond the journal canvas.
    svg_path = FIGURES / f"{name}.svg"
    fig.savefig(
        svg_path,
        metadata={"Date": None, "Creator": "CG-HIK deterministic figure builder"},
    )
    # Matplotlib formats path coordinates on separate lines with trailing spaces.
    # Normalize those lines so editable vectors pass repository whitespace checks.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        FIGURES / f"{name}.pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "CG-HIK deterministic figure builder",
        },
    )
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    plt.close(fig)


def draw_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    *,
    edgecolor: str | None = None,
    title_size: float = 6.8,
    body_size: float = 5.8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor or COLORS["ink"],
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.15,
    )


def arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    style: str = "-|>",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=0.75,
            color=color or COLORS["ink"],
            connectionstyle=connectionstyle,
        )
    )


def figure1_framework() -> None:
    """Schematic: learning allocates work; the verifier alone accepts."""
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.55),
        gridspec_kw={"width_ratios": [1.02, 1.18], "wspace": 0.09},
    )
    for ax in (left, right):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left.text(-0.02, 1.02, "a", fontsize=8.0, fontweight="bold", va="top")
    left.text(0.03, 1.02, "Development: action-complete evidence", fontsize=7.5,
              fontweight="bold", va="top")
    draw_box(left, 0.03, 0.77, 0.20, 0.14, "Query", "pose + history", COLORS["pale_blue"])
    draw_box(left, 0.31, 0.73, 0.28, 0.22, "All entries",
             "Easy · Medium · Hard\nsame query", COLORS["pale_orange"])
    draw_box(left, 0.67, 0.73, 0.30, 0.22, "Shared cascade",
             "entry-specific DLS\nrobust fallback", COLORS["pale_gray"])
    arrow(left, (0.23, 0.84), (0.31, 0.84))
    arrow(left, (0.59, 0.84), (0.67, 0.84))

    draw_box(left, 0.67, 0.47, 0.30, 0.15, "Verifier",
             "deterministic\npose · limits · velocity", COLORS["pale_green"], title_size=6.4)
    arrow(left, (0.82, 0.73), (0.82, 0.62))
    draw_box(left, 0.27, 0.43, 0.31, 0.23, "Complete record",
             "verified outcome · FEV\nfallback · raw repeats", COLORS["pale_blue"])
    arrow(left, (0.67, 0.545), (0.58, 0.545))

    role_y = 0.19
    role_w = 0.245
    for x, title, body in (
        (0.02, "Fit", "training role"),
        (0.29, "Calibrate", "calibration role"),
        (0.56, "Select", "policy role"),
    ):
        draw_box(left, x, role_y, role_w, 0.13, title, body, "white", title_size=6.4)
    arrow(left, (0.42, 0.43), (0.145, 0.32), connectionstyle="arc3,rad=0.14")
    arrow(left, (0.265, 0.255), (0.29, 0.255))
    arrow(left, (0.535, 0.255), (0.56, 0.255))
    draw_box(left, 0.73, 0.06, 0.25, 0.13, "Freeze", "model + policy", COLORS["pale_green"])
    arrow(left, (0.805, 0.19), (0.83, 0.18), connectionstyle="arc3,rad=-0.15")
    left.text(0.02, 0.07, "Disjoint roles", fontsize=5.8, color=COLORS["fixed_dark"])

    right.text(-0.02, 1.02, "b", fontsize=8.0, fontweight="bold", va="top")
    right.text(0.03, 1.02, "Runtime: frozen compute allocation", fontsize=7.5,
               fontweight="bold", va="top")
    draw_box(right, 0.03, 0.79, 0.23, 0.15, "Online query",
             "pose + diagnostics", COLORS["pale_blue"])
    draw_box(right, 0.34, 0.77, 0.28, 0.19, "Frozen predictor",
             "success · P50/P95\nportfolio failure", COLORS["pale_green"])
    draw_box(right, 0.70, 0.77, 0.27, 0.19, "Calibrated policy",
             "eligibility + abstention", COLORS["pale_orange"])
    arrow(right, (0.26, 0.865), (0.34, 0.865))
    arrow(right, (0.62, 0.865), (0.70, 0.865))

    draw_box(right, 0.03, 0.50, 0.27, 0.16, "Eligible",
             "choose lowest\npredicted P95", COLORS["pale_blue"])
    draw_box(right, 0.365, 0.50, 0.27, 0.16, "Uncertain",
             "defer to full cascade", COLORS["pale_gray"])
    draw_box(right, 0.70, 0.50, 0.27, 0.16, "Likely fail-all",
             "reject; skip\nnumerical solve", COLORS["pale_red"])
    arrow(right, (0.78, 0.77), (0.165, 0.66), connectionstyle="arc3,rad=0.13")
    arrow(right, (0.835, 0.77), (0.50, 0.66), connectionstyle="arc3,rad=0.05")
    arrow(right, (0.89, 0.77), (0.835, 0.66), connectionstyle="arc3,rad=-0.06")

    draw_box(right, 0.12, 0.26, 0.46, 0.14, "Shared numerical cascade",
             "selected entry  →  escalation  →  fallback", COLORS["pale_orange"])
    arrow(right, (0.165, 0.50), (0.28, 0.40))
    arrow(right, (0.50, 0.50), (0.43, 0.40))
    draw_box(right, 0.62, 0.23, 0.35, 0.20, "Deterministic verifier",
             "exclusive command authority\naccept or withhold", COLORS["pale_green"])
    arrow(right, (0.58, 0.33), (0.62, 0.33))
    arrow(right, (0.95, 0.50), (0.94, 0.15), color=COLORS["vermillion"],
          connectionstyle="arc3,rad=-0.20")

    draw_box(right, 0.62, 0.04, 0.35, 0.11, "Output",
             "verified command\nor no command", "white", title_size=6.4)
    arrow(right, (0.795, 0.23), (0.795, 0.15))
    right.text(
        0.03,
        0.065,
        "Learned allocation never\ncertifies a command.",
        fontsize=6.0,
        color=COLORS["proposed"],
        fontweight="bold",
        linespacing=1.2,
    )

    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.035, top=0.94)
    save(fig, "figure1_cghik_framework")


def routing_metric(rows: list[dict[str, str]], robot: str, metric: str) -> float:
    return number(one(rows, robot=robot, metric=metric), "value")


def figure2_heterogeneity() -> None:
    oracle = load_rows(
        "development_oracle_distribution.csv",
        ("robot", "entry", "count", "rate"),
        expected_count=6,
    )
    family = load_rows(
        "development_family_oracle.csv",
        ("robot", "family", "entry", "count", "rate", "family_success_count"),
        expected_count=30,
    )
    routing = load_rows(
        "development_routing_metrics.csv",
        ("robot", "metric", "value", "unit", "scope"),
        expected_count=64,
    )
    predicted_observed = load_rows(
        "development_predicted_observed_p95.csv",
        (
            "robot", "query_sha256", "family", "entry", "predicted_p95_ms",
            "observed_empirical_p95_ms", "selected_action", "semantic_success",
        ),
        expected_count=15000,
    )

    entries = ("easy", "medium", "hard")
    entry_handles = [
        mpl.patches.Patch(facecolor=COLORS[entry], label=entry.capitalize()) for entry in entries
    ]
    fig = plt.figure(figsize=(7.2, 6.2))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.13, 1.0],
        left=0.08,
        right=0.985,
        bottom=0.095,
        top=0.94,
        hspace=0.47,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[0.92, 1.23, 1.02], wspace=0.58)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.10], wspace=0.43)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(bottom[0, 0])
    ax_e = fig.add_subplot(bottom[0, 1])

    panel(ax_a, "a", "Empirical oracle entries", x=-0.24)
    for y, robot in enumerate(("panda", "ur5e")):
        start = 0.0
        for entry in entries:
            row = one(oracle, robot=robot, entry=entry)
            rate = number(row, "rate")
            ax_a.barh(y, rate * 100.0, left=start, height=0.55,
                      color=COLORS[entry], edgecolor="white", linewidth=0.5)
            text_color = "white" if entry == "hard" else COLORS["ink"]
            ax_a.text(start + rate * 50.0, y, f"{rate * 100:.1f}", ha="center", va="center",
                      fontsize=5.8, color=text_color, fontweight="bold")
            start += rate * 100.0
    ax_a.set_yticks([0, 1], ["Panda", "UR5e"])
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Successful queries (%)")
    ax_a.invert_yaxis()
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    grid_x(ax_a)
    ax_a.legend(handles=entry_handles, loc="lower center", bbox_to_anchor=(0.5, -0.31),
                frameon=False, ncol=3, handlelength=1.25, columnspacing=0.8)

    panel(ax_b, "b", "Query-family oracle share", x=-0.18)
    family_order = (
        ("panda", "hard_valid", "P · Hard-valid"),
        ("panda", "id", "P · In-distribution"),
        ("panda", "near_limit", "P · Near-limit"),
        ("panda", "near_singular", "P · Near-singular"),
        ("panda", "workspace_boundary", "P · Workspace edge"),
        ("ur5e", "hard_valid", "U · Hard-valid"),
        ("ur5e", "id", "U · In-distribution"),
        ("ur5e", "near_limit", "U · Near-limit"),
        ("ur5e", "near_singular", "U · Near-singular"),
        ("ur5e", "workspace_boundary", "U · Workspace edge"),
    )
    matrix = np.zeros((len(family_order), len(entries)), dtype=float)
    for row_index, (robot, family_name, _) in enumerate(family_order):
        total = 0.0
        for column_index, entry in enumerate(entries):
            row = one(family, robot=robot, family=family_name, entry=entry)
            rate = number(row, "rate")
            matrix[row_index, column_index] = rate * 100.0
            total += rate
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"family oracle rates do not sum to one: {robot}/{family_name}")
    heatmap = mpl.colors.LinearSegmentedColormap.from_list(
        "oracle_share", ["#F4F8FA", "#56B4E9", "#0072B2"]
    )
    image = ax_b.imshow(matrix, cmap=heatmap, vmin=0, vmax=100, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax_b.text(column_index, row_index, f"{value:.0f}", ha="center", va="center",
                      fontsize=5.2, color="white" if value >= 55 else COLORS["ink"])
    ax_b.axhline(4.5, color="white", linewidth=1.2)
    ax_b.set_yticks(np.arange(len(family_order)), [item[2] for item in family_order])
    ax_b.set_xticks(np.arange(len(entries)), [entry.capitalize() for entry in entries])
    ax_b.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                     length=0)
    ax_b.tick_params(axis="y", length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_b, orientation="horizontal", pad=0.075,
                            fraction=0.055, aspect=24)
    colorbar.set_label("Oracle share (%)", labelpad=1.5)
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.tick_params(labelsize=5.6, length=2)

    panel(ax_c, "c", "Gap to the per-query oracle", x=-0.20)
    gap_metrics = (
        ("best_fixed_to_oracle_gap_mean_ms", "Best-fixed mean"),
        ("easy_to_oracle_gap_mean_ms", "Easy-entry mean"),
        ("easy_to_oracle_gap_p95_ms", "Easy-entry P95"),
    )
    gap_y = np.arange(len(gap_metrics), dtype=float)
    robot_styles = {
        "panda": (COLORS["proposed"], "o", -0.10),
        "ur5e": (COLORS["orange"], "s", 0.10),
    }
    for robot, (color, marker, offset) in robot_styles.items():
        values = [routing_metric(routing, robot, metric) for metric, _ in gap_metrics]
        ax_c.hlines(gap_y + offset, 0, values, color=color, alpha=0.55, linewidth=1.0)
        ax_c.scatter(values, gap_y + offset, color=color, marker=marker, s=25,
                     edgecolor="white", linewidth=0.45, zorder=3,
                     label=ROBOT_LABELS[robot])
        for x_value, y_value in zip(values, gap_y + offset, strict=True):
            ax_c.text(x_value + 0.035, y_value, f"{x_value:.2f}", va="center",
                      fontsize=5.4, color=color)
    ax_c.set_yticks(gap_y, [label for _, label in gap_metrics])
    ax_c.set_xlim(0, 1.60)
    ax_c.set_xlabel("Excess latency (ms)")
    ax_c.invert_yaxis()
    ax_c.legend(frameon=False, loc="upper right")
    grid_x(ax_c)

    panel(ax_d, "d", "Frozen-policy routing regret", x=-0.18)
    regret_metrics = (
        ("routing_regret_median_ms", "Median"),
        ("routing_regret_mean_ms", "Mean"),
        ("routing_regret_p95_ms", "P95"),
    )
    y = np.arange(len(regret_metrics), dtype=float)
    for robot, (color, marker, offset) in robot_styles.items():
        values = [routing_metric(routing, robot, metric) for metric, _ in regret_metrics]
        ax_d.hlines(y + offset, 0, values, color=color, alpha=0.55, linewidth=1.1)
        ax_d.scatter(values, y + offset, color=color, marker=marker, s=27,
                     edgecolor="white", linewidth=0.45, zorder=3,
                     label=ROBOT_LABELS[robot])
        for x_value, y_value in zip(values, y + offset, strict=True):
            ax_d.text(x_value + 0.014, y_value, f"{x_value:.3f}", va="center", fontsize=5.6,
                      color=color)
    ax_d.axvline(0.15, color=COLORS["fixed_dark"], linestyle="--", linewidth=0.75)
    ax_d.text(0.154, 2.48, "0.15 ms", fontsize=5.6, color=COLORS["fixed_dark"], va="bottom")
    ax_d.set_yticks(y, [label for _, label in regret_metrics])
    ax_d.set_xlabel("Regret relative to empirical oracle (ms)")
    ax_d.set_xlim(0, 0.69)
    ax_d.set_ylim(-0.42, 2.55)
    ax_d.invert_yaxis()
    grid_x(ax_d)
    annotation = []
    for robot in ("panda", "ur5e"):
        n = int(round(routing_metric(routing, robot, "policy_successful_non_abstained_count")))
        agreement = routing_metric(routing, robot, "oracle_agreement_rate") * 100.0
        within = routing_metric(routing, robot, "regret_within_0_15ms_rate") * 100.0
        annotation.append(
            f"{ROBOT_LABELS[robot]}: n={n:,}; agreement {agreement:.1f}%; ≤0.15 ms {within:.1f}%"
        )
    ax_d.text(0.995, 0.99, "\n".join(annotation), transform=ax_d.transAxes,
              ha="right", va="top", fontsize=5.8, color=COLORS["ink"], linespacing=1.35)

    panel(ax_e, "e", "Predicted versus observed entry P95", x=-0.18)
    all_predicted = np.asarray(
        [number(row, "predicted_p95_ms") for row in predicted_observed], dtype=float
    )
    all_observed = np.asarray(
        [number(row, "observed_empirical_p95_ms") for row in predicted_observed], dtype=float
    )
    if np.any(all_predicted <= 0) or np.any(all_observed <= 0):
        raise ValueError("P95 calibration panel requires strictly positive latencies")
    ax_e.scatter(all_predicted, all_observed, s=1.4, color=COLORS["fixed"],
                 alpha=0.055, linewidths=0, rasterized=True)
    for robot, (color, marker, _) in robot_styles.items():
        selected = [row for row in predicted_observed if row["robot"] == robot]
        if len(selected) != 7500:
            raise ValueError(f"expected 7,500 entry-query actions for {robot}")
        predicted = np.asarray([number(row, "predicted_p95_ms") for row in selected])
        observed = np.asarray([number(row, "observed_empirical_p95_ms") for row in selected])
        order = np.argsort(predicted)
        bins = np.array_split(order, 12)
        binned_predicted = [float(np.median(predicted[index])) for index in bins]
        binned_observed = [float(np.median(observed[index])) for index in bins]
        ax_e.plot(binned_predicted, binned_observed, color=color, marker=marker,
                  markeredgecolor="white", markeredgewidth=0.4,
                  label=f"{ROBOT_LABELS[robot]} equal-count bins")
    ax_e.plot([1, 500], [1, 500], color=COLORS["ink"], linestyle=":", linewidth=0.8,
              label="Identity")
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    tick_values = [1, 2, 10, 100, 500]
    tick_labels = ["1", "2", "10", "100", "500"]
    ax_e.set_xticks(tick_values, tick_labels)
    ax_e.set_yticks(tick_values, tick_labels)
    ax_e.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_e.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_e.set_xlim(1, 500)
    ax_e.set_ylim(1, 500)
    ax_e.set_xlabel("Predicted P95 (ms)")
    ax_e.set_ylabel("Observed empirical P95 (ms)")
    ax_e.legend(frameon=False, loc="upper left")
    grid_y(ax_e)
    grid_x(ax_e)
    ax_e.text(0.98, 0.04, "All 15,000 entry-query actions shown;\ntrend points are 12 equal-count bins per robot.",
              transform=ax_e.transAxes, ha="right", va="bottom", fontsize=5.4,
              color=COLORS["fixed_dark"])

    fig.text(
        0.5,
        0.012,
        "Development only. Panels a–c use successful action-complete queries "
        "(Panda n=17,507; UR5e n=17,584); d uses successful non-abstained policy-validation "
        "queries.\nPanel e uses all three entry actions for 2,500 policy-validation queries per robot. "
        "Values are exact frozen-role summaries; no fresh-test claim is made.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["fixed_dark"],
    )
    save(fig, "figure2_heterogeneity_predictability")


def figure3_point_results() -> None:
    point = load_rows(
        "point_formal_results.csv",
        (
            "robot", "method", "method_label", "query_count", "verified_success_rate",
            "mean_fev", "p50_ms", "p95_ms", "p99_ms",
            "accepted_contract_violation_count",
        ),
        expected_count=14,
    )
    rejectable = load_rows(
        "point_rejectable_results.csv",
        (
            "robot", "method", "query_count", "command_reject_rate", "total_fev",
            "fev_avoided_fraction_vs_fixed", "formal_reject_recall",
            "accepted_contract_violation_count",
        ),
        expected_count=14,
    )
    compared = ("fixed_robust_cascade", "proposed_v4")
    for robot in ("panda", "ur5e"):
        for method in compared:
            one(point, robot=robot, method=method)
            one(rejectable, robot=robot, method=method)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    method_handles = [
        Line2D([], [], color=COLORS["fixed"], marker="o", linestyle="none",
               label="Fixed robust cascade"),
        Line2D([], [], color=COLORS["proposed"], marker="o", linestyle="none",
               label="CG-HIK"),
    ]

    panel(ax_a, "a", "Verified success is preserved")
    x = np.arange(2, dtype=float)
    for i, robot in enumerate(("panda", "ur5e")):
        base = number(one(point, robot=robot, method="fixed_robust_cascade"),
                      "verified_success_rate") * 100.0
        proposed = number(one(point, robot=robot, method="proposed_v4"),
                          "verified_success_rate") * 100.0
        ax_a.plot([i - 0.09, i + 0.09], [base, proposed], color=COLORS["grid"], linewidth=1.0)
        ax_a.scatter(i - 0.09, base, color=COLORS["fixed"], s=35, zorder=3)
        ax_a.scatter(i + 0.09, proposed, color=COLORS["proposed"], s=35, zorder=3)
        ax_a.text(i, min(base, proposed) - 0.18, f"Δ {proposed - base:+.2f} pp",
                  ha="center", va="top", fontsize=5.8, color=COLORS["ink"])
    ax_a.set_xticks(x, ["Panda", "UR5e"])
    ax_a.set_ylabel("Verified success (%)")
    ax_a.set_ylim(98.55, 100.18)
    ax_a.legend(handles=method_handles, frameon=False, loc="lower right")
    grid_y(ax_a)

    panel(ax_b, "b", "Numerical work falls")
    width = 0.32
    for j, method in enumerate(compared):
        values = [number(one(point, robot=robot, method=method), "mean_fev")
                  for robot in ("panda", "ur5e")]
        ax_b.bar(x + (j - 0.5) * width, values, width=width,
                 color=COLORS["fixed"] if method == "fixed_robust_cascade" else COLORS["proposed"],
                 edgecolor="white", linewidth=0.5)
    for i, robot in enumerate(("panda", "ur5e")):
        base = number(one(point, robot=robot, method="fixed_robust_cascade"), "mean_fev")
        proposed = number(one(point, robot=robot, method="proposed_v4"), "mean_fev")
        ax_b.text(i + width * 0.5, proposed + 0.28, f"{(proposed / base - 1) * 100:+.1f}%",
                  ha="center", va="bottom", fontsize=5.8, color=COLORS["proposed"],
                  fontweight="bold")
    ax_b.set_xticks(x, ["Panda", "UR5e"])
    ax_b.set_ylabel("Mean function evaluations")
    ax_b.set_ylim(0, 9.6)
    grid_y(ax_b)

    panel(ax_c, "c", "Latency trade-off is quantile-specific")
    percentiles = (("p50_ms", "P50"), ("p95_ms", "P95"), ("p99_ms", "P99"))
    positions = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    ratios: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    for robot in ("panda", "ur5e"):
        base_row = one(point, robot=robot, method="fixed_robust_cascade")
        prop_row = one(point, robot=robot, method="proposed_v4")
        for key, label in percentiles:
            ratio = number(prop_row, key) / number(base_row, key)
            ratios.append(ratio)
            labels.append(label)
            colors.append(COLORS["proposed"] if robot == "panda" else COLORS["proposed_light"])
    ax_c.bar(positions, ratios, width=0.66, color=colors, edgecolor="white", linewidth=0.5)
    ax_c.axhline(1.0, color=COLORS["ink"], linewidth=0.8)
    for xpos, ratio in zip(positions, ratios, strict=True):
        offset = 0.018 if ratio >= 1.0 else -0.025
        ax_c.text(xpos, ratio + offset, f"{ratio:.2f}×", ha="center",
                  va="bottom" if ratio >= 1.0 else "top", fontsize=5.8,
                  color=COLORS["ink"], fontweight="bold")
    ax_c.set_xticks(positions, labels)
    ax_c.set_ylabel("CG-HIK / fixed latency")
    ax_c.set_ylim(0.66, 1.23)
    ax_c.text(1, -0.20, "Panda", ha="center", va="top", fontsize=6.2,
              transform=ax_c.get_xaxis_transform())
    ax_c.text(5, -0.20, "UR5e", ha="center", va="top", fontsize=6.2,
              transform=ax_c.get_xaxis_transform())
    grid_y(ax_c)

    panel(ax_d, "d", "Known-infeasible work is avoided")
    metric_specs = (
        ("formal_reject_recall", "Reject recall", "o"),
        ("fev_avoided_fraction_vs_fixed", "FEV avoided", "s"),
    )
    for metric_index, (key, label, marker) in enumerate(metric_specs):
        values = [number(one(rejectable, robot=robot, method="proposed_v4"), key) * 100.0
                  for robot in ("panda", "ur5e")]
        ax_d.scatter(x + (metric_index - 0.5) * 0.18, values, s=42,
                     marker=marker, color=COLORS["proposed"] if metric_index == 0 else COLORS["orange"],
                     edgecolor="white", linewidth=0.5, label=label, zorder=3)
        for xpos, value in zip(x + (metric_index - 0.5) * 0.18, values, strict=True):
            ax_d.text(xpos, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax_d.set_xticks(x, ["Panda", "UR5e"])
    ax_d.set_ylabel("Known-infeasible queries / work (%)")
    ax_d.set_ylim(90, 99)
    ax_d.legend(frameon=False, loc="lower left")
    grid_y(ax_d)

    fig.text(
        0.5,
        0.008,
        "Fresh independent point evaluation: n=14,000 feasible and n=2,000 known-infeasible "
        "queries per robot. Points and bars are exact frozen-test aggregates;\n"
        "no replicate interval is implied. FEV denotes function evaluations.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["fixed_dark"],
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 1.0), h_pad=1.65, w_pad=1.5)
    save(fig, "figure3_fresh_point_results")


def figure4_trajectory_results() -> None:
    trajectory = load_rows(
        "trajectory_main_results.csv",
        (
            "robot", "method", "method_label", "trajectory_count",
            "whole_trajectory_completion_count", "total_cumulative_latency_seconds",
            "frame_p50_latency_ms", "frame_p95_latency_ms", "frame_p99_latency_ms",
            "mean_fev", "accepted_contract_violation_count",
        ),
        expected_count=6,
    )
    for robot in ("panda", "ur5e"):
        for method in METHOD_ORDER:
            row = one(trajectory, robot=robot, method=method)
            if int(row["trajectory_count"]) != 80:
                raise ValueError("trajectory figure expects 80 fresh trajectories per robot")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    x = np.arange(2, dtype=float)
    width = 0.245
    offsets = np.array([-width, 0.0, width])
    method_handles = [
        mpl.patches.Patch(facecolor=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for method in METHOD_ORDER
    ]

    panel(ax_a, "a", "Whole-trajectory completion")
    for j, method in enumerate(METHOD_ORDER):
        values = [number(one(trajectory, robot=robot, method=method),
                         "whole_trajectory_completion_count") for robot in ("panda", "ur5e")]
        bars = ax_a.bar(x + offsets[j], values, width=width, color=METHOD_COLORS[method],
                        edgecolor="white", linewidth=0.45)
        for bar, value in zip(bars, values, strict=True):
            ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{int(value)}/80",
                      ha="center", va="bottom", fontsize=5.6)
    ax_a.set_xticks(x, ["Panda", "UR5e"])
    ax_a.set_ylabel("Completed trajectories")
    ax_a.set_ylim(0, 48)
    ax_a.legend(handles=method_handles, frameon=False, loc="upper right")
    grid_y(ax_a)

    panel(ax_b, "b", "Aggregate cumulative latency")
    for j, method in enumerate(METHOD_ORDER):
        values = [number(one(trajectory, robot=robot, method=method),
                         "total_cumulative_latency_seconds") for robot in ("panda", "ur5e")]
        ax_b.bar(x + offsets[j], values, width=width, color=METHOD_COLORS[method],
                 edgecolor="white", linewidth=0.45)
    for i, robot in enumerate(("panda", "ur5e")):
        hard = number(one(trajectory, robot=robot, method="always_hard"),
                      "total_cumulative_latency_seconds")
        proposed = number(one(trajectory, robot=robot, method="counterfactual_cghik_v4"),
                          "total_cumulative_latency_seconds")
        ax_b.text(i + offsets[2], proposed + 12, f"{proposed / hard:.2f}×",
                  ha="center", va="bottom", fontsize=5.8, color=COLORS["proposed"],
                  fontweight="bold")
    ax_b.set_xticks(x, ["Panda", "UR5e"])
    ax_b.set_ylabel("Cumulative latency (s)")
    ax_b.set_ylim(0, 545)
    grid_y(ax_b)

    panel(ax_c, "c", "Mean numerical effort")
    for j, method in enumerate(METHOD_ORDER):
        values = [number(one(trajectory, robot=robot, method=method), "mean_fev")
                  for robot in ("panda", "ur5e")]
        ax_c.bar(x + offsets[j], values, width=width, color=METHOD_COLORS[method],
                 edgecolor="white", linewidth=0.45)
    for i, robot in enumerate(("panda", "ur5e")):
        hard = number(one(trajectory, robot=robot, method="always_hard"), "mean_fev")
        proposed = number(one(trajectory, robot=robot, method="counterfactual_cghik_v4"),
                          "mean_fev")
        ax_c.text(i + offsets[2], proposed + 1.8, f"{proposed / hard:.2f}×",
                  ha="center", va="bottom", fontsize=5.8, color=COLORS["proposed"],
                  fontweight="bold")
    ax_c.set_xticks(x, ["Panda", "UR5e"])
    ax_c.set_ylabel("Mean function evaluations")
    ax_c.set_ylim(0, 71)
    grid_y(ax_c)

    panel(ax_d, "d", "Frame-latency ratio to hard entry")
    percentile_specs = (
        ("frame_p50_latency_ms", "P50"),
        ("frame_p95_latency_ms", "P95"),
        ("frame_p99_latency_ms", "P99"),
    )
    block_positions = {"panda": np.array([0.0, 1.0, 2.0]),
                       "ur5e": np.array([4.0, 5.0, 6.0])}
    for robot in ("panda", "ur5e"):
        hard_row = one(trajectory, robot=robot, method="always_hard")
        for method, linestyle, marker in (
            ("fixed_robust_cascade", "--", "o"),
            ("counterfactual_cghik_v4", "-", "s"),
        ):
            row = one(trajectory, robot=robot, method=method)
            values = [number(row, key) / number(hard_row, key) for key, _ in percentile_specs]
            ax_d.plot(block_positions[robot], values, linestyle=linestyle, marker=marker,
                      color=METHOD_COLORS[method], linewidth=1.15,
                      markerfacecolor="white" if robot == "ur5e" else METHOD_COLORS[method],
                      markeredgewidth=0.8)
            if method == "counterfactual_cghik_v4":
                for xpos, value in zip(block_positions[robot], values, strict=True):
                    ax_d.text(xpos, value + (0.018 if value >= 1.0 else -0.025),
                              f"{value:.2f}×", ha="center",
                              va="bottom" if value >= 1.0 else "top", fontsize=5.6,
                              color=COLORS["proposed"])
    ax_d.axhline(1.0, color=COLORS["ink"], linewidth=0.75)
    ax_d.set_xticks(np.r_[block_positions["panda"], block_positions["ur5e"]],
                      ["P50", "P95", "P99", "P50", "P95", "P99"])
    ax_d.text(1, -0.20, "Panda", ha="center", va="top", fontsize=6.2,
              transform=ax_d.get_xaxis_transform())
    ax_d.text(5, -0.20, "UR5e", ha="center", va="top", fontsize=6.2,
              transform=ax_d.get_xaxis_transform())
    ax_d.set_ylabel("Method / hard-entry latency")
    ax_d.set_ylim(0.70, 1.18)
    ratio_handles = [
        Line2D([], [], color=COLORS["fixed"], linestyle="--", marker="o",
               label="Fixed robust cascade"),
        Line2D([], [], color=COLORS["proposed"], linestyle="-", marker="s",
               label="CG-HIK"),
        Line2D([], [], color=COLORS["ink"], linestyle="-", label="Hard-entry reference"),
    ]
    ax_d.legend(handles=ratio_handles, frameon=False, loc="lower left")
    grid_y(ax_d)

    fig.text(
        0.5,
        0.008,
        "One-shot transition-rich evaluation: n=80 trajectories and 12,000 frames per robot; "
        "each bar is an exact frozen-role aggregate. Ratios compare the same robot\n"
        "and frame quantile. P50 is report-only. FEV denotes function evaluations.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["fixed_dark"],
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 1.0), h_pad=1.7, w_pad=1.55)
    save(fig, "figure4_fresh_trajectory_results")


def phase_background(ax: mpl.axes.Axes, *, labels: bool = False) -> None:
    """Show the frozen 150-frame near-singular trajectory phase contract."""
    phases = (
        (0.00, 0.40, "Regular"),
        (0.40, 1.30, "Approach"),
        (1.30, 1.70, "Near-sing."),
        (1.70, 2.60, "Return"),
        (2.60, 3.00, "Regular"),
    )
    for index, (start, stop, label) in enumerate(phases):
        face = COLORS["pale_gray"] if index % 2 == 0 else COLORS["pale_orange"]
        ax.axvspan(start, stop, facecolor=face, alpha=0.48, zorder=-5)
        if labels:
            ax.text((start + stop) / 2, 1.08, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5.4, color=COLORS["fixed_dark"])
    for boundary in (0.40, 1.30, 1.70, 2.60):
        ax.axvline(boundary, color="white", linewidth=0.8, zorder=0)


def figure5_family_timeline() -> None:
    family = load_rows(
        "trajectory_family_results.csv",
        (
            "robot", "family", "family_label", "method", "trajectory_count",
            "completion_difference_vs_hard", "cumulative_latency_change_vs_hard",
            "mean_fev_change_vs_hard",
        ),
        expected_count=24,
    )
    trace = load_rows(
        "trajectory_representative_timeseries.csv",
        (
            "robot", "trajectory_uid", "selection_rule", "family", "frame",
            "time_seconds", "method", "method_label", "route", "latency_ms",
            "function_evaluations", "accepted", "fallback_used",
        ),
        expected_count=450,
    )
    proposed_family = [row for row in family if row["method"] == "counterfactual_cghik_v4"]
    if len(proposed_family) != 8:
        raise ValueError("family panel expects 2 robots × 4 families for CG-HIK")
    if len({row["trajectory_uid"] for row in trace}) != 1:
        raise ValueError("timeline must contain exactly one representative trajectory")
    for robot in ("panda", "ur5e"):
        for method in METHOD_ORDER:
            selected = [row for row in trace if row["robot"] == robot and row["method"] == method]
            if robot == "panda" and len(selected) != 150:
                raise ValueError(f"representative timeline missing frames for {method}")
            if robot == "ur5e" and selected:
                raise ValueError("representative source unexpectedly contains a second robot")

    fig = plt.figure(figsize=(7.2, 6.45))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.05, 0.46, 1.18, 1.18],
        left=0.205,
        right=0.985,
        bottom=0.11,
        top=0.94,
        hspace=0.54,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_c)

    panel(ax_a, "a", "Family-specific change relative to hard entry", x=-0.17)
    family_order = (
        ("panda", "smooth_fast_orientation_smooth", "Panda · Smooth–orientation"),
        ("panda", "regular_near_singular_regular", "Panda · Near-singular"),
        ("panda", "central_joint_limit_skim_return", "Panda · Joint-limit skim"),
        ("panda", "slow_high_curvature_high_speed_slow", "Panda · High-curvature/speed"),
        ("ur5e", "smooth_fast_orientation_smooth", "UR5e · Smooth–orientation"),
        ("ur5e", "regular_near_singular_regular", "UR5e · Near-singular"),
        ("ur5e", "central_joint_limit_skim_return", "UR5e · Joint-limit skim"),
        ("ur5e", "slow_high_curvature_high_speed_slow", "UR5e · High-curvature/speed"),
    )
    y = np.arange(len(family_order), dtype=float)
    latency_change = []
    fev_change = []
    for robot, family_name, _ in family_order:
        row = one(family, robot=robot, family=family_name, method="counterfactual_cghik_v4")
        if int(row["trajectory_count"]) != 20:
            raise ValueError("family panel expects 20 trajectories per robot/family")
        latency_change.append(number(row, "cumulative_latency_change_vs_hard") * 100.0)
        fev_change.append(number(row, "mean_fev_change_vs_hard") * 100.0)
    ax_a.axvspan(-90, 0, color=COLORS["pale_green"], alpha=0.55, zorder=-6)
    ax_a.axvspan(0, 10, color=COLORS["pale_red"], alpha=0.55, zorder=-6)
    ax_a.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax_a.hlines(y - 0.11, 0, latency_change, color=COLORS["proposed"], linewidth=1.0)
    ax_a.hlines(y + 0.11, 0, fev_change, color=COLORS["orange"], linewidth=1.0)
    ax_a.scatter(latency_change, y - 0.11, color=COLORS["proposed"], marker="o", s=30,
                 edgecolor="white", linewidth=0.45, label="Cumulative latency")
    ax_a.scatter(fev_change, y + 0.11, color=COLORS["orange"], marker="s", s=28,
                 edgecolor="white", linewidth=0.45, label="Mean FEV")
    for xpos, ypos in zip(latency_change, y - 0.11, strict=True):
        ax_a.text(xpos - 1.5 if xpos < 0 else xpos + 1.1, ypos, f"{xpos:+.1f}",
                  ha="right" if xpos < 0 else "left", va="center", fontsize=5.4,
                  color=COLORS["proposed"])
    ax_a.axhline(3.5, color=COLORS["ink"], linewidth=0.65)
    ax_a.set_yticks(y, [label for _, _, label in family_order])
    ax_a.set_xlim(-90, 10)
    ax_a.set_xlabel("Change from fixed hard-entry cascade (%)")
    ax_a.invert_yaxis()
    ax_a.legend(frameon=False, loc="upper left", ncol=2)
    grid_x(ax_a)
    ax_a.text(8.5, 6, "0/20 complete", ha="right", va="center", fontsize=5.6,
              color=COLORS["vermillion"], fontweight="bold")
    panel(ax_b, "b", "CG-HIK selected entry", x=-0.17)
    proposed_trace = sorted(
        (row for row in trace if row["method"] == "counterfactual_cghik_v4"),
        key=lambda row: int(row["frame"]),
    )
    route_code = {"easy": 0, "medium": 1, "hard": 2}
    codes = np.asarray([route_code[row["route"]] for row in proposed_trace], dtype=float)
    route_cmap = ListedColormap([COLORS["easy"], COLORS["medium"], COLORS["hard"]])
    ax_b.imshow(codes[np.newaxis, :], aspect="auto", interpolation="nearest",
                extent=(0, 3.0, 0, 1), cmap=route_cmap, vmin=-0.5, vmax=2.5)
    phase_background(ax_b)
    phase_labels = (
        (0.20, "Regular", COLORS["ink"]),
        (0.85, "Approach", "white"),
        (1.50, "Near-sing.", COLORS["ink"]),
        (2.15, "Return", "white"),
        (2.80, "Regular", COLORS["ink"]),
    )
    for xpos, label, color in phase_labels:
        ax_b.text(xpos, 0.5, label, ha="center", va="center", fontsize=5.4,
                  color=color, fontweight="bold")
    ax_b.set_yticks([])
    ax_b.set_xlim(0, 2.98)
    ax_b.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    route_handles = [
        mpl.patches.Patch(facecolor=COLORS["easy"], label="Easy"),
        mpl.patches.Patch(facecolor=COLORS["hard"], label="Hard"),
    ]
    ax_b.legend(handles=route_handles, frameon=False, loc="lower right",
                bbox_to_anchor=(1.0, 1.01), borderaxespad=0, ncol=2)

    method_line_styles = {
        "fixed_robust_cascade": (COLORS["fixed"], "--"),
        "always_hard": (COLORS["fixed_dark"], ":"),
        "counterfactual_cghik_v4": (COLORS["proposed"], "-"),
    }
    panel(ax_c, "c", "Frame latency", x=-0.17)
    phase_background(ax_c)
    for method in METHOD_ORDER:
        selected = sorted((row for row in trace if row["method"] == method),
                          key=lambda row: int(row["frame"]))
        times = np.asarray([number(row, "time_seconds") for row in selected])
        latency = np.asarray([number(row, "latency_ms") for row in selected])
        if np.any(latency <= 0):
            raise ValueError("log latency panel requires strictly positive values")
        color, linestyle = method_line_styles[method]
        ax_c.plot(times, latency, color=color, linestyle=linestyle,
                  label=METHOD_LABELS[method], zorder=2 if method == "counterfactual_cghik_v4" else 1)
    hard_trace = sorted((row for row in trace if row["method"] == "always_hard"),
                        key=lambda row: int(row["frame"]))
    rejected_times = [number(row, "time_seconds") for row in hard_trace if row["accepted"] == "False"]
    rejected_latency = [number(row, "latency_ms") for row in hard_trace if row["accepted"] == "False"]
    ax_c.scatter(rejected_times, rejected_latency, marker="x", s=10,
                 color=COLORS["vermillion"], linewidth=0.65, label="Rejected hard frame", zorder=3)
    ax_c.set_yscale("log")
    # Explicit plain-number ticks avoid sub-5-pt mathtext exponents in PDF.
    ax_c.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax_c.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_c.set_ylim(1.0, 130)
    ax_c.set_ylabel("Latency (ms)")
    ax_c.tick_params(axis="x", labelbottom=False)
    ax_c.legend(frameon=False, loc="upper left", ncol=2)
    grid_y(ax_c)

    panel(ax_d, "d", "Function evaluations", x=-0.17)
    phase_background(ax_d)
    for method in METHOD_ORDER:
        selected = sorted((row for row in trace if row["method"] == method),
                          key=lambda row: int(row["frame"]))
        times = np.asarray([number(row, "time_seconds") for row in selected])
        fev = np.asarray([number(row, "function_evaluations") for row in selected])
        color, linestyle = method_line_styles[method]
        ax_d.plot(times, fev, color=color, linestyle=linestyle,
                  zorder=2 if method == "counterfactual_cghik_v4" else 1)
    ax_d.set_xlim(0, 2.98)
    ax_d.set_ylim(0, 270)
    ax_d.set_xlabel("Trajectory time (s)")
    ax_d.set_ylabel("Function evaluations")
    grid_y(ax_d)

    fig.text(
        0.5,
        0.012,
        "a, n=20 fresh trajectories per robot and family; points are exact family aggregates. "
        "b–d, one deterministic Panda near-singular trajectory selected by the frozen rule\n"
        "(illustrative, not an independent statistical unit). Each method evolves its own "
        "closed-loop state; FEV denotes function evaluations.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["fixed_dark"],
    )
    save(fig, "figure5_family_and_timeline")


def main() -> None:
    figure1_framework()
    figure2_heterogeneity()
    figure3_point_results()
    figure4_trajectory_results()
    figure5_family_timeline()
    print("Created five main-paper figures in SVG, PDF, and 600-dpi PNG formats.")


if __name__ == "__main__":
    main()
