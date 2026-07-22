from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
FIGURE_DIR = PAPER_DIR / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLE_DIR / "paper1_robustness20_summary.csv"
NULL_PATH = TABLE_DIR / "paper1_balanced_split_null_summary.csv"

MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "RandomForest": "Random forest",
    "XGBoost": "XGBoost",
    "Ridge": "Ridge",
}

EFFECT_COLOR = "#2F5D8A"
BAR_COLOR = "#5B9BC4"
REFERENCE_COLOR = "#6E6E6E"


def draw_effect_panel(ax, frame: pd.DataFrame, title: str) -> None:
    frame = frame.copy()
    frame["label"] = frame["dataset"] + " — " + frame["model"].map(MODEL_LABELS)
    frame = frame.sort_values(["dataset", "model"], ascending=[False, False]).reset_index(drop=True)

    y = np.arange(len(frame))
    mean = frame["gap_reduction_mean"].to_numpy(dtype=float)
    low = frame["gap_reduction_ci95_low"].to_numpy(dtype=float)
    high = frame["gap_reduction_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([mean - low, high - mean])

    significant = frame["wilcoxon_p_holm"].to_numpy(dtype=float) < 0.05

    for i, is_significant in enumerate(significant):
        marker = "o" if is_significant else "s"
        marker_face = EFFECT_COLOR if is_significant else "white"
        ax.errorbar(
            mean[i],
            y[i],
            xerr=xerr[:, i : i + 1],
            fmt=marker,
            color=EFFECT_COLOR,
            markerfacecolor=marker_face,
            markeredgecolor=EFFECT_COLOR,
            markeredgewidth=1.2,
            capsize=3,
            markersize=5.5,
            linewidth=1.2,
        )

    ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlabel("Gap reduction (ordinary − target-balanced)", labelpad=7)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.grid(axis="x", alpha=0.22)
    ax.margins(x=0.12, y=0.10)

    note_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.92,
        "pad": 1.0,
    }
    ax.text(
        0.015,
        0.982,
        "← Larger balanced gap",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=note_box,
    )
    ax.text(
        0.985,
        0.982,
        "Smaller balanced gap →",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox=note_box,
    )


def main() -> None:
    robustness = pd.read_csv(SUMMARY_PATH)
    null_summary = pd.read_csv(NULL_PATH)

    classification = robustness[robustness["task_type"] == "classification"].copy()
    regression = robustness[robustness["task_type"] == "regression"].copy()

    fig = plt.figure(figsize=(13.4, 9.8), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.82, 1.18, 0.10],
        hspace=0.16,
        wspace=0.16,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ordered = null_summary.sort_values("balanced_improvement_percentile_mean", ascending=True)
    y = np.arange(len(ordered))
    percentile = 100.0 * ordered["balanced_improvement_percentile_mean"].to_numpy(dtype=float)
    minimum = 100.0 * ordered["balanced_improvement_percentile_min"].to_numpy(dtype=float)

    ax_a.barh(y, percentile, color=BAR_COLOR, alpha=0.90)
    ax_a.scatter(minimum, y, color=EFFECT_COLOR, marker="|", s=150, linewidths=2)
    ax_a.axvline(95.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(ordered["dataset"])
    ax_a.set_xlim(0, 102)
    ax_a.set_xlabel("Random scaffold assignments with worse target balance (%)", labelpad=7)
    ax_a.set_title(
        "A  Target-balanced scaffold assignments relative to the random-scaffold null",
        loc="left",
        fontweight="bold",
        pad=9,
    )
    ax_a.grid(axis="x", alpha=0.22)

    for i, value in enumerate(percentile):
        ax_a.text(value + 0.8, i, f"{value:.1f}%", va="center", fontsize=8)

    ax_a.text(
        0.995,
        0.025,
        "Bars: mean across 20 seeds; vertical marks: minimum across seeds",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax_b = fig.add_subplot(grid[1, 0])
    draw_effect_panel(ax_b, classification, "B  Classification datasets")

    ax_c = fig.add_subplot(grid[1, 1])
    draw_effect_panel(ax_c, regression, "C  Regression datasets")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=EFFECT_COLOR,
            markeredgecolor=EFFECT_COLOR,
            markersize=6,
            label="Holm-adjusted p < 0.05",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=EFFECT_COLOR,
            markeredgewidth=1.2,
            markersize=6,
            label="Not significant after Holm correction",
        ),
    ]

    ax_legend = fig.add_subplot(grid[2, :])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=2.0,
        handletextpad=0.7,
    )

    png_path = FIGURE_DIR / "figure_robustness20_split_effects.png"
    pdf_path = FIGURE_DIR / "figure_robustness20_split_effects.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print("saved", png_path)
    print("saved", pdf_path)


if __name__ == "__main__":
    main()
