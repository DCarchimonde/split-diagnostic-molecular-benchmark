from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_GAP_PATH = TABLE_DIR / "paper1_main_gap_table.csv"
SHIFT_PATH = TABLE_DIR / "paper1_split_shift_table.csv"
SIM_PATH = TABLE_DIR / "paper1_similarity_audit_compact.csv"
DATASET_SUMMARY_PATH = TABLE_DIR / "manuscript_table1_dataset_summary.csv"
RANKING_PATH = TABLE_DIR / "paper1_model_rankings_by_split.csv"
OUTLIER_PATH = TABLE_DIR / "paper1_outlier_cases.csv"

for path in [MAIN_GAP_PATH, SHIFT_PATH, SIM_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run upstream scripts first.")

main_gap = pd.read_csv(MAIN_GAP_PATH)
shift = pd.read_csv(SHIFT_PATH)
sim = pd.read_csv(SIM_PATH)
summary = pd.read_csv(DATASET_SUMMARY_PATH) if DATASET_SUMMARY_PATH.exists() else None

COLORS = {
    "random": "#5AA6C8",
    "ordinary": "#4C5B70",
    "balanced": "#63B39B",
    "accent": "#C77D2E",
    "missing": "#EEF0F2",
    "text": "#222222",
    "grid": "#D8DEE3",
}

SPLIT_LABELS = {
    "random": "Random",
    "split_random": "Random",
    "scaffold": "Ord. scaffold",
    "ordinary_scaffold": "Ord. scaffold",
    "split_scaffold": "Ord. scaffold",
    "balanced_scaffold": "Balanced scaffold",
    "target_balanced_scaffold": "Balanced scaffold",
}

MODEL_SHORT = {
    "RandomForest": "RF",
    "LogisticRegression": "LR",
    "XGBoost": "XGB",
    "Ridge": "Ridge",
}

MODEL_COLOR = {
    "RandomForest": COLORS["random"],
    "LogisticRegression": "#8B8FA8",
    "XGBoost": COLORS["balanced"],
    "Ridge": COLORS["ordinary"],
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 350,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    out = FIGURE_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str, size: int = 8) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=0.18,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=COLORS["text"], linespacing=1.2)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.04, 1.08, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


# -----------------------------------------------------------------------------
# Figure 1: workflow overview
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14.2, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1.0], hspace=0.42, wspace=0.30)

ax = fig.add_subplot(gs[0, 0])
ax.set_axis_off()
panel_label(ax, "A")
boxes = [
    (0.04, 0.68, "Public molecular\nproperty datasets", COLORS["random"]),
    (0.36, 0.68, "Canonical SMILES\nMorgan fingerprints", COLORS["balanced"]),
    (0.68, 0.68, "Random, scaffold,\nand balanced splits", COLORS["ordinary"]),
    (0.20, 0.36, "Models evaluated\nacross five seeds", COLORS["accent"]),
    (0.52, 0.36, "Split diagnostics\nand interpretation", COLORS["balanced"]),
]
for x, y, text, color in boxes:
    draw_box(ax, x, y, 0.24, 0.17, text, color)
for start, end in [((0.28, 0.765), (0.36, 0.765)), ((0.60, 0.765), (0.68, 0.765)), ((0.80, 0.68), (0.38, 0.53)), ((0.44, 0.445), (0.52, 0.445))]:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["text"], shrinkA=2, shrinkB=2))
ax.text(0.04, 0.17, "Central question", fontsize=10, fontweight="bold")
ax.text(0.04, 0.055, "Does a benchmark score reflect molecular generalization,\nor artifacts introduced by the train/test split?", fontsize=9.5, linespacing=1.25)

ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "B")
if summary is not None:
    overview = summary.copy()
    n_col = "after_dedup" if "after_dedup" in overview.columns else overview.columns[-1]
    overview["n"] = overview[n_col]
    colors = [COLORS["random"] if t == "classification" else COLORS["balanced"] for t in overview["task_type"]]
    ax.barh(overview["dataset"], overview["n"], color=colors, edgecolor="white")
    ax.set_xlabel("Molecules after preprocessing")
    ax.set_title("Six public molecular property datasets")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS["random"], label="Classification"), plt.Rectangle((0, 0), 1, 1, color=COLORS["balanced"], label="Regression")], frameon=False, loc="lower right")
else:
    ax.text(0.5, 0.5, "Dataset summary table not found", ha="center", va="center")
    ax.set_axis_off()

ax = fig.add_subplot(gs[1, 0])
ax.set_axis_off()
panel_label(ax, "C")
for i, (head, body) in enumerate([
    ("Target shift", "Train/test label or property mean gap"),
    ("Scaffold statistics", "Shared scaffolds, largest scaffold, singleton fraction"),
    ("Chemical similarity", "Maximum test-to-train Tanimoto similarity"),
    ("Model stability", "Split-dependent performance gaps and rankings"),
]):
    yy = 0.75 - i * 0.22
    draw_box(ax, 0.04, yy, 0.24, 0.13, head, COLORS["ordinary"], size=9)
    ax.text(0.33, yy + 0.065, body, va="center", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "D")
interpretation = pd.DataFrame({
    "Source": ["Random split", "Ordinary scaffold", "Target-balanced scaffold"],
    "Chemical separation": [0.25, 0.85, 0.65],
    "Target-shift control": [0.90, 0.35, 0.75],
})
x = np.arange(len(interpretation))
width = 0.34
ax.bar(x - width / 2, interpretation["Chemical separation"], width, label="Chemical separation", color=COLORS["ordinary"])
ax.bar(x + width / 2, interpretation["Target-shift control"], width, label="Target-shift control", color=COLORS["balanced"])
ax.set_xticks(x)
ax.set_xticklabels(interpretation["Source"], rotation=18, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Conceptual emphasis")
ax.set_title("Different splits answer different questions")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.94, hspace=0.42, wspace=0.30)
save(fig, "figure1_split_diagnostic_workflow.png", tight=False)

# -----------------------------------------------------------------------------
# Figure 2: generalization gap
# -----------------------------------------------------------------------------
plot_gap = main_gap.copy()
plot_gap["label"] = plot_gap["dataset"] + " / " + plot_gap["model"]
plot_gap = plot_gap.sort_values(["task_type", "dataset", "model"]).reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharex=False)
for ax, task, title in zip(axes, ["classification", "regression"], ["Classification: ROC-AUC gap", "Regression: RMSE gap"]):
    data = plot_gap[plot_gap["task_type"] == task].copy().sort_values("ordinary_gap_mean")
    y = np.arange(len(data))
    ax.barh(y - 0.18, data["ordinary_gap_mean"], height=0.34, color=COLORS["ordinary"], label="Ordinary scaffold")
    ax.barh(y + 0.18, data["balanced_gap_mean"], height=0.34, color=COLORS["balanced"], label="Target-balanced scaffold")
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_title(title)
    ax.set_xlabel("Generalization gap")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
axes[0].set_ylabel("Dataset / model")
fig.suptitle("Split-induced performance gaps are model- and dataset-dependent", fontsize=14, fontweight="bold")
save(fig, "figure2_generalization_gap.png")

# -----------------------------------------------------------------------------
# Figure 3: target shift
# -----------------------------------------------------------------------------
pivot = shift.pivot(index="dataset", columns="split_type", values="mean_abs_target_gap").reset_index()
for col in ["split_random", "split_scaffold", "balanced_scaffold"]:
    if col not in pivot.columns:
        pivot[col] = 0.0
pivot = pivot.sort_values("dataset").reset_index(drop=True)
x = np.arange(len(pivot))
fig, ax = plt.subplots(figsize=(11.5, 5.8))
ax.bar(x - 0.25, pivot["split_random"], width=0.25, label="Random", color=COLORS["random"])
ax.bar(x, pivot["split_scaffold"], width=0.25, label="Ordinary scaffold", color=COLORS["ordinary"])
ax.bar(x + 0.25, pivot["balanced_scaffold"], width=0.25, label="Target-balanced scaffold", color=COLORS["balanced"])
ax.set_xticks(x)
ax.set_xticklabels(pivot["dataset"])
ax.set_ylabel("Mean absolute target gap")
ax.set_title("Target shift is a hidden confounder of scaffold-based evaluation")
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
ax.legend(frameon=False)
save(fig, "figure3_target_shift.png")

# -----------------------------------------------------------------------------
# Figure 4: test-to-train similarity
# -----------------------------------------------------------------------------
sim_label = {"random": "Random", "ordinary_scaffold": "Ordinary scaffold", "scaffold": "Ordinary scaffold", "balanced_scaffold": "Target-balanced scaffold"}
sim_plot = sim[sim["split"].isin(sim_label.keys())].copy()
sim_plot["split_label"] = sim_plot["split"].map(sim_label)
sim_pivot = sim_plot.pivot(index="dataset", columns="split_label", values="mean_max_tanimoto").reset_index()
for col in ["Random", "Ordinary scaffold", "Target-balanced scaffold"]:
    if col not in sim_pivot.columns:
        sim_pivot[col] = 0.0
sim_pivot = sim_pivot.sort_values("dataset").reset_index(drop=True)
x = np.arange(len(sim_pivot))
fig, ax = plt.subplots(figsize=(11.5, 5.8))
ax.bar(x - 0.25, sim_pivot["Random"], width=0.25, label="Random", color=COLORS["random"])
ax.bar(x, sim_pivot["Ordinary scaffold"], width=0.25, label="Ordinary scaffold", color=COLORS["ordinary"])
ax.bar(x + 0.25, sim_pivot["Target-balanced scaffold"], width=0.25, label="Target-balanced scaffold", color=COLORS["balanced"])
ax.set_xticks(x)
ax.set_xticklabels(sim_pivot["dataset"])
ax.set_ylim(0, max(0.9, float(sim_pivot[["Random", "Ordinary scaffold", "Target-balanced scaffold"]].max().max()) + 0.08))
ax.set_ylabel("Mean maximum test-to-train Tanimoto")
ax.set_title("Random splits place test molecules closer to the training set")
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
ax.legend(frameon=False)
save(fig, "figure4_tanimoto_similarity.png")

# -----------------------------------------------------------------------------
# Figure S1: model-ranking stability matrix
# -----------------------------------------------------------------------------
if RANKING_PATH.exists():
    rank = pd.read_csv(RANKING_PATH)
    rank = rank[rank["rank"] == 1].copy()
    rank["split_label"] = rank["split"].map(SPLIT_LABELS).fillna(rank["split"])
    rank["model_short"] = rank["model"].map(MODEL_SHORT).fillna(rank["model"])

    datasets = sorted(rank["dataset"].unique())
    split_cols = ["Random", "Ord. scaffold", "Balanced scaffold"]
    top_models = sorted(rank["model"].unique())

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for yi, dataset in enumerate(datasets):
        for xi, split_name in enumerate(split_cols):
            row = rank[(rank["dataset"] == dataset) & (rank["split_label"] == split_name)]
            if row.empty:
                facecolor, text, text_color = COLORS["missing"], "NA", "#666666"
            else:
                model = row.iloc[0]["model"]
                facecolor = MODEL_COLOR.get(model, "#8B8FA8")
                text = row.iloc[0]["model_short"]
                text_color = "white"
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=2.0))
            ax.text(xi + 0.5, yi + 0.5, text, ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")

    ax.set_xlim(0, len(split_cols))
    ax.set_ylim(0, len(datasets))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(split_cols)) + 0.5)
    ax.set_xticklabels(split_cols)
    ax.set_yticks(np.arange(len(datasets)) + 0.5)
    ax.set_yticklabels(datasets)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Top-ranked model can change with the split protocol")
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR.get(m, "#8B8FA8"), label=MODEL_SHORT.get(m, m)) for m in top_models]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=min(len(handles), 4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.82)
    save(fig, "figureS1_model_ranking_stability.png", tight=False)
else:
    print(f"skipped Figure S1 because {RANKING_PATH} does not exist")

# -----------------------------------------------------------------------------
# Figure S2: outlier and sensitivity cases
# -----------------------------------------------------------------------------
if OUTLIER_PATH.exists():
    outliers = pd.read_csv(OUTLIER_PATH).copy()
    outliers["label"] = outliers["dataset"] + " / " + outliers["model"]
    outliers = outliers.sort_values("gap_reduction_after_balancing")
    y = np.arange(len(outliers))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.barh(y, outliers["gap_reduction_after_balancing"], color=np.where(outliers["gap_reduction_after_balancing"] >= 0, COLORS["balanced"], COLORS["accent"]), edgecolor="white")
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(outliers["label"])
    ax.set_xlabel("Gap reduction after target balancing")
    ax.set_title("Sensitivity cases reveal dataset- and model-dependent split effects")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    save(fig, "figureS2_outlier_sensitivity_cases.png")
else:
    print(f"skipped Figure S2 because {OUTLIER_PATH} does not exist")

# Rounded manuscript tables
rounded_gap = main_gap.copy()
for col in ["ordinary_gap_mean", "ordinary_gap_std", "balanced_gap_mean", "balanced_gap_std", "gap_reduction_after_balancing"]:
    if col in rounded_gap.columns:
        rounded_gap[col] = rounded_gap[col].round(3)
rounded_gap.to_csv(TABLE_DIR / "paper1_main_gap_table_rounded.csv", index=False)

rounded_shift = shift.copy()
if "mean_abs_target_gap" in rounded_shift.columns:
    rounded_shift["mean_abs_target_gap"] = rounded_shift["mean_abs_target_gap"].round(4)
rounded_shift.to_csv(TABLE_DIR / "paper1_split_shift_table_rounded.csv", index=False)

rounded_sim = sim_plot.copy()
for col in ["mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto", "p95_max_tanimoto", "frac_test_ge_0_7", "frac_test_ge_0_8", "frac_test_ge_0_9"]:
    if col in rounded_sim.columns:
        rounded_sim[col] = rounded_sim[col].round(3)
rounded_sim.to_csv(TABLE_DIR / "paper1_similarity_audit_compact_rounded.csv", index=False)

print("saved rounded manuscript tables")
