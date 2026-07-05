from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

main_gap = pd.read_csv(TABLE_DIR / "paper1_main_gap_table.csv")
perf = pd.read_csv(TABLE_DIR / "paper1_combined_split_performance.csv")
shift = pd.read_csv(TABLE_DIR / "paper1_split_shift_table.csv")

rows = []

for _, row in main_gap.iterrows():
    note = []
    if row["ordinary_gap_mean"] < 0:
        note.append("ordinary_gap_negative")
    if row["balanced_gap_mean"] < 0:
        note.append("balanced_gap_negative")
    if abs(row["gap_reduction_after_balancing"]) > 0.5:
        note.append("large_gap_change_after_balancing")
    if row["dataset"] == "ClinTox":
        note.append("class_imbalance_sensitive_dataset")
    if row["dataset"] == "ESOL" and row["model"] == "Ridge":
        note.append("ridge_behavior_differs_from_tree_models")
    if note:
        out = row.to_dict()
        out["flags"] = ";".join(note)
        rows.append(out)

outliers = pd.DataFrame(rows)
outliers_path = TABLE_DIR / "paper1_outlier_cases.csv"
outliers.to_csv(outliers_path, index=False)

# Model ranking sanity checks.
rank_rows = []
for (dataset, task_type, split), group in perf.groupby(["dataset", "task_type", "split"]):
    if task_type == "classification":
        metric = "roc_auc_mean"
        group = group.dropna(subset=[metric]).sort_values(metric, ascending=False)
    else:
        metric = "rmse_mean"
        group = group.dropna(subset=[metric]).sort_values(metric, ascending=True)
    for rank, (_, row) in enumerate(group.iterrows(), start=1):
        rank_rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "split": split,
            "rank": rank,
            "model": row["model"],
            "metric": metric,
            "value": float(row[metric]),
        })

rank_df = pd.DataFrame(rank_rows)
rank_path = TABLE_DIR / "paper1_model_rankings_by_split.csv"
rank_df.to_csv(rank_path, index=False)

print("saved", outliers_path)
print("saved", rank_path)
