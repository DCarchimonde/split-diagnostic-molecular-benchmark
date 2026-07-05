from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


many = read_csv_required(TABLE_DIR / "paper1_many_summary.csv")
balanced = read_csv_required(TABLE_DIR / "paper1_balanced_summary.csv")
ordinary_diag = read_csv_required(TABLE_DIR / "paper1_split_diagnostics.csv")
balanced_diag = read_csv_required(TABLE_DIR / "paper1_balanced_split_diagnostics.csv")
ordinary_gap = read_csv_required(TABLE_DIR / "paper1_gap_summary.csv")
balanced_gap = read_csv_required(TABLE_DIR / "paper1_balanced_gap_summary.csv")

ordinary_scaffold = many[many["split"].isin(["random", "scaffold"])].copy()
balanced_scaffold = balanced[balanced["split"].isin(["balanced_scaffold"])].copy()
combined_perf = pd.concat([ordinary_scaffold, balanced_scaffold], ignore_index=True, sort=False)
combined_perf_path = TABLE_DIR / "paper1_combined_split_performance.csv"
combined_perf.to_csv(combined_perf_path, index=False)

ordinary_gap = ordinary_gap.copy()
ordinary_gap["comparison"] = "random_vs_scaffold"
balanced_gap = balanced_gap.copy()
balanced_gap["comparison"] = "random_vs_balanced_scaffold"
combined_gap = pd.concat([ordinary_gap, balanced_gap], ignore_index=True, sort=False)
combined_gap_path = TABLE_DIR / "paper1_combined_gap_summary.csv"
combined_gap.to_csv(combined_gap_path, index=False)

ordinary_diag = ordinary_diag.copy()
ordinary_diag["comparison_group"] = "ordinary"
balanced_diag = balanced_diag.copy()
balanced_diag["comparison_group"] = "balanced"
combined_diag = pd.concat([ordinary_diag, balanced_diag], ignore_index=True, sort=False)
combined_diag_path = TABLE_DIR / "paper1_combined_split_diagnostics.csv"
combined_diag.to_csv(combined_diag_path, index=False)

# Compact manuscript-facing table: one row per dataset/model with ordinary and balanced gaps.
ordinary_small = ordinary_gap.rename(columns={"gap_mean": "ordinary_gap_mean", "gap_std": "ordinary_gap_std"})
balanced_small = balanced_gap.rename(columns={"gap_mean": "balanced_gap_mean", "gap_std": "balanced_gap_std"})
keys = ["dataset", "task_type", "model", "primary_metric"]
main_table = ordinary_small[keys + ["ordinary_gap_mean", "ordinary_gap_std"]].merge(
    balanced_small[keys + ["balanced_gap_mean", "balanced_gap_std"]],
    on=keys,
    how="outer",
)
main_table["gap_reduction_after_balancing"] = main_table["ordinary_gap_mean"] - main_table["balanced_gap_mean"]
main_table_path = TABLE_DIR / "paper1_main_gap_table.csv"
main_table.to_csv(main_table_path, index=False)

# Compact split shift table: mean absolute target gap by dataset and split type.
diag_rows = []
for dataset, group in ordinary_diag.groupby("dataset"):
    for split_col, sub in group.groupby("split_col"):
        diag_rows.append({
            "dataset": dataset,
            "split_type": split_col,
            "mean_abs_target_gap": float(sub["target_mean_gap_test_minus_train"].abs().mean()),
        })
for dataset, group in balanced_diag.groupby("dataset"):
    sub = group[group["split"] == "balanced_scaffold"]
    diag_rows.append({
        "dataset": dataset,
        "split_type": "balanced_scaffold",
        "mean_abs_target_gap": float(sub["target_mean_gap_test_minus_train"].abs().mean()),
    })
shift_table = pd.DataFrame(diag_rows)
shift_table_path = TABLE_DIR / "paper1_split_shift_table.csv"
shift_table.to_csv(shift_table_path, index=False)

print("saved", combined_perf_path)
print("saved", combined_gap_path)
print("saved", combined_diag_path)
print("saved", main_table_path)
print("saved", shift_table_path)
