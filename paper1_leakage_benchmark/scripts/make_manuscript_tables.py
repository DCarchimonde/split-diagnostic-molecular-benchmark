from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def require_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


# Table 1: dataset summary.
dataset = require_csv("dataset_summary.csv")
table1_cols = [
    "dataset",
    "task_type",
    "raw_rows",
    "after_dropna",
    "after_valid_smiles",
    "after_dedup",
    "duplicates_removed",
    "target_mean",
    "target_min",
    "target_max",
]
table1 = dataset[[c for c in table1_cols if c in dataset.columns]].copy()
for col in ["target_mean", "target_min", "target_max"]:
    if col in table1.columns:
        table1[col] = table1[col].round(4)
table1.to_csv(TABLE_DIR / "manuscript_table1_dataset_summary.csv", index=False)

# Table 2: split target shift and scaffold statistics.
scaffold = require_csv("paper1_scaffold_stats_summary.csv")
balanced_scaffold_path = TABLE_DIR / "paper1_balanced_scaffold_stats_summary.csv"
if balanced_scaffold_path.exists():
    scaffold = pd.concat([scaffold, pd.read_csv(balanced_scaffold_path)], ignore_index=True, sort=False)

keep = [
    "dataset",
    "task_type",
    "split",
    "n_scaffolds_total",
    "largest_scaffold_fraction",
    "singleton_scaffold_fraction",
    "n_train_scaffolds",
    "n_test_scaffolds",
    "n_shared_scaffolds",
    "shared_scaffold_fraction_test",
    "target_mean_gap_test_minus_train",
]
scaffold_simple = scaffold[[c for c in keep if c in scaffold.columns]].copy()
for col in [
    "n_scaffolds_total",
    "n_train_scaffolds",
    "n_test_scaffolds",
    "n_shared_scaffolds",
]:
    if col in scaffold_simple.columns:
        scaffold_simple[col] = scaffold_simple[col].round(1)
for col in [
    "largest_scaffold_fraction",
    "singleton_scaffold_fraction",
    "shared_scaffold_fraction_test",
    "target_mean_gap_test_minus_train",
]:
    if col in scaffold_simple.columns:
        scaffold_simple[col] = scaffold_simple[col].round(4)
scaffold_simple.to_csv(TABLE_DIR / "manuscript_table2_split_diagnostics.csv", index=False)

# Table 3: generalization gap.
gap = require_csv("paper1_main_gap_table_rounded.csv")
gap.to_csv(TABLE_DIR / "manuscript_table3_generalization_gap.csv", index=False)

# Table 4: similarity audit.
sim = require_csv("paper1_similarity_audit_compact_rounded.csv")
sim.to_csv(TABLE_DIR / "manuscript_table4_similarity_audit.csv", index=False)

# Optional appendix tables.
if (TABLE_DIR / "paper1_outlier_cases.csv").exists():
    require_csv("paper1_outlier_cases.csv").to_csv(TABLE_DIR / "appendix_table_outlier_cases.csv", index=False)
if (TABLE_DIR / "paper1_model_rankings_by_split.csv").exists():
    require_csv("paper1_model_rankings_by_split.csv").to_csv(TABLE_DIR / "appendix_table_model_rankings.csv", index=False)

print("saved manuscript_table1_dataset_summary.csv")
print("saved manuscript_table2_split_diagnostics.csv")
print("saved manuscript_table3_generalization_gap.csv")
print("saved manuscript_table4_similarity_audit.csv")
