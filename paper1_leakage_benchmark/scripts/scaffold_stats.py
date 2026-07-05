from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.splitting import add_random_split, add_scaffold_split, generate_scaffold

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 2024, 2026, 3407, 123]


def split_stats(df: pd.DataFrame, split_col: str, split_name: str, dataset: str, task_type: str, seed: int | None) -> dict:
    train = df[df[split_col] == "train"].copy()
    test = df[df[split_col] == "test"].copy()
    train_scaffolds = set(train["scaffold"].tolist())
    test_scaffolds = set(test["scaffold"].tolist())
    shared = train_scaffolds.intersection(test_scaffolds)
    scaffold_counts = df["scaffold"].value_counts()

    return {
        "dataset": dataset,
        "task_type": task_type,
        "seed": seed if seed is not None else "fixed",
        "split": split_name,
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_scaffolds_total": int(scaffold_counts.shape[0]),
        "largest_scaffold_size": int(scaffold_counts.iloc[0]),
        "largest_scaffold_fraction": float(scaffold_counts.iloc[0] / len(df)),
        "singleton_scaffold_fraction": float((scaffold_counts == 1).sum() / max(scaffold_counts.shape[0], 1)),
        "n_train_scaffolds": int(len(train_scaffolds)),
        "n_test_scaffolds": int(len(test_scaffolds)),
        "n_shared_scaffolds": int(len(shared)),
        "shared_scaffold_fraction_test": float(len(shared) / max(len(test_scaffolds), 1)),
        "train_target_mean": float(train["target"].mean()),
        "test_target_mean": float(test["target"].mean()),
        "target_mean_gap_test_minus_train": float(test["target"].mean() - train["target"].mean()),
    }


rows = []

for dataset, spec in DATASETS.items():
    path = PROCESSED_DIR / f"{dataset.lower()}_splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 02_featurize_and_split.py first.")
    base = pd.read_csv(path)
    if "scaffold" not in base.columns:
        base["scaffold"] = base["canonical_smiles"].map(generate_scaffold)

    fixed_scaffold = add_scaffold_split(base, smiles_col="canonical_smiles")
    rows.append(split_stats(fixed_scaffold, "split_scaffold", "ordinary_scaffold", dataset, spec.task_type, None))

    for seed in SEEDS:
        random_df = add_random_split(base, target_col="target", task_type=spec.task_type, random_state=seed)
        rows.append(split_stats(random_df, "split_random", "random", dataset, spec.task_type, seed))

stats = pd.DataFrame(rows)
stats_path = TABLE_DIR / "paper1_scaffold_stats.csv"
stats.to_csv(stats_path, index=False)

summary = stats.groupby(["dataset", "task_type", "split"], as_index=False).agg({
    "n_scaffolds_total": "mean",
    "largest_scaffold_fraction": "mean",
    "singleton_scaffold_fraction": "mean",
    "n_train_scaffolds": "mean",
    "n_test_scaffolds": "mean",
    "n_shared_scaffolds": "mean",
    "shared_scaffold_fraction_test": "mean",
    "target_mean_gap_test_minus_train": "mean",
})
summary_path = TABLE_DIR / "paper1_scaffold_stats_summary.csv"
summary.to_csv(summary_path, index=False)

print("saved", stats_path)
print("saved", summary_path)
