from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.splitting import generate_scaffold

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

rows = []
scaffold_rows = []

for name, spec in DATASETS.items():
    path = PROCESSED_DIR / f"{name.lower()}_splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 02_featurize_and_split.py first.")

    df = pd.read_csv(path)
    if "scaffold" not in df.columns:
        df["scaffold"] = df["canonical_smiles"].map(generate_scaffold)

    n_total = len(df)
    scaffold_counts = df["scaffold"].value_counts()
    n_scaffold = int(scaffold_counts.shape[0])
    largest_scaffold_size = int(scaffold_counts.iloc[0])
    largest_scaffold_frac = float(largest_scaffold_size / n_total)
    singleton_scaffold_frac = float((scaffold_counts == 1).sum() / n_scaffold)

    for split_col in ["split_random", "split_scaffold"]:
        train = df[df[split_col] == "train"].copy()
        test = df[df[split_col] == "test"].copy()
        train_scaffolds = set(train["scaffold"].tolist())
        test_scaffolds = set(test["scaffold"].tolist())
        shared_scaffolds = train_scaffolds.intersection(test_scaffolds)

        row = {
            "dataset": name,
            "task_type": spec.task_type,
            "split_col": split_col,
            "n_total": int(n_total),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_scaffolds_total": n_scaffold,
            "largest_scaffold_size": largest_scaffold_size,
            "largest_scaffold_fraction": largest_scaffold_frac,
            "singleton_scaffold_fraction": singleton_scaffold_frac,
            "n_train_scaffolds": int(len(train_scaffolds)),
            "n_test_scaffolds": int(len(test_scaffolds)),
            "n_shared_scaffolds": int(len(shared_scaffolds)),
            "shared_scaffold_fraction_test": float(len(shared_scaffolds) / max(len(test_scaffolds), 1)),
            "train_target_mean": float(train["target"].mean()),
            "test_target_mean": float(test["target"].mean()),
            "target_mean_gap_test_minus_train": float(test["target"].mean() - train["target"].mean()),
        }

        if spec.task_type == "classification":
            row["train_positive_rate"] = float(train["target"].mean())
            row["test_positive_rate"] = float(test["target"].mean())
            row["positive_rate_gap_test_minus_train"] = float(test["target"].mean() - train["target"].mean())

        rows.append(row)

    top_scaffolds = scaffold_counts.head(20).reset_index()
    top_scaffolds.columns = ["scaffold", "count"]
    for _, item in top_scaffolds.iterrows():
        scaffold_rows.append({
            "dataset": name,
            "scaffold": item["scaffold"],
            "count": int(item["count"]),
            "fraction": float(item["count"] / n_total),
        })

summary = pd.DataFrame(rows)
summary_path = TABLE_DIR / "paper1_split_diagnostics.csv"
summary.to_csv(summary_path, index=False)
print("saved", summary_path)

scaffold_summary = pd.DataFrame(scaffold_rows)
scaffold_path = TABLE_DIR / "paper1_top_scaffolds.csv"
scaffold_summary.to_csv(scaffold_path, index=False)
print("saved", scaffold_path)
