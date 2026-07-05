from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.splitting import generate_scaffold

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 2024, 2026, 3407, 123]


def make_balanced_split(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> pd.DataFrame:
    out = df.copy()
    if "scaffold" not in out.columns:
        out["scaffold"] = out["canonical_smiles"].map(generate_scaffold)
    target_n = int(round(len(out) * test_size))
    global_mean = float(out["target"].mean())
    target_std = float(out["target"].std(ddof=0)) or 1.0

    groups = []
    for _, sub in out.groupby("scaffold"):
        groups.append((sub.index.to_numpy(), int(len(sub)), float(sub["target"].sum())))

    best = None
    best_score = float("inf")
    for trial in range(300):
        rng = np.random.default_rng(seed * 1000 + trial)
        remaining = list(range(len(groups)))
        selected = []
        selected_n = 0
        selected_sum = 0.0
        while selected_n < max(1, int(target_n * 0.95)) and remaining:
            candidates = rng.choice(remaining, size=min(80, len(remaining)), replace=False)
            cand_best = None
            cand_score = float("inf")
            for c in candidates:
                _, n, s = groups[int(c)]
                new_n = selected_n + n
                if new_n > max(target_n * 1.25, target_n + 1) and selected_n > 0:
                    continue
                new_mean = (selected_sum + s) / max(new_n, 1)
                score = abs(new_n - target_n) / max(target_n, 1) + abs(new_mean - global_mean) / target_std
                if score < cand_score:
                    cand_score = score
                    cand_best = int(c)
            if cand_best is None:
                break
            _, n, s = groups[cand_best]
            selected.append(cand_best)
            selected_n += n
            selected_sum += s
            remaining.remove(cand_best)
        if selected_n == 0:
            continue
        score = abs(selected_n - target_n) / max(target_n, 1) + abs((selected_sum / selected_n) - global_mean) / target_std
        if score < best_score:
            best_score = score
            best = selected
    if best is None:
        raise RuntimeError("balanced split failed")
    test_idx = np.concatenate([groups[i][0] for i in best])
    out["split_balanced_scaffold"] = "train"
    out.loc[test_idx, "split_balanced_scaffold"] = "test"
    return out


def stats(df: pd.DataFrame, dataset: str, task_type: str, seed: int) -> dict:
    train = df[df["split_balanced_scaffold"] == "train"]
    test = df[df["split_balanced_scaffold"] == "test"]
    train_scaf = set(train["scaffold"])
    test_scaf = set(test["scaffold"])
    counts = df["scaffold"].value_counts()
    shared = train_scaf.intersection(test_scaf)
    return {
        "dataset": dataset,
        "task_type": task_type,
        "seed": seed,
        "split": "balanced_scaffold",
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_scaffolds_total": int(counts.shape[0]),
        "largest_scaffold_fraction": float(counts.iloc[0] / len(df)),
        "singleton_scaffold_fraction": float((counts == 1).sum() / max(counts.shape[0], 1)),
        "n_train_scaffolds": int(len(train_scaf)),
        "n_test_scaffolds": int(len(test_scaf)),
        "n_shared_scaffolds": int(len(shared)),
        "shared_scaffold_fraction_test": float(len(shared) / max(len(test_scaf), 1)),
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
    base["scaffold"] = base["canonical_smiles"].map(generate_scaffold)
    for seed in SEEDS:
        rows.append(stats(make_balanced_split(base, seed), dataset, spec.task_type, seed))

raw = pd.DataFrame(rows)
raw_path = TABLE_DIR / "paper1_balanced_scaffold_stats.csv"
raw.to_csv(raw_path, index=False)

numeric_cols = [
    "n_total",
    "n_train",
    "n_test",
    "n_scaffolds_total",
    "largest_scaffold_fraction",
    "singleton_scaffold_fraction",
    "n_train_scaffolds",
    "n_test_scaffolds",
    "n_shared_scaffolds",
    "shared_scaffold_fraction_test",
    "train_target_mean",
    "test_target_mean",
    "target_mean_gap_test_minus_train",
]
summary = raw.groupby(["dataset", "task_type", "split"], as_index=False)[numeric_cols].mean()
summary_path = TABLE_DIR / "paper1_balanced_scaffold_stats_summary.csv"
summary.to_csv(summary_path, index=False)

rounded = summary.copy()
for col in ["largest_scaffold_fraction", "singleton_scaffold_fraction", "shared_scaffold_fraction_test", "target_mean_gap_test_minus_train"]:
    rounded[col] = rounded[col].round(4)
rounded_path = TABLE_DIR / "paper1_balanced_scaffold_stats_summary_rounded.csv"
rounded.to_csv(rounded_path, index=False)

print("saved", raw_path)
print("saved", summary_path)
print("saved", rounded_path)
