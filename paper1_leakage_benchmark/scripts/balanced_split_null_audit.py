from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.splitting import add_scaffold_split, generate_scaffold
from robustness_20seeds import SEED_POOL, add_balanced_scaffold_split

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare target-balanced scaffold splits with random scaffold assignments.")
    parser.add_argument("--n-seeds", type=int, default=20, choices=range(5, len(SEED_POOL) + 1))
    parser.add_argument("--null-draws", type=int, default=5000)
    parser.add_argument("--datasets", type=str, default="all")
    return parser.parse_args()


def scaffold_groups(df: pd.DataFrame) -> list[dict]:
    work = df.copy()
    if "scaffold" not in work.columns:
        work["scaffold"] = work["canonical_smiles"].map(generate_scaffold)
    groups = []
    for scaffold, sub in work.groupby("scaffold"):
        groups.append({
            "scaffold": scaffold,
            "indices": sub.index.to_numpy(),
            "n": int(len(sub)),
        })
    return groups


def random_scaffold_assignment(
    df: pd.DataFrame,
    groups: list[dict],
    rng: np.random.Generator,
    test_size: float = 0.2,
) -> dict | None:
    target_n = int(round(len(df) * test_size))
    order = rng.permutation(len(groups))
    selected = []
    selected_n = 0

    for idx in order:
        group_n = groups[int(idx)]["n"]
        if selected_n >= max(1, int(target_n * 0.95)):
            break
        if selected_n > 0 and selected_n + group_n > max(target_n * 1.25, target_n + 1):
            continue
        selected.append(int(idx))
        selected_n += group_n

    if not selected:
        return None

    test_indices = np.concatenate([groups[i]["indices"] for i in selected])
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_indices] = False
    test_values = df.loc[test_indices, "target"].to_numpy(dtype=float)
    train_values = df.loc[train_mask, "target"].to_numpy(dtype=float)

    return {
        "n_test": int(len(test_indices)),
        "test_fraction": float(len(test_indices) / len(df)),
        "size_deviation": float(abs(len(test_indices) - target_n) / max(target_n, 1)),
        "target_gap": float(np.mean(test_values) - np.mean(train_values)),
        "abs_target_gap": float(abs(np.mean(test_values) - np.mean(train_values))),
    }


def split_diagnostics(df: pd.DataFrame, split_col: str) -> dict:
    test_mask = df[split_col].to_numpy() == "test"
    train_mask = ~test_mask
    test_values = df.loc[test_mask, "target"].to_numpy(dtype=float)
    train_values = df.loc[train_mask, "target"].to_numpy(dtype=float)
    target_n = int(round(len(df) * 0.2))
    return {
        "n_test": int(test_mask.sum()),
        "test_fraction": float(test_mask.mean()),
        "size_deviation": float(abs(test_mask.sum() - target_n) / max(target_n, 1)),
        "target_gap": float(np.mean(test_values) - np.mean(train_values)),
        "abs_target_gap": float(abs(np.mean(test_values) - np.mean(train_values))),
    }


def main() -> None:
    args = parse_args()
    seeds = SEED_POOL[: args.n_seeds]
    selected_datasets = list(DATASETS.keys()) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]

    null_rows = []
    balanced_rows = []

    for dataset_name in selected_datasets:
        print(f"\n========== {dataset_name} ==========")
        base_df = pd.read_csv(PROCESSED_DIR / f"{dataset_name.lower()}_splits.csv")
        groups = scaffold_groups(base_df)
        rng = np.random.default_rng(20260722 + sum(ord(c) for c in dataset_name))

        dataset_null = []
        for draw in range(args.null_draws):
            result = random_scaffold_assignment(base_df, groups, rng)
            if result is None:
                continue
            row = {
                "dataset": dataset_name,
                "task_type": DATASETS[dataset_name].task_type,
                "draw": draw,
            }
            row.update(result)
            null_rows.append(row)
            dataset_null.append(result["abs_target_gap"])

        null_values = np.asarray(dataset_null, dtype=float)
        ordinary_df = add_scaffold_split(base_df, smiles_col="canonical_smiles")
        ordinary_diag = split_diagnostics(ordinary_df, "split_scaffold")

        for seed in seeds:
            balanced_df, meta = add_balanced_scaffold_split(base_df, seed=seed)
            diag = split_diagnostics(balanced_df, "split_balanced_scaffold")
            fraction_null_as_good = float(np.mean(null_values <= diag["abs_target_gap"]))
            row = {
                "dataset": dataset_name,
                "task_type": DATASETS[dataset_name].task_type,
                "seed": seed,
                "ordinary_abs_target_gap": ordinary_diag["abs_target_gap"],
                "balanced_abs_target_gap": diag["abs_target_gap"],
                "balanced_size_deviation": diag["size_deviation"],
                "balanced_objective": meta["objective"],
                "fraction_null_assignments_as_good_or_better": fraction_null_as_good,
                "balanced_improvement_percentile": float(1.0 - fraction_null_as_good),
                "null_abs_target_gap_median": float(np.median(null_values)),
                "null_abs_target_gap_p05": float(np.quantile(null_values, 0.05)),
                "null_abs_target_gap_p95": float(np.quantile(null_values, 0.95)),
            }
            balanced_rows.append(row)

    null_df = pd.DataFrame(null_rows)
    balanced_df = pd.DataFrame(balanced_rows)

    null_path = TABLE_DIR / "paper1_scaffold_null_assignments.csv"
    balanced_path = TABLE_DIR / "paper1_balanced_split_null_percentiles.csv"
    summary_path = TABLE_DIR / "paper1_balanced_split_null_summary.csv"

    null_df.to_csv(null_path, index=False)
    balanced_df.to_csv(balanced_path, index=False)

    summary = balanced_df.groupby(["dataset", "task_type"], as_index=False).agg(
        n_seeds=("seed", "count"),
        ordinary_abs_target_gap=("ordinary_abs_target_gap", "mean"),
        balanced_abs_target_gap_mean=("balanced_abs_target_gap", "mean"),
        balanced_abs_target_gap_std=("balanced_abs_target_gap", "std"),
        balanced_improvement_percentile_mean=("balanced_improvement_percentile", "mean"),
        balanced_improvement_percentile_min=("balanced_improvement_percentile", "min"),
        balanced_size_deviation_mean=("balanced_size_deviation", "mean"),
        null_abs_target_gap_median=("null_abs_target_gap_median", "mean"),
    )
    summary.to_csv(summary_path, index=False)

    print("saved", null_path)
    print("saved", balanced_path)
    print("saved", summary_path)


if __name__ == "__main__":
    main()
