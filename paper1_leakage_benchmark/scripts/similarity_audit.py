from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

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
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return FP_GENERATOR.GetFingerprint(mol)


def max_tanimoto_to_train(test_fp, train_fps: list) -> float:
    if test_fp is None or len(train_fps) == 0:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    if len(sims) == 0:
        return float("nan")
    return float(max(sims))


def add_balanced_scaffold_split(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> pd.DataFrame:
    out = df.copy()
    if "scaffold" not in out.columns:
        out["scaffold"] = out["canonical_smiles"].map(generate_scaffold)

    target_n = int(round(len(out) * test_size))
    global_mean = float(out["target"].mean())
    target_std = float(out["target"].std(ddof=0))
    if target_std == 0 or np.isnan(target_std):
        target_std = 1.0

    groups = []
    for scaffold, sub in out.groupby("scaffold"):
        groups.append({
            "scaffold": scaffold,
            "indices": sub.index.to_numpy(),
            "n": int(len(sub)),
            "target_sum": float(sub["target"].sum()),
        })

    best = None
    best_score = float("inf")

    for trial in range(300):
        remaining = list(range(len(groups)))
        selected = []
        selected_n = 0
        selected_sum = 0.0
        local_rng = np.random.default_rng(seed * 1000 + trial)

        while selected_n < max(1, int(target_n * 0.95)) and remaining:
            sample_size = min(len(remaining), 80)
            candidates = local_rng.choice(remaining, size=sample_size, replace=False)
            best_candidate = None
            best_candidate_score = float("inf")

            for cand in candidates:
                g = groups[int(cand)]
                new_n = selected_n + g["n"]
                if new_n > max(target_n * 1.25, target_n + 1) and selected_n > 0:
                    continue
                new_sum = selected_sum + g["target_sum"]
                new_mean = new_sum / max(new_n, 1)
                size_score = abs(new_n - target_n) / max(target_n, 1)
                mean_score = abs(new_mean - global_mean) / target_std
                score = size_score + mean_score
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = int(cand)

            if best_candidate is None:
                break

            g = groups[best_candidate]
            selected.append(best_candidate)
            selected_n += g["n"]
            selected_sum += g["target_sum"]
            remaining.remove(best_candidate)

        if selected_n == 0:
            continue

        selected_mean = selected_sum / selected_n
        score = abs(selected_n - target_n) / max(target_n, 1) + abs(selected_mean - global_mean) / target_std
        if score < best_score:
            best_score = score
            best = selected

    if best is None:
        raise RuntimeError("Could not create balanced scaffold split")

    test_indices = np.concatenate([groups[i]["indices"] for i in best])
    out["split_balanced_scaffold"] = "train"
    out.loc[test_indices, "split_balanced_scaffold"] = "test"
    return out


def audit_split(df: pd.DataFrame, split_col: str, split_name: str, dataset: str, task_type: str, seed: int | str) -> tuple[list[dict], dict]:
    train = df[df[split_col] == "train"].copy()
    test = df[df[split_col] == "test"].copy()
    train_fps = [fp for fp in train["fp"].tolist() if fp is not None]

    rows = []
    sims = []
    for _, row in test.iterrows():
        sim = max_tanimoto_to_train(row["fp"], train_fps)
        sims.append(sim)
        rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "seed": seed,
            "split": split_name,
            "smiles": row["canonical_smiles"],
            "target": row["target"],
            "max_train_tanimoto": sim,
        })

    values = pd.Series(sims, dtype="float64").dropna()
    summary = {
        "dataset": dataset,
        "task_type": task_type,
        "seed": seed,
        "split": split_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mean_max_tanimoto": float(values.mean()) if len(values) else float("nan"),
        "median_max_tanimoto": float(values.median()) if len(values) else float("nan"),
        "p90_max_tanimoto": float(values.quantile(0.90)) if len(values) else float("nan"),
        "p95_max_tanimoto": float(values.quantile(0.95)) if len(values) else float("nan"),
        "frac_test_ge_0_7": float((values >= 0.7).mean()) if len(values) else float("nan"),
        "frac_test_ge_0_8": float((values >= 0.8).mean()) if len(values) else float("nan"),
        "frac_test_ge_0_9": float((values >= 0.9).mean()) if len(values) else float("nan"),
    }
    return rows, summary


all_detail_rows = []
summary_rows = []

for dataset, spec in DATASETS.items():
    print("auditing", dataset)
    path = PROCESSED_DIR / f"{dataset.lower()}_splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 02_featurize_and_split.py first.")

    base = pd.read_csv(path)
    if "scaffold" not in base.columns:
        base["scaffold"] = base["canonical_smiles"].map(generate_scaffold)
    base["fp"] = base["canonical_smiles"].map(fingerprint)

    scaffold_df = add_scaffold_split(base, smiles_col="canonical_smiles")
    detail, summary = audit_split(scaffold_df, "split_scaffold", "ordinary_scaffold", dataset, spec.task_type, "fixed")
    all_detail_rows.extend(detail)
    summary_rows.append(summary)

    for seed in SEEDS:
        random_df = add_random_split(base, target_col="target", task_type=spec.task_type, random_state=seed)
        detail, summary = audit_split(random_df, "split_random", "random", dataset, spec.task_type, seed)
        all_detail_rows.extend(detail)
        summary_rows.append(summary)

        balanced_df = add_balanced_scaffold_split(base, seed=seed)
        detail, summary = audit_split(balanced_df, "split_balanced_scaffold", "balanced_scaffold", dataset, spec.task_type, seed)
        all_detail_rows.extend(detail)
        summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(all_detail_rows)

summary_path = TABLE_DIR / "paper1_similarity_audit_summary.csv"
detail_path = TABLE_DIR / "paper1_similarity_audit_detail.csv"
summary_df.to_csv(summary_path, index=False)
detail_df.to_csv(detail_path, index=False)

compact = summary_df.groupby(["dataset", "task_type", "split"], as_index=False).agg({
    "mean_max_tanimoto": "mean",
    "median_max_tanimoto": "mean",
    "p90_max_tanimoto": "mean",
    "p95_max_tanimoto": "mean",
    "frac_test_ge_0_7": "mean",
    "frac_test_ge_0_8": "mean",
    "frac_test_ge_0_9": "mean",
})
compact_path = TABLE_DIR / "paper1_similarity_audit_compact.csv"
compact.to_csv(compact_path, index=False)

rounded = compact.copy()
for col in ["mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto", "p95_max_tanimoto", "frac_test_ge_0_7", "frac_test_ge_0_8", "frac_test_ge_0_9"]:
    rounded[col] = rounded[col].round(3)
rounded_path = TABLE_DIR / "paper1_similarity_audit_compact_rounded.csv"
rounded.to_csv(rounded_path, index=False)

print("saved", summary_path)
print("saved", detail_path)
print("saved", compact_path)
print("saved", rounded_path)
