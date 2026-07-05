from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.chem_features import build_feature_matrix
from shared_utils.dataset_registry import DATASETS
from shared_utils.splitting import add_random_split, add_scaffold_split, summarize_split

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

all_summaries = []

for name, spec in DATASETS.items():
    print("\n==========", name, "==========")
    clean_path = PROCESSED_DIR / f"{name.lower()}_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}. Run the data preparation script first.")

    df = pd.read_csv(clean_path)
    X, valid_indices = build_feature_matrix(df["canonical_smiles"].tolist())
    df = df.iloc[valid_indices].reset_index(drop=True)
    y = df["target"].to_numpy()

    df = add_random_split(df, target_col="target", task_type=spec.task_type)
    df = add_scaffold_split(df, smiles_col="canonical_smiles")

    feature_path = PROCESSED_DIR / f"{name.lower()}_features.npz"
    split_path = PROCESSED_DIR / f"{name.lower()}_splits.csv"

    np.savez_compressed(feature_path, X=X, y=y, smiles=df["canonical_smiles"].to_numpy())
    df.to_csv(split_path, index=False)

    print("saved", feature_path)
    print("saved", split_path)

    for split_col in ["split_random", "split_scaffold"]:
        item = summarize_split(df, split_col)
        item["dataset"] = name
        item["task_type"] = spec.task_type
        all_summaries.append(item)

summary = pd.DataFrame(all_summaries)
summary_path = TABLE_DIR / "split_summary.csv"
summary.to_csv(summary_path, index=False)
print("done", summary_path)
