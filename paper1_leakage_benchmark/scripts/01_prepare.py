from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.chem_features import canonicalize_smiles
from shared_utils.dataset_registry import DATASETS

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
RAW_DIR = PAPER_DIR / "data" / "raw"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []

for name, spec in DATASETS.items():
    raw_path = RAW_DIR / f"{name.lower()}_raw.csv"

    if raw_path.exists():
        print("loading local raw file", raw_path)
        df = pd.read_csv(raw_path)
    else:
        print("downloading public dataset", name)
        df = pd.read_csv(spec.url)
        df.to_csv(raw_path, index=False)
        print("saved raw file", raw_path)

    print("preparing", name)
    clean = df[[spec.smiles_col, spec.target_col]].copy()
    clean.columns = ["smiles", "target"]
    clean["dataset"] = name
    clean["task_type"] = spec.task_type

    n_raw = len(clean)
    clean = clean.dropna(subset=["smiles", "target"]).copy()
    n_dropna = len(clean)
    clean["canonical_smiles"] = clean["smiles"].map(canonicalize_smiles)
    clean = clean.dropna(subset=["canonical_smiles"]).copy()
    n_valid = len(clean)
    clean = clean.drop_duplicates(subset=["canonical_smiles"], keep="first").copy()
    n_final = len(clean)

    if spec.task_type == "classification":
        clean["target"] = clean["target"].astype(int)
    else:
        clean["target"] = clean["target"].astype(float)

    clean = clean.reset_index(drop=True)
    out_path = PROCESSED_DIR / f"{name.lower()}_clean.csv"
    clean.to_csv(out_path, index=False)
    print("saved", out_path, clean.shape)

    summary_rows.append({
        "dataset": name,
        "task_type": spec.task_type,
        "raw_rows": n_raw,
        "after_dropna": n_dropna,
        "after_valid_smiles": n_valid,
        "after_dedup": n_final,
        "duplicates_removed": n_valid - n_final,
        "target_mean": float(clean["target"].mean()),
        "target_min": float(clean["target"].min()),
        "target_max": float(clean["target"].max()),
    })

summary = pd.DataFrame(summary_rows)
summary_path = TABLE_DIR / "dataset_summary.csv"
summary.to_csv(summary_path, index=False)
print("saved", summary_path)
