"""Split utilities for molecular machine learning."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def add_random_split(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    out = df.copy()
    idx = np.arange(len(out))

    stratify = None
    if task_type == "classification":
        counts = out[target_col].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = out[target_col]

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    out["split_random"] = "unused"
    out.loc[train_idx, "split_random"] = "train"
    out.loc[test_idx, "split_random"] = "test"
    return out


def add_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    test_size: float = 0.2,
) -> pd.DataFrame:
    out = df.copy()
    out["scaffold"] = out[smiles_col].map(generate_scaffold)

    scaffold_groups = defaultdict(list)
    for i, scaffold in enumerate(out["scaffold"].tolist()):
        scaffold_groups[scaffold].append(i)

    groups = sorted(scaffold_groups.values(), key=len, reverse=True)
    target_test_n = int(round(len(out) * test_size))

    test_idx = []
    train_idx = []
    for group in groups:
        if len(test_idx) < target_test_n:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    out["split_scaffold"] = "unused"
    out.loc[train_idx, "split_scaffold"] = "train"
    out.loc[test_idx, "split_scaffold"] = "test"
    return out


def summarize_split(df: pd.DataFrame, split_col: str) -> dict:
    return {
        "split_col": split_col,
        "n_total": int(len(df)),
        "n_train": int((df[split_col] == "train").sum()),
        "n_test": int((df[split_col] == "test").sum()),
        "train_target_mean": float(df.loc[df[split_col] == "train", "target"].mean()),
        "test_target_mean": float(df.loc[df[split_col] == "test", "target"].mean()),
    }
