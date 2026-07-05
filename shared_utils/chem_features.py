from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> tuple[np.ndarray, list[int]]:
    features = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        fp = smiles_to_ecfp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        features.append(fp)
        valid_indices.append(idx)
    if not features:
        raise ValueError("No molecules were featurized.")
    return np.vstack(features), valid_indices
