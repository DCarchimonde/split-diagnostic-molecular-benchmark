"""Chemical feature utilities based on RDKit."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None if RDKit cannot parse the molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol. Return None for invalid molecules."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def smiles_to_ecfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray | None:
    """Convert a SMILES string to an ECFP/Morgan bit vector.

    Uses RDKit's newer MorganGenerator API to avoid the deprecated
    GetMorganFingerprintAsBitVect warning.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    fp = generator.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[int]]:
    """Build an ECFP matrix and return valid row indices."""
    features: list[np.ndarray] = []
    valid_indices: list[int] = []

    for idx, smiles in enumerate(smiles_list):
        fp = smiles_to_ecfp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        features.append(fp)
        valid_indices.append(idx)

    if not features:
        raise ValueError("No valid molecules were featurized.")

    return np.vstack(features), valid_indices
