from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    arr = np.zeros((n_bits,), dtype=np.int8)
    arr[list(fp.GetOnBits())] = 1
    return arr


if __name__ == "__main__":
    test_smiles = [
        "CCO",              # ethanol
        "CC(=O)O",          # acetic acid
        "c1ccccc1",         # benzene
        "CCN(CC)CC",        # triethylamine
    ]

    for smi in test_smiles:
        fp = smiles_to_ecfp(smi)
        print(smi, fp.shape, fp.sum())
