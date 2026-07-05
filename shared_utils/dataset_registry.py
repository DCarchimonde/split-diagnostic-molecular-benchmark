"""Dataset registry for Paper 1: leakage-aware AIDD benchmark.

This version uses classic MoleculeNet / DeepChem public datasets so the full
pipeline can run on a normal laptop before expanding to larger AIDD tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    smiles_col: str
    target_col: str
    task_type: TaskType
    positive_label: int | None = None
    citation_note: str = "MoleculeNet / DeepChem public dataset"


DATASETS: dict[str, DatasetSpec] = {
    "BBBP": DatasetSpec(
        name="BBBP",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        smiles_col="smiles",
        target_col="p_np",
        task_type="classification",
        positive_label=1,
    ),
    "BACE": DatasetSpec(
        name="BACE",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        smiles_col="mol",
        target_col="Class",
        task_type="classification",
        positive_label=1,
    ),
    "ClinTox": DatasetSpec(
        name="ClinTox",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        smiles_col="smiles",
        target_col="CT_TOX",
        task_type="classification",
        positive_label=1,
    ),
    "HIV": DatasetSpec(
        name="HIV",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        smiles_col="smiles",
        target_col="HIV_active",
        task_type="classification",
        positive_label=1,
    ),
    "ESOL": DatasetSpec(
        name="ESOL",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        smiles_col="smiles",
        target_col="measured log solubility in mols per litre",
        task_type="regression",
    ),
    "FreeSolv": DatasetSpec(
        name="FreeSolv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        smiles_col="smiles",
        target_col="expt",
        task_type="regression",
    ),
}


def get_dataset_names() -> list[str]:
    return list(DATASETS.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        valid = ", ".join(get_dataset_names())
        raise KeyError(f"Unknown dataset {name!r}. Valid choices: {valid}") from exc
