# Split-Diagnostic Molecular Benchmark

This repository contains the reproducibility package for the manuscript:

**Split-Diagnostic Evaluation of Molecular Property Prediction Benchmarks: Scaffold Effects, Target Shift, and Chemical Similarity in Structure-Property Modelling**

## Study purpose

This project audits how train/test splitting protocols affect molecular property prediction benchmarks. The study compares random, ordinary scaffold, and target-balanced scaffold splits across public molecular property datasets, and evaluates how split choice changes target distribution, scaffold composition, test-to-train chemical similarity, performance gaps, and model rankings.

The goal is not to propose a new molecular predictor or claim state-of-the-art performance. The goal is to make benchmark interpretation more transparent.

## Repository contents

```text
shared_utils/                         Shared data, chemistry, and split utilities
paper1_leakage_benchmark/scripts/     Reproducibility scripts
paper1_leakage_benchmark/results/     Generated result tables
paper1_leakage_benchmark/figures/     Manuscript figures
paper1_latex/                         LaTeX manuscript source
```

Raw downloaded datasets and trained model artifacts are intentionally not included.

## Reproducibility workflow

Run from the repository root:

```bash
python paper1_leakage_benchmark/scripts/01_prepare.py
python paper1_leakage_benchmark/scripts/02_featurize_and_split.py
python paper1_leakage_benchmark/scripts/train_many.py
python paper1_leakage_benchmark/scripts/train_balanced_scaffold.py
python paper1_leakage_benchmark/scripts/compare_splits.py
python paper1_leakage_benchmark/scripts/scaffold_stats.py
python paper1_leakage_benchmark/scripts/balanced_scaffold_stats.py
python paper1_leakage_benchmark/scripts/similarity_audit.py
python paper1_leakage_benchmark/scripts/outlier_inspection.py
python paper1_leakage_benchmark/scripts/make_paper_figures.py
python paper1_leakage_benchmark/scripts/make_manuscript_tables.py
```

## Main outputs

The workflow generates:

- dataset summary tables;
- split diagnostics;
- generalization-gap summaries;
- Tanimoto similarity audits;
- model-ranking and sensitivity analyses;
- manuscript figures.

## Environment

A Python environment with RDKit, scikit-learn, XGBoost, pandas, numpy, matplotlib, and scipy is required. See `requirements.txt` for the core package list.

## Citation

A formal citation will be added after journal submission or archival release.
