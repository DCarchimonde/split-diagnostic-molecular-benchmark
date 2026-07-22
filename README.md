# Split-Diagnostic Molecular Benchmark

This repository contains the reproducibility package for the manuscript:

**Chemometric Diagnosis of Data-Splitting Effects in Molecular Property Prediction Benchmarks**

## Study purpose

This project audits how train/test partitioning protocols affect molecular property prediction benchmarks. The study compares random, ordinary scaffold, and target-balanced scaffold partitions across six public classification and regression datasets, and evaluates how partition choice changes target-distribution shift, scaffold composition, test-to-train chemical similarity, generalization gaps, and model rankings.

The target-balanced scaffold procedure is used as a diagnostic counterfactual rather than a universal benchmark replacement. It preserves scaffold disjointness while minimizing normalized test-size deviation and target-mean deviation. Its behaviour is evaluated across 20 fixed seeds and compared with 5,000 target-blind random scaffold assignments per dataset.

The goal is not to introduce a new molecular predictor or claim state-of-the-art predictive performance. The goal is to make chemometric model validation and benchmark interpretation more transparent.

## Repository contents

```text
shared_utils/                         Shared data, chemistry, metric, and split utilities
paper1_leakage_benchmark/scripts/     Reproducibility, robustness, and audit scripts
paper1_leakage_benchmark/results/     Machine-readable result tables and generated figures
paper1_leakage_benchmark/figures/     Original manuscript figures
paper1_latex/                         LaTeX manuscript source
```

Raw downloaded datasets and trained model artifacts are intentionally not included.

## Main reproducibility workflow

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

## Twenty-seed robustness and null audits

```bash
python paper1_leakage_benchmark/scripts/balanced_split_null_audit.py --n-seeds 20 --null-draws 5000
python paper1_leakage_benchmark/scripts/robustness_20seeds.py --n-seeds 20 --bootstrap 5000 --resume
python paper1_leakage_benchmark/scripts/plot_robustness20_figure.py
```

These scripts generate:

- the 20-seed raw predictive results;
- seed-paired generalization-gap effects;
- bootstrap 95% confidence intervals;
- Wilcoxon signed-rank tests with Holm correction;
- target-gap robustness summaries;
- 5,000-draw random-scaffold null distributions;
- target-balanced improvement percentiles;
- the paired robustness figure used in the manuscript.

## Environment

A Python environment with RDKit, scikit-learn, XGBoost, pandas, numpy, matplotlib, and scipy is required. See `requirements.txt` for the core package list.

## Interpretation boundary

The 20 fixed seeds quantify robustness within the six included benchmark datasets. They are not treated as independent chemical populations. Target-balanced scaffold partitioning controls test size and the first moment of the target distribution, but it does not hold chemical composition, higher target moments, or every other source of partition difficulty constant.

## Citation

A formal citation will be added after journal submission or archival release.
