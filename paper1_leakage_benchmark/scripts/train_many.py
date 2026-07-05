from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.metrics import regression_metrics, safe_classification_metrics
from shared_utils.splitting import add_random_split, add_scaffold_split

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 2024, 2026, 3407, 123]

CLASSIFICATION_MODELS = ["RandomForest", "XGBoost", "LogisticRegression"]
REGRESSION_MODELS = ["RandomForest", "XGBoost", "Ridge"]


def make_models(task_type: str, seed: int):
    if task_type == "classification":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(
                    max_iter=10000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="liblinear",
                )),
            ]),
        }

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0)),
        ]),
    }


def get_score(model, x_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if len(proba.shape) == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def evaluate_one(model, task_type, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if task_type == "classification":
        return safe_classification_metrics(y_test, pred, get_score(model, x_test))
    return regression_metrics(y_test, pred)


def make_metric_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in ["accuracy", "f1", "roc_auc", "average_precision", "rmse", "mae", "r2"]
        if c in raw.columns
    ]
    rows = []
    group_cols = ["dataset", "task_type", "split", "model"]

    for keys, group in raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            values = group[metric].dropna()
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_gap_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_rows = []
    group_cols = ["dataset", "task_type", "seed", "model"]

    for keys, group in raw.groupby(group_cols):
        dataset, task_type, seed, model = keys
        random_row = group[group["split"] == "random"]
        scaffold_row = group[group["split"] == "scaffold"]
        if len(random_row) == 0 or len(scaffold_row) == 0:
            continue

        if task_type == "classification":
            metric = "roc_auc"
            random_value = float(random_row.iloc[0][metric])
            scaffold_value = float(scaffold_row.iloc[0][metric])
            gap = random_value - scaffold_value
        else:
            metric = "rmse"
            random_value = float(random_row.iloc[0][metric])
            scaffold_value = float(scaffold_row.iloc[0][metric])
            gap = scaffold_value - random_value

        gap_rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "seed": seed,
            "model": model,
            "primary_metric": metric,
            "random_value": random_value,
            "scaffold_value": scaffold_value,
            "generalization_gap": gap,
        })

    gap_raw = pd.DataFrame(gap_rows)
    summary_rows = []
    summary_cols = ["dataset", "task_type", "model", "primary_metric"]

    for keys, group in gap_raw.groupby(summary_cols):
        row = dict(zip(summary_cols, keys))
        values = group["generalization_gap"].dropna()
        row["gap_mean"] = float(values.mean())
        row["gap_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    gap_summary = pd.DataFrame(summary_rows)
    return gap_raw, gap_summary


all_rows = []

for data_name, spec in DATASETS.items():
    print("\n==========", data_name, "==========")
    data = np.load(PROCESSED_DIR / f"{data_name.lower()}_features.npz", allow_pickle=True)
    base_df = pd.read_csv(PROCESSED_DIR / f"{data_name.lower()}_splits.csv")
    x = data["X"]
    y = data["y"]

    for seed in SEEDS:
        random_df = add_random_split(
            base_df,
            target_col="target",
            task_type=spec.task_type,
            random_state=seed,
        )
        scaffold_df = add_scaffold_split(base_df, smiles_col="canonical_smiles")
        split_frames = {"random": random_df, "scaffold": scaffold_df}

        for split_name, split_df in split_frames.items():
            split_col = "split_random" if split_name == "random" else "split_scaffold"
            train_mask = split_df[split_col].values == "train"
            test_mask = split_df[split_col].values == "test"
            x_train = x[train_mask]
            x_test = x[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            for model_name, model in make_models(spec.task_type, seed).items():
                print(data_name, seed, split_name, model_name)
                metrics = evaluate_one(model, spec.task_type, x_train, x_test, y_train, y_test)
                row = {
                    "dataset": data_name,
                    "task_type": spec.task_type,
                    "seed": seed,
                    "split": split_name,
                    "model": model_name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }
                row.update(metrics)
                all_rows.append(row)
                print(row)

raw = pd.DataFrame(all_rows)
raw_path = TABLE_DIR / "paper1_many_raw.csv"
raw.to_csv(raw_path, index=False)
print("saved", raw_path)

summary = make_metric_summary(raw)
summary_path = TABLE_DIR / "paper1_many_summary.csv"
summary.to_csv(summary_path, index=False)
print("saved", summary_path)

gap_raw, gap_summary = make_gap_tables(raw)
gap_raw_path = TABLE_DIR / "paper1_gap_raw.csv"
gap_summary_path = TABLE_DIR / "paper1_gap_summary.csv"
gap_raw.to_csv(gap_raw_path, index=False)
gap_summary.to_csv(gap_summary_path, index=False)
print("saved", gap_raw_path)
print("saved", gap_summary_path)
