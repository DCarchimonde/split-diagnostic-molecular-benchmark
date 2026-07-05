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
from shared_utils.splitting import add_random_split, generate_scaffold

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 2024, 2026, 3407, 123]


def make_models(task_type: str, seed: int):
    if task_type == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=10000, class_weight="balanced", random_state=seed, solver="liblinear")),
            ]),
        }
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=seed, n_jobs=-1),
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


def add_balanced_scaffold_split(df: pd.DataFrame, task_type: str, seed: int, test_size: float = 0.2) -> pd.DataFrame:
    out = df.copy()
    if "scaffold" not in out.columns:
        out["scaffold"] = out["canonical_smiles"].map(generate_scaffold)

    target_n = int(round(len(out) * test_size))
    global_mean = float(out["target"].mean())
    target_std = float(out["target"].std(ddof=0))
    if target_std == 0 or np.isnan(target_std):
        target_std = 1.0

    groups = []
    for scaffold, sub in out.groupby("scaffold"):
        groups.append({
            "scaffold": scaffold,
            "indices": sub.index.to_numpy(),
            "n": int(len(sub)),
            "target_sum": float(sub["target"].sum()),
            "target_mean": float(sub["target"].mean()),
        })

    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")

    for trial in range(300):
        remaining = list(range(len(groups)))
        selected = []
        selected_n = 0
        selected_sum = 0.0
        local_rng = np.random.default_rng(seed * 1000 + trial)

        while selected_n < max(1, int(target_n * 0.95)) and remaining:
            sample_size = min(len(remaining), 80)
            candidates = local_rng.choice(remaining, size=sample_size, replace=False)
            best_candidate = None
            best_candidate_score = float("inf")

            for cand in candidates:
                g = groups[int(cand)]
                new_n = selected_n + g["n"]
                if new_n > max(target_n * 1.25, target_n + 1) and selected_n > 0:
                    continue
                new_sum = selected_sum + g["target_sum"]
                new_mean = new_sum / max(new_n, 1)
                size_score = abs(new_n - target_n) / max(target_n, 1)
                mean_score = abs(new_mean - global_mean) / target_std
                score = size_score + mean_score
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = int(cand)

            if best_candidate is None:
                break

            g = groups[best_candidate]
            selected.append(best_candidate)
            selected_n += g["n"]
            selected_sum += g["target_sum"]
            remaining.remove(best_candidate)

        if selected_n == 0:
            continue

        selected_mean = selected_sum / selected_n
        size_score = abs(selected_n - target_n) / max(target_n, 1)
        mean_score = abs(selected_mean - global_mean) / target_std
        score = size_score + mean_score

        if score < best_score:
            best_score = score
            best = selected

    if best is None:
        raise RuntimeError("Could not create balanced scaffold split")

    test_indices = np.concatenate([groups[i]["indices"] for i in best])
    out["split_balanced_scaffold"] = "train"
    out.loc[test_indices, "split_balanced_scaffold"] = "test"
    return out


def summarize_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["accuracy", "f1", "roc_auc", "average_precision", "rmse", "mae", "r2"] if c in raw.columns]
    rows = []
    keys = ["dataset", "task_type", "split", "model"]
    for key_values, group in raw.groupby(keys):
        row = dict(zip(keys, key_values))
        for metric in metrics:
            vals = group[metric].dropna()
            if len(vals) == 0:
                continue
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gap(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key_values, group in raw.groupby(["dataset", "task_type", "seed", "model"]):
        dataset, task_type, seed, model = key_values
        random_row = group[group["split"] == "random"]
        balanced_row = group[group["split"] == "balanced_scaffold"]
        if len(random_row) == 0 or len(balanced_row) == 0:
            continue
        if task_type == "classification":
            metric = "roc_auc"
            random_value = float(random_row.iloc[0][metric])
            balanced_value = float(balanced_row.iloc[0][metric])
            gap = random_value - balanced_value
        else:
            metric = "rmse"
            random_value = float(random_row.iloc[0][metric])
            balanced_value = float(balanced_row.iloc[0][metric])
            gap = balanced_value - random_value
        rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "seed": seed,
            "model": model,
            "primary_metric": metric,
            "random_value": random_value,
            "balanced_scaffold_value": balanced_value,
            "generalization_gap": gap,
        })

    gap_raw = pd.DataFrame(rows)
    summary_rows = []
    for key_values, group in gap_raw.groupby(["dataset", "task_type", "model", "primary_metric"]):
        row = dict(zip(["dataset", "task_type", "model", "primary_metric"], key_values))
        vals = group["generalization_gap"].dropna()
        row["gap_mean"] = float(vals.mean())
        row["gap_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)
    return gap_raw, pd.DataFrame(summary_rows)


all_rows = []
diagnostic_rows = []

for data_name, spec in DATASETS.items():
    print("\n==========", data_name, "==========")
    data = np.load(PROCESSED_DIR / f"{data_name.lower()}_features.npz", allow_pickle=True)
    base_df = pd.read_csv(PROCESSED_DIR / f"{data_name.lower()}_splits.csv")
    x = data["X"]
    y = data["y"]

    for seed in SEEDS:
        random_df = add_random_split(base_df, target_col="target", task_type=spec.task_type, random_state=seed)
        balanced_df = add_balanced_scaffold_split(base_df, task_type=spec.task_type, seed=seed)
        split_frames = {"random": random_df, "balanced_scaffold": balanced_df}

        for split_name, split_df in split_frames.items():
            split_col = "split_random" if split_name == "random" else "split_balanced_scaffold"
            train_mask = split_df[split_col].values == "train"
            test_mask = split_df[split_col].values == "test"
            x_train = x[train_mask]
            x_test = x[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            diagnostic_rows.append({
                "dataset": data_name,
                "task_type": spec.task_type,
                "seed": seed,
                "split": split_name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "train_target_mean": float(np.mean(y_train)),
                "test_target_mean": float(np.mean(y_test)),
                "target_mean_gap_test_minus_train": float(np.mean(y_test) - np.mean(y_train)),
            })

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
diag = pd.DataFrame(diagnostic_rows)
summary = summarize_metrics(raw)
gap_raw, gap_summary = summarize_gap(raw)

raw.to_csv(TABLE_DIR / "paper1_balanced_raw.csv", index=False)
summary.to_csv(TABLE_DIR / "paper1_balanced_summary.csv", index=False)
gap_raw.to_csv(TABLE_DIR / "paper1_balanced_gap_raw.csv", index=False)
gap_summary.to_csv(TABLE_DIR / "paper1_balanced_gap_summary.csv", index=False)
diag.to_csv(TABLE_DIR / "paper1_balanced_split_diagnostics.csv", index=False)

print("saved balanced scaffold experiment outputs")
