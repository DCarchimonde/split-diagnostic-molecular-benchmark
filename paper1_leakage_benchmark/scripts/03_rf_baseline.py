from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for name, spec in DATASETS.items():
    print("\n==========", name, "==========")
    data = np.load(PROCESSED_DIR / f"{name.lower()}_features.npz", allow_pickle=True)
    split_df = pd.read_csv(PROCESSED_DIR / f"{name.lower()}_splits.csv")
    X = data["X"]
    y = data["y"]

    for split_col in ["split_random", "split_scaffold"]:
        train_mask = split_df[split_col].values == "train"
        test_mask = split_df[split_col].values == "test"
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        split_name = split_col.replace("split_", "")

        if spec.task_type == "classification":
            model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            row = {
                "dataset": name,
                "task_type": spec.task_type,
                "split": split_name,
                "model": "RandomForest",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_score)),
            }
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            row = {
                "dataset": name,
                "task_type": spec.task_type,
                "split": split_name,
                "model": "RandomForest",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

        rows.append(row)
        print(row)

metrics = pd.DataFrame(rows)
metrics_path = TABLE_DIR / "paper1_rf_baseline_metrics.csv"
metrics.to_csv(metrics_path, index=False)
print("saved", metrics_path)
