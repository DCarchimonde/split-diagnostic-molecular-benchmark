from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.metrics import regression_metrics, safe_classification_metrics
from shared_utils.splitting import add_random_split, add_scaffold_split, generate_scaffold

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEED_POOL = [
    42, 123, 2024, 2026, 3407,
    7, 19, 71, 101, 211,
    307, 401, 503, 601, 701,
    809, 907, 1009, 1201, 1429,
    1601, 1801, 2003, 2203, 2503,
    2801, 3001, 3203, 3607, 4001,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split-robustness experiments over repeated seeds.")
    parser.add_argument("--n-seeds", type=int, default=20, choices=range(5, len(SEED_POOL) + 1))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing raw output file.")
    return parser.parse_args()


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
        metrics = safe_classification_metrics(y_test, pred, get_score(model, x_test))
        return "roc_auc", float(metrics["roc_auc"])
    metrics = regression_metrics(y_test, pred)
    return "rmse", float(metrics["rmse"])


def add_balanced_scaffold_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    n_trials: int = 300,
) -> tuple[pd.DataFrame, dict]:
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
        })

    best = None
    best_score = float("inf")
    best_meta = None

    for trial in range(n_trials):
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
                group = groups[int(cand)]
                new_n = selected_n + group["n"]
                if new_n > max(target_n * 1.25, target_n + 1) and selected_n > 0:
                    continue
                new_sum = selected_sum + group["target_sum"]
                new_mean = new_sum / max(new_n, 1)
                size_component = abs(new_n - target_n) / max(target_n, 1)
                mean_component = abs(new_mean - global_mean) / target_std
                candidate_score = size_component + mean_component
                if candidate_score < best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate = int(cand)

            if best_candidate is None:
                break

            group = groups[best_candidate]
            selected.append(best_candidate)
            selected_n += group["n"]
            selected_sum += group["target_sum"]
            remaining.remove(best_candidate)

        if selected_n == 0:
            continue

        selected_mean = selected_sum / selected_n
        size_component = abs(selected_n - target_n) / max(target_n, 1)
        mean_component = abs(selected_mean - global_mean) / target_std
        score = size_component + mean_component

        if score < best_score:
            best_score = score
            best = selected
            best_meta = {
                "objective": float(score),
                "size_component": float(size_component),
                "mean_component": float(mean_component),
                "target_test_n": int(target_n),
                "actual_test_n": int(selected_n),
            }

    if best is None or best_meta is None:
        raise RuntimeError("Could not create target-balanced scaffold split")

    test_indices = np.concatenate([groups[i]["indices"] for i in best])
    out["split_balanced_scaffold"] = "train"
    out.loc[test_indices, "split_balanced_scaffold"] = "test"
    return out, best_meta


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int = 20260722) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def summarize_paired(paired: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "task_type", "model", "primary_metric"]
    for keys, group in paired.groupby(group_cols):
        values = group["gap_reduction"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_seeds": int(len(values)),
            "ordinary_gap_mean": float(group["ordinary_gap"].mean()),
            "balanced_gap_mean": float(group["balanced_gap"].mean()),
            "gap_reduction_mean": float(np.mean(values)),
            "gap_reduction_median": float(np.median(values)),
            "gap_reduction_ci95_low": low,
            "gap_reduction_ci95_high": high,
            "wilcoxon_p": safe_wilcoxon(values),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_p_holm"] = holm_adjust(out["wilcoxon_p"])
    return out


def summarize_target_shift(diag: pd.DataFrame, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = diag.pivot_table(
        index=["dataset", "task_type", "seed"],
        columns="split",
        values="abs_target_gap",
        aggfunc="first",
    ).reset_index()
    pivot["target_gap_reduction"] = pivot["ordinary_scaffold"] - pivot["balanced_scaffold"]

    rows = []
    for keys, group in pivot.groupby(["dataset", "task_type"]):
        values = group["target_gap_reduction"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap)
        row = {
            "dataset": keys[0],
            "task_type": keys[1],
            "n_seeds": int(len(group)),
            "ordinary_abs_target_gap_mean": float(group["ordinary_scaffold"].mean()),
            "balanced_abs_target_gap_mean": float(group["balanced_scaffold"].mean()),
            "target_gap_reduction_mean": float(np.mean(values)),
            "target_gap_reduction_ci95_low": low,
            "target_gap_reduction_ci95_high": high,
            "wilcoxon_p": safe_wilcoxon(values),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["wilcoxon_p_holm"] = holm_adjust(summary["wilcoxon_p"])
    return pivot, summary


def main() -> None:
    args = parse_args()
    seeds = SEED_POOL[: args.n_seeds]
    selected_datasets = list(DATASETS.keys()) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]

    raw_path = TABLE_DIR / "paper1_robustness20_raw.csv"
    diag_path = TABLE_DIR / "paper1_robustness20_split_diagnostics.csv"

    if args.resume and raw_path.exists():
        raw = pd.read_csv(raw_path)
    else:
        raw = pd.DataFrame()

    if args.resume and diag_path.exists():
        diagnostics = pd.read_csv(diag_path)
    else:
        diagnostics = pd.DataFrame()

    completed = set()
    if not raw.empty:
        completed = set(zip(raw["dataset"], raw["seed"], raw["split"], raw["model"]))

    raw_rows = raw.to_dict("records") if not raw.empty else []
    diagnostic_rows = diagnostics.to_dict("records") if not diagnostics.empty else []

    for dataset_name in selected_datasets:
        spec = DATASETS[dataset_name]
        print(f"\n========== {dataset_name} ==========")
        data = np.load(PROCESSED_DIR / f"{dataset_name.lower()}_features.npz", allow_pickle=True)
        base_df = pd.read_csv(PROCESSED_DIR / f"{dataset_name.lower()}_splits.csv")
        x = data["X"]
        y = data["y"]

        for seed in seeds:
            random_df = add_random_split(base_df, target_col="target", task_type=spec.task_type, random_state=seed)
            ordinary_df = add_scaffold_split(base_df, smiles_col="canonical_smiles")
            balanced_df, balanced_meta = add_balanced_scaffold_split(base_df, seed=seed)

            split_frames = {
                "random": (random_df, "split_random", {}),
                "ordinary_scaffold": (ordinary_df, "split_scaffold", {}),
                "balanced_scaffold": (balanced_df, "split_balanced_scaffold", balanced_meta),
            }

            for split_name, (split_df, split_col, split_meta) in split_frames.items():
                train_mask = split_df[split_col].to_numpy() == "train"
                test_mask = split_df[split_col].to_numpy() == "test"
                y_train = y[train_mask]
                y_test = y[test_mask]

                diag_key = (dataset_name, seed, split_name)
                existing_diag_keys = {
                    (row["dataset"], int(row["seed"]), row["split"])
                    for row in diagnostic_rows
                }
                if diag_key not in existing_diag_keys:
                    diagnostic_rows.append({
                        "dataset": dataset_name,
                        "task_type": spec.task_type,
                        "seed": seed,
                        "split": split_name,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "train_target_mean": float(np.mean(y_train)),
                        "test_target_mean": float(np.mean(y_test)),
                        "target_gap": float(np.mean(y_test) - np.mean(y_train)),
                        "abs_target_gap": float(abs(np.mean(y_test) - np.mean(y_train))),
                        "objective": split_meta.get("objective", np.nan),
                        "size_component": split_meta.get("size_component", np.nan),
                        "mean_component": split_meta.get("mean_component", np.nan),
                        "target_test_n": split_meta.get("target_test_n", np.nan),
                        "actual_test_n": split_meta.get("actual_test_n", int(len(y_test))),
                    })

                for model_name, model in make_models(spec.task_type, seed).items():
                    key = (dataset_name, seed, split_name, model_name)
                    if key in completed:
                        continue
                    print(dataset_name, seed, split_name, model_name)
                    metric_name, metric_value = evaluate_one(
                        model,
                        spec.task_type,
                        x[train_mask],
                        x[test_mask],
                        y_train,
                        y_test,
                    )
                    raw_rows.append({
                        "dataset": dataset_name,
                        "task_type": spec.task_type,
                        "seed": seed,
                        "split": split_name,
                        "model": model_name,
                        "primary_metric": metric_name,
                        "metric_value": metric_value,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                    })
                    completed.add(key)

                pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
                pd.DataFrame(diagnostic_rows).to_csv(diag_path, index=False)

    raw = pd.DataFrame(raw_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    wide = raw.pivot_table(
        index=["dataset", "task_type", "seed", "model", "primary_metric"],
        columns="split",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    classification = wide["task_type"] == "classification"
    wide["ordinary_gap"] = np.where(
        classification,
        wide["random"] - wide["ordinary_scaffold"],
        wide["ordinary_scaffold"] - wide["random"],
    )
    wide["balanced_gap"] = np.where(
        classification,
        wide["random"] - wide["balanced_scaffold"],
        wide["balanced_scaffold"] - wide["random"],
    )
    wide["gap_reduction"] = wide["ordinary_gap"] - wide["balanced_gap"]

    paired_path = TABLE_DIR / "paper1_robustness20_paired.csv"
    summary_path = TABLE_DIR / "paper1_robustness20_summary.csv"
    target_raw_path = TABLE_DIR / "paper1_robustness20_target_shift_paired.csv"
    target_summary_path = TABLE_DIR / "paper1_robustness20_target_shift_summary.csv"

    wide.to_csv(paired_path, index=False)
    summarize_paired(wide, n_bootstrap=args.bootstrap).to_csv(summary_path, index=False)
    target_raw, target_summary = summarize_target_shift(diagnostics, n_bootstrap=args.bootstrap)
    target_raw.to_csv(target_raw_path, index=False)
    target_summary.to_csv(target_summary_path, index=False)

    print("saved", raw_path)
    print("saved", diag_path)
    print("saved", paired_path)
    print("saved", summary_path)
    print("saved", target_raw_path)
    print("saved", target_summary_path)


if __name__ == "__main__":
    main()
