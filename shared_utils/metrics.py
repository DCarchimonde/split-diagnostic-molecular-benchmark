"""Metric helpers for Paper 1 baseline experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


def safe_classification_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {}
    result["accuracy"] = float(accuracy_score(y_true, y_pred))
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if y_score is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = float("nan")
        try:
            result["average_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            result["average_precision"] = float("nan")

    return result


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
