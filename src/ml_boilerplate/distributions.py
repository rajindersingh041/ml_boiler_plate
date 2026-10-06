"""Pure distribution profiles: data in, JSON-serializable dicts out.

Chart rendering (plotly HTML) is layered on by callers; these functions
never touch disk so they stay trivially testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def data_distributions(df: pd.DataFrame, target: str) -> dict:
    return {
        "histograms": {c: df[c].dropna().tolist()[:1000]
                       for c in df.select_dtypes(include="number").columns},
        "target": df[target].value_counts(dropna=False).to_dict()
        if target in df.columns else {},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "cardinality": {c: int(df[c].nunique(dropna=True))
                        for c in df.select_dtypes(exclude="number").columns},
    }


def error_distributions(y_true, y_pred, y_proba, task_name: str) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task_name == "classification":
        from sklearn.metrics import confusion_matrix
        out: dict = {"confusion": confusion_matrix(y_true, y_pred).tolist()}
        if y_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba)))
            except ValueError:
                pass
        return out
    residuals = (y_true - y_pred).tolist()
    return {"residuals": residuals,
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}
