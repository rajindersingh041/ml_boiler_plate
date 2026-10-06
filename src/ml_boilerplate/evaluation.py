"""Final held-out test gate: metrics + pass/fail vs thresholds."""
from __future__ import annotations

from ml_boilerplate.metrics import classification_metrics, regression_metrics


def evaluate_on_test(y_true, y_pred, y_proba, task_name: str,
                     min_metrics: dict | None = None) -> dict:
    if task_name == "classification":
        metrics = classification_metrics(y_true, y_pred, y_proba)
    else:
        metrics = regression_metrics(y_true, y_pred)
    min_metrics = min_metrics or {}
    failed = [k for k, v in min_metrics.items() if metrics.get(k, float("-inf")) < v]
    return {"metrics": metrics, "passed": not failed, "failed": failed}
