"""Binary classification task."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ml_boilerplate.config import Config
from ml_boilerplate.metrics import classification_metrics
from ml_boilerplate.model import CLASSIFIER_REGISTRY
from ml_boilerplate.plotting import plot_confusion_matrix, plot_roc_curve
from ml_boilerplate.tasks.base import Task


class ClassificationTask(Task):
    name = "classification"

    def split(self, df: pd.DataFrame, cfg: Config):
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        return train_test_split(
            X, y,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state,
            stratify=y,
        )

    def model_registry(self) -> dict[str, type[BaseEstimator]]:
        return CLASSIFIER_REGISTRY

    def compute_metrics(self, y_true, y_pred, y_proba=None) -> dict[str, Any]:
        return classification_metrics(y_true, y_pred, y_proba)

    def plot_results(self, y_true, y_pred, y_proba, cfg: Config, context: dict) -> dict[str, str]:
        plots_dir = cfg.artifacts.plots_dir
        paths = {
            "confusion_matrix": plot_confusion_matrix(
                y_true, y_pred, f"{plots_dir}/confusion_matrix.html"
            )
        }
        if y_proba is not None:
            paths["roc_curve"] = plot_roc_curve(y_true, y_proba, f"{plots_dir}/roc_curve.html")
        return paths
