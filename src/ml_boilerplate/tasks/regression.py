"""Tabular regression task."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ml_boilerplate.config import Config
from ml_boilerplate.metrics import regression_metrics
from ml_boilerplate.model import REGRESSOR_REGISTRY
from ml_boilerplate.plotting import plot_predicted_vs_actual, plot_residuals
from ml_boilerplate.tasks.base import Task


class RegressionTask(Task):
    name = "regression"

    def split(self, df: pd.DataFrame, cfg: Config):
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        return train_test_split(
            X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
        )

    def model_registry(self) -> dict[str, type[BaseEstimator]]:
        return REGRESSOR_REGISTRY

    def compute_metrics(self, y_true, y_pred, y_proba=None) -> dict[str, Any]:
        return regression_metrics(y_true, y_pred)

    def plot_results(self, y_true, y_pred, y_proba, cfg: Config, context: dict) -> dict[str, str]:
        plots_dir = cfg.artifacts.plots_dir
        return {
            "predicted_vs_actual": plot_predicted_vs_actual(
                y_true, y_pred, f"{plots_dir}/predicted_vs_actual.html"
            ),
            "residuals": plot_residuals(y_true, y_pred, f"{plots_dir}/residuals.html"),
        }
