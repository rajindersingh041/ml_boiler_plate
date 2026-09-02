"""Task abstraction.

Everything that differs between classification, regression, and
timeseries forecasting lives on a Task subclass: how to split the
data, which models are valid, which metrics to compute, and which
result chart to draw. train.py/predict.py stay task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from ml_boilerplate.config import Config
from ml_boilerplate.features import build_preprocessor


class Task(ABC):
    name: str

    @abstractmethod
    def split(
        self, df: pd.DataFrame, cfg: Config
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return X_train, X_test, y_train, y_test."""

    @abstractmethod
    def model_registry(self) -> dict[str, type[BaseEstimator]]:
        """Name -> estimator class, used to resolve cfg.model.type."""

    @abstractmethod
    def compute_metrics(
        self, y_true, y_pred, y_proba=None
    ) -> dict[str, Any]:
        """Task-appropriate metrics dict."""

    @abstractmethod
    def plot_results(
        self,
        y_true,
        y_pred,
        y_proba,
        cfg: Config,
        context: dict[str, Any],
    ) -> dict[str, str]:
        """Render the task's result chart(s). Returns {name: path}."""

    def prepare_predict_features(
        self, df: pd.DataFrame, cfg: Config
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Turn raw input data into the feature matrix the model expects.

        Default: every column is a feature (dropping the target column if
        present). Returns (X, passthrough) where passthrough is an optional
        DataFrame of columns (e.g. a date) to reattach to the output
        alongside predictions. TimeSeriesTask overrides this to rebuild
        lag/rolling features from historical data.
        """
        X = df.drop(columns=[cfg.data.target_column], errors="ignore")
        return X, None

    def build_preprocessor(self, X: pd.DataFrame, cfg: Config) -> ColumnTransformer:
        """Shared default: numeric/categorical impute + encode + scale."""
        return build_preprocessor(X, cfg.features)
