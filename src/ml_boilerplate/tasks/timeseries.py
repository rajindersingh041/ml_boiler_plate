"""Single-series forecasting task.

Frames forecasting as supervised regression: lag + rolling-window +
date-part features predict the target, with a chronological (not
random) train/test split so the model is only ever evaluated on data
that comes after what it trained on.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from ml_boilerplate.config import Config, DataConfig
from ml_boilerplate.metrics import regression_metrics
from ml_boilerplate.model import REGRESSOR_REGISTRY
from ml_boilerplate.plotting import plot_forecast
from ml_boilerplate.tasks.base import Task


def build_lag_features(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Add lag, rolling-window, and date-part features; drop warm-up rows.

    Rolling/lag features use `.shift(1)` before rolling so no feature
    ever leaks the current row's target into itself.
    """
    date_col = cfg.date_column or "date"
    target = cfg.target_column

    out = df.sort_values(date_col).reset_index(drop=True).copy()
    out[date_col] = pd.to_datetime(out[date_col])

    for lag in cfg.n_lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for window in cfg.rolling_windows:
        shifted = out[target].shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()

    out["dayofweek"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day

    return out.dropna().reset_index(drop=True)


class TimeSeriesTask(Task):
    name = "timeseries"

    def __init__(self) -> None:
        self._dates: pd.Series | None = None
        self._split_index: int | None = None

    def split(self, df: pd.DataFrame, cfg: Config):
        date_col = cfg.data.date_column or "date"
        featured = build_lag_features(df, cfg.data)

        n_test = max(1, int(len(featured) * cfg.data.test_size))
        split_index = len(featured) - n_test
        if split_index <= 0:
            raise ValueError(
                "Not enough rows left after building lag/rolling features to "
                "leave any training data — reduce n_lags/rolling_windows or "
                "test_size, or provide more history."
            )

        feature_cols = [
            c for c in featured.columns if c not in (cfg.data.target_column, date_col)
        ]
        X = featured[feature_cols]
        y = featured[cfg.data.target_column]

        self._dates = featured[date_col]
        self._split_index = split_index

        return (
            X.iloc[:split_index],
            X.iloc[split_index:],
            y.iloc[:split_index],
            y.iloc[split_index:],
        )

    def model_registry(self) -> dict[str, type[BaseEstimator]]:
        return REGRESSOR_REGISTRY

    def prepare_predict_features(self, df: pd.DataFrame, cfg: Config):
        """Rebuild lag/rolling/date-part features from historical data.

        Expects `df` to look like the raw training data (date + target
        column history), not a single future row — lag/rolling features
        need that history to be computable. Returns predictions aligned
        to every row that has enough history (i.e. same warm-up rule as
        training).
        """
        date_col = cfg.data.date_column or "date"
        featured = build_lag_features(df, cfg.data)
        feature_cols = [
            c for c in featured.columns if c not in (cfg.data.target_column, date_col)
        ]
        return featured[feature_cols], featured[[date_col]]

    def compute_metrics(self, y_true, y_pred, y_proba=None) -> dict[str, Any]:
        return regression_metrics(y_true, y_pred)

    def plot_results(self, y_true, y_pred, y_proba, cfg: Config, context: dict) -> dict[str, str]:
        if self._dates is None or self._split_index is None:
            raise RuntimeError("plot_results called before split()")

        y_full = context["y_full"]
        y_pred_full = [float("nan")] * self._split_index + list(y_pred)

        plots_dir = cfg.artifacts.plots_dir
        return {
            "forecast": plot_forecast(
                self._dates.tolist(), list(y_full), y_pred_full, self._split_index,
                f"{plots_dir}/forecast.html",
            )
        }
