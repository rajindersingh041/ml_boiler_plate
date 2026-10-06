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

from ml_boilerplate.config import Config, DataConfig, FeatureConfig
from ml_boilerplate.engineering import add_lag_features
from ml_boilerplate.metrics import regression_metrics
from ml_boilerplate.model import REGRESSOR_REGISTRY
from ml_boilerplate.plotting import plot_forecast
from ml_boilerplate.tasks.base import Task


def build_lag_features(
    df: pd.DataFrame, cfg: DataConfig, feat: FeatureConfig | None = None
) -> pd.DataFrame:
    """Add lag, rolling-window, and date-part features; drop warm-up rows.

    Thin wrapper over engineering.add_lag_features. With default feature
    flags the output columns are exactly the legacy set (lag_N,
    rolling_mean/std_N, dayofweek/month/day plus the extended date parts).
    Rolling/lag features use `.shift(1)` before rolling so no feature
    ever leaks the current row's target into itself.
    """
    feat = feat or FeatureConfig()
    return add_lag_features(
        df,
        target=cfg.target_column,
        date_col=cfg.date_column or "date",
        n_lags=cfg.n_lags,
        rolling_windows=cfg.rolling_windows,
        rolling_stats=tuple(feat.rolling_stats),
        lag_diffs=feat.lag_diffs,
        date_parts=True,
        cyclical=feat.cyclical_encoding,
        holiday_country=feat.holiday_country,
        holiday_subdiv=feat.holiday_subdiv,
    )


class TimeSeriesTask(Task):
    name = "timeseries"

    def __init__(self) -> None:
        self._dates: pd.Series | None = None
        self._split_index: int | None = None

    def split(self, df: pd.DataFrame, cfg: Config):
        date_col = cfg.data.date_column or "date"
        featured = build_lag_features(df, cfg.data, cfg.features)

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
        featured = build_lag_features(df, cfg.data, cfg.features)
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
