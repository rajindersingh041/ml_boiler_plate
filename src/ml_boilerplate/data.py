"""Data loading.

Dispatches to a synthetic generator (one per task, for quick starts and
tests) or to `io.read_table` for real data (local path or http(s) URL,
csv or parquet). Train/test splitting is task-specific and lives on
each Task class in tasks/, since classification/regression use a random
split while timeseries needs a chronological one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from ml_boilerplate.config import DataConfig
from ml_boilerplate.io import read_table


def _synthetic_classification(cfg: DataConfig) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=cfg.random_state,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df[cfg.target_column] = y
    return df


def _synthetic_regression(cfg: DataConfig) -> pd.DataFrame:
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        noise=10.0,
        random_state=cfg.random_state,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df[cfg.target_column] = y
    return df


def _synthetic_timeseries(cfg: DataConfig) -> pd.DataFrame:
    """A single daily series with trend + weekly seasonality + noise."""
    rng = np.random.default_rng(cfg.random_state)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    trend = 0.05 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(scale=2.0, size=n)
    target = 50 + trend + seasonality + noise

    date_col = cfg.date_column or "date"
    return pd.DataFrame({date_col: dates, cfg.target_column: target})


_SYNTHETIC_GENERATORS = {
    "classification": _synthetic_classification,
    "regression": _synthetic_regression,
    "timeseries": _synthetic_timeseries,
}


def load_data(cfg: DataConfig, task: str) -> pd.DataFrame:
    """Load a dataset as a single DataFrame containing features + target."""
    if cfg.source == "file":
        if not cfg.path:
            raise ValueError("data.path must be set when data.source == 'file'")
        return read_table(cfg.path, cfg.format)

    if cfg.source == "synthetic":
        try:
            generator = _SYNTHETIC_GENERATORS[task]
        except KeyError as exc:
            raise ValueError(f"No synthetic generator for task {task!r}") from exc
        return generator(cfg)

    raise ValueError(f"Unknown data source: {cfg.source!r}")
