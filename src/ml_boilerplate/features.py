"""Feature preprocessing pipeline.

Builds a sklearn ColumnTransformer so numeric/categorical handling is
declared once and reused identically at train and predict time (no
train/serve skew). This is shared across all tasks (classification,
regression, timeseries) via Task.build_preprocessor.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_boilerplate.config import FeatureConfig


def _infer_columns(X: pd.DataFrame, cfg: FeatureConfig) -> tuple[list[str], list[str]]:
    numeric = cfg.numeric_columns or list(X.select_dtypes(include="number").columns)
    categorical = cfg.categorical_columns or list(
        X.select_dtypes(exclude="number").columns
    )
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, cfg: FeatureConfig) -> ColumnTransformer:
    """Build a ColumnTransformer fitted later inside the model pipeline."""
    numeric_cols, categorical_cols = _infer_columns(X, cfg)

    numeric_steps = [("impute", SimpleImputer(strategy=cfg.missing_strategy_numeric))]
    if cfg.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy=cfg.missing_strategy_categorical)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
