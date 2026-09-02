"""Model registries.

One registry per task family so a config's `model.type` always resolves
against the right set of estimators. Deliberately scikit-learn only
(no xgboost/lightgbm) to keep this dependency-light; swap in your own
registry entries if you need them.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from ml_boilerplate.config import ModelConfig

CLASSIFIER_REGISTRY: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "bagging": BaggingClassifier,
}

REGRESSOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "bagging": BaggingRegressor,
}


def build_model(cfg: ModelConfig, registry: dict[str, type[BaseEstimator]]) -> BaseEstimator:
    """Instantiate an estimator by name from the given registry."""
    try:
        estimator_cls = registry[cfg.type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model type {cfg.type!r}. Available: {list(registry)}"
        ) from exc
    return estimator_cls(**cfg.params)
