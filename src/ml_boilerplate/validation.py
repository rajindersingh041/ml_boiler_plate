"""Train/validation/test splitting: stratified for classification,
chronological (row order) for timeseries, random otherwise."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_boilerplate.config import Config


def split_train_val_test(X, y, cfg: Config, task_name: str):
    rs = cfg.data.random_state
    if task_name == "timeseries":
        n = len(X)
        n_test = int(n * cfg.data.test_size)
        n_val = int(n * cfg.validation.val_size)
        n_train = n - n_val - n_test
        return (
            X.iloc[:n_train], X.iloc[n_train:n_train + n_val], X.iloc[n_train + n_val:],
            y.iloc[:n_train], y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:],
        )
    stratify_test = y if task_name == "classification" else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=rs, stratify=stratify_test)
    val_frac = cfg.validation.val_size / (1 - cfg.data.test_size)
    stratify_val = y_temp if task_name == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=rs, stratify=stratify_val)
    return X_train, X_val, X_test, y_train, y_val, y_test
