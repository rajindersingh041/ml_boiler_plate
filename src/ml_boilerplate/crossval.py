"""K-fold CV over the full preprocess+model pipeline.

Uses StratifiedKFold for classification, TimeSeriesSplit for timeseries,
KFold otherwise. Metrics per fold come from task.compute_metrics so they
match train.py exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ml_boilerplate.config import Config
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task


def _splitter(task_name: str, cfg: Config):
    k = cfg.cross_validation.folds
    if task_name == "classification":
        return StratifiedKFold(n_splits=k, shuffle=cfg.cross_validation.shuffle,
                               random_state=cfg.data.random_state)
    if task_name == "timeseries":
        return TimeSeriesSplit(n_splits=k)
    shuffle = cfg.cross_validation.shuffle
    return KFold(n_splits=k, shuffle=shuffle,
                 random_state=cfg.data.random_state if shuffle else None)


def run_cv(X: pd.DataFrame, y: pd.Series, cfg: Config, task_name: str) -> dict:
    task = get_task(task_name)
    splitter = _splitter(task_name, cfg)
    split_args = (X, y) if task_name == "classification" else (X,)
    per_fold = []
    for train_idx, val_idx in splitter.split(*split_args):
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
        pipe = Pipeline([("preprocess", task.build_preprocessor(Xtr, cfg)),
                         ("model", build_model(cfg.model, task.model_registry()))])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xv)
        proba = None
        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(Xv)
            if p.ndim == 2 and p.shape[1] == 2:
                proba = p[:, 1]
        per_fold.append(task.compute_metrics(yv.to_numpy(), pred, proba))
    aggregate = {}
    for key in per_fold[0]:
        vals = np.array([f[key] for f in per_fold], dtype=float)
        aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {"per_fold": per_fold, "aggregate": aggregate}
