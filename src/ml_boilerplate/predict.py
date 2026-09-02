"""Inference: load a persisted pipeline and score new data.

Task-aware only for feature preparation (Task.prepare_predict_features) —
classification/regression treat every input column as a feature, while
timeseries rebuilds lag/rolling features from historical data. Input and
output both go through the generic io.read_table/write_table, so csv,
parquet, and http(s) URLs all work the same way as in training.
"""

from __future__ import annotations

import joblib
import pandas as pd

from ml_boilerplate.config import Config
from ml_boilerplate.io import read_table, write_table
from ml_boilerplate.tasks import get_task


def load_model(model_path: str):
    return joblib.load(model_path)


def run_prediction(cfg: Config, input_path: str, output_path: str) -> str:
    """Score `input_path` with the trained model and write predictions."""
    task = get_task(cfg.task)
    model = load_model(cfg.artifacts.model_path)

    df = read_table(input_path)
    X, passthrough = task.prepare_predict_features(df, cfg)

    out = passthrough.reset_index(drop=True) if passthrough is not None else pd.DataFrame()
    out["prediction"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            out["probability"] = proba[:, 1]

    write_table(out, output_path, cfg.output.format)
    return output_path
