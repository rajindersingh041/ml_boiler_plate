"""Serving: FastAPI over the promoted production model.

Training code never runs here — the app loads one joblib pipeline and
uses Task.prepare_predict_features for task-aware input handling.
"""
from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from ml_boilerplate.registry import load_production
from ml_boilerplate.tasks import get_task


def create_app(production_dir: str, task_name: str, target_column: str) -> FastAPI:
    model = load_production(production_dir)
    task = get_task(task_name)
    try:
        expected = set(model.feature_names_in_)
    except AttributeError:
        expected = set()
    app = FastAPI(title="ml-boilerplate")

    @app.get("/health")
    def health():
        return {"status": "ok", "task": task_name, "model": "production"}

    @app.post("/predict")
    def predict(payload: dict):
        rows = payload.get("rows")
        if not rows:
            raise HTTPException(status_code=422, detail="'rows' must be a non-empty list")
        df = pd.DataFrame(rows)
        if expected and not expected.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {sorted(expected - set(df.columns))}")
        X, _ = task.prepare_predict_features(
            df if target_column not in df.columns
            else df.drop(columns=[target_column]), _cfg(target_column))
        try:
            preds = model.predict(X)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"predictions": [float(p) if p is not None else None for p in preds]}

    return app


class _cfg:  # minimal stand-in exposing only what prepare_predict_features reads
    def __init__(self, target_column: str):
        from ml_boilerplate.config import DataConfig
        self.data = DataConfig(target_column=target_column)


PRODUCTION_DIR = os.environ.get("PRODUCTION_DIR", "artifacts/production")
TASK = os.environ.get("TASK", "classification")
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "target")

try:
    app = create_app(PRODUCTION_DIR, TASK, TARGET_COLUMN)
except FileNotFoundError as exc:
    app = None
    _STARTUP_ERROR = str(exc)
