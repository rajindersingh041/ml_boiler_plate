"""Training entrypoint: wires data -> [eda] -> task.split -> preprocess ->
model into one fitted sklearn Pipeline, evaluates it with task-appropriate
metrics, saves the pipeline + metrics, and renders the task's result chart.

Task-agnostic by design: swap cfg.task between "classification",
"regression", and "timeseries" and everything below adapts via the Task
abstraction in tasks/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_boilerplate.config import Config
from ml_boilerplate.data import load_data
from ml_boilerplate.eda import run_eda
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task

logger = logging.getLogger(__name__)


def run_training(cfg: Config) -> dict:
    """Run the full train pipeline and return the computed metrics."""
    task = get_task(cfg.task)
    logger.info("Task: %s", task.name)

    logger.info("Loading data (source=%s)", cfg.data.source)
    df = load_data(cfg.data, cfg.task)

    if cfg.eda.enabled:
        logger.info("Running EDA")
        run_eda(df, cfg.eda, report_path=f"{cfg.eda.output_dir}/eda_report.json")

    X_train, X_test, y_train, y_test = task.split(df, cfg)

    logger.info("Building pipeline (model=%s)", cfg.model.type)
    preprocessor = task.build_preprocessor(X_train, cfg)
    estimator = build_model(cfg.model, task.model_registry())
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    logger.info("Fitting on %d training rows", len(X_train))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] == 2:
            y_proba = proba[:, 1]

    metrics = task.compute_metrics(y_test.to_numpy(), y_pred, y_proba)
    logger.info("Test metrics: %s", metrics)

    model_path = Path(cfg.artifacts.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_path = Path(cfg.artifacts.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    context = {"y_full": pd.concat([y_train, y_test])}
    plots = task.plot_results(y_test.to_numpy(), y_pred, y_proba, cfg, context)
    logger.info("Saved plots: %s", plots)

    return metrics
