"""Smoke test: regression pipeline trains end-to-end on synthetic data."""

from pathlib import Path

from ml_boilerplate.config import Config
from ml_boilerplate.train import run_training


def test_regression_produces_metrics(tmp_path):
    cfg = Config(task="regression")
    cfg.model.type = "gradient_boosting"
    cfg.artifacts.model_path = str(tmp_path / "model.joblib")
    cfg.artifacts.metrics_path = str(tmp_path / "metrics.json")
    cfg.artifacts.plots_dir = str(tmp_path / "plots")
    cfg.eda.output_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)

    for key in ("mae", "rmse", "r2", "mape", "smape"):
        assert key in metrics
    assert Path(cfg.artifacts.model_path).exists()
    assert (Path(cfg.artifacts.plots_dir) / "predicted_vs_actual.html").exists()
    assert (Path(cfg.artifacts.plots_dir) / "residuals.html").exists()
