"""Smoke test: timeseries pipeline trains end-to-end on a synthetic series."""

from pathlib import Path

from ml_boilerplate.config import Config
from ml_boilerplate.train import run_training


def test_timeseries_produces_forecast(tmp_path):
    cfg = Config(task="timeseries")
    cfg.data.date_column = "date"
    cfg.artifacts.model_path = str(tmp_path / "model.joblib")
    cfg.artifacts.metrics_path = str(tmp_path / "metrics.json")
    cfg.artifacts.plots_dir = str(tmp_path / "plots")
    cfg.eda.output_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)

    for key in ("mae", "rmse", "mape", "smape"):
        assert key in metrics
    assert Path(cfg.artifacts.model_path).exists()
    assert (Path(cfg.artifacts.plots_dir) / "forecast.html").exists()
