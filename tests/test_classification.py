"""Smoke test: classification pipeline trains end-to-end on synthetic data."""

from pathlib import Path

from ml_boilerplate.config import Config
from ml_boilerplate.train import run_training


def test_classification_produces_metrics(tmp_path):
    cfg = Config(task="classification")
    cfg.artifacts.model_path = str(tmp_path / "model.joblib")
    cfg.artifacts.metrics_path = str(tmp_path / "metrics.json")
    cfg.artifacts.plots_dir = str(tmp_path / "plots")
    cfg.eda.output_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "roc_auc" in metrics
    assert Path(cfg.artifacts.model_path).exists()
    assert Path(cfg.artifacts.metrics_path).exists()
    assert (Path(cfg.artifacts.plots_dir) / "confusion_matrix.html").exists()
