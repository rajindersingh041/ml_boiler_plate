"""Sanity-check the shipped example configs parse correctly and point at
the intended real dataset (no network call — just YAML parsing)."""

from ml_boilerplate.config import Config


def test_classification_config_points_at_titanic():
    cfg = Config.from_yaml("configs/classification.yaml")
    assert cfg.task == "classification"
    assert cfg.data.source == "file"
    assert "titanic" in cfg.data.path.lower()
    assert cfg.data.target_column == "Survived"


def test_regression_config_points_at_housing():
    cfg = Config.from_yaml("configs/regression.yaml")
    assert cfg.task == "regression"
    assert cfg.data.source == "file"
    assert "housing" in cfg.data.path.lower()
    assert cfg.data.target_column == "median_house_value"


def test_timeseries_config_points_at_airline_passengers():
    cfg = Config.from_yaml("configs/timeseries.yaml")
    assert cfg.task == "timeseries"
    assert cfg.data.source == "file"
    assert "airline-passengers" in cfg.data.path.lower()
    assert cfg.data.date_column == "Month"
    assert cfg.data.target_column == "Passengers"
