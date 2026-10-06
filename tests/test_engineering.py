"""Unit tests: shared feature-engineering builders (datetime, lags, holidays)."""

import numpy as np
import pandas as pd
import pytest

from ml_boilerplate.engineering import (
    add_datetime_features,
    add_lag_features,
    featurize,
)


@pytest.fixture
def dated():
    return pd.DataFrame({"d": pd.to_datetime(["2024-07-03", "2024-07-04", "2024-07-06"])})


def test_date_parts(dated):
    out = add_datetime_features(dated, "d")
    # 2024-07-03 is a Wednesday, 2024-07-06 a Saturday
    assert out["d_dayofweek"].tolist() == [2, 3, 5]
    assert out["d_is_weekend"].tolist() == [0, 0, 1]
    assert out["d_month"].tolist() == [7, 7, 7]
    assert out["d_year"].tolist() == [2024, 2024, 2024]


def test_cyclical_encoding(dated):
    out = add_datetime_features(dated, "d", cyclical=True)
    expected = np.sin(2 * np.pi * 7 / 12)
    assert np.allclose(out["d_month_sin"], expected)
    assert np.allclose(out["d_month_sin"] ** 2 + out["d_month_cos"] ** 2, 1.0)


def test_us_holidays(dated):
    out = add_datetime_features(dated, "d", holiday_country="US")
    assert out["d_is_holiday"].tolist() == [0, 1, 0]  # 2024-07-04 only
    # next federal holiday after Sat 2024-07-06 is Labor Day 2024-09-02
    assert out["d_days_to_holiday"].tolist() == [1, 0, 58]


def test_unknown_country_raises_value_error(dated):
    with pytest.raises(ValueError, match="US-XX|XX"):
        add_datetime_features(dated, "d", holiday_country="XX")


def test_lag_and_rolling_use_shifted_values():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    out = add_lag_features(df, target="y", date_col="date", n_lags=[1, 2],
                           rolling_windows=[2], rolling_stats=["mean"],
                           date_parts=False)
    # warm-up: max(2, 2) rows dropped -> 4 rows left, starting at y == 30
    assert out["y"].tolist() == [30.0, 40.0, 50.0, 60.0]
    assert out["lag_1"].tolist() == [20.0, 30.0, 40.0, 50.0]
    # rolling mean over the two rows BEFORE the current one (shifted, no leak)
    assert out["rolling_mean_2"].tolist() == [15.0, 25.0, 35.0, 45.0]


def test_lag_diffs():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "y": [10.0, 15.0, 25.0, 40.0],
    })
    out = add_lag_features(df, target="y", date_col="date", n_lags=[1],
                           rolling_windows=[], lag_diffs=True, date_parts=False)
    assert out["lag_1_diff"].tolist() == [5.0, 10.0, 15.0]


def test_unsupported_rolling_stat_raises():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="skew"):
        add_lag_features(df, target="y", date_col="date", n_lags=[1],
                         rolling_windows=[2], rolling_stats=["skew"])


def test_featurize_drops_original_and_noops():
    df = pd.DataFrame({
        "d": ["2024-01-01", "2024-01-02"],
        "x": [1.0, 2.0],
        "target": [0, 1],
    })
    from ml_boilerplate.config import Config
    cfg = Config(task="classification")
    cfg.features.datetime_columns = ["d"]
    out = featurize(df, cfg, "classification")
    assert "d" not in out.columns
    assert "d_dayofweek" in out.columns
    assert out["x"].tolist() == [1.0, 2.0]

    cfg.features.datetime_columns = None
    same = featurize(df, cfg, "classification")
    assert list(same.columns) == ["d", "x", "target"]

    cfg.features.datetime_columns = ["d"]
    ts = featurize(df, cfg, "timeseries")  # lag path owns timeseries: no-op
    assert list(ts.columns) == ["d", "x", "target"]


def test_default_lag_output_matches_legacy_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "target": np.arange(30, dtype=float),
    })
    out = add_lag_features(df, target="target", date_col="date",
                           n_lags=[1, 7, 14], rolling_windows=[7, 14])
    # legacy set from the old inline builder: lags + rolling mean/std +
    # dayofweek/month/day (superset check: new date parts may add more)
    legacy = {"lag_1", "lag_7", "lag_14",
              "rolling_mean_7", "rolling_std_7",
              "rolling_mean_14", "rolling_std_14",
              "dayofweek", "month", "day"}
    assert legacy.issubset(set(out.columns))
    assert set(out.columns) == legacy | {"date", "target", "year", "quarter",
                                         "weekofyear", "dayofyear",
                                         "is_month_end", "is_weekend"}
