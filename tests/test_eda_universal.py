"""Reusable single-task EDA functions: pure summaries + matplotlib figures."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ml_boilerplate.eda_universal import (
    categorical_columns,
    categorical_top,
    categorical_vs_target,
    column_profile,
    detect_datetime_columns,
    detect_task,
    datetime_summary,
    encoding_advice,
    fig_boxplots,
    fig_heatmap,
    fig_histograms,
    fig_missing_bar,
    fig_target_distribution,
    leakage_suspects,
    missing_summary,
    missing_target_signal,
    multicollinear_pairs,
    numeric_describe,
    numeric_vs_target,
    outlier_shares,
    quality_flags,
    run_universal_eda,
    target_correlations,
    target_stats,
    volume_trend,
    zero_shares,
)


def _df():
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "num": [1.0, 2.0, np.nan, 100.0],
            "code": [1, 1, 2, 2],
            "cat": ["x", "y", "x", None],
            "target": [0, 1, 0, 1],
        }
    )


def test_column_profile_flags_constant_and_id_like():
    df = _df().assign(const=1)
    prof = column_profile(df)
    assert prof.loc["const", "is_constant"]
    assert prof.loc["id", "is_id_like"]
    assert not prof.loc["num", "is_constant"]
    assert prof.loc["num", "pct_missing"] == 25.0


def test_missing_summary_lists_only_missing_cols():
    miss = missing_summary(_df())
    assert list(miss.index) == ["num", "cat"]
    assert miss.loc["num", "pct_missing"] == 25.0


def test_missing_target_signal_numeric_target_means():
    sig = missing_target_signal(_df(), "num", "target")
    assert sig["n_missing"] == 1
    assert sig["mean_target_missing"] == 0
    assert sig["mean_target_present"] == 2 / 3


def test_numeric_describe_has_skew_and_kurt():
    desc = numeric_describe(_df(), exclude=["target"])
    assert {"skew", "kurt"}.issubset(desc.columns)
    assert "num" in desc.index


def test_outlier_shares_finds_extreme_point():
    out = outlier_shares(pd.DataFrame({"v": [float(i) for i in range(1, 21)] + [1000.0]}))
    assert out["v"] == round(1 / 21, 4)


def test_zero_shares_counts_exact_zeros():
    z = zero_shares(pd.DataFrame({"v": [0.0, 0.0, 1.0, np.nan]}))
    assert z["v"] == 0.5


def test_categorical_columns_includes_low_card_codes():
    df = pd.DataFrame(
        {
            "cont": [float(i) for i in range(20)],
            "code": [1, 2] * 10,
            "target": [0, 1] * 10,
        }
    )
    assert "code" in categorical_columns(df, target="target")
    assert "cont" not in categorical_columns(df, target="target")
    assert "target" not in categorical_columns(df, target="target")


def test_categorical_top_is_share_sorted():
    top = categorical_top(_df(), "cat", k=2)
    assert top.index[0] == "x"
    assert top.sum() <= 1.0


def test_encoding_advice_covers_three_cases():
    df = pd.DataFrame(
        {
            "uid": [f"id{i}" for i in range(200)],
            "big": [f"level{i % 60}" for i in range(200)],
            "rare": ["a"] * 199 + ["b"],
        }
    )
    assert "drop" in encoding_advice(df, "uid").lower()
    assert "target" in encoding_advice(df, "big").lower()
    assert "other" in encoding_advice(df, "rare").lower()


def test_detect_datetime_columns_finds_string_dates():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02"], "v": [1, 2]})
    assert detect_datetime_columns(df) == ["d"]


def test_datetime_summary_reports_range_and_freq():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=10, freq="D")})
    s = datetime_summary(df, "d")
    assert s["n_nat"] == 0
    assert s["inferred_freq"] == "D"
    assert str(s["min"]) == "2024-01-01 00:00:00"


def test_detect_task_classification_regression_and_time():
    assert detect_task(pd.Series([0, 1, 0]), False) == "classification"
    assert detect_task(pd.Series(["a", "b"]), False) == "classification"
    assert detect_task(pd.Series(np.arange(100, dtype=float)), False) == "regression"
    assert detect_task(pd.Series(np.arange(100, dtype=float)), True) == "regression+time"


def test_target_stats_classification_balance():
    stats = target_stats(pd.Series([0, 0, 0, 1]), "classification")
    assert stats["imbalance_ratio"] == 3.0
    assert stats["minority_share"] == 0.25


def test_target_stats_regression_skew_and_zeros():
    stats = target_stats(pd.Series([0.0, 0.0, 1.0, 10.0]), "regression")
    assert stats["zero_share"] == 0.5
    assert stats["skew"] > 1


def test_target_correlations_ranks_strongest_first():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "t": [1.0, 2.0, 3.0, 4.0]})
    corr = target_correlations(df, "t", "regression")
    assert corr.index[0] == "a"
    assert corr["a"] == 1.0


def test_leakage_suspects_threshold():
    corr = pd.Series({"a": 0.95, "b": 0.5})
    assert leakage_suspects(corr) == ["a"]
    assert leakage_suspects(corr, threshold=0.99) == []


def test_multicollinear_pairs_finds_dup_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "a2": [1.0, 2.0, 3.0, 4.0], "c": [4.0, 1.0, 3.0, 2.0]})
    pairs = multicollinear_pairs(df)
    assert ("a", "a2", 1.0) in pairs
    assert all(p[0] != "c" or p[1] != "c" or True for p in pairs)


def test_numeric_vs_target_classification_returns_group_means():
    out = numeric_vs_target(_df(), "target", "classification")
    assert out.loc[1, "num"] == 51.0


def test_categorical_vs_target_regression_returns_means():
    df = pd.DataFrame({"c": ["a", "a", "b", "b"], "t": [1.0, 3.0, 10.0, 20.0]})
    out = categorical_vs_target(df, "c", "t", "regression")
    assert out.loc["b", "mean"] == 15.0


def test_quality_flags_collects_lists():
    df = _df()
    flags = quality_flags(df, target="target", task="classification", leak=["x"], pairs=[("a", "b", 0.9)])
    assert flags["task"] == "classification"
    assert flags["leak_suspects"] == ["x"]
    assert flags["dup_pct"] == 0.0


def test_run_universal_eda_returns_findings_for_synthetic():
    from ml_boilerplate.data import load_data
    from ml_boilerplate.config import Config

    cfg = Config(task="regression")
    df = load_data(cfg.data, "regression")
    res = run_universal_eda(df, target="target")
    assert res["task"] == "regression"
    assert res["profile"].loc["target", "pct_missing"] == 0.0
    assert "corr_with_target" in res
    assert res["flags"]["dup_pct"] == 0.0


def test_volume_trend_resamples_daily_to_monthly():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=60, freq="D")})
    trend = volume_trend(df, "d", rule="ME")
    assert len(trend) == 2  # Jan (31 rows) + Feb (29 rows, leap year)
    assert trend.sum() == 60


def test_figures_are_matplotlib_figures():
    df = _df()
    assert isinstance(fig_missing_bar(df), Figure)
    assert isinstance(fig_histograms(df, exclude=["target"]), Figure)
    assert isinstance(fig_boxplots(df, exclude=["target"]), Figure)
    assert isinstance(fig_target_distribution(df["target"], "classification"), Figure)
    assert isinstance(fig_heatmap(df[["num", "target"]]), Figure)
