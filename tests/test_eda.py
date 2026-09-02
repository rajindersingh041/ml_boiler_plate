"""EDA produces a missing-value-aware report and the standard charts."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_boilerplate.config import EdaConfig
from ml_boilerplate.eda import profile_dataframe, run_eda


def _sample_df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": ["x", "y", "x", None],
            "target": [0, 1, 0, 1],
        }
    )


def test_profile_dataframe_reports_missing_values():
    profile = profile_dataframe(_sample_df())
    assert profile["n_rows"] == 4
    assert profile["missing"]["a"]["count"] == 1
    assert profile["missing"]["b"]["count"] == 1
    assert profile["categorical_cardinality"]["b"] == 2


def test_run_eda_writes_report_and_charts(tmp_path):
    cfg = EdaConfig(output_dir=str(tmp_path))
    report_path = str(tmp_path / "eda_report.json")

    paths = run_eda(_sample_df(), cfg, report_path=report_path)

    for key in ("report", "missing_values", "histograms", "correlation_heatmap"):
        assert Path(paths[key]).exists()
