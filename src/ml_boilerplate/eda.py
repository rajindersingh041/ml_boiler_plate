"""Exploratory data analysis: a JSON profile + a handful of standard charts.

Runnable standalone (`main.py eda`) or automatically at the start of
training when `cfg.eda.enabled` is true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_boilerplate.config import EdaConfig
from ml_boilerplate.plotting import (
    plot_correlation_heatmap,
    plot_histograms,
    plot_missing_values,
)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize shape, dtypes, missing values, and basic stats."""
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {
            col: {"count": int(missing_count[col]), "pct": float(missing_pct[col])}
            for col in df.columns
        },
        "numeric_summary": json.loads(numeric_df.describe().to_json()) if not numeric_df.empty else {},
        "categorical_cardinality": {
            col: int(categorical_df[col].nunique(dropna=True)) for col in categorical_df.columns
        },
    }


def run_eda(df: pd.DataFrame, cfg: EdaConfig, report_path: str = "artifacts/eda_report.json") -> dict[str, str]:
    """Write the JSON profile and the standard EDA charts. Returns output paths."""
    profile = profile_dataframe(df)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(profile, indent=2))

    output_dir = Path(cfg.output_dir)
    paths = {
        "report": report_path,
        "missing_values": plot_missing_values(df, str(output_dir / "missing_values.html")),
        "histograms": plot_histograms(df, str(output_dir / "histograms.html")),
        "correlation_heatmap": plot_correlation_heatmap(df, str(output_dir / "correlation_heatmap.html")),
    }
    return paths
