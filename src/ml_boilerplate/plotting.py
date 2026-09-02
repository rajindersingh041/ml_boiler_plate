"""Plotly chart builders, saved as standalone interactive HTML files.

Kept separate from eda.py/tasks/ so any of them can render a chart
without duplicating plotly boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve


def _save(fig: go.Figure, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def plot_missing_values(df: pd.DataFrame, path: str) -> str:
    missing = df.isna().mean().sort_values(ascending=False) * 100
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "column", "y": "% missing"},
        title="Missing values by column",
    )
    return _save(fig, path)


def plot_histograms(df: pd.DataFrame, path: str) -> str:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No numeric columns to plot")
        return _save(fig, path)
    melted = numeric_df.melt(var_name="column", value_name="value")
    fig = px.histogram(
        melted, x="value", facet_col="column", facet_col_wrap=3,
        title="Distributions of numeric columns",
    )
    fig.update_xaxes(matches=None)
    return _save(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: str) -> str:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for a correlation heatmap")
        return _save(fig, path)
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Correlation heatmap",
    )
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, labels=dict(x="predicted", y="actual"),
        title="Confusion matrix",
    )
    return _save(fig, path)


def plot_roc_curve(y_true, y_proba, path: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _save(fig, path)


def plot_predicted_vs_actual(y_true, y_pred, path: str) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="predictions"))
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line=dict(dash="dash")))
    fig.update_layout(title="Predicted vs actual", xaxis_title="actual", yaxis_title="predicted")
    return _save(fig, path)


def plot_residuals(y_true, y_pred, path: str) -> str:
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "predicted", "y": "residual"}, title="Residuals")
    fig.add_hline(y=0, line_dash="dash")
    return _save(fig, path)


def plot_forecast(dates, y_true, y_pred, split_index: int, path: str) -> str:
    """Actual vs predicted target over time, with the train/test boundary marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="actual"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="predicted"))
    if 0 <= split_index < len(dates):
        fig.add_vline(x=dates[split_index], line_dash="dash", annotation_text="train/test split")
    fig.update_layout(title="Forecast: actual vs predicted", xaxis_title="date", yaxis_title="target")
    return _save(fig, path)
