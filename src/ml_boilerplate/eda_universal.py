"""Reusable single-task EDA functions.

Each function does one thing and returns plain data (DataFrame / Series /
dict) or a matplotlib Figure — no printing, no file I/O. Import them
anywhere (notebooks, scripts, the CLI) instead of copy-pasting EDA cells.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


# ---------------------------------------------------------------- tables ---

def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype / missingness / uniqueness / constant / ID-like flags."""
    prof = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=False),
        }
    )
    prof["is_constant"] = prof["n_unique"] <= 1
    prof["is_id_like"] = prof["n_unique"] >= 0.95 * max(len(df), 1)
    return prof


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Only columns that have missing values, with count + percent."""
    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {"pct_missing": (miss * 100).round(2), "n_missing": df.isna().sum()[miss.index]}
    )


def missing_target_signal(df: pd.DataFrame, col: str, target: str) -> dict[str, Any]:
    """Is *being missing* in `col` predictive of `target`? (quick MNAR check)."""
    m = df[col].isna()
    y = df[target]
    return {
        "n_missing": int(m.sum()),
        "mean_target_missing": float(y[m].mean()) if pd.api.types.is_numeric_dtype(y) else None,
        "mean_target_present": float(y[~m].mean()) if pd.api.types.is_numeric_dtype(y) else None,
    }


def numeric_describe(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """describe() plus skew/kurt for numeric columns."""
    nums = [c for c in df.select_dtypes(include="number").columns if c not in (exclude or [])]
    desc = df[nums].describe().T if nums else pd.DataFrame()
    if not desc.empty:
        desc["skew"] = df[nums].skew()
        desc["kurt"] = df[nums].kurt()
    return desc


def outlier_shares(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, float]:
    """Share of points beyond 1.5xIQR per numeric column."""
    out: dict[str, float] = {}
    for c in df.select_dtypes(include="number").columns:
        if c in (exclude or []):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            out[c] = round(float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()), 4)
    return out


def zero_shares(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, float]:
    """Share of exact zeros per numeric column."""
    return {
        c: round(float((df[c] == 0).mean()), 4)
        for c in df.select_dtypes(include="number").columns
        if c not in (exclude or [])
    }


def categorical_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Object/category/bool cols plus low-cardinality numeric codes."""
    cats = [
        c
        for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c != target
    ]
    for c in df.select_dtypes(include="number").columns:
        if c != target and df[c].nunique(dropna=True) <= 10 and c not in cats:
            cats.append(c)
    return cats


def categorical_top(df: pd.DataFrame, col: str, k: int = 5) -> pd.Series:
    """Top-k value shares for one categorical column."""
    return (df[col].value_counts(dropna=False).head(k) / len(df)).round(4)


def encoding_advice(df: pd.DataFrame, col: str) -> str:
    """One-line encoding recommendation for a categorical column."""
    nu = df[col].nunique(dropna=False)
    if nu == len(df):
        return "unique per row: drop from features (ID-like)"
    if nu > 50:
        return "high cardinality: target/frequency encoding or hashing, not one-hot"
    rare = int((df[col].value_counts(normalize=True, dropna=False) < 0.01).sum())
    if rare:
        return f"{rare} rare level(s) <1%: group into 'Other'"
    return "low cardinality: one-hot encoding is fine"


# -------------------------------------------------------------- datetime ---

def detect_datetime_columns(df: pd.DataFrame, forced: str | None = None) -> list[str]:
    """Already-datetime dtypes plus object cols that mostly parse as dates."""
    if forced and forced in df.columns:
        return [forced]
    cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for c in df.select_dtypes(include="object").columns:
        try:
            s = pd.to_datetime(df[c].dropna().head(200), errors="coerce")
            if len(s.dropna()) / max(len(df[c].dropna().head(200)), 1) > 0.9 and c not in cols:
                cols.append(c)
        except Exception:
            pass
    return cols


def datetime_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Range, NaT count, gaps and inferred frequency for a datetime column."""
    t = pd.to_datetime(df[col], errors="coerce")
    dd = t.dropna().sort_values()
    gaps = dd.diff().dropna()
    return {
        "min": dd.min(),
        "max": dd.max(),
        "n_nat": int(t.isna().sum()),
        "median_gap": gaps.median() if len(gaps) else None,
        "max_gap": gaps.max() if len(gaps) else None,
        "inferred_freq": pd.infer_freq(dd.head(50)) if len(dd) else None,
    }


# ---------------------------------------------------------------- target ---

def detect_task(y: pd.Series, has_time: bool = False) -> str:
    """classification for bool/object/categorical/few-unique, else regression."""
    nuniq = y.nunique(dropna=True)
    if (
        pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or nuniq <= 20
    ):
        task = "classification"
    else:
        task = "regression"
    return task + "+time" if has_time else task


def target_stats(y: pd.Series, task: str) -> dict[str, Any]:
    """Balance (classification) or skew/zeros (regression) summary."""
    if task.startswith("classification"):
        vc = y.value_counts(dropna=False, normalize=True)
        return {
            "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1e-9)), 2),
            "minority_share": round(float(vc.min()), 4),
            "class_share": vc.round(4).to_dict(),
        }
    yn = pd.to_numeric(y, errors="coerce")
    return {
        "skew": round(float(yn.skew()), 3),
        "kurt": round(float(yn.kurt()), 3),
        "zero_share": round(float((yn == 0).mean()), 4),
    }


def target_correlations(df: pd.DataFrame, target: str, task: str) -> pd.Series:
    """Numeric features ranked by |corr| with the target."""
    nums = [c for c in df.select_dtypes(include="number").columns if c != target]
    tmp = df[nums + [target]].copy()
    if task.startswith("classification"):
        tmp[target] = pd.factorize(tmp[target])[0]
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    return tmp.corr(numeric_only=True)[target].drop(target, errors="ignore").sort_values(
        key=abs, ascending=False
    )


def leakage_suspects(corr: pd.Series, threshold: float = 0.9) -> list[str]:
    """|corr| above threshold — possibly computed after the outcome."""
    return corr[corr.abs() > threshold].index.tolist()


def multicollinear_pairs(
    df: pd.DataFrame, exclude: list[str] | None = None, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Inter-feature pairs with |r| above threshold."""
    nums = [c for c in df.select_dtypes(include="number").columns if c not in (exclude or [])]
    pairs: list[tuple[str, str, float]] = []
    if len(nums) > 1:
        c = df[nums].corr()
        for i in range(len(c.columns)):
            for j in range(i + 1, len(c.columns)):
                if abs(c.iloc[i, j]) > threshold:
                    pairs.append((c.columns[i], c.columns[j], round(float(c.iloc[i, j]), 3)))
    return pairs


def numeric_vs_target(df: pd.DataFrame, target: str, task: str) -> pd.DataFrame | pd.Series:
    """Group means (classification) or per-feature corr (regression)."""
    nums = [c for c in df.select_dtypes(include="number").columns if c != target][:8]
    if task.startswith("classification"):
        return df[nums + [target]].groupby(target, dropna=False).mean(numeric_only=True).round(3)
    yn = pd.to_numeric(df[target], errors="coerce")
    rs = {c: pd.Series(pd.to_numeric(df[c], errors="coerce")).corr(yn) for c in nums}
    return pd.Series(rs).sort_values(key=abs, ascending=False).round(3)


def categorical_vs_target(
    df: pd.DataFrame, col: str, target: str, task: str
) -> pd.DataFrame:
    """Crosstab (classification) or mean-target table (regression) for one cat column."""
    if task.startswith("classification"):
        return pd.crosstab(df[col].fillna("∅_missing"), df[target], normalize="index").round(3)
    return (
        df.assign(__y=pd.to_numeric(df[target], errors="coerce"))
        .groupby(col, dropna=False)["__y"]
        .agg(["mean", "count"])
        .sort_values("mean")
        .round(3)
    )


def quality_flags(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    leak: list[str] | None = None,
    pairs: list[tuple[str, str, float]] | None = None,
) -> dict[str, Any]:
    """Single go/no-go dict: dups, constants, IDs, heavy-missing, leakage, task."""
    return {
        "dup_pct": round(float(df.duplicated().mean()), 4),
        "constant": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "id_like": [c for c in df.columns if df[c].nunique(dropna=False) == len(df)],
        "missing_gt30": df.columns[df.isna().mean() > 0.3].tolist(),
        "leak_suspects": leak or [],
        "multicollinear": (pairs or [])[:6],
        "task": task,
        "target": target,
    }


def run_universal_eda(
    df: pd.DataFrame,
    target: str | None = None,
    date: str | None = None,
    id_cols: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run every check; return a findings dict (tables inline, no plots)."""
    dt_cols = detect_datetime_columns(df, forced=date)
    task = detect_task(df[target], bool(dt_cols)) if target in df.columns else None
    corr = target_correlations(df, target, task) if task else pd.Series(dtype=float)
    leak = leakage_suspects(corr) if task else []
    pairs = multicollinear_pairs(df, exclude=[target] if target else None)
    return {
        "shape": df.shape,
        "profile": column_profile(df),
        "missing": missing_summary(df),
        "numeric_describe": numeric_describe(df, exclude=[target] if target else None),
        "outliers": outlier_shares(df, exclude=[target] if target else None),
        "zeros": zero_shares(df, exclude=[target] if target else None),
        "categoricals": categorical_columns(df, target=target),
        "datetime_columns": dt_cols,
        "task": task,
        "target_stats": target_stats(df[target], task) if task else None,
        "corr_with_target": corr,
        "flags": quality_flags(df, target=target, task=task, leak=leak, pairs=pairs),
        "id_cols": [c for c in id_cols if c in df.columns],
    }


def volume_trend(df: pd.DataFrame, col: str, rule: str | None = None) -> pd.Series:
    """Row counts resampled over time (monthly for sparse dates, else daily)."""
    t = pd.to_datetime(df[col], errors="coerce")
    dd = pd.DataFrame({"t": t}).dropna().sort_values("t")
    if rule is None:
        rule = "ME" if dd["t"].diff().dropna().median() >= pd.Timedelta(days=20) else "D"
    return dd.set_index("t").assign(n=1)["n"].resample(rule).sum()


# ---------------------------------------------------------------- figures ---

def _fig(ncols: int = 1) -> tuple[Figure, Any]:
    sns.set_theme(style="whitegrid")
    return plt.subplots(1, ncols, figsize=(4 * ncols, 3))


def fig_missing_bar(df: pd.DataFrame) -> Figure:
    """Horizontal bar chart of per-column missingness."""
    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = _fig()
    if len(miss):
        miss.plot(kind="barh", ax=ax)
        ax.set_xlabel("fraction missing")
    ax.set_title("Missingness by column")
    fig.tight_layout()
    return fig


def fig_histograms(df: pd.DataFrame, exclude: list[str] | None = None, k: int = 4) -> Figure:
    """Histograms of the first k numeric columns."""
    nums = [c for c in df.select_dtypes(include="number").columns if c not in (exclude or [])][:k]
    fig, axes = _fig(max(len(nums), 1))
    for ax, c in zip(np.atleast_1d(axes), nums):
        ax.hist(df[c].dropna(), bins=30)
        ax.set_title(c)
    fig.tight_layout()
    return fig


def fig_boxplots(df: pd.DataFrame, exclude: list[str] | None = None, k: int = 4) -> Figure:
    """Boxplots of the first k numeric columns."""
    nums = [c for c in df.select_dtypes(include="number").columns if c not in (exclude or [])][:k]
    fig, axes = _fig(max(len(nums), 1))
    for ax, c in zip(np.atleast_1d(axes), nums):
        ax.boxplot(df[c].dropna(), vert=True)
        ax.set_title(c)
    fig.tight_layout()
    return fig


def fig_target_distribution(y: pd.Series, task: str) -> Figure:
    """Bar (classification) or histogram (regression) of the target."""
    fig, ax = _fig()
    if task.startswith("classification"):
        y.value_counts(dropna=False, normalize=True).sort_index().plot(kind="bar", ax=ax)
        ax.set_ylabel("share")
    else:
        ax.hist(pd.to_numeric(y, errors="coerce").dropna(), bins=50)
    ax.set_title("Target distribution")
    fig.tight_layout()
    return fig


def fig_heatmap(df: pd.DataFrame, exclude: list[str] | None = None, k: int = 12) -> Figure:
    """Correlation heatmap of numeric columns."""
    nums = [c for c in df.select_dtypes(include="number").columns if c not in (exclude or [])][:k]
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df[nums].corr(), cmap="coolwarm", center=0)
    plt.title("Feature correlation")
    fig.tight_layout()
    return fig
