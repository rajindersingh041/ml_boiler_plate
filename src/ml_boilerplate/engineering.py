"""Shared feature engineering: datetime parts, holidays, lags, rolling stats.

Pure DataFrame-in/DataFrame-out builders used by every task. Timeseries
delegates its lag/rolling/date-part construction here; tabular tasks use
`featurize` for row-local datetime derivation. The same functions run at
train and predict time, so there is no train/serve skew by construction.

Leakage rule (lags/rolling): every windowed statistic is computed on
`.shift(1)` first, so no feature ever sees the current row's target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_boilerplate.config import Config

_ROLLING_STATS = {"mean", "std", "min", "max"}


def _holiday_calendar(country: str, subdiv: str | None = None, years: list[int] | None = None):
    """Return a `holidays` calendar or raise ValueError for unknown countries.

    `years` must be passed explicitly: the calendar populates lazily, so
    without it `keys()` is empty and no date ever matches.
    """
    import holidays

    try:
        return holidays.country_holidays(country, subdiv=subdiv, years=years)
    except NotImplementedError as exc:
        available = sorted(holidays.list_supported_countries())
        raise ValueError(
            f"Unknown holiday country {country!r}. "
            f"Supported examples: {available[:10]} (full list via holidays.list_supported_countries())"
        ) from exc


def add_datetime_features(
    df: pd.DataFrame,
    col: str,
    prefix: str | None = None,
    cyclical: bool = False,
    holiday_country: str | None = None,
    holiday_subdiv: str | None = None,
) -> pd.DataFrame:
    """Derive numeric date-part (and optional cyclical/holiday) features.

    Unparseable values become NaT and flow into NaN parts, which the
    downstream imputer handles like any other missing numeric.
    """
    out = df.copy()
    ts = pd.to_datetime(out[col], errors="coerce")
    p = prefix or col

    out[f"{p}_year"] = ts.dt.year
    out[f"{p}_month"] = ts.dt.month
    out[f"{p}_day"] = ts.dt.day
    out[f"{p}_dayofweek"] = ts.dt.dayofweek
    out[f"{p}_quarter"] = ts.dt.quarter
    out[f"{p}_weekofyear"] = ts.dt.isocalendar().week.astype("Int64").astype(float)
    out[f"{p}_dayofyear"] = ts.dt.dayofyear
    out[f"{p}_is_month_end"] = ts.dt.is_month_end.astype(int)
    out[f"{p}_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    if cyclical:
        out[f"{p}_month_sin"] = np.sin(2 * np.pi * out[f"{p}_month"] / 12)
        out[f"{p}_month_cos"] = np.cos(2 * np.pi * out[f"{p}_month"] / 12)
        out[f"{p}_dow_sin"] = np.sin(2 * np.pi * out[f"{p}_dayofweek"] / 7)
        out[f"{p}_dow_cos"] = np.cos(2 * np.pi * out[f"{p}_dayofweek"] / 7)

    if holiday_country:
        # Pre-populate the calendar for the data's year range (+1 year so
        # days_to_holiday works for December dates).
        year_vals = ts.dt.year.dropna().astype(int)
        years = list(range(int(year_vals.min()), int(year_vals.max()) + 2)) if len(year_vals) else []
        cal = _holiday_calendar(holiday_country, holiday_subdiv, years)
        hol_set = set(cal.keys())  # datetime.date objects; compare as dates
        day = ts.dt.date  # (avoids datetime64-unit mismatches in np.isin)
        is_hol = day.isin(hol_set).to_numpy()
        out[f"{p}_is_holiday"] = is_hol.astype(int)
        hol_dates = np.array(sorted(pd.to_datetime(list(hol_set)).to_numpy(),
                                    )) if hol_set else np.array([], dtype="datetime64[D]")
        if len(hol_dates):
            days = day.to_numpy().astype("datetime64[D]")
            idx = np.searchsorted(hol_dates, days)
            nxt = np.where(idx < len(hol_dates),
                           hol_dates[np.minimum(idx, len(hol_dates) - 1)],
                           np.datetime64("NaT", "D"))
            delta = (nxt - days).astype("timedelta64[D]").astype(float)
            delta = np.where(is_hol, 0.0, delta)
        else:  # pragma: no cover - a country with zero holidays
            delta = np.full(len(out), np.nan)
        out[f"{p}_days_to_holiday"] = delta

    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    n_lags: list[int],
    rolling_windows: list[int],
    rolling_stats: tuple[str, ...] | list[str] = ("mean", "std"),
    lag_diffs: bool = False,
    date_parts: bool = True,
    cyclical: bool = False,
    holiday_country: str | None = None,
    holiday_subdiv: str | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Add lag, rolling-window, lag-diff, and date-part features.

    Rows without enough history are dropped (warm-up). Unknown rolling
    stats raise ValueError.
    """
    unknown = set(rolling_stats) - _ROLLING_STATS
    if unknown:
        raise ValueError(
            f"Unsupported rolling stats {sorted(unknown)}. Use any of {sorted(_ROLLING_STATS)}"
        )

    out = df.sort_values(date_col).reset_index(drop=True).copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    series = out[target]

    for lag in n_lags:
        out[f"{prefix}lag_{lag}"] = series.shift(lag)
    if lag_diffs:
        for lag in n_lags:
            out[f"{prefix}lag_{lag}_diff"] = series.diff(lag)
    shifted = series.shift(1)
    for window in rolling_windows:
        windowed = shifted.rolling(window)
        if "mean" in rolling_stats:
            out[f"{prefix}rolling_mean_{window}"] = windowed.mean()
        if "std" in rolling_stats:
            out[f"{prefix}rolling_std_{window}"] = windowed.std()
        if "min" in rolling_stats:
            out[f"{prefix}rolling_min_{window}"] = windowed.min()
        if "max" in rolling_stats:
            out[f"{prefix}rolling_max_{window}"] = windowed.max()

    if date_parts:
        out = add_datetime_features(
            out, date_col, prefix=None, cyclical=cyclical,
            holiday_country=holiday_country, holiday_subdiv=holiday_subdiv,
        )
        # add_datetime_features prefixes with the date column name; the lag
        # namespace uses bare names, so strip a "<date_col>_" prefix back off.
        out = out.rename(columns={
            c: c[len(date_col) + 1:] for c in out.columns
            if c.startswith(f"{date_col}_")
        })

    return out.dropna().reset_index(drop=True)


def featurize(df: pd.DataFrame, cfg: Config, task_name: str) -> pd.DataFrame:
    """Apply row-local datetime derivation for tabular tasks.

    No-op unless the task is classification/regression AND
    `features.datetime_columns` is set. Timeseries is always a no-op here:
    its lag path owns the date column. The original datetime columns are
    dropped after derivation (raw timestamps would one-hot explode).
    """
    cols = cfg.features.datetime_columns
    if task_name not in ("classification", "regression") or not cols:
        return df
    out = df.copy()
    for col in cols:
        out = add_datetime_features(
            out, col,
            cyclical=cfg.features.cyclical_encoding,
            holiday_country=cfg.features.holiday_country,
            holiday_subdiv=cfg.features.holiday_subdiv,
        )
        out = out.drop(columns=[col])
    return out
