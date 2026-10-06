"""Drift gate: PSI per numeric column of live batch vs training profile.

Exits nonzero (via the `monitor` CLI) when max PSI exceeds the threshold
so it doubles as a CI gate. PSI > 0.25 is the conventional "large shift".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)
    e_perc = e_counts / max(e_counts.sum(), 1) + 1e-6
    a_perc = a_counts / max(a_counts.sum(), 1) + 1e-6
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def check_drift(train_profile: dict, live_df: pd.DataFrame,
                psi_threshold: float = 0.25) -> dict:
    histograms = (train_profile or {}).get("histograms", {})
    psi: dict[str, float] = {}
    for col in live_df.select_dtypes(include="number").columns:
        s = live_df[col].dropna().to_numpy(dtype=float)
        if s.size == 0:
            continue
        expected = histograms.get(col)
        if not expected:  # no stored reference sample: nothing to compare against
            psi[col] = 0.0
            continue
        e = np.asarray(expected, dtype=float)
        e = e[~np.isnan(e)]
        if e.size == 0:
            psi[col] = 0.0
            continue
        psi[col] = _psi(e, s)
    drifted = any(v > psi_threshold for v in psi.values())
    return {"psi": psi, "drifted": drifted}
