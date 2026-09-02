"""Generic table I/O: local paths or http(s) URLs, csv or parquet.

Both read_table and write_table infer the format from the file
extension unless one is given explicitly, so the rest of the pipeline
never has to care where data came from or where it's going.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED = {"csv", "parquet"}


def _infer_format(path: str, format: str | None) -> str:
    if format:
        if format not in _SUPPORTED:
            raise ValueError(f"Unsupported format {format!r}. Use one of {_SUPPORTED}")
        return format
    suffix = Path(path).suffix.lstrip(".").lower()
    if suffix in _SUPPORTED:
        return suffix
    raise ValueError(
        f"Could not infer format from {path!r}; pass format explicitly "
        f"(one of {_SUPPORTED})"
    )


def read_table(path: str, format: str | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet table from a local path or an http(s) URL.

    pandas' csv reader supports http(s) URLs natively. Parquet works for
    local paths out of the box; reading parquet over a URL additionally
    requires `fsspec` to be installed.
    """
    fmt = _infer_format(path, format)
    if fmt == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_table(df: pd.DataFrame, path: str, format: str | None = None) -> None:
    """Write a DataFrame to CSV or Parquet, creating parent directories."""
    fmt = _infer_format(path, format)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
