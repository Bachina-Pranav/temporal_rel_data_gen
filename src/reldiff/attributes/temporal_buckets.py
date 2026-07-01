"""Canonical temporal bucket helpers for temporal attribute generation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd


def canonical_temporal_bucket(timestamps: Any, level: str) -> pd.Series:
    """Return canonical string buckets for timestamp-like values."""

    index = timestamps.index if isinstance(timestamps, pd.Series) else None
    values = pd.Series(pd.to_datetime(timestamps, errors="coerce"), index=index)
    if level == "global":
        return pd.Series(["global"] * len(values), index=values.index)
    if level == "date":
        return values.dt.strftime("%Y-%m-%d")
    if level == "year":
        return values.dt.strftime("%Y")
    if level == "year_month":
        return values.dt.strftime("%Y-%m")
    if level == "month":
        return values.dt.strftime("%m")
    raise ValueError("temporal bucket level must be year_month, month, date, year, or global")


def infer_bucket_format(bucket_values: Iterable[Any]) -> str:
    keys = [str(value) for value in bucket_values if value is not None and not pd.isna(value)]
    if not keys:
        return "unknown"
    if set(keys) == {"global"}:
        return "global"
    if all(is_year_month_day_key(key) for key in keys):
        return "YYYY-MM-DD"
    if all(is_year_month_key(key) for key in keys):
        return "YYYY-MM"
    if all(is_year_key(key) for key in keys):
        return "YYYY"
    if all(is_padded_month_key(key) for key in keys):
        return "MM"
    if all(is_legacy_month_number_key(key) for key in keys):
        return "legacy-month-number"
    return "unknown"


def normalize_legacy_month_buckets(bucket_values: Iterable[Any]) -> pd.Series:
    """Normalize month-of-year keys to zero-padded MM for diagnostics only."""

    rows = []
    for value in bucket_values:
        if value is None or pd.isna(value):
            rows.append(None)
            continue
        text = str(value).strip()
        if is_legacy_month_number_key(text) or is_padded_month_key(text):
            rows.append(f"{int(text):02d}")
        else:
            rows.append(text)
    return pd.Series(rows)


def is_legacy_bucket_format(bucket_values: Iterable[Any]) -> bool:
    return infer_bucket_format(bucket_values) == "legacy-month-number"


def is_year_month_day_key(key: str) -> bool:
    if len(key) != 10 or key[4] != "-" or key[7] != "-":
        return False
    try:
        year = int(key[:4])
        month = int(key[5:7])
        day = int(key[8:])
    except ValueError:
        return False
    return year >= 1 and 1 <= month <= 12 and 1 <= day <= 31


def is_year_month_key(key: str) -> bool:
    if len(key) != 7 or key[4] != "-":
        return False
    try:
        year = int(key[:4])
        month = int(key[5:])
    except ValueError:
        return False
    return year >= 1 and 1 <= month <= 12


def is_year_key(key: str) -> bool:
    if len(key) != 4:
        return False
    try:
        year = int(key)
    except ValueError:
        return False
    return year >= 1


def is_padded_month_key(key: str) -> bool:
    if len(key) != 2:
        return False
    try:
        month = int(key)
    except ValueError:
        return False
    return key == f"{month:02d}" and 1 <= month <= 12


def is_legacy_month_number_key(key: str) -> bool:
    try:
        month = int(key)
    except ValueError:
        return False
    return str(month) == key and 1 <= month <= 12


def expected_bucket_format(level: str) -> str:
    if level == "year_month":
        return "YYYY-MM"
    if level == "month":
        return "MM"
    if level == "date":
        return "YYYY-MM-DD"
    if level == "year":
        return "YYYY"
    if level == "global":
        return "global"
    raise ValueError("temporal bucket level must be year_month, month, date, year, or global")
