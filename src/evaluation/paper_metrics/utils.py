"""Shared helpers for single-event-table paper metrics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def normalize_value(value: Any) -> str:
    if is_null_like(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            numeric = float(text)
        except ValueError:
            return text
        if numeric.is_integer():
            return str(int(numeric))
    return text


def normalize_text(value: Any) -> str:
    if is_null_like(value):
        return ""
    return " ".join(str(value).strip().split())


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def datetime_numeric(series: pd.Series) -> pd.Series:
    parsed = datetime_series(series)
    numeric = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    numeric[parsed.isna().to_numpy()] = np.nan
    return pd.Series(numeric, index=series.index)


def token_lengths(series: pd.Series) -> pd.Series:
    return series.map(lambda value: len(normalize_text(value).split())).astype(float)


def char_lengths(series: pd.Series) -> pd.Series:
    return series.map(lambda value: len(normalize_text(value))).astype(float)


def ks_distance(left: Any, right: Any) -> float | None:
    a = pd.Series(left).dropna().astype(float).to_numpy()
    b = pd.Series(right).dropna().astype(float).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return None
    a = np.sort(a)
    b = np.sort(b)
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, values, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b))) if len(values) else 0.0


def total_variation(left: pd.Series, right: pd.Series, support: list[Any] | None = None) -> float:
    if support is None:
        support = sorted(set(left.dropna().map(str)).union(set(right.dropna().map(str))))
    l = left.dropna().map(str).value_counts(normalize=True).reindex([str(item) for item in support], fill_value=0.0)
    r = right.dropna().map(str).value_counts(normalize=True).reindex([str(item) for item in support], fill_value=0.0)
    return float(0.5 * np.abs(l.to_numpy(dtype=float) - r.to_numpy(dtype=float)).sum())


def wasserstein_1d(left: Any, right: Any) -> float | None:
    a = pd.Series(left).dropna().astype(float).to_numpy()
    b = pd.Series(right).dropna().astype(float).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, max(len(a), len(b)))
    return float(np.mean(np.abs(np.quantile(a, quantiles) - np.quantile(b, quantiles))))


def safe_corr(left: Any, right: Any) -> float | None:
    a = pd.Series(left).astype(float)
    b = pd.Series(right).astype(float)
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 2 or frame["a"].nunique() <= 1 or frame["b"].nunique() <= 1:
        return None
    return float(frame["a"].corr(frame["b"]))


def gini(values: Any) -> float | None:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return None
    if np.allclose(arr, 0):
        return 0.0
    arr = np.sort(np.clip(arr, 0.0, None))
    n = len(arr)
    return float((2.0 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - ((n + 1.0) / n))


def text_hash_embedding(text: Any, dim: int = 64) -> np.ndarray:
    tokens = normalize_text(text).lower().split()
    vec = np.zeros(int(dim), dtype=float)
    for token in tokens or [""]:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % int(dim)
        sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def flatten_dict(prefix: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        metric = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_dict(metric, value))
        else:
            rows.append({"metric": metric, "value": value})
    return rows
