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


def datetime_normalized(series: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    parsed = datetime_series(series)
    ref = datetime_series(reference) if reference is not None else parsed
    ref = ref.dropna()
    if len(ref) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ref_min = ref.min().value
    ref_span = max(float(ref.max().value - ref_min), 1.0)
    numeric = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    out = (numeric - float(ref_min)) / ref_span
    out[parsed.isna().to_numpy()] = np.nan
    return pd.Series(out, index=series.index)


def datetime_wasserstein_summary(real: pd.Series, synthetic: pd.Series) -> dict[str, float | None]:
    real_parsed = datetime_series(real).dropna()
    syn_parsed = datetime_series(synthetic).dropna()
    if len(real_parsed) == 0 or len(syn_parsed) == 0:
        return {
            "normalized_wasserstein": None,
            "wasserstein_days": None,
            "wasserstein_weeks": None,
            "wasserstein_months_approx": None,
        }
    normalized = wasserstein_1d(datetime_normalized(real_parsed, real_parsed), datetime_normalized(syn_parsed, real_parsed))
    days = wasserstein_1d(datetime_numeric(real_parsed) / (24 * 60 * 60 * 1e9), datetime_numeric(syn_parsed) / (24 * 60 * 60 * 1e9))
    return {
        "normalized_wasserstein": normalized,
        "wasserstein_days": days,
        "wasserstein_weeks": float(days / 7.0) if days is not None else None,
        "wasserstein_months_approx": float(days / 30.4375) if days is not None else None,
    }


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


def canonicalize_categorical_value(value: Any, column_config: dict[str, Any] | None = None) -> Any:
    column_config = column_config or {}
    if is_null_like(value):
        return pd.NA
    dtype = str(column_config.get("dtype", "")).lower()
    valid_values = column_config.get("valid_values")
    normalized = normalize_value(value)
    if is_binary_categorical(column_config):
        lowered = normalized.lower()
        if lowered in {"true", "t", "yes", "y"}:
            return 1
        if lowered in {"false", "f", "no", "n"}:
            return 0
        numeric = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric) in {0.0, 1.0}:
            return int(numeric)
    if dtype in {"int", "integer", "int64", "int32"} or valid_values_are_integer_like(valid_values):
        numeric = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric).is_integer():
            return int(numeric)
    if dtype in {"float", "double", "number", "numeric"}:
        numeric = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return float(numeric)
    return normalized


def canonicalize_categorical_series(series: pd.Series, column_config: dict[str, Any] | None = None) -> pd.Series:
    return series.map(lambda value: canonicalize_categorical_value(value, column_config))


def categorical_canonicalization_diagnostics(
    real: pd.Series,
    synthetic: pd.Series,
    column_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    column_config = column_config or {}
    real_after = canonicalize_categorical_series(real, column_config)
    syn_after = canonicalize_categorical_series(synthetic, column_config)
    valid = canonical_valid_values(column_config)
    return {
        "real_unique_before": unique_preview(real),
        "synthetic_unique_before": unique_preview(synthetic),
        "real_unique_after": unique_preview(real_after),
        "synthetic_unique_after": unique_preview(syn_after),
        "unmapped_real_count": unmapped_count(real_after, valid),
        "unmapped_synthetic_count": unmapped_count(syn_after, valid),
    }


def canonical_valid_values(column_config: dict[str, Any] | None = None) -> set[Any] | None:
    column_config = column_config or {}
    valid_values = column_config.get("valid_values")
    if valid_values is None:
        return None
    return {canonicalize_categorical_value(value, column_config) for value in valid_values}


def unique_preview(series: pd.Series, limit: int = 25) -> list[Any]:
    values = series.dropna().unique().tolist()
    values = sorted(values, key=lambda item: str(item))[: int(limit)]
    return jsonable(values)


def unmapped_count(series: pd.Series, valid_values: set[Any] | None) -> int:
    if valid_values is None:
        return 0
    mask = series.notna() & ~series.isin(valid_values)
    return int(mask.sum())


def is_binary_categorical(column_config: dict[str, Any]) -> bool:
    dtype = str(column_config.get("dtype", "")).lower()
    if dtype in {"bool", "boolean"}:
        return True
    valid_values = column_config.get("valid_values")
    if valid_values is None:
        return False
    normalized = {normalize_value(value).lower() for value in valid_values}
    return normalized.issubset({"0", "1", "true", "false"}) and normalized != set()


def valid_values_are_integer_like(valid_values: Any) -> bool:
    if valid_values is None:
        return False
    try:
        values = list(valid_values)
    except TypeError:
        return False
    if not values:
        return False
    for value in values:
        numeric = pd.to_numeric(pd.Series([normalize_value(value)]), errors="coerce").iloc[0]
        if pd.isna(numeric) or not float(numeric).is_integer():
            return False
    return True


def flatten_dict(prefix: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        metric = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_dict(metric, value))
        else:
            rows.append({"metric": metric, "value": value})
    return rows
