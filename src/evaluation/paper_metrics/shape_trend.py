"""Column shape and pair trend metrics for single event tables."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .utils import char_lengths, datetime_numeric, ks_distance, numeric_series, safe_corr, token_lengths, total_variation, wasserstein_1d


def shape_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, table_config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_column: dict[str, Any] = {}
    for column, cfg in (table_config.get("columns", {}) or {}).items():
        col_type = str((cfg or {}).get("type", "categorical")).lower()
        if column not in real or column not in synthetic:
            continue
        metric = column_shape_metric(real[column], synthetic[column], col_type, cfg)
        per_column[column] = metric
        rows.append({"column": column, **metric})
    all_errors = [row["shape_error"] for row in rows if row.get("shape_error") is not None]
    attr_errors = [
        row["shape_error"]
        for row in rows
        if row.get("shape_error") is not None and row.get("type") not in {"foreign_key", "primary_key"}
    ]
    text_errors = [row["shape_error"] for row in rows if row.get("shape_error") is not None and row.get("type") == "text"]
    payload = {
        "macro_shape_error": float(np.mean(all_errors)) if all_errors else None,
        "macro_attribute_shape_error": float(np.mean(attr_errors)) if attr_errors else None,
        "macro_text_shape_error": float(np.mean(text_errors)) if text_errors else None,
        "per_column": per_column,
    }
    return payload, pd.DataFrame(rows)


def column_shape_metric(real_col: pd.Series, syn_col: pd.Series, col_type: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if col_type == "categorical":
        support = cfg.get("valid_values")
        err = total_variation(real_col, syn_col, support=support)
        return {"type": col_type, "shape_error": err, "primary_statistic": "total_variation", "secondary_statistics": {}}
    if col_type in {"numerical", "numeric", "number"}:
        r = numeric_series(real_col)
        s = numeric_series(syn_col)
        err = ks_distance(r, s)
        return {
            "type": "numerical",
            "shape_error": err,
            "primary_statistic": "ks_distance",
            "secondary_statistics": {"wasserstein_distance": wasserstein_1d(r, s)},
        }
    if col_type == "datetime":
        r = datetime_numeric(real_col)
        s = datetime_numeric(syn_col)
        err = ks_distance(r, s)
        return {
            "type": col_type,
            "shape_error": err,
            "primary_statistic": "timestamp_ks_distance",
            "secondary_statistics": {"wasserstein_distance": wasserstein_1d(r, s)},
        }
    if col_type == "text":
        r_tok = token_lengths(real_col)
        s_tok = token_lengths(syn_col)
        err = ks_distance(r_tok, s_tok)
        return {
            "type": col_type,
            "shape_error": err,
            "primary_statistic": "token_length_ks",
            "secondary_statistics": {
                "char_length_ks": ks_distance(char_lengths(real_col), char_lengths(syn_col)),
                "token_length_mean_real": float(r_tok.mean()) if len(r_tok) else None,
                "token_length_mean_synthetic": float(s_tok.mean()) if len(s_tok) else None,
            },
        }
    if col_type == "foreign_key":
        r_counts = real_col.astype(str).value_counts()
        s_counts = syn_col.astype(str).value_counts()
        idx = r_counts.index.union(s_counts.index)
        err = ks_distance(r_counts.reindex(idx, fill_value=0), s_counts.reindex(idx, fill_value=0))
        return {"type": col_type, "shape_error": err, "primary_statistic": "active_fk_frequency_ks", "secondary_statistics": {}}
    return {"type": col_type, "shape_error": total_variation(real_col, syn_col), "primary_statistic": "total_variation", "secondary_statistics": {}}


def trend_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, table_config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    columns_cfg = dict(table_config.get("columns", {}) or {})
    pk = table_config.get("primary_key")
    pk_cols = {pk} if isinstance(pk, str) else set(pk or [])
    columns = [column for column in columns_cfg if column in real and column in synthetic and column not in pk_cols]
    rows: list[dict[str, Any]] = []
    for col_a, col_b in combinations(columns, 2):
        type_a = str(columns_cfg[col_a].get("type", "categorical")).lower()
        type_b = str(columns_cfg[col_b].get("type", "categorical")).lower()
        err, metric_used = pair_trend_error(real, synthetic, col_a, col_b, type_a, type_b)
        rows.append(
            {
                "col_a": col_a,
                "col_b": col_b,
                "type_a": type_a,
                "type_b": type_b,
                "trend_error": err,
                "metric_used": metric_used,
            }
        )
    values = [row["trend_error"] for row in rows if row.get("trend_error") is not None]
    attr_values = [
        row["trend_error"]
        for row in rows
        if row.get("trend_error") is not None and "foreign_key" not in {row.get("type_a"), row.get("type_b")}
    ]
    text_values = [
        row["trend_error"]
        for row in rows
        if row.get("trend_error") is not None and "text" in {row.get("type_a"), row.get("type_b")}
    ]
    payload = {
        "macro_trend_error": float(np.mean(values)) if values else None,
        "macro_attribute_trend_error": float(np.mean(attr_values)) if attr_values else None,
        "macro_text_cross_modal_trend_error": float(np.mean(text_values)) if text_values else None,
        "per_pair": rows,
    }
    return payload, pd.DataFrame(rows)


def pair_trend_error(real: pd.DataFrame, synthetic: pd.DataFrame, col_a: str, col_b: str, type_a: str, type_b: str) -> tuple[float | None, str]:
    a_real = feature_for_pair(real[col_a], type_a)
    b_real = feature_for_pair(real[col_b], type_b)
    a_syn = feature_for_pair(synthetic[col_a], type_a)
    b_syn = feature_for_pair(synthetic[col_b], type_b)
    if is_categorical_type(type_a) and is_categorical_type(type_b):
        return contingency_tvd(real[col_a], real[col_b], synthetic[col_a], synthetic[col_b]), "contingency_total_variation"
    corr_real = safe_corr(a_real, b_real)
    corr_syn = safe_corr(a_syn, b_syn)
    if corr_real is None or corr_syn is None:
        if corr_real is None and corr_syn is None:
            return 0.0, "correlation_unavailable_both"
        return 1.0, "correlation_unavailable_one_side"
    return float(min(abs(corr_real - corr_syn) / 2.0, 1.0)), "absolute_correlation_difference"


def is_categorical_type(col_type: str) -> bool:
    return col_type in {"categorical", "foreign_key"}


def feature_for_pair(series: pd.Series, col_type: str) -> pd.Series:
    if col_type in {"numerical", "numeric", "number"}:
        return numeric_series(series)
    if col_type == "datetime":
        return datetime_numeric(series)
    if col_type == "text":
        return token_lengths(series)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def contingency_tvd(real_a: pd.Series, real_b: pd.Series, syn_a: pd.Series, syn_b: pd.Series) -> float:
    real_pairs = real_a.astype(str) + "\u241f" + real_b.astype(str)
    syn_pairs = syn_a.astype(str) + "\u241f" + syn_b.astype(str)
    return total_variation(real_pairs, syn_pairs)
