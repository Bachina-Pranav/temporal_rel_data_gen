"""Column shape and pair trend metrics for single event tables."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    canonicalize_categorical_series,
    char_lengths,
    datetime_normalized,
    datetime_wasserstein_summary,
    ks_distance,
    numeric_series,
    safe_corr,
    token_lengths,
    total_variation,
    wasserstein_1d,
)


def shape_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_config: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pk_cols = primary_key_columns(table_config)
    rows: list[dict[str, Any]] = []
    per_column: dict[str, Any] = {}
    for column, cfg in (table_config.get("columns", {}) or {}).items():
        col_type = str((cfg or {}).get("type", "categorical")).lower()
        if column not in real or column not in synthetic:
            continue
        metric = column_shape_metric(real[column], synthetic[column], col_type, cfg)
        is_fk = col_type == "foreign_key"
        is_pk = column in pk_cols
        metric["headline_included"] = not (is_fk or is_pk)
        metric["shape_group"] = "structural" if is_fk or is_pk else ("text" if col_type == "text" else "attribute")
        per_column[column] = metric
        rows.append({"column": column, **metric})
    all_errors = [row["shape_error"] for row in rows if row.get("shape_error") is not None]
    non_id_errors = [
        row["shape_error"]
        for row in rows
        if row.get("shape_error") is not None and bool(row.get("headline_included", True))
    ]
    text_errors = [row["shape_error"] for row in rows if row.get("shape_error") is not None and row.get("type") == "text"]
    structural_errors = [
        row["shape_error"]
        for row in rows
        if row.get("shape_error") is not None and row.get("shape_group") == "structural"
    ]
    payload = {
        "macro_shape_error_all_columns": float(np.mean(all_errors)) if all_errors else None,
        "macro_shape_error": float(np.mean(all_errors)) if all_errors else None,
        "macro_attribute_shape_error": float(np.mean(non_id_errors)) if non_id_errors else None,
        "macro_non_id_shape_error": float(np.mean(non_id_errors)) if non_id_errors else None,
        "macro_text_shape_error": float(np.mean(text_errors)) if text_errors else None,
        "macro_structural_shape_error": float(np.mean(structural_errors)) if structural_errors else None,
        "per_column": per_column,
    }
    return payload, pd.DataFrame(rows)


def column_shape_metric(real_col: pd.Series, syn_col: pd.Series, col_type: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if col_type == "categorical":
        real_col = canonicalize_categorical_series(real_col, cfg)
        syn_col = canonicalize_categorical_series(syn_col, cfg)
        support = cfg.get("valid_values")
        if support is not None:
            support = canonicalize_categorical_series(pd.Series(support), cfg).dropna().drop_duplicates().tolist()
        err = total_variation(real_col, syn_col, support=support)
        secondary: dict[str, Any] = {}
        if bool(cfg.get("ordered", False)) or str(cfg.get("semantic_type", "")).lower() == "ordinal_categorical":
            secondary["ordinal_wasserstein_distance"] = wasserstein_1d(
                numeric_series(real_col),
                numeric_series(syn_col),
            )
        return {"type": col_type, "shape_error": err, "primary_statistic": "total_variation", "secondary_statistics": secondary}
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
        r = datetime_normalized(real_col, real_col)
        s = datetime_normalized(syn_col, real_col)
        err = ks_distance(r, s)
        return {
            "type": col_type,
            "shape_error": err,
            "primary_statistic": "normalized_timestamp_ks_distance",
            "secondary_statistics": datetime_wasserstein_summary(real_col, syn_col),
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


def trend_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_config: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns_cfg = dict(table_config.get("columns", {}) or {})
    trend_cfg = (config or {}).get("trend", {}) or {}
    pk_cols = primary_key_columns(table_config)
    high_cardinality_threshold = int(trend_cfg.get("high_cardinality_threshold", 100))
    columns = [column for column in columns_cfg if column in real and column in synthetic and column not in pk_cols]
    rows: list[dict[str, Any]] = []
    for col_a, col_b in combinations(columns, 2):
        type_a = str(columns_cfg[col_a].get("type", "categorical")).lower()
        type_b = str(columns_cfg[col_b].get("type", "categorical")).lower()
        err, metric_used = pair_trend_error(real, synthetic, col_a, col_b, type_a, type_b, columns_cfg[col_a], columns_cfg[col_b])
        headline_included = pair_in_headline(
            real,
            synthetic,
            col_a,
            col_b,
            type_a,
            type_b,
            pk_cols,
            high_cardinality_threshold,
        )
        rows.append(
            {
                "col_a": col_a,
                "col_b": col_b,
                "type_a": type_a,
                "type_b": type_b,
                "trend_error": err,
                "metric_used": metric_used,
                "headline_included": headline_included,
            }
        )
    values = [row["trend_error"] for row in rows if row.get("trend_error") is not None]
    non_id_values = [
        row["trend_error"]
        for row in rows
        if row.get("trend_error") is not None and "foreign_key" not in {row.get("type_a"), row.get("type_b")}
    ]
    headline_values = [row["trend_error"] for row in rows if row.get("trend_error") is not None and row.get("headline_included")]
    text_values = [
        row["trend_error"]
        for row in rows
        if row.get("trend_error") is not None and "text" in {row.get("type_a"), row.get("type_b")}
    ]
    payload = {
        "macro_trend_error_all_pairs": float(np.mean(values)) if values else None,
        "macro_trend_error": float(np.mean(values)) if values else None,
        "macro_attribute_trend_error": float(np.mean(non_id_values)) if non_id_values else None,
        "macro_non_id_trend_error": float(np.mean(non_id_values)) if non_id_values else None,
        "macro_headline_trend_error": float(np.mean(headline_values)) if headline_values else None,
        "macro_text_cross_modal_trend_error": float(np.mean(text_values)) if text_values else None,
        "per_pair": rows,
    }
    return payload, pd.DataFrame(rows)


def pair_trend_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col_a: str,
    col_b: str,
    type_a: str,
    type_b: str,
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
) -> tuple[float | None, str]:
    a_real = feature_for_pair(real[col_a], type_a, cfg_a)
    b_real = feature_for_pair(real[col_b], type_b, cfg_b)
    a_syn = feature_for_pair(synthetic[col_a], type_a, cfg_a)
    b_syn = feature_for_pair(synthetic[col_b], type_b, cfg_b)
    if is_categorical_type(type_a) and is_categorical_type(type_b):
        return contingency_tvd(
            categorical_for_trend(real[col_a], type_a, cfg_a),
            categorical_for_trend(real[col_b], type_b, cfg_b),
            categorical_for_trend(synthetic[col_a], type_a, cfg_a),
            categorical_for_trend(synthetic[col_b], type_b, cfg_b),
        ), "contingency_total_variation"
    corr_real = safe_corr(a_real, b_real)
    corr_syn = safe_corr(a_syn, b_syn)
    if corr_real is None or corr_syn is None:
        if corr_real is None and corr_syn is None:
            return 0.0, "correlation_unavailable_both"
        return 1.0, "correlation_unavailable_one_side"
    return float(min(abs(corr_real - corr_syn) / 2.0, 1.0)), "absolute_correlation_difference"


def is_categorical_type(col_type: str) -> bool:
    return col_type in {"categorical", "foreign_key"}


def feature_for_pair(series: pd.Series, col_type: str, cfg: dict[str, Any] | None = None) -> pd.Series:
    if col_type in {"numerical", "numeric", "number"}:
        return numeric_series(series)
    if col_type == "datetime":
        return datetime_normalized(series, series)
    if col_type == "text":
        return token_lengths(series)
    values = categorical_for_trend(series, col_type, cfg)
    codes, _ = pd.factorize(values.astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def categorical_for_trend(series: pd.Series, col_type: str, cfg: dict[str, Any] | None = None) -> pd.Series:
    if col_type == "categorical":
        return canonicalize_categorical_series(series, cfg or {})
    return series.astype(str)


def contingency_tvd(real_a: pd.Series, real_b: pd.Series, syn_a: pd.Series, syn_b: pd.Series) -> float:
    real_pairs = real_a.astype(str) + "\u241f" + real_b.astype(str)
    syn_pairs = syn_a.astype(str) + "\u241f" + syn_b.astype(str)
    return total_variation(real_pairs, syn_pairs)


def primary_key_columns(table_config: dict[str, Any]) -> set[str]:
    pk = table_config.get("primary_key")
    if pk is None:
        return set()
    return {pk} if isinstance(pk, str) else set(pk)


def pair_in_headline(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col_a: str,
    col_b: str,
    type_a: str,
    type_b: str,
    pk_cols: set[str],
    high_cardinality_threshold: int,
) -> bool:
    if col_a in pk_cols or col_b in pk_cols:
        return False
    if "foreign_key" in {type_a, type_b}:
        return False
    if is_high_cardinality_id_like(real, synthetic, col_a, type_a, high_cardinality_threshold):
        return False
    if is_high_cardinality_id_like(real, synthetic, col_b, type_b, high_cardinality_threshold):
        return False
    return True


def is_high_cardinality_id_like(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    column: str,
    col_type: str,
    threshold: int,
) -> bool:
    if col_type != "categorical":
        return False
    unique_count = max(real[column].nunique(dropna=True), synthetic[column].nunique(dropna=True))
    return int(unique_count) > int(threshold)
