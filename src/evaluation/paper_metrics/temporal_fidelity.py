"""Temporal fidelity metrics for single event tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import datetime_numeric, datetime_series, ks_distance, safe_corr, wasserstein_1d


def temporal_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    temporal_cfg = ((config.get("evaluation") or {}).get("temporal") or {})
    timestamp_columns = temporal_cfg.get("timestamp_columns") or [
        column for column, cfg in ((config.get("table") or {}).get("columns") or {}).items() if str((cfg or {}).get("type")) == "datetime"
    ]
    modes = ((temporal_cfg.get("binning") or {}).get("modes") or ["adaptive"])
    target_bins = int((temporal_cfg.get("binning") or {}).get("adaptive_target_bins", 50))
    rows: list[dict[str, Any]] = []
    per_timestamp: dict[str, Any] = {}
    for column in timestamp_columns:
        real_ts = datetime_series(real[column]).dropna()
        syn_ts = datetime_series(synthetic[column]).dropna()
        per_timestamp[column] = {}
        for mode in modes:
            metric = binned_time_metric(real_ts, syn_ts, mode, target_bins)
            per_timestamp[column][mode] = metric
            rows.append({"timestamp_column": column, "mode": mode, **metric})
    entity_temporal = entity_temporal_metrics(real, synthetic, timestamp_columns, temporal_cfg)
    distances = [
        item.get("adaptive", {}).get("total_variation_distance")
        for item in per_timestamp.values()
        if item.get("adaptive", {}).get("total_variation_distance") is not None
    ]
    payload = {
        "macro_temporal_event_distance": float(np.mean(distances)) if distances else None,
        "per_timestamp": per_timestamp,
        "entity_temporal": entity_temporal,
    }
    return payload, pd.DataFrame(rows)


def binned_time_metric(real_ts: pd.Series, syn_ts: pd.Series, mode: str, target_bins: int) -> dict[str, Any]:
    if len(real_ts) == 0 or len(syn_ts) == 0:
        return empty_time_metric(0)
    if mode == "adaptive":
        bins = np.linspace(real_ts.min().value, real_ts.max().value, max(2, int(target_bins) + 1))
        real_counts, _ = np.histogram(datetime_numeric(real_ts), bins=bins)
        syn_counts, _ = np.histogram(datetime_numeric(syn_ts), bins=bins)
    else:
        freq = {"daily": "D", "weekly": "W", "monthly": "M"}.get(mode, "D")
        real_counts = real_ts.dt.to_period(freq).value_counts().sort_index()
        syn_counts = syn_ts.dt.to_period(freq).value_counts().sort_index()
        idx = real_counts.index.union(syn_counts.index)
        real_counts = real_counts.reindex(idx, fill_value=0).to_numpy()
        syn_counts = syn_counts.reindex(idx, fill_value=0).to_numpy()
    return count_distribution_metric(real_counts, syn_counts, real_ts, syn_ts)


def count_distribution_metric(real_counts: np.ndarray, syn_counts: np.ndarray, real_ts: pd.Series, syn_ts: pd.Series) -> dict[str, Any]:
    real_total = max(float(real_counts.sum()), 1.0)
    syn_total = max(float(syn_counts.sum()), 1.0)
    rp = real_counts.astype(float) / real_total
    sp = syn_counts.astype(float) / syn_total
    return {
        "total_variation_distance": float(0.5 * np.abs(rp - sp).sum()),
        "wasserstein_distance": wasserstein_1d(datetime_numeric(real_ts), datetime_numeric(syn_ts)),
        "count_correlation": safe_corr(real_counts, syn_counts),
        "count_mae_normalized": float(np.mean(np.abs(rp - sp))) if len(rp) else None,
        "num_bins": int(len(real_counts)),
    }


def empty_time_metric(num_bins: int) -> dict[str, Any]:
    return {
        "total_variation_distance": None,
        "wasserstein_distance": None,
        "count_correlation": None,
        "count_mae_normalized": None,
        "num_bins": int(num_bins),
    }


def entity_temporal_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    timestamp_columns: list[str],
    temporal_cfg: dict[str, Any],
) -> dict[str, Any]:
    entity_cfg = temporal_cfg.get("entity_inter_event", {}) or {}
    if not bool(entity_cfg.get("enabled", False)):
        return {}
    out: dict[str, Any] = {}
    for entity_col in entity_cfg.get("entity_columns", []) or []:
        out[entity_col] = {}
        for ts_col in timestamp_columns:
            out[entity_col][ts_col] = entity_timestamp_metric(real, synthetic, entity_col, ts_col)
    return out


def entity_timestamp_metric(real: pd.DataFrame, synthetic: pd.DataFrame, entity_col: str, ts_col: str) -> dict[str, Any]:
    real_stats = per_entity_time_stats(real, entity_col, ts_col)
    syn_stats = per_entity_time_stats(synthetic, entity_col, ts_col)
    common = real_stats.index.intersection(syn_stats.index)
    return {
        "inter_event_time_ks": ks_distance(real_stats["gaps"], syn_stats["gaps"]),
        "active_window_ks": ks_distance(real_stats.loc[common, "window"], syn_stats.loc[common, "window"]) if len(common) else None,
        "first_event_time_ks": ks_distance(real_stats.loc[common, "first"], syn_stats.loc[common, "first"]) if len(common) else None,
        "last_event_time_ks": ks_distance(real_stats.loc[common, "last"], syn_stats.loc[common, "last"]) if len(common) else None,
        "num_entities_compared": int(len(common)),
    }


def per_entity_time_stats(frame: pd.DataFrame, entity_col: str, ts_col: str) -> pd.DataFrame:
    tmp = frame[[entity_col, ts_col]].copy()
    tmp[ts_col] = datetime_series(tmp[ts_col])
    tmp = tmp.dropna(subset=[entity_col, ts_col]).sort_values([entity_col, ts_col])
    rows = []
    gaps_all: list[float] = []
    for entity, group in tmp.groupby(entity_col):
        values = datetime_numeric(group[ts_col]).to_numpy(dtype=float)
        gaps = np.diff(values)
        gaps_all.extend(gaps.tolist())
        rows.append(
            {
                "entity": entity,
                "first": float(values.min()) if len(values) else np.nan,
                "last": float(values.max()) if len(values) else np.nan,
                "window": float(values.max() - values.min()) if len(values) >= 2 else 0.0,
                "gaps": float(np.mean(gaps)) if len(gaps) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("entity") if rows else pd.DataFrame(columns=["first", "last", "window", "gaps"])
