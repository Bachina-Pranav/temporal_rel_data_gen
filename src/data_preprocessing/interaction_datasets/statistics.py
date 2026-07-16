"""Statistics for interaction benchmark subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import InteractionDatasetAdapter


def gini(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or float(arr.sum()) == 0.0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1) / n)


def degree_summary(series: pd.Series) -> dict[str, Any]:
    values = series.astype(float).to_numpy()
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "gini": gini(values),
    }


def compute_statistics(
    adapter: InteractionDatasetAdapter,
    interactions: pd.DataFrame,
    *,
    full_counts: pd.Series | None = None,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(interactions[adapter.timestamp_column], utc=True, errors="coerce")
    source_degree = interactions.groupby(adapter.source_id_column).size()
    dest_degree = interactions.groupby(adapter.destination_id_column).size()
    stats: dict[str, Any] = {
        "dataset_name": adapter.benchmark_name,
        "domain": adapter.domain,
        "interaction_count": int(len(interactions)),
        "source_entity_count": int(interactions[adapter.source_id_column].nunique()),
        "destination_entity_count": int(interactions[adapter.destination_id_column].nunique()),
        "time_span": {
            "min": timestamps.min().isoformat() if len(timestamps) else None,
            "max": timestamps.max().isoformat() if len(timestamps) else None,
        },
        "source_degree": degree_summary(source_degree),
        "destination_degree": degree_summary(dest_degree),
        "unique_source_destination_pairs": int(interactions[[adapter.source_id_column, adapter.destination_id_column]].drop_duplicates().shape[0]),
        "repeated_pair_rate": float(1.0 - interactions[[adapter.source_id_column, adapter.destination_id_column]].drop_duplicates().shape[0] / max(len(interactions), 1)),
        "events_per_day": {str(k.date()): int(v) for k, v in timestamps.dt.floor("D").value_counts().sort_index().items()},
        "attribute_behavior": attribute_statistics(adapter, interactions),
    }
    if full_counts is not None and not full_counts.empty:
        stats["full_source_entity_count"] = int(len(full_counts))
        stats["full_interaction_count"] = int(full_counts.sum())
    return stats


def attribute_statistics(adapter: InteractionDatasetAdapter, interactions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column, semantic_type in adapter.attribute_types.items():
        if column not in interactions:
            continue
        if semantic_type in {"categorical", "ordinal_categorical", "boolean"}:
            out[column] = {
                "type": semantic_type,
                "distribution": {str(k): int(v) for k, v in interactions[column].astype(str).value_counts().sort_index().items()},
            }
        elif semantic_type in {"continuous_numerical", "count_numerical"}:
            values = pd.to_numeric(interactions[column], errors="coerce")
            out[column] = {
                "type": semantic_type,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "zero_rate": float((values == 0).mean()) if semantic_type == "count_numerical" else None,
            }
        elif semantic_type == "text":
            lengths = interactions[column].fillna("").astype(str).str.split().map(len)
            out[column] = {
                "type": semantic_type,
                "mean_tokens": float(lengths.mean()),
                "median_tokens": float(lengths.median()),
                "p95_tokens": float(lengths.quantile(0.95)),
                "empty_rate": float(interactions[column].fillna("").astype(str).str.len().eq(0).mean()),
            }
    return out


def write_statistics_markdown(stats: dict[str, Any], path: str | Path) -> None:
    lines = [
        f"# {stats['dataset_name']} Statistics",
        "",
        f"- Interactions: {stats['interaction_count']:,}",
        f"- Source entities: {stats['source_entity_count']:,}",
        f"- Destination entities: {stats['destination_entity_count']:,}",
        f"- Time span: {stats['time_span']['min']} to {stats['time_span']['max']}",
        f"- Unique source-destination pairs: {stats['unique_source_destination_pairs']:,}",
        f"- Repeated-pair rate: {stats['repeated_pair_rate']:.6f}",
        "",
        "## Attribute Behavior",
        "",
    ]
    for column, payload in stats.get("attribute_behavior", {}).items():
        lines.append(f"### {column}")
        lines.append("")
        for key, value in payload.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
