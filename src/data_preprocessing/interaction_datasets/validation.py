"""Validation checks for induced interaction subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import InteractionDatasetAdapter


def validate_subset(
    adapter: InteractionDatasetAdapter,
    subset_dir: str | Path,
    *,
    raw_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    subset_dir = Path(subset_dir)
    interactions = pd.read_csv(subset_dir / "interactions.csv")
    source = pd.read_csv(subset_dir / adapter.source_table_filename)
    destination = pd.read_csv(subset_dir / adapter.destination_table_filename)
    errors: list[str] = []
    warnings: list[str] = []
    if interactions.empty:
        errors.append("interactions.csv is empty")
    if interactions[adapter.event_id_column].astype(str).duplicated().any():
        errors.append("event_id values are not unique")
    timestamps = pd.to_datetime(interactions[adapter.timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        errors.append(f"{adapter.timestamp_column} contains unparsable timestamps")
    source_values = set(source[adapter.source_id_column].astype(str))
    dest_values = set(destination[adapter.destination_id_column].astype(str))
    child_sources = set(interactions[adapter.source_id_column].astype(str))
    child_dests = set(interactions[adapter.destination_id_column].astype(str))
    source_valid = child_sources.issubset(source_values)
    dest_valid = child_dests.issubset(dest_values)
    if not source_valid:
        errors.append("source foreign-key coverage is incomplete")
    if not dest_valid:
        errors.append("destination foreign-key coverage is incomplete")
    split_counts = interactions["split"].astype(str).value_counts().to_dict() if "split" in interactions else {}
    if set(split_counts) != {"train", "validation", "test"}:
        errors.append(f"split labels must be train/validation/test, got {sorted(split_counts)}")
    complete = True
    if raw_counts is not None:
        subset_counts = interactions.assign(
            **{adapter.source_id_column: interactions[adapter.source_id_column].astype(str)}
        ).groupby(adapter.source_id_column).size()
        for source_id, raw_count in raw_counts.items():
            got = int(subset_counts.get(str(source_id), 0))
            if got != int(raw_count):
                complete = False
                errors.append(f"incomplete source history for {source_id}: raw={raw_count}, subset={got}")
                break
    for column in adapter.generated_attributes:
        if column not in interactions.columns:
            errors.append(f"missing generated attribute column {column}")
    validate_dataset_specific(adapter, interactions, errors, warnings)
    return {
        "dataset_name": adapter.benchmark_name,
        "num_rows": int(len(interactions)),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
        "complete_source_histories": bool(complete),
        "foreign_key_valid": bool(source_valid and dest_valid),
        "source_fk_valid_rate": float(interactions[adapter.source_id_column].astype(str).isin(source_values).mean()) if len(interactions) else 0.0,
        "destination_fk_valid_rate": float(interactions[adapter.destination_id_column].astype(str).isin(dest_values).mean()) if len(interactions) else 0.0,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "timestamp_min": timestamps.min().isoformat() if len(timestamps) else None,
        "timestamp_max": timestamps.max().isoformat() if len(timestamps) else None,
    }


def validate_dataset_specific(adapter: InteractionDatasetAdapter, interactions: pd.DataFrame, errors: list[str], warnings: list[str]) -> None:
    name = adapter.dataset_name
    if name == "movielens":
        ratings = set(interactions["rating"].astype(str))
        expected = {"0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"}
        if not ratings.issubset(expected):
            errors.append(f"unexpected MovieLens rating values: {sorted(ratings - expected)}")
    elif name == "yelp":
        if not interactions["stars"].astype(float).between(1, 5).all():
            errors.append("Yelp stars must be in [1, 5]")
        for column in ["useful", "funny", "cool"]:
            values = pd.to_numeric(interactions[column], errors="coerce")
            if values.isna().any() or (values < 0).any() or not np.allclose(values, np.round(values)):
                errors.append(f"Yelp {column} must be nonnegative integers")
        if interactions["review_text"].fillna("").astype(str).str.len().eq(0).any():
            warnings.append("Yelp subset contains empty review_text values")
    elif name == "retailrocket":
        values = set(interactions["event_type"].astype(str))
        expected = {"view", "addtocart", "transaction"}
        if not values.issubset(expected):
            errors.append(f"unexpected RetailRocket event types: {sorted(values - expected)}")
    elif name == "hm":
        price = pd.to_numeric(interactions["price"], errors="coerce")
        if price.isna().any() or (price < 0).any() or np.isinf(price).any():
            errors.append("H&M price must be finite and nonnegative")
