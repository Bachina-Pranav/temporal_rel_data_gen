"""Training-only rank calibration for generated numerical attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .numerical_support import (
    infer_support_kind,
    nearest_support_values,
)


CALIBRATION_MODES = (
    "original",
    "global",
    "time_bucket",
    "destination_frequency_bucket",
    "destination_hierarchy",
)


@dataclass(frozen=True)
class CalibrationOptions:
    """Configuration for rank-preserving numerical calibration."""

    min_destination_rows: int = 20
    min_bucket_rows: int = 100
    num_time_buckets: int = 8
    num_frequency_buckets: int = 4
    project_to_training_support: bool | None = None


def calibrate_numerical_column(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    value_column: str,
    destination_column: str,
    timestamp_column: str,
    mode: str,
    options: CalibrationOptions | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate generated ranks against training-only target distributions."""

    options = options or CalibrationOptions()
    mode = str(mode).strip().lower()
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"Unknown numerical calibration mode {mode!r}; "
            f"expected one of {CALIBRATION_MODES}"
        )
    required_train = {
        value_column,
        destination_column,
        timestamp_column,
    }
    required_query = required_train
    missing_train = sorted(required_train.difference(train.columns))
    missing_query = sorted(required_query.difference(synthetic.columns))
    if missing_train or missing_query:
        raise ValueError(
            "Calibration inputs are missing required columns: "
            f"train={missing_train}, synthetic={missing_query}"
        )

    train_values = pd.to_numeric(
        train[value_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    generated_values = pd.to_numeric(
        synthetic[value_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    finite_train = train_values[np.isfinite(train_values)]
    if not len(finite_train):
        raise ValueError(
            f"Cannot calibrate {value_column!r} without finite training values"
        )
    output = generated_values.copy()
    source_counts: dict[str, int] = {}

    time_model = fit_time_bucket_model(
        train[timestamp_column],
        num_buckets=options.num_time_buckets,
    )
    train_time = apply_time_bucket_model(
        train[timestamp_column],
        time_model,
    )
    query_time = apply_time_bucket_model(
        synthetic[timestamp_column],
        time_model,
    )
    destination_counts = (
        train[destination_column].astype(str).value_counts()
    )
    frequency_mapping = fit_frequency_bucket_mapping(
        destination_counts,
        num_buckets=options.num_frequency_buckets,
    )
    train_destination = train[destination_column].astype(str)
    query_destination = synthetic[destination_column].astype(str)
    train_frequency = (
        train_destination.map(frequency_mapping).fillna("frequency_cold")
    )
    query_frequency = (
        query_destination.map(frequency_mapping).fillna("frequency_cold")
    )

    if mode == "original":
        source_counts["original"] = int(np.isfinite(output).sum())
    elif mode == "global":
        output = empirical_rank_map(generated_values, finite_train)
        source_counts["global"] = int(np.isfinite(output).sum())
    elif mode == "time_bucket":
        output, source_counts = calibrate_by_groups(
            generated_values,
            query_time,
            train_values,
            train_time,
            min_reference_rows=options.min_bucket_rows,
            global_reference=finite_train,
            source_prefix="time",
        )
    elif mode == "destination_frequency_bucket":
        output, source_counts = calibrate_by_groups(
            generated_values,
            query_frequency,
            train_values,
            train_frequency,
            min_reference_rows=options.min_bucket_rows,
            global_reference=finite_train,
            source_prefix="destination_frequency",
        )
    else:
        output, source_counts = calibrate_destination_hierarchy(
            generated_values=generated_values,
            query_destinations=query_destination,
            query_frequency=query_frequency,
            query_time=query_time,
            train_values=train_values,
            train_destinations=train_destination,
            train_frequency=train_frequency,
            train_time=train_time,
            min_destination_rows=options.min_destination_rows,
            min_bucket_rows=options.min_bucket_rows,
            global_reference=finite_train,
        )

    support, counts = np.unique(finite_train, return_counts=True)
    inferred = infer_support_kind(
        len(finite_train),
        support,
        counts,
    )
    project_support = options.project_to_training_support
    if project_support is None:
        project_support = inferred["label"] != "continuous"
    finite_output = np.isfinite(output)
    if project_support and finite_output.any():
        output[finite_output] = nearest_support_values(
            output[finite_output],
            support,
        )
    return output, {
        "mode": mode,
        "value_column": value_column,
        "destination_column": destination_column,
        "timestamp_column": timestamp_column,
        "mapping_fit_scope": "training_split_only",
        "generated_values_used_only_for_percentile_ranks": True,
        "num_training_rows": int(len(train)),
        "num_query_rows": int(len(synthetic)),
        "training_support_size": int(len(support)),
        "project_to_training_support": bool(project_support),
        "inferred_support_kind": inferred,
        "source_counts": source_counts,
        "time_bucket_model": time_model,
        "frequency_bucket_count": int(
            len(set(frequency_mapping.values()))
        ),
        "min_destination_rows": int(options.min_destination_rows),
        "min_bucket_rows": int(options.min_bucket_rows),
    }


def empirical_rank_map(
    generated_values: np.ndarray | pd.Series,
    reference_values: np.ndarray | pd.Series,
) -> np.ndarray:
    """Map generated percentiles to an empirical reference distribution."""

    generated = pd.to_numeric(
        pd.Series(generated_values),
        errors="coerce",
    ).to_numpy(dtype=float)
    reference = pd.to_numeric(
        pd.Series(reference_values),
        errors="coerce",
    ).to_numpy(dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    output = generated.copy()
    finite = np.isfinite(generated)
    if not finite.any() or not len(reference):
        return output
    ranks = (
        pd.Series(generated[finite])
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    quantiles = (ranks - 0.5) / max(len(ranks), 1)
    output[finite] = linear_empirical_quantiles(
        reference,
        np.clip(quantiles, 0.0, 1.0),
    )
    return output


def linear_empirical_quantiles(
    sorted_reference: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """NumPy-version-independent linear empirical quantiles."""

    reference = np.sort(np.asarray(sorted_reference, dtype=float))
    quantiles = np.asarray(quantiles, dtype=float)
    if len(reference) == 1:
        return np.full_like(quantiles, reference[0], dtype=float)
    positions = quantiles * float(len(reference) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = positions - lower
    return (
        reference[lower] * (1.0 - weight)
        + reference[upper] * weight
    )


def calibrate_by_groups(
    generated_values: np.ndarray,
    query_groups: pd.Series,
    train_values: np.ndarray,
    train_groups: pd.Series,
    *,
    min_reference_rows: int,
    global_reference: np.ndarray,
    source_prefix: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply rank calibration per group with a global fallback."""

    references = grouped_references(
        train_values,
        train_groups,
        min_rows=min_reference_rows,
    )
    output = np.asarray(generated_values, dtype=float).copy()
    source_counts: dict[str, int] = {}
    positions = pd.Series(
        np.arange(len(output), dtype=np.int64),
        index=query_groups.index,
    )
    for group, group_positions in positions.groupby(
        query_groups.astype(str),
        sort=False,
    ):
        index = group_positions.to_numpy(dtype=np.int64)
        reference = references.get(str(group))
        source = f"{source_prefix}:{group}"
        if reference is None:
            reference = global_reference
            source = "global"
        output[index] = empirical_rank_map(
            output[index],
            reference,
        )
        source_counts[source] = source_counts.get(source, 0) + int(len(index))
    return output, source_counts


def calibrate_destination_hierarchy(
    *,
    generated_values: np.ndarray,
    query_destinations: pd.Series,
    query_frequency: pd.Series,
    query_time: pd.Series,
    train_values: np.ndarray,
    train_destinations: pd.Series,
    train_frequency: pd.Series,
    train_time: pd.Series,
    min_destination_rows: int,
    min_bucket_rows: int,
    global_reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Destination, frequency, time, then global calibration fallback."""

    destination_references = grouped_references(
        train_values,
        train_destinations,
        min_rows=min_destination_rows,
    )
    frequency_references = grouped_references(
        train_values,
        train_frequency,
        min_rows=min_bucket_rows,
    )
    time_references = grouped_references(
        train_values,
        train_time,
        min_rows=min_bucket_rows,
    )
    source_keys: list[str] = []
    references: dict[str, np.ndarray] = {}
    for destination, frequency, time_bucket in zip(
        query_destinations.astype(str),
        query_frequency.astype(str),
        query_time.astype(str),
    ):
        if destination in destination_references:
            key = f"destination:{destination}"
            reference = destination_references[destination]
        elif frequency in frequency_references:
            key = f"destination_frequency:{frequency}"
            reference = frequency_references[frequency]
        elif time_bucket in time_references:
            key = f"time:{time_bucket}"
            reference = time_references[time_bucket]
        else:
            key = "global"
            reference = global_reference
        source_keys.append(key)
        references[key] = reference
    output = np.asarray(generated_values, dtype=float).copy()
    source_counts: dict[str, int] = {}
    source_series = pd.Series(source_keys)
    positions = pd.Series(np.arange(len(output), dtype=np.int64))
    for key, group_positions in positions.groupby(source_series, sort=False):
        index = group_positions.to_numpy(dtype=np.int64)
        output[index] = empirical_rank_map(
            output[index],
            references[str(key)],
        )
        source_counts[str(key)] = int(len(index))
    return output, source_counts


def grouped_references(
    values: np.ndarray,
    groups: pd.Series,
    *,
    min_rows: int,
) -> dict[str, np.ndarray]:
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "group": groups.astype(str).to_numpy(),
        }
    )
    frame = frame.loc[np.isfinite(frame["value"])]
    return {
        str(group): np.sort(part["value"].to_numpy(dtype=float))
        for group, part in frame.groupby("group", sort=False)
        if len(part) >= int(min_rows)
    }


def fit_time_bucket_model(
    timestamps: pd.Series,
    *,
    num_buckets: int,
) -> dict[str, Any]:
    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    finite = parsed.array.asi8[parsed.notna().to_numpy()]
    if not len(finite):
        return {
            "strategy": "training_timestamp_quantiles",
            "boundaries_ns": [],
            "num_buckets_requested": int(num_buckets),
        }
    requested = max(int(num_buckets), 1)
    boundaries = np.unique(
        np.quantile(
            finite.astype(float),
            np.linspace(0.0, 1.0, requested + 1)[1:-1],
        ).astype(np.int64)
    )
    return {
        "strategy": "training_timestamp_quantiles",
        "boundaries_ns": boundaries.tolist(),
        "num_buckets_requested": requested,
        "num_buckets_resolved": int(len(boundaries) + 1),
        "training_min": parsed.min().isoformat(),
        "training_max": parsed.max().isoformat(),
    }


def apply_time_bucket_model(
    timestamps: pd.Series,
    model: dict[str, Any],
) -> pd.Series:
    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    numeric = parsed.array.asi8.astype(float)
    boundaries = np.asarray(
        model.get("boundaries_ns", []),
        dtype=np.int64,
    )
    labels = np.full(len(parsed), "time_missing", dtype=object)
    finite = parsed.notna().to_numpy()
    labels[finite] = [
        f"time_q{bucket + 1}"
        for bucket in np.searchsorted(
            boundaries,
            numeric[finite],
            side="right",
        )
    ]
    return pd.Series(labels, index=timestamps.index)


def fit_frequency_bucket_mapping(
    counts: pd.Series,
    *,
    num_buckets: int,
) -> dict[str, str]:
    if not len(counts):
        return {}
    requested = min(max(int(num_buckets), 1), len(counts))
    try:
        buckets = pd.qcut(
            counts.astype(float).rank(method="first"),
            q=requested,
            labels=False,
            duplicates="drop",
        )
        return {
            str(entity): f"frequency_q{int(bucket) + 1}"
            for entity, bucket in buckets.items()
        }
    except ValueError:
        return {
            str(entity): "frequency_positive"
            for entity in counts.index
        }
