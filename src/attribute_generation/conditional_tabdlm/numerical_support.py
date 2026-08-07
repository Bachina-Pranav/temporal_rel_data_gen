"""Data-derived numerical support diagnostics and post-hoc projection."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


PROJECTION_MODES = (
    "none",
    "global_nearest",
    "global_stochastic",
    "entity_nearest",
    "learned_bins",
)


def numerical_support_profile(
    train: pd.Series,
    test: pd.Series,
    synthetic: pd.Series,
) -> dict[str, Any]:
    train_values = finite_array(train)
    test_values = finite_array(test)
    synthetic_values = finite_array(synthetic)
    support, counts = np.unique(train_values, return_counts=True)
    tolerance = infer_support_tolerance(support)
    test_distance = nearest_support_distances(test_values, support)
    synthetic_distance = nearest_support_distances(synthetic_values, support)
    return {
        "train": split_support_summary(train_values, support, tolerance),
        "test": split_support_summary(test_values, support, tolerance),
        "synthetic": split_support_summary(
            synthetic_values,
            support,
            tolerance,
        ),
        "training_support": {
            "size": int(len(support)),
            "tolerance": float(tolerance),
            "entropy_bits": empirical_entropy(counts),
            "normalized_entropy": normalized_entropy(counts),
            "spacing": spacing_summary(support),
            "decimal_precision_distribution": decimal_precision_distribution(
                support,
                weights=counts,
            ),
            "repeated_frequency": repeated_frequency_summary(counts),
            "most_frequent_values": most_frequent_values(support, counts),
            "inferred_support_kind": infer_support_kind(
                len(train_values),
                support,
                counts,
            ),
        },
        "test_nearest_training_support": distance_summary(test_distance),
        "synthetic_nearest_training_support": distance_summary(
            synthetic_distance
        ),
        "nearest_distance_histogram": shared_distance_histogram(
            test_distance,
            synthetic_distance,
        ),
    }


def project_numerical_support(
    train_values: pd.Series,
    generated_values: pd.Series,
    *,
    mode: str,
    seed: int,
    train_entities: pd.Series | None = None,
    query_entities: pd.Series | None = None,
    stochastic_neighbors: int = 8,
    stochastic_temperature: float = 1.0,
    min_entity_rows: int = 5,
    max_learned_bins: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode).strip().lower()
    if mode not in PROJECTION_MODES:
        raise ValueError(
            f"Unknown numerical support projection mode {mode!r}; "
            f"expected one of {PROJECTION_MODES}"
        )
    train = finite_array(train_values)
    generated = numeric_array_with_nan(generated_values)
    support, counts = np.unique(train, return_counts=True)
    if not len(support):
        raise ValueError("Cannot project numerical values without training support")
    if mode == "none":
        return generated.copy(), {
            "mode": mode,
            "rows_projected": 0,
            "support_size": int(len(support)),
        }
    finite = np.isfinite(generated)
    output = generated.copy()
    metadata: dict[str, Any] = {
        "mode": mode,
        "support_size": int(len(support)),
        "rows_requested": int(len(generated)),
        "rows_finite": int(finite.sum()),
        "seed": int(seed),
    }
    if mode == "global_nearest":
        output[finite] = nearest_support_values(generated[finite], support)
    elif mode == "global_stochastic":
        output[finite] = stochastic_support_values(
            generated[finite],
            support,
            counts,
            seed=seed,
            neighbors=stochastic_neighbors,
            temperature=stochastic_temperature,
        )
        metadata.update(
            {
                "neighbors": int(stochastic_neighbors),
                "temperature": float(stochastic_temperature),
            }
        )
    elif mode == "entity_nearest":
        if train_entities is None or query_entities is None:
            raise ValueError(
                "entity_nearest projection requires training and query entities"
            )
        output, entity_metadata = entity_conditioned_projection(
            train_values,
            train_entities,
            generated,
            query_entities,
            global_support=support,
            min_entity_rows=min_entity_rows,
        )
        metadata.update(entity_metadata)
    elif mode == "learned_bins":
        representatives = learned_support_representatives(
            train,
            max_bins=max_learned_bins,
        )
        output[finite] = nearest_support_values(
            generated[finite],
            representatives,
        )
        metadata.update(
            {
                "num_learned_bins": int(len(representatives)),
                "max_learned_bins": int(max_learned_bins),
                "representatives_are_observed_training_values": True,
            }
        )
    metadata["rows_projected"] = int(
        np.sum(finite & ~np.isclose(output, generated, rtol=0.0, atol=0.0))
    )
    metadata["output_exact_training_support_rate"] = float(
        np.mean(np.isin(output[finite], support))
    )
    return output, metadata


def entity_conditioned_projection(
    train_values: pd.Series,
    train_entities: pd.Series,
    generated_values: np.ndarray,
    query_entities: pd.Series,
    *,
    global_support: np.ndarray,
    min_entity_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_frame = pd.DataFrame(
        {
            "entity": train_entities.astype(str).to_numpy(),
            "value": pd.to_numeric(train_values, errors="coerce").to_numpy(),
        }
    ).dropna(subset=["value"])
    query = query_entities.astype(str).to_numpy()
    if len(query) != len(generated_values):
        raise ValueError(
            "Query entity count does not match generated numerical value count"
        )
    counts = train_frame["entity"].value_counts()
    eligible = set(counts[counts >= int(min_entity_rows)].index)
    entity_support = {
        str(entity): np.sort(group["value"].unique().astype(float))
        for entity, group in train_frame.groupby("entity", sort=False)
        if str(entity) in eligible
    }
    entity_bucket = frequency_bucket_mapping(counts)
    train_frame["frequency_bucket"] = train_frame["entity"].map(entity_bucket)
    bucket_support = {
        str(bucket): np.sort(group["value"].unique().astype(float))
        for bucket, group in train_frame.groupby(
            "frequency_bucket",
            dropna=True,
            sort=False,
        )
    }
    output = generated_values.copy()
    finite = np.isfinite(generated_values)
    source_counts = {
        "entity": 0,
        "entity_frequency_bucket": 0,
        "global": 0,
    }
    query_groups = pd.DataFrame(
        {
            "entity": query,
            "position": np.arange(len(query), dtype=np.int64),
        }
    ).groupby("entity", sort=False)["position"]
    for entity, position_series in query_groups:
        positions = position_series.to_numpy(dtype=np.int64)
        positions = positions[finite[positions]]
        if not len(positions):
            continue
        if entity in entity_support and len(entity_support[entity]):
            chosen_support = entity_support[entity]
            source = "entity"
        else:
            bucket = entity_bucket.get(entity)
            if bucket is not None and str(bucket) in bucket_support:
                chosen_support = bucket_support[str(bucket)]
                source = "entity_frequency_bucket"
            else:
                chosen_support = global_support
                source = "global"
        output[positions] = nearest_support_values(
            generated_values[positions],
            chosen_support,
        )
        source_counts[source] += int(len(positions))
    return output, {
        "min_entity_rows": int(min_entity_rows),
        "eligible_entities": int(len(entity_support)),
        "frequency_buckets": int(len(bucket_support)),
        "fallback_source_rows": source_counts,
        "fallback_hierarchy": [
            "entity",
            "entity_frequency_bucket",
            "global",
        ],
    }


def learned_support_representatives(
    train_values: np.ndarray,
    *,
    max_bins: int,
) -> np.ndarray:
    values = np.sort(np.asarray(train_values, dtype=float))
    support = np.unique(values)
    if len(support) <= int(max_bins):
        return support
    num_bins = min(
        int(max_bins),
        max(2, int(round(math.sqrt(len(support))))),
    )
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    boundaries = np.quantile(values, quantiles)
    representatives: list[float] = []
    for index in range(num_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == num_bins - 1:
            members = values[(values >= lower) & (values <= upper)]
        else:
            members = values[(values >= lower) & (values < upper)]
        if not len(members):
            continue
        median = float(np.median(members))
        representatives.append(
            float(nearest_support_values(np.asarray([median]), support)[0])
        )
    return np.unique(np.asarray(representatives, dtype=float))


def stochastic_support_values(
    values: np.ndarray,
    support: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
    neighbors: int,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(support) == 1:
        return np.full(len(values), support[0], dtype=float)
    neighbors = max(1, min(int(neighbors), len(support)))
    temperature = max(float(temperature), 1e-8)
    spacing = positive_spacings(support)
    scale = (
        float(np.median(spacing))
        if len(spacing)
        else max(float(np.std(support)), 1e-8)
    )
    scale = max(scale, infer_support_tolerance(support), 1e-12)
    rng = np.random.default_rng(int(seed))
    output = np.empty(len(values), dtype=float)
    insertion = np.searchsorted(support, values)
    radius = neighbors + 1
    for row, (value, center) in enumerate(zip(values, insertion)):
        start = max(0, int(center) - radius)
        end = min(len(support), int(center) + radius)
        candidates = np.arange(start, end)
        if len(candidates) > neighbors:
            distances = np.abs(support[candidates] - value)
            candidates = candidates[
                np.argsort(distances, kind="mergesort")[:neighbors]
            ]
        distances = np.abs(support[candidates] - value)
        logits = -distances / (temperature * scale)
        logits += 0.25 * np.log(np.maximum(counts[candidates], 1.0))
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        output[row] = support[
            int(rng.choice(candidates, p=probabilities))
        ]
    return output


def nearest_support_values(
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=float)
    if not len(support):
        raise ValueError("Numerical support cannot be empty")
    insertion = np.searchsorted(support, values)
    right = np.clip(insertion, 0, len(support) - 1)
    left = np.clip(insertion - 1, 0, len(support) - 1)
    left_distance = np.abs(values - support[left])
    right_distance = np.abs(values - support[right])
    choose_right = right_distance < left_distance
    return support[np.where(choose_right, right, left)]


def nearest_support_distances(
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.asarray([], dtype=float)
    nearest = nearest_support_values(values, support)
    return np.abs(values - nearest)


def infer_support_tolerance(support: np.ndarray) -> float:
    support = np.asarray(support, dtype=float)
    if not len(support):
        return 0.0
    scale = max(float(np.max(np.abs(support))), 1.0)
    machine = np.finfo(np.float64).eps * scale * 32.0
    spacing = positive_spacings(support)
    if not len(spacing):
        return machine
    return float(min(np.min(spacing) * 1e-6, np.median(spacing) * 1e-4) + machine)


def infer_support_kind(
    num_rows: int,
    support: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any]:
    from .numerical_type import infer_numerical_column_type

    support = np.asarray(support, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    observed_rows = int(counts.sum())
    if observed_rows <= 200_000:
        reconstructed = np.repeat(support, counts)
    else:
        rng = np.random.default_rng(42)
        reconstructed = rng.choice(
            support,
            size=200_000,
            replace=True,
            p=counts.astype(float) / observed_rows,
        )
    if int(num_rows) != observed_rows:
        num_rows = observed_rows
    report = infer_numerical_column_type(reconstructed, seed=42)
    report.update(
        {
            "quantized": report["label"] != "continuous",
            "unique_ratio": float(
                len(support) / max(int(num_rows), 1)
            ),
            "repeated_observation_rate": float(
                counts[counts > 1].sum() / max(counts.sum(), 1)
            ),
        }
    )
    return report


def split_support_summary(
    values: np.ndarray,
    train_support: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    distances = nearest_support_distances(values, train_support)
    return {
        "rows": int(len(values)),
        "unique_values": int(len(np.unique(values))),
        "unique_value_ratio": float(
            len(np.unique(values)) / max(len(values), 1)
        ),
        "exact_training_support_overlap_rate": float(
            np.mean(np.isin(values, train_support))
        )
        if len(values)
        else None,
        "tolerant_training_support_overlap_rate": float(
            np.mean(distances <= float(tolerance))
        )
        if len(values)
        else None,
        "decimal_precision_distribution": decimal_precision_distribution(
            values
        ),
    }


def distance_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "zero_rate": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "zero_rate": float(np.mean(values == 0.0)),
    }


def shared_distance_histogram(
    real_distances: np.ndarray,
    synthetic_distances: np.ndarray,
) -> dict[str, Any]:
    combined = np.concatenate(
        [
            np.asarray(real_distances, dtype=float),
            np.asarray(synthetic_distances, dtype=float),
        ]
    )
    positive = combined[combined > 0]
    if not len(positive):
        edges = np.asarray([0.0, 1.0], dtype=float)
    else:
        low = max(float(np.min(positive)), np.finfo(float).tiny)
        high = max(float(np.max(positive)), low * 10.0)
        edges = np.unique(
            np.concatenate(
                [
                    np.asarray([0.0]),
                    np.geomspace(low, high, num=12),
                    np.asarray([np.nextafter(high, float("inf"))]),
                ]
            )
        )
    return {
        "edges": edges.tolist(),
        "test_counts": np.histogram(real_distances, bins=edges)[0].tolist(),
        "synthetic_counts": np.histogram(
            synthetic_distances,
            bins=edges,
        )[0].tolist(),
    }


def spacing_summary(support: np.ndarray) -> dict[str, Any]:
    spacing = positive_spacings(support)
    if not len(spacing):
        return {
            "num_positive_gaps": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "num_positive_gaps": int(len(spacing)),
        "min": float(np.min(spacing)),
        "median": float(np.median(spacing)),
        "mean": float(np.mean(spacing)),
        "p95": float(np.quantile(spacing, 0.95)),
        "max": float(np.max(spacing)),
    }


def positive_spacings(support: np.ndarray) -> np.ndarray:
    support = np.unique(np.asarray(support, dtype=float))
    if len(support) < 2:
        return np.asarray([], dtype=float)
    return np.diff(support)[np.diff(support) > 0]


def decimal_precision_distribution(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, int]:
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(len(values), dtype=int)
    result: dict[str, int] = {}
    for value, weight in zip(values, weights):
        precision = decimal_precision(value)
        key = str(precision)
        result[key] = result.get(key, 0) + int(weight)
    return dict(sorted(result.items(), key=lambda pair: int(pair[0])))


def decimal_precision(value: float) -> int:
    decimal = Decimal(format(float(value), ".15g")).normalize()
    return max(0, -int(decimal.as_tuple().exponent))


def repeated_frequency_summary(counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=float)
    frequencies, support_value_counts = np.unique(
        counts.astype(np.int64),
        return_counts=True,
    )
    return {
        "support_values_seen_once": int(np.sum(counts == 1)),
        "support_values_repeated": int(np.sum(counts > 1)),
        "observations_on_repeated_values": int(counts[counts > 1].sum()),
        "count_p50": float(np.quantile(counts, 0.50)),
        "count_p95": float(np.quantile(counts, 0.95)),
        "count_max": int(np.max(counts)) if len(counts) else 0,
        "frequency_of_frequencies": {
            str(int(frequency)): int(num_support_values)
            for frequency, num_support_values in zip(
                frequencies,
                support_value_counts,
            )
        },
    }


def most_frequent_values(
    support: np.ndarray,
    counts: np.ndarray,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    order = np.argsort(-counts, kind="mergesort")[: int(limit)]
    total = max(int(np.sum(counts)), 1)
    return [
        {
            "value": float(support[index]),
            "count": int(counts[index]),
            "rate": float(counts[index] / total),
        }
        for index in order
    ]


def empirical_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    probabilities = counts / max(float(counts.sum()), 1.0)
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log2(positive)))


def normalized_entropy(counts: np.ndarray) -> float:
    if len(counts) <= 1:
        return 0.0
    return float(empirical_entropy(counts) / math.log2(len(counts)))


def frequency_bucket_mapping(counts: pd.Series) -> dict[str, str]:
    if not len(counts):
        return {}
    values = counts.astype(float)
    try:
        labels = pd.qcut(
            values.rank(method="first"),
            q=min(4, len(values)),
            labels=False,
            duplicates="drop",
        )
        return {
            str(entity): f"frequency_q{int(bucket) + 1}"
            for entity, bucket in labels.items()
        }
    except ValueError:
        return {str(entity): "frequency_positive" for entity in counts.index}


def finite_array(series: pd.Series | np.ndarray) -> np.ndarray:
    values = numeric_array_with_nan(series)
    return values[np.isfinite(values)]


def numeric_array_with_nan(
    series: pd.Series | np.ndarray,
) -> np.ndarray:
    return pd.to_numeric(
        pd.Series(series),
        errors="coerce",
    ).to_numpy(dtype=float)
