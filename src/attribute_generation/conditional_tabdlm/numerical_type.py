"""Training-only, multi-signal numerical-column type inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .numerical_support import (
    decimal_precision_distribution,
    empirical_entropy,
    normalized_entropy,
    positive_spacings,
)


NUMERICAL_TYPES = (
    "continuous",
    "repeated_or_quantized",
    "low_cardinality_discrete_numerical",
    "high_cardinality_structured_support",
)


@dataclass(frozen=True)
class NumericalTypeThresholds:
    """Configurable evidence thresholds for numerical type inference."""

    low_cardinality_max_values: int = 64
    low_cardinality_max_unique_ratio: float = 0.05
    direct_support_max_values: int = 8192
    hierarchical_support_max_values: int = 200_000
    repeated_max_unique_ratio: float = 0.25
    repeated_min_observation_mass: float = 0.50
    holdout_min_support_overlap: float = 0.75
    precision_min_dominant_mass: float = 0.70
    spacing_max_coefficient_of_variation: float = 2.0
    top_values_min_observation_mass: float = 0.10
    top_values_count: int = 32
    minimum_structured_signals: int = 3
    holdout_fraction: float = 0.20

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
    ) -> "NumericalTypeThresholds":
        raw = dict(config or {})
        allowed = {
            field
            for field in cls.__dataclass_fields__
        }
        return cls(
            **{
                key: raw[key]
                for key in allowed
                if key in raw
            }
        )


def infer_numerical_column_type(
    values: pd.Series | np.ndarray,
    *,
    thresholds: NumericalTypeThresholds | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Classify a numerical target from training values only."""

    thresholds = thresholds or NumericalTypeThresholds()
    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()
    array = numeric.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError(
            "Cannot infer numerical type from an empty finite training column"
        )
    support, counts = np.unique(array, return_counts=True)
    unique_ratio = float(len(support) / len(array))
    repeated_mass = float(
        counts[counts > 1].sum() / max(counts.sum(), 1)
    )
    holdout_overlap = deterministic_holdout_support_overlap(
        array,
        fraction=thresholds.holdout_fraction,
        seed=seed,
    )
    spacing = positive_spacings(support)
    spacing_cv = (
        float(np.std(spacing) / max(np.mean(spacing), 1e-12))
        if len(spacing)
        else 0.0
    )
    precision_counts = decimal_precision_distribution(
        support,
        weights=counts,
    )
    dominant_precision_mass = (
        float(max(precision_counts.values()) / len(array))
        if precision_counts
        else 0.0
    )
    top_count = min(
        int(thresholds.top_values_count),
        len(counts),
    )
    top_mass = float(
        np.sort(counts)[-top_count:].sum() / len(array)
    )
    signals = {
        "low_unique_ratio": (
            unique_ratio
            <= float(thresholds.repeated_max_unique_ratio)
        ),
        "high_repeated_observation_mass": (
            repeated_mass
            >= float(thresholds.repeated_min_observation_mass)
        ),
        "high_holdout_support_overlap": (
            holdout_overlap
            >= float(thresholds.holdout_min_support_overlap)
        ),
        "dominant_decimal_precision": (
            dominant_precision_mass
            >= float(thresholds.precision_min_dominant_mass)
        ),
        "regular_support_spacing": (
            spacing_cv
            <= float(
                thresholds.spacing_max_coefficient_of_variation
            )
        ),
        "common_values_cover_material_mass": (
            top_mass
            >= float(thresholds.top_values_min_observation_mass)
        ),
    }
    structured_votes = int(sum(bool(value) for value in signals.values()))
    low_cardinality = (
        len(support)
        <= int(thresholds.low_cardinality_max_values)
        and (
            unique_ratio
            <= float(thresholds.low_cardinality_max_unique_ratio)
            or repeated_mass
            >= float(thresholds.repeated_min_observation_mass)
        )
    )
    structured = (
        structured_votes
        >= int(thresholds.minimum_structured_signals)
        and len(support)
        <= int(thresholds.hierarchical_support_max_values)
    )
    if low_cardinality:
        label = "low_cardinality_discrete_numerical"
        recommended_head = "discrete_support"
    elif (
        structured
        and len(support)
        <= int(thresholds.direct_support_max_values)
    ):
        label = "repeated_or_quantized"
        recommended_head = "discrete_support"
    elif structured:
        label = "high_cardinality_structured_support"
        recommended_head = "hierarchical_support"
    else:
        label = "continuous"
        recommended_head = "continuous_baseline"
    return {
        "label": label,
        "recommended_head": recommended_head,
        "training_only": True,
        "num_rows": int(len(array)),
        "support_size": int(len(support)),
        "unique_value_ratio": unique_ratio,
        "repeated_observation_mass": repeated_mass,
        "holdout_support_overlap": holdout_overlap,
        "support_entropy_bits": empirical_entropy(counts),
        "normalized_support_entropy": normalized_entropy(counts),
        "spacing_coefficient_of_variation": spacing_cv,
        "dominant_decimal_precision_mass": dominant_precision_mass,
        "decimal_precision_distribution": precision_counts,
        "top_values_observation_mass": top_mass,
        "structured_signal_count": structured_votes,
        "structured_signals": signals,
        "thresholds": asdict(thresholds),
    }


def infer_numerical_types(
    train: pd.DataFrame,
    numerical_columns: list[str] | tuple[str, ...],
    *,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    thresholds = NumericalTypeThresholds.from_config(config)
    return {
        str(column): infer_numerical_column_type(
            train[column],
            thresholds=thresholds,
            seed=int(seed) + index,
        )
        for index, column in enumerate(numerical_columns)
    }


def deterministic_holdout_support_overlap(
    values: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> float:
    """Estimate support recurrence using a deterministic training-only split."""

    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 1.0
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(values))
    holdout_size = min(
        max(int(math.ceil(len(values) * float(fraction))), 1),
        len(values) - 1,
    )
    holdout = values[order[:holdout_size]]
    fit = values[order[holdout_size:]]
    return float(np.mean(np.isin(holdout, np.unique(fit))))
