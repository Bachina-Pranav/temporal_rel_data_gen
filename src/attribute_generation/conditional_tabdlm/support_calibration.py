"""Training-support probability calibration diagnostics and corrections."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .numerical_head import nearest_support_indices_numpy


def support_probability_table(
    train: pd.Series,
    validation: pd.Series,
    generated: pd.Series,
    *,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    train_values = finite_values(train)
    validation_values = finite_values(validation)
    generated_values = finite_values(generated)
    support, train_counts = np.unique(
        train_values,
        return_counts=True,
    )
    if not len(support):
        raise ValueError("Training numerical support is empty")
    validation_ids = nearest_support_indices_numpy(
        validation_values,
        support,
    )
    generated_ids = nearest_support_indices_numpy(
        generated_values,
        support,
    )
    validation_counts = np.bincount(
        validation_ids,
        minlength=len(support),
    )
    generated_counts = np.bincount(
        generated_ids,
        minlength=len(support),
    )
    train_probability = normalized(train_counts, epsilon=0.0)
    validation_probability = normalized(
        validation_counts,
        epsilon=0.0,
    )
    generated_probability = normalized(
        generated_counts,
        epsilon=0.0,
    )
    correction = np.log(train_probability + epsilon) - np.log(
        generated_probability + epsilon
    )
    return pd.DataFrame(
        {
            "support_value": support,
            "train_count": train_counts.astype(np.int64),
            "validation_count": validation_counts.astype(np.int64),
            "generated_count": generated_counts.astype(np.int64),
            "p_train": train_probability,
            "p_validation": validation_probability,
            "p_generated": generated_probability,
            "train_minus_generated": (
                train_probability - generated_probability
            ),
            "train_to_generated_ratio": (
                (train_probability + epsilon)
                / (generated_probability + epsilon)
            ),
            "delta_log_probability": correction,
        }
    )


def support_calibration_metrics(
    table: pd.DataFrame,
    *,
    num_frequency_buckets: int = 5,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    train = table["p_train"].to_numpy(dtype=float)
    generated = table["p_generated"].to_numpy(dtype=float)
    validation = table["p_validation"].to_numpy(dtype=float)
    support = table["support_value"].to_numpy(dtype=float)
    order = np.argsort(-train, kind="mergesort")
    entropy_train = entropy(train, epsilon)
    entropy_generated = entropy(generated, epsilon)
    median = weighted_quantile(support, train, 0.5)
    train_central_distance = float(
        np.sum(train * np.abs(support - median))
    )
    generated_central_distance = float(
        np.sum(generated * np.abs(support - median))
    )
    frequency_rank = pd.Series(train).rank(
        method="average"
    ).to_numpy(dtype=float)
    generated_rank = pd.Series(generated).rank(
        method="average"
    ).to_numpy(dtype=float)
    buckets = frequency_bucket_report(
        train,
        generated,
        num_buckets=num_frequency_buckets,
        epsilon=epsilon,
    )
    top_errors = {
        f"top_{count}_support_mass_error": top_mass_error(
            train,
            generated,
            order,
            count,
        )
        for count in (10, 50, 100)
    }
    top_width = min(100, len(train))
    tail_error = float(
        abs(
            generated[order[top_width:]].sum()
            - train[order[top_width:]].sum()
        )
    )
    rare = train <= np.quantile(train, 0.25)
    dominant = np.zeros(len(train), dtype=bool)
    dominant_width = min(
        10,
        max(1, int(np.ceil(0.10 * len(order)))),
    )
    dominant[order[:dominant_width]] = True
    flattening = entropy_generated > entropy_train + 1e-6
    concentrating = entropy_generated < entropy_train - 1e-6
    return {
        "support_size": int(len(table)),
        "total_variation_train_vs_generated": total_variation(
            train,
            generated,
        ),
        "total_variation_validation_vs_generated": total_variation(
            validation,
            generated,
        ),
        "jensen_shannon_train_vs_generated": jensen_shannon(
            train,
            generated,
            epsilon,
        ),
        "kl_train_to_generated": kl_divergence(
            train,
            generated,
            epsilon,
        ),
        "head_value_frequency_correlation": safe_correlation(
            train[order[: min(100, len(order))]],
            generated[order[: min(100, len(order))]],
        ),
        "support_value_rank_correlation": safe_correlation(
            frequency_rank,
            generated_rank,
        ),
        **top_errors,
        "tail_mass_error_after_top_100": tail_error,
        "train_entropy_nats": entropy_train,
        "generated_entropy_nats": entropy_generated,
        "entropy_difference_generated_minus_train": (
            entropy_generated - entropy_train
        ),
        "rare_support_mass_train": float(train[rare].sum()),
        "rare_support_mass_generated": float(generated[rare].sum()),
        "dominant_support_mass_train": float(train[dominant].sum()),
        "dominant_support_mass_generated": float(
            generated[dominant].sum()
        ),
        "central_absolute_distance_train": train_central_distance,
        "central_absolute_distance_generated": (
            generated_central_distance
        ),
        "calibration_by_support_frequency_bucket": buckets,
        "diagnosis": {
            "overproduces_rare_values": bool(
                generated[rare].sum() > train[rare].sum() + 0.01
            ),
            "underproduces_dominant_values": bool(
                generated[dominant].sum()
                < train[dominant].sum() - 0.01
            ),
            "flattens_support_distribution": bool(flattening),
            "concentrates_too_strongly": bool(concentrating),
            "shifts_toward_central_values": bool(
                generated_central_distance
                < train_central_distance - 1e-6
            ),
        },
    }


def corrected_logit_bias(
    table: pd.DataFrame,
    *,
    strength: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    return (
        float(strength)
        * table["delta_log_probability"].to_numpy(dtype=float)
    )


def frequency_bucket_report(
    target: np.ndarray,
    generated: np.ndarray,
    *,
    num_buckets: int,
    epsilon: float,
) -> list[dict[str, Any]]:
    count = min(max(int(num_buckets), 1), len(target))
    ranks = pd.Series(target).rank(method="first")
    try:
        bucket_ids = pd.qcut(
            ranks,
            q=count,
            labels=False,
            duplicates="drop",
        ).to_numpy(dtype=np.int64)
    except ValueError:
        bucket_ids = np.zeros(len(target), dtype=np.int64)
    output = []
    for bucket in sorted(np.unique(bucket_ids)):
        selected = bucket_ids == bucket
        target_mass = float(target[selected].sum())
        generated_mass = float(generated[selected].sum())
        output.append(
            {
                "bucket": int(bucket),
                "support_values": int(selected.sum()),
                "target_mass": target_mass,
                "generated_mass": generated_mass,
                "absolute_mass_error": abs(
                    target_mass - generated_mass
                ),
                "generated_to_target_ratio": (
                    (generated_mass + epsilon)
                    / (target_mass + epsilon)
                ),
            }
        )
    return output


def finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def normalized(counts: np.ndarray, *, epsilon: float) -> np.ndarray:
    values = np.asarray(counts, dtype=float) + float(epsilon)
    return values / max(float(values.sum()), 1e-12)


def entropy(probability: np.ndarray, epsilon: float) -> float:
    probability = np.asarray(probability, dtype=float)
    positive = probability > 0
    return float(
        -np.sum(
            probability[positive]
            * np.log(probability[positive] + epsilon)
        )
    )


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def jensen_shannon(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float,
) -> float:
    middle = 0.5 * (left + right)
    return 0.5 * (
        kl_divergence(left, middle, epsilon)
        + kl_divergence(right, middle, epsilon)
    )


def kl_divergence(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float,
) -> float:
    selected = left > 0
    return float(
        np.sum(
            left[selected]
            * (
                np.log(left[selected] + epsilon)
                - np.log(right[selected] + epsilon)
            )
        )
    )


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def top_mass_error(
    target: np.ndarray,
    generated: np.ndarray,
    order: np.ndarray,
    count: int,
) -> float:
    selected = order[: min(int(count), len(order))]
    return float(abs(target[selected].sum() - generated[selected].sum()))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, float(quantile), side="left"))
    return float(values[order[min(index, len(order) - 1)]])
