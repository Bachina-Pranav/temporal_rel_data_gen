"""Empirical-Bayes entity latent effects for temporal non-text attributes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .temporal_causal_features import load_block_maps, normalize_verified


EFFECT_COLUMNS = ["rating_effect", "verified_effect"]


@dataclass
class EntityEffectEstimate:
    customer_effects: pd.DataFrame
    product_effects: pd.DataFrame
    global_stats: Dict[str, Any]


def estimate_entity_latent_effects(
    reviews: pd.DataFrame,
    structure_debug_dir: str | Path | None = None,
    customer_id_col: str = "customer_id",
    product_id_col: str = "product_id",
    timestamp_col: str = "review_time",
    rating_col: str = "rating",
    verified_col: str = "verified",
    alpha_product_rating: str | float | None = "auto",
    alpha_customer_rating: str | float | None = "auto",
    alpha_product_verified: str | float | None = "auto",
    alpha_customer_verified: str | float | None = "auto",
) -> EntityEffectEstimate:
    """Estimate shrunk customer/product latent effects from real reviews."""

    frame = reviews.copy()
    required = [customer_id_col, product_id_col, timestamp_col, rating_col, verified_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Cannot estimate entity effects; missing columns: {missing}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=required).copy()
    frame["_rating"] = pd.to_numeric(frame[rating_col], errors="coerce")
    frame["_verified"] = normalize_verified(frame[verified_col])
    frame = frame.dropna(subset=["_rating", "_verified"]).copy()

    customer_blocks, product_blocks = load_block_maps(
        structure_debug_dir, customer_id_col, product_id_col
    )
    global_rating = float(frame["_rating"].mean()) if len(frame) else 0.0
    global_verified_rate = float(frame["_verified"].mean()) if len(frame) else 0.0
    global_verified_logit = logit(global_verified_rate)

    product_degree = frame.groupby(product_id_col).size()
    customer_degree = frame.groupby(customer_id_col).size()
    alpha_pr = resolve_alpha(alpha_product_rating, product_degree, default=10.0)
    alpha_cr = resolve_alpha(alpha_customer_rating, customer_degree, default=10.0)
    alpha_pv = resolve_alpha(alpha_product_verified, product_degree, default=10.0)
    alpha_cv = resolve_alpha(alpha_customer_verified, customer_degree, default=10.0)

    product_struct = compute_entity_structural_features(
        frame,
        entity_col=product_id_col,
        timestamp_col=timestamp_col,
        block_map=product_blocks,
        block_col="product_block",
        id_output_col=product_id_col,
    )
    customer_struct = compute_entity_structural_features(
        frame,
        entity_col=customer_id_col,
        timestamp_col=timestamp_col,
        block_map=customer_blocks,
        block_col="customer_block",
        id_output_col=customer_id_col,
    )
    product_effects = attach_effects(
        product_struct,
        frame,
        entity_col=product_id_col,
        rating_col="_rating",
        verified_col="_verified",
        global_rating=global_rating,
        global_verified_logit=global_verified_logit,
        alpha_rating=alpha_pr,
        alpha_verified=alpha_pv,
    )
    customer_effects = attach_effects(
        customer_struct,
        frame,
        entity_col=customer_id_col,
        rating_col="_rating",
        verified_col="_verified",
        global_rating=global_rating,
        global_verified_logit=global_verified_logit,
        alpha_rating=alpha_cr,
        alpha_verified=alpha_cv,
    )
    stats = {
        "global_mean_rating": global_rating,
        "global_verified_rate": global_verified_rate,
        "global_verified_logit": global_verified_logit,
        "alpha_product_rating": alpha_pr,
        "alpha_customer_rating": alpha_cr,
        "alpha_product_verified": alpha_pv,
        "alpha_customer_verified": alpha_cv,
        "num_reviews": int(len(frame)),
        "num_customers": int(customer_effects.shape[0]),
        "num_products": int(product_effects.shape[0]),
    }
    return EntityEffectEstimate(customer_effects, product_effects, stats)


def compute_entity_structural_features(
    spine: pd.DataFrame,
    entity_col: str,
    timestamp_col: str,
    block_map: Optional[Dict[Any, int]] = None,
    block_col: str = "block",
    id_output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute structural-only entity features from an event spine."""

    frame = spine[[entity_col, timestamp_col]].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=[entity_col, timestamp_col])
    block_map = dict(block_map or {})
    id_output_col = id_output_col or entity_col
    if frame.empty:
        return pd.DataFrame(
            columns=[
                id_output_col,
                block_col,
                "degree",
                "structural_degree",
                "total_degree",
                "log1p_total_degree",
                "lifecycle_start",
                "lifecycle_end",
                "first_review_time_normalized",
                "last_review_time_normalized",
                "activity_span_days",
                "activity_entropy_over_months",
                "fraction_activity_first_half",
                "fraction_activity_second_half",
            ]
        )

    min_time = frame[timestamp_col].min()
    max_time = frame[timestamp_col].max()
    total_span = max((max_time - min_time).total_seconds() / 86400.0, 1.0)
    midpoint = min_time + (max_time - min_time) / 2
    rows = []
    for entity_id, group in frame.groupby(entity_col, sort=False):
        times = group[timestamp_col].sort_values()
        degree = int(len(times))
        first = times.iloc[0]
        last = times.iloc[-1]
        months = times.dt.to_period("M").astype(str)
        month_probs = months.value_counts(normalize=True).to_numpy(dtype=float)
        entropy = float(-(month_probs * np.log(np.maximum(month_probs, 1e-12))).sum())
        first_half = float((times <= midpoint).mean())
        second_half = float((times > midpoint).mean())
        rows.append(
            {
                id_output_col: entity_id,
                block_col: int(block_map.get(entity_id, -1)),
                "degree": degree,
                "structural_degree": degree,
                "total_degree": degree,
                "log1p_total_degree": float(np.log1p(degree)),
                "lifecycle_start": str(first),
                "lifecycle_end": str(last),
                "first_review_time_normalized": float(
                    max((first - min_time).total_seconds() / 86400.0, 0.0) / total_span
                ),
                "last_review_time_normalized": float(
                    max((last - min_time).total_seconds() / 86400.0, 0.0) / total_span
                ),
                "activity_span_days": float(
                    max((last - first).total_seconds() / 86400.0, 0.0)
                ),
                "activity_entropy_over_months": entropy,
                "fraction_activity_first_half": first_half,
                "fraction_activity_second_half": second_half,
            }
        )
    return pd.DataFrame(rows)


def attach_effects(
    structural: pd.DataFrame,
    reviews: pd.DataFrame,
    entity_col: str,
    rating_col: str,
    verified_col: str,
    global_rating: float,
    global_verified_logit: float,
    alpha_rating: float,
    alpha_verified: float,
) -> pd.DataFrame:
    grouped = reviews.groupby(entity_col).agg(
        rating_mean=(rating_col, "mean"),
        rating_count=(rating_col, "count"),
        verified_rate=(verified_col, "mean"),
        verified_count=(verified_col, "count"),
    )
    output = structural.merge(
        grouped,
        left_on=structural.columns[0],
        right_index=True,
        how="left",
    )
    output[["rating_count", "verified_count"]] = output[
        ["rating_count", "verified_count"]
    ].fillna(0)
    output["rating_mean"] = output["rating_mean"].fillna(global_rating)
    output["verified_rate"] = output["verified_rate"].fillna(inv_logit(global_verified_logit))
    rating_shrink = output["rating_count"] / (
        output["rating_count"].astype(float) + float(alpha_rating)
    )
    verified_shrink = output["verified_count"] / (
        output["verified_count"].astype(float) + float(alpha_verified)
    )
    output["rating_effect"] = rating_shrink * (
        output["rating_mean"].astype(float) - float(global_rating)
    )
    output["verified_effect"] = verified_shrink * (
        output["verified_rate"].map(logit) - float(global_verified_logit)
    )
    output["rating_count"] = output["rating_count"].astype(int)
    output["verified_count"] = output["verified_count"].astype(int)
    return output.drop(columns=["rating_mean", "verified_rate"])


def save_entity_effect_estimate(estimate: EntityEffectEstimate, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    effects_dir = output_dir / "entity_effects"
    effects_dir.mkdir(parents=True, exist_ok=True)
    estimate.customer_effects.to_csv(effects_dir / "customer_effects.csv", index=False)
    estimate.product_effects.to_csv(effects_dir / "product_effects.csv", index=False)
    with (effects_dir / "global_effect_stats.json").open("w") as handle:
        json.dump(to_jsonable(estimate.global_stats), handle, indent=2)
        handle.write("\n")


def resolve_alpha(value: str | float | None, degree: pd.Series, default: float) -> float:
    if value is None or value == "auto":
        if len(degree) == 0:
            return float(default)
        median = float(np.median(degree.to_numpy(dtype=float)))
        return median if median > 0 else float(default)
    return float(value)


def logit(rate: float) -> float:
    clipped = float(np.clip(rate, 1e-4, 1.0 - 1e-4))
    return float(np.log(clipped / (1.0 - clipped)))


def inv_logit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
