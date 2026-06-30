#!/usr/bin/env python3
"""Evaluate generated temporal review attributes."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes.text_latents import TextLatentEncoder
from reldiff.generation.continuous_time_temporal_sbm import (
    empirical_ks_statistic,
    empirical_wasserstein_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal attribute generation.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--summary-col", default="summary")
    parser.add_argument("--review-text-col", default="review_text")
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory-bins", default="M")
    return parser.parse_args()


def load_reviews(path: str | Path, timestamp_col: str) -> pd.DataFrame:
    reviews = pd.read_csv(path)
    reviews[timestamp_col] = pd.to_datetime(reviews[timestamp_col], errors="coerce")
    return reviews.dropna(subset=[timestamp_col]).copy()


def evaluate_attributes(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "product_id",
    timestamp_col: str = "review_time",
    rating_col: str = "rating",
    verified_col: str = "verified",
    summary_col: str = "summary",
    review_text_col: str = "review_text",
    trajectory_bins: str = "M",
) -> Dict[str, Any]:
    real = real.copy()
    synthetic = synthetic.copy()
    real[timestamp_col] = pd.to_datetime(real[timestamp_col], errors="coerce")
    synthetic[timestamp_col] = pd.to_datetime(synthetic[timestamp_col], errors="coerce")
    real = real.dropna(subset=[timestamp_col])
    synthetic = synthetic.dropna(subset=[timestamp_col])

    results = {
        "categorical": {},
        "relational_attributes": {},
        "temporal": {},
        "text_latent": {},
        "text_lexical": {},
        "privacyish": {},
    }

    for column in [rating_col, verified_col]:
        if column in real.columns and column in synthetic.columns:
            results["categorical"][f"{column}_distribution_tv"] = total_variation(
                real[column], synthetic[column]
            )
            results["categorical"][f"{column}_distribution_js"] = js_divergence(
                real[column], synthetic[column]
            )
            results["categorical"][f"{column}_by_month_correlation"] = monthly_value_correlation(
                real, synthetic, timestamp_col, column, trajectory_bins
            )

    if rating_col in real.columns and rating_col in synthetic.columns:
        real_rating = numeric_series(real[rating_col])
        synthetic_rating = numeric_series(synthetic[rating_col])
        results["relational_attributes"].update(
            {
                "product_average_rating_correlation": grouped_mean_correlation(
                    real.assign(_rating=real_rating),
                    synthetic.assign(_rating=synthetic_rating),
                    product_col,
                    "_rating",
                ),
                "customer_average_rating_correlation": grouped_mean_correlation(
                    real.assign(_rating=real_rating),
                    synthetic.assign(_rating=synthetic_rating),
                    customer_col,
                    "_rating",
                ),
                "rating_vs_product_degree_correlation_real": degree_value_correlation(
                    real.assign(_rating=real_rating), product_col, "_rating"
                ),
                "rating_vs_product_degree_correlation_synthetic": degree_value_correlation(
                    synthetic.assign(_rating=synthetic_rating), product_col, "_rating"
                ),
                "rating_vs_customer_degree_correlation_real": degree_value_correlation(
                    real.assign(_rating=real_rating), customer_col, "_rating"
                ),
                "rating_vs_customer_degree_correlation_synthetic": degree_value_correlation(
                    synthetic.assign(_rating=synthetic_rating), customer_col, "_rating"
                ),
            }
        )
        results["temporal"]["monthly_average_rating_correlation"] = monthly_value_correlation(
            real.assign(_rating=real_rating),
            synthetic.assign(_rating=synthetic_rating),
            timestamp_col,
            "_rating",
            trajectory_bins,
        )
        results["temporal"]["product_rating_trajectory_correlation_top_products"] = (
            top_entity_trajectory_correlation(
                real.assign(_rating=real_rating),
                synthetic.assign(_rating=synthetic_rating),
                product_col,
                timestamp_col,
                "_rating",
                trajectory_bins,
            )
        )

    if verified_col in real.columns and verified_col in synthetic.columns:
        real_verified = bool_or_numeric_series(real[verified_col])
        synthetic_verified = bool_or_numeric_series(synthetic[verified_col])
        results["relational_attributes"].update(
            {
                "product_verified_rate_correlation": grouped_mean_correlation(
                    real.assign(_verified=real_verified),
                    synthetic.assign(_verified=synthetic_verified),
                    product_col,
                    "_verified",
                ),
                "customer_verified_rate_correlation": grouped_mean_correlation(
                    real.assign(_verified=real_verified),
                    synthetic.assign(_verified=synthetic_verified),
                    customer_col,
                    "_verified",
                ),
            }
        )
        results["temporal"]["monthly_verified_rate_correlation"] = monthly_value_correlation(
            real.assign(_verified=real_verified),
            synthetic.assign(_verified=synthetic_verified),
            timestamp_col,
            "_verified",
            trajectory_bins,
        )

    for column in [summary_col, review_text_col]:
        if column in real.columns and column in synthetic.columns:
            results["text_lexical"][column] = lexical_metrics(real[column], synthetic[column])
            results["privacyish"][f"{column}_exact_copy_rate"] = exact_copy_rate(
                real[column], synthetic[column]
            )
            results["text_latent"][column] = text_embedding_metrics(
                real[column], synthetic[column]
            )

    return results


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    real_counts = real.value_counts(normalize=True)
    synthetic_counts = synthetic.value_counts(normalize=True)
    index = real_counts.index.union(synthetic_counts.index)
    return float(
        0.5
        * np.abs(
            real_counts.reindex(index, fill_value=0)
            - synthetic_counts.reindex(index, fill_value=0)
        ).sum()
    )


def js_divergence(real: pd.Series, synthetic: pd.Series) -> float:
    real_counts = real.value_counts(normalize=True)
    synthetic_counts = synthetic.value_counts(normalize=True)
    index = real_counts.index.union(synthetic_counts.index)
    p = real_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    q = synthetic_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_or_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        return numeric
    lowered = series.fillna("").astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y", "verified"}).astype(float)


def grouped_mean_correlation(
    real: pd.DataFrame, synthetic: pd.DataFrame, group_col: str, value_col: str
) -> Optional[float]:
    real_values = real.groupby(group_col)[value_col].mean()
    synthetic_values = synthetic.groupby(group_col)[value_col].mean()
    index = real_values.index.intersection(synthetic_values.index)
    if len(index) < 2:
        return None
    return safe_corr(
        real_values.reindex(index).to_numpy(dtype=float),
        synthetic_values.reindex(index).to_numpy(dtype=float),
    )


def degree_value_correlation(
    df: pd.DataFrame, group_col: str, value_col: str
) -> Optional[float]:
    grouped = df.groupby(group_col).agg(degree=(value_col, "size"), value=(value_col, "mean"))
    if len(grouped) < 2:
        return None
    return safe_corr(grouped["degree"].to_numpy(dtype=float), grouped["value"].to_numpy(dtype=float))


def monthly_value_correlation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    freq: str,
) -> Optional[float]:
    real_values = real.set_index(timestamp_col)[value_col].resample(freq).mean()
    synthetic_values = synthetic.set_index(timestamp_col)[value_col].resample(freq).mean()
    index = real_values.index.union(synthetic_values.index)
    if len(index) < 2:
        return None
    return safe_corr(
        real_values.reindex(index).to_numpy(dtype=float),
        synthetic_values.reindex(index).to_numpy(dtype=float),
    )


def top_entity_trajectory_correlation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity_col: str,
    timestamp_col: str,
    value_col: str,
    freq: str,
    k: int = 20,
) -> Optional[float]:
    correlations = []
    for entity_id in real[entity_col].value_counts().head(k).index:
        real_entity = real[real[entity_col] == entity_id]
        synthetic_entity = synthetic[synthetic[entity_col] == entity_id]
        if synthetic_entity.empty:
            continue
        corr = monthly_value_correlation(
            real_entity, synthetic_entity, timestamp_col, value_col, freq
        )
        if corr is not None:
            correlations.append(corr)
    if not correlations:
        return None
    return float(np.mean(correlations))


def lexical_metrics(real: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
    real_texts = real.fillna("").astype(str).tolist()
    synthetic_texts = synthetic.fillna("").astype(str).tolist()
    real_lengths = np.asarray([len(text.split()) for text in real_texts], dtype=float)
    synthetic_lengths = np.asarray([len(text.split()) for text in synthetic_texts], dtype=float)
    return {
        "real_average_length": float(real_lengths.mean()) if len(real_lengths) else 0.0,
        "synthetic_average_length": float(synthetic_lengths.mean())
        if len(synthetic_lengths)
        else 0.0,
        "text_length_ks": empirical_ks_statistic(real_lengths, synthetic_lengths),
        "text_length_wasserstein": empirical_wasserstein_1d(real_lengths, synthetic_lengths),
        "real_vocabulary_size": len(vocabulary(real_texts)),
        "synthetic_vocabulary_size": len(vocabulary(synthetic_texts)),
        "synthetic_distinct_1": distinct_n(synthetic_texts, 1),
        "synthetic_distinct_2": distinct_n(synthetic_texts, 2),
        "synthetic_duplicate_text_rate": duplicate_text_rate(synthetic_texts),
    }


def text_embedding_metrics(real: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
    encoder = TextLatentEncoder(backend="hashing", latent_dim=128)
    real_embeddings = encoder.encode(real.fillna("").astype(str).tolist())
    synthetic_embeddings = encoder.encode(synthetic.fillna("").astype(str).tolist())
    real_center = real_embeddings.mean(axis=0)
    synthetic_center = synthetic_embeddings.mean(axis=0)
    normalized_real = normalize_rows(real_embeddings)
    normalized_synthetic = normalize_rows(synthetic_embeddings)
    similarities = normalized_synthetic @ normalized_real.T
    closest_cosine = similarities.max(axis=1) if similarities.size else np.asarray([])
    return {
        "mean_embedding_cosine_to_real_center": cosine(real_center, synthetic_center),
        "closest_text_embedding_cosine_mean": float(closest_cosine.mean())
        if len(closest_cosine)
        else None,
        "closest_text_embedding_cosine_min": float(closest_cosine.min())
        if len(closest_cosine)
        else None,
    }


def exact_copy_rate(real: pd.Series, synthetic: pd.Series) -> float:
    real_texts = set(real.fillna("").astype(str))
    synthetic_texts = synthetic.fillna("").astype(str)
    if len(synthetic_texts) == 0:
        return 0.0
    return float(synthetic_texts.isin(real_texts).mean())


def vocabulary(texts: List[str]) -> set[str]:
    vocab = set()
    for text in texts:
        vocab.update(re.findall(r"[A-Za-z0-9_']+", text.lower()))
    return vocab


def distinct_n(texts: List[str], n: int) -> float:
    grams = []
    for text in texts:
        tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())
        grams.extend(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    if not grams:
        return 0.0
    return float(len(set(grams)) / len(grams))


def duplicate_text_rate(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return float(1.0 - len(set(texts)) / len(texts))


def safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def cosine(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        return None
    return float(np.dot(x, y) / denom)


def main() -> None:
    args = parse_args()
    real = load_reviews(args.real_reviews, args.timestamp_col)
    synthetic = load_reviews(args.synthetic_reviews, args.timestamp_col)
    results = evaluate_attributes(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
        summary_col=args.summary_col,
        review_text_col=args.review_text_col,
        trajectory_bins=args.trajectory_bins,
    )
    print(json.dumps(results, indent=2))
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
