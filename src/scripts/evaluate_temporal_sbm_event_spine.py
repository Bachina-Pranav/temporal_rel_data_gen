#!/usr/bin/env python3
"""Evaluate a synthetic temporal review event spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation.continuous_time_temporal_sbm import (
    duplicate_pair_rate,
    empirical_ks_statistic,
    empirical_wasserstein_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal SBM event spine CSVs.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory-bins", default="M", help="Pandas time frequency for diagnostics.")
    return parser.parse_args()


def load_reviews(path: str | Path, timestamp_col: str) -> pd.DataFrame:
    reviews = pd.read_csv(path)
    reviews[timestamp_col] = pd.to_datetime(reviews[timestamp_col], errors="coerce")
    return reviews.dropna(subset=[timestamp_col]).copy()


def degree_counts(df: pd.DataFrame, column: str) -> np.ndarray:
    return df.groupby(column).size().to_numpy(dtype=float)


def total_variation(real: pd.Series, synthetic: pd.Series) -> float | None:
    real_counts = real.value_counts(normalize=True)
    syn_counts = synthetic.value_counts(normalize=True)
    index = real_counts.index.union(syn_counts.index)
    if len(index) == 0:
        return None
    return float(0.5 * np.abs(real_counts.reindex(index, fill_value=0) - syn_counts.reindex(index, fill_value=0)).sum())


def normalized_time_values(df: pd.DataFrame, timestamp_col: str) -> np.ndarray:
    if len(df) == 0:
        return np.asarray([], dtype=float)
    times = df[timestamp_col]
    min_time = times.min()
    max_time = times.max()
    span = (max_time - min_time).total_seconds()
    if span <= 0:
        return np.zeros(len(df), dtype=float)
    return ((times - min_time).dt.total_seconds() / span).to_numpy(dtype=float)


def inter_event_times(df: pd.DataFrame, group_col: str, timestamp_col: str) -> np.ndarray:
    intervals = []
    for _, group in df.sort_values(timestamp_col).groupby(group_col):
        values = group[timestamp_col].sort_values().astype("int64").to_numpy()
        if len(values) < 2:
            continue
        intervals.extend(np.diff(values) / 1e9 / 86400.0)
    return np.asarray(intervals, dtype=float)


def top_product_overlap(
    real: pd.DataFrame, synthetic: pd.DataFrame, product_col: str, k: int = 100
) -> float:
    real_top = set(real[product_col].value_counts().head(k).index)
    syn_top = set(synthetic[product_col].value_counts().head(k).index)
    if not real_top:
        return 0.0
    return float(len(real_top & syn_top) / len(real_top))


def edge_overlap_rate(
    real: pd.DataFrame, synthetic: pd.DataFrame, customer_col: str, product_col: str
) -> float:
    real_edges = set(map(tuple, real[[customer_col, product_col]].drop_duplicates().to_numpy()))
    syn_edges = set(map(tuple, synthetic[[customer_col, product_col]].drop_duplicates().to_numpy()))
    if not syn_edges:
        return 0.0
    return float(len(real_edges & syn_edges) / len(syn_edges))


def count_correlation(
    real: pd.DataFrame, synthetic: pd.DataFrame, timestamp_col: str, freq: str
) -> float | None:
    real_counts = real.set_index(timestamp_col).resample(freq).size()
    syn_counts = synthetic.set_index(timestamp_col).resample(freq).size()
    index = real_counts.index.union(syn_counts.index)
    if len(index) < 2:
        return None
    real_values = real_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    syn_values = syn_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    if real_values.std() == 0 or syn_values.std() == 0:
        return None
    return float(np.corrcoef(real_values, syn_values)[0, 1])


def top_product_trajectory_corr(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    product_col: str,
    timestamp_col: str,
    freq: str,
    k: int = 20,
) -> float | None:
    products = real[product_col].value_counts().head(k).index
    correlations = []
    for product_id in products:
        real_product = real[real[product_col] == product_id]
        syn_product = synthetic[synthetic[product_col] == product_id]
        corr = count_correlation(real_product, syn_product, timestamp_col, freq)
        if corr is not None:
            correlations.append(corr)
    if not correlations:
        return None
    return float(np.mean(correlations))


def evaluate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str,
) -> dict[str, Any]:
    real_time_x = normalized_time_values(real, timestamp_col)
    syn_time_x = normalized_time_values(synthetic, timestamp_col)

    real_customer_intervals = inter_event_times(real, customer_col, timestamp_col)
    syn_customer_intervals = inter_event_times(synthetic, customer_col, timestamp_col)
    real_product_intervals = inter_event_times(real, product_col, timestamp_col)
    syn_product_intervals = inter_event_times(synthetic, product_col, timestamp_col)

    real_days = real[timestamp_col].astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0
    syn_days = synthetic[timestamp_col].astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0

    return {
        "structural": {
            "num_reviews_real": int(len(real)),
            "num_reviews_synthetic": int(len(synthetic)),
            "active_customers_real": int(real[customer_col].nunique()),
            "active_customers_synthetic": int(synthetic[customer_col].nunique()),
            "active_products_real": int(real[product_col].nunique()),
            "active_products_synthetic": int(synthetic[product_col].nunique()),
            "customer_degree_ks": empirical_ks_statistic(
                degree_counts(real, customer_col), degree_counts(synthetic, customer_col)
            ),
            "product_degree_ks": empirical_ks_statistic(
                degree_counts(real, product_col), degree_counts(synthetic, product_col)
            ),
            "duplicate_customer_product_rate_real": duplicate_pair_rate(
                real, customer_col, product_col
            ),
            "duplicate_customer_product_rate_synthetic": duplicate_pair_rate(
                synthetic, customer_col, product_col
            ),
            "top_100_product_overlap": top_product_overlap(real, synthetic, product_col),
            "edge_overlap_rate": edge_overlap_rate(
                real, synthetic, customer_col, product_col
            ),
        },
        "temporal": {
            "global_timestamp_ks": empirical_ks_statistic(real_time_x, syn_time_x),
            "global_timestamp_wasserstein_days": empirical_wasserstein_1d(
                real_days, syn_days
            ),
            "hour_of_day_total_variation": total_variation(
                real[timestamp_col].dt.hour, synthetic[timestamp_col].dt.hour
            ),
            "day_of_week_total_variation": total_variation(
                real[timestamp_col].dt.dayofweek, synthetic[timestamp_col].dt.dayofweek
            ),
            "monthly_or_daily_count_correlation": count_correlation(
                real, synthetic, timestamp_col, trajectory_bins
            ),
            "product_inter_event_time_ks": empirical_ks_statistic(
                real_product_intervals, syn_product_intervals
            ),
            "customer_inter_event_time_ks": empirical_ks_statistic(
                real_customer_intervals, syn_customer_intervals
            ),
        },
        "joint_temporal_edge": {
            "top_product_trajectory_corr": top_product_trajectory_corr(
                real, synthetic, product_col, timestamp_col, trajectory_bins
            )
        },
    }


def main() -> None:
    args = parse_args()
    real = load_reviews(args.real_reviews, args.timestamp_col)
    synthetic = load_reviews(args.synthetic_reviews, args.timestamp_col)
    results = evaluate(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
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
