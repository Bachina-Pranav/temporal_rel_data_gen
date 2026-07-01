#!/usr/bin/env python3
"""Generate a review spine with ct_2k_sbm_temporal_kde_stubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation import ContinuousTime2KSBMTemporalKDEStubsGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate customer-product-review_time events with "
            "ct_2k_sbm_temporal_kde_stubs."
        )
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument(
        "--sbm-block-level",
        default="auto",
        help="SBM hierarchy level to use: auto, bottom, top, current, or an integer.",
    )
    parser.add_argument(
        "--timestamp-model",
        choices=[
            "auto",
            "smoothed_date_pmf",
            "block_pair_kde",
            "global_kde",
            "bootstrap_jitter",
        ],
        default="auto",
    )
    parser.add_argument("--timestamp-smoothing-alpha", default="auto")
    parser.add_argument("--timestamp-bandwidth", default="scott")
    parser.add_argument("--timestamp-min-block-count", type=int, default=20)
    parser.add_argument(
        "--pairing-mode",
        choices=["random", "temporal_sorted", "temporal_window_shuffle"],
        default="temporal_window_shuffle",
    )
    parser.add_argument("--temporal-window-size", default="auto")
    parser.add_argument("--avoid-real-edge-prob", type=float, default=0.95)
    parser.add_argument("--max-swap-attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_optional_int(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def parse_alpha(value: str):
    if value == "auto":
        return value
    return float(value)


def parse_bandwidth(value: str):
    if value in {"scott", "silverman"}:
        return value
    return float(value)


def main() -> None:
    args = parse_args()
    reviews = pd.read_csv(args.real_reviews)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ContinuousTime2KSBMTemporalKDEStubsGenerator.from_reviews(
        reviews,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        sbm_block_level=args.sbm_block_level,
        timestamp_model=args.timestamp_model,
        timestamp_smoothing_alpha=parse_alpha(args.timestamp_smoothing_alpha),
        timestamp_bandwidth=parse_bandwidth(args.timestamp_bandwidth),
        timestamp_min_block_count=args.timestamp_min_block_count,
        pairing_mode=args.pairing_mode,
        temporal_window_size=parse_optional_int(args.temporal_window_size),
        avoid_real_edge_prob=args.avoid_real_edge_prob,
        max_swap_attempts=args.max_swap_attempts,
    )
    generator.fit()
    synthetic = generator.generate(
        output_path=output_dir / "synthetic_review.csv",
        debug_dir=output_dir / "debug",
    )
    print(
        "Wrote "
        f"{len(synthetic):,} ct_2k_sbm_temporal_kde_stubs events to "
        f"{output_dir / 'synthetic_review.csv'}"
    )


if __name__ == "__main__":
    main()
