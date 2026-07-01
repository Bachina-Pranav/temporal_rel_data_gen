#!/usr/bin/env python3
"""Generate an Amazon-style review event spine with continuous-time temporal SBM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation import ContinuousTimeTemporalSBMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate customer-product-review_time events with ContinuousTimeTemporalSBMGenerator."
    )
    parser.add_argument("--customers", required=True, help="Path to customer table CSV.")
    parser.add_argument("--products", required=True, help="Path to product table CSV.")
    parser.add_argument("--reviews", required=True, help="Path to review table CSV.")
    parser.add_argument("--output", required=True, help="Output synthetic review CSV.")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--num-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument(
        "--sbm-block-level",
        default="auto",
        help="SBM hierarchy level to use: auto, bottom, top, current, or an integer.",
    )
    parser.add_argument(
        "--avoid-duplicate-pairs-same-time-neighborhood",
        action="store_true",
        help="Accepted for API completeness; duplicate pairs are allowed in this first version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ContinuousTimeTemporalSBMGenerator.from_csv(
        customers_path=args.customers,
        products_path=args.products,
        reviews_path=args.reviews,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        sbm_block_level=args.sbm_block_level,
    )
    generator.fit()
    synthetic = generator.generate(
        num_events=args.num_events,
        output_path=args.output,
        debug_dir=args.debug_dir,
        avoid_duplicate_pairs_same_time_neighborhood=(
            args.avoid_duplicate_pairs_same_time_neighborhood
        ),
    )
    print(f"Wrote {len(synthetic):,} synthetic review events to {args.output}")


if __name__ == "__main__":
    main()
