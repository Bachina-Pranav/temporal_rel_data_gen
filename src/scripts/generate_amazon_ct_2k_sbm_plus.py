#!/usr/bin/env python3
"""Generate an Amazon-style review spine with ct_2k_sbm_plus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation import ContinuousTime2KSBMPlusGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate customer-product-review_time events with ct_2k_sbm_plus."
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
        "--stub-pairing",
        choices=["random", "time_sorted"],
        default="time_sorted",
        help="How to pair customer/product endpoint stubs inside each block pair.",
    )
    parser.add_argument(
        "--pair-multiplicity-mode",
        choices=["none", "block_pair_empirical"],
        default="none",
        help="Optional repeated customer-product pair preservation mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ContinuousTime2KSBMPlusGenerator.from_csv(
        customers_path=args.customers,
        products_path=args.products,
        reviews_path=args.reviews,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        stub_pairing=args.stub_pairing,
        pair_multiplicity_mode=args.pair_multiplicity_mode,
    )
    generator.fit()
    synthetic = generator.generate(
        num_events=args.num_events,
        output_path=args.output,
        debug_dir=args.debug_dir,
    )
    print(f"Wrote {len(synthetic):,} ct_2k_sbm_plus events to {args.output}")


if __name__ == "__main__":
    main()
