#!/usr/bin/env python3
"""Generate an Amazon-style review spine with ct_2k_sbm_temporal_stubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation import ContinuousTime2KSBMTemporalStubsGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate customer-product-review_time events with "
            "ct_2k_sbm_temporal_stubs."
        )
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
        "--stub-pairing",
        choices=["random", "temporal_sorted", "temporal_window_shuffle"],
        default="temporal_window_shuffle",
        help="How to pair customer/product/timestamp stubs inside each block pair.",
    )
    parser.add_argument(
        "--timestamp-stub-mode",
        choices=["reuse_block_pair_timestamps", "kde_jitter"],
        default="reuse_block_pair_timestamps",
        help="How timestamp stubs are converted into generated review_time values.",
    )
    parser.add_argument(
        "--temporal-window-size",
        type=int,
        default=None,
        help="Optional local shuffle window size. Defaults to max(10, sqrt(m_ab)).",
    )
    parser.add_argument(
        "--avoid-real-edge-prob",
        type=float,
        default=0.95,
        help="Probability of trying a local product-stub swap for real edges.",
    )
    parser.add_argument(
        "--pair-multiplicity-mode",
        choices=["none", "empirical_block_pair"],
        default="none",
        help="Experimental repeated-pair adjustment mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ContinuousTime2KSBMTemporalStubsGenerator.from_csv(
        customers_path=args.customers,
        products_path=args.products,
        reviews_path=args.reviews,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        seed=args.seed,
        stub_pairing=args.stub_pairing,
        timestamp_stub_mode=args.timestamp_stub_mode,
        temporal_window_size=args.temporal_window_size,
        avoid_real_edge_prob=args.avoid_real_edge_prob,
        pair_multiplicity_mode=args.pair_multiplicity_mode,
        sbm_block_level=args.sbm_block_level,
    )
    generator.fit()
    synthetic = generator.generate(
        num_events=args.num_events,
        output_path=args.output,
        debug_dir=args.debug_dir,
    )
    print(
        f"Wrote {len(synthetic):,} ct_2k_sbm_temporal_stubs events to {args.output}"
    )


if __name__ == "__main__":
    main()
