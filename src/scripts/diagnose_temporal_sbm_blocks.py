#!/usr/bin/env python3
"""Diagnose temporal SBM block assignments and block-pair fidelity."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_sbm_event_spine import load_reviews  # noqa: E402
from reldiff.generation.block_diagnostics import (  # noqa: E402
    BLOCK_METADATA_WARNING,
    compute_all_block_diagnostics,
    load_block_map,
    load_block_maps_from_debug_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal SBM block and block-pair diagnostics."
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--customer-blocks", default=None)
    parser.add_argument("--product-blocks", default=None)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--block-pair-min-count", type=int, default=5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = load_reviews(args.real_reviews, args.timestamp_col)
    synthetic = load_reviews(args.synthetic_reviews, args.timestamp_col)

    customer_blocks = None
    product_blocks = None
    if args.customer_blocks is not None:
        customer_blocks = load_block_map(
            args.customer_blocks, args.customer_id_col, "customer_block"
        )
    if args.product_blocks is not None:
        product_blocks = load_block_map(
            args.product_blocks, args.product_id_col, "product_block"
        )
    if (
        (customer_blocks is None or product_blocks is None)
        and args.debug_dir is not None
    ):
        customer_blocks, product_blocks, _, _ = load_block_maps_from_debug_dir(
            args.debug_dir, args.customer_id_col, args.product_id_col
        )

    if customer_blocks is None or product_blocks is None:
        warnings.warn(BLOCK_METADATA_WARNING)
        raise SystemExit(2)

    diagnostics = compute_all_block_diagnostics(
        real,
        synthetic,
        customer_blocks,
        product_blocks,
        args.customer_id_col,
        args.product_id_col,
        args.timestamp_col,
        min_count=args.block_pair_min_count,
    )

    print("Temporal SBM block diagnostics")
    print(f"  customer blocks: {diagnostics['num_customer_blocks']}")
    print(f"  product blocks: {diagnostics['num_product_blocks']}")
    print(f"  nonzero real block pairs: {diagnostics['num_nonzero_block_pairs_real']}")
    print(
        "  nonzero synthetic block pairs: "
        f"{diagnostics['num_nonzero_block_pairs_synthetic']}"
    )
    print(
        "  valid timestamp KS block pairs: "
        f"{diagnostics['block_pair_timestamp_ks_num_pairs']}"
    )
    print(
        "  block-pair count exact match rate: "
        f"{diagnostics['block_pair_count_exact_match_rate']}"
    )
    if diagnostics.get("block_diagnostic_warnings"):
        print("  warnings:")
        for warning in diagnostics["block_diagnostic_warnings"]:
            print(f"    - {warning}")

    print(json.dumps(diagnostics, indent=2))
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(diagnostics, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
