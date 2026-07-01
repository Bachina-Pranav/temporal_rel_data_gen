#!/usr/bin/env python3
"""Evaluate ct_2k_sbm_temporal_stubs synthetic review spines."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_ct_2k_sbm_plus import evaluate_plus  # noqa: E402
from evaluate_temporal_sbm_event_spine import load_reviews  # noqa: E402
from reldiff.generation.block_diagnostics import (  # noqa: E402
    compute_all_block_diagnostics,
    load_block_map,
    load_block_maps_from_debug_dir,
    missing_block_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ct_2k_sbm_temporal_stubs event spines."
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
    parser.add_argument(
        "--fallback-single-block-pair",
        action="store_true",
        help="Explicitly collapse all rows to one block pair when no block metadata exists.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory-bins", default="M")
    return parser.parse_args()


def evaluate_temporal_stubs(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str = "M",
    customer_blocks: dict[Any, int] | None = None,
    product_blocks: dict[Any, int] | None = None,
    block_pair_min_count: int = 5,
    warn_missing_blocks: bool = True,
) -> dict[str, Any]:
    real = real.copy()
    synthetic = synthetic.copy()
    real[timestamp_col] = pd.to_datetime(real[timestamp_col], errors="coerce")
    synthetic[timestamp_col] = pd.to_datetime(synthetic[timestamp_col], errors="coerce")
    real = real.dropna(subset=[timestamp_col])
    synthetic = synthetic.dropna(subset=[timestamp_col])

    results = evaluate_plus(
        real,
        synthetic,
        customer_col=customer_col,
        product_col=product_col,
        timestamp_col=timestamp_col,
        trajectory_bins=trajectory_bins,
        customer_blocks=customer_blocks,
        product_blocks=product_blocks,
    )
    if customer_blocks is not None and product_blocks is not None:
        block_diagnostics = compute_all_block_diagnostics(
            real,
            synthetic,
            customer_blocks,
            product_blocks,
            customer_col,
            product_col,
            timestamp_col,
            min_count=block_pair_min_count,
        )
        for warning in block_diagnostics.get("block_diagnostic_warnings", []):
            warnings.warn(warning)
    else:
        block_diagnostics = missing_block_diagnostics(warn=warn_missing_blocks)
    results["additional"].update(block_diagnostics)
    return results


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
    if (
        (customer_blocks is None or product_blocks is None)
        and args.fallback_single_block_pair
    ):
        customer_blocks = {
            customer_id: 0
            for customer_id in pd.concat(
                [real[args.customer_id_col], synthetic[args.customer_id_col]]
            ).dropna().unique()
        }
        product_blocks = {
            product_id: 0
            for product_id in pd.concat(
                [real[args.product_id_col], synthetic[args.product_id_col]]
            ).dropna().unique()
        }

    results = evaluate_temporal_stubs(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        trajectory_bins=args.trajectory_bins,
        customer_blocks=customer_blocks,
        product_blocks=product_blocks,
        block_pair_min_count=args.block_pair_min_count,
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
