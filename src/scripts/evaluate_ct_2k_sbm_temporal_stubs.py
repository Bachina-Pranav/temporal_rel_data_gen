#!/usr/bin/env python3
"""Evaluate ct_2k_sbm_temporal_stubs synthetic review spines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_ct_2k_sbm_plus import (  # noqa: E402
    evaluate_plus,
    load_block_map,
)
from evaluate_temporal_sbm_event_spine import load_reviews  # noqa: E402
from reldiff.generation.continuous_time_temporal_sbm import (  # noqa: E402
    empirical_ks_statistic,
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory-bins", default="M")
    return parser.parse_args()


def block_pair_timestamp_ks_summary(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    customer_blocks: dict[Any, int],
    product_blocks: dict[Any, int],
) -> dict[str, float | int | None]:
    real_annotated = _annotate_with_blocks(
        real, customer_col, product_col, customer_blocks, product_blocks
    )
    synthetic_annotated = _annotate_with_blocks(
        synthetic, customer_col, product_col, customer_blocks, product_blocks
    )
    synthetic_groups = {
        (int(customer_block), int(product_block)): group
        for (customer_block, product_block), group in synthetic_annotated.groupby(
            ["customer_block", "product_block"]
        )
    }

    ks_values = []
    for (customer_block, product_block), real_group in real_annotated.groupby(
        ["customer_block", "product_block"], sort=True
    ):
        synthetic_group = synthetic_groups.get(
            (int(customer_block), int(product_block))
        )
        if synthetic_group is None or synthetic_group.empty:
            continue
        real_values = _timestamp_values(real_group[timestamp_col])
        synthetic_values = _timestamp_values(synthetic_group[timestamp_col])
        ks = empirical_ks_statistic(real_values, synthetic_values)
        if ks is not None:
            ks_values.append(ks)

    return {
        "block_pair_timestamp_ks_mean": float(np.mean(ks_values))
        if ks_values
        else None,
        "block_pair_timestamp_ks_median": float(np.median(ks_values))
        if ks_values
        else None,
        "block_pair_timestamp_ks_num_pairs": len(ks_values),
    }


def evaluate_temporal_stubs(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str = "M",
    customer_blocks: dict[Any, int] | None = None,
    product_blocks: dict[Any, int] | None = None,
) -> dict[str, Any]:
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
    block_pair_timestamp = {
        "block_pair_timestamp_ks_mean": None,
        "block_pair_timestamp_ks_median": None,
        "block_pair_timestamp_ks_num_pairs": 0,
    }
    if customer_blocks is not None and product_blocks is not None:
        block_pair_timestamp = block_pair_timestamp_ks_summary(
            real,
            synthetic,
            customer_col,
            product_col,
            timestamp_col,
            customer_blocks,
            product_blocks,
        )
    results["additional"].update(block_pair_timestamp)
    return results


def _annotate_with_blocks(
    df: pd.DataFrame,
    customer_col: str,
    product_col: str,
    customer_blocks: dict[Any, int],
    product_blocks: dict[Any, int],
) -> pd.DataFrame:
    annotated = df.copy()
    annotated["customer_block"] = annotated[customer_col].map(customer_blocks)
    annotated["product_block"] = annotated[product_col].map(product_blocks)
    return annotated.dropna(subset=["customer_block", "product_block"]).copy()


def _timestamp_values(timestamps: pd.Series) -> np.ndarray:
    return pd.to_datetime(timestamps).astype("int64").to_numpy(dtype=float)


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

    results = evaluate_temporal_stubs(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        trajectory_bins=args.trajectory_bins,
        customer_blocks=customer_blocks,
        product_blocks=product_blocks,
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
