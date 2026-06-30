#!/usr/bin/env python3
"""Evaluate ct_2k_sbm_plus synthetic review spines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_sbm_event_spine import evaluate, load_reviews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ct_2k_sbm_plus event spines.")
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


def exact_degree_match_rate(real: pd.DataFrame, synthetic: pd.DataFrame, column: str) -> float:
    real_counts = real[column].value_counts()
    synthetic_counts = synthetic[column].value_counts()
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) == 0:
        return 1.0
    return float(
        (
            real_counts.reindex(index, fill_value=0)
            == synthetic_counts.reindex(index, fill_value=0)
        ).mean()
    )


def load_block_map(path: str | Path, id_col: str, block_col: str) -> dict[Any, int]:
    df = pd.read_csv(path)
    return dict(zip(df[id_col], df[block_col].astype(int)))


def block_pair_count_exact_match_rate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    customer_blocks: dict[Any, int],
    product_blocks: dict[Any, int],
) -> float | None:
    def counts(df: pd.DataFrame) -> pd.Series:
        annotated = df[[customer_col, product_col]].copy()
        annotated["customer_block"] = annotated[customer_col].map(customer_blocks)
        annotated["product_block"] = annotated[product_col].map(product_blocks)
        annotated = annotated.dropna(subset=["customer_block", "product_block"])
        if annotated.empty:
            return pd.Series(dtype=int)
        return annotated.groupby(["customer_block", "product_block"]).size()

    real_counts = counts(real)
    synthetic_counts = counts(synthetic)
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) == 0:
        return None
    return float(
        (
            real_counts.reindex(index, fill_value=0)
            == synthetic_counts.reindex(index, fill_value=0)
        ).mean()
    )


def evaluate_plus(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str = "M",
    customer_blocks: dict[Any, int] | None = None,
    product_blocks: dict[Any, int] | None = None,
) -> dict[str, Any]:
    results = evaluate(
        real,
        synthetic,
        customer_col=customer_col,
        product_col=product_col,
        timestamp_col=timestamp_col,
        trajectory_bins=trajectory_bins,
    )
    additional = {
        "product_degree_exact_match_rate": exact_degree_match_rate(
            real, synthetic, product_col
        ),
        "customer_degree_exact_match_rate": exact_degree_match_rate(
            real, synthetic, customer_col
        ),
        "block_pair_count_exact_match_rate": None,
    }
    if customer_blocks is not None and product_blocks is not None:
        additional["block_pair_count_exact_match_rate"] = (
            block_pair_count_exact_match_rate(
                real,
                synthetic,
                customer_col,
                product_col,
                customer_blocks,
                product_blocks,
            )
        )
    results["additional"] = additional
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

    results = evaluate_plus(
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
