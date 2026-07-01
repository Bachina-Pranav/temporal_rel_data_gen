#!/usr/bin/env python3
"""Evaluate all structural temporal review-spine generators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_all_structure_methods import (  # noqa: E402
    block_pair_date_count_l1_metrics,
    exact_degree_match,
)
from evaluate_temporal_sbm_event_spine import evaluate, load_reviews  # noqa: E402
from reldiff.generation.block_diagnostics import (  # noqa: E402
    compute_all_block_diagnostics,
    load_block_maps_from_debug_dir,
    missing_block_diagnostics,
)


METHODS = [
    "continuous_time_temporal_sbm",
    "ct_2k_sbm_plus",
    "ct_2k_sbm_temporal_stubs",
    "ct_2k_sbm_temporal_kde_stubs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all structural method outputs under one root."
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--trajectory-bins", default="M")
    parser.add_argument("--block-pair-min-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_all_methods(
        real_reviews_path=args.real_reviews,
        outputs_root=args.outputs_root,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        trajectory_bins=args.trajectory_bins,
        block_pair_min_count=args.block_pair_min_count,
    )
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


def evaluate_all_methods(
    real_reviews_path: str | Path,
    outputs_root: str | Path,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str = "M",
    block_pair_min_count: int = 5,
) -> list[Dict[str, Any]]:
    real = load_reviews(real_reviews_path, timestamp_col)
    outputs_root = Path(outputs_root)
    rows = []
    for method in METHODS:
        method_dir = outputs_root / method
        synthetic_path = method_dir / "synthetic_review.csv"
        if not synthetic_path.exists():
            continue
        rows.append(
            evaluate_method(
                method,
                synthetic_path,
                method_dir / "debug",
                real,
                customer_col,
                product_col,
                timestamp_col,
                trajectory_bins,
                block_pair_min_count,
            )
        )
    return rows


def evaluate_method(
    method: str,
    synthetic_path: Path,
    debug_dir: Path,
    real: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    trajectory_bins: str,
    block_pair_min_count: int,
) -> Dict[str, Any]:
    synthetic = load_reviews(synthetic_path, timestamp_col)
    metrics = evaluate(
        real,
        synthetic,
        customer_col=customer_col,
        product_col=product_col,
        timestamp_col=timestamp_col,
        trajectory_bins=trajectory_bins,
    )
    customer_blocks = None
    product_blocks = None
    if debug_dir.exists():
        customer_blocks, product_blocks, _, _ = load_block_maps_from_debug_dir(
            debug_dir, customer_col, product_col
        )

    if customer_blocks is not None and product_blocks is not None:
        block_metrics = compute_all_block_diagnostics(
            real,
            synthetic,
            customer_blocks,
            product_blocks,
            customer_col,
            product_col,
            timestamp_col,
            min_count=block_pair_min_count,
        )
        block_metrics.update(
            block_pair_date_count_l1_metrics(
                real,
                synthetic,
                customer_blocks,
                product_blocks,
                customer_col,
                product_col,
                timestamp_col,
            )
        )
    else:
        block_metrics = missing_block_diagnostics(warn=False)
        block_metrics.update(
            {
                "block_pair_timestamp_count_l1_by_date_mean": None,
                "block_pair_timestamp_count_l1_by_date_weighted_mean": None,
            }
        )

    temporal = metrics["temporal"]
    row = {
        "method": method,
        **metrics["structural"],
        **block_metrics,
        **temporal,
        **metrics["joint_temporal_edge"],
        "product_lifecycle_start_corr": lifecycle_boundary_corr(
            real, synthetic, product_col, timestamp_col, boundary="start"
        ),
        "product_lifecycle_end_corr": lifecycle_boundary_corr(
            real, synthetic, product_col, timestamp_col, boundary="end"
        ),
        "preserves_customer_degree_exactly": exact_degree_match(
            real, synthetic, customer_col
        ),
        "preserves_product_degree_exactly": exact_degree_match(
            real, synthetic, product_col
        ),
        "preserves_timestamp_multiset_exactly": bool(
            temporal.get("timestamp_multiset_exact_match")
        ),
        "has_nontrivial_block_structure": bool(
            (block_metrics.get("num_customer_blocks") or 0) > 1
            or (block_metrics.get("num_product_blocks") or 0) > 1
        ),
        "generates_timestamps_from_kde": method == "ct_2k_sbm_temporal_kde_stubs",
        "reuses_exact_timestamps": bool(
            temporal.get("timestamp_multiset_exact_match")
        ),
    }
    return row


def lifecycle_boundary_corr(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    product_col: str,
    timestamp_col: str,
    boundary: str,
) -> float | None:
    if boundary == "start":
        real_values = real.groupby(product_col)[timestamp_col].min()
        synthetic_values = synthetic.groupby(product_col)[timestamp_col].min()
    else:
        real_values = real.groupby(product_col)[timestamp_col].max()
        synthetic_values = synthetic.groupby(product_col)[timestamp_col].max()
    index = real_values.index.intersection(synthetic_values.index)
    if len(index) < 2:
        return None
    real_days = (
        pd.to_datetime(real_values.loc[index]).astype("int64").to_numpy(dtype=float)
        / 1e9
        / 86400.0
    )
    synthetic_days = (
        pd.to_datetime(synthetic_values.loc[index]).astype("int64").to_numpy(dtype=float)
        / 1e9
        / 86400.0
    )
    if real_days.std() == 0 or synthetic_days.std() == 0:
        return None
    return float(np.corrcoef(real_days, synthetic_days)[0, 1])


if __name__ == "__main__":
    main()
