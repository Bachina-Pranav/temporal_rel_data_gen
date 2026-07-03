#!/usr/bin/env python3
"""Preflight checks for full Rel-Amazon time-biased stub matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_lowrank_temporal_event import load_entity_blocks  # noqa: E402
from generators.fast_temporal_activity import canonical_time_bucket  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight full Rel-Amazon event-spine generation.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-exact-affinity-cell-size", type=int, default=128)
    parser.add_argument("--allow-single-block-fallback", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_path = Path(args.real_reviews)
    if not review_path.exists():
        raise SystemExit(f"Missing review CSV: {review_path}")
    frame = pd.read_csv(review_path)
    required = [args.customer_id_col, args.product_id_col, args.timestamp_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(frame.columns)}", file=sys.stderr)
        raise SystemExit(1)

    summary: Dict[str, Any] = basic_summary(frame, args)
    debug = Path(args.structure_debug_dir)
    customer_blocks_path = debug / "customer_blocks.csv"
    product_blocks_path = debug / "product_blocks.csv"
    customer_blocks = load_entity_blocks(debug, "customer_blocks.csv", args.customer_id_col, "customer_block")
    product_blocks = load_entity_blocks(debug, "product_blocks.csv", args.product_id_col, "product_block")
    blocks_missing = not customer_blocks or not product_blocks
    summary["structure_debug_dir"] = str(debug)
    summary["customer_blocks_path"] = str(customer_blocks_path)
    summary["product_blocks_path"] = str(product_blocks_path)
    summary["structure_blocks_present"] = not blocks_missing

    if blocks_missing:
        warning = (
            "Missing structure blocks for full Rel-Amazon. Expected: "
            f"{customer_blocks_path} and {product_blocks_path}"
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        summary["warnings"] = [warning]
        summary.update(memory_estimates(len(frame), args.rank, 0, args.max_exact_affinity_cell_size))
        write_json(summary, args.output)
        print(json.dumps(summary, indent=2))
        if not args.allow_single_block_fallback:
            raise SystemExit(2)
        customer_blocks = {entity: 0 for entity in frame[args.customer_id_col].unique()}
        product_blocks = {entity: 0 for entity in frame[args.product_id_col].unique()}

    cell_stats = block_pair_time_stats(frame, customer_blocks, product_blocks, args)
    summary.update(cell_stats)
    summary.update(memory_estimates(len(frame), args.rank, int(cell_stats.get("max_cell_size", 0)), args.max_exact_affinity_cell_size))
    if int(summary.get("max_cell_size", 0)) > 50000:
        summary.setdefault("warnings", []).append("Very large cells detected. Ensure projection fallback is active.")
    elif int(summary.get("max_cell_size", 0)) > 5000:
        summary.setdefault("warnings", []).append("Large cells detected. Exact pairing will fallback to projection for those cells.")
    write_json(summary, args.output)
    print(json.dumps(summary, indent=2))


def basic_summary(frame: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    day = canonical_time_bucket(frame[args.timestamp_col], args.time_granularity)
    month = canonical_time_bucket(frame[args.timestamp_col], args.time_gate_granularity)
    parsed = pd.to_datetime(day, errors="coerce")
    customer_degree = frame[args.customer_id_col].value_counts()
    product_degree = frame[args.product_id_col].value_counts()
    return {
        "num_rows": int(len(frame)),
        "num_customers": int(frame[args.customer_id_col].nunique()),
        "num_products": int(frame[args.product_id_col].nunique()),
        "num_days": int(day.nunique()),
        "num_months": int(month.nunique()),
        "date_min": None if parsed.isna().all() else str(parsed.min().date()),
        "date_max": None if parsed.isna().all() else str(parsed.max().date()),
        **degree_summary(customer_degree, "customer_degree"),
        **degree_summary(product_degree, "product_degree"),
    }


def degree_summary(values: pd.Series, prefix: str) -> Dict[str, float]:
    arr = values.to_numpy(dtype=float)
    return {
        f"{prefix}_min": float(np.min(arr)) if len(arr) else 0.0,
        f"{prefix}_median": float(np.median(arr)) if len(arr) else 0.0,
        f"{prefix}_mean": float(np.mean(arr)) if len(arr) else 0.0,
        f"{prefix}_max": float(np.max(arr)) if len(arr) else 0.0,
    }


def block_pair_time_stats(
    frame: pd.DataFrame,
    customer_blocks: Dict[Any, int],
    product_blocks: Dict[Any, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tmp = pd.DataFrame(
        {
            "customer_block": frame[args.customer_id_col].map(customer_blocks).fillna(-1).astype(int),
            "product_block": frame[args.product_id_col].map(product_blocks).fillna(-1).astype(int),
            "time_bucket": canonical_time_bucket(frame[args.timestamp_col], args.time_granularity),
        }
    )
    counts = tmp.groupby(["customer_block", "product_block", "time_bucket"], sort=False).size().to_numpy(dtype=float)
    return {
        "num_customer_blocks": int(len(set(customer_blocks.values()))),
        "num_product_blocks": int(len(set(product_blocks.values()))),
        "num_block_pair_time_cells": int(len(counts)),
        "average_cell_size": float(np.mean(counts)) if len(counts) else 0.0,
        "max_cell_size": int(np.max(counts)) if len(counts) else 0,
        "p95_cell_size": float(np.percentile(counts, 95.0)) if len(counts) else 0.0,
        "p99_cell_size": float(np.percentile(counts, 99.0)) if len(counts) else 0.0,
    }


def memory_estimates(num_rows: int, rank: int, max_cell_size: int, max_exact_affinity_cell_size: int) -> Dict[str, Any]:
    sparse_matrix_nnz = int(num_rows)
    estimated_svd_memory = int((2 * num_rows * max(rank, 1)) * 8)
    estimated_slot_array_memory = int(num_rows * 9 * 8)
    largest_exact = min(int(max_cell_size), int(max_exact_affinity_cell_size))
    estimated_pairing_memory = int(largest_exact * largest_exact * 8)
    return {
        "sparse_matrix_nnz": sparse_matrix_nnz,
        "estimated_svd_memory_bytes": estimated_svd_memory,
        "estimated_slot_array_memory_bytes": estimated_slot_array_memory,
        "estimated_pairing_memory_for_largest_cell_bytes": estimated_pairing_memory,
    }


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
