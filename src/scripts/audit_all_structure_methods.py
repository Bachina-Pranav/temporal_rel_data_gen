#!/usr/bin/env python3
"""Audit structural temporal generators across an outputs root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_sbm_event_spine import (  # noqa: E402
    degree_counts,
    edge_overlap_rate,
    load_reviews,
)
from reldiff.generation.block_diagnostics import (  # noqa: E402
    compute_all_block_diagnostics,
    load_block_maps_from_debug_dir,
    missing_block_diagnostics,
)
from reldiff.generation.continuous_time_temporal_sbm import (  # noqa: E402
    duplicate_pair_rate,
    empirical_ks_statistic,
)


KNOWN_METHODS = [
    "continuous_time_temporal_sbm",
    "ct_2k_sbm_plus",
    "ct_2k_sbm_temporal_stubs",
    "product_time_ipf_temporal_auto_icl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all structural method outputs under one root."
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-pair-min-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = load_reviews(args.real_reviews, args.timestamp_col)
    outputs_root = Path(args.outputs_root)
    rows = []
    for method_dir in discover_method_dirs(outputs_root):
        row = audit_method(
            method_dir,
            real,
            args.customer_id_col,
            args.product_id_col,
            args.timestamp_col,
            min_count=args.block_pair_min_count,
        )
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame([flatten_for_csv(row) for row in rows]).to_csv(csv_path, index=False)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


def discover_method_dirs(outputs_root: Path) -> list[Path]:
    discovered = []
    for method in KNOWN_METHODS:
        path = outputs_root / method
        if (path / "synthetic_review.csv").exists():
            discovered.append(path)
    for path in sorted(outputs_root.iterdir()) if outputs_root.exists() else []:
        if path.is_dir() and (path / "synthetic_review.csv").exists() and path not in discovered:
            discovered.append(path)
    return discovered


def audit_method(
    method_dir: Path,
    real: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    min_count: int,
) -> Dict[str, Any]:
    method = method_dir.name
    synthetic_path = method_dir / "synthetic_review.csv"
    synthetic = load_reviews(synthetic_path, timestamp_col)
    debug_dir = method_dir / "debug"
    customer_blocks = None
    product_blocks = None
    block_files = {"customer_blocks": None, "product_blocks": None}
    if debug_dir.exists():
        customer_blocks, product_blocks, customer_path, product_path = load_block_maps_from_debug_dir(
            debug_dir, customer_col, product_col
        )
        block_files = {
            "customer_blocks": str(customer_path) if customer_path else None,
            "product_blocks": str(product_path) if product_path else None,
        }

    if customer_blocks is not None and product_blocks is not None:
        block_diagnostics = compute_all_block_diagnostics(
            real,
            synthetic,
            customer_blocks,
            product_blocks,
            customer_col,
            product_col,
            timestamp_col,
            min_count=min_count,
        )
    else:
        block_diagnostics = missing_block_diagnostics(warn=False)

    structural = {
        "customer_degree_ks": empirical_ks_statistic(
            degree_counts(real, customer_col), degree_counts(synthetic, customer_col)
        ),
        "product_degree_ks": empirical_ks_statistic(
            degree_counts(real, product_col), degree_counts(synthetic, product_col)
        ),
        "active_customers_synthetic": int(synthetic[customer_col].nunique()),
        "active_products_synthetic": int(synthetic[product_col].nunique()),
        "edge_overlap_rate": edge_overlap_rate(real, synthetic, customer_col, product_col),
        "duplicate_customer_product_rate": duplicate_pair_rate(
            synthetic, customer_col, product_col
        ),
        "timestamp_exact_reuse_rate": timestamp_exact_reuse_rate(
            real[timestamp_col], synthetic[timestamp_col]
        ),
    }
    learned_vs_preserved = learned_vs_preserved_summary(
        real, synthetic, customer_col, product_col, timestamp_col, block_diagnostics
    )
    interpretation = interpret_method(block_diagnostics, learned_vs_preserved, method)
    return {
        "method": method,
        "synthetic_review_path": str(synthetic_path),
        "debug_dir": str(debug_dir) if debug_dir.exists() else None,
        "block_files": block_files,
        "interpretation": interpretation,
        "learned_vs_preserved_summary": learned_vs_preserved,
        **structural,
        **block_diagnostics,
    }


def learned_vs_preserved_summary(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    block_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "preserves_customer_degree_exactly": exact_degree_match(real, synthetic, customer_col),
        "preserves_product_degree_exactly": exact_degree_match(real, synthetic, product_col),
        "preserves_timestamp_multiset_exactly": timestamp_exact_reuse_rate(
            real[timestamp_col], synthetic[timestamp_col]
        )
        == 1.0,
        "has_nontrivial_block_structure": bool(
            (block_diagnostics.get("num_customer_blocks") or 0) > 1
            or (block_diagnostics.get("num_product_blocks") or 0) > 1
        ),
        "generates_timestamps_from_kde": "temporal_sbm" in str(block_diagnostics),
        "reuses_exact_timestamps": timestamp_exact_reuse_rate(
            real[timestamp_col], synthetic[timestamp_col]
        )
        > 0.99,
    }


def interpret_method(
    block_diagnostics: Dict[str, Any],
    learned_vs_preserved: Dict[str, Any],
    method: str,
) -> str:
    if block_diagnostics.get("num_customer_blocks") is None:
        return "no_block_metadata"
    if learned_vs_preserved["has_nontrivial_block_structure"]:
        return "nontrivial_sbm_block_model"
    if (
        learned_vs_preserved["preserves_customer_degree_exactly"]
        and learned_vs_preserved["preserves_product_degree_exactly"]
    ):
        return "global_stub_rewiring"
    if "ipf" in method:
        return "marginal_calibration"
    return "unknown"


def exact_degree_match(real: pd.DataFrame, synthetic: pd.DataFrame, column: str) -> bool:
    real_counts = real[column].value_counts()
    synthetic_counts = synthetic[column].value_counts()
    index = real_counts.index.union(synthetic_counts.index)
    return bool(
        real_counts.reindex(index, fill_value=0).equals(
            synthetic_counts.reindex(index, fill_value=0)
        )
    )


def timestamp_exact_reuse_rate(real_times: pd.Series, synthetic_times: pd.Series) -> float:
    real_counts = pd.to_datetime(real_times).value_counts()
    synthetic_counts = pd.to_datetime(synthetic_times).value_counts()
    if synthetic_counts.sum() == 0:
        return 0.0
    reused = 0
    for timestamp, count in synthetic_counts.items():
        reused += min(int(count), int(real_counts.get(timestamp, 0)))
    return float(reused / synthetic_counts.sum())


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}.{subkey}"] = subvalue
        elif isinstance(value, list):
            flat[key] = json.dumps(value)
        else:
            flat[key] = value
    return flat


if __name__ == "__main__":
    main()
