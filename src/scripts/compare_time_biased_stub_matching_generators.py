#!/usr/bin/env python3
"""Compare time-biased block-stub matching generator outputs or metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine  # noqa: E402


KEYS = [
    "total_seconds",
    "events_per_second",
    "slot_build_seconds",
    "customer_stub_assignment_seconds",
    "product_stub_assignment_seconds",
    "dynamic_pairing_seconds",
    "customer_degree_ks",
    "product_degree_ks",
    "customer_degree_exact_match",
    "product_degree_exact_match",
    "daily_count_l1",
    "daily_count_corr",
    "block_pair_time_count_l1",
    "block_pair_time_exact_match",
    "duplicate_customer_product_rate",
    "real_edge_overlap_rate",
    "exact_event_overlap_rate",
    "product_first_time_corr",
    "product_last_time_corr",
    "product_peak_time_corr",
    "product_relative_age_ks",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "customer_peak_time_corr",
    "customer_relative_age_ks",
    "joint_coactive_window_rate",
    "mean_dynamic_affinity_real",
    "mean_dynamic_affinity_synthetic",
    "dynamic_affinity_distribution_ks",
    "event_tuple_c2st_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare time-biased block-stub matching event generator runs.")
    parser.add_argument("--real-reviews", default="data/original/rel-amazon-toy/review.csv")
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--runs", nargs="+", required=True, help="name=synthetic_review.csv or name=metrics.json entries")
    parser.add_argument("--inputs-are-metrics", action="store_true")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--output-json", default="outputs/amazon-toy/time_biased_block_stub_matching_comparison.json")
    parser.add_argument("--output-csv", default="outputs/amazon-toy/time_biased_block_stub_matching_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nested: Dict[str, Dict[str, Any]] = {}
    rows = []
    real = None if args.inputs_are_metrics else pd.read_csv(args.real_reviews)
    for item in args.runs:
        name, path_text = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        path = Path(path_text)
        if args.inputs_are_metrics:
            with path.open() as handle:
                metrics = json.load(handle)
        else:
            synthetic = pd.read_csv(path)
            metadata_path = path.parent / "metadata.json"
            metadata = None
            if metadata_path.exists():
                with metadata_path.open() as handle:
                    metadata = json.load(handle)
            metrics = evaluate_fast_event_spine(
                real,
                synthetic,
                structure_debug_dir=args.structure_debug_dir or path.parent / "debug",
                customer_col=args.customer_id_col,
                product_col=args.product_id_col,
                timestamp_col=args.timestamp_col,
                compute_c2st=args.compute_c2st,
                metadata=metadata,
            )
        nested[name] = metrics
        rows.append({"model": name, **{key: metrics.get(key) for key in KEYS}})
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(nested, handle, indent=2)
        handle.write("\n")
    table = pd.DataFrame(rows)
    table.to_csv(output_csv, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
