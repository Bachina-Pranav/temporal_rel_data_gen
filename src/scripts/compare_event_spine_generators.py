#!/usr/bin/env python3
"""Compare event-spine generator metric JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


KEYS = [
    "customer_degree_ks",
    "product_degree_ks",
    "daily_count_corr",
    "block_pair_time_count_l1",
    "product_first_time_corr",
    "product_last_time_corr",
    "product_peak_time_corr",
    "product_relative_age_ks",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "joint_coactive_window_rate",
    "event_tuple_c2st_accuracy",
    "real_edge_overlap_rate",
    "exact_event_overlap_rate",
    "duplicate_customer_product_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare event spine generators.")
    parser.add_argument("--metrics", nargs="+", required=True, help="name=metrics.json entries")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    nested: Dict[str, Dict[str, Any]] = {}
    for item in args.metrics:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        with Path(path).open() as handle:
            metrics = json.load(handle)
        nested[name] = metrics
        rows.append({"model": name, **{key: metrics.get(key) for key in KEYS}})
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(nested, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
