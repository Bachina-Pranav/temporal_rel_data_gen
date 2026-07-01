#!/usr/bin/env python3
"""Compare non-text attribute metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


KEY_METRICS = [
    "categorical.rating_distribution_js",
    "categorical.verified_distribution_js",
    "temporal.monthly_average_rating_correlation",
    "temporal.monthly_verified_rate_correlation",
    "relational.product_average_rating_correlation",
    "relational.customer_average_rating_correlation",
    "relational.product_verified_rate_correlation",
    "relational.customer_verified_rate_correlation",
    "relational.product_rating_trajectory_correlation_top_products",
    "entity_distribution.product_avg_rating_distribution_ks",
    "entity_distribution.customer_avg_rating_distribution_ks",
    "entity_distribution.product_verified_rate_distribution_ks",
    "entity_distribution.customer_verified_rate_distribution_ks",
    "entity_distribution.product_avg_rating_variance_ratio",
    "entity_distribution.customer_avg_rating_variance_ratio",
    "entity_distribution.product_verified_rate_variance_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare non-text attr metrics.")
    parser.add_argument("--metrics", nargs="+", required=True, help="name=path.json entries")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.metrics:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        with Path(path).open() as handle:
            metrics = json.load(handle)
        flat = flatten(metrics)
        row = {"model": name}
        for key in KEY_METRICS:
            row[key] = flat.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    print(df.to_string(index=False))


def flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


if __name__ == "__main__":
    main()
