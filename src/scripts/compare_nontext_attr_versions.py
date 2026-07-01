#!/usr/bin/env python3
"""Compare V1/V2/V3 non-text attribute metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


KEYS = [
    "categorical.rating_distribution_js",
    "categorical.verified_distribution_js",
    "temporal.monthly_average_rating_correlation",
    "temporal.monthly_verified_rate_correlation",
    "temporal.monthly_rating_distribution_js_mean",
    "temporal.monthly_verified_rate_mae",
    "entity_distribution.product_avg_rating_distribution_ks",
    "entity_distribution.customer_avg_rating_distribution_ks",
    "entity_distribution.product_verified_rate_distribution_ks",
    "entity_distribution.customer_verified_rate_distribution_ks",
    "entity_distribution.product_avg_rating_variance_ratio",
    "entity_distribution.customer_avg_rating_variance_ratio",
    "entity_distribution.product_verified_rate_variance_ratio",
    "entity_distribution.customer_verified_rate_variance_ratio",
    "block.block_pair_average_rating_correlation",
    "block.block_pair_verified_rate_correlation",
    "c2st.c2st_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare non-text attr versions.")
    parser.add_argument("--metrics", nargs="+", required=True, help="name=path.json entries")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.metrics:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        with Path(path).open() as handle:
            flat = flatten(json.load(handle))
        rows.append({"model": name, **{key: flat.get(key) for key in KEYS}})
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
