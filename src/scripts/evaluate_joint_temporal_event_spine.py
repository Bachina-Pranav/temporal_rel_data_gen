#!/usr/bin/env python3
"""Evaluate a joint temporal event spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.event_spine_metrics import evaluate_event_spine, write_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate joint temporal event-spine metrics.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    synthetic = pd.read_csv(args.synthetic_reviews)
    metrics = evaluate_event_spine(
        real,
        synthetic,
        structure_debug_dir=args.structure_debug_dir,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        compute_c2st=args.compute_c2st,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
