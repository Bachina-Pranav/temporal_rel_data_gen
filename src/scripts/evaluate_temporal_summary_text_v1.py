#!/usr/bin/env python3
"""Evaluate Text V1 generated summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textgen.text_eval import evaluate_summary_text_v1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal summary Text V1.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--text-col", default="summary")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--privacy-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    synthetic = pd.read_csv(args.synthetic_reviews)
    if args.text_col not in real.columns:
        raise ValueError(f"Real reviews missing text column {args.text_col!r}")
    if args.text_col not in synthetic.columns:
        raise ValueError(f"Synthetic reviews missing text column {args.text_col!r}")
    metrics = evaluate_summary_text_v1(
        real,
        synthetic,
        text_col=args.text_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
        timestamp_col=args.timestamp_col,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        privacy_sample_size=args.privacy_sample_size,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    pd.json_normalize(metrics).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
