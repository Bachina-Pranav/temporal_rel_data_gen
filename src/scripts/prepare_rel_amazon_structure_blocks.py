#!/usr/bin/env python3
"""Prepare normalized customer/product block files for Rel-Amazon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full Rel-Amazon structure block CSVs.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-debug-dir", required=True)
    parser.add_argument("--num-customer-blocks", type=int, default=5)
    parser.add_argument("--num-product-blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.real_reviews)
    required = [args.customer_id_col, args.product_id_col, args.timestamp_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(frame.columns)}", file=sys.stderr)
        raise SystemExit(1)
    output = Path(args.output_debug_dir)
    output.mkdir(parents=True, exist_ok=True)
    customer_blocks = degree_quantile_blocks(frame[args.customer_id_col], args.num_customer_blocks, args.seed)
    product_blocks = degree_quantile_blocks(frame[args.product_id_col], args.num_product_blocks, args.seed)
    pd.DataFrame(
        {
            args.customer_id_col: list(customer_blocks.keys()),
            "customer_block": list(customer_blocks.values()),
        }
    ).to_csv(output / "customer_blocks.csv", index=False)
    pd.DataFrame(
        {
            args.product_id_col: list(product_blocks.keys()),
            "product_block": list(product_blocks.values()),
        }
    ).to_csv(output / "product_blocks.csv", index=False)
    summary = {
        "method": "degree_quantile_blocks",
        "note": "Used only to prepare required structure-debug block files without running synthetic generation.",
        "num_customer_blocks_requested": int(args.num_customer_blocks),
        "num_product_blocks_requested": int(args.num_product_blocks),
        "num_customer_blocks_written": int(len(set(customer_blocks.values()))),
        "num_product_blocks_written": int(len(set(product_blocks.values()))),
        "num_customers": int(len(customer_blocks)),
        "num_products": int(len(product_blocks)),
        "seed": int(args.seed),
    }
    with (output / "structure_block_preparation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {output / 'customer_blocks.csv'}")
    print(f"[done] wrote {output / 'product_blocks.csv'}")


def degree_quantile_blocks(entity_series: pd.Series, num_blocks: int, seed: int) -> Dict[Any, int]:
    del seed
    degrees = entity_series.value_counts()
    num_blocks = max(int(num_blocks), 1)
    if len(degrees) == 0:
        return {}
    ordered = (
        pd.DataFrame({"entity": degrees.index.to_numpy(dtype=object), "degree": degrees.to_numpy(dtype=float)})
        .assign(_entity_text=lambda df: df["entity"].astype(str))
        .sort_values(["degree", "_entity_text"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranks = np.arange(len(ordered), dtype=np.int64)
    blocks = np.floor(ranks * num_blocks / max(len(ordered), 1)).astype(int)
    blocks = np.minimum(blocks, num_blocks - 1)
    return {entity: int(block) for entity, block in zip(ordered["entity"], blocks)}


if __name__ == "__main__":
    main()
