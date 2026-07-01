#!/usr/bin/env python3
"""Sample non-text attributes onto a synthetic temporal spine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalNonTextAttributeDiffusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample rating/verified/numerical attrs onto a review spine."
    )
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument(
        "--cat-sampling-strategy", choices=["sample", "argmax"], default="sample"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling-time-group", choices=["date", "exact", "window"], default="date"
    )
    parser.add_argument("--sampling-window-days", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic = TemporalNonTextAttributeDiffusion.sample_from_checkpoint(
        synthetic_spine_path=args.synthetic_spine,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        structure_debug_dir=args.structure_debug_dir,
        seed=args.seed,
        num_steps=args.num_diffusion_steps,
        cat_sampling_strategy=args.cat_sampling_strategy,
        temperature=args.temperature,
        sampling_time_group=args.sampling_time_group,
        sampling_window_days=args.sampling_window_days,
        device=args.device,
    )
    print(f"Wrote {len(synthetic):,} rows to {args.output}")


if __name__ == "__main__":
    main()
