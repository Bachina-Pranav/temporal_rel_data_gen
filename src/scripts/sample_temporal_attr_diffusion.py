#!/usr/bin/env python3
"""Sample review attributes onto an existing temporal review spine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalLatentTextAttributeDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rating/verified/text attributes for a synthetic spine."
    )
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--decoder-mode",
        choices=["nearest_neighbor"],
        default="nearest_neighbor",
    )
    parser.add_argument("--categorical-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = TemporalLatentTextAttributeDiffusion.sample_from_checkpoint(
        synthetic_spine_path=args.synthetic_spine,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=args.device,
        decoder_mode=args.decoder_mode,
        categorical_temperature=args.categorical_temperature,
    )
    print(f"Wrote {len(output):,} full synthetic reviews to {args.output}")


if __name__ == "__main__":
    main()
