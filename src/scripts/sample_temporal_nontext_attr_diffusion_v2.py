#!/usr/bin/env python3
"""Sample V2 non-text attributes onto a generated temporal spine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalNonTextAttributeDiffusionV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample rating/verified attrs with generated entity latents."
    )
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--entity-prior-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument("--cat-sampling-strategy", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-time-group", choices=["date", "exact", "window"], default="date")
    parser.add_argument("--sampling-window-days", type=float, default=1.0)
    parser.add_argument("--debug-use-posterior-effects", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = TemporalNonTextAttributeDiffusionV2.sample_from_checkpoint(
        synthetic_spine_path=args.synthetic_spine,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        structure_debug_dir=args.structure_debug_dir,
        entity_prior_dir=args.entity_prior_dir,
        seed=args.seed,
        num_steps=args.num_diffusion_steps,
        cat_sampling_strategy=args.cat_sampling_strategy,
        temperature=args.temperature,
        sampling_time_group=args.sampling_time_group,
        sampling_window_days=args.sampling_window_days,
        debug_use_posterior_effects=args.debug_use_posterior_effects,
        device=args.device,
    )
    print(f"Wrote {args.output} ({len(output)} rows)")
    print(f"Wrote {Path(args.output).with_name(Path(args.output).stem + '_metadata.json')}")
    print(f"Wrote sampled effects in {Path(args.output).parent}")


if __name__ == "__main__":
    main()
