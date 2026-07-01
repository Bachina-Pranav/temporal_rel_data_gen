#!/usr/bin/env python3
"""Sample V3 non-text attributes onto a temporal spine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample V3 non-text attributes.")
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
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
    parser.add_argument("--use-temporal-calibration", action="store_true")
    parser.add_argument("--temporal-calibration-strength", type=float, default=0.75)
    parser.add_argument("--debug-use-posterior-effects", action="store_true")
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--diagnostic-row-sample-size", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = TemporalNonTextAttributeDiffusionV3.sample_from_checkpoint(
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
        use_temporal_calibration=args.use_temporal_calibration,
        temporal_calibration_strength=args.temporal_calibration_strength,
        debug_use_posterior_effects=args.debug_use_posterior_effects,
        diagnostics_dir=args.diagnostics_dir,
        diagnostic_row_sample_size=args.diagnostic_row_sample_size,
        device=args.device,
    )
    print(f"Wrote {args.output} ({len(output)} rows)")
    print(f"Wrote {Path(args.output).with_name(Path(args.output).stem + '_metadata.json')}")
    print(f"Wrote sampled V3 effects in {Path(args.output).parent}")
    if args.diagnostics_dir:
        print(f"Wrote V3 diagnostics in {args.diagnostics_dir}")


if __name__ == "__main__":
    main()
