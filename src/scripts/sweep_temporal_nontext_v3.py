#!/usr/bin/env python3
"""Sample/evaluate V3 over sampling-time hyperparameter sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, flatten, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402
from reldiff.attributes.temporal_nontext_diffusion import to_jsonable  # noqa: E402
from reldiff.generation.block_diagnostics import load_block_maps_from_debug_dir  # noqa: E402


SUMMARY_KEYS = [
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
    "c2st.c2st_accuracy",
]

DEFAULT_SWEEPS = {
    "calibration_strength": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    "lambda_product_effect": [0.5, 0.75, 1.0],
    "lambda_customer_effect": [0.5, 0.75, 1.0, 1.25],
    "lambda_block": [0.5, 1.0],
    "customer_effect_scale": [1.0, 1.25, 1.5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 sampling/evaluation sweeps.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sweep", choices=sorted(DEFAULT_SWEEPS), default="calibration_strength")
    parser.add_argument("--values", nargs="*", type=float, default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument("--cat-sampling-strategy", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-time-group", choices=["date", "exact", "window"], default="date")
    parser.add_argument("--sampling-window-days", type=float, default=1.0)
    parser.add_argument("--temporal-calibration-strength", type=float, default=0.75)
    parser.add_argument("--no-temporal-calibration", action="store_true")
    parser.add_argument("--debug-use-posterior-effects", action="store_true")
    parser.add_argument("--diagnostic-row-sample-size", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = list(args.values) if args.values else DEFAULT_SWEEPS[args.sweep]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    real = load_reviews(args.real_reviews, args.timestamp_col)
    spine = pd.read_csv(args.synthetic_spine)
    customer_blocks = product_blocks = None
    if args.structure_debug_dir:
        customer_blocks, product_blocks, _, _ = load_block_maps_from_debug_dir(
            args.structure_debug_dir, args.customer_id_col, args.product_id_col
        )
    rows = []
    for value in values:
        row = run_one(args, value, output_root, real, spine, customer_blocks, product_blocks)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "sweep_summary.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "sweep_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output_root / 'sweep_summary.csv'}")


def run_one(
    args: argparse.Namespace,
    value: float,
    output_root: Path,
    real: pd.DataFrame,
    spine: pd.DataFrame,
    customer_blocks: Dict[Any, int] | None,
    product_blocks: Dict[Any, int] | None,
) -> Dict[str, Any]:
    run_name = f"{args.sweep}_{format_value(value)}"
    run_dir = output_root / run_name
    diagnostics_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = TemporalNonTextAttributeDiffusionV3.load_checkpoint(args.checkpoint, device=args.device)
    apply_sweep_value(generator, args.sweep, value)
    calibration_strength = (
        float(value) if args.sweep == "calibration_strength" else float(args.temporal_calibration_strength)
    )
    use_calibration = True if args.sweep == "calibration_strength" else not args.no_temporal_calibration
    synthetic = generator.sample(
        spine,
        structure_debug_dir=args.structure_debug_dir,
        sampled_effects_output_dir=run_dir,
        seed=args.seed,
        num_steps=args.num_diffusion_steps,
        cat_sampling_strategy=args.cat_sampling_strategy,
        temperature=args.temperature,
        sampling_time_group=args.sampling_time_group,
        sampling_window_days=args.sampling_window_days,
        use_temporal_calibration=use_calibration,
        temporal_calibration_strength=calibration_strength,
        debug_use_posterior_effects=args.debug_use_posterior_effects,
        diagnostics_dir=diagnostics_dir,
        diagnostic_row_sample_size=args.diagnostic_row_sample_size,
        device=args.device,
    )
    synthetic_path = run_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    metadata_path = run_dir / "synthetic_review_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(
            to_jsonable(generator.synthetic_metadata(synthetic_path, args.checkpoint, args.seed)),
            handle,
            indent=2,
        )
        handle.write("\n")
    metrics = evaluate_nontext_attrs(
        real,
        synthetic.copy(),
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        num_cols=args.num_cols,
        customer_blocks=customer_blocks,
        product_blocks=product_blocks,
        diagnostics_dir=diagnostics_dir,
    )
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2)
        handle.write("\n")
    flat = flatten(metrics)
    return {
        "run_name": run_name,
        "calibration_strength": calibration_strength,
        "lambda_product_effect": float(generator.lambda_product_effect),
        "lambda_customer_effect": float(generator.lambda_customer_effect),
        "lambda_block": float(generator.lambda_block),
        "customer_effect_scale": float(generator.customer_prior_v3.effect_scale),
        **{key.split(".")[-1]: flat.get(key) for key in SUMMARY_KEYS},
        "metrics_path": str(metrics_path),
        "synthetic_path": str(synthetic_path),
    }


def apply_sweep_value(generator: TemporalNonTextAttributeDiffusionV3, sweep: str, value: float) -> None:
    value = float(value)
    if sweep == "calibration_strength":
        return
    if sweep == "lambda_product_effect":
        generator.lambda_product_effect = value
        return
    if sweep == "lambda_customer_effect":
        generator.lambda_customer_effect = value
        return
    if sweep == "lambda_block":
        generator.lambda_block = value
        return
    if sweep == "customer_effect_scale":
        generator.customer_prior_v3.effect_scale = value
        return
    raise ValueError(f"Unknown sweep {sweep!r}")


def format_value(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
