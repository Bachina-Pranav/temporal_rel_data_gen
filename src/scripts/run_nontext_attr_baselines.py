#!/usr/bin/env python3
"""Run simple non-text attribute baselines for temporal review spines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BASELINES = [
    "marginal_attr_sampler",
    "time_marginal_attr_sampler",
    "product_history_sampler",
    "product_customer_history_sampler",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-text attribute baselines.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--v1-checkpoint", default=None)
    parser.add_argument("--v2-checkpoint", default=None)
    parser.add_argument("--entity-prior-dir", default=None)
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    real = pd.read_csv(args.real_reviews)
    spine = pd.read_csv(args.synthetic_spine)
    real[args.timestamp_col] = pd.to_datetime(real[args.timestamp_col], errors="coerce")
    spine[args.timestamp_col] = pd.to_datetime(spine[args.timestamp_col], errors="coerce")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for baseline in BASELINES:
        synthetic = run_baseline(
            baseline,
            real,
            spine,
            args.customer_id_col,
            args.product_id_col,
            args.timestamp_col,
            args.cat_cols,
            [col for col in args.num_cols if col in real.columns],
            rng,
        )
        path = output_dir / baseline / "synthetic_review.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(path, index=False)
        print(f"Wrote {path}")
    run_model_baselines(args, output_dir)


def run_baseline(
    baseline,
    real,
    spine,
    customer_col,
    product_col,
    timestamp_col,
    cat_cols,
    num_cols,
    rng,
):
    synthetic = spine[[customer_col, product_col, timestamp_col]].copy()
    for col in cat_cols + num_cols:
        if col not in real.columns:
            continue
        values = []
        for _, row in synthetic.iterrows():
            source = select_source(
                baseline, real, row, customer_col, product_col, timestamp_col
            )
            if source.empty:
                source = real
            values.append(sample_column(source[col], rng))
        synthetic[col] = values
    return synthetic


def select_source(baseline, real, row, customer_col, product_col, timestamp_col):
    if baseline == "marginal_attr_sampler":
        return real
    if baseline == "time_marginal_attr_sampler":
        month = pd.Timestamp(row[timestamp_col]).month
        return real[real[timestamp_col].dt.month == month]
    product_hist = real[
        (real[product_col] == row[product_col])
        & (real[timestamp_col] < row[timestamp_col])
    ]
    customer_hist = real[
        (real[customer_col] == row[customer_col])
        & (real[timestamp_col] < row[timestamp_col])
    ]
    if baseline == "product_history_sampler":
        return product_hist
    if baseline == "product_customer_history_sampler":
        both = pd.concat([product_hist, customer_hist], ignore_index=True)
        return both
    return real


def sample_column(values: pd.Series, rng: np.random.Generator):
    values = values.dropna().to_numpy(dtype=object)
    if len(values) == 0:
        return None
    return values[int(rng.integers(0, len(values)))]


def run_model_baselines(args, output_dir: Path) -> None:
    if args.v1_checkpoint:
        from reldiff.attributes import TemporalNonTextAttributeDiffusion

        path = output_dir / "causal_feature_mlp_no_entity_latents" / "synthetic_review.csv"
        TemporalNonTextAttributeDiffusion.sample_from_checkpoint(
            synthetic_spine_path=args.synthetic_spine,
            checkpoint_path=args.v1_checkpoint,
            output_path=path,
            structure_debug_dir=args.structure_debug_dir,
            seed=args.seed,
            num_steps=args.num_diffusion_steps,
        )
        print(f"Wrote {path}")
    if args.v2_checkpoint and args.entity_prior_dir:
        from reldiff.attributes import TemporalNonTextAttributeDiffusionV2

        path = output_dir / "sampled_entity_latent_diffusion" / "synthetic_review.csv"
        TemporalNonTextAttributeDiffusionV2.sample_from_checkpoint(
            synthetic_spine_path=args.synthetic_spine,
            checkpoint_path=args.v2_checkpoint,
            output_path=path,
            structure_debug_dir=args.structure_debug_dir,
            entity_prior_dir=args.entity_prior_dir,
            seed=args.seed,
            num_steps=args.num_diffusion_steps,
        )
        print(f"Wrote {path}")
        ub_path = output_dir / "posterior_effect_upper_bound" / "synthetic_review.csv"
        TemporalNonTextAttributeDiffusionV2.sample_from_checkpoint(
            synthetic_spine_path=args.synthetic_spine,
            checkpoint_path=args.v2_checkpoint,
            output_path=ub_path,
            structure_debug_dir=args.structure_debug_dir,
            entity_prior_dir=args.entity_prior_dir,
            seed=args.seed,
            num_steps=args.num_diffusion_steps,
            debug_use_posterior_effects=True,
        )
        print(f"Wrote {ub_path} (diagnostic upper bound)")


if __name__ == "__main__":
    main()
