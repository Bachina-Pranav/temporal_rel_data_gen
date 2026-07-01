#!/usr/bin/env python3
"""Train TemporalNonTextAttributeDiffusionV2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalNonTextAttributeDiffusionV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train non-text temporal attribute diffusion with entity latents."
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--entity-prior-dir", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temporal-split", action="store_true", default=True)
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--effect-noise-std", type=float, default=0.05)
    parser.add_argument("--effect-dropout", type=float, default=0.1)
    parser.add_argument("--model", default="temporal_feature_diffusion_v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model not in {
        "temporal_feature_diffusion_v2",
        "temporal_nontext_attr_diffusion_v2",
    }:
        raise ValueError("V2 currently supports temporal_feature_diffusion_v2.")
    result = TemporalNonTextAttributeDiffusionV2.train_from_csv(
        args.real_reviews,
        output_dir=args.output_dir,
        structure_debug_dir=args.structure_debug_dir,
        entity_prior_dir=args.entity_prior_dir,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        num_cols=args.num_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        effect_noise_std=args.effect_noise_std,
        effect_dropout=args.effect_dropout,
        random_split=args.random_split,
        device=args.device,
        seed=args.seed,
    )
    print(f"best_checkpoint={result.best_checkpoint}")
    print(f"latest_checkpoint={result.latest_checkpoint}")


if __name__ == "__main__":
    main()
