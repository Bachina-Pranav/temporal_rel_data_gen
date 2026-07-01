#!/usr/bin/env python3
"""Train temporal non-text attribute diffusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalNonTextAttributeDiffusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train non-text attribute diffusion conditioned on review spines."
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["temporal_feature_diffusion", "temporal_gnn_feature_diffusion"],
        default="temporal_feature_diffusion",
    )
    parser.add_argument("--temporal-split", action="store_true", default=True)
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--lambda-cat", type=float, default=1.0)
    parser.add_argument("--lambda-num", type=float, default=1.0)
    parser.add_argument("--mask-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "temporal_gnn_feature_diffusion":
        raise NotImplementedError(
            "temporal_gnn_feature_diffusion is reserved for a future extension; "
            "use --model temporal_feature_diffusion for v1."
        )
    result = TemporalNonTextAttributeDiffusion.train_from_csv(
        args.real_reviews,
        output_dir=args.output_dir,
        structure_debug_dir=args.structure_debug_dir,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        num_cols=args.num_cols,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        validation_fraction=args.validation_fraction,
        random_split=args.random_split,
        lambda_cat=args.lambda_cat,
        lambda_num=args.lambda_num,
        mask_schedule=args.mask_schedule,
        device=args.device,
    )
    print(f"Wrote best checkpoint to {result.best_checkpoint}")
    print(f"Wrote latest checkpoint to {result.latest_checkpoint}")


if __name__ == "__main__":
    main()
