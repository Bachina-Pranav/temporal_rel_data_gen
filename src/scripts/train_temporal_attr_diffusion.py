#!/usr/bin/env python3
"""Train temporal latent-text attribute diffusion for review attributes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes import TemporalLatentTextAttributeDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train attribute diffusion conditioned on temporal review spines."
    )
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--text-cols", nargs="+", default=["summary", "review_text"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--text-encoder-backend",
        choices=["auto", "sentence_transformers", "transformers", "hashing"],
        default="auto",
    )
    parser.add_argument(
        "--text-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text-latent-dim", type=int, default=384)
    parser.add_argument("--temporal-window-days", type=float, default=365.0)
    parser.add_argument("--max-customer-history", type=int, default=32)
    parser.add_argument("--max-product-history", type=int, default=32)
    parser.add_argument(
        "--temporal-mode",
        choices=["causal", "causal_window", "symmetric_window"],
        default="causal_window",
    )
    parser.add_argument("--lambda-cat", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--lambda-summary", type=float, default=0.5)
    parser.add_argument("--lambda-review-text", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--force-recompute-text-latents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = TemporalLatentTextAttributeDiffusion.train_from_csv(
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        text_cols=args.text_cols,
        temporal_window_days=args.temporal_window_days,
        max_customer_history=args.max_customer_history,
        max_product_history=args.max_product_history,
        temporal_mode=args.temporal_mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        text_encoder_backend=args.text_encoder_backend,
        text_model_name=args.text_model_name,
        text_latent_dim=args.text_latent_dim,
        lambda_cat=args.lambda_cat,
        lambda_text=args.lambda_text,
        lambda_summary=args.lambda_summary,
        lambda_review_text=args.lambda_review_text,
        validation_fraction=args.validation_fraction,
        force_recompute_text_latents=args.force_recompute_text_latents,
        device=args.device,
    )
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Latest checkpoint: {result.latest_checkpoint}")


if __name__ == "__main__":
    main()
