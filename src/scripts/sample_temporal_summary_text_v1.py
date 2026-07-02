#!/usr/bin/env python3
"""Sample Text V1 summaries for synthetic reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textgen.temporal_text_v1 import load_text_v1_checkpoint, write_text_v1_metadata  # noqa: E402
from textgen.text_conditioning import build_text_condition_features  # noqa: E402
from textgen.text_sampling import sample_summaries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample summary text from a Text V1 checkpoint. Does not accept real reviews."
    )
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-summary-tokens", type=int, default=32)
    parser.add_argument("--num-denoising-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic = pd.read_csv(args.synthetic_reviews)
    required = [
        args.customer_id_col,
        args.product_id_col,
        args.timestamp_col,
        args.rating_col,
        args.verified_col,
    ]
    missing = [col for col in required if col not in synthetic.columns]
    if missing:
        raise ValueError(f"Missing required synthetic columns: {missing}")
    loaded = load_text_v1_checkpoint(args.checkpoint, device=args.device)
    checkpoint = loaded["checkpoint"]
    if checkpoint.get("contains_training_text_bank", False):
        raise RuntimeError("Refusing to sample from a checkpoint that contains a training text bank.")
    sampled_effects_dir = Path(args.synthetic_reviews).parent
    features = build_text_condition_features(
        synthetic,
        structure_debug_dir=args.structure_debug_dir,
        mode="sample",
        sampled_effects_dir=sampled_effects_dir,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
    )
    condition_matrix = loaded["normalizer"].transform(features.features)
    summaries = sample_summaries(
        loaded["model"],
        loaded["tokenizer"],
        condition_matrix,
        max_summary_tokens=args.max_summary_tokens,
        num_denoising_steps=args.num_denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    output = synthetic.copy()
    output["summary"] = summaries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    write_text_v1_metadata(
        output_path.with_name(output_path.stem + "_metadata.json"),
        args.checkpoint,
        output_path,
        args.seed,
        decoding_strategy="iterative_masked_denoising_confidence_topk",
        max_summary_tokens=args.max_summary_tokens,
    )
    print(f"Wrote {len(output):,} rows to {output_path}")
    print(f"Wrote {output_path.with_name(output_path.stem + '_metadata.json')}")


if __name__ == "__main__":
    main()
