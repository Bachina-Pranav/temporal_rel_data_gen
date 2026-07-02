#!/usr/bin/env python3
"""Train Text V1 masked summary generator."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textgen.masked_summary_dataset import MaskedSummaryDataset, SimpleSummaryTokenizer  # noqa: E402
from textgen.masked_text_diffusion import TemporalSummaryMaskedDiffusionV1  # noqa: E402
from textgen.temporal_text_v1 import METHOD_ALIAS_TEXT_V1, METHOD_NAME_TEXT_V1, save_text_v1_checkpoint  # noqa: E402
from textgen.text_conditioning import ConditionFeatureNormalizer, build_text_condition_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal summary Text V1.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--text-col", default="summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="distilbert-base-uncased")
    parser.add_argument("--max-summary-tokens", type=int, default=32)
    parser.add_argument("--num-condition-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-token-frequency", type=int, default=1)
    parser.add_argument("--min-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-mask-prob", type=float, default=0.85)
    parser.add_argument("--mask-schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_csv(args.real_reviews)
    required = [
        args.customer_id_col,
        args.product_id_col,
        args.timestamp_col,
        args.rating_col,
        args.verified_col,
        args.text_col,
    ]
    missing = [col for col in required if col not in reviews.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    split = temporal_split(reviews, args.timestamp_col)
    features = build_text_condition_features(
        reviews,
        structure_debug_dir=args.structure_debug_dir,
        mode="train",
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
    )
    tokenizer = SimpleSummaryTokenizer().fit(
        split["train"][args.text_col].fillna(""),
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_token_frequency,
    )
    normalizer = ConditionFeatureNormalizer().fit(features.features.iloc[split["train_index"]])
    condition_matrix = normalizer.transform(features.features)
    train_dataset = make_dataset(args, split["train"], condition_matrix[split["train_index"]], tokenizer, seed_offset=0)
    val_dataset = make_dataset(args, split["val"], condition_matrix[split["val_index"]], tokenizer, seed_offset=100000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = TemporalSummaryMaskedDiffusionV1(
        vocab_size=tokenizer.vocab_size,
        condition_dim=condition_matrix.shape[1],
        max_summary_tokens=args.max_summary_tokens,
        num_condition_tokens=args.num_condition_tokens,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_config = vars(args).copy()
    train_config.update(
        {
            "method": METHOD_NAME_TEXT_V1,
            "method_alias": METHOD_ALIAS_TEXT_V1,
            "backbone_requested": args.backbone,
            "backbone_used": "lightweight_masked_summary_transformer",
            "no_nearest_neighbor_decoding": True,
            "no_text_retrieval": True,
            "contains_training_text_bank": False,
            "feature_metadata": features.metadata,
        }
    )
    with (output_dir / "train_config.json").open("w") as handle:
        json.dump(train_config, handle, indent=2)
        handle.write("\n")
    tokenizer.save(output_dir / "tokenizer.json")
    normalizer.save(output_dir / "condition_normalizer.json")

    history = []
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics = run_epoch(model, val_loader, None, args.device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        print(json.dumps(row))
        save_text_v1_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            tokenizer,
            normalizer,
            train_config,
            history,
            epoch,
            val_metrics,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            save_text_v1_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                tokenizer,
                normalizer,
                train_config,
                history,
                epoch,
                val_metrics,
            )
        else:
            stale += 1
        if stale >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break
    print(f"Wrote best checkpoint to {checkpoint_dir / 'best.pt'}")


def make_dataset(args: argparse.Namespace, frame: pd.DataFrame, features: np.ndarray, tokenizer: SimpleSummaryTokenizer, seed_offset: int) -> MaskedSummaryDataset:
    return MaskedSummaryDataset(
        frame,
        tokenizer,
        features,
        text_col=args.text_col,
        max_summary_tokens=args.max_summary_tokens,
        min_mask_prob=args.min_mask_prob,
        max_mask_prob=args.max_mask_prob,
        mask_schedule=args.mask_schedule,
        seed=args.seed + int(seed_offset),
    )


def run_epoch(model, loader: DataLoader, optimizer, device: str) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items() if key != "row_id"}
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output["loss"]
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        labels = batch["labels"]
        masked = labels != -100
        pred = output["logits"].argmax(dim=-1)
        total_correct += int((pred[masked] == labels[masked]).sum().detach().cpu()) if bool(masked.any()) else 0
        total_masked += int(masked.sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * max(int(masked.sum().detach().cpu()), 1)
    denom = max(total_masked, 1)
    mean_loss = total_loss / denom
    return {
        "loss": float(mean_loss),
        "masked_token_accuracy": float(total_correct / denom),
        "perplexity_on_masked_tokens": float(math.exp(min(mean_loss, 20.0))),
    }


def temporal_split(reviews: pd.DataFrame, timestamp_col: str) -> Dict[str, Any]:
    ordered = reviews.assign(_ts=pd.to_datetime(reviews[timestamp_col], errors="coerce")).sort_values("_ts")
    indices = ordered.index.to_numpy()
    n = len(indices)
    train_end = max(1, int(0.8 * n))
    val_end = max(train_end + 1, int(0.9 * n)) if n > 2 else n
    val_end = min(val_end, n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end] if train_end < val_end else indices[:train_end]
    test_idx = indices[val_end:] if val_end < n else indices[train_end:val_end]
    return {
        "train": reviews.loc[train_idx].reset_index(drop=True),
        "val": reviews.loc[val_idx].reset_index(drop=True),
        "test": reviews.loc[test_idx].reset_index(drop=True),
        "train_index": train_idx,
        "val_index": val_idx,
        "test_index": test_idx,
    }


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


if __name__ == "__main__":
    main()
