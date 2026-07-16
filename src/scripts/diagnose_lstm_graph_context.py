#!/usr/bin/env python3
"""Sampling-only graph-context diagnostic for the joint LSTM attribute generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.graph_dataset import build_temporal_history_index, write_temporal_graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import encode_conditions, load_lstm_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import resolve_device  # noqa: E402
from attribute_generation.conditional_tabdlm.constrained import mask_invalid_category_logits, normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import set_seed  # noqa: E402


MODES = ("correct", "zero", "shuffled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run correct/zero/shuffled graph-context LSTM diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--reference-table", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-rows", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    config = load_config(args.config)
    device = resolve_device(args.device)
    model, ckpt_config, vocabs, tokenizer, graph_encoder = load_lstm_checkpoint(args.checkpoint, device=device, include_graph=True)
    if graph_encoder is None:
        raise ValueError("Checkpoint/config does not have graph conditioning enabled")
    model.eval()
    graph_encoder.eval()
    spine = pd.read_csv(args.spine).head(int(args.num_rows)).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_history = build_temporal_history_index(spine, ckpt_config, seed=int(args.seed))
    write_temporal_graph_metadata(spine, ckpt_config, output_dir / "graph", source="diagnostic_spine", seed=int(args.seed), real_graph_used_at_sampling=False)
    rating_col = ckpt_config.schema.categorical_targets[0]
    outputs = {mode: [] for mode in MODES}
    logits_correct: list[np.ndarray] = []
    logits_zero: list[np.ndarray] = []
    logits_shuffled: list[np.ndarray] = []
    num_hash_buckets = int(ckpt_config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    with torch.no_grad():
        for start in range(0, len(spine), int(args.batch_size)):
            frame = spine.iloc[start : start + int(args.batch_size)]
            foreign_key_ids, datetime_values = encode_conditions(frame, ckpt_config.schema, num_hash_buckets, device)
            row_indices = list(range(start, start + len(frame)))
            correct = graph_encoder(graph_history.build_batch(row_indices, device=device, deterministic=True))
            contexts = {
                "correct": correct,
                "zero": torch.zeros_like(correct),
                "shuffled": correct[torch.randperm(correct.shape[0], device=correct.device)] if correct.shape[0] > 1 else correct,
            }
            for mode_idx, mode in enumerate(MODES):
                set_seed(int(args.seed) + mode_idx * 100_000 + start)
                generated = model.generate(
                    foreign_key_ids,
                    datetime_values,
                    vocabs,
                    tokenizer,
                    graph_context=contexts[mode],
                    temperature=float(ckpt_config.raw.get("sampling", {}).get("temperature", 0.9)),
                    top_p=float(ckpt_config.raw.get("sampling", {}).get("top_p", 0.95)),
                )
                outputs[mode].extend(generated["categorical"][rating_col])
            logits_correct.append(deterministic_logits(model, foreign_key_ids, datetime_values, contexts["correct"], rating_col, vocabs[rating_col]))
            logits_zero.append(deterministic_logits(model, foreign_key_ids, datetime_values, contexts["zero"], rating_col, vocabs[rating_col]))
            logits_shuffled.append(deterministic_logits(model, foreign_key_ids, datetime_values, contexts["shuffled"], rating_col, vocabs[rating_col]))
    summary = {
        "checkpoint": str(args.checkpoint),
        "spine": str(args.spine),
        "num_rows": int(len(spine)),
        "rating_col": rating_col,
        "modes": {},
        "logit_delta": {
            "correct_vs_zero": logit_delta_metrics(logits_correct, logits_zero, rating_values_for_vocab(vocabs[rating_col])),
            "correct_vs_shuffled": logit_delta_metrics(logits_correct, logits_shuffled, rating_values_for_vocab(vocabs[rating_col])),
        },
    }
    reference = pd.read_csv(args.reference_table) if args.reference_table else None
    for mode in MODES:
        output = spine.loc[:, [column for column in ["event_id", *ckpt_config.schema.condition_columns] if column in spine.columns]].copy()
        output[rating_col] = outputs[mode]
        path = output_dir / mode / "synthetic_interactions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        summary["modes"][mode] = {
            "output_path": str(path),
            "rating_distribution": normalized_counts(output[rating_col]),
        }
        if reference is not None and rating_col in reference:
            summary["modes"][mode]["rating_total_variation_vs_reference"] = total_variation(reference[rating_col], output[rating_col])
        print(f"Wrote {path}")
    summary_path = output_dir / "graph_context_diagnostic.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


def deterministic_logits(model, foreign_key_ids, datetime_values, graph_context, rating_col: str, vocab) -> np.ndarray:
    condition = model.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
    noise = torch.zeros(condition.shape[0], model.latent_noise_dim, dtype=condition.dtype, device=condition.device)
    row = model.row_latent(condition, noise=noise)
    logits = mask_invalid_category_logits(model.categorical_logits(row)[rating_col], rating_col, vocab)
    return logits.detach().float().cpu().numpy()


def mean_abs_delta(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    a = np.concatenate(left, axis=0)
    b = np.concatenate(right, axis=0)
    return float(np.mean(np.abs(a - b)))


def logit_delta_metrics(left: list[np.ndarray], right: list[np.ndarray], rating_values: list[float]) -> dict[str, float]:
    a = np.concatenate(left, axis=0)
    b = np.concatenate(right, axis=0)
    pa = softmax_np(a)
    pb = softmax_np(b)
    values = np.asarray(rating_values, dtype=float)
    if values.shape[0] != pa.shape[1]:
        values = np.arange(pa.shape[1], dtype=float)
    kl = np.sum(pa * (np.log(np.clip(pa, 1e-12, None)) - np.log(np.clip(pb, 1e-12, None))), axis=1)
    expected_a = pa @ values
    expected_b = pb @ values
    return {
        "mean_abs_logit_difference": float(np.mean(np.abs(a - b))),
        "mean_kl_divergence": float(np.mean(kl)),
        "argmax_changed_fraction": float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1))),
        "mean_abs_expected_rating_change": float(np.mean(np.abs(expected_a - expected_b))),
    }


def rating_values_for_vocab(vocab) -> list[float]:
    values = []
    valid = [normalize_rating_value(vocab.decode(idx)) for idx in range(vocab.size)]
    fallback = [float(value) for value in valid if value is not None]
    fallback_value = float(np.mean(fallback)) if fallback else 0.0
    for value in valid:
        values.append(float(value) if value is not None else fallback_value)
    return values


def softmax_np(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def normalized_counts(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts(normalize=True).sort_index()
    return {str(key): float(value) for key, value in counts.items()}


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    r = normalized_counts(real)
    s = normalized_counts(synthetic)
    keys = sorted(set(r) | set(s))
    return float(0.5 * sum(abs(r.get(key, 0.0) - s.get(key, 0.0)) for key in keys))


if __name__ == "__main__":
    main()
