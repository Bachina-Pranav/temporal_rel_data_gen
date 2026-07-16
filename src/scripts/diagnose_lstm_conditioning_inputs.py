#!/usr/bin/env python3
"""Sampling-only user/movie/time/graph conditioning diagnostics for LSTM ratings."""

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

from attribute_generation.conditional_tabdlm.constrained import mask_invalid_category_logits, normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_dataset import build_temporal_history_index  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import encode_conditions, load_lstm_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.train import resolve_device  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import set_seed  # noqa: E402


MODES = ("correct", "shuffled_user", "shuffled_movie", "shuffled_time", "zero_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose LSTM dependence on user/movie/time/graph inputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-rows", default="15000", help="Number of spine rows to use, or 'all'.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = resolve_device(args.device)
    model, config, vocabs, tokenizer, graph_encoder = load_lstm_checkpoint(args.checkpoint, device=device, include_graph=True)
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()
    num_rows = parse_num_rows(args.num_rows)
    spine = pd.read_csv(args.spine)
    if num_rows is not None:
        spine = spine.head(num_rows)
    spine = spine.reset_index(drop=True)
    if len(config.schema.foreign_key_columns) < 2:
        raise ValueError("Conditioning diagnostic requires two foreign-key columns")
    user_col, movie_col = config.schema.foreign_key_columns[:2]
    time_col = config.schema.datetime_columns[0]
    rating_col = config.schema.categorical_targets[0]
    frames = {
        "correct": spine.copy(),
        "shuffled_user": shuffled_column(spine, user_col, rng),
        "shuffled_movie": shuffled_column(spine, movie_col, rng),
        "shuffled_time": shuffled_column(spine, time_col, rng),
        "zero_graph": spine.copy(),
    }
    graph_history = build_temporal_history_index(spine, config, seed=int(args.seed)) if graph_encoder is not None else None
    logits_by_mode = {mode: [] for mode in MODES}
    samples_by_mode = {mode: [] for mode in MODES}
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    with torch.no_grad():
        for start in range(0, len(spine), int(args.batch_size)):
            row_indices = list(range(start, min(start + int(args.batch_size), len(spine))))
            graph_context = None
            if graph_encoder is not None and graph_history is not None:
                graph_context = graph_encoder(graph_history.build_batch(row_indices, device=device, deterministic=True))
            for mode in MODES:
                frame = frames[mode].iloc[start : start + len(row_indices)]
                foreign_key_ids, datetime_values = encode_conditions(frame, config.schema, num_hash_buckets, device)
                context = graph_context
                if mode == "zero_graph" and graph_context is not None:
                    context = torch.zeros_like(graph_context)
                logits = deterministic_rating_logits(model, foreign_key_ids, datetime_values, context, rating_col, vocabs[rating_col])
                logits_by_mode[mode].append(logits)
                generated = model.generate(
                    foreign_key_ids,
                    datetime_values,
                    vocabs,
                    tokenizer,
                    graph_context=context,
                    temperature=float(config.raw.get("sampling", {}).get("temperature", 0.9)),
                    top_p=float(config.raw.get("sampling", {}).get("top_p", 0.95)),
                )
                samples_by_mode[mode].extend(generated["categorical"][rating_col])
    summary = {
        "checkpoint": str(args.checkpoint),
        "spine": str(args.spine),
        "num_rows": int(len(spine)),
        "modes": {},
        "logit_delta_vs_correct": {},
    }
    correct_logits = logits_by_mode["correct"]
    rating_values = rating_values_for_vocab(vocabs[rating_col])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        output = frames[mode].loc[:, [column for column in ["event_id", *config.schema.condition_columns] if column in frames[mode].columns]].copy()
        output[rating_col] = samples_by_mode[mode]
        path = output_dir / mode / "synthetic_interactions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        summary["modes"][mode] = {
            "output_path": str(path),
            "rating_distribution": normalized_counts(output[rating_col]),
        }
        if mode != "correct":
            summary["logit_delta_vs_correct"][mode] = logit_delta_metrics(correct_logits, logits_by_mode[mode], rating_values)
    summary_path = output_dir / "conditioning_input_diagnostic.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)


def parse_num_rows(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "all":
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("--num-rows must be positive or 'all'")
    return parsed


def shuffled_column(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    out[column] = rng.permutation(out[column].to_numpy(copy=True))
    return out


def deterministic_rating_logits(model, foreign_key_ids, datetime_values, graph_context, rating_col: str, vocab) -> np.ndarray:
    condition = model.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
    noise = torch.zeros(condition.shape[0], model.latent_noise_dim, dtype=condition.dtype, device=condition.device)
    row = model.row_latent(condition, noise=noise)
    logits = mask_invalid_category_logits(model.categorical_logits(row)[rating_col], rating_col, vocab)
    return logits.detach().float().cpu().numpy()


def logit_delta_metrics(left: list[np.ndarray], right: list[np.ndarray], rating_values: list[float]) -> dict[str, float]:
    a = np.concatenate(left, axis=0)
    b = np.concatenate(right, axis=0)
    pa = softmax_np(a)
    pb = softmax_np(b)
    values = np.asarray(rating_values, dtype=float)
    kl = np.sum(pa * (np.log(np.clip(pa, 1e-12, None)) - np.log(np.clip(pb, 1e-12, None))), axis=1)
    finite_pair_mask = np.isfinite(a) & np.isfinite(b)
    finite_abs_delta = np.abs(a[finite_pair_mask] - b[finite_pair_mask])
    return {
        "mean_abs_logit_difference": float(np.mean(finite_abs_delta)) if finite_abs_delta.size else None,
        "mean_kl_divergence": float(np.mean(kl)),
        "argmax_changed_fraction": float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1))),
        "mean_abs_expected_rating_change": float(np.mean(np.abs((pa @ values) - (pb @ values)))),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def rating_values_for_vocab(vocab) -> list[float]:
    valid = [normalize_rating_value(vocab.decode(idx)) for idx in range(vocab.size)]
    fallback = [float(value) for value in valid if value is not None]
    fallback_value = float(np.mean(fallback)) if fallback else 0.0
    return [float(value) if value is not None else fallback_value for value in valid]


def normalized_counts(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts(normalize=True).sort_index()
    return {str(key): float(value) for key, value in counts.items()}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
