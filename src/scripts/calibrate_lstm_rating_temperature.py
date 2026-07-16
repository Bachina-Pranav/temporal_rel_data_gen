#!/usr/bin/env python3
"""Fit validation-only scalar temperature for LSTM rating logits."""

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

from attribute_generation.conditional_tabdlm.constrained import mask_invalid_category_logits  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import load_prepared_tables  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_dataset import build_temporal_history_index  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import encode_conditions, load_lstm_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import resolve_device  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate LSTM rating logits using validation data only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    base_config = load_config(args.config)
    device = resolve_device(args.device)
    model, config, vocabs, tokenizer, graph_encoder = load_lstm_checkpoint(args.checkpoint, device=device, include_graph=True)
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()
    train, valid, _ = load_prepared_tables(base_config)
    graph_frame = pd.concat([train, valid], ignore_index=True)
    graph_history = build_temporal_history_index(graph_frame, config, seed=int(args.seed)) if graph_encoder is not None else None
    rating_col = config.schema.categorical_targets[0]
    vocab = vocabs[rating_col]
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for start in range(0, len(valid), int(args.batch_size)):
            frame = valid.iloc[start : start + int(args.batch_size)]
            foreign_key_ids, datetime_values = encode_conditions(frame, config.schema, num_hash_buckets, device)
            graph_context = None
            if graph_encoder is not None and graph_history is not None:
                row_indices = list(range(len(train) + start, len(train) + start + len(frame)))
                graph_context = graph_encoder(graph_history.build_batch(row_indices, device=device, deterministic=True))
            condition = model.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
            noise = torch.zeros(condition.shape[0], model.latent_noise_dim, dtype=condition.dtype, device=condition.device)
            row_latent = model.row_latent(condition, noise=noise)
            logits = mask_invalid_category_logits(model.categorical_logits(row_latent)[rating_col], rating_col, vocab)
            labels = torch.tensor([vocab.encode(value) for value in frame[rating_col]], dtype=torch.long, device=device)
            logits_all.append(logits.detach().float().cpu())
            labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    results = []
    for temp in args.temperatures:
        metrics = calibration_metrics(logits / max(float(temp), 1e-8), labels)
        metrics["temperature"] = float(temp)
        results.append(metrics)
    best = min(results, key=lambda row: row["nll"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "uses_validation_data_only": True,
        "num_validation_rows": int(len(labels)),
        "temperature_grid": [float(value) for value in args.temperatures],
        "best_temperature": float(best["temperature"]),
        "best": best,
        "by_temperature": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=1)
    nll = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    pred = probs.argmax(dim=1)
    confidence = probs.max(dim=1).values
    accuracy = (pred == labels).float()
    return {
        "nll": float(nll.item()),
        "accuracy": float(accuracy.mean().item()),
        "ece": expected_calibration_error(confidence.numpy(), accuracy.numpy()),
        "mean_confidence": float(confidence.mean().item()),
    }


def expected_calibration_error(confidence: np.ndarray, accuracy: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(accuracy[mask])))
    return ece


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
