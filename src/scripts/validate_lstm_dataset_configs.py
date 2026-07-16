#!/usr/bin/env python3
"""Smoke-test LSTM dataset configs without full training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
    load_category_vocabs,
    load_numerical_metadata,
    load_prepared_tables,
    load_text_tokenizer,
)
from attribute_generation.conditional_tabdlm.graph_dataset import build_temporal_history_index  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_schema import graph_conditioning_enabled  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    build_lstm_model,
    encode_conditions,
    lstm_joint_loss,
    make_lstm_collate_fn,
    move_batch_to_device,
)
from attribute_generation.conditional_tabdlm.numerical import inverse_transform_numerical, sample_gaussian_params  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import build_graph_encoder, compute_graph_outputs, resolve_device  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import save_json, set_seed  # noqa: E402


CONFIGS = {
    "amazon_toy": "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_lstm_joint_full_review_text.yaml",
    "movielens_100k": "configs/attribute_generation/lstm_movielens_100k.yaml",
    "yelp_100k": "configs/attribute_generation/lstm_yelp_100k.yaml",
    "retailrocket_100k": "configs/attribute_generation/lstm_retailrocket_100k.yaml",
    "hm_100k": "configs/attribute_generation/lstm_hm_100k.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LSTM configs on tiny batches.")
    parser.add_argument("--datasets", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="outputs/lstm_dataset_config_validation.json")
    parser.add_argument("--run-forward-pass", action="store_true")
    parser.add_argument("--run-loss-pass", action="store_true")
    parser.add_argument("--run-sampling-smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    reports = {}
    failed = False
    for name in args.datasets:
        try:
            reports[name] = validate_one(name, CONFIGS[name], args, device)
        except Exception as exc:
            reports[name] = {"status": "failed", "error": str(exc)}
            failed = True
        print(json.dumps({name: reports[name]}, sort_keys=True, default=str))
    save_json(reports, args.output)
    if failed:
        raise SystemExit(1)


def validate_one(name: str, config_path: str, args: argparse.Namespace, device: str) -> dict[str, Any]:
    set_seed(17)
    config = load_config(config_path)
    train_frame, _, _ = load_prepared_tables(config)
    categorical_vocabs = load_category_vocabs(config)
    tokenizer = load_text_tokenizer(config)
    numerical_metadata = load_numerical_metadata(config)
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    dataset = ConditionalTABDLMDataset(
        train_frame.head(max(int(args.batch_size), 2)).copy(),
        config.schema,
        categorical_vocabs,
        tokenizer,
        num_hash_buckets,
        numerical_metadata=numerical_metadata,
    )
    loader = DataLoader(dataset, batch_size=min(int(args.batch_size), len(dataset)), shuffle=False, collate_fn=make_lstm_collate_fn)
    batch = move_batch_to_device(next(iter(loader)), device)
    model = build_lstm_model(config, categorical_vocabs, tokenizer).to(device)
    graph_encoder = None
    graph_history = None
    graph_context = None
    if graph_conditioning_enabled(config.raw):
        graph_encoder = build_graph_encoder(config, categorical_vocabs, tokenizer).to(device)
        graph_history = build_temporal_history_index(train_frame.head(len(dataset)).copy(), config, seed=17)
        graph_context, _ = compute_graph_outputs(graph_encoder, graph_history, batch, device, deterministic=True, config=config, training=True)
    report: dict[str, Any] = {
        "status": "ok",
        "config": config_path,
        "enabled_heads": {
            "categorical": list(config.schema.model_categorical_targets),
            "numerical": list(config.schema.numerical_targets),
            "text": list(config.schema.text_targets),
        },
        "tensor_shapes": {
            "foreign_key_ids": list(batch["foreign_key_ids"].shape),
            "datetime_values": list(batch["datetime_values"].shape),
            "categorical_ids": list(batch["categorical_ids"].shape),
            "numerical_values": list(batch["numerical_values"].shape),
            "text_ids": {column: list(value.shape) for column, value in batch["text_ids"].items()},
            "graph_context": list(graph_context.shape) if graph_context is not None else None,
        },
    }
    logits = None
    if args.run_forward_pass or args.run_loss_pass:
        logits = model(batch["foreign_key_ids"], batch["datetime_values"], batch["categorical_ids"], batch["text_ids"], graph_context=graph_context)
        report["forward_pass"] = "ok"
    if args.run_loss_pass:
        loss, component = lstm_joint_loss(
            logits,
            batch,
            config.schema,
            dict(config.raw.get("loss_weights", {})),
            tokenizer,
            {},
            config=config,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        report["loss_pass"] = "ok"
        report["loss"] = float(loss.detach().cpu())
        report["per_field_losses"] = component
    if args.run_sampling_smoke_test:
        smoke = run_sampling_smoke(model, config, categorical_vocabs, tokenizer, numerical_metadata, train_frame.head(2), graph_context[:2] if graph_context is not None else None, device)
        report["sampling_smoke_test"] = "ok"
        report["sampling_output_columns"] = list(smoke.columns)
    return report


def run_sampling_smoke(model, config, vocabs, tokenizer, numerical_metadata, frame: pd.DataFrame, graph_context, device: str) -> pd.DataFrame:
    foreign_key_ids, datetime_values = encode_conditions(frame, config.schema, int(config.raw.get("id_encoding", {}).get("num_buckets", 262144)), device)
    generated = model.generate(foreign_key_ids, datetime_values, vocabs, tokenizer, graph_context=graph_context)
    output = frame.loc[:, list(config.schema.condition_columns)].copy()
    for column in config.schema.categorical_targets:
        output[column] = generated["categorical"][column]
    for column in config.schema.numerical_targets:
        sampled = sample_gaussian_params(generated["numerical_params"][column])
        output[column] = inverse_transform_numerical(sampled, numerical_metadata.get(column, {})).detach().cpu().tolist()
    for column in config.schema.text_targets:
        output[column] = generated["text"][column]
    return output


if __name__ == "__main__":
    main()
