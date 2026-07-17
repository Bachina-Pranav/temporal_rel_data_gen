#!/usr/bin/env python3
"""Diagnose whether v4/v4.1 graph context changes model predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.graph_dataset import build_temporal_history_index, temporal_graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import encode_conditions, load_model_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import save_json, set_seed  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml"
DEFAULT_CHECKPOINT = "outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt"
DEFAULT_SPINE = "outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv"
DEFAULT_OUTPUT = "outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/graph_conditioning_diagnostic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v4 graph conditioning sensitivity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--synthetic-spine", default=DEFAULT_SPINE)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--num-rows", type=int, default=128)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    model, ckpt_config, vocabs, tokenizer, graph_encoder = load_model_checkpoint(args.checkpoint, device=device, include_graph=True)
    if graph_encoder is None:
        raise SystemExit("Checkpoint/config has no graph encoder; cannot diagnose graph conditioning.")
    graph_encoder.to(device).eval()
    model.to(device).eval()
    spine = pd.read_csv(args.synthetic_spine).head(args.num_rows).reset_index(drop=True)
    graph_history = build_temporal_history_index(spine, ckpt_config, seed=args.seed)
    row_indices = list(range(len(spine)))
    graph_batch = graph_history.build_batch(row_indices, device=device, deterministic=True)

    with torch.no_grad():
        correct = graph_encoder(graph_batch)
        variants = {
            "zero": torch.zeros_like(correct),
            "shuffled": correct[torch.randperm(correct.shape[0], device=correct.device)] if correct.shape[0] > 1 else correct,
            "identity_time_only": graph_encoder(zero_history_masks(graph_batch)),
            "fusion_disabled": None,
        }
        base_logits = run_logits(model, ckpt_config, vocabs, tokenizer, spine, correct, device)
        comparisons = {
            name: compare_logits(base_logits, run_logits(model, ckpt_config, vocabs, tokenizer, spine, context, device))
            for name, context in variants.items()
        }
        structured_counterfactual = structured_to_text_counterfactual(
            model,
            ckpt_config,
            vocabs,
            tokenizer,
            spine,
            correct,
            device,
        )

    grad_metrics = gradient_metrics(model, graph_encoder, ckpt_config, vocabs, tokenizer, spine, graph_batch, args.real_table, device)
    context_std = correct.float().std(dim=0).mean().item() if correct.numel() else 0.0
    context_pairwise = mean_pairwise_l2(correct)
    metadata = temporal_graph_metadata(spine, ckpt_config, source="synthetic_spine_diagnostic", seed=args.seed, real_graph_used_at_sampling=False)
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "synthetic_spine": str(args.synthetic_spine),
        "num_rows": int(len(spine)),
        "graph_context_mean_feature_std_across_rows": float(context_std),
        "graph_context_mean_pairwise_l2": float(context_pairwise),
        "context_variants": comparisons,
        "structured_to_text_counterfactual": structured_counterfactual,
        "gradient_metrics": grad_metrics,
        "temporal_graph_metadata": metadata,
        "safety": {
            "past_only_history_index": True,
            "future_events_excluded": True,
            "current_target_attributes_excluded_from_structure_only_graph": True,
            "note": "TemporalHistoryIndex asserts no future/current event in history during build_batch.",
        },
        "interpretation_note": "Nonzero logit differences prove graph signal reaches logits, not that graph conditioning improves generated-table quality.",
    }
    save_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_logits(
    model,
    config,
    vocabs,
    tokenizer,
    spine: pd.DataFrame,
    graph_context: torch.Tensor | None,
    device: str,
    categorical_input_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    schema = config.schema
    foreign_key_ids, datetime_values = encode_conditions(spine, schema, int(config.raw.get("id_encoding", {}).get("num_buckets", 262144)), device)
    if categorical_input_ids is None:
        cat_input = torch.empty((len(spine), len(schema.model_categorical_targets)), dtype=torch.long, device=device)
        for idx, column in enumerate(schema.model_categorical_targets):
            cat_input[:, idx] = vocabs[column].mask_id
    else:
        cat_input = categorical_input_ids.to(device=device, dtype=torch.long)
    text_input = {}
    text_attention = {}
    for column in schema.text_targets:
        length = int(schema.text_max_lengths[column])
        text_input[column] = torch.full((len(spine), length), tokenizer.mask_id, dtype=torch.long, device=device)
        text_input[column][:, 0] = tokenizer.bos_id
        text_attention[column] = torch.ones((len(spine), length), dtype=torch.long, device=device)
    t = torch.ones((len(spine),), dtype=torch.float32, device=device)
    return model(foreign_key_ids, datetime_values, cat_input, text_input, text_attention, t, graph_context)


def structured_to_text_counterfactual(model, config, vocabs, tokenizer, spine: pd.DataFrame, graph_context: torch.Tensor, device: str) -> dict[str, Any]:
    schema = config.schema
    target_column = None
    values: list[int] = []
    for column in schema.model_categorical_targets:
        vocab = vocabs[column]
        candidate_values = sorted(int(idx) for idx in vocab.id_to_token)
        if len(candidate_values) >= 2:
            target_column = column
            values = candidate_values[:2]
            break
    if target_column is None:
        return {"status": "skipped", "reason": "no categorical target has at least two valid values"}
    idx = schema.model_categorical_targets.index(target_column)
    cat_a = torch.empty((len(spine), len(schema.model_categorical_targets)), dtype=torch.long, device=device)
    cat_b = torch.empty_like(cat_a)
    for col_idx, column in enumerate(schema.model_categorical_targets):
        cat_a[:, col_idx] = vocabs[column].mask_id
        cat_b[:, col_idx] = vocabs[column].mask_id
    cat_a[:, idx] = int(values[0])
    cat_b[:, idx] = int(values[1])
    logits_a = run_logits(model, config, vocabs, tokenizer, spine, graph_context, device, categorical_input_ids=cat_a)
    logits_b = run_logits(model, config, vocabs, tokenizer, spine, graph_context, device, categorical_input_ids=cat_b)
    text_metrics = {
        column: tensor_compare(logits_a["text"][column], logits_b["text"][column])
        for column in schema.text_targets
    }
    return {
        "status": "ok",
        "counterfactual_field": target_column,
        "value_a": vocabs[target_column].decode(values[0]),
        "value_b": vocabs[target_column].decode(values[1]),
        "text": text_metrics,
        "mean_abs_change_text_logits": mean([metrics["mean_abs_change"] for metrics in text_metrics.values()]),
        "mean_kl_text_logits": mean([metrics["mean_kl"] for metrics in text_metrics.values()]),
        "interpretation_note": "This proves structured tokens can affect text logits; generated quality still requires ablation metrics.",
    }


def compare_logits(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"categorical": {}, "text": {}}
    cat_abs = []
    cat_kl = []
    for column, logits in base["categorical"].items():
        metrics = tensor_compare(logits, other["categorical"][column])
        out["categorical"][column] = metrics
        cat_abs.append(metrics["mean_abs_change"])
        cat_kl.append(metrics["mean_kl"])
    text_abs = []
    text_kl = []
    for column, logits in base["text"].items():
        metrics = tensor_compare(logits, other["text"][column])
        out["text"][column] = metrics
        text_abs.append(metrics["mean_abs_change"])
        text_kl.append(metrics["mean_kl"])
    out["mean_abs_change_structured_logits"] = mean(cat_abs)
    out["mean_kl_structured_logits"] = mean(cat_kl)
    out["mean_abs_change_text_logits"] = mean(text_abs)
    out["mean_kl_text_logits"] = mean(text_kl)
    return out


def tensor_compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_flat = a.detach().reshape(-1, a.shape[-1])
    b_flat = b.detach().reshape(-1, b.shape[-1])
    if a_flat.shape != b_flat.shape:
        raise ValueError(f"Cannot compare logits with different shapes: {tuple(a.shape)} vs {tuple(b.shape)}")
    rows = int(a_flat.shape[0])
    if rows == 0:
        return {"mean_abs_change": 0.0, "mean_kl": 0.0, "top1_change_rate": 0.0}

    chunk_size = max(1, min(256, rows))
    abs_sum = 0.0
    kl_sum = 0.0
    top1_changed = 0.0
    total_values = 0
    total_rows = 0
    for start in range(0, rows, chunk_size):
        a_chunk = a_flat[start : start + chunk_size].float()
        b_chunk = b_flat[start : start + chunk_size].float()
        p = torch.softmax(a_chunk, dim=-1).clamp_min(1e-12)
        q = torch.softmax(b_chunk, dim=-1).clamp_min(1e-12)
        abs_sum += float((a_chunk - b_chunk).abs().sum().detach().cpu())
        kl_sum += float((p * (p.log() - q.log())).sum(dim=-1).sum().detach().cpu())
        top1_changed += float((a_chunk.argmax(dim=-1) != b_chunk.argmax(dim=-1)).float().sum().detach().cpu())
        total_values += int(a_chunk.numel())
        total_rows += int(a_chunk.shape[0])
    return {
        "mean_abs_change": float(abs_sum / max(total_values, 1)),
        "mean_kl": float(kl_sum / max(total_rows, 1)),
        "top1_change_rate": float(top1_changed / max(total_rows, 1)),
    }


def gradient_metrics(model, graph_encoder, config, vocabs, tokenizer, spine, graph_batch, real_table, device) -> dict[str, Any]:
    if real_table is None:
        return {"status": "skipped", "reason": "--real-table not provided"}
    real = pd.read_csv(real_table, nrows=len(spine)).reset_index(drop=True)
    model.train()
    graph_encoder.train()
    for param in list(model.parameters()) + list(graph_encoder.parameters()):
        param.grad = None
    graph_context = graph_encoder(graph_batch)
    logits = run_logits(model, config, vocabs, tokenizer, spine, graph_context, device)
    losses = []
    for idx, column in enumerate(config.schema.model_categorical_targets):
        if column not in real:
            continue
        labels = torch.tensor([vocabs[column].encode(value) for value in real[column].tolist()], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits["categorical"][column], labels))
    if not losses:
        return {"status": "skipped", "reason": "no structured labels available"}
    torch.stack(losses).sum().backward()
    graph_grad = grad_norm(graph_encoder.parameters())
    fusion_params = [
        param for name, param in model.named_parameters()
        if "graph" in name or "condition" in name
    ]
    model.train(False)
    graph_encoder.train(False)
    return {
        "status": "ok",
        "graph_encoder_grad_norm": graph_grad,
        "graph_fusion_or_condition_grad_norm": grad_norm(fusion_params),
        "graph_embedding_not_detached": bool(graph_grad > 0),
    }


def zero_history_masks(graph_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in graph_batch.items()}
    for key in ["customer_history_mask", "product_history_mask"]:
        out[key] = torch.zeros_like(out[key], dtype=torch.bool)
    return out


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float((grad * grad).sum().cpu())
    return float(total ** 0.5)


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def mean_pairwise_l2(context: torch.Tensor) -> float:
    if context.shape[0] < 2:
        return 0.0
    diffs = context.float().unsqueeze(1) - context.float().unsqueeze(0)
    return float(torch.linalg.norm(diffs, dim=-1).mean().detach().cpu())


if __name__ == "__main__":
    main()
