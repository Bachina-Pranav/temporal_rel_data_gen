"""Training loop for Conditional TABDLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attribute_corruption import GraphAttributeStore, build_attribute_graph_batch
from .dataset import (
    ConditionalTABDLMDataset,
    auxiliary_target_values,
    load_category_vocabs,
    load_prepared_tables,
    load_text_tokenizer,
    make_collate_fn,
)
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_encoder import TemporalAttributeDenoisingGraphEncoder, TemporalStructureOnlyGraphEncoder, build_temporal_graph_encoder
from .graph_schema import (
    assert_valid_graph_conditioning,
    attribute_denoising_config,
    graph_conditioning_enabled,
    graph_metadata,
    graph_mode,
)
from .model import ConditionalTABDLM
from .schema import ConditionalTABDLMConfig
from .tokenization import CategoryVocab, SimpleTextTokenizer
from .utils import ensure_dir, save_json, save_yaml, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def train_from_config(config: ConditionalTABDLMConfig, device: str | None = None) -> Path:
    training = config.raw.get("training", {})
    diffusion = config.raw.get("diffusion", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    save_yaml(config.to_dict(), output_dir / "config_resolved.yaml")

    train_frame, valid_frame, _ = load_prepared_tables(config)
    train_frame = maybe_limit_rows(train_frame, training.get("max_rows"), seed)
    valid_frame = maybe_limit_rows(valid_frame, validation_row_cap(training.get("max_rows"), len(valid_frame)), seed + 1)
    use_graph_context = graph_conditioning_enabled(config.raw)
    if use_graph_context:
        assert_valid_graph_conditioning(config.raw)

    categorical_vocabs = load_category_vocabs(config)
    text_tokenizer = load_text_tokenizer(config)
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    train_dataset = ConditionalTABDLMDataset(
        train_frame,
        config.schema,
        categorical_vocabs,
        text_tokenizer,
        num_hash_buckets=num_hash_buckets,
    )
    valid_dataset = ConditionalTABDLMDataset(
        valid_frame,
        config.schema,
        categorical_vocabs,
        text_tokenizer,
        num_hash_buckets=num_hash_buckets,
    )
    collate_fn = make_collate_fn(
        config.schema,
        categorical_vocabs,
        text_tokenizer,
        min_mask_prob=float(diffusion.get("min_mask_prob", 0.05)),
        max_mask_prob=float(diffusion.get("max_mask_prob", 0.95)),
        mask_schedule=str(diffusion.get("mask_schedule", "linear")),
    )
    batch_size = int(training.get("batch_size", 128))
    num_workers = int(training.get("num_workers", training.get("workers", 0)))
    dataloader_timeout = int(training.get("dataloader_timeout_seconds", 0) or 0)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        dataloader_kwargs["timeout"] = dataloader_timeout
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    device = resolve_device(device or str(training.get("device", "auto")))
    print(
        "Training ConditionalTABDLM with "
        f"train_rows={len(train_dataset)}, valid_rows={len(valid_dataset)}, "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"dataloader_timeout_seconds={dataloader_timeout if num_workers > 0 else 0}, "
        f"device={device}"
    )
    model = build_model(config, categorical_vocabs, text_tokenizer).to(device)
    graph_encoder = build_graph_encoder(config, categorical_vocabs, text_tokenizer).to(device) if use_graph_context else None
    train_history_index = build_temporal_history_index(train_frame, config, seed=seed) if use_graph_context else None
    valid_graph_frame = pd.concat([train_frame, valid_frame], ignore_index=True) if use_graph_context else valid_frame
    valid_history_index = build_temporal_history_index(valid_graph_frame, config, seed=seed + 1) if use_graph_context else None
    valid_row_id_offset = len(train_frame) if use_graph_context else 0
    use_attr_denoising_graph = use_graph_context and graph_mode(config.raw) == "temporal_attribute_denoising"
    train_attr_store = (
        GraphAttributeStore.from_frame(train_frame, config, categorical_vocabs, text_tokenizer)
        if use_attr_denoising_graph
        else None
    )
    valid_attr_store = (
        GraphAttributeStore.from_frame(valid_graph_frame, config, categorical_vocabs, text_tokenizer)
        if use_attr_denoising_graph
        else None
    )
    metadata_dir = ensure_dir(output_dir / "metadata")
    if use_graph_context:
        graph_meta_dir = ensure_dir(output_dir / "graph")
        write_temporal_graph_metadata(train_frame, config, graph_meta_dir, source="real_training_rows", seed=seed)
        graph_flags = graph_metadata(config.raw, real_graph_used_at_sampling=False)
        save_json(graph_flags, output_dir / "graph_conditioning_flags.json")
        save_json(graph_flags, metadata_dir / "graph_conditioning.json")
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, graph_encoder),
        lr=float(training.get("learning_rate", training.get("lr", 3e-4))),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    loss_weights = dict(config.raw.get("loss_weights", {}))
    summary_token_loss_weights = dict(config.raw.get("summary_token_loss_weights", {}))
    length_class_weights = compute_summary_length_class_weights(
        train_frame,
        config,
        categorical_vocabs,
        text_tokenizer,
    )
    if length_class_weights is not None:
        save_json(length_class_weights["json"], metadata_dir / "summary_length_bucket_weights.json")
    use_amp = bool(training.get("mixed_precision", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    best_valid = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    epochs = int(training.get("epochs", 5))
    early_stopping_patience = int(training.get("early_stopping_patience", 0) or 0)
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0) or 0.0)
    epochs_without_improvement = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            loss_weights,
            text_tokenizer,
            summary_token_loss_weights,
            length_class_weights["tensor"].to(device) if length_class_weights is not None else None,
            graph_encoder=graph_encoder,
            graph_history_index=train_history_index,
            graph_deterministic=False,
            graph_attr_store=train_attr_store,
            config=config,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            None,
            scaler,
            device,
            use_amp,
            loss_weights,
            text_tokenizer,
            summary_token_loss_weights,
            length_class_weights["tensor"].to(device) if length_class_weights is not None else None,
            graph_encoder=graph_encoder,
            graph_history_index=valid_history_index,
            graph_deterministic=True,
            graph_row_id_offset=valid_row_id_offset,
            graph_attr_store=valid_attr_store,
            config=config,
        )
        length_calibration = compute_summary_length_calibration(
            model,
            valid_loader,
            config,
            categorical_vocabs,
            text_tokenizer,
            device,
            use_amp,
            graph_encoder=graph_encoder,
            graph_history_index=valid_history_index,
            graph_row_id_offset=valid_row_id_offset,
            graph_attr_store=valid_attr_store,
        )
        if length_calibration is not None:
            save_json(length_calibration, metadata_dir / "summary_length_calibration.json")
        current_valid = float(valid_metrics["total_loss"])
        improved = current_valid < (best_valid - early_stopping_min_delta)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "best_valid_total_loss": min(best_valid, current_valid),
            "best_epoch": best_epoch if not improved else epoch,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "epochs_without_improvement": 0 if improved else epochs_without_improvement + 1,
        }
        append_jsonl(log_path, row)
        print(json.dumps(row, sort_keys=True))
        save_checkpoint(
            last_path,
            model,
            config,
            categorical_vocabs,
            text_tokenizer,
            epoch,
            valid_metrics,
            length_calibration=length_calibration,
            summary_length_bucket_weights=length_class_weights["json"] if length_class_weights is not None else None,
            graph_encoder=graph_encoder,
        )
        if improved:
            best_valid = current_valid
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                config,
                categorical_vocabs,
                text_tokenizer,
                epoch,
                valid_metrics,
                length_calibration=length_calibration,
                summary_length_bucket_weights=length_class_weights["json"] if length_class_weights is not None else None,
                graph_encoder=graph_encoder,
            )
        else:
            epochs_without_improvement += 1
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping at "
                f"epoch={epoch}; best_epoch={best_epoch}; "
                f"best_valid_total_loss={best_valid:.6g}; "
                f"patience={early_stopping_patience}; "
                f"min_delta={early_stopping_min_delta}"
            )
            break
    print(f"Wrote best checkpoint to {best_path}")
    return best_path


def build_model(
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
) -> ConditionalTABDLM:
    model_cfg = config.raw.get("model", {})
    id_cfg = config.raw.get("id_encoding", {})
    dt_cfg = config.raw.get("datetime_encoding", {})
    return ConditionalTABDLM(
        schema=config.schema,
        categorical_vocabs=categorical_vocabs,
        text_tokenizer=text_tokenizer,
        num_hash_buckets=int(id_cfg.get("num_buckets", 262144)),
        id_embedding_dim=int(id_cfg.get("embedding_dim", 128)),
        datetime_embedding_dim=int(dt_cfg.get("embedding_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", model_cfg.get("hidden", 384))),
        num_layers=int(model_cfg.get("num_layers", model_cfg.get("layers", 6))),
        num_heads=int(model_cfg.get("num_heads", model_cfg.get("heads", 6))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        condition_dim=int(model_cfg.get("condition_dim", 256)),
        use_graph_context=bool(model_cfg.get("use_graph_context", graph_conditioning_enabled(config.raw))),
        graph_context_dim=int(model_cfg.get("graph_context_dim", config.raw.get("graph_conditioning", {}).get("graph_encoder", {}).get("output_dim", 256))),
    )


def build_graph_encoder(
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab] | None = None,
    text_tokenizer: SimpleTextTokenizer | None = None,
) -> TemporalStructureOnlyGraphEncoder:
    if graph_mode(config.raw) == "temporal_attribute_denoising":
        if categorical_vocabs is None or text_tokenizer is None:
            raise ValueError("temporal_attribute_denoising graph encoder requires categorical vocabs and text tokenizer")
        return build_temporal_graph_encoder(config.raw, config.schema, categorical_vocabs, text_tokenizer)
    return build_temporal_graph_encoder(
        config.raw,
        config.schema,
        categorical_vocabs or {},
        text_tokenizer or SimpleTextTokenizer(),
    )


def run_epoch(
    model: ConditionalTABDLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_amp: bool,
    loss_weights: dict[str, float] | None = None,
    text_tokenizer: SimpleTextTokenizer | None = None,
    summary_token_loss_weights: dict[str, float] | None = None,
    summary_length_class_weights: torch.Tensor | None = None,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
    graph_history_index: Any | None = None,
    graph_deterministic: bool = True,
    graph_row_id_offset: int = 0,
    graph_attr_store: GraphAttributeStore | None = None,
    config: ConditionalTABDLMConfig | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if graph_encoder is not None:
        graph_encoder.train(training)
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    corrects: dict[str, int] = {}
    graph_norm_sum = 0.0
    graph_norm_sq_sum = 0.0
    graph_norm_count = 0
    graph_grad_norm_sum = 0.0
    graph_grad_norm_count = 0
    extra_metric_sums: dict[str, float] = {}
    extra_metric_counts: dict[str, int] = {}
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if training else "valid")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            graph_context, graph_component = compute_graph_outputs(
                graph_encoder,
                graph_history_index,
                batch,
                device,
                deterministic=graph_deterministic or not training,
                row_id_offset=graph_row_id_offset,
                graph_attr_store=graph_attr_store,
                config=config,
                training=training,
            )
            if graph_context is not None:
                norms = graph_context.float().norm(dim=1).detach()
                graph_norm_sum += float(norms.sum().cpu())
                graph_norm_sq_sum += float((norms * norms).sum().cpu())
                graph_norm_count += int(norms.numel())
            logits = model(
                foreign_key_ids=batch["foreign_key_ids"],
                datetime_values=batch["datetime_values"],
                categorical_input_ids=batch["categorical_input_ids"],
                text_input_ids=batch["text_input_ids"],
                text_attention=batch["text_attention"],
                diffusion_t=batch["diffusion_t"],
                graph_context=graph_context,
            )
            loss, component = denoising_loss(
                logits,
                batch,
                model.schema,
                loss_weights or {},
                text_tokenizer=text_tokenizer,
                summary_token_loss_weights=summary_token_loss_weights or {},
                summary_length_class_weights=summary_length_class_weights,
            )
            aux_loss = graph_component.get("auxiliary_neighbor_denoising_loss_tensor")
            if aux_loss is not None:
                aux_weight = float((loss_weights or {}).get("auxiliary_neighbor_denoising", graph_component.get("auxiliary_neighbor_denoising_weight", 1.0)))
                loss = loss + aux_weight * aux_loss
                component["auxiliary_neighbor_denoising"] = {
                    "loss_sum": float(graph_component.get("auxiliary_neighbor_denoising_loss_sum", 0.0)),
                    "count": int(graph_component.get("auxiliary_neighbor_denoising_count", 1)),
                }
                for name, stats in graph_component.get("auxiliary_neighbor_denoising_components", {}).items():
                    component[f"aux_neighbor_{name}"] = {
                        "loss_sum": float(stats.get("loss_sum", 0.0)),
                        "count": int(stats.get("count", 1)),
                    }
            gate_reg = graph_component.get("summary_attr_gate_regularization_loss_tensor")
            if gate_reg is not None:
                loss = loss + gate_reg
                component["summary_attr_gate_regularization"] = {
                    "loss_sum": float(gate_reg.detach().cpu()),
                    "count": 1,
                }
            if "summary_attr_gate" in graph_component:
                extra_metric_sums["summary_attr_gate"] = extra_metric_sums.get("summary_attr_gate", 0.0) + float(graph_component["summary_attr_gate"])
                extra_metric_counts["summary_attr_gate"] = extra_metric_counts.get("summary_attr_gate", 0) + 1
            for diag_key in ["history_attr_mask_rate", "target_attr_mask_rate"]:
                if diag_key in graph_component:
                    component[diag_key] = {"loss_sum": float(graph_component[diag_key]), "count": 1}
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if graph_encoder is not None:
                graph_grad_norm_sum += parameter_grad_norm(graph_encoder.parameters())
                graph_grad_norm_count += 1
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model, graph_encoder), 1.0)
            scaler.step(optimizer)
            scaler.update()
        for key, stats in component.items():
            totals[key] = totals.get(key, 0.0) + float(stats["loss_sum"])
            counts[key] = counts.get(key, 0.0) + float(stats["count"])
            corrects[key] = corrects.get(key, 0) + int(stats.get("correct", 0))

    metrics = {}
    total_loss = 0.0
    for key in sorted(totals):
        count = max(float(counts.get(key, 0.0)), 1.0)
        metric_key = component_metric_name(key)
        component_loss = float(totals[key] / count)
        if str(metric_key).startswith("loss_"):
            metrics[metric_key] = component_loss
        else:
            metrics[f"{metric_key}_loss"] = component_loss
        if is_main_loss_component(key, model.schema) or key == "auxiliary_neighbor_denoising":
            total_loss += float(component_weight(key, loss_weights or {})) * component_loss
        if key in model.schema.model_categorical_targets:
            metrics[f"{metric_key}_accuracy"] = float(corrects.get(key, 0) / count)
    metrics["total_loss"] = float(total_loss)
    if graph_norm_count > 0:
        mean = graph_norm_sum / max(graph_norm_count, 1)
        var = max(graph_norm_sq_sum / max(graph_norm_count, 1) - mean * mean, 0.0)
        metrics["graph_context_norm_mean"] = float(mean)
        metrics["graph_context_norm_std"] = float(var ** 0.5)
    if graph_grad_norm_count > 0:
        metrics["graph_encoder_grad_norm"] = float(graph_grad_norm_sum / max(graph_grad_norm_count, 1))
    for key, value in extra_metric_sums.items():
        metrics[key] = float(value / max(extra_metric_counts.get(key, 0), 1))
    return metrics


def denoising_loss(
    logits: dict[str, Any],
    batch: dict[str, Any],
    schema,
    loss_weights: dict[str, float] | None = None,
    text_tokenizer: SimpleTextTokenizer | None = None,
    summary_token_loss_weights: dict[str, float] | None = None,
    summary_length_class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    return _denoising_loss_impl(
        logits,
        batch,
        schema,
        loss_weights or {},
        text_tokenizer=text_tokenizer,
        summary_token_loss_weights=summary_token_loss_weights or {},
        summary_length_class_weights=summary_length_class_weights,
    )


def _denoising_loss_impl(
    logits: dict[str, Any],
    batch: dict[str, Any],
    schema,
    loss_weights: dict[str, float],
    text_tokenizer: SimpleTextTokenizer | None = None,
    summary_token_loss_weights: dict[str, float] | None = None,
    summary_length_class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    losses: list[torch.Tensor] = []
    component: dict[str, dict[str, float | int]] = {}
    cat_labels = batch["categorical_labels"]
    for idx, column in enumerate(schema.model_categorical_targets):
        labels = cat_labels[:, idx]
        mask = labels != -100
        count = int(mask.sum().detach().cpu())
        if count == 0:
            continue
        class_weights = summary_length_class_weights if column == "summary_length_bucket" else None
        loss_sum = F.cross_entropy(
            logits["categorical"][column],
            labels,
            ignore_index=-100,
            reduction="sum",
            weight=class_weights,
        )
        mean_loss = loss_sum / max(count, 1)
        losses.append(float(component_weight(column, loss_weights)) * mean_loss)
        pred = logits["categorical"][column].argmax(dim=-1)
        correct = int((pred[mask] == labels[mask]).sum().detach().cpu())
        component[column] = {"loss_sum": float(loss_sum.detach().cpu()), "count": count, "correct": correct}

    for column in schema.text_targets:
        labels = batch["text_labels"][column]
        count = int((labels != -100).sum().detach().cpu())
        if count == 0:
            continue
        if text_tokenizer is not None:
            loss_sum, denom, subcomponents = weighted_summary_token_loss(
                logits["text"][column],
                labels,
                text_tokenizer,
                summary_token_loss_weights or {},
            )
            count = int(max(float(denom.detach().cpu()), 1.0))
            for subkey, stats in subcomponents.items():
                component[f"{column}_{subkey}_component"] = stats
        else:
            loss_sum = F.cross_entropy(
                logits["text"][column].reshape(-1, logits["text"][column].shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            denom = torch.tensor(float(count), device=loss_sum.device)
        mean_loss = loss_sum / denom.clamp_min(1.0)
        losses.append(float(component_weight(column, loss_weights)) * mean_loss)
        component[column] = {"loss_sum": float(loss_sum.detach().cpu()), "count": float(max(float(denom.detach().cpu()), 1.0))}

    if not losses:
        zero = batch["foreign_key_ids"].float().sum() * 0.0
        return zero, {}
    return torch.stack(losses).sum(), component


def component_metric_name(column: str) -> str:
    if column == "auxiliary_neighbor_denoising":
        return "auxiliary_neighbor_denoising"
    if str(column).startswith("aux_neighbor_"):
        return "loss_aux_neighbor_" + str(column)[len("aux_neighbor_") :]
    if column == "summary_length_bucket":
        return "summary_length"
    for suffix in ["_pad_component", "_eos_component", "_content_component"]:
        if str(column).endswith(suffix):
            return str(column)
    return str(column)


def component_weight(column: str, loss_weights: dict[str, float]) -> float:
    key = component_metric_name(column)
    return float(loss_weights.get(key, loss_weights.get(column, 1.0)))


def is_main_loss_component(column: str, schema) -> bool:
    return column in set(schema.model_categorical_targets + schema.text_targets)


def summary_token_weights(
    labels: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    weights_config: dict[str, float] | None = None,
) -> torch.Tensor:
    weights_config = weights_config or {}
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[labels == -100] = 0.0
    weights[labels == tokenizer.pad_id] = float(weights_config.get("pad", 0.15))
    weights[labels == tokenizer.eos_id] = float(weights_config.get("eos", 2.0))
    weights[labels == tokenizer.bos_id] = float(weights_config.get("bos", 0.0))
    weights[labels == tokenizer.unk_id] = float(weights_config.get("unk", 1.0))
    special = {
        tokenizer.pad_id,
        tokenizer.eos_id,
        tokenizer.bos_id,
        tokenizer.mask_id,
        tokenizer.unk_id,
        -100,
    }
    content_mask = torch.ones_like(labels, dtype=torch.bool)
    for token_id in special:
        content_mask &= labels != int(token_id)
    weights[content_mask] = float(weights_config.get("content", 1.0))
    return weights


def weighted_summary_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    weights_config: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float | int]]]:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    weights = summary_token_weights(labels, tokenizer, weights_config)
    loss_sum = (ce * weights).sum()
    denom = weights.sum().clamp_min(1.0)
    subcomponents: dict[str, dict[str, float | int]] = {}
    masks = {
        "pad": labels == tokenizer.pad_id,
        "eos": labels == tokenizer.eos_id,
        "content": (labels != -100)
        & (labels != tokenizer.pad_id)
        & (labels != tokenizer.eos_id)
        & (labels != tokenizer.bos_id)
        & (labels != tokenizer.mask_id),
    }
    for name, mask in masks.items():
        weighted = ce[mask] * weights[mask]
        subcomponents[name] = {
            "loss_sum": float(weighted.sum().detach().cpu()),
            "count": int(mask.sum().detach().cpu()),
        }
    return loss_sum, denom, subcomponents


def compute_summary_length_class_weights(
    train_frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
) -> dict[str, Any] | None:
    if "summary_length_bucket" not in config.schema.auxiliary_categorical_targets:
        return None
    length_cfg = config.raw.get("summary_length_loss", {})
    if not bool(length_cfg.get("class_balanced", False)):
        return None
    vocab = categorical_vocabs["summary_length_bucket"]
    values = auxiliary_target_values(train_frame, config.schema, tokenizer, "summary_length_bucket")
    ids = np.asarray([vocab.encode(value) for value in values], dtype=np.int64)
    counts = np.bincount(ids, minlength=vocab.size).astype(float)
    total = max(float(counts.sum()), 1.0)
    freqs = counts / total
    power = float(length_cfg.get("class_weight_power", 0.5))
    raw = np.zeros_like(freqs)
    nonzero = freqs > 0
    raw[nonzero] = np.power(1.0 / freqs[nonzero], power)
    if raw[nonzero].size:
        raw[nonzero] = raw[nonzero] / raw[nonzero].mean()
    raw[~nonzero] = float(length_cfg.get("max_class_weight", 5.0))
    weights = np.clip(
        raw,
        float(length_cfg.get("min_class_weight", 0.5)),
        float(length_cfg.get("max_class_weight", 5.0)),
    )
    id_to_token = vocab.id_to_token
    payload = {
        "class_balanced": True,
        "class_weight_power": power,
        "min_class_weight": float(length_cfg.get("min_class_weight", 0.5)),
        "max_class_weight": float(length_cfg.get("max_class_weight", 5.0)),
        "counts": {id_to_token[idx]: int(counts[idx]) for idx in range(vocab.size)},
        "frequencies": {id_to_token[idx]: float(freqs[idx]) for idx in range(vocab.size)},
        "weights": {id_to_token[idx]: float(weights[idx]) for idx in range(vocab.size)},
    }
    return {
        "tensor": torch.tensor(weights, dtype=torch.float32),
        "json": payload,
    }


@torch.no_grad()
def compute_summary_length_calibration(
    model: ConditionalTABDLM,
    loader: DataLoader,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    device: str,
    use_amp: bool,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
    graph_history_index: Any | None = None,
    graph_row_id_offset: int = 0,
    graph_attr_store: GraphAttributeStore | None = None,
) -> dict[str, Any] | None:
    if "summary_length_bucket" not in config.schema.model_categorical_targets:
        return None
    length_cfg = config.raw.get("summary_length", {})
    if not bool(length_cfg.get("calibrate_length_bucket_sampling", False)):
        return None
    length_idx = config.schema.model_categorical_targets.index("summary_length_bucket")
    vocab = categorical_vocabs["summary_length_bucket"]
    real_counts = torch.zeros(vocab.size, dtype=torch.float64, device=device)
    model_probs = torch.zeros(vocab.size, dtype=torch.float64, device=device)
    total = 0
    was_training = model.training
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        cat_input = initial_masked_categorical_inputs(batch["categorical_clean_ids"], config.schema, categorical_vocabs)
        text_input, text_attention = initial_masked_text_inputs(batch["text_clean_ids"], config.schema, tokenizer)
        with torch.cuda.amp.autocast(enabled=use_amp):
            graph_context, _ = compute_graph_outputs(
                graph_encoder,
                graph_history_index,
                batch,
                device,
                deterministic=True,
                row_id_offset=graph_row_id_offset,
                graph_attr_store=graph_attr_store,
                config=config,
                training=False,
            )
            logits = model(
                foreign_key_ids=batch["foreign_key_ids"],
                datetime_values=batch["datetime_values"],
                categorical_input_ids=cat_input,
                text_input_ids=text_input,
                text_attention=text_attention,
                diffusion_t=torch.ones(batch["foreign_key_ids"].shape[0], dtype=torch.float32, device=device),
                graph_context=graph_context,
            )
        labels = batch["categorical_clean_ids"][:, length_idx]
        real_counts += torch.bincount(labels, minlength=vocab.size).to(device=device, dtype=torch.float64)
        model_probs += torch.softmax(logits["categorical"]["summary_length_bucket"].float(), dim=-1).sum(dim=0).to(torch.float64)
        total += int(labels.numel())
    if was_training:
        model.train()
        if graph_encoder is not None:
            graph_encoder.train()
    if total <= 0:
        return None
    eps = float(length_cfg.get("calibration_epsilon", 1e-6))
    real_dist = (real_counts / max(float(real_counts.sum().item()), 1.0)).detach().cpu().numpy()
    model_dist = (model_probs / max(float(total), 1.0)).detach().cpu().numpy()
    ratio = real_dist / np.clip(model_dist, eps, None)
    strength = float(length_cfg.get("calibration_strength", 1.0))
    id_to_token = vocab.id_to_token
    return {
        "real_valid_bucket_distribution": {id_to_token[idx]: float(real_dist[idx]) for idx in range(vocab.size)},
        "model_valid_bucket_distribution": {id_to_token[idx]: float(model_dist[idx]) for idx in range(vocab.size)},
        "calibration_ratio": {id_to_token[idx]: float(ratio[idx]) for idx in range(vocab.size)},
        "calibration_strength": strength,
        "calibration_epsilon": eps,
    }


def initial_masked_categorical_inputs(
    clean_ids: torch.Tensor,
    schema,
    categorical_vocabs: dict[str, CategoryVocab],
) -> torch.Tensor:
    masked = clean_ids.clone()
    for idx, column in enumerate(schema.model_categorical_targets):
        masked[:, idx] = categorical_vocabs[column].mask_id
    return masked


def initial_masked_text_inputs(
    clean_text_ids: dict[str, torch.Tensor],
    schema,
    tokenizer: SimpleTextTokenizer,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_input: dict[str, torch.Tensor] = {}
    text_attention: dict[str, torch.Tensor] = {}
    for column in schema.text_targets:
        clean = clean_text_ids[column]
        values = torch.full_like(clean, tokenizer.mask_id)
        values[:, 0] = tokenizer.bos_id
        text_input[column] = values
        text_attention[column] = torch.ones_like(clean, dtype=torch.long)
    return text_input, text_attention


def save_checkpoint(
    path: str | Path,
    model: ConditionalTABDLM,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    epoch: int,
    valid_metrics: dict[str, float],
    length_calibration: dict[str, Any] | None = None,
    summary_length_bucket_weights: dict[str, Any] | None = None,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.to_config(),
            "raw_config": config.raw,
            "schema": config.schema.to_dict(),
            "categorical_vocabs": {column: vocab.to_dict() for column, vocab in categorical_vocabs.items()},
            "tokenizer_metadata": text_tokenizer.to_dict(),
            "epoch": int(epoch),
            "valid_metrics": valid_metrics,
            "summary_length_calibration": length_calibration,
            "summary_length_bucket_weights": summary_length_bucket_weights,
            "graph_encoder_state_dict": graph_encoder.state_dict() if graph_encoder is not None else None,
            "graph_encoder_config": graph_encoder.to_config() if graph_encoder is not None else None,
            "graph_conditioning_metadata": graph_metadata(config.raw, real_graph_used_at_sampling=False),
        },
        path,
    )


def trainable_parameters(
    model: ConditionalTABDLM,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
) -> list[torch.nn.Parameter]:
    params = list(model.parameters())
    if graph_encoder is not None:
        params.extend(list(graph_encoder.parameters()))
    return params


def compute_graph_outputs(
    graph_encoder: TemporalStructureOnlyGraphEncoder | None,
    graph_history_index: Any | None,
    batch: dict[str, Any],
    device: str,
    *,
    deterministic: bool = True,
    row_id_offset: int = 0,
    graph_attr_store: GraphAttributeStore | None = None,
    config: ConditionalTABDLMConfig | None = None,
    training: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if graph_encoder is None:
        return None, {}
    if graph_history_index is None:
        raise ValueError("graph_history_index is required when graph_encoder is enabled")
    row_ids = batch["row_id"]
    if int(row_id_offset) != 0:
        row_ids = row_ids + int(row_id_offset)
    graph_batch = graph_history_index.build_batch(
        row_ids,
        device=device,
        deterministic=deterministic,
    )
    component: dict[str, Any] = {}
    if graph_attr_store is not None and config is not None:
        attr_batch, attr_diag = build_attribute_graph_batch(
            graph_batch,
            batch,
            graph_attr_store,
            config,
            device=device,
            training=training,
        )
        graph_batch.update(attr_batch)
        component.update(attr_diag)
    context = graph_encoder(graph_batch)
    if isinstance(graph_encoder, TemporalAttributeDenoisingGraphEncoder):
        aux_cfg = attribute_denoising_config(config.raw if config is not None else {}).get("auxiliary_neighbor_denoising_loss", {})
        if bool(aux_cfg.get("enabled", False)) and graph_attr_store is not None:
            aux_loss, aux_component = graph_encoder.auxiliary_neighbor_loss(
                graph_batch,
                max_nodes=int(aux_cfg.get("max_neighbor_nodes_for_loss", 256)),
            )
            component["auxiliary_neighbor_denoising_loss_tensor"] = aux_loss
            component["auxiliary_neighbor_denoising_loss_sum"] = aux_component.get("loss_sum", 0.0)
            component["auxiliary_neighbor_denoising_count"] = aux_component.get("count", 1)
            component["auxiliary_neighbor_denoising_components"] = aux_component.get("components", {})
            component["auxiliary_neighbor_denoising_weight"] = float(aux_cfg.get("weight", 0.25))
        gate_value = graph_encoder.summary_attr_gate_value()
        if gate_value is not None:
            component["summary_attr_gate"] = float(gate_value)
        gate_reg = graph_encoder.summary_attr_gate_regularization_loss()
        if gate_reg is not None:
            component["summary_attr_gate_regularization_loss_tensor"] = gate_reg
    return context, component


def compute_graph_context(
    graph_encoder: TemporalStructureOnlyGraphEncoder | None,
    graph_history_index: Any | None,
    batch: dict[str, Any],
    device: str,
    *,
    deterministic: bool = True,
    row_id_offset: int = 0,
    graph_attr_store: GraphAttributeStore | None = None,
    config: ConditionalTABDLMConfig | None = None,
    training: bool = False,
) -> torch.Tensor | None:
    context, _ = compute_graph_outputs(
        graph_encoder,
        graph_history_index,
        batch,
        device,
        deterministic=deterministic,
        row_id_offset=row_id_offset,
        graph_attr_store=graph_attr_store,
        config=config,
        training=training,
    )
    return context


def parameter_grad_norm(parameters: Any) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).cpu())
    return float(total ** 0.5)


def move_batch_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    return value


def maybe_limit_rows(frame: pd.DataFrame, max_rows: Any, seed: int) -> pd.DataFrame:
    if max_rows in (None, "all"):
        return frame.reset_index(drop=True)
    max_rows = int(max_rows)
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(frame), size=max_rows, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def validation_row_cap(max_train_rows: Any, valid_rows: int) -> int | None:
    if max_train_rows in (None, "all"):
        return None
    max_train_rows = int(max_train_rows)
    if max_train_rows <= 0:
        return None
    return min(int(valid_rows), max(1024, max_train_rows // 20))


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU")
        return "cpu"
    return device


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")
