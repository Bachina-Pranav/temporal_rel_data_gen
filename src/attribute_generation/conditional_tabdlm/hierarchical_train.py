"""Training loop for hierarchical structured-then-text Conditional TABDLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ConditionalTABDLMDataset, load_category_vocabs, load_prepared_tables, load_text_tokenizer, make_collate_fn
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_schema import assert_valid_graph_conditioning, graph_conditioning_enabled, graph_metadata
from .hierarchical_schema import generation_plan_from_config
from .neighbor_cache import CachedTemporalHistoryIndex
from .pretokenized import PretokenizedLSTMDataset, load_pretokenized_bundle
from .sample import sample_categorical_logits, sample_length_bucket_logits
from .schema import ConditionalTABDLMConfig
from .tokenization import CategoryVocab, SimpleTextTokenizer
from .train import (
    build_graph_encoder,
    build_model,
    build_optimizer,
    configure_torch_runtime,
    compute_graph_outputs,
    denoising_loss,
    length_weight_tensors_to_device,
    maybe_compile_training_module,
    move_batch_to_device,
    save_checkpoint,
    text_token_loss_weights_by_column,
    trainable_parameters,
    unwrap_compiled_module,
    compute_length_class_weights,
    resolve_device,
)
from .utils import ensure_dir, save_json, save_yaml, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def train_hierarchical_from_config(config: ConditionalTABDLMConfig, device: str | None = None, resume: str | Path | None = None) -> Path:
    training_started = time.perf_counter()
    training = config.raw.get("training", {})
    diffusion = config.raw.get("diffusion", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    device = resolve_device(device or str(training.get("device", "auto")))
    configure_torch_runtime(training, device)
    plan = generation_plan_from_config(config.raw, config.schema)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    save_yaml(config.to_dict(), output_dir / "config_resolved.yaml")
    save_json(plan.to_dict(), output_dir / "metadata" / "generation_plan.json")

    use_graph_context = graph_conditioning_enabled(config.raw)
    if use_graph_context:
        assert_valid_graph_conditioning(config.raw)

    pretokenized_dir = training.get("pretokenized_dir") or config.raw.get("paths", {}).get("pretokenized_dir")
    if pretokenized_dir:
        bundle = load_pretokenized_bundle(pretokenized_dir, config.schema)
        train_dataset = PretokenizedLSTMDataset(bundle, "train")
        valid_dataset = PretokenizedLSTMDataset(bundle, "valid")
        categorical_vocabs = bundle.categorical_vocabs
        text_tokenizer = bundle.tokenizer
        train_frame = None
        valid_frame = None
    else:
        train_frame, valid_frame, _ = load_prepared_tables(config)
        categorical_vocabs = load_category_vocabs(config)
        text_tokenizer = load_text_tokenizer(config)
        num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
        train_dataset = ConditionalTABDLMDataset(train_frame, config.schema, categorical_vocabs, text_tokenizer, num_hash_buckets)
        valid_dataset = ConditionalTABDLMDataset(valid_frame, config.schema, categorical_vocabs, text_tokenizer, num_hash_buckets)
    collate_fn = make_collate_fn(
        config.schema,
        categorical_vocabs,
        text_tokenizer,
        min_mask_prob=float(diffusion.get("min_mask_prob", 0.05)),
        max_mask_prob=float(diffusion.get("max_mask_prob", 0.95)),
        mask_schedule=str(diffusion.get("mask_schedule", "linear")),
        mask_padding_in_attention=bool(
            training.get("mask_padding_in_attention", False)
        ),
    )
    batch_size = int(training.get("batch_size", 64))
    num_workers = int(training.get("num_workers", 0))
    loader_kwargs = dataloader_kwargs(training, device, num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **loader_kwargs,
    )

    model = build_model(config, categorical_vocabs, text_tokenizer).to(device)
    graph_encoder = build_graph_encoder(config, categorical_vocabs, text_tokenizer).to(device) if use_graph_context else None
    start_epoch = 1
    resume_checkpoint = None
    if resume is not None:
        resume_checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if graph_encoder is not None and resume_checkpoint.get("graph_encoder_state_dict") is not None:
            graph_encoder.load_state_dict(resume_checkpoint["graph_encoder_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
    model, compile_used = maybe_compile_training_module(model, bool(training.get("compile_model", False)))

    neighbor_cache_dir = training.get("neighbor_cache_dir") or config.raw.get("paths", {}).get("neighbor_cache_dir")
    if use_graph_context and neighbor_cache_dir:
        train_history_index = CachedTemporalHistoryIndex(neighbor_cache_dir)
        valid_history_index = train_history_index
        valid_row_id_offset = 0
    elif use_graph_context:
        if train_frame is None or valid_frame is None:
            raise ValueError("Pretokenized hierarchical graph training requires --neighbor-cache-dir")
        train_history_index = build_temporal_history_index(train_frame, config, seed=seed)
        valid_graph_frame = torch_load_concat_frames(train_frame, valid_frame)
        valid_history_index = build_temporal_history_index(valid_graph_frame, config, seed=seed + 1)
        valid_row_id_offset = len(train_frame)
    else:
        train_history_index = None
        valid_history_index = None
        valid_row_id_offset = 0
    if use_graph_context:
        if train_frame is not None:
            write_temporal_graph_metadata(train_frame, config, output_dir / "graph", source="real_training_rows", seed=seed)
        save_json(graph_metadata(config.raw, real_graph_used_at_sampling=False), output_dir / "metadata" / "graph_conditioning.json")

    optimizer = build_optimizer(
        trainable_parameters(model, graph_encoder),
        lr=float(training.get("learning_rate", training.get("lr", 3e-4))),
        weight_decay=float(training.get("weight_decay", 0.01)),
        fused=bool(training.get("fused_adamw", False)) and device.startswith("cuda"),
    )
    if resume is not None:
        if resume_checkpoint is not None and resume_checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

    use_amp = bool(training.get("mixed_precision", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_weights = dict(config.raw.get("loss_weights", {}))
    text_token_loss_weights = text_token_loss_weights_by_column(config)
    if train_frame is not None:
        length_weights = compute_length_class_weights(train_frame, config, categorical_vocabs, text_tokenizer)
    else:
        length_weights = compute_length_class_weights_from_pretokenized(train_dataset, config, categorical_vocabs)
    length_weight_tensors = length_weight_tensors_to_device(length_weights, device)
    log_path = output_dir / "train_log.jsonl"
    epochs = int(training.get("epochs", 5))
    early_stopping_patience = int(training.get("early_stopping_patience", 0) or 0)
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0) or 0.0)
    epochs_without_improvement = 0
    best_valid = float("inf")
    if resume_checkpoint is not None:
        best_valid = float((resume_checkpoint.get("valid_metrics") or {}).get("total_loss", best_valid))
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    skip_checkpoints = bool(training.get("skip_checkpoints", False))

    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        completed_epoch = epoch
        train_metrics = run_hierarchical_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            config,
            categorical_vocabs,
            text_tokenizer,
            loss_weights,
            text_token_loss_weights,
            length_weight_tensors,
            graph_encoder=graph_encoder,
            graph_history_index=train_history_index,
            graph_deterministic=False,
            training=True,
        )
        valid_metrics = run_hierarchical_epoch(
            model,
            valid_loader,
            None,
            scaler,
            device,
            use_amp,
            config,
            categorical_vocabs,
            text_tokenizer,
            loss_weights,
            text_token_loss_weights,
            length_weight_tensors,
            graph_encoder=graph_encoder,
            graph_history_index=valid_history_index,
            graph_deterministic=True,
            graph_row_id_offset=valid_row_id_offset,
            training=False,
        )
        current_valid = float(valid_metrics["total_loss"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "torch_compile_used": compile_used,
            "generation_plan": plan.to_dict(),
        }
        append_jsonl(log_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not skip_checkpoints:
            save_hierarchical_checkpoint(last_path, model, optimizer, config, categorical_vocabs, text_tokenizer, epoch, valid_metrics, graph_encoder)
        if current_valid < best_valid - early_stopping_min_delta:
            best_valid = current_valid
            epochs_without_improvement = 0
            if not skip_checkpoints:
                save_hierarchical_checkpoint(best_path, model, optimizer, config, categorical_vocabs, text_tokenizer, epoch, valid_metrics, graph_encoder)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_valid_total_loss": best_valid,
                            "patience": early_stopping_patience,
                            "min_delta": early_stopping_min_delta,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
    if skip_checkpoints:
        print(f"Skipped checkpoint writing; best checkpoint path would be {best_path}")
    else:
        print(f"Wrote best checkpoint to {best_path}")
    save_json(
        {
            "total_training_seconds": float(
                time.perf_counter() - training_started
            ),
            "start_epoch": int(start_epoch),
            "completed_epoch": int(completed_epoch),
            "configured_epochs": int(epochs),
            "best_valid_total_loss": (
                float(best_valid) if np.isfinite(best_valid) else None
            ),
            "early_stopping_patience": int(early_stopping_patience),
            "early_stopping_min_delta": float(early_stopping_min_delta),
            "train_rows": int(len(train_dataset)),
            "validation_rows": int(len(valid_dataset)),
            "batch_size": int(batch_size),
            "device": str(device),
            "mixed_precision": bool(use_amp),
            "checkpoint_path": str(best_path),
            "checkpoint_written": bool(
                not skip_checkpoints and best_path.exists()
            ),
        },
        output_dir / "metadata" / "training_runtime.json",
    )
    return best_path


def dataloader_kwargs(training: dict[str, Any], device: str, num_workers: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pin_memory": bool(training.get("pin_memory", str(device).startswith("cuda"))),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        timeout = int(training.get("dataloader_timeout_seconds", 0) or 0)
        if timeout > 0:
            kwargs["timeout"] = timeout
        prefetch = training.get("prefetch_factor")
        if prefetch is not None:
            kwargs["prefetch_factor"] = int(prefetch)
    return kwargs


def compute_length_class_weights_from_pretokenized(
    dataset: PretokenizedLSTMDataset,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
) -> dict[str, Any] | None:
    tensors: dict[str, torch.Tensor] = {}
    payloads: dict[str, Any] = {}
    for column in config.schema.length_bucket_targets:
        length_cfg = config.raw.get(length_loss_config_name(column), {})
        if not bool(length_cfg.get("class_balanced", False)):
            continue
        vocab = categorical_vocabs[column]
        col_idx = config.schema.model_categorical_targets.index(column)
        ids = np.asarray(dataset.categorical_ids[dataset.indices, col_idx], dtype=np.int64)
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
        tensors[column] = torch.tensor(weights, dtype=torch.float32)
        id_to_token = vocab.id_to_token
        payloads[column] = {
            "class_balanced": True,
            "column": column,
            "class_weight_power": power,
            "min_class_weight": float(length_cfg.get("min_class_weight", 0.5)),
            "max_class_weight": float(length_cfg.get("max_class_weight", 5.0)),
            "counts": {id_to_token[idx]: int(counts[idx]) for idx in range(vocab.size)},
            "frequencies": {id_to_token[idx]: float(freqs[idx]) for idx in range(vocab.size)},
            "weights": {id_to_token[idx]: float(weights[idx]) for idx in range(vocab.size)},
        }
    if not tensors:
        return None
    return {"tensor": tensors, "json": payloads}


def length_loss_config_name(column: str) -> str:
    if column == "summary_length_bucket":
        return "summary_length_loss"
    if column == "review_text_length_bucket":
        return "review_text_length_loss"
    return f"{column}_loss"


def run_hierarchical_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_amp: bool,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    loss_weights: dict[str, float],
    text_token_loss_weights: dict[str, dict[str, float]],
    length_class_weights: dict[str, torch.Tensor] | None,
    *,
    graph_encoder: torch.nn.Module | None = None,
    graph_history_index: Any | None = None,
    graph_deterministic: bool = True,
    graph_row_id_offset: int = 0,
    training: bool = True,
) -> dict[str, float]:
    model.train(training)
    if graph_encoder is not None:
        graph_encoder.train(training)
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    weighted_totals: dict[str, float] = {}
    weighted_counts: dict[str, float] = {}
    group_totals: dict[str, float] = {}
    objective_total = 0.0
    structured_objective_total = 0.0
    text_objective_total = 0.0
    objective_rows = 0
    gradient_audit_totals = {"structured": 0.0, "text": 0.0}
    gradient_audit_counts = {"structured": 0, "text": 0}
    mixture_counts = {"clean": 0, "corrupted": 0, "generated": 0}
    profile = bool(config.raw.get("training", {}).get("profile", False))
    max_batches_key = "max_train_batches" if training else "max_valid_batches"
    max_batches = config.raw.get("training", {}).get(max_batches_key)
    max_batches_int = int(max_batches) if max_batches not in (None, "all") else None
    timing_totals: dict[str, float] = {
        "batch_load": 0.0,
        "h2d": 0.0,
        "graph_context": 0.0,
        "structured_forward_loss": 0.0,
        "conditioning": 0.0,
        "text_forward_loss": 0.0,
        "backward_optimizer": 0.0,
        "total_step": 0.0,
    }
    timed_batches = 0
    iterator = tqdm(loader, leave=False, desc="hier_train" if training else "hier_valid") if tqdm is not None else loader
    previous_batch_end = time.perf_counter()
    for batch_idx, batch in enumerate(iterator):
        step_start = time.perf_counter()
        if profile:
            timing_totals["batch_load"] += step_start - previous_batch_end
        batch = move_batch_to_device(batch, device)
        h2d_end = time.perf_counter()
        if profile:
            timing_totals["h2d"] += h2d_end - step_start
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            graph_context, _ = compute_graph_outputs(
                graph_encoder,
                graph_history_index,
                batch,
                device,
                deterministic=graph_deterministic or not training,
                row_id_offset=graph_row_id_offset,
                config=config,
                training=training,
            )
            graph_end = time.perf_counter()
            if profile:
                timing_totals["graph_context"] += graph_end - h2d_end
            structured_loss, structured_component = structured_stage_loss(
                model,
                batch,
                config,
                categorical_vocabs,
                text_tokenizer,
                loss_weights,
                length_class_weights,
                graph_context,
            )
            structured_end = time.perf_counter()
            if profile:
                timing_totals["structured_forward_loss"] += structured_end - graph_end
            conditioning_mode = choose_text_conditioning_mode(config.raw.get("training", {}).get("text_conditioning", {}), training=training)
            mixture_counts[conditioning_mode] += int(batch["foreign_key_ids"].shape[0])
            cat_condition = structured_conditioning_values(
                model,
                batch,
                config,
                categorical_vocabs,
                text_tokenizer,
                graph_context,
                conditioning_mode,
            )
            conditioning_end = time.perf_counter()
            if profile:
                timing_totals["conditioning"] += conditioning_end - structured_end
            text_loss, text_component = text_stage_loss(
                model,
                batch,
                config,
                cat_condition,
                loss_weights,
                text_tokenizer,
                text_token_loss_weights,
                graph_context,
            )
            loss = structured_loss + text_loss
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(
                    "Non-finite hierarchical diffusion loss: "
                    f"structured={float(structured_loss.detach().cpu())}, "
                    f"text={float(text_loss.detach().cpu())}"
                )
            text_end = time.perf_counter()
            if profile:
                timing_totals["text_forward_loss"] += text_end - conditioning_end
        if optimizer is not None:
            gradient_audit_interval = int(
                config.raw.get("training", {}).get(
                    "modality_gradient_audit_interval", 0
                )
                or 0
            )
            if (
                gradient_audit_interval > 0
                and batch_idx % gradient_audit_interval == 0
            ):
                parameters = trainable_parameters(model, graph_encoder)
                gradient_audit_totals["structured"] += loss_gradient_norm(
                    structured_loss, parameters, retain_graph=True
                )
                gradient_audit_counts["structured"] += 1
                gradient_audit_totals["text"] += loss_gradient_norm(
                    text_loss, parameters, retain_graph=True
                )
                gradient_audit_counts["text"] += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model, graph_encoder), 1.0
            )
            if not bool(torch.isfinite(torch.as_tensor(clipped_norm)).all()):
                raise FloatingPointError(
                    "Non-finite gradient norm in hierarchical diffusion training"
                )
            scaler.step(optimizer)
            scaler.update()
        step_end = time.perf_counter()
        if profile:
            timing_totals["backward_optimizer"] += step_end - text_end
            timing_totals["total_step"] += step_end - step_start
            timed_batches += 1
        batch_rows = int(batch["foreign_key_ids"].shape[0])
        objective_total += float(loss.detach().cpu()) * batch_rows
        structured_objective_total += (
            float(structured_loss.detach().cpu()) * batch_rows
        )
        text_objective_total += float(text_loss.detach().cpu()) * batch_rows
        objective_rows += batch_rows
        for prefix, component in [
            ("structured", structured_component),
            ("text", text_component),
        ]:
            for key, stats in component.items():
                name = f"{prefix}_{key}"
                totals[name] = totals.get(name, 0.0) + float(stats["loss_sum"])
                counts[name] = counts.get(name, 0.0) + float(stats["count"])
                if "weighted_mean_loss" in stats:
                    weighted_totals[name] = weighted_totals.get(
                        name, 0.0
                    ) + float(stats["weighted_mean_loss"]) * batch_rows
                    weighted_counts[name] = weighted_counts.get(
                        name, 0.0
                    ) + batch_rows
                    group = str(stats.get("loss_group", prefix))
                    group_totals[group] = group_totals.get(group, 0.0) + (
                        float(stats["weighted_mean_loss"]) * batch_rows
                    )
        previous_batch_end = time.perf_counter()
        if max_batches_int is not None and batch_idx + 1 >= max_batches_int:
            break
    metrics = {
        f"loss_{key}": float(total / max(counts.get(key, 1.0), 1.0))
        for key, total in sorted(totals.items())
    }
    for key, total in sorted(weighted_totals.items()):
        metrics[f"weighted_loss_{key}"] = float(
            total / max(weighted_counts.get(key, 1.0), 1.0)
        )
    metrics["structured_objective_loss"] = float(
        structured_objective_total / max(objective_rows, 1)
    )
    metrics["text_objective_loss"] = float(
        text_objective_total / max(objective_rows, 1)
    )
    metrics["total_loss"] = float(objective_total / max(objective_rows, 1))
    for group, total in sorted(group_totals.items()):
        metrics[f"loss_group_{group}"] = float(
            total / max(objective_rows, 1)
        )
    for modality, total in gradient_audit_totals.items():
        count = gradient_audit_counts[modality]
        if count:
            metrics[f"gradient_norm_from_{modality}_loss"] = float(
                total / count
            )
    total_rows = max(sum(mixture_counts.values()), 1)
    for key, value in mixture_counts.items():
        metrics[f"text_conditioning_{key}_rate"] = float(value / total_rows)
    if profile and timed_batches > 0:
        for key, value in sorted(timing_totals.items()):
            metrics[f"runtime_{key}_seconds"] = float(value / timed_batches)
        metrics["runtime_num_timed_batches"] = float(timed_batches)
    return metrics


def structured_stage_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    loss_weights: dict[str, float],
    length_class_weights: dict[str, torch.Tensor] | None,
    graph_context: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    cat_input = batch["categorical_input_ids"]
    text_input, text_attention = inactive_text_inputs(batch, config, text_tokenizer)
    logits = model(batch["foreign_key_ids"], batch["datetime_values"], cat_input, text_input, text_attention, batch["diffusion_t"], graph_context)
    text_labels = {column: torch.full_like(batch["text_labels"][column], -100) for column in config.schema.text_targets}
    loss_batch = dict(batch)
    loss_batch["text_labels"] = text_labels
    return denoising_loss(
        logits,
        loss_batch,
        config.schema,
        loss_weights,
        text_tokenizer=text_tokenizer,
        length_class_weights=length_class_weights or {},
        loss_group_weights=dict(config.raw.get("loss_group_weights", {})),
        field_loss_groups=dict(
            config.raw.get("loss_groups", {}).get("field_groups", {})
        ),
    )


def text_stage_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: ConditionalTABDLMConfig,
    categorical_input_ids: torch.Tensor,
    loss_weights: dict[str, float],
    text_tokenizer: SimpleTextTokenizer,
    text_token_loss_weights: dict[str, dict[str, float]],
    graph_context: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    logits = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        categorical_input_ids,
        batch["text_input_ids"],
        batch["text_attention"],
        batch["diffusion_t"],
        graph_context,
    )
    cat_labels = torch.full_like(batch["categorical_labels"], -100)
    loss_batch = dict(batch)
    loss_batch["categorical_labels"] = cat_labels
    return denoising_loss(
        logits,
        loss_batch,
        config.schema,
        loss_weights,
        text_tokenizer=text_tokenizer,
        text_token_loss_weights=text_token_loss_weights,
        loss_group_weights=dict(config.raw.get("loss_group_weights", {})),
        field_loss_groups=dict(
            config.raw.get("loss_groups", {}).get("field_groups", {})
        ),
    )


def structured_conditioning_values(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    graph_context: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    if mode == "clean":
        return batch["categorical_clean_ids"]
    if mode == "corrupted":
        corruption_cfg = (
            config.raw.get("training", {})
            .get("text_conditioning", {})
            .get("corruption", {})
        )
        return corrupt_categorical_values(
            batch["categorical_clean_ids"],
            categorical_vocabs,
            config.schema,
            corruption_cfg,
        )
    if mode == "generated":
        with torch.no_grad():
            cat_input = batch["categorical_input_ids"]
            text_input, text_attention = inactive_text_inputs(batch, config, text_tokenizer)
            logits = model(batch["foreign_key_ids"], batch["datetime_values"], cat_input, text_input, text_attention, batch["diffusion_t"], graph_context)
            generated = batch["categorical_clean_ids"].clone()
            for idx, column in enumerate(config.schema.model_categorical_targets):
                if column in config.schema.length_bucket_targets:
                    sampled = sample_length_bucket_logits(logits["categorical"][column], column, categorical_vocabs[column], None, config.schema, temperature=1.0)
                else:
                    sampled = sample_categorical_logits(logits["categorical"][column], column, categorical_vocabs[column], temperature=1.0)
                generated[:, idx] = sampled
            return generated.detach()
    raise ValueError(f"Unsupported text conditioning mode: {mode}")


def choose_text_conditioning_mode(cfg: dict[str, Any], *, training: bool) -> str:
    if not training:
        eval_mode = str(cfg.get("validation_mode", cfg.get("eval_mode", "generated")))
        if eval_mode in {"clean", "corrupted", "generated"}:
            return eval_mode
        return "generated"
    mode = str(cfg.get("mode", "mixed"))
    if mode in {"clean", "corrupted", "generated"}:
        return mode
    clean = float(cfg.get("clean_probability", 0.5))
    corrupted = float(cfg.get("corrupted_probability", 0.25))
    generated = float(cfg.get("generated_probability", 0.25))
    total = max(clean + corrupted + generated, 1e-9)
    draw = torch.rand(()).item() * total
    if draw < clean:
        return "clean"
    if draw < clean + corrupted:
        return "corrupted"
    return "generated"


def corrupt_categorical_values(
    clean: torch.Tensor,
    categorical_vocabs: dict[str, CategoryVocab],
    schema: Any,
    corruption_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply configurable, schema-valid condition corruption."""

    cfg = dict(corruption_config or {})
    default_replace = float(cfg.get("replacement_probability", 1.0))
    default_mask = float(cfg.get("mask_probability", 0.0))
    length_step_probability = float(
        cfg.get("length_bucket_step_probability", 0.0)
    )
    per_field = dict(cfg.get("per_field", {}) or {})
    out = clean.clone()
    for idx, column in enumerate(schema.model_categorical_targets):
        vocab = categorical_vocabs[column]
        field_cfg = dict(per_field.get(column, {}) or {})
        allow_missing_replacement = bool(
            field_cfg.get(
                "allow_missing_replacement",
                cfg.get("allow_missing_replacement", False),
            )
        )
        replace_probability = float(
            field_cfg.get("replacement_probability", default_replace)
        )
        mask_probability = float(field_cfg.get("mask_probability", default_mask))
        if column in schema.length_bucket_targets and length_step_probability > 0:
            perturb = (
                torch.rand(clean.shape[0], device=clean.device)
                < float(
                    field_cfg.get(
                        "length_bucket_step_probability",
                        length_step_probability,
                    )
                )
            )
            direction = torch.where(
                torch.rand(clean.shape[0], device=clean.device) < 0.5,
                -torch.ones(clean.shape[0], dtype=torch.long, device=clean.device),
                torch.ones(clean.shape[0], dtype=torch.long, device=clean.device),
            )
            bucket_ids = [
                vocab.token_to_id[name]
                for name in schema.buckets_for_length_bucket(column)
                if name in vocab.token_to_id
            ]
            if not bucket_ids:
                raise ValueError(
                    f"No configured length buckets are present in vocab {column!r}"
                )
            bucket_tensor = torch.tensor(
                bucket_ids, dtype=torch.long, device=clean.device
            )
            distances = (
                clean[:, idx].view(-1, 1) == bucket_tensor.view(1, -1)
            )
            current_position = distances.long().argmax(dim=1)
            stepped_position = (current_position + direction).clamp(
                0, len(bucket_ids) - 1
            )
            stepped = bucket_tensor[stepped_position]
            out[perturb, idx] = stepped[perturb]
        replace = (
            torch.rand(clean.shape[0], device=clean.device)
            < replace_probability
        )
        valid_ids = [
            token_id
            for token, token_id in vocab.token_to_id.items()
            if allow_missing_replacement or token != "<missing>"
        ]
        if not valid_ids:
            valid_ids = list(range(vocab.size))
        candidate_ids = torch.tensor(
            valid_ids, dtype=torch.long, device=clean.device
        )
        sampled_position = torch.randint(
            0,
            len(valid_ids),
            (clean.shape[0],),
            dtype=torch.long,
            device=clean.device,
        )
        replacement = candidate_ids[sampled_position]
        if len(valid_ids) > 1:
            same = replacement == clean[:, idx]
            replacement[same] = candidate_ids[
                (sampled_position[same] + 1) % len(valid_ids)
            ]
        out[replace, idx] = replacement[replace]
        mask = torch.rand(clean.shape[0], device=clean.device) < mask_probability
        out[mask, idx] = vocab.mask_id
    return out


def inactive_text_inputs(batch: dict[str, Any], config: ConditionalTABDLMConfig, text_tokenizer: SimpleTextTokenizer) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_input = {}
    text_attention = {}
    for column in config.schema.text_targets:
        clean = batch["text_clean_ids"][column]
        values = torch.full_like(clean, text_tokenizer.pad_id)
        if values.shape[1] > 0:
            values[:, 0] = text_tokenizer.bos_id
        text_input[column] = values
        text_attention[column] = torch.zeros_like(clean, dtype=torch.long)
    return text_input, text_attention


def loss_gradient_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = torch.zeros((), dtype=torch.float32, device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().float().pow(2).sum()
    value = total.sqrt()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Non-finite modality gradient norm")
    return float(value.cpu())


def save_hierarchical_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    epoch: int,
    valid_metrics: dict[str, float],
    graph_encoder: torch.nn.Module | None,
) -> None:
    save_checkpoint(
        path,
        unwrap_compiled_module(model),
        config,
        categorical_vocabs,
        text_tokenizer,
        epoch,
        valid_metrics,
        graph_encoder=graph_encoder,
    )
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["hierarchical_diagnostics_version"] = 1
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = None
    checkpoint["generation_plan"] = generation_plan_from_config(config.raw, config.schema).to_dict()
    checkpoint["training_conditioning_mixture"] = config.raw.get("training", {}).get("text_conditioning", {})
    checkpoint["loss_weights"] = config.raw.get("loss_weights", {})
    checkpoint["loss_group_weights"] = config.raw.get(
        "loss_group_weights", {}
    )
    checkpoint["loss_groups"] = config.raw.get("loss_groups", {})
    checkpoint["text_token_loss_weights"] = {
        "summary_token_loss_weights": config.raw.get("summary_token_loss_weights", {}),
        "review_text_token_loss_weights": config.raw.get("review_text_token_loss_weights", {}),
    }
    torch.save(checkpoint, path)


def torch_load_concat_frames(train_frame, valid_frame):
    import pandas as pd

    return pd.concat([train_frame, valid_frame], ignore_index=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
