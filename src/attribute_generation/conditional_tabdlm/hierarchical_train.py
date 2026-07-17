"""Training loop for hierarchical structured-then-text Conditional TABDLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import ConditionalTABDLMDataset, load_category_vocabs, load_prepared_tables, load_text_tokenizer, make_collate_fn
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_schema import assert_valid_graph_conditioning, graph_conditioning_enabled, graph_metadata
from .hierarchical_schema import generation_plan_from_config
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

    train_frame, valid_frame, _ = load_prepared_tables(config)
    use_graph_context = graph_conditioning_enabled(config.raw)
    if use_graph_context:
        assert_valid_graph_conditioning(config.raw)

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

    train_history_index = build_temporal_history_index(train_frame, config, seed=seed) if use_graph_context else None
    valid_graph_frame = torch_load_concat_frames(train_frame, valid_frame) if use_graph_context else valid_frame
    valid_history_index = build_temporal_history_index(valid_graph_frame, config, seed=seed + 1) if use_graph_context else None
    valid_row_id_offset = len(train_frame) if use_graph_context else 0
    if use_graph_context:
        write_temporal_graph_metadata(train_frame, config, output_dir / "graph", source="real_training_rows", seed=seed)

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
    length_weights = compute_length_class_weights(train_frame, config, categorical_vocabs, text_tokenizer)
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

    for epoch in range(start_epoch, epochs + 1):
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
        save_hierarchical_checkpoint(last_path, model, optimizer, config, categorical_vocabs, text_tokenizer, epoch, valid_metrics, graph_encoder)
        if current_valid < best_valid - early_stopping_min_delta:
            best_valid = current_valid
            epochs_without_improvement = 0
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
    print(f"Wrote best checkpoint to {best_path}")
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
            text_end = time.perf_counter()
            if profile:
                timing_totals["text_forward_loss"] += text_end - conditioning_end
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model, graph_encoder), 1.0)
            scaler.step(optimizer)
            scaler.update()
        step_end = time.perf_counter()
        if profile:
            timing_totals["backward_optimizer"] += step_end - text_end
            timing_totals["total_step"] += step_end - step_start
            timed_batches += 1
        for prefix, component in [("structured", structured_component), ("text", text_component)]:
            for key, stats in component.items():
                name = f"{prefix}_{key}"
                totals[name] = totals.get(name, 0.0) + float(stats["loss_sum"])
                counts[name] = counts.get(name, 0.0) + float(stats["count"])
        previous_batch_end = time.perf_counter()
        if max_batches_int is not None and batch_idx + 1 >= max_batches_int:
            break
    metrics = {
        f"loss_{key}": float(total / max(counts.get(key, 1.0), 1.0))
        for key, total in sorted(totals.items())
    }
    metrics["total_loss"] = float(sum(metrics.values()))
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
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    cat_input = batch["categorical_input_ids"]
    text_input, text_attention = inactive_text_inputs(batch, config, text_tokenizer)
    logits = model(batch["foreign_key_ids"], batch["datetime_values"], cat_input, text_input, text_attention, batch["diffusion_t"], graph_context)
    text_labels = {column: torch.full_like(batch["text_labels"][column], -100) for column in config.schema.text_targets}
    loss_batch = dict(batch)
    loss_batch["text_labels"] = text_labels
    return denoising_loss(logits, loss_batch, config.schema, loss_weights, text_tokenizer=text_tokenizer, length_class_weights=length_class_weights or {})


def text_stage_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: ConditionalTABDLMConfig,
    categorical_input_ids: torch.Tensor,
    loss_weights: dict[str, float],
    text_tokenizer: SimpleTextTokenizer,
    text_token_loss_weights: dict[str, dict[str, float]],
    graph_context: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
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
    return denoising_loss(logits, loss_batch, config.schema, loss_weights, text_tokenizer=text_tokenizer, text_token_loss_weights=text_token_loss_weights)


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
        return corrupt_categorical_values(batch["categorical_clean_ids"], categorical_vocabs, config.schema)
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


def corrupt_categorical_values(clean: torch.Tensor, categorical_vocabs: dict[str, CategoryVocab], schema: Any) -> torch.Tensor:
    out = clean.clone()
    for idx, column in enumerate(schema.model_categorical_targets):
        vocab = categorical_vocabs[column]
        replacement = torch.randint(0, vocab.size, (clean.shape[0],), dtype=torch.long, device=clean.device)
        out[:, idx] = replacement
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
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = None
    checkpoint["generation_plan"] = generation_plan_from_config(config.raw, config.schema).to_dict()
    checkpoint["training_conditioning_mixture"] = config.raw.get("training", {}).get("text_conditioning", {})
    checkpoint["loss_weights"] = config.raw.get("loss_weights", {})
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
