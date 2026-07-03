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

from .dataset import (
    ConditionalTABDLMDataset,
    load_category_vocabs,
    load_prepared_tables,
    load_text_tokenizer,
    make_collate_fn,
)
from .model import ConditionalTABDLM
from .schema import ConditionalTABDLMConfig
from .tokenization import CategoryVocab, SimpleTextTokenizer
from .utils import ensure_dir, save_yaml, set_seed


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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = resolve_device(device or str(training.get("device", "auto")))
    model = build_model(config, categorical_vocabs, text_tokenizer).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", training.get("lr", 3e-4))),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    use_amp = bool(training.get("mixed_precision", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    best_valid = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    epochs = int(training.get("epochs", 5))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        valid_metrics = run_epoch(model, valid_loader, None, scaler, device, use_amp)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
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
        )
        if valid_metrics["total_loss"] < best_valid:
            best_valid = float(valid_metrics["total_loss"])
            save_checkpoint(
                best_path,
                model,
                config,
                categorical_vocabs,
                text_tokenizer,
                epoch,
                valid_metrics,
            )
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
    )


def run_epoch(
    model: ConditionalTABDLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train" if training else "valid")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                foreign_key_ids=batch["foreign_key_ids"],
                datetime_values=batch["datetime_values"],
                categorical_input_ids=batch["categorical_input_ids"],
                text_input_ids=batch["text_input_ids"],
                text_attention=batch["text_attention"],
                diffusion_t=batch["diffusion_t"],
            )
            loss, component = denoising_loss(logits, batch, model.schema)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        for key, stats in component.items():
            totals[key] = totals.get(key, 0.0) + float(stats["loss_sum"])
            counts[key] = counts.get(key, 0) + int(stats["count"])

    metrics = {}
    total_sum = 0.0
    total_count = 0
    for key in sorted(totals):
        count = max(counts.get(key, 0), 1)
        metrics[f"{key}_loss"] = float(totals[key] / count)
        total_sum += totals[key]
        total_count += counts.get(key, 0)
    metrics["total_loss"] = float(total_sum / max(total_count, 1))
    return metrics


def denoising_loss(
    logits: dict[str, Any],
    batch: dict[str, Any],
    schema,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    losses: list[torch.Tensor] = []
    component: dict[str, dict[str, float | int]] = {}
    cat_labels = batch["categorical_labels"]
    for idx, column in enumerate(schema.categorical_targets):
        labels = cat_labels[:, idx]
        count = int((labels != -100).sum().detach().cpu())
        if count == 0:
            continue
        loss_sum = F.cross_entropy(logits["categorical"][column], labels, ignore_index=-100, reduction="sum")
        losses.append(loss_sum)
        component[column] = {"loss_sum": float(loss_sum.detach().cpu()), "count": count}

    for column in schema.text_targets:
        labels = batch["text_labels"][column]
        count = int((labels != -100).sum().detach().cpu())
        if count == 0:
            continue
        loss_sum = F.cross_entropy(
            logits["text"][column].reshape(-1, logits["text"][column].shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        losses.append(loss_sum)
        component[column] = {"loss_sum": float(loss_sum.detach().cpu()), "count": count}

    if not losses:
        zero = batch["foreign_key_ids"].float().sum() * 0.0
        return zero, {}
    denom = sum(stats["count"] for stats in component.values())
    return torch.stack(losses).sum() / max(int(denom), 1), component


def save_checkpoint(
    path: str | Path,
    model: ConditionalTABDLM,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    epoch: int,
    valid_metrics: dict[str, float],
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
        },
        path,
    )


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

