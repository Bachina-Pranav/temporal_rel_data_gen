"""Joint row-latent LSTM generator for scalable full review-text attributes."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constrained import decode_category_id, mask_invalid_category_logits, validate_output_categoricals, valid_category_values
from .dataset import (
    ConditionalTABDLMDataset,
    load_category_vocabs,
    load_prepared_tables,
    load_text_tokenizer,
    prepare_rel_amazon_data,
)
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_encoder import TemporalStructureOnlyGraphEncoder
from .graph_schema import assert_valid_graph_conditioning, graph_conditioning_enabled, graph_metadata
from .model import DateTimeEncoder
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer, stable_hash_bucket
from .train import (
    build_graph_encoder,
    compute_graph_outputs,
    compute_length_class_weights,
    maybe_limit_rows,
    parameter_grad_norm,
    resolve_device,
    validation_row_cap,
)
from .utils import ensure_dir, jsonable, save_json, save_yaml, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class JointLSTMRelationalAttributeGenerator(nn.Module):
    """One shared row latent with categorical heads and autoregressive text decoders."""

    def __init__(
        self,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
        num_hash_buckets: int = 262144,
        id_embedding_dim: int = 128,
        datetime_embedding_dim: int = 64,
        graph_context_dim: int = 256,
        row_hidden_dim: int = 384,
        latent_noise_dim: int = 128,
        categorical_context_dim: int = 48,
        text_embedding_dim: int = 256,
        text_hidden_dim: int = 384,
        text_num_layers: int = 2,
        dropout: float = 0.1,
        decoder_type: str = "lstm",
        use_graph_context: bool = True,
    ):
        super().__init__()
        self.schema = schema
        self.num_hash_buckets = int(num_hash_buckets)
        self.id_embedding_dim = int(id_embedding_dim)
        self.datetime_embedding_dim = int(datetime_embedding_dim)
        self.graph_context_dim = int(graph_context_dim)
        self.row_hidden_dim = int(row_hidden_dim)
        self.latent_noise_dim = int(latent_noise_dim)
        self.categorical_context_dim = int(categorical_context_dim)
        self.text_embedding_dim = int(text_embedding_dim)
        self.text_hidden_dim = int(text_hidden_dim)
        self.text_num_layers = int(text_num_layers)
        self.dropout = float(dropout)
        self.decoder_type = str(decoder_type).lower()
        self.use_graph_context = bool(use_graph_context)
        self.text_vocab_size = int(text_tokenizer.vocab_size)
        self.text_pad_id = int(text_tokenizer.pad_id)

        self.foreign_key_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_hash_buckets, self.id_embedding_dim) for _ in schema.foreign_key_columns]
        )
        self.datetime_encoder = DateTimeEncoder(self.datetime_embedding_dim)
        cond_dim = len(schema.foreign_key_columns) * self.id_embedding_dim + len(schema.datetime_columns) * self.datetime_embedding_dim
        if self.use_graph_context:
            cond_dim += self.graph_context_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(cond_dim, self.row_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.row_hidden_dim),
        )
        self.row_encoder = nn.Sequential(
            nn.Linear(self.row_hidden_dim + self.latent_noise_dim, self.row_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.row_hidden_dim),
            nn.Linear(self.row_hidden_dim, self.row_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.row_hidden_dim),
        )
        self.categorical_heads = nn.ModuleDict(
            {
                column: nn.Linear(self.row_hidden_dim, categorical_vocabs[column].size)
                for column in schema.model_categorical_targets
            }
        )
        self.categorical_context_embeddings = nn.ModuleDict(
            {
                column: nn.Embedding(categorical_vocabs[column].size, self.categorical_context_dim)
                for column in schema.model_categorical_targets
            }
        )
        self.text_embedding = nn.Embedding(self.text_vocab_size, self.text_embedding_dim, padding_idx=text_tokenizer.pad_id)
        self.text_decoders = nn.ModuleDict()
        self.text_heads = nn.ModuleDict()
        self.text_initializers = nn.ModuleDict()
        decoder_context_dim = self.row_hidden_dim + len(schema.model_categorical_targets) * self.categorical_context_dim
        state_multiplier = 2 if self.decoder_type == "lstm" else 1
        for column in schema.text_targets:
            if self.decoder_type == "gru":
                self.text_decoders[column] = nn.GRU(
                    self.text_embedding_dim,
                    self.text_hidden_dim,
                    num_layers=self.text_num_layers,
                    batch_first=True,
                    dropout=self.dropout if self.text_num_layers > 1 else 0.0,
                )
            else:
                self.text_decoders[column] = nn.LSTM(
                    self.text_embedding_dim,
                    self.text_hidden_dim,
                    num_layers=self.text_num_layers,
                    batch_first=True,
                    dropout=self.dropout if self.text_num_layers > 1 else 0.0,
                )
            self.text_heads[column] = nn.Linear(self.text_hidden_dim, self.text_vocab_size)
            self.text_initializers[column] = nn.Sequential(
                nn.Linear(decoder_context_dim, self.text_hidden_dim * self.text_num_layers * state_multiplier),
                nn.Tanh(),
            )

    def encode_condition(
        self,
        foreign_key_ids: torch.Tensor,
        datetime_values: torch.Tensor,
        graph_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for idx, embedding in enumerate(self.foreign_key_embeddings):
            pieces.append(embedding(foreign_key_ids[:, idx]))
        datetime_features = self.datetime_encoder(datetime_values)
        for idx in range(datetime_features.shape[1]):
            pieces.append(datetime_features[:, idx, :])
        if self.use_graph_context:
            if graph_context is None:
                graph_context = torch.zeros(
                    foreign_key_ids.shape[0],
                    self.graph_context_dim,
                    dtype=torch.float32,
                    device=foreign_key_ids.device,
                )
            pieces.append(graph_context.float())
        return self.condition_mlp(torch.cat(pieces, dim=1))

    def row_latent(self, condition: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(condition.shape[0], self.latent_noise_dim, dtype=condition.dtype, device=condition.device)
        return self.row_encoder(torch.cat([condition, noise], dim=1))

    def categorical_logits(self, row_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {column: head(row_latent) for column, head in self.categorical_heads.items()}

    def categorical_context(self, row_latent: torch.Tensor, categorical_ids: torch.Tensor) -> torch.Tensor:
        pieces = [row_latent]
        for idx, column in enumerate(self.schema.model_categorical_targets):
            pieces.append(self.categorical_context_embeddings[column](categorical_ids[:, idx]))
        return torch.cat(pieces, dim=1)

    def initial_state(self, column: str, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        projected = self.text_initializers[column](context)
        batch = int(context.shape[0])
        if self.decoder_type == "gru":
            return projected.view(batch, self.text_num_layers, self.text_hidden_dim).transpose(0, 1).contiguous()
        projected = projected.view(batch, 2, self.text_num_layers, self.text_hidden_dim)
        hidden = projected[:, 0].transpose(0, 1).contiguous()
        cell = projected[:, 1].transpose(0, 1).contiguous()
        return hidden, cell

    def forward(
        self,
        foreign_key_ids: torch.Tensor,
        datetime_values: torch.Tensor,
        categorical_ids: torch.Tensor,
        text_ids: dict[str, torch.Tensor],
        graph_context: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        condition = self.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
        row = self.row_latent(condition, noise=noise)
        cat_logits = self.categorical_logits(row)
        context = self.categorical_context(row, categorical_ids)
        text_logits: dict[str, torch.Tensor] = {}
        for column in self.schema.text_targets:
            teacher = text_ids[column][:, :-1].contiguous()
            embedded = self.text_embedding(teacher)
            output, _ = self.text_decoders[column](embedded, self.initial_state(column, context))
            text_logits[column] = self.text_heads[column](output)
        return {"categorical": cat_logits, "text": text_logits, "row_latent": row}

    @torch.no_grad()
    def generate(
        self,
        foreign_key_ids: torch.Tensor,
        datetime_values: torch.Tensor,
        categorical_vocabs: dict[str, CategoryVocab],
        tokenizer: SimpleTextTokenizer,
        graph_context: torch.Tensor | None = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        min_tokens: dict[str, int] | None = None,
        repetition_penalty: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        min_tokens = min_tokens or {}
        repetition_penalty = repetition_penalty or {}
        condition = self.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
        row = self.row_latent(condition)
        cat_logits = self.categorical_logits(row)
        sampled_cat_columns: list[torch.Tensor] = []
        decoded_cats: dict[str, list[Any]] = {}
        for column in self.schema.model_categorical_targets:
            sampled = sample_categorical_column(cat_logits[column], column, categorical_vocabs[column], temperature=temperature, top_p=top_p)
            sampled_cat_columns.append(sampled)
            decoded_cats[column] = [decode_category_id(column, categorical_vocabs[column], idx) for idx in sampled.detach().cpu().tolist()]
        categorical_ids = torch.stack(sampled_cat_columns, dim=1) if sampled_cat_columns else torch.empty(
            (foreign_key_ids.shape[0], 0), dtype=torch.long, device=foreign_key_ids.device
        )
        context = self.categorical_context(row, categorical_ids)
        text_ids: dict[str, torch.Tensor] = {}
        decoded_text: dict[str, list[str]] = {}
        lengths: dict[str, list[int]] = {}
        for column in self.schema.text_targets:
            bucket_column = length_bucket_column_for_text(self.schema, column)
            bucket_names = decoded_cats.get(bucket_column, [None] * int(foreign_key_ids.shape[0])) if bucket_column else [None] * int(foreign_key_ids.shape[0])
            ids = self.generate_text_column(
                column,
                context,
                bucket_names,
                tokenizer,
                temperature=temperature,
                top_p=top_p,
                min_content_tokens=int(min_tokens.get(column, 0)),
                repetition_penalty=float(repetition_penalty.get(column, 1.0)),
            )
            text_ids[column] = ids
            decoded_text[column] = [tokenizer.decode(row_ids) for row_ids in ids.detach().cpu().tolist()]
            lengths[column] = [tokenizer.content_length(row_ids) for row_ids in ids.detach().cpu().tolist()]
        return {
            "categorical_ids": categorical_ids,
            "categorical": decoded_cats,
            "text_ids": text_ids,
            "text": decoded_text,
            "text_lengths": lengths,
        }

    def generate_text_column(
        self,
        column: str,
        context: torch.Tensor,
        bucket_names: list[Any],
        tokenizer: SimpleTextTokenizer,
        *,
        temperature: float,
        top_p: float,
        min_content_tokens: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        device = context.device
        batch = int(context.shape[0])
        max_len = int(self.schema.text_max_lengths[column])
        max_content = tokenizer.max_content_tokens(max_len)
        lows, highs = length_bounds_for_generation(self.schema, column, bucket_names, max_content, min_content_tokens)
        output = torch.full((batch, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
        output[:, 0] = tokenizer.bos_id
        active = torch.ones(batch, dtype=torch.bool, device=device)
        input_ids = torch.full((batch,), tokenizer.bos_id, dtype=torch.long, device=device)
        state = self.initial_state(column, context)
        max_steps = int(max(highs) + 1) if highs else max_content + 1
        max_steps = max(1, min(max_steps, max_len - 1))
        for step in range(1, max_steps + 1):
            active_idx = torch.where(active)[0]
            if int(active_idx.numel()) == 0:
                break
            step_input = input_ids.index_select(0, active_idx).view(-1, 1)
            active_state = select_state(state, active_idx, self.decoder_type)
            embedded = self.text_embedding(step_input)
            decoded, new_state = self.text_decoders[column](embedded, active_state)
            state = scatter_state(state, new_state, active_idx, self.decoder_type)
            logits = self.text_heads[column](decoded[:, -1, :])
            sampled = sample_text_step(
                logits,
                tokenizer,
                step=step,
                lows=[lows[int(idx)] for idx in active_idx.detach().cpu().tolist()],
                highs=[highs[int(idx)] for idx in active_idx.detach().cpu().tolist()],
                previous_ids=output.index_select(0, active_idx)[:, :step],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            output[active_idx, step] = sampled
            input_ids[active_idx] = sampled
            active_lows = torch.tensor([lows[int(idx)] for idx in active_idx.detach().cpu().tolist()], dtype=torch.long, device=device)
            active_highs = torch.tensor([highs[int(idx)] for idx in active_idx.detach().cpu().tolist()], dtype=torch.long, device=device)
            content_so_far = step - 1
            finished = ((sampled == tokenizer.eos_id) & (content_so_far >= active_lows)) | (content_so_far >= active_highs)
            if bool(finished.any()):
                finished_idx = active_idx[finished]
                missing_eos = output[finished_idx, step] != tokenizer.eos_id
                if bool(missing_eos.any()):
                    output[finished_idx[missing_eos], step] = tokenizer.eos_id
                active[finished_idx] = False
        if bool(active.any()):
            active_idx = torch.where(active)[0]
            eos_pos = torch.tensor([min(highs[int(idx)] + 1, max_len - 1) for idx in active_idx.detach().cpu().tolist()], dtype=torch.long, device=device)
            output[active_idx, eos_pos] = tokenizer.eos_id
        return output

    def to_config(self) -> dict[str, Any]:
        return {
            "model_family": "joint_lstm_generator",
            "num_hash_buckets": self.num_hash_buckets,
            "id_embedding_dim": self.id_embedding_dim,
            "datetime_embedding_dim": self.datetime_embedding_dim,
            "graph_context_dim": self.graph_context_dim,
            "row_hidden_dim": self.row_hidden_dim,
            "latent_noise_dim": self.latent_noise_dim,
            "categorical_context_dim": self.categorical_context_dim,
            "text_embedding_dim": self.text_embedding_dim,
            "text_hidden_dim": self.text_hidden_dim,
            "text_num_layers": self.text_num_layers,
            "dropout": self.dropout,
            "decoder_type": self.decoder_type,
            "use_graph_context": self.use_graph_context,
            "text_vocab_size": self.text_vocab_size,
        }


def make_lstm_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "foreign_key_ids": torch.stack([sample["foreign_key_ids"] for sample in samples], dim=0),
        "datetime_values": torch.stack([sample["datetime_values"] for sample in samples], dim=0),
        "categorical_ids": torch.stack([sample["categorical_ids"] for sample in samples], dim=0),
        "text_ids": {
            column: torch.stack([sample["text_ids"][column] for sample in samples], dim=0)
            for column in samples[0]["text_ids"]
        },
        "row_id": torch.stack([sample["row_id"] for sample in samples], dim=0),
    }


def build_lstm_model(
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
) -> JointLSTMRelationalAttributeGenerator:
    model_cfg = config.raw.get("model", {})
    id_cfg = config.raw.get("id_encoding", {})
    dt_cfg = config.raw.get("datetime_encoding", {})
    decoder_cfg = config.raw.get("text_decoder", {})
    return JointLSTMRelationalAttributeGenerator(
        schema=config.schema,
        categorical_vocabs=categorical_vocabs,
        text_tokenizer=tokenizer,
        num_hash_buckets=int(id_cfg.get("num_buckets", 262144)),
        id_embedding_dim=int(id_cfg.get("embedding_dim", 128)),
        datetime_embedding_dim=int(dt_cfg.get("embedding_dim", 64)),
        graph_context_dim=int(model_cfg.get("graph_context_dim", config.raw.get("graph_conditioning", {}).get("graph_encoder", {}).get("output_dim", 256))),
        row_hidden_dim=int(model_cfg.get("row_hidden_dim", model_cfg.get("hidden_dim", 384))),
        latent_noise_dim=int(model_cfg.get("latent_noise_dim", 128)),
        categorical_context_dim=int(model_cfg.get("categorical_context_dim", 48)),
        text_embedding_dim=int(decoder_cfg.get("embedding_dim", 256)),
        text_hidden_dim=int(decoder_cfg.get("hidden_dim", 384)),
        text_num_layers=int(decoder_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", decoder_cfg.get("dropout", 0.1))),
        decoder_type=str(decoder_cfg.get("type", "lstm")),
        use_graph_context=bool(model_cfg.get("use_graph_context", graph_conditioning_enabled(config.raw))),
    )


def train_lstm_from_config(config: ConditionalTABDLMConfig, device: str | None = None) -> Path:
    training = config.raw.get("training", {})
    requested_batch_size = int(training.get("batch_size", 256))
    auto_reduce = bool(training.get("auto_reduce_batch_size", False))
    batch_sizes = (
        candidate_train_batch_sizes(requested_batch_size, int(training.get("min_batch_size", requested_batch_size)))
        if auto_reduce
        else [requested_batch_size]
    )
    for attempt_idx, batch_size in enumerate(batch_sizes):
        retry_message = None
        try:
            return _train_lstm_from_config_once(
                config,
                device=device,
                batch_size_override=batch_size,
                requested_batch_size=requested_batch_size,
                auto_reduce_batch_size=auto_reduce,
            )
        except RuntimeError as exc:
            if not (auto_reduce and is_cuda_oom(exc) and attempt_idx < len(batch_sizes) - 1):
                raise
            next_batch_size = batch_sizes[attempt_idx + 1]
            retry_message = (
                f"CUDA OOM while training LSTM attribute generator at batch_size={batch_size}; "
                f"retrying with batch_size={next_batch_size}."
            )
        if retry_message is not None:
            clear_cuda_after_oom()
            print(retry_message, flush=True)
    raise RuntimeError("LSTM training failed before producing a checkpoint")


def _train_lstm_from_config_once(
    config: ConditionalTABDLMConfig,
    device: str | None = None,
    batch_size_override: int | None = None,
    requested_batch_size: int | None = None,
    auto_reduce_batch_size: bool = False,
) -> Path:
    training = config.raw.get("training", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    metadata_dir = ensure_dir(output_dir / "metadata")
    save_yaml(config.to_dict(), output_dir / "config_resolved.yaml")

    train_frame, valid_frame, _ = load_prepared_tables(config)
    train_frame = maybe_limit_rows(train_frame, training.get("max_rows"), seed)
    valid_frame = maybe_limit_rows(valid_frame, validation_row_cap(training.get("max_rows"), len(valid_frame)), seed + 1)
    use_graph_context = graph_conditioning_enabled(config.raw)
    if use_graph_context:
        assert_valid_graph_conditioning(config.raw)

    categorical_vocabs = load_category_vocabs(config)
    tokenizer = load_text_tokenizer(config)
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    train_dataset = ConditionalTABDLMDataset(train_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets)
    valid_dataset = ConditionalTABDLMDataset(valid_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets)
    batch_size = int(batch_size_override if batch_size_override is not None else training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=make_lstm_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=make_lstm_collate_fn)

    device = resolve_device(device or str(training.get("device", "auto")))
    model = build_lstm_model(config, categorical_vocabs, tokenizer).to(device)
    graph_encoder = build_graph_encoder(config, categorical_vocabs, tokenizer).to(device) if use_graph_context else None
    train_history_index = build_temporal_history_index(train_frame, config, seed=seed) if use_graph_context else None
    valid_graph_frame = pd.concat([train_frame, valid_frame], ignore_index=True) if use_graph_context else valid_frame
    valid_history_index = build_temporal_history_index(valid_graph_frame, config, seed=seed + 1) if use_graph_context else None
    valid_row_id_offset = len(train_frame) if use_graph_context else 0
    if use_graph_context:
        write_temporal_graph_metadata(train_frame, config, output_dir / "graph", source="real_training_rows", seed=seed)
    loss_weights = dict(config.raw.get("loss_weights", {}))
    length_class_weights = compute_length_class_weights(train_frame, config, categorical_vocabs, tokenizer)
    length_tensors = {
        column: tensor.to(device)
        for column, tensor in (length_class_weights or {}).get("tensor", {}).items()
    }
    if length_class_weights is not None:
        for column, payload in length_class_weights["json"].items():
            save_json(payload, metadata_dir / f"{column}_weights.json")
    write_lstm_model_metadata(config, metadata_dir)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(graph_encoder.parameters()) if graph_encoder is not None else []),
        lr=float(training.get("lr", training.get("learning_rate", 5e-4))),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    use_amp = bool(training.get("mixed_precision", True)) and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_norm = float(training.get("gradient_clip_norm", 1.0))
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    best_valid = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    epochs = int(training.get("epochs", 50))
    patience = int(training.get("early_stopping_patience", 0) or 0)
    min_delta = float(training.get("early_stopping_min_delta", 0.0) or 0.0)
    without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_lstm_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            loss_weights,
            tokenizer,
            length_tensors,
            graph_encoder=graph_encoder,
            graph_history_index=train_history_index,
            graph_deterministic=False,
            config=config,
            clip_norm=clip_norm,
        )
        valid_metrics = run_lstm_epoch(
            model,
            valid_loader,
            None,
            scaler,
            device,
            use_amp,
            loss_weights,
            tokenizer,
            length_tensors,
            graph_encoder=graph_encoder,
            graph_history_index=valid_history_index,
            graph_deterministic=True,
            graph_row_id_offset=valid_row_id_offset,
            config=config,
            clip_norm=clip_norm,
        )
        current = float(valid_metrics["total_loss"])
        improved = current < (best_valid - min_delta)
        row = {
            "epoch": int(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "best_valid_total_loss": min(best_valid, current),
            "epochs_without_improvement": 0 if improved else without_improvement + 1,
        }
        append_jsonl(log_path, row)
        print(json.dumps(row, sort_keys=True))
        save_lstm_checkpoint(last_path, model, config, categorical_vocabs, tokenizer, epoch, valid_metrics, graph_encoder=graph_encoder)
        if improved:
            best_valid = current
            without_improvement = 0
            save_lstm_checkpoint(best_path, model, config, categorical_vocabs, tokenizer, epoch, valid_metrics, graph_encoder=graph_encoder)
        else:
            without_improvement += 1
        if patience > 0 and without_improvement >= patience:
            print(f"Early stopping at epoch={epoch}; best_valid_total_loss={best_valid:.6g}")
            break
    save_json(
        {
            "train_batch_size_requested": int(requested_batch_size if requested_batch_size is not None else batch_size),
            "train_batch_size_used": int(batch_size),
            "auto_reduce_batch_size": bool(auto_reduce_batch_size),
            "min_batch_size": int(training.get("min_batch_size", batch_size)),
            "epochs_completed": int(epoch),
            "best_valid_total_loss": float(best_valid),
            "device": str(device),
            "mixed_precision_used": bool(use_amp),
        },
        metadata_dir / "training_runtime.json",
    )
    print(f"Wrote best checkpoint to {best_path}")
    return best_path


def candidate_train_batch_sizes(initial_batch_size: int, min_batch_size: int) -> list[int]:
    initial = max(1, int(initial_batch_size))
    floor = min(initial, max(1, int(min_batch_size)))
    sizes: list[int] = []
    current = initial
    while current >= floor:
        sizes.append(current)
        if current == floor:
            break
        current = max(floor, current // 2)
    return sizes


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(error)


def clear_cuda_after_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def run_lstm_epoch(
    model: JointLSTMRelationalAttributeGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_amp: bool,
    loss_weights: dict[str, float],
    tokenizer: SimpleTextTokenizer,
    length_class_weights: dict[str, torch.Tensor],
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
    graph_history_index: Any | None = None,
    graph_deterministic: bool = True,
    graph_row_id_offset: int = 0,
    config: ConditionalTABDLMConfig | None = None,
    clip_norm: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if graph_encoder is not None:
        graph_encoder.train(training)
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    corrects: dict[str, int] = {}
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, leave=False, desc="train_lstm" if training else "valid_lstm")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if training:
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
            logits = model(
                batch["foreign_key_ids"],
                batch["datetime_values"],
                batch["categorical_ids"],
                batch["text_ids"],
                graph_context=graph_context,
            )
            loss, component = lstm_joint_loss(logits, batch, model.schema, loss_weights, tokenizer, length_class_weights)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(graph_encoder.parameters()) if graph_encoder is not None else []), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        for key, stats in component.items():
            totals[key] = totals.get(key, 0.0) + float(stats["loss_sum"])
            counts[key] = counts.get(key, 0.0) + float(stats["count"])
            corrects[key] = corrects.get(key, 0) + int(stats.get("correct", 0))
    metrics: dict[str, float] = {}
    total = 0.0
    for key in sorted(totals):
        count = max(float(counts.get(key, 0.0)), 1.0)
        value = float(totals[key] / count)
        metrics[f"{key}_loss"] = value
        total += float(loss_weights.get(key, 1.0)) * value
        if key in model.schema.model_categorical_targets:
            metrics[f"{key}_accuracy"] = float(corrects.get(key, 0) / count)
    metrics["total_loss"] = float(total)
    return metrics


def lstm_joint_loss(
    logits: dict[str, Any],
    batch: dict[str, Any],
    schema: ConditionalTABDLMSchema,
    loss_weights: dict[str, float],
    tokenizer: SimpleTextTokenizer,
    length_class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    losses: list[torch.Tensor] = []
    component: dict[str, dict[str, float | int]] = {}
    cat_labels = batch["categorical_ids"]
    for idx, column in enumerate(schema.model_categorical_targets):
        labels = cat_labels[:, idx]
        class_weights = (length_class_weights or {}).get(column)
        loss_sum = F.cross_entropy(logits["categorical"][column], labels, reduction="sum", weight=class_weights)
        count = int(labels.numel())
        losses.append(float(loss_weights.get(metric_name_for_lstm(column), loss_weights.get(column, 1.0))) * (loss_sum / max(count, 1)))
        pred = logits["categorical"][column].argmax(dim=-1)
        correct = int((pred == labels).sum().detach().cpu())
        component[metric_name_for_lstm(column)] = {"loss_sum": float(loss_sum.detach().cpu()), "count": count, "correct": correct}
    for column in schema.text_targets:
        labels = batch["text_ids"][column][:, 1:].contiguous()
        mask = labels != tokenizer.pad_id
        count = int(mask.sum().detach().cpu())
        if count == 0:
            continue
        ce = F.cross_entropy(
            logits["text"][column].reshape(-1, logits["text"][column].shape[-1]),
            labels.reshape(-1),
            ignore_index=tokenizer.pad_id,
            reduction="sum",
        )
        key = "summary_text" if column == "summary" else column
        losses.append(float(loss_weights.get(key, loss_weights.get(column, 1.0))) * (ce / max(count, 1)))
        component[key] = {"loss_sum": float(ce.detach().cpu()), "count": count}
    if not losses:
        zero = batch["foreign_key_ids"].float().sum() * 0.0
        return zero, {}
    return torch.stack(losses).sum(), component


def metric_name_for_lstm(column: str) -> str:
    if column.endswith("_length_bucket"):
        return column[: -len("_bucket")]
    return column


def save_lstm_checkpoint(
    path: str | Path,
    model: JointLSTMRelationalAttributeGenerator,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    epoch: int,
    valid_metrics: dict[str, float],
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
            "tokenizer_metadata": tokenizer.to_dict(),
            "epoch": int(epoch),
            "valid_metrics": valid_metrics,
            "graph_encoder_state_dict": graph_encoder.state_dict() if graph_encoder is not None else None,
            "graph_encoder_config": graph_encoder.to_config() if graph_encoder is not None else None,
            "graph_conditioning_metadata": graph_metadata(config.raw, real_graph_used_at_sampling=False),
        },
        path,
    )


def load_lstm_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    include_graph: bool = False,
) -> tuple[JointLSTMRelationalAttributeGenerator, ConditionalTABDLMConfig, dict[str, CategoryVocab], SimpleTextTokenizer, TemporalStructureOnlyGraphEncoder | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint["raw_config"]
    schema = ConditionalTABDLMSchema.from_config_dict(raw_config)
    config = ConditionalTABDLMConfig(raw=raw_config, schema=schema, config_path=None)
    vocabs = {column: CategoryVocab.from_dict(data) for column, data in checkpoint["categorical_vocabs"].items()}
    tokenizer = SimpleTextTokenizer.from_dict(checkpoint["tokenizer_metadata"])
    model = build_lstm_model(config, vocabs, tokenizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    graph_encoder = None
    if include_graph and graph_conditioning_enabled(raw_config):
        graph_encoder = build_graph_encoder(config, vocabs, tokenizer).to(device)
        state = checkpoint.get("graph_encoder_state_dict")
        if state is not None:
            graph_encoder.load_state_dict(state)
        graph_encoder.eval()
    return model, config, vocabs, tokenizer, graph_encoder


@torch.no_grad()
def sample_lstm_from_config(
    config: ConditionalTABDLMConfig,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_rows: int | str | None = None,
    batch_size: int | str | None = None,
    device: str | None = None,
    seed: int | None = None,
    synthetic_spine_path: str | Path | None = None,
) -> Path:
    sampling = config.raw.get("sampling", {})
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_dir / "best.pt"
    output_path = Path(output_path) if output_path else config.output_dir / "synthetic_review_attrs.csv"
    seed = int(seed if seed is not None else sampling.get("seed", 42))
    device = resolve_device(device or str(sampling.get("device", "auto")))
    set_seed(seed)
    model, ckpt_config, vocabs, tokenizer, graph_encoder = load_lstm_checkpoint(checkpoint_path, device=device, include_graph=True)
    spine_path = Path(synthetic_spine_path) if synthetic_spine_path else config.synthetic_spine_path
    spine = pd.read_csv(spine_path)
    if num_rows not in (None, "all"):
        spine = spine.head(int(num_rows)).copy()
    spine = spine.reset_index(drop=True)
    batch_size_used = resolve_sampling_batch_size(batch_size if batch_size is not None else sampling.get("batch_size", "auto"), device)
    graph_history_index = None
    if graph_encoder is not None:
        graph_encoder.eval()
        graph_history_index = build_temporal_history_index(spine, ckpt_config, seed=seed)
        write_temporal_graph_metadata(spine, ckpt_config, output_path.parent / "graph", source="synthetic_spine", seed=seed, real_graph_used_at_sampling=False)
    id_cfg = ckpt_config.raw.get("id_encoding", {})
    num_hash_buckets = int(id_cfg.get("num_buckets", 262144))
    temperature = float(sampling.get("temperature", 0.9))
    top_p = float(sampling.get("top_p", 0.95))
    min_tokens = {
        "summary": int(sampling.get("min_summary_tokens", 1)),
        "review_text": int(sampling.get("min_review_text_tokens", 1)),
    }
    repetition = {
        "summary": float(sampling.get("summary_repetition_penalty", 1.10)),
        "review_text": float(sampling.get("review_text_repetition_penalty", 1.05)),
    }
    attrs: dict[str, list[Any]] = {column: [] for column in ckpt_config.schema.categorical_targets + ckpt_config.schema.text_targets}
    lengths: dict[str, list[int]] = {column: [] for column in ckpt_config.schema.text_targets}
    start_time = time.perf_counter()
    iterator = range(0, len(spine), batch_size_used)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(spine) + batch_size_used - 1) // batch_size_used, desc="sample_lstm")
    use_amp = bool(sampling.get("mixed_precision", True)) and str(device).startswith("cuda")
    for start in iterator:
        frame = spine.iloc[start : start + batch_size_used]
        foreign_key_ids, datetime_values = encode_conditions(frame, ckpt_config.schema, num_hash_buckets, device)
        graph_context = None
        if graph_encoder is not None:
            if graph_history_index is None:
                raise ValueError("graph_history_index is required when graph_encoder is enabled")
            row_indices = list(range(start, start + len(frame)))
            graph_context = graph_encoder(graph_history_index.build_batch(row_indices, device=device, deterministic=True))
        with torch.cuda.amp.autocast(enabled=use_amp):
            generated = model.generate(
                foreign_key_ids,
                datetime_values,
                vocabs,
                tokenizer,
                graph_context=graph_context,
                temperature=temperature,
                top_p=top_p,
                min_tokens=min_tokens,
                repetition_penalty=repetition,
            )
        for column in ckpt_config.schema.categorical_targets:
            attrs[column].extend(generated["categorical"][column])
        for column in ckpt_config.schema.text_targets:
            attrs[column].extend(generated["text"][column])
            lengths[column].extend(generated["text_lengths"][column])
    total_seconds = float(time.perf_counter() - start_time)
    output = spine.loc[:, list(ckpt_config.schema.condition_columns)].copy()
    for column in ckpt_config.schema.categorical_targets:
        output[column] = attrs[column]
    for column in ckpt_config.schema.text_targets:
        output[column] = attrs[column]
    output = validate_output_categoricals(
        output,
        {column: vocabs[column] for column in ckpt_config.schema.categorical_targets if column in vocabs},
        repair_invalid=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metadata_dir = ensure_dir(output_path.parent / "metadata")
    runtime = runtime_metadata(total_seconds, len(output), batch_size_used, device, use_amp, lengths)
    save_json(runtime, metadata_dir / "runtime_sampling.json")
    sample_metadata = {
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(spine_path),
        "output_path": str(output_path),
        "num_rows": int(len(output)),
        "batch_size": int(batch_size_used),
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in ckpt_config.schema.text_targets,
        "review_text_separate_stage": False,
        "uses_diffusion": False,
        "uses_transformer_backbone": False,
        "text_decoder_type": ckpt_config.raw.get("text_decoder", {}).get("type", "lstm"),
        **graph_metadata(ckpt_config.raw, real_graph_used_at_sampling=False),
        "synthetic_graph_history_source": "synthetic_spine",
        "graph_uses_clean_target_attributes": False,
        "graph_uses_clean_future_attributes": False,
        "valid_categorical_values": {
            column: valid_category_values(column, vocabs[column])
            for column in ckpt_config.schema.categorical_targets
            if column in vocabs
        },
    }
    save_json(sample_metadata, output_path.parent / "sample_metadata.json")
    print(f"Wrote {output_path}")
    print(json.dumps(runtime, sort_keys=True))
    return output_path


def write_lstm_model_metadata(config: ConditionalTABDLMConfig, metadata_dir: str | Path) -> None:
    graph_flags = graph_metadata(config.raw, real_graph_used_at_sampling=False)
    review_cfg = config.raw.get("review_text", {})
    auto_review = config.raw.get("_auto_text_length_metadata", {}).get("review_text", {})
    loss_weights = dict(config.raw.get("loss_weights", {}))
    metadata: dict[str, Any] = {
        "experiment_name": config.raw.get("experiment_name", Path(config.output_dir).name),
        "model_family": "joint_lstm_generator",
        "base_graph_model": "v2_structure_only_temporal_graph",
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in config.schema.text_targets,
        "review_text_separate_stage": False,
        "uses_diffusion": False,
        "uses_transformer_backbone": False,
        "text_decoder_type": config.raw.get("text_decoder", {}).get("type", "lstm"),
        "condition_columns": list(config.schema.condition_columns),
        "target_columns": {
            "categorical": list(config.schema.categorical_targets),
            "text": list(config.schema.text_targets),
        },
        "auxiliary_targets": list(config.schema.auxiliary_categorical_targets),
        "loss_weighting": {
            "mode": "fixed_manual",
            "mgda_enabled": False,
            "rating": float(loss_weights.get("rating", 1.0)),
            "verified": float(loss_weights.get("verified", 1.0)),
            "summary_length": float(loss_weights.get("summary_length", 1.0)),
            "review_text_length": float(loss_weights.get("review_text_length", 1.0)),
            "summary_text": float(loss_weights.get("summary_text", 1.0)),
            "review_text": float(loss_weights.get("review_text", 1.0)),
        },
        "graph_conditioning_mode": "structure_only_temporal",
        "temporal_filter_enabled": True,
        "temporal_filter_mode": "past_only",
        "graph_uses_future_events": False,
        "graph_uses_target_attributes": False,
        "graph_uses_clean_target_attributes": False,
        "graph_uses_clean_future_attributes": False,
        "real_graph_used_at_sampling": False,
        "synthetic_graph_history_source": "synthetic_spine",
        "review_text_max_tokens": int(config.schema.text_max_lengths.get("review_text", 0)),
        "review_text_max_tokens_strategy": review_cfg.get("max_tokens_strategy"),
        "review_text_length_cap_source": review_cfg.get("length_cap_source"),
        "review_text_coverage_rate_train": review_cfg.get("coverage_rate_train"),
        "review_text_truncation_rate_train": review_cfg.get("truncation_rate_train"),
        "review_text_length_stats_real": review_cfg.get("length_stats_real"),
        "review_text_length_bucket_edges": {name: list(bounds) for name, bounds in config.schema.review_text_length_buckets.items()},
    }
    metadata.update(graph_flags)
    metadata.update({"graph_uses_clean_target_attributes": False, "graph_uses_clean_future_attributes": False})
    metadata.update(auto_review)
    save_json(metadata, Path(metadata_dir) / "model_metadata.json")
    if auto_review:
        save_json(auto_review, Path(metadata_dir) / "review_text_length_stats.json")


def runtime_metadata(
    total_seconds: float,
    rows: int,
    batch_size: int,
    device: str,
    mixed_precision: bool,
    lengths: dict[str, list[int]],
) -> dict[str, Any]:
    rows_per_second = float(rows / max(total_seconds, 1e-9))
    review_lengths = lengths.get("review_text", [])
    summary_lengths = lengths.get("summary", [])
    projected_seconds = float(10_000_000 / max(rows_per_second, 1e-9))
    return {
        "total_sampling_seconds": float(total_seconds),
        "rows_generated": int(rows),
        "rows_per_second": rows_per_second,
        "seconds_per_1000_rows": float(1000.0 / max(rows_per_second, 1e-9)),
        "projected_seconds_for_10m_rows": projected_seconds,
        "projected_hours_for_10m_rows": float(projected_seconds / 3600.0),
        "average_generated_summary_tokens": float(np.mean(summary_lengths)) if summary_lengths else None,
        "average_generated_review_text_tokens": float(np.mean(review_lengths)) if review_lengths else None,
        "p95_generated_review_text_tokens": float(np.quantile(review_lengths, 0.95)) if review_lengths else None,
        "max_generated_review_text_tokens": int(max(review_lengths)) if review_lengths else None,
        "batch_size_used": int(batch_size),
        "device": str(device),
        "mixed_precision_used": bool(mixed_precision),
    }


def length_bounds_for_generation(
    schema: ConditionalTABDLMSchema,
    text_column: str,
    bucket_names: list[Any],
    max_content: int,
    min_content_tokens: int,
) -> tuple[list[int], list[int]]:
    bucket_column = length_bucket_column_for_text(schema, text_column)
    buckets = schema.buckets_for_length_bucket(bucket_column) if bucket_column else {}
    lows: list[int] = []
    highs: list[int] = []
    for name in bucket_names:
        if name is not None and str(name) in buckets:
            low, high = buckets[str(name)]
        else:
            low, high = 0, max_content
        lows.append(max(int(min_content_tokens), min(int(low), int(max_content))))
        highs.append(max(lows[-1], min(int(high), int(max_content))))
    return lows, highs


def length_bucket_column_for_text(schema: ConditionalTABDLMSchema, text_column: str) -> str | None:
    for column in schema.length_bucket_targets:
        try:
            if schema.text_column_for_length_bucket(column) == text_column:
                return column
        except (KeyError, IndexError):
            continue
    return None


def sample_categorical_column(
    logits: torch.Tensor,
    column: str,
    vocab: CategoryVocab,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    constrained = mask_invalid_category_logits(logits, column, vocab)
    return sample_from_logits(constrained, temperature=temperature, top_p=top_p)


def sample_text_step(
    logits: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    *,
    step: int,
    lows: list[int],
    highs: list[int],
    previous_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    filtered = logits.clone()
    forbidden_always = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.mask_id, tokenizer.unk_id]
    filtered[:, forbidden_always] = -float("inf")
    for row_idx, (low, high) in enumerate(zip(lows, highs)):
        content_so_far = int(step - 1)
        if content_so_far < int(low):
            filtered[row_idx, tokenizer.eos_id] = -float("inf")
        if content_so_far >= int(high):
            filtered[row_idx, :] = -float("inf")
            filtered[row_idx, tokenizer.eos_id] = 0.0
        if repetition_penalty > 1.0:
            previous = previous_ids[row_idx].detach().cpu().tolist()
            for token_id in set(int(idx) for idx in previous if int(idx) not in tokenizer.special_ids):
                filtered[row_idx, token_id] = filtered[row_idx, token_id] / float(repetition_penalty)
    return sample_from_logits(filtered, temperature=temperature, top_p=top_p)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits / max(float(temperature), 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(row_sums > 0, probs / row_sums.clamp_min(1e-12), torch.full_like(probs, 1.0 / probs.shape[-1]))
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def select_state(state: Any, index: torch.Tensor, decoder_type: str) -> Any:
    if str(decoder_type) == "gru":
        return state.index_select(1, index)
    hidden, cell = state
    return hidden.index_select(1, index), cell.index_select(1, index)


def scatter_state(state: Any, new_state: Any, index: torch.Tensor, decoder_type: str) -> Any:
    if str(decoder_type) == "gru":
        if new_state.dtype != state.dtype:
            new_state = new_state.to(dtype=state.dtype)
        state[:, index, :] = new_state
        return state
    hidden, cell = state
    new_hidden, new_cell = new_state
    if new_hidden.dtype != hidden.dtype:
        new_hidden = new_hidden.to(dtype=hidden.dtype)
    if new_cell.dtype != cell.dtype:
        new_cell = new_cell.to(dtype=cell.dtype)
    hidden[:, index, :] = new_hidden
    cell[:, index, :] = new_cell
    return hidden, cell


def encode_conditions(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    num_hash_buckets: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreign_keys = [
        [stable_hash_bucket(column, row[column], num_hash_buckets) for column in schema.foreign_key_columns]
        for _, row in frame.iterrows()
    ]
    datetimes = [
        [pd.Timestamp(row[column]).timestamp() for column in schema.datetime_columns]
        for _, row in frame.iterrows()
    ]
    return (
        torch.tensor(foreign_keys, dtype=torch.long, device=device),
        torch.tensor(datetimes, dtype=torch.float32, device=device),
    )


def resolve_sampling_batch_size(value: Any, device: str) -> int:
    if value in (None, "auto"):
        return 256 if str(device).startswith("cuda") else 64
    return int(value)


def move_batch_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    return value


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        json.dump(jsonable(row), handle, sort_keys=True)
        handle.write("\n")
