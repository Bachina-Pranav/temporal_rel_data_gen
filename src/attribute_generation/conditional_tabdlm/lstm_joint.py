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
    load_numerical_metadata,
    load_prepared_tables,
    load_text_tokenizer,
    prepare_rel_amazon_data,
)
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_encoder import TemporalStructureOnlyGraphEncoder
from .graph_schema import assert_valid_graph_conditioning, graph_conditioning_enabled, graph_metadata
from .model import DateTimeEncoder
from .numerical import gaussian_nll_from_params, inverse_transform_numerical, sample_gaussian_params
from .neighbor_cache import CachedTemporalHistoryIndex
from .pretokenized import PretokenizedLSTMDataset, load_pretokenized_bundle
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .temporal_stratified_sampler import TemporalStratifiedSampler
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
        review_text_conditioned_on_summary: bool = False,
        summary_condition_type: str = "final_hidden_plus_mean_pool",
        summary_condition_dim: int = 384,
        summary_condition_dropout: float = 0.1,
        decoder_input_token_dropout: dict[str, float] | None = None,
        decoder_input_token_dropout_replacement: str = "UNK",
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
        self.review_text_conditioned_on_summary = bool(review_text_conditioned_on_summary)
        self.summary_condition_type = str(summary_condition_type)
        self.summary_condition_dim = int(summary_condition_dim)
        self.summary_condition_dropout = float(summary_condition_dropout)
        self.decoder_input_token_dropout = dict(decoder_input_token_dropout or {})
        self.decoder_input_token_dropout_replacement = str(decoder_input_token_dropout_replacement)
        self.last_token_dropout_rates: dict[str, float] = {}

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
        self.numerical_heads = nn.ModuleDict(
            {
                column: nn.Linear(self.row_hidden_dim, 2)
                for column in schema.numerical_targets
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
        self.decoder_context_dim = int(decoder_context_dim)
        self.summary_condition_projector = (
            nn.Sequential(
                nn.Linear(self.text_hidden_dim * 2, self.summary_condition_dim),
                nn.GELU(),
                nn.Dropout(self.summary_condition_dropout),
                nn.LayerNorm(self.summary_condition_dim),
            )
            if self.review_text_conditioned_on_summary
            else None
        )
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
            initializer_context_dim = decoder_context_dim
            if column == "review_text" and self.review_text_conditioned_on_summary:
                initializer_context_dim += self.summary_condition_dim
            self.text_initializers[column] = nn.Sequential(
                nn.Linear(initializer_context_dim, self.text_hidden_dim * self.text_num_layers * state_multiplier),
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

    def numerical_params(self, row_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {column: head(row_latent) for column, head in self.numerical_heads.items()}

    def categorical_context(self, row_latent: torch.Tensor, categorical_ids: torch.Tensor) -> torch.Tensor:
        pieces = [row_latent]
        for idx, column in enumerate(self.schema.model_categorical_targets):
            pieces.append(self.categorical_context_embeddings[column](categorical_ids[:, idx]))
        return torch.cat(pieces, dim=1)

    def decoder_context(self, column: str, context: torch.Tensor, summary_repr: torch.Tensor | None = None) -> torch.Tensor:
        if column == "review_text" and self.review_text_conditioned_on_summary:
            if summary_repr is None:
                summary_repr = torch.zeros(
                    context.shape[0],
                    self.summary_condition_dim,
                    dtype=context.dtype,
                    device=context.device,
                )
            return torch.cat([context, summary_repr.to(dtype=context.dtype)], dim=1)
        return context

    def initial_state(
        self,
        column: str,
        context: torch.Tensor,
        summary_repr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        projected = self.text_initializers[column](self.decoder_context(column, context, summary_repr=summary_repr))
        batch = int(context.shape[0])
        if self.decoder_type == "gru":
            return projected.view(batch, self.text_num_layers, self.text_hidden_dim).transpose(0, 1).contiguous()
        projected = projected.view(batch, 2, self.text_num_layers, self.text_hidden_dim)
        hidden = projected[:, 0].transpose(0, 1).contiguous()
        cell = projected[:, 1].transpose(0, 1).contiguous()
        return hidden, cell

    def apply_decoder_input_token_dropout(self, column: str, teacher: torch.Tensor) -> torch.Tensor:
        rate = float(self.decoder_input_token_dropout.get(column, 0.0) or 0.0)
        self.last_token_dropout_rates[column] = 0.0
        if not self.training or rate <= 0.0 or teacher.numel() == 0:
            return teacher
        replacement = self.text_pad_id
        if self.decoder_input_token_dropout_replacement.upper() == "MASK":
            replacement = 2
        elif self.decoder_input_token_dropout_replacement.upper() == "UNK":
            replacement = 3
        special_ids = {self.text_pad_id, 1, 2, 3, 4}
        eligible = torch.ones_like(teacher, dtype=torch.bool)
        for token_id in special_ids:
            eligible &= teacher != int(token_id)
        mask = (torch.rand_like(teacher.float()) < rate) & eligible
        if bool(mask.any()):
            teacher = teacher.clone()
            teacher[mask] = int(replacement)
        denom = max(int(eligible.sum().detach().cpu()), 1)
        self.last_token_dropout_rates[column] = float(mask.sum().detach().cpu()) / float(denom)
        return teacher

    def summary_representation_from_output(
        self,
        summary_output: torch.Tensor,
        summary_state: Any,
        summary_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.summary_condition_projector is None:
            raise RuntimeError("summary_condition_projector is unavailable when review_text_conditioned_on_summary is false")
        if str(self.decoder_type) == "gru":
            final_hidden = summary_state[-1]
        else:
            final_hidden = summary_state[0][-1]
        mask = (summary_input_ids != self.text_pad_id).to(dtype=summary_output.dtype).unsqueeze(-1)
        pooled = (summary_output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        combined = torch.cat([final_hidden, pooled], dim=1)
        return self.summary_condition_projector(combined)

    def summary_representation_from_ids(self, context: torch.Tensor, summary_ids: torch.Tensor) -> torch.Tensor:
        teacher = summary_ids[:, :-1].contiguous()
        embedded = self.text_embedding(teacher)
        output, state = self.text_decoders["summary"](embedded, self.initial_state("summary", context))
        return self.summary_representation_from_output(output, state, teacher)

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
        numerical = self.numerical_params(row)
        context = self.categorical_context(row, categorical_ids)
        text_logits: dict[str, torch.Tensor] = {}
        summary_repr = None
        for column in self.schema.text_targets:
            teacher = text_ids[column][:, :-1].contiguous()
            teacher_input = self.apply_decoder_input_token_dropout(column, teacher)
            embedded = self.text_embedding(teacher_input)
            output, state = self.text_decoders[column](embedded, self.initial_state(column, context, summary_repr=summary_repr))
            text_logits[column] = self.text_heads[column](output)
            if column == "summary" and self.review_text_conditioned_on_summary:
                summary_repr = self.summary_representation_from_output(output, state, teacher)
        return {"categorical": cat_logits, "numerical": numerical, "text": text_logits, "row_latent": row}

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
        numerical_params = self.numerical_params(row)
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
        summary_repr = None
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
                summary_repr=summary_repr,
            )
            text_ids[column] = ids
            decoded_text[column] = [tokenizer.decode(row_ids) for row_ids in ids.detach().cpu().tolist()]
            lengths[column] = [tokenizer.content_length(row_ids) for row_ids in ids.detach().cpu().tolist()]
            if column == "summary" and self.review_text_conditioned_on_summary:
                summary_repr = self.summary_representation_from_ids(context, ids)
        return {
            "categorical_ids": categorical_ids,
            "categorical": decoded_cats,
            "numerical_params": numerical_params,
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
        summary_repr: torch.Tensor | None = None,
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
        state = self.initial_state(column, context, summary_repr=summary_repr)
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
            "numerical_targets": list(self.schema.numerical_targets),
            "review_text_conditioned_on_summary": self.review_text_conditioned_on_summary,
            "summary_condition_type": self.summary_condition_type,
            "summary_condition_dim": self.summary_condition_dim,
        }


def make_lstm_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if "numerical_values" in samples[0]:
        numerical_values = torch.stack([sample["numerical_values"] for sample in samples], dim=0)
    else:
        numerical_values = torch.empty((len(samples), 0), dtype=torch.float32)
    return {
        "foreign_key_ids": torch.stack([sample["foreign_key_ids"] for sample in samples], dim=0),
        "datetime_values": torch.stack([sample["datetime_values"] for sample in samples], dim=0),
        "categorical_ids": torch.stack([sample["categorical_ids"] for sample in samples], dim=0),
        "numerical_values": numerical_values,
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
    review_decoder_cfg = config.raw.get("review_text_decoder", {})
    token_dropout_cfg = config.raw.get("training_regularization", {}).get("decoder_input_token_dropout", {})
    token_dropout = {}
    if bool(token_dropout_cfg.get("enabled", False)):
        token_dropout = {
            "summary": float(token_dropout_cfg.get("summary", 0.0) or 0.0),
            "review_text": float(token_dropout_cfg.get("review_text", 0.0) or 0.0),
        }
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
        review_text_conditioned_on_summary=bool(review_decoder_cfg.get("condition_on_summary", False)),
        summary_condition_type=str(review_decoder_cfg.get("summary_condition_type", "final_hidden_plus_mean_pool")),
        summary_condition_dim=int(review_decoder_cfg.get("summary_condition_dim", decoder_cfg.get("hidden_dim", 384))),
        summary_condition_dropout=float(review_decoder_cfg.get("summary_condition_dropout", 0.1)),
        decoder_input_token_dropout=token_dropout,
        decoder_input_token_dropout_replacement=str(token_dropout_cfg.get("replacement", "UNK")),
    )


def train_lstm_from_config(config: ConditionalTABDLMConfig, device: str | None = None) -> Path:
    training = config.raw.get("training", {})
    requested_batch_size = int(training.get("physical_batch_size", training.get("batch_size", 256)))
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
    if not bool_value(training.get("epoch_mode", True)):
        return _train_lstm_fixed_step_from_config_once(
            config,
            device=device,
            batch_size_override=batch_size_override,
            requested_batch_size=requested_batch_size,
            auto_reduce_batch_size=auto_reduce_batch_size,
        )
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
    numerical_metadata = load_numerical_metadata(config)
    num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
    train_dataset = ConditionalTABDLMDataset(train_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets, numerical_metadata=numerical_metadata)
    valid_dataset = ConditionalTABDLMDataset(valid_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets, numerical_metadata=numerical_metadata)
    batch_size = int(batch_size_override if batch_size_override is not None else training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_lstm_collate_fn,
        **dataloader_performance_kwargs(training, num_workers, train=True),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_lstm_collate_fn,
        **dataloader_performance_kwargs(training, num_workers, train=False),
    )

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
    metrics_log_path = ensure_dir(output_dir / "logs") / "train_metrics.jsonl"
    if log_path.exists():
        log_path.unlink()
    if metrics_log_path.exists():
        metrics_log_path.unlink()
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
        append_jsonl(metrics_log_path, row)
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


def _train_lstm_fixed_step_from_config_once(
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

    pretokenized_dir = training.get("pretokenized_dir") or config.raw.get("paths", {}).get("pretokenized_dir")
    if pretokenized_dir:
        bundle = load_pretokenized_bundle(pretokenized_dir, config.schema)
        train_dataset = PretokenizedLSTMDataset(bundle, "train")
        valid_dataset = PretokenizedLSTMDataset(bundle, "valid")
        categorical_vocabs = bundle.categorical_vocabs
        tokenizer = bundle.tokenizer
        train_frame = None
        valid_frame = None
        train_rows_available = int(len(train_dataset))
    else:
        train_frame, valid_frame, _ = load_prepared_tables(config)
        train_frame = maybe_limit_rows(train_frame, training.get("max_rows"), seed)
        valid_frame = maybe_limit_rows(valid_frame, validation_row_cap(training.get("max_rows"), len(valid_frame)), seed + 1)
        categorical_vocabs = load_category_vocabs(config)
        tokenizer = load_text_tokenizer(config)
        numerical_metadata = load_numerical_metadata(config)
        num_hash_buckets = int(config.raw.get("id_encoding", {}).get("num_buckets", 262144))
        train_dataset = ConditionalTABDLMDataset(train_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets, numerical_metadata=numerical_metadata)
        valid_dataset = ConditionalTABDLMDataset(valid_frame, config.schema, categorical_vocabs, tokenizer, num_hash_buckets, numerical_metadata=numerical_metadata)
        train_rows_available = int(len(train_dataset))

    use_graph_context = graph_conditioning_enabled(config.raw)
    if use_graph_context:
        assert_valid_graph_conditioning(config.raw)

    batch_size = int(batch_size_override if batch_size_override is not None else training.get("physical_batch_size", training.get("batch_size", 64)))
    grad_accum = gradient_accumulation_steps_for(training, batch_size)
    effective_batch_size = int(batch_size * grad_accum)
    max_steps = int(training.get("max_steps", 50000))
    steps_per_eval = int(training.get("steps_per_eval", 2000))
    steps_per_checkpoint = int(training.get("steps_per_checkpoint", 5000))
    num_workers = int(training.get("num_workers", 0))
    max_microbatches = max(1, max_steps * grad_accum)
    train_loader = build_fixed_step_lstm_loader(
        train_dataset,
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        num_microbatches=max_microbatches,
        seed=seed,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_lstm_collate_fn,
        **dataloader_performance_kwargs(training, num_workers, train=False),
    )

    device = resolve_device(device or str(training.get("device", "auto")))
    model = build_lstm_model(config, categorical_vocabs, tokenizer).to(device)
    graph_encoder = build_graph_encoder(config, categorical_vocabs, tokenizer).to(device) if use_graph_context else None
    neighbor_cache_dir = training.get("neighbor_cache_dir") or config.raw.get("paths", {}).get("neighbor_cache_dir")
    if use_graph_context and neighbor_cache_dir:
        train_history_index = CachedTemporalHistoryIndex(neighbor_cache_dir)
        valid_history_index = train_history_index
        valid_row_id_offset = 0
    elif use_graph_context:
        if train_frame is None or valid_frame is None:
            raise ValueError("Fixed-step pretokenized graph training requires --neighbor-cache-dir")
        train_history_index = build_temporal_history_index(train_frame, config, seed=seed)
        valid_graph_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
        valid_history_index = build_temporal_history_index(valid_graph_frame, config, seed=seed + 1)
        valid_row_id_offset = len(train_frame)
    else:
        train_history_index = None
        valid_history_index = None
        valid_row_id_offset = 0
    if use_graph_context:
        if train_frame is not None:
            write_temporal_graph_metadata(train_frame, config, output_dir / "graph", source="real_training_rows", seed=seed)
        graph_flags = graph_metadata(config.raw, real_graph_used_at_sampling=False)
        save_json(graph_flags, output_dir / "graph_conditioning_flags.json")
        save_json(graph_flags, metadata_dir / "graph_conditioning.json")

    loss_weights = dict(config.raw.get("loss_weights", {}))
    if train_frame is not None:
        length_class_weights = compute_length_class_weights(train_frame, config, categorical_vocabs, tokenizer)
    else:
        length_class_weights = compute_length_class_weights_from_pretokenized(train_dataset, config, categorical_vocabs)
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
    metrics_log_path = ensure_dir(output_dir / "logs") / "train_metrics.jsonl"
    for path in [log_path, metrics_log_path]:
        if path.exists():
            path.unlink()
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    start = time.perf_counter()
    best_valid = float("inf")
    best_step = 0
    last_valid_metrics: dict[str, float] = {"total_loss": float("inf")}
    sampler = getattr(train_loader, "sampler", None)
    train_metrics = run_lstm_fixed_steps(
        model,
        train_loader,
        valid_loader,
        optimizer,
        scaler,
        device,
        use_amp,
        loss_weights,
        tokenizer,
        length_tensors,
        max_steps=max_steps,
        gradient_accumulation_steps=grad_accum,
        steps_per_eval=steps_per_eval,
        steps_per_checkpoint=steps_per_checkpoint,
        checkpoint_dir=checkpoint_dir,
        log_path=log_path,
        metrics_log_path=metrics_log_path,
        categorical_vocabs=categorical_vocabs,
        graph_encoder=graph_encoder,
        train_history_index=train_history_index,
        valid_history_index=valid_history_index,
        valid_row_id_offset=valid_row_id_offset,
        config=config,
        clip_norm=clip_norm,
        validation_max_batches=int(training.get("validation_max_batches", 100)),
    )
    elapsed = time.perf_counter() - start
    best_valid = float(train_metrics.get("best_valid_total_loss", best_valid))
    best_step = int(train_metrics.get("best_step", best_step))
    last_valid_metrics = dict(train_metrics.get("last_valid_metrics", last_valid_metrics))
    if not best_path.exists():
        save_lstm_checkpoint(best_path, model, config, categorical_vocabs, tokenizer, max_steps, last_valid_metrics, graph_encoder=graph_encoder)
    save_lstm_checkpoint(last_path, model, config, categorical_vocabs, tokenizer, max_steps, last_valid_metrics, graph_encoder=graph_encoder)
    if isinstance(sampler, TemporalStratifiedSampler):
        save_json(sampler.diagnostics().to_dict(), output_dir / "sampling_diagnostics.json")
    train_rows_seen = int(max_steps * effective_batch_size)
    train_subset_used = training.get("max_rows") not in (None, "all") or bool(pretokenized_dir)
    runtime = {
        "train_mode": "fixed_step",
        "epoch_mode": False,
        "max_steps": int(max_steps),
        "physical_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "effective_batch_size": int(effective_batch_size),
        "sampling_mode": str(training.get("sampling_mode", training.get("train_row_sampling", "temporal_stratified"))),
        "train_rows_available": int(train_rows_available),
        "train_rows_seen_approx": int(train_rows_seen),
        "full_epoch_equivalent_fraction": float(train_rows_seen / max(train_rows_available, 1)),
        "train_subset_used": bool(train_subset_used),
        "max_train_rows": training.get("max_rows"),
        "mixed_precision_used": bool(use_amp),
        "amp_dtype": str(training.get("amp_dtype", "fp16")),
        "best_checkpoint_path": str(best_path),
        "best_valid_total_loss": float(best_valid),
        "best_step": int(best_step),
        "total_training_seconds": float(elapsed),
        "train_batch_size_requested": int(requested_batch_size if requested_batch_size is not None else batch_size),
        "train_batch_size_used": int(batch_size),
        "auto_reduce_batch_size": bool(auto_reduce_batch_size),
        "min_batch_size": int(training.get("min_batch_size", batch_size)),
        "oom_retries": int(0 if requested_batch_size in (None, batch_size) else 1),
        "final_physical_batch_size": int(batch_size),
        "pretokenized_dir": str(pretokenized_dir) if pretokenized_dir else None,
        "neighbor_cache_dir": str(neighbor_cache_dir) if neighbor_cache_dir else None,
        "architecture_changed": False,
    }
    runtime.update({key: value for key, value in train_metrics.items() if key.startswith("avg_") or key.endswith("_seconds")})
    save_json(runtime, metadata_dir / "training_runtime.json")
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


def run_lstm_fixed_steps(
    model: JointLSTMRelationalAttributeGenerator,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_amp: bool,
    loss_weights: dict[str, float],
    tokenizer: SimpleTextTokenizer,
    length_class_weights: dict[str, torch.Tensor],
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    steps_per_eval: int,
    steps_per_checkpoint: int,
    checkpoint_dir: Path,
    log_path: Path,
    metrics_log_path: Path,
    categorical_vocabs: dict[str, CategoryVocab],
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
    train_history_index: Any | None = None,
    valid_history_index: Any | None = None,
    valid_row_id_offset: int = 0,
    config: ConditionalTABDLMConfig | None = None,
    clip_norm: float = 1.0,
    validation_max_batches: int | None = 100,
) -> dict[str, Any]:
    model.train(True)
    if graph_encoder is not None:
        graph_encoder.train(True)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    best_valid = float("inf")
    best_step = 0
    last_valid_metrics: dict[str, float] = {"total_loss": float("inf")}
    component_totals: dict[str, float] = {}
    component_counts: dict[str, float] = {}
    component_corrects: dict[str, int] = {}
    timer_totals = {
        "batch_load": 0.0,
        "h2d": 0.0,
        "graph_context": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    microbatches = 0
    progress = tqdm(total=int(max_steps), desc="train_lstm_fixed_step") if tqdm is not None else None
    train_start = time.perf_counter()
    for step in range(1, int(max_steps) + 1):
        step_start = time.perf_counter()
        for _ in range(int(gradient_accumulation_steps)):
            load_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            timer_totals["batch_load"] += time.perf_counter() - load_start
            h2d_start = time.perf_counter()
            batch = move_batch_to_device(batch, device)
            timer_totals["h2d"] += time.perf_counter() - h2d_start
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype_from_name(config.raw.get("training", {}).get("amp_dtype", "fp16") if config is not None else "fp16")):
                graph_start = time.perf_counter()
                graph_context, _ = compute_graph_outputs(
                    graph_encoder,
                    train_history_index,
                    batch,
                    device,
                    deterministic=False,
                    config=config,
                    training=True,
                )
                timer_totals["graph_context"] += time.perf_counter() - graph_start
                forward_start = time.perf_counter()
                logits = model(
                    batch["foreign_key_ids"],
                    batch["datetime_values"],
                    batch["categorical_ids"],
                    batch["text_ids"],
                    graph_context=graph_context,
                )
                loss, component = lstm_joint_loss(
                    logits,
                    batch,
                    model.schema,
                    loss_weights,
                    tokenizer,
                    length_class_weights,
                    config=config,
                )
                loss = loss / max(int(gradient_accumulation_steps), 1)
                timer_totals["forward"] += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            timer_totals["backward"] += time.perf_counter() - backward_start
            accumulate_lstm_components(component, component_totals, component_counts, component_corrects, model.schema)
            microbatches += 1
        optimizer_start = time.perf_counter()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(graph_encoder.parameters()) if graph_encoder is not None else []), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        timer_totals["optimizer"] += time.perf_counter() - optimizer_start
        if progress is not None:
            elapsed = time.perf_counter() - train_start
            progress.set_postfix(
                rows_seen=step * int(gradient_accumulation_steps) * int(train_loader.batch_size or 1),
                step_s=f"{(elapsed / max(step, 1)):.3f}",
            )
            progress.update(1)
        should_eval = step == int(max_steps) or (steps_per_eval > 0 and step % int(steps_per_eval) == 0)
        should_ckpt = step == int(max_steps) or (steps_per_checkpoint > 0 and step % int(steps_per_checkpoint) == 0)
        if should_eval:
            valid_metrics = run_lstm_epoch(
                model,
                valid_loader,
                None,
                scaler,
                device,
                use_amp,
                loss_weights,
                tokenizer,
                length_class_weights,
                graph_encoder=graph_encoder,
                graph_history_index=valid_history_index,
                graph_deterministic=True,
                graph_row_id_offset=valid_row_id_offset,
                config=config,
                clip_norm=clip_norm,
                max_batches=validation_max_batches,
            )
            last_valid_metrics = valid_metrics
            current = float(valid_metrics.get("total_loss", float("inf")))
            improved = current < best_valid
            if improved:
                best_valid = current
                best_step = step
                save_lstm_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    config,
                    categorical_vocabs,
                    tokenizer,
                    step,
                    valid_metrics,
                    graph_encoder=graph_encoder,
                )
            train_metrics = finalize_lstm_component_metrics(component_totals, component_counts, component_corrects, model.schema, loss_weights, config)
            row = {
                "step": int(step),
                "max_steps": int(max_steps),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                "best_valid_total_loss": float(best_valid),
                "best_step": int(best_step),
                "rows_seen_approx": int(step * int(gradient_accumulation_steps) * int(train_loader.batch_size or 1)),
                "wall_clock_seconds": float(time.perf_counter() - train_start),
                "step_seconds": float(time.perf_counter() - step_start),
            }
            append_jsonl(log_path, row)
            append_jsonl(metrics_log_path, row)
            print(json.dumps(jsonable(row), sort_keys=True), flush=True)
        if should_ckpt:
            save_lstm_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                config,
                categorical_vocabs,
                tokenizer,
                step,
                last_valid_metrics,
                graph_encoder=graph_encoder,
            )
    if progress is not None:
        progress.close()
    total_seconds = time.perf_counter() - train_start
    denom = max(float(microbatches), 1.0)
    return {
        "best_valid_total_loss": float(best_valid),
        "best_step": int(best_step),
        "last_valid_metrics": last_valid_metrics,
        "avg_step_seconds": float(total_seconds / max(int(max_steps), 1)),
        "avg_batch_load_seconds": float(timer_totals["batch_load"] / denom),
        "avg_h2d_seconds": float(timer_totals["h2d"] / denom),
        "avg_graph_context_seconds": float(timer_totals["graph_context"] / denom),
        "avg_forward_seconds": float(timer_totals["forward"] / denom),
        "avg_backward_seconds": float(timer_totals["backward"] / denom),
        "avg_optimizer_seconds": float(timer_totals["optimizer"] / max(int(max_steps), 1)),
        "total_loop_seconds": float(total_seconds),
    }


def accumulate_lstm_components(
    component: dict[str, dict[str, float | int]],
    totals: dict[str, float],
    counts: dict[str, float],
    corrects: dict[str, int],
    schema: ConditionalTABDLMSchema,
) -> None:
    for key, stats in component.items():
        totals[key] = totals.get(key, 0.0) + float(stats["loss_sum"])
        counts[key] = counts.get(key, 0.0) + float(stats["count"])
        if key in schema.model_categorical_targets or key in {metric_name_for_lstm(column) for column in schema.model_categorical_targets}:
            corrects[key] = corrects.get(key, 0) + int(stats.get("correct", 0))


def finalize_lstm_component_metrics(
    totals: dict[str, float],
    counts: dict[str, float],
    corrects: dict[str, int],
    schema: ConditionalTABDLMSchema,
    loss_weights: dict[str, float],
    config: ConditionalTABDLMConfig | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = 0.0
    cat_metric_names = {metric_name_for_lstm(column): column for column in schema.model_categorical_targets}
    for key in sorted(totals):
        count = max(float(counts.get(key, 0.0)), 1.0)
        value = float(totals[key] / count)
        metrics[f"{key}_loss"] = value
        total += float(loss_weights.get(key, 1.0)) * value
        if key in cat_metric_names:
            metrics[f"{key}_accuracy"] = float(corrects.get(key, 0) / count)
    smoothing = text_label_smoothing_from_config(config)
    for column, value in smoothing.items():
        metric_key = "label_smoothing_summary" if column == "summary" else f"label_smoothing_{column}"
        metrics[metric_key] = float(value)
    metrics["total_loss"] = float(total)
    return metrics


def build_fixed_step_lstm_loader(
    dataset: Any,
    config: ConditionalTABDLMConfig,
    *,
    batch_size: int,
    num_workers: int,
    num_microbatches: int,
    seed: int,
) -> DataLoader:
    training = config.raw.get("training", {})
    sampling_cfg = config.raw.get("sampling", {})
    mode = str(training.get("sampling_mode", training.get("train_row_sampling", sampling_cfg.get("mode", "temporal_stratified"))))
    timestamps = getattr(dataset, "timestamps_ns", None)
    sampler = None
    shuffle = True
    if timestamps is not None and mode in {"uniform", "temporal_stratified", "temporal_weighted", "hybrid"}:
        sampler = TemporalStratifiedSampler(
            np.asarray(timestamps, dtype=np.int64),
            mode=mode,
            num_time_bins=int(sampling_cfg.get("num_time_bins", training.get("num_time_bins", 128))),
            binning=str(sampling_cfg.get("binning", training.get("binning", "quantile"))),
            replacement=bool(sampling_cfg.get("replacement", True)),
            seed=seed,
            num_samples=int(num_microbatches) * int(batch_size),
            timestamp_column=config.schema.datetime_columns[0] if config.schema.datetime_columns else None,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=make_lstm_collate_fn,
        drop_last=True,
        **dataloader_performance_kwargs(training, num_workers, train=True),
    )


def dataloader_performance_kwargs(training: dict[str, Any], num_workers: int, *, train: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(training.get("pin_memory", training.get("dataloader", {}).get("pin_memory", True))),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(
            training.get("persistent_workers", training.get("dataloader", {}).get("persistent_workers", True))
        )
        kwargs["prefetch_factor"] = int(
            training.get("prefetch_factor", training.get("dataloader", {}).get("prefetch_factor", 4))
        )
        timeout = int(training.get("dataloader_timeout_seconds", 0) or 0)
        if timeout > 0:
            kwargs["timeout"] = timeout
    if not train:
        kwargs["drop_last"] = False
    return kwargs


def gradient_accumulation_steps_for(training: dict[str, Any], physical_batch_size: int) -> int:
    if training.get("gradient_accumulation_steps") is not None:
        return max(1, int(training.get("gradient_accumulation_steps")))
    target = training.get("target_effective_batch_size", training.get("effective_batch_size"))
    if target is None:
        return 1
    return max(1, int(math.ceil(float(target) / max(int(physical_batch_size), 1))))


def compute_length_class_weights_from_pretokenized(
    dataset: PretokenizedLSTMDataset,
    config: ConditionalTABDLMConfig,
    categorical_vocabs: dict[str, CategoryVocab],
) -> dict[str, Any] | None:
    tensors: dict[str, torch.Tensor] = {}
    payloads: dict[str, Any] = {}
    for column in config.schema.length_bucket_targets:
        if column not in categorical_vocabs:
            continue
        length_cfg = config.raw.get(f"{'summary_length' if column == 'summary_length_bucket' else 'review_text_length'}_loss", {})
        if not bool(length_cfg.get("class_balanced", False)):
            continue
        col_idx = config.schema.model_categorical_targets.index(column)
        values = np.asarray(dataset.categorical_ids[dataset.indices, col_idx], dtype=np.int64)
        vocab = categorical_vocabs[column]
        counts = np.bincount(values, minlength=vocab.size).astype(float)
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
            "counts": {id_to_token[idx]: int(counts[idx]) for idx in range(vocab.size)},
            "frequencies": {id_to_token[idx]: float(freqs[idx]) for idx in range(vocab.size)},
            "weights": {id_to_token[idx]: float(weights[idx]) for idx in range(vocab.size)},
        }
    if not tensors:
        return None
    return {"tensor": tensors, "json": payloads}


def amp_dtype_from_name(name: Any) -> torch.dtype:
    if str(name).lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    max_batches: int | None = None,
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
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
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
            loss, component = lstm_joint_loss(logits, batch, model.schema, loss_weights, tokenizer, length_class_weights, config=config)
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
        if training:
            for column, rate in model.last_token_dropout_rates.items():
                metric_key = f"token_dropout_{column}_rate"
                totals[metric_key] = totals.get(metric_key, 0.0) + float(rate)
                counts[metric_key] = counts.get(metric_key, 0.0) + 1.0
    metrics: dict[str, float] = {}
    total = 0.0
    for key in sorted(totals):
        count = max(float(counts.get(key, 0.0)), 1.0)
        value = float(totals[key] / count)
        if key.startswith("token_dropout_"):
            metrics[key] = value
            continue
        metrics[f"{key}_loss"] = value
        total += float(loss_weights.get(key, 1.0)) * value
        if key in model.schema.model_categorical_targets:
            metrics[f"{key}_accuracy"] = float(corrects.get(key, 0) / count)
    smoothing = text_label_smoothing_from_config(config)
    for column, value in smoothing.items():
        metric_key = "label_smoothing_summary" if column == "summary" else f"label_smoothing_{column}"
        metrics[metric_key] = float(value)
    metrics["total_loss"] = float(total)
    return metrics


def lstm_joint_loss(
    logits: dict[str, Any],
    batch: dict[str, Any],
    schema: ConditionalTABDLMSchema,
    loss_weights: dict[str, float],
    tokenizer: SimpleTextTokenizer,
    length_class_weights: dict[str, torch.Tensor] | None = None,
    config: ConditionalTABDLMConfig | None = None,
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
    numerical_values = batch.get("numerical_values")
    if numerical_values is not None:
        for idx, column in enumerate(schema.numerical_targets):
            target = numerical_values[:, idx]
            loss_values = gaussian_nll_from_params(logits["numerical"][column], target)
            loss_sum = loss_values.sum()
            count = int(target.numel())
            losses.append(float(loss_weights.get(column, 1.0)) * (loss_sum / max(count, 1)))
            component[column] = {"loss_sum": float(loss_sum.detach().cpu()), "count": count}
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
            label_smoothing=float(text_label_smoothing_for_column(config, column)),
        )
        key = "summary_text" if column == "summary" else column
        losses.append(float(loss_weights.get(key, loss_weights.get(column, 1.0))) * (ce / max(count, 1)))
        component[key] = {"loss_sum": float(ce.detach().cpu()), "count": count}
    if not losses:
        zero = batch["foreign_key_ids"].float().sum() * 0.0
        return zero, {}
    return torch.stack(losses).sum(), component


def text_label_smoothing_from_config(config: ConditionalTABDLMConfig | None) -> dict[str, float]:
    if config is None:
        return {}
    smoothing_cfg = config.raw.get("loss", {}).get("text_label_smoothing", {})
    if not bool(smoothing_cfg.get("enabled", False)):
        return {}
    return {
        "summary": float(smoothing_cfg.get("summary", 0.0) or 0.0),
        "review_text": float(smoothing_cfg.get("review_text", 0.0) or 0.0),
    }


def text_label_smoothing_for_column(config: ConditionalTABDLMConfig | None, column: str) -> float:
    return float(text_label_smoothing_from_config(config).get(column, 0.0))


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
    numerical_metadata = load_numerical_metadata(config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.to_config(),
            "raw_config": config.raw,
            "schema": config.schema.to_dict(),
            "categorical_vocabs": {column: vocab.to_dict() for column, vocab in categorical_vocabs.items()},
            "tokenizer_metadata": tokenizer.to_dict(),
            "numerical_metadata": numerical_metadata,
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
    if checkpoint.get("numerical_metadata") is not None:
        raw_config = dict(raw_config)
        raw_config["_numerical_metadata"] = checkpoint["numerical_metadata"]
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
    numerical_metadata = ckpt_config.raw.get("_numerical_metadata") or load_numerical_metadata(ckpt_config)
    attrs: dict[str, list[Any]] = {column: [] for column in ckpt_config.schema.categorical_targets + ckpt_config.schema.numerical_targets + ckpt_config.schema.text_targets}
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
        for column in ckpt_config.schema.numerical_targets:
            sampled = sample_gaussian_params(
                generated["numerical_params"][column],
                temperature=float(sampling.get("numerical_temperature", sampling.get("temperature", 0.9))),
            )
            values = inverse_transform_numerical(sampled, numerical_metadata.get(column, {})).detach().cpu().tolist()
            attrs[column].extend(values)
        for column in ckpt_config.schema.text_targets:
            attrs[column].extend(generated["text"][column])
            lengths[column].extend(generated["text_lengths"][column])
    total_seconds = float(time.perf_counter() - start_time)
    output = spine.loc[:, list(ckpt_config.schema.condition_columns)].copy()
    for column in ckpt_config.schema.categorical_targets:
        output[column] = attrs[column]
    for column in ckpt_config.schema.numerical_targets:
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
    review_decoder_cfg = config.raw.get("review_text_decoder", {})
    smoothing_cfg = config.raw.get("loss", {}).get("text_label_smoothing", {})
    token_dropout_cfg = config.raw.get("training_regularization", {}).get("decoder_input_token_dropout", {})
    no_repeat_cfg = config.raw.get("sampling", {}).get("no_repeat_ngram", {})
    overlap_cfg = config.raw.get("sampling", {}).get("exact_train_overlap_blocking", {})
    summary_temperature = sampling_value_for_metadata(config.raw.get("sampling", {}), "temperature", "summary", 0.9)
    review_temperature = sampling_value_for_metadata(config.raw.get("sampling", {}), "temperature", "review_text", 0.9)
    summary_top_p = sampling_value_for_metadata(config.raw.get("sampling", {}), "top_p", "summary", 0.95)
    review_top_p = sampling_value_for_metadata(config.raw.get("sampling", {}), "top_p", "review_text", 0.95)
    auto_review = config.raw.get("_auto_text_length_metadata", {}).get("review_text", {})
    loss_weights = dict(config.raw.get("loss_weights", {}))
    metadata: dict[str, Any] = {
        "experiment_name": config.raw.get("experiment_name", Path(config.output_dir).name),
        "base_experiment": config.raw.get("base_experiment"),
        "model_family": "joint_lstm_generator",
        "base_graph_model": "v2_structure_only_temporal_graph",
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in config.schema.text_targets,
        "review_text_separate_stage": False,
        "uses_diffusion": False,
        "uses_transformer_backbone": False,
        "text_decoder_type": config.raw.get("text_decoder", {}).get("type", "lstm"),
        "review_text_conditioned_on_summary": bool(review_decoder_cfg.get("condition_on_summary", False)),
        "summary_condition_type": review_decoder_cfg.get("summary_condition_type", "none"),
        "condition_columns": list(config.schema.condition_columns),
        "target_columns": {
            "categorical": list(config.schema.categorical_targets),
            "numerical": list(config.schema.numerical_targets),
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
        "regularization": {
            "text_label_smoothing_enabled": bool(smoothing_cfg.get("enabled", False)),
            "summary_label_smoothing": float(smoothing_cfg.get("summary", 0.0) or 0.0),
            "review_text_label_smoothing": float(smoothing_cfg.get("review_text", 0.0) or 0.0),
            "decoder_input_token_dropout_enabled": bool(token_dropout_cfg.get("enabled", False)),
            "summary_token_dropout": float(token_dropout_cfg.get("summary", 0.0) or 0.0),
            "review_text_token_dropout": float(token_dropout_cfg.get("review_text", 0.0) or 0.0),
        },
        "sampling_privacy_controls": {
            "exact_train_overlap_blocking_enabled": bool(overlap_cfg.get("enabled", False)),
            "no_repeat_ngram_enabled": bool(no_repeat_cfg.get("enabled", False)),
            "summary_no_repeat_ngram_size": int(no_repeat_cfg.get("summary_ngram_size", 0) or 0),
            "review_text_no_repeat_ngram_size": int(no_repeat_cfg.get("review_text_ngram_size", 0) or 0),
            "summary_temperature": float(summary_temperature),
            "review_text_temperature": float(review_temperature),
            "summary_top_p": float(summary_top_p),
            "review_text_top_p": float(review_top_p),
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


def sampling_value_for_metadata(sampling: dict[str, Any], key: str, column: str, default: float) -> float:
    value = sampling.get(key, default)
    if isinstance(value, dict):
        return float(value.get(column, value.get("default", default)))
    return float(value)


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
        return value.to(device, non_blocking=str(device).startswith("cuda"))
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    return value


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        json.dump(jsonable(row), handle, sort_keys=True)
        handle.write("\n")
