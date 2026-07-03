"""Conditional TABDLM-style masked denoising transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .schema import ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer


class DateTimeEncoder(nn.Module):
    """Continuous-time Fourier encoder for Unix timestamps."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        num_freqs = max(1, math.ceil(self.embedding_dim / 2))
        periods = torch.logspace(0.0, 4.0, steps=num_freqs)
        self.register_buffer("periods", periods, persistent=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        days = timestamps.float() / 86400.0
        angles = days.unsqueeze(-1) * (2.0 * math.pi / self.periods.view(1, 1, -1))
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return features[..., : self.embedding_dim]


class ConditionalTABDLM(nn.Module):
    """Masked denoising model for p(target attributes | FK/date conditions)."""

    def __init__(
        self,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
        num_hash_buckets: int = 262144,
        id_embedding_dim: int = 128,
        datetime_embedding_dim: int = 64,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        dropout: float = 0.1,
        condition_dim: int = 256,
    ):
        super().__init__()
        self.schema = schema
        self.num_hash_buckets = int(num_hash_buckets)
        self.id_embedding_dim = int(id_embedding_dim)
        self.datetime_embedding_dim = int(datetime_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.condition_dim = int(condition_dim)
        self.text_vocab_size = int(text_tokenizer.vocab_size)
        self.text_pad_id = int(text_tokenizer.pad_id)
        self.categorical_vocab_sizes = {
            column: categorical_vocabs[column].size for column in schema.model_categorical_targets
        }

        self.foreign_key_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_hash_buckets, self.id_embedding_dim) for _ in schema.foreign_key_columns]
        )
        self.foreign_key_projectors = nn.ModuleList(
            [nn.Linear(self.id_embedding_dim, self.condition_dim) for _ in schema.foreign_key_columns]
        )
        self.datetime_encoder = DateTimeEncoder(self.datetime_embedding_dim)
        self.datetime_projectors = nn.ModuleList(
            [nn.Linear(self.datetime_embedding_dim, self.condition_dim) for _ in schema.datetime_columns]
        )
        self.condition_column_embedding = nn.Embedding(
            max(1, len(schema.condition_columns)), self.condition_dim
        )
        self.condition_type_embedding = nn.Embedding(2, self.condition_dim)
        self.condition_norm = nn.LayerNorm(self.condition_dim)
        self.condition_to_hidden = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        self.categorical_embeddings = nn.ModuleDict(
            {
                column: nn.Embedding(categorical_vocabs[column].size + 1, self.hidden_dim)
                for column in schema.model_categorical_targets
            }
        )
        self.text_embeddings = nn.ModuleDict(
            {
                column: nn.Embedding(self.text_vocab_size, self.hidden_dim)
                for column in schema.text_targets
            }
        )
        self.target_column_embedding = nn.Embedding(max(1, len(schema.model_target_columns)), self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_target_positions, self.hidden_dim)
        self.diffusion_time = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.categorical_heads = nn.ModuleDict(
            {
                column: nn.Linear(self.hidden_dim, categorical_vocabs[column].size)
                for column in schema.model_categorical_targets
            }
        )
        self.text_heads = nn.ModuleDict(
            {column: nn.Linear(self.hidden_dim, self.text_vocab_size) for column in schema.text_targets}
        )

    @property
    def max_target_positions(self) -> int:
        return len(self.schema.model_categorical_targets) + sum(
            int(self.schema.text_max_lengths[column]) for column in self.schema.text_targets
        )

    def encode_conditions(
        self,
        foreign_key_ids: torch.Tensor,
        datetime_values: torch.Tensor,
    ) -> torch.Tensor:
        condition_tokens: list[torch.Tensor] = []
        token_index = 0
        for idx, embedding in enumerate(self.foreign_key_embeddings):
            token = self.foreign_key_projectors[idx](embedding(foreign_key_ids[:, idx]))
            token = token + self.condition_column_embedding.weight[token_index] + self.condition_type_embedding.weight[0]
            condition_tokens.append(token)
            token_index += 1
        datetime_features = self.datetime_encoder(datetime_values)
        for idx, projector in enumerate(self.datetime_projectors):
            token = projector(datetime_features[:, idx, :])
            token = token + self.condition_column_embedding.weight[token_index] + self.condition_type_embedding.weight[1]
            condition_tokens.append(token)
            token_index += 1
        stacked = torch.stack(condition_tokens, dim=1)
        return self.condition_to_hidden(self.condition_norm(stacked.mean(dim=1)))

    def forward(
        self,
        foreign_key_ids: torch.Tensor,
        datetime_values: torch.Tensor,
        categorical_input_ids: torch.Tensor,
        text_input_ids: dict[str, torch.Tensor],
        text_attention: dict[str, torch.Tensor] | None = None,
        diffusion_t: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch_size = foreign_key_ids.shape[0]
        device = foreign_key_ids.device
        if diffusion_t is None:
            diffusion_t = torch.ones(batch_size, dtype=torch.float32, device=device)
        condition = self.encode_conditions(foreign_key_ids, datetime_values)
        time_bias = self.diffusion_time(diffusion_t.float().view(-1, 1))
        shared_bias = (condition + time_bias).unsqueeze(1)

        embeddings: list[torch.Tensor] = []
        attention_parts: list[torch.Tensor] = []
        output_slices: dict[str, tuple[int, int]] = {}
        pos = 0
        target_col_idx = 0
        for col_idx, column in enumerate(self.schema.model_categorical_targets):
            token = self.categorical_embeddings[column](categorical_input_ids[:, col_idx])
            col_emb = self.target_column_embedding.weight[target_col_idx].view(1, 1, -1)
            embeddings.append((token.unsqueeze(1) + col_emb))
            attention_parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            output_slices[column] = (pos, pos + 1)
            pos += 1
            target_col_idx += 1

        for column in self.schema.text_targets:
            ids = text_input_ids[column]
            token = self.text_embeddings[column](ids)
            length = ids.shape[1]
            col_emb = self.target_column_embedding.weight[target_col_idx].view(1, 1, -1)
            embeddings.append(token + col_emb)
            if text_attention is None:
                attention_parts.append(torch.ones(batch_size, length, dtype=torch.bool, device=device))
            else:
                attention_parts.append(text_attention[column].bool())
            output_slices[column] = (pos, pos + length)
            pos += length
            target_col_idx += 1

        sequence = torch.cat(embeddings, dim=1)
        positions = torch.arange(sequence.shape[1], device=device).view(1, -1)
        sequence = sequence + self.position_embedding(positions) + shared_bias
        attention = torch.cat(attention_parts, dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~attention)
        encoded = self.output_norm(encoded)

        categorical_logits: dict[str, torch.Tensor] = {}
        for column in self.schema.model_categorical_targets:
            start, _ = output_slices[column]
            categorical_logits[column] = self.categorical_heads[column](encoded[:, start, :])
        text_logits: dict[str, torch.Tensor] = {}
        for column in self.schema.text_targets:
            start, end = output_slices[column]
            text_logits[column] = self.text_heads[column](encoded[:, start:end, :])
        return {"categorical": categorical_logits, "text": text_logits}

    def to_config(self) -> dict[str, Any]:
        return {
            "num_hash_buckets": self.num_hash_buckets,
            "id_embedding_dim": self.id_embedding_dim,
            "datetime_embedding_dim": self.datetime_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "condition_dim": self.condition_dim,
            "text_vocab_size": self.text_vocab_size,
            "categorical_vocab_sizes": dict(self.categorical_vocab_sizes),
        }
