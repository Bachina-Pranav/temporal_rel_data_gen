"""Lightweight conditional masked text diffusion model for Text V1."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSummaryMaskedDiffusionV1(nn.Module):
    """A small BERT-style encoder with soft condition prompt tokens."""

    def __init__(
        self,
        vocab_size: int,
        condition_dim: int,
        max_summary_tokens: int = 32,
        num_condition_tokens: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        sep_token_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.condition_dim = int(condition_dim)
        self.max_summary_tokens = int(max_summary_tokens)
        self.num_condition_tokens = int(num_condition_tokens)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.hidden_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.soft_prompt_projector = nn.Linear(self.hidden_dim, self.num_condition_tokens * self.hidden_dim)
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
        self.mlm_head = nn.Linear(self.hidden_dim, self.vocab_size)

    @property
    def max_sequence_length(self) -> int:
        return 1 + self.num_condition_tokens + 1 + self.max_summary_tokens + 1

    @property
    def summary_offset(self) -> int:
        return 1 + self.num_condition_tokens + 1

    def soft_prompt(self, condition_features: torch.Tensor) -> torch.Tensor:
        encoded = self.condition_encoder(condition_features)
        prompt = self.soft_prompt_projector(encoded)
        return prompt.view(condition_features.shape[0], self.num_condition_tokens, self.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, summary_len = input_ids.shape
        if summary_len > self.max_summary_tokens:
            raise ValueError("input_ids length exceeds max_summary_tokens")

        device = input_ids.device
        cls_ids = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=device)
        sep_ids = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)
        cls_emb = self.token_embedding(cls_ids)
        sep_emb = self.token_embedding(sep_ids)
        prompt_emb = self.soft_prompt(condition_features)
        content_emb = self.token_embedding(input_ids)
        embeddings = torch.cat([cls_emb, prompt_emb, sep_emb, content_emb, sep_emb], dim=1)
        positions = torch.arange(embeddings.shape[1], device=device).view(1, -1)
        embeddings = embeddings + self.position_embedding(positions)

        prefix_len = self.summary_offset
        prefix_mask = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
        suffix_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        full_attention = torch.cat([prefix_mask, attention_mask.long(), suffix_mask], dim=1)
        key_padding_mask = full_attention == 0

        encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        summary_states = encoded[:, self.summary_offset : self.summary_offset + summary_len, :]
        logits = self.mlm_head(self.output_norm(summary_states))
        output: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return output

    def to_config(self) -> Dict[str, int | float | str]:
        return {
            "model_type": "lightweight_masked_summary_transformer",
            "vocab_size": self.vocab_size,
            "condition_dim": self.condition_dim,
            "max_summary_tokens": self.max_summary_tokens,
            "num_condition_tokens": self.num_condition_tokens,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
        }

    @classmethod
    def from_config(cls, config: Dict[str, int | float | str]) -> "TemporalSummaryMaskedDiffusionV1":
        return cls(
            vocab_size=int(config["vocab_size"]),
            condition_dim=int(config["condition_dim"]),
            max_summary_tokens=int(config["max_summary_tokens"]),
            num_condition_tokens=int(config["num_condition_tokens"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            pad_token_id=int(config["pad_token_id"]),
            cls_token_id=int(config["cls_token_id"]),
            sep_token_id=int(config["sep_token_id"]),
        )
