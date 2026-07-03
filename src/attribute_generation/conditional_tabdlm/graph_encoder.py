"""Structure-only temporal graph encoder for Conditional TABDLM v2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .graph_schema import graph_encoder_config
from .model import DateTimeEncoder


class TemporalStructureOnlyGraphEncoder(nn.Module):
    """Encode past-only temporal customer/product histories for each target event."""

    def __init__(
        self,
        num_hash_buckets: int = 262144,
        entity_embedding_dim: int = 128,
        datetime_embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.num_hash_buckets = int(num_hash_buckets)
        self.entity_embedding_dim = int(entity_embedding_dim)
        self.datetime_embedding_dim = int(datetime_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.normalize_output = bool(normalize_output)

        self.customer_embedding = nn.Embedding(self.num_hash_buckets, self.entity_embedding_dim)
        self.product_embedding = nn.Embedding(self.num_hash_buckets, self.entity_embedding_dim)
        self.node_type_embedding = nn.Embedding(3, self.entity_embedding_dim)
        self.history_source_embedding = nn.Embedding(2, self.entity_embedding_dim)
        self.datetime_encoder = DateTimeEncoder(self.datetime_embedding_dim)
        event_input_dim = self.entity_embedding_dim * 4 + self.datetime_embedding_dim
        target_input_dim = self.entity_embedding_dim * 3 + self.datetime_embedding_dim
        self.event_projector = nn.Sequential(
            nn.Linear(event_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.target_projector = nn.Sequential(
            nn.Linear(target_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        fusion_layers: list[nn.Module] = []
        fusion_dim = self.hidden_dim * 3 + 4
        for layer_idx in range(max(1, self.num_layers)):
            in_dim = fusion_dim if layer_idx == 0 else self.hidden_dim
            fusion_layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.LayerNorm(self.hidden_dim),
                ]
            )
        fusion_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        if self.normalize_output:
            fusion_layers.append(nn.LayerNorm(self.output_dim))
        self.fusion = nn.Sequential(*fusion_layers)

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> "TemporalStructureOnlyGraphEncoder":
        id_cfg = raw_config.get("id_encoding", {})
        dt_cfg = raw_config.get("datetime_encoding", {})
        enc_cfg = graph_encoder_config(raw_config)
        return cls(
            num_hash_buckets=int(id_cfg.get("num_buckets", 262144)),
            entity_embedding_dim=int(id_cfg.get("embedding_dim", 128)),
            datetime_embedding_dim=int(dt_cfg.get("embedding_dim", 64)),
            hidden_dim=int(enc_cfg.get("hidden_dim", 256)),
            output_dim=int(enc_cfg.get("output_dim", raw_config.get("model", {}).get("graph_context_dim", 256))),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            dropout=float(enc_cfg.get("dropout", 0.1)),
            normalize_output=bool(enc_cfg.get("normalize_output", True)),
        )

    def forward(self, graph_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self._encode_target(
            graph_batch["target_customer_hash"],
            graph_batch["target_product_hash"],
            graph_batch["target_time"],
        )
        customer_hist = self._encode_history(
            graph_batch["customer_history_customer_hash"],
            graph_batch["customer_history_product_hash"],
            graph_batch["customer_history_time"],
            graph_batch["customer_history_mask"],
            source_type=0,
        )
        product_hist = self._encode_history(
            graph_batch["product_history_customer_hash"],
            graph_batch["product_history_product_hash"],
            graph_batch["product_history_time"],
            graph_batch["product_history_mask"],
            source_type=1,
        )
        customer_counts = graph_batch["customer_history_mask"].float().sum(dim=1, keepdim=True)
        product_counts = graph_batch["product_history_mask"].float().sum(dim=1, keepdim=True)
        count_features = torch.cat(
            [
                torch.log1p(customer_counts),
                torch.log1p(product_counts),
                (customer_counts > 0).float(),
                (product_counts > 0).float(),
            ],
            dim=1,
        )
        return self.fusion(torch.cat([target, customer_hist, product_hist, count_features], dim=1))

    def _encode_target(self, customer_hash: torch.Tensor, product_hash: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        batch_size = int(customer_hash.shape[0])
        event_type = self.node_type_embedding(torch.full((batch_size,), 2, dtype=torch.long, device=customer_hash.device))
        pieces = [
            self.customer_embedding(customer_hash),
            self.product_embedding(product_hash),
            event_type,
            self.datetime_encoder(timestamp.view(-1, 1)).squeeze(1),
        ]
        return self.target_projector(torch.cat(pieces, dim=1))

    def _encode_history(
        self,
        customer_hash: torch.Tensor,
        product_hash: torch.Tensor,
        timestamp: torch.Tensor,
        mask: torch.Tensor,
        *,
        source_type: int,
    ) -> torch.Tensor:
        if customer_hash.shape[1] == 0:
            return torch.zeros((customer_hash.shape[0], self.hidden_dim), dtype=torch.float32, device=customer_hash.device)
        batch_size, width = customer_hash.shape
        event_type = self.node_type_embedding(torch.full((batch_size, width), 2, dtype=torch.long, device=customer_hash.device))
        source = self.history_source_embedding(torch.full((batch_size, width), int(source_type), dtype=torch.long, device=customer_hash.device))
        features = torch.cat(
            [
                self.customer_embedding(customer_hash),
                self.product_embedding(product_hash),
                event_type,
                source,
                self.datetime_encoder(timestamp),
            ],
            dim=-1,
        )
        encoded = self.event_projector(features)
        weights = mask.float().unsqueeze(-1)
        summed = (encoded * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "temporal_structure_ego_encoder",
            "num_hash_buckets": self.num_hash_buckets,
            "entity_embedding_dim": self.entity_embedding_dim,
            "datetime_embedding_dim": self.datetime_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "normalize_output": self.normalize_output,
        }
