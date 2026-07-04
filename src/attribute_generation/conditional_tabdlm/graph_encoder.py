"""Structure-only temporal graph encoder for Conditional TABDLM v2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_schema import attribute_denoising_config, graph_encoder_config, graph_mode
from .model import DateTimeEncoder
from .schema import ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer


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


class ReviewEventAttributeStateEncoder(nn.Module):
    """Encode noised review-event attribute states without using clean target attrs."""

    def __init__(
        self,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
        hidden_dim: int,
        rating_dim: int = 64,
        verified_dim: int = 32,
        summary_length_dim: int = 32,
        summary_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.text_tokenizer = text_tokenizer
        self.categorical_embeddings = nn.ModuleDict()
        cat_dims = []
        for column in schema.model_categorical_targets:
            if column == "rating":
                dim = int(rating_dim)
            elif column == "verified":
                dim = int(verified_dim)
            elif column == "summary_length_bucket":
                dim = int(summary_length_dim)
            else:
                dim = int(summary_length_dim)
            self.categorical_embeddings[column] = nn.Embedding(categorical_vocabs[column].size + 1, dim)
            cat_dims.append(dim)
        self.text_embedding = nn.Embedding(text_tokenizer.vocab_size, int(summary_dim))
        input_dim = int(sum(cat_dims) + (int(summary_dim) if schema.text_targets else 0))
        self.projector = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(
        self,
        categorical_ids: torch.Tensor,
        text_ids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pieces = []
        for idx, column in enumerate(self.schema.model_categorical_targets):
            pieces.append(self.categorical_embeddings[column](categorical_ids[..., idx]))
        if self.schema.text_targets:
            column = self.schema.text_targets[0]
            ids = text_ids[column]
            emb = self.text_embedding(ids)
            mask = (ids != self.text_tokenizer.pad_id).float().unsqueeze(-1)
            pooled = (emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
            pieces.append(pooled)
        return self.projector(torch.cat(pieces, dim=-1))


class TemporalAttributeDenoisingGraphEncoder(TemporalStructureOnlyGraphEncoder):
    """Past-only temporal graph encoder with noised/generated attribute states."""

    def __init__(
        self,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
        **kwargs: Any,
    ):
        attr_cfg = kwargs.pop("attribute_denoising", {})
        super().__init__(**kwargs)
        self.schema = schema
        self.categorical_vocabs = categorical_vocabs
        self.text_tokenizer = text_tokenizer
        embedding_cfg = attr_cfg.get("attribute_embedding", {}) if isinstance(attr_cfg, dict) else {}
        self.attr_state_encoder = ReviewEventAttributeStateEncoder(
            schema,
            categorical_vocabs,
            text_tokenizer,
            hidden_dim=self.hidden_dim,
            rating_dim=int(embedding_cfg.get("rating_dim", 64)),
            verified_dim=int(embedding_cfg.get("verified_dim", 32)),
            summary_length_dim=int(embedding_cfg.get("summary_length_dim", 32)),
            summary_dim=int(embedding_cfg.get("summary_dim", 128)),
            dropout=float(embedding_cfg.get("dropout", self.dropout)),
        )
        self.aux_categorical_heads = nn.ModuleDict(
            {column: nn.Linear(self.hidden_dim, categorical_vocabs[column].size) for column in schema.model_categorical_targets}
        )
        self.aux_text_heads = nn.ModuleDict(
            {column: nn.Linear(self.hidden_dim, text_tokenizer.vocab_size) for column in schema.text_targets}
        )

    @classmethod
    def from_config(
        cls,
        raw_config: dict[str, Any],
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
    ) -> "TemporalAttributeDenoisingGraphEncoder":
        id_cfg = raw_config.get("id_encoding", {})
        dt_cfg = raw_config.get("datetime_encoding", {})
        enc_cfg = graph_encoder_config(raw_config)
        return cls(
            schema=schema,
            categorical_vocabs=categorical_vocabs,
            text_tokenizer=text_tokenizer,
            num_hash_buckets=int(id_cfg.get("num_buckets", 262144)),
            entity_embedding_dim=int(id_cfg.get("embedding_dim", 128)),
            datetime_embedding_dim=int(dt_cfg.get("embedding_dim", 64)),
            hidden_dim=int(enc_cfg.get("hidden_dim", 256)),
            output_dim=int(enc_cfg.get("output_dim", raw_config.get("model", {}).get("graph_context_dim", 256))),
            num_layers=int(enc_cfg.get("num_layers", 2)),
            dropout=float(enc_cfg.get("dropout", 0.1)),
            normalize_output=bool(enc_cfg.get("normalize_output", True)),
            attribute_denoising=attribute_denoising_config(raw_config),
        )

    def forward(self, graph_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self._encode_target(
            graph_batch["target_customer_hash"],
            graph_batch["target_product_hash"],
            graph_batch["target_time"],
        )
        if "target_categorical_ids" in graph_batch:
            target = target + self.attr_state_encoder(
                graph_batch["target_categorical_ids"],
                {column: graph_batch[f"target_text_ids_{column}"] for column in self.schema.text_targets},
            )
        customer_hist = self._encode_history_with_attr(graph_batch, "customer", source_type=0)
        product_hist = self._encode_history_with_attr(graph_batch, "product", source_type=1)
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

    def _encode_history_with_attr(self, graph_batch: dict[str, torch.Tensor], kind: str, *, source_type: int) -> torch.Tensor:
        base = self._encode_history(
            graph_batch[f"{kind}_history_customer_hash"],
            graph_batch[f"{kind}_history_product_hash"],
            graph_batch[f"{kind}_history_time"],
            graph_batch[f"{kind}_history_mask"],
            source_type=source_type,
        )
        cat_key = f"{kind}_history_categorical_ids"
        if cat_key not in graph_batch:
            return base
        history_attr = self.attr_state_encoder(
            graph_batch[cat_key],
            {column: graph_batch[f"{kind}_history_text_ids_{column}"] for column in self.schema.text_targets},
        )
        weights = graph_batch[f"{kind}_history_mask"].float().unsqueeze(-1)
        pooled = (history_attr * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return base + pooled

    def auxiliary_neighbor_loss(self, graph_batch: dict[str, torch.Tensor], max_nodes: int = 256) -> tuple[torch.Tensor, dict[str, Any]]:
        reps = []
        labels = []
        masks = []
        text_labels: dict[str, list[torch.Tensor]] = {column: [] for column in self.schema.text_targets}
        for kind, source_type in (("customer", 0), ("product", 1)):
            cat_key = f"{kind}_history_categorical_ids"
            clean_key = f"{kind}_history_clean_categorical_ids"
            if cat_key not in graph_batch or clean_key not in graph_batch:
                continue
            struct = self._encode_history_events(
                graph_batch[f"{kind}_history_customer_hash"],
                graph_batch[f"{kind}_history_product_hash"],
                graph_batch[f"{kind}_history_time"],
                source_type=source_type,
            )
            attr = self.attr_state_encoder(
                graph_batch[cat_key],
                {column: graph_batch[f"{kind}_history_text_ids_{column}"] for column in self.schema.text_targets},
            )
            reps.append(struct + attr)
            labels.append(graph_batch[clean_key])
            masks.append(graph_batch[f"{kind}_history_mask"])
            for column in self.schema.text_targets:
                text_labels[column].append(graph_batch[f"{kind}_history_clean_text_ids_{column}"])
        if not reps:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {"loss_sum": 0.0, "count": 0}
        rep = torch.cat([value.reshape(-1, self.hidden_dim) for value in reps], dim=0)
        label = torch.cat([value.reshape(-1, value.shape[-1]) for value in labels], dim=0)
        mask = torch.cat([value.reshape(-1) for value in masks], dim=0).bool()
        if max_nodes > 0 and int(mask.sum().item()) > int(max_nodes):
            keep_positions = torch.where(mask)[0][: int(max_nodes)]
            trimmed_mask = torch.zeros_like(mask)
            trimmed_mask[keep_positions] = True
            mask = trimmed_mask
        losses = []
        count = int(mask.sum().detach().cpu())
        for idx, column in enumerate(self.schema.model_categorical_targets):
            if count == 0:
                continue
            logits = self.aux_categorical_heads[column](rep)
            losses.append(F.cross_entropy(logits[mask], label[mask, idx], reduction="mean"))
        for column in self.schema.text_targets:
            clean = torch.cat([value.reshape(-1, value.shape[-1]) for value in text_labels[column]], dim=0)
            if count == 0:
                continue
            token_logits = self.aux_text_heads[column](rep[mask])
            repeated = token_logits.unsqueeze(1).expand(-1, clean.shape[1], -1)
            losses.append(
                F.cross_entropy(
                    repeated.reshape(-1, repeated.shape[-1]),
                    clean[mask].reshape(-1),
                    ignore_index=self.text_tokenizer.pad_id,
                    reduction="mean",
                )
            )
        if not losses:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {"loss_sum": 0.0, "count": 0}
        loss = torch.stack(losses).mean()
        return loss, {"loss_sum": float(loss.detach().cpu()) * max(count, 1), "count": max(count, 1)}

    def _encode_history_events(
        self,
        customer_hash: torch.Tensor,
        product_hash: torch.Tensor,
        timestamp: torch.Tensor,
        *,
        source_type: int,
    ) -> torch.Tensor:
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
        return self.event_projector(features)

    def to_config(self) -> dict[str, Any]:
        data = super().to_config()
        data["type"] = "temporal_attr_denoising_ego_encoder"
        return data


def build_temporal_graph_encoder(
    raw_config: dict[str, Any],
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
) -> TemporalStructureOnlyGraphEncoder:
    if graph_mode(raw_config) == "temporal_attribute_denoising":
        return TemporalAttributeDenoisingGraphEncoder.from_config(raw_config, schema, categorical_vocabs, text_tokenizer)
    return TemporalStructureOnlyGraphEncoder.from_config(raw_config)
