"""Residual diffusion model for V3 non-text attributes."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .nontext_diffusion_model import ResidualMLPBlock
from .nontext_diffusion_schedules import sinusoidal_time_embedding


class ResidualTemporalFeatureDiffusionModelV3(nn.Module):
    """Predict residual logits on top of temporal/block/entity base logits."""

    def __init__(
        self,
        cat_cols: List[str],
        cat_vocab_sizes: Dict[str, int],
        num_numerical: int,
        continuous_feature_dim: int,
        discrete_feature_vocab_sizes: Dict[str, int],
        rating_num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        token_dim: int = 32,
        time_dim: int = 32,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.cat_vocab_sizes = dict(cat_vocab_sizes)
        self.num_numerical = int(num_numerical)
        self.discrete_feature_names = list(discrete_feature_vocab_sizes)
        self.rating_num_classes = int(rating_num_classes)
        self.time_dim = int(time_dim)
        self.cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(int(cat_vocab_sizes[col]) + 1, token_dim)
                for col in self.cat_cols
            }
        )
        self.discrete_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(size), token_dim)
                for name, size in discrete_feature_vocab_sizes.items()
            }
        )
        base_dim = self.rating_num_classes + 2
        input_dim = (
            len(self.cat_cols) * token_dim
            + len(self.discrete_feature_names) * token_dim
            + int(continuous_feature_dim)
            + self.num_numerical
            + base_dim
            + time_dim
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(int(num_layers))]
        )
        self.rating_residual_head = nn.Linear(hidden_dim, self.rating_num_classes)
        self.verified_residual_head = nn.Linear(hidden_dim, 2)
        self.num_head: Optional[nn.Linear]
        self.num_head = nn.Linear(hidden_dim, self.num_numerical) if self.num_numerical else None

    def forward(
        self,
        cat_tokens: Dict[str, torch.Tensor],
        continuous_features: torch.Tensor,
        discrete_features: Dict[str, torch.Tensor],
        diffusion_t: torch.Tensor,
        base_rating_logits: torch.Tensor,
        base_verified_logits: torch.Tensor,
        numerical_noisy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pieces = []
        for col in self.cat_cols:
            pieces.append(self.cat_embeddings[col](cat_tokens[col].long()))
        for name in self.discrete_feature_names:
            pieces.append(self.discrete_embeddings[name](discrete_features[name].long()))
        pieces.append(continuous_features.float())
        if self.num_numerical:
            if numerical_noisy is None:
                numerical_noisy = continuous_features.new_zeros(
                    (continuous_features.shape[0], self.num_numerical)
                )
            pieces.append(numerical_noisy.float())
        pieces.append(base_rating_logits.float())
        pieces.append(base_verified_logits.float())
        pieces.append(sinusoidal_time_embedding(diffusion_t.float(), self.time_dim))
        h = self.backbone(self.input_proj(torch.cat(pieces, dim=1)))
        rating_residual = self.rating_residual_head(h)
        verified_residual = self.verified_residual_head(h)
        out: Dict[str, torch.Tensor] = {
            "rating_residual_logits": rating_residual,
            "verified_residual_logits": verified_residual,
            "rating_logits": base_rating_logits + rating_residual,
            "verified_logits": base_verified_logits + verified_residual,
        }
        if self.num_head is not None:
            out["num_pred"] = self.num_head(h)
        return out
