"""Feature-conditioned diffusion model for non-text review attributes."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .nontext_diffusion_schedules import sinusoidal_time_embedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalFeatureDiffusionModel(nn.Module):
    """Residual MLP over noisy attrs, causal features, block IDs, and time."""

    def __init__(
        self,
        cat_cols: List[str],
        cat_vocab_sizes: Dict[str, int],
        num_numerical: int,
        continuous_feature_dim: int,
        discrete_feature_vocab_sizes: Dict[str, int],
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
        input_dim = (
            len(self.cat_cols) * token_dim
            + len(self.discrete_feature_names) * token_dim
            + int(continuous_feature_dim)
            + self.num_numerical
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
        self.cat_heads = nn.ModuleDict(
            {
                col: nn.Linear(hidden_dim, int(cat_vocab_sizes[col]))
                for col in self.cat_cols
            }
        )
        self.num_head: Optional[nn.Linear]
        self.num_head = nn.Linear(hidden_dim, self.num_numerical) if self.num_numerical else None

    def forward(
        self,
        cat_tokens: Dict[str, torch.Tensor],
        continuous_features: torch.Tensor,
        discrete_features: Dict[str, torch.Tensor],
        diffusion_t: torch.Tensor,
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
        pieces.append(sinusoidal_time_embedding(diffusion_t.float(), self.time_dim))
        x = torch.cat(pieces, dim=1)
        h = self.backbone(self.input_proj(x))
        out: Dict[str, torch.Tensor] = {
            f"{col}_logits": self.cat_heads[col](h) for col in self.cat_cols
        }
        if self.num_head is not None:
            out["num_pred"] = self.num_head(h)
        return out


class TemporalGNNFeatureDiffusionModel(nn.Module):
    """Interface placeholder for a later temporal GNN implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "temporal_gnn_feature_diffusion is reserved for a future extension; "
            "use temporal_feature_diffusion for v1."
        )
