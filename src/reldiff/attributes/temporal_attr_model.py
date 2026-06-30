"""Small temporal graph-conditioned denoiser for review attributes."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float().view(-1, 1)
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                -torch.log(torch.tensor(10000.0, device=timesteps.device)),
                half_dim,
                device=timesteps.device,
            )
        )
        args = timesteps * frequencies.view(1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.cat(
                [embedding, torch.zeros((embedding.shape[0], 1), device=embedding.device)],
                dim=1,
            )
        return self.proj(embedding)


class TemporalRelDiffAttrModel(nn.Module):
    """Denoise categorical attributes and text latents for review nodes."""

    def __init__(
        self,
        cat_cols: List[str],
        num_classes: List[int],
        text_dims: Dict[str, int],
        time_feature_dim: int,
        context_feature_dim: int,
        hidden_dim: int = 128,
        cat_embed_dim: int = 16,
        time_embed_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.num_classes = [int(value) for value in num_classes]
        self.text_dims = {column: int(dim) for column, dim in text_dims.items()}
        self.time_feature_dim = int(time_feature_dim)
        self.context_feature_dim = int(context_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cat_embed_dim = int(cat_embed_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_class + 1, cat_embed_dim) for num_class in self.num_classes]
        )
        self.text_projections = nn.ModuleDict(
            {
                column: nn.Linear(dim, hidden_dim)
                for column, dim in self.text_dims.items()
            }
        )
        self.diffusion_time = SinusoidalTimeEmbedding(time_embed_dim)

        input_dim = (
            len(self.cat_cols) * cat_embed_dim
            + len(self.text_dims) * hidden_dim
            + self.time_feature_dim
            + self.context_feature_dim
            + time_embed_dim
        )
        layers = []
        current_dim = input_dim
        for _ in range(max(num_layers, 1)):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.cat_heads = nn.ModuleDict(
            {
                column: nn.Linear(hidden_dim, num_class)
                for column, num_class in zip(self.cat_cols, self.num_classes)
            }
        )
        self.text_heads = nn.ModuleDict(
            {
                column: nn.Linear(hidden_dim, dim)
                for column, dim in self.text_dims.items()
            }
        )

    def forward(
        self,
        cat_noisy: torch.Tensor,
        text_noisy: Dict[str, torch.Tensor],
        time_features: torch.Tensor,
        context_features: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        pieces = []
        if len(self.cat_cols) > 0:
            cat_pieces = []
            for column_index, embedding in enumerate(self.cat_embeddings):
                cat_pieces.append(embedding(cat_noisy[:, column_index].long()))
            pieces.append(torch.cat(cat_pieces, dim=1))

        for column in self.text_dims:
            pieces.append(F.silu(self.text_projections[column](text_noisy[column])))

        pieces.append(time_features.float())
        pieces.append(context_features.float())
        pieces.append(self.diffusion_time(diffusion_step.float()))
        hidden = self.trunk(torch.cat(pieces, dim=1))
        return {
            "cat_logits": {
                column: self.cat_heads[column](hidden) for column in self.cat_cols
            },
            "text_eps": {
                column: self.text_heads[column](hidden) for column in self.text_dims
            },
        }
