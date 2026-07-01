"""Diffusion schedules for non-text review attributes."""

from __future__ import annotations

import math

import torch


def mask_probability(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 1.0 - torch.cos(t * math.pi / 2.0)
    raise ValueError("schedule must be 'linear' or 'cosine'.")


def gaussian_sigma(t: torch.Tensor, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return sigma_min * (sigma_max / sigma_min) ** t


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return t.new_zeros((len(t), 0))
    half = dim // 2
    if half == 0:
        return t[:, None]
    frequencies = torch.exp(
        torch.linspace(0, -math.log(10_000), half, device=t.device)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding.shape[1] < dim:
        embedding = torch.cat([embedding, t[:, None]], dim=1)
    return embedding[:, :dim]
