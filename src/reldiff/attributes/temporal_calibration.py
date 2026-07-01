"""Temporal calibration helpers for V3 sampling."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .entity_latent_effects import logit


def calibrate_rating_logits_np(
    logits: np.ndarray,
    target_distribution: np.ndarray,
    strength: float = 0.75,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = softmax_np(logits)
    expected = probs.mean(axis=0)
    target = np.asarray(target_distribution, dtype=float)
    target = target / max(target.sum(), eps)
    correction = np.log(target + eps) - np.log(expected + eps)
    return logits + float(strength) * correction[None, :], correction


def calibrate_verified_logits_np(
    logits: np.ndarray,
    target_rate: float,
    strength: float = 0.75,
) -> Tuple[np.ndarray, float]:
    probs = softmax_np(logits)[:, 1]
    expected = float(np.mean(probs))
    correction = logit(float(target_rate)) - logit(expected)
    calibrated = logits.copy()
    calibrated[:, 1] += float(strength) * correction
    calibrated[:, 0] -= float(strength) * correction
    return calibrated, float(correction)


def calibrate_logits_torch(
    rating_logits: torch.Tensor,
    verified_logits: torch.Tensor,
    target_rating_distribution: np.ndarray,
    target_verified_rate: float,
    strength: float = 0.75,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = rating_logits.device
    rating_probs = torch.softmax(rating_logits, dim=1)
    expected = rating_probs.mean(dim=0).clamp_min(eps)
    target = torch.tensor(target_rating_distribution, dtype=rating_logits.dtype, device=device)
    target = target / target.sum().clamp_min(eps)
    correction = torch.log(target.clamp_min(eps)) - torch.log(expected)
    calibrated_rating = rating_logits + float(strength) * correction.view(1, -1)

    verified_probs = torch.softmax(verified_logits, dim=1)[:, 1]
    expected_rate = verified_probs.mean().clamp(eps, 1.0 - eps)
    target_rate = torch.tensor(float(target_verified_rate), dtype=verified_logits.dtype, device=device).clamp(eps, 1.0 - eps)
    verified_correction = torch.log(target_rate / (1.0 - target_rate)) - torch.log(expected_rate / (1.0 - expected_rate))
    calibrated_verified = verified_logits.clone()
    calibrated_verified[:, 1] += float(strength) * verified_correction
    calibrated_verified[:, 0] -= float(strength) * verified_correction
    norm = float(torch.norm(correction.detach()).cpu() + torch.abs(verified_correction.detach()).cpu())
    return calibrated_rating, calibrated_verified, norm


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
