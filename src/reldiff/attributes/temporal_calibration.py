"""Temporal calibration helpers for V3 sampling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .entity_latent_effects import logit


def js_divergence_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    return float(
        0.5 * np.sum(p[p_mask] * np.log2(p[p_mask] / np.maximum(m[p_mask], eps)))
        + 0.5 * np.sum(q[q_mask] * np.log2(q[q_mask] / np.maximum(m[q_mask], eps)))
    )


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


def calibration_group_stats_torch(
    group: Any,
    rating_logits_pre: torch.Tensor,
    rating_logits_post: torch.Tensor,
    verified_logits_pre: torch.Tensor,
    verified_logits_post: torch.Tensor,
    target_rating_distribution: np.ndarray,
    target_verified_rate: float,
    strength: float,
    rating_values: list[Any],
) -> Dict[str, Any]:
    """Summarize one temporal calibration group."""

    pre_rating = torch.softmax(rating_logits_pre, dim=1).mean(dim=0).detach().cpu().numpy()
    post_rating = torch.softmax(rating_logits_post, dim=1).mean(dim=0).detach().cpu().numpy()
    target_rating = np.asarray(target_rating_distribution, dtype=float)
    target_rating = target_rating / max(target_rating.sum(), 1e-12)
    pre_verified = float(torch.softmax(verified_logits_pre, dim=1)[:, 1].mean().detach().cpu())
    post_verified = float(torch.softmax(verified_logits_post, dim=1)[:, 1].mean().detach().cpu())
    verified_correction = logit(float(target_verified_rate)) - logit(pre_verified)
    row: Dict[str, Any] = {
        "group": str(group),
        "num_rows": int(rating_logits_pre.shape[0]),
        "rating_correction_norm": float(
            np.linalg.norm(np.log(target_rating + 1e-8) - np.log(pre_rating + 1e-8))
        ),
        "target_verified_rate": float(target_verified_rate),
        "expected_precal_verified_rate": pre_verified,
        "expected_postcal_verified_rate": post_verified,
        "verified_logit_correction": float(verified_correction),
        "calibration_strength": float(strength),
        "precal_rating_target_js": js_divergence_probs(pre_rating, target_rating),
        "postcal_rating_target_js": js_divergence_probs(post_rating, target_rating),
        "precal_verified_target_abs_error": float(abs(pre_verified - target_verified_rate)),
        "postcal_verified_target_abs_error": float(abs(post_verified - target_verified_rate)),
    }
    for idx, value in enumerate(rating_values):
        suffix = str(value)
        row[f"target_rating_p_{suffix}"] = float(target_rating[idx])
        row[f"expected_precal_rating_p_{suffix}"] = float(pre_rating[idx])
        row[f"expected_postcal_rating_p_{suffix}"] = float(post_rating[idx])
    return row


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
