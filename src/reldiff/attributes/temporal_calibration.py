"""Temporal calibration helpers for V3 sampling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .entity_latent_effects import logit


LOGIT_CLAMP = 30.0


def sanitize_logits_np(logits: np.ndarray, clamp: float = LOGIT_CLAMP) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return np.clip(np.nan_to_num(logits, nan=0.0, posinf=clamp, neginf=-clamp), -clamp, clamp)


def sanitize_logits_torch(logits: torch.Tensor, clamp: float = LOGIT_CLAMP) -> torch.Tensor:
    logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    return torch.nan_to_num(logits, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


def normalize_probability_vector(values: np.ndarray, size: int | None = None, eps: float = 1e-8) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    if size is not None and probs.shape != (size,):
        probs = np.ones(size, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= eps:
        return np.ones_like(probs, dtype=float) / max(int(probs.size), 1)
    return probs / total


def finite_probability(value: float, default: float = 0.5, eps: float = 1e-8) -> float:
    value = float(value)
    if not np.isfinite(value):
        value = float(default)
    return float(np.clip(value, eps, 1.0 - eps))


def safe_softmax_probs_torch(logits: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    logits = sanitize_logits_torch(logits)
    probs = torch.softmax(logits, dim=dim)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    sums = probs.sum(dim=dim, keepdim=True)
    invalid = (~torch.isfinite(sums)) | (sums <= eps)
    probs = probs / sums.clamp_min(eps)
    if bool(invalid.any().detach().cpu()):
        num_classes = max(int(logits.shape[dim]), 1)
        probs = torch.where(invalid, torch.full_like(probs, 1.0 / num_classes), probs)
    return probs


def _normalize_probability_tensor(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = probs.sum()
    if not bool(torch.isfinite(total).detach().cpu()) or float(total.detach().cpu()) <= eps:
        return torch.full_like(probs, 1.0 / max(int(probs.numel()), 1))
    return probs / total.clamp_min(eps)


def js_divergence_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize_probability_vector(p, eps=eps)
    q = normalize_probability_vector(q, size=p.size, eps=eps)
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
    logits = sanitize_logits_np(logits)
    probs = softmax_np(logits)
    expected = normalize_probability_vector(probs.mean(axis=0), size=logits.shape[1], eps=eps)
    target = normalize_probability_vector(target_distribution, size=logits.shape[1], eps=eps)
    correction = sanitize_logits_np(np.log(np.clip(target, eps, None)) - np.log(np.clip(expected, eps, None)))
    return sanitize_logits_np(logits + float(strength) * correction[None, :]), correction


def calibrate_verified_logits_np(
    logits: np.ndarray,
    target_rate: float,
    strength: float = 0.75,
) -> Tuple[np.ndarray, float]:
    logits = sanitize_logits_np(logits)
    probs = softmax_np(logits)[:, 1]
    expected = finite_probability(float(np.mean(probs)))
    target_rate = finite_probability(target_rate)
    correction = float(np.clip(np.nan_to_num(logit(target_rate) - logit(expected), nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP), -LOGIT_CLAMP, LOGIT_CLAMP))
    calibrated = logits.copy()
    calibrated[:, 1] += float(strength) * correction
    calibrated[:, 0] -= float(strength) * correction
    return sanitize_logits_np(calibrated), float(correction)


def calibrate_logits_torch(
    rating_logits: torch.Tensor,
    verified_logits: torch.Tensor,
    target_rating_distribution: np.ndarray,
    target_verified_rate: float,
    strength: float = 0.75,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = rating_logits.device
    rating_logits = sanitize_logits_torch(rating_logits)
    verified_logits = sanitize_logits_torch(verified_logits)
    rating_probs = safe_softmax_probs_torch(rating_logits, dim=1)
    if rating_probs.shape[0] == 0:
        expected = torch.full((rating_logits.shape[1],), 1.0 / max(int(rating_logits.shape[1]), 1), dtype=rating_logits.dtype, device=device)
    else:
        expected = _normalize_probability_tensor(rating_probs.mean(dim=0), eps=eps)
    target_np = normalize_probability_vector(target_rating_distribution, size=int(rating_logits.shape[1]), eps=eps)
    target = torch.tensor(target_np, dtype=rating_logits.dtype, device=device)
    correction = torch.log(target.clamp_min(eps)) - torch.log(expected)
    correction = sanitize_logits_torch(correction)
    calibrated_rating = sanitize_logits_torch(rating_logits + float(strength) * correction.view(1, -1))

    verified_probs = safe_softmax_probs_torch(verified_logits, dim=1)[:, 1]
    if verified_probs.shape[0] == 0:
        expected_rate = torch.tensor(0.5, dtype=verified_logits.dtype, device=device)
    else:
        expected_rate = torch.nan_to_num(verified_probs.mean(), nan=0.5, posinf=0.5, neginf=0.5).clamp(eps, 1.0 - eps)
    target_rate = torch.tensor(finite_probability(target_verified_rate, eps=eps), dtype=verified_logits.dtype, device=device)
    verified_correction = torch.log(target_rate / (1.0 - target_rate)) - torch.log(expected_rate / (1.0 - expected_rate))
    verified_correction = sanitize_logits_torch(verified_correction)
    calibrated_verified = verified_logits.clone()
    calibrated_verified[:, 1] += float(strength) * verified_correction
    calibrated_verified[:, 0] -= float(strength) * verified_correction
    calibrated_verified = sanitize_logits_torch(calibrated_verified)
    norm_tensor = torch.norm(correction.detach()) + torch.abs(verified_correction.detach())
    norm = float(torch.nan_to_num(norm_tensor, nan=0.0, posinf=LOGIT_CLAMP, neginf=0.0).cpu())
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

    pre_rating = safe_softmax_probs_torch(rating_logits_pre, dim=1).mean(dim=0).detach().cpu().numpy()
    post_rating = safe_softmax_probs_torch(rating_logits_post, dim=1).mean(dim=0).detach().cpu().numpy()
    target_rating = normalize_probability_vector(target_rating_distribution, size=len(rating_values), eps=1e-12)
    pre_verified = finite_probability(float(safe_softmax_probs_torch(verified_logits_pre, dim=1)[:, 1].mean().detach().cpu()))
    post_verified = finite_probability(float(safe_softmax_probs_torch(verified_logits_post, dim=1)[:, 1].mean().detach().cpu()))
    target_verified_rate = finite_probability(target_verified_rate)
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
    logits = sanitize_logits_np(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
    return np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
