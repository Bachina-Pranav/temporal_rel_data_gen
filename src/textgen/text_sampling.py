"""Iterative masked denoising sampler for Text V1 summaries."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .masked_summary_dataset import SimpleSummaryTokenizer


def sample_summaries(
    model,
    tokenizer: SimpleSummaryTokenizer,
    condition_features: np.ndarray,
    max_summary_tokens: int = 32,
    num_denoising_steps: int = 16,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> List[str]:
    """Generate summaries from all-mask initial states without retrieval."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = model.to(device)
    model.eval()
    features = np.asarray(condition_features, dtype=np.float32)
    outputs: List[str] = []
    forbidden = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.mask_token_id,
    ]
    for start in range(0, len(features), int(batch_size)):
        batch_features = torch.tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        generated = denoise_batch(
            model,
            tokenizer,
            batch_features,
            max_summary_tokens=max_summary_tokens,
            num_denoising_steps=num_denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden,
            device=device,
        )
        outputs.extend(tokenizer.decode_summary(row) or fallback_summary() for row in generated.detach().cpu().tolist())
    return outputs


def denoise_batch(
    model,
    tokenizer: SimpleSummaryTokenizer,
    condition_features: torch.Tensor,
    max_summary_tokens: int,
    num_denoising_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    forbidden_token_ids: Sequence[int],
    device: str,
) -> torch.Tensor:
    batch_size = condition_features.shape[0]
    input_ids = torch.full(
        (batch_size, int(max_summary_tokens)),
        tokenizer.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    masked = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    steps = max(int(num_denoising_steps), 1)
    for step in range(steps):
        with torch.no_grad():
            logits = model(input_ids, attention_mask, condition_features)["logits"]
            sampled, confidence = sample_tokens_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            )
        progress = 1.0 - math.cos(((step + 1) / steps) * math.pi / 2.0)
        target_unmasked = max(1, int(math.ceil(progress * int(max_summary_tokens))))
        current_unmasked = (~masked).sum(dim=1)
        for row in range(batch_size):
            need = int(max(target_unmasked - int(current_unmasked[row].item()), 0))
            candidates = torch.where(masked[row])[0]
            if need <= 0 or len(candidates) == 0:
                continue
            need = min(need, len(candidates))
            candidate_conf = confidence[row, candidates]
            selected = candidates[torch.topk(candidate_conf, k=need).indices]
            input_ids[row, selected] = sampled[row, selected]
            masked[row, selected] = False
    if bool(masked.any().detach().cpu()):
        with torch.no_grad():
            logits = model(input_ids, attention_mask, condition_features)["logits"]
            sampled, _ = sample_tokens_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            )
        input_ids = torch.where(masked, sampled, input_ids)
    return input_ids


def sample_tokens_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    forbidden_token_ids: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.clone()
    for token_id in forbidden_token_ids:
        if 0 <= int(token_id) < logits.shape[-1]:
            logits[..., int(token_id)] = -1e9
    if top_k and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        cutoff = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -1e9), logits)
    if top_p and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 0] = False
        filtered = torch.full_like(sorted_logits, -1e9)
        sorted_logits = torch.where(remove, filtered, sorted_logits)
        logits = torch.full_like(logits, -1e9).scatter(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    sums = probs.sum(dim=-1, keepdim=True)
    invalid = (~torch.isfinite(sums)) | (sums <= 1e-12)
    probs = probs / sums.clamp_min(1e-12)
    if bool(invalid.any().detach().cpu()):
        uniform = torch.ones_like(probs)
        for token_id in forbidden_token_ids:
            if 0 <= int(token_id) < uniform.shape[-1]:
                uniform[..., int(token_id)] = 0.0
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.where(invalid, uniform, probs)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).view(*probs.shape[:-1])
    confidence = probs.max(dim=-1).values
    return sampled, confidence


def fallback_summary() -> str:
    return "good product"
