"""Schema-driven numerical transformations and losses for LSTM attributes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def numerical_field_metadata(config: Any, column: str) -> dict[str, Any]:
    fields = (config.raw.get("schema") or {}).get("fields") or {}
    meta = dict(fields.get(column) or {})
    numerical_cfg = (config.raw.get("numerical") or {}).get(column) or {}
    meta.update(numerical_cfg)
    semantic = str(meta.get("semantic_type", "continuous_numerical"))
    if semantic == "count_numerical":
        meta.setdefault("preprocessing", "log1p_standardize")
        meta.setdefault("output_distribution", "log1p_gaussian")
    else:
        meta.setdefault("preprocessing", "standardize")
        meta.setdefault("output_distribution", "gaussian")
    return meta


def fit_numerical_transformers(frame: pd.DataFrame, config: Any) -> dict[str, Any]:
    transformers: dict[str, Any] = {}
    for column in config.schema.numerical_targets:
        meta = numerical_field_metadata(config, column)
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if values.isna().any():
            raise ValueError(f"Numerical target {column!r} contains NaN values")
        preprocessing = str(meta.get("preprocessing", "standardize"))
        transformed = np.log1p(np.clip(values.to_numpy(dtype=float), 0.0, None)) if preprocessing.startswith("log1p") else values.to_numpy(dtype=float)
        mean = float(np.mean(transformed))
        std = float(np.std(transformed))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        transformers[column] = {
            "column": column,
            "semantic_type": meta.get("semantic_type", "continuous_numerical"),
            "preprocessing": preprocessing,
            "output_distribution": meta.get("output_distribution", "gaussian"),
            "mean": mean,
            "std": std,
            "min_train": float(np.min(values)),
            "max_train": float(np.max(values)),
            "clip_to_train_range": bool(meta.get("clip_to_train_range", True)),
        }
    return transformers


def transform_numerical_value(value: Any, metadata: dict[str, Any]) -> float:
    numeric = float(value)
    if str(metadata.get("preprocessing", "standardize")).startswith("log1p"):
        numeric = float(np.log1p(max(numeric, 0.0)))
    return float((numeric - float(metadata.get("mean", 0.0))) / max(float(metadata.get("std", 1.0)), 1e-8))


def inverse_transform_numerical(values: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    out = values.float() * float(metadata.get("std", 1.0)) + float(metadata.get("mean", 0.0))
    if str(metadata.get("preprocessing", "standardize")).startswith("log1p"):
        out = torch.expm1(out).clamp_min(0.0)
    if bool(metadata.get("clip_to_train_range", True)):
        out = out.clamp(float(metadata.get("min_train", -float("inf"))), float(metadata.get("max_train", float("inf"))))
    if metadata.get("semantic_type") == "count_numerical":
        out = torch.round(out).clamp_min(0.0)
    return out


def gaussian_nll_from_params(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean = params[:, 0]
    log_std = params[:, 1].clamp(min=-7.0, max=5.0)
    var = torch.exp(2.0 * log_std)
    return 0.5 * ((target.float() - mean) ** 2 / var + 2.0 * log_std + np.log(2.0 * np.pi))


def sample_gaussian_params(params: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    mean = params[:, 0]
    std = torch.exp(params[:, 1].clamp(min=-7.0, max=5.0)) * max(float(temperature), 1e-6)
    return mean + torch.randn_like(mean) * std
