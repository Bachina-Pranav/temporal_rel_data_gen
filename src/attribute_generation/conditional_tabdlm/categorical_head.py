"""Training-only empirical-prior categorical output heads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def categorical_head_feature_enabled(raw: dict[str, Any]) -> bool:
    return bool(
        ((raw.get("categorical_heads") or {}).get("prior") or {}).get(
            "enabled",
            False,
        )
    )


def fit_categorical_head_metadata(
    config: Any,
    categorical_vocabs: dict[str, Any],
    *,
    train_frame: pd.DataFrame | None,
    train_dataset: Any | None,
) -> dict[str, Any]:
    """Fit categorical priors from training rows, never validation/test."""

    if not categorical_head_feature_enabled(config.raw):
        return {}
    raw = dict(config.raw.get("categorical_heads") or {})
    prior = dict(raw.get("prior") or {})
    configured = prior.get("columns")
    selected = (
        [str(column) for column in configured]
        if configured is not None
        else list(config.schema.categorical_targets)
    )
    invalid = sorted(
        set(selected).difference(config.schema.categorical_targets)
    )
    if invalid:
        raise ValueError(
            "Categorical prior columns must be generated categorical "
            f"targets (not auxiliary targets): {invalid}"
        )
    smoothing = float(prior.get("smoothing", 1.0))
    if smoothing < 0:
        raise ValueError("Categorical-prior smoothing must be nonnegative")
    columns: dict[str, Any] = {}
    for column in config.schema.model_categorical_targets:
        vocab = categorical_vocabs[column]
        enabled = column in selected
        counts = (
            categorical_training_counts(
                column,
                config,
                vocab,
                train_frame=train_frame,
                train_dataset=train_dataset,
            )
            if enabled
            else np.ones(int(vocab.size), dtype=np.float64)
        )
        smoothed = counts + smoothing * (counts > 0)
        probabilities = smoothed / max(
            float(smoothed.sum()),
            1e-12,
        )
        columns[column] = {
            "enabled": bool(enabled),
            "training_only": True,
            "counts": counts.astype(np.int64).tolist(),
            "probabilities": probabilities.tolist(),
            "smoothing": smoothing,
            "alpha": float(prior.get("alpha", 1.0)),
            "residual_weight": float(
                prior.get("residual_weight", 1.0)
            ),
            "residual_init_scale": float(
                prior.get("residual_init_scale", 1e-3)
            ),
            "epsilon": float(prior.get("epsilon", 1e-8)),
        }
    return {
        "version": 1,
        "training_only": True,
        "columns": columns,
    }


def categorical_training_counts(
    column: str,
    config: Any,
    vocab: Any,
    *,
    train_frame: pd.DataFrame | None,
    train_dataset: Any | None,
) -> np.ndarray:
    if train_frame is not None:
        if column not in train_frame:
            raise KeyError(
                f"Training table is missing categorical target {column!r}"
            )
        ids = np.asarray(
            [vocab.encode(value) for value in train_frame[column]],
            dtype=np.int64,
        )
    elif train_dataset is not None:
        index = config.schema.model_categorical_targets.index(column)
        indices = np.asarray(train_dataset.indices, dtype=np.int64)
        ids = np.asarray(
            train_dataset.categorical_ids[indices, index],
            dtype=np.int64,
        )
    else:
        raise ValueError(
            "Categorical-prior metadata requires training rows or a "
            "pretokenized training dataset"
        )
    return np.bincount(ids, minlength=int(vocab.size)).astype(np.float64)


class PriorAnchoredCategoricalHead(nn.Linear):
    """Linear residual plus a fixed empirical training-distribution prior."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        metadata: dict[str, Any],
    ):
        super().__init__(int(hidden_dim), int(output_dim))
        self.enabled = bool(metadata.get("enabled", False))
        self.alpha = float(metadata.get("alpha", 1.0))
        self.residual_weight = float(
            metadata.get("residual_weight", 1.0)
        )
        epsilon = float(metadata.get("epsilon", 1e-8))
        probabilities = torch.as_tensor(
            metadata.get("probabilities", [1.0] * int(output_dim)),
            dtype=torch.float32,
        )
        if probabilities.numel() != int(output_dim):
            raise ValueError(
                "Categorical-prior support size does not match the "
                f"vocabulary: {probabilities.numel()} != {output_dim}"
            )
        probabilities = probabilities / probabilities.sum().clamp_min(
            epsilon
        )
        log_probability = torch.where(
            probabilities > 0,
            torch.log(probabilities.clamp_min(epsilon)),
            torch.full_like(probabilities, -1.0e9),
        )
        self.register_buffer(
            "log_probability",
            log_probability,
            persistent=False,
        )
        if self.enabled:
            nn.init.normal_(
                self.weight,
                mean=0.0,
                std=max(
                    float(metadata.get("residual_init_scale", 1e-3)),
                    0.0,
                ),
            )
            nn.init.zeros_(self.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = super().forward(hidden)
        if not self.enabled:
            return residual
        return (
            self.alpha * self.log_probability.unsqueeze(0)
            + self.residual_weight * residual.float()
        )
