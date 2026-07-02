"""Checkpoint helpers for Text V1 temporal summary generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from .masked_summary_dataset import SimpleSummaryTokenizer
from .masked_text_diffusion import TemporalSummaryMaskedDiffusionV1
from .text_conditioning import ConditionFeatureNormalizer


METHOD_NAME_TEXT_V1 = "temporal_summary_text_v1"
METHOD_ALIAS_TEXT_V1 = "graph_conditioned_masked_summary_diffusion"


def save_text_v1_checkpoint(
    path: str | Path,
    model: TemporalSummaryMaskedDiffusionV1,
    tokenizer: SimpleSummaryTokenizer,
    normalizer: ConditionFeatureNormalizer,
    train_config: Dict[str, Any],
    history: list[Dict[str, Any]],
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Save model state without a training text bank or retrieval index."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "method": METHOD_NAME_TEXT_V1,
        "method_alias": METHOD_ALIAS_TEXT_V1,
        "model_state_dict": model.state_dict(),
        "model_config": model.to_config(),
        "tokenizer": tokenizer.to_dict(),
        "condition_normalizer": normalizer.to_dict(),
        "train_config": dict(train_config),
        "history": history,
        "epoch": int(epoch),
        "metrics": metrics,
        "contains_training_text_bank": False,
        "nearest_neighbor_decoder": False,
        "retrieval_augmented_generation": False,
        "text_column": train_config.get("text_col", "summary"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    torch.save(checkpoint, path)


def load_text_v1_checkpoint(path: str | Path, device: str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    tokenizer = SimpleSummaryTokenizer.from_dict(checkpoint["tokenizer"])
    normalizer = ConditionFeatureNormalizer.from_dict(checkpoint["condition_normalizer"])
    model = TemporalSummaryMaskedDiffusionV1.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "model": model,
        "tokenizer": tokenizer,
        "normalizer": normalizer,
    }


def write_text_v1_metadata(
    path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    seed: int,
    decoding_strategy: str,
    max_summary_tokens: int,
    conditioning_source: str = "synthetic_v3_nontext",
) -> None:
    metadata = {
        "method": METHOD_NAME_TEXT_V1,
        "method_alias": METHOD_ALIAS_TEXT_V1,
        "text_column": "summary",
        "no_nearest_neighbor_decoding": True,
        "no_text_retrieval": True,
        "contains_training_text_bank": False,
        "nearest_neighbor_decoder": False,
        "retrieval_augmented_generation": False,
        "conditions_on_generated_nontext": True,
        "conditioning_source": conditioning_source,
        "model_checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "seed": int(seed),
        "decoding_strategy": decoding_strategy,
        "max_summary_tokens": int(max_summary_tokens),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
