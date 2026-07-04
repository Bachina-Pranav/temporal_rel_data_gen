"""Noised attribute-state helpers for temporal attribute-denoising graph conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import auxiliary_target_values, normalize_frame
from .graph_schema import attribute_denoising_config
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer


@dataclass
class GraphAttributeStore:
    """Encoded clean or generated review attributes indexed by graph row id."""

    categorical_ids: np.ndarray
    text_ids: dict[str, np.ndarray]
    schema: ConditionalTABDLMSchema
    categorical_vocabs: dict[str, CategoryVocab]
    text_tokenizer: SimpleTextTokenizer

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        config: ConditionalTABDLMConfig,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
    ) -> "GraphAttributeStore":
        frame = normalize_frame(frame, config.schema)
        categorical = np.zeros((len(frame), len(config.schema.model_categorical_targets)), dtype=np.int64)
        for idx, column in enumerate(config.schema.model_categorical_targets):
            if column in config.schema.auxiliary_categorical_targets:
                values = auxiliary_target_values(frame, config.schema, text_tokenizer, column)
            else:
                values = frame[column]
            categorical[:, idx] = [categorical_vocabs[column].encode(value) for value in values]
        text_ids: dict[str, np.ndarray] = {}
        for column in config.schema.text_targets:
            rows = [text_tokenizer.encode(value, config.schema.text_max_lengths[column])[0] for value in frame[column]]
            text_ids[column] = np.asarray(rows, dtype=np.int64)
        return cls(
            categorical_ids=categorical,
            text_ids=text_ids,
            schema=config.schema,
            categorical_vocabs=categorical_vocabs,
            text_tokenizer=text_tokenizer,
        )

    @classmethod
    def empty_generated(
        cls,
        num_rows: int,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
    ) -> "GraphAttributeStore":
        categorical = np.zeros((int(num_rows), len(schema.model_categorical_targets)), dtype=np.int64)
        for idx, column in enumerate(schema.model_categorical_targets):
            categorical[:, idx] = categorical_vocabs[column].mask_id
        text_ids: dict[str, np.ndarray] = {}
        for column in schema.text_targets:
            length = int(schema.text_max_lengths[column])
            arr = np.full((int(num_rows), length), text_tokenizer.mask_id, dtype=np.int64)
            arr[:, 0] = text_tokenizer.bos_id
            text_ids[column] = arr
        return cls(
            categorical_ids=categorical,
            text_ids=text_ids,
            schema=schema,
            categorical_vocabs=categorical_vocabs,
            text_tokenizer=text_tokenizer,
        )

    def update_rows(
        self,
        row_indices: list[int],
        categorical_ids: torch.Tensor,
        text_ids: dict[str, torch.Tensor],
    ) -> None:
        rows = [int(idx) for idx in row_indices]
        self.categorical_ids[rows, :] = categorical_ids.detach().cpu().numpy().astype(np.int64)
        for column in self.schema.text_targets:
            self.text_ids[column][rows, :] = text_ids[column].detach().cpu().numpy().astype(np.int64)


def build_attribute_graph_batch(
    graph_batch: dict[str, torch.Tensor],
    batch: dict[str, Any],
    store: GraphAttributeStore,
    config: ConditionalTABDLMConfig,
    *,
    device: str | torch.device,
    training: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    attr_cfg = attribute_denoising_config(config.raw)
    history_cfg = attr_cfg["history_attribute_corruption"] if training else attr_cfg["sampling_history_attribute_corruption"]
    mask_prob = float(history_cfg.get("mask_prob", 0.15 if training else 0.0)) if bool(history_cfg.get("enabled", True)) else 0.0
    attr_batch: dict[str, torch.Tensor] = {
        "target_categorical_ids": batch["categorical_input_ids"].to(device),
    }
    for column in config.schema.text_targets:
        attr_batch[f"target_text_ids_{column}"] = batch["text_input_ids"][column].to(device)

    mask_stats = []
    target_stats = [target_mask_rate(batch, store)]
    for kind in ("customer", "product"):
        rows = graph_batch[f"{kind}_history_row_index"].detach().cpu().numpy()
        mask = graph_batch[f"{kind}_history_mask"].to(device)
        clean_cat = gather_categorical(store, rows, device=device)
        noised_cat, cat_rate = corrupt_categorical_history(clean_cat, mask, store, mask_prob=mask_prob)
        attr_batch[f"{kind}_history_categorical_ids"] = noised_cat
        attr_batch[f"{kind}_history_clean_categorical_ids"] = clean_cat
        text_rates = []
        for column in config.schema.text_targets:
            clean_text = gather_text(store, rows, column, device=device)
            noised_text, rate = corrupt_text_history(clean_text, mask, store.text_tokenizer, mask_prob=mask_prob)
            attr_batch[f"{kind}_history_text_ids_{column}"] = noised_text
            attr_batch[f"{kind}_history_clean_text_ids_{column}"] = clean_text
            text_rates.append(rate)
        mask_stats.append(cat_rate)
        mask_stats.extend(text_rates)

    diagnostics = {
        "history_attr_mask_rate": float(np.mean([rate for rate in mask_stats if rate is not None])) if mask_stats else 0.0,
        "target_attr_mask_rate": float(np.mean(target_stats)) if target_stats else 0.0,
    }
    return attr_batch, diagnostics


def gather_categorical(store: GraphAttributeStore, rows: np.ndarray, *, device: str | torch.device) -> torch.Tensor:
    safe_rows = np.where(rows >= 0, rows, 0)
    values = store.categorical_ids[safe_rows]
    values = torch.tensor(values, dtype=torch.long, device=device)
    if rows.size:
        invalid = torch.tensor(rows < 0, dtype=torch.bool, device=device)
        for idx, column in enumerate(store.schema.model_categorical_targets):
            values[:, :, idx] = torch.where(
                invalid,
                torch.full_like(values[:, :, idx], store.categorical_vocabs[column].mask_id),
                values[:, :, idx],
            )
    return values


def gather_text(
    store: GraphAttributeStore,
    rows: np.ndarray,
    column: str,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    safe_rows = np.where(rows >= 0, rows, 0)
    values = torch.tensor(store.text_ids[column][safe_rows], dtype=torch.long, device=device)
    if rows.size:
        invalid = torch.tensor(rows < 0, dtype=torch.bool, device=device).unsqueeze(-1)
        masked = torch.full_like(values, store.text_tokenizer.mask_id)
        if values.shape[-1] > 0:
            masked[:, :, 0] = store.text_tokenizer.bos_id
        values = torch.where(invalid, masked, values)
    return values


def corrupt_categorical_history(
    values: torch.Tensor,
    mask: torch.Tensor,
    store: GraphAttributeStore,
    *,
    mask_prob: float,
) -> tuple[torch.Tensor, float]:
    if mask_prob <= 0:
        return values.clone(), 0.0
    out = values.clone()
    valid = mask.unsqueeze(-1).expand_as(out).clone()
    random_mask = (torch.rand(out.shape, device=out.device) < float(mask_prob)) & valid
    for idx, column in enumerate(store.schema.model_categorical_targets):
        out[:, :, idx] = torch.where(random_mask[:, :, idx], torch.full_like(out[:, :, idx], store.categorical_vocabs[column].mask_id), out[:, :, idx])
    denom = int(valid.sum().detach().cpu())
    return out, float(random_mask.sum().detach().cpu() / max(denom, 1))


def corrupt_text_history(
    values: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    *,
    mask_prob: float,
) -> tuple[torch.Tensor, float]:
    if mask_prob <= 0:
        return values.clone(), 0.0
    out = values.clone()
    valid = mask.unsqueeze(-1).expand_as(out).clone()
    if out.shape[-1] > 0:
        valid[:, :, 0] = False
    random_mask = (torch.rand(out.shape, device=out.device) < float(mask_prob)) & valid
    out = torch.where(random_mask, torch.full_like(out, tokenizer.mask_id), out)
    denom = int(valid.sum().detach().cpu())
    return out, float(random_mask.sum().detach().cpu() / max(denom, 1))


def target_mask_rate(batch: dict[str, Any], store: GraphAttributeStore) -> float:
    rates = []
    cat = batch["categorical_input_ids"]
    for idx, column in enumerate(store.schema.model_categorical_targets):
        rates.append(float((cat[:, idx] == store.categorical_vocabs[column].mask_id).float().mean().detach().cpu()))
    for column in store.schema.text_targets:
        values = batch["text_input_ids"][column]
        candidate = torch.ones_like(values, dtype=torch.bool)
        if values.shape[1] > 0:
            candidate[:, 0] = False
        denom = max(int(candidate.sum().detach().cpu()), 1)
        rates.append(float(((values == store.text_tokenizer.mask_id) & candidate).sum().detach().cpu() / denom))
    return float(np.mean(rates)) if rates else 0.0
