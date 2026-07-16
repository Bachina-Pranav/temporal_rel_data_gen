"""Data preparation and PyTorch datasets for conditional TABDLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .numerical import fit_numerical_transformers, transform_numerical_value
from .tokenization import (
    CategoryVocab,
    SimpleTextTokenizer,
    normalize_text,
    stable_hash_bucket,
    summary_length_bucket_name,
)
from .utils import ensure_dir, load_json, read_dataframe, save_json, write_dataframe


@dataclass
class PreparedData:
    train_path: Path
    valid_path: Path
    test_path: Path
    schema_path: Path
    tokenizer_path: Path
    categorical_vocab_paths: dict[str, Path]
    numerical_metadata_path: Path | None = None


def prepare_rel_amazon_data(config: ConditionalTABDLMConfig) -> PreparedData:
    """Prepare the Rel-Amazon Exp1 table for conditional attribute generation."""

    output_dir = ensure_dir(config.data_dir)
    schema = config.schema
    frame = pd.read_csv(config.train_data_path)
    validate_columns(frame, schema)
    frame = normalize_frame(frame, schema)
    if schema.text_targets:
        for column in schema.text_targets:
            frame = frame[frame[column].map(normalize_text).str.len() > 0]
    frame = frame.dropna(subset=list(schema.condition_columns)).copy()
    sort_columns = list(schema.datetime_columns)
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    train, valid, test = split_prepared_frame(frame, schema)
    save_json(split_metadata(train, valid, test, schema), output_dir / "split_metadata.json")
    numerical_metadata = fit_numerical_transformers(train, config) if schema.numerical_targets else {}
    write_dataframe(train, output_dir / "train.parquet")
    write_dataframe(valid, output_dir / "valid.parquet")
    write_dataframe(test, output_dir / "test.parquet")

    save_json(schema.to_dict(), output_dir / "schema.json")
    save_json(numerical_metadata, output_dir / "numerical_metadata.json")
    token_cfg = config.raw.get("tokenizer", {})
    max_vocab_size = int(token_cfg.get("max_vocab_size", 30000))
    min_frequency = int(token_cfg.get("min_frequency", 1))
    tokenizer = SimpleTextTokenizer(lowercase=bool(token_cfg.get("lowercase", True)))
    if schema.text_targets:
        texts = []
        for column in schema.text_targets:
            texts.extend(train[column].tolist())
        tokenizer.fit(texts, max_vocab_size=max_vocab_size, min_frequency=min_frequency)

    vocab_paths: dict[str, Path] = {}
    for column in schema.model_categorical_targets:
        values = auxiliary_target_values(train, schema, tokenizer, column) if column in schema.auxiliary_categorical_targets else train[column]
        vocab = CategoryVocab.from_values(column, values)
        path = output_dir / f"vocab_{column}.json"
        save_json(vocab.to_dict(), path)
        vocab_paths[column] = path
    tokenizer_metadata = tokenizer.to_dict()
    tokenizer_metadata["text_max_lengths"] = dict(schema.text_max_lengths)
    for column in schema.text_targets:
        max_tokens = int(schema.text_max_lengths[column])
        tokenizer_metadata[f"{column}_max_tokens"] = max_tokens
        tokenizer_metadata[f"{column}_max_content_tokens"] = tokenizer.max_content_tokens(max_tokens)
        tokenizer_metadata[f"{column}_special_tokens"] = {
            "bos": f"{column.upper()}_BOS",
            "eos": f"{column.upper()}_EOS",
            "pad": f"{column.upper()}_PAD",
            "mask": f"{column.upper()}_MASK",
        }
    save_json(tokenizer_metadata, output_dir / "tokenizer_metadata.json")
    auto_meta = config.raw.get("_auto_text_length_metadata", {}).get("review_text")
    if auto_meta:
        metadata_dir = ensure_dir(config.output_dir / "metadata")
        save_json(auto_meta, metadata_dir / "review_text_length_stats.json")

    return PreparedData(
        train_path=output_dir / "train.parquet",
        valid_path=output_dir / "valid.parquet",
        test_path=output_dir / "test.parquet",
        schema_path=output_dir / "schema.json",
        tokenizer_path=output_dir / "tokenizer_metadata.json",
        categorical_vocab_paths=vocab_paths,
        numerical_metadata_path=output_dir / "numerical_metadata.json" if schema.numerical_targets else None,
    )


def validate_columns(frame: pd.DataFrame, schema: ConditionalTABDLMSchema) -> None:
    missing = [column for column in schema.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_frame(frame: pd.DataFrame, schema: ConditionalTABDLMSchema) -> pd.DataFrame:
    frame = frame.copy()
    for column in schema.foreign_key_columns:
        frame[column] = frame[column].astype(str)
    for column in schema.datetime_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in schema.categorical_targets:
        frame[column] = frame[column].astype(str)
    for column in schema.numerical_targets:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in schema.text_targets:
        frame[column] = frame[column].map(normalize_text)
    return frame.dropna(subset=list(schema.datetime_columns) + list(schema.numerical_targets)).reset_index(drop=True)


def time_aware_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * 0.90)
    valid_end = int(n * 0.95)
    return (
        frame.iloc[:train_end].reset_index(drop=True),
        frame.iloc[train_end:valid_end].reset_index(drop=True),
        frame.iloc[valid_end:].reset_index(drop=True),
    )


def split_prepared_frame(frame: pd.DataFrame, schema: ConditionalTABDLMSchema) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use an explicit preprocessing split when present, otherwise keep legacy behavior."""

    if "split" not in frame.columns:
        return time_aware_split(frame)
    labels = frame["split"].astype(str).str.strip().str.lower()
    aliases = {
        "train": "train",
        "training": "train",
        "valid": "validation",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    normalized = labels.map(aliases)
    unknown = sorted(set(labels[normalized.isna()].tolist()))
    if unknown:
        raise ValueError(f"Unknown split labels in train_data_path: {unknown}")
    frame = frame.assign(split=normalized)
    splits = []
    for split_name in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f"Explicit split column has no rows for {split_name!r}")
        splits.append(sort_for_training(split_frame, schema))
    return tuple(splits)  # type: ignore[return-value]


def sort_for_training(frame: pd.DataFrame, schema: ConditionalTABDLMSchema) -> pd.DataFrame:
    sort_columns = [column for column in schema.datetime_columns if column in frame.columns]
    if "event_id" in frame.columns:
        sort_columns.append("event_id")
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def split_metadata(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
) -> dict[str, Any]:
    split_frames = {"train": train, "validation": valid, "test": test}
    metadata: dict[str, Any] = {
        "split_source": "explicit_split_column" if "split" in train.columns else "legacy_time_aware_90_5_5",
        "row_counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "timestamp_bounds": {},
        "cold_start_foreign_keys": {},
    }
    timestamp_col = schema.datetime_columns[0] if schema.datetime_columns else None
    if timestamp_col is not None:
        for name, frame in split_frames.items():
            timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
            metadata["timestamp_bounds"][name] = {
                "min": timestamps.min().isoformat() if len(timestamps) else None,
                "max": timestamps.max().isoformat() if len(timestamps) else None,
            }
    for column in schema.foreign_key_columns:
        train_values = set(train[column].astype(str)) if column in train else set()
        column_meta: dict[str, Any] = {}
        for name, frame in [("validation", valid), ("test", test)]:
            values = set(frame[column].astype(str)) if column in frame else set()
            new_values = sorted(values.difference(train_values))
            column_meta[name] = {
                "num_distinct": int(len(values)),
                "num_first_seen_after_train": int(len(new_values)),
                "examples_first_seen_after_train": new_values[:20],
            }
        metadata["cold_start_foreign_keys"][column] = column_meta
    return metadata


def load_prepared_tables(config: ConditionalTABDLMConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = config.data_dir
    paths = [data_dir / "train.parquet", data_dir / "valid.parquet", data_dir / "test.parquet"]
    if not all(path.exists() for path in paths):
        prepare_rel_amazon_data(config)
    return tuple(read_dataframe(path) for path in paths)  # type: ignore[return-value]


def load_category_vocabs(config: ConditionalTABDLMConfig) -> dict[str, CategoryVocab]:
    vocabs: dict[str, CategoryVocab] = {}
    for column in config.schema.model_categorical_targets:
        path = config.data_dir / f"vocab_{column}.json"
        vocabs[column] = CategoryVocab.from_dict(load_json(path))
    return vocabs


def load_text_tokenizer(config: ConditionalTABDLMConfig) -> SimpleTextTokenizer:
    return SimpleTextTokenizer.from_dict(load_json(config.data_dir / "tokenizer_metadata.json"))


def load_numerical_metadata(config: ConditionalTABDLMConfig) -> dict[str, Any]:
    path = config.data_dir / "numerical_metadata.json"
    if not path.exists():
        return {}
    return load_json(path)


class ConditionalTABDLMDataset(Dataset):
    """Rows encoded as fixed condition tokens plus clean target tokens."""

    def __init__(
        self,
        frame: pd.DataFrame,
        schema: ConditionalTABDLMSchema,
        categorical_vocabs: dict[str, CategoryVocab],
        text_tokenizer: SimpleTextTokenizer,
        num_hash_buckets: int,
        numerical_metadata: dict[str, Any] | None = None,
    ):
        self.frame = normalize_frame(frame, schema).reset_index(drop=True)
        self.schema = schema
        self.categorical_vocabs = categorical_vocabs
        self.text_tokenizer = text_tokenizer
        self.num_hash_buckets = int(num_hash_buckets)
        self.numerical_metadata = dict(numerical_metadata or {})

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        foreign_keys = [
            stable_hash_bucket(column, row[column], self.num_hash_buckets)
            for column in self.schema.foreign_key_columns
        ]
        datetimes = [
            pd.Timestamp(row[column]).timestamp()
            for column in self.schema.datetime_columns
        ]
        categorical = []
        for column in self.schema.model_categorical_targets:
            value = self.auxiliary_value(row, column) if column in self.schema.auxiliary_categorical_targets else row[column]
            categorical.append(self.categorical_vocabs[column].encode(value))
        numerical = [
            transform_numerical_value(row[column], self.numerical_metadata.get(column, {}))
            for column in self.schema.numerical_targets
        ]
        text_ids: dict[str, list[int]] = {}
        text_attention: dict[str, list[int]] = {}
        for column in self.schema.text_targets:
            ids, attention = self.text_tokenizer.encode(row[column], self.schema.text_max_lengths[column])
            text_ids[column] = ids
            text_attention[column] = attention
        return {
            "foreign_key_ids": torch.tensor(foreign_keys, dtype=torch.long),
            "datetime_values": torch.tensor(datetimes, dtype=torch.float32),
            "categorical_ids": torch.tensor(categorical, dtype=torch.long),
            "numerical_values": torch.tensor(numerical, dtype=torch.float32),
            "text_ids": {column: torch.tensor(ids, dtype=torch.long) for column, ids in text_ids.items()},
            "text_attention": {column: torch.tensor(att, dtype=torch.long) for column, att in text_attention.items()},
            "row_id": torch.tensor(int(index), dtype=torch.long),
        }

    def auxiliary_value(self, row: pd.Series, column: str) -> str:
        if column not in {"summary_length_bucket", "review_text_length_bucket"}:
            raise KeyError(f"Unsupported auxiliary categorical target: {column}")
        if not self.schema.text_targets:
            return "len_0"
        text_col = self.schema.text_column_for_length_bucket(column)
        max_tokens = self.schema.text_max_lengths[text_col]
        buckets = self.schema.buckets_for_length_bucket(column)
        ids, _ = self.text_tokenizer.encode(row[text_col], max_tokens)
        content_length = self.text_tokenizer.content_length(ids)
        return summary_length_bucket_name(content_length, buckets)


def make_collate_fn(
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    min_mask_prob: float,
    max_mask_prob: float,
    mask_schedule: str = "linear",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_and_mask(
            samples,
            schema=schema,
            categorical_vocabs=categorical_vocabs,
            text_tokenizer=text_tokenizer,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            mask_schedule=mask_schedule,
        )

    return collate


def collate_and_mask(
    samples: list[dict[str, Any]],
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    min_mask_prob: float,
    max_mask_prob: float,
    mask_schedule: str = "linear",
) -> dict[str, Any]:
    foreign_key_ids = torch.stack([sample["foreign_key_ids"] for sample in samples], dim=0)
    datetime_values = torch.stack([sample["datetime_values"] for sample in samples], dim=0)
    categorical_clean = torch.stack([sample["categorical_ids"] for sample in samples], dim=0)
    numerical_values = torch.stack([sample["numerical_values"] for sample in samples], dim=0)
    batch_size = categorical_clean.shape[0]
    timesteps = torch.rand(batch_size, dtype=torch.float32)
    rates = mask_probability(timesteps, min_mask_prob, max_mask_prob, mask_schedule)

    categorical_input = categorical_clean.clone()
    categorical_labels = torch.full_like(categorical_clean, -100)
    categorical_mask = torch.rand(categorical_clean.shape) < rates.view(-1, 1)
    for col_idx, column in enumerate(schema.model_categorical_targets):
        categorical_input[categorical_mask[:, col_idx], col_idx] = categorical_vocabs[column].mask_id
        categorical_labels[categorical_mask[:, col_idx], col_idx] = categorical_clean[categorical_mask[:, col_idx], col_idx]

    text_input: dict[str, torch.Tensor] = {}
    text_labels: dict[str, torch.Tensor] = {}
    text_attention: dict[str, torch.Tensor] = {}
    masked_any = categorical_mask.any(dim=1)
    for column in schema.text_targets:
        clean = torch.stack([sample["text_ids"][column] for sample in samples], dim=0)
        attention = torch.stack([sample["text_attention"][column] for sample in samples], dim=0)
        candidate = attention.bool()
        if candidate.shape[1] > 0:
            candidate[:, 0] = False
        mask = (torch.rand(clean.shape) < rates.view(-1, 1)) & candidate
        noisy = clean.clone()
        labels = torch.full_like(clean, -100)
        noisy[mask] = text_tokenizer.mask_id
        labels[mask] = clean[mask]
        text_input[column] = noisy
        text_labels[column] = labels
        text_attention[column] = attention
        masked_any |= mask.any(dim=1)

    for row_idx in torch.where(~masked_any)[0].tolist():
        if len(schema.model_categorical_targets) > 0:
            col_idx = int(torch.randint(0, len(schema.model_categorical_targets), (1,)).item())
            column = schema.model_categorical_targets[col_idx]
            categorical_input[row_idx, col_idx] = categorical_vocabs[column].mask_id
            categorical_labels[row_idx, col_idx] = categorical_clean[row_idx, col_idx]
        else:
            column = schema.text_targets[0]
            candidates = torch.where(text_attention[column][row_idx].bool())[0]
            candidates = candidates[candidates != 0]
            pos = int(candidates[torch.randint(0, len(candidates), (1,)).item()].item())
            text_input[column][row_idx, pos] = text_tokenizer.mask_id
            text_labels[column][row_idx, pos] = torch.stack([sample["text_ids"][column] for sample in samples], dim=0)[row_idx, pos]

    return {
        "foreign_key_ids": foreign_key_ids,
        "datetime_values": datetime_values,
        "categorical_input_ids": categorical_input,
        "categorical_clean_ids": categorical_clean,
        "categorical_labels": categorical_labels,
        "numerical_values": numerical_values,
        "text_input_ids": text_input,
        "text_clean_ids": {
            column: torch.stack([sample["text_ids"][column] for sample in samples], dim=0)
            for column in schema.text_targets
        },
        "text_labels": text_labels,
        "text_attention": text_attention,
        "diffusion_t": timesteps,
        "row_id": torch.stack([sample["row_id"] for sample in samples], dim=0),
    }


def mask_probability(
    timesteps: torch.Tensor,
    min_mask_prob: float,
    max_mask_prob: float,
    mask_schedule: str,
) -> torch.Tensor:
    min_p = float(min_mask_prob)
    max_p = float(max_mask_prob)
    if mask_schedule == "cosine":
        curve = 1.0 - torch.cos(timesteps * torch.pi / 2.0)
    else:
        curve = timesteps
    return torch.clamp(min_p + curve * (max_p - min_p), 0.0, 1.0)


def auxiliary_target_values(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    tokenizer: SimpleTextTokenizer,
    column: str,
) -> list[str]:
    if column not in {"summary_length_bucket", "review_text_length_bucket"}:
        raise KeyError(f"Unsupported auxiliary categorical target: {column}")
    if not schema.text_targets:
        return ["len_0"] * len(frame)
    text_col = schema.text_column_for_length_bucket(column)
    max_tokens = schema.text_max_lengths[text_col]
    buckets = schema.buckets_for_length_bucket(column)
    values = []
    for text in frame[text_col]:
        ids, _ = tokenizer.encode(text, max_tokens)
        values.append(summary_length_bucket_name(tokenizer.content_length(ids), buckets))
    return values
