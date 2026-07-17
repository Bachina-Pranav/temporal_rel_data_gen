"""Pretokenized array datasets for scalable LSTM attribute training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer
from .utils import load_json


@dataclass
class PretokenizedBundle:
    root: Path
    metadata: dict[str, Any]
    schema: ConditionalTABDLMSchema
    categorical_vocabs: dict[str, CategoryVocab]
    tokenizer: SimpleTextTokenizer
    numerical_metadata: dict[str, Any]


class PretokenizedLSTMDataset(Dataset):
    """Dataset backed by memmaps/numpy arrays created before training."""

    def __init__(self, bundle: PretokenizedBundle, split: str):
        self.bundle = bundle
        self.root = bundle.root
        self.schema = bundle.schema
        self.split = str(split)
        split_path = self.root / f"{self.split}_indices.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing pretokenized split indices: {split_path}")
        self.indices = np.load(split_path, mmap_mode="r")
        self.foreign_key_ids = np.load(self.root / "foreign_key_ids.npy", mmap_mode="r")
        self.datetime_values = np.load(self.root / "datetime_values.npy", mmap_mode="r")
        self.categorical_ids = np.load(self.root / "categorical_ids.npy", mmap_mode="r")
        self.numerical_values = None
        numerical_path = self.root / "numerical_values.npy"
        if self.schema.numerical_targets:
            if not numerical_path.exists():
                raise FileNotFoundError(f"Missing pretokenized numerical values: {numerical_path}")
            self.numerical_values = np.load(numerical_path, mmap_mode="r")
        self.review_time_ns = np.load(self.root / "review_time_ns.npy", mmap_mode="r")
        self.text_ids: dict[str, np.memmap] = {}
        self.text_lengths: dict[str, np.ndarray] = {}
        text_meta = self.bundle.metadata.get("text_fields", {})
        for column in self.schema.text_targets:
            field_meta = text_meta.get(column, {})
            token_path = self.root / f"{column}_token_ids.memmap"
            if not token_path.exists():
                raise FileNotFoundError(f"Missing pretokenized token ids for {column!r}: {token_path}")
            shape = tuple(int(value) for value in field_meta.get("shape", ()))
            if len(shape) != 2:
                raise ValueError(f"Invalid pretokenized shape for {column!r}: {shape}")
            dtype = np.dtype(field_meta.get("dtype", "int32"))
            self.text_ids[column] = np.memmap(token_path, dtype=dtype, mode="r", shape=shape)
            self.text_lengths[column] = np.load(self.root / f"{column}_lengths.npy", mmap_mode="r")

    @property
    def timestamps_ns(self) -> np.ndarray:
        return np.asarray(self.review_time_ns[self.indices], dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_idx = int(self.indices[int(index)])
        text_ids = {
            column: torch.as_tensor(np.asarray(array[source_idx], dtype=np.int64).copy(), dtype=torch.long)
            for column, array in self.text_ids.items()
        }
        sample = {
            "foreign_key_ids": torch.as_tensor(
                np.asarray(self.foreign_key_ids[source_idx], dtype=np.int64).copy(),
                dtype=torch.long,
            ),
            "datetime_values": torch.as_tensor(
                np.asarray(self.datetime_values[source_idx], dtype=np.float32).copy(),
                dtype=torch.float32,
            ),
            "categorical_ids": torch.as_tensor(
                np.asarray(self.categorical_ids[source_idx], dtype=np.int64).copy(),
                dtype=torch.long,
            ),
            "text_ids": text_ids,
            "row_id": torch.tensor(source_idx, dtype=torch.long),
        }
        if self.numerical_values is not None:
            sample["numerical_values"] = torch.as_tensor(
                np.asarray(self.numerical_values[source_idx], dtype=np.float32).copy(),
                dtype=torch.float32,
            )
        return sample


def load_pretokenized_bundle(root: str | Path, schema: ConditionalTABDLMSchema) -> PretokenizedBundle:
    root = Path(root)
    metadata = load_json(root / "metadata.json")
    tokenizer_path = root / "tokenizer_metadata.json"
    tokenizer = SimpleTextTokenizer.from_dict(load_json(tokenizer_path))
    vocabs = {
        column: CategoryVocab.from_dict(load_json(root / f"vocab_{column}.json"))
        for column in schema.model_categorical_targets
    }
    numerical_path = root / "numerical_metadata.json"
    numerical_metadata = load_json(numerical_path) if numerical_path.exists() else {}
    return PretokenizedBundle(
        root=root,
        metadata=metadata,
        schema=schema,
        categorical_vocabs=vocabs,
        tokenizer=tokenizer,
        numerical_metadata=numerical_metadata,
    )


def pretokenized_row_count(root: str | Path) -> int:
    metadata = load_json(Path(root) / "metadata.json")
    return int(metadata.get("num_rows", 0))
