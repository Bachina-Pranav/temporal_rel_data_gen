#!/usr/bin/env python3
"""Pretokenize configured text fields for scalable fixed-step training."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempdir_bootstrap import configure_tempdir  # noqa: E402

configure_tempdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd

from attribute_generation.conditional_tabdlm.dataset import normalize_frame, validate_columns  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
    resolve_auto_review_text_config,
)
from attribute_generation.conditional_tabdlm.tokenization import (  # noqa: E402
    CategoryVocab,
    SimpleTextTokenizer,
    normalize_category,
    normalize_text,
    stable_hash_bucket,
    summary_length_bucket_name,
)
from attribute_generation.conditional_tabdlm.utils import ensure_dir, load_yaml, save_json, save_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize text fields for a single event table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    if args.real_table:
        raw.setdefault("paths", {})["train_data_path"] = args.real_table
    raw = resolve_auto_review_text_config(raw)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    config = ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=Path(args.config))
    output_dir = ensure_dir(args.output_dir)
    save_yaml(config.to_dict(), output_dir / "config_resolved.yaml")
    save_json(schema.to_dict(), output_dir / "schema.json")

    print("[pretokenize] pass 1/3: counting valid rows and timestamps", flush=True)
    timestamps_ns = collect_timestamps(config, int(args.chunk_size))
    split = split_indices_from_timestamps(timestamps_ns)
    save_split_indices(output_dir, split)

    print("[pretokenize] pass 2/3: fitting tokenizer and categorical vocabularies", flush=True)
    tokenizer, vocabs = fit_tokenizer_and_vocabs(config, split, int(args.chunk_size))
    tokenizer_metadata = tokenizer.to_dict()
    tokenizer_metadata["text_max_lengths"] = dict(schema.text_max_lengths)
    for column in schema.text_targets:
        tokenizer_metadata[f"{column}_max_tokens"] = int(schema.text_max_lengths[column])
        tokenizer_metadata[f"{column}_max_content_tokens"] = tokenizer.max_content_tokens(schema.text_max_lengths[column])
    save_json(tokenizer_metadata, output_dir / "tokenizer_metadata.json")
    for column, vocab in vocabs.items():
        save_json(vocab.to_dict(), output_dir / f"vocab_{column}.json")

    print("[pretokenize] pass 3/3: writing memmaps and arrays", flush=True)
    metadata = write_arrays(config, output_dir, tokenizer, vocabs, timestamps_ns, int(args.chunk_size))
    metadata.update(
        {
            "config_path": str(args.config),
            "real_table_path": str(config.train_data_path),
            "num_workers_requested": int(args.num_workers),
            "chunk_size": int(args.chunk_size),
            "train_rows": int(len(split["train"])),
            "valid_rows": int(len(split["valid"])),
            "test_rows": int(len(split["test"])),
        }
    )
    save_json(metadata, output_dir / "metadata.json")
    print(f"Wrote pretokenized arrays to {output_dir}", flush=True)


def iter_valid_chunks(
    config: ConditionalTABDLMConfig,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    schema = config.schema
    required = list(dict.fromkeys(schema.required_columns))
    first = True
    for chunk in pd.read_csv(
        config.train_data_path,
        usecols=required,
        chunksize=int(chunk_size),
        low_memory=False,
    ):
        if first:
            validate_columns(chunk, schema)
            first = False
        frame = normalize_frame(chunk, schema)
        for column in schema.text_targets:
            frame = frame[frame[column].map(normalize_text).str.len() > 0]
        frame = frame.dropna(subset=list(schema.condition_columns)).reset_index(drop=True)
        if len(frame):
            yield frame


def collect_timestamps(config: ConditionalTABDLMConfig, chunk_size: int) -> np.ndarray:
    timestamp_col = config.schema.datetime_columns[0]
    pieces: list[np.ndarray] = []
    for frame in iter_valid_chunks(config, chunk_size):
        pieces.append(frame[timestamp_col].to_numpy(dtype="datetime64[ns]").astype(np.int64))
        print(f"[pretokenize] valid rows counted: {sum(len(piece) for piece in pieces):,}", flush=True)
    if not pieces:
        raise ValueError("No valid rows found for pretokenization")
    return np.concatenate(pieces).astype(np.int64, copy=False)


def split_indices_from_timestamps(timestamps_ns: np.ndarray) -> dict[str, np.ndarray]:
    n = int(len(timestamps_ns))
    order = np.argsort(timestamps_ns, kind="mergesort")
    train_end = int(n * 0.90)
    valid_end = int(n * 0.95)
    return {
        "train": np.sort(order[:train_end]).astype(np.int64),
        "valid": np.sort(order[train_end:valid_end]).astype(np.int64),
        "test": np.sort(order[valid_end:]).astype(np.int64),
    }


def save_split_indices(output_dir: Path, split: dict[str, np.ndarray]) -> None:
    for name, indices in split.items():
        np.save(output_dir / f"{name}_indices.npy", indices.astype(np.int64, copy=False))


def fit_tokenizer_and_vocabs(
    config: ConditionalTABDLMConfig,
    split: dict[str, np.ndarray],
    chunk_size: int,
) -> tuple[SimpleTextTokenizer, dict[str, CategoryVocab]]:
    schema = config.schema
    token_cfg = config.raw.get("tokenizer", {})
    tokenizer = SimpleTextTokenizer(lowercase=bool(token_cfg.get("lowercase", True)))
    token_counts: Counter[str] = Counter()
    categorical_counts: dict[str, Counter[str]] = {
        column: Counter() for column in schema.model_categorical_targets
    }
    train_mask = np.zeros(max(int(max(split["train"], default=-1)) + 1, 0), dtype=bool)
    if len(split["train"]):
        train_mask = np.zeros(int(max(max(split["train"]) + 1, len(train_mask))), dtype=bool)
        train_mask[split["train"]] = True
    row_offset = 0
    for frame in iter_valid_chunks(config, chunk_size):
        row_ids = np.arange(row_offset, row_offset + len(frame), dtype=np.int64)
        row_offset += len(frame)
        mask = row_ids < len(train_mask)
        mask[mask] = train_mask[row_ids[mask]]
        train_frame = frame.loc[mask].reset_index(drop=True)
        for column in schema.text_targets:
            for text in train_frame[column]:
                token_counts.update(tokenizer.tokenize(text))
        for column in schema.categorical_targets:
            categorical_counts[column].update(normalize_category(value) for value in train_frame[column])
        for column in schema.auxiliary_categorical_targets:
            categorical_counts[column].update(auxiliary_values(train_frame, schema, tokenizer, column))
    add_tokens_from_counts(
        tokenizer,
        token_counts,
        max_vocab_size=int(token_cfg.get("max_vocab_size", 30000)),
        min_frequency=int(token_cfg.get("min_frequency", 1)),
    )
    return tokenizer, {
        column: vocab_from_counts(column, counts)
        for column, counts in categorical_counts.items()
    }


def write_arrays(
    config: ConditionalTABDLMConfig,
    output_dir: Path,
    tokenizer: SimpleTextTokenizer,
    vocabs: dict[str, CategoryVocab],
    timestamps_ns: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    schema = config.schema
    n = int(len(timestamps_ns))
    fk = np.lib.format.open_memmap(
        output_dir / "foreign_key_ids.npy",
        mode="w+",
        dtype=np.int64,
        shape=(n, len(schema.foreign_key_columns)),
    )
    dt = np.lib.format.open_memmap(
        output_dir / "datetime_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, len(schema.datetime_columns)),
    )
    cats = np.lib.format.open_memmap(
        output_dir / "categorical_ids.npy",
        mode="w+",
        dtype=np.int64,
        shape=(n, len(schema.model_categorical_targets)),
    )
    np.save(output_dir / "review_time_ns.npy", timestamps_ns.astype(np.int64, copy=False))
    np.save(output_dir / "normalized_review_time.npy", (timestamps_ns.astype(np.float64) / 1e9).astype(np.float32))
    for idx, column in enumerate(schema.foreign_key_columns):
        np.lib.format.open_memmap(output_dir / f"{column}_encoded.npy", mode="w+", dtype=np.int64, shape=(n,))
    text_arrays: dict[str, np.memmap] = {}
    text_meta: dict[str, Any] = {}
    for column in schema.text_targets:
        shape = (n, int(schema.text_max_lengths[column]))
        text_arrays[column] = np.memmap(
            output_dir / f"{column}_token_ids.memmap",
            dtype=np.int32,
            mode="w+",
            shape=shape,
        )
        np.lib.format.open_memmap(output_dir / f"{column}_lengths.npy", mode="w+", dtype=np.int32, shape=(n,))
        text_meta[column] = {"shape": list(shape), "dtype": "int32"}
    row_offset = 0
    fk_column_arrays = {
        column: np.load(output_dir / f"{column}_encoded.npy", mmap_mode="r+")
        for column in schema.foreign_key_columns
    }
    length_arrays = {
        column: np.load(output_dir / f"{column}_lengths.npy", mmap_mode="r+")
        for column in schema.text_targets
    }
    for frame in iter_valid_chunks(config, chunk_size):
        start = row_offset
        end = row_offset + len(frame)
        row_offset = end
        for col_idx, column in enumerate(schema.foreign_key_columns):
            values = [stable_hash_bucket(column, value, config.raw.get("id_encoding", {}).get("num_buckets", 262144)) for value in frame[column]]
            fk[start:end, col_idx] = np.asarray(values, dtype=np.int64)
            fk_column_arrays[column][start:end] = np.asarray(values, dtype=np.int64)
        for col_idx, column in enumerate(schema.datetime_columns):
            dt[start:end, col_idx] = frame[column].map(pd.Timestamp.timestamp).to_numpy(dtype=np.float32)
        for col_idx, column in enumerate(schema.model_categorical_targets):
            values = auxiliary_values(frame, schema, tokenizer, column) if column in schema.auxiliary_categorical_targets else frame[column]
            cats[start:end, col_idx] = np.asarray([vocabs[column].encode(value) for value in values], dtype=np.int64)
        for column in schema.text_targets:
            max_len = int(schema.text_max_lengths[column])
            rows = []
            lengths = []
            for text in frame[column]:
                ids, _ = tokenizer.encode(text, max_len)
                rows.append(ids)
                lengths.append(tokenizer.content_length(ids))
            text_arrays[column][start:end, :] = np.asarray(rows, dtype=np.int32)
            length_arrays[column][start:end] = np.asarray(lengths, dtype=np.int32)
        print(f"[pretokenize] encoded rows: {end:,}/{n:,}", flush=True)
    fk.flush()
    dt.flush()
    cats.flush()
    for arr in text_arrays.values():
        arr.flush()
    for col_idx, column in enumerate(schema.model_categorical_targets):
        if column in schema.categorical_targets:
            np.save(output_dir / f"{column}.npy", np.asarray(cats[:, col_idx], dtype=np.int64))
    return {
        "num_rows": n,
        "foreign_key_columns": list(schema.foreign_key_columns),
        "datetime_columns": list(schema.datetime_columns),
        "categorical_columns": list(schema.model_categorical_targets),
        "text_fields": text_meta,
        "model_family": "conditional_tabdlm_lstm_joint_full_text",
    }


def add_tokens_from_counts(
    tokenizer: SimpleTextTokenizer,
    counts: Counter[str],
    *,
    max_vocab_size: int,
    min_frequency: int,
) -> None:
    protected = set(tokenizer.vocab)
    candidates = sorted(
        (
            (token, count)
            for token, count in counts.items()
            if int(count) >= int(min_frequency) and token not in protected
        ),
        key=lambda item: (-item[1], item[0]),
    )
    for token, _ in candidates:
        if len(tokenizer.vocab) >= int(max_vocab_size):
            break
        tokenizer.vocab[token] = len(tokenizer.vocab)
    tokenizer.inv_vocab = {idx: token for token, idx in tokenizer.vocab.items()}


def vocab_from_counts(column: str, counts: Counter[str]) -> CategoryVocab:
    tokens = sorted(counts, key=lambda token: (-counts[token], token))
    if "<missing>" not in tokens:
        tokens.insert(0, "<missing>")
    return CategoryVocab(column=column, token_to_id={token: idx for idx, token in enumerate(tokens)})


def auxiliary_values(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    tokenizer: SimpleTextTokenizer,
    column: str,
) -> list[str]:
    if column not in {"summary_length_bucket", "review_text_length_bucket"}:
        raise KeyError(f"Unsupported auxiliary categorical target: {column}")
    if frame.empty:
        return []
    text_col = schema.text_column_for_length_bucket(column)
    max_content = tokenizer.max_content_tokens(schema.text_max_lengths[text_col])
    buckets = schema.buckets_for_length_bucket(column)
    values = []
    for text in frame[text_col]:
        length = min(len(tokenizer.tokenize(text)), max_content)
        values.append(summary_length_bucket_name(length, buckets))
    return values


if __name__ == "__main__":
    main()
