#!/usr/bin/env python3
"""Precompute past-only temporal neighbor inputs for graph-conditioned training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempdir_bootstrap import configure_tempdir  # noqa: E402

configure_tempdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd

from attribute_generation.conditional_tabdlm.dataset import normalize_frame, validate_columns  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_schema import temporal_filter_config  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_cache import validate_cache_temporal_safety  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema, resolve_auto_review_text_config  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import normalize_text, stable_hash_bucket  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import ensure_dir, load_yaml, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute temporal neighbor cache.")
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
    if len(schema.foreign_key_columns) < 2:
        raise ValueError("Temporal neighbor cache expects at least two configured foreign key columns")
    customer_col = schema.foreign_key_columns[0]
    product_col = schema.foreign_key_columns[1]
    temporal = temporal_filter_config(raw)
    timestamp_col = str(temporal.get("timestamp_column", schema.datetime_columns[0]))
    output_dir = ensure_dir(args.output_dir)
    frame = load_graph_frame(Path(raw["paths"]["train_data_path"]), schema, [customer_col, product_col, timestamp_col])
    num_buckets = int(raw.get("id_encoding", {}).get("num_buckets", 262144))
    print(f"[neighbor-cache] building TemporalHistoryIndex for {len(frame):,} rows", flush=True)
    history = TemporalHistoryIndex.from_config(
        frame,
        raw,
        customer_col=customer_col,
        product_col=product_col,
        num_hash_buckets=num_buckets,
        seed=int(raw.get("training", {}).get("seed", 42)),
    )
    metadata = {
        "num_rows": int(len(frame)),
        "customer_column": str(customer_col),
        "product_column": str(product_col),
        "timestamp_column": str(timestamp_col),
        "max_customer_history": int(history.max_customer_history),
        "max_product_history": int(history.max_product_history),
        "num_workers_requested": int(args.num_workers),
        "chunk_size": int(args.chunk_size),
    }
    write_cache(output_dir, history, customer_col, product_col)
    save_json(metadata, output_dir / "metadata.json")
    diagnostics = validate_cache_temporal_safety(output_dir, sample_rows=min(10000, len(frame)))
    metadata["temporal_safety_sample"] = diagnostics
    save_json(metadata, output_dir / "metadata.json")
    print(f"Wrote temporal neighbor cache to {output_dir}", flush=True)


def load_graph_frame(path: Path, schema: ConditionalTABDLMSchema, columns: list[str]) -> pd.DataFrame:
    required = list(dict.fromkeys(schema.required_columns))
    frame = pd.read_csv(path, usecols=required, low_memory=False)
    validate_columns(frame, schema)
    frame = normalize_frame(frame, schema)
    for column in schema.text_targets:
        frame = frame[frame[column].map(normalize_text).str.len() > 0]
    frame = frame.dropna(subset=list(schema.condition_columns)).reset_index(drop=True)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing graph columns after filtering: {missing}")
    return frame.loc[:, columns].reset_index(drop=True)


def write_cache(output_dir: Path, history: TemporalHistoryIndex, customer_col: str, product_col: str) -> None:
    n = int(history.num_rows)
    customer_hash = np.asarray(
        [stable_hash_bucket(customer_col, value, history.num_hash_buckets) for value in history.customers],
        dtype=np.int64,
    )
    product_hash = np.asarray(
        [stable_hash_bucket(product_col, value, history.num_hash_buckets) for value in history.products],
        dtype=np.int64,
    )
    np.save(output_dir / "customer_hash.npy", customer_hash)
    np.save(output_dir / "product_hash.npy", product_hash)
    np.save(output_dir / "timestamp_ns.npy", np.asarray(history.timestamps_ns, dtype=np.int64))
    np.save(output_dir / "timestamp_seconds.npy", np.asarray(history.timestamps_seconds, dtype=np.float32))
    counts = np.zeros((n, 2), dtype=np.int32)
    for kind, width, count_col in [
        ("customer", history.max_customer_history, 0),
        ("product", history.max_product_history, 1),
    ]:
        shape = (n, max(0, int(width)))
        indices = np.memmap(output_dir / f"{kind}_history_indices.memmap", dtype=np.int64, mode="w+", shape=shape)
        mask = np.memmap(output_dir / f"{kind}_history_mask.memmap", dtype=np.uint8, mode="w+", shape=shape)
        deltas = np.memmap(output_dir / f"{kind}_history_time_deltas.memmap", dtype=np.float32, mode="w+", shape=shape)
        indices[:] = -1
        mask[:] = 0
        deltas[:] = 0.0
        for row_idx in range(n):
            rows = history.history_for_row(row_idx, kind=kind, deterministic=True)
            rows = list(rows[-width:]) if width > 0 else []
            counts[row_idx, count_col] = len(rows)
            if rows:
                end = len(rows)
                indices[row_idx, :end] = np.asarray(rows, dtype=np.int64)
                mask[row_idx, :end] = 1
                target_ts = float(history.timestamps_seconds[row_idx])
                deltas[row_idx, :end] = target_ts - np.asarray(history.timestamps_seconds[rows], dtype=np.float32)
            if row_idx and row_idx % 100000 == 0:
                print(f"[neighbor-cache] {kind}: cached {row_idx:,}/{n:,} rows", flush=True)
        indices.flush()
        mask.flush()
        deltas.flush()
    np.save(output_dir / "history_counts.npy", counts)


if __name__ == "__main__":
    main()
