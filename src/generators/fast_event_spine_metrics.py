"""Metrics wrapper for fast low-rank temporal event-spine generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .event_spine_metrics import (
    block_key_frame,
    canonicalize,
    evaluate_event_spine,
    write_metrics,
)
from .joint_temporal_2k_sbm_event import load_blocks


RUNTIME_KEYS = [
    "fit_seconds",
    "sample_seconds",
    "total_seconds",
    "events_per_second",
    "num_cells_processed",
    "average_cell_size",
    "max_cell_size",
    "percent_large_cells_projection_sort",
]


def evaluate_fast_event_spine(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    structure_debug_dir: Optional[str | Path] = None,
    customer_col: str = "customer_id",
    product_col: str = "product_id",
    timestamp_col: str = "review_time",
    compute_c2st: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = evaluate_event_spine(
        real,
        synthetic,
        structure_debug_dir=structure_debug_dir,
        customer_col=customer_col,
        product_col=product_col,
        timestamp_col=timestamp_col,
        compute_c2st=compute_c2st,
    )
    metrics["block_pair_count_l1"] = block_pair_count_l1(
        real,
        synthetic,
        structure_debug_dir,
        customer_col,
        product_col,
        timestamp_col,
    )
    if metadata:
        for key in RUNTIME_KEYS:
            metrics[key] = metadata.get(key)
        metrics["method"] = metadata.get("method")
    return metrics


def block_pair_count_l1(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    structure_debug_dir: Optional[str | Path],
    customer_col: str,
    product_col: str,
    timestamp_col: str,
) -> Optional[float]:
    root = Path(structure_debug_dir) if structure_debug_dir else None
    customer_blocks = load_blocks(root, "customer_blocks.csv", [customer_col, "id", "customer_id", "entity_id"], ["customer_block", "block"])
    product_blocks = load_blocks(root, "product_blocks.csv", [product_col, "id", "product_id", "entity_id"], ["product_block", "block"])
    if not customer_blocks or not product_blocks:
        return None
    real_c = canonicalize(real, customer_col, product_col, timestamp_col)
    syn_c = canonicalize(synthetic, customer_col, product_col, timestamp_col)
    real_keys = block_key_frame(real_c, customer_col, product_col, "_time_bucket", customer_blocks, product_blocks)
    syn_keys = block_key_frame(syn_c, customer_col, product_col, "_time_bucket", customer_blocks, product_blocks)
    real_pair = real_keys.groupby(["customer_block", "product_block"]).size()
    syn_pair = syn_keys.groupby(["customer_block", "product_block"]).size()
    pairs = sorted(set(real_pair.index).union(set(syn_pair.index)))
    return float(sum(abs(real_pair.get(pair, 0) - syn_pair.get(pair, 0)) for pair in pairs) / max(len(real_c), 1))


def load_metadata(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


__all__ = ["evaluate_fast_event_spine", "write_metrics", "load_metadata"]
