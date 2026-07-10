"""Cached temporal neighbor inputs for graph-conditioned attribute training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import load_json


class CachedTemporalHistoryIndex:
    """A TemporalHistoryIndex-compatible wrapper around precomputed memmaps."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.metadata = load_json(self.root / "metadata.json")
        self.num_rows = int(self.metadata["num_rows"])
        self.max_customer_history = int(self.metadata.get("max_customer_history", 0))
        self.max_product_history = int(self.metadata.get("max_product_history", 0))
        self.customer_hash = np.load(self.root / "customer_hash.npy", mmap_mode="r")
        self.product_hash = np.load(self.root / "product_hash.npy", mmap_mode="r")
        self.timestamp_seconds = np.load(self.root / "timestamp_seconds.npy", mmap_mode="r")
        self.timestamp_ns = np.load(self.root / "timestamp_ns.npy", mmap_mode="r")
        self._history_indices: dict[str, np.memmap] = {}
        self._history_masks: dict[str, np.memmap] = {}
        for kind, width in [("customer", self.max_customer_history), ("product", self.max_product_history)]:
            shape = (self.num_rows, max(0, int(width)))
            self._history_indices[kind] = np.memmap(
                self.root / f"{kind}_history_indices.memmap",
                dtype=np.int64,
                mode="r",
                shape=shape,
            )
            self._history_masks[kind] = np.memmap(
                self.root / f"{kind}_history_mask.memmap",
                dtype=np.uint8,
                mode="r",
                shape=shape,
            )

    def build_batch(
        self,
        row_indices: list[int] | np.ndarray | torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        del deterministic
        rows = tensor_to_int_array(row_indices)
        if rows.size and (rows.min() < 0 or rows.max() >= self.num_rows):
            raise IndexError("Cached temporal history row index out of range")
        output: dict[str, torch.Tensor] = {
            "target_row_index": torch.as_tensor(rows, dtype=torch.long, device=device),
            "target_customer_hash": torch.as_tensor(self.customer_hash[rows], dtype=torch.long, device=device),
            "target_product_hash": torch.as_tensor(self.product_hash[rows], dtype=torch.long, device=device),
            "target_time": torch.as_tensor(self.timestamp_seconds[rows], dtype=torch.float32, device=device),
        }
        output.update(self._pack_history(rows, kind="customer", device=device))
        output.update(self._pack_history(rows, kind="product", device=device))
        return output

    def _pack_history(
        self,
        rows: np.ndarray,
        *,
        kind: str,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        raw_indices = np.asarray(self._history_indices[kind][rows], dtype=np.int64)
        raw_mask = np.asarray(self._history_masks[kind][rows], dtype=np.uint8).astype(bool)
        safe_indices = np.where(raw_indices >= 0, raw_indices, 0)
        prefix = f"{kind}_history"
        customer_hash = np.asarray(self.customer_hash[safe_indices], dtype=np.int64)
        product_hash = np.asarray(self.product_hash[safe_indices], dtype=np.int64)
        history_time = np.asarray(self.timestamp_seconds[safe_indices], dtype=np.float32)
        customer_hash[~raw_mask] = 0
        product_hash[~raw_mask] = 0
        history_time[~raw_mask] = 0.0
        return {
            f"{prefix}_row_index": torch.as_tensor(raw_indices, dtype=torch.long, device=device),
            f"{prefix}_customer_hash": torch.as_tensor(customer_hash, dtype=torch.long, device=device),
            f"{prefix}_product_hash": torch.as_tensor(product_hash, dtype=torch.long, device=device),
            f"{prefix}_time": torch.as_tensor(history_time, dtype=torch.float32, device=device),
            f"{prefix}_mask": torch.as_tensor(raw_mask, dtype=torch.bool, device=device),
        }


def tensor_to_int_array(row_indices: list[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(row_indices):
        values = row_indices.detach().cpu().numpy()
    else:
        values = np.asarray(row_indices)
    return values.astype(np.int64, copy=False).reshape(-1)


def validate_cache_temporal_safety(root: str | Path, sample_rows: int | None = None) -> dict[str, Any]:
    cache = CachedTemporalHistoryIndex(root)
    rows = np.arange(cache.num_rows, dtype=np.int64)
    if sample_rows is not None and cache.num_rows > int(sample_rows):
        rng = np.random.default_rng(17)
        rows = np.sort(rng.choice(rows, size=int(sample_rows), replace=False))
    violations = 0
    checked = 0
    for kind in ["customer", "product"]:
        indices = np.asarray(cache._history_indices[kind][rows], dtype=np.int64)
        masks = np.asarray(cache._history_masks[kind][rows], dtype=np.uint8).astype(bool)
        target_ts = np.asarray(cache.timestamp_ns[rows], dtype=np.int64)[:, None]
        safe_indices = np.where(indices >= 0, indices, 0)
        hist_ts = np.asarray(cache.timestamp_ns[safe_indices], dtype=np.int64)
        active = masks & (indices >= 0)
        violations += int(np.sum(active & (hist_ts >= target_ts)))
        checked += int(np.sum(active))
    return {
        "num_rows_checked": int(len(rows)),
        "num_history_edges_checked": int(checked),
        "future_or_same_time_violations": int(violations),
        "temporal_past_only": bool(violations == 0),
    }
