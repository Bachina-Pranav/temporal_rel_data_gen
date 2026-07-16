"""Past-only temporal ego-history sampling for graph-conditioned TABDLM."""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .graph_schema import temporal_filter_config
from .tokenization import stable_hash_bucket


@dataclass(frozen=True)
class TemporalHistoryStats:
    num_rows: int
    fraction_rows_with_customer_history: float
    fraction_rows_with_product_history: float
    fraction_rows_with_any_history: float
    mean_customer_history_count_used: float
    mean_product_history_count_used: float
    p90_customer_history_count_used: float
    p90_product_history_count_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_rows": int(self.num_rows),
            "fraction_rows_with_customer_history": float(self.fraction_rows_with_customer_history),
            "fraction_rows_with_product_history": float(self.fraction_rows_with_product_history),
            "fraction_rows_with_any_history": float(self.fraction_rows_with_any_history),
            "mean_customer_history_count_used": float(self.mean_customer_history_count_used),
            "mean_product_history_count_used": float(self.mean_product_history_count_used),
            "p90_customer_history_count_used": float(self.p90_customer_history_count_used),
            "p90_product_history_count_used": float(self.p90_product_history_count_used),
        }


class TemporalHistoryIndex:
    """Lookup structure for strict past-only customer/product event histories."""

    def __init__(
        self,
        frame: pd.DataFrame,
        customer_col: str,
        product_col: str,
        timestamp_col: str,
        num_hash_buckets: int,
        *,
        max_customer_history: int = 50,
        max_product_history: int = 100,
        allow_same_timestamp_events: bool = False,
        history_sampling_strategy: str = "recent",
        recent_fraction_if_mixed: float = 0.7,
        seed: int = 42,
    ):
        required = [customer_col, product_col, timestamp_col]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Temporal graph frame is missing columns: {missing}")
        self.customer_col = str(customer_col)
        self.product_col = str(product_col)
        self.timestamp_col = str(timestamp_col)
        self.num_hash_buckets = int(num_hash_buckets)
        self.max_customer_history = int(max_customer_history)
        self.max_product_history = int(max_product_history)
        self.allow_same_timestamp_events = bool(allow_same_timestamp_events)
        self.history_sampling_strategy = str(history_sampling_strategy)
        self.recent_fraction_if_mixed = float(recent_fraction_if_mixed)
        self.rng = random.Random(int(seed))

        normalized = frame.loc[:, required].copy().reset_index(drop=True)
        normalized[customer_col] = normalized[customer_col].astype(str)
        normalized[product_col] = normalized[product_col].astype(str)
        timestamps = pd.to_datetime(normalized[timestamp_col], errors="coerce")
        if timestamps.isna().any():
            bad = int(timestamps.isna().sum())
            raise ValueError(f"Temporal graph has {bad} invalid timestamps in {timestamp_col!r}")
        self.customers = normalized[customer_col].to_numpy(dtype=object)
        self.products = normalized[product_col].to_numpy(dtype=object)
        self.timestamps_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.timestamps_seconds = (self.timestamps_ns.astype(np.float64) / 1e9).astype(np.float32)
        self.num_rows = int(len(normalized))
        self.customer_histories = self._build_histories(self.customers)
        self.product_histories = self._build_histories(self.products)

    @classmethod
    def from_config(
        cls,
        frame: pd.DataFrame,
        raw_config: dict[str, Any],
        customer_col: str,
        product_col: str,
        *,
        num_hash_buckets: int,
        seed: int = 42,
    ) -> "TemporalHistoryIndex":
        temporal = temporal_filter_config(raw_config)
        return cls(
            frame,
            customer_col=customer_col,
            product_col=product_col,
            timestamp_col=str(temporal.get("timestamp_column", "review_time")),
            num_hash_buckets=num_hash_buckets,
            max_customer_history=int(temporal.get("max_history_events_per_customer", 50)),
            max_product_history=int(temporal.get("max_history_events_per_product", 100)),
            allow_same_timestamp_events=bool(temporal.get("allow_same_timestamp_events", False)),
            history_sampling_strategy=str(temporal.get("history_sampling_strategy", "recent")),
            recent_fraction_if_mixed=float(temporal.get("recent_fraction_if_mixed", 0.7)),
            seed=seed,
        )

    def _build_histories(self, keys: np.ndarray) -> dict[str, list[tuple[int, int]]]:
        histories: dict[str, list[tuple[int, int]]] = {}
        for row_idx, key in enumerate(keys):
            histories.setdefault(str(key), []).append((int(self.timestamps_ns[row_idx]), int(row_idx)))
        for values in histories.values():
            values.sort()
        return histories

    def history_for_row(
        self,
        row_idx: int,
        *,
        kind: str,
        deterministic: bool = True,
    ) -> list[int]:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= self.num_rows:
            raise IndexError(f"row_idx out of range: {row_idx}")
        if kind == "customer":
            key = str(self.customers[row_idx])
            histories = self.customer_histories
            max_history = self.max_customer_history
        elif kind == "product":
            key = str(self.products[row_idx])
            histories = self.product_histories
            max_history = self.max_product_history
        else:
            raise ValueError(f"Unsupported history kind: {kind!r}")
        items = histories.get(key, [])
        target_ts = int(self.timestamps_ns[row_idx])
        if self.allow_same_timestamp_events:
            cutoff = (target_ts, row_idx)
        else:
            cutoff = (target_ts, -1)
        end = bisect.bisect_left(items, cutoff)
        if max_history <= 0:
            return []
        if deterministic or self.history_sampling_strategy == "recent":
            start = max(0, end - max_history)
            selected = [event_idx for _, event_idx in items[start:end] if int(event_idx) != row_idx]
            if len(selected) > max_history:
                selected = selected[-max_history:]
            self._assert_temporal_safety(row_idx, selected)
            return selected
        candidates = [event_idx for _, event_idx in items[:end] if int(event_idx) != row_idx]
        selected = self._sample_history(candidates, max_history=max_history, deterministic=deterministic)
        self._assert_temporal_safety(row_idx, selected)
        return selected

    def _sample_history(self, candidates: list[int], *, max_history: int, deterministic: bool) -> list[int]:
        if max_history <= 0:
            return []
        if len(candidates) <= max_history:
            return list(candidates)
        if deterministic or self.history_sampling_strategy == "recent":
            return list(candidates[-max_history:])
        if self.history_sampling_strategy == "uniform_past":
            return sorted(self.rng.sample(candidates, k=max_history), key=lambda idx: (self.timestamps_ns[idx], idx))
        if self.history_sampling_strategy == "mixed_recent_uniform":
            recent_count = max(1, min(max_history, int(round(max_history * self.recent_fraction_if_mixed))))
            recent = candidates[-recent_count:]
            older = candidates[: max(0, len(candidates) - recent_count)]
            remaining = max_history - len(recent)
            sampled = self.rng.sample(older, k=min(remaining, len(older))) if remaining > 0 and older else []
            return sorted(sampled + recent, key=lambda idx: (self.timestamps_ns[idx], idx))
        raise ValueError(f"Unknown history_sampling_strategy: {self.history_sampling_strategy!r}")

    def _assert_temporal_safety(self, target_idx: int, history_indices: list[int]) -> None:
        target_ts = int(self.timestamps_ns[target_idx])
        for hist_idx in history_indices:
            if int(hist_idx) == int(target_idx):
                raise AssertionError("Temporal graph history included the target event itself")
            hist_ts = int(self.timestamps_ns[int(hist_idx)])
            if self.allow_same_timestamp_events:
                ok = hist_ts < target_ts or (hist_ts == target_ts and int(hist_idx) < int(target_idx))
            else:
                ok = hist_ts < target_ts
            if not ok:
                raise AssertionError(
                    "Temporal graph history included a same-time/future event: "
                    f"target_idx={target_idx}, hist_idx={hist_idx}, target_ts={target_ts}, hist_ts={hist_ts}"
                )

    def build_batch(
        self,
        row_indices: list[int] | np.ndarray | torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        if torch.is_tensor(row_indices):
            rows = [int(value) for value in row_indices.detach().cpu().tolist()]
        else:
            rows = [int(value) for value in row_indices]
        customer_histories = [self.history_for_row(row, kind="customer", deterministic=deterministic) for row in rows]
        product_histories = [self.history_for_row(row, kind="product", deterministic=deterministic) for row in rows]
        return {
            "target_row_index": torch.tensor(rows, dtype=torch.long, device=device),
            "target_customer_hash": self._hash_values([self.customers[row] for row in rows], self.customer_col, device),
            "target_product_hash": self._hash_values([self.products[row] for row in rows], self.product_col, device),
            "target_time": torch.tensor([self.timestamps_seconds[row] for row in rows], dtype=torch.float32, device=device),
            **self._pack_history(customer_histories, kind="customer", device=device),
            **self._pack_history(product_histories, kind="product", device=device),
        }

    def _pack_history(
        self,
        histories: list[list[int]],
        *,
        kind: str,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        width = self.max_customer_history if kind == "customer" else self.max_product_history
        width = max(int(width), 0)
        row_index = torch.full((len(histories), width), -1, dtype=torch.long, device=device)
        customer_hash = torch.zeros((len(histories), width), dtype=torch.long, device=device)
        product_hash = torch.zeros((len(histories), width), dtype=torch.long, device=device)
        times = torch.zeros((len(histories), width), dtype=torch.float32, device=device)
        mask = torch.zeros((len(histories), width), dtype=torch.bool, device=device)
        for batch_idx, history in enumerate(histories):
            clipped = list(history[-width:]) if width > 0 else []
            for pos, hist_idx in enumerate(clipped):
                row_index[batch_idx, pos] = int(hist_idx)
                customer_hash[batch_idx, pos] = stable_hash_bucket(self.customer_col, self.customers[hist_idx], self.num_hash_buckets)
                product_hash[batch_idx, pos] = stable_hash_bucket(self.product_col, self.products[hist_idx], self.num_hash_buckets)
                times[batch_idx, pos] = float(self.timestamps_seconds[hist_idx])
                mask[batch_idx, pos] = True
        prefix = f"{kind}_history"
        return {
            f"{prefix}_row_index": row_index,
            f"{prefix}_customer_hash": customer_hash,
            f"{prefix}_product_hash": product_hash,
            f"{prefix}_time": times,
            f"{prefix}_mask": mask,
        }

    def _hash_values(
        self,
        values: list[Any],
        column: str,
        device: str | torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [stable_hash_bucket(column, value, self.num_hash_buckets) for value in values],
            dtype=torch.long,
            device=device,
        )

    def diagnostics(self, sample_size: int | None = None) -> TemporalHistoryStats:
        rows = list(range(self.num_rows))
        if sample_size is not None and self.num_rows > int(sample_size):
            rng = random.Random(17)
            rows = sorted(rng.sample(rows, int(sample_size)))
        customer_counts = np.asarray([len(self.history_for_row(row, kind="customer")) for row in rows], dtype=float)
        product_counts = np.asarray([len(self.history_for_row(row, kind="product")) for row in rows], dtype=float)
        any_counts = (customer_counts > 0) | (product_counts > 0)
        return TemporalHistoryStats(
            num_rows=int(len(rows)),
            fraction_rows_with_customer_history=float(np.mean(customer_counts > 0)) if len(rows) else 0.0,
            fraction_rows_with_product_history=float(np.mean(product_counts > 0)) if len(rows) else 0.0,
            fraction_rows_with_any_history=float(np.mean(any_counts)) if len(rows) else 0.0,
            mean_customer_history_count_used=float(np.mean(customer_counts)) if len(rows) else 0.0,
            mean_product_history_count_used=float(np.mean(product_counts)) if len(rows) else 0.0,
            p90_customer_history_count_used=float(np.quantile(customer_counts, 0.9)) if len(rows) else 0.0,
            p90_product_history_count_used=float(np.quantile(product_counts, 0.9)) if len(rows) else 0.0,
        )
