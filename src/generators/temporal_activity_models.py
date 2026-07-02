"""Shrinkage temporal activity models for joint event-spine generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


def canonical_day_bucket(series: pd.Series) -> pd.Series:
    """Return canonical day buckets as YYYY-MM-DD strings."""

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        valid_min = parsed.dropna().min()
        parsed = parsed.fillna(valid_min if pd.notna(valid_min) else pd.Timestamp("1970-01-01"))
    return parsed.dt.normalize().dt.strftime("%Y-%m-%d")


class TemporalActivityModel:
    """Entity-time activity model with empirical-to-block shrinkage."""

    def __init__(
        self,
        alpha_time: float = 10.0,
        block_time_smoothing: float = 5.0,
        eps: float = 1e-12,
        entity_kind: str = "entity",
    ):
        self.alpha_time = float(alpha_time)
        self.block_time_smoothing = float(block_time_smoothing)
        self.eps = float(eps)
        self.entity_kind = entity_kind
        self.time_buckets: List[str] = []
        self.time_index: Dict[str, int] = {}
        self.entity_degree: Dict[Any, int] = {}
        self.entity_block: Dict[Any, int] = {}
        self.entity_time_counts: Dict[Any, Dict[str, int]] = {}
        self.block_time_probs: Dict[int, np.ndarray] = {}
        self.global_time_probs: np.ndarray = np.asarray([], dtype=float)
        self.weights: Dict[Any, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        entity_blocks: Mapping[Any, int],
    ) -> "TemporalActivityModel":
        frame = df[[entity_col, time_col]].copy()
        frame[time_col] = canonical_day_bucket(frame[time_col])
        self.time_buckets = sorted(frame[time_col].unique().tolist())
        self.time_index = {bucket: idx for idx, bucket in enumerate(self.time_buckets)}
        global_counts = frame[time_col].value_counts().reindex(self.time_buckets, fill_value=0).to_numpy(dtype=float)
        self.global_time_probs = normalize_probs(global_counts, eps=self.eps)
        degree = frame[entity_col].value_counts()
        self.entity_degree = {entity: int(count) for entity, count in degree.items()}
        self.entity_block = {entity: int(entity_blocks.get(entity, 0)) for entity in self.entity_degree}
        grouped = frame.groupby([entity_col, time_col]).size()
        self.entity_time_counts = {}
        for (entity, time_bucket), count in grouped.items():
            self.entity_time_counts.setdefault(entity, {})[time_bucket] = int(count)
        block_counts: Dict[int, np.ndarray] = {}
        for entity, block in self.entity_block.items():
            counts = np.zeros(len(self.time_buckets), dtype=float)
            for time_bucket, count in self.entity_time_counts.get(entity, {}).items():
                counts[self.time_index[time_bucket]] += float(count)
            block_counts.setdefault(block, np.zeros(len(self.time_buckets), dtype=float))
            block_counts[block] += counts
        self.block_time_probs = {}
        for block, counts in block_counts.items():
            smoothed = counts + self.block_time_smoothing * self.global_time_probs
            self.block_time_probs[int(block)] = normalize_probs(smoothed, eps=self.eps)
        self.weights = {
            entity: float(deg / (deg + self.alpha_time))
            for entity, deg in self.entity_degree.items()
        }
        return self

    @classmethod
    def fit_customer_activity(
        cls,
        df: pd.DataFrame,
        customer_col: str,
        time_col: str,
        customer_blocks: Mapping[Any, int],
        alpha_customer_time: float = 10.0,
        block_time_smoothing: float = 5.0,
    ) -> "TemporalActivityModel":
        return cls(alpha_customer_time, block_time_smoothing, entity_kind="customer").fit(
            df, customer_col, time_col, customer_blocks
        )

    @classmethod
    def fit_product_activity(
        cls,
        df: pd.DataFrame,
        product_col: str,
        time_col: str,
        product_blocks: Mapping[Any, int],
        alpha_product_time: float = 5.0,
        block_time_smoothing: float = 5.0,
    ) -> "TemporalActivityModel":
        return cls(alpha_product_time, block_time_smoothing, entity_kind="product").fit(
            df, product_col, time_col, product_blocks
        )

    def probability(self, entity_id: Any, time_bucket: Any) -> float:
        if not self.time_buckets:
            return 1.0
        time_key = canonical_one_day(time_bucket)
        idx = self.time_index.get(time_key)
        if idx is None:
            return self.eps
        degree = float(self.entity_degree.get(entity_id, 0))
        block = int(self.entity_block.get(entity_id, 0))
        block_probs = self.block_time_probs.get(block, self.global_time_probs)
        block_prob = float(block_probs[idx]) if len(block_probs) else 1.0 / max(len(self.time_buckets), 1)
        empirical = 0.0
        if degree > 0:
            empirical = float(self.entity_time_counts.get(entity_id, {}).get(time_key, 0)) / degree
        weight = float(self.weights.get(entity_id, 0.0))
        return max(weight * empirical + (1.0 - weight) * block_prob, self.eps)

    def probabilities(self, entity_ids: Iterable[Any], time_bucket: Any) -> np.ndarray:
        return np.asarray([self.probability(entity_id, time_bucket) for entity_id in entity_ids], dtype=float)

    def save_summary(self, path: str | Path) -> None:
        weights = np.asarray(list(self.weights.values()), dtype=float)
        degrees = np.asarray(list(self.entity_degree.values()), dtype=float)
        summary = {
            f"alpha_{self.entity_kind}_time": self.alpha_time,
            "block_time_smoothing": self.block_time_smoothing,
            f"num_{self.entity_kind}s": int(len(self.entity_degree)),
            "num_time_buckets": int(len(self.time_buckets)),
            f"mean_{self.entity_kind}_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
            f"median_{self.entity_kind}_degree": float(np.median(degrees)) if len(degrees) else 0.0,
            "min_weight": float(np.min(weights)) if len(weights) else 0.0,
            "max_weight": float(np.max(weights)) if len(weights) else 0.0,
            "mean_weight": float(np.mean(weights)) if len(weights) else 0.0,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")


def canonical_one_day(value: Any) -> str:
    return canonical_day_bucket(pd.Series([value])).iloc[0]


def normalize_probs(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= eps:
        return np.ones_like(values, dtype=float) / max(len(values), 1)
    return values / total
