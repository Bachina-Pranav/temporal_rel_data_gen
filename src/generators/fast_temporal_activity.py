"""Fast shrinkage temporal activity models for scalable event-spine generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


def canonical_time_bucket(series: pd.Series, granularity: str = "day") -> pd.Series:
    """Normalize timestamps to stable day or month string buckets."""

    if granularity not in {"day", "month"}:
        raise ValueError("granularity must be 'day' or 'month'")
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        valid_min = parsed.dropna().min()
        parsed = parsed.fillna(valid_min if pd.notna(valid_min) else pd.Timestamp("1970-01-01"))
    if granularity == "day":
        return parsed.dt.normalize().dt.strftime("%Y-%m-%d")
    return parsed.dt.to_period("M").astype(str)


def canonical_one_time(value: Any, granularity: str = "day") -> str:
    return canonical_time_bucket(pd.Series([value]), granularity=granularity).iloc[0]


def normalize_probs(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= eps:
        return np.ones_like(values, dtype=float) / max(len(values), 1)
    return values / total


def resolve_auto_alpha(alpha: Any, degrees: Iterable[int], default: float = 1.0) -> float:
    if str(alpha).lower() == "auto":
        values = np.asarray(list(degrees), dtype=float)
        if len(values) == 0:
            return float(default)
        return max(float(np.median(values)), 1e-6)
    return float(alpha)


class FastTemporalActivityModel:
    """Entity-time activity with empirical-to-block shrinkage and block/time caches."""

    def __init__(
        self,
        alpha: Any = "auto",
        block_time_smoothing: float = 5.0,
        granularity: str = "day",
        entity_kind: str = "entity",
        eps: float = 1e-12,
    ):
        self.alpha = alpha
        self.alpha_resolved = 1.0
        self.block_time_smoothing = float(block_time_smoothing)
        self.granularity = granularity
        self.entity_kind = entity_kind
        self.eps = float(eps)
        self.entity_ids: np.ndarray = np.asarray([], dtype=object)
        self.entity_to_index: Dict[Any, int] = {}
        self.entity_degree: Dict[Any, int] = {}
        self.entity_block: Dict[Any, int] = {}
        self.entity_time_counts: Dict[Any, Dict[str, int]] = {}
        self.time_buckets: List[str] = []
        self.time_index: Dict[str, int] = {}
        self.global_time_probs: np.ndarray = np.asarray([], dtype=float)
        self.block_time_probs: Dict[int, np.ndarray] = {}
        self.entities_by_block: Dict[int, np.ndarray] = {}
        self.degrees_by_block: Dict[int, np.ndarray] = {}
        self.weights_by_entity: Dict[Any, float] = {}
        self._cache: Dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def fit(
        self,
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        block_map: Mapping[Any, int],
        alpha: Optional[Any] = None,
    ) -> "FastTemporalActivityModel":
        frame = df[[entity_col, time_col]].copy()
        frame[time_col] = canonical_time_bucket(frame[time_col], self.granularity)
        degree_series = frame[entity_col].value_counts()
        self.alpha_resolved = resolve_auto_alpha(self.alpha if alpha is None else alpha, degree_series.to_numpy(dtype=int))
        self.entity_ids = np.asarray(sorted(degree_series.index.tolist()), dtype=object)
        self.entity_to_index = {entity: idx for idx, entity in enumerate(self.entity_ids)}
        self.entity_degree = {entity: int(degree_series.loc[entity]) for entity in self.entity_ids}
        self.entity_block = {entity: int(block_map.get(entity, 0)) for entity in self.entity_ids}
        self.time_buckets = sorted(frame[time_col].unique().tolist())
        self.time_index = {bucket: idx for idx, bucket in enumerate(self.time_buckets)}
        global_counts = frame[time_col].value_counts().reindex(self.time_buckets, fill_value=0).to_numpy(dtype=float)
        self.global_time_probs = normalize_probs(global_counts, eps=self.eps)
        grouped = frame.groupby([entity_col, time_col]).size()
        self.entity_time_counts = {}
        for (entity, bucket), count in grouped.items():
            self.entity_time_counts.setdefault(entity, {})[bucket] = int(count)
        block_counts: Dict[int, np.ndarray] = {}
        for entity in self.entity_ids:
            block = int(self.entity_block.get(entity, 0))
            counts = np.zeros(len(self.time_buckets), dtype=float)
            for bucket, count in self.entity_time_counts.get(entity, {}).items():
                counts[self.time_index[bucket]] += float(count)
            block_counts.setdefault(block, np.zeros(len(self.time_buckets), dtype=float))
            block_counts[block] += counts
        self.block_time_probs = {}
        for block, counts in block_counts.items():
            smoothed = counts + self.block_time_smoothing * self.global_time_probs
            self.block_time_probs[int(block)] = normalize_probs(smoothed, eps=self.eps)
        self.weights_by_entity = {
            entity: float(self.entity_degree[entity] / (self.entity_degree[entity] + self.alpha_resolved))
            for entity in self.entity_ids
        }
        by_block: Dict[int, List[Any]] = {}
        for entity in self.entity_ids:
            by_block.setdefault(int(self.entity_block.get(entity, 0)), []).append(entity)
        self.entities_by_block = {
            block: np.asarray(sorted(entities), dtype=object)
            for block, entities in by_block.items()
        }
        self.degrees_by_block = {
            block: np.asarray([self.entity_degree[entity] for entity in entities], dtype=float)
            for block, entities in self.entities_by_block.items()
        }
        self._cache = {}
        return self

    def probability(self, entity_id: Any, time_bucket: Any) -> float:
        if not self.time_buckets:
            return 1.0
        bucket = canonical_one_time(time_bucket, self.granularity)
        idx = self.time_index.get(bucket)
        if idx is None:
            return self.eps
        degree = float(self.entity_degree.get(entity_id, 0))
        block = int(self.entity_block.get(entity_id, 0))
        block_probs = self.block_time_probs.get(block, self.global_time_probs)
        block_prob = float(block_probs[idx]) if len(block_probs) else 1.0 / max(len(self.time_buckets), 1)
        empirical = 0.0
        if degree > 0:
            empirical = float(self.entity_time_counts.get(entity_id, {}).get(bucket, 0)) / degree
        weight = float(self.weights_by_entity.get(entity_id, 0.0))
        return max(weight * empirical + (1.0 - weight) * block_prob, self.eps)

    def probabilities(self, entity_ids: Iterable[Any], time_bucket: Any) -> np.ndarray:
        return np.asarray([self.probability(entity, time_bucket) for entity in entity_ids], dtype=float)

    def probabilities_for_block_time(
        self,
        block_id: int,
        time_bucket: Any,
        entity_ids: Optional[Iterable[Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        bucket = canonical_one_time(time_bucket, self.granularity)
        if entity_ids is not None:
            ids = np.asarray(list(entity_ids), dtype=object)
            probs = self.probabilities(ids, bucket)
            return ids, np.nan_to_num(probs, nan=self.eps, posinf=self.eps, neginf=self.eps).clip(min=0.0)
        key = (int(block_id), bucket)
        if key not in self._cache:
            ids = self.entities_by_block.get(int(block_id), np.asarray([], dtype=object))
            probs = self.probabilities(ids, bucket) if len(ids) else np.asarray([], dtype=float)
            probs = np.nan_to_num(probs, nan=self.eps, posinf=self.eps, neginf=self.eps).clip(min=0.0)
            self._cache[key] = (ids, probs)
        return self._cache[key]

    def degree_weights_for_block(self, block_id: int) -> tuple[np.ndarray, np.ndarray]:
        ids = self.entities_by_block.get(int(block_id), np.asarray([], dtype=object))
        degrees = self.degrees_by_block.get(int(block_id), np.asarray([], dtype=float))
        return ids, degrees

    def summary(self) -> Dict[str, Any]:
        degrees = np.asarray(list(self.entity_degree.values()), dtype=float)
        weights = np.asarray(list(self.weights_by_entity.values()), dtype=float)
        return {
            "entity_kind": self.entity_kind,
            "granularity": self.granularity,
            "alpha_requested": self.alpha,
            "alpha_resolved": float(self.alpha_resolved),
            "block_time_smoothing": float(self.block_time_smoothing),
            "num_entities": int(len(self.entity_degree)),
            "num_blocks": int(len(self.entities_by_block)),
            "num_time_buckets": int(len(self.time_buckets)),
            "mean_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
            "median_degree": float(np.median(degrees)) if len(degrees) else 0.0,
            "mean_shrinkage_weight": float(np.mean(weights)) if len(weights) else 0.0,
        }

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.summary(), handle, indent=2)
            handle.write("\n")
