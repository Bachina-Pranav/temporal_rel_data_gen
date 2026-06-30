"""Temporal neighborhood sampling for Amazon-style review graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


CONTEXT_FEATURE_NAMES = [
    "customer_history_frac",
    "product_history_frac",
    "customer_mean_delta_frac",
    "product_mean_delta_frac",
    "customer_decay_sum",
    "product_decay_sum",
    "customer_past_rating_mean",
    "product_past_rating_mean",
    "customer_past_verified_rate",
    "product_past_verified_rate",
]


@dataclass
class TemporalNeighborhood:
    target_index: int
    customer_id: Any
    product_id: Any
    target_time: pd.Timestamp
    customer_history_indices: np.ndarray
    product_history_indices: np.ndarray
    customer_delta_days: np.ndarray
    product_delta_days: np.ndarray
    customer_weights: np.ndarray
    product_weights: np.ndarray

    @property
    def review_indices(self) -> np.ndarray:
        return np.unique(
            np.concatenate([self.customer_history_indices, self.product_history_indices])
        )


class TemporalReviewNeighborSampler:
    """Sample temporally filtered review neighborhoods around target reviews."""

    def __init__(
        self,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        temporal_window_days: float = 365.0,
        max_customer_history: int = 32,
        max_product_history: int = 32,
        temporal_mode: str = "causal_window",
        tau_decay_days: float = 90.0,
    ):
        if temporal_mode not in {"causal", "causal_window", "symmetric_window"}:
            raise ValueError(
                "temporal_mode must be causal, causal_window, or symmetric_window."
            )
        self.reviews = reviews.copy().reset_index(drop=True)
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.temporal_window_days = float(temporal_window_days)
        self.max_customer_history = int(max_customer_history)
        self.max_product_history = int(max_product_history)
        self.temporal_mode = temporal_mode
        self.tau_decay_days = float(tau_decay_days)

        self.reviews[self.timestamp_col] = pd.to_datetime(
            self.reviews[self.timestamp_col], errors="coerce"
        )
        if self.reviews[self.timestamp_col].isna().any():
            raise ValueError("TemporalReviewNeighborSampler received invalid timestamps.")
        self.timestamps = self.reviews[self.timestamp_col].reset_index(drop=True)
        self._customer_index = self._build_index(self.customer_id_col)
        self._product_index = self._build_index(self.product_id_col)

    @property
    def context_feature_names(self) -> List[str]:
        return list(CONTEXT_FEATURE_NAMES)

    def sample(self, target_index: int) -> TemporalNeighborhood:
        target_index = int(target_index)
        row = self.reviews.iloc[target_index]
        target_time = pd.Timestamp(row[self.timestamp_col])
        customer_id = row[self.customer_id_col]
        product_id = row[self.product_id_col]

        customer_history = self._history_indices(
            self._customer_index.get(customer_id, np.asarray([], dtype=int)),
            target_index,
            target_time,
            self.max_customer_history,
        )
        product_history = self._history_indices(
            self._product_index.get(product_id, np.asarray([], dtype=int)),
            target_index,
            target_time,
            self.max_product_history,
        )
        customer_delta_days = self._delta_days(target_time, customer_history)
        product_delta_days = self._delta_days(target_time, product_history)

        return TemporalNeighborhood(
            target_index=target_index,
            customer_id=customer_id,
            product_id=product_id,
            target_time=target_time,
            customer_history_indices=customer_history,
            product_history_indices=product_history,
            customer_delta_days=customer_delta_days,
            product_delta_days=product_delta_days,
            customer_weights=self._decay_weights(customer_delta_days),
            product_weights=self._decay_weights(product_delta_days),
        )

    def sample_many(self, target_indices: Iterable[int]) -> List[TemporalNeighborhood]:
        return [self.sample(index) for index in target_indices]

    def assert_no_future(self, target_indices: Iterable[int]) -> None:
        if self.temporal_mode == "symmetric_window":
            return
        for target_index in target_indices:
            neighborhood = self.sample(int(target_index))
            target_time = neighborhood.target_time
            for neighbor_index in neighborhood.review_indices:
                neighbor_time = pd.Timestamp(self.timestamps.iloc[int(neighbor_index)])
                if neighbor_time > target_time:
                    raise AssertionError(
                        "Temporal sampler leaked a future review: "
                        f"neighbor={neighbor_time}, target={target_time}"
                    )

    def context_features(
        self,
        target_indices: Iterable[int],
        rating_values: Optional[np.ndarray] = None,
        verified_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rows = []
        for target_index in target_indices:
            neighborhood = self.sample(int(target_index))
            rows.append(
                self._context_row(neighborhood, rating_values, verified_values)
            )
        return np.asarray(rows, dtype=np.float32)

    def _build_index(self, column: str) -> Dict[Any, np.ndarray]:
        grouped: Dict[Any, List[int]] = {}
        for index, value in enumerate(self.reviews[column].to_numpy(dtype=object)):
            grouped.setdefault(value, []).append(index)

        sorted_grouped = {}
        for value, indices in grouped.items():
            indices_array = np.asarray(indices, dtype=int)
            order = np.argsort(
                self.timestamps.iloc[indices_array].astype("int64").to_numpy(),
                kind="mergesort",
            )
            sorted_grouped[value] = indices_array[order]
        return sorted_grouped

    def _history_indices(
        self,
        candidates: np.ndarray,
        target_index: int,
        target_time: pd.Timestamp,
        max_history: int,
    ) -> np.ndarray:
        if len(candidates) == 0:
            return np.asarray([], dtype=int)

        candidate_times = self.timestamps.iloc[candidates]
        if self.temporal_mode == "symmetric_window":
            delta_days = np.abs(
                (candidate_times - target_time).dt.total_seconds().to_numpy()
                / 86400.0
            )
            mask = (candidates != target_index) & (delta_days <= self.temporal_window_days)
        else:
            delta_days = (
                (target_time - candidate_times).dt.total_seconds().to_numpy()
                / 86400.0
            )
            mask = (candidates != target_index) & (delta_days >= 0)
            if self.temporal_mode == "causal_window":
                mask &= delta_days <= self.temporal_window_days

        selected = candidates[mask]
        if len(selected) == 0:
            return np.asarray([], dtype=int)

        selected_times = self.timestamps.iloc[selected].astype("int64").to_numpy()
        order = np.argsort(selected_times, kind="mergesort")[::-1]
        selected = selected[order]
        return selected[:max_history].astype(int)

    def _delta_days(
        self, target_time: pd.Timestamp, history_indices: np.ndarray
    ) -> np.ndarray:
        if len(history_indices) == 0:
            return np.asarray([], dtype=np.float32)
        deltas = (
            (target_time - self.timestamps.iloc[history_indices])
            .dt.total_seconds()
            .to_numpy()
            / 86400.0
        )
        if self.temporal_mode == "symmetric_window":
            deltas = np.abs(deltas)
        return deltas.astype(np.float32)

    def _decay_weights(self, delta_days: np.ndarray) -> np.ndarray:
        if len(delta_days) == 0:
            return np.asarray([], dtype=np.float32)
        tau = max(self.tau_decay_days, 1e-6)
        return np.exp(-np.maximum(delta_days, 0.0) / tau).astype(np.float32)

    def _context_row(
        self,
        neighborhood: TemporalNeighborhood,
        rating_values: Optional[np.ndarray],
        verified_values: Optional[np.ndarray],
    ) -> List[float]:
        customer_delta = neighborhood.customer_delta_days
        product_delta = neighborhood.product_delta_days
        window = max(self.temporal_window_days, 1.0)
        row = [
            min(len(customer_delta) / max(self.max_customer_history, 1), 1.0),
            min(len(product_delta) / max(self.max_product_history, 1), 1.0),
            float(customer_delta.mean() / window) if len(customer_delta) else 0.0,
            float(product_delta.mean() / window) if len(product_delta) else 0.0,
            float(neighborhood.customer_weights.sum() / max(self.max_customer_history, 1)),
            float(neighborhood.product_weights.sum() / max(self.max_product_history, 1)),
        ]
        row.extend(
            [
                self._history_mean(neighborhood.customer_history_indices, rating_values),
                self._history_mean(neighborhood.product_history_indices, rating_values),
                self._history_mean(neighborhood.customer_history_indices, verified_values),
                self._history_mean(neighborhood.product_history_indices, verified_values),
            ]
        )
        return row

    @staticmethod
    def _history_mean(indices: np.ndarray, values: Optional[np.ndarray]) -> float:
        if values is None or len(indices) == 0:
            return 0.0
        selected = np.asarray(values)[indices]
        if len(selected) == 0:
            return 0.0
        selected = selected.astype(np.float32)
        if np.isnan(selected).all():
            return 0.0
        return float(np.nanmean(selected))
