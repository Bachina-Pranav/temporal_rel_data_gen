"""Causal temporal aggregate features for non-text attribute generation."""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


CAUSAL_CONTINUOUS_FEATURES = [
    "customer_past_review_count",
    "customer_past_avg_rating",
    "customer_past_rating_std",
    "customer_past_verified_rate",
    "customer_days_since_last_review",
    "product_past_review_count",
    "product_past_avg_rating",
    "product_past_rating_std",
    "product_past_verified_rate",
    "product_days_since_last_review",
    "block_pair_past_review_count",
    "block_pair_past_avg_rating",
    "block_pair_past_verified_rate",
    "customer_block_past_review_count",
    "customer_block_past_avg_rating",
    "customer_block_past_verified_rate",
    "product_block_past_review_count",
    "product_block_past_avg_rating",
    "product_block_past_verified_rate",
    "global_past_review_count",
    "global_past_avg_rating",
    "global_past_verified_rate",
    "normalized_time",
    "year",
    "days_since_dataset_start",
    "customer_total_degree_from_spine",
    "product_total_degree_from_spine",
    "customer_degree_log1p",
    "product_degree_log1p",
]

CAUSAL_DISCRETE_FEATURES = [
    "customer_block",
    "product_block",
    "block_pair",
    "month",
    "day_of_week",
    "year_bucket",
]


@dataclass
class RunningAttrStats:
    count: int = 0
    rating_sum: float = 0.0
    rating_sq_sum: float = 0.0
    verified_sum: float = 0.0
    last_time: Optional[pd.Timestamp] = None

    def update(self, rating: float, verified: float, timestamp: pd.Timestamp) -> None:
        self.count += 1
        self.rating_sum += float(rating)
        self.rating_sq_sum += float(rating) ** 2
        self.verified_sum += float(verified)
        timestamp = pd.Timestamp(timestamp)
        if self.last_time is None or timestamp > self.last_time:
            self.last_time = timestamp

    def avg_rating(self, fallback: float) -> float:
        if self.count == 0:
            return float(fallback)
        return float(self.rating_sum / self.count)

    def rating_std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.rating_sum / self.count
        var = max(self.rating_sq_sum / self.count - mean**2, 0.0)
        return float(np.sqrt(var))

    def verified_rate(self, fallback: float) -> float:
        if self.count == 0:
            return float(fallback)
        return float(self.verified_sum / self.count)


@dataclass
class CausalFeatureState:
    customer: Dict[Any, RunningAttrStats] = field(default_factory=lambda: defaultdict(RunningAttrStats))
    product: Dict[Any, RunningAttrStats] = field(default_factory=lambda: defaultdict(RunningAttrStats))
    block_pair: Dict[Any, RunningAttrStats] = field(default_factory=lambda: defaultdict(RunningAttrStats))
    customer_block: Dict[Any, RunningAttrStats] = field(default_factory=lambda: defaultdict(RunningAttrStats))
    product_block: Dict[Any, RunningAttrStats] = field(default_factory=lambda: defaultdict(RunningAttrStats))
    global_stats: RunningAttrStats = field(default_factory=RunningAttrStats)


class TemporalCausalFeatureBuilder:
    """Build aggregate features using only strictly previous time groups.

    For date-only datasets, all rows with the same date form one time group and
    cannot see each other. For datetime datasets, the default grouping is exact
    timestamp.
    """

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        rating_col: str = "rating",
        verified_col: str = "verified",
        customer_blocks: Optional[Dict[Any, int]] = None,
        product_blocks: Optional[Dict[Any, int]] = None,
        date_only: Optional[bool] = None,
        marginal_rating: Optional[float] = None,
        marginal_verified: Optional[float] = None,
    ):
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.rating_col = rating_col
        self.verified_col = verified_col
        self.customer_blocks = dict(customer_blocks or {})
        self.product_blocks = dict(product_blocks or {})
        self.date_only = date_only
        self.marginal_rating = 0.0 if marginal_rating is None else float(marginal_rating)
        self.marginal_verified = (
            0.0 if marginal_verified is None else float(marginal_verified)
        )
        self.min_time: Optional[pd.Timestamp] = None
        self.max_time: Optional[pd.Timestamp] = None
        self.customer_total_degree: Counter = Counter()
        self.product_total_degree: Counter = Counter()
        self.state = CausalFeatureState()

    @classmethod
    def from_structure_debug_dir(
        cls,
        structure_debug_dir: str | Path | None,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        **kwargs: Any,
    ) -> "TemporalCausalFeatureBuilder":
        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, customer_id_col, product_id_col
        )
        return cls(
            customer_id_col=customer_id_col,
            product_id_col=product_id_col,
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
            **kwargs,
        )

    @property
    def continuous_feature_names(self) -> List[str]:
        return list(CAUSAL_CONTINUOUS_FEATURES)

    @property
    def discrete_feature_names(self) -> List[str]:
        return list(CAUSAL_DISCRETE_FEATURES)

    def fit_metadata(self, reviews: pd.DataFrame) -> "TemporalCausalFeatureBuilder":
        reviews = self._preprocess(reviews)
        self.min_time = reviews[self.timestamp_col].min()
        self.max_time = reviews[self.timestamp_col].max()
        self.date_only = detect_date_only(reviews[self.timestamp_col]) if self.date_only is None else self.date_only
        if self.rating_col in reviews.columns:
            self.marginal_rating = float(pd.to_numeric(reviews[self.rating_col]).mean())
        if self.verified_col in reviews.columns:
            self.marginal_verified = float(
                normalize_verified(reviews[self.verified_col]).mean()
            )
        self.customer_total_degree = Counter(reviews[self.customer_id_col])
        self.product_total_degree = Counter(reviews[self.product_id_col])
        return self

    def transform_training(self, reviews: pd.DataFrame) -> pd.DataFrame:
        reviews = self._preprocess(reviews)
        self.fit_metadata(reviews)
        self.reset_history()
        return self._transform_chronological(reviews, update_from_group=True)

    def prepare_sampling(self, spine: pd.DataFrame) -> None:
        spine = self._preprocess(spine)
        if self.min_time is None or self.max_time is None:
            self.min_time = spine[self.timestamp_col].min()
            self.max_time = spine[self.timestamp_col].max()
        if self.date_only is None:
            self.date_only = detect_date_only(spine[self.timestamp_col])
        self.customer_total_degree = Counter(spine[self.customer_id_col])
        self.product_total_degree = Counter(spine[self.product_id_col])
        self.reset_history()

    def transform_current_group(self, group: pd.DataFrame) -> pd.DataFrame:
        group = self._preprocess(group)
        rows = [self._feature_row(row) for _, row in group.iterrows()]
        return pd.DataFrame(rows, index=group.index)

    def update_history(self, generated_group: pd.DataFrame) -> None:
        generated_group = self._preprocess(generated_group)
        for _, row in generated_group.iterrows():
            if self.rating_col not in row or self.verified_col not in row:
                continue
            rating = float(row[self.rating_col])
            verified = float(normalize_verified(pd.Series([row[self.verified_col]])).iloc[0])
            timestamp = pd.Timestamp(row[self.timestamp_col])
            customer_id = row[self.customer_id_col]
            product_id = row[self.product_id_col]
            customer_block = self.customer_block(customer_id)
            product_block = self.product_block(product_id)
            block_pair = self.block_pair(customer_block, product_block)
            self.state.customer[customer_id].update(rating, verified, timestamp)
            self.state.product[product_id].update(rating, verified, timestamp)
            self.state.block_pair[block_pair].update(rating, verified, timestamp)
            self.state.customer_block[customer_block].update(rating, verified, timestamp)
            self.state.product_block[product_block].update(rating, verified, timestamp)
            self.state.global_stats.update(rating, verified, timestamp)

    def reset_history(self) -> None:
        self.state = CausalFeatureState()

    def iter_time_groups(self, df: pd.DataFrame) -> Iterable[Tuple[Any, pd.DataFrame]]:
        df = self._preprocess(df)
        group_key = self._time_group_values(df)
        grouped = df.assign(_time_group_key=group_key).sort_values(
            [self.timestamp_col], kind="mergesort"
        )
        for key, group in grouped.groupby("_time_group_key", sort=True):
            yield key, group.drop(columns=["_time_group_key"])

    def _transform_chronological(
        self, reviews: pd.DataFrame, update_from_group: bool
    ) -> pd.DataFrame:
        features = []
        for _, group in self.iter_time_groups(reviews):
            group_features = self.transform_current_group(group)
            features.append(group_features)
            if update_from_group:
                self.update_history(group)
        if not features:
            return pd.DataFrame(columns=self.continuous_feature_names + self.discrete_feature_names)
        return pd.concat(features).sort_index()

    def _feature_row(self, row: pd.Series) -> Dict[str, float]:
        timestamp = pd.Timestamp(row[self.timestamp_col])
        customer_id = row[self.customer_id_col]
        product_id = row[self.product_id_col]
        customer_block = self.customer_block(customer_id)
        product_block = self.product_block(product_id)
        block_pair = self.block_pair(customer_block, product_block)

        global_rating = self.state.global_stats.avg_rating(self.marginal_rating)
        global_verified = self.state.global_stats.verified_rate(self.marginal_verified)
        min_time = self.min_time or timestamp
        max_time = self.max_time or timestamp
        span_days = max((max_time - min_time).total_seconds() / 86400.0, 1.0)
        days_since_start = max((timestamp - min_time).total_seconds() / 86400.0, 0.0)

        customer_stats = self.state.customer[customer_id]
        product_stats = self.state.product[product_id]
        block_pair_stats = self.state.block_pair[block_pair]
        customer_block_stats = self.state.customer_block[customer_block]
        product_block_stats = self.state.product_block[product_block]

        customer_degree = int(self.customer_total_degree.get(customer_id, 0))
        product_degree = int(self.product_total_degree.get(product_id, 0))
        row_dict: Dict[str, float] = {
            "customer_past_review_count": float(customer_stats.count),
            "customer_past_avg_rating": customer_stats.avg_rating(global_rating),
            "customer_past_rating_std": customer_stats.rating_std(),
            "customer_past_verified_rate": customer_stats.verified_rate(global_verified),
            "customer_days_since_last_review": days_since_last(timestamp, customer_stats.last_time),
            "product_past_review_count": float(product_stats.count),
            "product_past_avg_rating": product_stats.avg_rating(global_rating),
            "product_past_rating_std": product_stats.rating_std(),
            "product_past_verified_rate": product_stats.verified_rate(global_verified),
            "product_days_since_last_review": days_since_last(timestamp, product_stats.last_time),
            "block_pair_past_review_count": float(block_pair_stats.count),
            "block_pair_past_avg_rating": block_pair_stats.avg_rating(global_rating),
            "block_pair_past_verified_rate": block_pair_stats.verified_rate(global_verified),
            "customer_block_past_review_count": float(customer_block_stats.count),
            "customer_block_past_avg_rating": customer_block_stats.avg_rating(global_rating),
            "customer_block_past_verified_rate": customer_block_stats.verified_rate(global_verified),
            "product_block_past_review_count": float(product_block_stats.count),
            "product_block_past_avg_rating": product_block_stats.avg_rating(global_rating),
            "product_block_past_verified_rate": product_block_stats.verified_rate(global_verified),
            "global_past_review_count": float(self.state.global_stats.count),
            "global_past_avg_rating": global_rating,
            "global_past_verified_rate": global_verified,
            "normalized_time": float(days_since_start / span_days),
            "month": float(timestamp.month),
            "day_of_week": float(timestamp.dayofweek),
            "year": float(timestamp.year),
            "year_bucket": float(timestamp.year),
            "days_since_dataset_start": float(days_since_start),
            "customer_total_degree_from_spine": float(customer_degree),
            "product_total_degree_from_spine": float(product_degree),
            "customer_degree_log1p": float(np.log1p(customer_degree)),
            "product_degree_log1p": float(np.log1p(product_degree)),
            "customer_block": float(customer_block),
            "product_block": float(product_block),
            "block_pair": float(block_pair),
        }
        return row_dict

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        return df.dropna(subset=[self.customer_id_col, self.product_id_col, self.timestamp_col])

    def _time_group_values(self, df: pd.DataFrame) -> pd.Series:
        if self.date_only:
            return df[self.timestamp_col].dt.floor("D")
        return df[self.timestamp_col]

    def customer_block(self, customer_id: Any) -> int:
        return int(self.customer_blocks.get(customer_id, -1))

    def product_block(self, product_id: Any) -> int:
        return int(self.product_blocks.get(product_id, -1))

    @staticmethod
    def block_pair(customer_block: int, product_block: int) -> int:
        return int((int(customer_block) + 1) * 1_000_003 + (int(product_block) + 1))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "customer_id_col": self.customer_id_col,
            "product_id_col": self.product_id_col,
            "timestamp_col": self.timestamp_col,
            "rating_col": self.rating_col,
            "verified_col": self.verified_col,
            "date_only": self.date_only,
            "marginal_rating": self.marginal_rating,
            "marginal_verified": self.marginal_verified,
            "min_time": str(self.min_time) if self.min_time is not None else None,
            "max_time": str(self.max_time) if self.max_time is not None else None,
            "continuous_feature_names": self.continuous_feature_names,
            "discrete_feature_names": self.discrete_feature_names,
        }


def load_block_maps(
    structure_debug_dir: str | Path | None,
    customer_id_col: str,
    product_id_col: str,
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    if structure_debug_dir is None:
        warnings.warn("No structure debug directory provided; block features set to unknown.")
        return {}, {}
    debug_dir = Path(structure_debug_dir)
    customer_path = debug_dir / "customer_blocks.csv"
    product_path = debug_dir / "product_blocks.csv"
    if not customer_path.exists() or not product_path.exists():
        warnings.warn(
            "Block assignment files not found in structure debug directory; "
            "block features set to unknown."
        )
        return {}, {}
    customer_df = pd.read_csv(customer_path)
    product_df = pd.read_csv(product_path)
    customer_blocks = {}
    product_blocks = {}
    if customer_id_col in customer_df.columns and "customer_block" in customer_df.columns:
        customer_blocks = dict(
            zip(customer_df[customer_id_col], customer_df["customer_block"].astype(int))
        )
    if product_id_col in product_df.columns and "product_block" in product_df.columns:
        product_blocks = dict(
            zip(product_df[product_id_col], product_df["product_block"].astype(int))
        )
    return customer_blocks, product_blocks


def detect_date_only(timestamps: pd.Series) -> bool:
    values = pd.to_datetime(timestamps, errors="coerce").dropna()
    if len(values) == 0:
        return False
    offsets = (values - values.dt.floor("D")).dt.total_seconds()
    return bool((offsets == 0).mean() >= 0.99)


def normalize_verified(values: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in {"true", "t", "yes", "y", "1"}:
                return 1.0
            if value_lower in {"false", "f", "no", "n", "0"}:
                return 0.0
        return float(bool(value)) if not isinstance(value, (int, float, np.number)) else float(value)

    return values.map(convert).astype(float)


def days_since_last(timestamp: pd.Timestamp, last_time: Optional[pd.Timestamp]) -> float:
    if last_time is None:
        return 0.0
    return float(max((pd.Timestamp(timestamp) - pd.Timestamp(last_time)).total_seconds() / 86400.0, 0.0))


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
