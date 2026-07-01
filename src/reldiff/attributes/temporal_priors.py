"""Temporal base priors for non-text review attributes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .entity_latent_effects import logit
from .temporal_buckets import (
    canonical_temporal_bucket,
    expected_bucket_format,
    infer_bucket_format,
    is_legacy_bucket_format,
)
from .temporal_causal_features import normalize_verified


class TemporalAttributePrior:
    """Smoothed temporal distributions for rating and verified."""

    def __init__(
        self,
        rating_values: List[Any],
        temporal_prior_level: str = "year_month",
        smoothing_alpha: str | float = "auto",
        eps: float = 1e-8,
    ):
        self.rating_values = list(rating_values)
        self.temporal_prior_level = temporal_prior_level
        self.smoothing_alpha = smoothing_alpha
        self.eps = float(eps)
        self.rating_global_distribution: List[float] = []
        self.verified_global_rate: float = 0.0
        self.per_bucket_rating_distribution: Dict[str, List[float]] = {}
        self.per_bucket_verified_rate: Dict[str, float] = {}
        self.bucket_counts: Dict[str, int] = {}
        self.bucket_format: str = expected_bucket_format(temporal_prior_level)
        self.num_buckets: int = 0
        self.bucket_examples: List[str] = []
        self.min_bucket: str | None = None
        self.max_bucket: str | None = None

    def fit(
        self,
        reviews: pd.DataFrame,
        timestamp_col: str = "review_time",
        rating_col: str = "rating",
        verified_col: str = "verified",
    ) -> "TemporalAttributePrior":
        frame = reviews.copy()
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
        frame = frame.dropna(subset=[timestamp_col, rating_col, verified_col])
        rating_index = {str(value): idx for idx, value in enumerate(self.rating_values)}
        global_counts = np.zeros(len(self.rating_values), dtype=float)
        for value in frame[rating_col]:
            key = str(value)
            if key in rating_index:
                global_counts[rating_index[key]] += 1.0
        if global_counts.sum() == 0:
            global_counts[:] = 1.0
        global_p = global_counts / global_counts.sum()
        self.rating_global_distribution = global_p.tolist()
        self.verified_global_rate = float(normalize_verified(frame[verified_col]).mean())
        buckets = temporal_bucket(frame[timestamp_col], self.temporal_prior_level)
        verified_values = normalize_verified(frame[verified_col])
        for bucket, group_idx in frame.groupby(buckets, sort=True).groups.items():
            indices = list(group_idx)
            counts = np.zeros(len(self.rating_values), dtype=float)
            for value in frame.loc[indices, rating_col]:
                key = str(value)
                if key in rating_index:
                    counts[rating_index[key]] += 1.0
            n = float(len(indices))
            alpha = self._alpha(n)
            p = (counts + alpha * global_p) / max(n + alpha, self.eps)
            verified_rate = float(
                (verified_values.loc[indices].sum() + alpha * self.verified_global_rate)
                / max(n + alpha, self.eps)
            )
            key = str(bucket)
            self.bucket_counts[key] = int(n)
            self.per_bucket_rating_distribution[key] = p.tolist()
            self.per_bucket_verified_rate[key] = verified_rate
        self._refresh_bucket_metadata()
        return self

    def rating_logits_for_timestamps(self, timestamps: pd.Series) -> np.ndarray:
        rows = []
        for bucket in temporal_bucket(pd.to_datetime(timestamps), self.temporal_prior_level):
            dist = self._rating_distribution_for_bucket(bucket)
            rows.append(np.log(np.asarray(dist, dtype=float) + self.eps))
        return np.vstack(rows).astype(np.float32)

    def verified_logits_for_timestamps(self, timestamps: pd.Series) -> np.ndarray:
        rows = []
        for bucket in temporal_bucket(pd.to_datetime(timestamps), self.temporal_prior_level):
            rate = self._verified_rate_for_bucket(bucket)
            rows.append(logit(rate))
        return np.asarray(rows, dtype=np.float32)

    def target_rating_distribution(self, bucket: Any) -> np.ndarray:
        return np.asarray(self._rating_distribution_for_bucket(bucket), dtype=np.float32)

    def target_verified_rate(self, bucket: Any) -> float:
        return float(self._verified_rate_for_bucket(bucket))

    def _rating_distribution_for_bucket(self, bucket: Any) -> List[float]:
        key = str(bucket)
        if key is None:
            return self.rating_global_distribution
        return self.per_bucket_rating_distribution.get(key, self.rating_global_distribution)

    def _verified_rate_for_bucket(self, bucket: Any) -> float:
        key = str(bucket)
        if key is None:
            return self.verified_global_rate
        return float(self.per_bucket_verified_rate.get(key, self.verified_global_rate))

    def _refresh_bucket_metadata(self) -> None:
        buckets = sorted(str(key) for key in self.per_bucket_rating_distribution)
        self.bucket_format = infer_bucket_format(buckets)
        self.num_buckets = len(buckets)
        self.bucket_examples = buckets[:5]
        self.min_bucket = buckets[0] if buckets else None
        self.max_bucket = buckets[-1] if buckets else None

    def uses_legacy_temporal_buckets(self) -> bool:
        return is_legacy_bucket_format(self.per_bucket_rating_distribution)

    def _alpha(self, n: float) -> float:
        if self.smoothing_alpha == "auto":
            return float(max(20.0, np.sqrt(max(n, 1.0))))
        return float(self.smoothing_alpha)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rating_values": self.rating_values,
            "rating_global_distribution": self.rating_global_distribution,
            "verified_global_rate": self.verified_global_rate,
            "per_bucket_rating_distribution": self.per_bucket_rating_distribution,
            "per_bucket_verified_rate": self.per_bucket_verified_rate,
            "bucket_counts": self.bucket_counts,
            "smoothing_alpha": self.smoothing_alpha,
            "temporal_prior_level": self.temporal_prior_level,
            "bucket_format": self.bucket_format,
            "num_buckets": self.num_buckets,
            "bucket_examples": self.bucket_examples,
            "min_bucket": self.min_bucket,
            "max_bucket": self.max_bucket,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalAttributePrior":
        prior = cls(
            data["rating_values"],
            temporal_prior_level=data.get("temporal_prior_level", "year_month"),
            smoothing_alpha=data.get("smoothing_alpha", "auto"),
            eps=data.get("eps", 1e-8),
        )
        prior.rating_global_distribution = data["rating_global_distribution"]
        prior.verified_global_rate = data["verified_global_rate"]
        prior.per_bucket_rating_distribution = data.get("per_bucket_rating_distribution", {})
        prior.per_bucket_verified_rate = data.get("per_bucket_verified_rate", {})
        prior.bucket_counts = data.get("bucket_counts", {})
        prior._refresh_bucket_metadata()
        return prior

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalAttributePrior":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))


def temporal_bucket(timestamps: pd.Series, level: str) -> pd.Series:
    return canonical_temporal_bucket(timestamps, level)


def check_temporal_bucket_consistency(
    prior: TemporalAttributePrior,
    synthetic_timestamps: pd.Series,
    evaluator_timestamps: pd.Series,
    level: str | None = None,
    sampling_level: str | None = None,
    evaluator_level: str | None = None,
) -> Dict[str, Any]:
    """Compare prior, sampling, and evaluator temporal bucket keys."""

    bucket_level = level or prior.temporal_prior_level
    sampling_level = sampling_level or bucket_level
    evaluator_level = evaluator_level or bucket_level
    prior_buckets = set(prior.per_bucket_rating_distribution)
    synthetic_buckets = set(temporal_bucket(synthetic_timestamps, sampling_level).dropna().astype(str))
    evaluator_buckets = set(temporal_bucket(evaluator_timestamps, evaluator_level).dropna().astype(str))
    prior_format = infer_bucket_format(prior_buckets)
    sampling_format = infer_bucket_format(synthetic_buckets)
    evaluator_format = infer_bucket_format(evaluator_buckets)
    common_overlap = prior_buckets & synthetic_buckets & evaluator_buckets
    all_formats_equal = prior_format == sampling_format == evaluator_format
    has_legacy = "legacy-month-number" in {prior_format, sampling_format, evaluator_format}
    is_consistent = bool(all_formats_equal and bool(common_overlap) and not has_legacy)
    return {
        "train_prior_num_buckets": len(prior_buckets),
        "sampling_num_buckets": len(synthetic_buckets),
        "evaluator_num_buckets": len(evaluator_buckets),
        "train_prior_bucket_examples": sorted(prior_buckets)[:5],
        "synthetic_bucket_examples": sorted(synthetic_buckets)[:5],
        "evaluator_bucket_examples": sorted(evaluator_buckets)[:5],
        "train_prior_bucket_format": prior_format,
        "sampling_bucket_format": sampling_format,
        "evaluator_bucket_format": evaluator_format,
        "buckets_missing_in_synthetic": sorted(prior_buckets - synthetic_buckets),
        "buckets_missing_in_prior": sorted((synthetic_buckets | evaluator_buckets) - prior_buckets),
        "buckets_missing_in_evaluator": sorted(prior_buckets - evaluator_buckets),
        "bucket_format": prior_format if all_formats_equal else "mixed",
        "sampling_temporal_bucket_level": sampling_level,
        "evaluator_temporal_bucket_level": evaluator_level,
        "num_overlapping_buckets": len(common_overlap),
        "is_consistent": is_consistent,
    }
