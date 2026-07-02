"""Affinity components for joint temporal event generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .temporal_activity_models import canonical_day_bucket, canonical_one_day, normalize_probs


AGE_BINS = ["pre_active", "early", "mid", "late", "post_active", "single_day"]


def day_index(value: Any) -> int:
    return int(pd.Timestamp(canonical_one_day(value)).toordinal())


def product_age_bin(first_time: Any, last_time: Any, time_bucket: Any) -> str:
    first = day_index(first_time)
    last = day_index(last_time)
    current = day_index(time_bucket)
    if current < first:
        return "pre_active"
    if current > last:
        return "post_active"
    span = max(last - first, 0)
    if span == 0:
        return "single_day"
    relative = (current - first) / span
    if relative <= 0.25:
        return "early"
    if relative <= 0.75:
        return "mid"
    return "late"


def product_lifecycle_table(
    df: pd.DataFrame,
    product_col: str,
    time_col: str,
    product_blocks: Mapping[Any, int],
) -> pd.DataFrame:
    frame = df[[product_col, time_col]].copy()
    frame[time_col] = canonical_day_bucket(frame[time_col])
    rows = []
    for product, group in frame.groupby(product_col):
        counts = group[time_col].value_counts()
        first = min(counts.index)
        last = max(counts.index)
        peak = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        span = max(day_index(last) - day_index(first), 0)
        probs = normalize_probs(counts.to_numpy(dtype=float))
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        rows.append(
            {
                "product_id": product,
                "product_block": int(product_blocks.get(product, 0)),
                "degree": int(len(group)),
                "first_time": first,
                "last_time": last,
                "peak_time": peak,
                "active_span_days": int(span),
                "activity_entropy": entropy,
            }
        )
    return pd.DataFrame(rows)


class ProductAgeAffinity:
    """Smoothed log-residual table A[customer_block, product_age_bin]."""

    def __init__(self, age_smoothing: float = 5.0, eps: float = 1e-12):
        self.age_smoothing = float(age_smoothing)
        self.eps = float(eps)
        self.log_residual: Dict[tuple[int, str], float] = {}
        self.table = pd.DataFrame()
        self.product_first_last: Dict[Any, tuple[str, str]] = {}
        self.global_probability: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        customer_col: str,
        product_col: str,
        time_col: str,
        customer_blocks: Mapping[Any, int],
        product_lifecycle: pd.DataFrame,
    ) -> "ProductAgeAffinity":
        frame = df[[customer_col, product_col, time_col]].copy()
        frame[time_col] = canonical_day_bucket(frame[time_col])
        lifecycle = {
            row["product_id"]: (row["first_time"], row["last_time"])
            for _, row in product_lifecycle.iterrows()
        }
        self.product_first_last = dict(lifecycle)
        rows = []
        for _, row in frame.iterrows():
            first, last = lifecycle.get(row[product_col], (row[time_col], row[time_col]))
            rows.append(
                {
                    "customer_block": int(customer_blocks.get(row[customer_col], 0)),
                    "age_bin": product_age_bin(first, last, row[time_col]),
                }
            )
        age_frame = pd.DataFrame(rows)
        global_counts = age_frame["age_bin"].value_counts().reindex(AGE_BINS, fill_value=0).astype(float)
        global_probs = normalize_probs(global_counts.to_numpy(dtype=float) + self.age_smoothing / len(AGE_BINS))
        self.global_probability = dict(zip(AGE_BINS, global_probs))
        table_rows = []
        for cblock, group in age_frame.groupby("customer_block"):
            counts = group["age_bin"].value_counts().reindex(AGE_BINS, fill_value=0).astype(float)
            smoothed = counts.to_numpy(dtype=float) + self.age_smoothing * global_probs
            probs = normalize_probs(smoothed)
            for age_bin, prob, global_prob in zip(AGE_BINS, probs, global_probs):
                residual = float(np.log(max(prob, self.eps)) - np.log(max(global_prob, self.eps)))
                self.log_residual[(int(cblock), age_bin)] = residual
                table_rows.append(
                    {
                        "customer_block": int(cblock),
                        "age_bin": age_bin,
                        "probability": float(prob),
                        "global_probability": float(global_prob),
                        "log_residual": residual,
                    }
                )
        self.table = pd.DataFrame(table_rows)
        return self

    def score(self, customer_block: int, product_id: Any, time_bucket: Any) -> float:
        first, last = self.product_first_last.get(product_id, (time_bucket, time_bucket))
        age_bin = product_age_bin(first, last, time_bucket)
        return float(self.log_residual.get((int(customer_block), age_bin), 0.0))

    def age_bin(self, product_id: Any, time_bucket: Any) -> str:
        first, last = self.product_first_last.get(product_id, (time_bucket, time_bucket))
        return product_age_bin(first, last, time_bucket)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(path, index=False)


class StaticCustomerProductAffinity:
    """Static F_static(u,i), used only as one component of event_score(u,i,t)."""

    def __init__(self, rank: int = 32, seed: int = 42):
        self.rank = int(rank)
        self.seed = int(seed)
        self.customer_index: Dict[Any, int] = {}
        self.product_index: Dict[Any, int] = {}
        self.user_emb: Optional[np.ndarray] = None
        self.item_emb: Optional[np.ndarray] = None
        self.customer_degree: Dict[Any, int] = {}
        self.product_degree: Dict[Any, int] = {}
        self.score_mean = 0.0
        self.score_std = 1.0
        self.fallback_used = False
        self.summary: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, customer_col: str, product_col: str) -> "StaticCustomerProductAffinity":
        counts = df.groupby([customer_col, product_col]).size().reset_index(name="count")
        customers = sorted(df[customer_col].unique().tolist())
        products = sorted(df[product_col].unique().tolist())
        self.customer_index = {customer: idx for idx, customer in enumerate(customers)}
        self.product_index = {product: idx for idx, product in enumerate(products)}
        self.customer_degree = df[customer_col].value_counts().astype(int).to_dict()
        self.product_degree = df[product_col].value_counts().astype(int).to_dict()
        nnz = int(len(counts))
        try:
            from scipy import sparse
            from sklearn.decomposition import TruncatedSVD

            rows = counts[customer_col].map(self.customer_index).to_numpy()
            cols = counts[product_col].map(self.product_index).to_numpy()
            data = np.log1p(counts["count"].to_numpy(dtype=float))
            matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(customers), len(products)))
            rank = max(1, min(self.rank, min(matrix.shape) - 1))
            if rank < 1:
                raise ValueError("matrix too small for SVD")
            svd = TruncatedSVD(n_components=rank, random_state=self.seed)
            user_emb = svd.fit_transform(matrix)
            item_emb = svd.components_.T
            self.user_emb = normalize_rows(user_emb)
            self.item_emb = normalize_rows(item_emb)
            sample_scores = self._calibration_scores(counts, customers, products)
            self.score_mean = float(np.mean(sample_scores)) if len(sample_scores) else 0.0
            self.score_std = float(np.std(sample_scores)) if len(sample_scores) and np.std(sample_scores) > 1e-8 else 1.0
            self.fallback_used = False
            method = "truncated_svd_log1p_interactions"
        except Exception as exc:
            self.user_emb = None
            self.item_emb = None
            self.score_mean = 0.0
            self.score_std = 1.0
            self.fallback_used = True
            method = f"degree_popularity_fallback: {exc}"
        self.summary = {
            "method": method,
            "rank": int(self.rank),
            "num_customers": int(len(customers)),
            "num_products": int(len(products)),
            "nnz": nnz,
            "score_mean": float(self.score_mean),
            "score_std": float(self.score_std),
            "fallback_used": bool(self.fallback_used),
        }
        return self

    def score(self, customer_ids: Sequence[Any], product_ids: Sequence[Any]) -> np.ndarray:
        if len(customer_ids) != len(product_ids):
            raise ValueError("customer_ids and product_ids must have same length")
        scores = np.asarray([self.score_one(u, i) for u, i in zip(customer_ids, product_ids)], dtype=float)
        return scores

    def score_one(self, customer_id: Any, product_id: Any) -> float:
        if self.user_emb is None or self.item_emb is None:
            return self.degree_popularity_score(customer_id, product_id)
        u_idx = self.customer_index.get(customer_id)
        i_idx = self.product_index.get(product_id)
        if u_idx is None or i_idx is None:
            return self.degree_popularity_score(customer_id, product_id)
        raw = float(np.dot(self.user_emb[u_idx], self.item_emb[i_idx]))
        return float((raw - self.score_mean) / max(self.score_std, 1e-8))

    def degree_popularity_score(self, customer_id: Any, product_id: Any) -> float:
        u_degree = float(self.customer_degree.get(customer_id, 0))
        i_degree = float(self.product_degree.get(product_id, 0))
        return float(np.log1p(u_degree) + np.log1p(i_degree))

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.summary, handle, indent=2)
            handle.write("\n")

    def _calibration_scores(self, counts: pd.DataFrame, customers: list[Any], products: list[Any]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        observed = counts.sample(min(len(counts), 2000), random_state=self.seed)
        obs_scores = []
        for _, row in observed.iterrows():
            u_idx = self.customer_index[row.iloc[0]]
            i_idx = self.product_index[row.iloc[1]]
            obs_scores.append(float(np.dot(self.user_emb[u_idx], self.item_emb[i_idx])))
        random_scores = []
        for _ in range(min(len(obs_scores), 2000)):
            u = customers[int(rng.integers(0, len(customers)))]
            i = products[int(rng.integers(0, len(products)))]
            random_scores.append(float(np.dot(self.user_emb[self.customer_index[u]], self.item_emb[self.product_index[i]])))
        return np.asarray(obs_scores + random_scores, dtype=float)


@dataclass
class EventScoreWeights:
    lambda_static: float = 1.0
    lambda_ut: float = 1.0
    lambda_it: float = 1.0
    lambda_age: float = 0.5
    lambda_deg: float = 0.1
    lambda_dup: float = 1.0
    lambda_mem: float = 2.0


class EventAffinityScorer:
    """Time-dependent decomposed score F_{u,i,t} for candidate events."""

    def __init__(
        self,
        static_affinity: StaticCustomerProductAffinity,
        customer_activity,
        product_activity,
        age_affinity: ProductAgeAffinity,
        customer_blocks: Mapping[Any, int],
        real_event_set: set[tuple[Any, Any, str]],
        weights: EventScoreWeights,
        eps: float = 1e-12,
    ):
        self.static_affinity = static_affinity
        self.customer_activity = customer_activity
        self.product_activity = product_activity
        self.age_affinity = age_affinity
        self.customer_blocks = dict(customer_blocks)
        self.real_event_set = real_event_set
        self.weights = weights
        self.eps = float(eps)

    def event_score(
        self,
        customer_id: Any,
        product_ids: Sequence[Any],
        time_bucket: Any,
        remaining_product_degrees: Sequence[float],
        duplicate_counts: Mapping[tuple[Any, Any], int],
    ) -> np.ndarray:
        """Return time-dependent F_{u,i,t}; F_static(u,i) is only one component."""

        time_key = canonical_one_day(time_bucket)
        product_ids = list(product_ids)
        customer_prob = self.customer_activity.probability(customer_id, time_key)
        product_probs = self.product_activity.probabilities(product_ids, time_key)
        customer_block = int(self.customer_blocks.get(customer_id, 0))
        static = self.static_affinity.score([customer_id] * len(product_ids), product_ids)
        age = np.asarray([self.age_affinity.score(customer_block, product_id, time_key) for product_id in product_ids], dtype=float)
        deg = np.log1p(np.asarray(remaining_product_degrees, dtype=float))
        duplicate = np.asarray([duplicate_counts.get((customer_id, product_id), 0) for product_id in product_ids], dtype=float)
        memory = np.asarray([(customer_id, product_id, time_key) in self.real_event_set for product_id in product_ids], dtype=float)
        return (
            self.weights.lambda_static * static
            + self.weights.lambda_ut * np.log(max(customer_prob, self.eps))
            + self.weights.lambda_it * np.log(np.clip(product_probs, self.eps, None))
            + self.weights.lambda_age * age
            + self.weights.lambda_deg * deg
            - self.weights.lambda_dup * duplicate
            - self.weights.lambda_mem * memory
        )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def stable_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0)
    scores = np.clip(scores / max(float(temperature), 1e-8), -30.0, 30.0)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return normalize_probs(exp)
