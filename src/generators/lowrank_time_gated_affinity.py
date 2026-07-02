"""Low-rank time-gated dynamic affinity for scalable event-spine pairing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .fast_temporal_activity import canonical_one_time, canonical_time_bucket, resolve_auto_alpha


class LowRankTimeGatedAffinity:
    """Dynamic score F_{u,i,t} = (z_u * g_t)^T z_i without dense tensors."""

    def __init__(
        self,
        rank: int = 32,
        alpha_time_gate: Any = "auto",
        time_gate_granularity: str = "month",
        seed: int = 42,
        eps: float = 1e-12,
    ):
        self.rank = int(rank)
        self.rank_effective = int(rank)
        self.alpha_time_gate = alpha_time_gate
        self.alpha_time_gate_resolved = 1.0
        self.time_gate_granularity = time_gate_granularity
        self.seed = int(seed)
        self.eps = float(eps)
        self.customer_index: Dict[Any, int] = {}
        self.product_index: Dict[Any, int] = {}
        self.customer_ids: np.ndarray = np.asarray([], dtype=object)
        self.product_ids: np.ndarray = np.asarray([], dtype=object)
        self.customer_embeddings = np.zeros((0, self.rank), dtype=float)
        self.product_embeddings = np.zeros((0, self.rank), dtype=float)
        self.time_gates: Dict[str, np.ndarray] = {}
        self.global_gate = np.ones(self.rank, dtype=float)
        self.time_gate_buckets: list[str] = []
        self.fallback_used = False
        self.fit_method = "unfit"
        self.num_events = 0

    def fit(
        self,
        df: pd.DataFrame,
        customer_col: str,
        product_col: str,
        timestamp_col: str,
        time_gate_granularity: Optional[str] = None,
    ) -> "LowRankTimeGatedAffinity":
        if time_gate_granularity is not None:
            self.time_gate_granularity = time_gate_granularity
        frame = df[[customer_col, product_col, timestamp_col]].copy()
        frame["_time_gate"] = canonical_time_bucket(frame[timestamp_col], self.time_gate_granularity)
        self.num_events = int(len(frame))
        self.customer_ids = np.asarray(sorted(frame[customer_col].unique().tolist()), dtype=object)
        self.product_ids = np.asarray(sorted(frame[product_col].unique().tolist()), dtype=object)
        self.customer_index = {customer: idx for idx, customer in enumerate(self.customer_ids)}
        self.product_index = {product: idx for idx, product in enumerate(self.product_ids)}
        counts = frame.groupby([customer_col, product_col]).size().reset_index(name="count")
        self._fit_embeddings(counts, customer_col, product_col)
        gate_counts = frame["_time_gate"].value_counts().sort_index()
        self.alpha_time_gate_resolved = resolve_auto_alpha(self.alpha_time_gate, gate_counts.to_numpy(dtype=int))
        self._fit_time_gates(frame, customer_col, product_col)
        return self

    def _fit_embeddings(self, counts: pd.DataFrame, customer_col: str, product_col: str) -> None:
        rng = np.random.default_rng(self.seed)
        n_customers = len(self.customer_ids)
        n_products = len(self.product_ids)
        if n_customers == 0 or n_products == 0:
            self.customer_embeddings = np.zeros((n_customers, self.rank), dtype=float)
            self.product_embeddings = np.zeros((n_products, self.rank), dtype=float)
            self.global_gate = np.ones(self.rank, dtype=float)
            self.fallback_used = True
            self.fit_method = "empty_fallback"
            return
        try:
            from scipy import sparse
            from sklearn.decomposition import TruncatedSVD

            rows = counts[customer_col].map(self.customer_index).to_numpy()
            cols = counts[product_col].map(self.product_index).to_numpy()
            data = np.log1p(counts["count"].to_numpy(dtype=float))
            matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_customers, n_products))
            effective = min(max(1, self.rank), max(1, min(matrix.shape) - 1))
            if min(matrix.shape) <= 1:
                raise ValueError("interaction matrix too small for truncated SVD")
            svd = TruncatedSVD(n_components=effective, random_state=self.seed)
            user_emb = svd.fit_transform(matrix)
            product_emb = svd.components_.T
            self.rank_effective = int(effective)
            self.customer_embeddings = pad_rank(normalize_rows(user_emb), self.rank)
            self.product_embeddings = pad_rank(normalize_rows(product_emb), self.rank)
            self.fallback_used = False
            self.fit_method = "truncated_svd_log1p_interactions"
        except Exception as exc:
            customer_degree = counts.groupby(customer_col)["count"].sum()
            product_degree = counts.groupby(product_col)["count"].sum()
            user_emb = rng.normal(size=(n_customers, self.rank))
            product_emb = rng.normal(size=(n_products, self.rank))
            for customer, degree in customer_degree.items():
                user_emb[self.customer_index[customer], 0] += np.log1p(float(degree))
            for product, degree in product_degree.items():
                product_emb[self.product_index[product], 0] += np.log1p(float(degree))
            self.rank_effective = int(self.rank)
            self.customer_embeddings = normalize_rows(user_emb)
            self.product_embeddings = normalize_rows(product_emb)
            self.fallback_used = True
            self.fit_method = f"random_popularity_fallback: {exc}"

    def _fit_time_gates(self, frame: pd.DataFrame, customer_col: str, product_col: str) -> None:
        gate_vectors: Dict[str, list[np.ndarray]] = {}
        all_vectors: list[np.ndarray] = []
        for _, row in frame.iterrows():
            u_idx = self.customer_index.get(row[customer_col])
            i_idx = self.product_index.get(row[product_col])
            if u_idx is None or i_idx is None:
                continue
            vector = self.customer_embeddings[u_idx] * self.product_embeddings[i_idx]
            gate = row["_time_gate"]
            gate_vectors.setdefault(gate, []).append(vector)
            all_vectors.append(vector)
        if all_vectors:
            global_gate = np.mean(np.vstack(all_vectors), axis=0)
        else:
            global_gate = np.ones(self.rank, dtype=float)
        self.global_gate = normalize_gate(global_gate, self.eps)
        self.time_gate_buckets = sorted(gate_vectors.keys())
        self.time_gates = {}
        for gate in self.time_gate_buckets:
            raw = np.mean(np.vstack(gate_vectors[gate]), axis=0)
            count = len(gate_vectors[gate])
            weight = float(count / (count + self.alpha_time_gate_resolved))
            shrunk = weight * raw + (1.0 - weight) * self.global_gate
            self.time_gates[gate] = normalize_gate(shrunk, self.eps)

    def score_pairs(self, customer_ids: Sequence[Any], product_ids: Sequence[Any], time_bucket: Any) -> np.ndarray:
        customers = self.transformed_customer_vectors(customer_ids, time_bucket)
        products = self.product_vectors(product_ids)
        if len(customers) == 0:
            return np.asarray([], dtype=float)
        return np.sum(customers * products, axis=1)

    def transformed_customer_vectors(self, customer_ids: Sequence[Any], time_bucket: Any) -> np.ndarray:
        ids = np.asarray(customer_ids, dtype=object)
        vectors = np.zeros((len(ids), self.rank), dtype=float)
        for pos, customer in enumerate(ids):
            idx = self.customer_index.get(customer)
            if idx is not None:
                vectors[pos] = self.customer_embeddings[idx]
        gate = self.gate_for_time(time_bucket)
        return vectors * gate

    def product_vectors(self, product_ids: Sequence[Any]) -> np.ndarray:
        ids = np.asarray(product_ids, dtype=object)
        vectors = np.zeros((len(ids), self.rank), dtype=float)
        for pos, product in enumerate(ids):
            idx = self.product_index.get(product)
            if idx is not None:
                vectors[pos] = self.product_embeddings[idx]
        return vectors

    def gate_for_time(self, time_bucket: Any) -> np.ndarray:
        gate_bucket = canonical_one_time(time_bucket, self.time_gate_granularity)
        return self.time_gates.get(gate_bucket, self.global_gate)

    def summary(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "rank_effective": int(self.rank_effective),
            "num_customers": int(len(self.customer_ids)),
            "num_products": int(len(self.product_ids)),
            "num_events": int(self.num_events),
            "num_time_gate_buckets": int(len(self.time_gate_buckets)),
            "time_gate_granularity": self.time_gate_granularity,
            "alpha_time_gate_requested": self.alpha_time_gate,
            "alpha_time_gate": float(self.alpha_time_gate_resolved),
            "fallback_used": bool(self.fallback_used),
            "fit_method": self.fit_method,
            "formula": "(z_u * g_t)^T z_i",
            "uses_dense_F_u_i_t": False,
        }

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.summary(), handle, indent=2)
            handle.write("\n")


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, eps, None)


def normalize_gate(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.mean(np.abs(values)))
    if not np.isfinite(scale) or scale <= eps:
        return np.ones_like(values, dtype=float)
    return values / scale


def pad_rank(values: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[1] == rank:
        return values
    if values.shape[1] > rank:
        return values[:, :rank]
    out = np.zeros((values.shape[0], rank), dtype=float)
    out[:, : values.shape[1]] = values
    return out
