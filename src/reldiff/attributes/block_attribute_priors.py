"""Block and block-pair attribute residual priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .entity_latent_effects import logit
from .temporal_causal_features import load_block_maps, normalize_verified


class BlockAttributePrior:
    """Smoothed rating/verified residuals for block hierarchy."""

    def __init__(self, rating_values: List[Any], smoothing_alpha: float = 20.0, eps: float = 1e-8):
        self.rating_values = list(rating_values)
        self.smoothing_alpha = float(smoothing_alpha)
        self.eps = float(eps)
        self.global_rating_distribution: List[float] = []
        self.global_verified_rate: float = 0.0
        self.block_pair_rating_residuals: Dict[str, List[float]] = {}
        self.customer_block_rating_residuals: Dict[str, List[float]] = {}
        self.product_block_rating_residuals: Dict[str, List[float]] = {}
        self.block_pair_verified_residuals: Dict[str, float] = {}
        self.customer_block_verified_residuals: Dict[str, float] = {}
        self.product_block_verified_residuals: Dict[str, float] = {}

    def fit(
        self,
        reviews: pd.DataFrame,
        structure_debug_dir: str | Path | None = None,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        rating_col: str = "rating",
        verified_col: str = "verified",
    ) -> "BlockAttributePrior":
        frame = reviews.copy()
        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, customer_id_col, product_id_col
        )
        frame["customer_block"] = frame[customer_id_col].map(customer_blocks).fillna(-1).astype(int)
        frame["product_block"] = frame[product_id_col].map(product_blocks).fillna(-1).astype(int)
        frame["block_pair"] = [
            block_pair_key(c, p) for c, p in zip(frame["customer_block"], frame["product_block"])
        ]
        rating_index = {str(value): idx for idx, value in enumerate(self.rating_values)}
        global_counts = count_ratings(frame[rating_col], rating_index)
        if global_counts.sum() == 0:
            global_counts[:] = 1.0
        global_p = global_counts / global_counts.sum()
        self.global_rating_distribution = global_p.tolist()
        self.global_verified_rate = float(normalize_verified(frame[verified_col]).mean())

        self.block_pair_rating_residuals, self.block_pair_verified_residuals = self._fit_group(
            frame, "block_pair", rating_col, verified_col, rating_index, global_p
        )
        self.customer_block_rating_residuals, self.customer_block_verified_residuals = self._fit_group(
            frame, "customer_block", rating_col, verified_col, rating_index, global_p
        )
        self.product_block_rating_residuals, self.product_block_verified_residuals = self._fit_group(
            frame, "product_block", rating_col, verified_col, rating_index, global_p
        )
        return self

    def residuals_for_rows(
        self,
        rows: pd.DataFrame,
        customer_blocks: Dict[Any, int],
        product_blocks: Dict[Any, int],
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
    ) -> Tuple[np.ndarray, np.ndarray]:
        rating_rows = []
        verified_rows = []
        zeros = np.zeros(len(self.rating_values), dtype=np.float32)
        for _, row in rows.iterrows():
            cb = int(customer_blocks.get(row[customer_id_col], -1))
            pb = int(product_blocks.get(row[product_id_col], -1))
            bp = block_pair_key(cb, pb)
            rating = (
                self.block_pair_rating_residuals.get(bp)
                or self.product_block_rating_residuals.get(str(pb))
                or self.customer_block_rating_residuals.get(str(cb))
                or zeros
            )
            verified = (
                self.block_pair_verified_residuals.get(bp)
                if bp in self.block_pair_verified_residuals
                else self.product_block_verified_residuals.get(str(pb), self.customer_block_verified_residuals.get(str(cb), 0.0))
            )
            rating_rows.append(np.asarray(rating, dtype=np.float32))
            verified_rows.append(float(verified))
        return np.vstack(rating_rows).astype(np.float32), np.asarray(verified_rows, dtype=np.float32)

    def _fit_group(self, frame, group_col, rating_col, verified_col, rating_index, global_p):
        rating_residuals = {}
        verified_residuals = {}
        global_log = np.log(global_p + self.eps)
        global_verified_logit = logit(self.global_verified_rate)
        verified = normalize_verified(frame[verified_col])
        for key, group in frame.groupby(group_col, sort=False):
            counts = count_ratings(group[rating_col], rating_index)
            n = float(len(group))
            p = (counts + self.smoothing_alpha * global_p) / max(n + self.smoothing_alpha, self.eps)
            residual = np.log(p + self.eps) - global_log
            residual = residual - residual.mean()
            rating_residuals[str(key)] = residual.tolist()
            rate = float((verified.loc[group.index].sum() + self.smoothing_alpha * self.global_verified_rate) / max(n + self.smoothing_alpha, self.eps))
            verified_residuals[str(key)] = float(logit(rate) - global_verified_logit)
        return rating_residuals, verified_residuals

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockAttributePrior":
        prior = cls(data["rating_values"], data.get("smoothing_alpha", 20.0), data.get("eps", 1e-8))
        prior.__dict__.update(data)
        return prior

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "BlockAttributePrior":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))


def count_ratings(values: pd.Series, rating_index: Dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(rating_index), dtype=float)
    for value in values:
        key = str(value)
        if key in rating_index:
            counts[rating_index[key]] += 1.0
    return counts


def block_pair_key(customer_block: int, product_block: int) -> str:
    return f"{int(customer_block)}:{int(product_block)}"
