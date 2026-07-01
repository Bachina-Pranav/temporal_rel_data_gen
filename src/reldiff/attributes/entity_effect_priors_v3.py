"""Multiclass entity effect priors for V3 non-text generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .entity_latent_effects import compute_entity_structural_features, logit, resolve_alpha
from .temporal_causal_features import load_block_maps, normalize_verified


class ConditionalGaussianEffectPriorV3:
    def __init__(
        self,
        entity_type: str,
        id_col: str,
        block_col: str,
        effect_cols: List[str],
        num_degree_bins: int = 4,
        min_entities_per_cell: int = 20,
        covariance_shrinkage: float = 0.2,
        covariance_eps: float = 1e-4,
        effect_scale: float = 1.0,
    ):
        self.entity_type = entity_type
        self.id_col = id_col
        self.block_col = block_col
        self.effect_cols = list(effect_cols)
        self.num_degree_bins = int(num_degree_bins)
        self.min_entities_per_cell = int(min_entities_per_cell)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.covariance_eps = float(covariance_eps)
        self.effect_scale = float(effect_scale)
        self.degree_quantiles: List[float] = []
        self.cells: Dict[str, Dict[str, Any]] = {}
        self.block_cells: Dict[str, Dict[str, Any]] = {}
        self.degree_cells: Dict[str, Dict[str, Any]] = {}
        self.global_cell: Dict[str, Any] = {}

    def fit(self, effects: pd.DataFrame) -> "ConditionalGaussianEffectPriorV3":
        frame = effects[[self.id_col, self.block_col, "degree"] + self.effect_cols].dropna().copy()
        frame["degree_bin"] = self.assign_degree_bins(frame["degree"], fit=True)
        self.global_cell = self._fit_cell(frame)
        self.block_cells = {
            str(block): self._fit_cell(group)
            for block, group in frame.groupby(self.block_col, sort=False)
            if len(group) >= max(2, min(self.min_entities_per_cell, len(frame)))
        }
        self.degree_cells = {
            str(bin_name): self._fit_cell(group)
            for bin_name, group in frame.groupby("degree_bin", sort=False)
            if len(group) >= max(2, min(self.min_entities_per_cell, len(frame)))
        }
        self.cells = {}
        for (block, bin_name), group in frame.groupby([self.block_col, "degree_bin"], sort=False):
            if len(group) >= self.min_entities_per_cell or len(group) >= max(2, len(frame) // 4):
                self.cells[self.cell_key(block, bin_name)] = self._fit_cell(group)
        return self

    def sample(self, structural: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        rng = rng or np.random.default_rng()
        frame = structural.copy()
        frame["degree_bin"] = self.assign_degree_bins(frame["degree"], fit=False)
        rows = []
        for _, row in frame.iterrows():
            cell, cell_used = self.resolve_cell(row[self.block_col], row["degree_bin"])
            sample = rng.multivariate_normal(
                np.asarray(cell["mean"], dtype=float),
                np.asarray(cell["cov"], dtype=float),
            ) * self.effect_scale
            out = {
                self.id_col: row[self.id_col],
                "block": int(row[self.block_col]),
                "degree": int(row["degree"]),
                "degree_bin": row["degree_bin"],
                "prior_cell_used": cell_used,
            }
            for col, value in zip(self.effect_cols, sample):
                out[f"sampled_{col}"] = float(value)
            rows.append(out)
        return pd.DataFrame(rows)

    def resolve_cell(self, block: Any, degree_bin: Any):
        key = self.cell_key(block, degree_bin)
        if key in self.cells:
            return self.cells[key], key
        if str(block) in self.block_cells:
            return self.block_cells[str(block)], f"block:{block}"
        if str(degree_bin) in self.degree_cells:
            return self.degree_cells[str(degree_bin)], f"degree_bin:{degree_bin}"
        return self.global_cell, "global"

    def assign_degree_bins(self, degrees: pd.Series, fit: bool) -> pd.Series:
        values = pd.to_numeric(degrees, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if fit:
            qs = np.linspace(0, 1, self.num_degree_bins + 1)[1:-1]
            self.degree_quantiles = sorted(float(q) for q in set(np.quantile(values, qs).tolist())) if len(values) else []
        return pd.Series(
            [f"q{int(np.searchsorted(self.degree_quantiles, value, side='right'))}" for value in values],
            index=degrees.index,
        )

    def _fit_cell(self, frame: pd.DataFrame) -> Dict[str, Any]:
        z = frame[self.effect_cols].to_numpy(dtype=float)
        if len(z) == 0:
            z = np.zeros((1, len(self.effect_cols)), dtype=float)
        mean = z.mean(axis=0)
        if len(z) <= 1:
            empirical = np.eye(len(self.effect_cols)) * self.covariance_eps
        else:
            empirical = np.cov(z, rowvar=False)
        diagonal = np.diag(np.diag(empirical))
        cov = (1.0 - self.covariance_shrinkage) * empirical + self.covariance_shrinkage * diagonal
        cov = cov + np.eye(len(self.effect_cols)) * self.covariance_eps
        return {"mean": mean.tolist(), "cov": cov.tolist(), "num_entities": int(len(frame))}

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__, prior_type="conditional_gaussian_v3")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalGaussianEffectPriorV3":
        prior = cls(
            data["entity_type"],
            data["id_col"],
            data["block_col"],
            data["effect_cols"],
            data.get("num_degree_bins", 4),
            data.get("min_entities_per_cell", 20),
            data.get("covariance_shrinkage", 0.2),
            data.get("covariance_eps", 1e-4),
            data.get("effect_scale", 1.0),
        )
        prior.__dict__.update({k: v for k, v in data.items() if k != "prior_type"})
        return prior

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalGaussianEffectPriorV3":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    @staticmethod
    def cell_key(block: Any, degree_bin: Any) -> str:
        return f"block={block}|degree_bin={degree_bin}"


def estimate_entity_effects_v3(
    reviews: pd.DataFrame,
    rating_values: List[Any],
    structure_debug_dir: str | Path | None = None,
    customer_id_col: str = "customer_id",
    product_id_col: str = "product_id",
    timestamp_col: str = "review_time",
    rating_col: str = "rating",
    verified_col: str = "verified",
    alpha_product_rating: str | float = "auto",
    alpha_customer_rating: str | float = "auto",
    alpha_product_verified: str | float = "auto",
    alpha_customer_verified: str | float = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    frame = reviews.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=[customer_id_col, product_id_col, timestamp_col, rating_col, verified_col])
    rating_index = {str(value): idx for idx, value in enumerate(rating_values)}
    global_counts = np.zeros(len(rating_values), dtype=float)
    for value in frame[rating_col]:
        if str(value) in rating_index:
            global_counts[rating_index[str(value)]] += 1
    if global_counts.sum() == 0:
        global_counts[:] = 1.0
    p_base = global_counts / global_counts.sum()
    global_verified = float(normalize_verified(frame[verified_col]).mean())
    customer_blocks, product_blocks = load_block_maps(structure_debug_dir, customer_id_col, product_id_col)
    product_degree = frame.groupby(product_id_col).size()
    customer_degree = frame.groupby(customer_id_col).size()
    product_struct = compute_entity_structural_features(frame, product_id_col, timestamp_col, product_blocks, "product_block", product_id_col)
    customer_struct = compute_entity_structural_features(frame, customer_id_col, timestamp_col, customer_blocks, "customer_block", customer_id_col)
    product = attach_vector_effects(product_struct, frame, product_id_col, product_id_col, "product_block", rating_col, verified_col, rating_values, rating_index, p_base, global_verified, resolve_alpha(alpha_product_rating, product_degree, 10), resolve_alpha(alpha_product_verified, product_degree, 10))
    customer = attach_vector_effects(customer_struct, frame, customer_id_col, customer_id_col, "customer_block", rating_col, verified_col, rating_values, rating_index, p_base, global_verified, resolve_alpha(alpha_customer_rating, customer_degree, 10), resolve_alpha(alpha_customer_verified, customer_degree, 10))
    stats = {
        "rating_values": rating_values,
        "rating_global_distribution": p_base.tolist(),
        "verified_global_rate": global_verified,
        "num_products": int(len(product)),
        "num_customers": int(len(customer)),
    }
    return customer, product, stats


def attach_vector_effects(structural, frame, entity_col, id_col, block_col, rating_col, verified_col, rating_values, rating_index, p_base, global_verified, alpha_rating, alpha_verified):
    rows = []
    verified = normalize_verified(frame[verified_col])
    base_log = np.log(p_base + 1e-8)
    global_verified_logit = logit(global_verified)
    for _, srow in structural.iterrows():
        entity_id = srow[id_col]
        group = frame[frame[entity_col] == entity_id]
        counts = np.zeros(len(rating_values), dtype=float)
        for value in group[rating_col]:
            if str(value) in rating_index:
                counts[rating_index[str(value)]] += 1
        n = float(len(group))
        p = (counts + float(alpha_rating) * p_base) / max(n + float(alpha_rating), 1e-8)
        effect = np.log(p + 1e-8) - base_log
        effect = effect - effect.mean()
        verified_rate = float((verified.loc[group.index].sum() + float(alpha_verified) * global_verified) / max(n + float(alpha_verified), 1e-8))
        out = srow.to_dict()
        out["block"] = int(out.get(block_col, -1))
        out["degree_bin"] = "unknown"
        for idx, value in enumerate(effect):
            out[f"rating_effect_{idx}"] = float(value)
        out["verified_effect"] = float(logit(verified_rate) - global_verified_logit)
        rows.append(out)
    return pd.DataFrame(rows)


def fit_entity_priors_v3(customer_effects, product_effects, customer_id_col="customer_id", product_id_col="product_id", num_degree_bins=4, min_entities_per_cell=20, product_effect_scale=1.0, customer_effect_scale=1.15):
    effect_cols = [col for col in product_effects.columns if col.startswith("rating_effect_")] + ["verified_effect"]
    customer_prior = ConditionalGaussianEffectPriorV3("customer", customer_id_col, "customer_block", effect_cols, num_degree_bins, min_entities_per_cell, effect_scale=customer_effect_scale).fit(customer_effects)
    product_prior = ConditionalGaussianEffectPriorV3("product", product_id_col, "product_block", effect_cols, num_degree_bins, min_entities_per_cell, effect_scale=product_effect_scale).fit(product_effects)
    return customer_prior, product_prior
