"""Generative priors over customer/product latent effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .entity_latent_effects import EFFECT_COLUMNS


@dataclass
class PriorSample:
    effects: pd.DataFrame
    diagnostics: Dict[str, Any]


class ConditionalGaussianEffectPrior:
    """Block and degree-bin conditioned Gaussian prior over entity effects."""

    def __init__(
        self,
        entity_type: str,
        id_col: str,
        block_col: str,
        num_degree_bins: int = 4,
        min_entities_per_cell: int = 20,
        covariance_shrinkage: float = 0.2,
        covariance_eps: float = 1e-4,
    ):
        self.entity_type = entity_type
        self.id_col = id_col
        self.block_col = block_col
        self.num_degree_bins = int(num_degree_bins)
        self.min_entities_per_cell = int(min_entities_per_cell)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.covariance_eps = float(covariance_eps)
        self.degree_quantiles: List[float] = []
        self.cells: Dict[str, Dict[str, Any]] = {}
        self.block_cells: Dict[str, Dict[str, Any]] = {}
        self.degree_cells: Dict[str, Dict[str, Any]] = {}
        self.global_cell: Dict[str, Any] = {}

    def fit(self, effects: pd.DataFrame) -> "ConditionalGaussianEffectPrior":
        required = [self.id_col, self.block_col, "degree"] + EFFECT_COLUMNS
        missing = [col for col in required if col not in effects.columns]
        if missing:
            raise ValueError(f"Cannot fit effect prior; missing columns: {missing}")
        frame = effects[required].copy()
        frame = frame.dropna(subset=EFFECT_COLUMNS)
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

    def sample(
        self,
        structural_features: pd.DataFrame,
        rng: Optional[np.random.Generator] = None,
    ) -> PriorSample:
        rng = rng or np.random.default_rng()
        frame = structural_features.copy()
        required = [self.id_col, self.block_col, "degree"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Cannot sample effect prior; missing columns: {missing}")
        frame["degree_bin"] = self.assign_degree_bins(frame["degree"], fit=False)
        rows = []
        cell_counts: Dict[str, int] = {}
        for _, row in frame.iterrows():
            cell, cell_used = self.resolve_cell(row[self.block_col], row["degree_bin"])
            sample = rng.multivariate_normal(
                np.asarray(cell["mean"], dtype=float),
                np.asarray(cell["cov"], dtype=float),
            )
            cell_counts[cell_used] = cell_counts.get(cell_used, 0) + 1
            rows.append(
                {
                    self.id_col: row[self.id_col],
                    "block": int(row[self.block_col]),
                    "degree": int(row["degree"]),
                    "degree_bin": row["degree_bin"],
                    "sampled_rating_effect": float(sample[0]),
                    "sampled_verified_effect": float(sample[1]),
                    "prior_cell_used": cell_used,
                }
            )
        return PriorSample(
            pd.DataFrame(rows),
            {
                "entity_type": self.entity_type,
                "num_sampled": int(len(rows)),
                "prior_cell_counts": cell_counts,
            },
        )

    def resolve_cell(self, block: Any, degree_bin: Any) -> Tuple[Dict[str, Any], str]:
        key = self.cell_key(block, degree_bin)
        if key in self.cells:
            return self.cells[key], key
        block_key = str(block)
        if block_key in self.block_cells:
            return self.block_cells[block_key], f"block:{block_key}"
        degree_key = str(degree_bin)
        if degree_key in self.degree_cells:
            return self.degree_cells[degree_key], f"degree_bin:{degree_key}"
        return self.global_cell, "global"

    def assign_degree_bins(self, degrees: pd.Series, fit: bool) -> pd.Series:
        values = pd.to_numeric(degrees, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if fit:
            if len(values) == 0:
                self.degree_quantiles = []
            else:
                qs = np.linspace(0, 1, self.num_degree_bins + 1)[1:-1]
                quantiles = np.quantile(values, qs).tolist() if len(qs) else []
                self.degree_quantiles = sorted(float(q) for q in set(quantiles))
        labels = []
        for value in values:
            bin_idx = int(np.searchsorted(self.degree_quantiles, value, side="right"))
            labels.append(f"q{bin_idx}")
        return pd.Series(labels, index=degrees.index)

    def _fit_cell(self, frame: pd.DataFrame) -> Dict[str, Any]:
        z = frame[EFFECT_COLUMNS].to_numpy(dtype=float)
        if len(z) == 0:
            z = np.zeros((1, len(EFFECT_COLUMNS)), dtype=float)
        mean = z.mean(axis=0)
        if len(z) <= 1:
            empirical = np.eye(len(EFFECT_COLUMNS)) * self.covariance_eps
        else:
            empirical = np.cov(z, rowvar=False)
            if empirical.ndim == 0:
                empirical = np.eye(len(EFFECT_COLUMNS)) * float(empirical)
        diagonal = np.diag(np.diag(empirical))
        cov = (
            (1.0 - self.covariance_shrinkage) * empirical
            + self.covariance_shrinkage * diagonal
            + np.eye(len(EFFECT_COLUMNS)) * self.covariance_eps
        )
        return {
            "mean": mean.tolist(),
            "cov": cov.tolist(),
            "num_entities": int(len(frame)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_type": "conditional_gaussian",
            "entity_type": self.entity_type,
            "id_col": self.id_col,
            "block_col": self.block_col,
            "effect_columns": EFFECT_COLUMNS,
            "num_degree_bins": self.num_degree_bins,
            "min_entities_per_cell": self.min_entities_per_cell,
            "covariance_shrinkage": self.covariance_shrinkage,
            "covariance_eps": self.covariance_eps,
            "degree_quantiles": self.degree_quantiles,
            "cells": self.cells,
            "block_cells": self.block_cells,
            "degree_cells": self.degree_cells,
            "global_cell": self.global_cell,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalGaussianEffectPrior":
        prior = cls(
            entity_type=data["entity_type"],
            id_col=data["id_col"],
            block_col=data["block_col"],
            num_degree_bins=data.get("num_degree_bins", 4),
            min_entities_per_cell=data.get("min_entities_per_cell", 20),
            covariance_shrinkage=data.get("covariance_shrinkage", 0.2),
            covariance_eps=data.get("covariance_eps", 1e-4),
        )
        prior.degree_quantiles = [float(value) for value in data.get("degree_quantiles", [])]
        prior.cells = data.get("cells", {})
        prior.block_cells = data.get("block_cells", {})
        prior.degree_cells = data.get("degree_cells", {})
        prior.global_cell = data.get("global_cell", {})
        return prior

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalGaussianEffectPrior":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    @staticmethod
    def cell_key(block: Any, degree_bin: Any) -> str:
        return f"block={block}|degree_bin={degree_bin}"


def fit_customer_product_priors(
    customer_effects: pd.DataFrame,
    product_effects: pd.DataFrame,
    customer_id_col: str = "customer_id",
    product_id_col: str = "product_id",
    num_degree_bins: int = 4,
    min_entities_per_cell: int = 20,
) -> Tuple[ConditionalGaussianEffectPrior, ConditionalGaussianEffectPrior, Dict[str, Any]]:
    customer_prior = ConditionalGaussianEffectPrior(
        "customer",
        id_col=customer_id_col,
        block_col="customer_block",
        num_degree_bins=num_degree_bins,
        min_entities_per_cell=min_entities_per_cell,
    ).fit(customer_effects)
    product_prior = ConditionalGaussianEffectPrior(
        "product",
        id_col=product_id_col,
        block_col="product_block",
        num_degree_bins=num_degree_bins,
        min_entities_per_cell=min_entities_per_cell,
    ).fit(product_effects)
    diagnostics = {
        "customer": prior_diagnostics(customer_prior),
        "product": prior_diagnostics(product_prior),
    }
    return customer_prior, product_prior, diagnostics


def save_customer_product_priors(
    output_dir: str | Path,
    customer_prior: ConditionalGaussianEffectPrior,
    product_prior: ConditionalGaussianEffectPrior,
    diagnostics: Dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    prior_dir = output_dir / "entity_effect_priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    customer_prior.save(prior_dir / "customer_prior.json")
    product_prior.save(prior_dir / "product_prior.json")
    with (prior_dir / "prior_diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")


def load_customer_product_priors(
    prior_dir: str | Path,
) -> Tuple[ConditionalGaussianEffectPrior, ConditionalGaussianEffectPrior]:
    root = Path(prior_dir)
    nested = root / "entity_effect_priors"
    if (nested / "customer_prior.json").exists():
        root = nested
    return (
        ConditionalGaussianEffectPrior.load(root / "customer_prior.json"),
        ConditionalGaussianEffectPrior.load(root / "product_prior.json"),
    )


def prior_diagnostics(prior: ConditionalGaussianEffectPrior) -> Dict[str, Any]:
    return {
        "entity_type": prior.entity_type,
        "num_block_degree_cells": len(prior.cells),
        "num_block_fallback_cells": len(prior.block_cells),
        "num_degree_fallback_cells": len(prior.degree_cells),
        "global_num_entities": prior.global_cell.get("num_entities", 0),
        "degree_quantiles": prior.degree_quantiles,
    }
