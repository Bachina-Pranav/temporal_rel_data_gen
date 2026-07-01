from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.entity_effect_priors import ConditionalGaussianEffectPrior  # noqa: E402


def test_conditional_gaussian_prior_sampling_shape_and_fallback():
    effects = pd.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(12)],
            "product_block": [0] * 6 + [1] * 6,
            "degree": [1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25],
            "rating_effect": np.linspace(-0.5, 0.5, 12),
            "verified_effect": np.linspace(0.4, -0.4, 12),
        }
    )
    prior = ConditionalGaussianEffectPrior(
        "product",
        id_col="product_id",
        block_col="product_block",
        num_degree_bins=3,
        min_entities_per_cell=3,
    ).fit(effects)
    synthetic_struct = pd.DataFrame(
        {
            "product_id": ["p100", "p101", "p102"],
            "product_block": [0, 1, 99],
            "degree": [2, 22, 7],
        }
    )
    sample = prior.sample(synthetic_struct, rng=np.random.default_rng(7)).effects
    assert list(sample.columns) == [
        "product_id",
        "block",
        "degree",
        "degree_bin",
        "sampled_rating_effect",
        "sampled_verified_effect",
        "prior_cell_used",
    ]
    assert len(sample) == 3
    assert sample[["sampled_rating_effect", "sampled_verified_effect"]].notna().all().all()
    assert sample["prior_cell_used"].notna().all()
